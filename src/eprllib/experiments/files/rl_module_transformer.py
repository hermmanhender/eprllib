import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override

# --- 1. Positional Encoding (Identical to your reference) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (seq_len, batch_size, d_model)
        return x + self.pe[:x.size(0), :]

# --- 2. Transformer Encoder Block (Identical to your reference) ---
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None,
                src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        # src shape: (batch_size, seq_len, d_model)
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# --- 3. The New RLModule ---
class TransformerRLModule(TorchRLModule):
    """
    RLlib RLModule implementation of a Transformer encoder.
    """
    def __init__(self, config: RLModuleConfig):
        super().__init__(config)

        # Access observation and action spaces from the config
        obs_space = config.observation_space
        action_space = config.action_space
        
        # In RLModule, model_config is a dictionary passed via the config object
        model_config = config.model_config

        # Determine dimensions from observation space
        # Assuming obs_space is Box(shape=(seq_len, feature_dim))
        self.seq_len = obs_space.shape[0]
        self.feature_dim = obs_space.shape[1]

        # Transformer Params
        self.d_model = model_config.get("d_model", 128)
        self.nhead = model_config.get("nhead", 4)
        self.num_encoder_layers = model_config.get("num_encoder_layers", 2)
        self.dim_feedforward = model_config.get("dim_feedforward", 512)
        self.dropout = model_config.get("dropout", 0.1)

        # If action space is Discrete, get number of actions
        if hasattr(action_space, "n"):
            self.num_outputs = action_space.n
        else:
            # Fallback or handle Box action space (Continuous)
            # For this example we assume Discrete or manual output size
            self.num_outputs = model_config.get("num_outputs", 2)

        # --- Layers ---
        self.input_embedding = nn.Linear(self.feature_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.seq_len)
        
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
            for _ in range(self.num_encoder_layers)
        ])

        # Policy Head (Action Logits)
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.num_outputs)
        )

        # Value Head (Value Function)
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 1)
        )

    def _common_forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Shared forward pass logic for inference, exploration, and training.
        """
        # 1. Get Observations
        # Shape: (batch_size, seq_len, feature_dim)
        obs = batch[Columns.OBS].float()

        # 2. Embedding
        # Shape: (batch_size, seq_len, d_model)
        x = self.input_embedding(obs)

        # 3. Positional Encoding
        # Needs (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        # Back to (batch_size, seq_len, d_model) for Transformer blocks
        x = x.permute(1, 0, 2)

        # 4. Transformer Blocks
        for layer in self.transformer_encoder_layers:
            x = layer(x)

        # 5. Pooling (Mean over sequence)
        # Shape: (batch_size, d_model)
        pooled_output = torch.mean(x, dim=1)

        # 6. Heads
        action_logits = self.policy_head(pooled_output)
        value_preds = self.value_head(pooled_output)

        # 7. Return Dictionary
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
            Columns.VF_PREDS: value_preds.squeeze(-1) # Remove last dim -> (batch_size,)
        }

    # --- RLModule Entry Points ---
    # These methods override the base class to route traffic to our logic.

    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._common_forward(batch)

    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._common_forward(batch)

    @override(TorchRLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return self._common_forward(batch)