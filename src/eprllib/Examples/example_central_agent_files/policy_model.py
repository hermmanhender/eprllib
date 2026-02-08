import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

# Importaciones específicas de RLlib, aunque el entorno y el entrenamiento se eliminan,
# estas son necesarias si este modelo se va a integrar en RLlib posteriormente.
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()

# --- 1. Definición de la Capa de Codificación Posicional ---
class PositionalEncoding(nn.Module):
    """
    Añade información posicional a las incrustaciones de entrada.
    Esto es crucial para que los Transformers entiendan el orden de la secuencia,
    ya que la auto-atención es inherentemente agnóstica al orden.

    Args:
        d_model (int): La dimensión de las incrustaciones de entrada y salida.
                       Debe coincidir con la dimensión del modelo Transformer.
        max_len (int): La longitud máxima esperada de la secuencia.
                       Determina el tamaño precalculado de la matriz de codificación posicional.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        # Inicializa una matriz de ceros para almacenar las codificaciones posicionales.
        # pe.shape: (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # Crea un tensor de posiciones para cada elemento de la secuencia.
        # position.shape: (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Calcula el término de división para las funciones seno y coseno.
        # Este término asegura que las ondas tengan diferentes frecuencias y longitudes de onda.
        # div_term.shape: (d_model / 2)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Aplica la función seno a las posiciones pares de d_model.
        pe[:, 0::2] = torch.sin(position * div_term)
        # Aplica la función coseno a las posiciones impares de d_model.
        pe[:, 1::2] = torch.cos(position * div_term)

        # Reorganiza la forma para que sea (seq_len, 1, d_model) para facilitar la adición a los lotes.
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Registra 'pe' como un buffer, lo que significa que no es un parámetro entrenable
        # pero es parte del estado del módulo y se moverá con el modelo a la GPU/CPU.
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Aplica la codificación posicional a la entrada.
        Aplica la codificación posicional a la entrada.

        Args:
            x (torch.Tensor): El tensor de entrada, típicamente las incrustaciones de la secuencia.
                              Esperado shape: (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: El tensor de entrada con la codificación posicional añadida.
                          Shape: (seq_len, batch_size, d_model).
        """
        # Suma la codificación posicional precalculada a la entrada.
        # Se recorta 'pe' a la longitud de la secuencia actual de 'x'.
        return x + self.pe[:x.size(0), :]

# --- 2. Definición del Bloque del Codificador Transformer ---
class TransformerEncoderBlock(nn.Module):
    """
    Un bloque básico del codificador Transformer, compuesto por una capa de auto-atención
    multi-cabeza y una red feed-forward, con conexiones residuales y normalización de capa.

    Args:
        d_model (int): La dimensión del modelo (dimensión de entrada y salida del bloque).
        nhead (int): El número de cabezas paralelas en la capa de auto-atención multi-cabeza.
        dim_feedforward (int): La dimensión de la capa oculta en la red feed-forward.
        dropout (float): La tasa de dropout a aplicar para la regularización.
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super(TransformerEncoderBlock, self).__init__()
        # Capa de auto-atención multi-cabeza.
        # batch_first=True indica que la entrada esperada es (batch_size, seq_len, features).
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Primera capa lineal de la red feed-forward.
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        # Capa de dropout para la red feed-forward.
        self.dropout = nn.Dropout(dropout)
        # Segunda capa lineal de la red feed-forward.
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Capas de normalización de capa para estabilizar el entrenamiento.
        self.norm1 = nn.LayerNorm(d_model) # Para la salida de la auto-atención
        self.norm2 = nn.LayerNorm(d_model) # Para la salida de la red feed-forward
        # Capas de dropout para las conexiones residuales.
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None) -> Tensor:
        """
        Pasa la entrada a través de un bloque del codificador Transformer.

        Args:
            src (torch.Tensor): El tensor de entrada al bloque del codificador.
                                Esperado shape: (batch_size, seq_len, d_model).
            src_mask (torch.Tensor, opcional): Una máscara para la auto-atención.
                                                Se usa para prevenir la atención a tokens futuros (en decodificadores)
                                                o a tokens enmascarados (padding).
            src_key_padding_mask (torch.Tensor, opcional): Una máscara para enmascarar
                                                            elementos de padding en la secuencia de entrada.

        Returns:
            torch.Tensor: La salida del bloque del codificador Transformer.
                          Shape: (batch_size, seq_len, d_model).
        """
        # 1. Sub-capa de auto-atención multi-cabeza:
        # src2 es la salida de la auto-atención.
        # src, src, src se usan como Query, Key y Value para la auto-atención.
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # Conexión residual y dropout, seguido de normalización de capa.
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # 2. Sub-capa de red feed-forward:
        # src2 es la salida de la red feed-forward.
        # Se aplica una activación ReLU después de la primera capa lineal.
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # Conexión residual y dropout, seguido de normalización de capa.
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# --- 3. Modelo Transformer Personalizado para RLlib ---
class CustomTransformerModel(TorchModelV2, nn.Module):
    """
    Un modelo de red neuronal personalizado para RLlib que incorpora una
    arquitectura de codificador Transformer. Este modelo está diseñado para
    procesar observaciones de secuencia y generar logits de política y valores.

    Args:
        obs_space (gym.spaces.Space): El espacio de observación del entorno.
                                     Se espera que sea un Box con shape=(seq_len, feature_dim).
        action_space (gym.spaces.Space): El espacio de acción del entorno.
        num_outputs (int): El número de salidas para la cabeza de la política (número de acciones discretas).
        model_config (dict): Un diccionario de configuración para el modelo,
                             que contiene parámetros específicos del Transformer.
        name (str): El nombre del modelo.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # Inicializa la clase base de TorchModelV2 de RLlib.
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        # Inicializa la clase base de nn.Module de PyTorch.
        nn.Module.__init__(self)

        # Extraer parámetros del modelo de la configuración proporcionada.
        # Estos parámetros definen la arquitectura del Transformer.
        # self.seq_len: Longitud de la secuencia de la observación.
        self.seq_len = obs_space.shape[0]
        # self.feature_dim: Dimensión de las características de cada elemento en la secuencia.
        self.feature_dim = obs_space.shape[1]
        # self.d_model: Dimensión del modelo Transformer. Todas las capas internas operan en esta dimensión.
        self.d_model = model_config.get("d_model", 128)
        # self.nhead: Número de cabezas de atención en la auto-atención multi-cabeza.
        self.nhead = model_config.get("nhead", 4)
        # self.num_encoder_layers: Número de bloques de codificador Transformer apilados.
        self.num_encoder_layers = model_config.get("num_encoder_layers", 2)
        # self.dim_feedforward: Dimensión de la capa oculta en la red feed-forward de cada bloque.
        self.dim_feedforward = model_config.get("dim_feedforward", 512)
        # self.dropout: Tasa de dropout para regularización.
        self.dropout = model_config.get("dropout", 0.1)

        # Capa de incrustación de entrada: Mapea las características de entrada
        # (feature_dim) a la dimensión del modelo (d_model) del Transformer.
        self.input_embedding = nn.Linear(self.feature_dim, self.d_model)

        # Codificación posicional: Añade información sobre la posición de los elementos en la secuencia.
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=self.seq_len)

        # Bloques del codificador Transformer: Una lista de módulos de TransformerEncoderBlock.
        # Cada bloque procesa la secuencia y refina sus representaciones.
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(self.d_model, self.nhead, self.dim_feedforward, self.dropout)
            for _ in range(self.num_encoder_layers)
        ])

        # Cabeza de la política: Transforma la representación de secuencia agregada
        # en logits para las acciones.
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, self.num_outputs)
        )

        # Cabeza de valor: Transforma la representación de secuencia agregada
        # en una estimación escalar de la función de valor.
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, 1)
        )

        # Variable interna para almacenar la salida de la función de valor.
        # Esto es necesario porque RLlib llama a `value_function()` por separado.
        self._value_out = None

    def forward(self, input_dict: dict, state: list, seq_lens: Tensor) -> tuple[Tensor, list]:
        """
        Define el paso hacia adelante del modelo.

        Args:
            input_dict (dict): Un diccionario que contiene las observaciones bajo la clave "obs".
                               input_dict["obs"] shape: (batch_size, seq_len, feature_dim).
            state (list): El estado recurrente del modelo (no utilizado en este Transformer puramente feed-forward).
            seq_lens (torch.Tensor): Las longitudes de las secuencias en el lote (no utilizado explícitamente aquí
                                     ya que no hay padding o RNNs, pero es un argumento estándar de RLlib).

        Returns:
            tuple[torch.Tensor, list]: Una tupla que contiene:
                                       - logits (torch.Tensor): Las salidas sin normalizar para las acciones.
                                                                Shape: (batch_size, num_outputs).
                                       - state (list): El estado recurrente actualizado (vacío en este caso).
        """
        # Extrae la observación del diccionario de entrada.
        # x shape inicial: (batch_size, seq_len, feature_dim)
        x = input_dict["obs"].float() # Asegura que el tensor de entrada sea de tipo float.

        # Pasa la observación a través de la capa de incrustación de entrada.
        # x shape después de incrustación: (batch_size, seq_len, d_model)
        x = self.input_embedding(x)

        # Transpone el tensor para que la dimensión de la secuencia sea la primera.
        # Esto es requerido por la clase PositionalEncoding (seq_len, batch_size, d_model).
        x = x.permute(1, 0, 2)

        # Añade la codificación posicional a la secuencia incrustada.
        x = self.pos_encoder(x)

        # Vuelve a transponer el tensor a su forma original (batch_size, seq_len, d_model)
        # para los bloques del codificador Transformer que esperan batch_first=True.
        x = x.permute(1, 0, 2)

        # Pasa la secuencia a través de cada bloque del codificador Transformer apilado.
        for layer in self.transformer_encoder_layers:
            x = layer(x) # x mantiene la forma (batch_size, seq_len, d_model)

        # Agregación de la secuencia:
        # Para obtener un vector de características de longitud fija para las cabezas de política y valor,
        # se calcula el promedio a lo largo de la dimensión de la secuencia (dim=1).
        # pooled_output shape: (batch_size, d_model)
        pooled_output = torch.mean(x, dim=1)

        # Calcula los logits de la política utilizando la cabeza de la política.
        # logits shape: (batch_size, num_outputs)
        logits = self.policy_head(pooled_output)

        # Calcula la estimación de valor utilizando la cabeza de valor.
        # value shape: (batch_size, 1)
        value = self.value_head(pooled_output)
        # Almacena la salida de valor (eliminando la dimensión extra de 1) para la función value_function().
        self._value_out = value.squeeze(1)

        # Retorna los logits y el estado (vacío en este caso, ya que no hay estado recurrente).
        return logits, state

    def value_function(self) -> Tensor:
        """
        Retorna la estimación de la función de valor calculada en el paso forward.
        Este método es llamado por RLlib para obtener el valor de la política actual.

        Returns:
            torch.Tensor: El tensor de valores estimados. Shape: (batch_size,).
        """
        # Asegura que forward() haya sido llamado antes de intentar obtener el valor.
        assert self._value_out is not None, "Debe llamar forward() antes de value_function()"
        return self._value_out


# Para utilizar este modelo, necesitarás:
# 1. Definir tu propio entorno gymnasium.Env que genere observaciones de secuencia.
# 2. Registrar este modelo con ModelCatalog.register_custom_model.
# 3. Configurar un algoritmo de RLlib (ej. PPO) para usar este modelo personalizado.