"""
SHAP Utilities to analyse policies
===================================


"""
import torch
import shap
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from numpy.typing import NDArray
from numpy import float32
from eprllib import logger


class eprllibSHAP:
    """
    eprllibSHAP is a utility class for integrating SHAP with RLlib policies.
    """
    def __init__(
        self,
        algorithm: Algorithm,
        policy_name: str
    ):
        """
        Initialize the eprllibSHAP class with an RLlib algorithm and policy name.
        
        Args:
            algorithm (Algorithm): The RLlib algorithm instance.
            policy_name (str): The name of the policy to be used for predictions.
        
        Raises:
            ValueError: If the specified policy name is not found in the algorithm.
        """
        try:
            self.POLICY: Policy = algorithm.get_policy(policy_name)
        except ValueError as e:
            msg = f"Policy '{policy_name}' not found in the algorithm."
            logger.error(msg)
            raise ValueError(msg) from e
        
    def model_predict(
        self,
        data: NDArray[float32],
    ):
        """
        Predict actions using the RLlib policy.
        
        Args:
            data (np.ndarray): The input data for prediction.
        
        Returns:
            np.ndarray: The predicted actions.
        """
        data_tensor = torch.tensor(data, dtype=torch.float32)
        with torch.no_grad():
            predictions: NDArray[float32] = self.POLICY.compute_actions(obs_batch=data_tensor.numpy(),exploration=False)[0]
        return predictions


    def EPExplainer(
        self,
        data: NDArray[float32],
        feature_names: List[str],
        sample_size: Optional[int] = None
    ) -> shap.KernelExplainer:
        """
        Create a SHAP KernelExplainer for the RLlib policy.
        
        Args:
            data (np.ndarray): The input data for the explainer.
            feature_names (list): List of feature names.
            sample_size (int, optional): Number of samples to use for the explainer. Defaults to None.
            
        Returns:
            shap.KernelExplainer: The SHAP KernelExplainer.
        """
        # apply sample data if given
        if sample_size is not None:
            assert isinstance(sample_size, int), "sample_size must be an integer."
            data = shap.sample(data, sample_size)
        
        return shap.KernelExplainer(self.model_predict, data, feature_names)