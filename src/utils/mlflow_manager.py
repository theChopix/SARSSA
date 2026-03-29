"""
Utility functions for interacting with MLflow runs.

This module provides helper functions to retrieve artifacts and parameters
from MLflow runs, including support for JSON, NPY, and NPZ file formats.
"""

import json
import numpy as np
import scipy.sparse as sp
import mlflow
from pathlib import Path
from typing import Any, Dict, Optional, Union


class MLflowRunLoader:
    """
    Helper class for loading artifacts and parameters from MLflow runs.
    
    Usage:
        loader = MLflowRunLoader(run_id="abc123")
        
        # Get parameters
        params = loader.get_parameters()
        param_value = loader.get_parameter("learning_rate")
        
        # Get artifacts
        metadata = loader.get_json_artifact("metadata.json")
        embeddings = loader.get_npy_artifact("embeddings.npy")
        matrix = loader.get_npz_artifact("matrix.npz")
        
        # Get artifact with custom path
        data = loader.get_json_artifact("subdir/config.json", artifact_path="custom")
    """
    
    def __init__(self, run_id: str):
        """
        Initialize the MLflow run loader.
        
        Args:
            run_id: MLflow run ID to load artifacts and parameters from
        """
        self.run_id = run_id
        self._run = None
        self._artifact_uri = None
    
    @property
    def run(self) -> mlflow.entities.Run:
        """Get the MLflow run object, loading it once and caching."""
        if self._run is None:
            self._run = mlflow.get_run(self.run_id)
        return self._run
    
    @property
    def artifact_uri(self) -> str:
        """Get the base artifact URI for this run."""
        if self._artifact_uri is None:
            base_uri = self.run.info.artifact_uri
            # Convert to local path if needed
            if base_uri.startswith('file://'):
                base_uri = base_uri[7:]
            elif 'mlruns' in base_uri:
                base_uri = './' + base_uri[base_uri.find('mlruns'):]
            self._artifact_uri = base_uri
        return self._artifact_uri
    
    def get_artifact_path(self, filename: str, artifact_path: Optional[str] = None) -> str:
        """
        Construct the full path to an artifact file.
        
        Args:
            filename: Name of the artifact file
            artifact_path: Optional subdirectory within artifacts (e.g., "dataset", "model")
            
        Returns:
            Full path to the artifact file
        """
        if artifact_path:
            return f"{self.artifact_uri}/{artifact_path}/{filename}"
        return f"{self.artifact_uri}/{filename}"
    
    def get_parameters(self) -> Dict[str, str]:
        """
        Get all parameters from the MLflow run.
        
        Returns:
            Dictionary of parameter names to values
        """
        return self.run.data.params
    
    def get_parameter(self, param_name: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a specific parameter from the MLflow run.
        
        Args:
            param_name: Name of the parameter to retrieve
            default: Default value if parameter doesn't exist
            
        Returns:
            Parameter value as string, or default if not found
        """
        return self.run.data.params.get(param_name, default)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get all metrics from the MLflow run.
        
        Returns:
            Dictionary of metric names to values
        """
        return self.run.data.metrics
    
    def get_metric(self, metric_name: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get a specific metric from the MLflow run.
        
        Args:
            metric_name: Name of the metric to retrieve
            default: Default value if metric doesn't exist
            
        Returns:
            Metric value as float, or default if not found
        """
        return self.run.data.metrics.get(metric_name, default)
    
    def get_json_artifact(
        self, 
        filename: str, 
        artifact_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a JSON artifact from the MLflow run.
        
        Args:
            filename: Name of the JSON file (e.g., "metadata.json")
            artifact_path: Optional subdirectory within artifacts
            
        Returns:
            Parsed JSON data as a dictionary
            
        Raises:
            FileNotFoundError: If the artifact file doesn't exist
            json.JSONDecodeError: If the file is not valid JSON
        """
        file_path = self.get_artifact_path(filename, artifact_path)
        with open(file_path, "r") as f:
            return json.load(f)
    
    def get_npy_artifact(
        self, 
        filename: str, 
        artifact_path: Optional[str] = None,
        allow_pickle: bool = False
    ) -> np.ndarray:
        """
        Load a NumPy array artifact (.npy) from the MLflow run.
        
        Args:
            filename: Name of the NPY file (e.g., "embeddings.npy")
            artifact_path: Optional subdirectory within artifacts
            
        Returns:
            NumPy array
            
        Raises:
            FileNotFoundError: If the artifact file doesn't exist
        """
        file_path = self.get_artifact_path(filename, artifact_path)
        return np.load(file_path, allow_pickle=allow_pickle)
    
    def get_npz_artifact(
        self, 
        filename: str, 
        artifact_path: Optional[str] = None,
        return_sparse: bool = True
    ) -> Union[sp.csr_matrix, np.lib.npyio.NpzFile]:
        """
        Load a compressed NumPy artifact (.npz) from the MLflow run.
        
        Args:
            filename: Name of the NPZ file (e.g., "matrix.npz")
            artifact_path: Optional subdirectory within artifacts
            return_sparse: If True, load as scipy sparse matrix (default: True)
                          If False, return raw npz file object
            
        Returns:
            Scipy sparse CSR matrix if return_sparse=True, otherwise NpzFile object
            
        Raises:
            FileNotFoundError: If the artifact file doesn't exist
        """
        file_path = self.get_artifact_path(filename, artifact_path)
        if return_sparse:
            return sp.load_npz(file_path)
        return np.load(file_path)
    
    def artifact_exists(
        self, 
        filename: str, 
        artifact_path: Optional[str] = None
    ) -> bool:
        """
        Check if an artifact file exists.
        
        Args:
            filename: Name of the artifact file
            artifact_path: Optional subdirectory within artifacts
            
        Returns:
            True if the artifact exists, False otherwise
        """
        file_path = self.get_artifact_path(filename, artifact_path)
        return Path(file_path).exists()


def get_run_parameters(run_id: str) -> Dict[str, str]:
    """
    Get all parameters from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Dictionary of parameter names to values
    """
    loader = MLflowRunLoader(run_id)
    return loader.get_parameters()


def get_run_parameter(run_id: str, param_name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get a specific parameter from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        param_name: Name of the parameter to retrieve
        default: Default value if parameter doesn't exist
        
    Returns:
        Parameter value as string, or default if not found
    """
    loader = MLflowRunLoader(run_id)
    return loader.get_parameter(param_name, default)


def get_json_artifact(
    run_id: str, 
    filename: str, 
    artifact_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a JSON artifact from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        filename: Name of the JSON file (e.g., "metadata.json")
        artifact_path: Optional subdirectory within artifacts
        
    Returns:
        Parsed JSON data as a dictionary
    """
    loader = MLflowRunLoader(run_id)
    return loader.get_json_artifact(filename, artifact_path)


def get_npy_artifact(
    run_id: str, 
    filename: str, 
    artifact_path: Optional[str] = None,
    allow_pickle: bool = False
) -> np.ndarray:
    """
    Load a NumPy array artifact (.npy) from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        filename: Name of the NPY file (e.g., "embeddings.npy")
        artifact_path: Optional subdirectory within artifacts
        
    Returns:
        NumPy array
    """
    loader = MLflowRunLoader(run_id)
    return loader.get_npy_artifact(filename, artifact_path, allow_pickle=allow_pickle)


def get_npz_artifact(
    run_id: str, 
    filename: str, 
    artifact_path: Optional[str] = None,
    return_sparse: bool = True
) -> Union[sp.csr_matrix, np.lib.npyio.NpzFile]:
    """
    Load a compressed NumPy artifact (.npz) from an MLflow run.
    
    Args:
        run_id: MLflow run ID
        filename: Name of the NPZ file (e.g., "matrix.npz")
        artifact_path: Optional subdirectory within artifacts
        return_sparse: If True, load as scipy sparse matrix (default: True)
                      If False, return raw npz file object
        
    Returns:
        Scipy sparse CSR matrix if return_sparse=True, otherwise NpzFile object
    """
    loader = MLflowRunLoader(run_id)
    return loader.get_npz_artifact(filename, artifact_path, return_sparse)
