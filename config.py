"""
Model configuration file for computer_vision project.

This module provides a comprehensive configuration management system for the computer_vision project.
It includes settings, parameters, and customization options for various components of the project.
"""

import logging
import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Config:
    """
    Model configuration class.

    This class provides a comprehensive configuration management system for the computer_vision project.
    It includes settings, parameters, and customization options for various components of the project.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration object.

        Args:
            config_file (Optional[str]): Path to the configuration file. Defaults to None.
        """
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """
        Load the configuration from the file.

        Returns:
            Dict: The loaded configuration.
        """
        if self.config_file is None:
            self.config_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
            logger.info(f'Using default configuration file: {self.config_file}')

        try:
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f'Loaded configuration from file: {self.config_file}')
                return config
        except FileNotFoundError:
            logger.error(f'Configuration file not found: {self.config_file}')
            return {}
        except yaml.YAMLError as e:
            logger.error(f'Error parsing configuration file: {e}')
            return {}

    def get_config(self) -> Dict:
        """
        Get the current configuration.

        Returns:
            Dict: The current configuration.
        """
        return self.config

    def update_config(self, config: Dict) -> None:
        """
        Update the configuration.

        Args:
            config (Dict): The new configuration.
        """
        self.config.update(config)
        self.save_config()

    def save_config(self) -> None:
        """
        Save the configuration to the file.
        """
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f'Saved configuration to file: {self.config_file}')

class ModelConfig(Config):
    """
    Model configuration class.

    This class provides a comprehensive configuration management system for the computer_vision project.
    It includes settings, parameters, and customization options for various components of the project.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the model configuration object.

        Args:
            config_file (Optional[str]): Path to the configuration file. Defaults to None.
        """
        super().__init__(config_file)
        self.model_config = self.load_model_config()

    def load_model_config(self) -> Dict:
        """
        Load the model configuration from the file.

        Returns:
            Dict: The loaded model configuration.
        """
        model_config_file = os.path.join(os.path.dirname(__file__), 'model_config.yaml')
        try:
            with open(model_config_file, 'r') as f:
                model_config = yaml.safe_load(f)
                logger.info(f'Loaded model configuration from file: {model_config_file}')
                return model_config
        except FileNotFoundError:
            logger.error(f'Model configuration file not found: {model_config_file}')
            return {}
        except yaml.YAMLError as e:
            logger.error(f'Error parsing model configuration file: {e}')
            return {}

    def get_model_config(self) -> Dict:
        """
        Get the current model configuration.

        Returns:
            Dict: The current model configuration.
        """
        return self.model_config

    def update_model_config(self, config: Dict) -> None:
        """
        Update the model configuration.

        Args:
            config (Dict): The new model configuration.
        """
        self.model_config.update(config)
        self.save_model_config()

    def save_model_config(self) -> None:
        """
        Save the model configuration to the file.
        """
        model_config_file = os.path.join(os.path.dirname(__file__), 'model_config.yaml')
        with open(model_config_file, 'w') as f:
            yaml.dump(self.model_config, f, default_flow_style=False)
            logger.info(f'Saved model configuration to file: {model_config_file}')

class AlgorithmConfig(Config):
    """
    Algorithm configuration class.

    This class provides a comprehensive configuration management system for the computer_vision project.
    It includes settings, parameters, and customization options for various components of the project.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the algorithm configuration object.

        Args:
            config_file (Optional[str]): Path to the configuration file. Defaults to None.
        """
        super().__init__(config_file)
        self.algorithm_config = self.load_algorithm_config()

    def load_algorithm_config(self) -> Dict:
        """
        Load the algorithm configuration from the file.

        Returns:
            Dict: The loaded algorithm configuration.
        """
        algorithm_config_file = os.path.join(os.path.dirname(__file__), 'algorithm_config.yaml')
        try:
            with open(algorithm_config_file, 'r') as f:
                algorithm_config = yaml.safe_load(f)
                logger.info(f'Loaded algorithm configuration from file: {algorithm_config_file}')
                return algorithm_config
        except FileNotFoundError:
            logger.error(f'Algorithm configuration file not found: {algorithm_config_file}')
            return {}
        except yaml.YAMLError as e:
            logger.error(f'Error parsing algorithm configuration file: {e}')
            return {}

    def get_algorithm_config(self) -> Dict:
        """
        Get the current algorithm configuration.

        Returns:
            Dict: The current algorithm configuration.
        """
        return self.algorithm_config

    def update_algorithm_config(self, config: Dict) -> None:
        """
        Update the algorithm configuration.

        Args:
            config (Dict): The new algorithm configuration.
        """
        self.algorithm_config.update(config)
        self.save_algorithm_config()

    def save_algorithm_config(self) -> None:
        """
        Save the algorithm configuration to the file.
        """
        algorithm_config_file = os.path.join(os.path.dirname(__file__), 'algorithm_config.yaml')
        with open(algorithm_config_file, 'w') as f:
            yaml.dump(self.algorithm_config, f, default_flow_style=False)
            logger.info(f'Saved algorithm configuration to file: {algorithm_config_file}')

def get_config() -> Config:
    """
    Get the current configuration.

    Returns:
        Config: The current configuration.
    """
    return Config()

def get_model_config() -> ModelConfig:
    """
    Get the current model configuration.

    Returns:
        ModelConfig: The current model configuration.
    """
    return ModelConfig()

def get_algorithm_config() -> AlgorithmConfig:
    """
    Get the current algorithm configuration.

    Returns:
        AlgorithmConfig: The current algorithm configuration.
    """
    return AlgorithmConfig()

if __name__ == '__main__':
    config = get_config()
    model_config = get_model_config()
    algorithm_config = get_algorithm_config()

    print(config.get_config())
    print(model_config.get_model_config())
    print(algorithm_config.get_algorithm_config())