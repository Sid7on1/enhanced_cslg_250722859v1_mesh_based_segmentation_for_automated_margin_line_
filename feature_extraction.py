import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    Feature Extractor class for computer vision tasks.
    Provides methods for loading models, extracting features, and saving/loading features.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initializes the FeatureExtractor.
        Args:
            model_path (str): Path to the trained model for feature extraction.
            device (str, optional): Device to load the model on (cpu or cuda). Defaults to 'cpu'.
        """
        self.model_path = model_path
        self.device = device
        self.model = self._load_model(model_path, device)

    def _load_model(self, model_path: str, device: str) -> torch.nn.Module:
        """
        Loads the trained model from the specified path onto the device.
        Args:
            model_path (str): Path to the model file.
            device (str): Device to load the model on (cpu or cuda).
        Returns:
            torch.nn.Module: Loaded model.
        """
        logger.info(f"Loading model from {model_path} onto {device}...")
        try:
            # Load the model using torch.load
            model = torch.load(model_path, map_location=device)
            model.eval()  # Set model to evaluation mode
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            sys.exit(1)

    def extract_features(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extracts features from a list of images using the loaded model.
        Args:
            images (List[np.ndarray]): List of input images of shape (H, W, C).
        Returns:
            np.ndarray: Array of extracted features of shape (N, feature_dim).
        """
        if not images:
            logger.warning("No images provided for feature extraction.")
            return np.array([])

        # Convert images to torch tensors and normalize
        images_tensor = self._preprocess_images(images)

        # Move tensors to the specified device
        images_tensor = images_tensor.to(self.device)

        # Extract features using the model
        features = self.model(images_tensor)
        features = features.squeeze(2).squeeze(2)  # Remove spatial dimensions

        return features.cpu().numpy()

    def _preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocesses a list of images by converting them to torch tensors and normalizing.
        Args:
            images (List[np.ndarray]): List of input images of shape (H, W, C).
        Returns:
            torch.Tensor: Tensor of preprocessed images of shape (N, C, H, W).
        """
        # Convert images to float tensors
        images_tensor = torch.as_tensor(images).float()

        # Normalize images
        mean = torch.as_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        images_tensor = (images_tensor / 255.0 - mean) / std

        # Transpose to match model input shape
        images_tensor = images_tensor.permute(0, 3, 1, 2)

        return images_tensor

    def save_features(self, features: np.ndarray, filenames: List[str], output_path: str) -> None:
        """
        Saves extracted features along with corresponding filenames to a CSV file.
        Args:
            features (np.ndarray): Array of extracted features of shape (N, feature_dim).
            filenames (List[str]): List of filenames corresponding to the images.
            output_path (str): Path to the output CSV file.
        """
        if features.shape[0] != len(filenames):
            logger.error("Number of features and filenames do not match.")
            return

        # Create a DataFrame with features and filenames
        data = {'filename': filenames, 'features': list(features)}
        df = pd.DataFrame(data)

        # Save to CSV file
        df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")

    def load_features(self, input_path: str) -> Dict[str, np.ndarray]:
        """
        Loads extracted features and corresponding filenames from a CSV file.
        Args:
            input_path (str): Path to the input CSV file.
        Returns:
            Dict[str, np.ndarray]: Dictionary with 'filenames' and 'features' as keys.
        """
        try:
            # Read the CSV file
            df = pd.read_csv(input_path)

            # Convert features column to numpy array
            features = df['features'].to_numpy()
            filenames = df['filename'].tolist()

            return {'filenames': filenames, 'features': features}
        except FileNotFoundError:
            logger.error(f"Features file not found at {input_path}.")
            return {'filenames': [], 'features': np.array([])}
        except Exception as e:
            logger.error(f"Error loading features: {e}")
            return {'filenames': [], 'features': np.array([])}

# Example usage
if __name__ == '__main__':
    extractor = FeatureExtractor(model_path='path_to_model.pth', device='cuda')  # Replace with your model path

    # Placeholder images for demonstration
    images = [np.random.rand(224, 224, 3) for _ in range(10)]

    # Extract features
    features = extractor.extract_features(images)
    print(features.shape)  # Should output: (10, feature_dim)

    # Save and load features (replace with your paths)
    extractor.save_features(features, ['image1.jpg', 'image2.jpg'], 'output.csv')
    loaded_features = extractor.load_features('output.csv')
    print(loaded_features['features'].shape)  # Should output: (10, feature_dim)