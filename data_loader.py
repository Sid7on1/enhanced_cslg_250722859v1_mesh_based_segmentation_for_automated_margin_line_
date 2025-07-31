import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define constants and configuration
@dataclass
class Config:
    data_dir: str
    batch_size: int
    num_workers: int
    image_size: Tuple[int, int]
    num_classes: int

@dataclass
class ImageMetadata:
    image_path: str
    label: int

class ImageDataset(Dataset):
    def __init__(self, config: Config, metadata: List[ImageMetadata]):
        self.config = config
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        metadata = self.metadata[index]
        image_path = metadata.image_path
        label = metadata.label

        image = cv2.imread(image_path)
        image = cv2.resize(image, self.config.image_size)
        image = image / 255.0

        return {
            'image': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data: List[ImageMetadata]) -> List[ImageMetadata]:
        pass

class ImageDataProcessor(DataProcessor):
    def process(self, data: List[ImageMetadata]) -> List[ImageMetadata]:
        return data

class DataLoader:
    def __init__(self, config: Config, data_processor: DataProcessor):
        self.config = config
        self.data_processor = data_processor
        self.dataset = ImageDataset(config, data_processor.process([]))

    def load_data(self, metadata: List[ImageMetadata]) -> DataLoader:
        self.dataset.metadata = metadata
        return DataLoader(self.config, self.data_processor, self.dataset)

    def __call__(self):
        return DataLoader(self.config, self.data_processor, self.dataset)

class DataBatcher:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def batch_data(self):
        data_loader = self.data_loader()
        return DataLoader(data_loader.config, data_loader.data_processor, data_loader.dataset)

def load_metadata(config: Config) -> List[ImageMetadata]:
    metadata = []
    for file in os.listdir(config.data_dir):
        file_path = os.path.join(config.data_dir, file)
        if os.path.isfile(file_path):
            image_path = file_path
            label = int(file.split('_')[-1].split('.')[0])
            metadata.append(ImageMetadata(image_path, label))
    return metadata

def main():
    config = Config(
        data_dir='path/to/data',
        batch_size=32,
        num_workers=4,
        image_size=(224, 224),
        num_classes=10
    )

    metadata = load_metadata(config)
    data_processor = ImageDataProcessor()
    data_loader = DataLoader(config, data_processor)
    batcher = DataBatcher(data_loader)

    batch = batcher.batch_data()
    for batch_data in batch:
        images = batch_data['image']
        labels = batch_data['label']
        logger.info(f'Batch size: {len(images)}')

if __name__ == '__main__':
    main()