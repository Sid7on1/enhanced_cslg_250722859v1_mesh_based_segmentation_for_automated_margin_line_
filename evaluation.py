import logging
import numpy as np
import torch
from torch import nn
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model: nn.Module, device: torch.device, config: Dict):
        self.model = model
        self.device = device
        self.config = config
        self.scaler = StandardScaler()
        self.metrics = {
            'accuracy': [],
            'auc': [],
            'f1_score': [],
            'precision': [],
            'recall': []
        }

    def evaluate(self, data_loader: torch.utils.data.DataLoader, labels: List[int]) -> Dict:
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                inputs, _ = batch
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        accuracy = accuracy_score(labels, predictions)
        auc_score = roc_auc_score(labels, predictions)
        f1_score = self.calculate_f1_score(labels, predictions)
        precision = self.calculate_precision(labels, predictions)
        recall = self.calculate_recall(labels, predictions)
        self.metrics['accuracy'].append(accuracy)
        self.metrics['auc'].append(auc_score)
        self.metrics['f1_score'].append(f1_score)
        self.metrics['precision'].append(precision)
        self.metrics['recall'].append(recall)
        return {
            'accuracy': accuracy,
            'auc': auc_score,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        }

    def calculate_f1_score(self, labels: List[int], predictions: List[int]) -> float:
        tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
        fp = sum([1 for i in range(len(labels)) if labels[i] == 0 and predictions[i] == 1])
        fn = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0])
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        return f1

    def calculate_precision(self, labels: List[int], predictions: List[int]) -> float:
        tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
        fp = sum([1 for i in range(len(labels)) in range(len(labels)) if labels[i] == 0 and predictions[i] == 1])
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        return precision

    def calculate_recall(self, labels: List[int], predictions: List[int]) -> float:
        tp = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 1])
        fn = sum([1 for i in range(len(labels)) if labels[i] == 1 and predictions[i] == 0])
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        return recall

    def plot_roc_curve(self, labels: List[int], predictions: List[int]) -> None:
        fpr, tpr, _ = roc_curve(labels, predictions)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def save_metrics(self, metrics: Dict, filename: str) -> None:
        with open(filename, 'w') as f:
            json.dump(metrics, f)

def load_config(filename: str) -> Dict:
    with open(filename, 'r') as f:
        config = json.load(f)
    return config

def main():
    config = load_config('config.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth', map_location=device)
    evaluator = ModelEvaluator(model, device, config)
    data_loader = torch.utils.data.DataLoader(torch.load('data.pt'), batch_size=config['batch_size'], shuffle=True)
    labels = torch.load('labels.pt')
    metrics = evaluator.evaluate(data_loader, labels)
    evaluator.save_metrics(metrics, 'metrics.json')
    evaluator.plot_roc_curve(labels, metrics['predictions'])

if __name__ == '__main__':
    main()