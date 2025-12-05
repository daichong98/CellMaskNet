
"""
PanNuke Nucleus Classifier
Train using centroid matching results from train_centroid_matcher.py
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
import argparse
from typing import List, Tuple, Dict


class Config:
    """Configuration Class"""

    def __init__(self):
        # Base path
        self.project_root = "path/to/project/root"

        # Feature file paths
        self.features = {
            'coattn': f"{self.project_root}/PanNuke_classification/step5_coattention/pannuke_coattention_features.csv",
            'morphological': f"{self.project_root}/PanNuke_classification/step6_morphological/pannuke_morphological_features.csv",
            'ring': f"{self.project_root}/PanNuke_classification/step7_ring/pannuke_ring_features.csv",
            'gat': f"{self.project_root}/PanNuke_classification/step7_gat/pannuke_gat_features.csv"
        }

        # Centroid matching results path
        self.centroid_results_dir = f"{self.project_root}/PanNuke_classification/step8_centroid_revised1"

        # Output path
        self.output_dir = f"{self.project_root}/PanNuke_classification/results_simple"

        # Model parameters
        self.hidden_dims = [512, 256, 128, 64]
        self.dropout = 0.3
        self.num_classes = 5

        # Training parameters
        self.learning_rate = 0.00001
        self.batch_size = 1024
        self.num_epochs = 500
        self.val_split = 0.2
        self.random_seed = 42
        self.early_stopping_patience = 30  # Add early stopping patience
        self.use_class_weights = False  # Use class weights to handle imbalance

        # Class names
        self.class_names = ["Neoplastic", "Inflammatory",
                            "Connective", "Dead", "Epithelial"]


class NucleusDataset(Dataset):
    """Nucleus Dataset"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SimpleMLP(nn.Module):
    """Simple MLP Classifier"""

    def __init__(self, input_dim, config):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, config.num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class DataLoader_Simple:
    """Simplified Data Loader"""

    def __init__(self, config):
        self.config = config

    def load_features(self, folds: List[int]) -> pd.DataFrame:
        """Load features for specified fold"""
        print(f"📊 Loading features for Fold {folds}...")

        # Load various feature files
        dfs = {}
        for name, path in self.config.features.items():
            if os.path.exists(path):
                df = pd.read_csv(path)
                # Filter specified fold
                fold_mask = df['image_name'].apply(
                    lambda x: self._extract_fold(x) in folds)
                df_filtered = df[fold_mask].drop_duplicates(
                    subset=['image_name', 'nucleus_id'])
                dfs[name] = df_filtered
                print(f"  {name}: {len(df_filtered)} samples")

        if not dfs:
            raise ValueError(f"Feature file for Fold {folds} not found")

        # Merge features
        merged = dfs['coattn']
        for name, df in dfs.items():
            if name != 'coattn':
                merged = merged.merge(
                    df, on=['image_name', 'nucleus_id'], how='inner')

        print(f"✅ Features after merge: {merged.shape}")
        return merged

    def _extract_fold(self, filename: str) -> int:
        """Extract fold info from filename"""
        match = re.search(r'fold(\d+)', str(filename))
        return int(match.group(1)) if match else -1

    def load_centroid_matches(self) -> Dict[str, pd.DataFrame]:
        """Load centroid matching results"""
        print("Loading centroid matching results...")

        if not os.path.exists(self.config.centroid_results_dir):
            raise FileNotFoundError(
                f"Centroid matching results directory does not exist: {self.config.centroid_results_dir}")

        match_files = glob.glob(os.path.join(
            self.config.centroid_results_dir, "*_matches.csv"))
        print(f"Found {len(match_files)} matching files")

        matches = {}
        for file in match_files:
            key = os.path.basename(file).replace('_matches.csv', '')
            matches[key] = pd.read_csv(file)

        return matches

    def match_features_labels(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get labels using centroid matching results"""
        print("Match features and labels...")

        matches = self.load_centroid_matches()

        # Extract feature columns
        exclude_cols = ['image_name', 'nucleus_id', 'fold']
        feature_cols = [col for col in features_df.columns
                        if col not in exclude_cols and
                        features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        matched_features = []
        matched_labels = []

        stats = {'total': 0, 'matched': 0, 'no_match': 0}

        for _, row in features_df.iterrows():
            stats['total'] += 1
            image_name = str(row['image_name'])
            nucleus_id = int(row['nucleus_id'])

            # Find matching key
            match_key = self._find_match_key(image_name, matches.keys())

            if match_key and match_key in matches:
                match_df = matches[match_key]
                # Find corresponding nucleus in matching results
                nucleus_match = self._find_nucleus_match(match_df, nucleus_id)

                if nucleus_match is not None:
                    label = int(nucleus_match['new_type']) - 1  # Convert to 0-4
                    if 0 <= label < 5:
                        features = row[feature_cols].values.astype(np.float32)
                        matched_features.append(features)
                        matched_labels.append(label)
                        stats['matched'] += 1
                    else:
                        stats['no_match'] += 1
                else:
                    stats['no_match'] += 1
            else:
                stats['no_match'] += 1

        print(
            f"Matching stats: Total={stats['total']}, Success={stats['matched']}, Failed={stats['no_match']}")
        print(f"Matching rate: {stats['matched']/stats['total']*100:.1f}%")

        features_array = np.array(matched_features)
        labels_array = np.array(matched_labels)

        # Show class distribution
        if len(labels_array) > 0:
            unique_labels, counts = np.unique(labels_array, return_counts=True)
            print("📊 Class distribution:")
            for label, count in zip(unique_labels, counts):
                print(f"  {self.config.class_names[label]}: {count}")

        return features_array, labels_array

    def _find_match_key(self, image_name: str, match_keys: List[str]) -> str:
        """Find matching key"""
        # Try multiple possible conversions
        possible_keys = [
            image_name.replace('.png', '_segmentation'),
            re.sub(r'_nucleus_\d+\.png', '_segmentation', image_name),
            re.sub(r'_\d+\.png', '_segmentation', image_name)
        ]

        for key in possible_keys:
            if key in match_keys:
                return key
        return None

    def _find_nucleus_match(self, match_df: pd.DataFrame, nucleus_id: int):
        """Find corresponding nucleus in matching results"""
        # Try different matching methods
        for id_col in ['hovernet_id', 'hovernet_idx']:
            if id_col in match_df.columns:
                matches = match_df[match_df[id_col] == nucleus_id]
                if len(matches) > 0:
                    return matches.iloc[0]
        return None


class NucleusClassifier:
    """Nucleus Classifier"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.model = None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Using device: {self.device}")

    def compute_class_weights(self, labels):
        """Compute class weights"""
        if not self.config.use_class_weights:
            return None

        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        return torch.FloatTensor(weights).to(self.device)

    def prepare_data(self, features, labels):
        """Prepare data"""
        print("📊 Preparing data...")

        # Check data
        if len(features) == 0:
            raise ValueError("No valid training data!")

        print(f"Sample count: {len(features)}")
        print(f"Feature dimension: {features.shape[1]}")

        # Split train/validation set
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels,
            test_size=self.config.val_split,
            random_state=self.config.random_seed,
            stratify=labels if len(np.unique(labels)) > 1 else None
        )

        # Normalize
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Create data loader
        train_dataset = NucleusDataset(X_train_scaled, y_train)
        val_dataset = NucleusDataset(X_val_scaled, y_val)


        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader