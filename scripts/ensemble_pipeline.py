#!/usr/bin/env python3
"""
Ensemble Pipeline Implementation

Following the pipeline directions:
1. Similar to evaluate_saved - load pre-trained model and apply to training data
2. Add lev_dist and ratio as columns to results_df
3. Write a file to train XGBoost tree on the 3 features (max_sim, ratio, lev_dist)
4. Get and save results (ROC_AUC, accuracy, etc.)
"""

import torch
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

# Import fuzzy string matching libraries
from fuzzywuzzy import fuzz
import Levenshtein

# Import existing utilities
from scripts.evaluation.evaluator import Evaluator
from scripts.baseline.baseline_tester import BaselineTester
from model_utils.models.learning.siamese import SiameseModelPairs
from utils.evals import find_best_threshold_youden, plot_roc_curve, plot_confusion_matrix

class EnsemblePipeline:
    """
    Ensemble pipeline that combines pre-trained model embeddings with traditional features.
    """
    
    def __init__(self, model_path, backbone='siglip', batch_size=32, device=None):
        """
        Initialize the ensemble pipeline.
        
        Args:
            model_path: Path to the saved .pt model file
            backbone: Backbone model type (clip, siglip, etc.)
            batch_size: Batch size for processing
            device: Device to run on
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.backbone = backbone
        self.model_path = model_path
        
        # Load the pre-trained model
        self._load_model()
        
    def _load_model(self):
        """Load the pre-trained model from the saved .pt file."""
        print(f"Loading pre-trained model from {self.model_path}...")
        
        # Load backbone
        tester = BaselineTester(model_type=self.backbone, batch_size=1, device=self.device)
        backbone_module = tester.model_wrapper
        
        # Load your model with matching dimensions
        # Note: You may need to adjust these dimensions based on your saved model
        embedding_dim = 768  # Adjust based on your model
        projection_dim = 768  # Adjust based on your model
        
        self.model = SiameseModelPairs(embedding_dim=embedding_dim, projection_dim=projection_dim, backbone=backbone_module).to(self.device)
        
        # Load saved weights
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print("Model loaded successfully!")
        
    def compute_levenshtein_distance(self, str1, str2):
        """Compute Levenshtein distance between two strings."""
        return Levenshtein.distance(str1, str2)
    
    def compute_token_set_ratio(self, str1, str2):
        """Compute token set ratio using fuzzywuzzy library."""
        return fuzz.token_set_ratio(str1, str2) / 100.0  # Convert to 0-1 scale
    
    def compute_partial_ratio(self, str1, str2):
        """Compute partial ratio using fuzzywuzzy library."""
        return fuzz.partial_ratio(str1, str2) / 100.0  # Convert to 0-1 scale
    
    def compute_token_sort_ratio(self, str1, str2):
        """Compute token sort ratio using fuzzywuzzy library."""
        return fuzz.token_sort_ratio(str1, str2) / 100.0  # Convert to 0-1 scale
    
    def apply_model_to_training_data(self, training_filepath):
        """
        Step 1: Apply the pre-trained model to training data (similar to evaluate_saved).
        
        Args:
            training_filepath: Path to training data
            
        Returns:
            DataFrame with model predictions and features
        """
        print("Applying pre-trained model to training data...")
        
        # Create evaluator
        evaluator = Evaluator(self.model, batch_size=self.batch_size, model_type='pair')
        
        # Evaluate on training data to get embeddings and similarities
        results_df, metrics = evaluator.evaluate(training_filepath, plot=False)
        
        print(f"Applied model to {len(results_df)} training samples")
        print(f"Training data metrics - Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        
        return results_df
    
    def add_traditional_features(self, results_df):
        """
        Step 2: Add fuzzy string matching features to results_df.
        
        Args:
            results_df: DataFrame with model results
            
        Returns:
            DataFrame with additional features
        """
        print("Adding fuzzy string matching features...")
        
        # Add Levenshtein distance
        results_df['lev_dist'] = results_df.apply(
            lambda row: self.compute_levenshtein_distance(
                str(row['fraudulent_name']), str(row['real_name'])
            ), axis=1
        )
        
        # Add token set ratio
        results_df['token_set_ratio'] = results_df.apply(
            lambda row: self.compute_token_set_ratio(
                str(row['fraudulent_name']), str(row['real_name'])
            ), axis=1
        )
        
        # Add partial ratio
        results_df['partial_ratio'] = results_df.apply(
            lambda row: self.compute_partial_ratio(
                str(row['fraudulent_name']), str(row['real_name'])
            ), axis=1
        )
        
        # Add token sort ratio
        results_df['token_sort_ratio'] = results_df.apply(
            lambda row: self.compute_token_sort_ratio(
                str(row['fraudulent_name']), str(row['real_name'])
            ), axis=1
        )
        
        print(f"Added features for {len(results_df)} samples")
        print(f"Levenshtein distance range: [{results_df['lev_dist'].min()}, {results_df['lev_dist'].max()}]")
        print(f"Token set ratio range: [{results_df['token_set_ratio'].min():.3f}, {results_df['token_set_ratio'].max():.3f}]")
        print(f"Partial ratio range: [{results_df['partial_ratio'].min():.3f}, {results_df['partial_ratio'].max():.3f}]")
        print(f"Token sort ratio range: [{results_df['token_sort_ratio'].min():.3f}, {results_df['token_sort_ratio'].max():.3f}]")
        
        return results_df
    

    
    def train_ensemble_model_with_test(self, train_results_df, test_results_df):
        """
        Train ensemble model using separate training and test data.
        
        Args:
            train_results_df: DataFrame with training data and features
            test_results_df: DataFrame with test data and features
            
        Returns:
            Trained ensemble model and test results
        """
        print("Training ensemble model with separate test data...")
        
        # Prepare training features
        feature_columns = ['max_similarity', 'token_set_ratio', 'lev_dist']
        X_train = train_results_df[feature_columns].values
        y_train = train_results_df['label'].values
        
        # Prepare test features
        X_test = test_results_df[feature_columns].values
        y_test = test_results_df['label'].values
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train Random Forest
        self.ensemble_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.ensemble_model.predict(X_test)
        y_proba = self.ensemble_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_proba)
        
        print(f"Ensemble model performance:")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        # Feature importance
        feature_importance = dict(zip(feature_columns, 
                                    self.ensemble_model.feature_importances_))
        print(f"Feature importance: {feature_importance}")
        
        return {
            'accuracy': test_accuracy,
            'auc': test_auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'feature_importance': feature_importance
        }
    
    def evaluate_and_save_results(self, test_results, output_dir="ensemble_results"):
        """
        Step 4: Get and save results (ROC_AUC, accuracy, etc.).
        
        Args:
            test_results: Results from ensemble model evaluation
            output_dir: Directory to save results
        """
        print("Evaluating and saving results...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble model
        model_filename = f"ensemble_model_{timestamp}.pkl"
        model_path = os.path.join(output_dir, model_filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.ensemble_model, f)
        print(f"Ensemble model saved to: {model_path}")
        
        # Save results
        results_filename = f"ensemble_results_{timestamp}.json"
        results_path = os.path.join(output_dir, results_filename)
        
        # Prepare results for JSON serialization
        results_data = {
            'accuracy': float(test_results['accuracy']),
            'auc': float(test_results['auc']),
            'feature_importance': {k: float(v) for k, v in test_results['feature_importance'].items()},
            'timestamp': timestamp,
            'model_path': self.model_path,
            'backbone': self.backbone
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Print detailed results
        print("\n" + "="*50)
        print("ENSEMBLE MODEL RESULTS")
        print("="*50)
        print(f"Accuracy: {test_results['accuracy']:.4f}")
        print(f"AUC: {test_results['auc']:.4f}")
        print(f"Feature Importance:")
        for feature, importance in test_results['feature_importance'].items():
            print(f"  {feature}: {importance:.4f}")
        print("="*50)
        
        return results_data
    
    def run_pipeline(self, training_filepath, test_filepath, output_dir="ensemble_results"):
        """
        Run the complete ensemble pipeline.
        
        Args:
            training_filepath: Path to training data
            test_filepath: Path to test data
            output_dir: Directory to save results
        """
        print("Starting ensemble pipeline...")
        print(f"Model path: {self.model_path}")
        print(f"Training data: {training_filepath}")
        print(f"Test data: {test_filepath}")
        print(f"Backbone: {self.backbone}")
        
        # Step 1: Apply model to training data
        train_results_df = self.apply_model_to_training_data(training_filepath)
        
        # Step 2: Add traditional features to training data
        train_results_df = self.add_traditional_features(train_results_df)
        
        # Step 3: Apply model to test data
        test_results_df = self.apply_model_to_training_data(test_filepath)
        test_results_df = self.add_traditional_features(test_results_df)
        
        # Step 4: Train ensemble model using training data only
        test_results = self.train_ensemble_model_with_test(train_results_df, test_results_df)
        
        # Step 5: Evaluate and save results
        final_results = self.evaluate_and_save_results(test_results, output_dir)
        
        print("Ensemble pipeline completed successfully!")
        return final_results


