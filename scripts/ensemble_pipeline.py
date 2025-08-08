#!/usr/bin/env python3
"""
Ensemble Pipeline Implementation

Following the pipeline directions:
1. Similar to evaluate_saved - load pre-trained model and apply to training data
2. Add lev_dist and ratio as columns to results_df
3. Write a file to train XGBoost tree on the 3 features (max_sim, ratio, lev_dist)
4. Get and save results (ROC_AUC, accuracy, etc.)
"""

import argparse
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
        m, n = len(str1), len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                    
        return dp[m][n]
    
    def compute_token_set_ratio(self, str1, str2):
        """Compute token set ratio (Jaccard similarity) between two strings."""
        # Convert to lowercase and split into tokens
        tokens1 = set(str1.lower().split())
        tokens2 = set(str2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union
    
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
        Step 2: Add lev_dist and ratio as columns to results_df.
        
        Args:
            results_df: DataFrame with model results
            
        Returns:
            DataFrame with additional features
        """
        print("Adding traditional features (lev_dist and ratio)...")
        
        # Add Levenshtein distance
        results_df['lev_dist'] = results_df.apply(
            lambda row: self.compute_levenshtein_distance(
                str(row['fraudulent_name']), str(row['real_name'])
            ), axis=1
        )
        
        # Add token set ratio
        results_df['ratio'] = results_df.apply(
            lambda row: self.compute_token_set_ratio(
                str(row['fraudulent_name']), str(row['real_name'])
            ), axis=1
        )
        
        print(f"Added features for {len(results_df)} samples")
        print(f"Levenshtein distance range: [{results_df['lev_dist'].min()}, {results_df['lev_dist'].max()}]")
        print(f"Token set ratio range: [{results_df['ratio'].min():.3f}, {results_df['ratio'].max():.3f}]")
        
        return results_df
    
    def train_ensemble_model(self, results_df, test_size=0.2, random_state=42):
        """
        Step 3: Train XGBoost/Random Forest on the 3 features (max_sim, ratio, lev_dist).
        
        Args:
            results_df: DataFrame with all features
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Trained ensemble model and test results
        """
        print("Training ensemble model (Random Forest) on 3 features...")
        
        # Prepare features
        X = results_df[['max_similarity', 'ratio', 'lev_dist']].values
        y = results_df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train Random Forest (using Random Forest instead of XGBoost for simplicity)
        self.ensemble_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
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
        feature_importance = dict(zip(['max_similarity', 'ratio', 'lev_dist'], 
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
    
    def run_pipeline(self, training_filepath, output_dir="ensemble_results"):
        """
        Run the complete ensemble pipeline.
        
        Args:
            training_filepath: Path to training data
            output_dir: Directory to save results
        """
        print("Starting ensemble pipeline...")
        print(f"Model path: {self.model_path}")
        print(f"Training data: {training_filepath}")
        print(f"Backbone: {self.backbone}")
        
        # Step 1: Apply model to training data
        results_df = self.apply_model_to_training_data(training_filepath)
        
        # Step 2: Add traditional features
        results_df = self.add_traditional_features(results_df)
        
        # Step 3: Train ensemble model
        test_results = self.train_ensemble_model(results_df)
        
        # Step 4: Evaluate and save results
        final_results = self.evaluate_and_save_results(test_results, output_dir)
        
        print("Ensemble pipeline completed successfully!")
        return final_results

def main():
    """Main function to run the ensemble pipeline."""
    parser = argparse.ArgumentParser(description='Ensemble Pipeline for Homoglyph Detection')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the saved .pt model file')
    parser.add_argument('--training_filepath', type=str, required=True,
                      help='Path to training data file')
    parser.add_argument('--backbone', type=str, default='siglip',
                      choices=['clip', 'siglip', 'coca', 'flava'],
                      help='Backbone model type')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for processing')
    parser.add_argument('--output_dir', type=str, default='ensemble_results',
                      help='Directory to save results')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
    
    # Check if training file exists
    if not os.path.exists(args.training_filepath):
        print(f"Error: Training file not found at {args.training_filepath}")
        return
    
    # Run the pipeline
    try:
        pipeline = EnsemblePipeline(
            model_path=args.model_path,
            backbone=args.backbone,
            batch_size=args.batch_size
        )
        
        results = pipeline.run_pipeline(
            training_filepath=args.training_filepath,
            output_dir=args.output_dir
        )
        
        print(f"\nPipeline completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
