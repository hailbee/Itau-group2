import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.evaluation.evaluator import Evaluator
from model_utils.models.model_factory import ModelFactory

class GeneralizedEmbeddingExtractor(nn.Module):
    """Generalized embedding extractor that works with any vision-language model."""
    
    def __init__(self, model_wrapper):
        super().__init__()
        self.model_wrapper = model_wrapper
    
    def forward(self, texts):
        return self.model_wrapper.encode_text(texts)
    
    def to(self, device):
        """Move the model wrapper to the specified device."""
        self.model_wrapper.to(device)
        return self

class BaselineTester:
    """
    Generalized interface for testing multiple vision-language model baselines.
    Uses the ModelFactory for cleaner model management.
    """
    
    def __init__(self, model_type='clip', model_name=None, batch_size=32, device=None):
        """
        Initialize baseline tester with specified model.
        
        Args:
            model_type: One of 'clip', 'coca', 'flava', 'siglip', 'openclip'
            model_name: Specific model name (optional, uses default if not provided)
            batch_size: Batch size for processing
            device: Device to run on (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_type = model_type

        print(f"[DEBUG] Requested model_type: {model_type}")

        # Use ModelFactory to create the model wrapper
        try:
            self.model_wrapper = ModelFactory.create_model(
                model_type, model_name, self.device
            )
            print(f"[DEBUG] Successfully loaded model: {model_type}")
        except Exception as e:
            print(f"[ERROR] Failed to load model {model_type}: {e}")
            raise e

        self.extractor = GeneralizedEmbeddingExtractor(self.model_wrapper)
        self.evaluator = Evaluator(self.extractor, batch_size=batch_size)

    def encode(self, texts):
        """Encode texts using the selected model."""
        return self.model_wrapper.encode_text(texts)

    def test(self, reference_filepath, test_filepath):
        """
        Test the selected model performance.
        
        Args:
            reference_filepath: Path to reference data
            test_filepath: Path to test data
            
        Returns:
            tuple: (results_df, metrics)
        """
        print(f"Testing {self.model_type.upper()} model...")
        results_df, metrics = self.evaluator.evaluate(reference_filepath, test_filepath)
        
        print(f"\n{self.model_type.upper()} Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Optimal threshold: {metrics['threshold']:.4f}")
        
        return results_df, metrics
    
    def test_all_models(self, reference_filepath, test_filepath):
        """
        Test all available models and compare their performance.
        
        Args:
            reference_filepath: Path to reference data
            test_filepath: Path to test data
            
        Returns:
            dict: Dictionary with results for each model
        """
        all_results = {}
        available_models = ModelFactory.get_available_models()
        
        for model_type in available_models:
            print(f"\n{'='*50}")
            print(f"Testing {model_type.upper()}")
            print(f"{'='*50}")
            print(f"[DEBUG] Attempting to load and test model_type: {model_type}")
            try:
                # Create new tester for each model
                tester = BaselineTester(model_type, batch_size=self.batch_size, device=self.device)
                results_df, metrics = tester.test(reference_filepath, test_filepath)
                all_results[model_type] = {
                    'results_df': results_df,
                    'metrics': metrics
                }
            except Exception as e:
                print(f"Error testing {model_type}: {str(e)}")
                all_results[model_type] = {
                    'error': str(e)
                }
        
        # Print comparison summary
        self._print_comparison_summary(all_results)
        
        return all_results
    
    def _print_comparison_summary(self, all_results):
        """Print a comparison summary of all model results."""
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'ROC AUC':<10}")
        print("-" * 60)
        
        for model_type, result in all_results.items():
            if 'error' in result:
                print(f"{model_type:<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
            else:
                metrics = result['metrics']
                print(f"{model_type:<12} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
                      f"{metrics['recall']:<10.4f} {metrics['roc_auc']:<10.4f}")
        
        print(f"{'='*60}")

    @staticmethod
    def get_available_models():
        """Get list of available model types."""
        return ModelFactory.get_available_models()
    
    @staticmethod
    def get_default_model_name(model_type):
        """Get the default model name for a given model type."""
        return ModelFactory.get_default_model_name(model_type)
