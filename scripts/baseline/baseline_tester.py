import torch
from transformers import CLIPModel, CLIPTokenizer
from scripts.evaluation.evaluator import Evaluator

class BaselineTester:
    """
    Unified interface for testing raw CLIP model performance.
    """
    def __init__(self, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Load CLIP model and tokenizer
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
        # Create evaluator
        self.evaluator = Evaluator(self.model, batch_size=batch_size)

    def encode(self, texts):
        """Encode texts using raw CLIP"""
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return features

    def test(self, reference_filepath, test_filepath):
        """
        Test raw CLIP model performance.
        
        Args:
            reference_filepath: Path to reference data
            test_filepath: Path to test data
            
        Returns:
            tuple: (results_df, metrics)
        """
        print("Testing raw CLIP model...")
        results_df, metrics = self.evaluator.evaluate(reference_filepath, test_filepath)
        
        print("\nRaw CLIP Performance:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"Optimal threshold: {metrics['threshold']:.4f}")
        
        return results_df, metrics 