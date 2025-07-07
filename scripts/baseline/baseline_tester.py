import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from transformers import CLIPModel, CLIPTokenizer, AutoModel, AutoTokenizer
from scripts.evaluation.evaluator import Evaluator

class BaseVisionLanguageModel(ABC):
    """Abstract base class for vision-language models."""
    
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """Load the model and tokenizer."""
        pass
    
    @abstractmethod
    def encode_text(self, texts):
        """Encode text inputs to embeddings."""
        pass
    
    def to(self, device):
        """Move model to specified device."""
        self.device = device
        if self.model:
            self.model = self.model.to(device)
        return self

class CLIPModelWrapper(BaseVisionLanguageModel):
    """Wrapper for OpenAI CLIP models."""
    
    def _load_model(self):
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
        return F.normalize(features, dim=1)

class CoCaModelWrapper(BaseVisionLanguageModel):
    """Wrapper for CoCa/GIT models."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # GIT/CoCa models use the last hidden state for text features
            if hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)  # Pool over sequence length
            elif hasattr(outputs, 'text_embeds'):
                features = outputs.text_embeds
            else:
                # Fallback to logits if available
                features = outputs.logits.mean(dim=1)
        return F.normalize(features, dim=1)

class FLAVAModelWrapper(BaseVisionLanguageModel):
    """Wrapper for FLAVA models."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # FLAVA models have different output structures
            if hasattr(outputs, 'text_embeds'):
                features = outputs.text_embeds
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            else:
                # Fallback to logits
                features = outputs.logits.mean(dim=1)
        return F.normalize(features, dim=1)

class ALIGNModelWrapper(BaseVisionLanguageModel):
    """Wrapper for ALIGN models."""
    
    def _load_model(self):
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # ALIGN models have different output structures
            if hasattr(outputs, 'text_embeds'):
                features = outputs.text_embeds
            elif hasattr(outputs, 'pooler_output'):
                features = outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state.mean(dim=1)
            else:
                # Fallback to logits
                features = outputs.logits.mean(dim=1)
        return F.normalize(features, dim=1)

class OpenCLIPModelWrapper(BaseVisionLanguageModel):
    """Wrapper for OpenCLIP models."""
    
    def _load_model(self):
        import open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model = self.model.to(self.device)
    
    def encode_text(self, texts):
        with torch.no_grad():
            text_tokens = self.tokenizer(texts).to(self.device)
            features = self.model.encode_text(text_tokens)
        return F.normalize(features, dim=1)

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
    """
    
    # Model configurations with fallbacks
    MODEL_CONFIGS = {
        'clip': {
            'class': CLIPModelWrapper,
            'name': 'openai/clip-vit-base-patch32'
        },
        'coca': {
            'class': CoCaModelWrapper,
            'name': 'microsoft/git-base-coco'  # GIT model (similar to CoCa)
        },
        'flava': {
            'class': FLAVAModelWrapper,
            'name': 'facebook/flava-base'  # FLAVA base model
        },
        'align': {
            'class': ALIGNModelWrapper,
            'name': 'kakaobrain/align-base'  # ALIGN model from KakaoBrain
        },
        'openclip': {
            'class': OpenCLIPModelWrapper,
            'name': 'ViT-L-14'  # OpenCLIP model name
        }
    }
    
    # Alternative model configurations for better compatibility
    ALTERNATIVE_MODELS = {
        'coca': [
            'microsoft/git-base-coco',
            'microsoft/git-base-textcaps',
            'microsoft/git-large-coco'
        ],
        'flava': [
            'facebook/flava-base',
            'facebook/flava-large',
            'facebook/flava-full'
        ],
        'align': [
            'kakaobrain/align-base',
            'google/align-base',
            'google/align-large'
        ]
    }
    
    def __init__(self, model_type='clip', batch_size=32, device=None):
        """
        Initialize baseline tester with specified model.
        
        Args:
            model_type: One of 'clip', 'coca', 'flava', 'align', 'openclip'
            batch_size: Batch size for processing
            device: Device to run on (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_type = model_type
        
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.MODEL_CONFIGS.keys())}")
        
        # Load model wrapper with fallback support
        config = self.MODEL_CONFIGS[model_type]
        self.model_wrapper = self._load_model_with_fallback(config, model_type)
        
        # Create embedding extractor and evaluator
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
        
        for model_type in self.MODEL_CONFIGS.keys():
            print(f"\n{'='*50}")
            print(f"Testing {model_type.upper()}")
            print(f"{'='*50}")
            
            try:
                # Create new tester for each model
                tester = BaselineTester(model_type, self.batch_size, self.device)
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

    def _load_model_with_fallback(self, config, model_type):
        """Load model with fallback to alternative models if primary fails."""
        primary_name = config['name']
        try:
            print(f"Loading {model_type.upper()} model: {primary_name}")
            return config['class'](primary_name, self.device)
        except Exception as e:
            print(f"Failed to load {primary_name}: {str(e)[:100]}...")
            # Try alternative models if available
            if model_type in self.ALTERNATIVE_MODELS:
                for alt_name in self.ALTERNATIVE_MODELS[model_type]:
                    if alt_name != primary_name:  # Skip the one we already tried
                        try:
                            print(f"Trying alternative {model_type.upper()} model: {alt_name}")
                            return config['class'](alt_name, self.device)
                        except Exception as alt_e:
                            print(f"Failed to load {alt_name}: {str(alt_e)[:100]}...")
                            continue
            # If all alternatives fail, raise the original error
            raise e 