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
            elif hasattr(outputs, 'text_outputs') and hasattr(outputs.text_outputs, 'last_hidden_state'):
                features = outputs.text_outputs.last_hidden_state.mean(dim=1)
            else:
                # Try to get any available tensor output
                for attr in dir(outputs):
                    if not attr.startswith('_') and hasattr(getattr(outputs, attr), 'shape'):
                        tensor = getattr(outputs, attr)
                        if len(tensor.shape) >= 2:
                            features = tensor.mean(dim=1) if len(tensor.shape) > 2 else tensor
                            break
                else:
                    raise ValueError(f"Could not extract features from FLAVA model output: {type(outputs)}")
        return F.normalize(features, dim=1)

class SigLIPModelWrapper(BaseVisionLanguageModel):
    """Wrapper for SigLIP text-only models."""
    def _load_model(self):
        try:
            print(f"Loading SigLIP model: {self.model_name}")
            self.model = AutoModel.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                torch_dtype=torch.float32
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            print(f"Successfully loaded SigLIP model: {self.model_name}")
        except Exception as e:
            print(f"Error loading SigLIP model {self.model_name}: {str(e)}")
            raise e
    def encode_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        print("[DEBUG] About to call SigLIP model with inputs:", inputs)
        with torch.no_grad():
            try:
                outputs = self.model(**inputs)
                print("[DEBUG] SigLIP model call returned.")
                print("[DEBUG] type(outputs):", type(outputs))
                print("[DEBUG] repr(outputs):", repr(outputs))
            except Exception as e:
                print("[DEBUG] Exception during SigLIP model call:", e)
                raise
            # If outputs is a tuple or list, try each element
            if isinstance(outputs, (tuple, list)):
                for o in outputs:
                    if o is not None and hasattr(o, 'shape') and len(o.shape) >= 2:
                        return F.normalize(o, dim=1)
            # If outputs is a dict, try each value
            if isinstance(outputs, dict):
                for v in outputs.values():
                    if v is not None and hasattr(v, 'shape') and len(v.shape) >= 2:
                        return F.normalize(v, dim=1)
            # Try known fields
            for key in ['text_embeds', 'pooler_output', 'last_hidden_state', 'logits']:
                if hasattr(outputs, key):
                    value = getattr(outputs, key)
                    if value is not None:
                        if key == 'last_hidden_state':
                            features = value.mean(dim=1)
                        elif key == 'logits':
                            features = value.mean(dim=1)
                        else:
                            features = value
                        return F.normalize(features, dim=1)
            # Try any tensor attribute
            for attr in dir(outputs):
                if not attr.startswith('_'):
                    tensor = getattr(outputs, attr)
                    if tensor is not None and hasattr(tensor, 'shape') and len(tensor.shape) >= 2:
                        features = tensor.mean(dim=1) if len(tensor.shape) > 2 else tensor
                        return F.normalize(features, dim=1)
            raise ValueError(f"Could not extract features from SigLIP model output: {type(outputs)}; output: {outputs}")

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
    
    # Model configurations
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
            'name': 'facebook/flava-full'  # FLAVA full model (actually exists)
        },
        'siglip': {
            'class': SigLIPModelWrapper,
            'name': 'google/siglip-base-patch16-224'  # SigLIP base model
        },
        'openclip': {
            'class': OpenCLIPModelWrapper,
            'name': 'ViT-L-14'  # OpenCLIP model name (valid)
        }
    }
    

    
    def __init__(self, model_type='clip', batch_size=32, device=None):
        """
        Initialize baseline tester with specified model.
        
        Args:
            model_type: One of 'clip', 'coca', 'flava', 'siglip', 'openclip'
            batch_size: Batch size for processing
            device: Device to run on (auto-detected if None)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model_type = model_type

        print(f"[DEBUG] Requested model_type: {model_type}")

        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[model_type]
        if model_type == 'siglip':
            siglip_candidates = [
                config['name'],
                'google/siglip-base-patch16-384',
                'google/siglip-large-patch16-224',
                'google/siglip-large-patch16-384',
                'google/siglip-so400m-patch14-224',
                'google/siglip-so400m-patch14-384',
            ]
            last_error = None
            for candidate in siglip_candidates:
                print(f"[DEBUG] Attempting to load SigLIP model: {candidate}")
                try:
                    self.model_wrapper = config['class'](candidate, self.device)
                    print(f"[DEBUG] Successfully loaded SigLIP model: {candidate}")
                    break
                except Exception as e:
                    print(f"[ERROR] Failed to load SigLIP model {candidate}: {e}")
                    last_error = e
            else:
                print("[ERROR] All SigLIP model candidates failed to load.")
                raise last_error
        else:
            try:
                self.model_wrapper = config['class'](config['name'], self.device)
                print(f"[DEBUG] Successfully loaded model: {model_type} ({config['name']})")
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
        
        for model_type in self.MODEL_CONFIGS.keys():
            print(f"\n{'='*50}")
            print(f"Testing {model_type.upper()}")
            print(f"{'='*50}")
            print(f"[DEBUG] Attempting to load and test model_type: {model_type}")
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

 