from .wrappers import (
    CLIPModelWrapper,
    CoCaModelWrapper,
    FLAVAModelWrapper,
    SigLIPModelWrapper,
)

class ModelFactory:
    """Factory class for creating different model wrappers."""
    
    MODEL_CONFIGS = {
        'clip': {
            'class': CLIPModelWrapper,
            'default_name': 'openai/clip-vit-base-patch32'
        },
        'coca': {
            'class': CoCaModelWrapper,
            'default_name': 'microsoft/git-base-coco'
        },
        'flava': {
            'class': FLAVAModelWrapper,
            'default_name': 'facebook/flava-full'
        },
        'siglip': {
            'class': SigLIPModelWrapper,
            'default_name': 'google/siglip-base-patch16-224'
        }
    }
    
    @classmethod
    def create_model(cls, model_type, model_name=None, device=None):
        """
        Create a model wrapper of the specified type.
        
        Args:
            model_type: One of 'clip', 'coca', 'flava', 'siglip'
            model_name: Specific model name (optional, uses default if not provided)
            device: Device to run on (auto-detected if None)
            
        Returns:
            Model wrapper instance
        """
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {list(cls.MODEL_CONFIGS.keys())}")
        
        config = cls.MODEL_CONFIGS[model_type]
        model_class = config['class']
        default_name = config['default_name']
        
        # Use provided model_name or default
        model_name = model_name or default_name
        
        try:
            return model_class(model_name, device)
        except ImportError as e:
            # Provide helpful error message for missing dependencies
            if model_type == 'siglip':
                raise ImportError(
                    f"Failed to create {model_type} model: {str(e)}\n"
                    "Please install required dependencies: pip install sentencepiece==0.2.0"
                )
            else:
                raise ImportError(f"Failed to create {model_type} model: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Failed to create {model_type} model: {str(e)}")
    
    @classmethod
    def get_available_models(cls):
        """Get list of available model types."""
        return list(cls.MODEL_CONFIGS.keys())
    
    @classmethod
    def get_default_model_name(cls, model_type):
        """Get the default model name for a given model type."""
        if model_type not in cls.MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}")
        return cls.MODEL_CONFIGS[model_type]['default_name'] 