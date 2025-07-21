"""
Model utilities for vision-language models.
This package provides modular wrappers for different vision-language models
and siamese learning architectures that can work with any of them.
"""

# Import model wrappers
from .wrappers import (
    CLIPModelWrapper,
    CoCaModelWrapper,
    FLAVAModelWrapper,
    SigLIPModelWrapper
)

# Import model factory
from .model_factory import ModelFactory

# Import base and learning models
from .base import BaseSiameseModel
from .learning import (
    SiameseModelInfoNCE,
    SiameseModelPairs,
    SiameseModelTriplet,
    SiameseModelSupCon
)

# Convenience function to create models
def create_model(model_type, model_name=None, device=None):
    """
    Convenience function to create a model wrapper.
    
    Args:
        model_type: One of 'clip', 'coca', 'flava', 'siglip'
        model_name: Specific model name (optional)
        device: Device to run on (auto-detected if None)
        
    Returns:
        Model wrapper instance
    """
    return ModelFactory.create_model(model_type, model_name, device)

def create_siamese_model(model_type, learning_type='pairs', embedding_dim=None, 
                        projection_dim=128, model_name=None, device=None):
    """
    Convenience function to create a siamese model with specified backbone.
    
    Args:
        model_type: Backbone model type ('clip', 'coca', etc.)
        learning_type: One of 'pairs', 'triplet', 'infonce', 'supcon'
        embedding_dim: Embedding dimension (auto-detected if None)
        projection_dim: Projection dimension
        model_name: Specific model name (optional)
        device: Device to run on (auto-detected if None)
        
    Returns:
        Siamese model instance
    """
    # Create backbone
    backbone = ModelFactory.create_model(model_type, model_name, device)
    
    # Auto-detect embedding dimension if not provided
    if embedding_dim is None:
        embedding_dim = backbone.embedding_dim
    
    # Create appropriate siamese model
    if learning_type == 'pairs':
        return SiameseModelPairs(embedding_dim, projection_dim, backbone)
    elif learning_type == 'triplet':
        return SiameseModelTriplet(embedding_dim, projection_dim, backbone)
    elif learning_type == 'infonce':
        return SiameseModelInfoNCE(embedding_dim, projection_dim, backbone)
    elif learning_type == 'supcon':
        return SiameseModelSupCon(embedding_dim, projection_dim, backbone)
    else:
        raise ValueError(f"Unknown learning type: {learning_type}. "
                       f"Supported types: pairs, triplet, infonce, supcon")

# Export all classes
__all__ = [
    # Model wrappers
    'CLIPModelWrapper',
    'CoCaModelWrapper', 
    'FLAVAModelWrapper',
    'SigLIPModelWrapper',
    
    # Factory
    'ModelFactory',
    
    # Base and learning models
    'BaseSiameseModel',
    'SiameseModelInfoNCE',
    'SiameseModelPairs',
    'SiameseModelTriplet',
    'SiameseModelSupCon',
    
    # Convenience functions
    'create_model',
    'create_siamese_model',
] 