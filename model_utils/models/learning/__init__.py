"""
Learning models for different siamese learning approaches.
These models work with any vision-language model wrapper as backbone.
"""

from .infonce import SiameseModelInfoNCE
from .siamese import SiameseModelPairs, SiameseModelTriplet
from .supcon import SiameseModelSupCon

__all__ = [
    'SiameseModelInfoNCE',
    'SiameseModelPairs',
    'SiameseModelTriplet',
    'SiameseModelSupCon',
] 