import numpy as np

def get_curriculum_ratios(epoch, total_epochs):
    """
    Return (easy, medium, hard) ratios for self-paced curriculum learning.

    Args:
        epoch (int): Current training epoch.
        total_epochs (int): Total number of training epochs.

    Returns:
        dict: A dictionary with keys 'easy', 'medium', 'hard' and corresponding float ratios.
    """
    # Normalize the current epoch to [0, 1]
    t = epoch / total_epochs

    # Use smooth cosine interpolation to define the curves
    easy = 0.5 * (1 + np.cos(np.pi * t)) * (1 - t)
    hard = 0.5 * (1 + np.cos(np.pi * (1 - t))) * t
    medium = 1.0 - (easy + hard)

    # Normalize to ensure they sum to 1
    total = easy + medium + hard
    return {
        "easy": easy / total,
        "medium": medium / total,
        "hard": hard / total
    }