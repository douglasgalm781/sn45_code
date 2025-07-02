import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosim(model, text1: str, text2: str) -> float:
    # Load the pre-trained sentence transformer model

    # Embed the texts
    embeddings = model.encode([text1, text2])

    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return similarity


def normalize_cosim(value, min_value=0.5, max_value=1.0, exponent=1.3):
    """
    Exponentially normalize the cosine similarity value to a range of 0 to 1.

    Parameters:
    value (float): The cosine similarity value to be normalized.
    min_value (float): The minimum value of the original range. Default is 0.5.
    max_value (float): The maximum value of the original range. Default is 1.0.
    exponent (float): The exponent to be used for the normalization. Default is 1.3.

    Returns:
    float: The exponentially normalized value in the range of 0 to 1, or 0 if the result is invalid.
    """
    if min_value == max_value:
        raise ValueError("min_value and max_value must be different")

    # First normalize linearly
    linear_normalized_value = (value - min_value) / (max_value - min_value)

    # Check for invalid linear_normalized_value (e.g., NaN or out of bounds)
    if (
        np.isnan(linear_normalized_value)
        or linear_normalized_value < 0
        or linear_normalized_value > 1
    ):
        return 0

    # Then apply the exponential transformation
    exponential_normalized_value = np.power(linear_normalized_value, exponent)

    return exponential_normalized_value
