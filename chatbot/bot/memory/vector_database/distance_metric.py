import math
from enum import Enum


class DistanceMetric(Enum):
    L2 = "l2"
    IP = "ip"
    COSINE = "cosine"


def cosine_relevance_score_fn(distance: float) -> float:
    """
    Computes the relevance score for cosine similarity.
    """
    return 1.0 - distance

def euclidean_relevance_score_fn(distance: float) -> float:
    """
    Computes the relevance score for Euclidean distance.
    """
    return 1.0 - distance / math.sqrt(2)

def max_inner_product_relevance_score_fn(distance: float) -> float:
    """
    Computes the relevance score for max inner product.
    """
    if distance > 0.0:
        return 1.0 - distance
    
    return -1.0 * distance


SUPPORTED_RELEVANCE_SCORE_FUNCTIONS = {
    DistanceMetric.COSINE: cosine_relevance_score_fn,
    DistanceMetric.L2: euclidean_relevance_score_fn,
    DistanceMetric.IP: max_inner_product_relevance_score_fn,
}

def get_relevance_score_fn(distance_metric: DistanceMetric):
    func = SUPPORTED_RELEVANCE_SCORE_FUNCTIONS.get(distance_metric)

    if func is None:
        raise KeyError(f"No supported normalization function for distance metric of type: {distance_metric}.")

    return func