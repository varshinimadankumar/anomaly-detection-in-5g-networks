import numpy as np

def fuse_scores(scores):
    """
    scores: dict { 'ae': arr, 'ocsvm': arr, 'rf': arr }
    """
    arrs = np.vstack(list(scores.values()))
    fused = arrs.mean(axis=0)
    return fused
