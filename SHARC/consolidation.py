import numpy as np
from scipy.stats import entropy

def lowest_entropy_consolidation(probabilities, threshold=None, label_lut=None, return_entropies=False, return_selected_classifiers=False):
    """ For each sample use the classification of the classifier with lowest entropy in the distribution of class labels.

    Parameters
    ----------
    probabilities : array-like, shape (n_classifiers, n_samples, n_classes)
        The probabilities predicted for each class by each classifier.
    threshold : float, default=None
        The entropy threshold. Samples with a post-consolidation entropy *above* this threshold with be classified as an outlier. 
        If `None` no thresholding will be applied.
    label_lut : array-like, shape (n_classes,)
        Lookup table for the labels.
    
    Returns
    -------
    labels : array-like, shape (n_samples,)
        The consolidated labels.
    entropies: array-like, shape (n_classifiers, n_samples)
        The computed entropy in the probabilities predicted for each class by each classifier. Only returned if `return_entropies=True`.
    selected_classification: array-like, shape (n_samples)
        An array consisting of indices corresponding to the classifiers that were used in the final classification of each sample.
        Only returned if `return_selected_classifiers=True` 
    
    """

    # compute entropy
    entropies = np.empty(probabilities.shape[:2])
    for i, probs in enumerate(probabilities):
        entropies[i] = entropy(probs, axis=-1)

    # for each sample find the classification that gave the lowest entropy
    selected_classification = np.argmin(entropies, axis=0)

    # use the selected classifications to make label predictions
    labels = np.empty(probabilities.shape[1], dtype=np.int64)
    for i, j in enumerate(selected_classification):
        if threshold is not None:
            # check for outliers
            if entropies[j,i] > threshold:
                labels[i] = -1
                continue
        labels[i] = np.argmax(probabilities[j,i])

        if label_lut is not None:
            labels[i] = label_lut[labels[i]]

    out  = [labels]
    if return_entropies:
        out += [entropies]
    if return_selected_classifiers:
        out += [selected_classification]
    return out

def alternative_consolidation(predictions):
    """ When the predictions by the different classifiers are in disagreement the sample is assigned to the post-consolidation outlier class.

    Parameters
    ----------
    predictions : array-like, shape (n_classifiers, n_samples)
        The predictions given by each classifier.
    
    Returns
    -------
    labels : array-like, shape (n_samples,)
        The consolidated labels.
    
    """

    labels = np.empty(predictions.shape[1], dtype=np.int64)
    for i, preds in enumerate(predictions.T):
        labels[i] = preds[0]
        for c in preds[1:]:
            if c != labels[i]:
                labels[i] = -1
                break
        
    return labels

def majority_vote_consolidation(predictions):
    """ Consolidation is done through a majority vote. When the vote is indecisive the sample is classified as an outlier.

    Parameters
    ----------
    predictions : array-like, shape (n_classifiers, n_samples)
        The predictions given by each classifier.
    
    Returns
    -------
    labels : array-like, shape (n_samples,)
        The consolidated labels.
    
    """

    labels = np.empty(predictions.shape[1], dtype=np.int64)

    # collect votes
    tmp_hist = np.zeros(np.max(predictions) + 1)
    for i, pred in enumerate(predictions.T):
        for c in pred:
            tmp_hist[int(c)] += 1

        max_idx = np.argwhere(tmp_hist == np.amax(tmp_hist)).ravel()
        if len(max_idx) > 1:
            labels[i] = -1
        else:
            labels[i] = max_idx[0]

        tmp_hist.fill(0)
        
    return labels

def average_probability_consolidation(probabilities, threshold=None, label_lut=None):
    """ Consolidation is done by averaging the probabilities for each class returned by each classifier.
    Samples are labelled by the class with the highest average probability.

    Parameters
    ----------
    probabilities : array-like, shape (n_classifiers, n_samples, n_classes)
        The probabilities predicted for each class by each classifier.
    threshold : float, default=None (optional)
        Optional probability threshold. Whenever, the highest average probability falls below the given threshold value the sample is classified as an outlier.
    label_lut : array-like, shape (n_classes,)
        Lookup table for the labels.
    
    Returns
    -------
    labels : array-like, shape (n_samples,)
        The consolidated labels.
    
    """

    # average probabilities
    probabilities = np.mean(probabilities, axis=0)
    labels = np.empty(probabilities.shape[0], dtype=np.int64)
    for i, probs in enumerate(probabilities):
        labels[i] = np.argmax(probs)
        if threshold is not None:
            if probs[int(labels[i])] < threshold:
                labels[i] = -1
                continue

        if label_lut is not None:
            labels[i] = label_lut[labels[i]]

    return labels

def multiplied_probability_consolidation(probabilities, threshold=None, label_lut=None):
    """ Consolidation is done by multiplying the probabilities for each class returned by each classifier.
    Samples are labelled by the class with the highest multiplied probability.

    Parameters
    ----------
    probabilities : array-like, shape (n_classifiers, n_samples, n_classes)
        The probabilities predicted for each class by each classifier.
    threshold : float, default=None (optional)
        Optional probability threshold. Whenever, the highest multiplied and normalized probability falls below the given threshold value the sample is classified as an outlier.
    label_lut : array-like, shape (n_classes,)
        Lookup table for the labels.
    
    Returns
    -------
    labels : array-like, shape (n_samples,)
        The consolidated labels.
    
    """

    probabilities = np.prod(probabilities, axis=0)
    labels = np.empty(probabilities.shape[0], dtype=np.int64)
    for i, probs in enumerate(probabilities):
        if np.sum(probs) == 0:
            labels[i] = -1
            continue

        probs /= np.sum(probs) # normalization such that probabilities add up to 1
        labels[i] = np.argmax(probs)
        
        if threshold is not None:
            if probs[int(labels[i])] < threshold:
                labels[i] = -1
                continue

        if label_lut is not None:
            labels[i] = label_lut[labels[i]]

    return labels