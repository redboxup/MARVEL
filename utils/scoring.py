import numpy as np
import sklearn.metrics as sk

# Evaluate Scores
recall_level_default = 0.95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError(
            "cumsum was found to be unstable: "
            "its last element does not correspond to sum"
        )
    return out


def fpr_and_fdr_at_recall(
    y_true, y_score, recall_level=recall_level_default, pos_label=None
):
    classes = np.unique(y_true)
    if pos_label is None and not (
        np.array_equal(classes, [0, 1])
        or np.array_equal(classes, [-1, 1])
        or np.array_equal(classes, [0])
        or np.array_equal(classes, [-1])
        or np.array_equal(classes, [1])
    ):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.0

    # make y_true a boolean vector
    y_true = y_true == pos_label

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = (
        np.r_[recall[sl], 1],
        np.r_[fps[sl], 0],
        np.r_[tps[sl], 0],
        thresholds[sl],
    )

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (
        np.sum(np.logical_not(y_true))
    )  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    """
    Calculate evaluation metrics for binary classification.

    This function computes the Area Under the Receiver Operating Characteristic Curve (AUROC),
    Average Precision Score (AUPR), and the False Positive Rate (FPR) at a specified recall level
    based on the provided positive and negative examples.

    Parameters:
    ----------
    _pos : array-like
        A collection of positive examples (true positives). This can be a list or a NumPy array
        containing scores or probabilities that represent the positive class instances.

    _neg : array-like
        A collection of negative examples (true negatives). This can also be a list or a NumPy array
        containing scores or probabilities that represent the negative class instances.

    recall_level : float, optional
        The desired recall level at which to compute the False Positive Rate (FPR).
        Defaults to a predefined value (`recall_level_default`).

    Returns:
    -------
    tuple
        A tuple containing three metrics:
        - AUROC : float
            The Area Under the Receiver Operating Characteristic Curve.
        - AUPR : float
            The Average Precision Score.
        - FPR : float
            The False Positive Rate at the specified recall level.

    Example:
    --------
    >>> import numpy as np

    >>> # Example positive and negative scores
    >>> _pos = [0.9, 0.85, 0.95]  # Positive class scores
    >>> _neg = [0.1, 0.2, 0.15]   # Negative class scores

    >>> # Call the function to get measures
    >>> auroc, aupr, fpr = get_measures(_pos, _neg, recall_level=0.8)

    >>> # Output the results
    >>> print(f"AUROC: {auroc}, AUPR: {aupr}, FPR at Recall Level: {fpr}")

    Notes:
    ------
    Ensure that the input collections for positive and negative examples are of appropriate length,
    as they will be combined to compute the metrics. The function expects binary classification data.
    If the inputs do not conform to expected formats or dimensions, it may raise errors.
    """
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[: len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return dict(auroc=auroc, aupr=aupr, fpr95=fpr)
