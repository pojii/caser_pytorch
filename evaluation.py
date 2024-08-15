import numpy as np
import torch

def _compute_apk(actual, predicted, k=10):
    """
    Compute the average precision at k.
    This function computes the average precision at k between two lists of items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def _compute_precision_recall(targets, predictions, k):
    """
    Helper function to compute precision and recall at k.
    """
    predictions_at_k = predictions[:k]
    num_hits = sum((p in targets) for p in predictions_at_k)

    precision = num_hits / k
    recall = num_hits / len(targets)

    return precision, recall

def _compute_mrr(targets, predictions, k):
    """
    Compute the Mean Reciprocal Rank at k.
    """
    predictions_at_k = predictions[:k]
    for i, p in enumerate(predictions_at_k):
        if p in targets:
            return 1.0 / (i + 1.0)
    return 0.0

def _compute_hit_rate(targets, predictions, k):
    """
    Compute the Hit Rate at k.
    """
    predictions_at_k = predictions[:k]
    return 1.0 if any(p in targets for p in predictions_at_k) else 0.0

def evaluate_ranking(model, test, train=None, k=10):
    """
    Compute Precision@k, Recall@k scores and average precision (AP).
    One score is given for every user with interactions in the test
    set, representing the AP, Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------
    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, rated items in
        interactions will be excluded.
    k: int or array of int,
        The maximum number of predicted items
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    test_users = list(range(test.shape[0]))

    if not isinstance(k, list):
        ks = [k]
    else:
        ks = k

    precisions = [list() for _ in range(len(ks))]
    recalls = [list() for _ in range(len(ks))]
    apks = list()
    mrrs = []
    hits = []

    for user_id in test_users:
        if user_id >= model.test_sequence.sequences.shape[0]:
            print(f"Skipping user_id {user_id} as it's out of bounds for test_sequence")
            continue
        
        if train is not None and user_id >= train.shape[0]:
            print(f"Skipping user_id {user_id} as it's out of bounds for train data")
            continue

        row = test[user_id]

        if not len(row.indices):
            continue

        try:
            predictions = -model.predict(user_id)
        except IndexError:
            print(f"Error predicting for user_id {user_id}. Skipping.")
            continue
        except RuntimeError as e:
            print(f"Runtime error during prediction for user_id {user_id}: {e}")
            continue

        if np.all(predictions == 0):
            continue

        predictions = predictions.argsort()

        if train is not None:
            rated = set(train[user_id].indices)
        else:
            rated = set()

        predictions = [p for p in predictions if p not in rated]

        targets = row.indices

        for i, _k in enumerate(ks):
            precision, recall = _compute_precision_recall(targets, predictions, _k)
            precisions[i].append(precision)
            recalls[i].append(recall)

            if _k == 10:  # Compute MRR@10 and Hit@10 for k=10
                mrr = _compute_mrr(targets, predictions, _k)
                hit = _compute_hit_rate(targets, predictions, _k)
                mrrs.append(mrr)
                hits.append(hit)

        apks.append(_compute_apk(targets, predictions, k=np.inf))

    precisions = [np.array(i) for i in precisions]
    recalls = [np.array(i) for i in recalls]

    if not isinstance(k, list):
        precisions = precisions[0]
        recalls = recalls[0]

    mean_aps = np.mean(apks) if apks else 0.0
    mean_mrr = np.mean(mrrs) if mrrs else 0.0
    mean_hit = np.mean(hits) if hits else 0.0

    print(f"Processed {len(apks)} valid users out of {test.shape[0]} total users")

    return precisions, recalls, mean_aps, mean_mrr, mean_hit

