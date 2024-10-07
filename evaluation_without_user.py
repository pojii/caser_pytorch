import numpy as np
import torch

def _compute_apk(actual, predicted, k=10):
    """
    Compute Average Precision at k.
    """
    if len(actual) == 0:
        return 0.0

    if k is None or k == np.inf:
        k = len(predicted)
    else:
        k = min(k, len(predicted))

    actual = set(actual)  # Convert to set for faster lookup
    predicted = predicted[:k]  # Only consider top k predictions

    score = 0.0
    num_hits = 0.0
    # print('actual',actual)
    for i, p in enumerate(predicted):
        if p in actual:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def _compute_precision_recall(actual, predicted, k):
    """
    Compute precision and recall at k.
    """
    if isinstance(actual, torch.Tensor):
        actual = actual.cpu().numpy()
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.cpu().numpy()
    
    actual = np.atleast_1d(actual)
    predicted = np.atleast_1d(predicted)

    predictions_at_k = predicted[:k]
    num_hits = sum(np.isin(predictions_at_k, actual))

    precision = num_hits / k if k > 0 else 0.0
    recall = num_hits / len(actual) if len(actual) > 0 else 0.0

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

def _compute_hit_rate(actual, predicted, k):
    """
    Compute Hit Rate at k.
    """
    actual_set = set(actual)
    # print('predicted',predicted[:k])
    predicted_set = set(predicted[:k])
    return float(len(actual_set & predicted_set) > 0)

def evaluate_ranking(model, test, train=None, k=[10]):
    if not isinstance(k, list):
        k = [k]

    precisions = [[] for _ in range(len(k))]
    recalls = [[] for _ in range(len(k))]
    hit_rates = [[] for _ in range(len(k))]
    apks = []
    mrrs = []

    user_metrics = {}

    for user_id in range(len(test.sequences.sequences)):
        sequence = test.sequences.sequences[user_id]
        if np.all(sequence == 0):
            continue

        if hasattr(test.sequences, 'targets') and user_id < len(test.sequences.targets):
            actual = test.sequences.targets[user_id]
            actual = actual[actual != 0]  # Remove padding
        else:
            actual = sequence[sequence != 0][-1:]

        if len(actual) == 0:
            continue

        # Clip sequence values to be within the valid range
        sequence = np.clip(sequence, 0, model._num_items - 1)

        try:
            if len(sequence.shape) == 1:
                sequence = sequence.reshape(1, -1)
            predictions = model.predict(sequence).squeeze()
            predictions = -predictions  # Invert predictions for ranking
        except RuntimeError as e:
            print(f"Runtime error during prediction for user_id {user_id}: {e}")
            continue

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        

        # Sort predictions
        top_k = np.argsort(predictions)[::-1]

        user_metrics[user_id] = {
            'predictions': {},
            'actual': actual.tolist() if isinstance(actual, np.ndarray) else actual,
            'precision': {},
            'recall': {},
            'hit_rate': {},
            'apk': 0,
            'mrr': 0
        }

        for i, _k in enumerate(k):
            precision, recall = _compute_precision_recall(actual, top_k, _k)
            hit_rate = _compute_hit_rate(actual, top_k, _k)
            
            precisions[i].append(precision)
            recalls[i].append(recall)
            hit_rates[i].append(hit_rate)

            user_metrics[user_id]['predictions'][_k] = top_k[:_k].tolist()
            user_metrics[user_id]['precision'][_k] = precision
            user_metrics[user_id]['recall'][_k] = recall
            user_metrics[user_id]['hit_rate'][_k] = hit_rate

        apk = _compute_apk(actual, top_k, k=np.inf)
        mrr = _compute_mrr(actual, top_k, k=max(k))
        
        apks.append(apk)
        mrrs.append(mrr)
        
        user_metrics[user_id]['apk'] = apk
        user_metrics[user_id]['mrr'] = mrr

    mean_precision = [np.mean(p) for p in precisions]
    mean_recall = [np.mean(r) for r in recalls]
    mean_hit_rate = [np.mean(h) for h in hit_rates]
    mean_apk = np.mean(apks)
    mean_mrr = np.mean(mrrs)

    print(f"Processed {len(user_metrics)} valid users out of {len(test.test_sequences.sequences)} total users")

    return mean_precision, mean_recall, mean_apk, mean_mrr, mean_hit_rate, user_metrics