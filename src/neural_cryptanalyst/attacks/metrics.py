import numpy as np
from typing import List, Union

def calculate_guessing_entropy(
    predictions: np.ndarray, correct_key: int, num_traces_list: List[int]
) -> np.ndarray:
    if predictions.ndim == 2:
        num_classes = predictions.shape[1]
        max_traces = min(max(num_traces_list), predictions.shape[0])
        result = np.zeros(len(num_traces_list))

        accumulated_probabilities = np.zeros((num_classes,))

        for trace_idx in range(max_traces):
            accumulated_probabilities += np.log(predictions[trace_idx] + 1e-36)

            if trace_idx + 1 in num_traces_list:
                sorted_probs = np.argsort(accumulated_probabilities)[::-1]
                key_rank = np.where(sorted_probs == correct_key)[0][0]
                result[num_traces_list.index(trace_idx + 1)] = key_rank

        return result

    elif predictions.ndim == 3:
        num_experiments, n_traces, num_classes = predictions.shape
        max_traces = min(max(num_traces_list), n_traces)
        result = np.zeros(len(num_traces_list))

        for exp_idx in range(num_experiments):
            accumulated_probabilities = np.zeros((num_classes,))

            for trace_idx in range(max_traces):
                accumulated_probabilities += np.log(predictions[exp_idx, trace_idx] + 1e-36)

                if trace_idx + 1 in num_traces_list:
                    sorted_probs = np.argsort(accumulated_probabilities)[::-1]
                    key_rank = np.where(sorted_probs == correct_key)[0][0]
                    result[num_traces_list.index(trace_idx + 1)] += key_rank

        result /= num_experiments
        return result

    else:
        return np.ones(len(num_traces_list)) * 128

def calculate_success_rate(
    predictions: np.ndarray, correct_key: int,
    num_traces_list: List[int], rank_threshold: int = 1
) -> np.ndarray:
    if predictions.ndim == 2:
        predictions = predictions[np.newaxis, :, :]

    num_experiments, n_traces, num_classes = predictions.shape
    max_traces = min(max(num_traces_list), n_traces)
    result = np.zeros(len(num_traces_list))

    for exp_idx in range(num_experiments):
        accumulated_probabilities = np.zeros((num_classes,))

        for trace_idx in range(max_traces):
            accumulated_probabilities += np.log(predictions[exp_idx, trace_idx] + 1e-36)

            if trace_idx + 1 in num_traces_list:
                sorted_probs = np.argsort(accumulated_probabilities)[::-1]
                key_rank = np.where(sorted_probs == correct_key)[0][0]

                if key_rank < rank_threshold:
                    result[num_traces_list.index(trace_idx + 1)] += 1

    result /= num_experiments
    return result

def calculate_mutual_information_analysis(traces: np.ndarray,
                                          predictions: np.ndarray,
                                          correct_key: int) -> float:
    from sklearn.metrics import mutual_info_score

    correct_probs = predictions[:, correct_key]

    n_bins = 10
    trace_means = traces.mean(axis=1)
    trace_bins = np.digitize(trace_means,
                             np.histogram(trace_means, bins=n_bins)[1])
    prob_bins = np.digitize(correct_probs,
                            np.histogram(correct_probs, bins=n_bins)[1])

    return mutual_info_score(trace_bins, prob_bins)

def key_rank(predictions: np.ndarray, correct_key: int) -> int:
    if predictions.ndim == 2:
        log_probs = np.log(predictions + 1e-36)
        aggregated = np.sum(log_probs, axis=0)
    else:
        aggregated = predictions

    sorted_indices = np.argsort(aggregated)[::-1]
    return int(np.where(sorted_indices == correct_key)[0][0])
