import numpy as np
from typing import Optional, List, Tuple, Dict
from sklearn.cluster import KMeans
from ..preprocessing import TracePreprocessor, FeatureSelector

class NonProfiledAttack:
    def __init__(self, preprocessor: Optional[TracePreprocessor] = None, feature_selector: Optional[FeatureSelector] = None):
        self.preprocessor = preprocessor or TracePreprocessor()
        self.feature_selector = feature_selector or FeatureSelector()
        self.key_hypotheses = list(range(256))

    def dpa_attack(self, traces: np.ndarray, plaintexts: np.ndarray, target_byte: int = 0, leakage_model: str = 'hw') -> np.ndarray:
        n_traces, n_samples = traces.shape
        correlations = np.zeros((256, n_samples))
        for key_guess in self.key_hypotheses:
            intermediate_values = np.zeros(n_traces)
            for i in range(n_traces):
                from ..utils.crypto import aes_sbox
                if plaintexts.ndim == 1:
                    sbox_out = aes_sbox(plaintexts[i] ^ key_guess)
                else:
                    sbox_out = aes_sbox(plaintexts[i, target_byte] ^ key_guess)
                if leakage_model == 'hw':
                    intermediate_values[i] = bin(sbox_out).count('1')
                else:
                    raise NotImplementedError(f"Leakage model {leakage_model} not implemented")
            for j in range(n_samples):
                if np.std(traces[:, j]) > 0 and np.std(intermediate_values) > 0:
                    correlations[key_guess, j] = np.abs(np.corrcoef(traces[:, j], intermediate_values)[0, 1])
        max_correlations = np.max(correlations, axis=1)
        return max_correlations

    def cpa_attack(self, traces: np.ndarray, plaintexts: np.ndarray, num_poi: int = 5000, target_byte: int = 0) -> List[int]:
        if not self.preprocessor._fitted:
            self.preprocessor.fit(traces)
        processed = self.preprocessor.preprocess_traces(traces)
        max_correlations = self.dpa_attack(processed, plaintexts, target_byte)
        return np.argsort(max_correlations)[::-1].tolist()

    def template_matching_attack(self, traces: np.ndarray, plaintexts: np.ndarray,
                                  n_clusters: int = 9,
                                  feature_window: Optional[Tuple[int, int]] = None) -> np.ndarray:
        n_traces, n_samples = traces.shape

        if feature_window is None:
            window_size = max(10, n_samples // 10)
            start = (n_samples - window_size) // 2
            end = start + window_size
        else:
            start, end = feature_window
            start = max(0, min(start, n_samples - 1))
            end = max(start + 1, min(end, n_samples))

        features = traces[:, start:end]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(features)

        cluster_hw_map = {}
        for cluster_id in range(n_clusters):
            cluster_mask = clusters == cluster_id
            if np.any(cluster_mask):
                cluster_traces = traces[cluster_mask]
                avg_power = np.mean(cluster_traces)
                cluster_hw_map[cluster_id] = avg_power

        sorted_clusters = sorted(cluster_hw_map.items(), key=lambda x: x[1])
        hw_estimates = np.zeros(len(traces))
        for i, (cluster_id, _) in enumerate(sorted_clusters):
            hw_estimates[clusters == cluster_id] = i

        return hw_estimates

    def higher_order_dpa_attack(self, traces: np.ndarray, plaintexts: np.ndarray,
                               order: int = 2, target_byte: int = 0,
                               combination_function: str = 'product',
                               window_size: int = 100) -> Tuple[np.ndarray, Dict]:

        if order < 2 or order > 5:
            raise ValueError(f"Order must be between 2 and 5, got {order}")

        print(f"Performing {order}-order DPA attack with {combination_function} combination...")

        n_traces, n_samples = traces.shape
        correlations = np.zeros((256, n_samples - (order - 1) * window_size))

        centered_traces = traces - np.mean(traces, axis=0)

        if order == 2:
            combined_traces = self._combine_second_order(centered_traces, combination_function, window_size)
        elif order == 3:
            combined_traces = self._combine_third_order(centered_traces, combination_function, window_size)
        elif order == 4:
            combined_traces = self._combine_fourth_order(centered_traces, combination_function, window_size)
        else:
            combined_traces = self._combine_higher_order(centered_traces, order, combination_function, window_size)

        for key_guess in self.key_hypotheses:
            intermediate_values = np.zeros(n_traces)

            for i in range(n_traces):
                from ..utils.crypto import aes_sbox
                if plaintexts.ndim == 1:
                    sbox_out = aes_sbox(plaintexts[i] ^ key_guess)
                else:
                    sbox_out = aes_sbox(plaintexts[i, target_byte] ^ key_guess)

                intermediate_values[i] = bin(sbox_out).count('1')

            for j in range(combined_traces.shape[1]):
                if np.std(combined_traces[:, j]) > 0 and np.std(intermediate_values) > 0:
                    corr = np.abs(np.corrcoef(combined_traces[:, j], intermediate_values)[0, 1])
                    correlations[key_guess, j] = corr

        max_correlations = np.max(correlations, axis=1)
        best_key = np.argmax(max_correlations)

        attack_info = {
            'order': order,
            'combination_function': combination_function,
            'window_size': window_size,
            'best_key': best_key,
            'max_correlation': max_correlations[best_key],
            'combined_trace_shape': combined_traces.shape,
            'success': max_correlations[best_key] > 0.1
        }

        return correlations, attack_info

    def template_attack(self, profiling_traces: np.ndarray, profiling_labels: np.ndarray,
                       attack_traces: np.ndarray, poi_indices: Optional[np.ndarray] = None,
                       pooled_covariance: bool = True) -> np.ndarray:
        from scipy.stats import multivariate_normal

        if poi_indices is None:
            from ..preprocessing import FeatureSelector
            selector = FeatureSelector()
            poi_indices, _ = selector.select_poi_sost(profiling_traces, profiling_labels, num_poi=20)

        profiling_poi = profiling_traces[:, poi_indices]
        attack_poi = attack_traces[:, poi_indices]

        templates = {}
        unique_labels = np.unique(profiling_labels)

        if pooled_covariance:
            pooled_cov = np.zeros((len(poi_indices), len(poi_indices)))

            for label in unique_labels:
                mask = profiling_labels == label
                label_traces = profiling_poi[mask]

                if len(label_traces) > 1:
                    mean_vec = np.mean(label_traces, axis=0)
                    cov_matrix = np.cov(label_traces.T)

                    pooled_cov += cov_matrix * (len(label_traces) - 1)

                    templates[label] = {
                        'mean': mean_vec,
                        'count': len(label_traces)
                    }

            total_df = sum(t['count'] - 1 for t in templates.values())
            pooled_cov /= total_df
            pooled_cov += np.eye(len(poi_indices)) * 1e-6

            for label in templates:
                templates[label]['cov'] = pooled_cov

        else:
            for label in unique_labels:
                mask = profiling_labels == label
                label_traces = profiling_poi[mask]

                if len(label_traces) > 1:
                    mean_vec = np.mean(label_traces, axis=0)
                    cov_matrix = np.cov(label_traces.T)
                    cov_matrix += np.eye(len(poi_indices)) * 1e-6

                    templates[label] = {
                        'mean': mean_vec,
                        'cov': cov_matrix,
                        'count': len(label_traces)
                    }

        n_attack = len(attack_traces)
        n_labels = len(templates)
        log_probs = np.zeros((n_attack, n_labels))

        for i, trace in enumerate(attack_poi):
            for j, (label, template) in enumerate(templates.items()):
                try:
                    rv = multivariate_normal(template['mean'], template['cov'])
                    log_probs[i, j] = rv.logpdf(trace)
                except (np.linalg.LinAlgError, ValueError):
                    diff = trace - template['mean']
                    inv_cov = np.linalg.pinv(template['cov'])
                    log_probs[i, j] = -0.5 * diff @ inv_cov @ diff

        return log_probs

    def template_attack_complete(self, profiling_traces: np.ndarray, profiling_labels: np.ndarray,
                                attack_traces: np.ndarray, poi_indices: Optional[np.ndarray] = None,
                                pooled_covariance: bool = True,
                                covariance_regularization: float = 1e-6) -> Tuple[np.ndarray, Dict]:

        from scipy.stats import multivariate_normal
        from scipy.linalg import inv, det, LinAlgError

        print("Building multivariate Gaussian templates...")

        if poi_indices is None:
            from ..preprocessing import FeatureSelector
            selector = FeatureSelector()
            poi_indices, _ = selector.select_poi_sost(profiling_traces, profiling_labels, num_poi=50)
            print(f"Automatically selected {len(poi_indices)} POIs")

        profiling_poi = profiling_traces[:, poi_indices]
        attack_poi = attack_traces[:, poi_indices]
        n_poi = len(poi_indices)

        templates = {}
        unique_labels = np.unique(profiling_labels)

        min_samples_required = n_poi + 1
        insufficient_data_classes = []

        for label in unique_labels:
            mask = profiling_labels == label
            n_samples = np.sum(mask)
            if n_samples < min_samples_required:
                insufficient_data_classes.append((label, n_samples))

        if insufficient_data_classes:
            print(f"Warning: {len(insufficient_data_classes)} classes have insufficient data")

        if pooled_covariance:
            print("Computing pooled covariance matrix...")
            pooled_cov = np.zeros((n_poi, n_poi))
            total_df = 0

            for label in unique_labels:
                mask = profiling_labels == label
                label_traces = profiling_poi[mask]

                if len(label_traces) > 1:
                    mean_vec = np.mean(label_traces, axis=0)
                    centered = label_traces - mean_vec
                    cov_matrix = (centered.T @ centered) / (len(label_traces) - 1)

                    pooled_cov += cov_matrix * (len(label_traces) - 1)
                    total_df += len(label_traces) - 1

                    templates[label] = {
                        'mean': mean_vec,
                        'n_samples': len(label_traces)
                    }

            if total_df > 0:
                pooled_cov /= total_df

            pooled_cov += np.eye(n_poi) * covariance_regularization

            try:
                pooled_cov_inv = inv(pooled_cov)
                pooled_cov_det = det(pooled_cov)
                for label in templates:
                    templates[label]['cov_inv'] = pooled_cov_inv
                    templates[label]['cov_det'] = pooled_cov_det
            except LinAlgError:
                print("Warning: Covariance matrix is singular, using pseudo-inverse")
                pooled_cov_inv = np.linalg.pinv(pooled_cov)
                pooled_cov_det = 1e-100
                for label in templates:
                    templates[label]['cov_inv'] = pooled_cov_inv
                    templates[label]['cov_det'] = pooled_cov_det

        else:
            print("Computing individual covariance matrices...")
            for label in unique_labels:
                mask = profiling_labels == label
                label_traces = profiling_poi[mask]

                if len(label_traces) > n_poi:
                    mean_vec = np.mean(label_traces, axis=0)
                    centered = label_traces - mean_vec
                    cov_matrix = (centered.T @ centered) / (len(label_traces) - 1)
                    cov_matrix += np.eye(n_poi) * covariance_regularization

                    try:
                        cov_inv = inv(cov_matrix)
                        cov_det = det(cov_matrix)
                    except LinAlgError:
                        cov_inv = np.linalg.pinv(cov_matrix)
                        cov_det = 1e-100

                    templates[label] = {
                        'mean': mean_vec,
                        'cov_inv': cov_inv,
                        'cov_det': cov_det,
                        'n_samples': len(label_traces)
                    }

        print("Computing attack probabilities...")
        n_attack = len(attack_traces)
        log_probs = np.full((n_attack, 256), -np.inf)

        log_2pi_term = -0.5 * n_poi * np.log(2 * np.pi)

        for i in range(n_attack):
            trace = attack_poi[i]
            for label, template in templates.items():
                diff = trace - template['mean']
                mahalanobis_sq = diff @ template['cov_inv'] @ diff
                log_prob = log_2pi_term - 0.5 * np.log(template['cov_det']) - 0.5 * mahalanobis_sq
                log_probs[i, label] = log_prob

        max_log_probs = np.max(log_probs, axis=1, keepdims=True)
        log_probs_normalized = log_probs - max_log_probs
        probs = np.exp(log_probs_normalized)
        probs /= np.sum(probs, axis=1, keepdims=True)

        predicted_keys = np.argmax(log_probs, axis=1)
        confidence_scores = np.max(probs, axis=1)

        attack_info = {
            'n_templates': len(templates),
            'n_poi': n_poi,
            'poi_indices': poi_indices,
            'pooled_covariance': pooled_covariance,
            'predicted_keys': predicted_keys,
            'average_confidence': np.mean(confidence_scores),
            'max_confidence': np.max(confidence_scores),
            'template_quality': {
                label: template['n_samples']
                for label, template in templates.items()
            }
        }

        return log_probs, attack_info

    def _combine_second_order(self, traces: np.ndarray, combination: str, window: int) -> np.ndarray:
        n_traces, n_samples = traces.shape
        n_combined = n_samples - window
        combined = np.zeros((n_traces, n_combined))

        if combination == 'product':
            for shift in range(1, window + 1):
                combined += traces[:, :-shift] * traces[:, shift:]
            combined /= window
        elif combination == 'abs_diff':
            for shift in range(1, window + 1):
                combined += np.abs(traces[:, :-shift] - traces[:, shift:])
            combined /= window
        elif combination == 'normalized_product':
            for i in range(n_traces):
                trace_std = np.std(traces[i])
                if trace_std > 0:
                    normalized = traces[i] / trace_std
                    for shift in range(1, window + 1):
                        combined[i] += normalized[:-shift] * normalized[shift:]
            combined /= window
        else:
            raise ValueError(f"Unknown combination function: {combination}")

        return combined

    def _combine_third_order(self, traces: np.ndarray, combination: str, window: int) -> np.ndarray:
        n_traces, n_samples = traces.shape
        n_combined = n_samples - 2 * window
        combined = np.zeros((n_traces, n_combined))

        if combination == 'product':
            shifts1 = np.linspace(1, window, min(10, window), dtype=int)
            shifts2 = np.linspace(1, window, min(10, window), dtype=int)

            for s1 in shifts1:
                for s2 in shifts2:
                    if s1 < s2:
                        combined += (traces[:, :-s2] *
                                   traces[:, s1:-s2+s1] *
                                   traces[:, s2:])
            combined /= (len(shifts1) * len(shifts2) / 2)
        elif combination == 'central_moment':
            for i in range(n_traces):
                for j in range(n_combined):
                    window_data = traces[i, j:j+3*window]
                    if len(window_data) >= 3:
                        mean = np.mean(window_data)
                        combined[i, j] = np.mean((window_data - mean)**3)
        else:
            return self._combine_third_order(traces, 'product', window)

        return combined

    def _combine_fourth_order(self, traces: np.ndarray, combination: str, window: int) -> np.ndarray:
        n_traces, n_samples = traces.shape
        n_combined = n_samples - 3 * window
        combined = np.zeros((n_traces, n_combined))

        if combination == 'product':
            sample_shifts = np.random.choice(window, size=min(5, window), replace=False)

            for s1 in sample_shifts:
                for s2 in sample_shifts:
                    for s3 in sample_shifts:
                        if s1 < s2 < s3:
                            combined += (traces[:, :-s3] *
                                       traces[:, s1:-s3+s1] *
                                       traces[:, s2:-s3+s2] *
                                       traces[:, s3:])

            num_combinations = len(sample_shifts) * (len(sample_shifts)-1) * (len(sample_shifts)-2) / 6
            combined /= max(1, num_combinations)
        elif combination == 'central_moment':
            for i in range(n_traces):
                for j in range(n_combined):
                    window_data = traces[i, j:j+4*window]
                    if len(window_data) >= 4:
                        mean = np.mean(window_data)
                        combined[i, j] = np.mean((window_data - mean)**4)
        else:
            return self._combine_fourth_order(traces, 'product', window)

        return combined

    def _combine_higher_order(self, traces: np.ndarray, order: int, combination: str, window: int) -> np.ndarray:
        n_traces, n_samples = traces.shape
        n_combined = max(1, n_samples - (order - 1) * window)
        combined = np.zeros((n_traces, n_combined))

        print(f"Computing {order}-th order combinations (this may take time)...")

        if combination == 'central_moment':
            for i in range(n_traces):
                for j in range(n_combined):
                    window_data = traces[i, j:j+order*window]
                    if len(window_data) >= order:
                        mean = np.mean(window_data)
                        combined[i, j] = np.mean((window_data - mean)**order)
        elif combination == 'cumulant':
            from scipy import stats
            for i in range(n_traces):
                for j in range(n_combined):
                    window_data = traces[i, j:j+order*window]
                    if len(window_data) >= order:
                        if order == 3:
                            combined[i, j] = stats.skew(window_data)
                        elif order == 4:
                            combined[i, j] = stats.kurtosis(window_data)
                        else:
                            mean = np.mean(window_data)
                            combined[i, j] = stats.moment(window_data, order)
        else:
            return self._combine_higher_order(traces, order, 'central_moment', window)

        return combined

    def multivariate_higher_order_attack(self, traces: np.ndarray,
                                       plaintexts: np.ndarray,
                                       order: int = 2,
                                       num_variates: int = 2,
                                       target_byte: int = 0) -> Dict:
        print(f"Performing {num_variates}-variate {order}-order attack...")

        if not 2 <= order <= 4:
            raise ValueError("Order must be between 2 and 4")
        if not 2 <= num_variates <= 4:
            raise ValueError("Number of variates must be between 2 and 4")

        selector = FeatureSelector()
        labels = plaintexts[:, target_byte] if plaintexts.ndim > 1 else plaintexts

        poi_sets = []
        remaining_traces = traces.copy()
        for v in range(num_variates):
            poi_idx, _ = selector.select_poi_sost(remaining_traces, labels, num_poi=200)
            poi_sets.append(poi_idx)
            remaining_traces[:, poi_idx] = 0

        print(f"Selected {num_variates} independent POI sets")

        variate_traces = []
        for poi_idx in poi_sets:
            variate_traces.append(traces[:, poi_idx])

        n_traces = traces.shape[0]
        combined_features = np.zeros(n_traces)

        if num_variates == 2:
            combined_features = self._combine_bivariate(
                variate_traces[0], variate_traces[1], order
            )
        elif num_variates == 3:
            combined_features = self._combine_trivariate(
                variate_traces[0], variate_traces[1], variate_traces[2], order
            )
        else:
            combined_features = self._combine_quadvariate(
                variate_traces[0], variate_traces[1], variate_traces[2], variate_traces[3], order
            )

        key_correlations = np.zeros(256)

        for key_guess in range(256):
            hypotheses = []
            for i in range(n_traces):
                from ..utils.crypto import aes_sbox
                plaintext_byte = plaintexts[i, target_byte] if plaintexts.ndim > 1 else plaintexts[i]
                hypothesis = bin(aes_sbox(plaintext_byte ^ key_guess)).count('1')
                hypotheses.append(hypothesis)

            if np.std(combined_features) > 0 and np.std(hypotheses) > 0:
                corr = np.abs(np.corrcoef(combined_features, hypotheses)[0, 1])
                key_correlations[key_guess] = corr

        best_key = np.argmax(key_correlations)
        best_correlation = key_correlations[best_key]
        sorted_correlations = np.sort(key_correlations)[::-1]
        margin = sorted_correlations[0] - sorted_correlations[1]

        results = {
            'best_key': best_key,
            'correlation': best_correlation,
            'margin': margin,
            'order': order,
            'num_variates': num_variates,
            'poi_sets': poi_sets,
            'key_correlations': key_correlations,
            'success': best_correlation > 0.15 and margin > 0.05
        }

        print(f"Best key: {best_key} (correlation: {best_correlation:.4f})")
        print(f"Margin over second best: {margin:.4f}")

        return results

    def _combine_bivariate(self, traces1: np.ndarray, traces2: np.ndarray,
                          order: int) -> np.ndarray:
        if order == 2:
            centered1 = traces1 - np.mean(traces1, axis=1, keepdims=True)
            centered2 = traces2 - np.mean(traces2, axis=1, keepdims=True)
            return np.mean(centered1 * centered2, axis=1)
        elif order == 3:
            centered1 = traces1 - np.mean(traces1, axis=1, keepdims=True)
            centered2 = traces2 - np.mean(traces2, axis=1, keepdims=True)
            return np.mean(centered1 * (centered2 ** 2), axis=1)
        else:
            centered1 = traces1 - np.mean(traces1, axis=1, keepdims=True)
            centered2 = traces2 - np.mean(traces2, axis=1, keepdims=True)
            return np.mean((centered1 ** 2) * (centered2 ** 2), axis=1)

    def _combine_trivariate(self, traces1: np.ndarray, traces2: np.ndarray,
                           traces3: np.ndarray, order: int) -> np.ndarray:
        centered1 = traces1 - np.mean(traces1, axis=1, keepdims=True)
        centered2 = traces2 - np.mean(traces2, axis=1, keepdims=True)
        centered3 = traces3 - np.mean(traces3, axis=1, keepdims=True)

        if order == 2:
            return np.mean(
                (centered1 * centered2 + centered1 * centered3 + centered2 * centered3) / 3,
                axis=1
            )
        elif order == 3:
            return np.mean(centered1 * centered2 * centered3, axis=1)
        else:
            return np.mean(
                centered1 * centered2 * centered3 * (centered1 + centered2 + centered3),
                axis=1
            )

    def _combine_quadvariate(self, traces1: np.ndarray, traces2: np.ndarray,
                            traces3: np.ndarray, traces4: np.ndarray,
                            order: int) -> np.ndarray:
        centered = [
            traces1 - np.mean(traces1, axis=1, keepdims=True),
            traces2 - np.mean(traces2, axis=1, keepdims=True),
            traces3 - np.mean(traces3, axis=1, keepdims=True),
            traces4 - np.mean(traces4, axis=1, keepdims=True)
        ]

        if order == 2:
            products = []
            for i in range(4):
                for j in range(i+1, 4):
                    products.append(np.mean(centered[i] * centered[j], axis=1))
            return np.mean(products, axis=0)
        elif order == 3:
            products = []
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        products.append(
                            np.mean(centered[i] * centered[j] * centered[k], axis=1)
                        )
            return np.mean(products, axis=0)
        else:
            return np.mean(
                centered[0] * centered[1] * centered[2] * centered[3],
                axis=1
            )
