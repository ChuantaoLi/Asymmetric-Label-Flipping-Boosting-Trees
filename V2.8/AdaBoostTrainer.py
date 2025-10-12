import numpy as np
from sklearn.neighbors import NearestNeighbors
from MutualNearestNeighbors import Mutual_Nearest_Neighbors
from NeighborAnalyzer import Neighbor_Analyzer
from AdaBoostClassifier import AdaBoost
from collections import Counter


class AdaBoost_Trainer:
    """
    非对称翻转提升树V3.1 (带有复合因子权重更新 和 动态参数调控)
    """

    def __init__(self, data_arr, class_labels, num_it):
        self.data_arr = data_arr
        self.class_labels = class_labels
        self.num_it = num_it

    def _density_factor(self, X: np.ndarray, y: np.ndarray, k_neighbors: int) -> np.ndarray:
        """
        计算每个样本的密度因子 (rho)。
        密度高的样本rho值接近1，密度低的样本rho值接近0。
        """
        n = X.shape[0]
        rho_prime = np.zeros(n, dtype=float)

        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) <= 1:
                rho_prime[idx] = 0.0
                continue

            k = min(k_neighbors, len(idx) - 1)
            if k == 0:
                rho_prime[idx] = 0.0
                continue

            nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn.fit(X[idx])
            dists, _ = nn.kneighbors(X[idx], return_distance=True)

            avg_d = dists[:, 1:].mean(axis=1)
            rho_prime[idx] = np.exp(-avg_d)

        mn, mx = rho_prime.min(), rho_prime.max()
        if mx - mn < 1e-12:
            return np.zeros_like(rho_prime)
        else:
            return (rho_prime - mn) / (mx - mn)

    def _confidence_factor(self, agg_class_est: np.ndarray, encoded_labels: np.ndarray) -> np.ndarray:
        """
        计算每个样本的置信因子 (delta)。
        """
        margin = np.multiply(encoded_labels, agg_class_est.A.flatten())
        H = 1.0 / (1.0 + np.exp(margin))
        mu = np.mean(H)
        sigma = np.std(H)
        if sigma < 1e-12: sigma = 1e-12
        delta = 1.0 - np.exp(-((H - mu) ** 2) / (2.0 * sigma ** 2))
        return delta

    def _balance_factor(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        计算每个样本的平衡因子 (gamma)。
        """
        counts = Counter(encoded_labels)
        majority_count = max(counts.values())
        gamma = np.array([counts[c] / majority_count for c in encoded_labels], dtype=float)
        return gamma

    def train(self, visualizer=None, **vis_kwargs):
        """
        训练AdaBoost (V3.1)
        引入置信、密度和平衡因子改进权重更新公式
        引入基于密度的动态参数调控
        """
        unique, counts = np.unique(self.class_labels, return_counts=True)
        minority_label = unique[np.argmin(counts)]
        majority_label = unique[np.argmax(counts)]

        encoded_labels = np.ones_like(self.class_labels)
        encoded_labels[self.class_labels == majority_label] = -1
        label_map = {1: minority_label, -1: majority_label}

        weak_class_arr = []
        alpha_list = []
        m = self.data_arr.shape[0]
        D = np.mat(np.ones((m, 1)) / m)

        # 邻域分析
        mns = Mutual_Nearest_Neighbors(self.data_arr, self.class_labels)
        mns.algorithm()
        analyzer = Neighbor_Analyzer(mns, self.class_labels)
        sample_types = analyzer.analyze_samples()

        # =========================================================================
        # =================== 核心改进：动态参数调控 ================================
        # =========================================================================

        # 1. 在循环外预先计算所有样本的密度因子 rho
        # 这个值将用于动态调整 cost_factor 和 static_flipping_gain
        print("Calculating density factor (rho) for dynamic parameter tuning...")
        rho = self._density_factor(self.data_arr, self.class_labels, k_neighbors=7)
        print("Density factor calculation complete.")
        print("-" * 50)

        # 2. 动态调整 Cost Factor (噪声抑制)
        cost_factor = np.ones(m)
        outlier_mask = (sample_types == analyzer.SAMPLE_TYPES['OUTLIER'])
        isolated_mask = (sample_types == analyzer.SAMPLE_TYPES['ISOLATED'])
        suppression_mask = outlier_mask | isolated_mask

        # 改进：不再使用固定的0.2。样本密度(rho)越低，越可能是噪声，抑制因子就越小，抑制作用越强。
        # rho本身就在[0,1]区间，可以直接作为抑制因子使用。
        cost_factor[suppression_mask] = rho[suppression_mask]
        print(f"Applied DYNAMIC weight suppression to {np.sum(suppression_mask)} outlier/isolated samples based on their density.")

        # 3. 动态调整 Static Flipping Gain (翻转增益)
        static_flipping_gain = np.zeros(m)

        # 3.1 边界多数类样本 (Boundary Majority)
        boundary_majority_mask = (sample_types == analyzer.SAMPLE_TYPES['BOUNDARY']) & (self.class_labels == majority_label)
        # 改进：不再使用固定的0.8。密度越低(rho越小)，边界越模糊，翻转价值越大，增益应越高。
        # 使用(1-rho)反转趋势，并映射到[0.5, 0.9]区间。
        min_gain_b, max_gain_b = 0.7, 0.9
        gain_range_b = max_gain_b - min_gain_b
        dynamic_gain_b = min_gain_b + gain_range_b * (1.0 - rho[boundary_majority_mask])
        static_flipping_gain[boundary_majority_mask] = dynamic_gain_b
        print(f"Assigned DYNAMIC flipping gain (range [{min_gain_b}, {max_gain_b}]) to {np.sum(boundary_majority_mask)} majority boundary samples.")

        # 3.2 安全多数类样本 (Safe Majority)
        safe_majority_mask = (sample_types == analyzer.SAMPLE_TYPES['SAFE']) & (self.class_labels == majority_label)
        # 改进：不再使用固定的0.3。逻辑同上，但整体增益区间要低得多，映射到[0.1, 0.4]区间。
        min_gain_s, max_gain_s = 0.05, 0.15
        gain_range_s = max_gain_s - min_gain_s
        dynamic_gain_s = min_gain_s + gain_range_s * (1.0 - rho[safe_majority_mask])
        static_flipping_gain[safe_majority_mask] = dynamic_gain_s
        print(f"Assigned DYNAMIC flipping gain (range [{min_gain_s}, {max_gain_s}]) to {np.sum(safe_majority_mask)} majority safe samples.")
        print("-" * 50)

        # 将cost_factor转为矩阵形式，以便后续乘法操作
        cost_factor_mat = np.mat(cost_factor).T

        # =========================================================================
        # =================== 动态参数调控结束 ====================================
        # =========================================================================

        # 涟漪拓展机制部分与之前版本保持不变
        EXPANSION_NEIGHBORS = 10
        MAX_PROXIMITY_BOOST = 1.5
        nn_model_for_expansion = NearestNeighbors(n_neighbors=EXPANSION_NEIGHBORS).fit(self.data_arr)
        last_flipped_indices = np.array([], dtype=int)

        print("--- Starting AdaBoost Training with Dynamic Parameters (V3.1) ---")
        agg_class_est = np.mat(np.zeros((m, 1)))

        for i in range(self.num_it):
            encoded_labels_for_stump = encoded_labels.copy()
            indices_to_flip_this_round = np.array([])

            # 翻转策略逻辑与之前版本保持不变
            if i > 0:
                # 改进建议：使用Sigmoid函数计算后验概率，物理意义更明确
                posterior_prob = 1.0 / (1.0 + np.exp(-agg_class_est.A.flatten()))

                proximity_gain = np.ones(m)
                if len(last_flipped_indices) > 0:
                    _, neighbor_indices = nn_model_for_expansion.kneighbors(self.data_arr[last_flipped_indices])
                    candidate_indices = np.unique(neighbor_indices.flatten())
                    majority_mask = (encoded_labels == -1)
                    candidate_mask_bool = np.zeros(m, dtype=bool)
                    candidate_mask_bool[candidate_indices] = True
                    final_points_indices = np.where(majority_mask & candidate_mask_bool)[0]
                    if len(final_points_indices) > 0:
                        _, neighbors_of_points = nn_model_for_expansion.kneighbors(self.data_arr[final_points_indices])
                        minority_counts = np.sum(self.class_labels[neighbors_of_points] == minority_label, axis=1)
                        gains = 1.0 + (MAX_PROXIMITY_BOOST - 1.0) * (minority_counts / EXPANSION_NEIGHBORS)
                        proximity_gain[final_points_indices] = gains

                majority_indices_local = np.where(encoded_labels == -1)[0]
                adjusted_probs = np.clip(
                    posterior_prob[majority_indices_local] *
                    static_flipping_gain[majority_indices_local] *
                    proximity_gain[majority_indices_local],
                    0, 1.0
                )
                flipped_mask = np.random.rand(len(majority_indices_local)) < adjusted_probs
                indices_to_flip_this_round = majority_indices_local[flipped_mask]
                last_flipped_indices = indices_to_flip_this_round
                if len(indices_to_flip_this_round) > 0:
                    encoded_labels_for_stump[indices_to_flip_this_round] *= -1

            best_stump, error = AdaBoost.build_stump(self.data_arr, encoded_labels_for_stump, D)
            error = max(error.item(), 1e-16)
            alpha = 0.5 * np.log((1.0 - error) / error)

            weak_class_arr.append(best_stump)
            alpha_list.append(alpha)

            class_est = AdaBoost.stump_classify(np.mat(self.data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])
            agg_class_est += alpha * class_est

            # --- 复合因子更新样本权重 ---
            delta = self._confidence_factor(agg_class_est, encoded_labels)
            # rho 已经提前计算，此处无需重复计算
            gamma = self._balance_factor(encoded_labels)

            adapt = np.exp(-delta * gamma * (1.0 - rho))

            original_predictions = AdaBoost.stump_classify(np.mat(self.data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])
            base_expon = np.multiply(-alpha * np.mat(encoded_labels_for_stump).T, original_predictions)

            D = np.multiply(D, np.exp(base_expon))
            D = np.multiply(D, np.mat(adapt).T)
            D = np.multiply(D, cost_factor_mat)  # 应用动态计算的抑制因子
            D /= D.sum()

            # 评估和可视化部分不变
            temp_model = AdaBoost()
            temp_model.classifiers = weak_class_arr
            temp_model.alphas = alpha_list
            temp_model.label_map = label_map
            current_predictions, _ = temp_model.predict(self.data_arr)
            g_mean = AdaBoost.calculate_gmean(self.class_labels, current_predictions, majority_label, minority_label)
            print(f"Iteration {i + 1}/{self.num_it}: G-Mean={g_mean:.3f}, Flipped={len(indices_to_flip_this_round)} samples")

            if visualizer and ((i + 1) % 10 == 0 or (i + 1) == self.num_it or i == 0):
                visualizer.plot_decision_boundary(
                    model=temp_model,
                    iteration=i + 1,
                    flipped_indices_in_train=indices_to_flip_this_round,
                    **vis_kwargs
                )

        final_model = AdaBoost()
        final_model.classifiers = weak_class_arr
        final_model.alphas = alpha_list
        final_model.label_map = label_map
        return final_model