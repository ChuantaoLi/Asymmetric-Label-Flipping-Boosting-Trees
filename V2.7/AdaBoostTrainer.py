import numpy as np
from sklearn.neighbors import NearestNeighbors
from MutualNearestNeighbors import Mutual_Nearest_Neighbors
from NeighborAnalyzer import Neighbor_Analyzer
from AdaBoostClassifier import AdaBoost
from collections import Counter


class AdaBoost_Trainer:
    """
    非对称翻转提升树V3.0 (带有复合因子权重更新)
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

        # 分别计算每个类别的内部密度
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) <= 1:
                rho_prime[idx] = 0.0
                continue

            # 确保 k 小于类别样本数
            k = min(k_neighbors, len(idx) - 1)
            if k == 0:
                rho_prime[idx] = 0.0
                continue

            nn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
            nn.fit(X[idx])
            dists, _ = nn.kneighbors(X[idx], return_distance=True)

            # 计算到k个邻居的平均距离 (忽略自身)
            avg_d = dists[:, 1:].mean(axis=1)
            rho_prime[idx] = np.exp(-avg_d)  # 距离越小，密度越大

        mn, mx = rho_prime.min(), rho_prime.max()
        # 归一化到[0, 1]
        if mx - mn < 1e-12:
            return np.zeros_like(rho_prime)
        else:
            return (rho_prime - mn) / (mx - mn)

    def _confidence_factor(self, agg_class_est: np.ndarray, encoded_labels: np.ndarray) -> np.ndarray:
        """
        计算每个样本的置信因子 (delta)。
        基于当前集成模型的预测分数计算，越难分类的样本delta值越高。
        """
        # agg_class_est 的符号代表预测类别，绝对值大小代表置信度
        # y*f(x) 的值越小，说明分类越不确定或错误
        margin = np.multiply(encoded_labels, agg_class_est.A.flatten())

        # 将 margin 映射到 (0, 1) 区间，作为“模糊度 H”
        # margin 趋近于-inf, H趋近于1 (非常不确定)
        # margin 趋近于+inf, H趋近于0 (非常确定)
        H = 1.0 / (1.0 + np.exp(margin))

        mu = np.mean(H)
        sigma = np.std(H)
        if sigma < 1e-12: sigma = 1e-12

        # delta是置信因子，基于H值的分布计算，值越大表示样本越难分类
        delta = 1.0 - np.exp(-((H - mu) ** 2) / (2.0 * sigma ** 2))
        return delta

    def _balance_factor(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        计算每个样本的平衡因子 (gamma)。
        少数类样本的gamma值更小，使得其权重更新时惩罚更轻。
        """
        counts = Counter(encoded_labels)
        majority_count = max(counts.values())

        # 类别数量越少，gamma值越小，在权重更新中越受“保护”
        gamma = np.array([counts[c] / majority_count for c in encoded_labels], dtype=float)
        return gamma

    def train(self, visualizer=None, **vis_kwargs):
        """
        训练AdaBoost (V3.0)
        引入置信、密度和平衡因子改进权重更新公式
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

        # (邻域分析部分与之前版本保持不变)
        mns = Mutual_Nearest_Neighbors(self.data_arr, self.class_labels)
        mns.algorithm()
        analyzer = Neighbor_Analyzer(mns, self.class_labels)
        sample_types = analyzer.analyze_samples()

        # (代价因子和翻转增益部分与之前版本保持不变)
        cost_factor = np.ones(m)
        outlier_mask = (sample_types == analyzer.SAMPLE_TYPES['OUTLIER'])
        isolated_mask = (sample_types == analyzer.SAMPLE_TYPES['ISOLATED'])
        suppression_mask = outlier_mask | isolated_mask
        cost_factor[suppression_mask] = 0.2
        cost_factor_mat = np.mat(cost_factor).T
        print(f"Applied weight suppression to {np.sum(suppression_mask)} outlier/isolated samples.")

        static_flipping_gain = np.zeros(m)
        boundary_majority_mask = (sample_types == analyzer.SAMPLE_TYPES['BOUNDARY']) & (self.class_labels == majority_label)
        static_flipping_gain[boundary_majority_mask] = 0.8
        safe_majority_mask = (sample_types == analyzer.SAMPLE_TYPES['SAFE']) & (self.class_labels == majority_label)
        static_flipping_gain[safe_majority_mask] = 0.3
        print(f"Assigned high static flipping gain to {np.sum(boundary_majority_mask)} majority boundary samples.")
        print("-" * 50)

        # (涟漪拓展机制部分与之前版本保持不变)
        EXPANSION_NEIGHBORS = 5
        MAX_PROXIMITY_BOOST = 1.25
        nn_model_for_expansion = NearestNeighbors(n_neighbors=EXPANSION_NEIGHBORS).fit(self.data_arr)
        last_flipped_indices = np.array([], dtype=int)

        print("--- Starting AdaBoost Training with Composite Factor Weight Update (V3.0) ---")
        agg_class_est = np.mat(np.zeros((m, 1)))

        for i in range(self.num_it):
            encoded_labels_for_stump = encoded_labels.copy()
            indices_to_flip_this_round = np.array([])

            # (翻转策略逻辑与之前版本保持不变)
            if i > 0:
                posterior_prob = 0.5 * (np.tanh(agg_class_est.A) + 1).flatten()
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

            # 更新集成模型预测分数
            class_est = AdaBoost.stump_classify(np.mat(self.data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])
            agg_class_est += alpha * class_est

            # --- V3.0 核心改进：引入复合因子更新样本权重 ---
            # 1. 计算各项因子
            delta = self._confidence_factor(agg_class_est, encoded_labels)
            rho = self._density_factor(self.data_arr, self.class_labels, k_neighbors=5)
            gamma = self._balance_factor(encoded_labels)

            # 2. 计算自适应调节项
            # 公式解读: 越难分的(delta大)、越稀疏的(rho小)、属于少数类的(gamma小)样本，调节项越小，从而权重惩罚越轻
            adapt = np.exp(-delta * gamma * (1.0 - rho))

            # 3. 计算标准AdaBoost的指数更新项
            original_predictions = AdaBoost.stump_classify(np.mat(self.data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])
            base_expon = np.multiply(-alpha * np.mat(encoded_labels_for_stump).T, original_predictions)

            # 4. 结合所有项进行最终权重更新
            # D_t+1 = adapt * D_t * exp(base_expon) * cost_factor
            D = np.multiply(D, np.exp(base_expon))
            D = np.multiply(D, np.mat(adapt).T)  # 应用自适应调节项
            D = np.multiply(D, cost_factor_mat)  # 应用离群点抑制
            D /= D.sum()
            # ----------------------------------------------------

            # (评估和可视化部分不变)
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