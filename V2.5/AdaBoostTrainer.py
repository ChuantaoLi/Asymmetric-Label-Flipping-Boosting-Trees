import numpy as np
from sklearn.neighbors import NearestNeighbors
from MutualNearestNeighbors import Mutual_Nearest_Neighbors
from NeighborAnalyzer import Neighbor_Analyzer
from AdaBoostClassifier import AdaBoost


class AdaBoost_Trainer:
    """
    非对称翻转提升树V2.5
    """

    def __init__(self, data_arr, class_labels, num_it):
        self.data_arr = data_arr
        self.class_labels = class_labels
        self.num_it = num_it

    def train(self, visualizer=None, **vis_kwargs):
        """
        训练AdaBoost
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

        # 初始化 MNN 和邻域分析器
        mns = Mutual_Nearest_Neighbors(self.data_arr, self.class_labels)
        mns.algorithm()
        analyzer = Neighbor_Analyzer(mns, self.class_labels)

        # 异常值抑制
        is_outlier = analyzer.identify_outliers()
        cost_factor = np.ones(m)
        cost_factor[is_outlier] = 0.1
        cost_factor_mat = np.mat(cost_factor).T

        # 静态边界增益
        boundary_scores = analyzer.identify_boundary_samples(majority_label, minority_label)
        static_flipping_gain = np.full_like(boundary_scores, fill_value=0.5)
        static_flipping_gain[boundary_scores == 1.0] = 1.0

        # 涟漪拓展机制
        EXPANSION_NEIGHBORS = 5
        MAX_PROXIMITY_BOOST = 2.0
        print(f"--- Building k-NN model for expansion mechanism (k={EXPANSION_NEIGHBORS}) ---")
        nn_model_for_expansion = NearestNeighbors(n_neighbors=EXPANSION_NEIGHBORS).fit(self.data_arr)
        last_flipped_indices = np.array([], dtype=int)

        print("--- Starting AdaBoost Training with Intelligent Boundary Flipping and Outlier Suppression ---")
        for i in range(self.num_it):
            encoded_labels_for_stump = encoded_labels.copy()
            indices_to_flip_this_round = np.array([])

            if i > 0:
                agg_class_est = np.mat(np.zeros((m, 1)))
                for j in range(i):
                    class_est = AdaBoost.stump_classify(np.mat(self.data_arr), weak_class_arr[j]["dim"], weak_class_arr[j]["thresh"], weak_class_arr[j]["ineq"])
                    agg_class_est += alpha_list[j] * class_est

                posterior_prob = 0.5 * (np.tanh(agg_class_est.A) + 1).flatten()

                # 实现涟漪拓展
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

                majority_indices = np.where(encoded_labels == -1)[0]
                adjusted_probs = np.clip(posterior_prob[majority_indices] * static_flipping_gain[majority_indices] * proximity_gain[majority_indices], 0, 1.0)

                flipped_mask = np.random.rand(len(majority_indices)) < adjusted_probs
                indices_to_flip_this_round = majority_indices[flipped_mask]
                last_flipped_indices = indices_to_flip_this_round

                if len(indices_to_flip_this_round) > 0:
                    encoded_labels_for_stump[indices_to_flip_this_round] *= -1

            best_stump, _ = AdaBoost.build_stump(self.data_arr, encoded_labels_for_stump, D)
            original_predictions = AdaBoost.stump_classify(np.mat(self.data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])

            err_arr = np.mat(np.ones((m, 1)))
            err_arr[original_predictions == np.mat(encoded_labels_for_stump).T] = 0
            error = max(float((D.T * err_arr).item()), 1e-16)
            alpha = 0.5 * np.log((1.0 - error) / error)

            weak_class_arr.append(best_stump)
            alpha_list.append(alpha)

            base_expon = np.multiply(-alpha * np.mat(encoded_labels_for_stump).T, original_predictions)
            cost_sensitive_expon = np.multiply(base_expon, cost_factor_mat)
            D = np.multiply(D, np.exp(cost_sensitive_expon))
            D /= D.sum()

            # 创建一个临时的模型用于评估和可视化
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

        # 返回最终模型
        final_model = AdaBoost()
        final_model.classifiers = weak_class_arr
        final_model.alphas = alpha_list
        final_model.label_map = label_map
        return final_model
