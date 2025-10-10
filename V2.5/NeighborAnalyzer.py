import numpy as np
from MutualNearestNeighbors import Mutual_Nearest_Neighbors


class Neighbor_Analyzer:
    """
    邻域识别器，划分边界样本、离群点和噪声点
    """

    def __init__(self, nn_obj: 'Mutual_Nearest_Neighbors', y: np.array):
        """
        初始化分析器
        """
        self.nn = nn_obj
        self.y_train = y
        self.num_samples = len(y)

    def identify_boundary_samples(self, majority_label, minority_label) -> np.array:
        """
        识别多数类中的边界样本
        如果一个多数类样本的相互近邻中至少有一个是少数类样本，则它被视为边界点
        """
        majority_indices = np.where(self.y_train == majority_label)[0]
        boundary_scores = np.zeros(self.num_samples)
        boundary_count = 0

        for i in majority_indices:
            # 获取样本 i 的相互近邻
            neighbors = self.nn.nan.get(i, set())
            if not neighbors:
                continue

            # 检查近邻中是否有少数类样本
            for neighbor_idx in neighbors:
                if self.y_train[neighbor_idx] == minority_label:
                    boundary_scores[i] = 1.0
                    boundary_count += 1
                    break  # 找到一个即可，跳出内层循环

        print(f"Identified {boundary_count} majority samples on the boundary.")
        return boundary_scores

    def identify_outliers(self) -> np.array:
        """
        识别所有类别中的潜在离群点和噪声点
        没有任何相互近邻的样本
        所有相互近邻的标签都与自身标签不同的样本
        """
        is_outlier_noise = np.zeros(self.num_samples, dtype=bool)
        outlier_count = 0

        for i in range(self.num_samples):
            neighbors = self.nn.nan.get(i, set())

            # 情况1: 没有相互近邻，被视为离群点
            if not neighbors:
                is_outlier_noise[i] = True
                outlier_count += 1
                continue

            # 情况2: 所有相互近邻的标签都与自己不同
            neighbor_labels = self.y_train[list(neighbors)]
            if np.all(neighbor_labels != self.y_train[i]):
                is_outlier_noise[i] = True
                outlier_count += 1

        print(f"Identified {outlier_count} potential outlier/noise samples.")
        print("-" * 50)
        return is_outlier_noise
