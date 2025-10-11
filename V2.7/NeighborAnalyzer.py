import numpy as np
from MutualNearestNeighbors import Mutual_Nearest_Neighbors


class Neighbor_Analyzer:
    """
    邻域识别器 (V2 - 改进版)
    功能：对每个样本进行类型划分，提供更精细的分析。
    类型定义：
    - SAFE: 所有相互近邻都与自身同类。
    - BOUNDARY: 相互近邻中存在异类样本。
    - OUTLIER: 所有相互近邻都与自身异类。
    - ISOLATED: 没有任何相互近邻。
    """
    SAMPLE_TYPES = {'SAFE': 0, 'BOUNDARY': 1, 'OUTLIER': 2, 'ISOLATED': 3}

    def __init__(self, nn_obj: 'Mutual_Nearest_Neighbors', y: np.array):
        """初始化分析器"""
        self.nn = nn_obj
        self.y_train = y
        self.num_samples = len(y)
        self.sample_classification = np.full(self.num_samples, -1, dtype=int)

    def analyze_samples(self) -> np.array:
        """
        对所有样本进行分类，返回一个包含每个样本类型的Numpy数组。
        """
        print("-" * 50)
        print("Starting comprehensive sample analysis...")
        for i in range(self.num_samples):
            my_label = self.y_train[i]
            neighbors = self.nn.nan.get(i, set())

            # 情况1: 没有相互近邻 -> 孤立点 (ISOLATED)
            if not neighbors:
                self.sample_classification[i] = self.SAMPLE_TYPES['ISOLATED']
                continue

            neighbor_labels = self.y_train[list(neighbors)]
            num_same_class = np.sum(neighbor_labels == my_label)
            num_diff_class = len(neighbors) - num_same_class

            # 情况2: 所有邻居都与自己不同 -> 离群点 (OUTLIER)
            if num_same_class == 0:
                self.sample_classification[i] = self.SAMPLE_TYPES['OUTLIER']
            # 情况3: 所有邻居都与自己相同 -> 安全点 (SAFE)
            elif num_diff_class == 0:
                self.sample_classification[i] = self.SAMPLE_TYPES['SAFE']
            # 情况4: 邻居中有不同类别的点 -> 边界点 (BOUNDARY)
            else:
                self.sample_classification[i] = self.SAMPLE_TYPES['BOUNDARY']

        # 打印统计信息
        counts = {name: np.sum(self.sample_classification == code) for name, code in self.SAMPLE_TYPES.items()}
        print("Sample Analysis Complete:")
        for name, count in counts.items():
            print(f"  -> Identified {count} '{name}' samples.")
        print("-" * 50)

        return self.sample_classification