from sklearn.neighbors import KDTree, NearestNeighbors
import numpy as np


class Mutual_Nearest_Neighbors(object):
    """
    相互近邻计算类
    不再需要手动设置固定的k值，而是根据样本局部密度自适应计算邻域
    """

    def __init__(self, X: np.array, y: np.array):
        self.target: np.array = y
        self.data: np.array = X
        self.knn = {}  # 存储每个点的k近邻
        self.nan = {}  # 存储每个点的相互近邻
        self.nan_edges = set()  # 存储相互近邻的边
        self.nan_num = {}  # 存储每个点相互近邻的数量

    def _initialize_structures(self):
        """初始化数据结构"""
        num_samples = len(self.data)
        self.knn = {j: set() for j in range(num_samples)}
        self.nan = {j: set() for j in range(num_samples)}
        self.nan_num = {j: 0 for j in range(num_samples)}
        self.nan_edges = set()

    def algorithm(self, k_selector: int = 5):
        """
        相互近邻算法主函数
        """

        self._initialize_structures()
        n_samples = self.data.shape[0]

        # 1. 找到每个点到其第 k_selector 个邻居的距离，作为其局部邻域范围 sigma
        nn = NearestNeighbors(n_neighbors=k_selector + 1).fit(self.data)
        distances, _ = nn.kneighbors(self.data)
        # 获取第 k_selector 个邻居的距离
        sigmas = distances[:, k_selector]

        # 2. 构建邻接关系：如果点 j 在点 i 的 sigma 范围内，则 j 是 i 的邻居
        tree = KDTree(self.data)
        all_knn_indices = tree.query_radius(self.data, r=sigmas)

        # 3. 构建初始邻居关系集合
        for i in range(n_samples):
            # 去掉点本身
            neighbors = set(all_knn_indices[i])
            neighbors.discard(i)
            self.knn[i] = neighbors

        # 4. 识别相互近邻关系
        for i in range(n_samples):
            for neighbor in self.knn[i]:
                # 确保 neighbor 是一个有效的整数索引
                neighbor = int(neighbor)
                # 如果 i 也在其邻居 neighbor 的邻居列表中
                # 并且这条边还没有被记录过
                if i in self.knn.get(neighbor, set()) and (i, neighbor) not in self.nan_edges:
                    # 添加相互近邻关系
                    self.nan[i].add(neighbor)
                    self.nan[neighbor].add(i)

                    # 记录边，防止重复计算
                    self.nan_edges.add((i, neighbor))
                    self.nan_edges.add((neighbor, i))

        # 5. 统计每个点的相互近邻数量
        for i in range(len(self.data)):
            self.nan_num[i] = len(self.nan[i])

        print(f"Adaptive MNN graph built using local scaling (k_selector={k_selector}).")
        isolated_points = sum(1 for count in self.nan_num.values() if count == 0)
        print(f"Found {isolated_points} isolated samples (with 0 MNNs).")