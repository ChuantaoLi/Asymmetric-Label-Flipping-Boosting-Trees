from sklearn.neighbors import KDTree
import numpy as np


class Mutual_Nearest_Neighbors(object):
    """相互近邻计算类"""

    def __init__(self, X: np.array, y: np.array):
        self.nan_edges = {}
        self.nan_num = {}
        self.target: np.array = y
        self.data: np.array = X
        self.knn = {}
        self.nan = {}
        self.relative_cox = []

    def asserts(self):
        """初始化相互近邻参数"""
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.nan[j] = set()
            self.nan_num[j] = 0

    def count(self):
        """统计没有相互近邻的样本数量"""
        nan_zeros = 0
        for x in self.nan_num:
            if self.nan_num[x] == 0:
                nan_zeros += 1
        return nan_zeros

    def findKNN(self, inst, r, tree):
        """查找最近邻"""
        _dist, ind = tree.query([inst], r + 1)
        return np.delete(ind[0], 0)

    def algorithm(self):
        """相互近邻算法主函数"""
        tree = KDTree(self.data)
        self.asserts()
        flag = 0
        r = 2

        while flag == 0 and r <= 10:
            for i in range(len(self.data)):
                if self.nan_num[i] == 0:
                    knn = self.findKNN(self.data[i], r, tree)
                    for n in knn:
                        self.knn[i].add(n)
                        if i in self.knn.get(n, set()) and (i, n) not in self.nan_edges:
                            self.nan_edges.add((i, n))
                            self.nan_edges.add((n, i))
                            self.nan[i].add(n)
                            self.nan[n].add(i)
                            self.nan_num[i] += 1
                            self.nan_num[n] += 1

            cnt_after = self.count()
            if cnt_after == 0:
                flag = 1
            else:
                r += 1