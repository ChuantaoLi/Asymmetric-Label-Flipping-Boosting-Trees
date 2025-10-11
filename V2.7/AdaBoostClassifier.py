import numpy as np
from sklearn.metrics import confusion_matrix


class AdaBoost:
    """
    二分类AdaBoost
    """

    def __init__(self):
        self.classifiers = []
        self.alphas = []
        self.label_map = {}

    @staticmethod
    def stump_classify(data_matrix, dim, thresh_val, thresh_ineq):
        """使用决策树桩对数据进行分类"""
        ret_array = np.ones((data_matrix.shape[0], 1))
        if thresh_ineq == "lt":
            ret_array[data_matrix[:, dim] <= thresh_val] = -1.0
        else:
            ret_array[data_matrix[:, dim] > thresh_val] = -1.0
        return ret_array

    @staticmethod
    def build_stump(data_arr, encoded_labels, D):
        """构建最佳的决策树桩"""
        data_matrix = np.mat(data_arr)
        label_mat = np.mat(encoded_labels).T
        m, n = data_matrix.shape
        best_stump = {}
        min_error = np.inf
        for i in range(n):
            feature_values = np.unique(data_matrix[:, i].A1)
            for thresh_val in feature_values:
                for inequal in ["lt", "gt"]:
                    predicted_vals = AdaBoost.stump_classify(data_matrix, i, thresh_val, inequal)
                    err_arr = np.mat(np.ones((m, 1)))
                    err_arr[predicted_vals == label_mat] = 0
                    weighted_error = D.T * err_arr
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_stump["dim"] = i
                        best_stump["thresh"] = thresh_val
                        best_stump["ineq"] = inequal
        return best_stump, min_error

    @staticmethod
    def calculate_gmean(y_true, y_pred, majority_label, minority_label):
        """计算 G-Mean 分数"""
        labels = [majority_label, minority_label]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        return np.sqrt(sens * spec)

    def predict(self, data_to_class):
        """使用训练好的多个弱分类器进行最终预测"""
        data_matrix = np.mat(data_to_class)
        m = data_matrix.shape[0]
        agg_class_est = np.mat(np.zeros((m, 1)))
        for i in range(len(self.classifiers)):
            class_est = self.stump_classify(
                data_matrix, self.classifiers[i]["dim"],
                self.classifiers[i]["thresh"], self.classifiers[i]["ineq"])
            agg_class_est += self.alphas[i] * class_est
        predictions_encoded = np.sign(agg_class_est).A1
        final_predictions = np.array([self.label_map[int(p)] for p in predictions_encoded])
        return final_predictions, agg_class_est.A.flatten()
