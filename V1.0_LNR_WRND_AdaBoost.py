import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
import warnings
import os
from sklearn.neighbors import KDTree

# --- 全局设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


class Natural_Neighbor(object):
    """自然邻居计算类"""

    def __init__(self, X: np.array, y: np.array):
        self.nan_edges = {}
        self.nan_num = {}
        self.target: np.array = y
        self.data: np.array = X
        self.knn = {}
        self.nan = {}
        self.relative_cox = []

    def asserts(self):
        """初始化自然邻居参数"""
        self.nan_edges = set()
        for j in range(len(self.data)):
            self.knn[j] = set()
            self.nan[j] = set()
            self.nan_num[j] = 0

    def count(self):
        """统计没有自然邻居的实例数量"""
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
        """自然邻居算法主函数"""
        tree = KDTree(self.data)
        self.asserts()
        flag = 0
        r = 2
        cnt_before = -1

        while flag == 0 and r <= 5:
            for i in range(len(self.data)):
                knn = self.findKNN(self.data[i], r, tree)
                n = knn[-1]
                self.knn[i].add(n)

                if i in self.knn[n] and (i, n) not in self.nan_edges:
                    self.nan_edges.add((i, n))
                    self.nan_edges.add((n, i))
                    self.nan[i].add(n)
                    self.nan[n].add(i)
                    self.nan_num[i] += 1
                    self.nan_num[n] += 1

            cnt_after = self.count()
            if cnt_after < np.sqrt(len(self.data)):
                flag = 1
            else:
                r += 1
            cnt_before = cnt_after
        return r, tree

    def RelativeDensity(self, min_i, maj_i):
        """计算相对密度权重"""
        self.relative_cox = [0] * len(self.target)
        for i, num in self.nan.items():
            if self.target[i] == min_i:
                if len(num) == 0:
                    self.relative_cox[i] = -2 # 离群样本
                else:
                    absolute_min, min_num, absolute_max, maj_num = 0, 0, 0, 0
                    maj_index = []

                    for j in iter(num):
                        if self.target[j] == min_i:
                            absolute_min += np.sqrt(np.sum(np.square(self.data[i] - self.data[j])))
                            min_num += 1
                        elif self.target[j] == maj_i:
                            absolute_max += np.sqrt(np.sum(np.square(self.data[i] - self.data[j])))
                            maj_num += 1
                            maj_index.append(j)

                    self.nan[i].difference_update(maj_index)

                    if min_num == 0:
                        self.relative_cox[i] = -3 # 噪声点
                    elif maj_num == 0: # 安全点
                        relative = min_num / absolute_min if absolute_min > 0 else 0
                        self.relative_cox[i] = relative
                    else: # 边界样本
                        relative_min = min_num / absolute_min if absolute_min > 0 else 0
                        relative_maj = maj_num / absolute_max if absolute_max > 0 else 0
                        self.relative_cox[i] = relative_min / relative_maj if relative_maj > 0 else relative_min


# --- 1. 核心算法模块 (无变化) ---
def stump_classify(data_matrix, dim, thresh_val, thresh_ineq):
    ret_array = np.ones((data_matrix.shape[0], 1))
    if thresh_ineq == "lt":
        ret_array[data_matrix[:, dim] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dim] > thresh_val] = -1.0
    return ret_array


# --- 1. 核心算法模块 (无变化) ---
def stump_classify(data_matrix, dim, thresh_val, thresh_ineq):
    ret_array = np.ones((data_matrix.shape[0], 1))
    if thresh_ineq == "lt":
        ret_array[data_matrix[:, dim] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dim] > thresh_val] = -1.0
    return ret_array


def build_stump(data_arr, encoded_labels, D):
    data_matrix = np.mat(data_arr)
    label_mat = np.mat(encoded_labels).T
    m, n = data_matrix.shape
    best_stump = {}
    min_error = np.inf
    for i in range(n):
        feature_values = np.unique(data_matrix[:, i].A1)
        for thresh_val in feature_values:
            for inequal in ["lt", "gt"]:
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = np.mat(np.ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T * err_arr
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump["dim"] = i
                    best_stump["thresh"] = thresh_val
                    best_stump["ineq"] = inequal
    return best_stump, min_error, None  # best_class_est is not needed outside


def calculate_gmean(y_true, y_pred, majority_label, minority_label):
    labels = [majority_label, minority_label]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return np.sqrt(sens * spec)


def ada_classify(data_to_class, classifier_arr, alpha_list, label_map):
    data_matrix = np.mat(data_to_class)
    m = data_matrix.shape[0]
    agg_class_est = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(
            data_matrix, classifier_arr[i]["dim"],
            classifier_arr[i]["thresh"], classifier_arr[i]["ineq"])
        agg_class_est += alpha_list[i] * class_est
    predictions_encoded = np.sign(agg_class_est).A1
    final_predictions = np.array([label_map[int(p)] for p in predictions_encoded])
    return final_predictions, agg_class_est.A.flatten()


# --- 2. 可视化模块 (无变化) ---
def plot_decision_boundary(classifiers, alphas, label_map, scaler, pca,
                           X_train_pca, y_train, X_test_pca, y_test,
                           iteration, flipped_indices_in_train):
    print(f"  -> Generating visualization for iteration {iteration}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Decision Boundary after Iteration {iteration}', fontsize=20)
    x_min, x_max = min(X_train_pca[:, 0].min(), X_test_pca[:, 0].min()) - 1, max(X_train_pca[:, 0].max(), X_test_pca[:, 0].max()) + 1
    y_min, y_max = min(X_train_pca[:, 1].min(), X_test_pca[:, 1].min()) - 1, max(X_train_pca[:, 1].max(), X_test_pca[:, 1].max()) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_points_high_dim_scaled = pca.inverse_transform(grid_points_2d)
    grid_points_high_dim_original = scaler.inverse_transform(grid_points_high_dim_scaled)
    Z, _ = ada_classify(grid_points_high_dim_original, classifiers, alphas, label_map)
    Z = Z.reshape(xx.shape)
    minority_label, majority_label = label_map[1], label_map[-1]
    ax1.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    ax1.scatter(X_train_pca[y_train == minority_label, 0], X_train_pca[y_train == minority_label, 1], c='red', edgecolor='k', s=60, label='Minority Class', alpha=0.9)
    ax1.scatter(X_train_pca[y_train == majority_label, 0], X_train_pca[y_train == majority_label, 1], c='blue', edgecolor='k', s=60, label='Majority Class', alpha=0.9)
    if flipped_indices_in_train.size > 0:
        ax1.scatter(X_train_pca[flipped_indices_in_train, 0], X_train_pca[flipped_indices_in_train, 1], s=150, facecolors='none', edgecolors='yellow', linewidth=2, marker='o',
                    label=f'Flipped Samples ({len(flipped_indices_in_train)})')
    ax1.set_title('Training Set')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend()
    ax1.grid(False)
    ax2.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
    ax2.scatter(X_test_pca[y_test == minority_label, 0], X_test_pca[y_test == minority_label, 1], c='red', edgecolor='k', s=60, label='Minority Class', alpha=0.9)
    ax2.scatter(X_test_pca[y_test == majority_label, 0], X_test_pca[y_test == majority_label, 1], c='blue', edgecolor='k', s=60, label='Majority Class', alpha=0.9)
    ax2.set_title('Testing Set')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.legend()
    ax2.grid(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if not os.path.exists('decision_boundaries'):
        os.makedirs('decision_boundaries')
    plt.savefig(f'V1.0决策边界可视化/V1.0_Boundary_iteration_{iteration:03d}.png', dpi=600)
    plt.close()


# --- 3. 改进权重更新机制的训练模块 ---
def ada_boost_train_ds_with_flipping(data_arr, class_labels, num_it):
    unique, counts = np.unique(class_labels, return_counts=True)
    minority_label = unique[np.argmin(counts)]
    majority_label = unique[np.argmax(counts)]
    encoded_labels = np.ones_like(class_labels)
    encoded_labels[class_labels == majority_label] = -1
    label_map = {1: minority_label, -1: majority_label}

    weak_class_arr = []
    alpha_list = []
    m = data_arr.shape[0]
    D = np.mat(np.ones((m, 1)) / m)

    # ==================== 新增模块：代价因子计算 ====================
    print("--- Analyzing training data with Natural Neighbors to create cost factors ---")
    nn = Natural_Neighbor(data_arr, class_labels)
    nn.algorithm()
    nn.RelativeDensity(minority_label, majority_label)
    cost_weights_raw = np.array(nn.relative_cox)

    # 初始化代价因子，默认都为1
    cost_factor = np.ones(m)

    # 找到所有少数类样本的索引
    minority_indices = np.where(class_labels == minority_label)[0]

    # 分离出边界/安全样本 (cox >= 0) 和 离群/噪声样本 (cox < 0)
    border_safe_indices = minority_indices[cost_weights_raw[minority_indices] >= 0]
    outlier_noise_indices = minority_indices[cost_weights_raw[minority_indices] < 0]

    # 1. 为边界/安全样本设置更大的代价
    if len(border_safe_indices) > 0:
        # 使用 StandardScaler 对正权重进行标准化，使其均值为0，方差为1
        # 然后通过 sigmoid(tanh) 函数映射到一个合适的范围(例如1-2之间)，作为增益因子
        border_safe_weights = cost_weights_raw[border_safe_indices].reshape(-1, 1)
        scaler_cost = StandardScaler()
        scaled_weights = scaler_cost.fit_transform(border_safe_weights)
        # Tanh 将值映射到(-1, 1)，再调整到(1, 2)
        # 使得权重越大的样本，其代价因子越高，最高接近2
        cost_factor[border_safe_indices] = 1.5 + 0.5 * np.tanh(scaled_weights.flatten())

    # 2. 为离群/噪声样本设置更小的代价
    DAMPENING_FACTOR = 0.2  # 抑制因子，可以调整
    cost_factor[outlier_noise_indices] = DAMPENING_FACTOR

    # 转换为矩阵形式，便于后续计算
    cost_factor_mat = np.mat(cost_factor).T

    print(f"Identified {len(outlier_noise_indices)} outlier/noise minority samples (cost suppressed to {DAMPENING_FACTOR}).")
    print(f"Identified {len(border_safe_indices)} borderline/safe minority samples (cost amplified).")
    print("-" * 50)
    # =================================================================

    print("--- Starting AdaBoost Training with Cost-Sensitive Weight Update ---")
    for i in range(num_it):
        encoded_labels_for_stump = encoded_labels.copy()
        indices_to_flip_this_round = np.array([])

        if i > 0:
            agg_class_est = np.mat(np.zeros((m, 1)))
            for j in range(i):
                class_est = stump_classify(np.mat(data_arr), weak_class_arr[j]["dim"], weak_class_arr[j]["thresh"], weak_class_arr[j]["ineq"])
                agg_class_est += alpha_list[j] * class_est
            posterior_prob = 0.5 * (np.tanh(agg_class_est.A) + 1)
            majority_indices = np.where(encoded_labels == -1)[0]
            flipping_probs = posterior_prob[majority_indices]
            random_values = np.random.rand(len(majority_indices))
            flipped_mask = random_values < flipping_probs.flatten()
            indices_to_flip_this_round = majority_indices[flipped_mask]
            encoded_labels_for_stump[indices_to_flip_this_round] *= -1

        best_stump, _, _ = build_stump(data_arr, encoded_labels_for_stump, D)
        original_predictions = stump_classify(np.mat(data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])
        err_arr = np.mat(np.ones((m, 1)))
        err_arr[original_predictions == np.mat(encoded_labels).T] = 0
        error = float((D.T * err_arr).item())
        error = max(error, 1e-16)
        alpha = 0.5 * np.log((1.0 - error) / error)

        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)
        alpha_list.append(alpha)

        # ==================== 核心修改：在权重更新中引入代价因子 ====================
        # 标准更新项: -alpha * y_i * h(x_i)
        # 改进后更新项: -alpha * y_i * h(x_i) * C(i)
        base_expon = np.multiply(-alpha * np.mat(encoded_labels).T, original_predictions)
        cost_sensitive_expon = np.multiply(base_expon, cost_factor_mat)

        D = np.multiply(D, np.exp(cost_sensitive_expon))
        D /= D.sum()
        # =======================================================================

        current_predictions, _ = ada_classify(data_arr, weak_class_arr, alpha_list, label_map)
        g_mean = calculate_gmean(class_labels, current_predictions, majority_label, minority_label)
        print(f"Iteration {i + 1}/{num_it}: G-Mean={g_mean:.3f}, Flipped={len(indices_to_flip_this_round)} samples")

        if (i + 1) % 20 == 0 or (i + 1) == num_it or i == 0:
            plot_decision_boundary(list(weak_class_arr), list(alpha_list), label_map, scaler, pca, X_train_pca, y_train, X_test_pca, y_test, i + 1, indices_to_flip_this_round)

    return weak_class_arr, alpha_list, label_map


# --- 4. 主执行流程 (已修正) ---
if __name__ == '__main__':
    file_path = "SPECTF Heart.csv"
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Data loaded: {X.shape[1]} features.")
    print(f"Training set size: {len(y_train)}, Testing set size: {len(y_test)}")
    print("-" * 50)

    num_iterations = 100
    trained_classifiers, alphas, label_map_final = ada_boost_train_ds_with_flipping(
        X_train, y_train, num_iterations
    )

    print("\n--- Model Training Finished ---")
    print("-" * 50)

    y_pred, y_scores = ada_classify(X_test, trained_classifiers, alphas, label_map_final)

    # 1. 从返回的 label_map 中提取标签，确保它们在当前作用域中已定义
    minority_class_label = label_map_final[1]
    majority_class_label = label_map_final[-1]

    # 2. 计算 G-Mean
    g_mean = calculate_gmean(y_test, y_pred, majority_class_label, minority_class_label)

    # 3. 计算 AUC
    y_test_binary = (y_test == minority_class_label).astype(int)
    auc = roc_auc_score(y_test_binary, y_scores)
    # ======================================================

    # 计算其他指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Performance Metrics on Test Set ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"G-Mean: {g_mean:.3f}")
    print(f"AUC: {auc:.3f}")
