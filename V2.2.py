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


# --- 1. 您提供的 Natural_Neighbor 类 ---
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
        print("--- Building Natural Neighbor graph ---")
        tree = KDTree(self.data)
        self.asserts()
        flag = 0
        r = 2

        while flag == 0 and r <= 10:  # 稍微放宽 r 的上限
            for i in range(len(self.data)):
                # 仅为还没有足够邻居的样本计算
                if self.nan_num[i] == 0:
                    knn = self.findKNN(self.data[i], r, tree)
                    for n in knn:
                        self.knn[i].add(n)
                        # 检查是否互为邻居
                        if i in self.knn.get(n, set()) and (i, n) not in self.nan_edges:
                            self.nan_edges.add((i, n))
                            self.nan_edges.add((n, i))
                            self.nan[i].add(n)
                            self.nan[n].add(i)
                            self.nan_num[i] += 1
                            self.nan_num[n] += 1

            cnt_after = self.count()
            print(f"r={r}, samples with 0 NaNs: {cnt_after}")
            if cnt_after == 0:  # 理想情况是所有点都有邻居
                flag = 1
            else:
                r += 1
        print("Natural Neighbor graph constructed.")


# --- 2. 新增的边界分数计算模块 (基于自然邻居) ---
def calculate_majority_boundary_scores_with_nn(nn_obj, majority_label, minority_label, y_train):
    """
    使用 Natural_Neighbor 对象的结果来计算多数类样本的边界分数。
    """
    print("--- Identifying boundary majority samples using Natural Neighbors ---")
    majority_indices = np.where(y_train == majority_label)[0]
    boundary_scores_all = np.zeros(len(y_train))

    boundary_count = 0
    for i in majority_indices:
        is_boundary = False
        # 检查其自然邻居中是否有少数类样本
        for neighbor_idx in nn_obj.nan.get(i, set()):
            if y_train[neighbor_idx] == minority_label:
                is_boundary = True
                break

        if is_boundary:
            boundary_scores_all[i] = 1.0
            boundary_count += 1

    print(f"Identified {boundary_count} majority samples on the boundary.")
    print("-" * 50)
    return boundary_scores_all


# --- 3. 核心算法模块 (无变化) ---
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
    return best_stump, min_error, None


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


# --- 4. 可视化模块 (无变化) ---
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
    plt.savefig(f'decision_boundaries/boundary_iteration_{iteration:03d}.png', dpi=300)
    plt.close()


from sklearn.neighbors import NearestNeighbors  # 确保导入 NearestNeighbors


# --- 5. 最终改进的训练模块 (实现“涟漪”拓展机制) ---
def ada_boost_train_ds_with_flipping(data_arr, class_labels, num_it=100):
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

    # 1. 构建自然邻居图，用于识别全局的、静态的边界区域
    nn = Natural_Neighbor(data_arr, class_labels)
    nn.algorithm()

    # 2. 根据自然邻居分析，计算一个静态的“翻转增益”
    #    边界点获得高增益，安全点获得一个较低的基础增益
    boundary_scores = calculate_majority_boundary_scores_with_nn(nn, majority_label, minority_label, class_labels)

    BASELINE_FLIP_GAIN = 0.3  # 安全点的基础翻转增益 (来自您的代码)
    static_flipping_gain = np.full_like(boundary_scores, fill_value=BASELINE_FLIP_GAIN)
    static_flipping_gain[boundary_scores == 1.0] = 1.0

    # ==================== 核心修改：为“涟漪”拓展机制做准备 ====================
    # a. 定义“拓展”的邻域大小和增益强度
    EXPANSION_NEIGHBORS = 10  # 拓展时查找的邻居数量
    PROXIMITY_BOOST_FACTOR = 3  # 对周边样本的概率增强倍数

    # b. 预构建k-NN模型，用于在迭代中快速查找周边样本
    print(f"--- Building k-NN model for expansion mechanism (k={EXPANSION_NEIGHBORS}) ---")
    nn_model_for_expansion = NearestNeighbors(n_neighbors=EXPANSION_NEIGHBORS).fit(data_arr)

    # c. 初始化变量，存储上一轮被翻转的样本索引
    last_flipped_indices = np.array([], dtype=int)
    # =======================================================================

    print("--- Starting AdaBoost Training with Expansionary Boundary Flipping ---")
    for i in range(num_it):
        encoded_labels_for_stump = encoded_labels.copy()
        indices_to_flip_this_round = np.array([])

        if i > 0:
            agg_class_est = np.mat(np.zeros((m, 1)))
            for j in range(i):
                class_est = stump_classify(np.mat(data_arr), weak_class_arr[j]["dim"], weak_class_arr[j]["thresh"], weak_class_arr[j]["ineq"])
                agg_class_est += alpha_list[j] * class_est

            posterior_prob = 0.5 * (np.tanh(agg_class_est.A) + 1).flatten()

            # ==================== 核心修改：实现“涟漪”拓展翻转逻辑 ====================
            # 1. 创建一个动态的“邻近度增益”数组，基础为1（无增益）
            proximity_gain = np.ones(m)

            # 2. 如果上一轮有翻转的样本，则对其周边进行增益
            if len(last_flipped_indices) > 0:
                # 找到上一轮翻转样本的邻居
                _, neighbor_indices = nn_model_for_expansion.kneighbors(data_arr[last_flipped_indices])
                # 将所有邻居索引合并并去重
                expansion_indices = np.unique(neighbor_indices.flatten())
                # 为这些“拓展区域”的样本施加概率增益
                proximity_gain[expansion_indices] = PROXIMITY_BOOST_FACTOR

            # 3. 计算最终翻转概率
            majority_indices = np.where(encoded_labels == -1)[0]

            base_probs = posterior_prob[majority_indices]
            static_gain = static_flipping_gain[majority_indices]
            dynamic_gain = proximity_gain[majority_indices]

            adjusted_flipping_probs = base_probs * static_gain * dynamic_gain
            adjusted_flipping_probs = np.clip(adjusted_flipping_probs, 0, 1.0)
            # =======================================================================

            random_values = np.random.rand(len(majority_indices))
            flipped_mask = random_values < adjusted_flipping_probs

            indices_to_flip_this_round = majority_indices[flipped_mask]
            last_flipped_indices = indices_to_flip_this_round

            if len(indices_to_flip_this_round) > 0:
                encoded_labels_for_stump[indices_to_flip_this_round] *= -1

        best_stump, _, _ = build_stump(data_arr, encoded_labels_for_stump, D)
        original_predictions = stump_classify(np.mat(data_arr), best_stump["dim"], best_stump["thresh"], best_stump["ineq"])

        # 在这里，我们必须使用训练时的标签(encoded_labels_for_stump)来计算误差
        err_arr = np.mat(np.ones((m, 1)))
        err_arr[original_predictions == np.mat(encoded_labels_for_stump).T] = 0
        error = float((D.T * err_arr).item())
        error = max(error, 1e-16)
        alpha = 0.5 * np.log((1.0 - error) / error)

        best_stump["alpha"] = alpha
        weak_class_arr.append(best_stump)
        alpha_list.append(alpha)

        # ####################################################################
        # ##                      >>> 核心修正点 <<<                      ##
        # ##  使用弱学习器训练时所用的“临时标签”来更新权重。               ##
        # ##  这样，对于被成功翻转的样本，它被视为“分类正确”，权重会减小。   ##
        # ####################################################################
        base_expon = np.multiply(-alpha * np.mat(encoded_labels_for_stump).T, original_predictions)
        D = np.multiply(D, 0.6 * np.exp(base_expon))
        D /= D.sum()

        current_predictions, _ = ada_classify(data_arr, weak_class_arr, alpha_list, label_map)
        g_mean = calculate_gmean(class_labels, current_predictions, majority_label, minority_label)
        print(f"Iteration {i + 1}/{num_it}: G-Mean={g_mean:.3f}, Flipped={len(indices_to_flip_this_round)} samples")

        if (i + 1) % 20 == 0 or (i + 1) == num_it or i == 0:
            plot_decision_boundary(list(weak_class_arr), list(alpha_list), label_map, scaler, pca, X_train_pca, y_train, X_test_pca, y_test, i + 1, indices_to_flip_this_round)

    return weak_class_arr, alpha_list, label_map


# --- 6. 主执行流程 ---
if __name__ == '__main__':
    file_path = "二分类数据集/值得实验的/balance.csv"
    df = pd.read_csv(file_path).dropna()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
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
        X_train, y_train, num_it=num_iterations
    )

    print("\n--- Model Training Finished ---")
    print("-" * 50)

    y_pred, y_scores = ada_classify(X_test, trained_classifiers, alphas, label_map_final)

    minority_class_label = label_map_final[1]
    majority_class_label = label_map_final[-1]

    g_mean = calculate_gmean(y_test, y_pred, majority_class_label, minority_class_label)
    y_test_binary = (y_test == minority_class_label).astype(int)
    if minority_class_label == label_map_final[-1]:
        y_scores = -y_scores
    auc = roc_auc_score(y_test_binary, y_scores)

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
