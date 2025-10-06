import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.neighbors import NearestNeighbors  # 导入 NearestNeighbors
import warnings
import os

# --- 全局设置 ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['legend.fontsize'] = 12
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["LOKY_MAX_CPU_COUNT"] = "8"


# --- 1. IBI³ 计算模块 ---
def _calculate_ibi3_for_single_sample(x_sample, X_all, y_all, k0, r, majority_label, nn_minority, all_indices_map):
    """为单个少数类样本计算IBI³值 (内部函数)"""
    # 在整个数据集中找到初始邻居
    nn_all = NearestNeighbors(n_neighbors=k0 + 1).fit(X_all)
    _, indices = nn_all.kneighbors([x_sample])
    neighbor_indices = indices[0][1:]  # 排除样本自身
    neighbor_labels = y_all[neighbor_indices]

    # 计算多数类邻居的数量
    M = np.sum(neighbor_labels == majority_label)
    k = k0

    # 应用 "flexible k" 策略
    if M == k0:
        # 情况B：所有k0个邻居都是多数类
        dist_to_nearest_minority, _ = nn_minority.kneighbors([x_sample], n_neighbors=2)
        search_radius = dist_to_nearest_minority[0][1]

        # 在该半径内重新统计多数类邻居
        nn_radius = NearestNeighbors(radius=search_radius).fit(X_all)
        radius_indices = nn_radius.radius_neighbors([x_sample], return_distance=False)[0]

        sample_idx = all_indices_map.get(tuple(x_sample.tolist()), -1)
        if sample_idx != -1:
            radius_indices = radius_indices[radius_indices != sample_idx]

        radius_neighbor_labels = y_all[radius_indices]
        M = np.sum(radius_neighbor_labels == majority_label)
        k = M + 1

    if k == 0: return 0.0

    # 估计局部概率得分
    fn = M / k
    fp = (k - M) / k
    fp_prime = r * fp

    # 计算 IBI³
    denom_imbalanced = fn + fp
    denom_balanced = fn + fp_prime
    p_imbalanced = fp / denom_imbalanced if denom_imbalanced > 0 else 0
    p_balanced = fp_prime / denom_balanced if denom_balanced > 0 else 0
    return p_balanced - p_imbalanced


def calculate_ibi3_factors(X, y, minority_label, majority_label, k0=5):
    """
    计算所有少数类样本的IBI³值，并返回一个包含所有样本的代价因子数组。
    """
    print("--- Calculating IBI^3 values to create cost factors ---")

    # 1. 准备数据
    minority_indices = np.where(y == minority_label)[0]
    X_minority = X[minority_indices]

    minority_count = len(X_minority)
    majority_count = len(y) - minority_count
    imbalance_ratio = majority_count / minority_count if minority_count > 0 else 1.0

    # 2. 为提高效率进行预计算
    nn_minority_model = NearestNeighbors(n_neighbors=2).fit(X_minority)
    all_indices_map = {tuple(row.tolist()): i for i, row in enumerate(X)}

    ibi3_scores = np.zeros(minority_count)
    for i, sample in enumerate(X_minority):
        score = _calculate_ibi3_for_single_sample(
            sample, X, y, k0, imbalance_ratio,
            majority_label, nn_minority_model, all_indices_map
        )
        ibi3_scores[i] = score

    print(f"Calculated IBI^3 for {minority_count} minority samples. Mean score: {np.mean(ibi3_scores):.4f}")

    # 3. 为所有样本创建最终的代价因子数组
    cost_factor = np.ones(len(y))
    # 代价因子 = 1 + IBI³ 分数。这会提升“困难”少数类样本的重要性。
    # 分数可能为负，我们将其裁剪为0，以避免意外降低权重。
    cost_factor[minority_indices] = 1 + np.maximum(0, ibi3_scores)

    print(f"Cost factors created. Min factor: {np.min(cost_factor):.3f}, Max factor: {np.max(cost_factor):.3f}")
    print("-" * 50)

    return cost_factor


# --- 2. 核心算法模块 ---
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


# --- 3. 可视化模块 (无变化) ---
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
    plt.savefig(f'V2.0_Boundary_iteration_{iteration:03d}.png', dpi=600)
    plt.close()


# --- 4. 改进权重更新机制的训练模块 ---
def ada_boost_train_ds_with_flipping(data_arr, class_labels, num_it=10):
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

    # ==================== 新增模块：IBI³ 代价因子计算 ====================
    cost_factor = calculate_ibi3_factors(data_arr, class_labels, minority_label, majority_label, k0=5)
    cost_factor_mat = np.mat(cost_factor).T
    # =================================================================

    print("--- Starting AdaBoost Training with IBI^3 Cost-Sensitive Update ---")
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
        base_expon = np.multiply(-alpha * np.mat(encoded_labels).T, original_predictions)
        cost_sensitive_expon = np.multiply(base_expon, cost_factor_mat)
        D = np.multiply(D, 0.6 * np.exp(cost_sensitive_expon))
        # =======================================================================

        D /= D.sum()

        current_predictions, _ = ada_classify(data_arr, weak_class_arr, alpha_list, label_map)
        g_mean = calculate_gmean(class_labels, current_predictions, majority_label, minority_label)
        print(f"Iteration {i + 1}/{num_it}: G-Mean={g_mean:.3f}, Flipped={len(indices_to_flip_this_round)} samples")

        if (i + 1) % 20 == 0 or (i + 1) == num_it or i == 0:
            plot_decision_boundary(list(weak_class_arr), list(alpha_list), label_map, scaler, pca, X_train_pca, y_train, X_test_pca, y_test, i + 1, indices_to_flip_this_round)

    return weak_class_arr, alpha_list, label_map


# --- 5. 主执行流程 ---
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
        X_train, y_train, num_it=num_iterations
    )

    print("\n--- Model Training Finished ---")
    print("-" * 50)

    y_pred, y_scores = ada_classify(X_test, trained_classifiers, alphas, label_map_final)

    minority_class_label = label_map_final[1]
    majority_class_label = label_map_final[-1]

    g_mean = calculate_gmean(y_test, y_pred, majority_class_label, minority_class_label)
    y_test_binary = (y_test == minority_class_label).astype(int)
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
