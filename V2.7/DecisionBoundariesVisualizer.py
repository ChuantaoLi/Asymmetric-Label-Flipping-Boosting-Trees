import matplotlib.pyplot as plt
import numpy as np
import os


class Visualizer:
    """
    生成和保存决策边界的可视化图像
    """

    def __init__(self, output_dir='decision_boundaries'):
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 18
        plt.rcParams['legend.fontsize'] = 12

    def plot_decision_boundary(self, model, scaler, pca, X_train_pca, y_train, X_test_pca, y_test, iteration, flipped_indices_in_train):
        """绘制并保存当前迭代次数下的决策边界图"""
        print(f"  -> Generating visualization for iteration {iteration}...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Decision Boundary after Iteration {iteration}', fontsize=20)

        # 创建网格
        x_min = min(X_train_pca[:, 0].min(), X_test_pca[:, 0].min()) - 1
        x_max = max(X_train_pca[:, 0].max(), X_test_pca[:, 0].max()) + 1
        y_min = min(X_train_pca[:, 1].min(), X_test_pca[:, 1].min()) - 1
        y_max = max(X_train_pca[:, 1].max(), X_test_pca[:, 1].max()) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

        # 对网格点进行预测
        grid_points_2d = np.c_[xx.ravel(), yy.ravel()]
        grid_points_high_dim_scaled = pca.inverse_transform(grid_points_2d)
        grid_points_high_dim_original = scaler.inverse_transform(grid_points_high_dim_scaled)

        # 使用传入的 model 对象进行预测
        Z, _ = model.predict(grid_points_high_dim_original)
        Z = Z.reshape(xx.shape)

        minority_label, majority_label = model.label_map[1], model.label_map[-1]

        # 绘制训练集
        ax1.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
        ax1.scatter(X_train_pca[y_train == minority_label, 0], X_train_pca[y_train == minority_label, 1], c='red', edgecolor='k', s=60, label='Minority Class', alpha=0.9)
        ax1.scatter(X_train_pca[y_train == majority_label, 0], X_train_pca[y_train == majority_label, 1], c='blue', edgecolor='k', s=60, label='Majority Class', alpha=0.9)

        if flipped_indices_in_train.size > 0:
            ax1.scatter(X_train_pca[flipped_indices_in_train, 0], X_train_pca[flipped_indices_in_train, 1], s=150, facecolors='none',
                        edgecolors='yellow', linewidth=2, marker='o',
                        label=f'Flipped Samples ({len(flipped_indices_in_train)})')

        ax1.set_title('Training Set')
        ax1.set_xlabel('Principal Component 1')
        ax1.set_ylabel('Principal Component 2')
        ax1.legend()
        ax1.grid(False)

        # 绘制测试集
        ax2.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.5)
        ax2.scatter(X_test_pca[y_test == minority_label, 0], X_test_pca[y_test == minority_label, 1], c='red', edgecolor='k', s=60, label='Minority Class', alpha=0.9)
        ax2.scatter(X_test_pca[y_test == majority_label, 0], X_test_pca[y_test == majority_label, 1], c='blue', edgecolor='k', s=60, label='Majority Class', alpha=0.9)
        ax2.set_title('Testing Set')
        ax2.set_xlabel('Principal Component 1')
        ax2.set_ylabel('Principal Component 2')
        ax2.legend()
        ax2.grid(False)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.output_dir}/boundary_iteration_{iteration:03d}.png', dpi=300)
        plt.close()
