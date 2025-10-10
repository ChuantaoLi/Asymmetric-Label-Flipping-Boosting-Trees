import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)
from AdaBoostClassifier import AdaBoost
from DecisionBoundariesVisualizer import Visualizer
from AdaBoostTrainer import AdaBoost_Trainer

if __name__ == '__main__':
    df = pd.read_csv(r'D:\毕业设计\二分类数据集\banana.csv').dropna()

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    num_iterations = 50
    trainer = AdaBoost_Trainer(X_train, y_train, num_it=num_iterations)
    visualizer = Visualizer(output_dir='Decision_Boundaries')

    vis_params = {
        'scaler': scaler,
        'pca': pca,
        'X_train_pca': X_train_pca,
        'y_train': y_train,
        'X_test_pca': X_test_pca,
        'y_test': y_test
    }

    trained_model = trainer.train(visualizer=visualizer, **vis_params)

    y_pred, y_scores = trained_model.predict(X_test)

    minority_class_label = trained_model.label_map[1]
    majority_class_label = trained_model.label_map[-1]

    g_mean = AdaBoost.calculate_gmean(y_test, y_pred, majority_class_label, minority_class_label)

    y_test_binary = (y_test == minority_class_label).astype(int)
    if minority_class_label == trained_model.label_map[-1]:
        y_scores = -y_scores
    auc = roc_auc_score(y_test_binary, y_scores)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n--- Performance Metrics on Test Set ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"G-Mean: {g_mean:.3f}")
    print(f"AUC: {auc:.3f}")
