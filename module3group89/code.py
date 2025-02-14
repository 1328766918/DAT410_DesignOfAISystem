import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score


class KMeansClassifier:
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters)
        self.cluster_labels = None

    def fit(self, features, labels):
        # Train KMeans on the feature data
        self.model.fit(features)
        # Initialize an array to store the majority label for each cluster
        self.cluster_labels = np.zeros(self.n_clusters)
        for cluster_id in range(self.n_clusters):
            # Retrieve labels corresponding to points assigned to the current cluster
            cluster_members = labels[self.model.labels_ == cluster_id]
            # If there are points in the cluster, assign the majority label to the cluster
            if len(cluster_members) > 0:
                self.cluster_labels[cluster_id] = np.bincount(cluster_members).argmax()

    def predict(self, features):
        # Predict cluster assignments for new data points
        assigned_clusters = self.model.predict(features)
        predictions = []
        for cluster in assigned_clusters:
            predictions.append(self.cluster_labels[cluster])
        return np.array(predictions)

    def score(self, true_labels, predicted_labels):
        # Calculate and return the accuracy score.
        return accuracy_score(true_labels, predicted_labels)


# Load datasets from CSV files
df_beijing = pd.read_csv('Beijing_labeled.csv')
df_shenyang = pd.read_csv('Shenyang_labeled.csv')
df_guangzhou = pd.read_csv('Guangzhou_labeled.csv')
df_shanghai = pd.read_csv('Shanghai_labeled.csv')

# Combine Beijing and Shenyang data for training/validation
training_df = pd.concat([df_beijing, df_shenyang])
X_train = training_df.drop(columns=['PM_HIGH'])
y_train = training_df['PM_HIGH']

# Separate test data for Guangzhou
X_test_guangzhou = df_guangzhou.drop(columns=['PM_HIGH'])
y_test_guangzhou = df_guangzhou['PM_HIGH']

# Separate test data for Shanghai
X_test_shanghai = df_shanghai.drop(columns=['PM_HIGH'])
y_test_shanghai = df_shanghai['PM_HIGH']

# Create and train the classifier with 2 clusters
kmeans_clf = KMeansClassifier(n_clusters=2)
kmeans_clf.fit(X_train, y_train)

# Evaluate performance on the training set
train_predictions = kmeans_clf.predict(X_train)
train_accuracy = kmeans_clf.score(y_train, train_predictions)

# Evaluate performance on the Guangzhou test set
guangzhou_predictions = kmeans_clf.predict(X_test_guangzhou)
guangzhou_accuracy = kmeans_clf.score(y_test_guangzhou, guangzhou_predictions)

# Evaluate performance on the Shanghai test set
shanghai_predictions = kmeans_clf.predict(X_test_shanghai)
shanghai_accuracy = kmeans_clf.score(y_test_shanghai, shanghai_predictions)

print(f"Training Accuracy (Beijing and Shenyang): {train_accuracy:.4f}")
print(f"Test Accuracy (Guangzhou): {guangzhou_accuracy:.4f}")
print(f"Test Accuracy (Shanghai): {shanghai_accuracy:.4f}")