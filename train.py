import pandas as pd
from collections import Counter

# Load the dataset
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv("data/test.csv")

# Get information about the train and test data
print(f"Training data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Display basic information about the dataset
train_info = train_data.info()

# Check for duplicates and missing values
duplicates = train_data.duplicated().sum()
missing_values = train_data.isnull().sum().sum()
print(f"Number of duplicate rows: {duplicates}")
print(f"Number of missing values: {missing_values}")

# Group and count main names of columns
column_groups = pd.DataFrame.from_dict(
    Counter([col.split('-')[0].split('(')[0] for col in train_data.columns]), 
    orient='index'
).rename(columns={0: 'count'}).sort_values('count', ascending=False)

print("Grouped column counts:")
print(column_groups)

import plotly.express as px

# Define colors for activities
activity_colors = {
    'WALKING': '#e60049',
    'WALKING_UPSTAIRS': '#0bb4ff',
    'WALKING_DOWNSTAIRS': '#50e991',
    'SITTING': '#e6d800',
    'STANDING': '#9b19f5',
    'LAYING': '#ffa300'
}

# Label distribution of activities
activity_counts = train_data['Activity'].value_counts().reset_index()
activity_counts.columns = ['Activity', 'Count']

# Pie chart for activity distribution
fig = px.pie(activity_counts, values='Count', names='Activity', title='Activity Distribution', 
             color='Activity', color_discrete_map=activity_colors)
fig.show()

import plotly.graph_objects as go

# Each subject's distribution of activity number of samples
subject_activity_counts = train_data.groupby(['subject', 'Activity']).size().unstack().fillna(0).reset_index()

# Prepare data for stacked bar chart
data = []
for activity in subject_activity_counts.columns[1:]:
    data.append(go.Bar(name=activity, x=subject_activity_counts['subject'], y=subject_activity_counts[activity],
                       marker_color=activity_colors[activity]))

# Bar chart for each subject's activity distribution
fig = go.Figure(data=data)
fig.update_layout(barmode='stack', title='Activity Distribution per Subject', 
                  xaxis_title='Subject', yaxis_title='Number of Samples')
fig.show()

# List of features to plot
features_to_plot = [
    'tBodyAcc-mean()-X', 'tBodyAcc-mean()-Y', 'tBodyAcc-mean()-Z',
    'tBodyAcc-std()-X', 'tBodyAcc-std()-Y', 'tBodyAcc-std()-Z',
    'tGravityAcc-mean()-X', 'tGravityAcc-mean()-Y', 'tGravityAcc-mean()-Z',
    'tGravityAcc-std()-X', 'tGravityAcc-std()-Y', 'tGravityAcc-std()-Z',
    'tBodyGyro-mean()-X', 'tBodyGyro-mean()-Y', 'tBodyGyro-mean()-Z',
    'tBodyGyro-std()-X', 'tBodyGyro-std()-Y', 'tBodyGyro-std()-Z',
    'tBodyAccMag-mean()', 'tBodyAccMag-std()',
    'angle(tBodyAccMean,gravity)', 'angle(tBodyAccJerkMean),gravityMean)',
    'angle(tBodyGyroMean,gravityMean)', 'angle(tBodyGyroJerkMean,gravityMean)',
    'angle(X,gravityMean)', 'angle(Y,gravityMean)', 'angle(Z,gravityMean)'
]

# Generate a KDE plot for each feature
for feature in features_to_plot:
    fig = go.Figure()

    for activity, color in activity_colors.items():
        subset = train_data[train_data['Activity'] == activity]
        x = subset[feature]
        kde = gaussian_kde(x, bw_method=0.3)
        x_eval = np.linspace(x.min() - 0.1, x.max() + 0.1, 1000)
        y_eval = kde(x_eval)
        fig.add_trace(go.Scatter(x=x_eval, y=y_eval, mode='lines', name=activity, line=dict(color=color)))

    # Update layout for better visualization
    fig.update_layout(
        title=f'Density Plot for {feature} by Activity',
        xaxis_title=feature,
        yaxis_title='Density',
        legend_title='Activity',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    fig.show()

    import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Separate features and target
X = train_data.drop(columns=['Activity', 'subject'])
y = train_data['Activity']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Threshold for high correlation
threshold = 0.75

# Function to find highly correlated features for each activity
def find_highly_correlated_features(df, activity, threshold):
    activity_indices = df['Activity'] == activity
    X_activity = df.loc[activity_indices].drop(columns=['Activity', 'subject'])
    
    corr_matrix = X_activity.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    return set(to_drop)

# Find highly correlated features for each activity
high_corr_features_per_activity = {}
for activity in y.unique():
    high_corr_features_per_activity[activity] = find_highly_correlated_features(train_data, activity, threshold)
    print(f"Number of highly correlated features for {activity}: {len(high_corr_features_per_activity[activity])}")

# Find intersection of highly correlated features across all activities
common_high_corr_features = set.intersection(*high_corr_features_per_activity.values())
print(f"Number of common highly correlated features: {len(common_high_corr_features)}")

# Cluster the common highly correlated features
X_common = X_scaled_df[list(common_high_corr_features)]
corr_matrix_common = X_common.corr().abs()

# Generate the linkage matrix
Z = linkage(corr_matrix_common, 'ward')

# Plot dendrogram
plt.figure(figsize=(15, 10))
dendrogram(Z, labels=corr_matrix_common.columns)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Features')
plt.ylabel('Distance')
plt.show()

# Cluster the features and select a representative feature for each cluster
clusters = fcluster(Z, t=1.15, criterion='distance')
cluster_dict = {}
for i, feature in enumerate(corr_matrix_common.columns):
    cluster_dict.setdefault(clusters[i], []).append(feature)

# Select representative feature (e.g., the first feature in each cluster)
selected_representative_features = [features[0] for features in cluster_dict.values()]

# Print clusters and selected representative features
for cluster, features in cluster_dict.items():
    print(f"Cluster {cluster}: {features}")
    print(f"Selected representative feature: {features[0]}")

# Print summary of clustering and feature elimination
num_clusters = len(cluster_dict)
num_features_eliminated = len(common_high_corr_features) - len(selected_representative_features)
print(f"Created {num_clusters} clusters of highly correlated features.")
print(f"Eliminated {num_features_eliminated} features, retaining {len(selected_representative_features)} representative features.")

# Combine non-highly correlated features with representative features
non_high_corr_features = set(X.columns) - common_high_corr_features
combined_features = list(non_high_corr_features) + selected_representative_features
print(f"Total number of features in the dataset after elimination: {len(combined_features)}")

# Filter dataset to include only the combined set of features
X_combined = X[combined_features]

# Standardize the combined feature set
X_combined_scaled = scaler.fit_transform(X_combined)

# Train and evaluate model before feature elimination using cross-validation
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores_before_elimination = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
print("Cross-validation performance before feature elimination:")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_before_elimination.mean(), scores_before_elimination.std() * 2))

# Train and evaluate model after feature elimination using cross-validation
scores_after_elimination = cross_val_score(clf, X_combined_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-validation performance after feature elimination:")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_after_elimination.mean(), scores_after_elimination.std() * 2))

# Get feature importances from the Random Forest
clf.fit(X_combined_scaled, y)
feature_importances = clf.feature_importances_

# Plot feature importances with Plotly
importances = pd.Series(feature_importances, index=combined_features)
importances = importances.sort_values(ascending=False)

fig = px.bar(importances, x=importances.index, y=importances.values, title='Feature Importances from Random Forest', labels={'x':'Features', 'y':'Importance'})
fig.update_layout(xaxis={'categoryorder':'total descending'})
fig.show()

# List top 10 features
top_10_features = importances.index[:10]
print("Top 10 features based on importance:")
print(top_10_features)

# Observe performance changes for different number of features
feature_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200]
performance_scores = []

for N in feature_counts:
    top_features = importances.index[:N]
    X_top = X_combined[top_features]
    X_top_scaled = scaler.fit_transform(X_top)
    scores_top_features = cross_val_score(clf, X_top_scaled, y, cv=5, scoring='accuracy')
    performance_scores.append(scores_top_features.mean())

# Plot performance changes
fig = go.Figure()
fig.add_trace(go.Scatter(x=feature_counts, y=performance_scores, mode='lines+markers', name='Performance'))
fig.update_layout(title='Performance Changes with Different Number of Features', xaxis_title='Number of Features', yaxis_title='Accuracy')
fig.show()

# Selected the top N features based on importance
N = 80
top_features = importances.index[:N]
X_top = X_combined[top_features]

X_top.shape

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np

# Separate features and target
X_train = X_top
y_train = train_data['Activity']
X_test = test_data[X_top.columns]
y_test = test_data['Activity']

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode categorical activities
activity_encoder = {activity: idx for idx, activity in enumerate(y_train.unique())}
y_train_encoded = y_train.map(activity_encoder)
y_test_encoded = y_test.map(activity_encoder)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded.values, dtype=torch.long)

# Reshape the data to fit into the CNN model
X_train_tensor = X_train_tensor.view(X_train_tensor.size(0), 1, X_train_tensor.size(1))
X_test_tensor = X_test_tensor.view(X_test_tensor.size(0), 1, X_test_tensor.size(1))

# Split training data into train and validation sets
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(
    X_train_tensor, y_train_tensor, test_size=0.2, random_state=42)

# Create DataLoader for train, validation and test sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * (X_train_tensor.size(2) // 2 // 2), 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.ReLU()(self.conv1(x)))
        x = self.pool(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.LogSoftmax(dim=1)(x)

# Initialize the model, loss function, and optimizer
num_classes = len(np.unique(y_train))
model = CNNModel(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100 * correct / total

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Train the model
train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=20)

# Evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = test_loss / len(test_loader.dataset)
    test_acc = 100 * correct / total

    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Evaluate the model
evaluate_model(model, test_loader)