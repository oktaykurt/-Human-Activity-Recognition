import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
from torch.utils.data import DataLoader, TensorDataset

# Define the CNN model
class CNNModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(128 * (X_test_tensor.size(2) // 2 // 2), 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)

# Load the pre-trained model and scaler
model = torch.load('model.pth')
scaler = pd.read_pickle('scaler.pkl')

# Load the test data
test_data = pd.read_csv("data/test.csv")

# Preprocessing test data
combined_features = [feature for feature in test_data.columns if feature not in ['Activity', 'subject']]  # Adjust accordingly
X_test = test_data[combined_features]
y_test = test_data['Activity']

X_test_scaled = scaler.transform(X_test)

activity_encoder = {activity: idx for idx, activity in enumerate(y_test.unique())}
activity_decoder = {idx: activity for activity, idx in activity_encoder.items()}
y_test_encoded = y_test.map(activity_encoder)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).view(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
y_test_tensor = torch.tensor(y_test_encoded.values, dtype=torch.long)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()

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
    return test_loss, test_acc, predicted, labels

# Evaluate the model
test_loss, test_acc, predictions, labels = evaluate_model(model, test_loader)

st.title('Test Set Evaluation Dashboard')

st.header('Test Set Summary')
st.write(f"Test set shape: {test_data.shape}")
st.write(f"Test Loss: {test_loss:.4f}")
st.write(f"Test Accuracy: {test_acc:.2f}%")

# Display activity distribution in the test set
st.header('Activity Distribution in Test Set')
activity_counts = test_data['Activity'].value_counts().reset_index()
activity_counts.columns = ['Activity', 'Count']

fig = px.pie(activity_counts, values='Count', names='Activity', title='Activity Distribution')
st.plotly_chart(fig)

# Dropdown for selecting an instance
st.header('Evaluate Individual Test Instance')
selected_index = st.selectbox('Select instance index', range(len(test_data)))

# Evaluate the selected instance
model.eval()
with torch.no_grad():
    input_tensor = X_test_tensor[selected_index].unsqueeze(0)
    label = y_test_tensor[selected_index].item()
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1).item()

st.write(f"Selected instance index: {selected_index}")
st.write(f"Actual Activity: {activity_decoder[label]}")
st.write(f"Predicted Activity: {activity_decoder[prediction]}")

# Interactive feature exploration
st.header('Feature Exploration')
selected_feature = st.selectbox('Select feature', combined_features)

fig = px.histogram(test_data, x=selected_feature, color='Activity', title=f'Distribution of {selected_feature} by Activity')
st.plotly_chart(fig)
