
# Human Activity Recognition Project

This project focuses on classifying human activities using data collected from smartphone sensors. The main components are data preprocessing, feature analysis, model training, and evaluation. The project includes scripts for data preparation, model training, and a Streamlit dashboard for model evaluation.

## Project Structure

- `train.py`: Script for data preprocessing, feature analysis, and model training.
- `app.py`: Streamlit app for evaluating the trained model and visualizing test set results.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- pandas
- numpy
- plotly
- scikit-learn
- torch
- streamlit

Install the required packages using:

\`\`\`bash
pip install pandas numpy plotly scikit-learn torch streamlit
\`\`\`

### Data

The dataset is expected to be in the `data/` directory with the following files:
- `train.csv`
- `test.csv`

## train.py

### Description

\`train.py\` handles the following tasks:
1. **Load and Inspect Data**: Load training and test datasets, and perform basic data inspection.
2. **Data Cleaning**: Check for duplicates and missing values.
3. **Exploratory Data Analysis**: Generate visualizations for activity distribution and feature analysis.
4. **Feature Selection**: Identify and select highly correlated features.
5. **Model Training**: Train a Random Forest classifier and a CNN model, evaluate their performance, and identify important features.

### Usage

Run the script using:

\`\`\`bash
python train.py
\`\`\`

### Key Functions

- Data loading and inspection
- Exploratory data analysis (EDA) using Plotly for visualizations
- Feature selection based on correlation
- Random Forest and CNN model training
- Model evaluation and feature importance analysis

## app.py

### Description

\`app.py\` is a Streamlit app for evaluating the trained model and visualizing test set results. It includes:
1. **Model Evaluation**: Evaluate the model on the test set and display loss and accuracy.
2. **Activity Distribution**: Visualize the distribution of activities in the test set.
3. **Individual Instance Evaluation**: Select and evaluate individual instances from the test set.
4. **Feature Exploration**: Interactive feature distribution visualization.

### Usage

Run the app using:

\`\`\`bash
streamlit run app.py
\`\`\`

### Key Features

- Displays test set summary including shape, loss, and accuracy
- Visualizes activity distribution in the test set
- Allows evaluation of individual test instances
- Provides interactive feature exploration

## Directory Structure

\`\`\`
.
├── data/
│   ├── train.csv
│   ├── test.csv
├── train.py
├── app.py
└── README.md
\`\`\`

## Notes

- Ensure the dataset files are in the \`data/\` directory.
- The trained model and scaler should be saved as \`model.pth\` and \`scaler.pkl\` respectively for use in the Streamlit app.

## Acknowledgments

This project uses the Human Activity Recognition dataset.
