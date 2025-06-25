pip install pandas numpy scikit-learn tensorflow matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

data = pd.read_csv("cardio_B.csv")
data.head()

data.columns

num_duplicates = data.duplicated().sum()
data = data.drop_duplicates(keep='first')

# data = data.drop('Unnamed: 0', axis=1)

from sklearn.impute import SimpleImputer

#Handling Missing Values:
numerical_features = data.select_dtypes(include=['number']).columns
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_features] = numerical_imputer.fit_transform(data[numerical_features])

# Convert age from days to years
data['age_years'] = (data['age'] / 365).astype(int)

# Calculate BMI = weight (kg) / height (m)^2
data['BMI'] = data['weight'] / ((data['height'] / 100) ** 2)

# Calculate systolic - diastolic blood pressure difference
data['ap_diff'] = data['ap_hi'] - data['ap_lo']

# Remove outliers where diastolic > systolic (impossible readings)
data = data[data['ap_hi'] > data['ap_lo']]
data = data[(data['ap_hi'] < 250) & (data['ap_lo'] > 50)]

# Reset index after cleaning
data = data.reset_index(drop=True)

# Optional: Drop original age if you only want to use age_years
data = data.drop('age', axis=1)

# Check correlation after adding new features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap After Feature Engineering')
plt.show()

data.tail()

for column in data.columns:
    # Check if the column is numeric
    if data[column].dtype in [np.int64, np.float64]:
        min_val = data[column].min()
        max_val = data[column].max()
        print(f"'{column}': Range [{min_val}, {max_val}]")
    else:
        unique_vals = data[column].unique()
        print(f"'{column}': Unique values {unique_vals}")

print(data.info())

print(data.describe())

# Visualize the data
sns.pairplot(data, hue='cardio')
plt.show()

features = data.drop('cardio', axis=1)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, data['cardio'], test_size=0.2, random_state=42
)

# 1. Isolation Forest
# iso_forest = IsolationForest(contamination=0.1)
# iso_pred = iso_forest.fit_predict(features_scaled)
# data['Isolation_Forest'] = (iso_pred == -1).astype(int)  # Mark anomalies as 1
# data['Isolation_Forest']

iso_forest = IsolationForest(
    n_estimators=300,       # Increased number of trees
    max_samples=0.8,        # Use 80% of samples
    contamination=0.1,      # Approximate contamination rate
    max_features=0.7,       # Use 70% of features for each tree
    bootstrap=True,         # Use bootstrapping for variability
    random_state=42         # Reproducibility
)
iso_pred = iso_forest.fit_predict(features_scaled)
data['Isolation_Forest'] = (iso_pred == -1).astype(int)
data['Isolation_Forest']

# 2. One-Class SVM (remove contamination parameter, it doesn't exist)
oc_svm = OneClassSVM(gamma='auto')
oc_svm_pred = oc_svm.fit_predict(features_scaled)
data['One_Class_SVM'] = (oc_svm_pred == -1).astype(int)  # Mark anomalies as 1
data['One_Class_SVM']

# 3. Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_pred = lof.fit_predict(features_scaled)
data['Local_Outlier_Factor'] = (lof_pred == -1).astype(int)  # Mark anomalies as 1
data['Local_Outlier_Factor']

# 4. Autoencoder
def create_autoencoder(input_shape):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(input_shape, activation='linear'))
    return model

autoencoder = create_autoencoder(features_scaled.shape[1])
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the Autoencoder
autoencoder.fit(features_scaled, features_scaled, epochs=80, batch_size=250, shuffle=True, validation_split=0.1)

# Use the autoencoder to find anomalies
reconstructed = autoencoder.predict(features_scaled)
mse = np.mean(np.power(features_scaled - reconstructed, 2), axis=1)
data['Autoencoder'] = mse

# Define threshold for anomalies
threshold = np.percentile(data['Autoencoder'], 95)  # 95th percentile as anomaly threshold
data['Autoencoder_Anomaly'] = (data['Autoencoder'] > threshold).astype(int)

# Show results
print(data[['cardio', 'Isolation_Forest', 'One_Class_SVM', 'Local_Outlier_Factor', 'Autoencoder_Anomaly']])

# Evaluate the anomaly detection (optional: based on the 'target' column)
print("\nIsolation Forest Classification Report:")
print(classification_report(data['cardio'], data['Isolation_Forest']))

print("\nOne-Class SVM Classification Report:")
print(classification_report(data['cardio'], data['One_Class_SVM']))

print("\nLocal Outlier Factor Classification Report:")
print(classification_report(data['cardio'], data['Local_Outlier_Factor']))

print("\nAutoencoder Anomaly Classification Report:")
print(classification_report(data['cardio'], data['Autoencoder_Anomaly']))

# Visualize the anomalies
plt.figure(figsize=(10, 6))
sns.countplot(x='cardio', hue='Isolation_Forest', data=data)
plt.title('Isolation Forest Anomalies')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='cardio', hue='One_Class_SVM', data=data)
plt.title('One-Class SVM Anomalies')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='cardio', hue='Local_Outlier_Factor', data=data)
plt.title('Local Outlier Factor Anomalies')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='cardio', hue='Autoencoder_Anomaly', data=data)
plt.title('Autoencoder Anomalies')
plt.show()

# Set a custom style for the plots
sns.set(style="whitegrid")

# Create a figure with subplots (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))  # 2 rows, 2 columns

# Adjust the layout for better spacing between subplots
plt.subplots_adjust(hspace=0.3, wspace=0.3)

# Define a color palette for each subplot
palette_isolation = sns.color_palette("coolwarm", 2)
palette_svm = sns.color_palette("Spectral", 2)
palette_lof = sns.color_palette("viridis", 2)
palette_autoencoder = sns.color_palette("Set2", 2)

# Function to add value annotations to bars
def annotate_bars(ax):
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10, color='black')

# Isolation Forest Anomalies
ax1 = sns.countplot(x='cardio', hue='Isolation_Forest', data=data, ax=axes[0, 0], palette=palette_isolation)
axes[0, 0].set_title('Isolation Forest Anomalies', fontsize=16, fontweight='bold')
axes[0, 0].set_xlabel('Heart Disease Target', fontsize=12)
axes[0, 0].set_ylabel('Count', fontsize=12)
annotate_bars(ax1)

# One-Class SVM Anomalies
ax2 = sns.countplot(x='cardio', hue='One_Class_SVM', data=data, ax=axes[0, 1], palette=palette_svm)
axes[0, 1].set_title('One-Class SVM Anomalies', fontsize=16, fontweight='bold')
axes[0, 1].set_xlabel('Heart Disease Target', fontsize=12)
axes[0, 1].set_ylabel('Count', fontsize=12)
annotate_bars(ax2)

# Local Outlier Factor Anomalies
ax3 = sns.countplot(x='cardio', hue='Local_Outlier_Factor', data=data, ax=axes[1, 0], palette=palette_lof)
axes[1, 0].set_title('Local Outlier Factor Anomalies', fontsize=16, fontweight='bold')
axes[1, 0].set_xlabel('Heart Disease Target', fontsize=12)
axes[1, 0].set_ylabel('Count', fontsize=12)
annotate_bars(ax3)

# Autoencoder Anomalies
ax4 = sns.countplot(x='cardio', hue='Autoencoder_Anomaly', data=data, ax=axes[1, 1], palette=palette_autoencoder)
axes[1, 1].set_title('Autoencoder Anomalies', fontsize=16, fontweight='bold')
axes[1, 1].set_xlabel('Heart Disease Target', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
annotate_bars(ax4)

# Final adjustments to make the plot look fancy
for ax in axes.flat:
    ax.legend(title='Anomaly Detection', loc='upper right', fontsize=10, title_fontsize='13')
    ax.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7)

# Show the plot
plt.show()

# Prepare the data for a stacked bar plot
methods = ['Isolation_Forest', 'One_Class_SVM', 'Local_Outlier_Factor', 'Autoencoder_Anomaly']

# Count the number of 0's and 1's for each method (Normal and Anomalies)
anomaly_counts = {method: data[method].value_counts() for method in methods}

# Create a DataFrame to hold the counts for plotting
anomaly_df = pd.DataFrame(anomaly_counts)
anomaly_df = anomaly_df.T  # Transpose the DataFrame to make methods the rows

# Plot the stacked bar chart
anomaly_df.plot(kind='bar', stacked=True, figsize=(12, 8), color=['#3498db', '#e74c3c'])

# Add labels and title
plt.title('Stacked Bar Plot of Anomalies vs Normal Instances Detected by Methods', fontsize=16, fontweight='bold')
plt.xlabel('Anomaly Detection Methods', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(['Normal (0)', 'Anomaly (1)'], loc='upper right', fontsize=10)

# Show the plot
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Select the anomaly detection columns
anomaly_methods = ['Isolation_Forest', 'One_Class_SVM', 'Local_Outlier_Factor', 'Autoencoder_Anomaly']

# Calculate the correlation matrix
correlation_matrix = data[anomaly_methods].corr()

# Create a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)

# Customize the plot
plt.title('Correlation Heatmap of Anomaly Detection Methods', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

# Show the plot
plt.show()

models = ['Isolation_Forest', 'One_Class_SVM', 'Local_Outlier_Factor', 'Autoencoder_Anomaly']

results = []

for model in models:
    print(f"\n📈 Evaluating {model}...")

    report = classification_report(data['cardio'], data[model], output_dict=True)
    accuracy = accuracy_score(data['cardio'], data[model])
    f1 = report['1']['f1-score'] if '1' in report else 0.0
    roc_auc = roc_auc_score(data['cardio'], data[model])

    results.append({
        'Model': model,
        'Accuracy': accuracy,
        'F1-Score': f1,
        'ROC-AUC': roc_auc
    })


results_df = pd.DataFrame(results)

# Sort models:
# - First by F1-Score (higher is better)
# - Then by ROC-AUC (higher is better)
results_df = results_df.sort_values(by=['F1-Score', 'ROC-AUC'], ascending=False)

print("\n🔵 Model Rankings (sorted by F1-Score and ROC-AUC):")
print(results_df)

best_model = results_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['Model']}")
print(f"   → Accuracy: {best_model['Accuracy']:.4f}")
print(f"   → F1-Score: {best_model['F1-Score']:.4f}")
print(f"   → ROC-AUC: {best_model['ROC-AUC']:.4f}")

import matplotlib.pyplot as plt

# Normalize column names to avoid whitespace issues
results_df.columns = results_df.columns.str.strip()

# Check if 'Accuracy' column exists
if 'Accuracy' in results_df.columns:
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    positions = range(len(results_df))

    plt.bar(positions, results_df['Accuracy'], width=bar_width, label='Accuracy', color='skyblue')
    plt.xlabel('Models', fontweight='bold', fontsize=12)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.xticks(positions, results_df['Model'], rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
else:
    print("Column 'Accuracy' not found in results_df.")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix

models = ['Isolation_Forest', 'One_Class_SVM', 'Local_Outlier_Factor', 'Autoencoder_Anomaly']

for model in models:
    y_true = data['cardio']
    y_pred = data[model]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    confusion_df = pd.DataFrame({
        'Predicted Positive': [f'TP = {tp}', f'FP = {fp}'],
        'Predicted Negative': [f'FN = {fn}', f'TN = {tn}']
    }, index=['Actual Positive', 'Actual Negative'])

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')

    table = ax.table(cellText=confusion_df.values,
                     colLabels=confusion_df.columns,
                     rowLabels=confusion_df.index,
                     loc='center',
                     cellLoc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)

    plt.title(f'Confusion Matrix of {model.replace("_", " ")}', fontsize=14, fontweight='bold')
    plt.show()