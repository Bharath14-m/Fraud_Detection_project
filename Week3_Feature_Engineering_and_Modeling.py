# WEEK 3: Feature Engineering & Baseline Modeling (Fixed for Kaggle Dataset)

# STEP 1: Import Libraries & Load Week 2 Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

sns.set(style='whitegrid')

# Load the Week 2 dataset
df = pd.read_csv('transactions_week2_ready.csv')
print("✅ Data Loaded Successfully")
print(df.shape)
print(df.head())

# STEP 2: Feature Engineering (for Kaggle Credit Card Fraud dataset)

# 1️⃣ Time-based features
df['Hour'] = (df['Time'] / 3600) % 24  # Convert seconds to hour of day

# 2️⃣ Log-transform skewed variables
df['Log_Amount'] = np.log1p(df['Amount'])

# 3️⃣ Create an interaction term
df['V1_Amount_Interaction'] = df['V1'] * df['Amount']

print("✅ Feature Engineering Complete")

# STEP 3: Train/Test Split
X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# STEP 4: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Feature Scaling Complete")

# STEP 5: Train Logistic Regression Model
model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)
print("✅ Model Training Complete")

# STEP 6: Evaluate Model
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# STEP 7: ROC Curve Plot
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend()
plt.show()

# STEP 8: Feature Importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(model.coef_[0])
}).sort_values(by='Coefficient', ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x='Coefficient', y='Feature', data=importance.head(10))
plt.title('Top 10 Important Features (Logistic Regression)')
plt.show()

# STEP 9: Save Week 3 Output
df.to_csv('transactions_week3_ready.csv', index=False)
print("✅ Week 3 dataset saved successfully as transactions_week3_ready.csv")
