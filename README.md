# 💳 Credit Card Fraud Detection Project

## 📘 Overview
This project aims to detect fraudulent credit card transactions using machine learning and visualize fraud patterns through an interactive Power BI dashboard.  
The project follows a **4-week structured workflow** covering data preprocessing, exploratory data analysis (EDA), feature engineering, model development, and business intelligence reporting.

---

## 🗂️ Dataset
**Source:** [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- Contains transactions made by European cardholders in September 2013.
- Total records: **284,807**
- Fraudulent transactions: **492 (0.17%)**
- Highly imbalanced dataset.

---

## 📅 Project Timeline

### 🧩 Week 1: Data Preparation & Initial Exploration
- Extracted data using **SQLAlchemy**.
- Loaded the dataset into a Pandas DataFrame.
- Performed basic profiling using `.info()` and `.describe()`.
- Checked for missing values and data types.
- Analyzed **class imbalance** between fraudulent and legitimate transactions.
- Saved cleaned data as:




---

### 🔍 Week 2: Exploratory Data Analysis (EDA)
- Conducted **univariate** and **bivariate** analyses:
- Distribution of `Amount`, `Time`, and `Class`.
- Comparison of fraudulent vs legitimate transaction patterns.
- Used visualizations:
- Histograms, boxplots, scatterplots, and correlation heatmaps.
- Derived insights on how fraud varies by **transaction amount** and **time of day**.
- Saved processed dataset as:




---

### ⚙️ Week 3: Feature Engineering & Baseline Modeling
- Created new engineered features:
- `Transaction_Hour` (derived from time)
- `Transaction_Count` (per user)
- `Average_Transaction_Amount`
- `Time_Since_Last_Transaction`
- Built a **Logistic Regression** baseline model.
- Evaluated using:
- Precision, Recall, F1-Score, ROC-AUC.
- Identified top predictors of fraud from model coefficients.
- Exported files:



---

### 📊 Week 4: Analysis, Reporting & Dashboard (Power BI)
- Imported model results and data into **Power BI**.
- Created an interactive **Dark Analytics Dashboard** featuring:
1. **KPI Cards** — Total Transactions, Fraud Count, Fraud Percentage.
2. **Donut Chart** — Fraud vs Legit Share.
3. **Line Chart** — Fraud Rate by Hour.
4. **Top Predictors Bar Chart** — Model feature importance.
5. **Geographic & Temporal Insights** (optional enhancements).
- Dashboard title:  
**"FRAUD DETECTION DASHBOARD"**

---

## 🧠 Key Insights
- Fraud transactions typically occur during **odd hours (midnight–early morning)**.
- Fraudulent amounts tend to be **slightly higher** on average.
- Top predictors of fraud include specific principal components (from PCA) and transaction time patterns.

---

## 🧰 Tech Stack
| Category | Tools Used |
|-----------|-------------|
| Programming | Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn) |
| Database | SQLite (via SQLAlchemy) |
| Visualization | Power BI |
| Modeling | Logistic Regression |
| Environment | Jupyter Notebook / VS Code |

---

## 📁 Repository Structure
