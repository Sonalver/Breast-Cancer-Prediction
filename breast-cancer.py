# IMPORT LIBRARIES
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# LOAD DATA
df = pd.read_csv('C:/Users/sonal/PycharmProjects/PythonProject/resources/breast_cancer.csv')
print(df.head())

# SHAPE AND NULL CHECK
print(df.shape)
print(df.isnull().sum())

# DROP NULL COLUMNS
df.dropna(axis=1, inplace=True)
print("Shape after dropping null columns:", df.shape)

# DIAGNOSIS COUNT (M / B)
print(df['diagnosis'].value_counts())

# LABEL ENCODING (M = 1, B = 0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

print(df.head())
print("Encoded Classes:", le.classes_)


# EDA
# 1️⃣ Diagnosis Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=df, palette='coolwarm')
plt.title("Diagnosis Distribution (0 = Benign, 1 = Malignant)")
plt.show()

# 2️⃣ Feature Histograms
df.hist(figsize=(18, 12), bins=30)
plt.tight_layout()
plt.show()

# 3️⃣ Pairplot of Top Predictors
corr = df.corr()['diagnosis'].abs().sort_values(ascending=False)
top_features = corr.index[1:6].tolist()

print("Top Predictors Based on Correlation:", top_features)

sns.pairplot(df[top_features + ['diagnosis']], hue='diagnosis', palette='coolwarm')
plt.show()

# 4️⃣ Full Correlation Heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Full Feature Correlation Heatmap")
plt.show()

# 5️⃣ Outlier Identification
# A) Boxplots for important features
key_cols = df.columns[2:12]   # First 10 features

plt.figure(figsize=(16,10))
for i, column in enumerate(key_cols, 1):
    plt.subplot(4, 3, i)
    sns.boxplot(y=df[column], color='lightblue')
    plt.title(f'Boxplot: {column}')
plt.tight_layout()
plt.show()

# B) IQR Method for Outlier Counts
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()

print("Outliers per feature:\n", outliers)

# M O D E L   B U I L D I N G
# SPLIT INTO X AND Y
X = df.iloc[:, 2:].values
Y = df.iloc[:, 1].values

# TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# TRAIN MODEL
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=5000)
classifier.fit(X_train, Y_train)

# PREDICTIONS
predictions = classifier.predict(X_test)

# CONFUSION MATRIX + HEATMAP
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(Y_test, predictions)
print("Confusion Matrix:\n", cm)

sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix Heatmap")
plt.show()

# ACCURACY
accuracy = accuracy_score(Y_test, predictions)
print("Model Accuracy:", accuracy)

print("Predictions:", predictions)

# IMPORTANT MORPHOLOGICAL FEATURES
import numpy as np

feature_names = df.columns[2:]
coefficients = classifier.coef_[0]

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(coefficients)
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Features:\n", importance_df.head(10))

plt.figure(figsize=(10,5))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', palette='viridis')
plt.title("Top Morphological Predictors")
plt.show()
