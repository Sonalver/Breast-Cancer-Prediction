# Breast-Cancer-Prediction
Breast cancer prediction using logistic regression machine learning model, using UCI machine learning repository dataset 
A complete machine learning workflow to classify breast tumors as Benign (0) or Malignant (1) using the Wisconsin Breast Cancer Dataset.

🚀 Features in This Project

Data loading & preprocessing
Diagnostic Exploratory Data Analysis (EDA)
Outlier detection
Correlation-based feature insights
Logistic Regression model
Model evaluation (Accuracy + Confusion Matrix)
Morphological feature importance visualization

📊 Exploratory Data Analysis (EDA)

Includes:
✔ Diagnosis Distribution
Shows count of benign vs malignant tumors.

✔ Feature Histograms
Distribution of all numeric features.

✔ Pairplot of Top Predictors
Top 5 features automatically selected using correlation.

✔ Full Correlation Heatmap
Identifies strongest linear relationships.

✔ Outlier Detection
Boxplots (first 10 numeric features)
IQR-based outlier count per feature

✔ Important Morphological Features
Based on Logistic Regression coefficients (radius_mean, concave points_mean, perimeter_mean, etc.)

🤖 Model Pipeline
Drop null columns
Encode diagnosis (M→1, B→0)
Split dataset (75% train / 25% test)
Scale features using StandardScaler
Train Logistic Regression (max_iter=5000)
Evaluate performance

Outputs:
Confusion Matrix (visualized)
Accuracy Score
Predictions
Feature Importance Plot

📈 Typical Performance
Accuracy: ~94–97%
Strong predictive features: concave points_mean, radius_mean, area_mean, perimeter_mean

Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
