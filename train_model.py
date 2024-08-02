import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the balanced dataset
data = pd.read_csv('data/linkedin-jobs-canada-balanced.csv')

# Drop the 'salary' column due to many missing values
data = data.drop(columns=['salary'])

# Ensure all data types are correct and there are no NaNs
print(data.info())
print(data.isnull().sum())

# Handle missing values (though ideally there shouldn't be any at this point)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Encode categorical variables
X = pd.get_dummies(data[['title', 'company', 'onsite_remote', 'location']])
columns = X.columns  # Save columns for future use

# Define the target variable
y = data['register']

# Check class distribution
print(y.value_counts())

# Ensure both classes are represented in the train and test sets
if y.value_counts().min() > 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
else:
    print("Not enough data to split while keeping both classes. Using the entire dataset for training.")
    X_train, X_test, y_train, y_test = X, X, y, y

# Train and evaluate the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf)}')
print(f'Precision: {precision_score(y_test, y_pred_rf, zero_division=1)}')
print(f'Recall: {recall_score(y_test, y_pred_rf, zero_division=1)}')
print(f'F1-Score: {f1_score(y_test, y_pred_rf, zero_division=1)}')

# Train and evaluate the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Results:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_gb)}')
print(f'Precision: {precision_score(y_test, y_pred_gb, zero_division=1)}')
print(f'Recall: {recall_score(y_test, y_pred_gb, zero_division=1)}')
print(f'F1-Score: {f1_score(y_test, y_pred_gb, zero_division=1)}')

# Choose the best model based on performance
best_model = rf_model if f1_score(y_test, y_pred_rf, zero_division=1) > f1_score(y_test, y_pred_gb, zero_division=1) else gb_model

# Save the best model and columns
joblib.dump(best_model, 'data/best_model.pkl')
joblib.dump(columns, 'data/model_columns.pkl')
