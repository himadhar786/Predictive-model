import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load the dataset
data = pd.read_csv('data/candidate_data.csv')

# Fill missing values
data.fillna('', inplace=True)

# Encode categorical variables
X = pd.get_dummies(data[['skills', 'experience', 'job_title', 'social_media_activity']])
columns = X.columns  # Save columns for future use

# Target variable
y = data['joined_picsume']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

# Best model
best_rf = grid_search.best_estimator_

# Model evaluation
y_pred = best_rf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1-Score: {f1_score(y_test, y_pred)}')

# Save the model and columns
joblib.dump(best_rf, 'best_rf_model.pkl')
joblib.dump(columns, 'model_columns.pkl')
