import pandas as pd

# Load the dataset
data = pd.read_csv('data/linkedin-jobs-canada.csv')

# Handle missing values
data.fillna('', inplace=True)

# Refined feature engineering: create a binary target variable 'register'
def registration_likelihood(row):
    keywords = ['Data', 'Engineer', 'Scientist', 'Developer', 'Manager']
    if any(keyword in row['title'] for keyword in keywords):
        return 1
    if row['company'] in ['Company A', 'Company B'] and row['location'] in ['Toronto, ON', 'Vancouver, BC']:
        return 1
    return 0

data['register'] = data.apply(registration_likelihood, axis=1)

# Check the distribution of the target variable
print(data['register'].value_counts())

# Balance the dataset
majority_class = data[data['register'] == 1]
minority_class = data[data['register'] == 0]

# Downsample the majority class
majority_downsampled = majority_class.sample(len(minority_class), random_state=42)

# Combine the downsampled majority class with the minority class
balanced_data = pd.concat([majority_downsampled, minority_class])

# Save the balanced dataset
balanced_data.to_csv('data/linkedin-jobs-canada-balanced.csv', index=False)

# Check the distribution of the target variable in the balanced dataset
print(balanced_data['register'].value_counts())
