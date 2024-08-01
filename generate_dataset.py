import pandas as pd
import random

# Seed for reproducibility
random.seed(42)

# Sample data
skills_list = [
    "Python, Machine Learning", "Java, Spring", "JavaScript, React", "SQL, Data Analysis",
    "C++, Embedded Systems", "R, Statistics", "Ruby, Rails", "HTML, CSS, JavaScript",
    "AWS, DevOps", "TensorFlow, Deep Learning"
]

job_titles = [
    "Data Scientist", "Software Engineer", "Web Developer", "Data Analyst",
    "Systems Engineer", "Statistician", "Backend Developer", "Frontend Developer",
    "DevOps Engineer", "AI Researcher"
]

# Generate synthetic data
data = {
    "skills": [random.choice(skills_list) for _ in range(100)],
    "experience": [random.randint(1, 15) for _ in range(100)],  # years of experience
    "job_title": [random.choice(job_titles) for _ in range(100)],
    "social_media_activity": [random.randint(0, 100) for _ in range(100)],  # activity score out of 100
    "joined_picsume": [random.randint(0, 1) for _ in range(100)]  # 1 if joined, 0 otherwise
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/candidate_data.csv', index=False)

print(df.head())
