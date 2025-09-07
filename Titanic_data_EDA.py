# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")

# Load datasets
train_df = pd.read_csv(r'C:\Users\Ruchi\Downloads\train.csv')
test_df = pd.read_csv(r'C:\Users\Ruchi\Downloads\test.csv')
gender_submission_df = pd.read_csv(r'C:\Users\Ruchi\Downloads\gender_submission.csv')

# --------------------------
# Data Cleaning
# --------------------------

# Fill missing 'Age' with median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)
test_df['Age'].fillna(test_df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with mode in train set
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Fill missing 'Fare' with median in test set
test_df['Fare'].fillna(test_df['Fare'].median(), inplace=True)

# Drop 'Cabin' as it has too many missing values
train_df.drop(columns=['Cabin'], inplace=True)
test_df.drop(columns=['Cabin'], inplace=True)

# Encode 'Sex' column
train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})

# One-hot encode 'Embarked'
train_df = pd.get_dummies(train_df, columns=['Embarked'], drop_first=True)
test_df = pd.get_dummies(test_df, columns=['Embarked'], drop_first=True)

# --------------------------
# Exploratory Data Analysis
# --------------------------

# 1. Survival distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Survived')
plt.title('Survival Distribution')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# 2. Survival by Sex
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Sex', hue='Survived')
plt.title('Survival by Sex')
plt.xlabel('Sex (0=Male, 1=Female)')
plt.ylabel('Count')
plt.show()

# 3. Survival by Pclass
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Pclass', hue='Survived')
plt.title('Survival by Passenger Class')
plt.xlabel('Pclass')
plt.ylabel('Count')
plt.show()

# 4. Age distribution by Survival
plt.figure(figsize=(8,5))
sns.kdeplot(data=train_df[train_df['Survived']==0], x='Age', label='Not Survived', fill=True)
sns.kdeplot(data=train_df[train_df['Survived']==1], x='Age', label='Survived', fill=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.ylabel('Density')
plt.legend()
plt.show()

# 5. Fare distribution by Survival
plt.figure(figsize=(8,5))
sns.kdeplot(data=train_df[train_df['Survived']==0], x='Fare', label='Not Survived', fill=True)
sns.kdeplot(data=train_df[train_df['Survived']==1], x='Fare', label='Survived', fill=True)
plt.title('Fare Distribution by Survival')
plt.xlabel('Fare')
plt.ylabel('Density')
plt.legend()
plt.show()
