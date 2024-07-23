import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the data
hill_valley_data = pd.read_csv('hill_valley_data.csv')  # Assuming you have this CSV file

# Separate features and target
X = hill_valley_data.drop('class', axis=1)
y = hill_valley_data['class']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
plt.ylabel("Value")
plt.show()
