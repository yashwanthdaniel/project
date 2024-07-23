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
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize a sample prediction
sample = X_test.iloc[0]
plt.plot(sample)
plt.title(f"Predicted: {'Hill' if model.predict([sample])[0] == 1 else 'Valley'}")
plt.xlabel("Feature")
plt.ylabel("Value")
plt.show()
