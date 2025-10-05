from sklearn.linear_model import LinearRegression
import numpy as np
import joblib

# Prepare data: years of experience and salary
X = np.array([[1], [2], [3], [4], [5]])  # Years of experience
y = np.array([30, 35, 40, 45, 50])      # Salary in thousands

# Train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to salary_model.pkl
joblib.dump(model, 'salary_model.pkl')
print('Model saved as salary_model.pkl')
