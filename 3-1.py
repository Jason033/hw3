import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Create data according to the given rule
X = np.arange(0, 1000, 1).reshape(-1, 1)  # X values from 0 to 999
Y = np.where((X > 500) & (X < 800), 1, 0).flatten()  # Label 1 for 500 < X < 800, else 0

# 2. Train Logistic Regression and SVM
log_reg = LogisticRegression()
log_reg.fit(X, Y)
svm = SVC(kernel='rbf', probability=True)
svm.fit(X, Y)

# 3. Predict outputs for each model
Y1_pred = log_reg.predict(X)
Y2_pred = svm.predict(X)

# 4. Plot results and decision boundaries
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Logistic Regression plot
ax1.scatter(X, Y, color='blue', label='True Labels')
ax1.plot(X, Y1_pred, color='red', label='Logistic Regression Prediction')
ax1.axvline(x=500, color='green', linestyle='--', label='Decision Boundary')
ax1.axvline(x=800, color='green', linestyle='--')
ax1.set_title('Logistic Regression Decision Boundary')
ax1.legend()

# SVM plot
ax2.scatter(X, Y, color='blue', label='True Labels')
ax2.plot(X, Y2_pred, color='purple', label='SVM Prediction')
ax2.axvline(x=500, color='green', linestyle='--', label='Decision Boundary')
ax2.axvline(x=800, color='green', linestyle='--')
ax2.set_title('SVM Decision Boundary')
ax2.legend()

plt.show()
