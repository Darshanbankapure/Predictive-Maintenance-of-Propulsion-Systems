import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from preprocess import preprocess_data

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, r2_score

train_df = preprocess_data()
X = train_df.drop(['id', 'cycle', 'label1', 'label2'], axis=1)
y = train_df['RUL']

# Split data into training and testing sets (70% training, 30% testing by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the SVM model
model = SVR() 
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate model performance using accuracy metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R-squared:", r2)


