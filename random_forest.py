from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

train_df = preprocess_data()
X = train_df.drop(['id', 'cycle', 'label1', 'label2'], axis=1)
y = train_df['RUL']

# Split data into training and testing sets (70% training, 30% testing by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestRegressor(n_estimators=100)  # Adjust n_estimators as needed
model.fit(X_train, y_train)

# Make predictions on the test data (X_test)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)