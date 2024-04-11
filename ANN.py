from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data
from sklearn.metrics import mean_squared_error, r2_score

train_df = preprocess_data()
X = train_df.drop(['id', 'cycle', 'label1', 'label2'], axis=1)
y = train_df['RUL']

# Split data into training and testing sets (70% training, 30% testing by default)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = keras.Sequential([
  layers.Dense(units=128, activation="relu", input_shape=(X_train.shape[1],)),  # Adjust units as needed
  layers.Dense(units=64, activation="relu"),
  layers.Dense(units=1)  # Output layer with 1 unit for regression
])

# Compile the model with optimizer (e.g., adam) and loss function (mean squared error)
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=10, batch_size=32)

y_pred = model.predict(X_test)

# Calculate MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Mean Squared Error:", mse)
print("R-squared:", r2)