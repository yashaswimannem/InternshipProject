import joblib  # Import joblib to save the model

# Existing code to train and evaluate the model...
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the Dataset
url = "https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Big%20Sales%20Data.csv"
data = pd.read_csv(url)

# Step 2: Data Exploration and Preprocessing
# Display the first few rows of the data
print("Data Preview:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Drop rows with missing values
data = data.dropna()

# Display data types and basic information
print("\nData Information:")
print(data.info())

# Convert categorical variables to dummy variables
data = pd.get_dummies(data, drop_first=True)

# Step 3: Feature Selection
# Assuming 'Sales' is the target variable
X = data.drop("Item_Outlet_Sales", axis=1)
y = data["Item_Outlet_Sales"]

# Step 4: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Model
# Initialize and fit the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 6: Make Predictions and Evaluate the Model
# Predict on the test set
y_pred = rf_model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R^2 Score: {r2}")

# Step 7: Identify High Revenue Items
# Add predictions to the test set for analysis
test_results = X_test.copy()
test_results['Predicted_Sales'] = y_pred

# Sort by predicted sales in descending order
high_revenue_items = test_results.sort_values(by='Predicted_Sales', ascending=False)

print("\nTop High Revenue Items (Predicted):")
print(high_revenue_items.head())

# Optional: Save results to a CSV
high_revenue_items.to_csv("High_Revenue_Items.csv", index=False)
print("\nHigh revenue items saved to 'High_Revenue_Items.csv'.")

# Step 8: Save the model as a .pkl file
joblib.dump(rf_model, 'randomforest.pkl')
print("\nModel saved as 'randomforest.pkl'.")

import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Predicted vs Actual Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # Ideal line
plt.title('Predicted vs Actual Sales')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.show()

# Step 2: Error Distribution Plot
errors = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(errors, kde=True, color='purple')
plt.title('Error Distribution')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.show()

# Step 3: Feature Importance Plot
feature_importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances, color='green')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

