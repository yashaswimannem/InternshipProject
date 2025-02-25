# app.py
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
# Load the trained model
model = joblib.load("randomforest.pkl")

# Load sample training data to match feature structure
sample_data = pd.read_csv("https://raw.githubusercontent.com/YBIFoundation/Dataset/main/Big%20Sales%20Data.csv")
sample_data = sample_data.dropna()  # Apply any preprocessing that was done on training data
sample_data = pd.get_dummies(sample_data, drop_first=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    item_identifier = request.form['item_identifier']
    item_weight = float(request.form['item_weight'])
    item_mrp = float(request.form['item_mrp'])
    
    # Prepare input data in the required format
    input_data = pd.DataFrame(columns=sample_data.columns.drop("Item_Outlet_Sales"))  # Drop target variable
    input_data.loc[0] = 0  # Initialize all columns to zero
    input_data['Item_Weight'] = item_weight
    input_data['Item_MRP'] = item_mrp
    
    # Set the specific dummy variable column for `Item Identifier`, if it exists
    identifier_column = f"Item_Identifier_{item_identifier}"
    if identifier_column in input_data.columns:
        input_data[identifier_column] = 1

    # Make prediction
    try:
        sales_prediction = model.predict(input_data)
        prediction_text = f"Predicted Sales: {sales_prediction[0]:.2f}"
    except Exception as e:
        prediction_text = "Error making prediction. Check the inputs and model requirements."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
