from flask import Flask, request, jsonify
from catboost import CatBoostRegressor
import pickle
import pandas as pd

app = Flask(__name__)

# Create a model CatBoostRegressor
model = CatBoostRegressor(
    learning_rate=0.26767727312827677,
    depth=10,
    l2_leaf_reg=2.320477868945964,
    iterations=911,
    random_state=42
)

# Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load the model from a file
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a server on Flask
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.read_json(data['df'])  # Parse the string into a DataFrame
        X = df.values  # Convert the DataFrame to a numpy array
        y_pred = model.predict(X)
        return jsonify({'y_pred': y_pred.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)