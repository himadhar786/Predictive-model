from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and columns
model = joblib.load('data/best_model.pkl')
model_columns = joblib.load('data/model_columns.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)  # Force=True to ensure that JSON is correctly parsed
        df = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df)

        # Reindex the dataframe to ensure it has the same columns as the training data
        df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(df_encoded)
        return jsonify({'register': int(prediction[0])})  # Adjust the key as needed
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
