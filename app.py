from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and columns
model = joblib.load('best_rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df_encoded = pd.get_dummies(df)

    # Reindex the dataframe to ensure it has the same columns as the training data
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df_encoded)
    return jsonify({'joined_picsume': int(prediction[0])})


if __name__ == '__main__':
    app.run(debug=True)
