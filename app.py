from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows frontend to communicate with backend

# Load pre-trained ML model
pickle.dump(model,open("food_predict.pkl", "wb"))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(input_features)[0]

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
