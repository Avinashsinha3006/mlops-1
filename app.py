from flask import Flask, jsonify, request
import json
import pickle
from src.models.preprocess import preprocess

webapp_root = "webapp"
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            data = request.json
            test_x = preprocess(data)
            with open('prediction_service/model/finalized_model.sav', 'rb') as f:
                rf_model = pickle.load(f)
            predictions = rf_model.predict(test_x)
            result = json.dumps({'result': predictions.tolist()})

        except Exception as e:
            result = jsonify({'error': str(e)})
        return result


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
