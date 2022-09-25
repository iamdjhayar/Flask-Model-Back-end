from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

model = pickle.load(open('model/naive_bayes.pkl', 'rb'))


@app.route('/')
def hello_world():
    return 'This is my first API call!'


@app.route('/predict', methods=["POST"])
def testpost():
    target = request.get_json()
    prediction = model.predict([[target["nitro"], target["phosp"], target["potas"], target["temp"], target["humid"], target["ph"], target["rain"]]])
    print(prediction)
    response = jsonify(message=format(prediction[0]))
    return response
