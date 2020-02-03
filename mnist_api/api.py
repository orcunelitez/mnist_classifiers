import pickle


from flask import Flask, request, redirect, url_for, flash, jsonify
from flask_cors import CORS
import numpy as np
import pickle as p
import json
from simple_convnet import convnet
from alexnet import alexnet
from perceptron import mlp
from keras import backend as K


app = Flask(__name__)
CORS(app)

@app.route('/api/', methods=['GET'])
def api_guide():
    with open("readme.md", "r") as f:
        content = f.read()
    return content

@app.route('/api/test_model/', methods=['POST'])
def predict_model(): 
    try:
        modelNum = int(request.args.get("model"))
    except:
        return "A correct model is not selected"
    K.clear_session()
    if modelNum is None:
        return "A correct model is not selected"
    elif modelNum == 1:
        model = p.load(open("models/model1", 'rb'))
        test_result = model.test()
        return json.dumps({"accuracy": test_result,
        "training_history": str(model.hist.history)})
    elif modelNum == 2:
        model = p.load(open("models/model2", 'rb'))
        test_result = model.test()
        return json.dumps({"accuracy": test_result,
        "training_history": str(model.hist.history)})
    elif modelNum == 3:
        model = p.load(open("models/model3", 'rb'))
        test_result = model.test()
        return json.dumps({"accuracy": test_result,
        "training_history": str(model.hist.history)})
    
    else:
        return "A correct model is not selected"


if __name__ == '__main__':
    app.run(host='0.0.0.0')