import pickle
import xgboost as xgb
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_1.bin'

with open (model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app= Flask('subscribe')
@app.route('/predict', methods =['POST'])


def predict():
        
    customer= request.get_json()
    X=dv.transform([customer])

    x=xgb.DMatrix(X, label=([0]), feature_names=dv.get_feature_names())


    pred=model.predict(x)

    result ={
        'The prediction of this customer to subscribe = ': float(pred[0])
    }
    return jsonify(result)





if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)