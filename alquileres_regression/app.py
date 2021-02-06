#!/usr/bin/env python
'''
API Machine Learning
---------------------------
Autor: Inove Coding School
Version: 1.0
 
Descripcion:
Se utiliza Flask para crear un WebServer que levanta un
modelo de inteligencia artificial con machine learning
y realizar predicciones o clasificaciones

Ejecución: Lanzar el programa y abrir en un navegador la siguiente dirección URL
http://127.0.0.1:5000/

'''

__author__ = "Inove Coding School"
__email__ = "INFO@INOVE.COM.AR"
__version__ = "1.0"

import traceback
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('alquileres.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    try:
        m2 = float(request.form.get('m2'))
        ambientes = int(request.form.get('ambientes'))

        features = [m2, ambientes]
        numpy_features = [np.array(features)]
        prediction = model.predict(numpy_features)

        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'El valor del alquiler debería ser ${output}')
    except:
        return jsonify({'trace': traceback.format_exc()})

if __name__ == "__main__":
    app.run(debug=True)