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
from flask import Flask, request, jsonify, render_template, url_for

app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl', 'rb'))
le = pickle.load(open('titanic_encoder.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


# Cuando se presione el botón el sistema llama a esta función
# para realizar la predicción con el modelo de inteligencia artificial
@app.route('/predict',methods=['POST'])
def predict():
    try:
        pclass = int(request.form.get('class'))
        sex = str(request.form.get('sex'))
        age = int(request.form.get('age'))
        siblings = int(request.form.get('siblings'))
        parch = int(request.form.get('parch'))

        # Castear el array que devuelve le.transform a int
        sex_encoded = int(le.transform([sex]))
        # Crear el array de entrada
        features = np.array([pclass, sex_encoded, age, siblings, parch])
        # El sistema espera 1 fila y N columnas, hay que hacer reshape
        numpy_features = features.reshape(1, -1)
        prediction = model.predict(numpy_features)

        if prediction[0] == 1:
            image = url_for('static', filename='media/titanic_survivors.jpg')
            return render_template('index.html', prediction_text='Sobreviviste!', prediction_image=image)
        else:
            image = url_for('static', filename='media/new_life.jpg')
            return render_template('index.html', prediction_text='Game Over!', prediction_image=image)
        
    except Exception as e:
        print(e)
        return render_template('index.html', prediction_text='Datos mal ingresados')

if __name__ == "__main__":
    app.run(debug=True)