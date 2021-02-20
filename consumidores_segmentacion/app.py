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
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')   # For multi thread, non-interactive backend (avoid run in main loop)
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from flask import Flask, request, jsonify, render_template, url_for

app = Flask(__name__)

def custumer_overview():
    df = pd.read_csv("Mall_Customers.csv")
    df = df.drop(['CustomerID', 'Gender'], axis=1)

    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], ax=ax)

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded_img = base64.encodebytes(output.getvalue())
    plt.close(fig)  # Cerramos la imagen para que no consuma memoria del sistema
    return encoded_img

def custumer_segmentation(n_clusters):
    df = pd.read_csv("Mall_Customers.csv")
    df = df.drop(['CustomerID', 'Gender'], axis=1)
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=0)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    score = silhouette_score(X, kmeans.labels_)
    df['custseg'] = labels

    fig = plt.figure()
    ax = fig.add_subplot()
    sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['custseg'], palette='bright', ax=ax)

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    encoded_img = base64.encodebytes(output.getvalue())
    plt.close(fig)  # Cerramos la imagen para que no consuma memoria del sistema
    return encoded_img, score


@app.route('/')
def home():
    encoded_img = custumer_overview()
    return render_template('index.html', overview_graph=encoded_img)


# Cuando se presione el botón el sistema llama a esta función
# para realizar la predicción con el modelo de inteligencia artificial
@app.route('/predict',methods=['POST'])
def predict():
    try:
        clusters = int(request.form.get('clusters'))
        encoded_img = custumer_overview()

        if clusters <= 0:
            return render_template('index.html', overview_graph=encoded_img, prediction_text='Datos mal ingresados')

        prediction_image, score = custumer_segmentation(clusters)

        return render_template('index.html', overview_graph=encoded_img,
         prediction_image=prediction_image, prediction_text=f'Score del resultado: {score:.2f}')

        
    except Exception as e:
        print(e)
        encoded_img = custumer_overview()
        return render_template('index.html', overview_graph=encoded_img, prediction_text='Datos mal ingresados')

if __name__ == "__main__":
    app.run(debug=True)