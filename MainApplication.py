import os
from PIL import Image
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from PIL import Image

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import base64, os
import numpy as np
import time
import cv2
import os, random
import sys
import os
import csv



app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open('mymodel.pkl','rb'))

@app.route('/')
def index():
    return render_template("Recognition.html")
@app.route('/predict',methods=['POST'])

def predict():
    return render_template("upload.html")


@app.route('/upload', methods=['POST'])

def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        '''new_X = cv2.imread(file)
        new_X = cv2.cvtColor(new_X, cv2.COLOR_BGR2GRAY)
        img =cv2.resize(new_X, ( 100, 100), interpolation=cv2.INTER_AREA)
        img = img/255.0
        plt.imshow(img,cmap = "gray")
        img = img.ravel()
        new_y = model.predict(img)'''
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)
        img_file = Image.open(file)
        width, height = img_file.size
        format = img_file.format
        mode = img_file.mode
        img_grey = img_file.convert('L')

        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))/255
        value = value.flatten()
        value=value.reshape(1,-1)
        var=int(model.predict(value))
        let=''
        if var==0:
            let='A'
        elif var==1:
            let='E'
        elif var==2:
            let='U'
        
        elif var==4:
            let='O'
        else:
            let='I'
        print()

    return render_template("prediction.html",prediction_text='predicted Letter:  "{}"'.format(let))

if __name__ == "__main__":
    app.run(port=4555, debug=True)