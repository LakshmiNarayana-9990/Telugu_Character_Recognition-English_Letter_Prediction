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
import cv2

import os

def createFileList(myDir, format='.png'):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            if name.endswith(format):
                fullName = os.path.join(root, name)
                fileList.append(fullName)
    return fileList


df = pd.read_csv('teluguletters.csv')

pix=[]
for i in range(1,10001):
    pix.append('pix-'+str(i))
features=pix
X = df.loc[:, features].values
y = df.loc[:,'class'].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 100)
y_train=y_train.ravel()
y_test=y_test.ravel()
svm_model = SVC(kernel = 'linear', C = 1, gamma=0.0001).fit(X_train, y_train)



    # Saving model to disk
pickle.dump(svm_model, open('mymodel.pkl','wb'))

    # Loading model to compare the results  
model = pickle.load(open('mymodel.pkl','rb'))
# load the original image
myFileList = createFileList(r'C:\Users\HP\Music\MLproject\images')
for i in myFileList:
    new_X = cv2.imread(i)
    new_X = cv2.cvtColor(new_X, cv2.COLOR_BGR2GRAY)
    img =cv2.resize(new_X, ( 100, 100), interpolation=cv2.INTER_AREA)
    img = img/255.0
    plt.imshow(img,cmap = "gray")
    img = img.ravel()
    new_y = model.predict([img])
    print("predicted label",int(new_y))
    #Another Way
    '''img_file = Image.open(i)
        
            # img_file.show()

    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode

            # Make image Greyscale
    img_grey = img_file.convert('L')
           #img_grey.save('result.png')
            #img_grey.show()

            # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))/255
    value = value.flatten()
    value=value.reshape(1,-1)


    print()
    print(model.predict(value))
    print()'''
accuracy = svm_model.score(X_test, y_test)
print('Accuracy: ',accuracy*100)