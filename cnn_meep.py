import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.io import wavfile
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
import glob, os
from pandas.tools.plotting import table
import random

class MeepCNN:  #Get hte data from the csv files created
   def __init__(self):
       self.totalXData = []
       self.totalYData = []
       os.chdir("csv_datas\\")
       for files in glob.glob("*.csv"):
           self.set_ydata(files)
           self.totalXData.append(pd.read_csv(files).T.as_matrix())
       print(self.totalYData)
       for files in glob.glob("*.csv"):
           print(files)
           self.totalXData.append(pd.read_csv(files).T.as_matrix())
       self.X_len = len(self.totalXData)
       print(self.X_len)

   def open_meep_data(self):  #Get the data from the sensors
       print(self.totalYData)
       self.totalXData = np.reshape(np.array(self.totalXData), (self.X_len, 20, 300, 1)).astype(float)
       miy = np.min(self.totalYData)
       self.totalYData = np.add(self.totalYData,abs(miy))
       may = np.max(self.totalYData)
       self.totalYData = np.divide(self.totalYData,may)
       print(self.totalYData.shape)
       self.totalYData = np.append(self.totalYData,self.totalYData,axis=0)
       print(self.totalYData.shape)


   def get_ydata(self,files):  #Get the obsctruction coordinate values 
       inez = files.find("ez") + 2
       indot = files.find(".csv")
       yname = files[inez:indot]
       if (len(yname) == 4):
           self.totalYData.append([int(yname[:2]), int(yname[2:])])
       elif (len(yname) == 3):
           if (yname.find("-") == 0):
               self.totalYData.append([int(yname[:2]), int(yname[2:])])
           else:
               self.totalYData.append([int(yname[:1]), int(yname[1:])])
       elif (len(yname) == 2):
           self.totalYData.append([int(yname[:1]), int(yname[1:])])
       print(yname)
       
       
   def cnn(self,type1, type2, type3,type4,epcs): #Create model
       X_train, X_test, y_train, y_test = train_test_split(self.totalXData, self.totalYData, test_size=0.2, random_state=234)
       model = Sequential()
       model.add(Conv2D(type4, (type1, type1), input_shape=(20, 300, 1), activation='tanh',))
       model.add(Conv2D(100, (type2, type2),activation='relu'))
       model.add(MaxPooling2D(pool_size=(type3,type3)))
       model.add(Flatten())
       model.add(Dense(2,activation='sigmoid'))
       model.add(Dense(2, activation='relu'))
       model.compile(loss = 'mean_squared_error', optimizer='rmsprop',metrics=['mae','acc'])
       model.fit(X_train, y_train,epochs=epcs) #Train model

       model.summary()

       score = model.evaluate(X_test, y_test, verbose=0)
       #model.save('cnn.model') #Save model

       prediction = model.predict(X_train)
       sum_err = [0,0]
       for i,val in enumerate(y_train[:-1]):  #check values
           print("P:",prediction[i], " V:",val," E:", val-prediction[i])
           sum_err += abs(val-prediction[i])
       #print(a, b, c, d, e)
       print('score', score)
       print(sum_err)
       return sum_err,score

   
