import pandas as pd
import numpy as np
from scipy.io import loadmat
import re
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
from tensorflow import set_random_seed
import string
import pandas as pd

class WifiCNN:
    def __init__(self):
        self.totalXData = []
        self.totalYData = []
        self.fileInputs = []
        random.seed(1234)
        set_random_seed(567)
        self.CheckFiles = 1
        ini_path = ''
        self.num_sims = 500
        self.size = 100



        #os.chdir("csv_datas\\")
        #for files in glob.glob("*.csv"):
        #    self.set_ydata(files)
        #    self.totalXData.append(pd.read_csv(files).T.as_matrix())
        #print(self.totalYData)
        #self.X_len = len(self.totalXData)
        #print(self.X_len)
        self.open_time_history()
        self.open_output()
        #print(self.totalXData)
        #print(self.totalYData)


    def input_output_checks(self,namefile,i):
        #print(re.sub(r'[A-z]+', '', namefile, re.I),"--",re.sub(r'[A-z]+', '', self.fileInputs[i], re.I))
        if(re.sub(r'[A-z]+', '', namefile, re.I) != re.sub(r'[A-z]+', '', self.fileInputs[i], re.I)):
            print("WRONG INPUT OUTPUT FILE INSERTION!",namefile, " does not match ",self.fileInputs[i])



    def open_time_history(self):
        xpath = "C:\\Users\\prpue\\PycharmProjects\\DefectDetection\\InputHistory1Defect"
        for i in range(self.num_sims):
            x = loadmat(os.path.join(xpath, "displacement"+str(i)+".mat"))
            x = np.array(x['displacement'][1:]).T
            x[[0, 7], :] = x[[7, 0], :]
            x[[1, 6], :] = x[[6, 1], :]
            x[[2, 5], :] = x[[5, 2], :]
            x[[3, 4], :] = x[[4, 3], :]

            self.totalXData.append(x)
        self.totalXData = np.array(self.totalXData)
        maxs  =np.amax(self.totalXData)
        self.totalXData = self.totalXData / (maxs)  # eliminate amplitude problem on CNN
        self.totalXData = self.totalXData[:,:,:self.size,np.newaxis]
        print("XShape----", self.totalXData.shape,maxs)

    def open_output(self):
        ypath = "C:\\Users\\prpue\\PycharmProjects\\DefectDetection\\Output1Defect"
        for i in range(self.num_sims):
            y = loadmat(os.path.join(ypath, "ControlPoints"+str(i)+".mat"))
            y = np.array(y['P'])
            self.totalYData.append(y.flatten())
        self.totalYData = np.array(self.totalYData)
        mins,maxss  =np.amin(self.totalYData), np.amax(self.totalYData)
        self.totalYData = self.totalYData + abs(mins)  # eliminate amplitude problem on CNN
        self.totalYData = self.totalYData / maxss  # eliminate amplitude problem on CNN
        print("YShape----",self.totalYData.shape,mins,maxss)


    def cnn(self,type1,type2,type3):
        X_train, X_test, y_train, y_test = train_test_split(self.totalXData, self.totalYData, test_size=0.3, random_state=2354)
        model = Sequential()

        model.add(Conv2D(500, (2, 2), input_shape=(32,self.size,1), activation='relu',))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(400, (2, 2),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Conv2D(200, (2, 2), activation='relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(150, activation=type3))
        model.add(Dense(14, activation='softmax'))
        model.compile(loss = 'mean_squared_error', optimizer='rmsprop',metrics=['mae','acc'])
        history = model.fit(X_train, y_train,epochs=20)
        #print(history.history.keys())
        model.summary()
        score = model.evaluate(X_test, y_test, verbose=0)
        print(score)
        print("CHECKS----")
        num = 40
        narr = np.array([self.totalXData[num]])
        print(narr.shape)
        print(model.predict_on_batch(narr))
        print(self.totalYData[num])


        return score
        #summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        #summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        #prediction = model.predict(X_train)
        '''
        model.save('cnn2.model')

        prediction = model.predict(X_train)
        sum_err = [0,0]
        for i,val in enumerate(y_train[:-1]):
            print("P:",prediction[i], " V:",val," E:", val-prediction[i])
            sum_err += abs(val-prediction[i])
        #print(a, b, c, d, e)
        print('score', score)
        print(sum_err)
        return sum_err,score,[type1, type2, type3, type4, epcs]
        '''

if __name__ == '__main__':
    wcnn = WifiCNN()

    adata = []
    types = ['sigmoid','tanh','relu']

    adata.append(wcnn.cnn('relu','relu','relu'))

    print(adata)

    arr = str(adata)
    more = True
    endx = 0
    minavs = 1000
    while (more):
        indx = arr.find('([', endx)
        if (indx == -1):
            break
        endx = arr.find('])', indx)
        val = arr[indx + 2:endx].split(', ')
        # print(val)
        val1, val2 = float(val[0]), float(val[1])
        #print(val1, val2)
        avg=np.average((val1, val2))
        print(avg)

    ''''''

    #wcnn.get_value_average()
    #for x in range(1):


