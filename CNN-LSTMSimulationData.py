import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.io import wavfile
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, TimeDistributed
from keras.layers import Conv2D, LSTM, MaxPooling2D, CuDNNLSTM, Bidirectional
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
import glob, os
from pandas.tools.plotting import table
import random

class WifiCNN:
    def __init__(self):
        self.totalXData = []
        self.totalYData = []
        random.seed(1234)

        '''
            np.array([#[-1, -1],
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
            [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
            [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4],
            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4],
            [1, 0], [1, 1], [1, 2], [1, 3], [1, 4],
            [2, 0], [2, 1], [2, 2], [2, 3], [2, 4],
            [3, 0], [3, 1], [3, 2], [3, 3], [3, 4],
            [4, 0], [4, 1], [4, 2], [4, 3], [4, 4]
        ])
        '''
        #self.totalYData = np.array([[256], [120], [136], [152], [168], [184],
        #                            [121], [137], [153], [169], [185],
        #                            [122], [138], [154], [170], [186],
        #                            [123], [139], [155], [171], [187],
        #                            [124], [140], [156], [172], [188]
        #                            ])

        os.chdir("csv_datas\\")
        for files in glob.glob("*.csv"):
            self.set_ydata(files)
            self.totalXData.append(pd.read_csv(files).T.as_matrix())
        print(self.totalYData)
        for files in glob.glob("*.csv"):
            print(files)
            self.totalXData.append(pd.read_csv(files).T.as_matrix())
        #for files in glob.glob("*.csv"):
        #    print(files)
        #    self.totalXData.append(pd.read_csv(files).T.as_matrix())
        self.X_len = len(self.totalXData)
        print(self.X_len)

    def open_wave_data(self):
        print(self.totalYData)
        self.totalXData = np.reshape(np.array(self.totalXData), (self.X_len, 20, 300, 1)).astype(float)
        miy = np.min(self.totalYData)
        self.totalYData = np.add(self.totalYData,abs(miy))
        may = np.max(self.totalYData)
        self.totalYData = np.divide(self.totalYData,may)
        print(self.totalYData.shape)
        self.totalYData = np.append(self.totalYData,self.totalYData,axis=0)
        print(self.totalYData.shape)


    def set_ydata(self,files):
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
        print("y",yname)

    def sequence_input(self,inp_data, out_data, t_step, axis, pred_stepdiff):
        ind = np.asarray(inp_data.shape)
        ind[axis] = inp_data.shape[axis] - 2 * t_step + 1 - pred_stepdiff
        ind2 = np.asarray(out_data.shape)
        ind2[axis] = out_data.shape[axis] - 2 * t_step + 1 - pred_stepdiff
        inp_datanew = np.zeros([ind[0], t_step, ind[1], ind[2], ind[3]])
        out_datanew = np.zeros([ind2[0], t_step, ind2[1]])

        for i in range(0, inp_datanew.shape[axis]):
            tn = i
            tnplus1 = i + t_step

            inp_datanew[i, :, :, :, :] = inp_data[tn:tnplus1, :, :, :]
            out_datanew[i, :, :] = out_data[tnplus1 + pred_stepdiff: tnplus1 + t_step + pred_stepdiff, :]
        #  print(i)
        return inp_datanew, out_datanew
    def modify_input(self):
        self.totalXData, totalYData = self.sequence_input(self.totalXData,self.totalYData,5,0,0)

        print("YESSSSSSSSSSSSSSSSSSSSSSSSSSS")
        data = np.array(self.totalXData)
        shapes = data.shape
        print(shapes)
        data = np.array(self.totalYData)
        shapes = data.shape
        print(shapes)
        '''
        for z in range(shapes[0]):  # 10
            datay = []
            for y in range(shapes[1]):  # 20
                datax = []
                for x in range(shapes[2] - 5):  # 300
                    datax.append(data[z][y][x:seq_len])
                    print(datax)
                datay.append(datax)
            end_data.append(datay)
        print(np.array(end_data).shape)
        '''


        #print(self.totalXData.shape)


    def cnn(self,a, b, c,epcs):
        self.modify_input()
        n_unitsFC1 = 100
        n_unitsFC2 = 100
        LSTM_neurons = 100
        n_convpool_layers = 1
        n_convlayers = 3
        n_reglayers = 1
        max_poolsize = (3, 3)
        train_set_ratio = 0.7
        valid_set_ratio = 0.15
        drop_rate = 0.0
        n_filters = 100
        kernel_size = (3, 3)
        input_strides = 1
        kernel_init = 'glorot_uniform'
        cost_function = 'mean_squared_error'
        batch_size = 10
        n_epochs = 30
        n_output = 2
        n_post_test = 5000
        early_stop_delta = 0.001  # 0.0025 change or above is considered improvement
        early_stop_patience = 10  # keep optimizing for 10 iterations under "no improvement"
        nred = 0
        n_train = int(105)
        x_train = self.totalXData[:n_train, :, nred:100 - nred, nred:100 - nred, :]
        x_test = self.totalXData[n_train:, :, nred:100 - nred, nred:100 - nred, :]
        y_train = self.totalYData[:n_train, :]
        y_test = self.totalYData[n_train:, :]
        #X_train, X_test, y_train, y_test = train_test_split(self.totalXData, self.totalYData, test_size=0.3, random_state=234)
        y_train = np.array(y_train)
        y_train = self.totalYData#y_train.reshape(y_train.shape[0]*y_train.shape[1],2)
        print(y_train.shape)
        model = Sequential()

        model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=input_strides,
                                         kernel_initializer=kernel_init,
                                         activation=type1), input_shape=(5, 20, 100, 1)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))

        model.add(TimeDistributed(Conv2D(filters=n_filters, kernel_size=kernel_size, activation=type2)))
        model.add(TimeDistributed(MaxPooling2D(pool_size=max_poolsize)))


        model.add(TimeDistributed(Flatten()))

        model.add(TimeDistributed(Dense(n_unitsFC1, activation=type3, kernel_initializer=kernel_init)))
        model.add(TimeDistributed(Dense(n_unitsFC1, activation=type4, kernel_initializer=kernel_init)))

        model.add(CuDNNLSTM(LSTM_neurons, return_sequences=False, kernel_initializer=kernel_init))

        model.add(Dense(n_output, activation='linear', kernel_initializer=kernel_init))

        model.summary()

        model.compile(loss=cost_function, optimizer='adam', metrics=['accuracy'])


        # Train the network
        print(np.array(x_train).shape,np.array(y_train).shape)
        history = model.fit(x_train, y_train[:105], epochs=epcs,
                            batch_size=batch_size, verbose=1)


        prediction = model.predict(x_train)
        sum_err = [0,0]
        for i,val in enumerate(y_train[:-1]):
            print("P:",prediction[i], " V:",val," E:", val-prediction[i])
            sum_err += abs(val-prediction[i])
        #print(a, b, c, d, e)
        print('score', score)
        print(sum_err)
        return sum_err,score,[type1, type2, type3, type4]

    def cnn_data(self):
        print("helo")



if __name__ == '__main__':
    wcnn = WifiCNN()
    wcnn.open_wave_data()
    adata = []
    types = ['sigmoid','tanh','relu']
    for a in range(1, 6):
        for b in range(1, 6):
            for c in range(1, 6):
                for epocs in range(1,40):
                    adata.append(wcnn.cnn(a, b, c, epocs))
#        for type2 in types:
#            for type3 in types:
#                for type4 in types:

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


    #wcnn.get_value_average()
    #for x in range(1):

