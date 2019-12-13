import h5py
import numpy as np
import pandas as pd


class GetSensors:
   def __init__(self):
       self.data = pd.DataFrame()

   def add_data(self,x,y,data): ##Get the files
       name = str(x)+"-"+str(y)
       self.data[name] = data


   def read_write(self,files,sensors): #Get the data from each sensor and save the information
       for fil in files:
           fileh5 = h5py.File("Datas/"+fil+".h5", 'r')
           print("Got file",fil,fileh5['ez'].shape)
           for location in sensors:
               print("getting sensor",location[0],location[1])
               self.add_data(location[0],location[1],fileh5['ez'][location[0]][location[1]])
           fileh5.close()
           self.data.to_csv("csv_datas\\"+fil+".csv",header=True,index=False)

   def print_pd(self): #in case you want to check the data before saving
       print(self.data)


if __name__ == '__main__':
    # Set the sensor locations
   sensors = [[600,150],[500,150],[400,150],[300,150],[200,150],

              [600,650],[500,650], [400,650],[300,650],[200,650],

              [650,200],[650,300],[650,400],[650,500],[650,600],

              [150,200],[150,300], [150,400],[150,500],[150,600]]
   #Set the files that have the data
   files = ["MeepSimulation-ez00", "MeepSimulation-ez01", "MeepSimulation-ez02", "MeepSimulation-ez03", "MeepSimulation-ez04",
            "MeepSimulation-ez10", "MeepSimulation-ez11", "MeepSimulation-ez12", "MeepSimulation-ez13", "MeepSimulation-ez14",
            "MeepSimulation-ez20", "MeepSimulation-ez21", "MeepSimulation-ez22", "MeepSimulation-ez23", "MeepSimulation-ez24",
            "MeepSimulation-ez30", "MeepSimulation-ez31", "MeepSimulation-ez32", "MeepSimulation-ez33", "MeepSimulation-ez34",
            "MeepSimulation-ez40", "MeepSimulation-ez41", "MeepSimulation-ez42", "MeepSimulation-ez43", "MeepSimulation-ez44"
            ]
   gs = GetSensors()
   gs.read_write(files = files,sensors = sensors)
