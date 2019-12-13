'''
This file generates 25 simulations in meep of defects in 25 different coordinates of the plates. 
'''

import numpy as np
import meep as mp
import matplotlib.pyplot as plt
import os

length = 150
for x in range(5):
	for y in range(5):
		#Simulate the plate
		cell = mp.Vector3(16,16,0)

		#Generate a defect
		geometry = [mp.Block(mp.Vector3(10,10,1),
				     center=mp.Vector3(0,0),
				     material=mp.Medium(epsilon=5)),
			    mp.Block(mp.Vector3(1,1,1),
				     center=mp.Vector3(x,y),  # It changes position on every loop
				     material=mp.Medium(epsilon=1))]
		pml_layers = [mp.PML(1.0)]
		resolution = 50
		#Generate a source of EMF
		sources = [mp.Source(mp.ContinuousSource(frequency=1,end_time=3),
				     component=mp.Ez,
				     center=mp.Vector3(-3,0),
				     size=mp.Vector3(0,0))]
		
		sim = mp.Simulation(cell_size=cell,
				    boundary_layers=pml_layers,
				    geometry=geometry,
				    sources=sources,
				    resolution=resolution)
		#Start the simulation
		sim.run(mp.at_beginning(mp.output_epsilon),
			mp.to_appended("ez"+str(x)+str(y), mp.at_every(0.5, mp.output_efield_z)),
			until=length)
		#Generate Data and gifs
		os.system("h5topng -t 0:"+str(length)+" -R -Zc dkbluered -a yarg -A MeepSimulation-eps-000000.00.h5 MeepSimulation-ez"+str(x)+str(y)+".h5")
		os.system("convert MeepSimulation-ez"+str(x)+str(y)+".t*.png ez"+str(x)+str(y)+".gif")
