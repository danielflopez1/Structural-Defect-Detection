# Structural-Defects-Sensor
Implemented convolutional and LSTM networks to detect structural defects in plates using electromagnetic wave generators and sensors. Generated training and testing data, simulated electromagnetic waves and their propagation in a 2D plate using finite-difference time-domain (FDTD) method using the MEEP library. This Model identified correctly the region of the location of the defect 89% of the time. [Project Details](https://docs.google.com/document/d/1AlJmcSzWoFh2aex3gz_YyJZfXLPGOyG6g7xSFFyxJ0c/edit?usp=sharing)

## Requirements

```
Tensorflow=1.14.x
Keras=2.2.5
numpy=1.15.4
meep: https://github.com/NanoComp/meep
```
## Simulations
These simulations were created using [meep's](https://meep.readthedocs.io/en/latest) finite-difference time-domain (FDTD) method. 
![](WaveSimulation1.gif)  ![](Wavesimulation3.gif)

Plate simulation parameters in the code are given as a plate of 10 by 10 centered at 0,0 in the model’s units, in a 16 by 16 simulation grid centered at 0,0 in model’s units. With a medium with elasticity epsilon 5. Each model unit are 50 pixels. The obstruction/defect is a hole (medium with elasticity 1) of 1 by 1 model’s units. It is put in different places to simulate multiple possible defects.
![](sensors.png)


## Networks 
CNN-LSTM 
CNN
LSTM
Fourier preprocessing to CNN


