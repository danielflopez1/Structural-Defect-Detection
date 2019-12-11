# Structural-Defects-Sensor
Implemented convolutional and LSTM networks to detect structural defects in plates using electromagnetic wave generators and sensors. To generate training and testing data, simulated electromagnetic waves and their propagation in a 2D plate using finite-difference time-domain (FDTD) method. Used the MEEP library. This Model identified correctly the region of the location of the defect 65% of the time. [Project Details](https://docs.google.com/document/d/1AlJmcSzWoFh2aex3gz_YyJZfXLPGOyG6g7xSFFyxJ0c/edit?usp=sharing)

## Requirements

```
Tensorflow=1.15.0
Keras=2.2.5
numpy=1.17.4
meep: https://github.com/NanoComp/meep
```
## Simulations
![](WaveSimulation1.gif)  ![](Wavesimulation3.gif)

These simulations were created using [meep's](https://meep.readthedocs.io/en/latest) finite-difference time-domain (FDTD) method. 
Every 10 time-states data was recorded in 
## Networks 
CNN-LSTM 
