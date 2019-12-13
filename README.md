# Structural-Defects-Sensor
Implemented convolutional and LSTM networks to detect structural defects in plates using electromagnetic wave generators and sensors. Generated training and testing data, simulated electromagnetic waves and their propagation in a 2D plate using finite-difference time-domain (FDTD) method using the MEEP library. This Model identified correctly the region of the location of the defect 89% of the time. 

## Requirements

```
Tensorflow=1.14.x
Keras=2.2.5
numpy=1.15.4
meep: https://github.com/NanoComp/meep
pandas
matplotlib
glob
scipy
sklearn
```
## Simulations
These simulations were created using [meep's](https://meep.readthedocs.io/en/latest) finite-difference time-domain (FDTD) method. 
![](WaveSimulation1.gif)

The simulation can be run in Linux. Here is Meep's [Installation Guide](https://meep.readthedocs.io/en/latest/Installation/#installation) and to generate the simulation above you may use the MeepSimulation.py python file.

## Training 
### CNN on Meep data
Use cnn_meep.py 
```
mcnn = MeepCNN()
mcnn.open_meep_data()
mcnn.cnn(2, 4, 4, 150, 3))
```


## Some Results 
|Network        | Accuracy |
|---------------|----------|
|CNN            | 89.5489  |
|Fourier-CNN    | 79.26414 |

### Report

[Project Details](https://docs.google.com/document/d/1AlJmcSzWoFh2aex3gz_YyJZfXLPGOyG6g7xSFFyxJ0c/edit?usp=sharing)

There are more neural networks in the report to be tested.
