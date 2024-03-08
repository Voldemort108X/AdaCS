# Adaptive Correspondence Scoring for Unsupervised Medical Image Registration

## Motivation
![Motivation](assets/motivation.png "Motivation")

## Framework
![Framework](assets/framework.png "Framework")

## Installation
```
conda create -f requirements.yml -n AdaCS_env # Create an environment named AdaCS_env
conda activate AdaCS_env
```

## Dataset


## Default directory structure
    ├── Dataset                   
    |   ├── ACDC       # Place the downloaded dataset here
    |   |   ├── train
    |   |   ├── val
    |   |   ├── test
    |   ├── CAMUS
    |   |   ├── Original_data
    |   |   |   ├── TrainingData_LVQuan19 # Place the downloaded dataset here
    |   |   ├── train
    |   |   ├── ...
    ├── Code
    |   ├── AdaCS
    |   |   ├── train_vxm.py
    |   |   ├── test_vxm.py
    |   |   ├── train_tsm.py
    |   |   ├── ...

## Train AdaCS

## Test

## Acknowledgement
We use implementation of Voxelmorph , Transmorph, Diffusemorph and c-LapIRN.