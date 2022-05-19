# Introduction
This repository implements the Local Dynamics Model in PyTorch.

# Code Structure
The structure of this project is based on this [MuZero implementation](https://github.com/werner-duvaud/muzero-general).

# Domains
- ***Block Stacking 3:*** 
- ***House Building 2:*** 
- ***Bottle Arrangement:*** 
- ***Bin Packing:*** 

# Getting Started

## Installation 

## Train

python train block_stacking_3 --results_path=model_name

## Test

python test block_stacking_3 model_name

## Config

# Model Architectures

## Local Dynamics Model
![LDM](/assets/local_dynamics_model.png)

## Q-Value Model
![Q](/assets/q_value_model.png)

## State-Value Model
![S](/assets/s_value_model.png)
