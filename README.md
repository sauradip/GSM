# Proposal-Free Temporal Action Localization with Global Segmentation Mask

This repository is the official pytorch implementation of "Proposal-Free Temporal Action Localization with Global Segmentation Mask". 


## Requirements

To install requirements:

```Environment Setup
pip install -r requirements.txt
```

>📋  Create a virtual environment in conda or pip and install the requirements

## Video Features

* In this repository, we have demonstrated the implementation on ActivityNet Dataset. We use the Kinetics Pretrained I3D features for ActivityNet from the following repository : [ACM-Net](https://github.com/ispc-lab/ACM-Net)
* Please download the video features (train/test) and unzip it in `features/` folder. (Remember to have atleast 30 GB of disk space)
* Update the feature path in `config/anet.yaml`

## Training

To train the model in the paper, run this command:

```train
python gsm_train.py 
```
> You can set all the training hyperparameters in `config/anet.yaml` file

## Evaluation

To evaluate our model on ActivityNetv1.3 dataset, run:

```inference
sh evaluation.sh
```

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results



