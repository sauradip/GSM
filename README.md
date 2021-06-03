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

For ease of inference, we have provided the pre-trained model for GSM on ActivityNet.
You can download pretrained models here:

- [GSM Pretrained](https://drive.google.com/drive/folders/1kG7b0hxktEWE_UmZDok4BN_RmKALhlH_?usp=sharing) trained on ActivityNetv1.3. 

>  Place the contents of the folder in `\output`  

## Performance

![](https://github.com/sauradip/GSM/blob/main/Screenshot%202021-06-03%20at%208.02.52%20PM.png)



