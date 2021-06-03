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
* The Training Curves will be as follows
![](https://github.com/sauradip/GSM/blob/main/loss_picture.png)

## Evaluation

To evaluate our model on ActivityNetv1.3 dataset, run:

```inference
sh evaluation.sh

Loading validation Video Information ...
100% 4728/4728 [00:00<00:00, 9221.40it/s] 
Inference start
Inference finished
Starting Post-Process
Ending Post-Process
Detection: average-mAP 35.954 mAP@0.50 54.968 mAP@0.55 51.479 mAP@0.60 47.625 mAP@0.65 44.342 
mAP@0.70 40.620 mAP@0.75 36.354 mAP@0.80 31.349 mAP@0.85 25.181 mAP@0.90 18.190 mAP@0.95 9.428
```

## Pre-trained Models

For ease of inference, we have provided the pre-trained model for GSM on ActivityNet.
You can download pretrained models here:

- [[Google Drive]](https://drive.google.com/drive/folders/1kG7b0hxktEWE_UmZDok4BN_RmKALhlH_?usp=sharing) trained on ActivityNetv1.3. 
>  Place the contents of the folder in `\output` and run evaluation

## Performance

![](https://github.com/sauradip/GSM/blob/main/Screenshot%202021-06-03%20at%208.02.52%20PM.png)



