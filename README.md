# 046211 - GPT2 sEMG gesture classifier 

<div align="center">
  <img src="https://github.com/stavb2009/OM2SEQ/blob/6ea42dc2063cd3708d465db35cab8aefc5538af1/Experiment%20Platform.jpg" alt="Experiment platform">
</div>

</h1>
<h1 align="center">


</h1>
  <p align="center">
    <a href="https://github.com/stavb2009">Stav Belyy</a> •
    <a href="https://github.com/yuval-gerzon">Yuval Gerzon</a>
  </p>

Our Project for classifying sEMG signal of hand gestures using transfer learning from GPT2  

Was inspired from the work:

Demir, F., Bajaj, V., Ince, M.C. et al [Surface EMG signals and deep transfer learning-based physical action
classification](https://music-classification.github.io/tutorial/landing-page.html), 2019



  * [Background](#background)
  * [Dataset](#Dataset)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Running the Network](#Running-the-Network)
  * [Results](#Results)


## Background
Surface Electromyography (sEMG) measures muscle activity, helping diagnose muscle issues by tracking contractions. Merging sEMG with machine learning aids in quicker, more accurate diagnoses, useful for making better prosthetics and rehab methods. Yet, the big hurdle is not having enough data for these smart models to learn well. Transfer learning helps here, using big, general datasets to boost smaller sEMG studies. This speeds up development and gets around the data shortage, making our tools smarter and more helpful.

Our project plans to leverage GPT-2, an advanced NLP model originally trained on a huge dataset, for something a bit out of the box: classifying sEMG signals. This approach aims to sidestep the data scarcity issue in our field, making use of GPT-2's vast training for tasks it wasn't directly designed for.


## Dataset

Our study focused on PutEMG dataset: 
[PutEMG](https://biolab.put.poznan.pl/putemg-dataset/)
* 44 subjects
* 8 different hand gestures (classes) + rest, labeled 0-3,6-9,-1
* collected using 24-elctored matrix - 24 channels, labeled EMG_1 - EMG_24
* samples in 5120Hz
* 394M data points
![Data sample](https://github.com/stavb2009/OM2SEQ/blob/f9cfefd33a30b2a6db82121c7d69268fb1747c56/EMG%20data%20sample.png "Data sample")  


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.8 `|
|`torch`|  `2.2.1`|
|`pandas`|  `2.0.3`|
|`scipy`|  `1.10.1`|
|`transformers`|  `4.38.2`|
|`matplotlib`|  `3.7.5`|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`EMG_03_data.parquet`| sample of the post-processed dataset for training|
|`best_model.ckpt`| example of a trained model|
|`Preprocessing.py`| python code for cleaning and XXX the data|
|`EMG_utils.py`| auxiliary functions,including training, validation loops and graph builders|
|`EMG_DATA_LOADER.py`|loading and orgnizing the data for training|
|`main.py`| main file|
|`ENV.py`| pre-declared parameters for the run. user should only change this file for its run|
|`requirements.txt`| basic python packages requirements|


## Running the Network
---  Preprocessing  --
use Preprocessing.py to create your own dataset or use EMG_03_data.parquet as an example.

---  define parameters  ---
use ENV.py to change parameters, such as location of the dataset, saving directory and model parameters

## Results
<div align="center">
  <img src="https://github.com/stavb2009/OM2SEQ/blob/6373c150ea8c965d4d382ef1f1d9afb8c1da53eb/Accuracy%20graph.png" alt="Train and Test accuracy">
</div>

<div align="center">
  <img src="https://github.com/stavb2009/OM2SEQ/blob/969ad3224aace224d99a592c05ae8cfb68faa185/confusion_matrix.png" alt="confusion matrix">
</div>



