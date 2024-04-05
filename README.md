# 046211 - GPT2 sEMG gesture classifier 
<h1 align="center">
  <br>
    
  <br>
  <img src="https://raw.githubusercontent.com/taldatech/ee046211-deep-learning/main/assets/nn_gumgum.gif" height="200">
</h1>
  <p align="center">
    <a href="https://github.com/stavb2009">Stav Belyy</a> •
    <a href="https://github.com/KarpAmit">Yuval Gerzon</a>
  </p>

Our Project for classifying sEMG signal of hand gestured using transfer learning from GPT2  

Was inspired from the work:

Demir, F., Bajaj, V., Ince, M.C. et al [Surface EMG signals and deep transfer learning-based physical action
classification](https://music-classification.github.io/tutorial/landing-page.html), 2019



  * [Background](#background)
  * [Dataset](#Dataset)
  * [Prerequisites](#prerequisites)
  * [Files in the repository](#files-in-the-repository)
  * [Downloading the data](#Downloading-the-data)
  * [Running the Network](#Running-the-Network)
  * [References](#references)

## Background
Surface Electromyography (sEMG) measures muscle activity, helping diagnose muscle issues by tracking contractions. Merging sEMG with machine learning aids in quicker, more accurate diagnoses, useful for making better prosthetics and rehab methods. Yet, the big hurdle is not having enough data for these smart models to learn well. Transfer learning helps here, using big, general datasets to boost smaller sEMG studies. This speeds up development and gets around the data shortage, making our tools smarter and more helpful.

Our project plans to leverage GPT-2, an advanced NLP model originally trained on a huge dataset, for something a bit out of the box: classifying sEMG signals. This approach aims to sidestep the data scarcity issue in our field, making use of GPT-2's vast training for tasks it wasn't directly designed for.


## Dataset
Our study focused on PutEMG dataset: 
[PutEMG](https://biolab.put.poznan.pl/putemg-dataset/)
* 44 subjects
* 8 different hand gestures (classes) + rest, labeled 0-3,6-9,-1
* collected using 24-elctored matrix, samples in 5120Hz
* each subject practicipated twice
* 394M data points


## Prerequisites
|Library         | Version |
|----------------------|----|
|`Python`|  `3.8 (Anaconda)`|
|`torch`|  `1.13`|
|`torchaudio`|  `0.13`|
|`scipy.io.wavfile`|  `1.12.0`|
|`tensorboardX`|  `1.5`|


## Files in the repository

|File name         | Purpsoe |
|----------------------|------|
|`images`| images of the network build and chosen results|
|`songs_split_txt`| an example of how how the Train/Val/Test split should look like|
|`DL music genre classifier.pptx`| pptx Presentaion of the network|
|`Deep_Network.py`| code for the network|
|`Stereo_to_mono_trans.py`| python code for transforming Stereo to Mono|
|`best_model.ckpt`| example of a trained model|
|`create_dataset_files.py`| python code for spltting data Train/Val/Test|
|`download_MTG_Jamendo.py`| code to download the MTG-Jamendo Dataset|
|`music_split.py`|python code for splitting the songs to chosen time samples|
|`Project report.pdf`| project report of our network|

## Downloading the data
```diff
- Notice: The Dataset uses around 750GB
```
All audio is distributed in 320kbps MP3 format. The audio files are split into folders packed into TAR archives. The dataset is hosted [online at MTG UPF](https://essentia.upf.edu/documentation/datasets/mtg-jamendo/).

The script to download and validate all files in the dataset. See its help message for more information:

```bash
python download.py -h
```
```
usage: download.py [-h] [--dataset {raw_30s,autotagging_moodtheme}]
                   [--type {audio,audio-low,melspecs,acousticbrainz}]
                   [--from {mtg,mtg-fast}] [--unpack] [--remove]
                   outputdir

Download the MTG-Jamendo dataset

positional arguments:
  outputdir             directory to store the dataset

options:
  -h, --help            show this help message and exit
  --dataset {raw_30s,autotagging_moodtheme}
                        dataset to download (default: raw_30s)
  --type {audio,audio-low,melspecs,acousticbrainz}
                        type of data to download (audio, audio in low quality,
                        mel-spectrograms, AcousticBrainz features) (default: audio)
  --from {mtg,mtg-fast}
                        download from MTG (server in Spain, slow),
                        or fast MTG mirror (Finland) (default: mtg-fast)
  --unpack              unpack tar archives (default: False)
  --remove              remove tar archives while unpacking one by one (use to
                        save disk space) (default: False)

```

For example: to download audio for the `autotagging_moodtheme.tsv` subset, unpack and validate all tar archives:

```
mkdir /path/to/download
python3 download.py --dataset autotagging_moodtheme --type audio /path/to/download --unpack --remove
```


Unpacking process is run after tar archive downloads are complete and validated. In the case of download errors, re-run the script to download missing files.

Due to the large size of the dataset, it can be useful to include the `--remove` flag to save disk space: in this case, tar archive are unpacked and immediately removed one by one.


## Running the Network

To run the network:
First run the splitter with chosen time length: 
```bash
python music_split.py -h
```

Transform all songs to Mono from Stereo:
```bash
python Stereo_to_mono_trans.py -h
```
Split data to train, val and test:
```bash
python create_dataset_files.py -h
```
Run the Network:
```bash
python Deep_Network.py -h
```

## References
* MinzWon, Andres Ferraro, Dmitry Bogdanov and Xavier Serra. “Evaluation of CNN-based Automatic Music Tagging Models”, 2020
* Khaled Koutin , Hamid Eghbal-zadeh and Gerhard Widmer. “Receptive Field Regularization Techniques for Audio Classification and Tagging with Deep Convolutional Neural Networks”, 2021



