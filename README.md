# predictTypeSignature

This repository contains code for my thesis *Predicting Haskell Type Signatures From Names*. The structured prediction model is in *master* and the unstructured prediction model, as the name suggests, is in the *unstructured_prediction* branch.

## Dataset
The `dataset` directory contains the datset we use: `train_simple_sigs_parsable_normalized.txt`, `dev_simple_sigs_parsable_normalized.txt`, `test_simple_sigs_parsable_normalized.txt` are for training, validation, and testing, respectively.

## Requirements
* pytorch 0.4.0
* python 3.5+

## How to run
```bash
python train.py --train_data=dataset/train_simple_sigs_parsable_normalized.txt --dev_data=dev_simple_sigs_parsable_normalized.txt --test_simple_sigs_parsable_normalized.txt
```
