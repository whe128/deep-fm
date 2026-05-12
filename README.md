# deep-fm

## Overview

This project implements a simple DeepFM model using PyTorch for CTR prediction.

The model combines:

- Linear sum
- Factorization Machine (FM)
- Deep Neural Network (DNN)

to learn both low-order and high-order feature interactions.

## Structure

- `dataset.py`
  - fake CTR dataset

- `model.py`
  - DeepFM model

- `train.py`
  - training and inference

## Run

```bash
pip install -r requirements.txt

python train.py
```
