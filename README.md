# Homework

## Description
This projects uses a simple unet to segment DCM image. Both train codes and val codes are provided.


## Requiements

```
# python == 3.10
pip install -r requirements.txt
```

## How to use

1. train the model
```
python run.py --train
```
2. eval the model on testset
```
python run.py --eval
```
3. eval the model on testset and trainset

```
python run.py --eval --eval_on_testset
```