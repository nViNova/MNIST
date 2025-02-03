# DS102 - MNIST

A short data science project that performs inferences on handwritten digits with a Neural Network Model trained on the MNIST Dataset using PyTorch.

Fulfilled as part of the requirements for DS102

## To Use

### Drawing Inference Application

**Note:** Requires `pyxel` and `pytorch` to run

Simply run `pip install pyxel pytorch` to install.

`draw.py` - A python script written with `pyxel`, that launches a drawing app that performs inferences whenever a user draws a digit.

To run simply run `pyxel run draw.py` in the root directory

#### Controls

- `LMB` Draw a white pixel with a radius of 2
- `RMB` Erase a white pixel with a radius of 2
- `Q` Clear the screen

### Training Notebook

`DS102_KANUPSIM_MNIST.ipynb` contains the 

## Attributions

### Training

PyTorch Docs were highly utilized to make the notebook for training the NN. Most functions for training and testing the model from the dataset were from the [quickstart guide](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html).
