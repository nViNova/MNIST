import pyxel
import torch
from torch import nn, load, tensor
import torchvision.models as models

import numpy as np

CLASSES = [str(n) for n in range(10)]

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class App:
    def __init__(self):
        self.r, self.c = 28, 28
        self.radius = 2

        pyxel.init(self.r, self.c, fps=165)

        self.grid = np.array([[[0 for y in range(self.c)] for x in range(self.r)]], dtype=np.float32)

        pyxel.mouse(True)

        self.model = NeuralNetwork().to(DEVICE)
        self.model.load_state_dict(load("model.pth", weights_only=True))
        self.model.eval()
        self.predicted = "None"

        pyxel.run(self.update, self.draw)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.r and 0 <= y < self.c
    
    def update_inference(self, is_clear = False):
        if(is_clear):
            self.predicted = "None"
            return
        grid = tensor(self.grid).to(DEVICE)
        pred = self.model(grid)
        self.predicted = CLASSES[pred[0].argmax(0)]

    def update(self):
        if(pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT, hold=1, repeat=1)):
            for x in range(self.radius):
                for y in range(self.radius):
                    if(self.in_bounds(pyxel.mouse_x + x, pyxel.mouse_y + y)):
                        self.grid[0][pyxel.mouse_y + y][pyxel.mouse_x + x] = 1 if (x == 0 and y == 0) else 1
            self.update_inference()
        
        if(pyxel.btnp(pyxel.MOUSE_BUTTON_RIGHT)):
            for x in range(self.r):
                for y in range(self.c):
                    self.grid[0][y][x] = 0
            self.update_inference(is_clear=True)
        
    def draw(self):
        pyxel.cls(0)
        for x in range(self.r):
            for y in range(self.c):
                pyxel.rect(x, y, 1, 1, 7 if self.grid[0][y][x] else 0)
        
        pyxel.text(0, 0, self.predicted, 7)

App()