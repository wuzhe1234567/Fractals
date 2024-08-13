import torch
import numpy as np
import matplotlib.pyplot as plt
iterations = 7
size = 2 ** iterations
image = torch.zeros((size, size))
def sierpinski(x, y, n):
    if n == 0:
        image[x, y] = 1
    else:
        sierpinski(x, y, n - 1)
        sierpinski(x + 2**(n - 1), y, n - 1)
        sierpinski(x + 2**(n - 1), y + 2**(n - 1), n - 1)
sierpinski(0, 0, iterations)
plt.imshow(image.numpy())
plt.show()
