#Import the library
import torch
import numpy as np
import matplotlib.pyplot as plt
print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# generate a 2D grid of coordinates.
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

gaussian = torch.exp(-(x**2+y**2)/2.0)
sinpart=torch.sin(x) * torch.sin(y)
cospart=torch.cos(x) * torch.cos(y)
z =gaussian*sinpart

#plot
plt.imshow(z.cpu().numpy())#Updated!
plt.tight_layout()
plt.show()