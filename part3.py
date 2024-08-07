import torch
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

canvas_size=512
canvas = torch.zeros((canvas_size, canvas_size), dtype=torch.float32, device=device)

vertices = torch.tensor([[canvas_size // 2, 0], [0, canvas_size], [canvas_size, canvas_size]], device=device)

depth = 6

def draw_sierpinski(x0, y0, x1, y1, x2, y2, depth):
    if depth == 0:
        vertices = torch.tensor([[x0, y0], [x1, y1], [x2, y2]], device=device)
        fill_triangle(vertices)
    else:
        x01, y01 = (x0 + x1) / 2, (y0 + y1) / 2
        x12, y12 = (x1 + x2) / 2, (y1 + y2) / 2
        x20, y20 = (x2 + x0) / 2, (y2 + y0) / 2

        draw_sierpinski(x0, y0, x01, y01, x20, y20, depth - 1)
        draw_sierpinski(x1, y1, x12, y12, x01, y01, depth - 1)
        draw_sierpinski(x2, y2, x20, y20, x12, y12, depth - 1)
def fill_triangle(vertices):
    x_min = int(vertices[:, 0].min().item())
    x_max = int(vertices[:, 0].max().item())
    y_min = int(vertices[:, 1].min().item())
    y_max = int(vertices[:, 1].max().item())
    for x in range(x_min, x_max + 1):
        for y in range(y_min, y_max + 1):
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                v0 = vertices[1] - vertices[0]
                v1 = vertices[2] - vertices[0]
                v2 = torch.tensor([x, y], device=device) - vertices[0]

                d00 = (v0 * v0).sum()
                d01 = (v0 * v1).sum()
                d11 = (v1 * v1).sum()
                d20 = (v2 * v0).sum()
                d21 = (v2 * v1).sum()
                denom = d00 * d11 - d01 * d01
                if denom == 0:
                    continue
                v = (d11 * d20 - d01 * d21) / denom
                w = (d00 * d21 - d01 * d20) / denom
                if v >= 0 and w >= 0 and (v + w) <= 1:
                    canvas[y, x] = 1.0

plt.figure(figsize=(10, 10))

draw_sierpinski(vertices[0, 0].item(), vertices[0, 1].item(),
                vertices[1, 0].item(), vertices[1, 1].item(),
                vertices[2, 0].item(), vertices[2, 1].item(), depth)

plt.imshow(canvas.cpu().numpy())
plt.tight_layout(pad=0)
plt.show()