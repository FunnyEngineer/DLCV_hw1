import sklearn
from PIL import Image
import numpy as np

image = Image.open('bird.jpg')

print(image.format)
print(image.size)
print(image.mode)

arr = np.asarray(image)
