from sklearn.cluster import KMeans
from PIL import Image
import numpy as np

image = Image.open('bird.jpg')

print(image.format)
print(image.size)
print(image.mode)

arr = np.asarray(image)
arr = arr.reshape(-1, 3)
new_arr = KMeans(n_clusters=2, random_state=0).fit_transform(arr)
new_arr = new_arr.reshape(1024, 1024, 3)
print(new_arr.shape)