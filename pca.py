import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import numpy as np
import cv2
from matplotlib import pyplot as plt

# create training set and testing set
dir_path = './p2_data'
train_data = []
test_data = []
for filename in os.listdir(dir_path):
    img = cv2.imread(os.path.join(dir_path , filename), cv2.IMREAD_GRAYSCALE).flatten()
    if filename.split('_')[1]  == '10.png':
        test_data.append(img)
    else:
        train_data.append(img)

#from list to array
train_data = np.array(train_data)
test_data = np.array(test_data)

#section 1
pca = PCA(n_components=40)
pca.fit(train_data)
#plot mean face
mean_face = pca.mean_.reshape(56, 46)
cv2.imwrite('mean_face.png', mean_face)

#plot first four eigen face
for i in range(4):
    