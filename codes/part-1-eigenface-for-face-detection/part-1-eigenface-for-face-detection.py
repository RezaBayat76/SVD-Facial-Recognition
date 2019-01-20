import os
import numpy as np
from PIL import Image

# Constants
PREFIX_IMAGE_PATH = '../../yalefaces/images/';
NUM_M = 165 # Images from a group of I=15 individuals
NUM_I = 15 # Number of individuals
NUM_D = 11 # Each individual took D=11 different pictures changing their facial expression for each of the I images.

# The First step: Read images.
# The second step: Find the average matrix relative to our data base
image_names = [image for image in os.listdir(PREFIX_IMAGE_PATH)]

# Assuming all images are the same size, get dimensions of first image
w, h = Image.open(PREFIX_IMAGE_PATH + image_names[0]).size
N = len(image_names)

# Create a numpy array of floats to store the average (assume RGB images)
average_face = np.zeros((h, w), np.float)

# Build up average pixel intensities, casting each image as an array of floats
index = 0
images = {}
for image_name in image_names:
    img = np.array(Image.open(PREFIX_IMAGE_PATH + image_name), dtype=np.float)
    average_face = average_face + img/N
    images[image_name] = img

# Round values in array and cast as 8-bit integer
average_face = np.array(np.round(average_face), dtype=np.uint8)

# Yhe third step: Subtract the average face from each of the existing faces in our grayscale data base
subtracted_images = {}
for image_name in image_names:
    subtracted_images[image_name] = images[image_name] - average_face

# The fourth step: Take each of the subtraced images and find its transpose.
transposed_images = {}
for image_name in image_names:
    transposed_images[image_name] = subtracted_images[image_name].transpose()

# The fifth step: Find the covariant image using the two previous steps
covariant_image = np.zeros((h, h), np.float)   
for image_name in image_names:
    mult_img = np.matmul(subtracted_images[image_name], transposed_images[image_name])
    covariant_image = covariant_image + mult_img/N

# The sixth step: Finding the eigen values and eigen vectors relative to the data base images. 
eigenvalues, eigenvectors = np.linalg.eig(np.array(covariant_image))
eigenfaces = {}

# Select 6 eigen vectors
for i in range (0, 6):
    index = 0
    tmp = np.zeros((h, w), np.float)
    for image_name in images:
        tmp = tmp + eigenvectors[i][index] * images[image_name]
        index = index + 1
    eigenfaces[i] = tmp