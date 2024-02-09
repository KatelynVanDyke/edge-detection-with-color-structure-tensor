"""
Author: Katelyn Van Dyke
Date: 8 Feb 2024
Course: Computer Vision
Professor: Dr. Feliz Bunyak Ersoy
Description: Edge detection with color structure tensor
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# Load the image
image = cv2.imread('../images/cells3.jpg')

# Create a data folder
output_folder = '../data'
os.makedirs(output_folder, exist_ok=True)

# Compute gaussian derivatives using Sobel filters
ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, scale=10)
iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, scale=10)

# Display Ix and Iy for each channel
for i in range(3):
    plt.figure()
    plt.imshow(ix[:, :, i], cmap='gray'), plt.title(f'Ix - Channel {i}')
    plt.savefig(f'../data/ix-{i}.png')
    plt.figure()
    plt.imshow(iy[:, :, i], cmap='gray'), plt.title(f'Iy - Channel {i}')
    plt.savefig(f'../data/iy-{i}.png')

# Compute 2D Color Structure Tensor
j11 = ix[:, :, 0] * ix[:, :, 0] + iy[:, :, 0] * iy[:, :, 0]
j12 = ix[:, :, 0] * ix[:, :, 1] + iy[:, :, 0] * iy[:, :, 1]
j21 = ix[:, :, 1] * ix[:, :, 0] + iy[:, :, 1] * iy[:, :, 0]
j22 = ix[:, :, 1] * ix[:, :, 1] + iy[:, :, 1] * iy[:, :, 1]

# Compute and display elements of 2D Color Structure Tensor
plt.figure()
plt.imshow(j11, cmap='gray'), plt.title('2D Color Structure Tensor - J11')
plt.savefig('../data/2dcolstruct-j11.png')
plt.figure()
plt.imshow(j12, cmap='gray'), plt.title('2D Color Structure Tensor -nJ12')
plt.savefig('../data/2dcolstruct-j12.png')
plt.figure()
plt.imshow(j21, cmap='gray'), plt.title('2D Color Structure Tensor - J21')
plt.savefig('../data/2dcolstruct-j21.png')
plt.figure()
plt.imshow(j22, cmap='gray'), plt.title('2D Color Structure Tensor - J22')
plt.savefig('../data/2dcolstruct-j22.png')

# Compute and display trace of 2D Color Structure Tensor
plt.figure()
trace_jc = j11 + j22
plt.imshow(trace_jc, cmap='gray'), plt.title('Trace(Jc)')
plt.imshow(trace_jc, cmap='gray'), plt.title('Scalar Edge Indicator (Trace(Jc))')
plt.savefig('../data/tracejc.png')

# Convert to greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gaussian derivatives using Sobel filters
gray_ix = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3, scale=10)
gray_iy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3, scale=10)

# Compute and display gradient magnitude
plt.figure()
gradient_magnitude = np.sqrt(gray_ix**2 + gray_iy**2)
plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
plt.savefig('../data/gradmag.png')
