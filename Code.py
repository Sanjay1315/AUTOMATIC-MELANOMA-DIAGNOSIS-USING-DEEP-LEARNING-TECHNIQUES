


import cv2
import csv
import collections
import numpy as np
from tracker import *
fr="dataset/test/benign/melanoma_9610.jpg"
import cv2  
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread(fr,0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Non Adaptive Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
# Initialize Tracker
tracker = EuclideanDistTracker()
import numpy as np
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
img = cv2.imread(fr)


dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)

plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
gray_image = skimage.color.rgb2gray(img)

# blur the image to denoise
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)

fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap="gray")

histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))

fig, ax = plt.subplots()
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
img = cv.imread(fr)
hist,bins = np.histogram(img.flatten(),256,[0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')
plt.show()
from PIL import Image
img = cv.imread(fr,0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite('clahe_2.jpg',cl1)
im = Image.open("clahe_2.jpg")
im.show()


import cv2
import numpy as np

img = cv2.imread(fr)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gaussian = cv2.GaussianBlur(gray,(3,3),0)

#canny
img_canny = cv2.Canny(img,100,200)

#sobel
img_sobelx = cv2.Sobel(img_gaussian,cv2.CV_8U,1,0,ksize=5)
img_sobely = cv2.Sobel(img_gaussian,cv2.CV_8U,0,1,ksize=5)
img_sobel = img_sobelx + img_sobely


#prewitt
kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)


cv2.imshow("Original Image", img)
cv2.imshow("Canny", img_canny)
cv2.imshow("Sobel X", img_sobelx)
cv2.imshow("Sobel Y", img_sobely)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Prewitt X", img_prewittx)
cv2.imshow("Prewitt Y", img_prewitty)
cv2.imshow("Prewitt", img_prewittx + img_prewitty)
cv2.waitKey(0)
cv2.destroyAllWindows()



from skimage.exposure import histogram
hist, hist_centers = histogram(img)

#Plotting the Image and the Histogram of gray values
fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].imshow(img, cmap=plt.cm.gray)
axes[0].axis('off')
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = fr  # Replace with the actual path to your image
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(image_rgb)
cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
segmented_image = cv2.bitwise_and(image_rgb, mask)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.subplot(1, 3, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Afected Area')
plt.subplot(1, 3, 3)
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresholded_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros_like(image_rgb)
cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
segmented_image = cv2.bitwise_and(image_rgb, mask)


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread(fr)

# Convert image to RGB (matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create a mask initialized with zeros for GrabCut
mask = np.zeros(image.shape[:2], np.uint8)

# Define the background and foreground models for GrabCut
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define the rectangle enclosing the region of interest (the skin lesion)
rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)

# Apply GrabCut algorithm to segment the skin lesion
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

# Create a mask where probable background and definite background are 0, others are 1
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Apply the mask to the original image to extract the segmented region
segmented = image * mask2[:, :, np.newaxis]

# Plot images
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(image_rgb)
axs[0].set_title('Original Image')
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
axs[1].set_title('Segmented Image')
axs[1].axis('off')
plt.show()
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np

# Load ground truth image
ground_truth = cv2.imread(fr)
segmented_image = segmented_image

# Convert images to grayscale (if needed)
ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
segmented_gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Compute Root Mean Squared Error (RMSE)
rmse = np.sqrt(((ground_truth_gray - segmented_gray) ** 2).mean())

# Compute Peak Signal-to-Noise Ratio (PSNR)
psnr = cv2.PSNR(ground_truth_gray, segmented_gray)

# Compute Structural Similarity Index (SSIM)
ssim_index, _ = ssim(ground_truth_gray, segmented_gray, full=True)

print("Root Mean Squared Error (RMSE):", rmse)
print("Peak Signal-to-Noise Ratio (PSNR):", psnr)
print("Structural Similarity Index (SSIM):", ssim_index)
