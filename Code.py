
import cv2
import csv
import collections
import numpy as np
from tracker import *
fr="dataset/test/benign/melanoma_9608.jpg"
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


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import random
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)
# Define a double convolution block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or x.size(1) != out.size(1):
            residual = nn.Conv2d(x.size(1), out.size(1), kernel_size=1, stride=self.stride, bias=False)(x)

        out += residual
        out = self.relu(out)

        return out
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

# Define the loss function
def dice_loss(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth)/(pred.sum() + target.sum() + smooth)
    return 1. - dice

# Define the performance metrics
def iou_score(pred, target, smooth=1.):
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    iou = (intersection + smooth) / (union + smooth)
    return iou

def dice_coef(pred, target, smooth=1.):
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth)/(pred.sum() + target.sum() + smooth)
    return dice
from tensorflow import keras
from tensorflow.keras import layers
IMG_HEIGHT = 256
IMG_WIDTH = 256
import torch
import torch.nn as nn
import torchvision
from einops import rearrange

# Vision Transformer Model
class VisionTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        # Patch embedding layer
        self.patch_embedding = nn.Conv2d(channels, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embedding
        self.positional_embedding = nn.Parameter(torch.randn(num_patches + 1, dim))

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim), num_layers=depth)

        # Classifier head
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embedding(x)
        x = rearrange(x, 'b c h w -> b (h w) c')  # Flatten the spatial dimensions into patches
        x = torch.cat((self.cls_token.expand(x.size(0), -1, -1), x), dim=1)  # Add classification token

        # Add positional embeddings
        x += self.positional_embedding
        x = x.permute(1, 0, 2)  # Transpose for transformer input

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Classification head
        x = x.mean(dim=0)  # Take mean over tokens
        x = self.fc(x)

        return x

# Example usage
image_size = 224
patch_size = 16
num_classes = 1000
dim = 512
depth = 6
heads = 8
mlp_dim = 2048

# Instantiate Vision Transformer model
vit_model = VisionTransformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)

# Example input
input_tensor = torch.randn(1, 3, image_size, image_size)  # Batch size 1, 3 channels (RGB), image size 224x224

# Forward pass
output = vit_model(input_tensor)
print("Output shape:", output.shape)

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense 
# ## Load Data
image_directory='dataset/test/'
no_tumor_images=os.listdir(image_directory+ 'benign/')
yes_tumor_images=os.listdir(image_directory+ 'malignant/')
print('Train: ', len(no_tumor_images))
print('Test: ',len(yes_tumor_images))
dataset=[]
label=[]
INPUT_SIZE=64
# ## Create labels 
for i , image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'benign/'+image_name)
        image=Image.fromarray(image,'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i , image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'malignant/'+image_name)
        image=Image.fromarray(image, 'RGB')
        image=image.resize((INPUT_SIZE,INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)
dataset=np.array(dataset)
label=np.array(label)
print('Dataset: ',len(dataset))
print('Label: ',len(label))
# ## Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.20, random_state=2523)
# ## Normalize the Data
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)
vit_model = VisionTransformer(image_size, patch_size, num_classes, dim, depth, heads, mlp_dim)
model=Sequential()
model.add(Conv2D(32, (3,3),activation='relu', input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=dice_loss, metrics=[dice_coef])


model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
history=model.fit(X_train, y_train, 
batch_size=32, 
verbose=1, epochs=20, 
validation_data=(X_test, y_test),
shuffle=False)
plt.figure(0)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# ## Save the Model

model.save('model.h5')


# ## Load Model 

model = load_model('model.h5')
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)

# ## Make Prediction on New Data

#show_result('1.png')
model.load_weights('model.h5')

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

acc = accuracy_score(y_test, y_pred.round())
scores = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
