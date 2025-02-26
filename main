import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import color

# Load the images
image_path_1 = "/mnt/data/2016-10-20-10_09_2017-01-27-23_59_Sentinel-2_L2A_True_color.jpg"
image_path_2 = "/mnt/data/2024-08-19-00_00_2025-02-19-23_59_Sentinel-2_L2A_True_color.jpg"

image1 = cv2.imread(image_path_1)
image2 = cv2.imread(image_path_2)

# Ensure images are loaded properly
if image1 is None or image2 is None:
    raise ValueError("One or both image paths are incorrect or images could not be loaded.")

# Resize images to same dimensions
size = (512, 512)
image1 = cv2.resize(image1, size)
image2 = cv2.resize(image2, size)

# Convert images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute absolute difference
diff = cv2.absdiff(gray1, gray2)

# Apply thresholding to highlight changes
_, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Perform morphological operations to reduce noise
kernel = np.ones((5, 5), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Highlight changes on the original image
highlighted = image2.copy()
highlighted[thresh == 255] = [0, 0, 255]  # Highlight changes in red

# Display the images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
ax[0].set_title("Image 1")
ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
ax[1].set_title("Image 2")
ax[1].axis("off")
ax[2].imshow(cv2.cvtColor(highlighted, cv2.COLOR_BGR2RGB))
ax[2].set_title("Detected Changes")
ax[2].axis("off")
plt.show()

# Feature Extraction using HOG
def extract_features(image):
    image_gray = color.rgb2gray(image)
    features = hog(image_gray, pixels_per_cell=(16, 16), cells_per_block=(1, 1), feature_vector=True)
    return features

features1 = extract_features(image1)
features2 = extract_features(image2)

# Ensure feature vectors are the same length
min_length = min(len(features1), len(features2))
features1 = features1[:min_length]
features2 = features2[:min_length]

# Compute feature difference
diff_features = np.abs(features1 - features2)
change_score = np.sum(diff_features)
change_score
