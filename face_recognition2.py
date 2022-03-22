import numpy as np
import skimage.color
import skimage.io
import os
import matplotlib.pyplot as plt

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = skimage.io.imread(fname=os.path.join(folder,filename), as_gray=True)
        if image is not None:
            images.append(image)
    return images

# read the image of a plant seedling as grayscale from the outset
# display the image
#fig, ax = plt.subplots()


BINS_NUM = 8
train_images = load_images_from_folder("train")
test_images = load_images_from_folder("test")
train = []
for train_image in train_images:
    train_img, bins, patches =  plt.hist(train_image.flatten(), bins=BINS_NUM)
    train.append(train_img)

test = []
for test_image in test_images:
    test_img, bins, patches =  plt.hist(test_image.flatten(), bins=BINS_NUM)
    test.append(test_img)
res_idx = []
for test_arr in test:
    min_diff = 513
    idx = 0
    min_idx = -1
    for train_arr in train:    
        diff =  (np.max(np.subtract(train_arr,test_arr)) - np.min(np.subtract(train_arr,test_arr))) / 2
        if(diff < min_diff):
            min_diff = diff
            min_idx = idx
        idx += 1
    res_idx.append(min_idx)
plt.clf()

rows = 2
columns = 2
idx = 0
for elem in res_idx:
    fig, ax = plt.subplots()
    fig.add_subplot(rows, columns, 1)
  
    # showing image
    plt.imshow(test_images[idx],cmap='gray')
    plt.axis('off')
    plt.title("Test")
  
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(train_images[elem],cmap='gray')
    plt.axis('off')
    plt.title("Train")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 3)
    plt.hist(test_images[idx].flatten(),bins = BINS_NUM)
    plt.axis('off')

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 4)
    plt.hist(train_images[elem].flatten(),bins = BINS_NUM)
    plt.axis('off')
    plt.show()
    idx += 1

# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("pixel count")
# train1, bins, patches =  plt.hist(image.flatten(), bins=128)  # <- or here
# plt.show()