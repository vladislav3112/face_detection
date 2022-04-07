from xmlrpc.client import MAXINT
import numpy as np
import skimage.color
import skimage.io
import os
import matplotlib.pyplot as plt
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = skimage.io.imread(fname=os.path.join(folder,filename), as_gray=True)
        image = cv2.imread(os.path.join(folder,filename), 0)
        if image is not None:
            images.append(image)
    return images

img_abstract = load_images_from_folder("D:\study+projects\\face recognition\paintings\\abstract")
img_barokko = load_images_from_folder("D:\study+projects\\face recognition\paintings\\barokko")
img_impress = load_images_from_folder("D:\study+projects\\face recognition\paintings\\impressionism")

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from template_matching import *
import tkinter
import pathlib, os

SOURCE_URL = ""
def open_img(row_pos):
    global SOURCE_URL
    # Select the Imagename  from a folder
    x = openfilename()
    SOURCE_URL = x
    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.grid(row=2,column=row_pos)


def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')
    return filename

def calculate_res():
    number = painting_type_breif(SOURCE_URL)
    print("OK")
    #current_dir = pathlib.Path(__file__).parent.resolve() # current directory
    #img_path = os.path.join(current_dir, "result1.jpg")
    #my_img = Image.open(img_path)
    #img = my_img.resize((250, 250), Image.ANTIALIAS)
    #img = ImageTk.PhotoImage(img)
    #panel1 = Label(root, image=img)

    # set the image as img
    #panel1.image = img
    #panel1.grid(row=2,column=2,sticky="NEWS")
    #print(img_path)
    
def painting_type_breif(SOURCE_URL):
    training_type = []
    matches_by_type = []
    for idx in range(3):
        if(idx == 0):
            training_type = img_abstract
        if(idx == 1):
            training_type = img_barokko
        if(idx == 2):
            training_type = img_impress
        total_matches = 0
        for i in range(len(training_type)):
            training_image = training_type[i]
            training_image = cv2.resize(training_image,(250, 250), Image.ANTIALIAS)
            test_image = cv2.imread(SOURCE_URL)
            test_image = cv2.resize(test_image,(250, 250), Image.ANTIALIAS)
            training_gray = training_image
            test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
            fast = cv2.FastFeatureDetector_create() 
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

            train_keypoints = fast.detect(training_gray, None)
            test_keypoints = fast.detect(test_gray, None)

            train_keypoints, train_descriptor = brief.compute(training_gray, train_keypoints)
            test_keypoints, test_descriptor = brief.compute(test_gray, test_keypoints)

            #keypoints_without_size = np.copy(training_image)
            #keypoints_with_size = np.copy(training_image)

            #cv2.drawKeypoints(training_image, train_keypoints, keypoints_without_size, color = (0, 255, 0))

            #cv2.drawKeypoints(training_image, train_keypoints, keypoints_with_size, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display image with and without keypoints size
            #fx, plots = plt.subplots(1, 1, figsize=(20,10))

            #plots[0].set_title("Train keypoints With Size")
            #plots[0].imshow(keypoints_with_size, cmap='gray')

            #plots[1].set_title("Train keypoints Without Size")
            #plots[1].imshow(keypoints_without_size, cmap='gray')

            # Print the number of keypoints detected in the training image
            print("Number of Keypoints Detected In The Training Image: ", len(train_keypoints))

            # Print the number of keypoints detected in the query image
            print("Number of Keypoints Detected In The Query Image: ", len(test_keypoints))


            # ## Matching Keypoints

            # Create a Brute Force Matcher object.
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

            # Perform the matching between the BRIEF descriptors of the training image and the test image
            matches = bf.match(train_descriptor, test_descriptor)

            # The matches with shorter distance are the ones we want.
            matches = sorted(matches, key = lambda x : x.distance)
            score = 0.0
            for elem in matches:
                score += 1.0 / (elem.distance)
            print("Total matches: ", len(matches))
            if (total_matches != max(total_matches, score)):
                total_matches = max(total_matches, score)
                result = cv2.drawMatches(training_image, train_keypoints, test_gray, test_keypoints, matches, test_gray, flags = 2)
                
            # Display the best matching points
                plt.rcParams['figure.figsize'] = [14.0, 7.0]
                plt.title('Best Matching Points')
        plt.imshow(result)
        plt.show()
        matches_by_type.append(total_matches)
            #
            #plt.show()
    print(matches_by_type)

def painting_type_pca(SOURCE_URL):
    return 0
# Create a window
root = Toplevel()
# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("550x300+300+150")



# Allow Window to be resizable
root.resizable(width=True, height=True)

# Create a button and place it into the window using grid layout
var1 = tkinter.IntVar()
var2 = tkinter.IntVar()
btn1 = Button(root, text='open source image', command=lambda : open_img(row_pos=0))
btn3 = Button(root, text='calculate res', command=lambda: calculate_res())

btn1.grid(row=1,column=0)
btn3.grid(row=1,column=2)

root.mainloop()


# img = cv2.imread(SOURCE_URL,0)
#     # Initiate FAST detector
#     star = cv2.xfeatures2d.StarDetector_create()
#     # Initiate BRIEF extractor
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
#     # find the keypoints with STAR
#     kp = star.detect(img,None)
#     # compute the descriptors with BRIEF
#     kp, des = brief.compute(img, kp)
#     print( brief.descriptorSize() )
#     print( des.shape )
#     return 1