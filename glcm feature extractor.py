import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import graycomatrix,graycoprops
from scipy.stats import entropy

#print(help('cv2'))
print(os.listdir("paintings_2"))
from skimage import io


#Resize images to
SIZE = 128

#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
#for directory_path in glob.glob("cell_images/train/*"):
for directory_path in glob.glob("paintings_2/train/*"):
    label = directory_path.split("\\")[-1]
    #print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        #print(img_path)
        img = cv2.imread(img_path, 0) #Reading color images
        img = cv2.resize(img, (SIZE, SIZE)) #Resize images
        train_images.append(img)
        train_labels.append(label)
        
train_images = np.array(train_images)
train_labels = np.array(train_labels)

#Do exactly the same for test/validation images
# test
test_images = []
test_labels = []
#for directory_path in glob.glob("cell_images/test/*"): 
for directory_path in glob.glob("paintings_2/test/*"):
    fruit_label = directory_path.split("\\")[-1]
    print(fruit_label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
        print(img_path)
        img = cv2.imread(img_path, 0)
        img = cv2.resize(img, (SIZE, SIZE))
        test_images.append(img)
        test_labels.append(fruit_label)
        
test_images = np.array(test_images)
test_labels = np.array(test_labels)

#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

# Normalize pixel values to between 0 and 1
#x_train, x_test = x_train / 255.0, x_test / 255.0

###################################################################
# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor_glcm(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        img = dataset[image, :,:]
        ################################################################
        #START ADDING DATA TO THE DATAFRAME
  
                
         #Full image
        #GLCM = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
        GLCM = graycomatrix(img, [1], [0])       
        GLCM_Energy = graycoprops(GLCM, 'energy')[0]
        df['Energy'] = GLCM_Energy
        GLCM_corr = graycoprops(GLCM, 'correlation')[0]
        df['Corr'] = GLCM_corr       
        GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
        df['Diss_sim'] = GLCM_diss       
        GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
        df['Homogen'] = GLCM_hom       
        GLCM_contr = graycoprops(GLCM, 'contrast')[0]
        df['Contrast'] = GLCM_contr


        GLCM2 = graycomatrix(img, [3], [0])       
        GLCM_Energy2 = graycoprops(GLCM2, 'energy')[0]
        df['Energy2'] = GLCM_Energy2
        GLCM_corr2 = graycoprops(GLCM2, 'correlation')[0]
        df['Corr2'] = GLCM_corr2       
        GLCM_diss2 = graycoprops(GLCM2, 'dissimilarity')[0]
        df['Diss_sim2'] = GLCM_diss2       
        GLCM_hom2 = graycoprops(GLCM2, 'homogeneity')[0]
        df['Homogen2'] = GLCM_hom2       
        GLCM_contr2 = graycoprops(GLCM2, 'contrast')[0]
        df['Contrast2'] = GLCM_contr2

        GLCM3 = graycomatrix(img, [5], [0])       
        GLCM_Energy3 = graycoprops(GLCM3, 'energy')[0]
        df['Energy3'] = GLCM_Energy3
        GLCM_corr3 = graycoprops(GLCM3, 'correlation')[0]
        df['Corr3'] = GLCM_corr3       
        GLCM_diss3 = graycoprops(GLCM3, 'dissimilarity')[0]
        df['Diss_sim3'] = GLCM_diss3       
        GLCM_hom3 = graycoprops(GLCM3, 'homogeneity')[0]
        df['Homogen3'] = GLCM_hom3       
        GLCM_contr3 = graycoprops(GLCM3, 'contrast')[0]
        df['Contrast3'] = GLCM_contr3

        GLCM4 = graycomatrix(img, [0], [np.pi/4])       
        GLCM_Energy4 = graycoprops(GLCM4, 'energy')[0]
        df['Energy4'] = GLCM_Energy4
        GLCM_corr4 = graycoprops(GLCM4, 'correlation')[0]
        df['Corr4'] = GLCM_corr4       
        GLCM_diss4 = graycoprops(GLCM4, 'dissimilarity')[0]
        df['Diss_sim4'] = GLCM_diss4       
        GLCM_hom4 = graycoprops(GLCM4, 'homogeneity')[0]
        df['Homogen4'] = GLCM_hom4       
        GLCM_contr4 = graycoprops(GLCM4, 'contrast')[0]
        df['Contrast4'] = GLCM_contr4
        
        GLCM5 = graycomatrix(img, [0], [np.pi/2])       
        GLCM_Energy5 = graycoprops(GLCM5, 'energy')[0]
        df['Energy5'] = GLCM_Energy5
        GLCM_corr5 = graycoprops(GLCM5, 'correlation')[0]
        df['Corr5'] = GLCM_corr5       
        GLCM_diss5 = graycoprops(GLCM5, 'dissimilarity')[0]
        df['Diss_sim5'] = GLCM_diss5       
        GLCM_hom5 = graycoprops(GLCM5, 'homogeneity')[0]
        df['Homogen5'] = GLCM_hom5       
        GLCM_contr5 = graycoprops(GLCM5, 'contrast')[0]
        df['Contrast5'] = GLCM_contr5
        
        #Add more filters as needed
        #entropy = shannon_entropy(img)
        #df['Entropy'] = entropy

        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        if(dataset.shape[0]==1):
            print("OK")
            #sns.heatmap(np.reshape(GLCM,(256,256)), annot=True)
            #splt.show()
        
    return image_dataset

def feature_extractor_sift(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        img = dataset[image, :,:]
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img,None)
        # showing image
        #print("Image")
        #imshow(gaussian)
        #show()
        # getting Speeded-Up Robust Features
        print("No of  points: {}".format(len(kp)))
        #df['Points'] = kp[:10]
        keypoints = sorted(kp, key = lambda x : x.size)
        df['Octaves'] = [o.octave for o in keypoints[:25]]
        df['Angles'] = [o.angle for o in keypoints[:25]]
        df['PointsX'] = [o.pt[0] for o in keypoints[:25]]
        df['PointsY'] = [o.pt[1] for o in keypoints[:25]]
        #df['Sizes'] = [o.size for o in keypoints[:20]]
        desflatten = [np.mean(arr) for arr in des]
        if(dataset.shape[0]==1):
            img=cv2.drawKeypoints(img,kp,image)
            ph_img = ImageTk.PhotoImage(Image.fromarray(img).resize((250, 250), Image.ANTIALIAS))
            panel2 = Label(root,image=ph_img)
            panel2.image=ph_img
            # set the image as img
            panel2.grid(row=3,column=0,sticky="NEWS")
            #plt.imshow(img)
            #plt.show()
        #df['Descriptor'] = desflatten
        ################################################################
        #START ADDING DATA TO THE DATAFRAME
        image_dataset = image_dataset.append(df)
    return image_dataset

def feature_extractor_shi_tomasi(dataset):
    image_dataset = pd.DataFrame()
    
    for image in range(dataset.shape[0]):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        training_img = dataset[image,:,:,]
        corners = cv2.goodFeaturesToTrack(image=training_img,maxCorners=58,minDistance=4,qualityLevel=0.04)
        corners = np.int0(corners)
        if(dataset.shape[0]) == 1:
            for i in corners:
                x,y = i.ravel()
                cv2.circle(training_img,(x,y),3,255,-1)
            ph_img = ImageTk.PhotoImage(Image.fromarray(training_img).resize((250, 250), Image.ANTIALIAS))
            panel2 = Label(root,image=ph_img)
            panel2.image=ph_img
            # set the image as img
            panel2.grid(row=3,column=0,sticky="NEWS")
        df['PointsX'] = [o[0][0] for o in corners]
        df['PointsY'] = [o[0][1] for o in corners]
        #df['descriptor'] = train_descriptor            
        image_dataset = image_dataset.append(df)
    return image_dataset


####################################################################
#Extract features from training images
image_features = feature_extractor_shi_tomasi(x_train)

 

X_for_ML = image_features
#Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

#Define the classifier
from sklearn.ensemble import RandomForestClassifier
#RF_model = RandomForestClassifier(n_estimators = 300,random_state = 42)#best for shi_tomasi
RF_model = RandomForestClassifier(n_estimators = 100, random_state = 42)#best for glcm,sift

#Can also use SVM but RF is faster and may be more accurate.
#from sklearn import svm
#SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
#SVM_model.fit(X_for_ML, y_train)

# Fit the model on training data
RF_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding

#Predict on Test data
#Extract features from test data and reshape, just like training data
test_features = feature_extractor_shi_tomasi(x_test.copy())
test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))

#Predict on test
test_prediction = RF_model.predict(test_for_RF)
#test_prediction=np.argmax(test_prediction, axis=1)
#Inverse le transform to get original label back. 
test_prediction = le.inverse_transform(test_prediction)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
plt.show()



## INTERFACE PART:
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

from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from template_matching import *
import tkinter
import pathlib, os

SOURCE_URL = ""
def open_img(row_pos):
    global SOURCE_URL
    global POS
    # Select the Imagename  from a folder
    x = openfilename()
    # opens the image
    img = Image.open(x)
    code = x[x.find("__")+2:x.find(".")]
    print(code)
    POS = int(code)#get pos of numper in path
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

def calculate_res(idx):
    img = x_test[idx].copy()

    #Extract features and reshape to right dimensions
    input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
    input_img_features=feature_extractor_shi_tomasi(input_img)
    input_img_features = np.expand_dims(input_img_features, axis=0)
    input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
    #Predict
    img_prediction = RF_model.predict(input_img_for_RF)
    #img_prediction=np.argmax(img_prediction, axis=1)
    img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
    print("The prediction for this image is: ", img_prediction)
    print("The actual label for this image is: ", test_labels[idx])

    print("OK")
    str = "The prediction for this image is: " + img_prediction[0]+'\n'+"The actual label for this image is: " + test_labels[idx]
    #current_dir = pathlib.Path(__file__).parent.resolve() # current directory
    #img_path = os.path.join(current_dir, "result1.jpg")
    #my_img = Image.open(img_path)
    #img = my_img.resize((250, 250), Image.ANTIALIAS)
    #img = ImageTk.PhotoImage(img)
    panel1 = Label(root,text=str)

    # set the image as img
    panel1.grid(row=2,column=3,sticky="NEWS")
    #print(img_path)
    
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
var_check = tkinter.IntVar()
btn1 = Button(root, text='open source image', command=lambda : open_img(row_pos=0))
btn3 = Button(root, text='calculate res', command=lambda: calculate_res(idx=POS))

btn1.grid(row=1,column=0)
btn3.grid(row=1,column=2)
R1 = Radiobutton(root, text="glcm", variable=var_check, value=1)
R2 = Radiobutton(root, text="SIFT", variable=var_check, value=2)
R3 = Radiobutton(root, text="Shi-Tomashi", variable=var_check, value=3)
R1.grid(row=2,column=2)
R2.grid(row=3,column=2)
R3.grid(row=4,column=2)
root.mainloop()