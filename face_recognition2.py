from scipy.stats import mode
from xmlrpc.client import MAXINT
import numpy as np
import skimage.color
import skimage.io
import os
import matplotlib.pyplot as plt
import cv2

DISPLAY_PARAM = 0 # display res images or return score
CALCULATE_BEST = 1
PARALLEL_METHOD = 0 # when we use parallel method it is 0

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = skimage.io.imread(fname=os.path.join(folder,filename), as_gray=True)
        image = cv2.imread(os.path.join(folder,filename), 0)
        if image is not None:
            images.append(image)
    return images
def load_images_from_folders(folders_num,start_idx,train_len):
    test_images = []
    train_images = []
    folder_base = "cross_val"
    for folder_idx in range (folders_num):
        folder = folder_base + "/" + str(folder_idx+1)
        curr_idx = 0
        for filename in os.listdir(folder):
            image = skimage.io.imread(fname=os.path.join(folder,filename), as_gray=True)
            image = cv2.imread(os.path.join(folder,filename), 0)
            if image is not None:
                if(curr_idx >= start_idx and curr_idx < start_idx + train_len):
                    train_images.append(image)
                else:
                    test_images.append(image)
            curr_idx += 1
    return train_images, test_images
def face_recognition_hist(BINS_NUM):
    
    scores =[]
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
        scores.append(min_diff)
    plt.clf()
    if(not DISPLAY_PARAM):
        return scores
    if(PARALLEL_METHOD):
        return res_idx
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
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        idx += 1
def face_recognition_scale(scale_param):
    
    scores = []
    train = []
    for train_image in train_images:
        scaled_img = cv2.resize(train_image.copy(), (0, 0), fx=1/scale_param, fy=1/scale_param)
        train.append(scaled_img)

    test = []
    for test_image in test_images:
        scaled_img = cv2.resize(test_image.copy(), (0, 0), fx=1/scale_param, fy=1/scale_param)
        test.append(scaled_img)
    res_idx = []
    for test_arr in test:
        min_diff = MAXINT
        idx = 0
        min_idx = -1
        for train_arr in train:    
            diff =  abs(np.double(np.sum(train_arr)) - np.double(np.sum(test_arr)))
            if(diff < min_diff):
                min_diff = diff
                min_idx = idx
            idx += 1
        #print(min_diff)
        res_idx.append(min_idx)
        scores.append(min_diff)
    plt.clf()
    if(not DISPLAY_PARAM):
        return scores
    if(PARALLEL_METHOD):
        return res_idx

    rows = 2
    columns = 2
    idx = 0
    for elem in res_idx:
        fig, ax = plt.subplots()
        fig.add_subplot(rows, columns, 1)
        plt.draw()
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
        plt.imshow(test[idx],cmap='gray')
        plt.axis('off')
        plt.title("Test scaled")

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        plt.imshow(train[elem],cmap='gray')
        plt.axis('off')
        plt.title("Train scaled")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        idx += 1

def face_recognition_dft(components_num):
    
    scores = []
    train = []
    for train_image in train_images:
        imf = np.float32(train_image)/255.0 # the dft +scaling
        imgcv = cv2.dft(imf)#*255.0)
        train.append(np.abs(imgcv[0:components_num, 0:components_num]))

    test = []
    for test_image in test_images:
        imf = np.float32(test_image)/255.0
        imgcv = cv2.dft(imf)#*255.0)# the dft +scaling
        test.append(np.abs(imgcv[0:components_num, 0:components_num]))
    res_idx = []
    for test_arr in test:
        min_diff = MAXINT
        idx = 0
        min_idx = -1
        for train_arr in train:    
            diff =  abs(np.double(np.sum(train_arr)) - np.double(np.sum(test_arr)))
            if(diff < min_diff):
                min_diff = diff
                min_idx = idx
            idx += 1
        #print(min_diff)
        res_idx.append(min_idx)
        scores.append(min_diff)
    plt.clf()
    if(not DISPLAY_PARAM):
        return scores
    if(PARALLEL_METHOD):
        return res_idx
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
        plt.imshow(test[idx],cmap='gray')
        #plt.axis('off')
        plt.title("Test scaled")

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        plt.imshow(train[elem],cmap='gray')
        #plt.axis('off')
        plt.title("Train scaled")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        idx += 1

def face_recognition_dct(components_num):
    
    scores = []
    train = []
    for train_image in train_images:
        imf = np.float32(train_image)/255.0 # the dft +scaling
        imgcv = cv2.dct(imf)
        train.append(imgcv[0:components_num, 0:components_num])

    test = []
    for test_image in test_images:
        imf = np.float32(test_image)/255.0
        imgcv = cv2.dct(imf)# the dft +scaling
        test.append(imgcv[0:components_num, 0:components_num])
    res_idx = []
    for test_arr in test:
        min_diff = MAXINT
        idx = 0
        min_idx = -1
        for train_arr in train:    
            diff =  abs(np.double(np.sum(train_arr)) - np.double(np.sum(test_arr)))
            if(diff < min_diff):
                min_diff = diff
                min_idx = idx
            idx += 1
        #print(min_diff)
        res_idx.append(min_idx)
        scores.append(min_diff)
    plt.clf()
    if(not DISPLAY_PARAM):
        return scores
    if(PARALLEL_METHOD):
        return res_idx
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
        plt.imshow(test[idx],cmap='gray')
        plt.axis('off')
        plt.title("Test scaled")

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        plt.imshow(train[elem],cmap='gray')
        plt.axis('off')
        plt.title("Train scaled")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        idx += 1

def face_recognition_grad(width):
    
    scores = []
    train = []
    for train_image in train_images: 
        lines_arr = []
        for idx in range(width,train_image.shape[1]-width):
            curr_line1 = train_image[idx - width]
            curr_line2 = train_image[idx + width]
            lines_arr.append(abs(np.double(np.sum(curr_line1)) - np.double(np.sum(curr_line2))))
        avg_elem = np.average(lines_arr)
        res = []
        for elem in lines_arr:
            if(elem < avg_elem):
                res.append(0)
            else:
                res.append(1)
        train.append(res)

    test = []
    for test_image in test_images:
        lines_arr = []
        for idx in range(width,test_image.shape[1]-width):
            curr_line1 = test_image[idx - width]
            curr_line2 = test_image[idx + width]
            lines_arr.append(abs(np.double(np.sum(curr_line1)) - np.double(np.sum(curr_line2))))
        avg_elem = np.average(lines_arr)
        res = []
        for elem in lines_arr:
            if(elem < avg_elem):
                res.append(0)
            else:
                res.append(1)
        test.append(res)
    res_idx = []
    for test_arr in test:
        min_diff = MAXINT
        idx = 0
        min_idx = -1
        for train_arr in train:    
            diff =  abs(np.double(np.sum(train_arr)) - np.double(np.sum(test_arr)))
            if(diff < min_diff):
                min_diff = diff
                min_idx = idx
            idx += 1
        #print(min_diff)
        res_idx.append(min_idx)
        scores.append(min_diff)
    plt.clf()
    if(not DISPLAY_PARAM):
        return scores
    if(PARALLEL_METHOD):
        return res_idx
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
        #plt.imshow(test[idx],cmap='gray')
        plt.axis('off')
        plt.title("Test scaled")

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 4)
        #plt.imshow(train[elem],cmap='gray')
        plt.axis('off')
        plt.title("Train scaled")
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()
        idx += 1


#interface part
from violajones import *
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from template_matching import *
import tkinter
import pathlib, os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg




def openfolder(is_train):
    global TRAIN_PATH
    global TEST_PATH
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    folder_selected = filedialog.askdirectory()
    if(is_train):
        TRAIN_PATH = folder_selected
    else:
        TEST_PATH = folder_selected
    return folder_selected



best_param_vec = [108,5.5,25,9,24]
def calculate_res(train_path, test_path):
    global train_images
    global test_images
    
    text = text_edit.get(0.3,END)
    print(text)
    try:
        param = int(text)
        label_param.config(text = "Write parameter value (now set as " + text + " )")
    except Exception:
        label_param.config(text = "Write parameter value (number only!):")
    train_images, test_images = load_images_from_folders(folders_num=5,start_idx=0, train_len=2)
    #train_images = load_images_from_folder(train_path)
    #test_images = load_images_from_folder(test_path)
    plot.clear()
    if(var.get() == 1):
        face_recognition_hist(BINS_NUM=param)
        if(CALCULATE_BEST):
            
            best = MAXINT
            best_param = -1
            scores = []
            for i in range(32,128,2):
                curr_score = np.array(face_recognition_hist(BINS_NUM=i)).mean()
                scores.append(curr_score)
                if (curr_score < best):
                    best_param = i
                    best = curr_score
            label_param.config(text = "Best parameter value: " + str(best_param))
            plot.plot(range(32,128,2), scores, color="blue", marker="o")
    elif(var.get() == 2):
        face_recognition_scale(scale_param=param)
        best = MAXINT
        best_param = -1
        scores = []
        if(CALCULATE_BEST):
            for i in np.arange(2,6,0.5):
                curr_score = np.array(face_recognition_scale(scale_param=i)).mean()
                scores.append(curr_score)
                if (curr_score < best):
                    best_param = i
                    best = curr_score
            label_param.config(text = "Best parameter value: " + str(best_param))
            plot.plot(np.arange(2,6,0.5), scores, color="blue", marker="o")
    elif(var.get() == 3):
        face_recognition_dct(components_num=param)
        best = MAXINT
        best_param = -1
        scores = []
        if(CALCULATE_BEST):
            for i in np.arange(4,32):
                curr_score = np.array(face_recognition_dct(components_num=i)).mean()
                scores.append(curr_score)
                if (curr_score < best):
                    best_param = i
                    best = curr_score
            label_param.config(text = "Best parameter value: " + str(best_param))
            plot.plot(np.arange(4,32), scores, color="blue", marker="o")
    elif(var.get() == 4):
        face_recognition_dft(components_num=param)
        best = MAXINT
        best_param = -1
        scores = []
        if(CALCULATE_BEST):
            for i in np.arange(4,32):
                curr_score = np.array(face_recognition_dft(components_num=i)).mean()
                scores.append(curr_score)
                if (curr_score < best):
                    best_param = i
                    best = curr_score
            label_param.config(text = "Best parameter value: " + str(best_param))
            plot.plot(np.arange(4,32), scores, color="blue", marker="o")
    elif(var.get() == 5):
        face_recognition_grad(width=param)
        best = MAXINT
        best_param = -1
        scores = []
        if(CALCULATE_BEST):
            for i in np.arange(1,30):
                curr_score = np.array(face_recognition_grad(width=i)).mean()
                scores.append(curr_score)
                if (curr_score < best):
                    best_param = i
                    best = curr_score
            label_param.config(text = "Best parameter value: " + str(best_param))
            plot.plot(np.arange(1,30), scores, color="blue", marker="o")
    else:
        global PARALLEL_METHOD
        PARALLEL_METHOD = 1
        vec1 = face_recognition_hist(BINS_NUM=best_param_vec[0])
        vec2 = face_recognition_scale(scale_param=best_param_vec[1])
        vec3 = face_recognition_dft(components_num=best_param_vec[2])
        vec4 = face_recognition_dct(components_num=best_param_vec[3])
        vec5 = face_recognition_grad(width=best_param_vec[4])
        result = mode([vec1,vec2,vec3,vec4,vec5])[0][0]
        print(result)
        #plotting result:
        rows = 1
        columns = 2
        idx = 0
        for elem in result:
            fig, ax = plt.subplots()
            fig.add_subplot(rows, columns, 1)
            plt.draw()
            # showing image
            plt.imshow(test_images[idx],cmap='gray')
            plt.axis('off')
            plt.title("Test")
  
            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)
            plt.imshow(train_images[elem],cmap='gray')
            plt.axis('off')
            plt.title("Train voted")
            plt.get_current_fig_manager().full_screen_toggle()
            plt.show(block=False)
            plt.pause(0.5)
            plt.close()
            idx += 1    
        print("OK")



# Create a window
root = Tk()
var = IntVar()
# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("550x300+300+150")

#init plot
figure = Figure(figsize=(4, 3), dpi=100)
plot = figure.add_subplot(1, 1, 1)

# Allow Window to be resizable
root.resizable(width=True, height=True)

# Create a button and place it into the window using grid layout
def sel():
   selection = "You selected the option " + str(var.get())
   label.config(text = selection)
R1 = Radiobutton(root, text="hist", variable=var, value=1,
                  command=sel)
R2 = Radiobutton(root, text="scale", variable=var, value=2,
                  command=sel)
R3 = Radiobutton(root, text="dct", variable=var, value=3,
                  command=sel)
R4 = Radiobutton(root, text="dft", variable=var, value=4,
                  command=sel)
R5 = Radiobutton(root, text="gradient", variable=var, value=5,
                  command=sel)
R6 = Radiobutton(root, text="parallel system", variable=var, value=6,
                  command=sel)
label = Label(root)
label.grid(row=2,column=0)
#write param to window:
label_param = Label(root)
label_param.config(text = "Write parameter value:")
label_param.grid(row=3,column=1)
#label for best param:
label_best = Label(root)
label_best.grid(row=3,column=3)


text_edit = Text(root, width=4, height=1)

text_edit.grid(row=3,column=2)

btn1 = Button(root, text='select train folder', command=lambda : openfolder(is_train=True))
btn2 = Button(root, text='select test folder', command=lambda : openfolder(is_train=False))
btn3 = Button(root, text='calculate res', command=lambda: calculate_res(TRAIN_PATH,TEST_PATH))

btn1.grid(row=1,column=0)
btn2.grid(row=1,column=2)
btn3.grid(row=1,column=4)

R1.grid(row=3,column=0)
R2.grid(row=4,column=0)
R3.grid(row=5,column=0)
R4.grid(row=6,column=0)
R5.grid(row=7,column=0)
R6.grid(row=8,column=0)

#set loaction of plot
canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().grid(row=8,column=4)
root.mainloop()
