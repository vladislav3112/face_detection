from violajones import *
from re import TEMPLATE
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from template_matching import *
import tkinter
import pathlib, os

SOURCE_URL = ""
def open_img(row_pos):
    # Select the Imagename  from a folder
    x = openfilename()
    
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
    viola_jones(SOURCE_URL)
    print("OK")
    current_dir = pathlib.Path(__file__).parent.resolve() # current directory
    img_path = os.path.join(current_dir, "result1.jpg")
    my_img = Image.open(img_path)
    img = my_img.resize((250, 250), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel1 = Label(root, image=img)

    # set the image as img
    panel1.image = img
    panel1.grid(row=2,column=2,sticky="NEWS")
    print(img_path)
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