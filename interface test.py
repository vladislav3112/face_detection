from re import TEMPLATE
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from template_matching import *
import tkinter
import pathlib, os

TEMPLATE_URL =""
SOURCE_URL = ""
def open_img(row_pos, is_template=False):
    global TEMPLATE_URL
    global SOURCE_URL
    # Select the Imagename  from a folder
    x = openfilename()
    
    # opens the image
    img = Image.open(x)

    # resize the image and apply a high-quality down sampling filter
    #img = img.resize((250, 250), Image.ANTIALIAS)

    # PhotoImage class is used to add image to widgets, icons etc
    img = ImageTk.PhotoImage(img)

    # create a label
    panel = Label(root, image=img)

    # set the image as img
    panel.image = img
    panel.grid(row=row_pos)
    if(is_template):
        TEMPLATE_URL = x
        var2.set(1)
    else:
        SOURCE_URL = x
        var1.set(1)


def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')
    return filename

def calculate_res():
    template_matching(sourse_url=SOURCE_URL,template_url=TEMPLATE_URL)
    print("OK")
    current_dir = pathlib.Path(__file__).parent.resolve() # current directory
    img_path = os.path.join(current_dir, "result.jpg")
    img1 = ImageTk.PhotoImage(file=img_path)
    panel1 = Label(root, image=img1)

    # set the image as img
    panel1.image = img1
    panel1.grid(row=2,column=4)
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
btn1 = Button(root, text='open source image', command=lambda : open_img(row_pos=2))
btn2 = Button(root, text='open template image', command=lambda: open_img(row_pos=4,is_template=True))
btn3 = Button(root, text='calculate res', command=lambda: calculate_res())

btn1.grid(row=1,column=0)
btn2.grid(row=1,column=2)
btn3.grid(row=1,column=4)

root.mainloop()