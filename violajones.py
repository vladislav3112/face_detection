import cv2
from tkinter import filedialog
def openfilename():
    # open file dialog box to select image
    # The dialogue box has a title "Open"
    filename = filedialog.askopenfilename(title='"pen')
    return filename

url = "340_34.jpg"
#window_name = f"Detected Objects in {url}"
#original_image = cv2.imread(openfilename())

def viola_jones(url): 
    # Convert the image to grayscale for easier computation
    original_image = cv2.imread(url)
    image_grey = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    eye_classifier = cv2.CascadeClassifier("__xml/haarcascade_eye.xml")
    face_classifier = cv2.CascadeClassifier("__xml/haarcascade_frontalface_alt.xml")


    detected_face = face_classifier.detectMultiScale(image_grey, minSize=(50, 50))
    detected_eye = eye_classifier.detectMultiScale(image_grey, minSize=(50, 50))

    # Draw rectangles on eyes
    if len(detected_eye) != 0:
        for (x, y, width, height) in detected_eye:
            cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 2)
    # Draw rectangles on face
    if len(detected_face) != 0:
        for (x, y, width, height) in detected_face:
            cv2.rectangle(original_image, (x, y),
                      (x + height, y + width),
                      (255, 0, 0), 2)
    cv2.imwrite("result1.jpg",original_image)
#cv2.namedWindow(window_name, cv2.WINDOW_KEEPRATIO)
#cv2.imshow(window_name, original_image)
#cv2.resizeWindow(window_name, 400, 400)
cv2.waitKey(0)
cv2.destroyAllWindows()