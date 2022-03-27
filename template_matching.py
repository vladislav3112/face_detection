import cv2
import numpy as np
from matplotlib import pyplot as plt

url1 = "train/1.jpg"
url2 = "train/2.jpg"
def template_matching(sourse_url, template_url):
    THRESHOLD = 0.3 #For TM_CCOEFF_NORMED, larger values = good fit.

    img_rgb = cv2.imread(sourse_url,0)
    #img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_url, 0)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_rgb,template,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(img_rgb,top_left, bottom_right, (255,0,0), 2) 

    cv2.imwrite("result.jpg",img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()
template_matching(url1, url2)