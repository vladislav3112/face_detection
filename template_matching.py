import cv2
import numpy as np
from matplotlib import pyplot as plt

url1 = "different lighting.jpg"
url2 = 'different lighting template.jpg'
def template_matching(sourse_url, template_url):
    THRESHOLD = 0.3 #For TM_CCOEFF_NORMED, larger values = good fit.

    img_rgb = cv2.imread(sourse_url)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_url, 0)

    height, width = template.shape[::]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    plt.imshow(res, cmap='gray')

    
    loc = np.where( res >= THRESHOLD)  

    for pt in zip(*loc[::-1]): 
        cv2.rectangle(img_rgb, pt, (pt[0] + width, pt[1] + height),(255, 255, 0), 3) 

    cv2.imwrite("result.jpg",img_rgb)
    cv2.waitKey()
    cv2.destroyAllWindows()
template_matching(url1, url2)