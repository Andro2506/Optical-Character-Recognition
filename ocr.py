# pip install torch torchvision torchaudio easyocr opencv-python

# Example-1 (Optical Character Recognition)[OCR]

import easyocr
import cv2 as cv2
import numpy as np
from google.colab.patches import cv2_imshow

IMAGE_PATH = 'second.jpg'
canvas = np.ones((300, 200, 3))

reader = easyocr.Reader(['en'] , gpu=False)
result = reader.readtext(IMAGE_PATH)
print(result)

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX


img = cv2.imread(IMAGE_PATH)
spacer = 100
for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    canvas = cv2.putText(canvas,text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    
    spacer+=15
    
cv2_imshow(img)
cv2_imshow(canvas)

cv2.waitKey(0)

cv2.destroyAllWindows()



# Example-2 (Optical Character Recognition)[OCR]

IMAGE_PATH = 'first.png'
canvas = np.ones((300, 200, 3))

reader = easyocr.Reader(['en'] , gpu=False)
result = reader.readtext(IMAGE_PATH)
print(result)

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX


img = cv2.imread(IMAGE_PATH)
spacer = 100
for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    canvas = cv2.putText(canvas,text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    
    spacer+=15
    
cv2_imshow(img)
cv2_imshow(canvas)

cv2.waitKey(0)

cv2.destroyAllWindows()


# Example-3 (Optical Character Recognition)[OCR]

IMAGE_PATH = 'third.jpg'
canvas = np.ones((300, 200, 3))

reader = easyocr.Reader(['en'] , gpu=False)
result = reader.readtext(IMAGE_PATH)
print(result)

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX


img = cv2.imread(IMAGE_PATH)
spacer = 100
for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    # img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    canvas = cv2.putText(canvas,text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    
    spacer+=15
    
cv2_imshow(img)
cv2_imshow(canvas)

cv2.waitKey(0)

cv2.destroyAllWindows()



# Example-4 (Optical Character Recognition)[OCR]

IMAGE_PATH = 'four.jpg'
canvas = np.ones((300, 200, 3))

reader = easyocr.Reader(['en'] , gpu=False)
result = reader.readtext(IMAGE_PATH)
print(result)

top_left = tuple(result[0][0][0])
bottom_right = tuple(result[0][0][2])
text = result[0][1]
font = cv2.FONT_HERSHEY_SIMPLEX


img = cv2.imread(IMAGE_PATH)
spacer = 100
for detection in result: 
    top_left = tuple(detection[0][0])
    bottom_right = tuple(detection[0][2])
    text = detection[1]
    img = cv2.rectangle(img,top_left,bottom_right,(0,255,0),3)
    canvas = cv2.putText(canvas,text,(20,spacer), font, 0.5,(0,255,0),2,cv2.LINE_AA)
    
    spacer+=15
    
cv2_imshow(img)
cv2_imshow(canvas)

cv2.waitKey(0)

cv2.destroyAllWindows()