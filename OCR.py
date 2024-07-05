# Import Dependencies
from easyocr import Reader
from typing import List

# Initiate EasyOCR
reader = Reader(['en'])

def get_text(img:List) -> List:
    txt = reader.readtext(img)
    return txt


if __name__=="__main__":
    import cv2
    img = cv2.imread("test/1.jpg")
    print("Response:\n", get_text(img))
