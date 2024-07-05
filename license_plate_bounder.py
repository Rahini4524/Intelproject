# Import Dependencies
from ultralytics import YOLO
from typing import List

model = YOLO("weights/best.pt")

def YoloPredict(img_path:str) -> List:
    predicted = model(img_path)
    return predicted

if __name__=="__main__":
    while True:
        file_path = input("path:")
        print(YoloPredict(file_path))