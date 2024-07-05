from typing import List
from license_plate_bounder import predict
from OCR import get_text
import cv2

if __name__ == "__main__":
    image_path:str = input("Enter the path of the image:")
    bounders:List = predict(img_path=image_path)
    img = cv2.imread(image_path)
    res = []
    for bounder in bounders:
        for box in bounder.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            res.append(img[y1:y2, x1:x2])
    print(len(res))
    for crop_img in res:
        print(get_text(crop_img))

            






# print(predict("D:\\Rahini\\dataset\\test\\images\\33746_jpg.rf.240a2a6b1149ba8da713476095a53520.jpg"))
