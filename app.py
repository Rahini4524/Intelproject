import glob
import os
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET
import cv2
from torchvision.transforms import Normalize
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from typing import List
from license_plate_bounder import YoloPredict
from OCR import get_text
import numpy as np
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 -> input channels, 6 -> output channels, 5 -> kernal_size
        self.pool = nn.MaxPool2d(2, 2)   # 2 -> 2X2 kernal, 2 -> stride
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*7*12, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        # -> n, 3, 40, 60
        out = self.pool(F.relu(self.conv1(x)))    # -> n, 6, 36, 56  -> n, 6, 18, 28
        out = self.pool(F.relu(self.conv2(out)))  # -> n, 16, 14, 24 -> n, 16, 7, 12
        out = out.view(-1, 16*7*12)               # -> n, 1344
        out = F.relu(self.fc1(out))               # -> n, 50
        out = F.sigmoid(self.fc2(out))            # -> n, 1
        return out


model = torch.load("models/CNNmodel.pth", map_location=device)

model.eval()
model.to(device)   

def predict(images):
    return model(images.to(device))


tree = ET.parse('annotations\02376171-6ec8-48bd-80f7-718e9a9137db.xml')
root = tree.getroot()
bound_boxes = []
for item in root.findall('object'):
    boundbox = item.find('bndbox')
    xmin = int(boundbox.find('xmin').text)
    ymin = int(boundbox.find('ymin').text)
    xmax = int(boundbox.find('xmax').text)
    ymax = int(boundbox.find('ymax').text)
    bound_boxes.append((xmin, ymin, xmax, ymax))

test_image = glob.glob("Test Imagea/*.jpg")


selected = option_menu(
    options = ["Occupancy analysis", "Number plate analysis"],
    orientation="horizontal",
    menu_title=None,
)

def process(image_path):
    img = cv2.imread(image_path)

    grids = []
    annotation_dict = {}
    normalizer = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    for idx, boxes in enumerate(bound_boxes):

        x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]

        cropped_img = img[y1:y2, x1:x2]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        cropped_img = cv2.resize(cropped_img,(40, 60))
        cropped_img = torch.tensor(cropped_img, dtype=torch.float32).permute(2, 0, 1) / 255 # Normalize to [0, 1]
        cropped_img = normalizer(cropped_img)

        grids.append(cropped_img)

    tensor_grids = torch.stack(grids, dim=0)
    # print(tensor_grids.shape)

    predicted = predict(tensor_grids)
    # print(bound_boxes)
    result = [0 if val <.5  else 1 for val in predicted]
    # print(result)

    for box, occupy in zip(bound_boxes, result):
        x1, y1, x2, y2 = box

        if occupy == 1:
            cv2.rectangle(img, (x1, y1), (x2, y2), ( 0, 0, 255), 2)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite("result.jpg", img)
    return result
def to_percentage(values):
    total = sum(values)
    if total == 0:
        # Avoid division by zero if the total sum is zero
        return [0] * len(values)
    
    percentage_values = [(value / total) * 100 for value in values]
    return percentage_values

def save_uploaded_file(uploaded_file, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        return file_path


def main():
    data1 = {
        'position': ['plot 1', 'plot 2', 'plot 3', 'plot 4', 'plot 5', 'plot 6', 'plot 7'],
        'status': ['Null', 'Null', 'Null', 'Null', 'Null', 'Null', 'Null']
    }
    df1 = pd.DataFrame(data1)

    data2 = {
        'position': ['plot 8', 'plot 9', 'plot 10', 'plot 11', 'plot 12', 'plot 13', 'plot 14'],
        'status': ['Null', 'Null', 'Null', 'Null', 'Null', 'Null', 'Null']
    }
    df2 = pd.DataFrame(data2)

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css")

    st.title("Image Processing App")
    if selected == "Occupancy analysis":

        st.markdown("**Parking lot occupancy analysis**")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        st.markdown("""
    **Note:** **Please select an image from the `cropped_images` folder.** 
    """)
        if uploaded_file is not None:
            uploaded_file_path = save_uploaded_file(uploaded_file, "uploaded_files")

            image = Image.open(uploaded_file)
            border_width = 5
            border_color = 'black'
            st.write("**UPLOADED IMAGE**")
            bordered_image = ImageOps.expand(image, border=border_width, fill=border_color)
            st.image(bordered_image, use_column_width=True)
            st.markdown("""
                    <style>
                    .stButton button {
                        width: 600px;
                    }
                    </style>
                """, unsafe_allow_html=True)
            
            if st.button("click here for  update status"):
                print(f"Uploaded File Path: {uploaded_file_path}")
                result = process(uploaded_file_path)
                history = pd.read_csv("history.csv")
                currentHistory = pd.DataFrame([result], columns=history.columns)
                history = pd.concat([history, currentHistory],ignore_index=True)
                history.to_csv("history.csv", index=False)
                total =  history.sum(axis=0)
                result_percent = to_percentage(result)
                max_index = np.argmax(total.values)
                ratio = to_percentage(total.values)

                # Create a bar plot
                fig, ax = plt.subplots(figsize=(10, 5))
                plt.bar(range(len(ratio)), ratio, color='green')
                plt.xlabel('Parking Plots')
                plt.ylabel('Occupancy Ratio (%)')
                plt.title('Vehicle Movement Analysis')
                

                ax.set_xticks(range(len(ratio)))
                ax.set_xticklabels(range(1, len(ratio) + 1))
                
                

                # Function to color the updated values
                def color_cell(val):
                    if val == 'available':
                        return 'color: green'
                    
                    elif val == 'occupied':
                        return 'color: red'
                    else:
                        return 'color: inherit'

                # Apply styles to the specific column
                def apply_styles(df, column_name):
                    styled_df = df.style.applymap(color_cell, subset=[column_name])
                    return styled_df

                # Get the styled DataFrames
                styled_df1 = apply_styles(df1, 'status')
                styled_df2 = apply_styles(df2, 'status')
                
                for i in range(7):
                    
                    df1.loc[i,"status"] = "available" if result[i] == 0 else "occupied"
                    
                for i, j in zip(range(7), range(7, 14)):
                    df2.loc[i,"status"] = "available" if result[j] == 0 else "occupied"

                # Display the styled DataFrames side by side using st.columns
                col1, col2 = st.columns(2)

                with col1:
                    st.write(styled_df1.to_html(escape=False), unsafe_allow_html=True)

                with col2:
                    st.write(styled_df2.to_html(escape=False), unsafe_allow_html=True)

                processed_image = Image.open(r"result.jpg")

                st.write("**PROCESSED IMAGE**")
                border_width = 5
                border_color = 'black'
                bordered_image_1 = ImageOps.expand(processed_image)
                st.image(bordered_image_1, use_column_width=True)

                st.title("Occupancy Analysis")

                
                st.pyplot(fig)
                st.markdown(f"<span style='font-size:20px;'> <b>Frequently Accessed Plot Number:</b> {max_index+1}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:20px;'><b>Percentage:</b> {result_percent[max_index]:.2f}%</span>", unsafe_allow_html=True)
                st.write("**Go to the next slide (`Number plate analysis`) at the top: ⬆️**")
    if selected == "Number plate analysis":     
            st.title("Number plate recognition")
            st.markdown("""
    **Note:** Please select an image from the `ocr_test` folder or upload any car image. 
    """)
            uploaded_file_1 = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"],key="uploader2")\
            
            if uploaded_file_1 is not None:
                uploaded_file_path_2 = save_uploaded_file(uploaded_file_1, "uploaded_files_2")

                image = Image.open(uploaded_file_1)
                st.write("**UPLOADED IMAGE**")
                bordered_image = ImageOps.expand(image)
                st.image(bordered_image)
                print(f"Uploaded File Path: {uploaded_file_path_2}")
            
                bounders = YoloPredict(uploaded_file_path_2)
                if st.button(" click here for predict number plate "):
                    img = cv2.imread(uploaded_file_path_2)
                    res = []
                    for bounder in bounders:
                        for box in bounder.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            res.append(img[y1:y2, x1:x2])
                    st.title("Predicted Number plates")
                    if res:
                        for crop_img in res:
                            # print(get_text(crop_img))
                            st.image(crop_img)
                            for predictions in  get_text(crop_img):
                                st.write(f"##[Percentage: {predictions[2]:.3f}]")
                                detected_number_plate = predictions[1]

                                st.markdown(f"""
                                    <style>
                                    
                                    
                                    .number-plate-box {{
                                        border: 2px solid #4CAF50;
                                        padding: 10px;
                                        margin-top: 20px;
                                        width: fit-content;
                                        font-size: 20px;
                                        font-family: Arial, sans-serif;
                                        background-color: #f9f9f9;
                                        border-radius: 5px;
                                    }}
                                    .number-plate-box strong {{
                                        font-weight: bold;
                                    }}
                                    </style>
                                    """, unsafe_allow_html=True)

                                st.markdown(f"""
                                    <div class="number-plate-box">
                                        Detected Number Plate: <strong>{detected_number_plate}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)
                    else:
                        st.write("No License plate Detected!")
            
if __name__ == "__main__":
    main()
