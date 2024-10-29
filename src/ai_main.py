from ultralytics import YOLO
import os
import cv2 as cv
from ai_tool_functions import *

def main():
    model_folder = 'res/train_models/training_model1/'
    model_file = "weights/best.pt"
    
    model_path = model_folder + model_file
    
    img_folder = "dataset/raw/test/camera_color_image_raw/"
    img_path_list = os.listdir(img_folder)
    
    img_list = []
    for i in range(len(os.listdir(img_folder))):
        img_path = img_folder + img_path_list[i]
        img = cv.imread(img_path)
        img_list.append(img)
    
    model = YOLO(model_path)
    
    n_img = len(img_list)
    for i, img in enumerate(img_list):
        boxes_info_list = compute_yolo_boxes(model,
                                        img,
                                        target_classes_id = [0],
                                        target_classes_names = ['extinguisher'],
                                        )
        n_boxes = len(boxes_info_list)
        print(f'image {i+1}/{n_img}, {n_boxes} boxes:')
        show_result_yolo_boxes(img, boxes_info_list)
        print('') 

if __name__ == '__main__':
	main()