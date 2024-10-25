import cv2 as cv
import numpy as np
from ultralytics import YOLO
import os
from tool_functions import *

#TODO know if we keep this function
def match_color_depth_img(color_folder: str,
                          depth_folder: str,
                          show_result = False) -> list[int]:
    """
    Matches the color images with the depth images regarding to the date of the photo.
    
    Parameters
    ----------
    color_folder: str
        name of the folder containing the color images
    depth_folder: str
        name of the folder containing the depth images
    show_result: bool
        determines if the result should be showed after computation
    
    Returns
    -------
    depth_foreach_color: np.ndarray
        list of indices: i-th color image will be associated to depth_foreach_color[i]-th depth images of their respective folder
    depth_foreach_color_strict: np.ndarray
        same as depth_foreach_color, but only closest to each depth image is kept (other indices are None)
    """
    image_folder = color_folder
    depth_image_folder = depth_folder

    img_path_list = os.listdir(image_folder)
    depth_img_path_list = os.listdir(depth_image_folder)

    depth_foreach_color = []
    depth_foreach_color_strict = [None] * len(img_path_list)

    min_j = -1
    for i, depth_img_path in enumerate(depth_img_path_list):
        depth_img_num = int(depth_img_path.replace('.png', '').split('_')[-1])
        min_diff = np.inf
        for j, img_path in enumerate(img_path_list):
            img_num = int(img_path.replace('.png', '').split('_')[-1])
            diff = np.abs(img_num - depth_img_num)
            if diff < min_diff:
                min_j = j
                min_diff = diff
            else:
                break
        depth_foreach_color_strict[min_j] = i

    min_j = -1
    for i, img_path in enumerate(img_path_list):
        img_num = int(img_path.replace('.png', '').split('_')[-1])
        min_diff = np.inf
        for j, depth_img_path in enumerate(depth_img_path_list):
            depth_img_num = int(depth_img_path.replace('.png', '').split('_')[-1])
            diff = np.abs(img_num - depth_img_num)
            if diff < min_diff:
                min_j = j
                min_diff = diff
            else:
                break
        depth_foreach_color.append(min_j)

    if show_result:
        print('Brute force match depth to color images')
        for i, corresponding_ind in enumerate(depth_foreach_color):
            img_path = image_folder + img_path_list[i]
            depth_img_path = depth_image_folder + depth_img_path_list[corresponding_ind]
            
            img = cv.imread(img_path)
            depth_img = cv.imread(depth_img_path)
            
            cv.imshow('color', img)
            cv.imshow('depth', depth_img)
            
            waitNextKey()

        cv.destroyAllWindows()

        print('Best match depth to color images')
        for i, corresponding_ind in enumerate(depth_foreach_color):
            if corresponding_ind is not None:
                img_path = image_folder + img_path_list[i]
                depth_img_path = depth_image_folder + depth_img_path_list[corresponding_ind]
                
                img = cv.imread(img_path)
                depth_img = cv.imread(depth_img_path)
                
                cv.imshow('color', img)
                cv.imshow('depth', depth_img)
                
                waitNextKey()

        cv.destroyAllWindows()
    
    return depth_foreach_color, depth_foreach_color_strict

def compute_yolo_boxes(yolo_model: YOLO,
                   img: np.ndarray,
                   target_classes_id: list[int],
                   target_classes_names: list[str],
                   target_classes_colors: list[tuple] | None = None,
                   show_result: bool = False) -> list[dict]:
    """
    Compute the detected boxes of the targeted classes using YOLO model.
    
    Parameters
    ----------
    yolo_model: YOLO
        YOLO model that will be used for the object detection
    img: np.ndarray
        image on which we want to detect objects
    target_classes_id: list[int]
        list of indices representing the id of the classes that should be kept during the detection
    target_classes_names: list[str]
        list of the corresponding names of the targeted classes
    target_classes_colors: list[tuple]
        list of the corresponding colors of the targeted classes
    show_result: bool
        determines if the result should be plotted
    Returns
    -------
    boxes_info_list: list[dict]
        list of dictionnarires with useful informations about the boxes:
        
        for each dictionnary box_info, we have:
        - box_info['class_id'] contains the id of the class of the detected object
        - box_info['class_name'] contains the name of the class of the detected object
        - box_info['score'] contains the confiance score
        - box_info['coord'] contains the coordinates as the following list: [x1, y1]
        - box_info['size'] contains the size of the box as the following tuple: w_box, h_box
        - box_info['class_color'] contains the color associated to the class of the detected object
    """ 
    BLUE = (255, 0, 0)
    
    if target_classes_colors is None:
        target_classes_colors = [BLUE]*len(target_classes_id)

    results = yolo_model(img)
    
    detections = results[0]
    boxes = detections.boxes
    scores = boxes.conf
    classes = boxes.cls
    
    box_info_list = []
    im = img.copy()
    h, w = im.shape[:2]
    for i, box in enumerate(boxes):
        class_id = classes[i].item()
        if int(class_id) in target_classes_id:
            class_index = target_classes_id.index(int(class_id))
            
            box_info = {}
            
            score = scores[i].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(max(x1, 0))
            y1 = int(max(y1, 0))
            x2 = int(min(x2, w-1))
            y2 = int(min(y2, h-1))
            
            box_info['class_id'] = class_id
            box_info['class_name'] = target_classes_names[class_index]
            box_info['class_color'] = target_classes_colors[class_index]
            box_info['score'] = score
            box_info['coord'] = [x1, y1]
            box_info['size'] = x2-x1, y2-y1

        box_info_list.append(box_info)
            
    return box_info_list

def show_result_yolo_boxes(img: np.ndarray,
                           boxes_info_list: list[dict]) -> None:
    """
    Show the boxes info regarding the boxes_info_list and the original image.
    
    Parameters
    ----------
    img: np.ndarray
        original image
    boxes_info_list: list[dict]
        list of boxes info associated to the image
    """
    WHITE = (255, 255, 255)
    TEXT_COLOR = WHITE
    RECTANGLE_THICKNESS = 2
    FONT = cv.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.6
    TEXT_THICKNESS = 1
    
    im = img.copy()
    
    n_boxes = len(boxes_info_list)
    for i, box_info in enumerate(boxes_info_list):
        score = box_info['score']
        x1, y1 = box_info['coord']
        w_box, h_box = box_info['size']
        
        class_name = box_info['class_name']
        
        print(f'\tbox {i+1}/{n_boxes}:')
        print(f'\t\tdetected object: {class_name}')
        print(f'\t\tscore: {score}')
        print(f'\t\tupper left corner: x1: {x1}, y1: {y1}')
        print(f'\t\tsize: w: {w_box}, h: {h_box}')        
        
        class_color = box_info['class_color']

        cv.rectangle(im, (x1, y1), (x1 + w_box, y1 + h_box), class_color, RECTANGLE_THICKNESS)
        label = class_name + ': ' + str(round(score, 4))
        (text_width, text_height), baseline = cv.getTextSize(label, FONT, TEXT_SCALE, TEXT_THICKNESS)
        cv.rectangle(im, (x1, y1 - text_height), (x1 + text_width, y1 + baseline), class_color, cv.FILLED)
        cv.putText(im, label, (x1, y1), FONT, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
    cv.imshow('object detection', im)
    waitNextKey()
    
def choose_new_name(train_model_folder: str, name: str, default_name: str = 'training_model') -> str:
    """
    Choosing a new name for the model.
    
    Parameters
    ----------
    train_model_folder: str
        folder containing all the models
    name: str
        desired name from the user
    default_name: str
        default_name for a model
    """
    
    model_folder_list = os.listdir(train_model_folder)
    if name in model_folder_list:
        i = 1
        new_name = default_name + str(i)
        while new_name in model_folder_list:
            i+=1
            new_name = default_name + str(i)
    else:
        new_name = name
    return new_name