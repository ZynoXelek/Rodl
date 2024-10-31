from ultralytics import YOLO
from ai_tool_functions import *
import time as t

def train():
    dataset_folder = 'dataset/FireExtinguisher/'
    
    train_models_folder = 'res/train_models/'
    new_model_name = 'new_model'
    
    reference_folder = 'res/train_models/reference_models/'
    reference_model = 'yolo11n.pt'
    
    yaml_file = 'data.yaml'
    
    ref_model_path = reference_folder + reference_model

    yaml_path = dataset_folder + yaml_file
    
    new_model_name = choose_new_name(train_models_folder, new_model_name)
    
    # Load a COCO-pretrained YOLO11n mode
    model = YOLO(ref_model_path)
    
    t1 = t.time()

    # Train the model on the dataset
    model.train(data = yaml_path, epochs = 5, imgsz = 640, project = train_models_folder, name = new_model_name)
    
    delta_t = t.time() - t1
    hour, min, sec = convert_seconds(int(delta_t))
    print(f'Training executed in {hour} hours, {min} mins and {sec} secs.')
        
if __name__ == '__main__':
	train()