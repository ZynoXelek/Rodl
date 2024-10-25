from ultralytics import YOLO
from ai_tool_functions import *

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

    # Train the model on the dataset
    model.train(data = yaml_path, epochs = 10, imgsz = 640, project = train_models_folder, name = new_model_name)

if __name__ == '__main__':
	train()