# Rodl
Project about Real-time Object Detection and Localization, for the Computer Vision and Image Processing course at LTU.

## Folder organization

The Python code of the project is located in the `src/` folder.
The resources used in the process, which are mostly templates, are located in the `res/` folder, along with the deep learning IA models.
The dataset images to test the process on should be located in the `dataset/` folder.

### Dataset

Images used in this project are not uploaded to the github repository.
You should first download them [here](https://ltuse-my.sharepoint.com/:f:/g/personal/nikolaos_stathoulopoulos_ltu_se/EotXFpRj5GJGoW5qUDHrcigB46BdZ-9OX-i4M0yRLtvonQ?e=dogFnr) in the `Project_1` folder.

You should then place the `Project_1` folder's content in the `dataset/` folder of the project, such that it contains the `raw` and `rosbags` folders.

### Deep Learning dataset

No training dataset is provided in the project. We used the dataset which can be found [here](https://universe.roboflow.com/fire-extinguisher/fireextinguisher-z5atr) to train our models.
The trained models are located in the `res/train_models` folder.

## Running the code

To run the program, you simply have to run the `res/main.py` Python script.

If you wish to modify parameters, please do so in this `main.py` file.
Parameters that can easily be modified are part of the `#? XXXXX (can be modified) -------------` blocks of code.

Before going to the main process loop, the program first shows the base template that will be used for the matching step of the process.
Press any key to continue to the main processing loop, or `[Escape]` to cancel the process.

During the processing, press `[Space]` to switch from the image per image mode to the continuous mode.
- During the image per image mode, press any key to go to the next image.
- During the continuous mode, the processing keeps going as long as it can
During any of these mods, press `[Escape]` to stop the program early and quit.
