import os
import cv2 as cv
import numpy as np
import time

from glob import glob

from tool_functions import *
from template_advanced_matcher import *



def readProjectionMatrixInInfoFile(info_file_path: str) -> np.ndarray:
    """Read the projection matrix in the camera info file."""
    lines = []
    with open(info_file_path, "r") as f:
        lines = f.readlines()
    
    matrix_string = ""
    for line in lines:
        if "P:" in line:
            matrix_string = line.replace("P:", "").strip()
            break
    
    return loadMatrixFromText(matrix_string, shape=(3, 4))




def main():
    # Get all images
    
    #? File paths (can be modified) ----------------------------------------------------------------
    initial_folder_path = "dataset/raw/test/"
    color_folder_path = os.path.join(initial_folder_path, "camera_color_image_raw/")
    depth_folder_path = os.path.join(initial_folder_path, "camera_depth_image_raw/")
    color_info_folder_path = os.path.join(initial_folder_path, "camera_color_camera_info/")
    depth_info_folder_path = os.path.join(initial_folder_path, "camera_depth_camera_info/")

    image_path = os.path.join(color_folder_path, "*.png")
    file_paths = glob(image_path)
    depth_image_path = os.path.join(depth_folder_path, "*.png")
    depth_info_path = os.path.join(depth_info_folder_path, "*.txt")
    depth_file_paths = glob(depth_image_path)
    depth_info_file_paths = glob(depth_info_path)
    #? -----------------------------------------------------------------------------------------------
    
    depth_foreach, depth_foreach_strict = match_color_depth_img(color_folder_path, depth_folder_path, color_info_folder_path, depth_info_folder_path)
    
    
    # color movie
    color_movie = [cv.cvtColor(cv.imread(file), cv.COLOR_BGR2BGRA)    #* Use BGRA images since the template is also RGBA
                   for file in file_paths]
    
    
    #? Base Template: Size reduced and process the alpha channel (can be modified) -------------------
    base_template = cv.imread("res/extinguisher-body-template.png", cv.IMREAD_UNCHANGED)
    base_template = crop(base_template)
    base_template = setTransparentPixelsTo(base_template,
                                        #    color=(255, 255, 255, 0),
                                           )
    factor = 1/7
    base_template = cv.resize(base_template, None, None, fx=factor, fy=factor)
    #? -----------------------------------------------------------------------------------------------
    
    cv.imshow("Base template", base_template)
    waitNextKey(0)
    
    #? Matcher modes (can be modified) --------------------------------------
    
    # Create advance matcher
    matcher_mode = TemplateAdvancedMatcher.AI_MODE  # TemplateAdvancedMatcher.AI_MODE or TemplateAdvancedMatcher.CLASSIC_MODE
    reliability_mode = TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE # TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE or TemplateAdvancedMatcher.RELIABILITY_DEPTH_MODE
    matcher = TemplateAdvancedMatcher(matcher_mode, reliability_mode)
    
    
    #? Matching parameters (can be modified) --------------------------------
    matching_mode = MatchingMode.SQDIFF
    range_fx = np.arange(0.2, 1.21, 0.2)
    range_fy = np.arange(0.4, 1.51, 0.2)
    range_theta = np.arange(-10, 11, 10)
    
    custom_matching_method = None               # None, or custom_matching if you want to use a custom matching method
    custom_case = None                          # None or the corresponding Case.MIN, Case.MAX if you are using a custom matching method
    #? ----------------------------------------------------------------------
    
    
    #? Pre-treatment parameters (can be modified) --------------------------------
    
    #* In case of an AI mode
    if matcher_mode == TemplateAdvancedMatcher.AI_MODE:
        # None values means it will use the default values
        matcher.setAIModeVariables(model_path="res/train_models/training_model2/weights/best.pt",
                                target_classes_id=None,
                                target_classes_names=None,)
    #* In case of a classic mode
    elif matcher_mode == TemplateAdvancedMatcher.CLASSIC_MODE:
        # None values means it will use the default values
        matcher.setClassicModeVariables(nb_attempts=None,
                                        nb_clusters=None,
                                        nb_background_colors=None,
                                        exclude_single_pixels=None,
                                        erosion_size=None,
                                        dilation_size=None,
                                        box_excluding_size=None,
                                        classic_display=False,)
    #? Global parameters (can be modified) ---------------------------------
    
    TemplateAdvancedMatcher.DISPLAY_DEPTH_MATCHING_STEPS = False #True
    
    #? Scoring parameters (can be modified) --------------------------------
    
    #* In case of a background scoring mode
    if reliability_mode == TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE:
        # None values means it will use the default values
        matcher.setScoringBackgroundVariables(template_size_factor=None,
                                              reliability_high_threshold=None,
                                              reliability_low_threshold=None,)
    #* In case of a depth scoring mode
    elif reliability_mode == TemplateAdvancedMatcher.RELIABILITY_DEPTH_MODE:
        # None values means it will use the default values
        matcher.setScoringDepthVariables(outbox_factor=None,
                                         difference_threshold=None,
                                         reliability_threshold=None,
                                         display_reliability_points=False,)
    #? ----------------------------------------------------------------------
    
    
    #* Start the treatment
    base_waiting_time = 1   # Continuous mode
    waiting_time = 0        # Starts on image per image mode
    
    for i in range(len(color_movie)):
        im = color_movie[i]
        
        depth_im = None
        projection_matrix = None
        
        #? Depth image to color image matching (can be modified) -------------------------------
        # #* Strict depth matching version
        # if i+1 < len(depth_foreach_strict) and depth_foreach_strict[i+1] is not None:
        #     depth_index = depth_foreach_strict[i+1]
        #     depth_im = cv.imread(depth_file_paths[depth_index], cv.IMREAD_GRAYSCALE)
        #     projection_matrix = readProjectionMatrixInInfoFile(depth_info_file_paths[depth_index])
        
        #* Normal depth matching version (will always map a depth image to a color image)
        if i+1 < len(depth_foreach) and depth_foreach[i+1] is not None:
            depth_index = depth_foreach[i+1]
            depth_im = cv.imread(depth_file_paths[depth_index], cv.IMREAD_GRAYSCALE)
            projection_matrix = readProjectionMatrixInInfoFile(depth_info_file_paths[depth_index])
        #? -------------------------------------------------------------------------------------
        
        print(" --------------------------------------------------------------- ")
        print("image: ", i, " - depth_image found? ", depth_im is not None)
        print(projection_matrix)
        
        t = time.time()
        final_im, valid_matches, similarity_maps, similarity_stats =\
            matcher.fullMatch(im,
                              base_template,
                              matching_mode=matching_mode,
                              range_fx=range_fx,
                              range_fy=range_fy,
                              range_theta=range_theta,
                              depth_image=depth_im,
                              projection_matrix=projection_matrix,
                              display_depth_matching=True,
                              custom_matching_method=custom_matching_method,
                              custom_case=custom_case,
                              show_progress = True,)
        print("Time elapsed to compute the final image: ", time.time() - t)
        
        cv.imshow("Original image", im)
        cv.imshow("Final image", final_im)
        k = waitNextKey(waiting_time)
        
        # Switch between 0 and the base waiting time if space is pressed
        if k == 32:
            waiting_time = base_waiting_time if waiting_time == 0 else 0
    
    print("Treatment finished!")
    waitNextKey(0)
    cv.destroyAllWindows()



if __name__ == '__main__':
	main()
