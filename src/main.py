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
    
    depth_foreach, depth_foreach_strict = match_color_depth_img(color_folder_path, depth_folder_path, color_info_folder_path, depth_info_folder_path)
    
    
    # color movie
    # color_movie = [cv.imread(file) for file in file_paths]              #* Use BGR images
    color_movie = [cv.cvtColor(cv.imread(file), cv.COLOR_BGR2BGRA)    #* Use BGRA images
                   for file in file_paths]
    
    
    # Base Template
    base_template = cv.imread("res/extinguisher-tube-template.png", cv.IMREAD_UNCHANGED)   #"res/extinguisher-tube-template.png""
    base_template = crop(base_template)
    base_template = setTransparentPixelsTo(base_template,
                                        #    color=(255, 255, 255, 0),
                                           )
    factor = 1/7
    
    base_template = cv.resize(base_template, None, None, fx=factor, fy=factor)
    
    # base_template = cv.cvtColor(base_template, cv.COLOR_BGRA2BGR)       #* Use BGR template
    
    cv.imshow("Base template", base_template)
    waitNextKey(0)
    
    
    # create advance matcher
    matcher_mode = TemplateAdvancedMatcher.AI_MODE  # TemplateAdvancedMatcher.AI_MODE or TemplateAdvancedMatcher.CLASSIC_MODE
    reliability_mode = TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE # TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE or TemplateAdvancedMatcher.RELIABILITY_DEPTH_MODE
    matcher = TemplateAdvancedMatcher(matcher_mode, reliability_mode)
    
    matching_mode = MatchingMode.SQDIFF
    range_fx = np.arange(0.2, 1.21, 0.2)
    range_fy = np.arange(0.4, 1.51, 0.2)
    range_theta = np.arange(-10, 11, 10)
    
    # custom_matching_method = custom_matching        # or None
    # custom_case = Case.MIN                          # or Case.MAX
    
    base_waiting_time = 1
    waiting_time = base_waiting_time
    
    # for im in color_movie[30:40]:
    for i in range(len(color_movie)):
        im = color_movie[i]
        
        depth_im = None
        projection_matrix = None
        
        #* Strict depth matching version
        if i+1 < len(depth_foreach_strict) and depth_foreach_strict[i+1] is not None:
            depth_index = depth_foreach_strict[i+1]
            depth_im = cv.imread(depth_file_paths[depth_index], cv.IMREAD_GRAYSCALE)
            projection_matrix = readProjectionMatrixInInfoFile(depth_info_file_paths[depth_index])
            # cv.imshow("Depth image", normalize(depth_im))
        
        
        # #* Normal depth matching version (will always map a depth image to a color image)
        # if i+1 < len(depth_foreach) and depth_foreach[i+1] is not None:
        #     depth_index = depth_foreach[i+1]
        #     depth_im = cv.imread(depth_file_paths[depth_index], cv.IMREAD_GRAYSCALE)
        #     projection_matrix = readProjectionMatrixInInfoFile(depth_info_file_paths[depth_index])
        #     # cv.imshow("Depth image", normalize(depth_im))
        
        
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
                              # custom_matching_method=custom_matching_method,
                              # custom_case=custom_case,
                              show_progress = True)
        print("Time elapsed to compute the final image: ", time.time() - t)
        
        #TODO:
        #
        # Add new results to the list of matches
        # - If a depth map is associated to this image:
        #           Use it to compute the distance to the camera, and therefore the position with the Projection matrix written in the camera calibration file
        #           To do so, take the green box and compute the distance to the center of the box
        # - Else if not, either:
        #           ignore this image for depth computation (+)
        #           use the previous depth map to do the same computation
        #           try to match to the previous detected objects and use the same position
        #
        #
        # Apply a score to the matches
        # - If the score is too low, ignore the match (classic case, use similarity not normalized?)
        # We can then normalize each similarity map if we want to.
        # - If the score is high enough, it is a potential extinguisher
        # - Among potential extinguishers, try to compute delta distance to the sides to detect fake extinguishers?
        #
        
        
        cv.imshow("Color movie", im)
        # cv.imshow("Object image", objects_im)
        cv.imshow("Final image", final_im)
        k = waitNextKey(waiting_time)
        
        # Switch between 0 and the base waiting time if space is pressed
        if k == 32:
            waiting_time = base_waiting_time if waiting_time == 0 else 0
        # cv.destroyAllWindows()
    
    print("Treatment finished!")
    waitNextKey(0)
    cv.destroyAllWindows()



if __name__ == '__main__':
	main()
