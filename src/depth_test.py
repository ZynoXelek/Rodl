import os
import cv2 as cv
import numpy as np
import time

from glob import glob

from tool_functions import *
from template_advanced_matcher import *



# def test():
    
#     im = cv.imread("res/testing-ref.png")
#     im = (im / 255.0).astype(np.float32)
    
#     depth_test1 = normalize(cv.imread("res/depth_testing_ref(1).png", cv.IMREAD_GRAYSCALE)).astype(np.float32)
#     depth_test2 = normalize(cv.imread("res/depth_testing_ref(2).png", cv.IMREAD_GRAYSCALE)).astype(np.float32)
    
#     print(im.shape, im.dtype)
#     print(depth_test1.shape, depth_test1.dtype)
#     print(depth_test2.shape, depth_test2.dtype)
#     cv.imshow("Original image", im)
#     cv.imshow("Depth test 1", depth_test1)
#     cv.imshow("Depth test 2", depth_test2)
    
#     waitNextKey(0)
    
    
#     seg_im = kmeans(im, nKlusters=11)
#     seg_depth_im = kmeans(depth_test1, nKlusters=len(np.unique(depth_test1)))
    
    
    
    
#     print("Segmented image shape: ", seg_im.shape, " and type: ", seg_im.dtype)
#     print("Segmented depth image shape: ", seg_depth_im.shape, " and type: ", seg_depth_im.dtype)
    
#     cv.imshow("KMeans clustering", seg_im)
#     cv.imshow("KMeans clustering on depth", seg_depth_im)
    
    
    
#     # Count unique colors on segmented image
#     unique_colors, counts = np.unique(seg_im.reshape(-1, 3), axis=0, return_counts=True)
    
#     # sort both arrays by counts (descending)
#     sorted_indices = np.argsort(counts)[::-1]
#     unique_colors = unique_colors[sorted_indices]
#     counts = counts[sorted_indices]
    
#     print("Unique colors on segmented image: ", len(unique_colors))
#     print("Counts: ", counts)
    
    
    
    
#     waitNextKey(0)
#     cv.destroyAllWindows()






def main_kmeans():
    # Get all images
    initial_folder_path = "dataset/raw/test/"
    color_folder_path = os.path.join(initial_folder_path, "camera_color_image_raw/")
    
    
    image_path = os.path.join(color_folder_path, "*.png")
    file_paths = glob(image_path)
    
    # color movie
    color_movie = [cv.imread(file) for file in file_paths]              #* Use BGR images
    # color_movie = [cv.cvtColor(cv.imread(file), cv.COLOR_BGR2BGRA)    #* Use BGRA images
    #                for file in file_paths]
    
    
    # Base Template
    base_template = cv.imread("res/extinguisher-template.png", cv.IMREAD_UNCHANGED)
    base_template = crop(base_template)
    base_template = setTransparentPixelsTo(base_template,
                                        #    color=(255, 255, 255, 0),
                                           )
    factor = 1/7
    
    base_template = cv.resize(base_template, None, None, fx=factor, fy=factor)
    
    base_template = cv.cvtColor(base_template, cv.COLOR_BGRA2BGR)       #* Use BGR template
    
    cv.imshow("Base template", base_template)
    waitNextKey(0)
    
    
    # create advance matcher
    matcher = TemplateAdvancedMatcher(TemplateAdvancedMatcher.AI_MODE)
    
    range_fx = np.arange(0.2, 1.21, 0.2)
    range_fy = np.arange(0.4, 1.51, 0.2)
    range_theta = np.arange(-30, 31, 10)
    
    for im in color_movie:
        
        t = time.time()
        final_im = matcher.fullMatch(im,
                                     base_template,
                                     range_fx=range_fx,
                                     range_fy=range_fy,
                                     range_theta=range_theta,
                                     show_progress = True)
        print("Time elapsed to compute the final image: ", time.time() - t)
        
        
        cv.imshow("Color movie", im)
        # cv.imshow("Object image", objects_im)
        cv.imshow("Final image", final_im)
        waitNextKey(1)
    
    print("Treatment finished!")
    waitNextKey(0)
    cv.destroyAllWindows()



# def main_superpixels(): #! Not working well, extinguishers are noise
#     # Get all images
#     initial_folder_path = "dataset/raw/test/"
#     color_folder_path = os.path.join(initial_folder_path, "camera_color_image_raw/")
    
    
#     image_path = os.path.join(color_folder_path, "*.png")
#     file_paths = glob(image_path)
    
#     # color movie
#     color_movie = [cv.imread(file) for file in file_paths]
    
#     for im in color_movie:
        
#         remove_background_superpixels(im)
        
        
        
        
        
#         cv.imshow("Color movie", im)
#         waitNextKey(0)
    
#     print("Treatment finished!")
#     waitNextKey(0)
#     cv.destroyAllWindows()
    



def main():
    """Main function of the program."""
    main_kmeans()



if __name__ == "__main__":
	main()
