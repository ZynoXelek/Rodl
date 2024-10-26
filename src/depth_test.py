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
    color_movie = [cv.cvtColor(cv.imread(file), cv.COLOR_BGR2BGRA)
                   for file in file_paths]
    
    
    # Base Template
    base_template = cv.imread("res/extinguisher-template.png", cv.IMREAD_UNCHANGED)
    base_template = crop(base_template)
    base_template = setTransparentPixelsTo(base_template,
                                        #    color=(255, 255, 255, 0),
                                           )
    factor = 1/6
    
    # # Temporary test
    # th, tw = base_template.shape[:2]
    # for t in np.arange(-180, 180, 1):
    #     expected_size = computeRotatedRequiredSize((tw, th), t)
    #     actual_size = rotateImageWithoutLoss(base_template, t).shape[:2][::-1]
        
    #     print("Angle:", t,
    #           " Expected size: ", expected_size,
    #           " Actual size: ", actual_size,
    #           " difference: ", np.array(expected_size) - np.array(actual_size))
    
    # cv.imshow("Base template, rotated (1)", rotateImage(base_template, -46))
    # cv.imshow("Base template, rotated (2)", rotateImageWithoutLoss(base_template, -116, (255, 255, 255, 0)))
    base_template = cv.resize(base_template, None, None, fx=factor, fy=factor)
    th, tw = base_template.shape[:2]
    
    cv.imshow("Base template", base_template)
    waitNextKey(0)
    
    
    # box_factor = 0.5
    # box_limits = (int(tw * box_factor), int(th * box_factor))
    # print("Box limits that will be used: ", box_limits)
    
    # create advance matcher
    matcher = TemplateAdvancedMatcher(TemplateAdvancedMatcher.CLASSIC_MODE)
    
    for im in color_movie:
        
        # objects_im, contours = remove_background_kmeans(im,
        #                                                 # replacing_color=(255, 255, 255),
        #                                                 exclude_single_pixels=True,
        #                                                 #  box_excluding_size=box_limits
        #                                                 )
        
        
        # for contour in contours:
        #     x, y, w, h = cv.boundingRect(contour)
        #     cv.rectangle(objects_im, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        
        t = time.time()
        final_im = matcher.fullMatch(im, base_template, show_progress = True)
        print("Time elapsed to compute the final image: ", time.time() - t)
        
        
        cv.imshow("Color movie", im)
        # cv.imshow("Object image", objects_im)
        cv.imshow("Final image", final_im)
        waitNextKey(1)
    
    print("Treatment finished!")
    waitNextKey(0)
    cv.destroyAllWindows()



def main_superpixels():
    # Get all images
    initial_folder_path = "dataset/raw/test/"
    color_folder_path = os.path.join(initial_folder_path, "camera_color_image_raw/")
    
    
    image_path = os.path.join(color_folder_path, "*.png")
    file_paths = glob(image_path)
    
    # color movie
    color_movie = [cv.imread(file) for file in file_paths]
    
    for im in color_movie:
        
        remove_background_superpixels(im)
        
        
        
        
        
        cv.imshow("Color movie", im)
        waitNextKey(0)
    
    print("Treatment finished!")
    waitNextKey(0)
    cv.destroyAllWindows()
    



def main():
    """Main function of the program."""
    main_kmeans()
    # main_superpixels() #! Not working well, extinguishers are noise



if __name__ == "__main__":
	main()
