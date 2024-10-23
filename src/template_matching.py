import cv2 as cv
import numpy as np

#? Tool functions

def setTransparentPixelsTo(im: np.ndarray, color: tuple = (0, 0, 0, 0)) -> np.ndarray:
    """
    Set transparent pixels `[b, g, r, 0]` to the given color.
    The input image is not modified, a copy is made and returned.
    
    Parameters
    ---------
    im: np.ndarray
        image with transparency (4 channels)
    color: tuple
        color to set transparent pixels to (default: black)
    
    Returns
    -------
    im: np.ndarray
        image with transparent pixels set to black
    
    Raises:
        ValueError: if input image does not have 4 channels (RGBA)
    """
    im = im.copy()
    im_shape = im.shape
    
    if len(im_shape) != 3 or im_shape[2] != 4:
        raise ValueError("Input image must have 4 channels (RGBA)")
    
    # Make alpha 0 pixels black
    shape = im.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if im[i, j, 3] == 0:
                im[i, j] = color
    
    return im

def crop(im: np.ndarray) -> np.ndarray:
    """
    Crops the image so that we only keep the important part.

    Parameters
    ----------
    im: np.ndarray
        Original image to be cropped.
        
    Returns
    -------
    cropped_im: np.ndarray
        Cropped image.
    """
    shape = im.shape
    
    # Row indexes
    min_row_index = None
    max_row_index = None
    
    # Column indexes
    min_col_index = None
    max_col_index = None
    
    # Finding the min and max rows
    for i in range(shape[0]):
        if min_row_index is None:
            row = im[i]
            if np.any(row[:, 3] != 0):
                min_row_index = i
        
        if max_row_index is None:
            row = im[-1-i]
            if np.any(row[:, 3] != 0):
                max_row_index = shape[0] - i
        
        if min_row_index is not None and max_row_index is not None:
            break

    for j in range(shape[1]):
        if min_col_index is None:
            col_left = im[:, j]
            if np.any(col_left[:, 3] != 0):
                min_col_index = j
        
        if max_col_index is None:
            col_right = im[:, -1-j]
            if np.any(col_right[:, 3] != 0):
                max_col_index = shape[1] - j
        
        if min_col_index is not None and max_col_index is not None:
            break
    
    # If something went wrong
    if min_row_index is None:
        min_row_index = 0
    if max_row_index is None:
        max_row_index = shape[0]
    if min_col_index is None:
        min_col_index = 0
    if max_col_index is None:
        max_col_index = shape[1]
    
    print(shape)
    print("Cropping indexes: ", min_row_index, max_row_index, min_col_index, max_col_index)
    
    cropped_im = im[min_row_index:max_row_index, min_col_index:max_col_index]
    return cropped_im

#? Main code

#* Read clean template with transparency
template_im = cv.imread("res/extinguisher-template.png", cv.IMREAD_UNCHANGED)
img = cv.imread("dataset/raw/test/camera_color_image_raw/camera_color_image_1727164479163392418.png")
h, w = img.shape[:2]
img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)

#* Get a clean template
template_im = setTransparentPixelsTo(template_im)
template_im = crop(template_im)

# template_im = cv.cvtColor(template_im, cv.COLOR_BGRA2BGR)

RED = (0, 0, 255, 255)
testing_red_diff = np.zeros((h, w))
for i in range(h):
    for j in range(w):
        testing_red_diff[i, j] = np.linalg.norm(img[i, j, :] - RED)

print("Check transparency on first pixel (template): ", template_im[0, 0])
print("Check transparency on first pixel (image): ", img[0, 0])

#* Write template
cv.imwrite("res/clean-extinguisher-template.png", template_im)


print("Original image: ")
print(img.shape)
print(img.dtype)

print("Template: ")
print(template_im.shape)
print(template_im.dtype)


factor = 1/6
template_im = cv.resize(template_im, None, None, fx=factor, fy=factor)

ht, wt = template_im.shape[:2]

cv.imshow("Original image", img)
cv.imshow("Template resized", template_im)
cv.imshow("Testing red diff (normalized)", (1 - testing_red_diff/np.max(testing_red_diff)))
cv.waitKey(0)

#? Actual template matching

#* Template modes
MIN_CASE = 0, 2
MAX_CASE = 1, 3

M1 = cv.TM_SQDIFF, MIN_CASE
M1N = cv.TM_SQDIFF_NORMED, MIN_CASE
M2 = cv.TM_CCORR, MAX_CASE
M2N = cv.TM_CCORR_NORMED, MAX_CASE
M3 = cv.TM_CCOEFF, MAX_CASE
M3N = cv.TM_CCOEFF_NORMED, MAX_CASE

mode = M1
similarity = cv.matchTemplate(img, template_im, mode[0])

#* Normalize the similarity map to display it
similarity = similarity/np.max(similarity)
print("Normalized Similarity data: min, max, mean, median (dtype:", similarity.dtype, ")")
print(np.min(similarity), np.max(similarity), np.mean(similarity), np.median(similarity))


cv.imshow('similarity', similarity)

#* minVal, maxVal, minLoc, maxLoc
sim_extrema = cv.minMaxLoc(similarity)

value = sim_extrema[mode[1][0]]
location = sim_extrema[mode[1][1]]

result_im = img.copy()
cv.rectangle(result_im, location, (location[0] + wt, location[1] + ht), (0, 255, 0), 3)

cv.imshow("Result of template matching", result_im)
cv.waitKey(0)
cv.destroyAllWindows()
