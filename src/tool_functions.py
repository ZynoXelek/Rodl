import cv2 as cv
import numpy as np

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
    
    cropped_im = im[min_row_index:max_row_index, min_col_index:max_col_index]
    return cropped_im



def custom_similarity(image: np.ndarray,
                      template: np.ndarray,
                      display_advancement: bool = False,
                      show: bool = False,
                      show_threshold: float = 1.01
) -> np.ndarray:
    """Compute the squared difference map between the two images and show each map.

    Args:
        image (np.ndarray): original image (HxW)
        template (np.ndarray): template image (h x w)
        display_advancement (bool, optional): whether to show the progress. Defaults to True.
        show (bool, optional): whether to show the difference map. Defaults to False.
        show_threshold (float, optional): relative threshold divided by the minimum value found
                                          under which the corresponding value is shown. Defaults to 1.2.

    Returns:
        np.ndarray: squared difference map
    
    Raises:
		ValueError: if input image is not a color image
		ValueError: if input image and template have different number of channels
		ValueError: if template image is larger than the original image
    """
    
    if len(image.shape) < 2:
        raise ValueError("Input image must be a color image")
    
    if len(image.shape) != len(template.shape):
        raise ValueError("Input image and template must have the same number of channels")
    
    im_size = image.shape[:2]#[::-1]
    t_size = template.shape[:2]#[::-1]
    
    if t_size[0] > im_size[0] or t_size[1] > im_size[1]:
        raise ValueError("Template image must be smaller than the original image")
    
    
    # Compute squared difference map for each possible position
    percentage = 0
    last_percentage = None
    
    row_iter = im_size[0] - t_size[0] + 1
    col_iter = im_size[1] - t_size[1] + 1
    nb_iter = row_iter * col_iter
    
    result = np.zeros((row_iter, col_iter), dtype=np.float32)
    
    if display_advancement:
        print("Processing the difference map...")
    
    for i in range(row_iter):
        for j in range(col_iter):
            cropped_image = image[i:i+t_size[0], j:j+t_size[1]]
            diff = cropped_image - template
            result[i, j] = np.sum(np.square(diff))

            if display_advancement:
                percentage = round((i * col_iter + j) / nb_iter * 100)
                if percentage != last_percentage:
                    last_percentage = percentage
                    print(f"\tAdvancement: {percentage}%", end='\r')
    
    
    # Test if the result looks correct
    open_cv_result = cv.matchTemplate(image, template, cv.TM_SQDIFF)
    if not np.allclose(result, open_cv_result, atol=1e-10):
        print("Results do not match!")
    else:
        print("Custom result match with the one from opencv!")
    
    
    
    if display_advancement:
        print(f"\tAdvancement: 100%")
    
    
    
    if show:
        min_value = np.min(result)
        max_value = np.max(result)
        min_pos = np.unravel_index(np.argmin(result), result.shape)
        print("min pos =", min_pos, " correct? ", min_value == result[min_pos])
        indices = [min_pos, (164, 243)]
        # indices = np.argwhere(result <= min_value * show_threshold)
        
        nb = len(indices)
        print("Number of pixels respecting the threshold:", nb)
        
        if nb > 0:
            cv.imshow('Template', template)
        
        for i in range(nb):
            x, y = indices[i] # in numpy style, so x is the row and y is the column
            print(f"Pixel {i+1}/{nb} located at ({x}, {y})")
            
            cropped_image = image[x:x+t_size[0], y:y+t_size[1]]
            diff = cropped_image - template
            norm_diff = np.sum(np.square(diff), axis=-1)
            norm_diff = diff/max_value
            
            s = result[x, y]
            print(s == np.sum(norm_diff))
            
            print("Resulting value:", s, " which makes a ratio of ", s/min_value, " compared to the minimum value")
            
            cv.imshow('Normalized Difference', norm_diff/np.max(norm_diff))
            
            
            im_copy = image.copy()
            # On rectangle, width and height are inverted so x, y becomes y, x
            cv.rectangle(im_copy, (y, x), (y+t_size[1], x+t_size[0]), (0, 255, 0), 2)
            cv.imshow("Location in the image", im_copy)
            
            k = cv.waitKey(0)
            if k == 27:
                break
    
    cv.destroyAllWindows()
    
    return result












if __name__ == "__main__":
    # Test the function
    image = np.random.rand(100, 100, 4).astype(np.float32)
    template = np.random.rand(10, 10, 4).astype(np.float32)

    custom_result = custom_similarity(image, template, display_advancement=False, show=False)
    cv_result = cv.matchTemplate(image, template, cv.TM_SQDIFF)

    # Normalize the results for comparison
    custom_result_norm = (custom_result - np.min(custom_result)) / (np.max(custom_result) - np.min(custom_result))
    cv_result_norm = (cv_result - np.min(cv_result)) / (np.max(cv_result) - np.min(cv_result))

    assert np.allclose(custom_result_norm, cv_result_norm, atol=1e-5), "Results do not match!"
    print("Results match!")


