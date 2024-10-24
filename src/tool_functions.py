import cv2 as cv
import numpy as np


def waitNextKey(delay: int = 0) -> None:
    """
    Waits for next opencv key for the given delay.
    Quits if Esc or Q is pressed.
    
    Parameters
    ----------
    delay: int
        delay in milliseconds (default: 0)
    """
    k = cv.waitKey(delay)
    if k == 27 or k == ord('q'):
        cv.destroyAllWindows()
        exit(0)



def normalize(im: np.ndarray) -> np.ndarray:
    """
    Normalize the image to have values between 0 and 1.
    
    Parameters
    ----------
    im: np.ndarray
        image to normalize
    
    Returns
    -------
    normalized_im: np.ndarray
        normalized image
    """
    max_value = np.max(im)
    min_value = np.min(im)
    return (im - min_value) / (max_value - min_value)



def diffTransparency(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    Compute the difference between two images with transparency.
    The difference is computed on the RGB channels only, and then multiplied by (1 - diff_alpha).
    The result is that difference with transparent pixels are not taken into account.
    
    Parameters
    ----------
    im1: np.ndarray
        first image with transparency (4 channels)
    im2: np.ndarray
        second image with transparency (4 channels)
    
    Returns
    -------
    result: np.ndarray
        difference between the two images
    
    Raises:
        ValueError: if the input images do not have the same shape
        ValueError: if the input images do not have 4 channels (RGBA)
    """
    if im1.shape != im2.shape:
        raise ValueError("Input images must have the same shape")
    
    if len(im1.shape) != 3 or im1.shape[2] != 4:
        raise ValueError("Input images must have 4 channels (RGBA)")
    
    diff = im1 - im2
    result = diff[..., :3] * (1 - diff[..., 3:])
    return result



def alphaColorSquaredNormOnPixel(pixel: np.ndarray) -> float | np.ndarray:
    """
    Compute the custom alpha color squared norm of a pixel, where this norm is defined by the following:
    
    If the pixel is of color (r, g, b, a), then the norm is defined as:
    norm = (r^2 + g^2 + b^2) * a^2
    r, g, b, a are in [0., 1.]
    
    If the input is an array with the last axis of size 4, the norm will be computed on each pixel.

    Parameters
    ----------
    pixel: np.ndarray
        pixel whose squared norm should be computed
        
    Returns
    -------
    norm: float or np.ndarray
        squared norm of the considered pixel, or the array of squared norms
    
    Raises
    ------
        ValueError: if the input has not the right dtype (float or uint8)
        ValueError: if the input pixel does not have 4 channels  (RGBA)
    """
    if pixel.shape[-1] != 4:
        raise ValueError("Input pixel must have 4 channels (RGBA)")

    if pixel.dtype == np.uint8:
        pixel = (pixel / 255.0).astype(np.float32)
    elif not np.issubdtype(pixel.dtype, np.floating):
        raise ValueError("Input pixel must be a float or uint8 array")

    squared_rgb = np.square(pixel[..., :3])
    squared_alpha = np.square(pixel[..., 3])
    squared_norm = np.sum(squared_rgb, axis=-1) * squared_alpha
    return squared_norm



def alphaColorNormOnPixel(pixel: np.ndarray) -> float | np.ndarray:
    """
    Compute the custom alpha color norm of a pixel, where this norm is defined by the following:
    
    If the pixel is of color (r, g, b, a), then the norm is defined as:
    norm = sqrt(r^2 + g^2 + b^2) * a
    r, g, b, a are in [0., 1.]
    
    If the input is an array with the last axis of size 4, the norm will be computed on each pixel.

    Parameters
    ----------
    pixel: np.ndarray
        pixel whose norm should be computed
        
    Returns
    -------
    norm: float or np.ndarray
        norm of the considered pixel, or the array of norms
    
    Raises
    ------
        ValueError: if the input pixel does not have 4 channels  (RGBA)
    """
    return np.sqrt(alphaColorSquaredNormOnPixel(pixel))



def alphaColorSquaredNorm(im: np.ndarray) -> np.ndarray:
    """
    Compute the alpha color squared norm on each pixel of an image.
    If the image has no transparency, it will use the traditional euclidean norm.
    If the image is gray scale, the squared value of the pixel will be used.
    
    Parameters
    ----------
    im: np.ndarray
        image with transparency (4 channels)
    
    Returns
    -------
    norm_im: np.ndarray
        image with alpha color norm squared applied
    
    Raises:
        ValueError: if the input has not the right dtype (float or uint8)
        ValueError: if the input is not an image
        ValueError: if input image does not have 4 channels (RGBA)
        ValueError: if the input image has more than 3 dimensions
    """
    
    if im.dtype == np.uint8:
        im = (im.copy() / 255.0).astype(np.float32)
    elif not np.issubdtype(im.dtype, np.floating):
        raise ValueError("Input im must be a float or uint8 array")
    
    squared_norm_function = alphaColorSquaredNormOnPixel

    if len(im.shape) < 2:
        raise ValueError("Input image must be an image. It should have at least 2 dimensions.")
    
    if len(im.shape) == 2:
        squared_norm_function = np.square
    
    if len(im.shape) == 3:
        if im.shape[2] == 3:
            squared_norm_function = lambda x: np.sum(np.square(x), axis=-1)
        elif im.shape[2] != 4:
            raise ValueError("Input image must have 1, 3 or 4 channels (RGBA).")
    
    else:
        raise ValueError("Input image should either be gray scale or color image, which makes 2 or 3D.")
    
    square_norm_im = squared_norm_function(im[:, :])
    return square_norm_im



def alphaColorNorm(im: np.ndarray) -> np.ndarray:
    """
    Compute the alpha color norm on each pixel of an image.
    If the image has no transparency, it will be computed as if it was 100% opaque.
    If the image is gray scale, the absolute value of the pixel will be used.
    
    Parameters
    ----------
    im: np.ndarray
        image with transparency (4 channels)
    
    Returns
    -------
    norm_im: np.ndarray
        image with alpha color norm applied
    
    Raises:
        ValueError: if the input is not an image
        ValueError: if input image does not have 4 channels (RGBA)
        ValueError: if the input image has more than 3 dimensions
    """
    
    norm_im = np.sqrt(alphaColorSquaredNorm(im))
    return norm_im



def setTransparentPixelsTo(im: np.ndarray, color: tuple = (0, 0, 0, 0)) -> np.ndarray:
    """
    Set transparent pixels `[b, g, r, 0]` to the given color.
    The input image is not modified, a copy is made and returned.
    
    Parameters
    ----------
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
                      weight_euclidean: float = 1.0,
                      weight_alpha: float = 1.0,
                      normalized: bool = True,
                      display_advancement: bool = False,
                      show: bool = False,
                      show_threshold: float = 1.01
) -> np.ndarray:
    """Compute the custom squared difference map between the two images.
    It uses the transparency difference of the template and each cropped image.
    It uses the traditional euclidean norm combined with the alpha norm. The alpha norm is defined as:
    norm = sqrt(r^2 + g^2 + b^2) * a
    r, g, b, a are in [0., 1.]
    This combination uses the two given weights to compute the weighted average norm.

    Args:
        image (np.ndarray): original image (HxW)
        template (np.ndarray): template image (h x w)
        weight_euclidean (float, optional): weight for the euclidean norm. Defaults to 1.0.
        weight_alpha (float, optional): weight for the alpha norm. Defaults to 1.0.
        normalized (bool, optional): whether to normalize the squared difference map. Defaults to True.
        display_advancement (bool, optional): whether to show the progress. Defaults to True.
        show (bool, optional): whether to show the difference map. Defaults to False.
        show_threshold (float, optional): relative threshold divided by the minimum value found
                                          under which the corresponding value is shown. Defaults to 1.2.

    Returns:
        np.ndarray: squared difference map
    
    Raises:
        ValueError: if the input has not the right dtype (float or uint8)
		ValueError: if input image is not a color image
		ValueError: if input image and template have different number of channels
		ValueError: if template image is larger than the original image
        ValueError: if both weights are 0
    """
    
    if len(image.shape) < 2:
        raise ValueError("Input image must be a color image")
    
    if len(image.shape) != len(template.shape):
        raise ValueError("Input image and template must have the same number of channels")
    
    im_size = image.shape[:2]#[::-1]
    t_size = template.shape[:2]#[::-1]
    
    if t_size[0] > im_size[0] or t_size[1] > im_size[1]:
        raise ValueError("Template image must be smaller than the original image")
    
    if weight_euclidean < 0:
        print("Warning: weight for the euclidean norm is negative, it will be set to 0")
        weight_euclidean = 0
    
    if weight_alpha < 0:
        print("Warning: weight for the alpha norm is negative, it will be set to 0")
        weight_alpha = 0
    
    if weight_alpha + weight_euclidean == 0:
        raise ValueError("At least one of the weights must be non-zero. They must be positive or null.")
    
    if image.dtype == np.uint8:
        image = (image.copy() / 255.0).astype(np.float32)
    elif not np.issubdtype(image.dtype, np.floating):
        raise ValueError("Input im must be a float or uint8 array")
    
    if template.dtype == np.uint8:
        template = (template.copy() / 255.0).astype(np.float32)
    elif not np.issubdtype(template.dtype, np.floating):
        raise ValueError("Input template must be a float or uint8 array")
    
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
            # diff = diffTransparency(cropped_image, template)
            
            if weight_euclidean == 0.:
                euclidean_norm = 0.
            else:
                euclidean_norm = np.sum(np.square(diff))
            
            if weight_alpha == 0.:
                alpha_norm = 0.
            else:
                alpha_norm = np.sum(alphaColorSquaredNorm(diff))
            
            result[i, j] = (weight_alpha * alpha_norm + weight_euclidean * euclidean_norm) / (weight_alpha + weight_euclidean)
            if normalized:
                template_squared_norm = np.sum(np.square(template))
                cropped_image_squared_norm = np.sum(np.square(cropped_image))
                result[i, j] /= np.sqrt(template_squared_norm * cropped_image_squared_norm)
            
            # result[i,j] = cv.matchTemplate(cropped_image, template, cv.TM_CCOEFF_NORMED)
            
            if display_advancement:
                percentage = round((i * col_iter + j) / nb_iter * 100)
                if percentage != last_percentage:
                    last_percentage = percentage
                    print(f"\tAdvancement: {percentage}%", end='\r')
    
    
    
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
            
            # diff = cropped_image - template
            diff = np.abs(diffTransparency(cropped_image, template))
            
            print(diff.shape)
            print(diff[0, 0])
            print(diff[-1, -1])
            print(diff[diff.shape[0]//2, diff.shape[1]//2])
            print(cropped_image[cropped_image.shape[0]//2, cropped_image.shape[1]//2])
            print(template[template.shape[0]//2, template.shape[1]//2])
            
            # norm_diff = np.sum(np.square(diff), axis=-1)
            norm_diff = alphaColorSquaredNorm(diff)

            
            print(norm_diff.shape)
            
            s = result[x, y]
            print(s == np.sum(norm_diff))
            
            diff = diff / np.max(diff)
            norm_diff = norm_diff/np.max(norm_diff)
            print(norm_diff.shape)
            
            print("Resulting value:", s, " which makes a ratio of ", s/min_value, " compared to the minimum value")
            
            cv.imshow('Difference', diff)
            cv.imshow('Normalized Difference', norm_diff)
            
            
            im_copy = image.copy()
            # On rectangle, width and height are inverted so x, y becomes y, x
            cv.rectangle(im_copy, (y, x), (y+t_size[1], x+t_size[0]), (0, 255, 0), 2)
            cv.imshow("Location in the image", im_copy)
            
            waitNextKey(0)
    
    cv.destroyAllWindows()
    
    return result
