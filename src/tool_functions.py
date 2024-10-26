import cv2 as cv
import numpy as np
from sklearn.cluster import DBSCAN
from typing import Any

from enum import Enum



#?################################################################################################
#?                                                                                               #
#?                                           GENERIC                                             #
#?                                                                                               #
#?################################################################################################


class ImageType(Enum):
    GRAY_SCALE = 1
    COLOR = 2
    COLOR_WITH_ALPHA = 3
    
    @staticmethod
    def getImageType(image: np.ndarray) -> 'ImageType':
        """
        Detects the type of the image (gray scale, color, color with alpha).
        
        Parameters
        ----------
        image: np.ndarray
            image to detect the type of
        
        Returns
        -------
        image_type: ImageType
            type of the image, or None if it is not a correct image
        """
        image_type = None
        
        if len(image.shape) == 2:
            image_type = ImageType.GRAY_SCALE
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                image_type = ImageType.COLOR
            elif image.shape[2] == 4:
                image_type = ImageType.COLOR_WITH_ALPHA
        
        return image_type



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



def nanNormalize(im: np.ndarray) -> np.ndarray:
    """
    Normalize the image to have values between 0 and 1.
    It will ignore the nan values.
    
    Parameters
    ----------
    im: np.ndarray
        image to normalize
    
    Returns
    -------
    normalized_im: np.ndarray
        normalized image
    """
    max_value = np.nanmax(im)
    min_value = np.nanmin(im)
    return (im - min_value) / (max_value - min_value)



def getBlackColor(image_type: ImageType) -> tuple:
    match image_type:
        case ImageType.GRAY_SCALE:
            return 0
        case ImageType.COLOR:
            return (0, 0, 0)
        case ImageType.COLOR_WITH_ALPHA:
            return (0, 0, 0, 0)




def rotateImage(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate an image by the given angle (in degrees).
    The image is rotated around its center and keeps the same size.
    This rotation method ignores np.nan values, and restore them in the rotated result.
    
    Parameters
    ----------
    image: np.ndarray
        image to rotate
    angle: float
        angle in degrees to rotate the image
    
    Returns
    -------
    result: np.ndarray
        rotated image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    return cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)



def _rotateImageIgnoreNan(image: np.ndarray, angle: float) -> np.ndarray:
    # Create a mask of the np.nan values
    nan_mask = np.isnan(image).astype(np.float32)

    # Compute the rotation matrix
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)

    # Apply the affine transformation to the image and the mask
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    
    b_value = 1
    image_type = ImageType.getImageType(image)
    
    if image_type is None:
        raise ValueError("Input image must be a correct image, either gray scale or color image with or without transparency")
    
    match image_type:
        case ImageType.GRAY_SCALE:
            b_value = 1
        case ImageType.COLOR:
            b_value = (1, 1, 1)
        case ImageType.COLOR_WITH_ALPHA:
            b_value = (1, 1, 1, 1)
    transformed_nan_mask = cv.warpAffine(nan_mask, rot_mat, image.shape[1::-1], flags=cv.INTER_NEAREST, borderValue=b_value)

    # Restore the np.nan values in the result using the transformed mask
    result[transformed_nan_mask > 0] = np.nan
    
    return result



def computeRotatedRequiredSize(image_size: tuple[int], angle: float) -> tuple[int]:
    """
    Compute the required size for an image after rotation by the given angle.
    The image is rotated around its center.
    
    Parameters
    ----------
    image_size: tuple[int]
        size of the image, in the order (width, height)
    angle: float
        angle in degrees to rotate the image
    
    Returns
    -------
    (W, H): tuple[int]
        required size for the rotated image: (Width, Height)
    """
    image_size = np.array(image_size)
    w, h = image_size[..., 0], image_size[..., 1]
    rad_angle = np.radians(angle)
    c = np.cos(rad_angle)
    s = np.sin(rad_angle)
    
    H = (np.ceil(h * abs(c) + w * abs(s))).astype(int)
    W = (np.ceil(h * abs(s) + w * abs(c))).astype(int)
    
    new_image_size = np.stack((W, H), axis=-1)
    
    return new_image_size



def rotateImageWithoutLoss(image: np.ndarray, angle: float, color: int | float | tuple = None):
    """
    Rotate an image by the given angle.
    The resulting image is resized so that no information is lost. The added pixels are filled with the given color (Black by default).
    
    Parameters
    ----------
    image: np.ndarray
        image to rotate
    angle: float
        angle in degrees to rotate the image
    color: int | float | tuple, optional
        color to fill the added pixels with (default: black)
    
    Returns
    -------
    result: np.ndarray
        rotated image
    
    Raises
    ------
        ValueError: if the input image is not a correct image
    """
    
    image_type = ImageType.getImageType(image)
    
    if image_type is None:
        raise ValueError("Input image must be a correct image, either gray scale or color image with or without transparency")
    
    
    # Get default color
    if color is None:
        color = getBlackColor(image_type)
    else:
        match image_type:
            case ImageType.GRAY_SCALE:
                if not isinstance(color, (int, float)):
                    raise ValueError("Color must be an int or a float for a gray scale image.")
            case ImageType.COLOR:
                if not isinstance(color, tuple) or len(color) != 3:
                    raise ValueError("Color must be a tuple of 3 elements for a color image.")
            case ImageType.COLOR_WITH_ALPHA:
                if not isinstance(color, tuple) or len(color) != 4:
                    raise ValueError("Color must be a tuple of 4 elements for a color image with transparency.")
    
    # Compute new image size
    h, w = image.shape[:2]
    W, H = computeRotatedRequiredSize((w, h), angle)
    
    temp_w, temp_h = max(w, W), max(h, H)
    
    
    dummy_value = np.nan
    new_image = np.full((temp_h, temp_w, 4), dummy_value, dtype=np.float64)
    
    # Add the original image in its center
    begin_h = (temp_h - h) // 2
    begin_w = (temp_w - w) // 2
    converted_image = None
    match image_type:
        case ImageType.GRAY_SCALE:
            converted_image = cv.cvtColor(image, cv.COLOR_GRAY2BGRA)
        case ImageType.COLOR:
            converted_image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
        case ImageType.COLOR_WITH_ALPHA:
            converted_image = image
    new_image[begin_h:begin_h + h, begin_w:begin_w + w] = converted_image
   
    # Rotate the image
    new_image = _rotateImageIgnoreNan(new_image, angle)
    
    # Crop the image
    new_image = cropOnNan(new_image)
    
    # Verify the size
    #print("Cropped image size: ", new_image.shape, " while W, H: ", W, H)
    
    # Replacing transparent by wanted color
    wanted_color = np.zeros(4)
    match image_type:
        case ImageType.COLOR_WITH_ALPHA:
            wanted_color = color
        case _:
            wanted_color[:3] = color
    # Convert np nans to wanted color
    new_image = np.nan_to_num(new_image, nan=wanted_color)
    
    new_image = new_image.astype(image.dtype)
    
    # Convert back to original image type
    match image_type:
        case ImageType.GRAY_SCALE:
            new_image = cv.cvtColor(new_image, cv.COLOR_BGRA2GRAY)
        case ImageType.COLOR:
            new_image = cv.cvtColor(new_image, cv.COLOR_BGRA2BGR)
        case ImageType.COLOR_WITH_ALPHA:
            pass
    
    return new_image
    

#?################################################################################################
#?                                                                                               #
#?                                      TRANSPARENCY SUPPORT                                     #
#?                                                                                               #
#?################################################################################################







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
    Crops an image supporting transparency so that we only keep the biggest non transparent sub-image.

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



def cropOnValue(im: np.ndarray, value: Any) -> np.ndarray:
    """
    Crops an image so that we only keep the biggest sub-image with no rows nor columns
    full of value.

    Parameters
    ----------
    im: np.ndarray
        Original image to be cropped.
    value: Any
        Value that should not be kept when cropping.
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
            if np.any(row != value):
                min_row_index = i
        
        if max_row_index is None:
            row = im[-1-i]
            if np.any(row != value):
                max_row_index = shape[0] - i
        
        if min_row_index is not None and max_row_index is not None:
            break

    for j in range(shape[1]):
        if min_col_index is None:
            col_left = im[:, j]
            if np.any(col_left != value):
                min_col_index = j
        
        if max_col_index is None:
            col_right = im[:, -1-j]
            if np.any(col_right != value):
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



def cropOnNan(im: np.ndarray) -> np.ndarray:
    """
    Crops an image so that we only keep the biggest sub-image with no rows nor columns
    full of nans.

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
            if not np.all(np.isnan(row)):
                min_row_index = i
        
        if max_row_index is None:
            row = im[-1-i]
            if not np.all(np.isnan(row)):
                max_row_index = shape[0] - i
        
        if min_row_index is not None and max_row_index is not None:
            break

    for j in range(shape[1]):
        if min_col_index is None:
            col_left = im[:, j]
            if not np.all(np.isnan(col_left)):
                min_col_index = j
        
        if max_col_index is None:
            col_right = im[:, -1-j]
            if not np.all(np.isnan(col_right)):
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


#?################################################################################################
#?                                                                                               #
#?                                 CLUSTERING AND SEGMENTATION                                   #
#?                                                                                               #
#?################################################################################################





#* K-Means clustering





def kmeans(im: np.ndarray,
           criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2),
           nClusters = 7,
           attempts = 30,
           flags = cv.KMEANS_PP_CENTERS) -> np.ndarray:
    
    shape = im.shape
    
    image_type = ImageType.getImageType(im)
    
    if image_type is None:
        raise ValueError("Input image must be a correct image, either gray scale or color image with or without transparency")
    
    nb_channels = 1
    match image_type:
        case ImageType.GRAY_SCALE:
            nb_channels = 1
        case ImageType.COLOR:
            nb_channels = 3
        case ImageType.COLOR_WITH_ALPHA:
            nb_channels = 4
    
    flatten_im = np.float32(im.reshape(-1, nb_channels))
    
    _, labels, centers = cv.kmeans(flatten_im, nClusters, None, criteria, attempts, flags)
    
    # Display the results
    COLORS = np.random.randint(0, 255, size=(nClusters, 3), dtype=np.uint8)
    
    centers = np.uint8(centers)
    
    segmented_image = COLORS[labels.flatten()].reshape(shape[:2] + (3,))
    
    return segmented_image





def remove_background_kmeans(im: np.ndarray,
                             nb_attempts: int = 7,
                             nb_clusters: int = 7,
                             nb_background_colors: int = 4,
                             replacing_color: tuple | float | int = None,
                             exclude_single_pixels: bool = False,
                             erosion_size: int = 3,
                             dilation_size: int = 3,
                             box_excluding_size: tuple[int] = (10, 10),
) -> np.ndarray:
    """
    Remove the background of an image.
    This is done by segmenting the image using KMeans clustering and considering the two most common colors as the background. (ground and wall/roof)
    Then, it detects the contours of the relevant objects.
    
    Parameters
    ----------
    im : np.ndarray
        The image to process.
    nb_attempts : int, optional
        The number of attempts for the KMeans algorithm. Defaults to 7.
    nb_clusters : int, optional
        The number of clusters for the KMeans algorithm. Defaults to 7.
    nb_background_colors : int, optional
        The number of most common colors to consider as the background. Defaults to 4.
    replacing_color : tuple | float | int, optional
        The color to replace the background with. If None, the default color is black.
        For grayscale images, replacing_color must be an int or float.
        For color images, replacing_color must be a tuple of three elements.
    exclude_single_pixels : bool, optional
        If True, single pixels will be removed from the objects by applying erosion followed by dilation.
        Defaults to False
    erosion_size : int, optional
        The size of the erosion kernel. Only used if exclude_single_pixels is True. Defaults to 3.
    dilation_size : int, optional
        The size of the dilation kernel. Only used if exclude_single_pixels is True. Defaults to 3.
    box_excluding_size : tuple[int], optional
        The size of the box to exclude objects that are too small.
        The box is defined by (width, height). Defaults to (10, 10).
    
    Returns
    -------
    np.ndarray
        The image with the background removed.
    contours : list
        The list of contours of the relevant non background objects detected.
        
    Raises
    ------
    ValueError
        If the image has an incorrect shape
        If replacing_color is not of the correct type.
        If the erosion and dilation sizes are not odd numbers greater than or equal to 1.
        If the box excluding size is not an integer or is less than 1.
    """
    image_type = ImageType.getImageType(im)
    
    if image_type is None:
        raise ValueError("Input image must be a correct image, either gray scale or color image with or without transparency")
    
    # Get the right replacing color and verify the given one
    if replacing_color is not None:
        match image_type:
            case ImageType.GRAY_SCALE:
                if not isinstance(replacing_color, (int, float)):
                    raise ValueError("The replacing color must be an int or a float for a gray scale image.")
            case ImageType.COLOR:
                if not isinstance(replacing_color, tuple) or len(replacing_color) != 3:
                    raise ValueError("The replacing color must be a tuple of three elements for a color image.")
            case ImageType.COLOR_WITH_ALPHA:
                if not isinstance(replacing_color, tuple) or len(replacing_color) != 4:
                    raise ValueError("The replacing color must be a tuple of four elements for a color image with transparency.")
    else:
        replacing_color = getBlackColor(image_type)
    
    
    if not isinstance(nb_attempts, int) or nb_attempts < 1:
        raise ValueError("The number of attempts must be an integer greater than or equal to 1.")
    if not isinstance(nb_clusters, int) or nb_clusters < 1:
        raise ValueError("The number of clusters must be an integer greater than or equal to 1.")
    if not isinstance(nb_background_colors, int) or nb_background_colors < 1 or nb_background_colors > nb_clusters:
        raise ValueError("The number of background colors must be an integer between 1 and the number of clusters.")
    
    
    if erosion_size < 1:
        raise ValueError("The erosion size must be an odd number greater than or equal to 1.")
    if erosion_size % 2 == 0:
        raise ValueError("The erosion size must be an odd number.")
    
    if dilation_size < 1:
        raise ValueError("The dilation size must be an odd number greater than or equal to 1.")
    if dilation_size % 2 == 0:
        raise ValueError("The dilation size must be an odd number.")
    
    
    if not isinstance(box_excluding_size, tuple):
        if isinstance(box_excluding_size, int):
            box_excluding_size = (box_excluding_size, box_excluding_size)
        else:
            raise ValueError("The box excluding size must be an integer or a tuple of two integers.")
    else:
        if not isinstance(box_excluding_size[0], int) or not isinstance(box_excluding_size[1], int):
            raise ValueError("The box excluding size must be an integer or a tuple of two integers.")
    
    if box_excluding_size[0] < 1 or box_excluding_size[1] < 1:
        raise ValueError("The box excluding dimensions must both be greater or equal to 1.")
    
    
    NB_ATTEMPTS = nb_attempts
    NB_CLUSTERS = nb_clusters
    NB_BACKGROUND_COLORS = nb_background_colors
    
    EROSION_SIZE = erosion_size
    DILATION_SIZE = dilation_size
    
    EXCLUDING_SIZE = box_excluding_size
    
    #TODO: Add edge detection to help segmenting the image
    # canny_edges = cv.Canny(im, 100, 200)
    # cv.imshow("Canny edges", canny_edges)
    
    seg_im = kmeans(im, nClusters=NB_CLUSTERS, attempts=NB_ATTEMPTS)
    
    #TODO: temporary
    # cv.imshow("After kmeans", seg_im)
    # seg_canny_edges = cv.Canny(seg_im, 100, 200)
    # cv.imshow("Segmented Canny edges", seg_canny_edges)
    
    # Get all unique colors and their counts
    unique_colors, counts = np.unique(seg_im.reshape(-1, 3), axis=0, return_counts=True)
    
    # Sort both arrays by counts (descending)
    sorted_indices = np.argsort(counts)[::-1]
    unique_colors = unique_colors[sorted_indices]
    counts = counts[sorted_indices]
    
    # Mask the background
    object_colors = unique_colors[NB_BACKGROUND_COLORS:]
    
    
    # Creating a new image with the replacing color and adding the object colors on top
    final_im = np.full(im.shape[:3], replacing_color, dtype=np.uint8)
    
    mask_int = (np.isin(seg_im.reshape(-1, 3), object_colors).all(axis=-1).reshape(seg_im.shape[:2])).astype(np.uint8)
    
    #TODO: temporary
    cv.imshow("After kmeans: initial object mask", mask_int * 255)
    
    if exclude_single_pixels:
        # Remove single object pixels by erosion + dilation    
        mask_int = cv.dilate(mask_int, np.ones((DILATION_SIZE, DILATION_SIZE), np.uint8), iterations=1)
        mask_int = cv.erode(mask_int, np.ones((EROSION_SIZE, EROSION_SIZE), np.uint8), iterations=1)
        
    #     #TODO: temporary
    #     cv.imshow("Post erosion", mask_int * 255)
        
    # Then detect objects' contours
    contours, hierarchy = cv.findContours(mask_int * 255, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    relevant_contours = []
    
    
    #TODO: temporary
    contour_im = np.zeros(im.shape[:2], dtype=np.uint8)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        
        if w >= EXCLUDING_SIZE[0] and h >= EXCLUDING_SIZE[1]:
            relevant_contours.append(contour)
            cv.drawContours(contour_im, [contour], -1, 255, 1)
    
    #TODO: temporary
    cv.imshow("Post contour", contour_im)
    
    mask = mask_int.astype(bool)
    
    final_im[mask] = im[mask]
    
    return final_im, relevant_contours






#* Superpixels


def slic(im: np.ndarray,
         region_size: int = 30,
         ruler: float = 10.0,
         iterations: int = 10,
         min_element_size: int = 25,
) -> np.ndarray:
    """
    Perform SLIC (Simple Linear Iterative Clustering) superpixel segmentation on an image.
    
    Parameters
    -----------
    im : np.ndarray
        Input image on which superpixel segmentation is to be performed.
    region_size : int, optional
        Size of the superpixel regions. Default is 30.
    ruler : float, optional
        Compactness parameter for the SLIC algorithm. Default is 10.0.
    iterations : int, optional
        Number of iterations for the SLIC algorithm. Default is 10.
    min_element_size : int, optional
        Minimum element size to enforce label connectivity. Default is 25.
    
    Returns
    --------
    mask_slic : np.ndarray
        Mask of the superpixel boundaries.
    labels : np.ndarray
        Array of labels for each pixel.
    num_labels : int
        Number of superpixels generated.
    """
    SLIC = cv.ximgproc.SLIC
    SLICO = cv.ximgproc.SLICO
    MSLIC = cv.ximgproc.MSLIC

    algorithm = MSLIC # can be SLIC, SLICO or MSLIC
    
    
    # Apply gaussian filter to the image
    im = cv.GaussianBlur(im, (5, 5), 0)
    
    if len(im.shape) == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2BGR)
    elif len(im.shape) == 3 and im.shape[2] == 4:
        im = cv.cvtColor(im, cv.COLOR_BGRA2BGR)
    elif len(im.shape) != 3:
        raise ValueError("Input image must be a color image")
    
    # Convert im to LAB color space
    im = cv.cvtColor(im, cv.COLOR_BGR2LAB)
    
    # Process SLIC
    slic = cv.ximgproc.createSuperpixelSLIC(im, algorithm, region_size=region_size, ruler=ruler)
    
    slic.iterate(iterations)
    slic.enforceLabelConnectivity(min_element_size)
    
    mask_slic = slic.getLabelContourMask()
    labels = slic.getLabels()
    num_labels = slic.getNumberOfSuperpixels()
    
    return mask_slic, labels, num_labels






def remove_background_superpixels(im: np.ndarray,
) -> np.ndarray:
    
    mask_slic, labels, num_labels = slic(im)
    
    #TODO: temporary
    contour_color = (0, 255, 0)
    superpixel_im = im.copy()
    superpixel_im[mask_slic != 0] = contour_color
    
    cv.imshow("Superpixel image", superpixel_im)
    
    
    # Compute average color of each superpixel
    average_colors = []
    for i in range(num_labels):
        mask = (labels == i)
        avg_color = cv.mean(im, mask.astype(np.uint8) * 255)[:3]
        average_colors.append(avg_color)
    average_colors = np.array(average_colors)
    
    # Clustering using DBSCAN
    dbscan = DBSCAN(eps=10, min_samples=2, metric="euclidean")
    clusters = dbscan.fit_predict(average_colors)
    
    output_image = np.zeros_like(im)
    for i in range(num_labels):
        cluster_id = clusters[i]
        if cluster_id != -1:
            mask = (labels == i)
            output_image[mask] = average_colors[i]
        else:
            mask = (labels == i)
            output_image[mask] = (0, 0, 255)
    
    #! COMMENTS: Extinguishers are almost always classified as noise
    #! Superpixels won't be used therefore.
    
    
    cv.imshow("Clustered image", output_image)






#?################################################################################################
#?                                                                                               #
#?                                 COMPUTING SIMILARITY MAPS                                     #
#?                                                                                               #
#?################################################################################################





# TODO: Remove?
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


# TODO: Remove?
def custom_matching(im: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Compute the squared difference map between the two images.
    It only consider the RGB channels, where the template transparency is not 0.
    
    Parameters
    ----------
    im: np.ndarray
        original image
    template: np.ndarray
        template image
    
    Returns
    -------
    similarity_map: np.ndarray
        squared difference map
    """
    
    h, w = im.shape[:2]
    th, tw = template.shape[:2]
    
    sim_h, sim_w = h - th + 1, w - tw + 1
    
    similarity_map = np.zeros((sim_h, sim_w), dtype=np.float32)
    mask = template[:, :, 3] > 0
    
    for i in range(sim_h):
        for j in range(sim_w):
            cropped_im = im[i:i+th, j:j+tw]
            diff = cropped_im - template
            squared_diff = np.sum(np.square(diff[mask]))
            similarity_map[i, j] = squared_diff
    
    return similarity_map