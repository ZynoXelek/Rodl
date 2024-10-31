from typing import Callable
import cv2 as cv
import numpy as np

from enum import Enum

import os
from ultralytics import YOLO

from tool_functions import *
from ai_tool_functions import *

# #?: these lists are useful to try to tune reliability thresholds
# DEPTH_TESTING_LIST = []
# BACKGROUND_TESTING_LIST = []

class Case(Enum):
    """
    CASES are indexes of min/max value and min/max location in the similarity stats list.
    If you are using a custom method for the advanced template matching, you should provide the corresponding case, which means that
    the trustful value is the min/max value of the resulting similarity map.
    """
    MIN = 0, 2
    MAX = 1, 3



class MatchingMode(Enum):
    """MatchingMode of the template matching"""
    SQDIFF = cv.TM_SQDIFF, Case.MIN
    NORMED_SQDIFF = cv.TM_SQDIFF_NORMED, Case.MIN
    CCORR = cv.TM_CCORR, Case.MAX
    NORMED_CCORR = cv.TM_CCORR_NORMED, Case.MAX
    CCOEFF = cv.TM_CCOEFF, Case.MAX
    NORMED_CCOEFF = cv.TM_CCOEFF_NORMED, Case.MAX



class TemplateAdvancedMatcher():
    """
    Class that handles the advanced template matching.
    This matcher uses a pre-process method on each image to determine the areas of interest to look for objects.
    It can either use the classic method or the deep learning method to do so.
    Then, it tries to match the given template on each of these areas of interest.
    Finally, it scores the result to determine the reliability of the detected objects and it tries to detect fake ones.
    To do so, it can either use the depth information or the background information.
    
    Additionally, it computes the approximate position of each detected object in the image if provided a depth image.
    
    A lot of parameters uses the default values specified as class attributes. Though, most of them can be modified using the corresponding setter methods,
    for each of the defined modes.
    
    Attributes
    ----------
    mode: int
        The mode of the template matching. It can either be CLASSIC_MODE or AI_MODE.
    reliability_mode: int
        The mode of the reliability scoring. It can either be RELIABILITY_DEPTH_MODE or RELIABILITY_BACKGROUND_MODE.
    final_image: np.ndarray
        The final image with the detected objects.
    """
    
    #* Constants
    
    RECTANGLE_COLOR = (150, 255, 150)
    RECTANGLE_THICKNESS = 1
    
    BOX_LIMIT_RATIO = 0.5
    
    TEXT_COLOR = (255, 255, 255)
    FONT = cv.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.4
    TEXT_THICKNESS = 1
    
    DISPLAY_DEPTH_MATCHING_STEPS = False # Whether the steps of the depth matching should be displayed or not.
    
    # --- Modes ---
    
    CLASSIC_MODE = 0
    AI_MODE = 1
    
    AVAILABLE_MODS = [CLASSIC_MODE, AI_MODE]
    
    # --- Classic ---
    
    CLASSIC_DEFAULT_NB_ATTEMPTS = 7 # Number of attempts for the classic template matching, recommended around 10
    CLASSIC_DEFAULT_NB_CLUSTERS = 7 # Number of clusters for the kmeans segmentation, recommended to be set to 3 or 7
    CLASSIC_DEFAULT_NB_BACKGROUND_COLORS = 4 # Number of background colors to remove, recommended to be set to 2 or 4
    CLASSIC_DEFAULT_EXCLUDE_SINGLE_PIXELS = True # If True, the single pixels will be excluded from the background removal
    CLASSIC_DEFAULT_EROSION_SIZE = 3 # Size of the erosion kernel for the background removal
    CLASSIC_DEFAULT_DILATION_SIZE = 3 # Size of the dilation kernel for the background removal
    CLASSIC_DEFAULT_BOX_EXCLUDING_SIZE = (10, 10) # Size of the box to exclude from the background removal
    CLASSIC_DEFAULT_DISPLAY = False # If True, the steps of the classic method will be displayed
    
    # --- AI ---
    
    AI_MODEL_FOLDER = 'res/train_models/training_model2/'
    AI_MODEL_FILE = 'weights/best.pt'
    AI_DEFAULT_TARGET_CLASSES_ID = [0]
    AI_DEFAULT_TARGET_CLASSES_NAMES = ['extinguisher']
    
    # --- Reliability Scoring ---
    
    RELIABILITY_DEPTH_MODE = 0
    RELIABILITY_BACKGROUND_MODE = 1
    
    RELIABILITY_AVAILABLE_MODES = [RELIABILITY_DEPTH_MODE, RELIABILITY_BACKGROUND_MODE]
    
    # Using Depth
    DEPTH_OUTBOX_FACTORS = (1.8, 1.4)   # Factors to multiply the pre-treatment box' dimensions to get the dimensions of the box to extract the depth values from the edges
                                        # 1 means that the box will be of the same size as the original box
    DEPTH_DIFFERENCE_THRESHOLD = 1  # Threshold above (strictly) which the depth difference makes the pixel increase the reliability score
    DEPTH_RELIABILITY_THRESHOLD = 0.30    # Threshold above which the object is considered valid
    DEPTH_DISPLAY_RELIABILITY_POINTS = False # Whether the reliability points should be displayed or not while scoring each object
    
    # Using Background
    BACKGROUND_TEMPLATE_SIZE_FACTOR = 1.2 # Factor by which the template size is scaled (but does not scale the image itself) to add white pixels around
    BACKGROUND_RELIABILITY_HIGH_THRESHOLD = 0.40 # Threshold above which the object is considered reliable
    BACKGROUND_RELIABILITY_LOW_THRESHOLD = 0.30 # Threshold below which the object is considered unreliable
    
    #* Constructor
    
    def __init__(self, mode: int = CLASSIC_MODE, reliability_mode: int = RELIABILITY_BACKGROUND_MODE) -> None:
        """
        Constructor for a template advanced matcher object with the given mode.
        
        Parameters
        ----------
        mode: int
            The mode of the template matching. It can either be CLASSIC_MODE or AI_MODE.
            Defaults to CLASSIC_MODE.
        reliability_mode: int
            The mode of the reliability scoring. It can either be RELIABILITY_DEPTH_MODE or RELIABILITY_BACKGROUND_MODE.
            Defaults to RELIABILITY_BACKGROUND_MODE.
        
        Raises
        ------
        ValueError: If the mode is not valid.
        """
        if mode not in TemplateAdvancedMatcher.AVAILABLE_MODS:
            raise ValueError("The mode is not valid. Please use one of the available modes: CLASSIC_MODE or AI_MODE.")
        
        self.reset()
        self.setMode(mode)
        self.setReliabilityMode(reliability_mode)
    
    
    
    #* Getters and setters
    
    def setMode(self,
                mode: int = CLASSIC_MODE,
                *args, **kwargs,
    ) -> None:
        """
        Setter for the object-detection mode.
        You can additionally set the mode variables with the given arguments and keyword arguments.
        You can also do so by calling the setClassicModeVariables() or setAIModelVariables() methods afterwards.
        
        Parameters
        ----------
        mode: int
            Integer that represents the chosen mode.
        args: list
            List of arguments to set the mode variables.
        kwargs: dict
            Dictionary of keyword arguments to set the mode variables.
        
        Raises
        ------
        ValueError: If the mode does not exist.
        """
        if mode not in TemplateAdvancedMatcher.AVAILABLE_MODS:
            raise ValueError("The mode is not valid. Please use one of the available modes: CLASSIC_MODE or AI_MODE.")
        
        self.mode = mode
        match mode:
            case TemplateAdvancedMatcher.CLASSIC_MODE:
                self._pre_matching_method = self._classicBoxesDetection
                self.resetClassicModeVariables()
                self.setClassicModeVariables(*args, **kwargs)
            case TemplateAdvancedMatcher.AI_MODE:
                self._pre_matching_method = self._deepLearningBoxesDetection
                self.resetAIModeVariables()
                self.setAIModeVariables(*args, **kwargs)
    
    def setReliabilityMode(self,
                       reliability_mode: int = RELIABILITY_BACKGROUND_MODE,
                       *args, **kwargs,
    ) -> None:
        """
        Setter for the reliability scoring mode.
        You can additionally set the mode variables with the given arguments and keyword arguments.
        
        Parameters
        ----------
        reliability_mode: int
            Integer that represents the chosen mode.
        args: list
            List of arguments to set the mode variables.
        kwargs: dict
            Dictionary of keyword arguments to set the mode variables.
        
        Raises
        ------
        ValueError: If the mode does not exist.
        """
        if reliability_mode not in TemplateAdvancedMatcher.RELIABILITY_AVAILABLE_MODES:
            raise ValueError("The mode is not valid. Please use one of the available modes: RELIABILITY_DEPTH_MODE or RELIABILITY_BACKGROUND_MODE.")
        
        self.reliability_mode = reliability_mode
        match reliability_mode:
            case TemplateAdvancedMatcher.RELIABILITY_DEPTH_MODE:
                self.resetScoringDepthVariables()
                self.setScoringDepthVariables(*args, **kwargs)
            case TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE:
                self.resetScoringBackgroundVariables()
                self.setScoringBackgroundVariables(*args, **kwargs)
    
    def reset(self) -> None:
        """
        Reset the private attributes of the template advanced matcher object.
        """
        # Reset private attributes for future template matching
        self._original_template = None
        self._original_range_fx = None
        self._original_range_fy = None
        self._original_range_theta = None
        self._original_template_size_table = None
        self._range_fx = None
        self._range_fy = None
        self._range_theta = None
        self._template_size_table = None
        self._valid_sizes = None
        self._min_th = None
        self._min_tw = None
        
        
        self._similarity_stats = [] # list of [min_val, max_val, min_index, max_index, mean_val, median_val] stats for each similarity map
        self._similarity_maps = []
        self.final_image = None
        
        # Reset mode
        self.mode = None
        self._pre_matching_method = None
        
        # Reset scoring mode
        self.reliability_mode = None
    
    
    
    def resetClassicModeVariables(self) -> None:
        """
        Reset the variables for the classic mode of the template matching to use the default class ones.
        """
        self._classic_nb_attempts = TemplateAdvancedMatcher.CLASSIC_DEFAULT_NB_ATTEMPTS
        self._classic_nb_clusters = TemplateAdvancedMatcher.CLASSIC_DEFAULT_NB_CLUSTERS
        self._classic_nb_background_colors = TemplateAdvancedMatcher.CLASSIC_DEFAULT_NB_BACKGROUND_COLORS
        self._classic_exclude_single_pixels = TemplateAdvancedMatcher.CLASSIC_DEFAULT_EXCLUDE_SINGLE_PIXELS
        self._classic_erosion_size = TemplateAdvancedMatcher.CLASSIC_DEFAULT_EROSION_SIZE
        self._classic_dilation_size = TemplateAdvancedMatcher.CLASSIC_DEFAULT_DILATION_SIZE
        self._classic_box_excluding_size = TemplateAdvancedMatcher.CLASSIC_DEFAULT_BOX_EXCLUDING_SIZE
        self._classic_display = TemplateAdvancedMatcher.CLASSIC_DEFAULT_DISPLAY
    
    
    
    def setClassicModeVariables(self,
                                nb_attempts: int = None,
                                nb_clusters: int = None,
                                nb_background_colors: int = None,
                                exclude_single_pixels: bool = None,
                                erosion_size: int = None,
                                dilation_size: int = None,
                                box_excluding_size: tuple[int] = None,
                                classic_display: bool = None,
    ) -> None:
        """
        Sets the variables for the classic mode of the template matching.
        If a variable is not provided, the value will not change.
        
        Parameters
        ----------
        nb_attempts: int
            The number of attempts for the classic template matching.
            Default class value is 7.
        nb_clusters: int
            The number of clusters for the kmeans segmentation.
            Default class value is 7.
        nb_background_colors: int
            The number of background colors to remove.
            Default class value is 4.
        exclude_single_pixels: bool
            If True, the single pixels will be excluded from the background removal.
            Default class value is True.
        erosion_size: int
            The size of the erosion kernel for the background removal.
            Default class value is 3.
        dilation_size: int
            The size of the dilation kernel for the background removal.
            Default class value is 3.
        box_excluding_size: tuple[int]
            The size of the boxes to exclude from the background removal result.
            No box of interest of a size smaller than this value will be considered.
            Default class value is (10, 10).
        classic_display: bool
            If True, the steps of the classic method will be displayed.
            Default class value is False.
        """
        self._classic_nb_attempts = nb_attempts if nb_attempts is not None else self._classic_nb_attempts
        self._classic_nb_clusters = nb_clusters if nb_clusters is not None else self._classic_nb_clusters
        self._classic_nb_background_colors = nb_background_colors if nb_background_colors is not None else self._classic_nb_background_colors
        self._classic_exclude_single_pixels = exclude_single_pixels if exclude_single_pixels is not None else self._classic_exclude_single_pixels
        self._classic_erosion_size = erosion_size if erosion_size is not None else self._classic_erosion_size
        self._classic_dilation_size = dilation_size if dilation_size is not None else self._classic_dilation_size
        self._classic_box_excluding_size = box_excluding_size if box_excluding_size is not None else self._classic_box_excluding_size
        self._classic_display = classic_display if classic_display is not None else self._classic_display
    
    
    
    def resetAIModeVariables(self) -> None:
        """
        Reset the variables for the AI mode of the template matching to use the default class ones.
        """
        self._ai_model_path = os.path.join(TemplateAdvancedMatcher.AI_MODEL_FOLDER, TemplateAdvancedMatcher.AI_MODEL_FILE)
        self._target_classes_id = TemplateAdvancedMatcher.AI_DEFAULT_TARGET_CLASSES_ID
        self._target_classes_names = TemplateAdvancedMatcher.AI_DEFAULT_TARGET_CLASSES_NAMES
    
    
    
    
    def setAIModeVariables(self,
                           model_path: str = None,
                           target_classes_id: list[int] = None,
                           target_classes_names: list[str] = None,
    ) -> None:
        """
        Sets the variables for the AI mode of the template matching.
        If a variable is not provided, the value will not change.
        
        Parameters
        ----------
        model_path: str
            The path to the model to use for the object detection.
            Default class value is 'res/train_models/training_model2/weights/best.pt'.
        target_classes_id: list[int]
            The list of target classes IDs to detect.
            Default class value is [0].
        target_classes_names: list[str]
            The list of target classes names to detect.
            Default class value is ['extinguisher'].
        """
        self._ai_model_path = model_path if model_path is not None else self._ai_model_path
        self._target_classes_id = target_classes_id if target_classes_id is not None else self._target_classes_id
        self._target_classes_names = target_classes_names if target_classes_names is not None else self._target_classes_names
    
    
    
    def resetScoringDepthVariables(self) -> None:
        """
        Reset the variables for the depth scoring mode of the template matching to use the default class ones.
        """
        self._depth_outbox_factors = TemplateAdvancedMatcher.DEPTH_OUTBOX_FACTORS
        self._depth_difference_threshold = TemplateAdvancedMatcher.DEPTH_DIFFERENCE_THRESHOLD
        self._depth_reliability_threshold = TemplateAdvancedMatcher.DEPTH_RELIABILITY_THRESHOLD
        self._depth_display_reliability_points = TemplateAdvancedMatcher.DEPTH_DISPLAY_RELIABILITY_POINTS
    
    
    
    def setScoringDepthVariables(self,
                                 outbox_factor: float = None,
                                 difference_threshold: float = None,
                                 reliability_threshold: float = None,
                                 display_reliability_points: bool = None,
    ) -> None:
        """
        Sets the variables for the depth scoring mode.
        If a variable is not provided, the value will not change.
        
        Parameters
        ----------
        outbox_factor: float
            The factor to multiply the pre-treatment box' dimensions to get the dimensions of the box to extract the depth values from the edges.
            Default class value is (1.8, 1.4).
        difference_threshold: float
            The threshold above (strictly) which the depth difference makes the pixel increase the reliability score.
            Default class value is 1.
        reliability_threshold: float
            The threshold above which the object is considered valid.
            Default class value is 0.30.
        display_reliability_points: bool
            If True, the reliability points should be displayed while scoring each object.
            Default class value is False.
        """
        self._depth_outbox_factors = outbox_factor if outbox_factor is not None else self._depth_outbox_factors
        self._depth_difference_threshold = difference_threshold if difference_threshold is not None else self._depth_difference_threshold
        self._depth_reliability_threshold = reliability_threshold if reliability_threshold is not None else self._depth_reliability_threshold
        self._depth_display_reliability_points = display_reliability_points if display_reliability_points is not None else self._depth_display_reliability_points
    
    def resetScoringBackgroundVariables(self) -> None:
        """
        Reset the variables for the background scoring mode of the template matching to use the default class ones.
        """
        self._background_template_size_factor = TemplateAdvancedMatcher.BACKGROUND_TEMPLATE_SIZE_FACTOR
        self._background_reliability_high_threshold = TemplateAdvancedMatcher.BACKGROUND_RELIABILITY_HIGH_THRESHOLD
        self._background_reliability_low_threshold = TemplateAdvancedMatcher.BACKGROUND_RELIABILITY_LOW_THRESHOLD
    
    
    def setScoringBackgroundVariables(self,
                                      template_size_factor: float = None,
                                      reliability_high_threshold: float = None,
                                      reliability_low_threshold: float = None,
    ) -> None:
        """
        Sets the variables for the background scoring mode.
        If a variable is not provided, the value will not change.
        
        Parameters
        ----------
        template_size_factor: float
            The factor by which the template size is scaled (but does not scale the image itself) to add white pixels around.
            Default class value is 1.2.
        reliability_high_threshold: float
            The threshold above which the object is considered reliable.
            Default class value is 0.40.
        reliability_low_threshold: float
            The threshold below which the object is considered unreliable.
            Default class value is 0.30.
        """
        self._background_template_size_factor = template_size_factor if template_size_factor is not None else self._background_template_size_factor
        self._background_reliability_high_threshold = reliability_high_threshold if reliability_high_threshold is not None else self._background_reliability_high_threshold
        self._background_reliability_low_threshold = reliability_low_threshold if reliability_low_threshold is not None else self._background_reliability_low_threshold
    
    
    
    #* Private helper methods
    @staticmethod
    def _computeTemplateSizeTable(template_width: int,
                                  template_height: int,
                                  range_fx: np.ndarray,
                                  range_fy: np.ndarray,
                                  range_theta: np.ndarray,
                                  resize_rotated: bool = True,
    ) -> np.ndarray:
        """
        This method returns the table of the different sizes of the template with the given scaling factors.
        
        Parameters
        -----------
        template_width: int
            The width of the template.
        template_height: int
            The height of the template.
        range_fx : np.ndarray
            The range of scaling factors in the x-dimension, of shape lx.
        range_fy : np.ndarray
            The range of scaling factors in the y-dimension, of shape ly.
        range_theta : np.ndarray
            The range of thetas in the theta-dimension, of shape lt.
        resize_rotated: bool
            If True, the sizes will be computed for images rotated without loss. Default is True.
        
        Returns
        -------
        template_size_table: np.ndarray
            The table of the different sizes of the template with the given scaling factors and rotation values.
            Its shape is (lx, ly, lt, 2).
        """
        
        lx = len(range_fx)
        ly = len(range_fy)
        lt = len(range_theta)
        
        template_size_table = np.zeros((lx, ly, lt, 2), dtype=np.uint32)
        
        # Create size table with the two given ranges to compute the base factor table
        factor_table = np.zeros((lx, ly, 2))
        for i in range(lx):
            factor_table[i, :, 1] = range_fy
        for j in range(ly):
            factor_table[:, j, 0] = range_fx
        
        base_sizes = np.zeros((lx, ly, 2), dtype=np.uint32)
        base_sizes[:, :, 0] = np.round(template_width * factor_table[:, :, 0], decimals=0)
        base_sizes[:, :, 1] = np.round(template_height * factor_table[:, :, 1], decimals=0)
        
        if resize_rotated:
            for k, theta in enumerate(range_theta):
                template_size_table[:, :, k] = computeRotatedRequiredSize(base_sizes, theta, assert_exact=True)
        else:
            for k in range(lt):
                template_size_table[:, :, k] = base_sizes
        
        return template_size_table
    
    @staticmethod
    def _extractValidTemplateSizeTable(template_size_table: np.ndarray,
                                       image_width: int,
                                       image_height: int,
                                       range_fx: np.ndarray,
                                       range_fy: np.ndarray,
                                       range_theta: np.ndarray,
    ) -> list[np.ndarray]:
        """
        Extracts a valid template size table by ensuring that the template sizes do not exceed the image dimensions.
        
        Parameters
        -----------
        template_size_table: np.ndarray
            The original table of the different sizes of the template with the given scaling factors, of shape (lx, ly, lt, 2).
        image_width: int
            The width of the image to match with the template.
        image_height: int
            The height of the image to match with the template.
        range_fx : np.ndarray
            The range of scaling factors in the x-dimension, of shape lx.
        range_fy : np.ndarray
            The range of scaling factors in the y-dimension, of shape ly.
        range_theta : np.ndarray
            The range of thetas in the theta-dimension, of shape lt.
        
        Returns
        --------
        new_template_size_table: np.ndarray
            The new table corresponding to the new ranges.
            Its shape is (lx', ly', lt', 2).
        valid_sizes: np.ndarray
            A valid_sizes table where each value tells if the corresponding template size is valid or not.
            Its shape is (lx', ly', lt').
        new_range_fx: np.ndarray
            The new range of scaling factors in the x-dimension, of shape lx'.
        new_range_fy: np.ndarray
            The new range of scaling factors in the y-dimension, of shape ly'.
        new_range_theta: np.ndarray
            The new range of angles to rotate the image with, of shape lt'.
        """
                
        valid_sizes = (template_size_table[:, :, :, 0] <= image_width) & (template_size_table[:, :, :, 1] <= image_height)

        # Get indexes of valid sizes
        valid_fx_mask = np.any(valid_sizes, axis=(1, 2))
        valid_fy_mask = np.any(valid_sizes, axis=(0, 2))
        valid_theta_mask = np.any(valid_sizes, axis=(0, 1))
        
        # New ranges
        new_range_fx = range_fx[valid_fx_mask]
        new_range_fy = range_fy[valid_fy_mask]
        new_range_theta = range_theta[valid_theta_mask]
        
        # New template size table
        new_template_size_table = template_size_table[valid_fx_mask, :, :][:, valid_fy_mask, :][:, :, valid_theta_mask]
        valid_sizes = valid_sizes[valid_fx_mask, :, :][:, valid_fy_mask, :][:, :, valid_theta_mask]

        return new_template_size_table, valid_sizes, new_range_fx, new_range_fy, new_range_theta
    
    
    
    
    @staticmethod
    def _computeValidTemplateSizeTable(image_width: int,
                                       image_height: int,
                                       template_width: int,
                                       template_height: int,
                                       range_fx: np.ndarray,
                                       range_fy: np.ndarray,
                                       range_theta: np.ndarray,
                                       resize_rotated: bool = True,
    ) -> list[np.ndarray]:
        """
        Computes a valid template size table by ensuring that the template sizes do not exceed the image dimensions.
        
        
        Parameters
        -----------
        image_width: int
            The width of the image to match with the template.
        image_height: int
            The height of the image to match with the template.
        template_width: int
            The width of the template.
        template_height: int
            The height of the template.
        range_fx : np.ndarray
            The range of scaling factors in the x-dimension, of shape lx.
        range_fy : np.ndarray
            The range of scaling factors in the y-dimension, of shape ly.
        range_theta : np.ndarray
            The range of thetas in the theta-dimension, of shape lt.
        resize_rotated: bool
            If True, the sizes will be computed for images rotated without loss. Default is True.
        
        Returns
        --------
        new_template_size_table: np.ndarray
            The new table corresponding to the new ranges.
            Its shape is (lx', ly', lt', 2).
        valid_sizes: np.ndarray
            A valid_sizes table where each value tells if the corresponding template size is valid or not.
            Its shape is (lx', ly', lt').
        new_range_fx: np.ndarray
            The new range of scaling factors in the x-dimension, of shape lx'.
        new_range_fy: np.ndarray
            The new range of scaling factors in the y-dimension, of shape ly'.
        new_range_theta: np.ndarray
            The new range of angles to rotate the image with, of shape lt'.
        """
        
        template_size_table =\
            TemplateAdvancedMatcher._computeTemplateSizeTable(template_width, template_height, range_fx, range_fy, range_theta, resize_rotated)
        
        new_template_size_table, valid_sizes, new_range_fx, new_range_fy, new_range_theta =\
            TemplateAdvancedMatcher._extractValidTemplateSizeTable(template_size_table, image_width, image_height, range_fx, range_fy, range_theta)
        
        return new_template_size_table, valid_sizes, new_range_fx, new_range_fy, new_range_theta
    
    
    
    @staticmethod
    def computeSimilarityStats(similarity_map: np.ndarray) -> list[float]:
        """
        This method computes the statistics of the given similarity map.
        
        Returns
        -------
        similarity_stats: list[float]
            The similarity statistics with the following order: [min_val, max_val, min_index, max_index, mean_val, median_val]
        """
        if similarity_map.size == 0 or np.all(np.isnan(similarity_map)):
            return [np.nan] * 6
        
        min_val = np.nanmin(similarity_map)
        min_index = np.nanargmin(similarity_map)
        min_index = np.unravel_index(min_index, similarity_map.shape)

        max_val = np.nanmax(similarity_map)
        max_index = np.nanargmax(similarity_map)
        max_index = np.unravel_index(max_index, similarity_map.shape)
        
        mean_val = np.nanmean(similarity_map)
        median_val = np.nanmedian(similarity_map)
        
        similarity_stats = [min_val, max_val, min_index, max_index, mean_val, median_val]
        
        return similarity_stats
    
    
    
    def _classicBoxesDetection(self, image: np.ndarray) -> list[tuple[int]]:
        """
        Detect relevant parts of the image to look for objects using the classic method.
        
        Parameters
        ----------
        image: np.ndarray
            The image to detect the relevant parts of.
        
        Returns
        -------
        relevant_boxes: list[tuple[int]]
            The list of relevant boxes found in the image in the format (x, y, w, h).
        """
        
        _, contours = remove_background_kmeans(image,
                                               nb_attempts=self._classic_nb_attempts,
                                               nb_clusters=self._classic_nb_clusters,
                                               nb_background_colors=self._classic_nb_background_colors,
                                               exclude_single_pixels=self._classic_exclude_single_pixels,
                                               erosion_size=self._classic_erosion_size,
                                               dilation_size=self._classic_dilation_size,
                                               box_excluding_size=self._classic_box_excluding_size,
                                               display_steps=self._classic_display,
                                               )
        relevant_boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            relevant_boxes.append((x, y, w, h))
        
        return relevant_boxes

    
    
    def _deepLearningBoxesDetection(self, image: np.ndarray) -> list[tuple[int]]:
        """
        Detect relevant parts of the image to look for objects using the deep learning method.
        
        Parameters
        ----------
        image: np.ndarray
            The image to detect the relevant parts of.
        
        Returns
        -------
        relevant_boxes: list[tuple[int]]
            The list of relevant boxes found in the image in the format (x, y, w, h).
        """
        relevant_boxes = []
        
        model = YOLO(self._ai_model_path)
        
        boxes_info_list = compute_yolo_boxes(model,
                                        image,
                                        target_classes_id = self._target_classes_id,
                                        target_classes_names = self._target_classes_names,
                                        )
        
        for box_info in boxes_info_list:
            x, y = box_info['coord']
            w_box, h_box = box_info['size']
            relevant_boxes.append((x, y, w_box, h_box))
        
        return relevant_boxes
    
    
    
    @staticmethod
    def _processSingleMatch(image: np.ndarray,
                            template: np.ndarray,
                            matching_method: Callable,
    ) -> np.ndarray:
        th, tw = template.shape[:2]
        
        # Dividing the similarity values by the number of pixels in the template to limit big templates from being
        # penalized compared to small templates
        similarity = matching_method(image, template) / ((tw * th)**(3/2))
        
        # #TODO: temporary test to add shape information to the similarity, but was not good, especially as it made the time computation longer
        # colors = np.array([(i, i, i) for i in [50, 150, 250]], dtype=np.uint8)
        # segmented_template = kmeans(template, nClusters=2, colors=colors)
        # template_borders = cv.Canny(segmented_template, 100, 200)
        # segmented_image = kmeans(image, nClusters=3, colors=colors)
        # image_borders = cv.Canny(segmented_image, 100, 200)
        # similarity += cv.matchTemplate(image_borders, template_borders, cv.TM_CCOEFF_NORMED)
        
        return similarity
    
    
    
    @staticmethod
    def _processAllMatches(image: np.ndarray,
                           base_template: np.ndarray,
                           valid_sizes: np.ndarray,
                           range_fx: np.ndarray,
                           range_fy: np.ndarray,
                           range_theta: np.ndarray,
                           single_similarity_width: int,
                           single_similarity_height: int,
                           matching_method: Callable,
                           resize_rotated: bool = True,
                           show_progress: bool = False,
                           normalizedResult: bool = True,
    ) -> np.ndarray:
        """
        Compute the map of similarity images between the original image, and the base template scaled and rotated.
        The result is a map of similarity images with the shape (lx, ly, lt, single_similarity_height, single_similarity_width).
        Most similarity images inside the map have smaller dimensions than single_similarity_height and single_similarity_width due
        to the scaling of the template and its varying size. These images are therefore padded with np.nan values.
        
        Parameters
        ----------
        image: np.ndarray
            The image to match with the template.
        base_template: np.ndarray
            The template to match with the image.
        valid_sizes: np.ndarray
            A valid_sizes table where each value tells if the corresponding template size is valid or not.
            Its shape is (lx, ly, lt).
        range_fx: np.ndarray
            The range of scaling factors in the x-dimension, of shape lx.
        range_fy: np.ndarray
            The range of scaling factors in the y-dimension, of shape ly.
        range_theta: np.ndarray
            The range of thetas in the theta-dimension, of shape lt.
        single_similarity_width: int
            The width of the similarity images.
        single_similarity_height: int
            The height of the similarity images.
        matching_method: Callable
            The method to use for the template matching.
        resize_rotated: bool
            If True, the template will be rotated without loss, by extending the image size to make it fit. Default is True.
        show_progress: bool
            If True, the progress of the computation will be shown. Default is False.
        normalizedResult: bool
            If True, the resulting similarity map will be normalized. Default is True.
        
        Returns
        -------
        similarity_map: np.ndarray
            The map of similarity images between the original image, and the base template scaled and rotated.
            Its shape is (lx, ly, lt, single_similarity_height, single_similarity_width).
        """
        
        lx = len(range_fx)
        ly = len(range_fy)
        lt = len(range_theta)

        similarity_map = np.nan * np.ones((lx, ly, lt, single_similarity_height, single_similarity_width))
        
        
        if show_progress:
            last_percent = None
            print("Beginning computation of the similarity array...")
        
        for i in range(lx):
            fx = range_fx[i]
            
            for j in range(ly):
                fy = range_fy[j]
                    
                for k in range(lt):
                    theta = range_theta[k]
                    
                    if not valid_sizes[i, j, k]:
                        continue
                    
                    template = cv.resize(base_template, None, None, fx = fx, fy = fy)
                    
                    if theta != 0:
                        if resize_rotated:
                            template = rotateImageWithoutLoss(template, theta)
                        else:
                            template = rotateImage(template, theta)
                    
                    similarity = TemplateAdvancedMatcher._processSingleMatch(image, template, matching_method)
                    
                    #* Padding
                    sim_h, sim_w = similarity.shape[:2]
                    padded_sim = np.pad(similarity,
                                        ((0, single_similarity_height - sim_h), (0, single_similarity_width - sim_w)),
                                        'constant',
                                        constant_values=np.nan)
                    similarity_map[i, j, k] = padded_sim
                    
                    if show_progress:
                        percent = round(((i * ly + j) * lt + k) / (lx * ly * lt) * 100)
                        if percent != last_percent:
                            print("Progress: {0}%".format(percent), end="\r")
                        last_percent = percent
        
        if show_progress:
            print("Progress: 100%")
            print("Similarity computations finished!")
        
        if normalizedResult:
            # Normalize the similarity map
            similarity_map = nanNormalize(similarity_map)
        
        return similarity_map
    
    @staticmethod
    def _get_depth_image_match_box(image: np.ndarray,
                                   depth_image: np.ndarray,
                                   image_matching_box: tuple[int],
                                   box_template_factor: float = 1.5,
                                   box_search_area_factor: float = 2,
                                   display_steps: bool = False,
    ) -> np.ndarray:
        """
        Matches the found box from the image in the corresponding depth image.
        This method is made so that it tries to compensate a difference of timing between the depth and color images.
        
        Parameters
        ----------
        image: np.ndarray
            The color image to match with the depth image.
        depth_image: np.ndarray
            The depth image to match with the color image.
        image_matching_box: tuple[int]
            The box from the color image to be found in the depth image, in the format (x, y, w, h).
        box_template_factor: float, optional
            The factor to multiply the box dimensions to get the dimensions of the template to match. Default is 1.5.
        box_search_area_factor: float, optional
            The factor to multiply the box dimensions to get the dimensions of the search area. Default is 2.
        display_steps: bool, optional
            If True, the steps of the process will be displayed. Default is False.
        """
        
        
        x, y, w, h = image_matching_box
        
        #* Get a larger cropped image from the depth image (30% larger than the actual match)
        template_factor = box_template_factor
        
        # Use the entire large box which will be matched in the depth image
        t_used_w = w
        t_used_h = h
        t_used_x = x
        t_used_y = y
        
        
        cropping_w_template = int(t_used_w * template_factor)
        cropping_h_template = int(t_used_h * template_factor)
        cropping_x_template = max(0, t_used_x - (cropping_w_template - t_used_w) // 2)
        cropping_y_template = max(0, t_used_y - (cropping_h_template - t_used_h) // 2)
        corresponding_template = image[cropping_y_template:cropping_y_template + cropping_h_template,
                                        cropping_x_template:cropping_x_template + cropping_w_template]
        
        
        #? If better, we can use image segmentation before determining the contours
        use_segmentation = True
        n_clusters = 5
        # 4 parts from the background + 1 from the extinguisher
        # OR: 4 parts because of the 4 corners in the background + 3 because 3 distinctive parts on the extinguisher (tube, body, head)
        n_attempts = 5
        
        if use_segmentation:
            # Uses a k-means clustering to segment the image
            # However, modify the original data so that the position of each pixel is taken into account.
            colors = np.array([
                (i, i, i) for i in np.linspace(0, 255, n_clusters)
            ], dtype=np.uint8)
            
            initial_shape = corresponding_template.shape
            width, height, channels = initial_shape
            custom_data = np.zeros((width, height, channels + 2), dtype=np.float32)
            
            custom_data[:, :, :-2] = normalize(corresponding_template)
            
            # Add normalized pixel position to the data (x / width, y / height)
            for i in range(width):
                for j in range(height):
                    custom_data[i, j, -2] = i / width
                    custom_data[i, j, -1] = j / height
            
            segmented_image = kmeans(custom_data, nClusters=n_clusters, attempts=n_attempts, colors=colors)
            corresponding_template = segmented_image
        
        corresponding_template_contours = cv.Canny(corresponding_template, 100, 200)
        
        
        #* Get a larger cropped image from the depth image (30% larger than the actual match)
        depth_factor = box_search_area_factor
        
        # Use the outer box
        d_used_w = w
        d_used_h = h
        d_used_x = x
        d_used_y = y
        
        depth_image = (normalize(depth_image.copy()) * 255).astype(np.uint8)
        
        cropping_w_depth = int(d_used_w * depth_factor)
        cropping_h_depth = int(d_used_h * depth_factor)
        cropping_x_depth = max(0, d_used_x - (cropping_w_depth - d_used_w) // 2)
        cropping_y_depth = max(0, d_used_y - (cropping_h_depth - d_used_h) // 2)
        cropped_depth_image = depth_image[cropping_y_depth:cropping_y_depth + cropping_h_depth,
                                                cropping_x_depth:cropping_x_depth + cropping_w_depth]
        
        cropped_depth_image_contours = cv.Canny(cropped_depth_image, 60, 110)
        
        sim_depth_map = cv.matchTemplate(cropped_depth_image_contours, corresponding_template_contours, cv.TM_CCOEFF_NORMED)
        
        _, box_max_val, _, box_max_loc = cv.minMaxLoc(sim_depth_map)
        box_match_x, box_match_y = box_max_loc
        
        large_box_x = cropping_x_depth + box_match_x
        large_box_y = cropping_y_depth + box_match_y
        
        
        # Get back to the original box dimensions and compute the correct box position
        found_box_x = large_box_x + (cropping_w_template - t_used_w) // 2
        found_box_y = large_box_y + (cropping_h_template - t_used_h) // 2
        
        
        
        #? Visualization
        if display_steps:
            cv.rectangle(cropped_depth_image,
                            (box_match_x, box_match_y),
                            (box_match_x + cropping_w_template, box_match_y + cropping_h_template),
                            (255, 0, 0), 2)
            
            cv.rectangle(depth_image,
                            (cropping_x_depth, cropping_y_depth),
                            (cropping_x_depth + cropping_w_depth, cropping_y_depth + cropping_h_depth),
                            (255, 0, 0), 2)
            cv.rectangle(depth_image,
                            (large_box_x, large_box_y),
                            (large_box_x + cropping_w_template, large_box_y + cropping_h_template),
                            (200, 0, 0), 2)
            
            # Reset windows before displaying since the size often changes
            try:
                cv.destroyWindow("Image part to find in the area (template)")
                cv.destroyWindow("Template contours")
                cv.destroyWindow("Area in depth image to find the image part")
                cv.destroyWindow("Area depth contours")
            except cv.error:
                # These windows were not opened yet
                pass
            
            cv.imshow(f"Image part to find in the area (template)", corresponding_template)
            cv.imshow(f"Template contours", corresponding_template_contours)
            cv.imshow(f"Area in depth image to find the image part", cropped_depth_image)
            cv.imshow(f"Area depth contours", cropped_depth_image_contours)
        
        return found_box_x, found_box_y, t_used_w, t_used_h
    
    def _get_reliability_score_on_depth(self,
                                        depth_image: np.ndarray,
                                        depth_value: float,
                                        box_center_position: tuple[int],
                                        box_size: tuple[int],
                                        size_factors: tuple[float] = None,
                                        threshold: float = None,
                                        display_points: bool = False,
    ) -> float:
        """
        Computes the reliability score of the object detected in the given box, based on the depth image.
        It is the ratio of valid pixels on the box edges compared to the total number of pixels on the box edges.
        
        Parameters
        ----------
        depth_image: np.ndarray
            The depth image to compute the reliability score from.
        depth_value: float
            The depth value of the center of the box.
        box_center_position: tuple[int]
            The center position of the box in the format (x, y).
        box_size: tuple[int], optional
            The size of the box in the format (w, h).
        size_factors: tuple[float]
            The factors to multiply the box dimensions to get the dimensions of the box to extract the depth values from the edges.
            If not provided, the class variable will be used.
        threshold: float
            The threshold above (strictly) which the depth difference makes the pixel increase the reliability score.
            If not provided, the class variable will be used.
        display_points: bool, optional
            If True, the points used to compute the reliability score will be displayed. Default is False.
        
        Returns
        -------
        reliability_score: float
            The reliability score of the object detected in the given box.
        """
        
        if size_factors is None:
            size_factors = TemplateAdvancedMatcher.DEPTH_OUTBOX_FACTORS
        
        if threshold is None:
            threshold = TemplateAdvancedMatcher.DEPTH_DIFFERENCE_THRESHOLD
        
        cbw, cbh = box_center_position
        w, h = box_size
        
        out_coeffs = np.array(size_factors) / 2  # Divide by 2 to get the coefficient for each side
        
        # Avoid getting out of the image
        top_edge = max(0, cbh - int(h * out_coeffs[1]))
        bottom_edge = min(depth_image.shape[0] - 1, cbh + int(h * out_coeffs[1]))
        left_edge = max(0, cbw - int(w * out_coeffs[0]))
        right_edge = min(depth_image.shape[1] - 1, cbw + int(w * out_coeffs[0]))
        
        pixels_to_check_width = np.array([
            [(i, top_edge) for i in range(left_edge, right_edge + 1)],
            [(i, bottom_edge) for i in range(left_edge, right_edge + 1)],
        ])
        pixels_to_check_height = np.array([
            [(left_edge, j) for j in range(top_edge, bottom_edge + 1)],
            [(right_edge, j) for j in range(top_edge, bottom_edge + 1)],
        ])
        
        #* Avoid to reach the loop limit from the uint8 dtype so convert it to int16 first
        diff_array_width = np.abs(depth_image[pixels_to_check_width[:, :, 1], pixels_to_check_width[:, :, 0]].astype(np.int16) - depth_value)
        diff_array_height = np.abs(depth_image[pixels_to_check_height[:, :, 1], pixels_to_check_height[:, :, 0]].astype(np.int16) - depth_value)
        
        #? Visualization
        if display_points:
            color_depth_im = normalize(cv.cvtColor(depth_image.copy(), cv.COLOR_GRAY2BGR))
            for i in range(2):
                for j in range(len(pixels_to_check_width[i])):
                    cv.circle(color_depth_im, pixels_to_check_width[i, j], 1, (1., 0, 0), -1)
                for j in range(len(pixels_to_check_height[i])):
                    cv.circle(color_depth_im, pixels_to_check_height[i, j], 1, (0, 0, 1.), -1)
            cv.imshow("Color depth image", color_depth_im)
            # waitNextKey(0)
        
        reliable_pixels_width = (diff_array_width > threshold)
        reliable_pixels_height = (diff_array_height > threshold)
        
        #TODO: Add weights to pixels? Corners may be more relevant than edges?
        reliability_score = (np.sum(reliable_pixels_width) + np.sum(reliable_pixels_height)) / (reliable_pixels_width.size + reliable_pixels_height.size)
        
        return reliability_score
    
    @staticmethod
    def _get_reliability_score_on_background(image: np.ndarray,
                                             base_template: np.ndarray,
                                             box_properties: tuple[int],
    ) -> float:
        """
        Compute the reliability score of the object detected in the given box, based on the background image.
        This method requires a template that supports transparency.
        It will create a custom template with a white background where the template pixels are transparent.
        Then it will compute the two similarity values with the original template and the white template, using squared difference.
        The reliability score is the ratio of the difference of scores between the original template and white template,
        compared to the maximum of the two.
        The higher it is, the more the object is reliable.
        
        Parameters
        ----------
        image: np.ndarray
            The image to compute the reliability score from.
        base_template: np.ndarray
            The template to compute the reliability score from.
        box_properties: tuple[int]
            The properties of the box in the format (x, y, w, h).
        
        Returns
        -------
        reliability_score: float
            The reliability score of the object detected in the given box.
        """
        
        template_type = ImageType.getImageType(base_template)
        if template_type != ImageType.COLOR_WITH_ALPHA:
            raise ValueError("The template must be a color image with an alpha channel.")
        
        x, y, w, h = box_properties
        cropped_image = normalize(image[y:y+h, x:x+w])
        
        base_template_shape = base_template.shape[:2]
        delta_w = w - base_template_shape[1]
        delta_h = h - base_template_shape[0]
        
        original_template = normalize(base_template)
        if delta_h != 0 or delta_w != 0:
            # Adapt the template to the cropped image size
            original_template = np.zeros((h, w, 4), dtype=np.float64)
            original_template[delta_h//2:delta_h//2 + base_template_shape[0], delta_w//2:delta_w//2 + base_template_shape[1]] = normalize(base_template)
        
        
        
        # Create the white template of the same size as the cropped image
        white_template = np.zeros((h, w, 4), dtype=np.float64)
        white_template[delta_h//2:delta_h//2 + base_template_shape[0], delta_w//2:delta_w//2 + base_template_shape[1]] = normalize(base_template)
        white_template = setTransparentPixelsTo(white_template, (1., 1., 1., 1.))
        # # ? Verification
        # cv.imshow("White template", white_template)
        # waitNextKey(0)
        
        original_difference_value = np.sum((original_template - cropped_image)**2)
        white_difference_value = np.sum((white_template - cropped_image)**2)
        
        # Higher value when the white differences are higher than the original differences
        max_value = max(original_difference_value, white_difference_value)
        reliability_score = 1 + (white_difference_value - original_difference_value) / max_value
        reliability_score /= 2 # normalized to [0 - 1]
        
        return reliability_score
        
    
    @staticmethod
    def _extractBestMatch(template_size_table: np.ndarray,
                          similarity_stats: list[float],
                          similarity_case: Case) -> tuple[float]: 
        """
        This method extracts the best match from the similarity statistics.
        It returns the best box properties, the best value and the ranges of the scaling factors and thetas corresponding to the best match.
        
        Parameters
        ----------
        template_size_table: np.ndarray
            The table of the different sizes of the template with the given scaling factors.
        similarity_stats: list[float]
            The similarity statistics.
        similarity_case: Case
            Represents the of the matching mode.
        Returns
        -------
        best_box_properties: tuple[int]
            tuple containing the best match's box properties: upper left corner position and box size.
        best_value: float
            similarity score associated to the best match.
        ranges_values: tuple[np.ndarray]
            tuple containing the ranges of the scaling factors and thetas corresponding to the best match.
        """
        min_max_value = similarity_stats[similarity_case[0]]
        min_max_index = similarity_stats[similarity_case[1]] # (fx_i, fy_j, theta_k, height, width)
        
        
        best_format = template_size_table[*min_max_index[0:3]]
        best_tw, best_th = best_format
        best_loc = min_max_index[3:][::-1] # (width, height)
        best_x, best_y = best_loc

        return (best_x, best_y, best_tw, best_th), min_max_value, min_max_index[0:3]
    
    
    #* Main methods
    def computeSimilarityMap(self,
                             image: np.ndarray,
                             base_template: np.ndarray,
                             matching_mode: MatchingMode = MatchingMode.NORMED_SQDIFF,
                             range_fx: np.ndarray = np.arange(0.6, 1.41, 0.1),
                             range_fy: np.ndarray = np.arange(0.6, 1.41, 0.1),
                             range_theta: np.ndarray = np.arange(-10, 11, 5),
                             resize_rotated: bool = True,
                             custom_matching_method: Callable = None,
                             custom_case: Case = None,
                             show_progress: bool = False,
                             normalizedResult: bool = True,
    ) -> np.ndarray:
        """
        This method compute the similarity between the image, and the base template with different scaling factors and rotation angles.
        
        Parameters
        ----------
        image: np.ndarray
            The image to match with the template.
        base_template: np.ndarray
            The template to match with the image.
        matching_mode: MatchingMode
            The matching mode of the template matching.
        range_fx: np.ndarray
            The range of the scaling factor fx.
        range_fy: np.ndarray
            The range of the scaling factor fy.
        range_theta: np.ndarray
            The range of the rotation angle theta.
        resize_rotated: bool
            If True, the rotated template will be resized so that it does not lose any pixel. Default is True.
        custom_matching_method: Callable
            A custom method to compute the similarity map. If not None, the openCV method will not be used.
            This method should be of the form: custom_matching_method(image: np.ndarray, template: np.ndarray) -> np.ndarray
        custom_case: Case
            The case to compute the similarity map. This is only used if a custom method is provided.
            It corresponds to the trustful value of the similarity map (Case.MIN/Case.MAX).
        show_progress: bool
            If True, the progress of the computation will be shown. Default is False.
        normalizedResult: bool
            If True, the resulting similarity map will be normalized. Default is True.
        
        Returns
        -------
        similarity_map: np.ndarray
            The similarity map of the image with the base template.
            
        Raises
        ------
        ValueError
            If the image is not set yet
            If the matching mode is not valid
            If the template size is too large for the image
        
        Note
        ----
        If there are not any similarity that can be computed because of non-valid sizes, the returned similarity map and stats will be filled with NaN values.
        """
        
        matching_method = None
        
        if custom_matching_method is None:
            if matching_mode not in MatchingMode:
                raise ValueError("The matching mode is not valid. Please use one of the MatchingMode objects or openC")
            else:
                matching_mode = matching_mode.value[0]

            matching_method = lambda x,y: cv.matchTemplate(x, y, matching_mode)
        else:
            matching_method = custom_matching_method
            
            if custom_case is None:
                raise ValueError("A custom method requires a case to compute the similarity map. Please provide a case from the Case enum.")


        image_h, image_w = image.shape[:2]
        template_h, template_w = base_template.shape[:2]
        
        template_size_table = None
        valid_sizes = None
        min_tw = None
        min_th = None
        
        # Avoid computing the template size table if it is already computed
        if self._original_template is not None and self._original_template is base_template:
            if self._range_fx is range_fx and self._range_fy is range_fy and self._range_theta is range_theta:
                template_size_table = self._template_size_table
                valid_sizes = self._valid_sizes
                min_tw = self._min_tw
                min_th = self._min_th
        
        if valid_sizes is None:
            template_size_table, valid_sizes, range_fx, range_fy, range_theta =\
                TemplateAdvancedMatcher._computeValidTemplateSizeTable(image_w, image_h, template_w, template_h, range_fx, range_fy, range_theta, resize_rotated)
        
        if min_tw is None:
            min_tw = np.min(template_size_table[valid_sizes, 0])
        if min_th is None:
            min_th = np.min(template_size_table[valid_sizes, 1])
        
        self._template_size_table = template_size_table
        self._valid_sizes = valid_sizes
        self._min_tw = min_tw
        self._min_th = min_th
        
        # Get max similarity map size
        max_sim_h = image_h - min_th + 1
        max_sim_w = image_w - min_tw + 1
        
        # Compute the similarity map
        similarity_map = TemplateAdvancedMatcher._processAllMatches(image,
                                                                    base_template,
                                                                    valid_sizes,
                                                                    range_fx,
                                                                    range_fy,
                                                                    range_theta,
                                                                    max_sim_w,
                                                                    max_sim_h,
                                                                    matching_method,
                                                                    resize_rotated,
                                                                    show_progress,
                                                                    normalizedResult,)
        
        similarity_stats = TemplateAdvancedMatcher.computeSimilarityStats(similarity_map)

        return similarity_map, similarity_stats
    
    
    
    def fullMatch(self,
                  image: np.ndarray,
                  base_template: np.ndarray,
                  matching_mode: MatchingMode = MatchingMode.SQDIFF,
                  range_fx: np.ndarray = np.arange(0.6, 1.41, 0.1),
                  range_fy: np.ndarray = np.arange(0.6, 1.41, 0.1),
                  range_theta: np.ndarray = np.arange(-10, 11, 5),
                  depth_image: np.ndarray = None,
                  projection_matrix: np.ndarray = None,
                  custom_matching_method: Callable = None,
                  custom_case: Case = None,
                  show_progress: bool = False,
                  display_depth_matching: bool = False,
    ):
        """"
        Match the base template on the image using pre-treatment to reduce the range of the search.
        This methods first gets the interesting rectangles of the image (i.e. object parts or template parts) and then match
        the given template in several sizes and rotation inside these boxes to get the best match.
        Ti then returns the final image, the list of best matches of each smaller parts, the similarity maps and the similarity stats.
        
        Parameters
        ----------
        image: np.ndarray
            The image to match with the template.
        base_template: np.ndarray
            The template to match with the image.
        matching_mode: MatchingMode
            The matching mode of the template matching.
        range_fx: np.ndarray
            The range of the scaling factor fx.
        range_fy: np.ndarray
            The range of the scaling factor fy.
        range_theta: np.ndarray
            The range of the rotation angle theta.
        depth_image: np.ndarray, optional
            The depth image to match with the template. If provided, the depth of each match will be printed and returned.
            Default is None.
        projection_matrix: np.ndarray, optional
            The projection matrix to compute the position of each match in the 3D space. Default is None.
            Will be ignored if no depth image is provided.
        display_depth_matching: bool, optional
            If True, the depth matching will be displayed. Default is False.
        custom_matching_method: Callable, optional
            A custom method to compute the similarity map. If not None, the openCV method will not be used.
            This method should be of the form: custom_matching_method(image: np.ndarray, template: np.ndarray) -> np.ndarray
        custom_case: Case, optional
            The case to compute the similarity map. This is only used if a custom method is provided.
            It corresponds to the trustful value of the similarity map (Case.MIN/Case.MAX).
        show_progress: bool, optional
            If True, the progress of the computation will be shown. Default is False.
        
        Returns
        -------
        final_image: np.ndarray
            The final image showing the detected elements regarding to the matches
        found_matches: list[tuple[int]]
            The list of found matches found in the image in the format (x, y, w, h).
        similarity_maps: list[np.ndarray]
            The list of non normalized similarity maps for each cropped image.
        similarity_stats: list[list[float]]
            The list of similarity statistics for each similarity map.
            
        Raises
        ------
        ValueError
            If the image is not valid
            If the template is not valid
            If the image and the template do not have the same type
            If the matching mode is not valid
            If the custom method is provided but not the custom case
        """
        
        mode = self.mode
        
        if mode not in TemplateAdvancedMatcher.AVAILABLE_MODS:
            raise ValueError("The mode is not valid. Please use one of the available modes: CLASSIC_MODE or AI_MODE.")
        
        image_type = ImageType.getImageType(image)
        if image_type is None:
            raise ValueError("The image is not valid. Please provide a valid image.")
        
        if depth_image is not None:
            depth_image_type = ImageType.getImageType(depth_image)
            
            if depth_image_type is None:
                raise ValueError("The depth image is not valid. Please provide a valid depth image.")
            
            height, width = image.shape[:2]
            depth_height, depth_width = depth_image.shape[:2]
            if height != depth_height or width != depth_width:
                raise ValueError("The depth image and the image should have the same dimensions.")
        
        
        template_type = ImageType.getImageType(base_template)
        if template_type is None:    
            raise ValueError("The template is not valid. Please provide a valid template.")
        
        if image_type != template_type:
            raise ValueError("The image and the template should have the same type.")
        
        
        if matching_mode not in MatchingMode:
            raise ValueError("The matching mode is not valid. Please use one of the available matching modes (ie. MatchingMode Enum).")
        
        if custom_matching_method is None:
            similarity_case = matching_mode.value[1].value
        else:
            if custom_case is None:
                raise ValueError("A custom method requires a case to compute the similarity map. Please provide a case from the Case enum.")
            similarity_case = custom_case.value
        
        self._original_template = base_template
        template_h, template_w = base_template.shape[:2]
        
        #* Pre-matching method: detect sub-images of the current image to do matching on
        match mode:
            case TemplateAdvancedMatcher.CLASSIC_MODE:
                relevant_boxes = self._pre_matching_method(image)
            case TemplateAdvancedMatcher.AI_MODE:
                supported_im = None
                match image_type:
                    case ImageType.COLOR_WITH_ALPHA:
                        supported_im = cv.cvtColor(image, cv.COLOR_RGBA2RGB)
                    case ImageType.GRAY_SCALE:
                        supported_im = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
                relevant_boxes = self._pre_matching_method(supported_im)
                
        
        cropped_images = []
        for box in relevant_boxes:
            x, y, w, h = box
            cropped_images.append(image[y:y+h, x:x+w]) # Numpy uses height, width while boxes are in width, height format
    
        #* Compute the template size table for the base template if it is not already computed
        
        original_template_size_table = None
        
        
        if self._original_range_fx is range_fx and self._original_range_fy is range_fy and self._original_range_theta is range_theta:
            original_template_size_table = self._original_template_size_table
        
        if original_template_size_table is None:
            self._original_range_fx = range_fx
            self._original_range_fy = range_fy
            self._original_range_theta = range_theta
            
            original_template_size_table = TemplateAdvancedMatcher._computeTemplateSizeTable(template_w, template_h, range_fx, range_fy, range_theta)
            self._original_template_size_table = original_template_size_table
        
        #* Matching on each cropped image to get similarity maps, stats and best matches
        
        matching_crop_indexes = []
        self._similarity_maps = []
        self._similarity_stats = []
        best_matches = []
        
        for i in range(len(cropped_images)):
            cropped_img = cropped_images[i]
            
            cropped_im_h, cropped_im_w = cropped_img.shape[:2]
            
            # Get the template size table for this cropped image
            template_size_table, valid_sizes, new_range_fx, new_range_fy, new_range_theta =\
                TemplateAdvancedMatcher._extractValidTemplateSizeTable(original_template_size_table, cropped_im_w, cropped_im_h, range_fx, range_fy, range_theta)
            
            self._range_fx = new_range_fx
            self._range_fy = new_range_fy
            self._range_theta = new_range_theta
            self._template_size_table = template_size_table
            self._valid_sizes = valid_sizes
            
            # Skip this cropped image if the template size table is empty
            if template_size_table.size == 0:
                continue
            
            self._min_tw = np.min(template_size_table[:, :, :, 0][valid_sizes])
            self._min_th = np.min(template_size_table[:, :, :, 1][valid_sizes])
            
            similarity_map, similarity_stats = self.computeSimilarityMap(cropped_img,
                                                                         base_template,
                                                                         matching_mode,
                                                                         new_range_fx,
                                                                         new_range_fy,
                                                                         new_range_theta,
                                                                         custom_matching_method=custom_matching_method,
                                                                         custom_case=custom_case,
                                                                         show_progress=show_progress,
                                                                         normalizedResult=False,)
            
            if similarity_map.size == 0 or np.all(np.isnan(similarity_map)):
                continue
            
            matching_crop_indexes.append(i)
            self._similarity_maps.append(similarity_map)
            self._similarity_stats.append(similarity_stats)
            
            best_match = TemplateAdvancedMatcher._extractBestMatch(template_size_table, similarity_stats, similarity_case)
            best_match_factors_indexes = best_match[2]
            best_fx = new_range_fx[best_match_factors_indexes[0]]
            best_fy = new_range_fy[best_match_factors_indexes[1]]
            best_theta = new_range_theta[best_match_factors_indexes[2]]
            
            best_matches.append((best_match[0], best_match[1], (best_fx, best_fy, best_theta))) # (box_properties, value, factors)
        
       
        #* Score the best matches and get the final image
        
        # Sorting the best matches by their score
        new_indexes = np.argsort([match[1] for match in best_matches])
        best_matches = [best_matches[i] for i in new_indexes]
        matching_crop_indexes = [matching_crop_indexes[i] for i in new_indexes]
        
        
        
        #? We should get rid of the boxes where the score is too far from the best score in order to try do keep only
        #? the boxes where there is actually an extinguisher.
        #! However, this does not work properly and we did not find a correct threshold or way to do it.
        #! The score for small black objects are always better than the one of the extinguisher.
        # if self.mode == TemplateAdvancedMatcher.CLASSIC_MODE:
        #     # Try to eliminate boxes where there are no real match
            
        #     best_score_index = 0 if similarity_case == Case.MIN else -1
        #     best_score = best_matches[best_score_index][1]
            
        #     indexes_to_keep = []
        #     if similarity_case == Case.MIN:
        #         for i in range(len(best_matches)):
        #             if best_matches[i][1] < 1.2 * best_score:
        #                 indexes_to_keep.append(i)
        #     else:
        #         for i in range(len(best_matches)):
        #             if best_matches[i][1] > 0.8 * best_score:
        #                 indexes_to_keep.append(i)
            
        #     best_matches = [best_matches[i] for i in indexes_to_keep]
        #     matching_crop_indexes = [matching_crop_indexes[i] for i in indexes_to_keep]
        
        
        # list of valid matches, where a valid match is (match_box, score) in the image coordinates
        valid_matches = []
        
        
        if depth_image is not None:
            norm_depth_image = (normalize(depth_image) * 255).astype(np.uint8)
        
        
        final_image = image.copy()
        for i in range(len(matching_crop_indexes)):
            best_match = best_matches[i]
            box = relevant_boxes[matching_crop_indexes[i]]
            
            x, y, w, h = box
            best_match_box, score, best_match_factors = best_match
            x_best, y_best, w_best, h_best = best_match_box
            
            center_box_width = x + x_best + w_best // 2
            center_box_height = y + y_best + h_best // 2
            
            
            
            final_x = x + x_best
            final_y = y + y_best
            final_w = w_best
            final_h = h_best
            final_box = (final_x, final_y, final_w, final_h)
            
            # Verify that the box is not too close to another box
            limit_ratio = TemplateAdvancedMatcher.BOX_LIMIT_RATIO # the ratio of the box size to consider as a limit (0 to 1/2)
            
            if limit_ratio != 0.5:
                points_to_check = [
                    (x + x_best + limit_ratio * w_best, y + y_best),
                    (x + x_best + (1 - limit_ratio) * w_best, y + y_best),
                    (x + x_best, y + y_best + limit_ratio * h_best),
                    (x + x_best, y + y_best + (1 - limit_ratio) * h_best),
                ]
            else:
                points_to_check = [(center_box_width, center_box_height)]
            
            valid_box = True
            
            for match in valid_matches:
                valid_box, _ = match
                
                for p in points_to_check:
                    if isPointInBox(p, valid_box):
                        # Skip this box
                        valid_box = False
                        break
                
                if not valid_box:
                    break
            
            if not valid_box:
                continue
            
            
            
            #* Box is valid
            valid_matches.append((final_box, score))
            
            
            #* Get depth and try to detect if the object is a true match
            
            depth_value = None
            valid_object = None # Undefined by default
            position2D = (center_box_width, center_box_height)
            homogeneous_position_3D = None
            
            if depth_image is not None:
                
                #* Get the adapted matching box from the depth image
                depth_box = TemplateAdvancedMatcher._get_depth_image_match_box(image,
                                                                               depth_image,
                                                                               box,
                                                                               box_template_factor=1.5,
                                                                               box_search_area_factor=2,
                                                                               display_steps=TemplateAdvancedMatcher.DISPLAY_DEPTH_MATCHING_STEPS,)
                
                depth_box_x, depth_box_y, depth_box_w, depth_box_h = depth_box
                
                if display_depth_matching:
                    cv.rectangle(norm_depth_image,
                                    (depth_box_x, depth_box_y),
                                    (depth_box_x + depth_box_w, depth_box_y + depth_box_h),
                                    (250, 250, 250), 2)
                    
                    cv.rectangle(norm_depth_image,
                                    (depth_box_x + x_best, depth_box_y + y_best),
                                    (depth_box_x + x_best + w_best, depth_box_y + y_best + h_best),
                                    (0, 0, 0), 2)
                
                
                bx = depth_box_x + x_best
                by = depth_box_y + y_best
                bw = w_best
                bh = h_best
                
                cbw = bx + bw // 2 # center of this box (width)
                cbh = by + bh // 2 # center of this box (height)
                
                position2D = (cbw, cbh)
                
                depth_value = depth_image[cbh, cbw]
                if display_depth_matching:
                    # Draw a marker on the right spot
                    cv.circle(norm_depth_image, (cbw, cbh), 3, (255, 0, 0), -1)
                
                if depth_value == 0:
                    # look pixels in 5x5 area around the center of the box and get the minimum non zero value
                    depth_area = depth_image[max(0, cbh - 2):min(depth_image.shape[0], cbh + 3),
                                             max(0, cbw - 2):min(depth_image.shape[1], cbw + 3)]
                    non_zero_values = depth_area[depth_area != 0]
                    if non_zero_values.size > 0:
                        depth_value = np.min(depth_area[depth_area != 0])
                    else:
                        depth_value = None
                
                if depth_value is not None:
                    if projection_matrix is not None:
                        homogenous_position_2D = np.array([cbw, cbh, 1]) * depth_value
                        # extract relevant matrix 3x3 from the projection matrix
                        projection_matrix = projection_matrix[:3, :3]
                        homogeneous_position_3D = np.linalg.solve(projection_matrix, homogenous_position_2D)
                        # axis are still in the same directions as on the image plane
                        # +---> x
                        # |
                        # V
                        # y
                        #* This is the same as this:
                        # K0 = projection_matrix
                        # f = K0[0, 0] # This is f/rho_w
                        # Z = depth_value
                        # X = (cbw - K0[0, 2]) * Z / f
                        # Y = (cbh - K0[1, 2]) * Z / f
                        
                    
                    
                    #? Checking reliability on depth
                    if self.reliability_mode == TemplateAdvancedMatcher.RELIABILITY_DEPTH_MODE:
                        reliability_score = self._get_reliability_score_on_depth(depth_image,
                                                                                depth_value,
                                                                                (cbw, cbh),
                                                                                (w_best, h_best),
                                                                                size_factors=self._depth_outbox_factors,
                                                                                threshold=self._depth_difference_threshold,
                                                                                display_points=self._depth_display_reliability_points,)
                        
                        # #? Used to tune thresholds
                        # DEPTH_TESTING_LIST.append(reliability_score)
                        # print("DEPTH TESTING LIST: ", DEPTH_TESTING_LIST)
                        # test_list = np.array(DEPTH_TESTING_LIST)
                        # print("Stats in the order: min, max, mean, std")
                        # print(np.min(test_list), np.max(test_list), np.mean(test_list), np.std(test_list))
                        
                        valid_object = (reliability_score >= self._depth_reliability_threshold)
                        print("Valid extinguisher: ", valid_object, " (reliability score = ", reliability_score, ")")
                        
                        #? Works quite fine, but is still not perfect due to the fact that depth maps are not aligned,
                        #? and since during the time we are behind the extinguisher, the match is not really good.
            
            
            final_point_1 = (final_x, final_y)
            final_point_2 = (final_x + final_w, final_y + final_h)
            
            cv.rectangle(final_image,
                         final_point_1,
                         final_point_2,
                         TemplateAdvancedMatcher.RECTANGLE_COLOR,
                         TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
           
            #? Checking reliability on background
            if self.reliability_mode == TemplateAdvancedMatcher.RELIABILITY_BACKGROUND_MODE:
                best_fx, best_fy, best_theta = best_match_factors
                corresponding_template = rotateImageWithoutLoss(cv.resize(base_template, None, None, fx=best_fx, fy=best_fy), best_theta)
                # build the correct box
                box_width = int(final_w * self._background_template_size_factor)
                box_height = int(final_h * self._background_template_size_factor)
                box_x = final_x - (box_width - final_w) // 2
                box_y = final_y - (box_height - final_h) // 2
                
                # Avoid box from going out of the image
                box_x = max(0, box_x)
                box_y = max(0, box_y)
                box_width = min(image.shape[1] - box_x, box_width)
                box_height = min(image.shape[0] - box_y, box_height)
                
                background_reliability_score = TemplateAdvancedMatcher._get_reliability_score_on_background(image,
                                                                                                            corresponding_template,
                                                                                                            (box_x, box_y, box_width, box_height))

                if background_reliability_score >= self._background_reliability_high_threshold:
                    valid_object = True
                elif background_reliability_score <= self._background_reliability_low_threshold:
                    valid_object = False
                else:
                    # Undetermined
                    valid_object = None
                
                # #? Used to tune thresholds
                # BACKGROUND_TESTING_LIST.append(background_reliability_score)
                # print("BACKGROUND TESTING LIST: ", BACKGROUND_TESTING_LIST)
                # back_test_list = np.array(BACKGROUND_TESTING_LIST)
                # print("Stats in the order: min, max, mean, std")
                # print(np.min(back_test_list), np.max(back_test_list), np.mean(back_test_list), np.std(back_test_list))
            
            
            
            label = "Object"
            position_label = ""
            class_color = (150, 0, 0)
            if valid_object is not None:
                valid_str = "Valid" if valid_object else "Invalid"
                label += f" ({valid_str})"
                class_color = (0, 150, 0) if valid_object else (0, 0, 150)
            if homogeneous_position_3D is not None:
                hx, hy, hz = homogeneous_position_3D
                position_label += "position: ({:.1f}, {:.1f}, {:.1f})".format(hx, hy, hz)
            
            
            cv.rectangle(final_image, (x, y), (x + w, y + h), class_color, TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
            
            #* Add text on matching boxes
            (label_width, label_height), label_baseline = cv.getTextSize(label,
                                                                 TemplateAdvancedMatcher.FONT,
                                                                 TemplateAdvancedMatcher.TEXT_SCALE,
                                                                 TemplateAdvancedMatcher.TEXT_THICKNESS)
            
            (pos_label_width, pos_label_height), pos_label_baseline = cv.getTextSize(position_label,
                                                                    TemplateAdvancedMatcher.FONT,
                                                                    TemplateAdvancedMatcher.TEXT_SCALE,
                                                                    TemplateAdvancedMatcher.TEXT_THICKNESS)
            
            final_label_width = max(label_width, pos_label_width)
            final_label_height = label_height + pos_label_height
            final_label_baseline = label_baseline + pos_label_baseline
            
            # Get the best position to put the text
            box_top_diff = y - final_label_height
            box_bottom_diff = image.shape[0] - (y + h)
            
            
            l_rect_x = x
            l_rect_w = final_label_width
            l_rect_y = None
            l_rect_h = final_label_height
            if box_bottom_diff > 0 and box_bottom_diff <= box_top_diff:
                # Display the text at the bottom of the box
                l_rect_y = y + h
            else:
                # Display the text at the top of the box
                l_rect_y = y - final_label_height
            
            
            cv.rectangle(final_image,
                         (l_rect_x, l_rect_y),
                         (l_rect_x + l_rect_w, l_rect_y + l_rect_h + final_label_baseline),
                         class_color,
                         cv.FILLED)
            # Display base label
            cv.putText(final_image, label,
                       (l_rect_x, l_rect_y + label_height),
                       TemplateAdvancedMatcher.FONT,
                       TemplateAdvancedMatcher.TEXT_SCALE,
                       TemplateAdvancedMatcher.TEXT_COLOR,
                       TemplateAdvancedMatcher.TEXT_THICKNESS)
            # Display position label
            cv.putText(final_image, position_label,
                       (l_rect_x, l_rect_y + final_label_height + label_baseline),
                       TemplateAdvancedMatcher.FONT,
                       TemplateAdvancedMatcher.TEXT_SCALE,
                       TemplateAdvancedMatcher.TEXT_COLOR,
                       TemplateAdvancedMatcher.TEXT_THICKNESS)

            # waitNextKey(0)
        
        
        if depth_image is not None and display_depth_matching:
            cv.imshow(f"Normalized Depth image", norm_depth_image)
        
        self.final_image = final_image
        
        return final_image, valid_matches, self._similarity_stats, self._similarity_maps
