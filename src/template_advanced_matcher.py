from typing import Callable
import cv2 as cv
import numpy as np

from enum import Enum

# import os
# from ultralytics import YOLO

from tool_functions import *
# from ia_tool_functions import *

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
    #TODO: Update
    """
    Class that handles the advanced template matching.
    
    Attributes
    ----------
    final_image: np.ndarray
        final image showing the detected elements regarding to the matches
    """
    
    #* Constants
    
    RECTANGLE_COLOR = (0, 255, 0)
    RECTANGLE_THICKNESS = 1
    
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
    
    # --- AI ---
    
    AI_MODEL_FOLDER = 'res/train_models/training_model1/'
    AI_MODEL_FILE = 'weights/best.pt'
    
    #* Constructor
    
    def __init__(self, mode: int = CLASSIC_MODE) -> None:
        """
        Constructor for a template advanced matcher object with the given mode.
        
        Parameters
        ----------
        mode: int
            The mode of the template matching. It can either be CLASSIC_MODE or AI_MODE.
            Defaults to CLASSIC_MODE.
        
        Raises
        ------
        ValueError: If the mode is not valid.
        """
        if mode not in TemplateAdvancedMatcher.AVAILABLE_MODS:
            raise ValueError("The mode is not valid. Please use one of the available modes: CLASSIC_MODE or AI_MODE.")
        
        self.reset()
        self.setMode(mode)
    
    
    
    #* Getters and setters
    
    def setMode(self,
                mode: int = CLASSIC_MODE,
                *args, **kwargs,
    ) -> None:
        """
        Setter for the object-detection mode.
        You can additionally set the mode variables with the given arguments and keyword arguments.
        You can also do so by calling the setClassicModeVariables() method afterwards.
        
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
                self._pre_matching_method = TemplateAdvancedMatcher._deepLearningBoxesDetection
    
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
    
    
    
    def setClassicModeVariables(self,
                                nb_attempts: int = None,
                                nb_clusters: int = None,
                                nb_background_colors: int = None,
                                exclude_single_pixels: bool = None,
                                erosion_size: int = None,
                                dilation_size: int = None,
                                box_excluding_size: tuple[int] = None,
    ) -> None:
        """
        Sets the variables for the classic mode of the template matching.
        If a variable is not provided, the value will not change.
        """
        self._classic_nb_attempts = nb_attempts if nb_attempts is not None else self._classic_nb_attempts
        self._classic_nb_clusters = nb_clusters if nb_clusters is not None else self._classic_nb_clusters
        self._classic_nb_background_colors = nb_background_colors if nb_background_colors is not None else self._classic_nb_background_colors
        self._classic_exclude_single_pixels = exclude_single_pixels if exclude_single_pixels is not None else self._classic_exclude_single_pixels
        self._classic_erosion_size = erosion_size if erosion_size is not None else self._classic_erosion_size
        self._classic_dilation_size = dilation_size if dilation_size is not None else self._classic_dilation_size
        self._classic_box_excluding_size = box_excluding_size if box_excluding_size is not None else self._classic_box_excluding_size
    
    
    
    def getSimilarityMap(self) -> np.ndarray:
        """
        Getter for the similarity map.
        
        Returns
        -------
        similarity_map: np.ndarray
            The similarity map.
        """
        return self._similarity_map
    
    
    def getSimilarityStats(self) -> list[float]:
        """
        Getter for the similarity statistics.
        It is ordered this way: [min_val, max_val, min_index, max_index, mean_val, median_val]
        
        Returns
        -------
        similarity_stats: list[float]
            The similarity statistics.
        """
        return self._similarity_stats
    
    
    
    
    
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
                template_size_table[:, :, k] = computeRotatedRequiredSize(base_sizes, theta)
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
        
        min_val = np.min(similarity_map)
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
                                               )
        relevant_boxes = []
        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            relevant_boxes.append((x, y, w, h))
        
        #TODO: temporary
        boxes_im = image.copy()
        for box in relevant_boxes:
            x, y, w, h = box
            cv.rectangle(boxes_im, (x, y), (x + w, y + h), TemplateAdvancedMatcher.RECTANGLE_COLOR, TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
        cv.imshow("Boxes", boxes_im)
        
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
        raise NotImplementedError("The deep learning method is not implemented yet.")
        # relevant_boxes = []
        
        # model_path = os.path.join(TemplateAdvancedMatcher.AI_MODEL_FOLDER, TemplateAdvancedMatcher.AI_MODEL_FILE)
        
        # model = YOLO(model_path)
        
        # boxes_info_list = compute_yolo_boxes(model,
        #                                 image,
        #                                 target_classes_id = [0],
        #                                 target_classes_names = ['extinguisher'],
        #                                 show_result = True,
        #                                 )
        
        # for box_in boxes_info_list:
        #     x, y = box_info['coord']
        #     w_box, h_box = box_info['size']
        #     relevant_boxes.append((x, y, w_box, h_box))
        
        # return relevant_boxes
    
    
    
    @staticmethod
    def _processSingleMatch(image: np.ndarray,
                            template: np.ndarray,
                            matching_method: Callable,
    ) -> np.ndarray:
        th, tw = template.shape[:2]
        
        # Dividing the similarity values by the number of pixels in the template to limit big templates from being
        # penalized compared to small templates
        similarity = matching_method(image, template) / (tw * th) #TODO: Division to be removed?
        
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
    ) -> np.ndarray:
        #TODO: doc
        """
        
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
                    # #TODO: temporary
                    # print("Image n° {0}/{1} with fx = {2} and fy = {3}".format(i * len(range_fy) + j + 1, lx * ly, fx, fy))
                    
                    if not valid_sizes[i, j, k]:
                        continue
                    
                    template = cv.resize(base_template, None, None, fx = fx, fy = fy)
                    
                    if theta != 0:
                        if resize_rotated:
                            template = rotateImageWithoutLoss(template, theta)
                        else:
                            template = rotateImage(template, theta)
                    
                    similarity = TemplateAdvancedMatcher._processSingleMatch(image, template, matching_method)
                    
                    # sim_extrema = cv.minMaxLoc(similarity)
            
                    # value = sim_extrema[similarity_case[0]]
                    # location = sim_extrema[similarity_case[1]]
                    # x, y = location
                    
                    #* Padding
                    sim_h, sim_w = similarity.shape[:2]
                    padded_sim = np.pad(similarity,
                                        ((0, single_similarity_height - sim_h), (0, single_similarity_width - sim_w)),
                                        'constant',
                                        constant_values=np.nan)
                    similarity_map[i, j, k] = padded_sim

                    # #TODO: Temporary
                    # temp_im = image.copy()
                    # cv.imshow('similarity', similarity.copy()/np.max(similarity))
                    # cv.rectangle(temp_im, location, (x + tw, y + th), TemplateAdvancedMatcher.RECTANGLE_COLOR, TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
                    # cv.imshow('Result', temp_im)
                    
                    # waitNextKey(10)
                    
                    if show_progress:
                        percent = round(((i * ly + j) * lt + k) / (lx * ly * lt) * 100)
                        if percent != last_percent:
                            print("Progress: {0}%".format(percent), end="\r")
                        last_percent = percent
        
        if show_progress:
            print("Progress: 100%")
            print("Similarity computations finished!")
        
        #TODO: Remove the normalization? What about the threshold factor and the score?
        # Normalize the similarity map
        similarity_map = nanNormalize(similarity_map)
        
        return similarity_map
    
    @staticmethod
    def _extractBestMatch(template_size_table: np.ndarray,
                          similarity_stats: list[float],
                          similarity_case: Case) -> tuple[float]: 
        """
        This method compute the similarity between the image, and the base template with different scaling factors and rotation angles.
        
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
        """
        
        min_max_value = similarity_stats[similarity_case[0]]
        min_max_index = similarity_stats[similarity_case[1]] # (fx_i, fy_j, theta_k, height, width)
        
        
        best_format = template_size_table[*min_max_index[0:3]]
        best_tw, best_th = best_format
        best_loc = min_max_index[3:][::-1] # (width, height)
        best_x, best_y = best_loc

        return (best_x, best_y, best_tw, best_th), min_max_value
        
    
    
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
        
        #TODO: verif on sizes, shapes, etc...
        
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
                                                                    show_progress)
        
        similarity_stats = TemplateAdvancedMatcher.computeSimilarityStats(similarity_map)

        return similarity_map, similarity_stats
    
    
    
    def fullMatch(self,
                  image: np.ndarray,
                  base_template: np.ndarray,
                  matching_mode: MatchingMode = MatchingMode.SQDIFF,
                  range_fx: np.ndarray = np.arange(0.6, 1.41, 0.1),
                  range_fy: np.ndarray = np.arange(0.6, 1.41, 0.1),
                  range_theta: np.ndarray = np.arange(-10, 11, 5),
                  custom_matching_method: Callable = None,
                  custom_case: Case = None,
                  show_progress: bool = False,
    ):
        #TODO: doc
        """"
        
        """
        
        mode = self.mode
        
        if mode not in TemplateAdvancedMatcher.AVAILABLE_MODS:
            raise ValueError("The mode is not valid. Please use one of the available modes: CLASSIC_MODE or AI_MODE.")
        
        image_type = ImageType.getImageType(image)
        if image_type is None:
            raise ValueError("The image is not valid. Please provide a valid image.")
        
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
        relevant_boxes = self._pre_matching_method(image)
        
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
        
        matching_crop_index = []
        self._similarity_maps = []
        self._similarity_stats = []
        best_matches = []
        
        #TODO: temporary
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
                                                                         show_progress=show_progress)
            
            if similarity_map.size == 0 or np.all(np.isnan(similarity_map)):
                continue
            
            matching_crop_index.append(i)
            self._similarity_maps.append(similarity_map)
            self._similarity_stats.append(similarity_stats)
            best_matches.append(TemplateAdvancedMatcher._extractBestMatch(template_size_table, similarity_stats, similarity_case))
        
        #* Score the best matches and get the final image
        #TODO: temp
        print("Matches found on ", len(best_matches), " images.")
        
        final_image = image.copy()
        for i in range(len(matching_crop_index)):
            best_match = best_matches[i]
            box = relevant_boxes[matching_crop_index[i]]
            
            x, y, w, h = box
            best_match_box, score = best_match
            x_best, y_best, w_best, h_best = best_match_box
            cv.rectangle(final_image,
                         (x + x_best, y + y_best),
                         (x + x_best + w_best, y + y_best + h_best),
                         TemplateAdvancedMatcher.RECTANGLE_COLOR,
                         TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
            #TODO: Temporary
            cv.rectangle(final_image, (x, y), (x + w, y + h), (0, 0, 255), TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
        
        self.final_image = final_image
        
        return final_image#, best_matches, self._similarity_stats, self._similarity_maps
    
    
    #TODO:
    # Je propose:
    # - On enlève l'image de l'objet, ainsi que quasi tous les attributs.
    # - Toutes les méthodes déjà crées là peuvent devenir statiques, et on vire les get et self.
    # - On ajoute une méthode d'instance qui sera 'all-in' et qui prendra tous les paramètres nécessaires: la vidéo en entière ou chaque image.
    # - Elle traite le tout et stocke cette fois dans l'objets les résultats uniquement.
    # - On y récupèrera les images finales, les stats, les scores de confiance, etc...
    # Du coup soit on fait un traitement à postériori avec tout d'un coup, soit au frame par frame à voir.
    # Et du coup on devra sans doutes garder quelques attributs privés pour ne pas tout recalculer à chaque fois. Mais ils seront cachés de l'utilisateur quoi.
    # Je pense que dans l'idée, la méthode ne renvoie que les données, pas l'image finale.
    # Et on a une fonction statique qui prend exactement (ou partiellement) ces données là et qui construit l'image finale pour l'affichage.
    # Avec peut-être d'autre fonctions statiques pour reconstruire les images des étapes intermédiaires si on veut voir ce qu'il se passe.
