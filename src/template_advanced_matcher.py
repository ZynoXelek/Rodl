from typing import Callable
import cv2 as cv
import numpy as np

from enum import Enum

from tool_functions import *


class Case(Enum):
    """
    CASES are indexes of min/max value and min/max location in the similarity stats list.
    If you are using a custom method for the advanced template matching, you should provide the corresponding case, which means that
    the trustful value is the min/max value of the resulting similarity map.
    """
    MIN = 0, 2
    MAX = 1, 3



class Mode(Enum):
    """Mode of the template matching"""
    SQDIFF = cv.TM_SQDIFF, Case.MIN
    NORMED_SQDIFF = cv.TM_SQDIFF_NORMED, Case.MIN
    CCORR = cv.TM_CCORR, Case.MAX
    NORMED_CCORR = cv.TM_CCORR_NORMED, Case.MAX
    CCOEFF = cv.TM_CCOEFF, Case.MAX
    NORMED_CCOEFF = cv.TM_CCOEFF_NORMED, Case.MAX



class TemplateAdvancedMatcher():
    """
    Class that handles the advanced template matching.
    
    Attributes
    ----------
    similarity_map: np.ndarray
        map of similarities for the different rescaled templates
    final_image: np.ndarray
        final image showing the detected elements regarding to the matches
    """
    
    RECTANGLE_COLOR = (0, 255, 0)
    RECTANGLE_THICKNESS = 1
    
    
    #* Constructor
    
    def __init__(self, image: np.ndarray) -> None:
        self.setImage(image)
    
    
    
    
    
    #* Getters and setters
    
    def setImage(self, image: np.ndarray) -> None:
        """
        Setter for the image.
        
        Parameters
        ----------
        image: np.ndarray
            The image to set.
        """
        self.reset()
        
        self._image = image.copy()
    
    
    def reset(self) -> None:
        self._image = None
        
        # Reset private attributes for future template matching
        self._original_template = None
        self._mode = None
        self._range_fx = None
        self._range_fy = None
        self._template_size_table = None
        self._similarity_stats = [None] * 6 # [min_val, max_val, min_index, max_index, mean_val, median_val]
        
        self.similarity_map = None
        self.final_image = None
        
    
    
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
            The similarity statistics
        """
        return self._similarity_stats
    
    
    
    
    
    #* Private helper methods
    
    def _getTemplateSizeTable(self,
                              base_template: np.ndarray,
                              range_fx: np.ndarray,
                              range_fy: np.ndarray,
    ) -> np.ndarray:
        """
        This method returns the table of the different sizes of the template with the given scaling factors.
        
        Parameters
        -----------
        base_template: np.ndarray
            The template to match with the image.
        range_fx: np.ndarray
            The range of the scaling factor fx.
        range_fy: np.ndarray
            The range of the scaling factor fy.
        
        Returns
        -------
        template_size_table: np.ndarray
            The table of the different sizes of the template with the given scaling factors.
        """
        
        lx = len(range_fx)
        ly = len(range_fy)
        
        template_size_table = np.zeros((lx, ly, 2), dtype=np.uint32)
        
        base_template_h, base_template_w = base_template.shape[:2]
        
        # Create size table with the two given ranges
        factor_table = np.zeros((lx, ly, 2))
        for i in range(lx):
            factor_table[i, :, 0] = range_fx
        for j in range(ly):
            factor_table[:, j, 1] = range_fy
        
        template_size_table = np.zeros((lx, ly, 2), dtype=np.uint32)
        template_size_table[:, :, 0] = np.round(base_template_w * factor_table[:, :, 0], decimals=0)
        template_size_table[:, :, 1] = np.round(base_template_h * factor_table[:, :, 1], decimals=0)
        
        return template_size_table
    
    
    
    def _computeSimilarityStats(self) -> list[float]:
        """
        This method computes the statistics of the similarity map found by the `advance_match()` method.
        
        Returns
        -------
        similarity_stats: list[float]
            The similarity statistics with the following order: [min_val, max_val, min_index, max_index, mean_val, median_val]
        """
        
        sim_map = self._similarity_map
        if sim_map is None:
            raise ValueError("The similarity map is not computed yet. Please run the `advance_match()` method before.")
        
        min_val = np.nanmin(sim_map)
        min_index = np.nanargmin(sim_map)
        min_index = np.unravel_index(min_index, sim_map.shape)

        max_val = np.nanmax(sim_map)
        max_index = np.nanargmax(sim_map)
        max_index = np.unravel_index(max_index, sim_map.shape)
        
        mean_val = np.nanmean(sim_map)
        median_val = np.nanmedian(sim_map)
        
        self._similarity_stats = [min_val, max_val, min_index, max_index, mean_val, median_val]
        
        return self._similarity_stats
    
    
    
    
    
    
    #* Main methods

    def advanceMatch(self,
                    base_template: np.ndarray,
                    mode: Mode = Mode.NORMED_SQDIFF,
                    range_fx: np.ndarray = np.arange(0.6, 1.41, 0.1),
                    range_fy: np.ndarray = np.arange(0.6, 1.41, 0.1),
                    # rotation_range: np.ndarray = np.arange(-30, 30, 5),
                    custom_method: Callable = None,
                    custom_case: Case = None,
                    show_progress: bool = False,
    ) -> np.ndarray:
        """
        This method compute the similarity between the image that has previously been given to the matcher,
        and the given base template with different scaling factors.
        It then set the matcher attributes to the correct values.
        Finally, it returns the final image with the best matches.
        
        Parameters
        ----------
        base_template: np.ndarray
            The template to match with the image.
        mode: Mode
            The mode of the template matching.
        range_fx: np.ndarray
            The range of the scaling factor fx.
        range_fy: np.ndarray
            The range of the scaling factor fy.
        custom_method: Callable
            A custom method to compute the similarity map. If not None, the openCV method will not be used.
            This method should be of the form: custom_method(image: np.ndarray, template: np.ndarray) -> np.ndarray
        custom_case: Case
            The case to compute the similarity map. This is only used if a custom method is provided.
            It corresponds to the trustful value of the similarity map (Case.MIN/Case.MAX).
        show_progress: bool
            If True, the progress of the computation will be shown. Default is False.
        
        Returns
        -------
        best_image: np.ndarray
            The final image with the best matches.
            
        Raises
        ------
            ValueError: If the image is not set yet.
        """
        
        #TODO: verif on sizes, shapes, etc...
        
        if self._image is None:
            raise ValueError("The image is not set yet. Please set the image before running the `advance_match()` method.")
        
        matching_method = None
        similarity_case = None
        
        if custom_method is None:
            if not isinstance(mode, Mode):
                match mode:
                    case cv.TM_SQDIFF:
                        similarity_case = Case.MIN.value
                    case cv.TM_SQDIFF_NORMED:
                        similarity_case = Case.MIN.value
                    case cv.TM_CCORR:
                        similarity_case = Case.MAX.value
                    case cv.TM_CCORR_NORMED:
                        similarity_case = Case.MAX.value
                    case cv.TM_CCOEFF:
                        similarity_case = Case.MAX.value
                    case cv.TM_CCOEFF_NORMED:
                        similarity_case = Case.MAX.value
                    case _:
                        raise ValueError("The mode is not valid. Please use one of the Mode objects or openC")
            else:
                mode_val = mode.value
                mode = mode_val[0]
                similarity_case = mode_val[1].value

            matching_method = lambda x,y: cv.matchTemplate(x, y, mode)
        else:
            matching_method = custom_method
            
            if custom_case is None:
                raise ValueError("A custom method requires a case to compute the similarity map. Please provide a case from the Case enum.")
            
            similarity_case = custom_case.value

            

        im = self._image
        
        lx = len(range_fx)
        ly = len(range_fy)
        
        image_h, image_w = self._image.shape[:2]
        base_template_h, base_template_w = base_template.shape[:2]
        
        template_size_table = self._getTemplateSizeTable(base_template, range_fx, range_fy)
        
        # Check if every size is correct
        min_tw = np.inf
        min_th = np.inf
        
        for i in range(lx):
            for j in range(ly):
                tw, th = template_size_table[i, j]
                min_tw, min_th = min(min_tw, tw), min(min_th, th)
                
                if tw > image_w or th > image_h:
                    err_mess = "Template size is too large for the image with scaling factors fx = {0} and fy = {1}.".format(range_fx[i], range_fy[j])
                    err_mess += "(Original template size: {0}x{1}, Image size: {2}x{3})".format(base_template_w, base_template_h, image_w, image_h)
                    raise ValueError(err_mess)
        
        
        self._original_template = base_template
        self._mode = mode
        self._range_fx = range_fx
        self._range_fy = range_fy
        self._template_size_table = template_size_table
        
        # Get max similarity map size
        max_sim_h = image_h - min_th + 1
        max_sim_w = image_w - min_tw + 1

        similarity_map = np.nan * np.ones((lx, ly, max_sim_h, max_sim_w))
        
        
        if show_progress:
            last_percent = None
            print("Beginning computation of the similarity array...")
        for i in range(lx):
            fx = range_fx[i]
            
            for j in range(ly):
                fy = range_fy[j]
                
                # #TODO: temporary
                # print("Image n° {0}/{1} with fx = {2} and fy = {3}".format(i * len(range_fy) + j + 1, lx * ly, fx, fy))
                
                template = cv.resize(base_template, None, None, fx = fx, fy = fy)
                #TODO: temporary
                # cv.imshow('template', template)
                th, tw = template.shape[:2]
                
                # Dividing the similarity values by the number of pixels in the template to limit big templates from being
                # penalized compared to small templates
                similarity = matching_method(im, template) / (tw * th) #TODO: Division to be removed?
                sim_extrema = cv.minMaxLoc(similarity)
                
                value = sim_extrema[similarity_case[0]]
                location = sim_extrema[similarity_case[1]]
                x, y = location
                
                sim_h, sim_w = similarity.shape[:2]
                padded_sim = np.pad(similarity, ((0, max_sim_h - sim_h), (0, max_sim_w - sim_w)), 'constant', constant_values=np.nan)
                similarity_map[i, j] = padded_sim

                # #TODO: Temporary
                # temp_im = im.copy()
                # cv.imshow('similarity', similarity.copy()/np.max(similarity))
                # cv.rectangle(temp_im, location, (x + tw, y + th), TemplateAdvancedMatcher.RECTANGLE_COLOR, TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
                # cv.imshow('Result', temp_im)
                
                # waitNextKey(10)
                
                if show_progress:
                    percent = round((i * ly + j + 1) / (lx * ly) * 100)
                    if percent != last_percent:
                        print("Progress: {0}%".format(percent), end="\r")
                    last_percent = percent
        
        if show_progress:
            print("Progress: 100%")
            print("Similarity computations finished!")
        
        # Normalize the similarity map
        similarity_map = nannormalize(similarity_map)
        
        self._similarity_map = similarity_map
        
        similarity_stats = self._computeSimilarityStats()

        min_max_index = similarity_stats[similarity_case[1]] # (i_fx, j_fy, height, width)
        
        best_format = template_size_table[*min_max_index[0:2]]
        best_tw, best_th = best_format
        best_loc = min_max_index[2:][::-1] # (width, height)
        best_x, best_y = best_loc

        best_im = im.copy()
        cv.rectangle(best_im, best_loc, (best_x + best_tw, best_y + best_th), TemplateAdvancedMatcher.RECTANGLE_COLOR, TemplateAdvancedMatcher.RECTANGLE_THICKNESS)
        self._best_im = best_im
        
        return best_im