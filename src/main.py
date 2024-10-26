import cv2 as cv
import time

from tool_functions import *
from template_advanced_matcher import *


def main():
    """Main function of the program."""
    template_im = cv.imread("res/extinguisher-template.png", cv.IMREAD_UNCHANGED)
    im = cv.imread("res/testing-ref.png")
    
    # Convert to float32
    template_im = (template_im / 255.0).astype(np.float32)
    im = (im / 255.0).astype(np.float32)
    
    im = cv.cvtColor(im, cv.COLOR_BGR2BGRA)

    # Template pre-treatment
    template_im = setTransparentPixelsTo(template_im)
    template_im = crop(template_im)
    factor = 1/6 #1/6
    template_im = cv.resize(template_im, None, None, fx=factor, fy=factor)


    cv.imshow("Original template", template_im)
    waitNextKey(0)

    matcher = TemplateAdvancedMatcher(im)
    
    t = time.time()
    print("Beginning advanced matching...")
    final_im = matcher.advanceMatch(template_im, mode=MatchingMode.SQDIFF,
                                    range_fx=np.arange(0.8, 1.21, 0.1), range_fy=np.arange(0.8, 1.21, 0.1),
                                    # custom_method=custom_matching, custom_case=Case.MIN,
                                    show_progress=True,
                                    )
    
    delta_t = time.time() - t
    print("Time elapsed to compute the final image: ", delta_t)
    
    
    print("Global Statistics on the full similarity map:")
    min_val, max_val, min_index, max_index, mean_val, median_val = matcher.getSimilarityStats()
    print("Min value: ", min_val, " found at index: ", min_index)
    print("Max value: ", max_val, " found at index: ", max_index)
    print("Mean value: ", mean_val)
    print("Median value: ", median_val)
    
    
    
    
    print("Show the best match:")
    cv.imshow("Best match", final_im)
    
    waitNextKey(0)
    cv.destroyAllWindows()



if __name__ == '__main__':
	main()
