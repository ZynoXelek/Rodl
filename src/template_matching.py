import cv2 as cv
import numpy as np

from tool_functions import *

#? Main code

#* Read clean template with transparency
template_im = cv.imread("res/extinguisher-template.png", cv.IMREAD_UNCHANGED)
img = cv.imread("res/testing-ref.png")#"dataset/raw/test/camera_color_image_raw/camera_color_image_1727164479163392418.png")
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
cv.imshow("Testing red diff (normalized, negative)", (1 - testing_red_diff/np.max(testing_red_diff)))
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
# #* Testing something
similarity_bis = custom_similarity(img, template_im)#, display_advancement=False, show=False, show_threshold=1.01)


#* Normalize the similarity map to display it
similarity = similarity/np.max(similarity)
similarity_bis = similarity_bis/np.max(similarity_bis)
print("Normalized Similarity data: min, max, mean, median (dtype:", similarity.dtype, ")")
print(np.min(similarity), np.max(similarity), np.mean(similarity), np.median(similarity))
print("Normalized Similarity data (bis): min, max, mean, median (dtype:", similarity_bis.dtype, ")")
print(np.min(similarity_bis), np.max(similarity_bis), np.mean(similarity_bis), np.median(similarity_bis))


cv.imshow('similarity', similarity)
cv.imshow('similarity_bis', similarity_bis)

#* minVal, maxVal, minLoc, maxLoc
sim_extrema = cv.minMaxLoc(similarity)
sin_extrema_bis = cv.minMaxLoc(similarity_bis)

value = sim_extrema[mode[1][0]]
location = sim_extrema[mode[1][1]]
location_bis = sin_extrema_bis[mode[1][1]]
x, y = location
x_bis, y_bis = location_bis
print("location: ", location)

result_im = img.copy()
result_im_bis = img.copy()
cv.rectangle(result_im, (x, y), (x + wt, y + ht), (0, 255, 0), 3)
cv.rectangle(result_im_bis, (x_bis, y_bis), (x_bis + wt, y_bis + ht), (0, 255, 0), 3)

cv.imshow("Result of template matching", result_im)
cv.imshow("Result of template matching (bis)", result_im_bis)
cv.waitKey(0)
cv.destroyAllWindows()
