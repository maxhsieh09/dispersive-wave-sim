import numpy as np
import cv2

def fold_reflections(wave: np.ndarray):
    size = wave.shape[0] // 2, wave.shape[1] // 2

    # fold the first axis
    folded = wave[:size[0], :] + wave[-1:size[0]-1:-1, :]
    # fold the second axis
    folded = folded[:, :size[1]] + folded[:, -1:size[1]-1:-1]

    return folded

image = cv2.imread("test_image.jpg", cv2.IMREAD_GRAYSCALE)
folded = fold_reflections(image)
cv2.imshow("folded", folded)
cv2.waitKey(0)
