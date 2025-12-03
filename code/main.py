import cv2
import numpy as np
from matplotlib import pyplot as plt

original = cv2.imread('cells.png')
cells = cv2.imread('cells.png', 0)
cells = cv2.bitwise_not(cells)

cellsBlurred = cv2.GaussianBlur(cells, (21, 21), 0)
_, cellsBinary = cv2.threshold(cellsBlurred, 60, 255, cv2.THRESH_BINARY)


floodfill = cellsBinary.copy()
h, w = cellsBinary.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(floodfill, mask, (0, 0), 255)
holes = cv2.bitwise_not(floodfill)
filled_cells = cv2.bitwise_or(cellsBinary, holes)


kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
cellsOpened = cv2.morphologyEx(filled_cells,cv2.MORPH_OPEN,kernel_opening,iterations=3)


kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cellsFinal = cv2.erode(cellsOpened, kernel_erosion, iterations=4)


contours, _, = cv2.findContours(cellsFinal.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cellCount = len(contours)
cellsContours = cv2.cvtColor(cellsFinal, cv2.COLOR_GRAY2BGR)
cv2.drawContours(cellsContours, contours, -1, (0, 255, 0), 2)

plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1);plt.imshow(original, cmap="gray");plt.title("Cells (Original)")
plt.subplot(3, 3, 2);plt.imshow(cellsBlurred, cmap="gray");plt.title("Negative and Blurred Image")
plt.subplot(3, 3, 3);plt.imshow(cellsBinary, cmap="gray");plt.title("Binary Image")
plt.subplot(3, 3, 4);plt.imshow(filled_cells, cmap="gray");plt.title("FloodFill")
plt.subplot(3, 3, 5);plt.imshow(cellsOpened, cmap="gray");plt.title("Opening")
plt.subplot(3, 3, 6);plt.imshow(cellsFinal, cmap="gray");plt.title("Erosion")
plt.subplot(3, 3, 8);plt.imshow(cellsContours, cmap="gray");plt.title(f'Cell Count: {cellCount}')

plt.show()
