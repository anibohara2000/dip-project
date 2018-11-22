import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':
	img = cv.imread('images/index.jpg', cv.IMREAD_COLOR)
	imgRGB = img.copy()
	imgRGB[:, :, 0] = img[:, :, 2]
	imgRGB[:, :, 2] = img[:, :, 0]
	plt.imshow(imgRGB)
	plt.xticks([]), plt.yticks([])
	plt.show()
	rows, cols, chans = img.shape
	mask = np.zeros([rows,cols])
	intensities = np.zeros(chans)
	threshold  = 1000
	for i in range(chans):
		intensities[i] = int(input())

	for i in range(rows):
		for j in range(cols):
			if sum((img[i,j,:] - intensities)*(img[i,j,:] - intensities)) <= threshold :
				mask[i,j]=255

	cv.imshow('mask', mask)
	cv.waitKey()
