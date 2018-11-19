import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

if __name__ == '__main__':

	#############################
	img = cv.imread('images/barbara.jpg', cv.IMREAD_COLOR)
	# cv.imshow('barbara', img)
	# cv.waitKey()
	#############################

	###########################
	# myimg = np.zeros((512, 512), dtype=np.uint8)
	# myimg[100:200, 50:60] = 255
	# gradx = cv.Sobel(myimg, cv.CV_16S, 1, 0, ksize=1)
	# grady = cv.Sobel(myimg, cv.CV_16S, 0, 1, ksize=1)

	# plt.subplot(1, 3, 1), plt.imshow(myimg, cmap='gray')
	# plt.title('Original'), plt.xticks([]), plt.yticks([])
	# plt.subplot(1, 3, 2), plt.imshow(gradx, cmap='gray')
	# plt.title('X grad'), plt.xticks([]), plt.yticks([])
	# plt.subplot(1, 3, 3), plt.imshow(grady, cmap='gray')
	# plt.title('Y grad'), plt.xticks([]), plt.yticks([])
	# plt.show()
	############################

	# img is available
	rows, cols, chans = img.shape

	gradx = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=1)
	grady = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=1)
	grad = np.zeros((rows, cols, chans, 2))
	grad[:, :, :, 0] = gradx
	grad[:, :, :, 1] = grady 

	G = np.zeros((rows, cols, 2, 2))
	for row in range(rows):
		for col in range(cols):
			for chan in range(chans):
				i_grad = grad[row, col, chan, :].reshape(2, 1)
				# print(i_grad.shape)
				G[row, col, :, :] += i_grad @ i_grad.T


