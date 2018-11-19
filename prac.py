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





	##################################################
	# img is available
	rows, cols, chans = img.shape

	gradx = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=1)
	grady = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=1)
	# grad = np.zeros((rows, cols, chans, 2))
	# grad[:, :, :, 0] = gradx
	# grad[:, :, :, 1] = grady 

	gradxsq = gradx * gradx
	gradysq = grady * grady
	gradxy = gradx * grady


	## Computing Structure Tensor

	G = np.zeros((rows, cols, 2, 2))

	eig_value_large = np.zeros((rows, cols))
	eig_value_small = np.zeros((rows, cols))

	eig_vector_large = np.zeros((rows, cols, 2))
	eig_vector_small = np.zeros((rows, cols, 2))

	for chan in range(chans):
		G[:, :, 0, 0] += gradxsq[:, :, chan]
		G[:, :, 0, 1] += gradxy[:, :, chan]
		G[:, :, 1, 0] += gradxy[:, :, chan]
		G[:, :, 1, 1] += gradysq[:, :, chan]


	## Computing Hessian Matrix of image

	H = np.zeros((rows, cols, chans, 2, 2))

	H[:, :, :, 0, 0] = cv.Sobel(gradx, cv.CV_16S, 1, 0, ksize=1)
	H[:, :, :, 0, 1] = cv.Sobel(grady, cv.CV_16S, 1, 0, ksize=1)
	H[:, :, :, 1, 0] = cv.Sobel(gradx, cv.CV_16S, 0, 1, ksize=1)
	H[:, :, :, 1, 1] = cv.Sobel(grady, cv.CV_16S, 0, 1, ksize=1)

	for row in range(rows):
		for col in range(cols):
			G[row, col, :, :] = cv.GaussianBlur(G[row, col, :, :], (3, 3), 0)
			eig_values, eig_vectors = np.linalg.eig(G[row, col, :, :])
			if (eig_values[0] > eig_values[1]):
				eig_value_large[row, col] = eig_values[0]
				eig_value_small[row, col] = eig_values[1]
				eig_vector_large[row, col, :] = eig_vectors[:, 0]
				eig_vector_small[row, col, :] = eig_vectors[:, 1]
			else:
				eig_value_large[row, col] = eig_values[1]
				eig_value_small[row, col] = eig_values[0]
				eig_vector_large[row, col, :] = eig_vectors[:, 1]
				eig_vector_small[row, col, :] = eig_vectors[:, 0]


	


	######################################################

