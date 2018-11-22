import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt




if __name__ == '__main__':
	sigma = 40.0
	iterations = 10000
	img = cv.imread('images/inpainting/corrupt_images/spectacles_corrupt.png', cv.IMREAD_COLOR)
	img = img.astype(float)
	mask = cv.imread('images/inpainting/masks/spectacles_mask.png',cv.IMREAD_GRAYSCALE)
	rows, cols, chans = img.shape
	for row in range(rows):
		for col in range(cols):
			if mask[row,col] >= 127:
				for chan in range(chans):
					img[row,col,chan]=127

	for i in range(iterations):
	#############################
		
		# cv.imshow('barbara', img)
		# cv.waitKey()
		#############################

		###########################
		# myimg = np.zeros((512, 512), dtype=np.uint8)
		# myimg[100:200, 50:60] = 255
		# gradx = cv.Sobel(myimg, cv.CV_64F, 1, 0, ksize=1)
		# grady = cv.Sobel(myimg, cv.CV_64F, 0, 1, ksize=1)

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
		

		gradx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=1)
		grady = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=1)
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
		T = np.zeros((2,2))
		#final_image=img.copy()

		for chan in range(chans):
			G[:, :, 0, 0] += gradxsq[:, :, chan]
			G[:, :, 0, 1] += gradxy[:, :, chan]
			G[:, :, 1, 0] += gradxy[:, :, chan]
			G[:, :, 1, 1] += gradysq[:, :, chan]


		## Computing Hessian Matrix of image

		H = np.zeros((rows, cols, chans, 2, 2))

		H[:, :, :, 0, 0] = cv.Sobel(gradx, cv.CV_64F, 1, 0, ksize=1)
		H[:, :, :, 0, 1] = cv.Sobel(grady, cv.CV_64F, 1, 0, ksize=1)
		H[:, :, :, 1, 0] = cv.Sobel(gradx, cv.CV_64F, 0, 1, ksize=1)
		H[:, :, :, 1, 1] = cv.Sobel(grady, cv.CV_64F, 0, 1, ksize=1)

		


		for row in range(rows):
			for col in range(cols):
				if mask[row,col] >= 127:
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
					if (1+eig_value_small[row,col]+eig_value_large[row,col])<0:
						print(eig_value_small[row,col])

		for row in range(rows):
			for col in range(cols):
				if mask[row,col] >= 127:
					c1 = 1.0*(1.0/(1+max(eig_value_large[row, col]+eig_value_small[row, col],0)))
					c2 = 1.0*(1.0/math.sqrt(1+max(eig_value_large[row, col]+eig_value_small[row, col],0)))
					T= c1*(np.reshape(eig_vector_large[row, col, :],(2,1)) @ np.reshape(np.transpose(eig_vector_large[row, col, :]),(1,2))) + c2*(np.reshape(eig_vector_small[row, col, :],(2,1))@ np.reshape(np.transpose(eig_vector_small[row, col, :]),(1,2)))
					for chan in range(chans):
						x = np.trace(T @ H[row,col,chan,:,:])
						img[row,col,chan] += x*(math.exp(-1.0*x*x/(2*sigma*sigma)))

		print(str(i)+" iterations done")
		if i%200 ==0:
			imgRGB = img.copy()
			imgRGB = imgRGB.astype(np.uint8)
			imgRGB[:, :, 0] = img[:, :, 2]
			imgRGB[:, :, 2] = img[:, :, 0]
			plt.imshow(imgRGB)
			plt.xticks([]), plt.yticks([])
			plt.savefig(str(i) + "iterations-spec.png")


		img[img < 0]=0
		img[img > 255]=255


	img = img.astype(np.uint8)
	imgRGB = img.copy()
	imgRGB[:, :, 0] = img[:, :, 2]
	imgRGB[:, :, 2] = img[:, :, 0]
	plt.imshow(imgRGB)
	plt.xticks([]), plt.yticks([])
	plt.show()







	


	######################################################

