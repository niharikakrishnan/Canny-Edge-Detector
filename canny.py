import numpy as np
import math
from math import degrees, pi
import cv2
import matplotlib.pyplot as plt

def convolution(image, kernel, normalisation):
	image_row, image_col = image.shape
	kernel_row, kernel_col = kernel.shape
	output = np.zeros(image.shape)

	pad_height = int((kernel_row - 1) / 2)
	pad_width = int((kernel_col - 1) / 2)
	print("pad_height; ", pad_height, "pad_width", pad_width)

	padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))
	padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

	print("padded_image", padded_image)
	for row in range(1,image_row-1):
		for col in range(1,image_col-1):
			output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
			if normalisation:
				output[row, col] /= 140
	print(output)
	return output

def gaussian_smoothing(image, kernel, display):
	
	#Checking if image is color or grayscale format and converting to grayscale format
	print(image.shape)
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		print("Converted to Gray Channel. Size : {}".format(image.shape))

	#Applying Gaussian Smoothing to the input image
	gaussian_image = convolution(image, kernel, normalisation=True)
	
	if display:
		plt.imshow(gaussian_image)
		plt.title("Output Image using {}X{} Kernel".format(kernel.shape[0], kernel.shape[1]))
		plt.show()
	return gaussian_image

def gradient_operation(image, edge_filter, convert_to_degree, display):

	horizontal_gradient = convolution(image, edge_filter, normalisation=False)
	
	if display:
		plt.imshow(horizontal_gradient)
		plt.title("Normalised horizontal gradient using Prewitt Operator")
		plt.show()

	vertical_gradient = convolution(image, np.flip(edge_filter.T, axis=0), normalisation=False)
	
	if display:
		plt.imshow(vertical_gradient)
		plt.title("Normalised vertical gradient using Prewitt Operator")
		plt.show()

	#gradient_magnitude = abs(horizontal_gradient) + abs(vertical_gradient)

	#Using the formula, gradient magnitude = Square Root of Squares of Horizontal and Vertical Gradient
	gradient_magnitude = np.sqrt(np.square(horizontal_gradient) + np.square(vertical_gradient))
	
	#Calculating gradient angle -> tan inverse (vertical gradient/horizontal gradient) in radians
	gradient_direction = np.arctan2(vertical_gradient, horizontal_gradient)
	
	#Converting gradient angle from radians to degree. Required to perform non-maxima suppression
	gradient_direction = np.rad2deg(gradient_direction)
	gradient_direction += 180

	print(gradient_direction)
	return gradient_magnitude, gradient_direction

def non_maxima_suppression(gradient_magnitude, gradient_direction, display):
	image_row, image_col = gradient_magnitude.shape
	output = np.zeros(gradient_magnitude.shape)
	sector = np.zeros(gradient_magnitude.shape)
	
	#Applying non maima suppression to all pixels other than the border
	for row in range(1, image_row - 1):
		for col in range(1, image_col - 1):
			angle = gradient_direction[row, col]

			#Mapping to Sector 0
			if (0 <= angle < 22.5) or (337.5 <= angle <= 360):
				before_pixel = gradient_magnitude[row, col - 1]
				after_pixel = gradient_magnitude[row, col + 1]

			#Mapping to Sector 1
			elif (22.5 <= angle < 67.5) or (202.5 <= angle < 247.5):
				before_pixel = gradient_magnitude[row + 1, col - 1]
				after_pixel = gradient_magnitude[row - 1, col + 1]

			#Mapping to Sector 2
			elif (67.5 <= angle < 112.5) or (247.5 <= angle < 292.5):
				before_pixel = gradient_magnitude[row - 1, col]
				after_pixel = gradient_magnitude[row + 1, col]
			
			#Mapping to Sector 3
			else:
				before_pixel = gradient_magnitude[row - 1, col - 1]
				after_pixel = gradient_magnitude[row + 1, col + 1]

			if ((gradient_magnitude[row, col] > before_pixel) and (gradient_magnitude[row, col] > after_pixel)):
				output[row, col] = gradient_magnitude[row, col]

		#print(sector[row])
	if display:
		plt.imshow(output)
		plt.title("Output Image with Non Maxima Suppression")
		plt.show()

	print(output)
	return output

def threshold(image, display):
	image_row, image_col = image.shape
	output = np.zeros(image.shape)
	
	threshold_25 = np.quantile(image.flatten(), 0.25)
	threshold_50 = np.quantile(image.flatten(), 0.50)
	threshold_75 = np.quantile(image.flatten(), 0.75)
	print(threshold_25)
	print(threshold_50)
	print(threshold_75)

	threshold = [threshold_25, threshold_50, threshold_75]

	for threshold_value in threshold:
		for row in range(image_row):
			for col in range(image_col):
				if image[row, col] < threshold_value:
					output[row, col] = 0
				else:
					output[row, col] = 255
		if display:
			plt.imshow(output)
			plt.title("Output Image with threshold")
			plt.show()

	print(output)
	return output


#if __name__ == 'main'
frame = cv2.imread('images/house.bmp')
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#image = rgb2gray(frame)
#print(image)

#image = np.array([[1,1,1,1,1,1,5],[1,1,1,1,1,5,9],[1,1,1,1,5,9,9],[1,1,1,5,9,9,9],[1,1,5,9,9,9,9],[1,5,9,9,9,9,9],[5,9,9,9,9,9,9]], dtype='int')

#Defining Prewitt's operator filter
edge_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype = 'int')

kernel = np.array([[1,1,2,2,2,1,1],[1,2,2,4,2,2,1],[2,2,4,8,4,2,2],[2,4,8,16,8,4,2],[2,2,4,8,4,2,2],[1,2,2,4,2,2,1],[1,1,2,2,2,1,1]], dtype = 'int')


gaussian_image = gaussian_smoothing(image, kernel, display=True)
gradient_magnitude, gradient_direction = gradient_operation(image, edge_filter, convert_to_degree=True, display=True)
nms_image = non_maxima_suppression(gradient_magnitude, gradient_direction, display=True)
threshold_image = threshold(nms_image, display = True)


