import numpy as np
import math
from math import degrees, pi
import cv2
import matplotlib.pyplot as plt
import argparse
import os

def convolution(image, mask):
	
	#Getting the number of rows and columns of the image and the mask
	image_row, image_col = image.shape
	mask_row, mask_col = mask.shape
	print("Convoluting image of shape {} with kernel size of {}".format(image.shape, mask.shape))
	
	#Initialising the convoluted 2D array with zeroes
	convoluted_image = np.zeros(image.shape)

	padding_height = int((mask_row - 1) / 2)
	padding_width = int((mask_col - 1) / 2)

	padded_image = np.zeros((image_row + (2 * padding_height), image_col + (2 * padding_width)))
	padded_image[padding_height:padded_image.shape[0] - padding_height, padding_width:padded_image.shape[1] - padding_width] = image

	#Matrix multiplication - Convoluting image with kernel
	for row in range(1, image_row-1):
		for col in range(1, image_col-1):
			convoluted_image[row, col] = np.sum(mask * padded_image[row:row + mask_row, col:col + mask_col])
	
	return convoluted_image

def gaussianSmoothing(image, mask):
	print("Applying Gaussian Smoothing to input image")
	
	image_row, image_col = image.shape

	#Applying Gaussian Smoothing to the input image
	gaussian_image = convolution(image, mask)

	#Normalising the gaussian smoothened image by using the sum of total pixels
	for row in range(1, image_row-1):
		for col in range(1, image_col-1):
			if gaussian_image[row, col] > 255:
				gaussian_image[row, col] /= 140
	return gaussian_image

def gradientOperation(image, edge_filter):
	print("Computing Horizontal Gradient of the Smoothened Image.")
	image_row, image_col = image.shape

	#Computing the horizontal gradient
	horizontal_gradient = convolution(image, edge_filter)

	print("Computing Vertical Gradient.")
	#Transforming and flipping the horizontal gradient edge_filter to calculate vertical gradient
	vertical_edge_filter = np.flip(edge_filter.T, axis=0)
	vertical_gradient = convolution(image, vertical_edge_filter)

	print("Computing Gradient Magnitude.")
	#Using the formula, gradient magnitude = Square Root of Squares of Horizontal and Vertical Gradient
	gradient_magnitude = np.sqrt(np.square(horizontal_gradient) + np.square(vertical_gradient))

	#Normalising the Gradient Magnitude by 1/3 of the original value
	for row in range(1, image_row-1):
		for col in range(1, image_col-1):
			if gradient_magnitude[row, col] > 255:
				gradient_magnitude[row, col] /= 3

	print("Computing Gradient Angle.")
	#Calculating gradient angle -> tan inverse (vertical gradient/horizontal gradient) in radians
	gradient_angle = np.arctan2(vertical_gradient, horizontal_gradient)
	
	print("Computing Gradient Direction.")
	#Converting gradient angle from radians to degree. Required to perform non-maxima suppression
	gradient_direction = np.rad2deg(gradient_angle)

	return horizontal_gradient, vertical_gradient, gradient_magnitude, gradient_direction 

def nonMaximaSuppression(gradient_magnitude, gradient_direction):
	print("Applying Non Maxima Suppression")
	gradient_row, gradient_col = gradient_magnitude.shape
	
	nms_output = np.zeros(gradient_magnitude.shape)
	
	#Applying non maxima suppression to all pixels other than the border
	for row in range(1, gradient_row-1):
		for col in range(1, gradient_col-1):
			angle = gradient_direction[row, col]
			
			angle += 180

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
				nms_output[row, col] = gradient_magnitude[row, col]
	return nms_output

def simpleThresholding(image):
	print("Applying Simple Thresholding")
	
	image_row, image_col = image.shape
	thresholding = {}

	threshold_25 = np.percentile(list(set(image.flatten())), 25)
	threshold_50 = np.percentile(list(set(image.flatten())), 50)
	threshold_75 = np.percentile(list(set(image.flatten())), 75)
	threshold = {"threshold_25": threshold_25, "threshold_50": threshold_50, "threshold_75": threshold_75}

	thresholding_output = {}

	#Applying simple thresholding with threshold values at 25, 50 and 75th percentile
	for threshold_key, threshold_value in threshold.items():
		output = np.zeros(image.shape)

		for row in range(image_row):
			for col in range(image_col):
				if image[row, col] > threshold_value:
					output[row, col] = 255
		thresholding_output[threshold_key] = output
	return thresholding_output

if __name__ == '__main__':
	print("Running Canny Edge Detector!")
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True, help="Path to the image")
	args = vars(ap.parse_args())

	#Reading and opening input image and coverting to gray scale
	frame = cv2.imread(args['image'])
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#Creating path to write output images of Canny Edge Detector
	folder, fname_with_extension = os.path.split(args['image'])
	fname , extension = os.path.splitext(fname_with_extension)
	path = str(fname) + "_output"
	access = 0o755

	#Checking whether the specified output path exists or not
	isExist = os.path.exists(path)

	#Creating a new directory if it does not already exist
	if not isExist:
		os.makedirs(path,access)
		print("Output directory created.")

	#Defining 7 x 7 Gaussian mask as mentioned in the Project requirement
	mask = np.array(
        [[1, 1, 2, 2, 2, 1, 1], 
         [1, 2, 2, 4, 2, 2, 1], 
         [2, 2, 4, 8, 4, 2, 2], 
         [2, 4, 8, 16, 8, 4, 2], 
         [2, 2, 4, 8, 4, 2, 2],
         [1, 2, 2, 4, 2, 2, 1], 
         [1, 1, 2, 2, 2, 1, 1]], dtype='int')
	
	#Applying Gaussian smoothing to input image with given mask
	gaussian_image = gaussianSmoothing(image, mask)
	cv2.imwrite(path + "/"+str(fname)+"_GaussianSmoothing.bmp", gaussian_image)

	#Defining Prewitt's Edge Operator
	edge_filter = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype='int')

	#Applying gradient operation to compute horizontal gradient, vertical gradient, gradient magnitude and gradient angle
	horizontal_gradient, vertical_gradient, gradient_magnitude, gradient_direction = gradientOperation(image, edge_filter)
	cv2.imwrite(path + "/"+str(fname)+"_HorizontalGradient.bmp", horizontal_gradient)
	cv2.imwrite(path + "/"+str(fname)+"_VerticalGradient.bmp", vertical_gradient)
	cv2.imwrite(path + "/"+str(fname)+"_GradientMagnitude.bmp", gradient_magnitude)

	#Applying Non Maima Suppression to the gradient magnitude
	nonmaxima_image = nonMaximaSuppression(gradient_magnitude, gradient_direction)
	cv2.imwrite(path + "/"+str(fname)+"_NonMaximaSuppression.bmp", nonmaxima_image)

	#Applying Simple thresholding with thresholds chosen at 25th, 50th and 75th percentile
	threshold_image = simpleThresholding(nonmaxima_image)
	for threshold_name, threshold_value in threshold_image.items():
		cv2.imwrite(path + "/"+str(fname)+ "_"+threshold_name+".bmp", threshold_value)

	print("Canny Edge Detector Implemented and output images stored in the directory!")




