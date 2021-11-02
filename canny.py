import numpy as np
import math
from math import degrees, pi
import cv2
import matplotlib.pyplot as plt

frame = cv2.imread('images/house.bmp')
image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#image = rgb2gray(frame)
print(image)

#image = np.array([[1,1,1,1,1,1,5],[1,1,1,1,1,5,9],[1,1,1,1,5,9,9],[1,1,1,5,9,9,9],[1,1,5,9,9,9,9],[1,5,9,9,9,9,9],[5,9,9,9,9,9,9]], dtype='int')
gradient_x = np.array([[-1,0,1], [-1,0,1], [-1,0,1]], dtype = 'int')
gradient_y = np.array([[1,1,1], [0,0,0], [-1,-1,-1]], dtype = 'int')
threshold = 3

def calculate_gradient_magnitude(image, gradient_x, gradient_y):
	horizontal_gradient = np.zeros(image.shape)
	vertical_gradient = np.zeros(image.shape)
	image_row, image_col = image.shape
	gradient_x_row, gradient_x_col = gradient_x.shape
	gradient_y_row, gradient_y_col = gradient_y.shape

	for row in range(image_row-2):
		for col in range(image_col-2):
			horizontal_gradient[row+1,col+1] = np.sum(gradient_x * image[row:row+gradient_x_row, col:col+gradient_x_col])
			vertical_gradient[row+1,col+1] = np.sum(gradient_y * image[row:row+gradient_y_row, col:col+gradient_y_col])
	
	gradient_magnitude = abs(horizontal_gradient) + abs(vertical_gradient)
	plt.imshow(gradient_magnitude, cmap='gray')
	plt.title("threshold")
	plt.show()
	return horizontal_gradient, vertical_gradient, gradient_magnitude

def calculate_gradient_angle(horizontal_gradient, vertical_gradient):
	gradient_direction = np.arctan2(vertical_gradient, horizontal_gradient)
	gradient_angle_matrix = np.rad2deg(gradient_direction)
	return gradient_angle_matrix

def calculate_non_maxima_suppression(gradient_angle_matrix, gradient_magnitude):
	row, column = gradient_angle_matrix.shape
	print(gradient_magnitude)

	gradient_magnitude_updated = gradient_magnitude
	print(gradient_magnitude_updated)
	for i_x in range(row):
		for i_y in range(column):
			if gradient_angle_matrix[i_x,i_y]<0:
				gradient_angle = 360 + gradient_angle_matrix[i_x,i_y]
			else:
				gradient_angle = gradient_angle_matrix[i_x,i_y]

			#gradient_angle = abs(gradient_angle_matrix[i_x,i_y]-180) if abs(gradient_angle_matrix[i_x,i_y])>180 else abs(gradient_angle_matrix[i_x,i_y])
			print(gradient_angle)
			#gradient_angle = -45
			if (0 <= gradient_angle <= 22.5) or (337.5 <= gradient_angle <= 360) or (157.5 <= gradient_angle <= 202.5): #Sector 0
				neighb_1_x, neighb_1_y = i_x - 1, i_y
				neighb_2_x, neighb_2_y = i_x + 1, i_y
		             
		    # top right (diagonal-1) direction #Sector 1
			elif (22.5 <= gradient_angle <= 67.5) or (202.5 <= gradient_angle <= 247.5):
				neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
				neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
		     
		    # In y-axis direction #Sector 2
			elif ( 67.5 <=gradient_angle <= 112.5) or (247.5 <= gradient_angle <= 292.5):
				neighb_1_x, neighb_1_y = i_x, i_y - 1
				neighb_2_x, neighb_2_y = i_x, i_y + 1
		     
		    # top left (diagonal-2) direction #Sector 3
			elif (112.5 <= gradient_angle <= 157.5) or (292.5 <=gradient_angle <= 337.5):
				print("Inside sector 3")
				neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
				neighb_2_x, neighb_2_y = i_x + 1, i_y + 1
				print(gradient_magnitude[neighb_1_x,neighb_1_y])
				print(gradient_magnitude[i_x,i_y])
				print(gradient_magnitude[neighb_2_x,neighb_2_y])
				print("*******")

			#if width>neighb_1_x>= 0 and height>neighb_1_y>= 0:
			if ((gradient_magnitude[neighb_1_x,neighb_1_y]>=gradient_magnitude[i_x, i_y]) or (gradient_magnitude[neighb_2_x,neighb_2_y])>=gradient_magnitude[i_x, i_y]):
				gradient_magnitude_updated[i_x, i_y]= 0
	
	print(gradient_magnitude_updated)

horizontal_gradient, vertical_gradient, gradient_magnitude = calculate_gradient_magnitude(image, gradient_x, gradient_y)
gradient_angle_matrix = calculate_gradient_angle(horizontal_gradient, vertical_gradient)
#print(gradient_angle_matrix)
non_maxima_suppression = calculate_non_maxima_suppression(gradient_angle_matrix, gradient_magnitude)


