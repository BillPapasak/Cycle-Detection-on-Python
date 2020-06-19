import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt
import sys
from scipy.misc import imsave
from collections import defaultdict
import math 
import time


class sobelEdgeDetector:

	def __init__(self, imagePath, threshold):
		self.imagePath = imagePath
		self.threshold = threshold
	
	def openImage(self):
		return Image.open(self.imagePath)
	
	def saveImage(self, image):
		
		return imsave("edge_map_of "+self.imagePath, image)

	def createNewImage(self, rows, columns):
		return np.zeros((rows, columns))
	
	
	def getAverageGrey(self, pixel):
		return int((int(pixel[0])+int(pixel[1])+int(pixel[2]))/3)

	def rgbToGrayscale(self, image):
		rows, columns = image.size
		image = np.array(image)
		newImage = self.createNewImage(columns, rows)
		for column in range(columns):
			for row in range(rows):
				newImage[column][row] = self.getAverageGrey(image[column][row])
		return newImage

	def padImage(self, image):
		return np.pad(image, (1,1), "edge")
	
	def sobelImageGradient(self, image):
		rows, columns = image.shape
		xDirectionKernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
		yDirectionKernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
		imageGradientX = np.zeros((rows, columns))
		imageGradientY = np.zeros((rows, columns))
		imageDirection = np.zeros((rows, columns))
		paddedImage = self.padImage(image)
		rows, columns = paddedImage.shape
		for i in range(1, rows-1):
			for j in range(1, columns-1):

				imageGradientX[i-1][j-1] = (xDirectionKernel[0][0] * paddedImage[i-1][j-1]) + (xDirectionKernel[0][1] * paddedImage[i-1][j]) + (xDirectionKernel[0][2] * paddedImage[i-1][j+1]) +\
					(xDirectionKernel[1][0] * paddedImage[i][j-1])   + (xDirectionKernel[1][1] * paddedImage[i][j])   + (xDirectionKernel[1][2] * paddedImage[i][j+1]) +\
					(xDirectionKernel[2][0] * paddedImage[i+1][j-1]) + (xDirectionKernel[2][1] * paddedImage[i+1][j]) + (xDirectionKernel[2][2] * paddedImage[i+1][j+1])

				imageGradientY[i-1][j-1] = (yDirectionKernel[0][0] * paddedImage[i-1][j-1]) + (yDirectionKernel[0][1] * paddedImage[i-1][j]) + (yDirectionKernel[0][2] * paddedImage[i-1][j+1]) +\
					(yDirectionKernel[1][0] * paddedImage[i][j-1])   + (yDirectionKernel[1][1] * paddedImage[i][j])   + (yDirectionKernel[1][2] * paddedImage[i][j+1]) +\
					(yDirectionKernel[2][0] * paddedImage[i+1][j-1]) + (yDirectionKernel[2][1] * paddedImage[i+1][j]) + (yDirectionKernel[2][2] * paddedImage[i+1][j+1])
				imageDirection[i-1][j-1] = math.atan2(imageGradientX[i-1][j-1], imageGradientY[i-1][j-1])
		
		imX = imageGradientX * imageGradientX
		imY = imageGradientY * imageGradientY
		imageGradient = np.sqrt(imX + imY)
		imageGradient *= 255.0 / np.amax(imageGradient)
		#self.plotImage(imageGradient)
		imageGradient = self.thinningEdges(imageGradient, imageDirection)

		return imageGradient
				
	def thresholdingSobelGradient(self, image):
		maxIdensity = np.amax(image)
		rows, columns = image.shape
		edgeMapImage = np.zeros((rows, columns))
		image[image>maxIdensity*np.float64(self.threshold)] = 255
		image[image<maxIdensity*np.float64(self.threshold)] = 0
	
		return image
					
	def plotImage(self, image):
		plt.imshow(image, cmap = "gray")
		plt.show()
	
									
	def thinningEdges(self, imageGradient, imageDirection):
		rows, columns = imageGradient.shape
		finalImageGradient = np.copy(imageGradient)
		for row in range(1, rows-1):
			for column in range(1, columns-1):
				if imageDirection[row][column] >= 0:
					angle = imageDirection[row][column]
				else:
					angle = imageDirection[row][column] + math.pi
				roundAngle = round(angle/ (math.pi/4))
				p = imageGradient[row][column]
				if ((roundAngle == 0 or roundAngle == 4) and (imageGradient[row-1][column] > p or imageGradient[row+1][column] > p)
					or (roundAngle == 1 and (imageGradient[row-1][column-1] > p or imageGradient[row+1][column+1] > p))
					or (roundAngle == 2 and (imageGradient[row][column-1] > p or imageGradient[row][column+1] > p))
					or (roundAngle == 3 and (imageGradient[row+1][column-1] > p or imageGradient[row-1][column+1] > p))):
					finalImageGradient[row][column] = 0
		return finalImageGradient
					
	
	def main(self):
		image = self.openImage()
		grayscaleImage = self.rgbToGrayscale(image)
		imageEdgeMap = self.thresholdingSobelGradient(self.sobelImageGradient(grayscaleImage))
		self.saveImage(imageEdgeMap)
		#self.plotImage(imageEdgeMap)
		return imageEdgeMap	








	
sobel = sobelEdgeDetector(sys.argv[1], sys.argv[2])
sobel.main()
	

