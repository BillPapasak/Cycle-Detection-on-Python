from sobelFilter import sobelEdgeDetector
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

class cycleDetector:
	def __init__(self, imagePath, threshold):
		self.imagePath = imagePath
		self.threshold = threshold

	def plotImage(self, image):
		plt.imshow(image, cmap = "gray")
		plt.show()
		
	def commitNeighPixelSum(self, perThreshold):
		if 0.01 <= float(perThreshold) <= 0.04:
			return 41.2
		if 0.05 <= float(perThreshold) <= 0.14:
			return 26
		elif 0.15 <= float(perThreshold) <= 0.18:
			return 22.2
		elif 0.19 <= float(perThreshold) <= 0.24:
			return 21
		elif  0.25 <= float(perThreshold) <= 0.27:
			return 20.5
		elif 0.28 <= float(perThreshold) <= 0.40:
			return 20.3

	def circlesHoughTransformation(self, image):
		maxIdensity = np.amax(image)
		accumulatorThreshold = 180
		rows, columns = image.shape
		sinAngles = dict()
		cosAngles = dict()
		circles = list()
		centers = list()
		notInCenters = True
		circlesFound = 0
		for angle in range(0, 360):
			sinAngles[angle] = np.sin(angle * np.pi/180)
			cosAngles[angle] = np.cos(angle * np.pi/180)	
		radius = [rad for rad in range(10, 21)]
		
		start = time.time()
		accSum = self.commitNeighPixelSum(self.threshold)
		for rad in radius:#gia kathe diaforetikh akina diaforetiko accumulator array
			print("For Radius: %d" % rad)
			accumulatorArray = np.zeros((rows, columns))
			for row in range(rows):
				for column in range(columns):
					if image[row][column] == 255:
						for angle in range(0, 360):
							a = np.uint64(row - round(rad * sinAngles[angle]))
							b = np.uint64(column - round(rad * cosAngles[angle]))
							if a >=0 and a < rows and b >=0 and b < columns:
								accumulatorArray[a][b] += 1
			#self.plotImage(accumulatorArray)
			globalMaximum = round(np.amax(accumulatorArray))
			if globalMaximum > accumulatorThreshold:
				accumulatorArray[accumulatorArray < accumulatorThreshold] = 0
				for i in range(rows):
					for j in range(columns):
						if i > 0 and j > 0 and i < rows-1 and j < columns-1:
							avgNeighboursPixelSum = np.float64((accumulatorArray[i][j]+accumulatorArray[i-1][j]+accumulatorArray[i+1][j]+accumulatorArray[i][j-1]+accumulatorArray[i][j+1] + \
												accumulatorArray[i-1][j-1]+accumulatorArray[i-1][j+1]+accumulatorArray[i+1][j-1]+accumulatorArray[i+1][j+1])/9)
							if avgNeighboursPixelSum >= accSum:
									#elegxos an uparxei hdh kuklos me kontino kentro
									for center in centers:
											x, y, r = center[0], center[1], center[2]
											neighbourCenters = []
											neighbourCenters.extend(((x-1,y-1), (x-1,y), (x-1, y+1), (x, y-1), (x,y), (x, y+1), (x+1, y-1), (x+1,y), (x+1,y+1) \
														, (x-2,y-2), (x-2,y-1), (x-2, y), (x-2, y+1), (x-2,y+2), (x-1, y-2), (x-1, y+2), (x,y-2) \
														, (x,y+2), (x+1,y+2), (x+1, y-2) ,(x+2, y-2), (x+2, y-1), (x+2,y), (x+2,y+1), (x+2,y+2) \
														, (x-3, y-3), (x-3, y-2), (x-3, y-1), (x-3, y), (x-3, y+1), (x-3, y+2), (x-3, y+3), (x-2, y-3) \
														, (x-1, y-3),(x, y-3), (x+1, y-3), (x+2, y-3), (x+3, y-3), (x+3, y-2), (x+3, y-1), (x+3, y) \
														, (x+3, y+1), (x+3, y+2), (x+3, y+3), (x+2, y+3), (x+1, y+3), (x, y+3), (x-1, y+3), (x-2, y+3)))
											if (i,j) in neighbourCenters:
												if rad > r:
													circles.remove((x,y,r))
													centers.remove((x,y,r))
													circlesFound -= 1
													#break
												else:
													notInCenters = False
													break
											
									if notInCenters:
										print("One circle with Radius %d found" % rad)
										circlesFound += 1
										circles.append((i,j,rad))
										centers.append((i,j,rad))
									accumulatorArray[i:i+5, j:j+5] = 0
									#j = j+5
									notInCenters = True
					#i = i+5
		end = time.time()
		finalExecutionTime = end - start
		finalExecutionTime = finalExecutionTime%3600/60
		
		self.printInformation(finalExecutionTime, circlesFound)		
		return circles					
									
					
	def printInformation(self, executionTime, numberOfCircles):
		print("Hough Circle Detection in %lf minutes" % executionTime )
		print("%d Circles Found" % numberOfCircles)	
		
	def main(self):
		sobel = sobelEdgeDetector(self.imagePath, self.threshold)
		originalImage = sobel.openImage()
		imageEdgeMap = sobel.main()
		imageWithCircles = Image.new("RGB", originalImage.size)
		imageWithCircles.paste(originalImage)
		drawResult = ImageDraw.Draw(imageWithCircles)
		circles = self.circlesHoughTransformation(imageEdgeMap)
		for circle in circles:
			drawResult.ellipse((circle[1]-circle[2], circle[0]-circle[2], circle[1]+circle[2], circle[0]+circle[2]), outline=(0,255,0,0))

		imageWithCircles.save("circles"+self.imagePath)	
		#self.plotImage(imageWithCircles)
cycleDetector = cycleDetector(sys.argv[1], sys.argv[2])
cycleDetector.main()

