from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import svm

from skimage.feature import hog
from skimage import data, exposure


def example():
	image = data.astronaut()

	fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8,8),
		cells_per_block=(3,3), visualize=True, multichannel=True)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('input image')

	#resize
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('hog image')

	plt.show()


def imageToHog(imageList):
	resultArray = np.array()
	for img in imageList:
		fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8,8), cells_per_block=(3,3),
		visualize=False, multichannel=True)
		np.append(resultArray, [fd], axis=0)
	return resultArray





#main
def main():
	example()

main()