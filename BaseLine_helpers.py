import cv2
import numpy as np 
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import itertools
import imutils

class ImageHelpers:
	def __init__(self):
		pass

	def raw_pixel(self, image, size=(32, 32)):
		# resize the image to a fixed size, then flatten the image into
		# a list of raw pixel intensities
		return cv2.resize(image, size).flatten()

	def color_histogram(self, image, bins=(8, 8, 8)):
		# extract a 3D color histogram from the HSV color space using
	    # the supplied number of `bins` per channel
		hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
				[0, 180, 0, 256, 0, 256])

	    # handle normalizing the histogram if we are using OpenCV 2.4.X
		if imutils.is_cv2():
			hist = cv2.normalize(hist)

	    # otherwise, perform "in place" normalization in OpenCV 3 (I
	    # personally hate the way this is done
		else:
			cv2.normalize(hist, hist)

	    # return the flattened histogram as the feature vector
		return hist.flatten()

class BLHelpers:
	def __init__(self):
		self.clf  = SVC(probability=False, cache_size=200,
						kernel="rbf", C=1.0, gamma= 'auto')	

	def train(self, hist, train_labels):
		"""
		uses sklearn.svm.SVC classifier (SVM) 

		"""
		print ("Training SVM")
		print (self.clf)
		print ("Train labels", train_labels)
		self.clf.fit(hist, train_labels)
		print ("Training completed")

	def predict(self, iplist):
		predictions = self.clf.predict(iplist)
		return predictions

	def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
	    """
	    This function prints and plots the confusion matrix.
	    Normalization can be applied by setting `normalize=True`.
	    """
	    if normalize:
	        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	        print("Normalized confusion matrix")
	    else:
	        print('Confusion matrix, without normalization')

	    print(cm)

	    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	    plt.title(title)
	    plt.colorbar()
	    tick_marks = np.arange(len(classes))
	    plt.xticks(tick_marks, classes, rotation=45)
	    plt.yticks(tick_marks, classes)

	    fmt = '.2f' if normalize else 'd'
	    thresh = cm.max() / 2.
	    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	        plt.text(j, i, format(cm[i, j], fmt),
	                 horizontalalignment="center",
	                 color="white" if cm[i, j] > thresh else "black")

	    plt.tight_layout()
	    plt.ylabel('True label')
	    plt.xlabel('Predicted label')


class FileHelpers:

	def __init__(self):
		pass

	def getFiles(self, path):
		"""
		- returns  a dictionary of all files 
		having key => value as  objectname => image path

		- returns total number of files.

		"""
		imlist = {}
		count = 0
		for each in glob(path + "*"):
			word = each.split("/")[-1]
			print (" #### Reading image category ", word, " ##### ")
			imlist[word] = []
			for imagefile in glob(path+word+"/*"):
				print ("Reading file ", imagefile)
				im = cv2.imread(imagefile)
				imlist[word].append(im)
				count +=1 

		return [imlist, count]

