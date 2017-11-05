import cv2
import numpy as np 
from glob import glob 
import argparse
from BaseLine_helpers import *
from matplotlib import pyplot as plt
from sklearn import metrics



class BaseLine:
    def __init__(self):
        #self.no_clusters = no_clusters
        self.train_path = None
        self.test_path = None
        self.im_helper = ImageHelpers()
        self.bl_helper = BLHelpers()
        self.file_helper = FileHelpers()
        self.images = None
        self.trainImageCount = 0
        self.train_labels = np.array([])
        self.name_dict = {}
        self.descriptor_list = []

    def trainModel_Hist(self):
        """
        This method contains the entire module 
        required for training the bag of visual words model

        Use of helper functions will be extensive.

        """

        # read file. prepare file lists.
        self.images, self.trainImageCount = self.file_helper.getFiles(self.train_path)
        # extract color histogram Features from each image
        label_count = 0 
        for word, imlist in self.images.items():
            self.name_dict[str(label_count)] = word
            print ("Computing Color Histogram Features for ", word)
            for im in imlist:
                self.train_labels = np.append(self.train_labels, label_count)
                hist = self.im_helper.color_histogram(im)
                self.descriptor_list.append(hist)
            label_count += 1

        self.bl_helper.train(self.descriptor_list, self.train_labels)


    def recognize_Hist(self,test_img, test_image_path=None):

        """ 
        This method recognizes a single image 
        It can be utilized individually as well.
        """

        hist = self.im_helper.color_histogram(test_img).reshape(1, -1)
        #.transform(np.atleast_2d(hist))

        '''
        # generate vocab for test image
        vocab = np.array([ 0 for i in range(self.no_clusters)])
        # locate nearest clusters for each of 
        # the visual word (feature) present in the image
        
        # test_ret =<> return of kmeans nearest clusters for N features
        test_ret = self.bl_helper.kmeans_obj.predict(des)

        for each in test_ret:
            vocab[each] += 1

        # Scale the features
        vocab = self.bl_helper.scale.transform(np.atleast_2d(vocab))
        
        '''

        # predict the class of the image
        lb = self.bl_helper.predict(hist)
        print ("Image belongs to class : ", self.name_dict[str(int(lb[0]))])
        return lb



    def testModel_Hist(self):
        """ 
        This method is to test the trained classifier

        read all images from testing path 
        use BOVHelpers.predict() function to obtain classes of each image

        """

        self.testImages, self.testImageCount = self.file_helper.getFiles(self.test_path)

        predictions = []
        test_labels = []
        pridict_labels = []
        class_name = []

        for word, imlist in self.testImages.items():
            print ("processing " ,word)
            class_name.append(word)

            for im in imlist:
                cl = self.recognize_Hist(im)
                predictions.append({
                    'image':im,
                    'class':cl,
                    'object_name':self.name_dict[str(int(cl[0]))]
                    })
                test_labels.append(word)
                pridict_labels.append(self.name_dict[str(int(cl[0]))])

        
        cnf_matrix = metrics.confusion_matrix(test_labels, pridict_labels)
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        self.bl_helper.plot_confusion_matrix(cnf_matrix, classes=class_name,
                              title='Confusion matrix, without normalization')
        #plt.show()
        plt.savefig("confusion_matrix_Hist.png")

        # Print accuracy
        print("Accuracy: %0.4f" % metrics.accuracy_score(test_labels, pridict_labels))
        
    def print_vars(self):
        pass



if __name__ == '__main__':

    # parse cmd args
    parser = argparse.ArgumentParser(
            description=" Bag of visual words example"
        )
    parser.add_argument('--train', action="store", dest="train", required=True)
    parser.add_argument('--test', action="store", dest="test", required=True)
    args =  vars(parser.parse_args())
    
    bl = BaseLine()
    # set training paths
    bl.train_path = args['train'] 
    # set testing paths
    bl.test_path = args['test']
    # train the model
    bl.trainModel_Hist()
    # test model
    bl.testModel_Hist()