import tensorflow as tf
import numpy as np

'''
loadMNISTData: load The data
getTestData: get the test data, parameter <= MNIST Loaded Test data
getTrainData: get the test data, parameter <= MNIST Loaded Train data
getBatch: get the batch from train or test data, parameter <= train/test mnist data , batch size
getImageAndLabelAsNumpyArray: trans to image and label numpy array
'''

def loadMNISTData(): 
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("MNIST_data/" ,one_hot = True)

def getTestData(mnistData): 
    return mnistData.test

def getTrainData(mnistData):
    return mnistData.train

def getBatch(mnistData, batchSize=100): 
    return mnistData.next_batch(batchSize)

def getImageAndLabelAsNumpyArray(batchedmnistData, labelData):
    image = (np.array(batchedmnistData) > 0.1).astype(int)
    label = np.array(labelData)
    label = [np.dot(i , [0,1,2,3,4,5,6,7,8,9]) for i in label]
    return image, label

def showImageAndValue(img, label): 
    strd = [ str(i) for i in img ]
    h= ''
    print 'Value is : ' , label
    for i in range(len(strd)): 
        if i % 28 == 0 and i != 0: 
            print h
            h = strd[i]
        else: 
            h += strd[i]

def defaultLoadDataRun(max=100): 
    batcheddata = getBatch(getTestData(loadMNISTData()) , max)
    image, label = getImageAndLabelAsNumpyArray(batcheddata[0], batcheddata[1])
    return image, label, max


class MNISTBatchedData: 
    def __init__(self, data, maxvalue=100): 
        self.data = data
        self.max = maxvalue
        self.image, self.label = self.getbatch(maxvalue)

    def getbatch(self, maxvalue=100): 
        batchedData = getBatch(self.data, maxvalue)
        return getImageAndLabelAsNumpyArray(batchedData[0], batchedData[1])
    
    def printImageAndValue(self, value=0): 
        if value < 0: 
            value = 0
        elif value > self.max: 
            value = self.max
        showImageAndValue(self.image[value], self.label[value])

class MNISTData:
    def __init__(self, maxvalue=100): 
        self.raw_mnist = loadMNISTData()
        self.max = maxvalue
        self.test = self.raw_mnist.test
        self.train = self.raw_mnist.train
        self.batched_train = MNISTBatchedData( getTrainData(self.raw_mnist), maxvalue)
        self.batched_test = MNISTBatchedData(getTestData(self.raw_mnist), maxvalue)


def main():
    pass

if __name__ == '__main__':
    main()
