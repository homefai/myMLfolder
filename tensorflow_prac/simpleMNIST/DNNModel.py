import tensorflow as tf
import numpy as np
import mnistdata as md


class simpleModel: 
    def __init__(self , weightsize = [28*28, 10], inputsize=28*28, outputsize=10): 
        self.learningValue = 0.5
        self.x = tf.placeholder(tf.float32, shape = [None, inputsize])
        self.y_ = tf.placeholder(tf.float32, shape = [None, outputsize] ) 

        '''
        self.weight1 = tf.Variable(tf.zeros([weightsize[0], weightsize[1]]))
        self.bias1 = tf.Variable(tf.zeros([weightsize[1]] ))
        self.layer1 = tf.add( tf.matmul( self.x, self.weight1), self.bias1)
        '''

        self.weightout = tf.Variable(tf.zeros([ weightsize[0] , weightsize[1]]))
        self.biasout = tf.Variable(tf.zeros([weightsize[1]]))
        self.layerOut = tf.add(tf.matmul(self.x , self.weightout), self.biasout) 

        self.crossEntropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.layerOut, self.y_))
        self.optimzer = tf.train.GradientDescentOptimizer(self.learningValue).minimize(self.crossEntropy)
        self.sess = tf.Session()
        
    
    def prediction( self, x_input, sess): 
        return sess.run(  self.layerOut , feed_dict={ self.x: x_input} )
    
    def startOptimzer(self, loadedData, label): 
        self.sess.run(tf.global_variables_initializer())
        print 'here'
        for i in range(len(label)): 
            print i
            if i ==0: 
                print 'loadedData'
                print loadedData[i]
                print 'label' , label[i] 
            self.sess.run(self.optimzer, feed_dict= {  self.x:[np.array(loadedData[i])] , self.y_: [label[i]] } )
        print 'end of optimization'
    
    def getLabelValue(self, y_):
        return np.dot(y_, [0,1,2,3,4,5,6,7,8,9])

    def acc_count(self, x_input, label): 
        total = 0
        correct = 0
        #confusionMat = np.zeros(10,10)
        
        for i in range(len(label)):
            ans = self.sess.run(self.layerOut, feed_dict={ self.x : [x_input[i]]} )
            ans_index = np.argmax(ans)
            value = self.getLabelValue(label[i])
                #confusionMat[value][ans_index] += 1
            total += 1
            if value == ans_index: 
                correct += 1
        return total, correct, correct / float(total)
    
    def loadData(self): 
        max_value = 10000
        mnistLoader = md.MNISTData(maxvalue=max_value)
        trainData = mnistLoader.train.next_batch(max_value)
        testData = mnistLoader.test.next_batch(max_value)
        trainLabel = trainData[1]
        trainData = trainData[0]
        testLabel = testData[1]
        testData = testData[0]

        return trainData, trainLabel, testData, testLabel

    
    def defaultRun(self): 
        trainData , trainLabel, testData, testLabel = self.loadData()
        self.startOptimzer(trainData, trainLabel)
        total , correct , acc = self.acc_count(testData, testLabel)
        print acc
    
    def testselflayerOut(self): 
        test = np.random.rand(784)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        v = sess.run(self.layerOut, feed_dict= {self.x: [test] } )
        print v
        sess.close()


def classTesting():
    testclass = simpleModel()
    testclass.defaultRun()
    
def tryTheNumpy(): 
    listElement = np.random.randint(0, 2, 784)
    print listElement


def main():
    #print 'This is DNNLib'
    #classTesting()
    #tryTheNumpy()
    pass

if __name__ == '__main__':
    main()