import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

###Construction Phase###
tfImages = tf.placeholder(tf.float32, shape = [None, 784])
tfLabels = tf.placeholder(tf.int64, shape = [None, 10])
dropoutKeep = tf.placeholder(tf.float32)

learningRate = .001
batchSize = 100
features = 32
kernelSize = 3
trainingSteps = 100 #500

tPerformance = np.zeros(trainingSteps)
vPerformance = np.zeros(trainingSteps)
tCross = np.zeros(trainingSteps)
vCross = np.zeros(trainingSteps)

def plotResults():
    #Plots the performance of the training and validation data, as well as the cross entropy at the end of the training phase
    vpFinal = vPerformance
    vcFinal = vCross
    xaxis = np.arange(trainingSteps)
    indicies = []
    for i in range(trainingSteps):
        if vPerformance[i] == 0:
            indicies.append(i)
    vpFinal = np.delete(vpFinal, indicies)
    vcFinal = np.delete(vcFinal, indicies)
    xaxis = np.delete(xaxis, indicies)
    plt.plot(tPerformance, 'b-', label='Training')
    plt.plot(xaxis, vpFinal, 'r-', label='Validation')
    plt.title('Performance on Training and Validation Data During Training')
    plt.xlabel('Training step')
    plt.ylabel('Percent Correct')
    plt.legend(loc=4)
    plt.show()
    plt.plot(tCross, 'b-', label='Training')
    plt.plot(xaxis, vcFinal, 'r-', label='Validation')
    plt.title('Cross Entropy of Training and Validation Data During Training')
    plt.xlabel('Training step')
    plt.ylabel('Cross Entropy')
    plt.legend(loc=1)
    plt.show()

def convLayer(x, features, initial):
    #Creates a convolutional layer.  Takes in the 'initial' variable which tells the function if the layer is the first
    #convolutional layer or not, and therefore uses a different kernel/feature size for each layer.  It then applies
    #Relu and 2x2 max pooling.
    if initial == True:
        cKernel = tf.Variable(tf.truncated_normal([kernelSize,kernelSize,1,features],stddev=.1))
        cBias = tf.Variable(tf.constant(0, shape=[features], dtype=tf.float32))
        x = tf.reshape(x, [-1, 28, 28, 1])
        cConvolve = tf.nn.conv2d(x, cKernel, strides=[1, 1, 1, 1], padding='SAME')
        cRelu = tf.nn.relu(cConvolve + cBias)
        cPooling = tf.nn.max_pool(cRelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    else:
        cKernel = tf.Variable(tf.truncated_normal([kernelSize,kernelSize,features,features*2],stddev=.1)) #second layer features must be 2xfirst
        cBias = tf.Variable(tf.constant(0, shape=[features*2], dtype=tf.float32))
        cConvolve = tf.nn.conv2d(x, cKernel, strides=[1, 1, 1, 1], padding='SAME')
        cRelu = tf.nn.relu(cConvolve + cBias)
        cPooling = tf.nn.max_pool(cRelu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return cPooling, cKernel

def ffLayer(x, layers, type):
    #Creates a feedforward layer (i.e. fully connected layer).  Takes in type which tells it which layer to create, and
    #layers which is how many inputs per neuron
    if type == 'first':
        fWeights = tf.Variable(tf.truncated_normal([7*7*layers,1024], stddev=.1))
        fBias = tf.Variable(tf.constant(0, shape=[1024], dtype=tf.float32))
        fLogits = tf.matmul(x, fWeights) + fBias
        fActivation = tf.nn.dropout(tf.nn.relu(fLogits), dropoutKeep)
        return fActivation, fWeights
    elif type == 'second':
        fWeights = tf.Variable(tf.truncated_normal([layers, int(layers/2)], stddev=.1))
        fBias = tf.Variable(tf.constant(0, shape=[layers/2], dtype=tf.float32))
        fLogits = tf.matmul(x, fWeights) + fBias
        fActivation = tf.nn.dropout(tf.nn.relu(fLogits), dropoutKeep)
        return fActivation, fWeights
    else:
        fWeights = tf.Variable(tf.truncated_normal([512,10], stddev=.1))
        fBias = tf.Variable(tf.constant(0, shape=[10], dtype=tf.float32))
        fLogits = tf.matmul(x, fWeights) + fBias
        return fLogits, fWeights


conv1, ck1L2 = convLayer(tfImages, features, True)
conv2, ck2L2 = convLayer(conv1, features, False)
convolution = tf.reshape(conv2,[-1, 7*7*(features*2)])
ff1, fw1L2 = ffLayer(convolution, features*2,'first')
ff2, fw2L2 = ffLayer(ff1, 1024,'second')
output, fw3L2 = ffLayer(ff2,features, 'end')


crossEntropy = tf.nn.softmax_cross_entropy_with_logits(logits = output, labels = tfLabels)
# Cross entropy is normalized with the L2 norms of the weights
crossEntropy += .001 * tf.add_n((tf.nn.l2_loss(ck1L2),
                                 tf.nn.l2_loss(ck2L2),
                                 tf.nn.l2_loss(fw1L2),
                                 tf.nn.l2_loss(fw2L2),
                                 tf.nn.l2_loss(fw3L2)))
crossEntropy = tf.reduce_mean(crossEntropy)
trainStep = tf.train.AdamOptimizer(learningRate).minimize(crossEntropy)


match = tf.equal(tf.argmax(output,1), tf.argmax(tfLabels,1))
accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

###Training Phase###

pseudoEpochs = 0
timeStart = time.time()
with tf.Session() as session:
    print('Initializing...')
    session.run(tf.global_variables_initializer())
    print('Network initialized')
    # saver = tf.train.Saver()
    for step in range(trainingSteps):
        batch = mnist.train.next_batch(batchSize)
        images = batch[0]
        labels = batch[1]
        tPerformance[step], tCross[step], weights = session.run(
            [accuracy, crossEntropy, trainStep], feed_dict={tfImages: images, tfLabels: labels, dropoutKeep: .7})
        print('Training performance: {}.  Cross entropy: {}\nStep: {}/{}'.format(tPerformance[step], tCross[step], step+1, trainingSteps))
        if step%50==0 or step==trainingSteps-1:
            images, labels = mnist.validation.images, mnist.validation.labels
            vPerformance[step], vCross[step] = session.run([accuracy, crossEntropy], feed_dict={tfImages: images, tfLabels: labels, dropoutKeep: 1})
            # took out the weights calc in the validation
            print('Validation performance: {}. Cross entropy: {}'.format(vPerformance[step], vCross[step]))
            # saver.save(session, 'path', global_step=step)
        plt.plot(tPerformance, 'b-', label='Training')
        plt.pause(0.0001) # make the plot live
    images, labels = mnist.test.images, mnist.test.labels
    testAccuracy = session.run([accuracy], feed_dict={tfImages: images[:100], tfLabels: labels[:100], dropoutKeep: 1}) #slicing only to make debugging faster
    pseudoEpochs = math.floor((trainingSteps * batchSize) / 55000)


    print('Final test performance:', max(testAccuracy))


timeEnd = time.time()

print('Time elapsed in minutes: {}'.format(round((timeEnd-timeStart)/60)))

print('Learning rate:', learningRate)
print('Batch size:', batchSize)
print('Training steps:', trainingSteps)
print('Pseudo epochs:', pseudoEpochs)
print('Features for first layer:', features, 'and for second layer:', features * 2)
print('Kernel size: {}x{}'.format(kernelSize,kernelSize))
print('Dropout rate: .7')

plt.close() # close the live plot and start the analysis plots
plotResults()
