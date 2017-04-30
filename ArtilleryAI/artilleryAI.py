# Framework to aim an artillery shell to hit a target.  Can apply various learning algorithms to see which learns the task best
# Artillery Artifical Learning
# ArtyAL
# AAL
# All forces are defined with f* so that they can be searched quickly

###Dependencies###
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


class Artillery:
    """
    Creates an artillery piece that has a barrel angle and shot force
    """

    def __init__(self):
        self.force = 1000000  # newtons
        self.angle = 60  # degrees
        self.contactTime = .0001  # seconds
        self.shellMass = 10  # kilograms

        # def changeAngle(self):
        # changes the barrel angle

    def shoot(self):
        speed = (self.force * self.contactTime) / self.shellMass
        return speed, self.angle


class Target:
    """
    Creates a target some distance away from the artillery piece
    """

    def __init__(self, algo, phase):
        self.targetDistance = None  # meters
        self.minDistance = 10  # meters
        self.maxDistance = 10000  # meters
        if algo == "ANN":  # if the network is an ANN, generate train/test sets and properly set up target creation
            self.generateANNSets()
            # if phase == 'train':
            #     self.setup = 'annTrain'
            # elif phase == 'validation':
            #     self.setup = 'annValid'
            # elif phase == 'test':
            #     self.setup = 'annTest'
            # else:
            #     raise ValueError('Invalid ANN phase type, e.g. train/test/etc.')

    def newTarget(self, batch, phase):
        if phase == 'annTrain':
            randomi = np.random.choice(len(self.training), batch, replace=False)
            return self.training[randomi]
        elif phase == 'annValid':
            randomi = np.random.choice(len(self.validation), batch, replace=False)
            return self.validation[randomi]
        elif phase == 'annTest':
            return self.test

    def generateANNSets(self):
        setSizes = 1000
        used = np.empty(setSizes * 3)
        cycler = 0
        self.training, self.validation, self.test = np.zeros(setSizes), np.zeros(setSizes), np.zeros(setSizes)
        for i in range(setSizes * 3):
            j = i % 3  # counts by 3
            k = i - j
            k = int(k / 3)
            if cycler == 0:
                r = np.random.randint(self.minDistance, self.maxDistance)
                while r in used:
                    r = np.random.randint(self.minDistance, self.maxDistance)
                used[i] = r
                self.training[k] = r
                cycler += 1
            if cycler == 1:
                r = np.random.randint(self.minDistance, self.maxDistance)
                while r in used:
                    r = np.random.randint(self.minDistance, self.maxDistance)
                used[i] = r
                self.validation[k] = r
                cycler += 1
            if cycler == 2:
                r = np.random.randint(self.minDistance, self.maxDistance)
                while r in used:
                    r = np.random.randint(self.minDistance, self.maxDistance)
                used[i] = r
                self.test[k] = r
                cycler = 0


class Physics:
    """
    Creates the world the artillery is in, including gravity and wind, as well as their interactions
    """

    def __init__(self):
        self.gravity = 9.81  # meters/sec
        self.windSpeedAgainst = None  # meters/sec; windspeed against the flight path of the shell
        self.windSpeedPush = None  # meters/sec; windspeed pushing the flight path of the shell
        self.airResistance = None  # newtons?????; force pushing back against the shell (simplification of air resistance)

    def fGrav(self, mass):
        return mass * self.gravity

    def ballistic(self, v, a):
        dist_maxX = (v ** 2 * np.sin(
            np.deg2rad(2 * a)) / self.gravity)  # equation for horizontal distance covered based on angle and v
        return dist_maxX


def learningMethodTarget(algo):
    if algo == 'ANN':
        target = Target(algo)
        return target

def learnedParameter():
    '''
    Return either 'angle', 'power', or 'mass'
    '''
    return 'angle'


# rewardScaled = 100*((target-abs(target-landed))/target) #get's percentage close to target as a score



class ANN:
    def __init__(self):
        self.batchSize = 100
        self.arty = Artillery()
        self.tgt = learningMethodTarget("ANN")
        self.env = Physics()
        self.parameter = learnedParameter()
        self.phase = 'annTrain'
        self.tfPhase = tf.Variable(True)
        self.trainingSteps = 100
        self.labels #DEFINE THIS
        self.tPerformance = np.zeros(self.trainingSteps)
        self.vPerformance = np.zeros(self.trainingSteps)
        self.tCross = np.zeros(self.trainingSteps)
        self.vCross = np.zeros(self.trainingSteps)
        self.tfBatch, self.tfLabels =self.tensorInit()

    def getBatch(self, phase):
        batch = self.tgt.newTarget(self.batchSize, phase)
        return batch

    ###Neural Network Section###
    def tensorInit(self):
        tfb = tf.placeholder(tf.float32, shape=[None, 3])
        tfl = tf.placeholder(tf.int64, shape=[None, 2])
        return tfb, tfl

    def feedForward(self, input, num, activation, trainStatus):
        dropOutRate = .2
        layer = tf.layers.dense(inputs=input, units=num, activation=activation)
        dropout = tf.layers.dropout(inputs=layer, rate=dropOutRate, training=trainStatus)
        return dropout

    def finalLayer(self, input):
        logits = tf.layers.dense(input, 2)

    def learn(self, input, labels):
        self.onehotlabels = tf.one_hot(indicies=tf.cast(labels, tf.int32), depth=2)
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=onehotlabels, logits=input)
        train = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(),
                                                   learning_rate=0.001, optimizer="Adam")
        return train, onehotlabels

    def buildGraph(self):
        inputLayer = self.feedForward(self.tfBatch, 3, tf.nn.relu, self.tfPhase)
        hiddenLayer = self.feedForward(inputLayer, 4, tf.nn.relu, self.tfPhase)
        outputLayer = self.finalLayer(hiddenLayer)
        update, lbls = self.learn(outputLayer, self.labels)
        match = tf.equal(tf.argmax(tf.nn.softmax(outputLayer), 1), self.tfLabels)
        self.accuracy = tf.reduce_mean(tf.cast(match, tf.float32))

    def activateLearning(self):
        self.buildGraph()
        with tf.Session() as session:
            print('Initializing...')
            session.run(tf.global_variables_initializer())
            print('Network initialized')
            saver = tf.train.Saver()
            accuracy = 0
            crossEntropy = 0
            trainStep = 1

            for step in range(self.trainingSteps):
                batch = self.getBatch('annTrain')
                self.tPerformance[step], self.tCross[step] = session.run([self.accuracy, self.loss, trainStep],
                                                                    feed_dict={self.tfBatch: batch,
                                                                               self.tfLabels: self.onehotlabels,
                                                                               self.tfPhase: True})
                print('Training performance: {}.  Cross entropy: {}\nStep: {}'.format(self.tPerformance[step], self.tCross[step],
                                                                                  step + 1))
    #         if step % 100 == 0 or step == trainingSteps - 1:
    #             images, labels = validation, validationLabels
    #             vPerformance[step], vCross[step] = session.run([accuracy, crossEntropy],
    #                                                            feed_dict={tfImages: images, tfLabels: labels,
    #                                                                       dropoutKeep: 1})
    #             # took out the weights calc in the validation
    #             print('Validation performance: {}. Cross entropy: {}'.format(vPerformance[step], vCross[step]))
    #             saver.save(session, 'C:/Users/John/Desktop/Tensorflow Project Data/Saved Weights/leaf.ckpt',
    #                        global_step=step)
    #     images, labels = test, testLabels
    #     testAccuracy = session.run([accuracy], feed_dict={tfImages: images, tfLabels: labels, dropoutKeep: 1})
    #     pseudoEpochs = math.floor((trainingSteps * batchSize) / 1152)
    #
    #     print('Final test performance: ', max(testAccuracy))
    #
    # timeEnd = time.time()

    # print('Time elapsed in minutes: {}'.format(round((timeEnd - timeStart) / 60)))
    #
    # print('Batch type:', batchType)
    # print('Learning rate:', learningRate)
    # print('Batch size:', batchSize)
    # print('Training steps:', trainingSteps)
    # print('Pseudo epochs:', pseudoEpochs)
    # print('Features for first layer:', features, 'and for second layer:', features * 2)
    # print('Kernel size: {}x{}'.format(kernelSize, kernelSize))
    # print('Dropout rate: .9')