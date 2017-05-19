# Framework to aim an artillery shell to hit a target.  Can apply various learning algorithms to see which learns the task best
# Artillery Artifical Learning
# ArtyAL
# AAL
# All forces are defined with f* so that they can be searched quickly

###Dependencies###
import numpy as np
import ArtyAlgorithms as algo
# import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


class Artillery:
    """
    Creates an artillery piece that has a barrel angle and shot force
    """

    def __init__(self):
        self.force = 1000000  # newtons
        self.angle = 60  # degrees
        self.contactTime = .001  # seconds
        self.shellMass = 10  # kilograms

    def changeAngle(self, size):
        '''
        Changes the angle of the barrel
        If training the ANN, returns an array the size of the dataset of angle values'''
        if size != None:
            angles = np.random.randint(1, 90, size) #not inclusive of 90
            return angles
        else:
            self.angle = np.random.randint(1, 90)

    def shoot(self):
        speed = (self.force * self.contactTime) / self.shellMass
        return speed, self.angle



class Target:
    """
    Creates a target some distance away from the artillery piece
    """

    def __init__(self):#, algo):#, phase):
        self.targetDistance = None  # meters
        self.minDistance = 35.58  # meters, based on 1000000 N, .001 s, and 10 kg
        self.maxDistance = 1019.37  # meters, based on 1000000 N, .001 s, and 10 kg
        self.setSizes = 2000

    # def newTarget(self, batch, phase):
    #     if phase == 'annTrain':
    #         randomi = np.random.choice(len(self.training), batch, replace=False)
    #         return self.training[randomi]
    #     elif phase == 'annValid':
    #         randomi = np.random.choice(len(self.validation), batch, replace=False)
    #         return self.validation[randomi]
    #     elif phase == 'annTest':
    #         return self.test

    def generateANNSets(self):
        '''
        Generates training and test labels (target distances).  Distances are multiplied by 100 to get an int, then
        divided by 100 again to get it within the range of the artillery.
        '''
        used = []
        cycler = 0
        self.training, self.test = np.zeros(self.setSizes), np.zeros(self.setSizes)
        for i in range(self.setSizes * 2):
            j = i % 2  # counts by 2
            k = i - j
            k = int(k / 2)
            if cycler == 0:
                r = np.random.randint(self.minDistance * 100, self.maxDistance * 100)
                r /= 100
                while r in used:
                    r = np.random.randint(self.minDistance * 100, self.maxDistance * 100)
                    r /= 100
                used.append(r)
                self.training[k] = r
                cycler += 1
            if cycler == 1:
                r = np.random.randint(self.minDistance * 100, self.maxDistance * 100)
                r /= 100
                while r in used:
                    r = np.random.randint(self.minDistance * 100, self.maxDistance * 100)
                    r /= 100
                used.append(r)
                self.test[k] = r
                cycler = 0
        return self.training, self.test


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

    def generateANNParameters(self, train, test):
        single_test_params = []
        multi_test_params = []

        for x in range(len(test)):
            #get the angles needed for test data
            pass
        for x in range(len(single_test_params)):
            #get the angles need for the test data (already calculated, turned into tuples with the other parameters attached
            pass




class Environment:
    """
    Instantiates all classes together into one class
    """
    def __init__(self):
        self.arty = Artillery()
        self.target = Target()
        self.physics = Physics()

    def learningMethodTarget(self, algo):
        #change this because I want ANN learning of the function to be sepearte
        if algo == 'ANN':
            target = Target(algo)
            return target

    def learnedParameter(self):
        '''
        Return either 'angle', 'power', or 'mass'
        '''
        return 'angle'

    def learnPhysics(self):
        '''
        Generates datasets, trains NN on it, and returns a network that can correctly tell if a target will be hit or
        not given the input parameters
        '''
        algo.ANN(self.arty, self.target, self.physics)



# rewardScaled = 100*((target-abs(target-landed))/target) #get's percentage close to target as a score
