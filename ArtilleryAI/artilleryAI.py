# Framework to aim an artillery shell to hit a target.  Can apply various learning algorithms to see which learns the task best
# Artillery Artifical Learning
# ArtyAL
# AAL
# All forces are defined with f* so that they can be searched quickly

###Global Dependencies###
import numpy as np


class Artillery:
    """
    Creates an artillery piece that has a barrel angle and shot force
    """

    def __init__(self):
        self.force = 1000000  # newtons
        self.angle = 60  # degrees
        self.contactTime = .0001  # seconds
        self.shellWeight = 10  # kilograms

        # def changeAngle(self):
        # changes the barrel angle

    def shoot(self):
        speed = (self.force * self.contactTime) / self.shellWeight
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
            if phase == 'train':
                self.setup = 'annTrain'
            elif phase == 'validation':
                self.setup = 'annValid'
            elif phase == 'test':
                self.setup = 'annTest'
            else:
                raise ValueError('Invalid ANN phase type, e.g. train/test/etc. (kwargs "phase" variable)')

    def newTarget(self, batch):
        if self.setup == 'annTrain':
            randomi = np.random.choice(len(self.training), batch, replace=False)
            return self.training[randomi]
        elif self.setup == 'annValid':
            randomi = np.random.choice(len(self.validation), batch, replace=False)
            return self.validation[randomi]
        elif self.setup == 'annTest':
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
        import tensorflow as tf
        target = Target(algo, 'train')
        return target


algorithm = 'ANN'
target = learningMethodTarget(algorithm)
trainingSample = target.newTarget(10)

print(trainingSample)

arty = Artillery()
v, a = arty.shoot()
env = Physics()
distance = env.ballistic(v, a)

print(distance)

# rewardScaled = 100*((target-abs(target-landed))/target) #get's percentage close to target as a score