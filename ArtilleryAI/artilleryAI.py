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
        self.force = 1000  # newtons????? of force
        self.angle = 45
        self.contactTime = .1  # seconds
        self.shellWeight = 10  # kilograms
        self.height = 0  # meters (0 currently for ease of calculation)
        # def changeAngle(self):
        # changes the barrel angle

    def shoot(self):
        acceleration = self.force / self.shellWeight
        speed = acceleration / self.shellWeight  # (F*t)/m or v = at
        thirdAngle = 180 - 90 - self.angle
        equal = speed / np.sin(90)
        verticalSpeed = equal * np.sin(self.angle)
        horizontalSpeed = equal * np.cos(thirdAngle)
        print(horizontalSpeed, verticalSpeed, 'h,v')  # debugging
        return verticalSpeed, horizontalSpeed, self.height


class Target:
    """
    Creates a target some distance away from the artillery piece
    """

    def __init__(self):
        self.targetDistance = None  # meters
        self.minDistance = 10  # meters
        self.maxDistance = 10000  # meters

    def newTarget(self, **kwargs):
        if kwargs["algo"] == "ANN":
            if kwargs["phase"] == 'train':
                randomi = np.random.choice(len(self.training), kwargs["batch"], replace=False)
                return self.training[randomi]
            elif kwargs["phase"] == 'validation':
                randomi = np.random.choice(len(self.validation), kwargs["batch"], replace=False)
                return self.validation[randomi]
            elif kwargs["phase"] == 'test':
                return self.test
            else:
                raise ValueError('Invalid ANN phase type, e.g. train/test/etc. (kwargs "phase" variable)')

    def generateANNSets(self):
        setSizes = 1000
        used = np.empty(setSizes * 3)
        cycler = 0
        self.training, self.validation, self.test = np.zeros(setSizes), np.zeros(setSizes), np.zeros(setSizes)
        for i in range(setSizes * 3):
            j = i % 3
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

    def ballistic(self, vS, hS, h):
        t_peak = (vS - 0) / self.gravity
        dist_maxY = (vS * t_peak) - (.5 * self.gravity * t_peak)
        dist_maxX = hS * (t_peak * 2)
        return dist_maxX, dist_maxY, t_peak  # delete y and time distance later, just for debugging


def learningMethod(algo):
    if algo == 'ANN':
        import tensorflow as tf
        target = Target()
        target.generateANNSets()
        return target


algorithm = 'ANN'
target = learningMethod(algorithm)
target.generateANNSets()
trainingSample = target.newTarget(algo=algorithm, phase="train", batch=10)

arty = Artillery()
v, h, height = arty.shoot()
env = Physics()
distance, yyy, ttt = env.ballistic(v, h, height)

print(distance, yyy, ttt)
# http://hyperphysics.phy-astr.gsu.edu/hbase/traj.html
# rewardScaled = 100*((target-abs(target-landed))/target) #get's percentage close to target as a score