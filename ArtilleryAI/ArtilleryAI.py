# Framework to aim an artillery shell to hit a target.  Can apply various learning algorithms to see which learns the task best
# Artillery Artifical Learning
# ArtyAL
# AAL
# All forces are defined with f* so that they can be searched quickly

###Dependencies###
import numpy as np
from sklearn import preprocessing
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout



class Artillery:
    """
    Creates an artillery piece that has a barrel angle and shot force
    """

    def __init__(self):
        self.force = 1000000  # newtons
        self.angle = 60  # degrees
        self.contact_time = .001  # seconds
        self.shell_mass = 10  # kilograms

    def change_angle(self, size=None, change=60):
        """
        Changes the angle of the barrel.  Either returns a list of random angle changes, or sets it to the user's
        desire.
        :param size:
        :param change:
        :returns list of angles if desired:
        """
        if size != None:
            angles = np.random.randint(1, 90, size) #not inclusive of 90
            return angles
        else:
            self.angle = change


    def change_force(self, size=None, change=60):
        """
        Changes the force of the shot.  Either returns a list of random force changes, or sets it to the user's
        desire.
        :param size:
        :param change:
        :returns list of forces if desired:
        """
        if size != None:
            forces = np.random.randint(10000, 100000000, size)
            return forces
        else:
            self.force = change


    def change_mass(self, size=None, change=60):
        """
        Changes the shell mass of the shot.  Either returns a list of random mass changes, or sets it to the user's
        desire.
        :param size:
        :param change:
        :returns list of masses if desired:
        """
        if size != None:
            masses = np.random.randint(1, 100, size)
            return masses
        else:
            self.shell_mass = change


    def shoot(self, force, mass):
        """
        Shoots the artillery with current specs or with given specs
        :return speed and angle:
        """
        if type(force)==type(np.array((1))):  # quick hacky solution
            shots = []
            for shot in range(len(force)):
                shots.append((force[shot] * self.contact_time) / mass[shot])
            return shots
        else:
            speed = (self.force * self.contact_time) / self.shell_mass
            return speed, self.angle


class Target:
    """
    Creates a target some distance away from the artillery piece
    """
    def __init__(self, set_size, phys, angle, shots):#, algo):#, phase):
        self.set_sizes = set_size
        self.phys = phys
        self.angles = angle
        self.shots = shots
        if len(self.angles) != len(self.shots):
            raise ValueError("Length of angles and shots not equal")
        if self.set_sizes != len(self.angles):
            raise ValueError("Length of setsizes and angle/shots not equal")


    def generate_targets(self):
        target_distances = []  # meters
        for shot in range(self.set_sizes):
            target_distances.append(self.phys.ballistic(self.shots[shot], self.angles[shot]))
        return target_distances


class Physics:
    """
    Creates the world the artillery is in, including gravity and wind, as well as their interactions
    """
    def __init__(self):
        self.gravity = 9.81  # meters/sec


    def ballistic(self, v, a):
        dist_maxX = (v ** 2 * np.sin(
            np.deg2rad(2 * a)) / self.gravity)  # equation for horizontal distance covered based on angle and v
        return dist_maxX


class Environment:
    """
    Instantiates all classes together into one class
    """
    def __init__(self):
        self.arty = Artillery()
        self.physics = Physics()


    def generate_set(self, size):
        angles = self.arty.change_angle(size)
        forces = self.arty.change_force(size)
        masses = self.arty.change_mass(size)
        shots = self.arty.shoot(forces, masses)
        target = Target(size, self.physics, angles, shots)
        targets = np.asarray(target.generate_targets())

        angles = (angles - np.mean(angles)) / np.std(angles)
        forces =(forces - np.mean(forces)) / np.std(forces)
        masses = (masses - np.mean(masses)) / np.std(masses)
        targets = (targets - np.mean(targets)) / np.std(targets)
        shots = (shots - np.mean(shots)) / np.std(shots)

        set = np.zeros((3, size))
        set[0] = angles
        set[1] = forces
        set[2] = masses
        # set[3] = shots
        set = set.T

        return set, targets


class ANN:
    def __init__(self, env):
        self.env = env
        self.model = Sequential()
        self.model.add(Dense(units=4, input_dim=3, activation='relu'))
        # self.model.add(Dropout(0.5))
        self.model.add(Dense(3, activation='relu'))
        self.model.add(Dense(3, activation='relu'))
        self.model.add(Dense(units=1))
        self.model.compile(loss='mean_squared_error',
                           optimizer='sgd',
                           metrics=['accuracy'])

    def generate_data(self, size):
        self.training, self.training_tgt = self.env.generate_set(size)
        self.test, self.test_tgt = self.env.generate_set(size)

        # print(self.training[0])
        # print("AH")
        # print(self.training[1])
        # print("SH")
        # print(self.training[2])

    def learn_physics(self):
        self.model.fit(self.training, self.training_tgt, batch_size=1, epochs=300, verbose=1, shuffle=False)
        self.model.evaluate(self.test, self.test_tgt, batch_size=1, verbose=1)

    def predict(self, x):
        prediction = self.model.predict(x, verbose=1)
        return prediction

env = Environment()
network = ANN(env)
network.generate_data(2)

tester = [60,1000000,10]
tester = np.asarray(tester)
tester = np.expand_dims(tester, 0)
# print(tester.shape)

network.learn_physics()

arty = Artillery()
shot,ang = arty.shoot(1000000,10)
phys = Physics()
dist = phys.ballistic(shot, ang)
# network.predict(np.array(([60],[1000000],[10])))
# tester = [60,1000000,10]
# tester = np.asarray(tester)
# print(tester.shape)
pred = network.predict(tester)
print(dist, 'comes out as',pred)
