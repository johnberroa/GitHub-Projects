import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


class ANN:
    """Creates an artificial neural network to learn the projectile motion function"""

    def __init__(self, arty, target, physics):
        self.arty = arty
        self.target = target
        self.physics = physics
        self.training, self.test = target.generateANNSets()
        self.angles = self.arty.changeAngle(self.target.setSizes)

    def generateModel(self, type):
        """
        Generates a Keras ANN and returns it
        """
        if type == 'singular':
            self.model = Sequential()
            self.model.add(Dense(units=10, input_dim=1, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(units=1, activation='softmax'))
            self.model.compile(loss='mean_squared_error',
                               optimizer='adam',
                               metrics=['accuracy'])
        elif type == 'multi':
            self.model = Sequential()
            self.model.add(Dense(units=10, input_dim=4, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(units=1, activation='softmax'))
            self.model.compile(loss='mean_squared_error',
                          optimizer='adam',
                          metrics=['accuracy'])
        else:
            print("Invalid model type (Turn this into an exception)")
        return self.model

    def fitModel(self, model, type):
        if type == 'singular':
            model.fit(self.angles, self.training, epochs=5, batch_size=50)
            loss_and_metrics = model.evaluate(ANGLES, self.test, batch_size=100)
        elif type == 'multi':
            model.fit(INPUTS, self.training, epochs=5, batch_size=50)
            loss_and_metrics = model.evaluate(INPUTS, self.test, batch_size=100)
        else:
            print('SAME AS ABOVE')
