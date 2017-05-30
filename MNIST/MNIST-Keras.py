import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization


class MNIST_Classifier:
    """Classifies the MNIST dataset utilizing a Convnet with Keras"""

    def __init__(self):
        self.load()
        self.first_filter = 32
        self.second_filter = 64
        self.kernel = (3,3)
        self.dropout_rate = .7
        self.batch_size = 100
        self.epochs = 1

    def load(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
        x, y = self.x_train.shape[1:]
        self.image_shape = (x, y, 1)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], x, y, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], x, y, 1)

    def generate_network(self):
        model = Sequential()
        model.add(Conv2D(self.first_filter, self.kernel, input_shape=self.image_shape,
                                          strides=(1, 1), padding='valid',
                                          activation='relu', use_bias=True))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                      beta_initializer='zeros', gamma_initializer='ones',
                                                      moving_mean_initializer='zeros',
                                                      moving_variance_initializer='ones', beta_regularizer=None,
                                                      gamma_regularizer=None, beta_constraint=None,
                                                      gamma_constraint=None))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

        model.add(Conv2D(self.second_filter, self.kernel,
                                            strides=(1, 1), padding='valid',
                                            activation='relu', use_bias=True))
        model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                                      beta_initializer='zeros', gamma_initializer='ones',
                                                      moving_mean_initializer='zeros',
                                                      moving_variance_initializer='ones', beta_regularizer=None,
                                                      gamma_regularizer=None, beta_constraint=None,
                                                      gamma_constraint=None))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu', use_bias=True))
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(512, activation='relu', use_bias=True))
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(10, activation='softmax'))

        model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=.001),
                      metrics=['accuracy'])
        return model

    def train_model(self, model):
        model.fit(self.x_train, self.y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(self.x_test, self.y_test))

        performance = model.evaluate(self.x_test, self.y_test, verbose=1)
        return performance

    def report_results(self, results):
        print('Test cross entropy:', results[0])
        print('Test accuracy:', results[1])


network = MNIST_Classifier()
model = network.generate_network()
results = network.train_model(model)
network.report_results(results)
