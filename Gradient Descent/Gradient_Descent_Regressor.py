##################
# Gradient Descent for Regression
# ## Purpose
# To show the principle behind gradient descent by fitting a regression line through data.
# ## Limitations
# -The gradients seem to explode with to fast a learning rate
# -Complex data sets either take forever to solve, or diverge
##################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('test.csv')
x = data['row'].values
y = data['col'].values


class GradientRegressor():
    """
    Calculates the best fit line for a regression task by using gradient descent
    """

    def __init__(self, x, y, learning_rate=.01):
        self.x = x
        self.y = y
        self.N = len(x)  # getting the length of the dataset
        self.m = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.learning_rate_list = []
        self.simple_speedchange_rate = 5  # hypermeter turned through trial and error
        self.m_avg_grad = []
        self.b_avg_grad = []
        self.training_step = 0
        self.epsilon = 1e-10  # stopping condition
        self.training = True


    def calculate_error(self):
        """
        Calculates the MSE on all datapoints
        """
        squared_error = (self.y - (self.m * self.x + self.b)) ** 2
        mse = np.mean(squared_error)
        return mse


    def gradient_descent(self):
        """
        Calculates the gradients for m and b and then changes m and b in relation to those gradients
        :return:
        """
        m_gradient = (2 / self.N) * np.sum(-self.x * (self.y - (self.m * self.x + self.b)))
        b_gradient = (2 / self.N) * np.sum(-(self.y - (self.m * self.x + self.b)))
        self.m_avg_grad.append(m_gradient)
        self.b_avg_grad.append(b_gradient)
        learning_rate = self.optimize_learningrate(m_gradient, b_gradient)
        self.m -= m_gradient * learning_rate
        self.b -= b_gradient * learning_rate


    def optimize_learningrate(self, m_grad, b_grad):
        """
        Increases learning rate if gradient is shallower than average;
        decreases learning rate if gradient is steeper than average
        :param m_grad:
        :param b_grad:
        :return learning_rate:
        """
        learning_rate = self.learning_rate
        if np.abs(m_grad) > np.abs(np.mean(self.m_avg_grad[-10:]) * 1.2) or np.abs(b_grad) > np.abs(
                        np.mean(self.b_avg_grad[-10:]) * 1.2):
            learning_rate *= 1 / self.simple_speedchange_rate
        elif np.abs(m_grad) < np.abs(np.mean(self.m_avg_grad[-10:]) * .8) or np.abs(b_grad) < np.abs(
                        np.mean(self.b_avg_grad[-10:]) * .8):
            learning_rate *= self.simple_speedchange_rate
        self.learning_rate_list.append(learning_rate)
        return learning_rate


    def regress(self):
        """
        Continually runs gradient descent until the mean of the previous MSEs
        are no different than epsilon from the current mean
        :return slope, intercept, mses, learning_rates:
        """
        mse_list = [100, 100]
        mse_list.append(self.calculate_error())
        while np.abs(mse_list[-1] - np.mean(mse_list[-3:])) > self.epsilon:
            self.gradient_descent()
            mse_list.append(self.calculate_error())
            self.training_step += 1
            self.plot_results()

        print("Training complete")
        print("Training steps",self.training_step)
        print("Error:", mse_list[-1])
        self.training = False
        self.plot_results(mses=mse_list)


    def plot_results(self, mses=None):
        """
        Plots the results in 3 plots--
        1. Best fit line through data
        2. MSE
        3. Learning rate
        :param mses:
        """
        if self.training == True:
            y_out = lambda points: self.m * points + self.b
            plt.plot(self.x, y_out(self.x), 'g', alpha=.1)
        else:
            y_out = lambda points: self.m * points + self.b
            plt.plot(self.x, y_out(self.x), 'r', alpha=1)
            plt.scatter(self.x, self.y)
            plt.title("Data ({} Training Steps)".format(self.training_step))
            plt.show()

            plt.figure(1)
            plt.subplot(211)
            plt.plot(np.arange(len(mses) - 3), mses[3:])
            plt.title("Mean Squared Error")
            plt.ylabel("MSE")

            plt.subplot(212)
            plt.plot(np.arange(len(self.learning_rate_list)), self.learning_rate_list)
            plt.title("Learning Rate")
            plt.ylabel("Rate")

            plt.tight_layout()
            plt.show()



if __name__ == '__main__':
    reg = GradientRegressor(x,y)
    reg.regress()

