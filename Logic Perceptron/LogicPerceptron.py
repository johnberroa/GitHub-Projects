import numpy as np


class LogicPerceptron:
    """
    Creates a perceptron that learns logical operators AND, OR, NAND, and NOR
    """

    def __init__(self):
        self.epsilon = .1
        self.dimensions = 2
        self.training_size = 100
        self.test_size = 100
        self.weights = np.random.random(self.dimensions + 1)  # +1 because of adding a bias


    def generate_datasets(self):
        """
        Generates training and test data sets as 1s and 0s with 1s as the final column for bias
        :return training_set, test_set:
        """
        # generate training set
        training_set = np.random.randint(2, size=(self.training_size, self.dimensions))  # pairs of 1s and 0s
        bias = np.ones(self.training_size)  # shape (100,)
        bias = np.expand_dims(bias, axis=1)  # shape (100,1)
        training_set = np.concatenate((bias, training_set), axis=1)

        # generate test set
        test_set = np.random.randint(2, size=(self.test_size, self.dimensions))  # pairs of 1s and 0s
        bias = np.ones(self.test_size)  # shape (100,)
        bias = np.expand_dims(bias, axis=1)  # shape (100,1)
        test_set = np.concatenate((bias, test_set), axis=1)
        return training_set, test_set


    def generate_labels(self, function, dataset):
        """
        Passes through the datapoints to get the correct classification based on the logical function provided
        :param function:
        :param dataset:
        :return labels:
        """
        labels = []
        for datapoint in dataset:
            labels.append(function(datapoint[0], datapoint[1]))
        return labels


    def threshold(self, activation):
        """
        Simple step function threshold
        :param activation:
        :return 1 or 0:
        """
        if activation >= 0:
            return 1
        else:
            return 0


    def functions_to_learn(self, selector):
        """
        Functional definitions for the perceptron to learn
        :param selector (selects which function):
        :return function:
        """
        if selector == 'and':
            function = lambda x1, x2: x1 and x2
            return function
        elif selector == 'or':
            function = lambda x1, x2: x1 or x2
            return function
        elif selector == 'nand':
            function = lambda x1, x2: not (x1 and x2)
            return function
        elif selector == 'nor':
            function = lambda x1, x2: not (x1 or x2)
            return function
        else:
            raise ValueError("Incorrect function to learn type.  Pick and/or/nand/nor.  Input was:", selector)


    def infer(self, datapoint):
        """
        Passes datapoint through the network to get an answer
        :param datapoint:
        :return output (the answer):
        """
        activation = np.dot(self.weights, datapoint)
        output = self.threshold(activation)
        return output


    def learn(self, outputs, labelset, datapoint):
        """
        Implements the perceptron learning rule
        :param outputs:
        :param labelset:
        :param dataset:
        """
        delta_w = self.epsilon * ((labelset - outputs) * datapoint)
        self.weights += delta_w # perceptron learning rule


    def train(self, function_string, epochs):
        """
        Trains the perceptron for a certain amount of epochs, then tests it and returns the result
        :param function_string:
        :param epochs:
        :return performance (on test set):
        """
        training_set, test_set = self.generate_datasets()
        function = self.functions_to_learn(function_string)
        labels = self.generate_labels(function, training_set)
        for e in range(epochs):
            for i, step in enumerate(training_set):
                output = self.infer(step)
                self.learn(output, labels[i], step)
        performance_test = self.test(function, test_set)
        return performance_test


    def test(self, function, test_set):
        """
        Tests the performance of the current weights against the training set
        :param function_string:
        :param test_set:
        :return performance:
        """
        labels = self.generate_labels(function, test_set)
        results = []
        for i, step in enumerate(test_set):
            output = self.infer(step)
            if labels[i] == output:
                results.append(1)
            else:
                results.append(0)
        correct_results = np.count_nonzero(results)
        performance = correct_results / self.test_size
        return performance


if __name__ == "__main__":
    perceptron = LogicPerceptron()
    performancetst = perceptron.train('and', 5) # can do 'and', 'or', 'nand', and 'nor'
    print("\n\nTest performance after training:", performancetst)

