
class Data():
    def __init__(self, datasize, plot=False):
        self.datasize = datasize
        self.generate_data(self.datasize, plot)


    def generate_data(self, datasize, plot):
        """
        Creates a circular dataset with one label on the outside and anothe ron the inside.
        Credit this code to the University of Osnabrück Machine Learning Course
        :param datasize, plot:
        :return :
        """
        def uniform(a, b, n=1):
            """Returns n floats uniformly distributed between a and b."""
            return (b - a) * np.random.random_sample(n) + a

        n = datasize
        radius = 5
        r = np.append(uniform(0, radius * .5, n // 2),
                      uniform(radius * .7, radius, n // 2))
        angle = uniform(0, 2 * np.pi, n)
        x = r * np.sin(angle) + uniform(-radius, radius, n)
        y = r * np.cos(angle) + uniform(-radius, radius, n)
        self.inputs = np.vstack((x, y)).T
        self.targets = np.less(np.linalg.norm(self.inputs, axis=1), radius * .5)
        if plot:
            plt.figure('Data')
            plt.suptitle('Labeled Data')
            plt.scatter(*self.inputs.T, 2, c=self.targets, cmap='RdYlBu')
            plt.show()


    def generate_sets(self, split):
        train = np.random.randint(0,self.datasize,int(self.datasize*split))
        print(len(train))
        self.training_set = self.inputs[train]
        self.training_set_labels = self.targets[train]
        holder_data = np.array(self.inputs)
        holder_label = np.array(self.targets)
        self.test_set = [self.inputs[i] for i in range(len(self.inputs)) if i not in train]  # this does not work exactly as planned
        self.test_set_labels = [self.targets[i] for i in range(len(self.inputs)) if i not in train]


    def get_batch(self, mode, batchsize='empty'):
        if mode == 'stoch':
            i = np.random.randint(0, len(self.training_set))
            return self.training_set[i], self.training_set_labels[i]
        elif mode == 'minibatch':
            try:
                i = np.random.randint(0, len(self.training_set), batchsize)
                return self.training_set[i], self.training_set_labels[i]
            except:
                raise AttributeError("Need to input a batchsize when using 'minibatch'")
        elif mode == 'batch':
            return self.training_set, self.training_set_labels
        else:
            raise ValueError("Invalid mode variable.  Input: {}; required: 'stoch', 'minibatch', or 'batch'.".format(mode))