import numpy as np
from sklearn.utils import shuffle


class LVQ1(object):
    def __init__(self, bmu_metric='euclidean', random_protos=True,
                 init_protos=None, alpha_decay='linear',
                 n_protos_per_class=4, n_class=3, record_pt_evolve=False):
        self.bmu_metric = bmu_metric
        self.n_protos = n_protos_per_class * n_class
        self.prototypes = init_protos
        self.proto_labels = np.repeat(np.arange(n_class), n_protos_per_class)
        self.sum_errors = []
        self.random_protos = random_protos
        self.train_errors = 0
        self.test_errors = 0
        self.alphas = 0
        self.alpha_decay = alpha_decay
        self.proto_evolve = [[] for _ in range(n_class)]
        self.w_max = 150
        self.w_min = -200
        self.record_pt_evolve = record_pt_evolve

    # Locate the best matching unit
    def get_best_matching_unit(self, vector):
        ind = 0
        if self.bmu_metric == 'euclidean':
            ind = np.argmin(np.sum((self.prototypes - vector) ** 2, axis=1))
        elif self.bmu_metric == 'dot_product':
            ind = np.argmax(np.dot(self.prototypes, vector))
        return self.prototypes[ind], ind

    def set_prototypes(self,  prototypes=None):
        self.prototypes = prototypes

    # Create a random proto vector
    def init_prototypes_from_data(self, data):
        self.prototypes = np.around(np.random.normal(np.mean(data, axis=0),
                                                     np.std(data, axis=0),
                                                     size=(self.n_protos,
                                                           len(data[0]))),
                                    1)

        return self.prototypes

    # # Train a set of proto vectors
    def train_prototypes(self, x_train, y_train, x_test, y_test,
                         alpha_start, n_epochs, test_each_epoch=False):
        np.set_printoptions(precision=2, suppress=True)

        self.train_errors = np.zeros((n_epochs, 1))
        self.test_errors = np.zeros((n_epochs, 1))

        if self.random_protos:
            self.init_prototypes_from_data(x_train)

        print("Initial prototypes:\n", self.prototypes)

        epoch_list = np.linspace(0, 1, n_epochs)
        if self.alpha_decay == 'hill':
            self.alphas = alpha_start / (1 + (epoch_list / 0.5) ** 4)
        else:
            self.alphas = np.linspace(alpha_start, 0, n_epochs)

        for epoch in range(n_epochs):
            n_train_errors = 0
            mse = 0.0
            x_train, y_train = shuffle(x_train, y_train)
            for ind, vec in enumerate(x_train):

                bmu, bmu_ind = self.get_best_matching_unit(vec)
                error = vec - bmu
                mse += np.sqrt(np.sum(error**2))
                if self.record_pt_evolve:
                    self.proto_evolve[bmu_ind].append([epoch, bmu.copy()])

                if self.proto_labels[bmu_ind] == y_train[ind]:
                    self.prototypes[bmu_ind] += self.alphas[epoch] * error

                else:
                    self.prototypes[bmu_ind] -= self.alphas[epoch] * error
                    n_train_errors += 1

                self.prototypes[bmu_ind][self.prototypes[bmu_ind] >
                                         self.w_max] = self.w_max

                self.prototypes[bmu_ind][self.prototypes[bmu_ind] <
                                         self.w_min] = self.w_min

            if test_each_epoch:
                test_acc = self.predict(x_test, y_test)
                self.test_errors[epoch] = 1 - test_acc

            train_error = n_train_errors / x_train.shape[0]
            self.sum_errors.append(mse)
            self.train_errors[epoch] = train_error

            print(
                    '>epoch=%d, lrate=%.3f, error=%.3f, tr_err=%.3f, '
                    'test_err=%.3f' %
                    (epoch,
                     self.alphas[epoch],
                     mse,
                     train_error,
                     self.test_errors[epoch]))

        self.sum_errors = np.array(self.sum_errors)
        self.prototypes = np.around(self.prototypes, 2)
        print("Final prototypes:\n", self.prototypes)
        return self.prototypes
    
    def predict(self, x_test, y_test):
        n_test_errors = 0
        for ind, vec in enumerate(x_test):
            bmu, bmu_ind = self.get_best_matching_unit(vec)
            # print("Inferred Label:", self.proto_labels[bmu_ind],
            #       "\nActual Label:  ", y_test[ind])
            if self.proto_labels[bmu_ind] != y_test[ind]:
                n_test_errors += 1
        acc = 1 - n_test_errors/x_test.shape[0]
        # print("accuracy:", acc)
        return acc
