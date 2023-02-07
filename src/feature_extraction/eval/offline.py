import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class OfflineEvaluator(self, classifier='KNN', dataset='WRGBD', n_neighbors=1, bmu_metrics = ["euclidean", "dot_product", "cosine"],
                       max_iter=1000):
    def __init__(self):
        self.classifier = classifier
        self.dataset = dataset
        
        # k-NN params
        self.n_neighbors = n_neighbors if classifier=='KNN' else None
        self.bmu_metrics = bmu_metrics if classifier=='KNN' else ['linear']
        
        # SVM params
        self.max_iter = max_iter if classifier=='SVM' else None
        
    def offline_eval(X, y, n_repeat=1, split='random', test_size=0.4, obj_labels=None,  ojb_level=False, view_labels=None)
        n_repeat = n_repeat
        accs = np.zeros(shape=(len(bmu_metrics),n_repeat))

        X = X.copy()
        y = y.copy()     
            
        for bm, bmu_metric in enumerate(bmu_metrics): 
            for r in range(n_repeat):
                if split == 'custom':
                    train_inds, test_inds = self._gen_w_rgbd_train_test_inds(y, obj_labels, ojb_level, view_labels)
                    X_train = X[train_inds,:]
                    X_test = X[test_inds,:]
                    y_train = y[train_inds]
                    y_test = y[test_inds]

                else split == 'random':
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True)
                    
                if classifier=='KNN':

                    if bmu_metric != "dot_product":
                        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=bmu_metric)
                        knn.fit(X_train, y_train)
                        y_pred_test=knn.predict(X_test)

                    else:
                        sim_X_train = np.matmul(X_train,X_train.T)
                        sim_X_test = np.matmul(X_test,X_train.T)

                        sim_X_train = np.max(sim_X_train) - sim_X_train
                        sim_X_test = np.max(sim_X_test) - sim_X_test

                        knn = KNeighborsClassifier(n_neighbors=1, algorithm="brute",  metric="precomputed")
                        knn.fit(sim_X_train, y_train)

                        y_pred_test=knn.predict(sim_X_test)
                        
                else:
                    clf = svm.LinearSVC(max_iter=self.max_iter)
                    # clf = svm.LinearSVC(max_iter=10000)
                    clf.fit(X_train, y_train)

                    # y_pred_train=clf.predict(X_train)
                    y_pred_test=clf.predict(X_test)

                # Model Accuracy: how often is the classifier correct?
                acc = metrics.accuracy_score(y_test, y_pred_test)        
                accs[bm, r] = acc
                
        return accs

            
    def _gen_w_rgbd_train_test_inds(y, obj_labels=None,  ojb_level=False, view_labels=None):

        n_obj_per_cat = np.array([ 7,  3,  6,  8,  6,  5,  6,  5,  6,  5, 14,  4, 12,  9,  6,  8,  8,
                                   4,  9,  4,  4,  3,  4,  5,  5,  5,  6,  4,  3,  4,  7,  5,  8,  4,
                                   6,  3,  6, 12,  6,  6,  5,  7,  4,  6,  3,  6,  5,  5,  5,  8,  5])

        if obj_level:
            # Leave one (elevation) viewing angle out of three (30, 45, 60) in the object identification experiments

            test_inds = np.where(view_labels==2)[0]
            train_inds = np.delete(np.arange(len(y)), test_inds)

        else:
            # Leave one instance from each category for testing, in the categorization experiments
            # Choose single object instance from each category as test instance

            test_obj_ind = []

            cum_obj_count = [0] + np.cumsum(n_obj_per_cat).tolist()

            for i in range(len(n_obj_per_cat)):
                test_obj_ind.append(np.random.choice(np.arange(cum_obj_count[i], cum_obj_count[i+1])))

            test_inds = np.where(np.isin(obj_labels,test_obj_ind))[0]
            train_inds = np.delete(np.arange(len(y)), test_inds)

            return train_inds, test_inds
