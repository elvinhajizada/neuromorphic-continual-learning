from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import os

import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torch.nn.functional import one_hot
from torch.linalg import norm
torch.set_printoptions(precision=2)

class CLP(BaseEstimator, ClassifierMixin):
    
    def __init__(
        self, 
        sim_metric='cosine',
        n_protos=10,
        num_classes=2,
        alpha_init=1,
        sim_th_init=0.5, 
        max_allowed_mistakes=3,                  
        k_hit=1, 
        k_miss=1, 
        tau_sim_th_pos=100, 
        tau_sim_th_neg=100,
        k_sim_th_pos=1, 
        k_sim_th_neg=1,
        device='cpu', 
        verbose=1):
        
        self.sim_metric = sim_metric
        self.n_protos = n_protos
        self.num_classes = num_classes
        self.alpha_init = alpha_init
        self.sim_th_init = sim_th_init
        self.max_allowed_mistakes = max_allowed_mistakes
        self.k_hit = k_hit
        self.k_miss = k_miss
        self.tau_sim_th_pos = tau_sim_th_pos
        self.tau_sim_th_neg = tau_sim_th_neg
        self.k_sim_th_pos = k_sim_th_pos
        self.k_sim_th_neg = k_sim_th_neg
        self.device = device
        self.verbose = verbose
        
    def fit(self, X, y):
        y = y.cpu()
        n_samples, n_features = X.shape
        
        self.prototypes_ = 1*torch.ones(self.n_protos, n_features).to(self.device)
        self.proto_labels_ = self.num_classes * torch.ones((self.n_protos, 1), dtype=int)
        self.alphas_ = self.alpha_init*torch.ones((self.n_protos, 1)).to(self.device)
        self.sim_th_ = self.sim_th_init*torch.ones((self.n_protos, 1)).to(self.device)
        self.hits_ = torch.ones((self.n_protos, 1)).to(self.device)
        self.misses_ = torch.ones((self.n_protos, 1)).to(self.device)
        self.n_ignored_inst_ = 0    # Num of ignored instances
        self.classes_ = [self.num_classes]  # "Unknown" class already in the list
        
        for x, y in zip(X, y):
            
            self.mistaken_proto_inds_ = [] # inds of protos that made incorrect inferences for current sample
            
            if self.sim_metric == 'cosine': # normalize x, if we use cosine similarity
                x = x / norm(x, 2)
                
            n_mistakes = 0
            y = torch.tensor(int(y))    
            
            while True:
                
                bmu_ind, max_sim = self._get_best_matching_unit(x)
                bmu_ind = bmu_ind.item()
                max_sim = max_sim.item()

                if y.item() not in self.classes_:
                    self.classes_.append(y.item())
                    if self.verbose >= 1:
                        print("Novel Label!")
                    self._allocate(x, y)
                    break

                # Novel instance --> Allocate
                # if no winner, because all similarities are below the given threshold, then allocate
                if bmu_ind == -1:
                    if self.verbose >= 1:
                        print("Novel Instance!")
                        print(max_sim, bmu_ind)
                    self._allocate(x, y)
                    break

                # Get the winner prototype
                bmu = self.prototypes_[bmu_ind]

                # Calculate Error
                if self.sim_metric == 'euclidean':
                    error = x - bmu

                elif self.sim_metric == 'cosine':
                    error = x

                # If winner not assigned to a label, then assign it to
                # the training instance's label
                if self.proto_labels_[bmu_ind] == self.num_classes:
                    if self.verbose >= 1:
                        print("Unsupervised allocating...")
                    self.proto_labels_[bmu_ind] = y
                    self.prototypes_[bmu_ind] += self.alphas_[bmu_ind] * error

                    # update the threshold towards max_sim-eps
                    self.sim_th_[bmu_ind] = self.sim_th_[bmu_ind] + \
                    (self.k_sim_th_pos*max_sim - self.sim_th_[bmu_ind]) / self.tau_sim_th_pos

                    self.hits_[bmu_ind]+=1
                    self.alphas_[bmu_ind] = self.misses_[bmu_ind]/self.hits_[bmu_ind]
                    break

                # Update the winner based on its inference
                # If CORRECT prediction
                elif self.proto_labels_[bmu_ind] == y:
                    if self.verbose == 2:
                        print("Correct")
                        
                    self.prototypes_[bmu_ind] += self.alphas_[bmu_ind] * error

                    # update the threshold towards max_sim-eps
                    self.sim_th_[bmu_ind] = self.sim_th_[bmu_ind] + \
                    (self.k_sim_th_pos*max_sim - self.sim_th_[bmu_ind]) / self.tau_sim_th_pos             

                    self.hits_[bmu_ind] += self.k_hit
                    self.alphas_[bmu_ind] = self.misses_[bmu_ind]/self.hits_[bmu_ind]
                    # self._bound_weights(bmu_ind)
                    break

                # if INCORRECT prediction 
                else:
                    n_mistakes += 1

                    self.mistaken_proto_inds_.append(bmu_ind)
                    # update the mistaken prototype
                    self.prototypes_[bmu_ind] -= self.alphas_[bmu_ind] * error

                    # update the threshold of this prototype
                    self.sim_th_[bmu_ind] = self.sim_th_[bmu_ind] + \
                    (self.k_sim_th_neg*max_sim - self.sim_th_[bmu_ind]) / self.tau_sim_th_neg


                    self.misses_[bmu_ind] += self.k_miss
                    self.alphas_[bmu_ind] = self.misses_[bmu_ind]/self.hits_[bmu_ind]

                    if self.verbose >= 1:
                        print("Mistaken prototype:", self.proto_labels_[bmu_ind].item(), bmu_ind)
                        print("sim_th & alpha:", self.sim_th_[bmu_ind].item(), self.alphas_[bmu_ind].item())

                    # If more misses than hits, then forget this prototype, reset it
                    # if self.alphas_[bmu_ind] > 1: 
                    #     self._forget(bmu_ind)

                    # Try again if you have not checked all top matches
                    if n_mistakes < self.max_allowed_mistakes:
                        # print("First Mistake trying again...")
                        continue

                    # Ignore or allocate a new non-winning prototype if maximum number of allowed
                    # mistakes are passed
                    # TODO: decide to allocate or ignore
                    elif n_mistakes == self.max_allowed_mistakes:
                        if self.verbose >= 1:
                            print("Ignoring the instance")
                        self.n_ignored_inst_ += 1
                        # self._allocate(x, y) 
                        break
        return self
    
    def predict(self, x):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """

        # Compute the winner prototype, return this and its index
        bmu_inds, _ = self._get_best_matching_unit(x) 
        # Find the predicted labels
        preds = self.proto_labels_[bmu_inds]
        # Infer "Unknown Instance" label (as label == n_classes) as predictions
        preds[bmu_inds==-1] = self.num_classes
        
        # return predictions
        return np.array(preds.cpu())
    
    # Locate the best matching unit
    def _get_best_matching_unit(self, x):

        ind = 0
        
        similarities = self._calc_similarities(x)
        similarities[self.mistaken_proto_inds_] -= 10000
        sims = similarities.clone().detach()
        
        th_passing_check  = torch.gt(sims, self.sim_th_.tile((1, sims.shape[1])))
        sims_sorted, inds_sorted = torch.sort(sims, 0, descending=True)
        th_passing_sorted = torch.gather(th_passing_check, 0, inds_sorted)
        
        bmu_inds = torch.zeros(size=(1, sims.shape[1]))
        max_sims = torch.zeros(size=(1, sims.shape[1]))
        
        for i in range(sims.shape[1]):
            top_th_passing_inds = inds_sorted[th_passing_sorted[:,i],i]
            max_th_passing_sims = sims_sorted[th_passing_sorted[:,i],i]
            if len(top_th_passing_inds) > 0:
                bmu_inds[0,i] = top_th_passing_inds[0]
                max_sims[0,i] = max_th_passing_sims[0]
            else:
                bmu_inds[0,i] = -1
                max_sims[0,i] = 0
        
        inds_sorted = inds_sorted.long()
        top_sims, top_inds = sims_sorted[:5,:], inds_sorted[:5,:]
        if self.verbose == 2:
            for i in range(0,top_sims.shape[1], 3): 
                print("-----------------------------------------------------------")
                print("sims:  ",top_sims[:,i].t().data)
                print("simth: ",self.sim_th_[top_inds[:,i]].t().data)
                print("labels:",self.proto_labels_[top_inds[:,i]].t().data)
                print("alphas:",self.alphas_[top_inds[:,i]].t().data)
        
        bmu_inds = bmu_inds.squeeze()
        max_sims = max_sims.squeeze()
        
        return bmu_inds.long(), max_sims
    
    
    def _calc_similarities(self, x):
        
        similarities = 0
        
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        if self.sim_metric == 'euclidean':
            similarities = -torch.cdist(self.prototypes_, x, p=2)
            
        elif self.sim_metric == 'dot_product':
            similarities = torch.mm(self.prototypes_, x.T)          
            
        elif self.sim_metric == 'cosine':
            similarities = pairwise_cosine_similarity(self.prototypes_, x)
            
        return similarities

    def _allocate (self, x, y):
        # print("Mistake again, allocating...")
        similarities = self._calc_similarities(x)
        similarities[self.proto_labels_<self.num_classes] -= 100000
        
        bmu_ind = torch.argmax(similarities, dim=0)
        self.proto_labels_[bmu_ind] = y
        
        error = x - self.prototypes_[bmu_ind]
        self.prototypes_[bmu_ind] += self.alphas_[bmu_ind] * error
            
        self.hits_[bmu_ind]+=1
        self.alphas_[bmu_ind] = self.misses_[bmu_ind]/self.hits_[bmu_ind]
    
    def init_prototypes_from_data(self, data):
        
        num_bins = 100
        x = data[0,:].cpu()
        x[np.absolute(x)<0.02]=0

        counts, bins = np.histogram(x, bins=num_bins)
        bins = bins[:-1]
        probs = counts/float(counts.sum())
        
        self.prototypes_ = torch.tensor(np.random.choice(bins, size=(self.n_protos, data.shape[1]), replace=True, p=probs)).to(self.device)
        # torch.abs(torch.round(
        #     torch.normal(mean=torch.mean(data, axis=0).tile(self.n_protos, 1),
        #                  std=torch.std(data, axis=0).tile(self.n_protos, 1),
        #                  out=self.prototypes), decimals=1))

        return self.prototypes_