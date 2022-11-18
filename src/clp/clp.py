import warnings
from typing import Optional, Sequence
import numpy as np
import os

import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torch.nn.functional import one_hot
from torch.linalg import norm



from avalanche.training.plugins import SupervisedPlugin
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.models import FeatureExtractorBackbone


class CLP(SupervisedTemplate):
    """Deep Streaming Linear Discriminant Analysis.

    This strategy does not use backpropagation.
    Minibatches are first passed to the pretrained feature extractor.
    The result is processed one element at a time to fit the LDA.
    Original paper:
    "Hayes et. al., Lifelong Machine Learning with Deep Streaming Linear
    Discriminant Analysis, CVPR Workshop, 2020"
    https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf
    """

    def __init__(
            self,
            clvq_model,
            n_protos,
            bmu_metric,
            criterion,
            alpha_start,
            tau_alpha_decay,
            tau_alpha_growth,
            sim_th,
            w_max,
            w_min,
            input_size,
            num_classes,
            max_allowed_mistakes,
            output_layer_name=None,
            train_epochs: int = 1,
            train_mb_size: int = 1,
            eval_mb_size: int = 1,
            device="cpu",
            plugins: Optional[Sequence["SupervisedPlugin"]] = None,
            evaluator=default_evaluator,
            eval_every=-1,
    ):
        """Init function for the SLDA model.

        :param slda_model: a PyTorch model
        :param criterion: loss function
        :param output_layer_name: if not None, wrap model to retrieve
            only the `output_layer_name` output. If None, the strategy
            assumes that the model already produces a valid output.
            You can use `FeatureExtractorBackbone` class to create your custom
            SLDA-compatible model.
        :param input_size: feature dimension
        :param num_classes: number of total classes in stream
        :param train_mb_size: batch size for feature extractor during
            training. Fit will be called on a single pattern at a time.
        :param eval_mb_size: batch size for inference
        :param shrinkage_param: value of the shrinkage parameter
        :param streaming_update_sigma: True if sigma is plastic else False
            feature extraction in `self.feature_extraction_wrapper`.
        :param plugins: list of StrategyPlugins
        :param evaluator: Evaluation Plugin instance
        :param eval_every: run eval every `eval_every` epochs.
            See `BaseTemplate` for details.
        """

        if plugins is None:
            plugins = []

        clvq_model = clvq_model.eval()
        if output_layer_name is not None:
            clvq_model = FeatureExtractorBackbone(
                    clvq_model.to(device), output_layer_name
            ).eval()

        super().__init__(
                clvq_model,
                None,
                criterion,
                train_mb_size,
                train_epochs,
                eval_mb_size,
                device=device,
                plugins=plugins,
                evaluator=evaluator,
                eval_every=eval_every,
        )
        
        self.device = device
        
        # CLVQ parameters
        self.bmu_metric = bmu_metric
        self.alpha_start = alpha_start
        self.alpha_dc = np.e ** (-1 / tau_alpha_decay)
        self.alpha_gr = np.e ** (-1 / tau_alpha_growth)
        self.max_allowed_mistakes = max_allowed_mistakes
        self.num_classes = num_classes
        self.sim_th = sim_th
        self.w_max = w_max
        self.w_min = w_min
        
        # setup weights for CLVQ
        self.n_protos = n_protos
        self.prototypes = torch.zeros(n_protos, input_size).to(device)
        self.proto_labels = num_classes * torch.ones((n_protos, 1), dtype=int)
        self.alphas = torch.ones((n_protos, 1)).to(self.device)
        self.sims = []

    def forward(self, return_features=False):
        """Compute the model's output given the current mini-batch."""
        self.model.eval()
        feat = self.model(self.mb_x).flatten(start_dim=1, end_dim=-1)
        out = one_hot(self.predict(feat), self.num_classes+1).squeeze(1).float().to(self.device)
        
        if return_features:
            return out, feat
        else:
            return out

    def training_epoch(self, **kwargs):
        """
        Training epoch.
        :param kwargs:
        :return:
        """
        for _, self.mbatch in enumerate(self.dataloader):
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.loss = torch.tensor([1], dtype=float).to(self.device)

            # Forward
            self._before_forward(**kwargs)
            # compute output on entire minibatch
            self.mb_output, feats = self.forward(return_features=True)
            self._after_forward(**kwargs)

            # Loss & Backward
            # self.loss += self._criterion(self.mb_output, self.mb_y)
            self.loss += 1

            # Optimization step
            self._before_update(**kwargs)
            # process one element at a time
            for f, y in zip(feats, self.mb_y):
                # f = f.squeeze()
                self.fit(f, y.unsqueeze(0))
            self._after_update(**kwargs)

            self._after_training_iteration(**kwargs)

    def make_optimizer(self):
        """Empty function.
        Deep SLDA does not need a Pytorch optimizer."""
        pass

    @torch.no_grad()
    def fit(self, x, y):
        """
        Fit the SLDA model to a new sample (x,y).
        :param x: a torch tensor of the input data (must be a vector)
        :param y: a torch tensor of the input label
        :return: None
        """
        n_mistakes = 0
        mistake = False
        y = torch.tensor(int(y))
        while True:
            
            bmu, bmu_ind = self.get_best_matching_unit(x)
            
            # Novel instance --> Allocate
            # if no winner, because all similarities are below the given threshold, then allocate
            if bmu_ind == None:
                self._allocate(x, y)
                break
            
            # Novel label --> Allocate
            if y not in self.proto_labels:
                self._allocate(x, y)
                break
                
            # Calculate Error
            if self.bmu_metric == 'euclidean':
                error = x - bmu
                
            elif self.bmu_metric == 'cosine':
                bmu_n = norm(bmu, 2)
                x_n = norm(x, 2)
                error = bmu_n * x_n * (x / (bmu_n * x_n) - torch.dot(bmu.squeeze(), x)/(bmu_n**3 * x_n)*bmu)
                
            # print("winner label: ", self.proto_labels[bmu_ind])
            # print("Target label: ", y)
            
            # If winner not assigned to a label, then assign it to
            # the training instance's label
            if self.proto_labels[bmu_ind] == self.num_classes:
                # print("Unsupervised llocating...")
                mistake = False
                self.proto_labels[bmu_ind] = y
                self.prototypes[bmu_ind] += self.alphas[bmu_ind] * error
                self.alphas[bmu_ind] = self.alpha_dc * self.alphas[bmu_ind]
                # self._bound_weights(bmu_ind)
                break

            # Update the winner based on its inference
            # If CORRECT prediction
            elif self.proto_labels[bmu_ind] == y:
                mistake = False
                # print("Correct")
                # print(self.alphas[bmu_ind])
                self.prototypes[bmu_ind] += self.alphas[bmu_ind] * error
                self.alphas[bmu_ind] = self.alpha_dc * self.alphas[bmu_ind]
                # self._bound_weights(bmu_ind)
                break

            # if INCORRECT prediction 
            else:
                # print("Mistake")
                mistake = True
                n_mistakes += 1
                self.prototypes[bmu_ind] -= self.alphas[bmu_ind] * error
                self.alphas[bmu_ind] = min(
                        (self.alphas[bmu_ind] * self.alpha_gr),
                        self.alpha_start)
                
                # self._bound_weights(bmu_ind)


                if n_mistakes < self.max_allowed_mistakes:
                    # print("First Mistake trying again...")
                    continue

                # Allocate a new non-winning prototype if maximum number of allowed
                # mistakes are passed
                elif n_mistakes == self.max_allowed_mistakes:
                    self._allocate(x, y)
                    break
    
    def _allocate (self, x, y):
        # print("Mistake again, allocating...")
        similarities = self._calc_similarities(x)
        similarities[self.proto_labels<self.num_classes] -= 100000
        bmu_ind = torch.argmax(similarities, dim=0)
        self.proto_labels[bmu_ind] = y
        error = x - self.prototypes[bmu_ind]
        self.prototypes[bmu_ind] += self.alphas[bmu_ind] * error
        self.alphas[bmu_ind] = self.alpha_dc * self.alphas[bmu_ind]
        # self._bound_weights(bmu_ind)
        
    def _bound_weights(self, bmu_ind):
        over_w_max_inds = (self.prototypes[bmu_ind] > self.w_max)
        below_w_min_inds = (self.prototypes[bmu_ind] < self.w_min)
        
        if torch.count_nonzero(over_w_max_inds) > 0:
            inds = over_w_max_inds.nonzero().squeeze()
            print(inds)
            self.prototypes[bmu_ind][inds] = self.w_max
        if torch.count_nonzero(below_w_min_inds) > 0:    
            inds = below_w_min_inds.nonzero().squeeze()
            print(inds)
            self.prototypes[bmu_ind][inds] = self.w_min
        
    @torch.no_grad()
    def predict(self, x):
        """
        Make predictions on test data X.
        :param X: a torch tensor that contains N data samples (N x d)
        :param return_probas: True if the user would like probabilities instead
        of predictions returned
        :return: the test predictions or probabilities
        """

        # Compute the winner prototype, return this and its index
        bmu, bmu_ind = self.get_best_matching_unit(x)

        # Find the predicted labels
        preds = self.proto_labels[bmu_ind]

        # return predictions
        return preds

    # Locate the best matching unit
    def get_best_matching_unit(self, x):

        ind = 0
        
        similarities = self._calc_similarities(x)
        
        (max_sim, ind) = torch.max(similarities, dim=0, keepdim=False)
        self.sims.append(max_sim)
        if torch.count_nonzero(max_sim>self.sim_th) == 0:
                return None, None
        else:
            return self.prototypes[ind], ind
            
#         if self.bmu_metric == 'euclidean':
#             euc_dist = torch.cdist(self.prototypes, x, p=2)
#             (min_dist, ind) = torch.min(euc_dist, dim=0, keepdim=False)
#             self.sims.append(min_dist)
#             if torch.count_nonzero(min_dist > self.sim_th) > 0:
#                 return None, None
            
#         elif self.bmu_metric == 'dot_product':
#             dp = torch.mm(self.prototypes, x.T)
#             if torch.count_nonzero(dp>self.sim_th) > 0:
#                 ind = torch.argmax(dp, dim=0)
#                 self.sims.append(torch.max(dp, dim=0))
#             else:
#                 return None, None
            
#         elif self.bmu_metric == 'cosine':
#             ind = torch.argmax(pairwise_cosine_similarity(self.prototypes, x),
#                                dim=0)
        
        
    
    def _calc_similarities(self, x):
        
        similarities = 0
        
        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        if self.bmu_metric == 'euclidean':
            similarities = -torch.cdist(self.prototypes, x, p=2)
            
        elif self.bmu_metric == 'dot_product':
            similarities = torch.mm(self.prototypes, x.T)          
            
        elif self.bmu_metric == 'cosine':
            similarities = pairwise_cosine_similarity(self.prototypes, x)
            
        return similarities

    def init_prototypes_from_data(self, data):
        
        num_bins = 100
        x = data[0,:].cpu()
        x[np.absolute(x)<0.02]=0

        counts, bins = np.histogram(x, bins=num_bins)
        bins = bins[:-1]
        probs = counts/float(counts.sum())
        
        self.prototypes = torch.tensor(np.random.choice(bins, size=(self.n_protos, data.shape[1]), replace=True, p=probs)).to(self.device)
        # torch.abs(torch.round(
        #     torch.normal(mean=torch.mean(data, axis=0).tile(self.n_protos, 1),
        #                  std=torch.std(data, axis=0).tile(self.n_protos, 1),
        #                  out=self.prototypes), decimals=1))

        return self.prototypes
    
    def criterion(self):
        """Loss function."""
        return 0

    def fit_base(self, X, y):
        """
        Fit the SLDA model to the base data.
        :param X: an Nxd torch tensor of base initialization data
        :param y: an Nx1-dimensional torch tensor of the associated labels for X
        :return: None
        """
        print("\nFitting Base...")

        # update class means
        for k in torch.unique(y):
            self.muK[k] = X[y == k].mean(0)
            self.cK[k] = X[y == k].shape[0]
        self.num_updates = X.shape[0]

        print("\nEstimating initial covariance matrix...")
        from sklearn.covariance import OAS

        cov_estimator = OAS(assume_centered=True)
        cov_estimator.fit((X - self.muK[y]).cpu().numpy())
        self.Sigma = (
                torch.from_numpy(cov_estimator.covariance_).float().to(
                    self.device)
        )

    def save_model(self, save_path, save_name):
        """
        Save the model parameters to a torch file.
        :param save_path: the path where the model will be saved
        :param save_name: the name for the saved file
        :return:
        """
        # grab parameters for saving
        d = dict()
        d["muK"] = self.muK.cpu()
        d["cK"] = self.cK.cpu()
        d["Sigma"] = self.Sigma.cpu()
        d["num_updates"] = self.num_updates

        # save model out
        torch.save(d, os.path.join(save_path, save_name + ".pth"))

    def load_model(self, save_path, save_name):
        """
        Load the model parameters into StreamingLDA object.
        :param save_path: the path where the model is saved
        :param save_name: the name of the saved file
        :return:
        """
        # load parameters
        d = torch.load(os.path.join(save_path, save_name + ".pth"))
        self.muK = d["muK"].to(self.device)
        self.cK = d["cK"].to(self.device)
        self.Sigma = d["Sigma"].to(self.device)
        self.num_updates = d["num_updates"]

    def _check_plugin_compatibility(self):
        """Check that the list of plugins is compatible with the template.

        This means checking that each plugin impements a subset of the
        supported callbacks.
        """
        # I don't know if it's possible to do it in Python.
        ps = self.plugins

        def get_plugins_from_object(obj):
            def is_callback(x):
                return x.startswith("before") or x.startswith("after")

            return filter(is_callback, dir(obj))

        cb_supported = set(get_plugins_from_object(self.PLUGIN_CLASS))
        cb_supported.remove("before_backward")
        cb_supported.remove("after_backward")
        for p in ps:
            cb_p = set(get_plugins_from_object(p))

            if not cb_p.issubset(cb_supported):
                warnings.warn(
                        f"Plugin {p} implements incompatible callbacks for template"
                        f" {self}. This may result in errors. Incompatible "
                        f"callbacks: {cb_p - cb_supported}",
                )
                return

