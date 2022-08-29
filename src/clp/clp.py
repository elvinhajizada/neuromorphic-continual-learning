import warnings
from typing import Optional, Sequence
import numpy as np
import os

import torch
from torchmetrics.functional import pairwise_cosine_similarity
from torch.nn.functional import one_hot


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

        # CLVQ parameters
        self.bmu_metric = bmu_metric
        self.alpha_start = alpha_start
        self.alpha_dc = np.e ** (-1 / tau_alpha_decay)
        self.alpha_gr = np.e ** (-1 / tau_alpha_growth)
        self.max_allowed_mistakes = max_allowed_mistakes
        self.num_classes = num_classes

        # setup weights for CLVQ
        self.n_protos = n_protos
        self.prototypes = torch.zeros(n_protos, input_size)
        self.proto_labels = num_classes * torch.ones((n_protos, 1), dtype=int)
        self.alphas = torch.ones((n_protos, 1)).to(self.device)
        self.w_max = 350
        self.w_min = 0

    def forward(self, return_features=False):
        """Compute the model's output given the current mini-batch."""
        self.model.eval()
        feat = self.model(self.mb_x).flatten(start_dim=1, end_dim=-1)
        out = one_hot(self.predict(feat), self.num_classes+1).squeeze(1).float()
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

            self.loss = torch.tensor([1], dtype=float)

            # Forward
            self._before_forward(**kwargs)
            # compute output on entire minibatch
            self.mb_output, feats = self.forward(return_features=True)
            self._after_forward(**kwargs)

            # Loss & Backward
            self.loss += self._criterion(self.mb_output, self.mb_y)
            # self.loss += 1

            # Optimization step
            self._before_update(**kwargs)
            # process one element at a time
            for f, y in zip(feats, self.mb_y):
                f = f.squeeze()
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
            error = x - bmu
            # print("winner id: ", bmu_ind, "error: ", np.mean(error))

            # If winner not assigned to a label, then assign it to
            # the training instance's label
            if self.proto_labels[bmu_ind] == self.num_classes:
                mistake = False
                self.proto_labels[bmu_ind] = y
                self.alphas[bmu_ind] = self.alpha_dc * self.alphas[bmu_ind]

            # Update the winner based on its inference
            # If CORRECT prediction
            if self.proto_labels[bmu_ind] == y:
                mistake = False
                # print("Correct")

                self.prototypes[bmu_ind] += self.alphas[bmu_ind] * error
                self.alphas[bmu_ind] = self.alpha_dc * self.alphas[bmu_ind]

            # if INCORRECT prediction
            else:
                # print("Mistake")
                mistake = True
                n_mistakes += 1
                # print(self.alphas[bmu_ind])
                self.prototypes[bmu_ind] -= self.alphas[bmu_ind] * x
                self.alphas[bmu_ind] = min(
                        (self.alphas[bmu_ind] * self.alpha_gr),
                        self.alpha_start)
                # n_train_errors += 1
                # print("After update: ", self.prototypes[bmu_ind])

            # Bound weights between [w_min, w_max]
            # self.prototypes[bmu_ind][self.prototypes[bmu_ind] >
            #                          self.w_max] = self.w_max
            #
            # self.prototypes[bmu_ind][self.prototypes[bmu_ind] <
            #                          self.w_min] = self.w_min

            # if self.rec_alpha_evolve:
            #     for i in range(self.n_protos):
            #         self.alpha_evolve[i].append(self.alphas[i])

            if mistake and n_mistakes == 1:
                # n_train_errors += 1
                # print("First Mistake:", self.proto_labels[bmu_ind])
                continue

            elif mistake and n_mistakes <= self.max_allowed_mistakes:
                # print("Second Mistake:", self.proto_labels[bmu_ind])
                continue

            # Allocate a new non-winning prototype if maximum number of allowed
            # mistakes are passed
            elif mistake and n_mistakes == (self.max_allowed_mistakes + 1):

                next_proto_ind = np.where(self.proto_labels == self.num_classes)[0][0]
                self.proto_labels[next_proto_ind] = y
                error = x - self.prototypes[next_proto_ind]
                self.prototypes[next_proto_ind] += self.alphas[
                                                       next_proto_ind] * error
                self.alphas[next_proto_ind] = self.alpha_dc * self.alphas[
                    bmu_ind]
                # print("New Proto:", next_proto_ind)
            else:
                break

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

        if len(x.shape) == 1:
            x = x.unsqueeze(dim=0)

        if self.bmu_metric == 'euclidean':
            ind = torch.argmin(torch.cdist(self.prototypes, x, p=2), dim=0)
        elif self.bmu_metric == 'dot_product':
            ind = torch.argmax(torch.mm(self.prototypes, x.T), dim=0)
        elif self.bmu_metric == 'cosine':
            ind = torch.argmax(pairwise_cosine_similarity(self.prototypes, x),
                               dim=0)

        return self.prototypes[ind], ind

    def init_prototypes_from_data(self, data):
        torch.abs(torch.round(
            torch.normal(mean=torch.mean(data, axis=0).tile(self.n_protos, 1),
                         std=torch.std(data, axis=0).tile(self.n_protos, 1),
                         out=self.prototypes), decimals=1))

        return self.prototypes

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

