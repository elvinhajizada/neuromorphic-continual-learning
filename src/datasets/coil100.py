import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torch
from avalanche.benchmarks.datasets.downloadable_dataset import SimpleDownloadableDataset


class Coil100Dataset(Dataset):
    def __init__(self, root_dir, obj_list=np.arange(100), transform=None, size=64, train=True,
                 test_size=0.2, seed=1234):
        """
        this function builds a data frame which contains the path to image
        and the tag/object name using the prefix of the image name
        """
        self.targets = []
        self.paths = []
        self.size = size
        self.root_dir = root_dir
        self.transform = transform

        path = root_dir + '/coil-100/*.png'

        # list files
        files = glob.glob(path)

        for file in tqdm(files):
            self.targets.append(
                int(file.split("/")[-1].split("__")[0].split("j")[1]) - 1)
            self.paths.append(file)

        self.targets = np.array(self.targets)
        self.paths = np.array(self.paths)
        
        sorted_inds = np.argsort(self.targets)
        self.targets = self.targets[sorted_inds]
        self.paths = self.paths[sorted_inds]
        
        obj_slices = np.concatenate([np.arange(ind*72,(ind+1)*72) for ind in obj_list])
        n_classes = len(obj_list)
        # self.targets = self.targets[obj_slices]
        self.targets = self.targets[:72*n_classes]
        self.paths = self.paths[obj_slices]
        
        train_indices, test_indices, _, _ = train_test_split(
            range(len(self.targets)),
            self.targets,
            stratify=self.targets,
            test_size=test_size,
            random_state=seed,
            shuffle=True)
        if train:
            self.targets = self.targets[train_indices]
            self.paths = self.paths[train_indices]
        else:
            self.targets = self.targets[test_indices]
            self.paths = self.paths[test_indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        img_path, img_label = self.paths[idx], self.targets[idx]
        img = Image.open(img_path)
        if self.transform:
            img = transforms.Resize(size=[self.size, self.size])(img)
            img = self.transform(img)
        return img, img_label


def show_batch(dl):
    for images, labels, _ in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break
