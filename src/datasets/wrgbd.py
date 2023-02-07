import glob
from tqdm import tqdm
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pickle

from sklearn.model_selection import train_test_split

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import torch
from avalanche.benchmarks.datasets.downloadable_dataset import SimpleDownloadableDataset
from torchvision.transforms.functional import pad


class WRGBD(Dataset):
    def __init__(self, root_dir, obj_level=True, transform=None, train_test_split='random', subset='train', depth_mask = False,
                 subsampling_factor=1, test_size=0.2, seed=1234, read_target_lists=False, save_target_lists=False):
        """
        this function builds a data frame which contains the path to image
        and the tag/object name using the prefix of the image name
        """
        
        self.root_dir = root_dir
        self.transform = transform
        self.targets_fname = "target_lists.pkl"
        self.depth_mask = depth_mask
        
        self.targets = []
        self.paths = []
        self.mask_paths = []
        self.cat_targets = []
        self.obj_targets = []
        self.view_targets = []
        self.pose_targets = []
        self.cat_names = []
        
        self.n_obj_per_cat = []
        
        self.pil2tensor = transforms.ToTensor()
        self.tensor2pil = transforms.ToPILImage()
        np.random.seed(seed)
        
        if read_target_lists:
            # pickle load the list object back in (senses protocol)
            with open(self.targets_fname, "rb") as fin:
                target_lists = pickle.load(fin)
                
            [self.cat_targets, self.obj_targets, self.view_targets, self.pose_targets, self.paths, self.mask_paths] = target_lists
        
            print("All target lists are loaded")
            
        else:
            dataset_dir = self.root_dir + "/*/"
            cat_dirs = glob.glob(dataset_dir)
            last_obj_ind = 0

            for cat_dir in cat_dirs:
                
                cat_name = cat_dir.split("/")[-2]
                obj_dirs = glob.glob(cat_dir+"*/")
                
                for obj_dir in obj_dirs:
                    
                    files = glob.glob(obj_dir+"**/*_crop.png", recursive=True)
                    
                    for file in files:
                        target_vals = file.split("/")[-1].split("_")
                        self.cat_targets.append(cat_name)
                        self.obj_targets.append(last_obj_ind)
                        self.view_targets.append(int(target_vals[-3]))
                        self.pose_targets.append(int(target_vals[-2]))
                        self.paths.append(file)
                        self.mask_paths.append(file[:-8]+'mask'+file[-8:])
                        
                    last_obj_ind += 1
                    
                n_obj = len(obj_dirs)
                self.n_obj_per_cat.append(n_obj)
                # print(str(last_obj_ind) + " object are loaded")

                self.cat_names.append(cat_dir.split("/")[-2])
            
            cat_map_dict = dict(zip(self.cat_names, range(len(self.cat_names))))
            self.cat_targets = [cat_map_dict[cat_name] for cat_name in self.cat_targets] 
            print("All target lists are created")
            
            
        self.paths = np.array(self.paths)
        self.mask_paths = np.array(self.mask_paths)
        self.obj_targets = np.array(self.obj_targets)
        self.cat_targets = np.array(self.cat_targets)
        self.view_targets = np.array(self.view_targets)
        
        if save_target_lists:
            self._write_target_lists_to_file(self.targets_fname)
        
        if obj_level:
            self.targets = self.obj_targets.copy()
        else:
            self.targets = self.cat_targets.copy()
            
        # sorted_inds = np.argsort(self.targets)
        # self.targets = self.targets[sorted_inds]
        # self.paths = self.paths[sorted_inds]
        
        ## Choosing some objects
        # obj_slices = np.concatenate([np.arange(ind*72,(ind+1)*72) for ind in obj_list])
        # n_classes = len(obj_list)
        # # self.targets = self.targets[obj_slices]
        # self.targets = self.targets[:72*n_classes]
        # self.paths = self.paths[obj_slices]
        
        if subsampling_factor != 1:
            ss_inds = list(range(0,len(self.targets),subsampling_factor))
            self.targets = self.targets[ss_inds]
            self.paths = self.paths[ss_inds]
            self.mask_paths = self.mask_paths[ss_inds]
            self.obj_targets = self.obj_targets[ss_inds]
            self.cat_targets = self.cat_targets[ss_inds]
            self.view_targets = self.view_targets[ss_inds]
            
        if subset!='all':
            
            if train_test_split == 'custom':
                
                if obj_level:
                    test_indices = np.where(self.view_targets==2)[0]
                    train_indices = np.delete(np.arange(len(self.targets)), test_indices)

                else:
                    test_obj_ind = []
                    cum_obj_count = [0] + np.cumsum(self.n_obj_per_cat).tolist()
                    for i in range(len(self.n_obj_per_cat)):
                        test_obj_ind.append(np.random.choice(np.arange(cum_obj_count[i], cum_obj_count[i+1])))

                    test_indices = np.where(np.isin(self.obj_targets,test_obj_ind))[0]
                    train_indices = np.delete(np.arange(len(self.targets)), test_indices)
                    
            else:      
                # Random train-test sets
                train_indices, test_indices, _, _ = train_test_split(range(len(self.targets)),
                                                                    self.targets,
                                                                    stratify=self.targets,
                                                                    test_size=test_size,
                                                                    random_state=seed,
                                                                    shuffle=True)
            
            if subset=='train':
                self.targets = self.targets[train_indices]
                self.paths = self.paths[train_indices]
                self.mask_paths = self.mask_paths[train_indices]
                self.obj_targets = self.obj_targets[train_indices]
                self.cat_targets = self.cat_targets[train_indices]
            elif subset=='test':
                self.targets = self.targets[test_indices]
                self.paths = self.paths[test_indices]
                self.mask_paths = self.mask_paths[test_indices]
                self.obj_targets = self.obj_targets[test_indices]
                self.cat_targets = self.cat_targets[test_indices]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):

        img_path, img_label  = self.paths[idx], self.targets[idx]
        img = Image.open(img_path)
        
        if self.depth_mask:
            try:
                img_mask = self.pil2tensor(Image.open(self.mask_paths[idx]))
                img = self.tensor2pil(self.pil2tensor(img) * img_mask)
            except:
                pass
            
        if self.transform:
            img = self.transform(img)
        
        return img, img_label
    
    def _write_target_lists_to_file(self, targets_fname = "target_lists.pkl"):
        target_lists = [self.cat_targets,
                        self.obj_targets,
                        self.view_targets,
                        self.pose_targets,
                        self.paths,
                        self.mask_paths]
        
        with open(targets_fname, "wb") as fout:
            # default protocol is zero
            # -1 gives highest prototcol and smallest data file size
            pickle.dump(target_lists, fout, protocol=-1)


