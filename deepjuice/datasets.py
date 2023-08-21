import pandas as pd
import numpy as np
import torch
import os, sys
from PIL import Image
import gdown, tarfile

from torch.utils.data import Dataset, DataLoader

### DeepDiveData Classes -----------------------------------------------

class DeepJuiceData:
    
    def get_sample_stimulus(self, key='stimulus', index = None):
        if index is None:
            index = np.random.randint(self.n_stimuli)

        sample_stimulus = self.stimulus_data['stimulus'][index]
        
        image_exts = ['.jpg','.jpeg','.png','.webp']
        
        if any([ext in sample_stimulus.lower() for ext in image_exts]):
            return Image.open(sample_stimulus)
        
        return sample_stimulus
    
    get_stimulus = get_sample_stimulus
    
    def get_rdm_indices(self, group_vars, index = 'row_number',
                        metadata = None, response_data = None):
        
        if index is None:
            index = self.index_name
        
        if metadata is None:
            metadata = self.metadata.reset_index()
            
        if response_data is None:
            response_data = self.response_data
        
        assert np.array_equal(metadata.index, response_data.index)
        
        if index == 'row_number':
            metadata['row_number'] = metadata.index
        
        return iterative_subset(metadata, index, group_vars)
    
    def download_data(self, path_dir, dataset_name, drive_id):
        print('Downloading data from Google Drive to {}'.format(path_dir))
        tar_file = '{}/{}.tar.bz'.format(path_dir, dataset_name)
        download_url = ('https://drive.google.com/uc?export' + 
                        'uc?export=download&id={}'.format(drive_id))
        
        gdown.download(download_url, quiet = False, output = tar_file)
        extract_tar(tar_file, path_dir, 'neural_data', delete = True)

### Stimulus Management ------------------------------------------------

class ImageSet(Dataset):
    def __init__(self, image_paths, transforms=None):
        self.images = image_paths
        self.transforms = transforms

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.transforms:
            img = self.transforms(img)
        return img
    
    def __len__(self):
        return np.array(self.images).shape[0]
    
def get_image_loader(image_set, transforms, batch_size = 64):
    if isinstance(image_set, pd.Series) or isinstance(image_set, list):
        return DataLoader(ImageSet(image_set, transforms), batch_size)
    if isinstance(image_set, np.ndarray):
        return DataLoader(Array2ImageSet(image_set, transforms), batch_size)
    
class TextSet(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding = True, 
                                   return_tensors = 'pt')

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for 
                key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings[next(iter(self.encodings))])
    
def get_text_loader(text_set, tokenizer = None, batch_size = 64):
    if tokenizer is None:
        return DataLoader(Tokens2TextSet(text_set), batch_size)
    if tokenizer is not None:
        return DataLoader(TextSet(text_set, tokenizer), batch_size)
    
### Downloads ----------------------------------------------------

def extract_tar(tar_file, dest_path, subfolder, delete = False):
    tar = tarfile.open(tar_file)
    if not subfolder:
        tar.extractall(dest_path)
    if subfolder:
        def members(tar):
            l = len(subfolder+'/')
            for member in tar.getmembers():
                if member.path.startswith(subfolder+'/'):
                    member.path = member.path[l:]
                    yield member
        tar.extractall(dest_path, members = members(tar))
        
    tar.close()
    os.remove(tar_file)
    
### Helper Functions -----------------------------------------------

def iterative_subset(df, index, cat_list):
    out_dict = {}
    df = df[[index] + cat_list]
    for row in df.itertuples():
        current = out_dict
        for i, col in enumerate(cat_list):
            cat = getattr(row, col)
            if cat not in current:
                if i+1 < len(cat_list):
                    current[cat] = {}
                else:
                    current[cat] = []
            current = current[cat]
            if i+1 == len(cat_list):
                current += [getattr(row, index)]
    
    return out_dict

def convert_subsets_to_index(subset_dict, index_type = pd.Index):
    new_subset_dict = {}
    def convert_subset_to_index(subset_dict, new_subset_dict):
        for key, value in subset_dict.items():
            if isinstance(value, dict):
                new_subset_dict[key] = {}
                convert_subset_to_index(subset_dict[key], new_subset_dict[key])
            if isinstance(value, list):
                new_subset_dict[key] = index_type(value)
                
    convert_subset_to_index(subset_dict, new_subset_dict)
    
    return new_subset_dict