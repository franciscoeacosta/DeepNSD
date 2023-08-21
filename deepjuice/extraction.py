import pandas as pd
import numpy as np
import torch

import traceback
import itertools
import torchinfo
import torchlens

from copy import copy, deepcopy
from tqdm.auto import tqdm

from typing import Mapping, Iterable, Union
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from .structural import get_fn_kwargs, parse_fns, apply_fn
from .structural import make_filter, apply_filter
from .structural import parse_uid_keys, modify_keys, drop_constant_columns
from .systemops.memory import get_available_device_memory
from .systemops.memory import MemoryStats, convert_memory_unit

### The Super Functions: Get Feature Maps + Metadata -----------------------------------

def get_feature_maps(model, inputs, batch_dim=0, keep=None, skip=None, metadata=None, 
                     device='auto', backend='torchinfo', remove_duplicates=True,
                     save_inputs=False, flatten=False, to_numpy=False, progress=True):
    
    keep_list = [] if keep is None else ([keep] if isinstance(keep, str) else keep)
    skip_list = [] if skip is None else ([skip] if isinstance(skip, str) else skip)
    
    if len(keep_list) >= 1 and len(skip_list) >= 1:
        if any([skip in keep_list for skip in skip_list]):
            raise ValueError('overlap in keep and skip args; please revise.')
        
    
    sample_inputs = get_sample_inputs(inputs, batch_dim, n=3)
    input_shape = (sample_inputs.shape[:batch_dim]+
                   sample_inputs.shape[batch_dim+1:])
    
    check_backend(backend)
        
    if backend=='torchinfo':
        
        necessary_metas = ['feature_module', 'module_uid']
        
        if metadata is None:
            metadata = get_torchinfo_metadata(model, sample_inputs, batch_dim, device)
            
        if len(keep_list) >= 1:
            skip_list = [uid for uid in metadata['feature_map_uid'].tolist() 
                         if uid not in keep_list]
        
        if remove_duplicates:
            skip_list += metadata.query('is_duplicate == True')['feature_map_uid'].tolist()
        
        feature_map_shapes = {row['feature_map_uid']: row['output_shape'] 
                              for _, row in metadata.iterrows()
                              if row['feature_map_uid'] not in skip_list}
        
        extraction_function = torchinfo_extract
        extraction_kwargs = {'skip_list': skip_list, 'output_type': 'feature_maps'}
        
    if isinstance(inputs, DataLoader):
        batch_size, dataset_size, start_index = inputs.batch_size, len(inputs.dataset), 0
        description = 'Feature Extraction (Batch)'
        
        input_iterator = inputs if not progress else tqdm(inputs, desc = description)
        feature_maps = {map_uid: torch.empty(dataset_size, *feature_map_shapes[map_uid])
                            for map_uid in feature_map_shapes}
    
        if save_inputs:
            saved_inputs = torch.empty(dataset_size, *input_shape)
        
        for batch_inputs in input_iterator:
            batch_feature_maps = extraction_function(model, batch_inputs, **extraction_kwargs)
            
            if save_inputs:
                saved_inputs[start_index:start_index+batch_size,...] = batch_inputs.cpu()
            
            for map_uid in feature_map_shapes:
                feature_maps[map_uid][start_index:start_index+batch_size,...] = batch_feature_maps[map_uid]
                
            start_index += batch_size
        
        feature_maps = feature_maps if not save_inputs else {'Input': saved_inputs, **feature_maps}
            
    if isinstance(inputs, torch.Tensor):
        feature_maps = extraction_function(model, inputs, **extraction_kwargs)
        
        if save_inputs:
            feature_maps = {'Input': inputs.to('cpu'), **feature_maps}
            
    feature_maps = {map_uid: feature_map for map_uid, feature_map in feature_maps.items()}
    
    if flatten == True:
        for feature_map_uid, feature_map in feature_maps.items():
            feature_maps[feature_map_uid] = feature_map.reshape(feature_map.shape[0], -1)
            
    if to_numpy == True:
        for feature_map_uid, feature_map in feature_maps.items():
            feature_maps[feature_map_uid] = feature_map.numpy()
    
    return feature_maps

def get_feature_map_metadata(model, input_data, batch_dim=0, device='auto', 
                             backend='torchinfo', handler=False, max_samples=3):
        
    check_backend(backend)
    
    input_data = get_sample_inputs(input_data, batch_dim, max_samples)
    
    if backend=='torchinfo':
        
        metadata_args = (model, input_data, batch_dim, device) 
        metadata = get_torchinfo_metadata(*metadata_args)
    
    if not handler:
        return metadata
    if handler:
        return MetadataHandle(metadata)


### Convenience Classes: Metadata + FeatureMapLoader ---------------------------------------- 

class MetadataHandle():
    def __init__(self, metadata, uid_key='feature_map_uid',
                 output='uids', embedded_data=True,
                 chain_filters=True, **filter_kwargs):
        
        self.check_input_data(metadata, uid_key)
        self.uid_key = uid_key
        self.data = metadata
        self.is_filtered = False
        self.applied_filters = []
        
        self.uids = self.get_uids(filtered=False)
        self.original_uids = deepcopy(self.uids)
        self.original_data = deepcopy(self.data)
        
        if 'uid' in output:
            self.output = 'uids'
        if 'data' in output:
            self.output = 'data'
            
        self.chain_filters = chain_filters
        self.filter_kwargs = self.check_filter(filter_kwargs)
        
        self.embedded_data = embedded_data
        if self.embedded_data:
            self.embed_data() # add metadata to handle as attributes
            
    def _repr_(self):
        return self.data._repr_()
            
    def _repr_html_(self):
        return self.data._repr_html_()
    
    def check_input_data(self, metadata, uid_key):
        raise_flag = True
        if isinstance(metadata, pd.DataFrame):
            if uid_key in metadata.columns:
                raise_flag = False
        if isinstance(metadata, dict):
            if uid_key in next(iter(metadata.values())):
                raise_flag = False 
        if raise_flag:
            raise ValueError('metadata must be either a dict of dicts or ' + 
                             f'pd.DataFrame and must include {self.uid_key} key')
        
    def _check_filtered(self, filtered=False):
        if filtered and not self.is_filtered:
            raise Warning('filter not yet applied, output is the original data unfiltered')
            
    def embed_data(self, embed_type=pd.Series):
        if isinstance(self.data, pd.DataFrame):
            for col in self.data.columns:
                attr_data = self.data[col].copy()
                attr_data.index = copy(self.uids)
                if embed_type==list:
                    attr_data = attr_data.to_list()
                if embed_type==dict:
                    attr_vals = attr_data.to_list()
                    attr_data = {self.uids[i]: attr_vals[i]
                                 for i in range(len(self.uids))}
                    
                setattr(self, col, attr_data)
                
        if isinstance(self.data, dict):
            attr_data = {}
            for uid, metas in self.data.items():
                for key, value in metas.items():
                    if key not in attr_data:
                        attr_data['key'] = [value]
                    elif key in attr_data:
                        attr_data['key'] += [value]
                        
            for attr, data in attr_data.items():
                setattr(self, attr, embed_type(attr_data))
    
    def get_uids(self, filtered=True):
        self._check_filtered(filtered)
        
        if isinstance(self.data, pd.DataFrame):
            self.uids = self.data[self.uid_key].tolist()
        if isinstance(self.data, dict):
            self.uids = [entry[self.uid_key] for entry 
                            in self.data.values()]
        
        if not filtered:
            if not self.is_filtered:
                return self.uids
            return self.original_uids
            
        return self.uids
    
    def get_data(self, filtered=True):
        self._check_filtered(filtered)
        
        if filtered:
            return self.data
        
        if not filtered:
            return self.original_data
        
    def get_meta(self, meta_key, output_type=list):
        if isinstance(self.data, pd.DataFrame):
            meta = self.data[meta_key].to_list()
        if isinstance(self.data, dict):
            meta = [metas[meta_key] for metas in self.data.values()]
        
        if output_type == pd.Series:
            return pd.Series(meta)
        if output_type == dict:
            return {self.uids[i]: meta[i] for i in range(len(self.uids))}
        
        return meta
        
    def convert_to_dict(self, filtered=True):
        output = self.get_data(filtered)
        if isinstance(output, dict):
            return output
        output = output.set_index(self.uid_key).to_dict(orient='index')
        return {k: {self.uid_key: k, **v} for k, v in output.items()}
    
    def convert_to_pandas(self, filtered=True):
        output = self.get_data(filtered)
        if isinstance(output, pd.DataFrame):
            return output
        return pd.DataFrame(list(output.values()))

    def check_filter(self, filter_kwargs={}):
        if len(filter_kwargs)==0 or filter_kwargs is None:
            return {} # no filter if none set
        
        filter_kwargs = {key.replace('filter_', '').replace('key','query'): 
                         value for key, value in filter_kwargs.items()}
        
        if not all([kwarg in filter_kwargs for kwarg in ['value','query']]):
                raise ValueError('filtering requires value and query (or key)')
                
        return filter_kwargs
    
    def remove_filters(self):
        self.data = deepcopy(self.original_data)
        self.uids = deepcopy(self.original_uids)
        
    def apply_filter(self, **filter_kwargs):
        if len(filter_kwargs)==0:
            filter_kwargs = self.filter_kwargs
        filter_kwargs = self.check_filter(filter_kwargs)
        
        if not self.chain_filters:
            self.remove_filters()
            
        if self.chain_filters is False:
            self.applied_filters = []
            
        self.data = apply_filter(self.data, **filter_kwargs)
        self.is_filtered = True
        self.uids = self.get_uids(filtered=True)
        self.applied_filters += filter_kwargs
        
        if self.embedded_data:
            self.embed_data() # update attributes
        
        return self # filtered metadata handler

def convert_metadata(metadata, uid_key='feature_map_uid', output='dict'):
    if isinstance(metadata, pd.DataFrame):
        if output == 'pandas':
            return metadata
        if output == 'dict':
            return metadata.set_index(uid_key).to_dict(orient='index')
        
    if isinstance(metadata, dict):
        if output == 'dict':
            return metadata
        if output == 'pandas':
            pd.DataFrame([{uid_key: key, **value} for key, value in metadata.items()])
            
class FeatureMapLoader(MetadataHandle):
    def __init__(self, model, inputs, metadata=None, batch_dim=0,
                 uid_key='feature_map_uid', 
                 shapes_key='output_shape', 
                 memory_key='output_bytes', max_memory_load='auto', 
                 remove_duplicates=True, device='auto', **kwargs):
        
        if metadata is None:
            metadata = get_feature_map_metadata(model, inputs, batch_dim, device)
            
        super().__init__(metadata, uid_key)
        
        if remove_duplicates:
            self.apply_filter(**{'key': 'is_duplicate', 'value': True, 'logic': '!='})
        
        self.get_feature_maps_kwargs = get_fn_kwargs(get_feature_maps, kwargs)
        self.memory_stats = MemoryStats(output_units='auto', readable=True)
        
        self.model, self.inputs = model, inputs
        self.shapes_key = shapes_key
        self.memory_key = memory_key
        self.n_inputs = len(self.inputs)
        if isinstance(self.inputs, DataLoader):
            self.n_inputs = len(self.inputs.dataset)
        
        self.shapes = self.get_shapes()
        self.memory = self.get_memory()
        self.memory_load = self.memory_stats.convert_from_bytes(self.memory)
        
        self.max_memory = self.get_max_memory(max_memory_load)
        self.batches = self.set_batches()
        self.n_batches = len(self.set_batches())
        
        self.cpu_oversize = self.get_oversized_maps(device='cpu')
        self.gpu_oversize = self.get_oversized_maps(device='gpu')
            
        self.max_memory_load = self.get_max_memory(max_memory_load, readable=True)
           
    def get_shapes(self, add_input_size=True):
        shapes = self.get_meta(self.shapes_key)
            
        if add_input_size:
            shapes = [[self.n_inputs] + list(shape) for shape in shapes]
            
        return {self.uids[i]: shapes[i] for i in range(len(self.uids))}
    
    def get_memory(self, add_input_size=True):
        memory = self.get_meta(self.memory_key)
            
        if add_input_size:
            memory = [mem * self.n_inputs for mem in memory]
            
        return {self.uids[i]: memory[i] for i in range(len(self.uids))}
    
    def get_max_memory(self, bound='auto', device=None, 
                       usage_ratio=0.75, readable=False):
        
        if 'auto' in bound:
            if ':' in bound:
                usage_ratio = float(max_memory_load.split(':')[-1])
            if usage_ratio <= 1:
                bound = get_available_device_memory(device) * usage_ratio
            if usage_ratio > 1:
                bound = get_available_device_memory(device) // usage_ratio
    
        if readable:
            return convert_memory_unit(bound, output_units='auto') 
        if not readable:
            return convert_memory_unit(bound, output_units='B', readable=False)
            
    
    def get_oversized_maps(self, max_memory_load=None, device='cpu'):
        if max_memory_load is None:
            max_memory_load = self.max_memory
            
        return {k: list(self.memory_load.values())[i] for i, (k, v) 
                in enumerate(self.memory.items()) 
                if v >= self.memory_stats._convert_for_stats(max_memory_load)}
    
    def get_batches(self):
        return self.batches
    
    def set_batches(self, max_memory_load=None, device='cpu'):
        if max_memory_load is None:
            max_memory_load = self.max_memory
            
        max_memory_load = self.memory_stats._convert_for_stats(max_memory_load)

        # Partition the feature maps into batches based on memory constraints
        batches, current_batch, current_batch_memory = [], [], 0

        for key, memory in self.memory.items():
            if current_batch_memory+memory > max_memory_load:
                batches.append(current_batch)
                current_batch = []
                current_batch_memory = 0
            current_batch.append(key)
            current_batch_memory += memory

        if current_batch:
            batches.append(current_batch)

        return batches
        
    def generate_batches(self, *args, **kwargs):
        description = 'Feature Map Loader'
        for current_batch in tqdm(self.batches, desc=description):
            kwargs = {**kwargs, **self.get_feature_maps_kwargs}
            kwargs['keep'] = current_batch
            if not 'progress' in kwargs:
                kwargs['progress'] = False
            yield get_feature_maps(self.model, self.inputs, **kwargs)
            
    def __iter__(self):
        self._batch_generator = self.generate_batches()
        return self

    def __next__(self):
        return next(self._batch_generator)
            
### Prep for Extraction + Tensor Processing --------------------------------------------    

def prep_for_extraction(model, inputs=None, device='auto', dtype=None, enforce_eval=True):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    if device is None:
        device = 'cpu'
    
    if enforce_eval:
        model = model.eval()
        
    if 'cuda' in device:
        if not next(model.parameters()).is_cuda:
            model = model.to(device)

    if inputs is None:
        return(model)
    
    device = next(model.parameters()).device
    
    if inputs is not None:
        action_fn = lambda inputs: inputs.to(device)
        inputs = process_input_data(inputs, action_fn)
        
        if dtype is not None:
            action_fn = lambda inputs: inputs.to(dtype)
            inputs = process_input_data(inputs, action_fn)
            
    return model, inputs

# adapted from torchinfo's traverse_input_data function
def process_input_data(data, action_fn, aggregate_fn=None):
    if aggregate_fn is None:
        aggregate_fn = lambda data: data
    
    processing_fns = (action_fn, aggregate_fn)
    
    if isinstance(data, torch.Tensor):
        data = action_fn(data)
        
    elif isinstance(data, np.ndarray):
        data = action_fn(torch.from_numpy(data))
        
    elif isinstance(data, DataLoader):
        data = aggregate_fn([process_input_data(dat, *processing_fns) 
                                for dat in data])

    elif isinstance(data, Mapping):
        data = aggregate_fn({k: process_input_data(v, *processing_fns) 
                                 for k, v in data.items()})
        
    elif isinstance(data, Iterable) and not isinstance(data, str):
        data = aggregate_fn([process_input_data(dat, *processing_fns) 
                             for dat in data])
        
    elif isinstance(data, tuple) and hasattr(input_data, "_fields"): 
        data = aggregate_fn(*(process_input_data(dat, *processing_fns) 
                              for dat in data))
        
    return data

def get_sample_inputs(inputs, batch_dim=0, n=3):
    def action_fn(data):
        if isinstance(data, torch.Tensor) or isinstance(data, np.ndarray):
            selector = [slice(None)] * data.ndim
            selector[batch_dim] = slice(n)
            return data[tuple(selector)]
        return data
    
    if isinstance(inputs, DataLoader):
        inputs = next(iter(inputs))

    return process_input_data(inputs, action_fn)

def process_tensor(tensor, output_type='tensor', keep_on_device=False, 
                   tensor_fns=[], tensor_fn_kwargs=None, **other_kwargs):
    
    output_types = ['tensor', 'size', 'shape', 'dtype',
                    'element_size', 'device', 'bytes', 'summary']
    
    if output_type not in output_types:
        raise ValueError(f'output_type should be one of {output_types}')
        
    tensor_fns = parse_fns(tensor_fns, tensor_fn_kwargs, **other_kwargs)

    if isinstance(tensor, torch.Tensor):
        if output_type == 'tensor':
            tensor = tensor.detach()
            for fn, fn_kwargs in tensor_fns.items():
                tensor = fn(tensor, **fn_kwargs)
            if isinstance(tensor, torch.Tensor):
                if keep_on_device: 
                    return tensor
                return tensor.to('cpu', non_blocking=True)
            return tensor # functional output
        
        if output_type == 'size':
            return list(tensor.size())

        if output_type == 'shape':
            return list(tensor.shape)

        if output_type == 'dtype':
            return tensor.dtype

        if output_type == 'device':
            return tensor.device.__name__

        if output_type == 'element_size':
            return tensor.element_size()

        if output_type == 'bytes':
            return (tensor.element_size() * 
                    np.prod(list(tensor.shape)))
        
    process_kwargs = {'output_type': output_type, 
                      'keep_on_device': keep_on_device,
                      'tensor_fns': tensor_fns,
                      'tensor_fn_kwargs': tensor_fn_kwargs}
                
    if isinstance(tensor, (list, tuple)):
        output = [process_tensor(x, **process_kwargs) for x in tensor]
        return output[0] if len(output) == 1 else tuple(output)
    
    if isinstance(tensor, dict):
        return {key: process_tensor(value, **process_kwargs)
                    for key, value in tensor.items()}
    
### Get + Remove Duplicate Feature Maps ----------------------------------------

def get_matched_feature_maps(feature_maps, match_to_keep=None, metadata=None, filter_kwargs=None):
    
    if isinstance(feature_maps, list):
        feature_maps = [(i, value.flatten()) for i, value in enumerate(feature_maps)]
    if isinstance(feature_maps, dict):
        feature_maps = [(k, value.flatten()) for k, value in feature_maps.items()]
        
    tensor_lengths = [len(feature_map[1]) for feature_map in feature_maps]
    random_tensor = torch.rand(max(tensor_lengths))

    tensor_hash_dict = OrderedDict()
    for (tensor_name, target_tensor) in feature_maps:
        tensor_dot = torch.dot(target_tensor, random_tensor[:len(target_tensor)])
        tensor_hash = tensor_dot.numpy().tobytes()
        tensor_hash_dict.setdefault(tensor_hash, []).append(tensor_name)
    
    match_list = [tuple(match) for match in tensor_hash_dict.values() if len(match) > 1]
    
    if match_to_keep in [None, 'list', list]:
        return match_list
        
    if match_to_keep in ['dict', list]:
        match_dict = {k[0]: [] for k in feature_maps}
        for feature_map in feature_maps:
            uid_or_index = feature_map[0]
            matched_uids_or_idx = []
            for match_set in match_list:
                if uid_or_index in match_set:
                    matched_uids_or_idx += [match for match in match_set]
            match_dict[uid_or_index] = matched_uids_or_idx

        return match_dict
    
    index_to_keep = 0 if match_to_keep != 'last' else -1
        
    if metadata is None:
        return {match_set: match_set[index_to_keep] for match_set in match_list}

    if metadata is not None:
        if filter_kwargs is None:
            raise ValueError('please provide filter_kwargs to keep matches based on metadata')

        filtered_uids = MetadataHandle(metadata, **filter_kwargs).apply_filter().get_uids()
        
        matches_to_keep = {}
        for match_set in match_list:
            filtered_matches = [match for match in match_set if match in filtered_uids]
            matches_to_keep[match_set] = match_set[index_to_keep]
            if len(filtered_matches) >= 1:
                matches_to_keep[match_set] = filtered_matches[index_to_keep]
    
        return matches_to_keep
    
    raise ValueError('match_to_keep, metadata, and filter_kwargs produced no valid outputs')
                  
def remove_duplicate_feature_maps(feature_maps, match_to_keep='first'):
    feature_maps_to_keep = get_matched_feature_maps(feature_maps, match_to_keep)
    
    feature_maps_to_drop = [[key for key in list(keys) if key != value] 
                            for keys, value in feature_maps_to_keep.items()]

    return {key:value for (key,value) in feature_maps.items()
            if key not in itertools.chain(*feature_maps_to_drop)}


### Feature Map Grouping -------------------------------------------------------

    
### Backend Checks -------------------------------------------------------------

def check_backend(backend):
    available_backends=['torchinfo']
    if backend not in available_backends:
        raise ValueError(f'currently, only one of backend={available_backends} is available')
        
    return backend

### TorchInfo Metadata + Extraction -------------------------------------------- 

from torchinfo.layer_info import LayerInfo, get_children_layers 
from torchinfo.torchinfo import construct_pre_hook
from torchinfo.torchinfo import add_missing_container_layers, set_children_layers

def get_torchinfo_feature_maps(model, inputs, device='auto', **kwargs):
    return torchinfo_extract(model, inputs, 'feature_maps', device, **kwargs)

def get_torchinfo_metadata(model, inputs, batch_dim=0, device='auto', **kwargs):
    
    if 'preprocess' in kwargs:
        input_data = kwargs.pop('preprocess')(input_data)
    
    info = torchinfo_extract(model, inputs, 'metadata', device=device)
    return parse_torchinfo(info, batch_dim=batch_dim, **kwargs)

class TorchInfoFeatureMapTracker():
    def __init__(self, uid_keys=None, complete_uid=False, track_parenting=True):
        self.default_uid_keys = ['module_class','nest_depth','class_index', 
                                   'feature_map_index','recursion_depth']
        
        self.other_uid_keys = ['nest_level', 'nest_index', 'module_index',
                               'global_index', 'leaf_index', 'leaf_name', 
                               'module_uid', 'torchinfo_id', 'feature_map_name']
        
        self.available_uid_keys = self.default_uid_keys + self.other_uid_keys
        
        self.uid_keys = uid_keys
        self.uid_data = None
        self.complete_uid = complete_uid
        self.track_parenting = track_parenting
        
        if self.uid_keys is not None:
            self.uid_keys = self.parse_uid_keys(self.uid_keys, parent_leaf_split=True)
        
        if self.uid_keys is None:
            if self.complete_uid:
                self.uid_keys = self.parse_uid_keys(self.default_uid_keys, parent_leaf_split=True)
        
            if not self.complete_uid:
                self.uid_keys = {'leaf': self.default_uid_keys, 
                                 'parent': ['module_class', 'nest_depth', 'class_index']}
        
        uid_key_vals = list(itertools.chain(*self.uid_keys.values()))
        self.default_uid_keys = self.uid_keys
        self.all_uid_keys = pd.unique(list(uid_key_vals)).tolist()
   
        self.key_mods = {'class_name': 'module_class',
                         'depth': 'nest_depth',
                         'depth_index': 'nest_index',
                         'var_name': 'layer_name'}
        
        self.feature_map_uids = []
        self.feature_map_index = 0
        self.module_uids = []
        self.module_index = -1
        self.global_index = -1
        self.recusion_depth = 0
        
        self.global_class_counts = {}
        self.parent_at_depth = {}
        self.current_uid_vals = {}
        self.is_leaf_layer = False
        self.is_recursive = False
        self.layer_is_model = False
        
        if self.uid_keys != self.default_uid_keys:
            self.track_parenting = True
        
        if self.track_parenting:
            self.parent_at_depth = {}
            self.model_name = None
            self.parent_uid = None
            self.leaf_index = None
            self.current_depth = -1
            self.feature_map_name = None
            
        self.current_layer_info = None
        
    @staticmethod
    def index_to_level(index):
        return '' if index == -1 else chr(index+64+1)
    
    def parse_uid_keys(self, uid_keys=None, parent_leaf_split=True):
        if uid_keys is None:
            return None
    
        def split_uid_keys(uid_keys):
            return {'leaf': uid_keys, 'parent': uid_keys}

        if isinstance(uid_keys, str):
            uid_keys = [uid_keys]

        elif isinstance(uid_keys, list):
            if len(uid_keys) == 0:
                return None
            if parent_leaf_split:
                return split_uid_keys(uid_keys)

        elif isinstance(uid_keys, dict):
            if 'parent' not in uid_keys and 'leaf' not in uid_keys:
                raise ValueError("dict of uid_keys requires 'parent' and 'leaf' entries.")

            uid_keys = {entry: parse_uid_keys(keys, False) 
                        for entry, keys in uid_keys.items()}

            if uid_keys['parent'] is None or uid_keys['leaf'] is None:
                raise ValueError("at least one of 'parent' or 'leaf' uid_keys is invalid.")

            return uid_keys
    
    def get_uid_data(self, layer_info):
        self.check_uid_data(layer_info)
        return {key: layer_info[key] for key in self.all_uid_keys}
    
    def set_uid_data(self, layer_info):
        self.uid_data = self.get_uid_data(layer_info)
    
    def check_uid_data(self, layer_info):
        if not all([key in layer_info.keys() for key in self.all_uid_keys]):
            raise ValueError(f'uid_keys for uid construction includes keys not in {layer_info.keys()}')
        
    def update(self, layer_info):
        self.global_index += 1
        
        if isinstance(layer_info, LayerInfo):
            layer_info = deepcopy(layer_info).__dict__
            
        layer_info = modify_keys(layer_info, self.key_mods, True)
        module_object = layer_info.pop('module', None)
        
        module_class = layer_info['module_class']
        nest_depth = layer_info['nest_depth']
        nest_index = layer_info['nest_index']
        nest_level = self.index_to_level(nest_depth)
        layer_info['nest_level'] = nest_level
        self.is_leaf_layer = layer_info['is_leaf_layer']
        
        if layer_info['nest_depth'] == 0:
            self.model_name = module_class
            self.module_index -= 1
            self.layer_is_model = True
        if layer_info['nest_depth'] > 0:
            self.layer_is_model = False
            layer_info['nest_depth'] -= 0
        
        self.is_recursive = True
        if not layer_info['is_recursive']:
            self.is_recursive = False
            self.module_index += 1
            self.recursion_depth = 0
            
        if self.is_recursive:
            self.recursion_depth += 1
            
        self.feature_map_index += int(self.is_leaf_layer)
        
        layer_info['feature_map_index'] = self.feature_map_index
        layer_info['global_index'] = self.global_index
        layer_info['module_index'] = self.module_index
        layer_info['recursion_depth'] = self.recursion_depth
        if not self.is_recursive: 
            layer_info['recursion_depth'] = None
            
        if module_class not in self.global_class_counts:
            self.global_class_counts[module_class] = 0
        if not self.is_recursive:
            self.global_class_counts[module_class] += 1
            
        layer_info['class_index'] = self.global_class_counts[module_class]
        
        layer_info['torchinfo_id'] = f'{module_class}-{nest_depth}-{nest_index}'
        if nest_depth == 0: 
            layer_info['torchinfo_id'] = module_class
            
        self.set_uid_data(layer_info)
        feature_map_uid = self.construct_uid()
        
        if self.track_parenting:
            
            is_parent = len(layer_info.pop('children', [])) > 0
            parent_info = layer_info.pop('parent_info', None)
            
            if isinstance(parent_info, LayerInfo):
                parent_info = deepcopy(parent_info).__dict__
            
            if is_parent or self.current_depth > nest_depth:
                self.parent_uid = feature_map_uid
                self.parent_at_depth[nest_depth] = self.parent_uid
                self.leaf_index = None # leaf index for parent

            self.current_depth = nest_depth

            depth_parent_uid = self.parent_at_depth.get(nest_depth-1)
            if depth_parent_uid == self.parent_uid:
                layer_info['parent_uid'] = self.parent_uid
            if depth_parent_uid != self.parent_uid:
                layer_info['parent_uid'] = depth_parent_uid
            parenting_uids = [parent_uid for depth, parent_uid 
                              in self.parent_at_depth.items()
                              if depth != self.current_depth]

            layer_info['parenting_uids'] = None if len(parenting_uids)==0 else parenting_uids

            parent_name = None
            leaf_name = None
        
            if self.is_leaf_layer:
                if parent_info:
                    parent_name = parent_info['var_name']
                if parent_name == self.model_name:
                    leaf_name = layer_info['layer_name']
                if parent_name != self.model_name:
                    if parent_name is None:
                        leaf_name = layer_info['layer_name']
                    if parent_name is not None:
                        leaf_name = f"{parent_name}-{layer_info['layer_name']}"
                self.leaf_index = 0 if pd.isna(self.leaf_index) else self.leaf_index + 1

            layer_info['parent_name'] = parent_name
            layer_info['leaf_name'] = leaf_name
            layer_info['leaf_index'] = self.leaf_index
            layer_info['feature_map_name'] = None     
            
            if is_parent and not self.layer_is_model:
                layer_info['feature_map_name'] = layer_info['layer_name']

            if leaf_name is not None:
                if self.is_recursive:
                    leaf_name += f'_{self.recursion_depth}'
                layer_info['feature_map_name'] = leaf_name.replace('-','.')
        
        layer_info['feature_map_uid'] = feature_map_uid
        self.current_layer_info = layer_info
        self.uid_data = self.get_uid_data(layer_info)
        
        return layer_info
        
    def construct_uid(self, names_sep='-', uid_na='X'):
        if self.uid_data is None:
            raise ValueError('class must be updated before constructing uid.')
        
        current_uid_keys = self.uid_keys['leaf']
        if not self.is_leaf_layer:
            current_uid_keys = self.uid_keys['parent']
            
        if self.layer_is_model:
            current_uid_keys = ['module_class']
            
        uid_construct = []
        for key in current_uid_keys:
            if pd.isna(self.uid_data[key]):
                if self.complete_uid:
                    uid_construct.append(uid_na)
            if not pd.isna(self.uid_data[key]):
                uid_construct.append(str(self.uid_data[key]))
            
        feature_map_uid = '-'.join([construct for construct in uid_construct])
        if feature_map_uid not in self.feature_map_uids:
            self.feature_map_uids.append(feature_map_uid)
        
        return feature_map_uid
    
    def get_uid(self, names_sep='-', uid_na='X'):
        return self.construct_uid(names_sep, uid_na)
    
    def get_key_order(self):
        return ['feature_map_uid'] + self.available_uid_keys
        
    def update_and_construct_uid(self, layer_info):
        return self.update(layer_info), self.construct_uid()

def torchinfo_extract(model, inputs, output_type='metadata', device='auto', 
                      skip_list=[], keep_on_device=False, tensor_fns=None, 
                      model_kwargs={}, tensor_fn_kwargs={}, **other_kwargs):
        
    class ModifiedLayerInfo(LayerInfo):
        def __init__(self, var_name, module, depth, parent_info, 
                     keep_input=False, keep_output=True):
            super().__init__(var_name, module, depth, parent_info)

            if keep_input:
                self.input_tensor = None

            if keep_output:
                self.output_tensor = None

            for tensor_name in ['input', 'output']:
                for attr in ['size', 'bytes', 'dtype']:
                    setattr(self, f'{tensor_name}_{attr}', None)
                    
    def hook_data_report(inputs, outputs, info_types=None):
        if info_types is None:
            info_types = ['shape', 'dtype', 'bytes']

        for arg in ['inputs', 'outputs']:
            report = f'{arg.upper()}:'
            for info_type in info_types:
                info = process_tensor(eval(arg), info_type)
                report += f' {info_type}: {info}'

        print(report) # input output report with key information
    
    def construct_hook(global_layer_info, tensors_to_keep=['output'], 
                       keep_on_device=False, tensor_fns=None, tensor_fn_kwargs=None):
        
        if tensor_fn_kwargs is None:
            tensor_fn_kwargs = {}
        
        def hook(module, inputs, outputs):
            info = global_layer_info[id(module)]
            
            if other_kwargs.get('hook_report', False):
                hook_data_report(inputs, outputs)
                
            if info.contains_lazy_param:
                info.calculate_num_params()
            
            tensor_fns_dict = tensor_fns
            if not isinstance(tensor_fns_dict, dict):
                tensor_fns_dict = {}
                for tensor_name in tensors_to_keep:
                    tensor_fns_dict[tensor_name] = tensor_fns
                
            for tensor_name in ['input','output']:
                for info_type in ['size','dtype','bytes']:
                    setattr(info, f'{tensor_name}_{info_type}',
                            process_tensor(eval(tensor_name+'s'), info_type))
                    
                if any([tensor_name in name for name in tensors_to_keep]):
                    _tensor_fns = tensor_fns_dict[tensor_name]
                    setattr(info, f'{tensor_name}_tensor', 
                            process_tensor(eval(tensor_name+'s'), 'tensor', 
                                           keep_on_device, _tensor_fns, tensor_fn_kwargs))
                
            info.executed = True
            info.calculate_macs()
                
        return hook
        
    def apply_hooks(model, input_data, skip_list, tensors_to_keep=['outputs'], 
                    keep_on_device=False, tensor_fns=None, tensor_fn_kwargs=None):
        
        if skip_list is None: skip_list = []
        
        model_name = model.__class__.__name__
        summary_list, hooks = [], {}
        layer_ids = set()
        global_layer_info = {}
        global_index = -1
        stack = [(model_name, model, 0, None)]
        while stack:
            var_name, module, curr_depth, parent_info = stack.pop()
            for tensor_name in ['input','output']:
                tensor_attribute = f'{tensor_name}_output'
                if hasattr(parent_info, tensor_attribute):
                    delattr(parent_info, tensor_attribute)
            
            module_id = id(module)
            global_index += 1
            module_uid = str(module_id)+'-'+str(global_index)
            
            if module_uid not in skip_list:
            
                layer_info_args = (var_name, module, curr_depth, parent_info,
                                   *(tensor_name in tensors_to_keep 
                                     for tensor_name in ['input', 'output']))

                global_layer_info[module_id] = ModifiedLayerInfo(*layer_info_args)

                pre_hook_args = (layer_ids, var_name, curr_depth, parent_info)
                pre_hook = construct_pre_hook(global_layer_info, summary_list, *pre_hook_args)

                if module_id in hooks:
                    for hook in hooks[module_id]:
                        hook.remove()

                hook_args = (global_layer_info, tensors_to_keep, 
                             keep_on_device, tensor_fns, tensor_fn_kwargs)
                hooks[module_id] = (module.register_forward_pre_hook(pre_hook), 
                                    module.register_forward_hook(construct_hook(*hook_args)))

                stack += [(name, mod, curr_depth + 1, global_layer_info[module_id]) 
                          for name, mod in reversed(module._modules.items()) if mod is not None]

        return summary_list, hooks

    def forward_pass(model, x, output_type, model_kwargs, hook_kwargs):
            
        summary_list, hooks = apply_hooks(model, x, **hook_kwargs)
        exception_traceback = None
        
        if model_kwargs is None: 
            model_kwargs = {}

        try: # prevents mangled hooks
            with torch.no_grad():
                if isinstance(x, torch.Tensor):
                    _ = model(x, **model_kwargs)
                elif isinstance(x, (list, tuple)):
                    _ = model(*x, **model_kwargs)
                elif isinstance(x, dict):
                    _ = model(**x, **model_kwargs)
                    
        except Exception as e:
            print(f'feature_extraction failed: {e}; removing hooks...')
            exception_traceback = traceback.format_exc()
            
        finally:
            if hooks:
                for pre_hook, hook in hooks.values():
                    pre_hook.remove(); hook.remove()
    
        if 'metadata' in output_type:
            held_tensors = {'input': [], 'output': []}
            for summary_item in summary_list:
                for tensor_name in ['input', 'output']:
                    tensor_attr = f'{tensor_name}_tensor'
                if hasattr(summary_item, tensor_attr):
                    tensor = summary_item.__dict__.pop(tensor_attr, None)
                    held_tensors[tensor_name] += [tensor]
                
            add_missing_container_layers(summary_list)
            set_children_layers(summary_list)
            
            held_tensors = {key: None if len(value) == 0 else value
                            for key, value in held_tensors.items()}
            
            for i, summary_item in enumerate(summary_list):
                for tensor_name in ['input', 'output']:
                    tensor_attr = f'{tensor_name}_tensor'
                    tensor_vals = held_tensors[tensor_name]
                    if tensor_vals is not None:
                        setattr(summary_item, tensor_attr, tensor_vals[i])
                
        if exception_traceback:
            print(f'Traceback: {exception_traceback}')
                    
        return summary_list
            
    if skip_list is None: skip_list = []
    
    hook_kwargs = {key: value for key, value in locals().items() if key in 
                      ['skip_list', 'keep_on_device', 'tensor_fns', 'tensor_fn_kwargs']}
    
    if other_kwargs.pop('tensors_to_keep', None) is None:
        hook_kwargs['tensors_to_keep'] = ['output']
    
    pandas_output = other_kwargs.pop('to_pandas', False)
    if pandas_output or output_type == 'metadata_mini':
        hook_kwargs['tensors_to_keep'] = []
    
    model, inputs = prep_for_extraction(model, inputs, device)
    summary_list = forward_pass(model, inputs, output_type, model_kwargs, hook_kwargs)
    
    map_tracker = TorchInfoFeatureMapTracker()
    parsed_output_list = []
    for global_index, layer_info in enumerate(summary_list):
        
        if getattr(layer_info, 'layer_id', None) is None:
            raise ValueError("something's off! no module_id found...")
            
        module_uid = str(layer_info.layer_id) + '-' + str(global_index)
        setattr(layer_info, 'module_uid', module_uid)
        
        if hasattr(layer_info, 'inner_layers'):
            delattr(layer_info, 'inner_layers')
            
        layer_info = map_tracker.update(layer_info)
        feature_map_uid = map_tracker.construct_uid()
        
        if output_type in ['metadata','metadata_mini','metadata_list']:
            parsed_output_list += [(feature_map_uid, layer_info)]
            
        if output_type == 'feature_maps':
            feature_map = layer_info['output_tensor']
            if feature_map_uid not in skip_list:
                parsed_output_list += [(feature_map_uid, feature_map)]
                
    if output_type == 'metadata_list':
        return [output[1] for output in parsed_output_list]
    
    if pandas_output == True:
        output_keys = pd.unique(map_tracker.get_key_order() + list(layer_info.keys()))
        return pd.DataFrame([output[1] for output in parsed_output_list])[output_keys]
        
    return {output[0]: output[1] for output in parsed_output_list}

def parse_torchinfo(info, uid_keys=None, complete_uid=False, to_pandas=True,
                    batch_dim=0, remove_batch_dim=True, fewer_id_keys=True,
                    remove_duplicates=False, filters=None, output_keys=None,
                    output_kwargs={'drop_model': True, 'drop_constants': True}):
    
    info_is_torchinfo_summary = False
    if hasattr(info, 'summary_list'):
        info_is_torchinfo_summary = True
        info_list = deepcopy(summary).summary_list
    elif isinstance(info, list):
        info_list = deepcopy(info)
    elif isinstance(info, dict):
        info_list = deepcopy(info).values()
    
    info_tracker = TorchInfoFeatureMapTracker(uid_keys, complete_uid, track_parenting=True)
    
    keys_to_extract = ['class_name', 'var_name', 'depth', 'depth_index', 
                       'output_size', 'output_bytes', 'output_dtype', 
                       'input_size', 'trainable_params', 'num_params', 'macs']

    key_mods = {'input_size': 'input_shape',
                'output_size': 'output_shape',
                'num_params': 'total_params',
                'params_percent': 'percent_params'}
        
    id_keys = [*[f'feature_map_{x}' for x in ['uid','index','depth']],
                   'module_uid', 'module_class', 'module_index', 
                   'class_index', 'recursion_depth', 'recursion_source_uid', 
                   'global_index', 'nest_depth', 'nest_index', 'nest_level',
                   'parent_uid', 'parenting_uids', 'parent_name', 
                   'leaf_index', 'leaf_name', 'feature_map_name']
    
    id_keys += ['torchinfo_id', 'torchlens_id', 'torchextract_id']
    
    if fewer_id_keys:
        keys_to_remove = ['module_uid', 'module_index', 'global_index', 'torchinfo_id', 
                          'leaf_index', 'leaf_name', 'nest_depth', 'nest_index', 'nest_level']
        
        id_keys = [key for key in id_keys if key not in keys_to_remove]
        
    data_keys = ['effective_depth', 'input_shape', 'output_shape', 'total_features', 
                 'output_bytes', 'output_dtype', 'total_params', 'trainable_params', 
                 'kernel_size', 'macs',  'is_recursive', 'duplicate_uids', 'is_duplicate']
    
    if output_keys is None:
        output_keys = info_tracker.all_uid_keys + id_keys + data_keys
        
    uid_na = output_kwargs.get('uid_na', 'X')
    if output_kwargs.get('uid_sync', False):
        output_keys += ['module_uid']
    
    output_keys = pd.unique(['feature_map_uid'] + output_keys).tolist()
    
    layer_info_list = []
    feature_maps = {}
    all_available_keys = []
    for global_index, layer_info in enumerate(info_list):
        if info_is_torchinfo_summary:
            layer_info = info_tracker.update(layer_info)
            
        feature_map_uid = layer_info['feature_map_uid']
        layer_info = modify_keys(layer_info, key_mods)
        
        is_leaf_layer = layer_info.get('is_leaf_layer', False)
        is_recursive =  layer_info.get('is_recursive', False)
        output_tensor = layer_info.pop('output_tensor', None)
        
        if output_tensor is not None:
            feature_maps[feature_map_uid] = output_tensor
            
        batch_size = layer_info['output_shape'][batch_dim]
        layer_info['total_features'] = np.prod(layer_info['output_shape'])
        
        if remove_batch_dim:
            for key in ['input_shape', 'output_shape']:
                layer_info[key] = layer_info[key][:batch_dim]+layer_info[key][batch_dim+1:]

            layer_info['output_bytes'] //= batch_size
            layer_info['total_features'] //= batch_size
            
        all_available_keys += [key for key in layer_info if key not in all_available_keys]
        
        layer_info_list.append(layer_info)
    
    layer_info_dict = {layer_info['feature_map_uid']: layer_info for
                           layer_info in layer_info_list}
    
    matched_feature_maps = {}
    if len(feature_maps) > 0:
        match_kwargs = {'match_to_keep': 'first', 'metadata': layer_info_dict, 
                        'filter_kwargs': {'key': 'is_leaf_layer', 'value': True}}
        
        matched_feature_maps = get_matched_feature_maps(feature_maps, **match_kwargs)

    feature_map_index = 0
    layer_info_list = []
    nonduplicate_index = 0
                                                
    for layer_info in layer_info_dict.values():
        feature_map_uid = layer_info['feature_map_uid']
        
        if layer_info['is_leaf_layer']:
            feature_map_index += 1
        
        layer_info['duplicate_uids'] = pd.NA
        layer_info['is_duplicate'] = False
        for match_set, match_to_keep in matched_feature_maps.items():
            if feature_map_uid in match_set:
                layer_info['duplicate_uids'] = match_set
                
                if feature_map_uid != match_to_keep:
                    layer_info['is_duplicate'] = True
                    
                for key in ['duplicate_uids', 'is_duplicate']:
                     if key not in all_available_keys:
                            all_available_keys += [key]
            
                break; # match identified; continue processing
        
        layer_info['effective_depth'] = nonduplicate_index
        if not layer_info['is_duplicate']:
            nonduplicate_index += 1
                
        keep_layer = True
        if remove_duplicates:
            keep_layer = False
            if not layer_info['is_duplicate']:
                keep_layer = True
            
        if filters is not None:
            for key, filter_kwargs in filters.items():
                keep_layer = False
                if make_filter(layer_info[key], **filter_kwargs):
                    keep_layer = True
                
        if keep_layer:
            layer_info_list.append(layer_info)

    for effective_index, layer_info in enumerate(layer_info_list):
        feature_map_depth = layer_info['feature_map_index']
        layer_info['effective_depth'] /= (nonduplicate_index-1)
        if layer_info['is_leaf_layer']:
            layer_info_dict[layer_info['parent_uid']] = layer_info['effective_depth']
        layer_info['feature_map_depth'] = feature_map_depth / (feature_map_index)
        
        for key in ['feature_map_depth', 'effective_depth']:
            if key not in all_available_keys:
                all_available_keys += [key]
    
    all_available_keys = pd.unique(all_available_keys).tolist()
    output_keys = [key for key in output_keys if key in all_available_keys]
    
    if not to_pandas:
        output_keys = [key for key in output_keys if key != 'feature_map_uid']
        layer_info = {layer_info.pop('feature_map_uid', f'features_{i}'): 
                      {k: layer_info[k] for k in output_keys} 
                      for i, layer_info in enumerate(layer_info_list)}
        
    if to_pandas:
        model_name = layer_info_list[0]['module_class']
        layer_info_list = [{key: pd.NA if value is None else value 
                            for key, value in layer_info.items()} 
                           for layer_info in layer_info_list] #for NAs
        
        if output_kwargs.get('uid_keys_first', False):
            reordered_keys = ([key for key in id_keys if key in output_keys] +
                              [key for key in data_keys if key in output_keys])
            output_keys = pd.unique(['feature_map_uid'] + reordered_keys).tolist()
    
        layer_info = pd.DataFrame(layer_info_list)[output_keys]   
        layer_info = layer_info.where(~pd.isna(layer_info), '')
        
        if complete_uid:
            for key in info_tracker.all_uid_keys:
                layer_info[key] = layer_info[key].apply(lambda x: uid_na if x == '' else x)
        
        if output_kwargs.get('drop_constants', False):
            layer_info = drop_constant_columns(layer_info)
            
        if output_kwargs.get('drop_model', False):
            layer_info = layer_info.query('feature_map_uid != @model_name')
        
    return layer_info

### Feature Map Grouping + Indexing ----------------------------------------

def get_feature_maps_by_index(feature_maps, index=0, verbose=True):
    feature_map_uids = list(feature_maps.keys())
    
    if isinstance(index, list):
        output_uids = [feature_map_uids[i] for i in index]
        output_maps = [feature_maps[feature_map_uids[i]]
                           for i in index]
    
    if not isinstance(index, list):
        output_uids = feature_map_uids[index]
        output_maps = feature_maps[feature_map_uids[index]]
    
    if verbose: print(f'Getting {output_uids}...'); return output_maps

def get_feature_maps_by_depth(feature_maps, depth_lower=None, depth_upper=None, uids_only=False):
    
    if depth_lower is None: depth_lower = 0
    if depth_upper is None: depth_upper = 1
    
    n_feature_maps = len(feature_maps)
    
    map_depths = {uid: index / n_feature_maps for index, uid 
                          in enumerate(feature_maps)}
    
    output_maps = {uid: features for uid, features in feature_maps.items()
                   if map_depths[uid] >= depth_lower
                   and map_depths[uid] <= depth_upper}
    
    if not uids_only:
        return output_maps
    
    return list(output_maps.keys())

def get_feature_map_groups(metadata, feature_maps=None, group_var='parent_uid', 
                           uid_key='feature_map_uid', rename_bool=True, 
                           tensor_fns={}, group_fns={}, **other_kwargs):
    
    if group_var not in metadata.columns:
        raise ValueError('requested group_var is not a valid metadata key')
    
    tensor_fns = parse_fns(tensor_fns, **other_kwargs)
    group_fns = parse_fns(group_fns, **other_kwargs)
    
    if metadata[group_var].dtype == bool and rename_bool:
        metadata = metadata.copy()
        name = group_var
        if 'is_' in group_var:
            name = name.split('_')[-1]
        lambda_fn = lambda x: f'is_{name}' if x==True else f'is_not_{name}'
        metadata[group_var] = metadata[group_var].apply(lambda_fn)
        
    group_dict = metadata.groupby(group_var)[uid_key].apply(list).to_dict()
    
    if feature_maps is None:
        return group_dict
    
    if feature_maps is not None:
        group_dict = {key: {uid: feature_maps[uid] for uid in value if uid in feature_maps}
                      for key, value in group_dict.items()}
        
        group_dict = {key: value for key, value in group_dict.items() if len(value) >= 1}
        
        if len(tensor_fns) >= 1:
            group_dict = process_tensor(group_dict, tensor_fns=tensor_fns)
        
        if len(group_fns) >= 1:
            for group_fn, fn_kwargs in group_fns.items():
                group_dict = {key: group_fn(group, **fn_kwargs)
                              for key, group in group_dict.items()} 
            
        return group_dict