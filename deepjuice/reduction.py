import numpy as np
import cupy as cp
import torch, scipy

from tqdm.auto import tqdm
import multiprocessing
from mtalg.random import MultithreadedRNG

from .structural import TorchSKLearn
                
### Dimensionality Reduction Meta -----------------------------------------------------

class FeatureMapRedux(TorchSKLearn):
    def __init__(self, n_srps = 'auto', eps = 0.1, seed = 0,
                 fit_pca = False, n_pcs = None, 
                 device = 'cpu', verbose = False):
        super().__init__()
        
        self.n_srps = n_srps
        if n_srps == 'auto':
            self.eps = eps
            self.seed = seed
            
        self.fit_pca = fit_pca
        self.n_pcs = n_pcs
        self.verbose = True
        self.device = device
        
        self.srp_fitted = False
        self.pca_fitted = False
        
        cuda_report = (f'using {device.upper()} for computation,'+
                       f'outputting data on {device.upper()}.')
        
        if n_srps == 'auto' and verbose:
            print(f'Computing # SRPs with JL-Lemma + epsilon = {eps}, {cuda_report}')
        if isinstance(n_srps, int):
            print('Overriding default (recommended) use of JL-Lemma, ' + 
                  f'and computing {n_srps} SRPs for all input data.')
        
        if n_pcs is not None and verbose:
            if isinstance(n_pcs, int):
                print(f'Computing # {PCs} after sparse random projection, {cuda_report}')
            if n_pcs == 'auto':
                print(f'Computing all PCs after sparse random projection, {cuda_report}')
        
            
    def fit(self, feature_map, srp_kwargs={}, pca_kwargs={}):
        feature_map = self.parse_input_data(feature_map)
        self.n_samples, self.n_features = feature_map.shape
        
        srp = TorchSRP(self.n_srps, self.eps, self.seed, self.device, **srp_kwargs)
        srp.fit(feature_map)
        self.srp_fitted = True
        
        self.n_srps = srp.n_components
        self.srp_matrix = srp.srp_matrix
            
        if self.verbose: print('Sparse random projection complete.')
        
        if self.fit_pca or self.n_pcs is not None:
            feature_map_srp = srp.transform(feature_map)
            
            n_components = self.n_pcs
            if not isinstance(self.n_pcs, int):
                n_components = self.n_samples
 
            pca = TorchPCA(n_components, self.device, **pca_kwargs)
            pca.fit(feature_map_srp)
            self.pca_fitted = True
            self.pca_matrix = pca.components_
            self.pca_mean = pca.mean
            self.pca_data = pca.to_pandas(pca)
            
            if self.n_pcs == 'auto':
                self.pca_matrix = pca.get_top_n_components()
                self.n_pcs = self.pca_matrix.shape[1]
                pc_report = f'Returning top {self.n_pcs} PCs, explaining 90% variance.'
                
            if isinstance(self.n_pcs, float):
                ev_threshold = self.n_pcs
                self.pca_matrix = pca.get_top_n_components(ev_threshold=ev_threshold)
                self.n_pcs = self.pca_matrix.shape[1]
                pc_report = f'Returning top {self.n_pcs} PCs, explaining {ev_threshold} variance.'
            
            else: self.n_pcs = n_components; pc_report = 'Returning all PCs.'

            if self.verbose: print(f'Principal components analysis complete. {pc_report}') 
            
        return self
    
    def transform(self, feature_map):
        feature_map = self.parse_input_data(feature_map)
        srp_map = torch.sparse.mm(self.srp_matrix, feature_map.T).T
        if self.pca_fitted: 
            srp_map -= self.pca_mean
            return torch.mm(srp_map, self.pca_matrix)
        return srp_map
    
    def fit_transform(self, feature_map):
        feature_map = self.parse_input_data(feature_map)
        self.fit(feature_map)
        return self.transform(feature_map)
    
    def inverse_transform(self, feature_map):
        feature_map = self.parse_input_data(feature_map)
        if self.pca_fitted:
            feature_map = torch.mm(X, self.components_.T) + self.pca_mean
        inverse_srp_matrix = torch.pinverse(self.srp_matrix.to_dense())
        return torch.mm(X, inverse_srp_matrix.T)
                
### Sparse Random Projection ----------------------------------------------------------

class TorchSRP(TorchSKLearn):
    def __init__(self, n_components = 'auto', eps = 0.1, seed = 0, device='cpu'):
        super().__init__(device=device)
        
        self.n_components = n_components
        if n_components == 'auto':
            self.eps = eps
            self.seed = seed
            
        self.device = device
        
    def fit(self, X, **srp_kwargs):
        X = self.parse_input_data(X)
        
        self.n_samples, self.n_features = X.shape
        if self.n_components == 'auto':
            self.n_components = jl_lemma(self.n_samples, eps=self.eps)
        
        srp_matrix_args = [self.n_components, self.n_features, self.seed]
        
        if 'cuda' in self.device:
            srp_kwargs = {'srp_matrix_type': 'torch.csr', **srp_kwargs}
            srp_matrix_args += [self.device]
            self.srp_matrix = make_srp_matrix_cuda(*srp_matrix_args, **srp_kwargs)
            
            cp.get_default_memory_pool().free_all_blocks()
            
        if self.device == 'cpu':
            self.srp_matrix = make_srp_matrix(*srp_matrix_args, **srp_kwargs)
            
        return self
            
    def transform(self, X):
        X = self.parse_input_data(X)
        srp_matrix = self.srp_matrix
        return torch.sparse.mm(srp_matrix, X.T).T
    
    def fit_transform(self, X):
        X = self.parse_input_data(X)
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        inverse_srp_matrix = torch.pinverse(self.srp_matrix.to_dense())
        X = self.parse_input_data(X)
        return torch.mm(X, inverse_srp_matrix.T)

def jl_lemma(n_samples, *, eps = 0.1):
    eps, n_samples = np.asarray(eps), np.asarray(n_samples)
    denominator = (eps**2 / 2) - (eps**3 / 3)
    return (4 * np.log(n_samples) / denominator).astype(np.int64)

def parse_device(device, backend='torch'):
    if isinstance(device, int):
        device = device
    elif 'cuda' in device:
        try: device = int(device.split(':')[-1])
        except: device = 0
    else:
        raise ValueError(f"Unrecognized device: {device}. " + 
                         "Expected a cuda device as 'cuda:X' or a 'cpu'.")
        
    if backend.lower() == 'torch':
        device = 'cuda:' + str(device) if device != 'cpu' else 'cpu'
    elif backend.lower() == 'cupy':
        device = device if device != 'cpu' else -1

    return device   

def init_precision(precision):
    if precision not in [16, 32, 64]:
        raise ValueError("Precision should be one of the following: 16, 32, 64")

    dtypes = {numtype: {bits: numtype+str(bits) for bits in [16,32,64]} 
              for numtype in ['int','float']}
    
    return {library: {numtype: getattr(eval(library), dtypes[numtype][precision]) 
                      for numtype in ['int','float']} for library in ['np','cp','torch']}

def make_srp_matrix_cuda(n_components, n_features, seed=0, device='cuda', precision=32, 
                         batch_size = 256, srp_matrix_type = 'cupy.csr'):
    
    # parse the device argument for use in cupy
    cp.cuda.Device(parse_device(device, 'cupy')).use()
    dtypes = init_precision(precision)

    if seed is not None: cp.random.seed(seed)
    
    matrix_size = (n_components, n_features)
    density = 1 / cp.sqrt(n_features)
    
    srp_matrix_format = srp_matrix_type.split('.')[1]

    col_indices = cp.array([], dtype=dtypes['cp']['int'])
    if srp_matrix_format == 'coo':
        row_indices = cp.array([ ], dtype=dtypes['cp']['int'])
    if srp_matrix_format == 'csr':
        row_indices = cp.array([0], dtype=dtypes['cp']['int'])
    for i in range(0, n_components, batch_size):
        # Determine the size of the current batch
        current_batch_size = min(batch_size, n_components - i)

        # Generate a mask for non-zero elements in the current batch
        nonzero_mask = cp.random.binomial(1, density, size=(current_batch_size, n_features)).astype(bool)
        
        # Get the indices of non-zero elements in the current batch
        nonzero_row_indices, nonzero_col_indices = cp.nonzero(nonzero_mask)
        
        # Extend row + col indices with the current batch's non-zero indices
        col_indices = cp.concatenate((col_indices, nonzero_col_indices))

        if srp_matrix_format == 'coo':
            row_indices = cp.concatenate((row_indices, nonzero_row_indices + i))

        if srp_matrix_format == 'csr':
            # Get the number of non-zero elements per row; update indptr
            n_nonzero_per_row = cp.sum(nonzero_mask, axis=1)
            row_indices = cp.concatenate((row_indices, cp.cumsum(n_nonzero_per_row) + row_indices[-1]))

    # Among non zero components the probability of the sign is 50%/50%
    if srp_matrix_format == 'coo':
        nonzero_size = row_indices.size
    if srp_matrix_format == 'csr':
        nonzero_size = col_indices.size
        
    scale_factor = cp.sqrt(1 / density) / cp.sqrt(n_components)
        
    data = ((cp.random.binomial(1, 0.5, size=nonzero_size) * 2 - 1)
            .astype(dtypes['cp']['float']) * scale_factor)
    
    if 'torch' in srp_matrix_type:
        data = torch.from_dlpack((data).toDlpack()).to(dtypes['torch']['float'])
        
    if srp_matrix_type == 'cupy.coo':
        indices = cp.vstack((row_indices, col_indices))
        srp_matrix = cp.sparse.coo_matrix((data, indices), dtype = data.dtype, shape = matrix_size)

    if srp_matrix_type == 'cupy.csr':
        data_csr_indices = (data, col_indices, row_indices)
        srp_matrix = cp.sparse.csr_matrix(data_csr_indices, dtype = data.dtype, shape = matrix_size)
        
    if srp_matrix_type == 'torch.coo':
        indices = torch.from_dlpack(cp.vstack((row_indices, col_indices)).toDlpack())
        srp_matrix = torch.sparse_coo_tensor(indices, data, dtype = data.dtype,
                                             size = matrix_size, device = device)
    
    if srp_matrix_type == 'torch.csr':
        row_indices = torch.from_dlpack(row_indices.toDlpack())
        col_indices = torch.from_dlpack(col_indices.toDlpack())
        srp_matrix = torch.sparse_csr_tensor(row_indices, col_indices, data, dtype = data.dtype,
                                             size = matrix_size, device = device)
    
    # Clear CuPy remnants from GPU memory
    cp.get_default_memory_pool().free_all_blocks()
    
    return srp_matrix

def make_srp_matrix(n_components, n_features, seed=0, precision=32,
                    n_threads='auto', srp_matrix_type='torch.csr'):

    if n_threads == 'auto':
        n_threads = int(multiprocessing.cpu_count() / 2) 
        
    dtypes = init_precision(precision)
        
    if seed is not None: np.random.seed(seed)
    
    matrix_size = (n_components, n_features)
    mrng = MultithreadedRNG(seed=seed, num_threads=n_threads)
    rng = np.random.default_rng(seed)
    density = 1 / np.sqrt(n_features)
    n_nonzero = mrng.binomial(n_features, density, size=n_components).astype(dtypes['np']['int'])

    # Generate location of non zero elements
    n_nonzero = np.random.binomial(n_features, density, size=n_components).astype(dtypes['np']['int'])
    col_indices = []
    for i in range(n_components):
        indices_i = np.sort(rng.choice(n_features, n_nonzero[i], replace=False))
        col_indices.extend(indices_i)

    row_indices = np.insert(np.cumsum(n_nonzero), 0, 0)
    
    scale_factor = np.sqrt(1 / density) / np.sqrt(n_components)

    # Among non zero components the probability of the sign is 50%/50%
    data = ((np.random.binomial(1, 0.5, size=int(n_nonzero.sum())) * 2 - 1)
            .astype(dtypes['np']['float']) * scale_factor)

    col_indices = np.array(col_indices)
    
    if 'coo' in srp_matrix_type:
        row_indices = np.repeat(np.arange(n_components), np.diff(row_indices))
    
    if 'torch' in srp_matrix_type:
        data = torch.from_numpy(data).to(dtypes['torch']['float'])

    if srp_matrix_type == 'scipy.coo':
        indices = np.vstack((row_indices, col_indices))
        srp_matrix = scipy.sparse.coo_matrix((data, indices), dtype = data.dtype, shape = matrix_size)

    if srp_matrix_type == 'scipy.csr':
        data_csr_indices = (data, col_indices, row_indices)
        srp_matrix = scipy.sparse.csr_matrix(data_csr_indices, dtype = data.dtype, shape = matrix_size)
        
    if srp_matrix_type == 'torch.coo':
        indices = torch.from_numpy(np.vstack((row_indices, col_indices)))
        srp_matrix = torch.sparse_coo_tensor(indices, data, dtype = data.dtype,
                                             size = matrix_size, device = 'cpu')
    
    if srp_matrix_type == 'torch.csr':
        srp_matrix = torch.sparse_csr_tensor(row_indices, col_indices, data, 
                                             dtype = data.dtype,
                                             size = matrix_size, device = 'cpu')

    return srp_matrix

def sparse_random_projection(feature_matrix, eps = 0.1, seed = 0, device = 'cuda', 
                             output_format = 'torch', keep_srp_matrix=False, **srp_kwargs):
    
    n_samples, n_features = feature_matrix.shape
    n_components = jl_lemma(n_samples, eps = eps)
    
    if 'cuda' in device:
        srp_matrix = make_srp_matrix_cuda(n_components, n_features, seed, device, **srp_kwargs)
        if isinstance(feature_matrix, torch.Tensor):
            if feature_matrix.is_cuda:
                feature_matrix = cp.fromDlpack(feature_matrix.to_dlpack())
        
        if not isinstance(feature_matrix, cp.ndarray):
            feature_matrix = cp.asarray(feature_matrix)
            
        feature_srp = feature_matrix @ srp_matrix.T
        feature_srp = torch.from_dlpack(feature_srp.toDlpack())
        
        cp.get_default_memory_pool().free_all_blocks()
    
    if device == 'cpu':
        srp_matrix = make_srp_matrix(n_components, n_features, seed, **srp_kwargs)
        
        if isinstance(srp_matrix, (scipy.sparse.csr.csr_matrix, scipy.sparse.coo.coo_matrix)):
            if isinstance(feature_matrix, torch.Tensor):
                feature_matrix = feature_matrix.numpy()
                
            feature_srp = feature_matrix @ srp_matrix.T
            
        if isinstance(srp_matrix, torch.Tensor):
            if isinstance(feature_matrix, np.ndarray):
                feature_matrix = torch.from_numpy(feature_matrix)
                
            feature_srp = torch.sparse.mm(srp_matrix, feature_matrix.T).T
        
    if output_format == 'numpy':
        feature_srp = feature_srp.cpu().numpy()
    
    return (feature_srp, srp_matrix) if keep_srp_matrix else feature_srp

def get_feature_map_srps(feature_maps, eps=0.1, seed=0, device = 'cuda', cpu = True, **srp_kwargs):
    srp_args = [eps, seed, device]
    
    if isinstance(feature_maps, list):
        feature_map_srps = []
        for feature_map in tqdm(feature_maps, desc = 'SR Projection (Feature Map):'):
            feature_map_srp = sparse_random_projection(feature_map, *srp_args, **srp_kwargs)
            if isinstance(feature_map_srp, torch.Tensor):
                if cpu: feature_map_srp = feature_map_srp.cpu()
                
            feature_map_srps.append(feature_map_srp)
    
    elif isinstance(feature_maps, dict):
        feature_map_srps = {}
        for model_layer, feature_map in tqdm(feature_maps.items(), 'SR Projection (Layer)'):
            feature_map_srp = sparse_random_projection(feature_map, *srp_args, **srp_kwargs)
            if isinstance(feature_map_srp, torch.Tensor):
                if cpu: feature_map_srp = feature_map_srp.cpu()
                
            feature_map_srps[model_layer] = feature_map_srp
            
    else: feature_map_srps = sparse_random_projection(feature_maps, *srp_args, **srp_kwargs)
    
    return feature_map_srps

### Principal Components Analysis (PCA) -----------------------------------------------------

class TorchPCA(TorchSKLearn):
    def __init__(self, n_components=None, ev_threshold = None, device='cpu'):
        super().__init__(device=device)
        self.n_components = n_components
        self.ev_threshold = ev_threshold
        
    def fit(self, X):
        X = self.parse_input_data(X)
        self.mean = X.mean(dim=0)
        X = X - self.mean
        _, S, V = torch.svd(X)
        if self.n_components is None:
            self.n_components = X.shape[1]
        self.components_ = V[:, :self.n_components]
        
        ev = (S**2 / (X.size(0) - 1))
        ev_ratio = ev / ev[:self.n_components].sum()
        cumulative_ev_ratio = torch.cumsum(ev_ratio, dim=0)
        
        if self.ev_threshold is not None:
            self.n_components = (cumulative_ev_ratio <= 0.9).sum().item()
            
        self.ev_ = ev[:self.n_components]
        self.ev_ratio_ = self.ev_ / ev.sum()
        self.cumulative_ev_ = torch.cumsum(self.ev_, dim=0)
        self.cumulative_ev_ratio_ = torch.cumsum(self.ev_ratio_, dim=0)
        self.total_ev_ = self.cumulative_ev_[-1].item()
        self.total_ev_ratio_ = self.cumulative_ev_ratio_[-1].item()
        
        if self.replace_acronyms:
            self._replace_acronyms('ev_', 'explained_variance_')
        
        return self

    def transform(self, X):
        X = self.parse_input_data(X)
        X = X - self.mean
        return torch.mm(X, self.components_)

    def fit_transform(self, X):
        X = self.parse_input_data(X)
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        X = self.parse_input_data(X)
        return torch.mm(X, self.components_.T) + self.mean
    
    def get_top_n_components(self, n_components = 'auto', ev_threshold = 0.9):
        ev_name = 'explained_variance' if self.replace_acronyms else 'ev'
        ev_var = f'cumulative_{ev_name}_ratio_'
        if n_components == 'auto':
            n_components = (self.__dict__[ev_var] <= ev_threshold).sum().item()
        return self.components_[:, :n_components]
    
    def transform_by_top_n_components(self, X, n_components='auto', ev_threshold = 0.9):
        X = self.parse_input_data(X)
        X = X - self.mean
        top_n_components = self.get_top_n_components(n_components, ev_threshold)
        return torch.mm(X, top_n_components)
        
    
    def to_pandas(self, kind='explained_variance', feature_names=None, use_acronyms=True):
        if not isinstance(kind, list):
            kind = [kind]
        pandas_dataframes = []
        
        PCIDs = range(self.components_.shape[1])
        
        if 'explained_variance' in kind:
            data_keys = [e.replace('ev', 'explained_variance') for e in
                         ['ev','ev_ratio','cumulative_ev','cumulative_ev_ratio']]
            data_dict = {f'{key}': self.__dict__[key+'_'].cpu() for key in data_keys}
            dataframe = pd.DataFrame(data_dict)
            dataframe.insert(0,'pc_id', PCIDs)
            if use_acronyms:
                dataframe.columns = [col.replace('explained_variance_', 'ev_') 
                                     for col in dataframe.columns]
                
            pandas_dataframes.append(dataframe)
            
        if 'loadings' in kind:
            dataframe = pd.DataFrame(self.components_)
            dataframe.columns = [f'PC{i}' for i in PCIDs]
            if feature_names is not None:
                dataframe.index = feature_names
                
            pandas_dataframes.append(dataframe)
            
        outputs = pandas_dataframes
            
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    
def get_feature_map_pcs(feature_maps, n_components=None, ev_threshold=None, device = 'cuda', cpu = True):
    pca_args = [n_components, ev_threshold, device]
    
    if isinstance(feature_maps, list):
        feature_map_pcas = []
        for feature_map in tqdm(feature_maps, desc = 'PC Extraction (Feature Map):'):
            feature_map_pca = TorchPCA(*pca_args).fit_transform(feature_map)
            if isinstance(feature_map_srp, torch.Tensor):
                if cpu: feature_map_pca = feature_map_pca.cpu()
                
            feature_map_pcas.append(feature_map_pca)
    
    elif isinstance(feature_maps, dict):
        feature_map_srps = {}
        for model_layer, feature_map in tqdm(feature_maps.items(), 'PC Extraction (Layer)'):
            feature_map_pca = TorchPCA(*pca_args).fit_transform(feature_map)
            if isinstance(feature_map_srp, torch.Tensor):
                if cpu: feature_map_srp = feature_map_srp.cpu()
                
            feature_map_pcas[model_layer] = feature_map_pca
            
    else: feature_map_pcas = TorchPCA(*pca_args)(feature_maps)
    
    return feature_map_pcas