import torch, math
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
from torchmetrics.functional import concordance_corrcoef, explained_variance

from .structural import convert_to_tensor

### Setup: Data Handling + Helpers -------------------------------------------------
    
def unify_dtypes(*args, target_dtype=None, precision='lowest'):
    dtypes_order = [torch.float16, torch.float32, torch.float64]
    dtypes = [arg.dtype for arg in args if isinstance(arg, torch.Tensor)]
    if not dtypes:
        return args
    if not target_dtype:
        if precision == 'highest':
            target_dtype = max(dtypes, key=dtypes_order.index)
        elif precision == 'lowest':
            target_dtype = min(dtypes, key=dtypes_order.index)
    result = tuple(arg.clone().to(dtype=target_dtype) 
                   if isinstance(arg, torch.Tensor) else arg for arg in args)
    return result[0] if len(result) == 1 else result
    
class FeatureMapDataset(Dataset):
    def __init__(self, feature_maps, splithalf=False):
        self.model_layers = [key for key in feature_maps.keys()]
        self.feature_maps = [val for val in feature_maps.values()]
        
        self.splithalf = splithalf

    def __len__(self):
        return len(self.feature_maps)

    def __getitem__(self, idx):
        if self.splithalf:
            model_layer = self.model_layers[idx]
            feature_map1 = self.feature_maps[idx,0::2]
            feature_map2 = self.feature_maps[idx,1::2]

            return model_layer, (feature_map1, feature_maps1)
            
        if not self.splithalf:
            model_layer = self.model_layers[idx]
            feature_map = self.feature_maps[idx]
            
            return model_layer, feature_map

### Encoding Models: Setup ---------------------------------------------------------

# score functions from torchmetrics.functional
_score_functions = {'spearmanr': spearman_corrcoef,
                    'pearsonr': pearson_corrcoef,
                    'concordance': concordance_corrcoef,
                    'explained_variance': explained_variance}

def get_scorer(score_type):
    return _score_functions[score_type]

class TorchEstimator:
    def __init__(self, dtype = None, device='cpu'):
        self.dtype = dtype
        self.device = device

    def to(self, device):
        self.device = device
        for attr, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.__setattr__(attr, value.to(device))
    def cuda(self):
        self.to('cuda')

    def cpu(self):
        self.to('cpu')
        
    def remove_from_cuda(self):
        self.to('cpu')
    
    def parse_input_data(self, *args, copy=False):
        args = convert_to_tensor(*args, copy=copy, device=self.device)
        if isinstance(args, torch.Tensor):
            args = (args, )
        args = unify_dtypes(*args, target_dtype=self.dtype)
        return args
        
    def preprocess_data(self, X, y, center=[], scale=[], scaler='standard',
                        output=None, save_to_class=False, **kwargs):
    
        stats = {f'{var}_{stat}': None for stat in ['mean','std','offset','scale'] for var in ['X', 'y']}

        X, y = self.parse_input_data(X, y)
        
        def parse_preprocessing_args(*args):
            parsed_args = []
            for arg in args:
                if arg is None or len(arg) == 0:
                    parsed_args.append('none')
                elif isinstance(arg, list):
                    parsed_args.append(''.join(arg))
                else:
                    parsed_args.append(arg)
            return tuple(parsed_args)

        center, scale, output = parse_preprocessing_args(center, scale, output)
        
        if kwargs.get('fit_intercept', False):
            center += 'x'

        if 'x' in center.lower():
            stats['X_mean'] = X.mean(dim = 0)
        if 'y' in center.lower():
            stats['y_mean'] = y.mean(dim = 0) if y.ndim > 1 else y.mean()
            
        if 'x' in scale.lower():
            stats['X_std'] = X.std(dim=0, correction=1)
            stats['X_std'][stats['X_std'] == 0.0] = 1.0  
        if 'y' in scale.lower():
            stats['y_std'] = y.std(dim=0, correction=1)
            stats['y_std'][stats['y_std'] == 0.0] = 1.0 
        
        if 'x' in center.lower():
            X -= stats['X_mean']
        if 'y' in center.lower():
            y -= stats['y_mean']
            
        if 'x' in scale.lower():
            X /= stats['X_std']
        if 'y' in scale.lower():
            y /= stats['y_std']

        if output == 'mean_std':
            if stats['X_mean'] is None:
                stats['X_mean'] = X.mean(dim=0)
            if stats['y_mean'] is None:
                stats['y_mean'] = y.mean(dim = 0) if y.ndim > 1 else y.mean()
            if stats['X_std'] is None:
                stats['X_std'] = torch.ones(X.shape[1], dtype=X.dtype,  device=X.device)
            if stats['y_std'] is None:
                stats['y_std'] = torch.ones(y.shape[1], dtype=y.dtype,  device=y.device)
                
        if output == 'offset_scale':
            stats['X_offset'] = stats.pop('X_mean', None)
            stats['y_offset'] = stats.pop('y_mean', None)
            stats['X_scale'] = stats.pop('X_std', None)
            if stats['X_offset'] is None:
                stats['X_offset'] = torch.zeros(X.shape[1], dtype=X.dtype, device=X.device)
            if stats['y_offset'] is None:
                stats['y_offset'] = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
            if stats['X_scale'] is None:
                stats['X_scale'] = torch.ones(X.shape[1], dtype=X.dtype,  device=X.device)

        if save_to_class:
            for stat, value in stats.items():
                if value is not None:
                    setattr(self, stat, value)

            return X, y

        if not save_to_class:
            if output == 'offset_scale':
                return X, y, stats['X_offset'], stats['y_offset'], stats['X_scale']

            if output == 'mean_std':
                return X, y, stats['X_mean'], stats['y_mean'], stats['X_std'], stats['y_std']
            
            return X, y

### Encoding Models: TorchRidgeGCV ---------------------------------------------------------
      
class TorchRidgeGCV(TorchEstimator):
    """Ridge regression with built-in Leave-one-out Cross-Validation. """
    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        fit_intercept=True,
        scale_X=True,
        scoring='pearsonr',
        store_cv_values=False,
        alpha_per_target=True,
        device = 'cpu'
    ):
        super().__init__()
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.scale_X = scale_X
        self.scoring = scoring
        self.device = device
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target
            
        if isinstance(scoring, str):
            self.scorer = get_scorer(self.scoring)
        
    @staticmethod
    def _decomp_diag(v_prime, Q):
        return (v_prime * Q**2).sum(axis=-1)
    
    @staticmethod
    def _diag_dot(D, B):
        if len(B.shape) > 1:
            D = D[(slice(None),) + (None,) * (len(B.shape) - 1)]
        return D * B
    
    @staticmethod
    def _find_smallest_angle(query, vectors):
        abs_cosine = torch.abs(torch.matmul(query, vectors))
        return torch.argmax(abs_cosine).item()
    
    def _compute_gram(self, X, sqrt_sw):
        X_mean = torch.zeros(X.shape[1], dtype=X.dtype, device = X.device)
        return X.matmul(X.T), X_mean
    
    def _eigen_decompose_gram(self, X, y, sqrt_sw):
        K, X_mean = self._compute_gram(X, sqrt_sw)
        if self.fit_intercept:
            K += torch.outer(sqrt_sw, sqrt_sw)
        eigvals, Q = torch.linalg.eigh(K)
        QT_y = torch.matmul(Q.T, y)
        return X_mean, eigvals, Q, QT_y
    
    def _solve_eigen_gram(self, alpha, y, sqrt_sw, X_mean, eigvals, Q, QT_y):
        w = 1.0 / (eigvals + alpha)
        if self.fit_intercept:
            normalized_sw = sqrt_sw / torch.linalg.norm(sqrt_sw)
            intercept_dim = self._find_smallest_angle(normalized_sw, Q)
            w[intercept_dim] = 0  # cancel regularization for the intercept

        c = torch.matmul(Q, self._diag_dot(w, QT_y))
        G_inverse_diag = self._decomp_diag(w, Q)
        # handle case where y is 2-d
        if len(y.shape) != 1:
            G_inverse_diag = G_inverse_diag[:, None]
        return G_inverse_diag, c
    
    def fit(self, X, y):
        self.alphas = torch.as_tensor(self.alphas, dtype=torch.float32)
        
        preprocessing_kwargs = {'output': 'offset_scale'}
        if self.fit_intercept:
            preprocessing_kwargs['center'] = 'x'
        if self.scale_X:
            preprocessing_kwargs['scale'] = 'x'

        X, y, X_offset, y_offset, X_scale = self.preprocess_data(X, y, **preprocessing_kwargs)

        decompose = self._eigen_decompose_gram
        solve = self._solve_eigen_gram

        sqrt_sw = torch.ones(X.shape[0], dtype=X.dtype, device = X.device)

        X_mean, *decomposition = decompose(X, y, sqrt_sw)

        n_y = 1 if len(y.shape) == 1 else y.shape[1]
        n_alphas = 1 if self.alphas.ndim == 0 else len(self.alphas)

        if self.store_cv_values:
            self.cv_values_ = torch.empty((*y.shape, n_alphas), dtype=X.dtype, device=X.device)

        best_alpha, best_coef, best_score, best_y_pred = [None]*4

        for i, alpha in enumerate(torch.atleast_1d(self.alphas)):
            G_inverse_diag, coef = solve(float(alpha), y, sqrt_sw, X_mean, *decomposition)
            y_pred = y - (coef / G_inverse_diag)
            if self.store_cv_values:
                self.cv_values_[:,:,i] = y_pred

            score = self.scorer(y, y_pred)
            if not self.alpha_per_target:
                score = self.scorer(y, y_pred).mean()

            # Keep track of the best model
            if best_score is None: 
                best_alpha = alpha
                best_coef = coef
                best_score = score
                best_y_pred = y_pred
                if self.alpha_per_target and n_y > 1:
                    best_alpha = torch.full((n_y,), alpha)
                    
            else: 
                if self.alpha_per_target and n_y > 1:
                    to_update = score > best_score
                    best_alpha[to_update] = alpha
                    best_coef[:, to_update] = coef[:, to_update]
                    best_score[to_update] = score[to_update]
                    best_y_pred[:, to_update] = y_pred[:, to_update]
                    
                elif alpha_score > best_score:
                    best_alpha, best_coef, best_score, best_y_pred = alpha, coef, score, y_pred

        self.alpha_ = best_alpha
        self.score_ = best_score
        self.dual_coef_ = best_coef
        self.coef_ = self.dual_coef_.T.matmul(X) 
        self.cv_y_pred_ = best_y_pred

        X_offset += X_mean * X_scale
        if self.fit_intercept:
            self.coef_ = self.coef_ / X_scale
            self.intercept_ = y_offset - torch.matmul(X_offset, self.coef_.T)
        else:
            self.intercept_ = torch.zeros(1, device=self.coef_.device)

        return self
    
    def predict(self, X):
        X = self.parse_input_data(X)
        return X.matmul(self.coef_.T) + self.intercept_
    
    def score(self, X, y):
        X, y = self.parse_input_data(X)
        return self.scorer(y, self.predict(X))


### RSA: Compare RDMS ------------------------------------------------------------

_compare_rdms_by = {'spearman': spearman_corrcoef,
                    'pearson': pearson_corrcoef,
                    'concordance': concordance_corrcoef}

def extract_rdv(X):
    return X[torch.triu(torch.ones_like(X, dtype=bool), diagonal=1)]

def compare_rdms(rdm1, rdm2, method = 'pearson', device = None, **method_kwargs):
    rdm1, rdm2 = convert_to_tensor(rdm1, rdm2)
    
    if device is None:
        rdm2 = rdm2.to(rdm1.device)
    else:
        rdm1 = rdm1.to(device)
        rdm2 = rdm2.to(device)
        
    rdm1_triu = extract_rdv(rdm1)
    rdm2_triu = extract_rdv(rdm2)
    
    return _compare_rdms_by[method](rdm1_triu, rdm2_triu, **method_kwargs).item()

def fisherz(r, eps=1e-5):
    return torch.arctanh(r-eps)

def fisherz_inv(z):
    return torch.tanh(z)

def average_rdms(rdms):
    return (1 - fisherz_inv(fisherz(torch.stack([1 - rdm for rdm in rdms]))
                            .mean(axis = 0, keepdims = True).squeeze()))

### RSA: Calculate RDMS -----------------------------------------------------------

def compute_rdm(data, method = 'pearson', norm=False, device=None, **rdm_kwargs):
    rdm_args = (data, norm, device)
    if method == 'euclidean':
        return compute_euclidean_rdm(*rdm_args, **rdm_kwargs)
    if method == 'pearson':
        return compute_pearson_rdm(*rdm_args, **rdm_kwargs)
    if method == 'spearman':
        return compute_spearman_rdm(*rdm_args, **rdm_kwargs)
    if method == 'mahalanobis':
        return compute_mahalanobis_rdm(*rdm_args, **rdm_kwargs)
    if method == 'concordance':
        return compute_concordance_rdm(*rdm_args, **rdm_kwargs)

def compute_mahalanobis_rdm(data, norm = False, device=None):
    data = convert_to_tensor(data, device=device)
    cov_matrix = torch.cov(data.T)
    inv_cov_matrix = torch.inverse(cov_matrix)
    centered_data = data - torch.mean(data, axis=0)
    kernel = centered_data @ inv_cov_matrix @ centered_data.T
    rdm = torch.diag(kernel).unsqueeze(1) + torch.diag(kernel).unsqueeze(0) - 2 * kernel
    return rdm / data.shape[1] if norm else rdm
    
def compute_pearson_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    rdm = 1 - torch.corrcoef(data)
    return rdm / data.shape[1] if norm else rdm

def compute_spearman_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    rank_data = data.argsort(dim=1).argsort(dim=1)
    rdm = 1 - torch.corrcoef(rank_data)
    return rdm / data.shape[1] if norm else rdm
    
def compute_euclidean_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    rdm = torch.cdist(data, data, p=2.0)
    rdm = rdm.fill_diagonal_(0)
    return rdm / data.shape[1] if norm else rdm

def compute_concordance_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    mean_matrix = data.mean(dim=1)
    var_matrix = data.var(dim=1)
    std_matrix = data.std(dim=1)
    corr_matrix = torch.corrcoef(data)
    numerator = 2 * corr_matrix * std_matrix[:, None] * std_matrix[None, :]
    denominator1 = var_matrix[:, None] + var_matrix[None, :]
    denominator2 = (mean_matrix[:, None] - mean_matrix[None, :]) ** 2
    rdm =  1 - (numerator / (denominator1 + denominator2))
    return rdm / data.shape[1] if norm else rdm

def get_rdms_by_indices(data, indices, method='pearson', device='cpu', **rdm_kwargs):
    def get_rdm(data, index):
        if isinstance(data, (np.ndarray, torch.Tensor)):
            rdm_data = data[index, :].T
        if isinstance(data, pd.DataFrame):
            rdm_data = data.loc[index].to_numpy().T
        return compute_rdm(rdm_data, method, device=device, **rdm_kwargs)

    if isinstance(indices, (np.ndarray, torch.Tensor)):
        return get_rdm(data, indices)

    if isinstance(indices, dict):
        rdms_dict = {}
        for key, index in indices.items():
            rdms_dict[key] = get_rdms_by_indices(data, index, method, device, **rdm_kwargs)
        return rdms_dict
    
def clean_nan_rdms(rdm_data):
    def rdm_nan_check(rdm):
        return rdm if torch.sum(torch.isnan(rdm) == 0) else None
    
    if isinstance(rdm_data, (np.ndarray, torch.Tensor)):
        return rdm_nan_check(rdm_data)

    if isinstance(rdm_data, dict):
        cleaned_dict = {}
        for key, data in rdm_data.items():
            cleaned_data = clean_nan_rdms(data)
            if cleaned_data is not None:  
                cleaned_dict[key] = cleaned_data
        return cleaned_dict
    
def get_traintest_rdms(rdm_data, test_idx=None):
    def get_traintest_rdm(rdm, test_idx):
        if test_idx is not None:
            if isinstance(test_idx, np.ndarray):
                test_idx = torch.tensor(test_idx)
                
            train_idx = torch.ones(rdm.shape[0], dtype=torch.bool)
            train_idx[test_idx] = False

            return {'train': rdm[train_idx, train_idx], 
                    'test': rdm[test_idx, test_idx]}
        
        return {'train': rdm[::2, ::2], 'test': rdm[1::2, 1::2]}

    if isinstance(rdm_data, (np.ndarray, torch.Tensor)):
        return get_traintest_rdm(rdm_data, test_idx)

    if isinstance(rdm_data, dict):
        rdms_dict = {}
        for key, data in rdm_data.items():
            rdms_dict[key] = get_traintest_rdms(data, test_idx)
        return rdms_dict

### CKA Methods -----------------------------------------------------------

class TorchCKA():
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
