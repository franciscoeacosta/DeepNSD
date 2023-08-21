### Auxiliary Functions: Feature Extraction ---------------------------------------------------------

from torchvision.utils import make_grid
from feature_extraction import *

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_weights_dtype(model):
    module = list(model.children())[0]
    if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
        return module.weight.dtype
    if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
        return get_weights_dtype(module)

def get_dataloader_sample(dataloader, nrow = 5, figsize = (5,5), title=None):
    dataloader.dataset.transforms = None
    image_batch = next(iter(dataloader))
    batch_size = image_batch.shape[0]
    image_grid = make_grid(image_batch, nrow = batch_size // nrow)
    if reverse_transforms:
        image_grid = reverse_transforms(image_grid)
    plt.figure(figsize=figsize)
    plt.imshow(image_grid)
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
### Auxiliary Functions: Mapping Methods ---------------------------------------------------------

from mapping_methods import *

def get_best_alpha_index(regression):
    best_score = 0; best_alpha_index = 0
    for alpha_index, alpha_value in enumerate(regression.alphas):
        score = score_func(xy['train']['y'], regression.cv_values_[:, :, alpha_index].squeeze()).mean()
        if score >= best_score:
            best_alpha_index, best_score = alpha_index, score
            
    return best_alpha_index