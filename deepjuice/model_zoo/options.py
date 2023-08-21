import pandas as pd
import numpy as np
import os, sys, torch
import importlib

from logging import warning
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
filepath = os.path.dirname(os.path.abspath(__file__))

custom_code_dir = filepath + '/custom_registry'
sys.path.append(custom_code_dir)

model_registry = pd.read_csv(filepath + '/registry.csv')
model_lookup = model_registry.set_index('model_uid').to_dict(orient='index')

### Setup ----------------------------------------------------------------------------------

def get_model_info(model_uid):
    return model_lookup[model_uid]

def subset_registry(model_source):
    return model_registry[model_registry['model_source'] == model_source].copy()

CACHED_TRANSFORMS = {} # a dictionary for storing transforms

def cache_transforms(model_uid, transforms):
    CACHED_TRANSFORMS[model_uid] = transforms
    
def get_model_options(filters = None, output = 'pandas', match = False):
    pd.Series.str.search = pd.Series.str.fullmatch if match else pd.Series.str.contains
    
    model_options = model_registry.copy()
    
    if isinstance(filters, str):
        def search(row): return row.astype(str).str.search(filters, na = False).any()
        query = model_registry.apply(lambda row: search(row), axis=1)
        model_options = model_registry[query].copy()
        
    if isinstance(filters, dict):
        model_options = model_registry.copy()
        for key, value in filters.items():
            if not key in model_options.columns:
                raise ValueError('{} not available in metadata'.format(key))
            model_options = model_options[model_options[key].str.search(value, na=False)]
        
    model_options.dropna(axis=1, how='all', inplace = True)
    
    if len(model_options) == 0:
        return 'No model_options available with this query.'
        
    if output == 'pandas':
        return model_options
    if output == 'list':
        return model_options['model_uid'].to_list()
    if output == 'dict':
        return model_options.set_index('model_uid').to_dict(orient='index')
    
def check_deepjuice_uid(model_uid):
    model_options = get_model_options()
    
    if model_uid not in model_options.model_uid.to_list():
        raise ValueError('No reference available for this model_uid.')
    

def get_deepjuice_model(model_uid, output = 'model_and_transforms'):
    check_deepjuice_uid(model_uid)

    model_info = get_model_info(model_uid)
    model_source = model_info['model_source']
    
    return eval('get_{}_model'.format(model_source))(model_uid, output)

def get_deepjuice_transforms(model_uid):
    check_deepjuice_uid(model_uid)
    
    model_info = get_model_info(model_uid)
    model_source = model_info['model_source']
    
    transforms_call = eval('get_{}_model'.format(model_source))
    
    if transforms_call in globals():
        return eval(transforms_call)
    
    if transforms_call not in globals():
        return get_transforms_only(model_source, model_uid)
    
    
### Transform Functions -------------------------------------------------------------------------

def get_transforms_only(model_source, model_uid):
    if model_uid in CACHED_TRANSFORMS:
        return CACHED_TRANSFORMS[model_uid]
    
    if not model_uid in CACHED_TRANSFORMS:
        return eval('get_{}_model'.format(model_source))(model_uid, output = 'transforms_only')

def get_imagenet_transforms(input_type = 'PIL'):
    import torchvision.transforms as transforms
    
    imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std':  [0.229, 0.224, 0.225]}
    
    imagenet_transforms = [transforms.Resize((224,224)), 
                           transforms.ToTensor(),
                           transforms.Normalize(**imagenet_stats)]
    
    if input_type == 'numpy':
        imagenet_transforms = [transforms.ToPILImage()] + imagenet_transforms
        
    return transforms.Compose(imagenet_transforms)

def replace_center_crop(transforms_compose):
    from torchvision.transforms import Compose
    if not isinstance(transforms_compose, Compose):
        raise ValueError('This functions works only a torchvisions.transforms.Compose object')
        
    transforms_list = transforms_compose.transforms
    
    if any(['CenterCrop' in str(transform) for transform in transforms_list]):
    
        crop_index, crop_size = next((index, transform.size) for index, transform 
                                     in enumerate(transforms_list) 
                                     if 'CenterCrop' in str(transform))

        resize_index, resize  = next((index, transform.size) for index, transform 
                                     in enumerate(transforms_list) 
                                     if 'Resize' in str(transform))

        transforms_list[resize_index].size = crop_size
        transforms_list.pop(crop_index)
        
    return Compose(transforms_list)

### Torchvision Models -------------------------------------------------------------------------

def get_torchvision_model(model_uid, output = 'model_and_transforms'):
    from torchvision.transforms import Resize, ToTensor, Normalize, Compose
    import torchvision.models as models
    
    model_info = get_model_info(model_uid)

    model_name = model_info['model_name']
    weights_dir, weights_name = model_info['weights_url'].split('.')

    model_directory = models.__dict__
    for subdir in ['detection','segmentation','video']:
        if subdir in model_directory:
            model_type = subdir
            subdir = model_directory[subdir].__dict__
            if model_name in subdir:
                model_directory = subdir

    weights = model_directory[weights_dir].__dict__[weights_name]
    model_call = model_directory[model_name]
    
    preprocess = weights.transforms()
    
    if isinstance(preprocess, Compose):
        preprocess = replace_center_crop(preprocess)
    
    if hasattr(preprocess, 'crop_size'):
        crop_size = preprocess.crop_size
        mean, std = preprocess.mean, preprocess.std
        preprocess = Compose([Resize(crop_size*2),
                              ToTensor(),
                              Normalize(mean, std)])
    
    if model_type in ['detection','segmentation']:
        standard_image_size = (256,256)
        warning('deepjuice uses a standard_image_size of {} for {} models'
                .format(standard_image_size, model_type))
        
        preprocess = Compose([Resize(standard_image_size), preprocess])
    
    CACHED_TRANSFORMS[model_uid] = preprocess
    
    if output == 'transforms_only':
        return preprocess
    
    model = model_call(weights = weights)
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess

    
### Timm Models -------------------------------------------------------------------------------

def get_timm_model(model_uid, output = 'model_and_transforms'):
    from timm.data.transforms_factory import create_transform
    from torchvision.transforms import Resize, Compose
    from timm.data import resolve_data_config
    from timm import create_model, list_models
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    model = create_model(model_name, pretrained = True)
    
    config = resolve_data_config({}, model = model)
    preprocess = create_transform(**config)
    
    if isinstance(preprocess, Compose):
        preprocess = replace_center_crop(preprocess)
    
    CACHED_TRANSFORMS[model_uid] = preprocess
    
    if output == 'transforms_only':
        return preprocess
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess


### Taskonomy Models ---------------------------------------------------------------------------

def get_taskonomy_transforms(image):
    return (functional.to_tensor(functional.resize(image, (256,256))) * 2 - 1)

def get_taskonomy_model(model_name, output = 'model_and_transforms'):
    from visualpriors.taskonomy_network import TASKONOMY_PRETRAINED_URLS
    from visualpriors import taskonomy_network
    
    model_info = get_model_info(model_name)
    model_name = model_info['model_name']
    
    preprocess = get_taskonomy_transforms
    
    if output == 'transforms_only':
        return preprocess
    
    model = taskonomy_network.TaskonomyEncoder()
    
    if 'random_weights' not in model_name:
        weights_url = TASKONOMY_PRETRAINED_URLS[model_name + '_encoder'] 
        weights = torch.utils.model_zoo.load_url(weights_url)
        model.load_state_dict(weights['state_dict'])
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess

    
### OpenAI-CLIP Models -------------------------------------------------------------------------

def get_openai_clip_model(model_uid, output = 'model_and_transforms'):
    from clip import load as load_clip_model
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    model, preprocess = load_clip_model(model_name, device = 'cpu')
    
    CACHED_TRANSFORMS[model_uid] = preprocess
    
    if output == 'transforms_only':
        return preprocess
    
    if output == 'model_only':
        return model.visual
    
    if output == 'model_and_transforms':
        return model.visual, preprocess


### OpenCLIP Models ---------------------------------------------------------------------------

def get_openclip_model(model_uid, output = 'model_and_transforms'):
    from open_clip import create_model_and_transforms
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    weights_name = model_info['weights']
    
    model, _, preprocess = create_model_and_transforms(model_name, pretrained = weights_name)
    
    CACHED_TRANSFORMS[model_uid] = preprocess
    
    if output == 'transforms_only':
        return preprocess
    
    if output == 'model_only':
        return model.visual
    
    if output == 'model_and_transforms':
        return model.visual, preprocess


### VISSL Models -------------------------------------------------------------------------------

get_vissl_transforms = get_imagenet_transforms

def get_vissl_model(model_uid, output = 'model_and_transforms'):
    from torch.hub import load_state_dict_from_url
    
    model_info = get_model_info(model_uid)
    weights_url = model_info['weights_url']
    
    preprocess = get_vissl_transforms()
    
    if output == 'transforms_only':
        return preprocess
    
    weights = load_state_dict_from_url(weights_url, map_location = torch.device('cpu'))
    
    def replace_module_prefix(state_dict, prefix, replace_with = ''):
        return {(key.replace(prefix, replace_with, 1) if key.startswith(prefix) else key): val
                      for (key, val) in state_dict.items()}

    def convert_model_weights(model):
        if "classy_state_dict" in model.keys():
            model_trunk = model["classy_state_dict"]["base_model"]["model"]["trunk"]
        elif "model_state_dict" in model.keys():
            model_trunk = model["model_state_dict"]
        else:
            model_trunk = model
        return replace_module_prefix(model_trunk, "_feature_blocks.")

    converted_weights = convert_model_weights(weights)
    excess_weights = ['fc','projection', 'prototypes']
    converted_weights = {key:value for (key,value) in converted_weights.items()
                             if not any([x in key for x in excess_weights])}
    
    if 'module' in next(iter(converted_weights)):
        converted_weights = {key.replace('module.',''):value for (key,value) in converted_weights.items()
                             if 'fc' not in key}
    
    from torchvision.models import resnet50
    import torch.nn as nn

    class Identity(nn.Module):
        def __init__(self):
            super(Identity, self).__init__()

        def forward(self, x):
            return x

    model = resnet50()
    model.fc = Identity()

    model.load_state_dict(converted_weights)
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess

    
### DINO Models -------------------------------------------------------------------------------

get_dino_transforms = get_imagenet_transforms

def get_dino_model(model_uid, output = 'model_and_transforms'):
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    preprocess = get_dino_transforms()
    
    if output == 'transforms_only':
        return preprocess
    
    model = torch.hub.load('facebookresearch/dino:main', model_name)
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess


### VICReg Models ------------------------------------------------------------------------------

get_vicreg_transforms = get_imagenet_transforms

def get_vicreg_model(model_uid, output = 'model_and_transforms'):
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    model_source = model_info['model_source']
    
    preprocess = get_vicreg_transforms()
    
    if output == 'transforms_only':
        return preprocess
    
    github_tag = 'facebookresearch/vicreg{}:main'.format('l' if 'alpha' in model_name else '')
    
    model = torch.hub.load(github_tag, model_name)
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess
    

### MiDaS Models -------------------------------------------------------------------------------

def get_midas_model(model_uid, output = 'model_and_transforms'):
    from torchvision.transforms import Resize, Compose
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    hub_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')

    midas_transforms = hub_transforms.default_transform
    
    if 'DPT' in model_name:
        midas_transforms = hub_transforms.dpt_transform
    if '_small' in model_name:
        midas_transforms = hub_transforms.small_transform
        
    transforms_lambda = ([lambda img: np.array(img)] + 
                         midas_transforms.transforms +
                         [lambda tensor: tensor.squeeze()])
    
    standard_image_size = (384,384)
    warning('deepjuice uses a standard_image_size of {} for MiDaS models'
            .format(standard_image_size))
        
    target_transforms = Compose([Resize(standard_image_size)] + transforms_lambda)
    
    preprocess = target_transforms
    
    CACHED_TRANSFORMS[model_uid] = preprocess
        
    if output == 'transforms_only':
        return preprocess
    
    model = torch.hub.load('intel-isl/MiDaS', model_name)
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess

    
### YOLOV5 Models -------------------------------------------------------------------------------

def get_yolov5_transforms():
    def yolov5_transforms(pil_image, size = (256,256)):
        img = np.asarray(pil_image.resize(size, Image.BICUBIC))
        if img.shape[0] < 5:  # image in CHW
            img = img.transpose((1, 2, 0))
        img = img[:, :, :3] if img.ndim == 3 else np.tile(img[:, :, None], 3)
        img = img if img.data.contiguous else np.ascontiguousarray(img)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        img_tensor = torch.from_numpy(img) / 255.
        
        return img_tensor
    
    return yolov5_transforms

def get_yolov5_model(model_uid, output = 'model_and_transforms'):
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    preprocess = get_yolov5_transforms()
    
    if output == 'transforms_only':
        return preprocess
    
    model = torch.hub.load('ultralytics/yolov5', model_name, autoshape = False)
    
    if output == 'model_only':
        return model
    
    return model, preprocess


### Detectron Models ---------------------------------------------------------------------------

def _get_detectron_transforms(cfg, model, input_type = 'PIL'):
    import detectron2.data.transforms as detectron_transform
    
    augment = detectron_transform.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, 
                                                      cfg.INPUT.MIN_SIZE_TEST], 
                                                      cfg.INPUT.MAX_SIZE_TEST)
    
    standard_image_size = (256,256)
    warning('deepjuice uses a standard_image_size of {} for Detectron2 models'
            .format(standard_image_size))
    
    if standard_image_size:
        augment = detectron_transform.Resize(standard_image_size)
    
    def detectron_transforms(original_image):
        if input_type == 'PIL':
            original_image = np.asarray(original_image)
        original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = augment.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}
        return model.preprocess_image([inputs]).tensor.squeeze()

    return detectron_transforms

def get_detectron_model(model_uid, output = 'model_and_transforms'):
    
    from detectron2.modeling import build_model
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    
    model_info = get_model_info(model_uid)
    weights_url = model_info['weights_url']
    
    cfg = model_zoo.get_config(weights_url)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_url)
    
    cfg_clone = cfg.clone()
    model = build_model(cfg_clone)
    model = model.eval()

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    preprocess = _get_detectron_transforms(cfg_clone, model)
    
    CACHED_TRANSFORMS[model_uid] = preprocess
    
    backbone_only = True
    if backbone_only:
        model = model.backbone
        
    if output == 'transforms_only':
        return preprocess
    
    if output == 'model_only':
        return model
    
    if output == 'model_and_transforms':
        return model, preprocess

    
### SLIP Models -------------------------------------------------------------------------------

def setup_slip_codebase():
    from git import Repo as gitrepo
    from gdown import download
    
    output_dir = '{}/slip_codebase_'.format(custom_code_dir)
    if os.path.exists(output_dir):   
        shutil.rmtree(output_dir)
    
    if os.path.exists(output_dir[:-1]):
        shutil.rmtree(output_dir[:-1])
    
    os.makedirs(output_dir[:-1])
    
    gitrepo.clone_from('https://github.com/facebookresearch/SLIP', output_dir)
    files_to_keep = ['losses.py','models.py','tokenizer.py','utils.py','LICENSE','README.md']
    for file in os.listdir(output_dir):
        if file in files_to_keep:
            src = os.path.join(output_dir, file)
            dst = os.path.join(output_dir[:-1], file)
            if not os.path.isdir(src):
                shutil.copy(src, dst)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            
    shutil.rmtree(output_dir)
    
get_slip_transforms = get_imagenet_transforms

def get_slip_model(model_uid, output = 'model_and_transforms'):
    sys.path.append('{}/slip_codebase'.format(custom_code_dir))
    import models
    from collections import OrderedDict
    from tokenizer import SimpleTokenizer
    
    model_info = get_model_info(model_uid)
    weights_url = model_info['weights_url']
    
    preprocess = get_slip_transforms()
    
    if output == 'transforms_only':
        return preprocess
    
    ckpt = torch.utils.model_zoo.load_url(weights_url, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    ssl_mlp_dim, ssl_emb_dim = ckpt['args'].ssl_mlp_dim, ckpt['args'].ssl_emb_dim
    model_call = getattr(models, ckpt['args'].model)
    model = model_call(rand_embed=False, ssl_mlp_dim = ssl_mlp_dim, ssl_emb_dim = ssl_emb_dim)
    model.load_state_dict(state_dict, strict=True)
    
    if output == 'model_only':
        return model.visual
    
    if output == 'model_and_transforms':
        return model.visual, preprocess
    

### IPCL Models -------------------------------------------------------------------------------

def get_ipcl_model(model_uid, output = 'model_and_transforms'):
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    model, preprocess = torch.hub.load("harvard-visionlab/open_ipcl", model_name)
    
    if output == 'transforms_only':
        return preprocess
    
    if output == 'model_only':
        return model
    
    return model, preprocess


### SEER Models -------------------------------------------------------------------------------

get_seer_transforms = get_imagenet_transforms

def get_seer_model(model_uid, output = 'model_and_transforms'):
    from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
    from vissl.models import build_model
    from classy_vision.generic.util import load_checkpoint
    from vissl.utils.checkpoint import init_model_from_consolidated_weights

    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    weights_url = model_info['weights_url']
    weights_name = weights_url.split('/')[-1]
    weights_path = os.path.join(torch.hub.get_dir(), 'checkpoints', weights_name)
    
    preprocess = get_seer_transforms()
    
    if output == 'transforms_only':
        return preprocess

    model_config = (model_name.replace('SEER-RegNet-','regnet')
                    .replace('GF','Gf')
                    .replace('-INFT',''))
    
    if '256Gf' in model_config:
        model_config += '_1'

    weights = torch.utils.model_zoo.load_url(weights_url, map_location='cpu')
    cfg = ['config=benchmark/linear_image_classification/imagenet1k/eval_resnet_8gpu_transfer_in1k_linear', 
           '+config/benchmark/linear_image_classification/imagenet1k/models={}'.format(model_config), 
           'config.MODEL.WEIGHTS_INIT.PARAMS_FILE={}'.format(weights_path)]

    cfg = compose_hydra_configuration(cfg)
    _, cfg = convert_to_attrdict(cfg)

    model = build_model(cfg.MODEL, cfg.OPTIMIZER)
    model = model.eval()

    weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

    init_model_from_consolidated_weights(config=cfg, model=model, 
                                         state_dict=weights, 
                                         skip_layers = [],
                                         state_dict_key_name="classy_state_dict")
    
    if output == 'model_only':
        return model
    
    return model, preprocess


### BiT Expert Models -------------------------------------------------------------------------

get_bit_expert_transforms = get_imagenet_transforms

def get_bit_expert_model(model_uid, output = 'model_and_transforms'):
    from bit_experts import ResNetV2
    import tensorflow_hub
    
    model_info = get_model_info(model_uid)
    expertise = model_uid.split('_')[-1]
    
    preprocess = get_bit_expert_transforms()
    
    if output == 'transforms_only':
        return preprocess
    
    model_url = 'https://tfhub.dev/google/experts/bit/r50x1/in21k/{}/1'.format(expertise)
    expert_weights = tensorflow_hub.KerasLayer(model_url).weights

    expert_weights_dict = {expert_weights[i].name.replace(':0',''): np.array(expert_weights[i]) 
                           for i in range(len(expert_weights))}

    model = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=1000, 
                     zero_head = True, remove_head = True).load_from(expert_weights_dict)
    
    if output == 'model_only':
        return model
    
    return model, preprocess

### LAVIS Models -------------------------------------------------------------------------

def setup_lavis_codebase():
    from git import Repo as gitrepo
    from gdown import download
    
    output_dir = '{}/lavis_codebase_'.format(custom_code_dir)
    if os.path.exists(output_dir):   
        shutil.rmtree(output_dir)
    
    if os.path.exists(output_dir[:-1]):
        shutil.rmtree(output_dir[:-1])
    
    os.makedirs(output_dir[:-1])
    
    gitrepo.clone_from('https://github.com/salesforce/LAVIS', output_dir)
    files_to_keep = ['lavis','LICENSE','README.md']
    for file in os.listdir(output_dir):
        if file in files_to_keep:
            src = os.path.join(output_dir, file)
            dst = os.path.join(output_dir[:-1], file)
            if not os.path.isdir(src):
                shutil.copy(src, dst)
            if os.path.isdir(src):
                shutil.copytree(src, dst)
            
    shutil.rmtree(output_dir)

def get_lavis_model(model_uid, output = 'model_and_transforms'):
    sys.path.append('{}/lavis_codebase'.format(custom_code_dir))
    from lavis.models import load_model_and_preprocess

    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    weights = model_info['weights']
    
    model, processors, _ = load_model_and_preprocess(name = model_name, 
                                                     model_type = weights, 
                                                     is_eval=True, device='cpu')
    
    preprocess = processors['eval']
    model = model.visual_encoder
    
    if output == 'model_only':
        return model
    
    if output == 'transforms_only':
        return preprocess
    
    if output == 'model_and_transforms':
        return model, preprocess