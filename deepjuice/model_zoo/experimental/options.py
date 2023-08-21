from model_options import *
    
_FILEPATH = os.path.dirname(os.path.abspath(__file__))

main_registry = pd.read_csv('/'.join(_FILEPATH.split('/')[:-1]) + '/model_registry.csv')
test_registry = pd.read_csv(_FILEPATH + '/latest_registry.csv')
model_registry = pd.concat([main_registry, test_registry], axis = 0)
model_lookup = model_registry.set_index('model_uid').to_dict(orient='index')

### Overwrite Standard DeepDive Functionality ------------------------------------------------

def get_model_info(model_uid):
    return model_lookup[model_uid]

def subset_registry(model_source):
    return model_registry[model_registry['model_source'] == model_source].copy()
    
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
    
def check_deepdive_uid(model_uid):
    model_options = get_model_options()
    
    if model_uid not in model_options.model_uid.to_list():
        raise ValueError('No reference available for this model_uid.')
    

def get_deepdive_model(model_uid, output = 'model_and_transforms'):
    check_deepdive_uid(model_uid)

    model_info = get_model_info(model_uid)
    model_source = model_info['model_source']
    
    return eval('get_{}_model'.format(model_source))(model_uid, output)

def get_deepdive_transforms(model_uid):
    check_deepdive_uid(model_uid)
    
    model_info = get_model_info(model_uid)
    model_source = model_info['model_source']
    
    transforms_call = eval('get_{}_model'.format(model_source))
    
    if transforms_call in globals():
        return eval(transforms_call)
    
    if transforms_call not in globals():
        return get_transforms_only(model_source, model_uid)
    
### Updated Model Calls ------------------------------------------------
    
def get_timm_model(model_uid, output = 'model_and_transforms'):
    from timm.data.transforms_factory import create_transform
    from torchvision.transforms import Resize, Compose
    from timm.data import resolve_data_config
    from timm import create_model, list_models
    
    model_info = get_model_info(model_uid)
    model_name = model_info['model_name']
    
    if not pd.isna(model_info['weights']):
        model_name = model_name + '.' + model_info['weights']
        
    print(f'Loading pytorch-image-models: {model_name}')
    
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