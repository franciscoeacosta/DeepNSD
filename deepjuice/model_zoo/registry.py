from copy import copy
import re, json
import pandas as pd

from importlib.util import find_spec

stamped_models = {}

model_registry_dflist = []

column_order = ['model_uid','model','architecture','train_task','train_data',
                'architecture_class','task_cluster','modality',
                'display_name', 'description', 'model_source', 'model_source_url','weights_url']

# Torchvision Models ----------------------------------------------------------------------- 

if find_spec('torchvision'):
    print('Torchvision found!')
    
    torchvision_tables = pd.read_html('https://pytorch.org/vision/stable/models.html')

    torchvision_dictlist = []

    imagenet_weights = torchvision_tables[1].Weight.to_list()
    transformer_archs = ['vit','swin']
    exclusions = ['swag_linear', 'swag_e2e']

    train_data_text = {'imagenet1k': 'ImageNet1K'}

    for weight in imagenet_weights:

        weight = weight.replace('_Weights', '')

        dict_i = {'model': weight.replace('.','_').lower(), 
                  'pretrained': True,
                  'architecture': weight.split('.')[0].lower(),
                  'train_task': 'classification', 
                  'train_data': weight.split('.')[1].split('_')[0].lower(),
                  'display_name': weight.split('.')[0].replace('_','-')}

        dict_i['architecture_class'] = 'Convolutional'
        dict_i['modality'] = 'Vision'
        dict_i['task_cluster'] = 'Supervised'
        dict_i['input_resolution'] = 224

        if any([transformer in dict_i['architecture'] for
                transformer in transformer_archs]):
            dict_i['architecture_class'] = 'Transformer'

        dict_i['description'] = ('{} trained on image classification with the {} dataset'
                                 .format(dict_i['display_name'], 
                                         train_data_text[dict_i['train_data']]))

        dict_i['model_source'] = 'torchvision'
        dict_i['model_source_url'] = 'pytorch.org/docs/stable/torchvision/models.html'

        dict_i = {'model_uid': 'torch' + '_' + dict_i['model'], **dict_i}

        if not any([exclusion in dict_i['model_uid'] for
                    exclusion in exclusions]):
            torchvision_dictlist.append(dict_i)

    torchvision_dictlist_ = copy(torchvision_dictlist)
    for dict_i in torchvision_dictlist:
        dict_i = copy(dict_i)
        dict_i['model_uid'] = re.sub('imagenet1k_v[1,2]', 'random', dict_i['model_uid'])
        dict_i['model'] = re.sub('imagenet1k_v[1,2]', 'random', dict_i['model'])
        dict_i['pretrained'] = False
        dict_i['train_task'] = None
        dict_i['train_data'] = None
        dict_i['description'] = '{} randomly initialized'.format(dict_i['display_name'])
    torchvision_dictlist_.append(dict_i)
    
    torchvision_dictlist = torchvision_dictlist_

    torchvision_df = pd.DataFrame(torchvision_dictlist)
    
    model_registry_dflist.append(torchvision_df)
    
# Pytorch-Image-Models (Timm) --------------------------------------------------------------

if find_spec('timm'):
    import timm
    
    timm_models = timm.list_models(pretrained = True)

    def extract_architecture(model_name):
        train_data_tags = ['_miil', '_in21k','_in22k','_22k','_in21ft1k','_in22ft1k','_22kft1k']
        train_task_tags = ['adv_','ens_','ig_','_ap','_ns','_dino','_teacher']

        for tag in train_data_tags + train_task_tags:
            model_name = model_name.replace(tag, '')

        return model_name

    def rename_timm_architecture(model_name):
        return (model_name.replace('_','-').upper()
                .replace('NET','Net')
                .replace('NEXT','Next')
                .replace('NEST','Nest')
                .replace('CONV','Conv')
                .replace('RES', 'Res')
                .replace('REG', 'Reg')
                .replace('REP', 'Rep')
                .replace('IT', 'iT')
                .replace('viT', 'ViT')
                .replace('VIS', 'Vis')
                .replace('MAX','Max')
                .replace('MIX','Mix')
                .replace('MixER','Mixer')
                .replace('EDGE', 'Edge')
                .replace('WIDE', 'Wide')
                .replace('PICO', 'Pico')
                .replace('NANO', 'Nano')
                .replace('TINY', 'Tiny')
                .replace('LiTE', 'Lite')
                .replace('-TI-', '-Ti-')
                .replace('MINI', 'Mini')
                .replace('SMALL', 'Small')
                .replace('BASE', 'Base')
                .replace('MEDIUM', 'Medium')
                .replace('LARGE', 'Large')
                .replace('HUGE', 'Huge')
                .replace('BIG','Big')
                .replace('BOT','Bot')
                .replace('POOL','Pool')
                .replace('HALO','Halo')
                .replace('DARK','Dark')
                .replace('DEiT','DeiT')
                .replace('CAiT','CaiT')
                .replace('COAT','CoaT')
                .replace('CROSS', 'Cross')
                .replace('DENSE', 'Dense')
                .replace('LEViT', 'LeViT')
                .replace('TWINS','Twins')
                .replace('SWIN','Swin')
                .replace('PATCH', 'P')
                .replace('WINDOW', 'W')
                .replace('2TO2', '2to2')
                .replace('NetAL', 'NetA-L')
                .replace('PRUNED', 'Pruned')
                .replace('GLUON','Gluon')
                .replace('FOCUS','Focus')
                .replace('LEGACY','Legacy')
                .replace('FORMER','Former')
                .replace('LAMBDA', 'Lambda')
                .replace('MOBILE', 'Mobile')
                .replace('RELPOS', 'RelPos')
                .replace('MINIMAL', 'Minimal')
                .replace('HARDCORE', 'HardCoRe')
                .replace('XCEPTION', 'Xception')
                .replace('SELECSLS','SelecSLS')
                .replace('ADV-','Adversarial-')
                .replace('MOBILeViT', 'MobileViT')
                .replace('SEQUENCER','Sequencer')
                .replace('INCEPTION','Inception')
                .replace('EFFICIENT','Efficient')
                .replace('DISTILLED', 'Distilled')
                .replace('DIST','Distilled'))

    url = 'https://raw.githubusercontent.com/rwightman/pytorch-image-models/main/results/model_metadata-in1k.csv'
    pretrain_data = pd.read_csv(url)
    pretrain_dict = {model: data['pretrain'] for model, data in 
                     pretrain_data.set_index('model').to_dict(orient='index').items()}

    def register_timm_model(model_name):

        model_entry = {'model': model_name}

        def add_to_list(original, addition):
            if not isinstance(original, list):
                original = [original]

            return original + [addition]

        architecture = extract_architecture(model_name)
        architecture_text = rename_timm_architecture(architecture)

        model_entry['architecture'] = extract_architecture(model_name)

        train_task = 'classification'
        train_text = 'image classification'

        pretrain_text = []
        pretraining = []
        if model_name in pretrain_dict:
            pretraining = pretrain_dict[model_name]

        if '_dist' in model_name or 'dist' in pretraining:
            train_task = add_to_list(train_task, 'distillation')
            pretrain_text.append('knowledge distillation')

        if 'ssl_' in model_name or pretraining in ['ig1b_ssl','yfc-semisl']:
            train_task = add_to_list(train_task, 'ssl_pretraining')
            pretrain_text.append('semi-supervised pretraining')

        if 'swsl' in model_name or pretraining == 'ig1b_swsl':
            train_task = add_to_list(train_task, 'swsl_pretraining')
            pretrain_text.append('semi-weakly-supervised pretraining')

        if '_ap' in model_name or pretraining in ['in1k-ap','in1k-adv']:
            train_task = add_to_list(train_task, 'adversarial_pretraining')
            pretrain_text.append('adversarial pretraining')

        if '_ns' in model_name or pretraining == 'jft300m-ns' :
            train_task = add_to_list(train_task, 'noisy_student')
            pretrain_text.append('noisy student pretraining')

        if '_sam' in model_name:
            train_task = add_to_list(train_task, 'sam_pretraining')
            pretrain_text.append('sharpness-aware pretraining')

        if 'beit_' in model_name:
            train_task = add_to_list(train_task,'masked_pretraining')
            pretrain_text.append('masked-input (self-supervised) pretraining')

        if '_bitm' in model_name:
            train_task = add_to_list(train_task, 'big_transfer')
            pretrain_text.append('big transfer pretraining')

        if '_dino' in model_name:
            train_task = add_to_list(train_task, 'dino_pretraining')
            pretrain_text.append('DINO-style self-supervised pretraining')

        if len(pretrain_text) >= 3:
            pretrain_text = ' with ' + ', '.join(pretrain_text[:-1]) + 'and ' + pretrain_text[-1]
        if len(pretrain_text) == 2:
            pretrain_text = ' and '.join(pretrain_text)
        if len(pretrain_text) == 1:
            pretrain_text = ' with ' + pretrain_text[0]
        if len(pretrain_text) == 0:
            pretrain_text = ''

        if not isinstance(train_task, list):
            model_entry['train_task'] = train_task
        if isinstance(train_task, list):
            model_entry['train_task'] = ','.join(train_task)

        train_data = 'imagenet1k'
        data_text = ['ImageNet1K']

        imagenet21k_tags = ['_bitm','_miil','in21k','in22k','_22k','_in21ft1k','_in22ft1k','_22kft1k']
        if any([tag in model_name or tag in pretraining for tag in imagenet21k_tags]):
            train_data = add_to_list(train_data, 'imagenet21k')
            data_text.append('ImageNet21K')

        instagram_tags = ['ig_','ssl','swsl']
        if any([tag in model_name for tag in instagram_tags]):
            train_data = add_to_list(train_data, 'instagram1B')
            data_text.append('Instagram1B')

        if pretraining == 'jft300m-ns':
            train_data = add_to_list(train_data, 'jft300m')
            data_text.append('JFT300M')

        if pretraining == 'yfc-semisl':
            train_data = add_to_list(train_data, 'yfc100m')
            data_text.append('YFC100M')

        if len(data_text) >= 3:
            data_text = ', '.join(data_text[:-1]) + 'and ' + data_text[-1] + ' datasets'
        if len(data_text) == 2:
            data_text = ' and '.join(data_text) + ' datasets'
        if len(data_text) == 1:
            data_text = data_text[0] + ' dataset'

        if not isinstance(train_data, list):
            model_entry['train_data'] = train_data
        if isinstance(train_data, list):
            model_entry['train_data'] = ','.join(train_data)

        description = f'{architecture_text} trained on {train_text}{pretrain_text} using the {data_text}.'
        model_entry['description'] = description
        model_entry['display_name'] = architecture_text

        filters = {'transformer': ['vit','deit','xcit','jx_nest','coat','cait','volo','twins','tnt'],
                   'mixer': ['resmlp','mixer'],
                   'lstm': ['sequencer'],
                   'hybrid': ['convit','convmixer','levit']}

        architecture_type = 'convolutional'

        transformer_tags = ['vit','beit','deit','xcit','jx_nest','coat','cait','volo','twins','tnt']
        if any([tag in model_name for tag in transformer_tags]):
            architecture_type = 'transformer'

        mixer_tags = ['resmlp','mixer']
        if any([tag in model_name for tag in mixer_tags]):
            architecture_type = 'mixer'

        lstm_tags = ['sequencer']
        if any([tag in model_name for tag in lstm_tags]):
            architecture_type = 'mixer'

        hybrid_tags = ['convit','convmixer','levit']
        if any([tag in model_name for tag in hybrid_tags]):
            architecture_type = 'hybrid'

        model_entry['architecture_type'] = architecture_type

        input_resolution = 224

        resolutions = [50, 75, 100, 125, 150, 175, 200,
                       224, 240, 256, 384, 408, 448, 512]

        resolution_tags = ['_' + str(tag).zfill(3) for tag in resolutions]
        for tag in resolution_tags:
            if tag in model_name:
                input_resolution = int(tag.replace('_',''))

        model_entry['input_resolution'] = input_resolution

        return model_entry
    
# Taskonomy Models -------------------------------------------------------------------------

# Model Concatenation ----------------------------------------------------------------------
    
model_registry = pd.concat(model_registry_dflist)
    
if __name__ == "__main__":
    
    model_registry.to_csv('model_registry.csv', index = None)
    
    