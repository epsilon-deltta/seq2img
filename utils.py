from torch import nn
def get_loss(name:str='mae',task_type='reg'):
    loss = None
    name = name.lower()
    if name.endswith('loss'):
        name = name.replace('loss','')
        
    if task_type == 'cls':
        if name == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        elif name == 'multimargin':
            loss = nn.MultiMarginLoss()
        elif name == 'nll':
            loss = nn.NLLLoss()
        else:
            ValueError(f'there is No {name} loss!!!')
    
    elif task_type == 'reg':
        if name == 'mse':
            loss = nn.MSELoss()
        elif name == 'bce':
            loss = nn.BCELoss()
        elif name == 'bcewithlogits':
            loss = nn.BCEWithLogitsLoss()
        elif name == 'hingeembedding':
            loss = nn.HingeEmbeddingLoss()
        elif name == 'huber':
            loss = nn.HuberLoss()
        elif name == 'kldiv':
            loss = nn.KLDivLoss()
        elif name == 'l1':
            loss = nn.L1Loss()
        elif name == 'multilabelsoftmargin':
            loss = nn.MultiLabelSoftMarginLoss()
        elif name == 'poissonnll':
            loss = nn.PoissonNLLLoss()
        elif name == 'smoothl1':
            loss = nn.SmoothL1Loss()
        elif name == 'softmargin':
            loss = nn.SoftMarginLoss()
            
        else:
            ValueError(f'there is No {name} loss!!!')
    return loss




import argparse

# default < model config < specified option in cmd
def reconfig(config: argparse.Namespace,parser: argparse.ArgumentParser)-> argparse.Namespace: 
    
    default_args = get_default_args(parser)
    args         = parser.parse_args()
    
    default_args_dict = dict( default_args._get_kwargs() )
    config_dict       = dict( config._get_kwargs() )
    args_dict         = dict( args._get_kwargs() )
    nargs             = dict()
    # args < config_args < default_args
    # 
    for key,default in default_args_dict.items():
        config_value = config_dict[key]
        args_value   = args_dict[key]

        if (default == config_value) and (default == args_value): # change x
            n_value = default
        elif (default != config_value) and (default == args_value): # model config change 
            n_value = config_value
        else: # custom param change
            n_value = args_value
        nargs[key] = n_value
        
    nargs = argparse.Namespace(**nargs)
    return nargs
    
def get_default_args(parser):
    default = {}
    for key,value in dict(parser.parse_args()._get_kwargs() ).items():
        default[key] = parser.get_default(key)
    default = argparse.Namespace(**default)
    return default

