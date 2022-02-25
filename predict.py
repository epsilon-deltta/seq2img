# arg param
from options.predict import args

# load model and its weight
from model import get_model
import torch

md = torch.load(args.model)
nargs = md['param']
model,config_args = get_model(nargs.model,nargs)

model.load_state_dict(md['weight'])

model = model.to(args.device)

# DataLoader
from dataset import Seq2imgDataset
tedt  = Seq2imgDataset(args.data)
tedl  = torch.utils.data.DataLoader(tedt, batch_size=1, num_workers=1)

# predict
from tqdm import tqdm
def get_results(dl,model,device='cuda'):
    pre_values = []
    for x,y in tqdm(dl):
        x,y = x.to(device),y.to(device)
        pre = model(x)
        pre_values.append(float(pre ))
    return pre_values

results = get_results(tedl,model,args.device)

# save the result in the file
results = [str(x)+'\n' for x in results]
if args.filename == '':
    import os
    fname = os.path.splitext(args.model)[0]
    fname = os.path.basename(fname)
    fpath = os.path.join("results", fname)
    file_path = f'{fpath}_output.txt'
else:
    file_path = args.filename
    
with open(file_path,'w') as f:
    f.writelines(results)
    print(f'The result is saved in {file_path}')