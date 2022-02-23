from options.eval import args

if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
# load model and its weight
import torch
md = torch.load(args.input)
nargs = md['param']
from model import get_model
model,config_args = get_model(nargs.model,nargs)
model.load_state_dict(md['weight'])
model = model.to(args.device)

# dataloader
from dataset import Seq2imgDataset
tedt = Seq2imgDataset('./split/test.txt')
tedl  = torch.utils.data.DataLoader(tedt, batch_size=args.num_workers, num_workers=args.num_workers)

# evaluation metric
from utils import get_loss
loss = get_loss(name=nargs.loss,task_type='reg')

# evaluate
from train import test
acc,val_loss = test(tedl,model,lossf=loss,exist_acc=False,device=args.device)


# Display&Save the result
test_perf = val_loss
print(f'{nargs.loss}: {test_perf}')
import os
fname = os.path.splitext(args.input)[0]
fname = os.path.basename(fname)
fpath = os.path.join("results", fname)

with open(f'{fpath}_evaluation.txt','w') as f:
    f.write(str(test_perf))
    print(f'the result is saved in {fpath}.txt ')