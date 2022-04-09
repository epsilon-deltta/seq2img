import torch
from tqdm import tqdm
def train(dl,model,lossf,opt,device='cuda'):
    model.train()
    for x,y in tqdm(dl):
        x,y = x.to(device),y.to(device)
        opt.zero_grad()
        pre = model(x)
        loss = lossf(pre,y)
        loss.backward()
        opt.step()

@torch.no_grad()
def test(dl,model,lossf,epoch=None,exist_acc=True,device='cuda'):
    model.eval()
    size, acc , losses = len(dl.dataset) ,0,0
    with torch.no_grad():
        for x,y in tqdm(dl):
            x,y = x.to(device),y.to(device)
            pre = model(x)
            loss = lossf(pre,y)
            
            if exist_acc: 
                acc += (pre.argmax(1)==y).type(torch.float).sum().item()
            losses += loss.item()
    if exist_acc:
        accuracy = round(acc/size,4)
    else:
        accuracy = None
    val_loss = round(losses/size,6)
    print(f'[{epoch}] acc/loss: {accuracy}/{val_loss}' if exist_acc else f'[{epoch}] loss: {val_loss}')
    return accuracy,val_loss 

import copy
def run(trdl,valdl,model,loss,opt,epoch=100,patience = 5,exist_acc=False,device='cuda'):
    val_losses = {0:1}
    model = model.to(device)
    for i in range(epoch):
        train(trdl,model,loss,opt,device=device)
        acc,val_loss = test(valdl,model,loss,epoch=i,exist_acc=exist_acc,device=device)

        
        if min(val_losses.values() ) > val_loss:
            best_model = copy.deepcopy(model)
        val_losses[i] = val_loss
        if i == min(val_losses,key=val_losses.get)+patience:
            break
    return best_model,list(val_losses.values())

if __name__ == '__main__':

    from options.train import parser
    from model import get_model
    
    # args init
    args = parser.parse_args()
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model load
    import copy
    model, model_args = get_model(args.model,copy.deepcopy(args))
    
    
    # re-config (default arg < model-specific arg < user-specified arg)
    from utils import reconfig
    nargs = reconfig(model_args,parser)
    

    # training settings
    from utils import get_loss
    loss = get_loss(name=nargs.loss,task_type='reg')
    params = [p for p in model.parameters() if p.requires_grad]
    opt  = torch.optim.Adam(params,lr=nargs.lr)
    
    # dataset and loaders
    
    from dataset import Seq2imgDataset
    
    trdt  = Seq2imgDataset(nargs.train)#  './split/train.txt'
    valdt = Seq2imgDataset(nargs.val) # './split/val.txt'
    # tedt  = ProteinDataset('./data/split/test.csv' ,transform=transform)

    trdl  = torch.utils.data.DataLoader(trdt, batch_size=nargs.batch_size, num_workers=nargs.num_workers)
    valdl  = torch.utils.data.DataLoader(valdt, batch_size=nargs.batch_size, num_workers=nargs.num_workers)
    # tedl  = torch.utils.data.DataLoader(tedt, batch_size=batch_size, num_workers=4)


    # train/validate
    best_model,val_losses = run(trdl,valdl,model,loss,opt,epoch=nargs.epoch,device=args.device) #,exist_acc=config['exist_acc'])
    
    # Final Logging for Best Model
    import numpy as np
    best_val_epoch = np.argmin(np.array(val_losses))
    best_val = val_losses[best_val_epoch]
    print('Finished training!')
    print('Best validation score: {}'.format(best_val))


    # save the best model
    best_model = best_model.to('cpu')
    import os
    if not nargs.filename == '':
        file_path = nargs.filename
    else:
        model_dir = './models'
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        file_name = nargs.model + '_' + str(best_val_epoch) + '.pt'
        file_path = os.path.join(model_dir, file_name)

    torch.save({'weight':best_model.state_dict(),'param':nargs},file_path)
