import argparse
import torch
exam_code = '''
e.g)  
'''
parser = argparse.ArgumentParser("Train datasets",epilog=exam_code)   


parser.add_argument('--input','-i',type=str,help='model path')
parser.add_argument('--num_workers'   ,default=4,type=int     ,help='')


# legacy
parser.add_argument('--device'   ,default=None,type=str     ,help='cpu | gpu')

parser.add_argument('-m'  ,'--model'   ,default='vit' ,metavar='{...}'    ,help='model name')


parser.add_argument('--epoch'   ,default=100,type=int     ,help='total number of epochs')
parser.add_argument('--batch_size'   ,default=4,type=int     ,help='batch size')


parser.add_argument('--lr'     ,default=0.001,type=float     ,help='Learning Rate')
parser.add_argument('--loss'   ,default='mse',type=str     ,help='which loss crossentropy(default)|multimargin|nll ?')

parser.add_argument('--filename'   ,default='',type=str     ,help='filepath to save the result')
parser.add_argument('-f'       ,help='for ipynb')
args = parser.parse_args()

# args.model = args.model.lower()
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    