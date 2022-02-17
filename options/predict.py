import argparse
import torch
exam_code = '''
e.g)  

'''
parser = argparse.ArgumentParser("Train datasets",epilog=exam_code)   


parser.add_argument('--model','-m',type=str,help='model path')
parser.add_argument('--data','-d',type=str,help='data path')


# legacy
parser.add_argument('--device'   ,default=None,type=str     ,help='cpu | gpu e.g.,) cuda:0, cuda:1')
parser.add_argument('--num_workers'   ,default=4,type=int     ,help='')

parser.add_argument('--filename'   ,default='',type=str     ,help='filepath to save the output')
parser.add_argument('-f'       ,help='for ipynb')
args = parser.parse_args()

# args.model = args.model.lower()
if args.device is None:
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    