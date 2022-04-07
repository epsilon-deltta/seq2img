import argparse

exam_code = '''
python split.py --origin_data ./data/data.txt --split_folder ktest
'''
parser = argparse.ArgumentParser("Split dataset",epilog=exam_code)   

parser.add_argument('--origin_data'   ,default='./data/data.txt',type=str     ,help='the path where the orignal data exists')
parser.add_argument('--split_folder'   ,default='split',type=str     ,help='where to save the train/val/test set')
args = parser.parse_args()



with open(args.origin_data,'r') as f:
    lines = f.readlines()
    head  = lines[0]
    lines = lines[1:]

# split ratio: train:val:test = 6:2:2 = 3:1:1 
from sklearn.model_selection import train_test_split
tr,val   = train_test_split(lines,train_size=.6)
val,test = train_test_split(lines,train_size=.5)

import os
if not os.path.exists(args.split_folder):
    os.mkdir(args.split_folder)
    
train_path = os.path.join(args.split_folder,'train.txt')
val_path   = os.path.join(args.split_folder,'val.txt'  )
test_path  = os.path.join(args.split_folder,'test.txt' )

with open(train_path,'w') as f:
    f.writelines(tr)
with open(val_path,'w')   as f:
    f.writelines(val)
with open(test_path,'w')  as f:
    f.writelines(test)
    
    
