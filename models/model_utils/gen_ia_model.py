from subprocess import check_output
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, impute_lu, conv_cat_num, conv_cat_str
from model_utils.evaluator import evaluate

def gen_model(df, out_dir, model):
  if not os.path.exists(out_dir):
    os.mkdir(os.path.join(out_dir))

  cmd = [
    'node',
    'node_modules/preprocessing/preprocessing/cli/produce-aggregate-data-files.js', 
    '-m',
    model,
    '-o',
    out_dir,
    '-p',
    train_src,
    'node_modules/preprocessing/data/mouza-names.csv',
  ]
  stdout = check_output(cmd).decode(sys.stdout.encoding).replace('\n', '')
  print(stdout)

if __name__ == '__main__':
  train_src = './well_data/train.csv'
  out_dir = './models/model3/model/'
  model = 'model3'

  gen_model(train_src, out_dir, model)
