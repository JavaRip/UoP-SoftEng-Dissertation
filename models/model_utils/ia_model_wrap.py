from subprocess import check_output
import pandas as pd
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.evaluator import gen_eval, print_eval 
from model_utils.utils import gen_labels

def gen_predictions(filepath, stain_color, model):
  cmd_arr = [
    'node',
    './models/model_utils/iarsenic-wrapper.js',
    filepath,
    stain_color,
    model,
  ]

  stdout = check_output(cmd_arr).decode(sys.stdout.encoding).replace('\n', '')
  df = pd.read_csv(stdout)

  df['Prediction'].replace('highlyPolluted', 'polluted', inplace=True)
  df['Prediction'].replace('We do not have enough data to make an estimate for your well', 'polluted', inplace=True)

  return df['Prediction']

if __name__ == '__main__':
  stain_color = 'Red'
  filepath = './well_data/test.csv'
  model = 'model5'

  df = pd.read_csv(filepath)
  df['Prediction'], outfile = gen_predictions(filepath, stain_color, model)
  df['Label'] = gen_labels(df)

  eval = gen_eval(df)
  print_eval(eval)

  print(f'written to {outfile}')
