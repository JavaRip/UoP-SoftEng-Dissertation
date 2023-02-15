import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import run_ia_model
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm5'

def main(test_src='./well_data/test.csv', k_fold=1, stain='Red'):
  return run_ia_model(test_src, stain, 'model5', k_fold)
  
if __name__ == '__main__':
  test_src ='./well_data/k1.csv'
  outfile = f'./prediction_data/model5-Red.csv'

  df = main(test_src, 1, 'Red')

  df.to_csv(outfile, index=False)

  eval = gen_eval(df)
  print_eval(eval)

  print(f'written to {outfile}')