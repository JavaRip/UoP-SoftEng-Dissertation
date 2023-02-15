import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import run_ia_model
from model_utils.evaluator import gen_eval, print_eval 

def get_name():
  return 'm4'

def main(model='model4', stain='Red', test_src='./well_data/test.csv'):
  return run_ia_model(model, stain, test_src)
  
if __name__ == '__main__':
  model = 'model4'
  stain = 'Red'
  test_src ='./well_data/test.csv'
  outfile = f'./prediction_data/{model}-{stain}.csv'

  df = main(model, stain, test_src)

  df.to_csv(outfile, index=False)

  eval = gen_eval(df)
  print_eval(eval)

  print(f'written to {outfile}')