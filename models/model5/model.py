import pandas as pd
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.evaluator import evaluate
from model_utils.utils import gen_labels, gen_ia_predictions

def main():
  stain_color = 'Red'
  filepath = './well_data/test.csv'
  model = 'model5'

  df = pd.read_csv(filepath)
  df['Prediction'], outfile = gen_ia_predictions(filepath, stain_color, model)
  df['Label'] = gen_labels(df)

  df.to_csv(outfile, index=False)
  evaluate(df)
  print(f'written to {outfile}')

if __name__ == '__main__':
  main()