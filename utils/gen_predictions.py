# todo allow model to be selected with cmdargs
from subprocess import check_output
import pandas as pd
import json
import sys

def predict(filepath, stain_color, model):
  cmd_arr = [
    'node',
    './utils/iarsenic-wrapper.js',
    filepath,
    stain_color,
    model,
  ]

  stdout = check_output(cmd_arr).decode(sys.stdout.encoding).replace('\n', '')
  return pd.read_csv(stdout), stdout

if __name__ == '__main__':
  stain_color = sys.argv[1]
  filepath = sys.argv[2]
  model = sys.argv[3]

  df = pd.read_csv(filepath)
  df['predictions'], outfile = predict(filepath, stain_color, model)
  df.to_csv(outfile, index=False)
  print(f'written to {outfile}')
