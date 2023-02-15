from subprocess import check_output
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)

def gen_model(train_src, out_dir, model):
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
