import sys
import os
import pandas as pd
from subprocess import check_output

from models.model_utils.evaluator import gen_eval, print_eval
from utils.src_to_test_train import main as src_tt
from models.model3 import model as model3
from models.model4 import model as model4
from models.model5 import model as model5
from models.model6 import model as model6
from models.model7 import model as model7
from models.model8 import model as model8
from models.model9 import model as model9

def get_predictions(model):
  m_name = model.get_name()
  print(m_name)
  pred_path = f'./prediction_data/{m_name}.csv'

  if os.path.exists(pred_path):
    print('prediction file found')
    return pd.read_csv(pred_path)
  else:
    return model.main()

def build_ia_model(m):
  cmd_arr = [
    'mkdir',
    f'./models/{m}/model',
  ]

  check_output(cmd_arr)

  cmd_arr = [
    'node',
    'node_modules/preprocessing/preprocessing/cli/produce-aggregate-data-files.js',
    '-m',
    f'{m}',
    '-o',
    f'./models/{m}/model',
    '-p',
    'well_data/train.csv',
    'node_modules/preprocessing/data/mouza-names.csv',
  ]

  return check_output(cmd_arr).decode(sys.stdout.encoding)

def run_model(model):
  m_name = model.get_name()
  print(f'running {m_name}')

  test_out=f'./prediction_data/{m_name}.csv'

  pred_df = get_predictions(model)
  eval = gen_eval(pred_df)
  print_eval(eval)

  pred_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')

def build_model(m):
  if os.path.exists(f'./models/{m}/model/'):
    print(f'{m} model built')
  else:
    print(f'{m} not built, building…')
    print(build_ia_model(m))

def extract_ia_data():
  if os.path.exists('./well_data/src_data.json'):
    return 'src data ready'
  else:
    print('preparing src data…')

    cmd_arr = [
      'npm',
      'run',
      'load-src-data',
    ]

    return check_output(cmd_arr).decode()

def gen_test_train():
  src_in = './well_data/src_data.json'
  src_out = './well_data/src_data.csv'
  train_out = './well_data/train.csv'
  test_out = './well_data/test.csv'

  src_tt(src_in, src_out, train_out, test_out)
  return 'test train split generated'

def unzip_geodata():
  if os.path.exists('./geodata/'):
    return 'geodata ready'
  else:
    print('unzipping geodata…')

    cmd_arr = [
      'npm',
      'run',
      'unzip-geodata',
    ]

    return check_output(cmd_arr).decode()

if __name__ == '__main__':
  print('\n______unzipping geodata______\n')
  print(unzip_geodata())

  print('\n______extracting data from iarsenic______\n')
  print(extract_ia_data())

  print('\n______extracting data from iarsenic______\n')
  print(extract_ia_data())

  print('\n______create test train split______\n')
  print(gen_test_train())

  print('\n______building ia models______\n')
  ia_models = ['model3', 'model4', 'model5']
  for m in ia_models:
    build_model(m)

  print('\n______running models______\n')
  models = [model3, model4, model5, model6, model7, model8, model9]
  for m in models:
    run_model(m)