import sys
import dill
import os
import pandas as pd
from subprocess import check_output
import multiprocess as mp 
import time

from models.model_utils.evaluator import gen_eval, print_eval
from utils.src_to_test_train import main as src_tt
from models.model3 import model as model3
from models.model4 import model as model4
from models.model5 import model as model5
from models.model6 import model as model6
from models.model7 import model as model7
from models.model8 import model as model8
from models.model9 import model as model9
from models.model10 import model as model10
from models.model11 import model as model11

def get_predictions(model, k_fold, pred_path):
  if os.path.exists(pred_path):
    print(f'{model.get_name()}-k{k_fold} prediction file found')
    return pd.read_csv(pred_path)
  else:
    return model.main(f'./well_data/k{k_fold}.csv', k_fold)

def get_eval(predictions, m_name, k_fold, eval_path):
  if os.path.exists(eval_path):
    print(f'{m_name}-k{k_fold} evaluation file found')
    return pd.read_csv(eval_path)
  else:
    eval = gen_eval(predictions)
    eval_df = pd.DataFrame(data={
      'model': m_name,
      'k': k_fold,
      'sensitivity': eval['sensitivity'],
      'f1': eval['f1'],
      'accuracy': eval['accuracy'],
      'precision': eval['precision'],
      'specificity': eval['specificity'],
    }, index=[0])

    return eval_df 

def build_model(m, k_fold):
  if not os.path.exists(f'./models/{m}/model/'):
    os.mkdir(f'./models/{m}/model/')

  # create k fold model
  if os.path.exists(f'./models/{m}/model/k{k_fold}'):
    print(f'{m} k{k_fold} model built')
  else:
    print(f'{m} k{k_fold} not built, building…')
    build_ia_model(m, k_fold)

def build_ia_model(m, k_fold):
  print(f'building {m} k{k_fold}')

  folds = [1, 2, 3, 4, 5]
  folds.remove(k_fold)

  os.mkdir(f'./models/{m}/model/k{k_fold}/')

  cmd_arr = [
    'node',
    'node_modules/preprocessing/preprocessing/cli/produce-aggregate-data-files.js',
    '-m',
    f'{m}',
    '-o',
    f'./models/{m}/model/k{k_fold}/',
    '-p',
    f'./well_data/k{folds[0]}.csv',
    f'./well_data/k{folds[1]}.csv',
    f'./well_data/k{folds[2]}.csv',
    f'./well_data/k{folds[3]}.csv',
    './node_modules/preprocessing/data/mouza-names.csv',
  ]

  return check_output(cmd_arr)

def run_model(model, k_fold):
  m_name = model.get_name()
  print(f'running {m_name} k{k_fold}')

  test_out=f'./prediction_data/{m_name}-k{k_fold}.csv'
  eval_out=f'./evaluation_data/{m_name}-k{k_fold}.csv'

  pred_df = get_predictions(model, k_fold, test_out)
  eval_df = get_eval(pred_df, m_name, k_fold, eval_out)

  pred_df.to_csv(test_out, index=False)
  eval_df.to_csv(eval_out, index=False)

  print(f'predictions written to {test_out}')
  print(f'evaluation written to {eval_out}')

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
  load_tt = False

  for x in [1, 2, 3, 4, 5]:
    if os.path.exists(f'./well_data/k{x}.csv'):
      print(f'k{x} split already exists')
    else:
      load_tt = True
      print(f'k{x} not found, generating k folds…')
      break

  if (load_tt):
    for f in os.listdir('./well_data/'):
      if f.endswith('csv'):
        os.remove(os.path.join('./well_data/', f))

    src_tt()

  return

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

  print('\n______create test train split______\n')
  print(gen_test_train())

  print('\n______building ia models______\n')
  ia_models = ['model3', 'model4', 'model5']
  bj = [] # BuildJobs

  for m in ia_models:
    for x in [1, 2, 3, 4, 5]:
      p = mp.Process(target=build_model, args=(m, x,))
      p.start()
      bj.append(p)
      time.sleep(0.05) # pause so logs come out in order
  
  for j in bj:
    j.join()

  print('\n______running models______\n')
  models = [model3, model4, model5, model6, model7, model8, model9, model10, model11]
  rj = [] # RunJobs
  for m in models:
    for x in [1, 2, 3, 4, 5]:
      p = mp.Process(target=run_model, args=(m, x,))
      p.start()
      rj.append(p)

  for j in rj:
    j.join()