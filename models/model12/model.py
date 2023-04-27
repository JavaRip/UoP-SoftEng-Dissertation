import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import time
import sys
import os
import inspect
import multiprocess as mp 
import dill

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentparentdir) 

from model_utils.utils import cat_int_enc, gen_labels, conv_cat_num, conv_cat_str, load_k_train
from model_utils.evaluator import gen_eval, print_eval 

from model3 import model as model3
from model4 import model as model4
from model5 import model as model5
from model6 import model as model6
from model7 import model as model7
from model8 import model as model8
from model9 import model as model9
from model10 import model as model10
from model11 import model as model11

def get_name():
  return 'm12'

def gen_pred_dict():
  print('generating prediction dictionary')
  return {
    'k1': {
      'm3': pd.read_csv(f'./prediction_data/m3-k1.csv'),
      'm4': pd.read_csv(f'./prediction_data/m4-k1.csv'),
      'm5': pd.read_csv(f'./prediction_data/m5-k1.csv'),
      'm6': pd.read_csv(f'./prediction_data/m6-k1.csv'),
      'm7': pd.read_csv(f'./prediction_data/m7-k1.csv'),
      'm8': pd.read_csv(f'./prediction_data/m8-k1.csv'),
      'm9': pd.read_csv(f'./prediction_data/m9-k1.csv'),
      'm10': pd.read_csv(f'./prediction_data/m10-k1.csv'),
      'm11': pd.read_csv(f'./prediction_data/m11-k1.csv'),
    },
    'k2': {
      'm3': pd.read_csv(f'./prediction_data/m3-k2.csv'),
      'm4': pd.read_csv(f'./prediction_data/m4-k2.csv'),
      'm5': pd.read_csv(f'./prediction_data/m5-k2.csv'),
      'm6': pd.read_csv(f'./prediction_data/m6-k2.csv'),
      'm7': pd.read_csv(f'./prediction_data/m7-k2.csv'),
      'm8': pd.read_csv(f'./prediction_data/m8-k2.csv'),
      'm9': pd.read_csv(f'./prediction_data/m9-k2.csv'),
      'm10': pd.read_csv(f'./prediction_data/m10-k2.csv'),
      'm11': pd.read_csv(f'./prediction_data/m11-k2.csv'),
    },
    'k3': {
      'm3': pd.read_csv(f'./prediction_data/m3-k3.csv'),
      'm4': pd.read_csv(f'./prediction_data/m4-k3.csv'),
      'm5': pd.read_csv(f'./prediction_data/m5-k3.csv'),
      'm6': pd.read_csv(f'./prediction_data/m6-k3.csv'),
      'm7': pd.read_csv(f'./prediction_data/m7-k3.csv'),
      'm8': pd.read_csv(f'./prediction_data/m8-k3.csv'),
      'm9': pd.read_csv(f'./prediction_data/m9-k3.csv'),
      'm10': pd.read_csv(f'./prediction_data/m10-k3.csv'),
      'm11': pd.read_csv(f'./prediction_data/m11-k3.csv'),
    }, 
    'k4': {
      'm3': pd.read_csv(f'./prediction_data/m3-k4.csv'),
      'm4': pd.read_csv(f'./prediction_data/m4-k4.csv'),
      'm5': pd.read_csv(f'./prediction_data/m5-k4.csv'),
      'm6': pd.read_csv(f'./prediction_data/m6-k4.csv'),
      'm7': pd.read_csv(f'./prediction_data/m7-k4.csv'),
      'm8': pd.read_csv(f'./prediction_data/m8-k4.csv'),
      'm9': pd.read_csv(f'./prediction_data/m9-k4.csv'),
      'm10': pd.read_csv(f'./prediction_data/m10-k4.csv'),
      'm11': pd.read_csv(f'./prediction_data/m11-k4.csv'),
    }, 
    'k5': {
      'm3': pd.read_csv(f'./prediction_data/m3-k5.csv'),
      'm4': pd.read_csv(f'./prediction_data/m4-k5.csv'),
      'm5': pd.read_csv(f'./prediction_data/m5-k5.csv'),
      'm6': pd.read_csv(f'./prediction_data/m6-k5.csv'),
      'm7': pd.read_csv(f'./prediction_data/m7-k5.csv'),
      'm8': pd.read_csv(f'./prediction_data/m8-k5.csv'),
      'm9': pd.read_csv(f'./prediction_data/m9-k5.csv'),
      'm10': pd.read_csv(f'./prediction_data/m10-k5.csv'),
      'm11': pd.read_csv(f'./prediction_data/m11-k5.csv'),
    }
  }

def get_perf(predictions):
  eval = gen_eval(predictions)
  return eval['accuracy']

def gen_bum_file(bum_file):
  # generate df of best model for each upa for each kfold
  well_df = pd.read_csv('./well_data/src_data.csv')
  res_df = pd.DataFrame(columns=['upa', 'k1', 'k2', 'k3', 'k4', 'k5', 'best'])
  
  upa_count = 0
  pred_dict = gen_pred_dict()
  for upa in well_df['Upazila'].unique():
    upa_count += 1
    print(f'{upa_count} / 445')

    bms = [] # best models

    for k in ['k1', 'k2', 'k3', 'k4', 'k5']:
      m3_df = pred_dict[k]['m3']
      m4_df = pred_dict[k]['m4']
      m5_df = pred_dict[k]['m5']
      m6_df = pred_dict[k]['m6']
      m7_df = pred_dict[k]['m7']
      m8_df = pred_dict[k]['m8']
      m9_df = pred_dict[k]['m9']
      m10_df = pred_dict[k]['m10']
      m11_df = pred_dict[k]['m11']

      evals = {
        'm3': get_perf(m3_df[m3_df['Upazila'] == upa]),
        'm4': get_perf(m4_df[m4_df['Upazila'] == upa]),
        'm5': get_perf(m5_df[m5_df['Upazila'] == upa]),
        'm6': get_perf(m6_df[m6_df['Upazila'] == upa]),
        'm7': get_perf(m7_df[m7_df['Upazila'] == upa]),
        'm8': get_perf(m8_df[m8_df['Upazila'] == upa]),
        'm9': get_perf(m9_df[m9_df['Upazila'] == upa]),
        'm10': get_perf(m10_df[m10_df['Upazila'] == upa]),
        'm11': get_perf(m11_df[m11_df['Upazila'] == upa]),
      } 

      best_model = ''
      best_score = 0
      for key in evals:
        if evals[key] >= best_score:
          best_score = evals[key]
          best_model = key

      bms.append(best_model)

    row = [upa, bms[0], bms[1], bms[2], bms[3], bms[4]]
    # mx_v / model x votes
    votes = {
      'm3': 0,
      'm4': 0,
      'm5': 0,
      'm6': 0,
      'm7': 0,
      'm8': 0,
      'm9': 0,
      'm10': 0,
      'm11': 0,
    }

    for m in bms:
      if m == 'm3':
        votes['m3'] += 1
      if m == 'm4':
        votes['m4'] += 1
      if m == 'm5':
        votes['m5'] += 1
      if m == 'm6':
        votes['m6'] += 1
      if m == 'm7':
        votes['m7'] += 1
      if m == 'm8':
        votes['m8'] += 1
      if m == 'm9':
        votes['m9'] += 1
      if m == 'm10':
        votes['m10'] += 1
      if m == 'm11':
        votes['m11'] += 1

      voted_model = '' 
      most_votes = 0
      for key in votes:
        if votes[key] >= most_votes:
          voted_model = key 
          most_votes = votes[key] 

    row.append(voted_model)
    res_df.loc[len(res_df)] = row
    
    print(row)
    
  res_df.to_csv(bum_file, index=False)
    

def gen_predictions(train_df=None, test_df=None):
  # best upazila model file
  bum_file = './models/model12/best_model_by_upa.csv'
  if not os.path.exists(bum_file):
    print('generating best upazila model file')
    gen_bum_file(bum_file)
  
  print('best upazila model file generated')
  bum_df = pd.read_csv(bum_file)
  well_df = pd.read_csv('./well_data/k1.csv')
  df = pd.merge(well_df, bum_df, left_on='Upazila', right_on='upa')
  df.drop(['k1', 'k2', 'k3', 'k4', 'k5', 'upa'], axis=1, inplace=True)

  model_dict = [ 
    {'name': 'm3', 'model': model3 },
    {'name': 'm4', 'model': model4 },
    {'name': 'm5', 'model': model5 },
    {'name': 'm6', 'model': model6 },
    {'name': 'm7', 'model': model7 },
    {'name': 'm8', 'model': model8 },
    {'name': 'm9', 'model': model9 },
    {'name': 'm10', 'model': model10 },
    {'name': 'm11', 'model': model11 },
  ]

  for m in model_dict:
    # model dataframe
    m_df = df[df['best'] == m['name']]

    print(m['name'])
    print(len(m_df))
    print('________________________________')

    # buffer dataframe
    b_df = m_df.copy()

    b_df.drop(['Prediction'], axis=1, inplace=True, errors='ignore')
    b_df.drop(['best'], axis=1, inplace=True)

    b_df.to_csv('./models/model12/buffer.csv', index=False)

    predictions = m['model'].main('./models/model12/buffer.csv', 1)
    df.loc[df['best'] == m['name'], ['Prediction']] = predictions['Prediction']
    df.info()
    print(df.head())
    print('--------------------------------')

  df['Labels'] = gen_labels(df)
  df.to_csv('./models/model12/predictions.csv', index=False)
  print_eval(gen_eval(df))


  


def main(
  test_src='./well_data/k1.csv',
  k_fold=1,
):

  # train_df = load_k_train(k_fold)
  # test_df = pd.read_csv(test_src) 

  # train_df['Label'] = gen_labels(train_df)
  # test_df['Label'] = gen_labels(test_df)

  # test_df['Prediction'] = gen_predictions(train_df, test_df)

  # return test_df
  gen_predictions()

if __name__ == '__main__':
  # test_out=f'./prediction_data/model6-{time.time() / 1000}.csv',
  test_df = main()

  # eval = gen_eval(test_df)
  # print_eval(eval)

  # print(f'written to {test_out}')