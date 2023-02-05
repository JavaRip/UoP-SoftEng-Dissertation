import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

def cat_int_enc(df):
  dfc = df.copy()

  for header in list(dfc.columns.values):
      if dfc[header].dtype == 'object':
          dfc[header] = pd.Categorical(dfc[header]).codes

  return dfc

def gen_predictions(train_df, test_df):
  train = cat_int_enc(train_df)
  test = cat_int_enc(test_df)

  train['l'].fillna((train['l'].mode()), inplace=True)
  train['u'].fillna((train['u'].mode()), inplace=True)
  test['l'].fillna((test['l'].mode()), inplace=True)
  test['l'].fillna((test['l'].mode()), inplace=True)

  train['Label'] = np.where(train['Arsenic'] > 10, 'polluted', 'safe')
  test['Label'] = np.where(test['Arsenic'] > 10, 'polluted', 'safe')

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  test_y = test['Label']

  rf_model = RandomForestClassifier(random_state=99)
  rf_model.fit(train_X, train_y)

  return rf_model.predict(test_X)

def load_data(train_src, test_src):
  return pd.read_csv(train_src), pd.read_csv(test_src)

if __name__ == '__main__':
  train_src = './models/model7/train.csv'
  test_src ='./models/model7/test.csv'
  test_out = f'./prediction_data/model7-{time.time() / 1000}.csv';

  train_df, test_df = load_data(train_src, test_src)

  predictions = gen_predictions(train_df, test_df)

  test_df['predictions'] = predictions

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
