import pandas as pd
import numpy as np

def cat_int_enc(df):
  for header in list(df.columns.values):
    if df[header].dtype == 'object':
      df[header] = pd.Categorical(df[header]).codes

def gen_labels(df):
  return np.where(df['Arsenic'] > 10, 'polluted', 'safe')

def impute_lower_and_median(df):
  df['l'].fillna((df['l'].mode()), inplace=True)
  df['u'].fillna((df['u'].mode()), inplace=True)

def gen_centroids(df, gdf):
  gdf['lon'] = gdf.centroid.x
  gdf['lat'] = gdf.centroid.y

  dfm = df.merge(
    gdf,
    left_on=['Division', 'District', 'Upazila', 'Union', 'Mouza'],
    right_on=['div', 'dis', 'upa', 'uni', 'mou'],
    how='left',
  )

  return dfm['lon'], dfm['lat']

def append_test_train(test, train):
  test['tid'] = 1
  train['tid'] = 0
  return pd.concat([test, train])

def split_test_train(df):
  test = df[df['tid'] == 1]
  train = df[df['tid'] == 0]

  train = pd.DataFrame(train.drop(columns=['tid']))
  test = pd.DataFrame(test.drop(columns=['tid']))

  return test, train

def conv_cat_num(df, col_name):
  df[col_name].replace('polluted', 1, inplace=True)
  df[col_name].replace('safe', 0, inplace=True)

def conv_cat_str(df, col_name):
  df[col_name].replace(1, 'polluted', inplace=True)
  df[col_name].replace(0, 'safe', inplace=True)