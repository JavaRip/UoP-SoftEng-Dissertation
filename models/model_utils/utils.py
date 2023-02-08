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

  return test.copy(), train.copy()

def conv_cat_num(df, col_name):
  df[col_name].replace('polluted', 1, inplace=True)
  df[col_name].replace('safe', 0, inplace=True)

def conv_cat_str(df, col_name):
  df[col_name].replace(1, 'polluted', inplace=True)
  df[col_name].replace(0, 'safe', inplace=True)

def impute_lu(df):
  df['l'].fillna((df['l'].mode()), inplace=True)
  df['u'].fillna((df['u'].mode()), inplace=True)

def stratify(df):
  df.loc[df['Depth'].between(0, 15.3, 'both'), 'Strata'] = 's15'
  df.loc[df['Depth'].between(15.3, 45, 'right'), 'Strata'] = 's45'
  df.loc[df['Depth'].between(45, 65, 'right'), 'Strata'] = 's65'
  df.loc[df['Depth'].between(65, 90, 'right'), 'Strata'] = 's90'
  df.loc[df['Depth'].between(90, 150, 'right'), 'Strata'] = 's150'
  df.loc[df['Depth'].gt(150), 'Strata'] = 'sD'

def enumerate_stratas(df):
  stratas = ['s15', 's45', 's65', 's90', 's150', 'sD']
  for x in range(len(stratas)):
    df['Strata'] = np.where(df['Strata'] == stratas[x], x, df['Strata'])

  pd.to_numeric(df['Strata'])

def get_test_mlu(train, test, level):
  drop_cols = [
    'Division',
    'District',
    'Upazila',
    'Union',
    'Mouza',
    'Depth',
    'Arsenic',
    'Label',
  ]

  drop_cols.remove(level)

  # get df containing just mlu & region name
  mlu_df = train.dropna().drop(columns=drop_cols).drop_duplicates(subset=level)

  # create df of rows containing null in test
  testna = test[test.isna().any(axis=1)].drop(columns=['m','l','u'])

  # remove rows containing na from test
  test = test.dropna()

  # get mlu values into na rows in testna
  testna = testna.merge(
    mlu_df,
    on=[level],
    how='left',
  )

  # join test rows that contained na and test
  return pd.concat([testna, test])
