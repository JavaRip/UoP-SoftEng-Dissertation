import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import geopandas as gpd


def cat_int_enc(df):
  dfc = df.copy()

  for header in list(dfc.columns.values):
    if dfc[header].dtype == 'object':
      dfc[header] = pd.Categorical(dfc[header]).codes

  return dfc

def add_centroids(gdf, df):
  gdf['lon'] = gdf.centroid.x
  gdf['lat'] = gdf.centroid.y

  dfm = df.merge(
    gdf,
    left_on=['Division', 'District', 'Upazila', 'Union', 'Mouza'],
    right_on=['div', 'dis', 'upa', 'uni', 'mou'],
    how='left',
  )

  return dfm.drop(
    columns=[
      'div',
      'Division',
      'dis',
      'District',
      'upa',
      'Upazila',
      'uni',
      'Union',
      'mou',
      'Mouza',
      'area',
      'geometry'
    ],
    axis='columns'
  )

def gen_predictions(train, test):
  train['Label'] = np.where(train['Arsenic'] > 10, 1, 0)
  test['Label'] = np.where(test['Arsenic'] > 10, 1, 0)

  # scale
  train['tid'] = 1
  test['tid'] = 0

  # lower precision for lat lon
  test.sort_values(by=['lat'], inplace=True)
  test['lat'] = pd.Categorical(test['lat']).codes
  test.sort_values(by=['lon'], inplace=True)
  test['lon'] = pd.Categorical(test['lon']).codes

  train.sort_values(by=['lat'], inplace=True)
  train['lat'] = pd.Categorical(train['lat']).codes
  print('----------')
  print(train['lat'].nunique())
  train.sort_values(by=['lon'], inplace=True)
  train['lon'] = pd.Categorical(train['lon']).codes
  print(train['lon'].nunique())

  scale_df = pd.concat([test, train])
  scale_df = pd.DataFrame(MinMaxScaler().fit_transform(scale_df), columns=scale_df.columns)

  train = scale_df[scale_df['tid'] == 1]
  test = scale_df[scale_df['tid'] == 0]

  train = pd.DataFrame(train.drop(columns=['tid']))
  test = pd.DataFrame(test.drop(columns=['tid']))

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  test_y = test['Label']

  clf = MLPClassifier(
    solver='adam',
    alpha=0.0001,
    hidden_layer_sizes=(2500, 625, 150, 38),
    learning_rate='adaptive',
    random_state=99,
    verbose=1,
  )

  clf.fit(train_X, train_y)

  return clf.predict(test_X), test, train

def load_data(train_src, test_src):
  return pd.read_csv(train_src), pd.read_csv(test_src)

if __name__ == '__main__':
  train_src = './well_data/train.csv'
  test_src ='./well_data/test.csv'
  test_out = f'./prediction_data/model9.csv';
  geo_src = './geodata/mou/mou-c005-s010-vw-pr.geojson'

  train_df, test_df = load_data(train_src, test_src)
  arsenic = test_df['Arsenic']

  gdf = gpd.read_file(geo_src)
  train_df = add_centroids(gdf, train_df)
  test_df = add_centroids(gdf, test_df)

  predictions, test_df, train_df = gen_predictions(train_df, test_df)

  test_df['predictions'] = predictions

  test_df['Label'].replace(1, 'polluted', inplace=True)
  test_df['Label'].replace(0, 'safe', inplace=True)
  test_df['predictions'].replace(1, 'polluted', inplace=True)
  test_df['predictions'].replace(0, 'safe', inplace=True)
  test_df['Arsenic'] = arsenic
  test_df.info()

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
