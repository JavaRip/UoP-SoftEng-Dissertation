import pandas as pd
import numpy as np
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import time
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import cat_int_enc, gen_labels, gen_centroids

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

# TODO scale_lat_lon is a bad name because it scales every column
def scale_lat_lon(df): 
  df.sort_values(by=['lat'], inplace=True)
  df['lat'] = pd.Categorical(df['lat']).codes

  df.sort_values(by=['lon'], inplace=True)
  df['lon'] = pd.Categorical(df['lon']).codes
  return pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)

def conv_label_num(df):
  df['Label'].replace('polluted', 1, inplace=True)
  df['Label'].replace('safe', 0, inplace=True)

def gen_predictions(train_df, test_df, gdf):
  train = train_df.copy()
  test = test_df.copy()

  tt_df = append_test_train(test, train)

  tt_df.drop(
    columns=['Division', 'District', 'Upazila', 'Union', 'Mouza'], 
    inplace=True,
  )

  tt_df['lon'], tt_df['lat'] = gen_centroids(train, gdf)
  tt_df['Label'] = gen_labels(tt_df)
  tt_df['Label'] = conv_label_num(tt_df)
  tt_df = scale_lat_lon(tt_df)
  tt_df.info()

  test, train = split_test_train(tt_df)

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  test_y = test['Label']

  clf = MLPClassifier(
    solver='adam',
    alpha=0.0001,
    hidden_layer_sizes=(100),
    learning_rate='adaptive',
    random_state=99,
    verbose=1,
  )

  clf.fit(train_X, train_y)

  test_X['predictions'] = clf.predict(test_X)
  test_X['predictions'].replace(1, 'polluted', inplace=True)
  test_X['predictions'].replace(0, 'safe', inplace=True)

  return test_X['predictions']

if __name__ == '__main__':
  train_src = './well_data/train.csv'
  test_src ='./well_data/test.csv'
  test_out = f'./prediction_data/model9.csv';
  geo_src = './geodata/mou/mou-c005-s010-vw-pr.geojson'

  gdf = gpd.read_file(geo_src)
  train_df = pd.read_csv(train_src)
  test_df = pd.read_csv(test_src)

  test_df['predictions'] = gen_predictions(train_df, test_df, gdf)

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
