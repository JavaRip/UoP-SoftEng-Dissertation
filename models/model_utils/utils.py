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
