import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
import time
from sklearn.preprocessing import OneHotEncoder


def cat_int_enc(df):
  dfc = df.copy()

  for header in list(dfc.columns.values):
    if dfc[header].dtype == 'object':
      dfc[header] = pd.Categorical(dfc[header]).codes

  return dfc

def gen_predictions(train_df, test_df):
  train = train_df
  test = test_df

  train['l'].fillna((train['l'].mode()), inplace=True)
  train['u'].fillna((train['u'].mode()), inplace=True)
  test['l'].fillna((test['l'].mode()), inplace=True)
  test['u'].fillna((test['u'].mode()), inplace=True)
  # TODO, make a project proposal to see if this code could literally be less dry

  train['Strata'] = np.where(train['Strata'] == 's15', 1, train['Strata'])
  train['Strata'] = np.where(train['Strata'] == 's45', 2, train['Strata'])
  train['Strata'] = np.where(train['Strata'] == 's65', 3, train['Strata'])
  train['Strata'] = np.where(train['Strata'] == 's90', 4, train['Strata'])
  train['Strata'] = np.where(train['Strata'] == 's150', 5, train['Strata'])
  train['Strata'] = np.where(train['Strata'] == 'sD', 6, train['Strata'])
  train['Strata'] = pd.to_numeric(train['Strata'])

  test['Strata'] = np.where(test['Strata'] == 's15', 1, test['Strata'])
  test['Strata'] = np.where(test['Strata'] == 's45', 2, test['Strata'])
  test['Strata'] = np.where(test['Strata'] == 's65', 3, test['Strata'])
  test['Strata'] = np.where(test['Strata'] == 's90', 4, test['Strata'])
  test['Strata'] = np.where(test['Strata'] == 's150', 5, test['Strata'])
  test['Strata'] = np.where(test['Strata'] == 'sD', 6, test['Strata'])
  test['Strata'] = pd.to_numeric(test['Strata'])

  train['Label'] = np.where(train['Arsenic'] > 10, 'polluted', 'safe')
  test['Label'] = np.where(test['Arsenic'] > 10, 'polluted', 'safe')

  train_X = train.drop(['Arsenic', 'Label'], axis='columns')
  train_y = train['Label']

  test_X = test.drop(['Arsenic', 'Label'], axis='columns')
  test_y = test['Label']

  enc = OneHotEncoder(handle_unknown='ignore')
  test_div_enc = pd.DataFrame(enc.fit_transform(test_X[['Division']]).toarray())
  test_div_enc.columns = enc.get_feature_names_out(['Division'])
  test_dis_enc = pd.DataFrame(enc.fit_transform(test_X[['District']]).toarray())
  test_dis_enc.columns = enc.get_feature_names_out(['District'])
  test_upa_enc = pd.DataFrame(enc.fit_transform(test_X[['Upazila']]).toarray())
  test_upa_enc.columns = enc.get_feature_names_out(['Upazila'])
  # test_uni_enc = pd.DataFrame(enc.fit_transform(test_X[['Union']]).toarray())
  # test_uni_enc.columns = enc.get_feature_names_out(['Union'])
  # test_mou_enc = pd.DataFrame(enc.fit_transform(test_X[['Mouza']]).toarray())
  # test_mou_enc.columns = enc.get_feature_names_out(['Mouza'])

  test_X = test_X.drop(
    columns=[
      'Division',
      'District',
      'Upazila',
      'Union',
      'Mouza',
    ]
  )


  test_X = test_X.join(test_div_enc)
  test_X = test_X.join(test_dis_enc)
  test_X = test_X.join(test_upa_enc)
  # test_X = test_X.join(test_uni_enc)
  # test_X = test_X.join(test_mou_enc)

  train_div_enc = pd.DataFrame(enc.fit_transform(train_X[['Division']]).toarray())
  train_div_enc.columns = enc.get_feature_names_out(['Division'])
  train_dis_enc = pd.DataFrame(enc.fit_transform(train_X[['District']]).toarray())
  train_dis_enc.columns = enc.get_feature_names_out(['District'])
  train_upa_enc = pd.DataFrame(enc.fit_transform(train_X[['Upazila']]).toarray())
  train_upa_enc.columns = enc.get_feature_names_out(['Upazila'])
  # train_uni_enc = pd.DataFrame(enc.fit_transform(train_X[['Union']]).toarray())
  # train_uni_enc.columns = enc.get_feature_names_out(['Union'])
  # train_mou_enc = pd.DataFrame(enc.fit_transform(train_X[['Mouza']]).toarray())
  # train_mou_enc.columns = enc.get_feature_names_out(['Mouza'])

  train_X = train_X.drop(
    columns=[
      'Division',
      'District',
      'Upazila',
      'Union',
      'Mouza',
    ]
  )

  train_X = train_X.join(train_div_enc)
  train_X = train_X.join(train_dis_enc)
  train_X = train_X.join(train_upa_enc)
  # train_X = train_X.join(train_uni_enc)
  # train_X = train_X.join(train_mou_enc)

  l_missing_cols = list(set(train_X.columns) - set(test_X.columns))
  r_missing_cols = list(set(test_X.columns) - set(train_X.columns))

  for col in l_missing_cols:
    test_X[col] = 0

  for col in r_missing_cols:
    train_X[col] = 0

  train_X = cat_int_enc(train_X)
  test_X = cat_int_enc(test_X)

  train_X = train_X.reindex(sorted(train_X.columns), axis=1)
  test_X = test_X.reindex(sorted(test_X.columns), axis=1)

  clf = MLPClassifier(
    solver='adam',
    alpha=0.0001,
    hidden_layer_sizes=(250),
    learning_rate='adaptive',
    random_state=99
  )

  clf.fit(train_X, train_y)

  return clf.predict(test_X)

def load_data(train_src, test_src):
  return pd.read_csv(train_src), pd.read_csv(test_src)

if __name__ == '__main__':
  train_src = './models/model8/train.csv'
  test_src ='./models/model8/test.csv'
  test_out = f'./prediction_data/model8.csv';

  train_df, test_df = load_data(train_src, test_src)

  predictions = gen_predictions(train_df, test_df)

  test_df['predictions'] = predictions

  test_df.to_csv(test_out, index=False)
  print(f'predictions written to {test_out}')
