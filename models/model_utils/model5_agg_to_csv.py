import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from os import listdir

# TODO update src_out
def main(
  train_src = './models/model5/aggregate-data/',
  test_src = './models/model5-trained-on-test-data/aggregate-data/',
  train_out = './models/model7/train.csv',
  test_out = './models/model7/test.csv',
  train_label_src = './well_data/train.csv',
  test_label_src = './well_data/test.csv'
):
  train_df = agg_data_to_df(train_src)
  label_train_df = label_agg_data(train_df, train_label_src)

  test_df = agg_data_to_df(test_src)
  label_test_df = label_agg_data(test_df, test_label_src)

  label_train_df.to_csv(train_out, index=False)
  label_test_df.to_csv(test_out, index=False)

def label_agg_data(agg_df, label_src):
  label_df = pd.read_csv(label_src)

  label_df.loc[label_df['Depth'].between(0, 15.3, 'both'), 'Strata'] = 's15'
  label_df.loc[label_df['Depth'].between(15.3, 45, 'right'), 'Strata'] = 's45'
  label_df.loc[label_df['Depth'].between(45, 65, 'right'), 'Strata'] = 's65'
  label_df.loc[label_df['Depth'].between(65, 90, 'right'), 'Strata'] = 's90'
  label_df.loc[label_df['Depth'].between(90, 150, 'right'), 'Strata'] = 's150'
  label_df.loc[label_df['Depth'].gt(150), 'Strata'] = 'sD'

  return pd.merge(
    label_df,
    agg_df,
    how='left',
    on=['Division', 'District', 'Upazila', 'Union', 'Mouza', 'Strata']
  )

def agg_data_to_df(agg_src):
  files = listdir(agg_src)

  csv_arr = []
  for filename in files:
    if not filename.endswith('.json'):
      continue

    div = filename.split('-')[0]
    dis = filename.split('-')[1].split('.')[0]

    file = open(f'{agg_src}{filename}', 'r')
    raw = file.read()
    data = json.loads(raw)

    for div in data:
      div_dict = data[div]
      for dis in div_dict['districts']:
         dis_dict = div_dict['districts'][dis]
         for upa in dis_dict['upazilas']:
           upa_dict = dis_dict['upazilas'][upa]
           for uni in upa_dict['unions']:
             uni_dict = upa_dict['unions'][uni]
             for mou in uni_dict['mouzas']:
               mou_dict = uni_dict['mouzas'][mou]
               for strata in mou_dict.keys():
                 if not 'm' in mou_dict[strata]:
                   mou_dict[strata]['m'] = ''
                 if not 'l' in mou_dict[strata]:
                   mou_dict[strata]['l'] = ''
                 if not 'u' in mou_dict[strata]:
                   mou_dict[strata]['u'] = ''

                 csv_arr.append([
                    div,
                    dis,
                    upa,
                    uni,
                    mou,
                    strata,
                    mou_dict[strata]['m'],
                    mou_dict[strata]['l'],
                    mou_dict[strata]['u'],
                 ])

    df = pd.DataFrame(
      csv_arr,
      columns=[
        'Division',
        'District',
        'Upazila',
        'Union',
        'Mouza',
        'Strata',
        'm',
        'l',
        'u',
      ],
    )

  return df

if __name__ == '__main__':
  main()
