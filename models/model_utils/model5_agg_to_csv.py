import json
import pandas as pd
from sklearn.model_selection import train_test_split
from os import listdir
import sys
import os

sys.path.append(
  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
)
from model_utils.utils import stratify

def main(
  train_src='./models/model5/model/aggregate-data/',
  train_out='./well_data/model5-train.csv',
  train_label_src='./well_data/train.csv',
):
  label_df = pd.read_csv(train_label_src)

  train_df = agg_data_to_df(train_src)
  label_train_df = label_agg_data(train_df, label_df)

  label_train_df.to_csv(train_out, index=False)

def label_agg_data(agg_df, label_df):
  stratify(label_df)

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
                   mou_dict[strata]['m'] = None
                 if not 'l' in mou_dict[strata]:
                   mou_dict[strata]['l'] = None
                 if not 'u' in mou_dict[strata]:
                   mou_dict[strata]['u'] = None

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
