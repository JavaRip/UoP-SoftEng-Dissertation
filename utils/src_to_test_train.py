import numpy as np
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def main():
  file = open('./well_data/src_data.json', 'r')
  raw = file.read()
  data = json.loads(raw)

  csv_arr = []

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
             for well in mou_dict['wells']:
                csv_arr.append([div, dis, upa, uni, mou, well['depth'], well['arsenic']])

  df = pd.DataFrame(
    csv_arr,
    columns=[
      'Division',
      'District',
      'Upazila',
      'Union',
      'Mouza',
      'Depth',
      'Arsenic',
    ],
  )

  # shuffle data
  df = df.sample(frac=1)

  df.to_csv('./well_data/src_data.csv', index=False)

  # split test train into 5 k folds
  df_split = np.array_split(df, 5)

  df_split[0].to_csv('./well_data/k1.csv', index=False)
  df_split[1].to_csv('./well_data/k2.csv', index=False)
  df_split[2].to_csv('./well_data/k3.csv', index=False)
  df_split[3].to_csv('./well_data/k4.csv', index=False)
  df_split[4].to_csv('./well_data/k5.csv', index=False)

if __name__ == '__main__':
  main()
