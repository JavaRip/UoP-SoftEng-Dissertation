import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from os import listdir

# TODO update src_out
def main(agg_src='./models/model5/aggregate-data/', src_out='./output.csv'):
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

    df.to_csv(src_out, index=False)

if __name__ == '__main__':
  main()
