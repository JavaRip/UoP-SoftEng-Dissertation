import json
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def main(src_in, src_out, train_out, test_out):
  file = open(src_in, 'r')
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

  df.to_csv(src_out, index=False)

  # TODO make this a function which you pass the well limit to & return df

  # remove mouzas with less than 200 wells
  # todo set with cmd args
  # df_agg = df.groupby(df['Mouza'], as_index=False).count()
  # df_agg['mou_count'] = df_agg['Depth']
  # df_agg.drop(columns=['Depth'])

  # df['mou_count'] = pd.merge(
  #     df_agg[['Mouza', 'mou_count']],
  #     df,
  #     on='Mouza'
  # )['mou_count']

  # min_wells = 200
  # df = df[df['mou_count'] > min_wells]

  train, test = train_test_split(
    df,
    test_size=0.2,
    random_state=99,
    # stratify=df['mou'], # what to do with mouzas containing less than 1 well
  )

  train.to_csv(train_out, index=False)
  test.to_csv(test_out, index=False)

if __name__ == '__main__':

  # cli flags could be better than this
  # cli options for min wells in mouza
  # cli option for num rows to split (to make smaller test trains)
  src_in = sys.argv[1]
  src_out = sys.argv[2]
  train_out = sys.argv[3]
  test_out = sys.argv[4]

  main(src_in, src_out, train_out, test_out)
