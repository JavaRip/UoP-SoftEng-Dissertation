import json
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
  def main(src_path='./well_data/src_data.json'):
    file = open(src_path, 'r')
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
      columns=['div', 'dis', 'upa', 'uni', 'mou', 'depth', 'as'],
    )

    train, test = train_test_split(
      df,
      test_size=0.2,
      random_state=99,
      # stratify=df['mou'], # what to do with mouzas containing less than 1 well
    )

    train.to_csv('./well_data/train.csv')
    test.to_csv('./well_data/test.csv')

  main()
