import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def cat_int_enc(df):
    dfc = df.copy()

    for header in list(dfc.columns.values):
        if dfc[header].dtype == 'object':
            dfc[header] = pd.Categorical(dfc[header]).codes

    return dfc

train = cat_int_enc(pd.read_csv('./well_data/train.csv'))
test = cat_int_enc(pd.read_csv('./well_data/test.csv'))

train['Labels'] = np.where(train['Arsenic'] > 10, 0, 1)
print(train.drop(['Labels', 'Arsenic'], axis='columns').info())

rf_model = RandomForestClassifier(random_state=99)
rf_model.fit(
    train.drop(
        ['Labels', 'Arsenic'],
        axis='columns',
    ),
    train['Labels'],
)

test['Labels'] = rf_model.predict(
    test.drop(
        'Arsenic',
        axis='columns',
    ),
)

from sklearn import metrics
accuracy = metrics.accuracy_score(test['Labels'], test['Labels'])

print(accuracy)
