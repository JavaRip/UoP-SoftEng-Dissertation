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

train['Label'] = np.where(train['Arsenic'] > 10, 'Unsafe', 'Safe')
test['Label'] = np.where(test['Arsenic'] > 10, 'Unsafe', 'Safe')

train_X = train.drop(['Arsenic', 'Label'], axis='columns')
train_y = train['Label']
test_X = test.drop(['Arsenic', 'Label'], axis='columns')
test_y = test['Label']

rf_model = RandomForestClassifier(random_state=99)
rf_model.fit(train_X, train_y)

predictions = rf_model.predict(test_X)

from sklearn import metrics
accuracy = metrics.accuracy_score(predictions, test_y)

print(accuracy)
