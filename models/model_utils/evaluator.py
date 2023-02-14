import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

def evaluate(df, label_col='Label', pred_col='Prediction'):
  confusion_matrix = metrics.confusion_matrix(df[label_col], df[pred_col])

  cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, 
    display_labels=[False, True]
  ) 

  cm_display.plot()
  # plt.show()

  accuracy = metrics.accuracy_score(df[label_col], df[pred_col])
  precision = metrics.precision_score(df[label_col], df[pred_col], pos_label='safe')
  sensitivity = metrics.recall_score(df[label_col], df[pred_col], pos_label='safe')
  specificity = metrics.recall_score(df[label_col], df[pred_col], pos_label='polluted')
  f1_score = metrics.f1_score(df[label_col], df[pred_col], pos_label='safe')
 
  print(f'accuracy: {accuracy}') 
  print(f'precision: {precision}') 
  print(f'specificity: {specificity}') 
  print('') # newline
  print(f'sensitivity: {sensitivity}') 
  print(f'f1_score: {f1_score}') 

if __name__ == '__main__':
  src = sys.argv[1]
  df = pd.read_csv(src)
  evaluate(df)
