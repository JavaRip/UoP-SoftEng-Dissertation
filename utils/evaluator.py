import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
  src = sys.argv[1]

  safe_threshold = 10
  df = pd.read_csv(src)
  df['Actual'] = np.where(df['Arsenic'] > 10, 'Unsafe', 'Safe')
  df['Predictions'] = np.where(df['predictions'] != 'safe', 'Unsafe', 'Safe')

  # TODO make predictions column consistent with header names (Predictions)
  confusion_matrix = metrics.confusion_matrix(df['Actual'], df['Predictions'])
  cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, 
    display_labels=[False, True]
  ) 

  cm_display.plot()
#  plt.show()

  # print(df['Predictions'].unique())
  # print(df['Predictions'].head())
  # print(df['Actual'].unique())
  # print(df['Actual'].head())
  accuracy = metrics.accuracy_score(df['Actual'], df['Predictions'])
  precision = metrics.precision_score(df['Actual'], df['Predictions'], pos_label='Safe')
  sensitivity = metrics.recall_score(df['Actual'], df['Predictions'], pos_label='Safe')
  specificity = metrics.recall_score(df['Actual'], df['Predictions'], pos_label='Unsafe')
  f1_score = metrics.f1_score(df['Actual'], df['Predictions'], pos_label='Safe')
 
  print(f'accuracy: {accuracy}') 
  print(f'precision: {precision}') 
  print(f'specificity: {specificity}') 
  print('') # newline
  print(f'sensitivity: {sensitivity}') 
  print(f'f1_score: {f1_score}') 
