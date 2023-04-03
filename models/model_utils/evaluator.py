import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import sys

def gen_conf_mat(df, label_col='Label', pred_col='Prediction'):
  confusion_matrix = metrics.confusion_matrix(df[label_col], df[pred_col])

  cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion_matrix, 
    display_labels=[False, True]
  ) 

  cm_display.plot()
  plt.show()

def gen_eval(df, label_col='Label', pred_col='Prediction'):
  accuracy = metrics.accuracy_score(df[label_col], df[pred_col])
  precision = metrics.precision_score(df[label_col], df[pred_col], pos_label='polluted')
  sensitivity = metrics.recall_score(df[label_col], df[pred_col], pos_label='polluted')
  specificity = metrics.recall_score(df[label_col], df[pred_col], pos_label='safe')
  f1_score = metrics.f1_score(df[label_col], df[pred_col], pos_label='polluted')
 
  return {
    'accuracy': accuracy,
    'precision': precision,
    'specificity': specificity,
    'sensitivity': sensitivity,
    'f1': f1_score,
  }

def print_eval(eval):
  print(f'accuracy: {eval["accuracy"]}') 
  print(f'precision: {eval["precision"]}') 
  print(f'specificity: {eval["specificity"]}') 
  print('') # newline
  print(f'sensitivity: {eval["sensitivity"]}') 
  print(f'f1_score: {eval["f1"]}') 


if __name__ == '__main__':
  src = sys.argv[1]
  df = pd.read_csv(src)
  gen_eval(df)
