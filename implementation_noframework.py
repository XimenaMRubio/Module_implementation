# -*- coding: utf-8 -*-
"""Implementation_noframework

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K5YIYqL_bMKzahcWJqTqmTxCfyq8dr9E
"""

if __name__ == "__main__":
  import pandas as pd
  import numpy as np
  from math import sqrt
  import statistics as st
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score, confusion_matrix

  def euclidean_d(row1,row2):
    distance = 0
    distance = sqrt(sum((row1 - row2)**2)) #get the euclidean distance from 2 vectors
    return(distance)

  def neighbors(data_train,new_row,num_neighbors): #get neighbors for one row
    data_train = pd.DataFrame(data_train)
    list_distance = []
    for row in range(len(data_train)):
      list_distance.append((euclidean_d(data_train.iloc[row][:-1],new_row),list(data_train.iloc[row])))
    list_distance.sort()
    neighbors = []
    for favs in range(num_neighbors):
      neighbors.append(list_distance[favs])
    return(neighbors)

  def classificationknn(data_train,new_row,num_neighbors):
    neighbors_final = list(neighbors(data_train,new_row,num_neighbors)) #list of neighbors
    classes= []
    for vec in range(len(neighbors_final)):
      classes.append(neighbors_final[vec][1][-1])
    class_pred = st.multimode(classes)
    if len(class_pred) > 1: #multimode case
      second_review=[] #list with the rows of multimode
      for rep in range(len(neighbors_final)):
        if neighbors_final[rep][1][-1] in class_pred:
          second_review.append(neighbors_final[rep])
      min_d = []
      for clean in range(len(second_review)):
        min_d.append((second_review[clean][0],second_review[clean][1][-1]))
      clean_distance = min(min_d)
      return(clean_distance[1])
    else:
      return(int(class_pred[0]))

  iris = load_iris()
  x=pd.DataFrame(iris.data)
  x.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
  y = iris.target
  Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, random_state=5)
  Xtrain['class'] = ytrain

  for knn_v in [1,2,3,4,5,6,7,8,9,10]:
    final_classifier=[]
    for xtest in range(len(Xtest)):
      trial = classificationknn(Xtrain,list(Xtest.iloc[xtest]),knn_v)
      final_classifier.append(trial)
    print('\n Using',knn_v,'neighbors:')
    print('The confusion matrix is','\n',confusion_matrix(ytest, final_classifier))
    print('The accuracy is',accuracy_score(ytest, final_classifier))