# Module implementation
Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework.
En este caso se utiliza KNN y se hacen pruebas con diferentes números de vecinos.

Se emplean tres funciones.

1# Función que calcula la distancia auclidiana entre dos vectores

2# Función que calcula los vecinos, donde se le da la base de entrenamiento,el nuevo registro a clasificar y el número de vecinos

3# Función que clasifica empleando knn. Esta función toma en cuenta una posible multimoda entre los vecinos seleccionados.


if __name__ == "__main__":
  import pandas as pd
  import numpy as np
  from math import sqrt
  import statistics as st
  from sklearn.datasets import load_iris
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import accuracy_score, confusion_matrix

#función que calcula la distancia auclidiana entre dos vectores
  def euclidean_d(row1,row2):
    distance = 0
    distance = sqrt(sum((row1 - row2)**2))
    return(distance)

#función que calcula los vecinos, donde se le da la base de entrenamiento,el nuevo registro a clasificar y el número de vecinos
  def neighbors(data_train,new_row,num_neighbors): #get neighbors for one row
    data_train = pd.DataFrame(data_train)
    list_distance = []
    for row in range(len(data_train)):
      list_distance.append((euclidean_d(data_train.iloc[row][:-1],new_row),list(data_train.iloc[row])))
    list_distance.sort()
    neighbors = []
    for favs in range(num_neighbors): #toma el número de vecinos seleccionados.
      neighbors.append(list_distance[favs])
    return(neighbors)

#función que clasifica empleando knn
  def classificationknn(data_train,new_row,num_neighbors):
    neighbors_final = list(neighbors(data_train,new_row,num_neighbors)) #lista de vecinos
    classes= []
    for vec in range(len(neighbors_final)):
      classes.append(neighbors_final[vec][1][-1])
    class_pred = st.multimode(classes)
    if len(class_pred) > 1: #caso multimoda
      second_review=[] #lista con las modas multimolda (si es que hay)
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

#baja la base de datos iris y separa el dataset en prueba y entrenamiento
  iris = load_iris()
  x=pd.DataFrame(iris.data)
  x.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
  y = iris.target
  Xtrain, Xtest, ytrain, ytest = train_test_split(x,y, random_state=5)
  Xtrain['class'] = ytrain

#Se toman de 1 a 10 vecinos y se calcula la matriz de confusión y el accuracy de cada uno de los casos.
  for knn_v in [1,2,3,4,5,6,7,8,9,10]:
    final_classifier=[]
    for xtest in range(len(Xtest)):
      trial = classificationknn(Xtrain,list(Xtest.iloc[xtest]),knn_v)
      final_classifier.append(trial)
    print('\n Using',knn_v,'neighbors:')
    print('The confusion matrix is','\n',confusion_matrix(ytest, final_classifier))
    print('The accuracy is',accuracy_score(ytest, final_classifier))
