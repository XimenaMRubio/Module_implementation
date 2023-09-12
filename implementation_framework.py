if __name__ == "__main__":
    #pip install imbalanced-learn #instalar en caso de que marque un error con la librería

    #Importando las librerías
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statistics as st
    from sklearn.model_selection import train_test_split,learning_curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix,classification_report, mean_absolute_error, mean_squared_error, accuracy_score
    from imblearn.metrics import specificity_score
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    #cargando los datos
    iris = load_iris()
    x=pd.DataFrame(iris.data)
    x.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']
    y = iris.target

    #Divide el set en train, test & validation
    Xtrain, divXtest, ytrain, divytest = train_test_split(x,y,test_size=.8) #el test y training se dividen
    # test se divide nuevamente para obtener el set de validación
    Xvalidation, Xtest, yvalidation, ytest = train_test_split(divXtest,divytest,test_size=.7)

    #Creando una gráfica comparativa de la distribución de clases en training, test y validation
    plt.subplot(3, 2, 1) 
    plt.hist(ytrain,bins=3,color='green', edgecolor='black')
    plt.xlabel('Diabetes')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de categorías (Entrenamiento)')
    #Arreglando detalles para que se vea bien
    plt.text(.35,5,str(pd.DataFrame(ytrain).value_counts()[0]), ha='center', va='bottom', fontsize=10, color='black')
    plt.text(1,5,str(pd.DataFrame(ytrain).value_counts()[1]), ha='center', va='bottom', fontsize=10, color='black')
    plt.text(1.7,5,str(pd.DataFrame(ytrain).value_counts()[2]), ha='center', va='bottom', fontsize=10, color='black')

    #Configurándolas como subplots
    plt.subplot(3, 2, 2) 
    plt.hist(ytest,bins=3,color='green', edgecolor='black')
    plt.xlabel('Diabetes')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de categorías (Prueba)')
    plt.text(.35,5,str(pd.DataFrame(ytest).value_counts()[0]), ha='center', va='bottom', fontsize=10, color='black')
    plt.text(1,5,str(pd.DataFrame(ytest).value_counts()[1]), ha='center', va='bottom', fontsize=10, color='black')
    plt.text(1.7,5,str(pd.DataFrame(ytest).value_counts()[2]), ha='center', va='bottom', fontsize=10, color='black')

    #Configurándolas como subplots
    plt.subplot(3, 2, 3) 
    plt.hist(ytest,bins=3,color='green', edgecolor='black')
    plt.xlabel('Diabetes')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de categorías (Prueba)')
    plt.text(.35,5,str(pd.DataFrame(yvalidation).value_counts()[0]), ha='center', va='bottom', fontsize=10, color='black')
    plt.text(1,5,str(pd.DataFrame(yvalidation).value_counts()[1]), ha='center', va='bottom', fontsize=10, color='black')
    plt.text(1.7,5,str(pd.DataFrame(yvalidation).value_counts()[2]), ha='center', va='bottom', fontsize=10, color='black')

    plt.subplots_adjust(hspace=1.5, wspace=1.5)
    plt.show()

    #obtener los porcentajes
    total =  x.shape[0]
    labels = ['Training', 'Testing']
    sizes = [Xtrain.shape[0]/total, divXtest.shape[0]/total]
    # Crear un pie chart
    plt.title('Dataset: parcial division')
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=140)
    plt.axis('equal')  #Configurando detalles del plot

    plt.show()

    #obtener porcentajes
    total =  x.shape[0]
    labels = ['Training', 'Testing', 'Validation']
    sizes = [Xtrain.shape[0]/total, Xtest.shape[0]/total, Xvalidation.shape[0]/total]
    
    #print de los detalles de dataset
    print('El dataset tiene ',total, ' datos en total.')
    print('El set de training tiene ',Xtrain.shape[0],' datos.')
    print('El set de testing tiene ',Xtest.shape[0],' datos.')
    print('El set de validación tiene ',Xvalidation.shape[0],' datos.\n')

    # Crear un pie chart
    plt.title('Dataset division')
    plt.pie(sizes, labels=labels, autopct='%1.0f%%', startangle=140)
    plt.axis('equal')
    plt.show()

    #diccionario de parámetros 
    param_grid = {
	'n_estimators': [50, 100, 200, 350],
	'max_features': ['sqrt', 'log2', None],
	'max_depth': [3,5,10],
	'max_leaf_nodes': [3, 6, 9],
    }

    #GRID SEARCH Y RANDOM SEARCH (se comentan ya que tardan un poco en cargar y solo fue suficiente
    # correrlos una vez para obtener la mejor configuración de acuerdo a los datos proporcionados.)

    #Buscando los mejores parámetros utilizando grid search
    # grid_search = GridSearchCV(RandomForestClassifier(),
    # 						param_grid=param_grid)
    # grid_search.fit(Xtrain, ytrain)
    # print(grid_search.best_estimator_)

    #Buscando los mejores parámetros utilizando random search
    # random_search = RandomizedSearchCV(RandomForestClassifier(),
    # 								param_grid)
    # random_search.fit(Xtrain, ytrain)
    # print(random_search.best_estimator_)

    #función de implementación del modelo
    def implement_model(model,Xtrain,ytrain,Xtest,ytest,Xvalidation,yvalidation): #Xtest,ytest #Xvalidation,yvalidation
       model.fit(Xtrain, ytrain)
       #Realiza la predicción
       ypredict = model.predict(Xtest)

       #Metricas
       print('La configuración de este modelo es: ',model,'\n') #Muestra la configuración del modelo
       cr= classification_report(ytest,ypredict,zero_division=0) #Se crea el classification report
       print(cr)
       print(f"MAE: {mean_absolute_error(ytest, ypredict)}") #Se calcula el MAE
       print(f"MSE: {mean_squared_error(ytest, ypredict)}") #Se calcula el MSE
       print('Overall specificity score: ',specificity_score(ytest,ypredict,average='weighted'),'\n','\n') #Calculando el specificity
       return(accuracy_score(ytest, ypredict))
    
    #función para obtener la curva de aprendizaje
    def plot_learning_curve(model, title, X, y, train_sizes=np.linspace(.1, 1.0, 5)):
        estimator = model
        plt.figure()
        plt.title(title)
        plt.xlabel("Tamaño del Conjunto de Entrenamiento")
        plt.ylabel("Puntuación")
        #se obtienen los scores para poder crear la gráfica
        #se calcula la mean y la desviación del test y train score
        train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        #configuración de la gráfica para que se vea bien
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Puntuación de Entrenamiento")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Puntuación de Validación")

        plt.legend(loc="best")
        return (plt.show)

    #Se crea una lista de todas las configuraciones a probar
    model_list=[RandomForestClassifier(max_depth=1, n_estimators=3),
                #Esta fue la primera estrategia de mejora del modelo
                RandomForestClassifier(max_depth=1, n_estimators=5),
                #Estas dos últimas configuraciones fueron el resultado de aplicar grid y random search
                RandomForestClassifier(max_depth=3, max_leaf_nodes=3, n_estimators=50), #grid search
                RandomForestClassifier(max_depth=5, max_leaf_nodes=6, n_estimators=50)] #random search
            
    list_accuracy = []
    list_accuracy_val = []
    for i in range(len(model_list)): #se itera sobre la lista para ir probando modelo por modelo
        #Se llama a las funciones definidas previamente para obtener las métricas y las gráficas correctas
        print('MODELO',i+1)
        list_accuracy.append(implement_model(model_list[i],Xtrain,ytrain,Xtest,ytest))
        plot_learning_curve(model_list[i], str(model_list[i]), x, y)

    #creando el plot de accuracy
    plt.plot(list_accuracy)
    plt.xlabel("Models")
    plt.xticks(size = 10)
    plt.title("Accuracy using test set")
    plt.show()
