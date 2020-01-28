#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fecha:
    Enero 2020
Asignatura:
    Estadística Multivariante
Documentación sobre clustering en Python:
    http://scikit-learn.org/stable/modules/clustering.html
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
    http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
    http://www.learndatasci.com/k-means-clustering-algorithms-python-intro/
"""
import time

import matplotlib.pyplot as plt
import pandas as pd
import os

from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering

from sklearn import metrics
from sklearn import preprocessing
from math import floor
import seaborn as sns

from scipy.cluster import hierarchy
import warnings

def norm_to_zero_one(df):
  return (df - df.min()) * 1.0 / (df.max() - df.min())

# Dibujar Scatter Matrix
def ScatterMatrix(X, name, path):
  print("\nGenerando scatter matrix...")
  sns.set()
  variables = list(X)
  variables.remove('cluster')
  sns_plot = sns.pairplot(X, vars=variables, hue="cluster", palette='Paired', 
                          plot_kws={"s": 25}, diag_kind="hist") 
  #en 'hue' indicamos que la columna 'cluster' define los colores
  sns_plot.fig.subplots_adjust(wspace=.03, hspace=.03);
   
  plt.savefig(path+"scatmatrix"+name+".png")
  plt.clf()

# Dibujar Heatmap
def Heatmap(X, name, path, dataset, labels):
  print("\nGenerando heat-map...")
  cluster_centers = X.groupby("cluster").mean()
  centers = pd.DataFrame(cluster_centers, columns=list(dataset))
  centers_desnormal = centers.copy()
  #se convierten los centros a los rangos originales antes de normalizar
  for var in list(centers):
    centers_desnormal[var] = dataset[var].min()+centers[var]*(dataset[var].max()-dataset[var].min())

  plt.figure(figsize=(11, 13))
  sns.heatmap(centers, cmap="YlGnBu", annot=centers_desnormal, fmt='.3f')
  plt.savefig(path+"heatmap"+name+".png")
  plt.clf()

# Dibujar Dendogramas (con y sin scatter matrix)
def Dendrograms(X, name, path):
  print("\nGenerando dendogramas...")
  #Para sacar el dendrograma en el jerárquico, no puedo tener muchos elementos.
  #Hago un muestreo aleatorio para quedarme solo con 1000, 
  #aunque lo ideal es elegir un caso de estudio que ya dé un tamaño así
  if len(X)>1000:
     X = X.sample(1000, random_state=seed)
  #Normalizo el conjunto filtrado
  X_filtrado_normal = preprocessing.normalize(X, norm='l2')
  linkage_array = hierarchy.ward(X_filtrado_normal)
  #Saco el dendrograma usando scipy, que realmente vuelve a ejecutar el clustering jerárquico
  hierarchy.dendrogram(linkage_array, orientation='left')
    
  plt.savefig(path+"dendrogram"+name+".png")
  plt.clf()
    
  X_filtrado_normal_DF = pd.DataFrame(X_filtrado_normal,index=X.index,columns=usadas)
  sns.clustermap(X_filtrado_normal_DF, method='ward', col_cluster=False, figsize=(20,10), cmap="YlGnBu", yticklabels=False)

  plt.savefig(path+"dendscat"+name+".png")
  plt.clf()
  
# Dibujar KdePlot
def KPlot(X, name, k, usadas, path):
  print("\nGenerando kplot...")
  n_var = len(usadas)
  fig, axes = plt.subplots(k, n_var, sharex='col', figsize=(15,10))
  fig.subplots_adjust(wspace=0.2)
  colors = sns.color_palette(palette=None, n_colors=k, desat=None)

  for i in range(k):
    dat_filt = X.loc[X['cluster']==i]
    for j in range(n_var):
      sns.kdeplot(dat_filt[usadas[j]], shade=True, color=colors[i], ax=axes[i,j])
  
  plt.savefig(path+"kdeplot"+name+".png")
  plt.clf()
  
# Dibujar BoxPlot
def BoxPlot(X, name, k, usadas, path):
  print("\nGenerando boxplot...")
  n_var = len(usadas)
  fig, axes = plt.subplots(k, n_var, sharey=True, figsize=(16, 16))
  fig.subplots_adjust(wspace=0.4, hspace=0.4)
  colors = sns.color_palette(palette=None, n_colors=k, desat=None)
  rango = []

  for i in range(n_var):
    rango.append([X[usadas[i]].min(), X[usadas[i]].max()])

  for i in range(k):
    dat_filt = X.loc[X['cluster']==i]
    for j in range(n_var):
      ax = sns.boxplot(dat_filt[usadas[j]], color=colors[i], ax=axes[i, j])
      ax.set_xlim(rango[j][0], rango[j][1])
      
  plt.savefig(path+"boxplot"+name+".png")
  plt.clf()


def ejecutarAlgoritmos(algoritmos, X, etiq, usadas, path):
    
  # Crea el directorio si no existe  
  try:
    os.stat(path)
  except:
    os.mkdir(path)    
    
  X_normal = X.apply(norm_to_zero_one)

  # Listas para almacenar los valores
  nombres = []
  tiempos = []
  numcluster = []
  metricaCH = []
  metricaSC = []
    
  for name,alg in algoritmos:    
    print(name,end='')
    t = time.time()
    cluster_predict = alg.fit_predict(X_normal) 
    tiempo = time.time() - t
    k = len(set(cluster_predict))
    print(": clusters: {:3.0f}, ".format(k),end='')
    print("{:6.2f} segundos".format(tiempo))

    # Calculamos los valores de cada métrica
    metric_CH = metrics.calinski_harabaz_score(X_normal, cluster_predict)
    print("\nCalinski-Harabaz Index: {:.3f}, ".format(metric_CH), end='')
    #el cálculo de Silhouette puede consumir mucha RAM. 
    #Si son muchos datos, más de 10k, se puede seleccionar una muestra, p.ej., el 20%
    if len(X) > 10000:
      m_sil = 0.2
    else:
      m_sil = 1.0
    metric_SC = metrics.silhouette_score(X_normal, cluster_predict, metric='euclidean', 
                                         sample_size=floor(m_sil*len(X)), random_state=seed)
    print("Silhouette Coefficient: {:.5f}".format(metric_SC))
    
    #se convierte la asignación de clusters a DataFrame
    clusters = pd.DataFrame(cluster_predict,index=X.index,columns=['cluster'])
    #y se añade como columna a X
    X_cluster = pd.concat([X_normal, clusters], axis=1)
 
    print("\nTamaño de cada cluster:\n")
    size = clusters['cluster'].value_counts()

    for num,i in size.iteritems():
      print('%s: %5d (%5.2f%%)' % (num,i,100*i/len(clusters)))
    
    nombre = name+str(etiq)
    
    # Dibujamos el Scatter Matrix
    ScatterMatrix(X = X_cluster, name = nombre, path = path)
    # Dibujamos el Heatmap
    Heatmap(X = X_cluster, name = nombre, path = path, dataset=X, labels = cluster_predict)
    
    # Dibujamos KdePlot
    KPlot(X = X_cluster, name = nombre, k = k, usadas = usadas, path = path)
    
    # Dibujamos BoxPlot
    BoxPlot(X = X_cluster, name = nombre, k = k, usadas = usadas, path = path)

    if name=='AggCluster':
      #Filtro quitando los elementos (outliers) que caen en clusters muy pequeños en el jerárquico
      min_size = 5
      X_filtrado = X_cluster[X_cluster.groupby('cluster').cluster.transform(len) > min_size]
      k_filtrado = len(set(X_filtrado['cluster']))
      print("De los {:.0f} clusters hay {:.0f} con más de {:.0f} elementos. Del total de {:.0f} elementos, se seleccionan {:.0f}".format(k,k_filtrado,min_size,len(X),len(X_filtrado)))
      X_filtrado = X_filtrado.drop('cluster', 1)
      Dendrograms(X = X_filtrado, name = nombre, path = path)
    
    # Almacenamos los datos para generar la tabla comparativa
    nombres.append(name)   
    tiempos.append(tiempo)
    numcluster.append(len(set(cluster_predict)))
    metricaCH.append(metric_CH)
    metricaSC.append(metric_SC)
    
    print("\n-------------------------------------------\n")
    
  # Generamos la tabla comparativa  
  resultados = pd.concat([pd.DataFrame(nombres, columns=['Name']), 
                          pd.DataFrame(numcluster, columns=['Num Clusters']), 
                          pd.DataFrame(metricaCH, columns=['CH']), 
                          pd.DataFrame(metricaSC, columns=['SC']), 
                          pd.DataFrame(tiempos, columns=['Time'])], axis=1)
  print(resultados)

if __name__ == '__main__':  
    
  datos = pd.read_csv('iris.csv')
  seed = 12345
  warnings.filterwarnings(action='ignore', category=FutureWarning)
  warnings.filterwarnings(action='ignore', category=RuntimeWarning)
  
  # Crea el directorio images si no existe  
  try:
    os.stat("./imagenes/")
  except:
    os.mkdir("./imagenes/")    
  
  for col in datos:
    datos[col].fillna(datos[col].mean(), inplace=True)
      
  #********************************************   
  subset = datos
  usadas = ['PETAL-LENGTH', 'SEPAL-LENGTH', 'SEPAL-WIDTH', 'PETAL-WIDTH']
  X = subset[usadas]
  #********************************************
       
  # En clustering hay que normalizar para las métricas de distancia
  X_normal = preprocessing.normalize(X, norm='l2')
   
  # Algoritmos de clustering utilizados 
  k_means = KMeans(init='k-means++', n_clusters=3, n_init=5, random_state=seed)
  ms = MeanShift(bandwidth=estimate_bandwidth(X_normal, quantile=0.67, n_samples=400), bin_seeding=True)
  ward = AgglomerativeClustering(n_clusters=3, linkage="ward", affinity='euclidean')
  db = DBSCAN(eps=0.12, min_samples=5)
  
  algoritmos1 = {('K-Means', k_means), ('MeanShift', ms), 
               ('AggCluster', ward), ('DBSCAN', db)}
  
  path = "./imagenes/"
  
  print("\nIRIS DATABASE , tamaño: "+str(len(X))+"\n")
  print("-------------------------------------------\n")
  ejecutarAlgoritmos(algoritmos1, X, "IRIS", usadas, path)