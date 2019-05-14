"""
Main file for learning K-means clustering algo and Gaussian mixture models clustering algo using Scikit-learn.
Author: Ísak Steingrimsson - fev004
"""
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plot
import numpy as np

#Reading in Excelfile
xls = pd.ExcelFile("Engelsberger_short.xlsx")
#Choosing sheet and parsing in all data into and dataframe called Ed(Engelsberger data)
Ed = xls.parse("Engelsberger et al_tpj_4848_sm_")
"""
Engelsberger data column names: ['Accession (AGI code)' 'Sequence with modifications' 'treatment' 0 3 5 10
 30]. First three can be seleced with string but 0, 3, 5, 10, 30 are accesed with the integer numbers.
 Example Ed[5] for integers and Ed["treatment"] where header is an string.
"""

"""
Data processing stage. These dataframes will be globals since I will have to use them
in both the Kmeans and gaussian mixture clustering. There are 2 dataframes that I will test:
EdFilledNaN is a dataframed where I fill in missing values with the mean value of all values in x column
EdRemovedNaN is a dataframe where I removed every row that contained a NaN value. Original row count is 640 rows,
after removing rows with NaN values the row count is 83 rows.  
"""
#Dropping non numerical columns on original dataframe
Ed = Ed.drop(columns = ["Accession (AGI code)", "Sequence with modifications", "treatment"])

EdFilledNaN = Ed.copy()
#Fills in the missing values with the mean value from that column
EdFilledNaN.fillna(EdFilledNaN.mean(), inplace = True)

#Drops every row that contains a null-NaN value
EdRemovedNaN = Ed.dropna()
"""
Initiating the Principal component analysis. 
n_components is how many dimensions we want to end up with. In my  case I want 2 dimensions to plot
on my scatterplot.
"""
#pca remove NaN
#pcar = PCA(n_components = 2)
#pcar.fit(EdRemovedNaN)
#EdRemovedNaNPCA = pcar.transform(EdRemovedNaN)
#pca filled NaN
#pcaf = PCA(n_components = 2)
#pcaf.fit(EdFilledNaN)
#EdFilledNaNPCA = pcaf.transform(EdFilledNaN)


#print(len(EdRemovedNaNPCA))
#print("PCA components:  " + str(EdRemovedNaNPCA))
#print(type(EdRemovedNaNPCA))

"""
Parameters for Kmeans function:
    dataframe: Pass in either EdFilledNaN or EdRemovedNaN
"""
def Kmeans(dataframe):

    km = KMeans(n_clusters = 3, n_init = 30)
    km.fit(dataframe)
    print("Cluster centers(centroids): " + str(km.cluster_centers_))
    print("Cluster nembership:\n{}".format(km.labels_))
    print("Number of iterations: " + str(km.n_iter_))
    #Getting out the data so that I can run it through PCA. dataT = data Transformed
    #Use this in PCA or original dataframe?
    dataT = km.transform(dataframe)
    print(type(dataT))
    """
    Initiating the Principal component analysis. 
    n_components is how many dimensions we want to end up with. In my  case I want 2 dimensions to plot
    on my scatterplot.
    """
    pca = PCA(n_components = 2)
    pca.fit(dataframe)
    dataPCA = pca.transform(dataframe)

    pcac = PCA(n_components = 2)
    pcac.fit(km.cluster_centers_)
    CentroidsPCA = pcac.transform(km.cluster_centers_)
    
    #I wanted to color in the centroids X with its cluster-color aswell
    centroidsColor = []
    for i in range(km.n_clusters):
        centroidsColor.append(i)
    
    #dataframe[:,0] Means take everything at index 0 in each of the nested arrays
    plot.scatter(dataPCA[:,0], dataPCA[:,1], c = km.labels_)
    plot.scatter(CentroidsPCA[:,0], CentroidsPCA[:,1], c = centroidsColor, marker = "x", s = 80)
    plot.title("X are cluster centroids\nDatapoints are colored after which cluster they are in.")
    plot.xlabel("PCA1")
    plot.ylabel("PCA2")
    plot.show()


"""
Parameters for Gaussian mixture function:
    dataframe: Pass in either EdFilledNaN or EdRemovedNaN
"""
def gaussianMixture(dataframe):
    GM = GaussianMixture(n_components = 3, covariance_type = "diag")
    GM.fit(dataframe)
    print("The mean of each mixture components: " + str(GM.means_))
    print("The covariance of each mixture components" + str(GM.covariances_))
    print("The weights of each mixture components: " + str(GM.weights_))
    print("Number of iterations(steps): " + str(GM.n_iter_))
    
    pca = PCA(n_components = 2)
    pca.fit(dataframe)
    dataPCA = pca.transform(dataframe)

    pcam = PCA(n_components = 2)
    pcam.fit(GM.means_)
    meanPCA = pcam.transform(GM.means_)

    #Labels is very usefull since it allows me to color each point after which component they are in
    labels = GM.predict(dataframe)

    meansColor = []
    for i in range(GM.n_components):
        meansColor.append(i)

    plot.scatter(dataPCA[:,0], dataPCA[:,1], c = labels)
    plot.scatter(meanPCA[:,0], meanPCA[:,1], marker = "x", s = 80, c = meansColor)
    plot.title("X are means")
    plot.xlabel("PCA1")
    plot.ylabel("PCA2")
    plot.show()


#Kmeans(EdRemovedNaN)
gaussianMixture(EdRemovedNaN)

