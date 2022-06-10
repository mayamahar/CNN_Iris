import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.inspection import DecisionBoundaryDisplay

from run import klasifikasi

n_neighbors = 1

def ambildataset():
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X = X[:, [0, 2]]
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.7, random_state=42)
def fungsi():
    for name, clf in zip(nama.names, klasifikasi.classifiers):
        clf.fit(ambildataset.X_train, ambildataset.y_train)
        score = clf.score(ambildataset.X_test, ambildataset.y_test)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        ambildataset.X,
        cmap=colour.cmap_light,
        alpha=0.8,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        shading="auto",
    )    
def colour():
    cmap_light = ListedColormap(["#FFAAAA", "#AAFFAA", "#AAAAFF"])
    cmap_bold = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

def nama():
    names = ["KNN", "NCA, KNN"]