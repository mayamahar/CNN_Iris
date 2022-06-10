import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.inspection import DecisionBoundaryDisplay

from CNN import ambildataset
from CNN import colour
from CNN import fungsi
from CNN import nama

n_neighbors = 1

def klasifikasi():
    classifiers = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("nca", NeighborhoodComponentsAnalysis()),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
        ]
    ),
]

def training():
    plt.scatter(ambildataset.X[:, 0], ambildataset.X[:, 1], c=ambildataset.y, cmap=colour.cmap_bold, edgecolor="k", s=20)
    plt.title("{} (k = {})".format(nama.name, n_neighbors))
    plt.text(
        0.9,
        0.1,
        "{:.2f}".format(fungsi.score),
        size=15,
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

plt.show()
