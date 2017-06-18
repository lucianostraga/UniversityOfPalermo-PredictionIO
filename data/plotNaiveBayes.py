import numpy as np
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np

labels , features = np.loadtxt('data.txt', delimiter = ',',dtype=str, unpack=True)

np.savetxt('features.txt',features, fmt="%s")
np.savetxt('labels.txt',labels , fmt="%s")

labels = np.loadtxt('labels.txt',dtype=int)
features = np.loadtxt('features.txt',dtype=int , delimiter=' ')

os.remove("labels.txt")
os.remove("features.txt")

X_train, X_test, y_train, y_test = train_test_split(
features, labels, test_size=0.2, random_state=0)

clf = MultinomialNB()
clf.fit(X_train,y_train)

pred = clf.predict(X_test)

accuracy = accuracy_score(pred, y_test)

print(accuracy)

#X = features[:, [5, 6]] 
X = features[:, :2]

#X = features
y = labels

pca = PCA(n_components=2)
X_r = pca.fit_transform(features)

tsne = TSNE(n_components=2, random_state=0)
X_r3 = tsne.fit_transform(features,y)

target_names  = ["Passed","Fail"]

plt.figure()
colors = ['navy', 'darkorange']
lw = 2

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('FIST TWO VARIABLES PLOT - RAW DATA')
plt.xlabel('x')
plt.ylabel('y')

plt.figure()

for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('TRANING SET - PCA PLOT')
plt.xlabel('x')
plt.ylabel('y')

plt.figure()
for color, i, target_name in zip(colors, [0, 1], target_names):
    plt.scatter(X_r3[y == i, 0], X_r3[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('TRAINING SET - t-SNE PLOT')
plt.xlabel('x')
plt.ylabel('y')

plt.show()

