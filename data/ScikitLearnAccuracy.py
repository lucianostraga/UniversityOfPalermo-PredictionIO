import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#CALCULATES ACCURACY WITH RAW DATA

labels , features = np.loadtxt('data.txt', delimiter = ',',dtype=str, unpack=True)

np.savetxt('features.txt',features, fmt="%s")
np.savetxt('labels.txt',labels , fmt="%s")

labels = np.loadtxt('labels.txt',dtype=int)
features = np.loadtxt('features.txt',dtype=int , delimiter=' ')

os.remove("labels.txt")
os.remove("features.txt")

X_train, X_test, y_train, y_test = train_test_split(
features, labels, test_size=0.2, random_state=0)


clf = GaussianNB()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
accuracy = accuracy_score(pred, y_test)
print(accuracy)

#CALCULATES ACCURACY WITH PCA TRANSFORMATION

pca = PCA(n_components=2)
X_r = pca.fit_transform(features)
X_train2, X_test2, y_train2, y_test2 = train_test_split(
X_r, labels, test_size=0.2, random_state=0)

clf = GaussianNB()
clf.fit(X_train2,y_train2)
pred2 = clf.predict(X_test2)
accuracy = accuracy_score(pred2, y_test2)
print(accuracy)

#CALCULATES ACCURACY WITH t-SNE TRANSFORMATION

tsne = TSNE(n_components=2, random_state=0)
X_r3 = tsne.fit_transform(features,labels)
X_train3, X_test3, y_train3, y_test3 = train_test_split(
X_r3, labels, test_size=0.2, random_state=0)

clf = GaussianNB()
clf.fit(X_train3,y_train3)
pred3 = clf.predict(X_test3)
accuracy = accuracy_score(pred3, y_test3)
print(accuracy)