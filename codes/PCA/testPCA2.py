
from sklearn.decomposition import PCA
import numpy as np


from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

X = iris.data
y = iris.target
print (X.shape)

iris_pca = PCA(n_components=2,copy=False,random_state=8)

X = iris_pca.fit_transform(X)
print (X.shape)

plt.scatter(X[:,0],X[:,1],c=y)
plt.show()


# print (iris_pca.components)

from sklearn.datasets import load_digits
digits = load_digits()
digit_data = digits.data

sub_data = digit_data[0:100,:]
print (sub_data.shape)

fig, axe = plt.subplots(1,12,subplot_kw=dict(xticks=[], yticks=[]))
for i in range(0,12):
    axe[i].imshow(sub_data[i,:].reshape((8,8)),cmap=plt.cm.binary, interpolation='nearest')

plt.show()