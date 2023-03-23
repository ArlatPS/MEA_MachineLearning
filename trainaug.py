import numpy as np
from sklearn import decomposition

def generateData(pca, x, start):
    original = pca.components_.copy()
    ncomp = pca.components_.shape[0]
    a = pca.transform(x)
    for i in range(start, ncomp):
        pca.components_[i,:] += np.random.normal(scale=0.1, size=ncomp)
        b = pca.inverse_transform(a)
        pca.components_ = original.copy()
        return b

x_trn = np.load("x_trn.npy")
y_trn = np.load("y_trn.npy")

pca = decomposition.PCA(n_components=120)
pca.fit(x_trn)
print(pca.explained_variance_ratio_)
print(x_trn.shape)

start = 4
nsets = 10
nsamp = x_trn.shape[0]
new_x = np.zeros((nsets*nsamp, x_trn.shape[1]))
new_y = np.zeros((nsets*nsamp))

for i in range(nsets):
    if (i == 0):
        new_x[0:nsamp,:] = x_trn
        new_y[0:nsamp] = y_trn
    else:
        new_x[(i * nsamp):((i + 1) * nsamp), :] = generateData(pca,x_trn,start)
        new_y[(i*nsamp):((i + 1)*nsamp)] = y_trn

idx = np.argsort(np.random.random(nsets * nsamp))
new_x = new_x[idx]
new_y = new_y[idx]

np.save("x_trn_aug", new_x)
np.save("y_trn_aug", new_y)