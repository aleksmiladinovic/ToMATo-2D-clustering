# Application of the ToMATo algorithm for image segmentation

import numpy as np
import matplotlib.pyplot as plt

# Image segmentation

img_rgb = plt.imread('SampleImage.png')

# Reducing the resolution
gap = 4
img_rgb_new = np.asarray([[[255*x for x in list(img_rgb[i,j,:])] for j in range(0,img_rgb.shape[1],gap)] for i in range(0,img_rgb.shape[0],gap)]).astype('uint8')

plt.title('RGB image')
plt.imshow(img_rgb)
plt.show()

plt.title('RGB image new')
plt.imshow(img_rgb_new)
plt.show()

# Converting to LUV
import cv2

img_luv = cv2.cvtColor(img_rgb_new, cv2.COLOR_RGB2Luv)
plt.title('LUV image')
plt.imshow(img_luv)
plt.show()

l, u, v = [], [], []
for i in range(0,img_luv.shape[0]):
    for j in np.arange(0,img_luv.shape[1]):
        l.append(img_luv[i,j,0])
        u.append(img_luv[i,j,1])
        v.append(img_luv[i,j,2])

# Data

pts = np.vstack((l,u,v)).T


# Finding the best parameters for density estimation

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
#import matplotlib as mpl
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits import mplot3d

#bdws = 10**np.linspace(-1, 1, 10)
bdws = np.linspace(0.1, 1, 10)
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bdws}, cv=10)
grid.fit(pts)
bw = grid.best_params_['bandwidth']
kde = grid.best_estimator_

vec = np.exp(kde.score_samples(pts))

plt.fill_between(np.arange(len(pts)), vec, alpha=0.5)
plt.show()


# Finding the best value for the radius parameter in the Rips graph

from sklearn.neighbors import KDTree

kdt = KDTree(pts, metric='euclidean')

rads = np.linspace(0, 100, 20)
# Average number of neighbors depending on the radius
cnbs = [np.mean([kdt.query_radius([pts[i]], rads[j], return_distance=False)[0].shape[0] for i in np.arange(pts.shape[0])]) for j in range(len(rads))]
plt.plot(rads, cnbs)
plt.title("Average number of neighbors vs. the radius")
plt.xlabel("Radius")
plt.ylabel("Average number of neighbors")
plt.show()

# Using the elbow method we use the value 0.6 as the best one
rad = 58


# Optimizing

# Sorted values of estimated density values and their indexes
ids = [i for (e,i) in sorted([(e,i) for i,e in enumerate(vec)])]
vecs = {i:e for (e,i) in sorted([(e,i) for i,e in enumerate(vec)])}

m = len(vec)
groups = {}           # Groups of points
ptso = []   # Points of the optimized set
veco = []   # Vec values of the optimized set
pts_idx = {}          # Point indexes
markedp = m*[False]   # Marked points

for i in reversed(range(m)):
    ix = ids[i]
    if markedp[ix] == False:
        markedp[ix] = True
        nbs = kdt.query_radius([pts[ix]], rad/20, return_distance = False)[0]
        for j in nbs:
            markedp[j] = True
        groups[ix] = [j for j in nbs]
        #ptso = np.append(ptso, [[pts[i][0], pts[i][1], pts[i][2]]])
        ptso.append(pts[ix])
        pts_idx[len(veco)] = ix
        veco.append(vecs[ix])

# Sorted values of optimized estimated density values and their indexes
ids = [i for (e,i) in sorted([(e,i) for i,e in enumerate(veco)])]
vecs = {i:e for (e,i) in sorted([(e,i) for i,e in enumerate(veco)])}


# Disjoint set structure

n = len(veco)
uf = {}
births = {}
deaths = {}

def FindClusters( rad, tau ):
    uf = {}
    births = {}
    deaths = {}
    rt = np.array([-1] * n)
    for i in reversed(range(n)):
        ix = ids[i]
        nbs = kdt.query_radius([pts[ix]], rad, return_distance=False)[0]
        s = [x for x in nbs if x in ids[i + 1:]] # Neighbors with the greater density function value
        if not s:
            uf[ix] = np.array([ix])  # Adding new component
            births[ix] = vecs[ix] # Adding new class
            rt[ix] = ix  # setting the root of the component
        else:
            p = s[np.asarray([vecs[j] for j in s]).argmax()] # Finding the parent
            rt[ix] = rt[p] # Setting the root
            uf[rt[p]] = np.append(uf[rt[p]], [ix])
            for x in s: # Merging sets if the relevance is less than tau
                if (rt[x] != rt[p]) & (vecs[rt[x]]-vecs[ix] < tau):
                    tomerge, todel = rt[p], rt[x]
                    if vecs[tomerge] < vecs[todel]:  # We merge into the component with the higher density function value
                        tomerge, todel = todel, tomerge
                    for y in uf[todel]:
                        rt[y] = tomerge
                    uf[tomerge] = np.append(uf[tomerge], uf[todel])
                    uf.pop(todel)
                    deaths[todel] = vecs[ix]
            phv = rt[p] # Finding the parent with the highest vec value
            for x in s:
                if vecs[rt[x]] > vecs[phv]:
                    phv = rt[x]
                elif (vecs[rt[x]] == vecs[phv]) & (rt[x]<phv):
                    phv = rt[x]
            if (phv != rt[p]) & (vecs[rt[p]]-vecs[ix] < tau):
                crid = rt[p] # Current root id
                for y in v[crid]:
                    rt[y] = phv
                uf[phv] = np.append(uf[phv], uf[crid])
                uf.pop(crid)
                deaths[crid] = vecs[ix]

    # Setting the death values to the components that have been born but had not died to 0 (equivalent to infinity)
    for ix in list(births.keys()):
        if ix not in list(deaths.keys()):
            deaths[ix] = 0

    return uf, births, deaths


uf, births, deaths = UnionFind( rad=rad, tau = np.inf)

# Forming persistence classes
persistc = np.array([[births[i], deaths[i]] for i in list(births.keys())])

plt.scatter(persistc[:,0], persistc[:,1], color = 'black')
plt.plot([0, max(persistc[:,0])], [0, max(persistc[:,0])], 'r--')
plt.title("Persistence diagram")
plt.xlabel("Birth")
plt.ylabel("Death")
plt.show()

# Finding the best merging parameter from the graph
tau = 0.001

uf, births, deaths = UnionFind( rad, tau)


# Clustering
cls = np.array([i for i in list(deaths.keys()) if deaths[i] == 0]) # Cluster indexes
n_cl = len(cls) # Number of clusters

clc = [len(uf[i]) + np.sum([len(groups[pts_idx[j]]) for j in uf[i]]) for i in cls] # Cluster cardinality list

# Determining the optimal treshhold value
plt.scatter(np.arange(n_cl), clc)
plt.title("Cluster cardinalities")
plt.xlabel("Cluster no.")
plt.ylabel("Cluster cardinality")
plt.show()

trshold = 50

# Coloring the clusters
colors = [] # Generating colors
for i in range(n_cl):
    if clc[i] > trshold:
        np.random.seed(i+123)
        colors.append(list(np.random.choice(range(256), size=3,)/255))
    else:
        colors.append([0, 0, 0])
clcolors = np.zeros((len(vec),3))
for i in range(n_cl):
    for j in uf[cls[i]]:
        clcolors[pts_idx[j]] = colors[i]
        for k in groups[pts_idx[j]]:
            clcolors[k] = colors[i]

# Final result
img_result = np.reshape(clcolors, img_luv.shape)
plt.title('Final result')
plt.imshow(img_result)
plt.show()


