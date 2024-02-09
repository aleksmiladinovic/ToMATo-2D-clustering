# Application of the ToMATo algorithm on a 2D set of data

# There are comments next to some variables which suggest which values should be taken
# depending on which test case we are processing, eg. Case1: rad=2, Case2: rad=0.2, Case3: rad=1
# meaning that we should take rad=2 when processing the first test case, rad=0.2 when dealing with the second
# and rad=1 when exercising the third test case

# Reading and visualizing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pts = pd.read_csv('points2d.csv').to_numpy( dtype = float ) # Here we can type points2d1, points2d2 or points2d3 depending on the test case we want
plt.scatter(pts[:,0],pts[:,1], s = 0.8)
plt.title("Set of points")
plt.show()

# Finding the best parameters for density estimation

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

bdws = np.linspace(0.1, 2, 20)
grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bdws}, cv=10)
grid.fit(pts)
bw = grid.best_params_['bandwidth']
kde = grid.best_estimator_
vec = np.exp(kde.score_samples(pts))

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(pts[:,0], pts[:,1], vec)
plt.title("Estimated density function")
plt.show()

# Finding the best value for the radius parameter in the Rips graph

from sklearn.neighbors import KDTree

kdt = KDTree(pts, metric='euclidean')
rads = np.linspace(0, 10, 20)     # Case1: (0,10,20), Case2: (0,1,20), Case3: (0,5,20)
# Average number of neighbors depending on the radius
cnbs = [np.mean([kdt.query_radius([pts[i]], rads[j], return_distance=False)[0].shape[0] for i in np.arange(pts.shape[0])]) for j in range(len(rads))]
plt.plot(rads, cnbs)
plt.title("Average number of neighbors vs. the radius")
plt.xlabel("Radius")
plt.ylabel("Average number of neighbors")
plt.show()

# Using the elbow method we determine the optimal radius value
rad = 2     # Case1: rad=2, Case2: rad=0.2, Case3: rad=1

# Sorted values of estimated density values and their indexes
ids = [i for (e,i) in sorted([(e,i) for i,e in enumerate(vec)])]
vecs = {i:e for (e,i) in sorted([(e,i) for i,e in enumerate(vec)])}

# Disjoint set structure

n = len(vec)
v = {}
births = {}
deaths = {}


# Finding clusters

def FindClusters( vecs, rad, tau ):
    v = {}
    births = {}
    deaths = {}
    rt = np.array([-1] * n)
    for i in reversed(range(n)):
        ix = ids[i]
        nbs = kdt.query_radius([pts[ix]], rad, return_distance=False)[0]
        s = [x for x in nbs if x in ids[i + 1:]] # Neighbors with the greater density function value
        if not s:
            v[ix] = np.array([ix])  # Adding new component
            births[ix] = vecs[ix] # Adding new class
            rt[ix] = ix  # setting the root of the component
        else:
            p = s[np.asarray([vecs[j] for j in s]).argmax()] # Finding the parent
            rt[ix] = rt[p] # Setting the root
            v[rt[p]] = np.append(v[rt[p]], [ix])
            for x in s: # Merging sets if the relevance is less than tau
                if (rt[x] != rt[p]) & (vecs[rt[x]]-vecs[ix] < tau):
                    tomerge, todel = rt[p], rt[x]
                    if vecs[tomerge] < vecs[todel]:  # We merge into the component with the higher density function value
                        tomerge, todel = todel, tomerge
                    for y in v[todel]:
                        rt[y] = tomerge
                    v[tomerge] = np.append(v[tomerge], v[todel])
                    v.pop(todel)
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
                v[phv] = np.append(v[phv], v[crid])
                v.pop(crid)
                deaths[crid] = vecs[ix]

    # Setting the death values to the components that have been born but had not died to 0 (equivalent to infinity)
    for ix in list(births.keys()):
        if ix not in list(deaths.keys()):
            deaths[ix] = 0

    return v, births, deaths

# Running the algorithm with tau equal to infinity is equivalent to finding all the persistence classes
v, births, deaths = UnionFind(vecs = vecs, rad=rad, tau = np.inf)

# Forming persistence classes
persistc = np.array([[births[i], deaths[i]] for i in list(births.keys())])

plt.scatter(persistc[:,0], persistc[:,1], color = 'black')
plt.plot([0, max(persistc[:,0])], [0, max(persistc[:,0])], 'r--')
plt.title("Persistence diagram")
plt.xlabel("Birth")
plt.ylabel("Death")
plt.show()

# Finding the best merging parameter from the graph
tau = 0.004     # Case1: tau=0.004, Case2: tau=0.2, Case3: tau=0.0035

# Running the algorithm again for the parameter tau
v, births, deaths = UnionFind(vecs, rad, tau)

# Forming clusters
cls = np.array([i for i in list(deaths.keys()) if deaths[i] == 0]) # Cluster indexes
n_cl = len(cls) # Number of clusters
clc = [len(v[i]) for i in cls] # Cluster cardinality list

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
    for j in v[cls[i]]:
        clcolors[j] = colors[i]

# Final result

plt.scatter(pts[:,0], pts[:,1], c = clcolors)
plt.title("Final result")
plt.show()
