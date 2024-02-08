# Application of the ToMATo algorithm on a 2D set of data

# Generating 2D set
'''
import random
with open("points2d.csv", "w") as f:
    f.write('x,y\n')
    for i in range(50):
        f.write(str(random.uniform(0,100))+","+str(random.uniform(0,100))+"\n")
    for i in range(200):
        f.write(str(random.uniform(10, 30)) + "," + str(random.uniform(10, 40)) + "\n")
    for i in range(200):
        f.write(str(random.uniform(90, 95)) + "," + str(random.uniform(50, 100)) + "\n")
    for i in range(250):
        f.write(str(random.uniform(40, 60)) + "," + str(random.uniform(40, 60)) + "\n")
    for i in range(200):
        f.write(str(random.uniform(0, 5)) + "," + str(random.uniform(0, 5)) + "\n")
    for i in range(300):
        f.write(str(random.uniform(40, 95)) + "," + str(random.uniform(90, 95)) + "\n")
    for i in range(100):
        f.write(str(random.uniform(10, 30)) + "," + str(random.uniform(70, 90)) + "\n")
'''

# Generating second 2D set
'''
import random
with open("points2d2.csv", "w") as f:
    f.write('x,y\n')
    for i in range(50):
        f.write(str(random.uniform(0,10))+","+str(random.uniform(0,10))+"\n")
    for i in range(50):
        f.write(str(random.uniform(90,100))+","+str(random.uniform(90,100))+"\n")
    for i in range(50):
        f.write(str(random.uniform(0,10))+","+str(random.uniform(90,100))+"\n")
    for i in range(50):
        f.write(str(random.uniform(90,100))+","+str(random.uniform(0,10))+"\n")
    f.write("50,50\n")
    f.write("50,0\n")
    f.write("0,50\n")
    f.write("100,50\n")
    f.write("50,100\n")
'''

# Genrating third 2D set
'''
from sklearn import datasets
import numpy as np

X, y = datasets.make_blobs(n_samples=3000, random_state=170)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
px = aniso[0]
#py = aniso[1]
with open("points2d3.csv","w") as f:
    f.write("x,y\n")
    for i in range(len(px)):
        f.write(str(px[i][0])+","+str(px[i][1])+"\n")
'''

# Generating fourth 2D set
'''
import numpy as np

n_samples = 1500
theta = np.sqrt(np.random.rand(n_samples))*2*np.pi # np.linspace(0,2*pi,100)
r_a = 2.5*theta + np.pi
data_a = np.array([np.cos(theta)*r_a, np.sin(theta)*r_a]).T
x_a = data_a + np.random.randn(n_samples,2)
r_b = -2.5*theta - np.pi
data_b = np.array([np.cos(theta)*r_b, np.sin(theta)*r_b]).T
x_b = data_b + np.random.randn(n_samples,2)
res_a = np.append(x_a, np.zeros((n_samples,1)), axis=1)
res_b = np.append(x_b, np.ones((n_samples,1)), axis=1)
res = np.append(res_a, res_b, axis=0)
np.random.shuffle(res)
px = res[:,:2]

with open("points2d4.csv","w") as f:
    f.write("x,y\n")
    for i in range(len(px[:,0])):
        f.write(str(px[i][0])+","+str(px[i][1])+"\n")
'''

# Reading and visualizing data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pts = pd.read_csv('points2d4.csv').to_numpy( dtype = float )
plt.scatter(pts[:,0],pts[:,1], s = 0.8)
plt.show()

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

fig = plt.figure()
ax = plt.axes(projection ='3d')
ax.scatter(pts[:,0], pts[:,1], vec)
plt.show()

# Finding the best value for the radius parameter in the Rips graph
from sklearn.neighbors import KDTree

kdt = KDTree(pts, metric='euclidean')
rads = np.linspace(0, 10, 20)
# Average number of neighbors depending on the radius
cnbs = [np.mean([kdt.query_radius([pts[i]], rads[j], return_distance=False)[0].shape[0] for i in np.arange(pts.shape[0])]) for j in range(len(rads))]
plt.plot(rads, cnbs)
plt.show()

# Using the elbow method we use the value 0.6 as the best one
rad =1

# Sorted values of estimated density values and their indexes
ids = [i for (e,i) in sorted([(e,i) for i,e in enumerate(vec)])]
vecs = {i:e for (e,i) in sorted([(e,i) for i,e in enumerate(vec)])}

with open("vecs.txt", "w") as f:
    duz = len(vec)
    for i in range(duz):
        f.write(str(ids[i])+":"+str(vecs[ids[i]])+"\n")

'''
ind = True
for i in range(len(vec)):
    if vec[ids[i]] != vecs[ids[i]]:
        print('Pretpostavka o sortiranju nije tacna')
        ind = False
        break
if ind:
    print("Pretpostavka o sortiranju je tacna")
'''
# Pokusaj upotrebe gudhi biblioteke
# Deluje da ne moze da pomogne. Provera sledi.
'''
import gudhi as gd

rc = gd.RipsComplex(points = pts, max_edge_length = rad)
st = rc.create_simplex_tree( max_dimension = 1 )

pers = st.persistence()

perc = []
#for i in range(len(pers)):
#    if pers[i][0] == 0:
#        perc.append(pers[i][1])
perc = np.array([pers[i][1] for i in range(len(pers)) if pers[i][0] == 0])
plt.scatter(perc[:,0], perc[:,1])
plt.show()

for i in range(len(pts)):
    st.assign_filtration([i], filtration = vec[i])
st.make_filtration_non_decreasing()
pers = st.persistence()
perc = np.array([pers[i][1] for i in range(len(pers)) if pers[i][0] == 0])

plt.scatter(perc[:,0], perc[:,1])
plt.show()
'''
# Izbacuje nesto jako cudno. Vredi pogledati.

# Disjoint set structure

n = len(vec)
v = {}
births = {}
deaths = {}


# Finding all the persistence classes

def UnionFind( vecs, rad, tau ):
    v = {}
    births = {}
    deaths = {}
    rt = np.array([-1] * n)
    for i in reversed(range(n)):
        ix = ids[i]
        nbs = kdt.query_radius([pts[ix]], rad, return_distance=False)[0]
        s = [x for x in nbs if x in ids[i + 1:]]
        if not s:
            v[ix] = np.array([ix])  # Adding new component
            births[ix] = vecs[ix] # Adding new class
            rt[ix] = ix  # setting the root of the component
        else:
            p = s[np.asarray([vecs[j] for j in s]).argmax()] # Finding the parent
            #print(str(p)+' '+str(rt[p]))
            rt[ix] = rt[p] # Setting the root
            v[rt[p]] = np.append(v[rt[p]], [ix])
            for x in s: # Merging sets if the relevance is less than tau
                if (rt[x] != rt[p]) & (vecs[rt[x]]-vecs[ix] < tau):
                    if rt[x] != rt[p]:
                        #crid = rt[x] # Current root id
                        #if vecs[rt[x]]-vecs[ix] < tau:
                        tomerge, todel = rt[p], rt[x]
                        if vecs[tomerge] < vecs[todel]:
                            tomerge, todel = todel, tomerge
                        for y in v[todel]:
                            rt[y] = tomerge
                        v[tomerge] = np.append(v[tomerge], v[todel])
                        v.pop(todel)
                        deaths[todel] = vecs[ix]
                    #else:
                    #    deaths[rt[x]] = vecs[ix]
            phv = rt[p] # Finding the parent with the highest vec value
            for x in s:
                if vecs[rt[x]] > vecs[phv]:
                    phv = rt[x]
                else:
                    if (vecs[rt[x]] == vecs[phv]) & (rt[x]<phv):
                        phv = rt[x]

            if phv == ix:
                print('Greska!!!')
                print('ix: '+str(ix)+" : vecs"+str(vecs[ix]))
                for k in s:
                    print('nbs: '+str(k)+", rt: "+str(rt[k])+', vecs: '+str(vecs[rt[k]]))
                print('==============================')
            if (phv != rt[p]) & (vecs[rt[p]]-vecs[ix] < tau):
                #print('Obrisan: ' + str(rt[p]) + ' root phv: ' + str(phv))
                crid = rt[p] # Remember root
                if crid != phv:
                    for y in v[crid]:
                        rt[y] = phv
                    v[phv] = np.append(v[phv], v[crid])
                    v.pop(crid)
                    deaths[crid] = vecs[ix]

    return v, births, deaths

def define_modes(vecs, rad):
    v = {}
    n = len(vec)
    rt = np.array([-1] * n)
    for i in reversed(range(n)):
        ix = ids[i]
        nbs = kdt.query_radius([pts[ix]], rad, return_distance=False)[0]
        S = [x for x in nbs if x in ids[i + 1:]]
        if not S:
            v[ix] = np.array([ix])  # Adding new component
            births[ix] = vecs[ix]  # Adding new class
            rt[ix] = ix  # setting the root of the component
        else:
            # p = S[np.asarray([vecs[j] for j in S]).argmax()] # Finding the parent
            p = ix
            for j in S:
                if vecs[p] < vecs[j]:
                    p = j
            rt[ix] = rt[p]  # Setting the root
            v[rt[p]] = np.append(v[rt[p]], [ix])
    return v

v, births, deaths = UnionFind(vecs = vecs, rad=rad, tau = np.inf)
#v = define_modes(vecs = vecs, rad=0.2)

for ix in list(births.keys()):
    if ix not in list(deaths.keys()):
        deaths[ix] = 0

persistc = np.array([[births[i], deaths[i]] for i in list(births.keys())])

plt.scatter(persistc[:,0], persistc[:,1], color = 'black')
plt.plot([0, max(persistc[:,0])], [0, max(persistc[:,0])], 'r--')
plt.show()

# Finding the best merging parameter from the graph
tau = 0.002

#print('=========================================================')
v, births, deaths = UnionFind(vecs, rad, tau)

for ix in list(births.keys()):
    if ix not in list(deaths.keys()):
        deaths[ix] = 0

cls = np.array([i for i in list(deaths.keys()) if deaths[i] == 0]) # Cluster indexes
n_cl = len(cls) # Number of clusters
clc = [len(v[i]) for i in cls] # Cluster cardinality list

print('Number of clusters:'+str(n_cl))
print('Cluster cardinalities')
for i in range(n_cl):
    print(clc[i])

# Determining the optimal treshhold value
plt.scatter(np.arange(n_cl), clc)
plt.show()

trshold = 10

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

plt.scatter(pts[:,0], pts[:,1], c = clcolors)
plt.show()



