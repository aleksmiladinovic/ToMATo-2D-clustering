# ToMATo algorithm

We will demonstrate the work of the **ToMATo** (Topological Mode Analysis Tool) algorithm for clustering data.  
First, we will consider the estimated density function on the data. The mode detection is done via the graph-based hill-climbing scheme.  
The novelty of this approach lies in the use of topological persistence for determining the relevance of the modes, forming the hierarchy of the modes as well as the merging of the clusters based on this information.  
Additionally, this algorithm offers visualisation of the data, most notably in the form of persistence diagrams which depict the importance of the modes of estimated density function.  
This information allows us to choose the parameter values based on which the algorithm will give us the exact number of clusters.

The file "ToMATo2D.py" contains the code for the ToMATo algorithm for the data read from a single .csv file.  
The folders "Clustering1", "Clustering2" and "Clustering3" each contain .csv file of points as well as the results of ToMATo algorithm.

The file "ToMATo_image_segmentation.py" represents the modification of the code presented in the previous file adapted for the image segmentation.  
The folder "ImageSegmentation" contains a sample image as well as the results of the ToMATo algorithm used for image segmentation.
