from utils.ply import read_ply, write_ply
import numpy as np
from sklearn.neighbors import KDTree

class Point_cloud:
    def __init__(self):
        """
        path : path to the ply files
        """

        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.n = 0
        self.kdtree = None
        self.points = None
        self.nn = None

    def save(self,path):
        """
        Write the point cloud to a file
        params:
            path: output path
        """
        write_ply(path,self.points,["x","y","z"])

    def init_from_ply(self,path):
        """
        Initialize point cloud from ply file
        params:
            path: input path
        """
        tmp = read_ply(path)
        self.points = np.vstack((tmp['x'],tmp['y'],tmp['z'])).T
        self._init()


    def init_from_transfo(self, initial, R = None,t = None):
        """
        Initialize a point cloud from another point cloud and a tranformation R x + t
        params:
            initial: input point cloud object
            R : rotation matrix to apply to initial
            t : transormation
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        self.points = initial.points @ R.T + t
        self._init()

    def init_from_points(self,points):
        """
        Initialize a point cloud from a list of points
        params:
            points: numpy array
        """
        self.points = points.copy()
        self._init()

    def _init(self):
        """
        Common function for every initialization function
        """
        self.kdtree = KDTree(self.points)
        self.n  = self.points.shape[0]
        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.nn = None

    def transform(self,R,T):
        """
        Apply transformation Rx+t to the point cloud
        """
        self.points = self.points @ R.T + T
        if not self.all_eigenvectors is None:
            self.all_eigenvectors = self.all_eigenvectors @ R.T

    def neighborhood_PCA(self, radius = 0.005):
        """
        Returns the eigenvalues, eigenvectors for each points on his neighbors
        The neigbours are computed with query_radius of kdtree

        params:
            radius: radius to find the list of nearest neighbors

        returns:
            tuple(eigenvalues, eigenvectors) of dimension (n,3) and (n,3,3)
        """
        if self.nn is None:
            self.nn = self.kdtree.query_radius(self.points,r = radius, return_distance = False)

        all_eigenvalues = np.zeros((self.n, 3))
        all_eigenvectors = np.zeros((self.n, 3, 3))

        for i in range(self.n):
            if len(self.nn[i]) < 3:
                all_eigenvalues[i], all_eigenvectors[i] = (np.array([0,0,0]),np.eye(3))
            else:
                all_eigenvalues[i], all_eigenvectors[i] = np.linalg.eigh(np.cov(self.points[self.nn[i]].T))

        return all_eigenvalues, all_eigenvectors

    def get_eigenvectors(self, radius = 0.005):
        """
        Returns the eigenvectors on the neighbors PCA. Used for computing it only
        once

        params:
            radius: radius to find the list of nearest neighbors NB: if the PCA have
                already been performed once, the PCA is not done again even if
                radius is different than the original
        """
        if self.all_eigenvectors is None:
            self.all_eigenvalues,self.all_eigenvectors = self.neighborhood_PCA(radius)
        return self.all_eigenvectors

    def get_projection_matrix_point2plane(self, indexes = None):
        """
        Get the projection matrix on the plane defined at each point for point2plane

        params:
            indexes: integer np array, indexes for which the projection matrix
            must be computed

        Returns:
            array of Projection matrices on each of the normals
        """
        if indexes is None:
            indexes = np.arange(self.n)

        all_eigenvectors = self.get_eigenvectors()
        normals = all_eigenvectors[:,:,0]
        normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)
        return np.array([normals[i,:,None]*normals[i,None,:] for i in indexes])

    def get_covariance_matrices_plane2plane(self, epsilon  = 1e-3,indexes = None):
        """
        Returns C_A covariance matrix used for plane2plane

        params:
            epsilon: value of the covariance in the normal direction
            indexes: integer np array, indexes for which the covariance matrix
            must be computed

        returns: array of covariance matrices for each point of indexes
        """
        if indexes is None:
            indexes = np.arange(self.n)
        d = 3
        new_n = indexes.shape[0]
        cov_mat = np.zeros((new_n,d,d))
        all_eigenvectors = self.get_eigenvectors()
        dz_cov_mat = np.eye(d)
        dz_cov_mat[0,0] = epsilon
        for i in range(new_n):
            U = all_eigenvectors[indexes[i]]
            cov_mat[i,:,:] = U @ dz_cov_mat @ U.T

        return cov_mat
