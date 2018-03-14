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
        """
        write_ply(path,self.points,["x","y","z"])

    def init_from_ply(self,path):
        """
        Initialize point cloud from ply file
        """
        tmp = read_ply(path)
        self.points = np.vstack((tmp['x'],tmp['y'],tmp['z'])).T
        self._init()


    def init_from_transfo(self, initial, R = None,t = None):
        """
        Initialize a point cloud from another point cloud and a tranformation
        """
        if R is None:
            R = np.eye(3)
        if t is None:
            t = np.zeros(3)

        self.points = initial.points @ R.T + t
        self._init()

    def init_from_points(self,points):
        self.points = points.copy()
        self._init()

    def _init(self):
        self.kdtree = KDTree(self.points)
        self.n  = self.points.shape[0]
        self.all_eigenvalues = None
        self.all_eigenvectors = None
        self.nn = None

    def transform(self,R,T):
        self.points = self.points @ R.T + T
        if not self.all_eigenvectors is None:
            self.all_eigenvectors = self.all_eigenvectors @ R.T

    def neighborhood_PCA(self):
        """
        Returns the eigenvalues, eigenvectors for each points on his neighbors
        """
        if self.nn is None:
            self.nn = self.kdtree.query_radius(self.points,radius = 0.005, return_distance = False)

        all_eigenvalues = np.zeros((self.n, 3))
        all_eigenvectors = np.zeros((self.n, 3, 3))

        for i in range(self.n):
            if len(self.nn[i]) < 3:
                all_eigenvalues[i], all_eigenvectors[i] = (np.array([0,0,0]),np.eye(3))
            else:
                all_eigenvalues[i], all_eigenvectors[i] = np.linalg.eigh(np.cov(self.points[self.nn[i]].T))

        return all_eigenvalues, all_eigenvectors

    def get_eigenvectors(self):
        """
        Returns the eigenvectors on the neighbors PCA. Used for computing it only
        once
        """
        if self.all_eigenvectors is None:
            self.all_eigenvalues,self.all_eigenvectors = self.neighborhood_PCA()
        return self.all_eigenvectors

    def get_projection_matrix_point2plane(self, indexes = None):
        """
        Get the projection matrix on the plane defined at each point
        Possibility to have it only on indexes
        """
        if indexes is None:
            indexes = np.arange(self.n)

        all_eigenvectors = self.get_eigenvectors()
        normals = all_eigenvectors[:,:,0]
        normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)
        return np.array([normals[i,:,None]*normals[i,None,:] for i in indexes])

    def get_covariance_matrices_plane2plane(self, epsilon  = 1e-3,indexes = None):
        """
        Returns C_A covariance matrix used
        Possibility to have it only on indexes
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
