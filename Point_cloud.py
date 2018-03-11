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

    def init_from_transfo(self, initial, R,t):
        """
        Initialize a point cloud from another point cloud and a tranformation
        """
        self.points = initial.points @ R.T + t
        self._init()

    def _init(self):
        self.kdtree = KDTree(self.points)
        self.n  = self.points.shape[0]

    def neighborhood_PCA(self):
        """
        Returns the eigenvalues, eigenvectors for each points on his neighbors
        """
        if self.nn is None:
            self.nn = self.kdtree.query(self.points,k = 20, return_distance = False)

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

    def get_projection_matrix_point2plane(self):
        print("Computing projection matrices")
        all_eigenvectors = self.get_eigenvectors()
        normals = all_eigenvectors[:,:,0]
        normals = normals / np.linalg.norm(normals, axis = 1, keepdims = True)
        return np.array([n[:,None]*n[None,:] for n in normals])

    def get_covariance_matrices_plane2plane(self, epsilon  = 1e-3):
        """
        Returns C_A covariance matrix used
        """
        print("Computing covariance matrices for each point")
        d = 3
        cov_mat = np.zeros((self.n,d,d))
        all_eigenvectors = self.get_eigenvectors()
        dz_cov_mat = np.eye(d)
        dz_cov_mat[0,0] = epsilon
        for i in range(self.n):
            U = all_eigenvectors[i]
            cov_mat[i,:,:] = U @ dz_cov_mat @ U.T

        print("Done")
        return cov_mat
