import numpy as np
from functions import *
from Transformation import *
from Point_cloud import Point_cloud
from Generalized_ICP import *

if __name__ == '__main__':

    if False:
        theta = np.array([2,-1,0.5])
        print(rot_mat(theta))

        grm = grad_rot_mat(theta)

        epsilon  = 1e-8
        for i in range(3):
            print("check gradient "+str(i))
            epsilon_vec = np.zeros(3)
            epsilon_vec[i] = epsilon
            print(grm[i])
            print((rot_mat(theta + epsilon_vec) - rot_mat(theta - epsilon_vec))/2/epsilon)


    if True:
        # Cloud paths
        NDDC_1_path = './data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = './data/Notre_Dame_Des_Champs_2.ply'


        data = Point_cloud()
        data.init_from_ply(NDDC_1_path)

        ref = Point_cloud()
        ref.init_from_ply(NDDC_2_path)
        R, T = ICP(data,ref, method = "point2plane", sampling_limit = 50000)
        bunny_trans = Point_cloud()
        bunny_trans.init_from_transfo(data, R ,T)
        bunny_trans.save('./bunny_transformed.ply')
