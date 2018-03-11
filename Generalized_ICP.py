from scipy.optimize import fmin_cg
import numpy as np
from Transformation import rot_mat, grad_rot_mat
from Point_cloud import Point_cloud

def ICP(data, ref, method = "point2point"):

    n = data.n
    if method == "point2point":
        x0 = np.zeros(6)
        M = np.array([np.eye(3) for i in range(n)])
        f = lambda x: loss(x,data.points,ref.points,M)
        df = lambda x: grad_loss(x,data.points,ref.points,M)

        x = fmin_cg(f = f,x0 = x0,fprime = df)

    elif method == "point2plane":
        x0 = np.zeros(6)
        M = ref.get_projection_matrix_point2plane()
        print(M.shape)
        f = lambda x: loss(x,data.points,ref.points,M)
        df = lambda x: grad_loss(x,data.points,ref.points,M)

        x = fmin_cg(f = f,x0 = x0,fprime = df)

    elif method == "plane2plane":
        cov_data = data.get_covariance_matrices_plane2plane()
        cov_ref = ref.get_covariance_matrices_plane2plane()

        last_min = np.inf
        cpt = 0
        n_iter_max = 10
        x = np.zeros(6)
        tol = 1e-6
        while cpt < n_iter_max:
            cpt = cpt+1
            R = rot_mat(x[3:])
            M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])

            f = lambda x: loss(x,data.points,ref.points,M)
            df = lambda x: grad_loss(x,data.points,ref.points,M)

            out = fmin_cg(f = f, x0 = x, fprime = df, disp = False, full_output = True)

            x = out[0]
            f_min = out[1]

            if last_min - f_min < tol:
                break
            else:
                last_min = f_min
                print("Successful iteration with loss {}".format(f_min))

    else:
        print("Error, unknown method : {}".format(method))
        return

    t = x[0:3]
    R = x[3:]

    return t, rot_mat(R)

def loss(x,a,b,M):
    """
    loss for parameter x
    a : data to align n*3
    b : ref point cloud n*3
    M : central matrix in the formal n*3*3
    """
    t = x[:3]
    R = rot_mat(x[3:])
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
    return np.sum(residual * tmp)

def grad_loss(x,a,b,M):
    """
    Gradient on x of the loss
    """
    t = x[:3]
    R = rot_mat(x[3:])
    g = np.zeros(6)
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d

    g[:3] = - 2*np.sum(tmp, axis = 0)

    grad_R = - 2* (tmp.T @ a) # shape d*d
    grad_R_euler = grad_rot_mat(x[3:]) # shape 3*d*d
    g[3:] = np.sum(grad_R[None,:,:] * grad_R_euler, axis = (1,2)) # chain rule
    return g
