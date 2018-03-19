from scipy.optimize import fmin_cg
import numpy as np
from Transformation import rot_mat, grad_rot_mat
from Point_cloud import Point_cloud

def best_transform(data, ref, method = "point2point", indexes_d = None, indexes_r = None, verbose = True):
    """
    Returns the best transformation computed for the two aligned point clouds

    params:
        data: point cloud to align (shape n*3)
        ref: reference point cloud (shape n*3)
        method: must be one of : point2point, point2plane, plane2plane
        indexes_d: integer array Indexes and order to take into account in data
        indexes_r: integer array Indexes and order to take into account in ref
        verbose: Whether to plot the result of the iterations of conjugate gradient in plane2plane

    Returns:
        R: a rotation matrix (shape 3*3)
        t: translation (length 3 vector)
    """

    if indexes_d is None:
        indexes_d = np.arange(data.shape[0])
    if indexes_r is None:
        indexes_r = np.arange(ref.shape[0])

    assert(indexes_d.shape == indexes_r.shape)
    n = indexes_d.shape[0]
    if method == "point2point":
        x0 = np.zeros(6)
        M = np.array([np.eye(3) for i in range(n)])
        f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
        df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

        x = fmin_cg(f = f,x0 = x0,fprime = df, disp = False)

    elif method == "point2plane":
        x0 = np.zeros(6)
        M = ref.get_projection_matrix_point2plane(indexes = indexes_r)
        f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
        df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

        x = fmin_cg(f = f,x0 = x0,fprime = df, disp = False)

    elif method == "plane2plane":
        cov_data = data.get_covariance_matrices_plane2plane(indexes = indexes_d)
        cov_ref = ref.get_covariance_matrices_plane2plane(indexes = indexes_r, epsilon = 0.01)

        last_min = np.inf
        cpt = 0
        n_iter_max = 50
        x = np.zeros(6)
        tol = 1e-6
        while True:
            cpt = cpt+1
            R = rot_mat(x[3:])
            M = np.array([np.linalg.inv(cov_ref[i] + R @ cov_data[i] @ R.T) for i in range(n)])

            f = lambda x: loss(x,data.points[indexes_d],ref.points[indexes_r],M)
            df = lambda x: grad_loss(x,data.points[indexes_d],ref.points[indexes_r],M)

            out = fmin_cg(f = f, x0 = x, fprime = df, disp = False, full_output = True)

            x = out[0]
            f_min = out[1]
            if verbose:
                print("\t\t EM style iteration {} with loss {}".format(cpt,f_min))

            if last_min - f_min < tol:
                if verbose:
                    print("\t\t\t Stopped EM because not enough improvement or not at all")
                break
            elif cpt >= n_iter_max:
                if verbose:
                    print("\t\t\t Stopped EM because maximum number of iterations reached")
                break
            else:
                last_min = f_min

    else:
        print("Error, unknown method : {}".format(method))
        return

    t = x[0:3]
    R = x[3:]

    return rot_mat(R),t

def loss(x,a,b,M):
    """
    loss for parameter x

    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)

    returns:
        Value of the loss function
    """
    t = x[:3]
    R = rot_mat(x[3:])
    residual = b - a @ R.T -t[None,:] # shape n*d
    tmp = np.sum(M * residual[:,None,:], axis = 2) # shape n*d
    return np.sum(residual * tmp)

def grad_loss(x,a,b,M):
    """
    Gradient of the loss loss for parameter x

    params:
        x : length 6 vector of transformation parameters
            (t_x,t_y,t_z, theta_x, theta_y, theta_z)
        a : data to align n*3
        b : ref point cloud n*3 a[i] is the nearest neibhor of Rb[i]+t
        M : central matrix for each data point n*3*3 (cf loss equation)

    returns:
        Value of the gradient of the loss function
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

def ICP(data,ref,method, exclusion_radius = 0.5, sampling_limit = None, verbose = True):
    """
    Full algorithm
    Aligns the two point cloud by iteratively matching the closest points
    params:
        data: point cloud to align (shape N*3)
        ref:
        method: one of point2point, point2plane, plane2plane
        exclusion_radius: threshold to discard pairs of point with too high distance
        sampling_limit: number of point to consider for huge point clouds
        verbose: whether to plot the results of the iterations and verbose of intermediate functions

    returns:
        R: rotation matrix (shape 3*3)
        T: translation (length 3)
        rms_list: list of rms at the end of each ICP iteration
    """

    data_aligned = Point_cloud()
    data_aligned.init_from_transfo(data)

    rms_list = []
    cpt = 0
    max_iter = 50
    dist_threshold = exclusion_radius
    RMS_threshold = 1e-4
    diff_thresh = 1e-3
    rms = np.inf
    while(True):
        if sampling_limit is None:
            samples = np.arange(data.n)
        else:
            samples = np.random.choice(data.n,size = sampling_limit,replace = False)

        dist,neighbors = ref.kdtree.query(data_aligned.points[samples], return_distance = True)

        dist = dist.flatten()
        neighbors = neighbors.flatten()

        indexes_d = samples[dist < dist_threshold]
        indexes_r = neighbors[dist < dist_threshold]

        R, T = best_transform(data, ref, method, indexes_d, indexes_r, verbose = verbose)
        data_aligned.init_from_transfo(data, R,T)
        new_rms = np.sqrt(np.mean(np.sum((data_aligned.points[samples]-ref.points[neighbors])**2,axis = 0)))
        rms_list.append(new_rms)
        if verbose:
            print("Iteration {} of ICP complete with RMS : {}".format(cpt+1,new_rms))

        if new_rms < RMS_threshold :
            if verbose:
                print("\t Stopped because very low rms")
            break
        elif rms - new_rms < 0:

            if verbose:
                print("\t Stopped because increasing rms")
            break
        elif rms-new_rms < diff_thresh:
            if verbose:
                print("\t Stopped because convergence of the rms")
            break
        elif cpt >= max_iter:
            if verbose:
                print("\t Max iter reached")
            break
        else:
            rms = new_rms
            cpt = cpt+1

    return R,T, rms_list
