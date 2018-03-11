import numpy as np

def elementary_rot_mat(theta):
    """
    Returns the 3 rotations around each axe
    """


    R_x = np.array([[1,         0,                 0                ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])



    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                   1,      0                 ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])

    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                   0,                    1]
                    ])

    return R_x, R_y, R_z

def rot_mat(theta) :
    """
    Returns the Rotation matrix for the rotation parametrized with theta
    Convention X, Y, Z
    """
    R_x,R_y,R_z = elementary_rot_mat(theta)
    R = R_z @ R_y @ R_x

    return R


def grad_rot_mat(theta):
    """
    Computes the gradient of the rotation matrix w.r.t the X,Y,Z euler angles
    Returns res[i,j,k] = dR_jk/theta_i
    """
    res = np.zeros((3,3,3))

    R_x,R_y,R_z = elementary_rot_mat(theta)

    g_x = np.array([[0,         0,                  0                ],
                    [0,         -np.sin(theta[0]), -np.cos(theta[0]) ],
                    [0,         np.cos(theta[0]),  -np.sin(theta[0]) ]
                    ])

    g_y = np.array([[-np.sin(theta[1]),   0,      np.cos(theta[1])  ],
                    [0,                   0,      0                 ],
                    [-np.cos(theta[1]),   0,      -np.sin(theta[1]) ]
                    ])

    g_z = np.array([[-np.sin(theta[2]),   -np.cos(theta[2]),    0],
                    [np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [0,                   0,                    0]
                    ])

    res[0,:,:] = R_z @ R_y @ g_x
    res[1,:,:] = R_z @ g_y @ R_x
    res[2,:,:] = g_z @ R_y @ R_x

    return res
