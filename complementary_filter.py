# %% Imports

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %%

def complementary_filter_update(initial_rotation, angular_velocity, linear_acceleration, dt):
    """
    Implements a complementary filter update

    :param initial_rotation: rotation_estimate at start of update
    :param angular_velocity: angular velocity vector at start of interval in radians per second
    :param linear_acceleration: linear acceleration vector at end of interval in meters per second squared
    :param dt: duration of interval in seconds
    :return: final_rotation - rotation estimate after update
    """

    #TODO Your code here - replace the return value with one you compute
    # print(angular_velocity)
    # print(dt)

    R_12 = Rotation.from_rotvec(angular_velocity*dt)
    R_1k = initial_rotation.as_matrix() @ R_12.as_matrix()  #rotation esti
    
    g_prime = R_1k @ linear_acceleration/9.8
    g_prime_norm = g_prime/np.linalg.norm(g_prime) #g_prime close to ex

    g_x = g_prime_norm[0]
    g_y = g_prime_norm[1]
    g_z = g_prime_norm[2]

    delta_q_acc = np.array([0,g_z/np.sqrt(2*(g_x+1)),-g_y/np.sqrt(2*(g_x+1)),np.sqrt((g_x+1)/2)])

    e_m = np.abs(np.linalg.norm(linear_acceleration/9.8)-1)
    alpha = 0
    if e_m<0.1:
        alpha = 1
    elif e_m >= 0.1 and e_m < 0.2:
        alpha = 1-(e_m-0.1)*10

    q_I = np.array([0,0,0,1])  #(x, y, z, w) as opposed to (w, x, y,z)
    delta_q_acc_prime = (1-alpha)*q_I+alpha*delta_q_acc
    delta_q_acc_prime_norm = delta_q_acc_prime/np.linalg.norm(delta_q_acc_prime)

    R = Rotation.from_quat(delta_q_acc_prime_norm).as_matrix() @ R_1k
    # print(R)
    R = Rotation.from_matrix(R)

    return R
