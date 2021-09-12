# %% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state
    R = q.as_matrix()
    new_p = p + v * dt + 1 / 2 * (R @ (a_m - a_b) + g) * dt ** 2
    new_v = v + (R @ (a_m - a_b) + g) * dt
    new_q = q * Rotation.from_rotvec(((w_m - w_b) * dt).reshape(1, 3))

    # YOUR CODE HERE

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    R = np.array(q.as_matrix())
    F_x = np.zeros((18, 18))
    I = np.identity(3)
    a_difference = (a_m - a_b).flatten()
    a_difference_matrix = np.array([[0, -a_difference[2], a_difference[1]],
                                    [a_difference[2], 0, -a_difference[0]],
                                    [-a_difference[1], a_difference[0], 0]])

    F_x[0:3, 0:3] = I
    F_x[0:3, 3:6] = I * dt
    F_x[3:6, 3:6] = I
    F_x[3:6, 6:9] = -q.as_matrix() @ a_difference_matrix * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[3:6, 15:18] = I * dt
    F_x[6:9, 6:9] = Rotation.from_rotvec(((w_m - w_b) * dt).flatten()).as_matrix().T
    F_x[6:9, 12:15] = -I * dt
    F_x[9:12, 9:12] = I
    F_x[12:15, 12:15] = I
    F_x[15:18, 15:18] = I

    F_i = np.zeros((18, 12))
    F_i[3:6, 0:3] = I
    F_i[6:9, 3:6] = I
    F_i[9:12, 6:9] = I
    F_i[12:15, 9:12] = I

    V_i = accelerometer_noise_density ** 2 * dt ** 2 * I
    theta_i = gyroscope_noise_density ** 2 * dt ** 2 * I
    A_i = accelerometer_random_walk ** 2 * dt ** 2 * I
    omega_i = gyroscope_random_walk ** 2 * dt ** 2 * I
    Q_i = np.zeros((12, 12))
    Q_i[0:3, 0:3] = V_i
    Q_i[3:6, 3:6] = theta_i
    Q_i[6:9, 6:9] = A_i
    Q_i[9:12, 9:12] = omega_i

    new_P = F_x @ error_state_covariance @ F_x.T + F_i @ Q_i @ F_i.T

    # YOUR CODE HERE

    return new_P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    R = np.array(q.as_matrix().reshape(3, 3))
    P_c_0 = (R.T @ ((Pw - p).reshape(3, 1)))
    P_c_0_flatten = P_c_0.flatten()
    X_c = P_c_0_flatten[0]
    Y_c = P_c_0_flatten[1]
    Z_c = P_c_0_flatten[2]
    u = X_c / Z_c
    v_ = Y_c / Z_c
    innovation = uv - np.array([[u], [v_]])

    if np.linalg.norm(innovation) >= error_threshold:  # judge inlier
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    dz_t_dP_c = np.array(1 / Z_c * np.array([[1, 0, -u], [0, 1, -v_]]))
    dP_c_ddtheta = np.array([[0, -Z_c, Y_c],
                             [Z_c, 0, -X_c],
                             [-Y_c, X_c, 0]])
    dP_c_ddp = -1 * R.T
    dz_t_ddtheta = dz_t_dP_c @ dP_c_ddtheta
    dz_t_ddp = dz_t_dP_c @ dP_c_ddp

    H_t = np.zeros((2, 18))
    H_t[0:2, 0:3] = dz_t_ddp
    H_t[0:2, 6:9] = dz_t_ddtheta

    K_t = error_state_covariance @ H_t.T @ np.linalg.inv(H_t @ error_state_covariance @ H_t.T + Q)
    dx = K_t @ innovation  # update nominal state

    p = p + dx[0:3]
    v = v + dx[3:6]
    q_euler = q.as_euler('xyz') + dx[6:9].flatten()
    q = Rotation.from_euler('xyz', q_euler)
    a_b = a_b + dx[9:12]
    w_b = w_b + dx[12:15]
    g = g + dx[15:18]

    I = np.identity(18)

    error_state_covariance = (I - K_t @ H_t) @ error_state_covariance @ (I - K_t @ H_t).T + K_t @ Q @ K_t.T

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

