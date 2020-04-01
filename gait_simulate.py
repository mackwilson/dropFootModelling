"""
Model of gait including the foot, shank, and upper leg segments, the knee and hip joints,
and two muscles that create moments about the ankles, tibialis anterior and soleus.

BME 355 Project code modified from Assignment 2 code.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from musculoskeletal import HillTypeMuscle, get_velocity_single_val, force_length_tendon_single_val


def soleus_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: soleus length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(theta):
    """
    :param theta: body angle (up from prone horizontal)
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment(theta):
    """
    :param theta: angle of body segment (up from prone)
    :return: moment about ankle due to force of gravity on body
    """
    mass = 75 # body mass (kg; excluding feet)
    centre_of_mass_distance = 1 # distance from ankle to body segment centre of mass (m)
    g = 9.81 # acceleration of gravity
    return mass * g * centre_of_mass_distance * np.sin(theta - np.pi / 2)


def dynamics(x, soleus, tibialis, control):
    """
    :param x: state vector (ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length)
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector
    """
    '''
    x1 = ankle angle = x[0]
    x2 = angular velocity = x[1]
    x3 = soleus norm CE length = x[2]
    x4 = TA norm CE length = x[3]
    '''
    # constants
    I_ankle = 90 
    f_max_soleus = 16000
    f_max_tibant = 2000
    tibant_activation = 0.4
    soleus_activation = 0.05
    soleus_moment_arm = .05 # values from class notes, and simulate() (here)
    tibant_moment_arm = .03 # values from class notes 

    # calculated
    length_soleus = soleus_length(x[0])
    length_tibant = tibialis_length(x[0])
    norm_tendon_length_soleus = soleus.norm_tendon_length(length_soleus, x[2])
    norm_tendon_length_tibant = tibialis.norm_tendon_length(length_tibant, x[3])
    torque_soleus = force_length_tendon_single_val(norm_tendon_length_soleus) * f_max_soleus * soleus_moment_arm
    torque_tibant = force_length_tendon_single_val(norm_tendon_length_tibant) * f_max_tibant * tibant_moment_arm
    external_moment = 0
    if control: 
        if x[0] > np.pi/2: 
            soleus_activation = 0
            tibant_activation = 0.4 * (x[0]- np.pi/2)
        else:
            tibant_activation = 0
            soleus_activation = 0.05 * (np.pi/2 - x[0])
        d_ext = 0.5
        f_ext = 10000*(x[0] - np.pi/2)
        external_moment = f_ext * d_ext * np.cos(x[0] - np.pi/2)

    # derivative of state: 
    xd = [0, 0, 0, 0]
    xd[0] = x[1]
    xd[1] = (torque_soleus - torque_tibant - external_moment + gravity_moment(x[0])) / I_ankle 
    xd[2] = get_velocity_single_val(soleus_activation, x[2], norm_tendon_length_soleus)
    xd[3] = get_velocity_single_val(tibant_activation, x[3], norm_tendon_length_tibant)
    return xd


def simulate(control, T):
    """
    Runs a simulation of the model and plots results.
    :param control: True if balance should be controlled
    :param T: total time to simulate, in seconds
    """
    rest_length_soleus = soleus_length(np.pi/2)
    rest_length_tibialis = tibialis_length(np.pi/2)

    soleus = HillTypeMuscle(16000, .6*rest_length_soleus, .4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, soleus, tibialis, control)

    sol = solve_ivp(f, [0, T], [np.pi/2-0.001, 0, 1, 1], rtol=1e-5, atol=1e-8)
    time = sol.t
    theta = sol.y[0,:]
    soleus_norm_length_muscle = sol.y[2,:]
    tibialis_norm_length_muscle = sol.y[3,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    for th, ls, lt in zip(theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(soleus_moment_arm * soleus.get_force_single_val(soleus_length(th), ls))
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force_single_val(tibialis_length(th), lt))

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(time, sol.y[0,:])
    plt.ylabel('Body angle (rad)')
    plt.ylim(1.4, 1.8)
    plt.subplot(2,1,2)
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.ylim(-3, 4)
    plt.tight_layout()
    plt.savefig("fig.png")


