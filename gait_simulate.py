"""
Model of gait including the foot, shank, and upper leg segments, the knee and hip joints,
and two muscles that create moments about the ankles, tibialis anterior and soleus.

BME 355 Project code modified from Assignment 2 code.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from muscle import HillTypeMuscle, get_velocity_single_val, force_length_tendon_single_val
from regression import tibialis_activation, soleus_activation, hip_torque, knee_torque, hip_angle, knee_angle

# GLOBAL CONSTANTS
# TODO: UPDATE THESE !
constants = {
    I_ANKLE: 90, 
    F_MAX_SOLEUS: 16000,
    F_MAX_TIBIALIS: 2000,
    MOMENT_ARM_SOLEUS: .05,
    MOMENT_ARM_TIBIALIS: .03,
    CYCLE_PERIOD: 2, # seconds 
    SHANK_LENGTH: 1,
    THIGH_LENGTH: 1,
    FOOT_LENGTH: 1, 
    D_COM_THIGH_KNEE: 0.343, # m
    D_COM_SHANK_ANKLE: 0.314,
    D_COM_FOOT_ANKLE: 0.130
}

def soleus_length(beta, theta):
    # TODO: update 
    """
    :param theta: body angle (up from prone horizontal)
    :return: soleus length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(beta, theta):
    # TODO: update 
    """
    :param theta: body angle (up from prone horizontal)
    :return: tibialis anterior length
    """
    rotation = [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def thigh_centre_mass_distance(theta, t):
    """
    :param theta: angle of foot relative to shank
    :param t: current time in seconds 
    :return distance from angle to centre of mass of thigh
    """
    l_s = constants.SHANK_LENGTH
    dcom_th = constants.D_COM_THIGH_KNEE
    return np.sqrt( 
        dcom_th**2 + l_s**2 - (2*dcom_th*l_s*np.cos(np.pi - hip_angle(t) - theta))
    )


def gravity_moment(beta, theta, t):
    """
    :param theta: angle of foot relative to shank
    :param beta: angle of shank relative to vertical 
    :param t: current time in seconds 
    :return moment about ankle due to force of gravity on body
    """
    dcom_foot = constants.D_COM_FOOT_ANKLE
    dcom_shank = constants.D_COM_SHANK_ANKLE
    dcom_leg = thigh_centre_mass_distance(beta, t)
    H = hip_angle(t)
    mass = 75 # body mass (kg; excluding feet)
    g = 9.81 # acceleration of gravity
    pass 
    # return: gravity moment on rest of body - gravity moment from foot 


def dynamics(x, t, soleus, tibialis, footDrop, fes):
    """
    :param x: state vector,[ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length]
    :param soleus: soleus muscle (HillTypeModel)
    :param tibialis: tibialis anterior muscle (HillTypeModel)
    :param control: True if balance should be controlled
    :return: derivative of state vector, xd
    """
    '''
    x1 = ankle angle (beta) = x[0]
    x2 = angular velocity of ankle = x[1]
    x3 = shank angle from vert (theta) = x[2]
    x4 = angular velocity of shank relative to vert = x[3]
    x5 = soleus norm CE length = x[4]
    x6 = TA norm CE length = x[5]
    '''
    activation_tibialis, activation_soleus = get_muscle_activations(footDrop, fes)

    # calculated
    l_soleus = soleus_length(x[0], x[2])
    l_tibialis = tibialis_length(x[0], x[2])
    norm_tendon_length_soleus = soleus.norm_tendon_length(l_soleus, x[2])
    norm_tendon_length_tibant = tibialis.norm_tendon_length(l_tibialis, x[3])
    torque_soleus = force_length_tendon_single_val(norm_tendon_length_soleus) * constants.F_MAX_SOLEUS * constants.MOMENT_ARM_SOLEUS
    torque_tibant = force_length_tendon_single_val(norm_tendon_length_tibant) * constants.F_MAX_TIBIALIS * constants.MOMENT_ARM_TIBIALIS

    # derivative of state: 
    xd = [0, 0, 0, 0, 0, 0]
    xd[0] = x[1]
    xd[1] = (torque_soleus - torque_tibant + hip_torque(t) + knee_torque(t) + gravity_moment(x[0], t)) / constants.I_ANKLE 
    xd[2] = x[3]
    xd[3] = 1 # insert equation here 
    xd[4] = get_velocity_single_val(activation_soleus, x[4], norm_tendon_length_soleus)
    xd[5] = get_velocity_single_val(activation_tibialis, x[5], norm_tendon_length_tibant)
    return xd

def get_muscle_activations(footDrop, fes):
    """
    :param footDrop: boolean flag for foot drop simulation case
    :param fes: boolean flag for FES simulation case (assumes footDrop to be true)
    :return the tibialis and soleus activations (which are not constant and are 
            from the regression model) 
    """
    # TODO: complete for all cases 
    if not footDrop and not fes:
        return tibialis_activation(), soleus_activation()

def simulate(N, footDrop=False, fes=False):
    """
    Runs a simulation of the model and plots results.
    :param N: Number of gait cycles to simulate.
    :param footDrop: boolean flag for foot drop simulation scenario
    :param fes: boolean flag for FES simulation scenario
    """
    rest_length_soleus = soleus_length(np.pi/2, 0)
    rest_length_tibialis = tibialis_length(np.pi/2, 0)

    soleus = HillTypeMuscle(16000, .6*rest_length_soleus, .4*rest_length_soleus)
    tibialis = HillTypeMuscle(2000, .6*rest_length_tibialis, .4*rest_length_tibialis)

    def f(t, x):
        return dynamics(x, soleus, tibialis, footDrop, fes)

    T = N * constants.CYCLE_PERIOD
    x0 = [np.pi/2, 0, 0, 0, 1, 1] # TODO: initial values need to be determined 
    """
    state = [shank-foot angle, angular velocity, normalized CE length of soleus, normalized CE length of tibialis]
    """
    sol = solve_ivp(f, [0, T], x0, rtol=1e-5, atol=1e-8)
    time = sol.t
    beta = sol.y[0,:]
    theta = sol.y[2,:]
    soleus_norm_length_muscle = sol.y[4,:]
    tibialis_norm_length_muscle = sol.y[5,:]

    soleus_moment_arm = .05
    tibialis_moment_arm = .03
    soleus_moment = []
    tibialis_moment = []
    toe_height = []
    ankle_height = []
    for t, b, th, ls, lt in zip(time, beta, theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
        soleus_moment.append(soleus_moment_arm * soleus.get_force_single_val(soleus_length(b, th), ls))
        tibialis_moment.append(-tibialis_moment_arm * tibialis.get_force_single_val(tibialis_length(b, th), lt))
        ankle_height.append(np.cos(hip_angle(t))*constants.SHANK_LENGTH + constants.THIGH_LENGTH*np.cos(th))
        toe_height.append() # TODO COMPLETE

    # plot results 
    plt.figure()
    plt.title("Physical angles over the simulation")
    plt.plot(time, sol.y[0,:], 'r')
    plt.plot(time, sol.y[2,:], 'g')
    plt.legend(('foot-shank', 'shank-vertical'))
    plt.ylabel('Angle (rad)')
    plt.tight_layout()
    plt.savefig("outputs/angles.png")

    plt.figure()
    plt.title("Moments from various sources over the simulation")
    plt.plot(time, soleus_moment, 'r')
    plt.plot(time, tibialis_moment, 'g')
    plt.plot(time, gravity_moment(sol.y[0,:]), 'k')
    plt.legend(('soleus', 'tibialis', 'gravity'))
    plt.xlabel('Time (s)')
    plt.ylabel('Torques (Nm)')
    plt.ylim(-3, 4)
    plt.tight_layout()
    plt.savefig("outputs/moments.png")

    plt.figure()
    plt.title("Height of body parts off the ground")
    plt.plot(time, toe_height, 'r')
    plt.plot(time, ankle_height, 'g')
    plt.legend(('toes', 'ankle'))
    plt.xlabel('Time (s)')
    plt.ylabel('Height (m)')
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig("outputs/heights.png")


