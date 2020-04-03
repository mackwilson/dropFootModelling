"""
Model of gait including the foot, shank, and upper leg segments, the knee and hip joints,
and two muscles that create moments about the ankles, tibialis anterior and soleus.

BME 355 Project code modified from Assignment 2 code.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from muscle import HillTypeMuscle, get_velocity_single_val, force_length_tendon_single_val
import regression as r 


# ******** GLOBAL CONSTANTS ********
# TODO: UPDATE THESE ! Some are still missing 
I_ANKLE = 90
I_KNEE = 120
F_MAX_SOLEUS = 16000
F_MAX_TIBIALIS = 2000
MOMENT_ARM_SOLEUS = .05
MOMENT_ARM_TIBIALIS = .03
CYCLE_PERIOD = 1.7 # seconds 
SHANK_LENGTH = 0.553 # m
THIGH_LENGTH = 0.605 # m
FOOT_LENGTH = 0.260 # m 
HIP_HEIGHT = 1.158 # m 
D_COM_THIGH_KNEE = 0.343 # m
D_COM_SHANK_ANKLE = 0.314 # m
D_COM_FOOT_ANKLE = 0.130 # m
MASS_FOOT = 1.03 # kg
MASS_SHANK = 3.45 # kg
MASS_THIGH = 8.45 # kg
MASS_TORSO = 50.42 # kg 
GRAVITY = 9.81 # N/kg


class GaitSimulator:
    def __init__(self, footDrop=False, fes=False, Kp=0, Kd=0):
        """
        :param N: number of gait cycles to simulate for 
        :param footDrop: boolean flag for foot drop simulation
        :param fes: boolean flag for FES to assist foot drop 
        :param Kp: proportional gain for the FES
        :param Kd: derivative gain for FES control
        """
        # CLASS ATTRIBUTES
        self.footDrop = footDrop 
        self.fes = fes 
        self.Kp = Kp 
        self.Kd = Kd 

        rest_length_soleus = soleus_length(np.pi/2)
        rest_length_tibialis = tibialis_length(np.pi/2)

        self.soleus = HillTypeMuscle(F_MAX_SOLEUS, .6*rest_length_soleus, .4*rest_length_soleus)
        self.tibialis = HillTypeMuscle(F_MAX_TIBIALIS, .6*rest_length_tibialis, .4*rest_length_tibialis)

    # ******** CLASS METHODS ********
    def get_soleus_activation(self, t, beta, theta, d_beta, d_theta):
        """
        :param t: current time 
        :param beta: angle from foot to shank
        :param theta: angle from shank to vertical
        :param d_beta: derivative of beta
        :param d_theta: derivative of theta
        :return the practical soleus activation for the type of simulation
        """
        footDropScaler = 1
        appliedExcitation = 0
        if self.footDrop:
            # reduce by a certain amount
            footDropScaler = 0.1
        if self.fes:
            # add excitation
            appliedExcitation = self.get_soleus_excitation(t, beta, theta, d_beta, d_theta)

        return r.soleus_activation(t) * footDropScaler + appliedExcitation

    def get_tibialis_activation(self, t, beta, theta, d_beta, d_theta):
        """
        :param t: current time 
        :param beta: angle from foot to shank
        :param theta: angle from shank to vertical
        :param d_beta: derivative of beta
        :param d_theta: derivative of theta
        :return the practical tibialis activation for the type of simulation
        """
        footDropScaler = 1
        appliedExcitation = 0
        if self.footDrop:
            # reduce by a certain amount
            footDropScaler = 0.1
        if self.fes:
            # add excitation
            appliedExcitation = self.get_tibialis_excitation(t, beta, theta, d_beta, d_theta)

        return r.tibialis_activation(t) * footDropScaler + appliedExcitation 

    def get_soleus_excitation(self, t, beta, theta, d_beta, d_theta):
        """
        :param t: current time
        :param beta: angle from foot to shank
        :param theta: angle from shank to vertical
        :param Kp: proportional gain for excitation control
        :param Kd: integrator gain for excitation control
        :return the applied excitation for the soleus as an additional normalized activation  
        """
        # TODO: complete
        return 0.5

    def get_tibialis_excitation(self, t, beta, theta, d_beta, d_theta):
        """
        :param t: current time
        :param beta: angle from foot to shank
        :param theta: angle from shank to vertical
        :param Kp: proportional gain for excitation control
        :param Kd: integrator gain for excitation control
        :return the applied excitation for the tibialis as an additional normalized activation  
        """
        # TODO: complete
        return 0.5


    # - - - - -  DYNAMICS - - - - - - 
    def dynamics(self, x, t):
        """
        :param x: state vector [ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length]
        :param soleus: soleus muscle (HillTypeModel)
        :param tibialis: tibialis anterior muscle (HillTypeModel)
        :param control: True if balance should be controlled
        :return: derivative of state vector, xd
        """
        # state 
        beta = x[0]
        d_beta = x[1]
        theta = x[2]
        d_theta = x[3]
        l_ce_norm_soleus = x[4]
        l_ce_norm_tibialis = x[5]
        
        activation_tibialis = self.get_tibialis_activation(t, beta, theta, d_beta, d_theta)
        activation_soleus = self.get_soleus_activation(t, beta, theta, d_beta, d_theta)

        # calculated
        l_soleus = soleus_length(beta)
        l_tibialis = tibialis_length(beta)
        l_tendon_norm_soleus = self.soleus.norm_tendon_length(l_soleus, l_ce_norm_soleus)
        l_tendon_norm_tibialis = self.tibialis.norm_tendon_length(l_tibialis, l_ce_norm_tibialis)
        torque_soleus = force_length_tendon_single_val(l_tendon_norm_soleus) * F_MAX_SOLEUS * MOMENT_ARM_SOLEUS
        torque_tibant = force_length_tendon_single_val(l_tendon_norm_tibialis) * F_MAX_TIBIALIS * MOMENT_ARM_TIBIALIS
        # print(l_ce_norm_soleus)

        # derivative of state: 
        xd = [0, 0, 0, 0, 0, 0]
        xd[0] = d_beta
        xd[1] = (torque_soleus - torque_tibant + gravity_moment_ankle(beta, theta, r.hip_angle(t))) / I_ANKLE 
        xd[2] = d_theta
        xd[3] = (r.knee_torque(t) - gravity_moment_knee(beta, theta)) / I_KNEE
        xd[4] = get_velocity_single_val(activation_soleus, l_ce_norm_soleus, l_tendon_norm_soleus)
        xd[5] = get_velocity_single_val(activation_tibialis, l_ce_norm_tibialis, l_tendon_norm_tibialis)
        # print("X = {}".format(x))
        return xd


    # - - - - -  SIMULATE - - - - - - 
    def simulate(self):
        """
        Runs a simulation of the model and plots results.
        :param N: Number of gait cycles to simulate.
        :param footDrop: boolean flag for foot drop simulation scenario
        :param fes: boolean flag for FES simulation scenario
        :param Kp: proportional gain for FES control
        :param Kd: integrator gain for FES control 
        """
        print("\n\n * * * * STARTING NEW SIMULATION * * * * ")
        print("\nParameters:")
        print("Foot Drop = {}".format(self.footDrop))
        print("FES = {}".format(self.fes))
        if(self.fes):
            print("Kp = {}, Kd = {}".format(self.Kp, self.Kd))

        print("\nSimulating...")
        def f(t, x):
            return self.dynamics(x, t)

        """
        state = [
                    shank-foot angle, 
                    angular velocity for above, 
                    shank-vertical angle, 
                    angular velocity for above, 
                    normalized CE length of soleus, 
                    normalized CE length of tibialis
                ]
        """
        t0 = 0.6*CYCLE_PERIOD
        H0 = r.hip_angle(t0)
        th0 = r.knee_angle(t0) - H0
        h_ankle = HIP_HEIGHT - np.cos(np.pi - H0)*THIGH_LENGTH - SHANK_LENGTH*np.cos(th0)
        beta0 = 1/np.sin(h_ankle/FOOT_LENGTH) - th0 - np.pi/2

        x0 = [beta0, 0, -th0, 0, 1, 1] # TODO: initial values need to be determined 
        sol = solve_ivp(
                        f, 
                        [t0, CYCLE_PERIOD], 
                        x0, 
                        # rtol=1e-5, 
                        # atol=1e-8
                    )

        self.get_outputs(sol)

    def get_outputs(self, sol):
        """
        :param sol: ivp solution matrix 
        :return plots of outputs 
        """
        time = sol.t
        print(time)
        beta = sol.y[0,:]
        d_beta = sol.y[1,:]
        theta = sol.y[2,:]
        d_theta = sol.y[3,:]
        soleus_norm_length_muscle = sol.y[4,:]
        tibialis_norm_length_muscle = sol.y[5,:]

        soleus_moment = []
        tibialis_moment = []
        grav_ankle_moment = []
        toe_height = []
        ankle_height = []
        act_soleus = []
        act_tibialis = []
        excit_soleus = []
        excit_tibialis = []

        print("Number of data points = {}".format(len(time)))

        print("\nFinding outputs...")
        for t, b, d_b, th, d_th, ls, lt in zip(time, beta, d_beta, theta, d_theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
            soleus_moment.append(MOMENT_ARM_SOLEUS * self.soleus.get_force_single_val(soleus_length(b), ls))
            tibialis_moment.append(-MOMENT_ARM_TIBIALIS * self.tibialis.get_force_single_val(tibialis_length(b), lt))
            ankle_height.append(HIP_HEIGHT - np.cos(np.pi - r.hip_angle(t))*THIGH_LENGTH - SHANK_LENGTH*np.cos(th))
            toe_height.append(HIP_HEIGHT - np.cos(np.pi - r.hip_angle(t))*THIGH_LENGTH - SHANK_LENGTH*np.cos(th) - FOOT_LENGTH*np.sin(np.pi/2 - b + th)) 
            excit_soleus.append(self.get_soleus_excitation(t, b, th, d_b, d_th))
            excit_tibialis.append(self.get_tibialis_excitation(t, b, th, d_b, d_th))
            act_soleus.append(self.get_soleus_activation(t, b, th, d_b, d_th)[0])
            act_tibialis.append(self.get_tibialis_activation(t, b, th, d_b, d_th))
            grav_ankle_moment.append(gravity_moment_ankle(b, th, r.hip_angle(t)))

        print("\nMuscle fatigue:")
        print("Soleus = {}".format(np.trapz(act_soleus)))
        print("Tibialis Anterior = {}".format(np.trapz(act_tibialis)))

        # plot results 
        plt.figure()
        plt.title("Physical angles over the simulation")
        plt.plot(time, beta, 'r')
        plt.plot(time, theta, 'g')
        plt.legend(('foot-shank', 'shank-vertical'))
        plt.ylabel('Angle (rad)')
        plt.tight_layout()
        plt.savefig("outputs/angles.png")

        plt.figure()
        plt.title("Moments around the ankle")
        plt.plot(time, soleus_moment, 'r')
        plt.plot(time, tibialis_moment, 'g')
        plt.plot(time, grav_ankle_moment, 'k')
        plt.plot(time, np.add(np.add(tibialis_moment, soleus_moment), grav_ankle_moment), 'b')
        plt.legend(('soleus', 'tibialis', 'gravity', 'total moment'))
        plt.xlabel('Time (s)')
        plt.ylabel('Torques (Nm)')
        # plt.ylim(-3, 4)
        plt.tight_layout()
        plt.savefig("outputs/moments.png")

        plt.figure()
        plt.title("Height of body parts off the ground")
        plt.plot(time, toe_height, 'r')
        plt.plot(time, ankle_height, 'g')
        plt.legend(('toes', 'ankle'))
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        # plt.ylim(0, 0.5)
        plt.tight_layout()
        plt.savefig("outputs/heights.png")




# ******** NON-CLASS UTIL FUNCTIONS ********
def soleus_length(beta):
    """
    :param beta: angle of foot relative to shank
    :return: soleus length
    """
    rotation = [[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]]
    origin = np.dot(rotation, [.3, .03])
    insertion = [-.05, -.02]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def tibialis_length(beta):
    """
    :param beta: angle of foot relative to shank
    :return: tibialis anterior length
    """
    rotation = [[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]]
    origin = np.dot(rotation, [.3, -.03])
    insertion = [.06, -.03]
    difference = origin - insertion
    return np.sqrt(difference[0]**2 + difference[1]**2)


def gravity_moment_ankle(beta, theta, H):
    """
    :param theta: angle of foot relative to shank
    :param beta: angle of shank relative to vertical 
    :param t: current time in seconds 
    :return moment about ankle due to force of gravity on body
    """
    torso_moment = MASS_TORSO*GRAVITY*(THIGH_LENGTH*np.sin(np.pi - H) + SHANK_LENGTH*np.sin(theta))
    foot_moment = MASS_FOOT*GRAVITY*D_COM_SHANK_ANKLE*np.cos(np.pi/2 - beta + theta)
    shank_moment = MASS_SHANK*GRAVITY*D_COM_SHANK_ANKLE*np.sin(theta)
    thigh_moment = MASS_THIGH*GRAVITY*(D_COM_THIGH_KNEE*np.sin(np.pi - H) + SHANK_LENGTH*np.sin(theta))
    return torso_moment + shank_moment + thigh_moment - foot_moment 


def gravity_moment_knee(beta, theta):
    # note: if stuff isnt working, check that we dont also want thigh and torso in this equation
    # note 2: can also check that the knee torques from regression model were measured with gravity too 
    """
    :param theta: angle of foot relative to shank
    :param beta: angle of shank relative to vertical 
    :param t: current time in seconds 
    :return moment about knee due to force of gravity on body
    """
    foot_moment = MASS_FOOT*GRAVITY*(D_COM_FOOT_ANKLE*np.cos(np.pi/2 + theta - beta) + SHANK_LENGTH*np.sin(theta))
    shank_moment = MASS_SHANK*GRAVITY*D_COM_SHANK_ANKLE*np.sin(theta)
    return  foot_moment + shank_moment

