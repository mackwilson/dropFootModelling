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
I_ANKLE = 0.5
F_MAX_SOLEUS = 16000
F_MAX_TIBIALIS = 2000
MOMENT_ARM_SOLEUS = .048
MOMENT_ARM_TIBIALIS = .036
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
DESIRED_BETA_TRAJECTORY = np.pi/2 # rad



class GaitSimulator:
    def __init__(self, footDrop=False, fes=False, Kp=0, Kd=0, Ki=0):
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
        self.Ki = Ki

        rest_length_soleus = soleus_length(np.pi/2)
        rest_length_tibialis = tibialis_length(np.pi/2)

        self.soleus = HillTypeMuscle(F_MAX_SOLEUS, .6*rest_length_soleus, .4*rest_length_soleus)
        self.tibialis = HillTypeMuscle(F_MAX_TIBIALIS, .6*rest_length_tibialis, .4*rest_length_tibialis)

        self.total_error = 0
        self.last_error = None

    # ******** CLASS METHODS ********
    def suffix(self):
        if self.fes:
            return "fes"
        if self.footDrop:
            return "footdrop"
        else:
            return "normal"

        
    def get_soleus_activation(self, t):
        """
        :param t: current time 
        :return the practical soleus activation for the type of simulation
        """
        return r.soleus_activation(t)[0]

    def get_tibialis_activation(self, t, beta):
        """
        :param t: current time 
        :param beta: angle from foot to shank
        :return the resulting tibialis stimulation 
        """
        return self.get_natural_tibialis_activation(t) + self.get_tibialis_excitation(t, beta) 

    def get_natural_tibialis_activation(self, t):
        """
        :param t: current time
        :return the natural tibialis activation from the muscle for the type of simulation
        """
        scaler = 1
        if self.footDrop:
            # reduce by a certain amount
            scaler = 0.05
        return r.tibialis_activation(t)*scaler

    def get_tibialis_excitation(self, t, beta):
        """
        :param t: current time
        :param beta: angle from foot to shank, used to calculate error from desired trajectory
        :return the applied excitation for the tibialis as an additional normalized activation  
        """
        if self.fes:
            # apply excitation proportional to how much beta > pi/2 
            # this should drive beta to pi/2 
            err = beta - DESIRED_BETA_TRAJECTORY
            self.last_error = err 
            self.total_error = self.total_error + abs(err)

            # ** do nothing if beta <= pi/2 **
            if err > 0:
                return self.Kp*err

        # if no fes, return 0 excitation applied
        return 0 


    # - - - - -  DYNAMICS - - - - - - 
    def dynamics(self, x, t):
        """
        :param x: state vector [ankle angle, angular velocity, soleus normalized CE length, TA normalized CE length]
        :param t: current time 
        :return: derivative of state vector, xd
        """
        # state 
        beta = x[0]
        d_beta = x[1]
        l_ce_norm_soleus = x[2]
        l_ce_norm_tibialis = x[3]
        H = r.hip_angle(t)
        K = r.knee_angle(t)
        theta = K - H
        
        activation_tibialis = self.get_tibialis_activation(t, beta)
        activation_soleus = self.get_soleus_activation(t)

        # calculated
        l_soleus = soleus_length(beta)
        l_tibialis = tibialis_length(beta)
        l_tendon_norm_soleus = self.soleus.norm_tendon_length(l_soleus, l_ce_norm_soleus)
        l_tendon_norm_tibialis = self.tibialis.norm_tendon_length(l_tibialis, l_ce_norm_tibialis)
        torque_soleus = force_length_tendon_single_val(l_tendon_norm_soleus) * F_MAX_SOLEUS * activation_soleus * MOMENT_ARM_SOLEUS
        torque_tibant = force_length_tendon_single_val(l_tendon_norm_tibialis) * F_MAX_TIBIALIS * activation_tibialis * MOMENT_ARM_TIBIALIS
        vm_soleus = get_velocity_single_val(activation_soleus, l_ce_norm_soleus, l_tendon_norm_soleus)
        vm_tibialis = get_velocity_single_val(activation_tibialis, l_ce_norm_tibialis, l_tendon_norm_tibialis)


        # derivative of state: 
        xd = [0, 0, 0, 0]
        xd[0] = d_beta
        xd[1] = -(torque_tibant - torque_soleus - gravity_moment_ankle(beta, theta)) / I_ANKLE
        xd[2] = vm_soleus
        xd[3] = vm_tibialis
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
            print("Kp = {}, Kd = {}, Ki = {}".format(self.Kp, self.Kd, self.Ki))

        print("\nSimulating...")
        def f(t, x):
            return self.dynamics(x, t)

        """
        state = [
                    shank-foot angle, 
                    angular velocity for above, 
                    normalized CE length of soleus, 
                    normalized CE length of tibialis
                ]
        """
        t0 = 0.6*CYCLE_PERIOD
        H0 = r.hip_angle(t0)
        th0 = r.knee_angle(t0) - H0
        h_ankle = HIP_HEIGHT - np.cos(np.pi - H0)*THIGH_LENGTH - SHANK_LENGTH*np.cos(th0)
        beta0 = np.pi - 1/np.cos(h_ankle/FOOT_LENGTH) + th0

        # plot_regressions(t0, CYCLE_PERIOD)

        x0 = [beta0, 0, 1.1, 0.95]
        sol = solve_ivp(
                        f, 
                        [t0, CYCLE_PERIOD], 
                        x0
                    )

        self.get_outputs(sol)

    def get_outputs(self, sol):
        """
        :param sol: ivp solution matrix 
        :return plots of outputs 
        """
        time = sol.t
        beta = sol.y[0,:]
        d_beta = sol.y[1,:]
        soleus_norm_length_muscle = sol.y[2,:]
        tibialis_norm_length_muscle = sol.y[3,:]
        theta = np.subtract(r.knee_angle(time), r.hip_angle(time))

        soleus_moment = []
        tibialis_moment = []
        grav_ankle_moment = []
        toe_height = []
        ankle_height = []
        act_soleus = []
        act_tibialis = []
        excit_tibialis = []
        stim_tibialis = []

        print("Number of data points = {}".format(len(time)))

        print("\nFinding outputs...")
        for t, b, d_b, th, ls, lt in zip(time, beta, d_beta, theta, soleus_norm_length_muscle, tibialis_norm_length_muscle):
            l_tendon_norm_soleus = self.soleus.norm_tendon_length(soleus_length(b), ls)
            l_tendon_norm_tibialis = self.tibialis.norm_tendon_length(tibialis_length(b), lt)
            torque_soleus = force_length_tendon_single_val(l_tendon_norm_soleus) * F_MAX_SOLEUS * self.get_soleus_activation(t) * MOMENT_ARM_SOLEUS
            torque_tibant = force_length_tendon_single_val(l_tendon_norm_tibialis) * F_MAX_TIBIALIS * self.get_tibialis_activation(t, b) * MOMENT_ARM_TIBIALIS
            
            soleus_moment.append(-torque_soleus)
            tibialis_moment.append(torque_tibant)
            ankle_height.append(HIP_HEIGHT - np.cos(np.pi - r.hip_angle(t))*THIGH_LENGTH - SHANK_LENGTH*np.cos(abs(th)))
            toe_height.append(HIP_HEIGHT - np.cos(np.pi - r.hip_angle(t))*THIGH_LENGTH - SHANK_LENGTH*np.cos(abs(th)) + FOOT_LENGTH*np.sin(np.pi/2 - b + th)) 
            excit_tibialis.append(self.get_tibialis_excitation(t, b))
            act_soleus.append(self.get_soleus_activation(t))
            act_tibialis.append(self.get_natural_tibialis_activation(t))
            stim_tibialis.append(self.get_tibialis_activation(t, b))
            grav_ankle_moment.append(gravity_moment_ankle(b, th))

        print("\nMuscle fatigue:")
        print("Soleus = {}".format(np.trapz(act_soleus)))
        print("Tibialis Anterior = {}".format(np.trapz(stim_tibialis)))

        if self.fes:
            print("\nControl success:")
            print("Total error (integral) = {} rads".format(self.total_error))
            print("Steady state error = {} rads".format(self.last_error))

        # plot results 
        plt.figure()
        plt.title("Physical angles for {} simulation".format(self.suffix()))
        plt.plot(time, beta, 'r')
        plt.plot(time, theta, 'g')
        plt.legend(('foot-shank (beta)', 'shank-vertical (theta)'))
        plt.ylabel('Angle (rad)')
        plt.xlabel('Time (s)')
        plt.tight_layout()
        plt.savefig("outputs/angles_{}.png".format(self.suffix()))
        plt.close()

        plt.figure()
        plt.title("Moments around the ankle for {} simulation".format(self.suffix()))
        plt.plot(time, soleus_moment, 'r')
        plt.plot(time, tibialis_moment, 'g')
        plt.plot(time, grav_ankle_moment, 'k')
        plt.plot(time, np.add(np.add(tibialis_moment, soleus_moment), grav_ankle_moment), 'b')
        plt.legend(('soleus', 'tibialis', 'gravity', 'total moment'))
        plt.xlabel('Time (s)')
        plt.ylabel('Torques (Nm)')
        # plt.ylim(-3, 4)
        plt.tight_layout()
        plt.savefig("outputs/moments_{}.png".format(self.suffix()))
        plt.close()

        plt.figure()
        plt.title("Height off the ground for {} simulation".format(self.suffix()))
        plt.plot(time, toe_height, 'r')
        plt.plot(time, ankle_height, 'g')
        plt.legend(('toes', 'heel'))
        plt.xlabel('Time (s)')
        plt.ylabel('Height (m)')
        # plt.ylim(0, 0.5)
        plt.tight_layout()
        plt.savefig("outputs/heights_{}.png".format(self.suffix()))
        plt.close()

        plt.figure()
        plt.title("Activations and excitations for {} simulation".format(self.suffix()))
        plt.plot(time, act_soleus, 'g')
        plt.plot(time, act_tibialis, 'r')
        plt.plot(time, excit_tibialis, 'b')
        plt.plot(time, stim_tibialis, 'k')
        plt.legend(("soleus activation", "natural tibialis activation", "tibialis excitation", "total tibialis stimulation"))
        plt.ylabel("Activation as decimal")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("outputs/activations_{}.png".format(self.suffix()))
        plt.close()

        plt.figure()
        plt.title("Normalized CE lengths for {} simulation".format(self.suffix()))
        plt.plot(time, soleus_norm_length_muscle, 'r')
        plt.plot(time, tibialis_norm_length_muscle, 'g')
        plt.legend(("soleus", "tibialis"))
        plt.ylabel("Normalized length")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig("outputs/lengths_{}.png".format(self.suffix()))
        plt.close()




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


def gravity_moment_ankle(beta, theta):
    """
    :param theta: angle of foot relative to shank
    :param beta: angle of shank relative to vertical 
    :param t: current time in seconds 
    :return moment about ankle due to force of gravity on body
    """
    return MASS_FOOT*GRAVITY*D_COM_SHANK_ANKLE*np.cos(beta - theta - np.pi/2)


def plot_regressions(t0, tf):
    """
    Plot the important regression models for this code.
    :param t0: initial time
    :param tf: final time 
    """
    t = np.linspace(t0, tf, 30)
    plt.figure()
    plt.plot(t, r.hip_angle(t), 'r')
    plt.plot(t, r.knee_angle(t), 'g')
    plt.plot(t, np.subtract(r.knee_angle(t), r.hip_angle(t)), 'k')
    plt.legend(('hip (H)','knee (K)','theta'))
    plt.ylabel("Angle (rad)")
    plt.xlabel("Time (s)")
    plt.title("Joint angles throughout gait from regression model")
    plt.savefig("outputs/regression_angle.png")
    plt.close()

    plt.figure()
    plt.plot(t, r.hip_torque(t), 'r')
    plt.plot(t, r.knee_torque(t),'g')
    plt.legend(('hip (TH)','knee (TK)'))
    plt.ylabel("Torque (Nm/kg)")
    plt.xlabel("Time (s)")
    plt.title("Joint torques throughout gait from regression model")
    plt.savefig("outputs/regression_torque.png")
    plt.close()

    plt.figure()
    plt.plot(t, r.tibialis_activation(t), 'r')
    plt.plot(t, r.soleus_activation(t),'g')
    plt.legend(('tibialis (aTA)','soleus (aS)'))
    plt.ylabel("Activation (decimal)")
    plt.xlabel("Time (s)")
    plt.title("Normalized activations for normal muscle from regression model")
    plt.savefig("outputs/regression_activation.png")
    plt.close()
