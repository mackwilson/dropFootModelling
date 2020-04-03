import collections
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import regression as r 



class HillTypeMuscle:
    """
    Damped Hill-type muscle model adapted from Millard et al. (2013). The
    dynamic model is defined in terms of normalized length and velocity.
    To model a particular muscle, scale factors are needed for force, CE
    length, and SE length. These are given as constructor arguments.
    """

    def __init__(self, f0M, resting_length_muscle, resting_length_tendon):
        """
        :param f0M: maximum isometric force
        :param resting_length_muscle: actual length (m) of muscle (CE) that corresponds to normalized length of 1
        :param resting_length_tendon: actual length of tendon (m) that corresponds to normalized length of 1
        """
        self.f0M = f0M
        self.resting_length_muscle = resting_length_muscle
        self.resting_length_tendon = resting_length_tendon

    def norm_tendon_length(self, muscle_tendon_length, normalized_muscle_length):
        """
        :param muscle_tendon_length: non-normalized length of the full muscle-tendon
            complex (typically found from joint angles and musculoskeletal geometry)
        :param normalized_muscle_length: normalized length of the contractile element
            (the state variable of the muscle model)
        :return: normalized length of the tendon
        """
        return (muscle_tendon_length - self.resting_length_muscle * normalized_muscle_length) / self.resting_length_tendon

    def get_force(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon(self.norm_tendon_length(total_length, norm_muscle_length))

    def get_force_single_val(self, total_length, norm_muscle_length):
        """
        :param total_length: muscle-tendon length (m)
        :param norm_muscle_length: normalized length of muscle (the state variable)
        :return: muscle tension (N)
        """
        return self.f0M * force_length_tendon_single_val(self.norm_tendon_length(total_length, norm_muscle_length))


def get_velocity(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    B = 0.1 # damping coefficient (see damped model in Millard et al.)

    def f(vm, a, lm, lt):
        Fv = r.force_velocity_muscle(vm)
        Ft = force_length_tendon(lt)
        Fp = force_length_parallel(lm)
        Fl = r.force_length_muscle(lm)
        
        return (a*Fl*Fv + Fp + B*vm)-Ft
            
    vm = fsolve(f, [0,1], args=(a,lm,lt))
    
    return vm[0]

def get_velocity_single_val(a, lm, lt):
    """
    :param a: activation (between 0 and 1)
    :param lm: normalized length of muscle (contractile element)
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized lengthening velocity of muscle (contractile element)
    """
    B = 0.1 # damping coefficient (see damped model in Millard et al.)

    def f(vm, a, lm, lt):
        Fv = r.force_velocity_muscle(vm)
        Ft = force_length_tendon_single_val(lt)
        Fp = force_length_parallel_single_val(lm)
        Fl = r.force_length_muscle(lm)
        
        return (a*Fl*Fv + Fp + B*vm)-Ft
            
    vm = fsolve(f, [0,1], args=(a,lm,lt))
    
    return vm[0]
    

def force_length_tendon(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """    
    normalized = []
    for l in lt:
        if l < 1:
            normalized.append(0)
        else:
            normalized.append((10*(l - 1)) + (240 * (l - 1)**2))
    
    return normalized

def force_length_tendon_single_val(lt):
    """
    :param lt: normalized length of tendon (series elastic element)
    :return: normalized tension produced by tendon
    """    
    if lt < 1:
        return 0
    else:
        return (10*(lt - 1)) + (240 * (lt - 1)**2)
    

def force_length_parallel(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """    
    # slack pe length = 1
    normalized = []
    
    for l in lm:
        if l < 1:
            normalized.append(0)
        else:
            normalized.append((3 * (l - 1)**2) / (0.6 + l - 1))

    return normalized

def force_length_parallel_single_val(lm):
    """
    :param lm: normalized length of muscle (contractile element)
    :return: normalized force produced by parallel elastic element
    """    
    # slack pe length = 1
    if lm < 1:
        return 0
    else:
        return (3 * (lm - 1)**2) / (0.6 + lm - 1)

def plot_curves():
    """
    Plot force-length, force-velocity, SE, and PE curves.
    """
    lm = np.arange(0, 1.8, .01)
    vm = np.arange(-1.2, 1.2, .01)
    lt = np.arange(0, 1.07, .01)
    plt.subplot(2,1,1)
    plt.plot(lm, r.force_length_muscle(lm), 'r')
    plt.plot(lm, force_length_parallel(lm), 'g')
    plt.plot(lt, force_length_tendon(lt), 'b')
    plt.legend(('CE', 'PE', 'SE'))
    plt.xlabel('Normalized length')
    plt.ylabel('Force scale factor')
    plt.subplot(2, 1, 2)
    plt.plot(vm, r.force_velocity_muscle(vm), 'k')
    plt.xlabel('Normalized muscle velocity')
    plt.ylabel('Force scale factor')
    plt.tight_layout()
    plt.savefig("outputs/hilltypemuscle.png")

