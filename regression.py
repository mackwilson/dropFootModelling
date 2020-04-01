import numpy as np 
from sklearn.linear_model import Ridge
from scipy.special import expit


class Gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-(x-self.mu)**2/2/self.sigma**2)

class Sigmoid:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return expit((x-self.mu) / self.sigma)

class Polynomial:
    def __init__(self, mu, sigma):
        # TODO HERE

    def __call__(self, x):
        # TODO HERE
        pass

class Regression():
    """
    1D regression model with Gaussian basis functions.
    """

    def __init__(self, x, t, centres, width, regularization_weight=1e-6, sigmoids=False):
        """
        :param x: samples of an independent variable
        :param t: corresponding samples of a dependent variable
        :param centres: a vector of Gaussian centres (should have similar range of values as x)
        :param width: sigma parameter of Gaussians
        :param regularization_weight: regularization strength parameter
        """
        if sigmoids:
            self.basis_functions = [Sigmoid(centre, width) for centre in centres]
        else:
            self.basis_functions = [Gaussian(centre, width) for centre in centres]
        self.ridge = Ridge(alpha=regularization_weight, fit_intercept=False)
        self.ridge.fit(self._get_features(x), t)

    def eval(self, x):
        """
        :param x: a new (or multiple samples) of the independent variable
        :return: the value of the curve at x
        """
        return self.ridge.predict(self._get_features(x))

    def _get_features(self, x):
        if not isinstance(x, collections.Sized):
            x = [x]

        phi = np.zeros((len(x), len(self.basis_functions)))
        for i, basis_function in enumerate(self.basis_functions):
            phi[:,i] = basis_function(x)
        return phi


def get_muscle_force_velocity_regression():
    data = np.array([
        [-1.0028395556708567, 0.0024834319945283845],
        [-0.8858611825192801, 0.03218792009622429],
        [-0.5176245843258415, 0.15771090304473967],
        [-0.5232565269687035, 0.16930496922242444],
        [-0.29749770052593094, 0.2899790099290114],
        [-0.2828848376217543, 0.3545364496120378],
        [-0.1801231103040022, 0.3892195938775034],
        [-0.08494610976156225, 0.5927831890757294],
        [-0.10185137142991896, 0.6259097662790973],
        [-0.0326643239546236, 0.7682365981934388],
        [-0.020787245583830716, 0.8526638522676352],
        [0.0028442725407418212, 0.9999952831301149],
        [0.014617579774061973, 1.0662107025777694],
        [0.04058866536166583, 1.124136223202283],
        [0.026390887007381902, 1.132426122025424],
        [0.021070257776939272, 1.1986556920827338],
        [0.05844673474682183, 1.2582274002971627],
        [0.09900238201929201, 1.3757434966156459],
        [0.1020023112662436, 1.4022310794556732],
        [0.10055894908138963, 1.1489210160137733],
        [0.1946227683309354, 1.1571212943090965],
        [0.3313459588217258, 1.152041225442796],
        [0.5510200231126625, 1.204839508502158]
    ])

    velocity = data[:,0]
    force = data[:,1]

    centres = np.arange(-1, 0, .2)
    width = .15
    result = Regression(velocity, force, centres, width, .1, sigmoids=True)

    return result


def get_muscle_force_length_regression():
    """
    CE force-length data samples from Winters et al. (2011) Figure 3C,
    normalized so that max force is ~1 and length at max force is ~1.
    The samples were taken form the paper with WebPlotDigitizer, and
    cut-and-pasted here.. 
    """
    data = np.array([
        [38.5662866635083, -17.050877714929527],
        [39.34921710553933, 2.3013572394286825],
        [40.24181873789377, 21.18058887534403],
        [41.148129269038634, 42.706798641414764],
        [41.976756040371086, 62.57227427394798],
        [42.75359363849526, 80.23227304852449],
        [43.60811499643185, 99.81330589014934],
        [44.592109287389135, 121.49311581696566],
        [46.197573656845755, 136.3134823356109],
        [48.320929758385155, 150.86474929623475],
        [50.44428585992456, 166.51252515570178],
        [52.56764196146396, 179.0919811214747],
        [54.69099806300336, 192.00130151892154],
        [56.81435416454277, 199.17197374228763],
        [58.91181567947803, 191.36660533822595],
        [60.82801508818432, 170.93419969716115],
        [62.66653073707819, 153.42305597662534],
        [64.42736262615965, 135.60188586609115],
        [66.23998368844939, 116.68684472661886],
        [68.07849933734326, 97.21869350276233],
        [69.81343663982057, 79.33144136735879],
        [71.60016311550618, 61.27497651183572],
        [73.36099500458762, 43.83672283741831],
        [75.14772148027322, 24.865625908400716],
        [76.9344479559588, 7.115103462499121],
        [78.64349067183198, -10.414623100896051],
        [79.86053624222652, -23.598685884949987]
    ])

    length = data[:,0]
    force = data[:,1]

    # normalize
    max_idx = np.argmax(force)
    force = np.divide(force, force[max_idx])
    length = np.divide(length, length[max_idx])

    # regression -> values for step, dith and weight were found through trial 
    #   and error 
    step = 0.2
    centres = np.arange(np.min(length), np.max(length), step)
    width = 0.05
    weight = 0.1
    result = Regression(length, force, centres, width, weight, sigmoids=True)
    return result

def get_tibialis_activation_regression():
    pass

def get_soleus_activation_regression():
    pass

def get_hip_angle_regression():
    pass

def get_hip_torque_regression():
    pass

def get_knee_angle_regression():
    pass

def get_knee_torque_regression():
    pass 



# REGRESSION RESULTS 
force_length_regression = get_muscle_force_length_regression()
force_velocity_regression = get_muscle_force_velocity_regression()
hip_angle_regression = get_hip_angle_regression()
hip_torque_regression = get_hip_torque_regression()
knee_angle_regression = get_knee_angle_regression()
knee_torque_regression = get_knee_torque_regression()
soleus_activation_regression = get_soleus_activation_regression()
tibialis_activation_regression = get_tibialis_activation_regression()

def force_length_muscle(lm):
    """
    :param lm: muscle (contracile element) length
    :return: force-length scale factor
    """
    return force_length_regression.eval(lm)

def force_velocity_muscle(vm):
    """
    :param vm: muscle (contractile element) velocity)
    :return: force-velocity scale factor
    """
    return np.maximum(0, force_velocity_regression.eval(vm))

def knee_angle():
    """
    :param: 
    :return: knee angle 
    """
    # TODO
    pass

def knee_torque():
    """
    :param: 
    :return: knee angle 
    """
    # TODO
    pass

def hip_angle():
    """
    :param: 
    :return: knee angle 
    """
    # TODO
    pass

def hip_torque():
    """
    :param: 
    :return: knee angle 
    """
    # TODO
    pass

def soleus_activation():
    """
    :param: 
    :return: knee angle 
    """
    # TODO
    pass

def tibialis_activation():
    """
    :param: 
    :return: knee angle 
    """
    # TODO
    pass