import numpy as np
import collections
from sklearn.linear_model import Ridge
from scipy.special import expit
import matplotlib.pylab as plt


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
    

class Polynomial_Regression():

    def __init__(self,x,y, degree):
        self.degree = degree
        self.powers = np.polyfit(x,y,self.degree)
        
    def eval(self, x):
        yprime = np.polyval(self.powers, x)
        plt.plot(x,yprime)
        plt.show()
        return yprime     


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

    # regression -> values for step, width and weight were found through trial 
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
    data = np.array([
        [0.710059171597635, 36.043956043956],
        [4.615384615384617, 36.043956043956],
        [9.230769230769234, 34.28571428571425],
        [12.781065088757398, 31.64835164835162],
        [15.266272189349117, 28.571428571428555],
        [18.106508875739642, 25.05494505494505],
        [20.591715976331365, 21.538461538461547],
        [22.721893491124263, 18.021978021978015],
        [25.207100591715978, 14.50549450549451],
        [27.6923076923077, 10.989010989010978],
        [30.532544378698233, 7.032967032967036],
        [33.37278106508876, 3.516483516483504],
        [36.21301775147929, 0],
        [38.69822485207101, -2.19780219780219],
        [41.53846153846154, -4.835164835164818],
        [44.73372781065089, -7.472527472527446],
        [48.28402366863906, -8.791208791208817],
        [51.4792899408284, -8.791208791208817],
        [54.31952662721893, -7.032967032967008],
        [56.804733727810664, -4.835164835164818],
        [58.934911242603555, -1.318681318681314],
        [60.71005917159763, 1.758241758241752],
        [62.13017751479291, 4.39560439560438],
        [63.905325443786985, 8.35164835164835],
        [66.03550295857988, 12.747252747252759],
        [67.81065088757398, 16.7032967032967],
        [69.94082840236686, 20.659340659340643],
        [72.07100591715977, 24.175824175824175],
        [73.84615384615387, 26.373626373626365],
        [76.68639053254438, 29.45054945054943],
        [79.88165680473372, 32.08791208791206],
        [83.7869822485207, 34.28571428571425],
        [88.04733727810651, 34.72527472527469],
        [91.24260355029585, 35.164835164835125],
        [94.4378698224852, 35.164835164835125],
        [97.27810650887574, 35.60439560439556]
            ])
    
    x = data[:,0]
    y = data[:,1]
    #Normalize x to be from 0 to time_step instead of 0 to 100% gait cycle
    #TO DO ONCE VALUE IS FOUND (x/100 * time)
    
    degree = 6
    result = Polynomial_Regression(x,y,degree)
    return result

def get_hip_torque_regression():
    data = np.array([
        [0.616740088105729, 0.2499999999999991],
        [1.9797875097175393, 0.3823529411764701],
        [3.70562321845037, 0.4852941176470589],
        [6.198497019953322, 0.41176470588235237],
        [8.701736201088352, 0.3088235294117636],
        [11.899455817569304, 0.235294117647058],
        [15.449598341539229, 0.16176470588235237],
        [17.947654832858234, 0.07352941176470562],
        [21.155739828971207, -0.029411764705883137],
        [24.348276755636164, -0.08823529411764763],
        [27.540813682301092, -0.14705882352941302],
        [31.080590826638996, -0.1911764705882364],
        [34.64109873024097, -0.29411764705882426],
        [38.21197201347496, -0.4264705882352944],
        [40.37315366675301, -0.5588235294117654],
        [42.18191241254209, -0.6911764705882364],
        [44.33791137600414, -0.8088235294117654],
        [46.13630474216117, -0.9117647058823533],
        [48.950505312257036, -0.8970588235294121],
        [50.68152371080589, -0.8088235294117654],
        [52.4021767297227, -0.6911764705882364],
        [54.12801243845553, -0.5882352941176476],
        [59.046385073853315, -0.5441176470588243],
        [61.49779735682816, -0.5000000000000009],
        [64.95983415392587, -0.3235294117647065],
        [66.33842964498575, -0.23529411764705976],
        [68.75874578906453, -0.10294117647058876],
        [71.18942731277531, 0],
        [75.06089660533817, 0.014705882352941124],
        [78.94273127753303, 0],
        [82.44622959315885, 0.0588235294117645],
        [84.52448820938065, 0.16176470588235237],
        [87.95024617776625, 0.4411764705882346],
        [91.05467737755893, 0.6323529411764701],
        [93.88960870691886, 0.5882352941176467],
        [96.05597305001294, 0.4411764705882346],
        [98.62658719875614, 0.14705882352941124]
            ])
    x = data[:,0]
    y = data[:,1]
    #Normalize x to be from 0 to time_step instead of 0 to 100% gait cycle
    #TO DO ONCE VALUE IS FOUND (x/100 * time)
    
    degree = 8
    result = Polynomial_Regression(x,y,degree)
    return result

def get_knee_angle_regression():        
    data = np.array([
        [2.831858407079647, 6.666666666666686],
        [5.309734513274336, 11.111111111111086],
        [7.787610619469028, 14.4444444444444],
        [9.557522123893804, 16.66666666666663],
        [11.327433628318584, 17.77777777777777],
        [13.09734513274336, 17.77777777777777],
        [15.221238938053098, 16.66666666666663],
        [16.991150442477874, 16.66666666666663],
        [18.761061946902657, 14.4444444444444],
        [22.30088495575221, 10],
        [25.13274336283186, 8.888888888888857],
        [27.610619469026553, 5.555555555555543],
        [30.08849557522123, 2.2222222222222285],
        [32.92035398230088, 1.1111111111111427],
        [35.752212389380524, -1.1111111111111427],
        [38.584070796460175, 0],
        [41.769911504424776, 0],
        [44.24778761061946, 1.1111111111111427],
        [49.557522123893804, 7.7777777777777715],
        [51.68141592920354, 12.222222222222229],
        [53.80530973451327, 17.77777777777777],
        [56.28318584070797, 24.444444444444457],
        [58.05309734513274, 30],
        [59.46902654867256, 36.66666666666663],
        [61.94690265486726, 42.22222222222223],
        [66.54867256637166, 53.333333333333314],
        [69.73451327433628, 55.55555555555554],
        [73.6283185840708, 53.333333333333314],
        [76.81415929203538, 45.55555555555554],
        [80.35398230088495, 34.4444444444444],
        [83.18584070796459, 24.444444444444457],
        [86.72566371681415, 12.222222222222229],
        [91.32743362831859, 1.1111111111111427],
        [96.63716814159292, -1.1111111111111427],
        [98.76106194690264, 1.1111111111111427]
        ])
    x = data[:,0]
    y = data[:,1]
    #Normalize x to be from 0 to time_step instead of 0 to 100% gait cycle
    #TO DO ONCE VALUE IS FOUND (x/100 * time)
    
    degree = 8
    result = Polynomial_Regression(x,y,degree)
    return result

def get_knee_torque_regression():
    
    data = np.array([
        [1.061946902654853, -0.1538461538461533],
        [3.893805309734489, 0],
        [6.725663716814125, 0.26373626373626413],
        [10.619469026548671, 0.5384615384615383],
        [15.22123893805309, 0.4945054945054945],
        [18.407079646017678, 0.3406593406593408],
        [20.53097345132744, 0.19780219780219843],
        [23.716814159292028, 0.043956043956044244],
        [27.610619469026545, -0.1098901098901095],
        [30.44247787610618, -0.1868131868131866],
        [33.274336283185846, -0.26373626373626324],
        [36.46017699115043, -0.32967032967032894],
        [40.7079646017699, -0.36263736263736224],
        [43.53982300884957, -0.30769230769230704],
        [47.433628318584084, -0.12087912087912045],
        [50.97345132743362, 0.032967032967033294],
        [53.45132743362831, 0.12087912087912134],
        [55.22123893805312, 0.16483516483516514],
        [59.11504424778761, 0.24175824175824223],
        [62.300884955752224, 0.23076923076923128],
        [68.31858407079648, 0.16483516483516514],
        [71.15044247787611, 0.10989010989011039],
        [76.81415929203541, 0.032967032967033294],
        [81.76991150442481, -0.0219780219780219],
        [85.6637168141593, -0.1538461538461533],
        [89.55752212389382, -0.36263736263736224],
        [93.45132743362834, -0.4395604395604389],
        [95.22123893805312, -0.40659340659340604],
        [96.63716814159292, -0.30769230769230704],
        [98.4070796460177, -0.19780219780219754],
        [99.46902654867259, -0.0879120879120876]
            ])
    
    x = data[:,0]
    y = data[:,1]
    #Normalize x to be from 0 to time_step instead of 0 to 100% gait cycle
    #TO DO ONCE VALUE IS FOUND (x/100 * time)
    
    degree = 5
    result = Polynomial_Regression(x,y,degree)
    return result


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

def knee_angle(t):
    """
    :param t: time of evaluation  
    :return: knee angle 
    """
    return knee_angle_regression.eval(t)

def knee_torque(t):
    """
    :param: 
    :return: knee angle 
    """
    return knee_torque_regression.eval(t)

def hip_angle(t):
    """
    :param t: time of evaluation 
    :return: knee angle 
    """
    return hip_angle_regression.eval(t)

def hip_torque(t):
    """
    :param t: time of evaluation 
    :return: knee angle 
    """
    return hip_torque_regression.eval(t)

def soleus_activation(t):
    """
    :param t: time of evaluation 
    :return: knee angle 
    """
    return soleus_activation_regression.eval(t)

def tibialis_activation(t):
    """
    :param t: time of evaluation 
    :return: knee angle 
    """
    return tibialis_activation_regression.eval(t)