import numpy as np
import matplotlib.pyplot as plt 

"""
This file is used to plot manually extracted performance metrics for the FES simulations. 
The K values reflect gains for the PID control used by FES.
These values could have been extracted automatically, given more time to alter the simulation file.
"""

kp = {
    "vals": [1, 2, 3, 4, 5],
    "errs": [88.7867, 57.9171, 53.0463, 54.9969, 56.5750],
    "fatigues": [4.9491, 3.7243, 3.5406, 3.2082, 2.5382]
}

kd = {
    "vals": [0, 0.001, 0.002, 0.005, 0.01, 0.02],
    "errs": [54.9969, 55.2986, 55.07, 54.32, 49.75, 54.64],
    "fatigues": [3.2082, 3.23, 3.25, 3.22, 3.17, 3.33]
}

ki = {
    "vals": [0, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005],
    "errs": [49.75, 49.72, 51.38, 53.37, 54.73, 55.53],
    "fatigues": [3.17, 3.17, 3.20, 3.27, 3.31, 3.36]
}

fig, axs = plt.subplots(2)
fig.suptitle("Optimal Kp value to minimize simulation results")
axs[0].plot(kp["vals"], kp["errs"])
axs[1].plot(kp["vals"], kp["fatigues"])
axs[0].set(ylabel="total error")
axs[1].set(ylabel="tibialis fatigue", xlabel='Kp values')
plt.savefig("outputs/Kp.png")
plt.close()

fig, axs = plt.subplots(2)
fig.suptitle("Optimal Kd value given Kp=4")
axs[0].plot(kd["vals"], kd["errs"])
axs[1].plot(kd["vals"], kd["fatigues"])
axs[0].set(ylabel="total error")
axs[1].set(ylabel="tibialis fatigue", xlabel='Kd values')
plt.savefig("outputs/Kd.png")
plt.close()

fig, axs = plt.subplots(2)
fig.suptitle("Optimal Ki value given Kp=4, Kd=0.01")
axs[0].plot(ki["vals"], ki["errs"])
axs[1].plot(ki["vals"], ki["fatigues"])
axs[0].set(ylabel="total error")
axs[1].set(ylabel="tibialis fatigue", xlabel='Ki values')
plt.savefig("outputs/Ki.png")
plt.close()