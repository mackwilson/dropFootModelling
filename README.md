BME 355 course project. Modelling the shank-foot complex through the gait cycles, for normal gait and with foot drop syndrome.

The model consists of rigid body physics for the food, shank, and upper leg segments. The upper leg is simplified and only torques from the knee and hip are considered in the forces. The model is simulated from 60% to 100% of the gait cycle for normal gait, foot drop, and foot drop with FES scenarios. FES is implemented as PID controller. 

**To Run**
Requires: `numpy, `scipy, `sklearn`, and `matplotlib` libraries to be installed
Run with `python main.py`.
Edit and run `python controlPlots.py` to optimize your controller gains. 

