from gait_simulate import GaitSimulator 
import muscle

# GaitSimulator().simulate()

# GaitSimulator(footDrop=True).simulate()

GaitSimulator(footDrop=True, fes=True, Kp=2, Kd=1/200, Ki=0).simulate()
