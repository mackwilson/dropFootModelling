from gait_simulate import GaitSimulator 
import muscle

GaitSimulator().simulate()

GaitSimulator(footDrop=True).simulate()

GaitSimulator(footDrop=True, fes=True, Kp=4, Kd=1/100, Ki=0).simulate()

