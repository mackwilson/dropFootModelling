from gait_simulate import GaitSimulator 
import muscle

GaitSimulator().simulate()

GaitSimulator(footDrop=True).simulate()

GaitSimulator(footDrop=True, fes=True, Kp=2, Kd=1/10, Ki=1/100).simulate()
