import mujoco as mj 
import numpy as np 

model = mj.MjModel.from_xml_path("./1dof.xml")
data = mj.MjData(model)
sensor = data.sensor("torque")

qvel_prev = data.qvel
torques = []
q_accs = []

steps = 10000
for step in range(steps):
    data.ctrl = (np.random.rand() * 2) - 1
    mj.mj_step(model, data)

    torque = sensor.data[0]
    torques.append(torque)

    qvel = data.qvel.copy() 
    q_acc = (qvel - qvel_prev)/ model.opt.timestep
    q_accs.append(q_acc)
    
    qvel_prev = qvel

T = np.array([torques]).T
A = np.stack(q_accs)

Inertia, residuals, rank, s = np.linalg.lstsq(A, T)
print(Inertia)
