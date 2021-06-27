import numpy
import pinocchio as pin
import hppfcl as fcl
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import sys
import config_double_pendulum as conf


def deg(arg):
    return degrees(arg)


def saturation(val, lmt):
    return max(min(val, lmt), -lmt)


# endof deg

show_plots = True

# Controller: # choose: pd | pdff | pdg | id | acc | imp
ctrl_type = 'nn'
Kp = np.eye(conf.Model.nv) * 380.0
Kd = np.eye(conf.Model.nv) * 35.0

if ctrl_type == 'id':
    Kp[1, 1] *= 0.25
    Kd[1, 1] *= 0.30
# print(Kp)
# acceleration-based controller: PI(acc) -> PD(vel)
velKp = np.array([[60, 0], [0, 40]])
velKd = np.eye(conf.Model.nv) * 0.20

# Impedance Controller:
des_inertia = np.array([[0.2, 0], [0, 0.2]])
des_damping = np.array([[12.0, 0], [0, 10.0]])
des_stiffness = np.array([[14, 0], [0, 8]])

# Physical parameters
jointsFriction = np.array([[1.1, 0], [0, 2.4]])

# Input
input_type = 'sin'
freqs = np.array([0.45, 0.4])
amps = np.array([0.8, 0.3])
phs = np.array([-.0 * pi * (90 / 180), .0 * pi * (25 / 180)])
ampsxfreqs = np.multiply(amps, freqs)
# print(ampsxfreqs)

# Environment Interaction
# Task Space Int Dyn
tStiffness = np.array([20.0, .0, .0])  # N/m
tDamping = np.array([0.23, .0, .0])  # N.s/m
# Joint Space Int Dyn
jStiffness = np.eye(conf.Model.nv) * 10.40  # N/rad
jDamping = np.eye(conf.Model.nv) * 0.104  # N.s/rad
Ksea = np.eye(conf.Model.nv) * 104.0  # N/rad
SeaMax = 104.0 * pi * (7/180)

# Desired states variables
q_des = np.array([pi * (178 / 180), pi * (90 / 180)])
dq_des = np.zeros(conf.Model.nv)
ddq_des = np.zeros(conf.Model.nv)

# q = pin.randomConfiguration(conf.Model)
# Initial states, q0, dq0
q = np.array([pi * (178 / 180), pi * (25 / 180)]) + np.array([amps[0] * sin(phs[0]), amps[1] * sin(phs[1])])
dq = np.zeros(conf.Model.nv)
ddq = np.zeros(conf.Model.nv)
q0 = q.copy()
dq0 = 2 * pi * np.array([ampsxfreqs[0], ampsxfreqs[1]])
q_rlx = np.array([pi, .0])

# Auxiliar state variables for integration
dq_last = np.zeros(conf.Model.nv)
ddq_last = dq_last.copy()

# Logging variables
downsmpl_log = 0
downsmpl_factor = 4
q_log = np.empty([1, 1 + conf.Model.nq]) * nan
dq_log = q_log.copy()
ddq_log = q_log.copy()
qdes_log = np.empty([1, conf.Model.nq]) * nan
dqdes_log = qdes_log.copy()
ddqdes_log = qdes_log.copy()

p_log = np.empty([1, 2 * conf.Model.nq]) * nan

data_sim = conf.Model.createData()

# print(conf.Model.getFrameId('bar_1'))
# base_frame = conf.Model.frames[0]
# print(base_frame)

# plt.figure()
# plt.draw()

# Create oscillatory reference data before simulation:
for k in range(conf.sim_steps):
    if input_type == 'sin':  # Oscillatory reference
        t = k*conf.dt
        q_des = q0 + np.array([amps[0] * sin(2 * pi * freqs[0] * t + phs[0]),
                               amps[1] * sin(2 * pi * freqs[1] * t + phs[1])])
        dq_des = 2 * pi * np.array([ampsxfreqs[0] * cos(2 * pi * freqs[0] * t),
                                    ampsxfreqs[1] * cos(2 * pi * freqs[1] * t)])
        ddq_des = -(2 * pi) ** 2 * np.array([ampsxfreqs[0] * freqs[0] * sin(2 * pi * freqs[0] * t),
                                             ampsxfreqs[1] * freqs[1] * sin(2 * pi * freqs[1] * t)])
    # endof if
    # Log variables
    downsmpl_log += 1
    if downsmpl_log > downsmpl_factor and show_plots:
        qdes_log = np.vstack((qdes_log, [deg(q_des[0]), deg(q_des[1])]))
        dqdes_log = np.vstack((dqdes_log, [deg(dq_des[0]), deg(dq_des[1])]))
        ddqdes_log = np.vstack((ddqdes_log, [deg(ddq_des[0]), deg(ddq_des[1])]))
        downsmpl_log = 0
# endof ref log


t = 0.00
tau_control = np.zeros(conf.Model.nv)

# SIMULATION:
for k in range(conf.sim_steps):

    loop_tbegin = time.time()
    pin.computeAllTerms(conf.Model, data_sim, q, dq)
    Mq = data_sim.M
    hq = data_sim.C
    grav = data_sim.g

    # inputs
    if input_type == 'stp':  # Step reference
        if t > conf.step_input_time:
            q_des = np.array([pi * (80 / 180), pi * (65 / 180)])
            dq_des = [0, 0]
            ddq_des = [0, 0]
    if input_type == 'sin':  # Oscillatory reference
        q_des = q0 + np.array([amps[0] * sin(2 * pi * freqs[0] * t + phs[0]),
                               amps[1] * sin(2 * pi * freqs[1] * t + phs[1])])
        dq_des = 2 * pi * np.array([ampsxfreqs[0] * cos(2 * pi * freqs[0] * t),
                                    ampsxfreqs[1] * cos(2 * pi * freqs[1] * t)])
        ddq_des = -(2 * pi) ** 2 * np.array([ampsxfreqs[0] * freqs[0] * sin(2 * pi * freqs[0] * t),
                                             ampsxfreqs[1] * freqs[1] * sin(2 * pi * freqs[1] * t)])
    # endof inputs

    # Task Space Interaction
    # pin.updateFramePlacements(conf.Model, data_sim)  # already done in 'computeAllTerms'
    #    J6 = pin.getJointJacobian(conf.Model, data_sim, 2, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
    #    # take first 3 rows of J6 cause we have a point contact (??)
    #    J = J6[:3, :]
    p = data_sim.oMi[2].translation
    #    int_force = np.multiply(tStiffness, p)
    #    int_tau = J.transpose().dot(int_force)  # tau = J^T x F
    # Joint Space Int
    tau_int = jStiffness.dot(q_des - q)

    # plt.plot(t, p[0])
    # plt.draw()
    # plt.pause(0.001)
    # print(int_tau)

    # SEA:
    # tau_sea = Ksea.dot(q - q_rlx)  # ? or:
    tau_sea = Ksea.dot(q_des - q)
    tau_sea[0] = saturation(tau_sea[0], SeaMax)
    tau_sea[1] = saturation(tau_sea[1], SeaMax)

    # PD Control
    if ctrl_type == 'pd':
        tau_control = Kp.dot(q_des - q) + Kd.dot(dq_des - dq)
    # PD+Feedforward Control
    if ctrl_type == 'pdff':
        tau_control = Kp.dot(q_des - q) + Kd.dot(dq_des - dq) + np.multiply(np.diag(Mq), ddq_des)
    # PD Control + Grav Compensation
    if ctrl_type == 'pdg':
        tau_control = Kp.dot(q_des - q) + Kd.dot(dq_des - dq) + grav
    # Inverse Dynamics Control, data_sim.nle contains C + g
    if ctrl_type == 'id':
        tau_control = Mq.dot(ddq_des + Kp.dot(q_des - q) + Kd.dot(dq_des - dq)) + data_sim.nle
    if ctrl_type == 'acc':
        tau_control = Mq.dot(ddq_des) + velKp.dot(dq_des - dq) + velKd.dot(ddq_des - ddq) + data_sim.nle
    if ctrl_type == 'imp':
        tau_control = des_stiffness.dot(q_des - q) + des_damping.dot(dq_des - dq) \
                      + Mq.dot(ddq_des) + data_sim.nle

    # Joints Friction
    tau_frict = jointsFriction.dot(dq) + 0.05*np.multiply(dq, dq)
    # Tau = tau_int + tau_sea + tau_control
    # Tau = tau_int + tau_control
    Tau = tau_control - tau_frict

    # Forward Dynamics (simulation)
    ddq = pin.aba(conf.Model, data_sim, q, dq, Tau)
    # Mq_inv = np.linalg.inv(Mq)
    # ddq = Mq_inv.dot(- data_sim.nle)

    dq += ddq*conf.dt
    q = pin.integrate(conf.Model, q, dq*conf.dt)
    # Forward Euler Integration with Trapeziodal Rule
    #dq += (ddq_last + ddq) * conf.dt * 0.5
    #ddq_last = ddq.copy()
    #q += (dq_last + dq) * conf.dt * 0.5
    #dq_last = dq.copy()

    # Log variables
    downsmpl_log += 1
    if downsmpl_log > downsmpl_factor and show_plots:
        # States log
        q_log = np.vstack((q_log, [t, deg(q[0]), deg(q[1])]))
        dq_log = np.vstack((dq_log, [t, deg(dq[0]), deg(dq[1])]))
        ddq_log = np.vstack((ddq_log, [t, deg(ddq[0]), deg(ddq[1])]))

        # p_log = np.vstack((p_log, [p[0], p[2], p[0], p[2]]))
        p_log = np.vstack((p_log, [p[0], tau_int[1], p[0], p[2]]))

        downsmpl_log = 0
    # endof logging

    conf.viz.display(q)
    loop_tend = time.time()
    ellapsed = loop_tend - loop_tbegin

    sleep_dt = max(0, conf.dt - ellapsed)
    time.sleep(sleep_dt)
    t += conf.dt
# END OF SIMULATION

plt.figure()
plt.plot(q_log[:, 0], p_log[:, 1])
plt.grid()
plt.show()

if show_plots:
    print("Simulation ended, here comes the plots...")
    plt.figure()
    plt.suptitle('Joint Positions')
    plt.subplot(2, 1, 1)
    plt.plot(q_log[:, 0], q_log[:, 1])
    plt.plot(q_log[:, 0], qdes_log[:, 0])
    plt.plot(q_log[:, 0], deg(q0[0]) * np.ones(q_log.shape))
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(q_log[:, 0], q_log[:, 2])
    plt.plot(q_log[:, 0], qdes_log[:, 1])
    plt.plot(q_log[:, 0], deg(q0[1]) * np.ones(q_log.shape))
    plt.grid()
    plt.show()

    plt.figure()
    plt.suptitle('Joint Velocities')
    plt.subplot(2, 1, 1)
    plt.plot(dq_log[:, 0], dq_log[:, 1])
    plt.plot(q_log[:, 0], dqdes_log[:, 0])
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.plot(dq_log[:, 0], dq_log[:, 2])
    plt.plot(q_log[:, 0], dqdes_log[:, 1])
    plt.grid()
    plt.show()
# endof plots

# FINALLY!!!
