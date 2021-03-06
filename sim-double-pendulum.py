import numpy
import pinocchio as pin
import hppfcl as fcl
import numpy as np
import matplotlib.pyplot as plt
from math import *
import time
import sys
# controllers_funct already imports conf and AdmShaping
from controllers_funct import *

def deg(arg):
    return degrees(arg)


def saturation(val, lmt):
    return max(min(val, lmt), -lmt)


# endof deg

show_plots = True
interaction_enable = True

# * Geometric parameters for joint error association *
lgth_1 = 1.
lgth_2 = 1.

# Physical parameters
jointsFriction = np.array([[1.1, 0], [0, 2.4]])

# Input
input_type = 'sin'
freqs = np.array([0.45, 0.4])
amps = np.array([0.8, 0.6])
phs = np.array([.0 * pi * (90 / 180), pi * (45 / 180)])
ampsxfreqs = np.multiply(amps, freqs)
# print(ampsxfreqs)

# Environment Interaction
# Task Space Int Dyn
tStiffness = np.array([20.0, .0, .0])  # N/m
tDamping = np.array([0.23, .0, .0])  # N.s/m
# Joint Space Int Dyn
intK = 100*104.
jStiffness = np.array([[intK, 0], [intK, intK]])  # N/rad, including here the joint angle error association
jDamping = np.eye(conf.Model.nv) * 0.104  # N.s/rad
Ksea = np.eye(conf.Model.nv) * 104.0  # N/rad
SeaMax = 104.0 * pi * (7/180)

# Desired states variables
q_des = np.array([pi * (178 / 180), pi * (90 / 180)])
dq_des = np.zeros(conf.Model.nv)
ddq_des = np.zeros(conf.Model.nv)

# q = pin.randomConfiguration(conf.Model)
# Initial states, q0, dq0
q   = np.array([pi * (178 / 180), pi * (25 / 180)]) + np.array([amps[0] * sin(phs[0]), amps[1] * sin(phs[1])])
dq  = np.zeros(conf.Model.nv)
ddq = np.zeros(conf.Model.nv)
q0  = q.copy()
dq0 = 2 * pi * np.array([ampsxfreqs[0], ampsxfreqs[1]])

# Human states, "qh", and properties (K, D)
qh   = q0
dqh  = np.zeros(conf.humModel.nv)
ddqh = np.zeros(conf.humModel.nv)
humStiffness = conf.humStiffness
humDamping = conf.humDamping

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
humjstates_log = np.empty([1, 1 + 3*conf.humModel.nq]) * nan

p_log = np.empty([1, 2 * conf.Model.nq]) * nan

data_sim = conf.Model.createData()
data_hum = conf.humModel.createData()
# print(conf.Model.getFrameId('bar_1'))
# base_frame = conf.Model.frames[0]
# print(base_frame)

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
    # Pinocchio Data for the Robot
    pin.computeAllTerms(conf.Model, data_sim, q, dq)
    Mq = data_sim.M
    hq = data_sim.C
    grav = data_sim.g
    # Pinocchio Data for the Human
    pin.computeAllTerms(conf.humModel, data_hum, qh, dqh)
    humMq = data_hum.M

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

    # SEA:
    # tau_sea = Ksea.dot(q - q_rlx)  # ? or:
    tau_sea = Ksea.dot(q_des - q)
    tau_sea[0] = saturation(tau_sea[0], SeaMax)
    tau_sea[1] = saturation(tau_sea[1], SeaMax)

    # Joints Friction
    tau_frict = jointsFriction.dot(dq)  # + 0.05*np.multiply(dq, dq)
    tau_frict_h = jointsFriction.dot(dqh)

    # Joint Space Int - Human-Exo Interaction
    tau_int = jStiffness.dot(qh - q)
    # angular error association:
    tau_int_corr = np.array([0, (lgth_1/lgth_2) * (sin(q[0]) - sin(qh[0]))]) * intK
    tau_int += tau_int_corr
    #tau_int[0] = saturation(tau_int[0], SeaMax)
    #tau_int[1] = saturation(tau_int[1], SeaMax)

    # Controller
    UserStates = [qh, dqh, ddqh]
    RobotStates = [q, dq, ddq]
    DesiredStates = [q_des, dq_des, ddq_des]
    tau_control = robot_controller(data_sim, DesiredStates, UserStates, RobotStates, tau_int)

    # -- Human Body Control: -- #
    hum_input = humMq.dot(ddq_des + humStiffness.dot(q_des - qh) + humDamping.dot(dq_des - dqh)) + data_hum.nle

    # TODO: modelo do hum est?? super lento, investigar!!
    if interaction_enable:
        Tau   = tau_control + tau_int
        Tau_h = hum_input - tau_int
    else:
        Tau = tau_control - tau_frict
        Tau_h = hum_input - tau_frict_h

    # Forward Dynamics (simulation)
    ddq  = pin.aba(conf.Model, data_sim, q, dq, Tau)
    ddqh = pin.aba(conf.humModel, data_hum, qh, dqh, Tau_h)
    # Mq_inv = np.linalg.inv(Mq)
    # ddq = Mq_inv.dot(Tau - data_sim.nle)

    dq += ddq*conf.dt
    q = pin.integrate(conf.Model, q, dq*conf.dt)
    dqh = ddqh*conf.dt
    qh  = pin.integrate(conf.humModel, qh, dqh*conf.dt)
    # Forward Euler Integration with Trapeziodal Rule
    #dq += (ddq_last + ddq) * conf.dt * 0.5
    #ddq_last = ddq.copy()
    #q += (dq_last + dq) * conf.dt * 0.5
    #dq_last = dq.copy()

    int_2 = 104. * (qh[0] - q[0] + qh[1] - q[1])
    # Log variables
    downsmpl_log += 1
    if downsmpl_log > downsmpl_factor and show_plots:
        # Robot States log
        q_log = np.vstack((q_log, [t, deg(q[0]), deg(q[1])]))
        dq_log = np.vstack((dq_log, [t, deg(dq[0]), deg(dq[1])]))
        ddq_log = np.vstack((ddq_log, [t, deg(ddq[0]), deg(ddq[1])]))

        # human States log
        humjstates_log = np.vstack((humjstates_log, [t, deg(qh[0]),   deg(qh[1]),\
                                                        deg(dqh[0]),  deg(dqh[1]),\
                                                        deg(ddqh[0]), deg(ddqh[1])]))

        # p_log = np.vstack((p_log, [p[0], p[2], p[0], p[2]])) # log x,z
        p_log = np.vstack((p_log, [tau_int[0], tau_int[1], tau_int_corr[1], tau_control[1]]))
        #print(humMq)
        downsmpl_log = 0
    # endof logging

    conf.viz.display(q)
    conf.viz_hum.display(qh)
    loop_tend = time.time()
    ellapsed = loop_tend - loop_tbegin

    sleep_dt = max(0, conf.dt - ellapsed)
    #print(sleep_dt)
    time.sleep(sleep_dt)
    t += conf.dt
# END OF SIMULATION

IntRms1 = (1/intK)*np.sqrt(np.mean(p_log[1:, 0]**2)) # dispensar o primeiro pois eh nan
IntRms1 = deg(IntRms1)
IntRms2 = (1/intK)*np.sqrt(np.mean(p_log[1:, 1]**2))
IntRms2 = deg(IntRms2)
print('Delta Ang RMS: ' + str(IntRms1) + ', ' + str(IntRms2))

plt.figure()
plt.plot(q_log[:, 0], p_log[:, 0])
plt.plot(q_log[:, 0], p_log[:, 1])
plt.grid()
plt.show(block=False)

if show_plots:
    print("Simulation ended, here comes the plots...")
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title('Robot q (deg)')
    axs[0, 0].plot(q_log[:, 0], q_log[:, 1])
    axs[0, 0].plot(humjstates_log[:, 0], humjstates_log[:, 1],'--')
    # axs[0, 0].plot(q_log[:, 0], deg(q0[0]) * np.ones(q_log.shape))
    axs[0, 0].grid()

    axs[1, 0].plot(q_log[:, 0], q_log[:, 2])
    axs[1, 0].plot(humjstates_log[:, 0], humjstates_log[:, 2],'--')
    # axs[1, 0].plot(q_log[:, 0], deg(q0[1]) * np.ones(q_log.shape))
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 0].grid()

    axs[0, 1].set_title('Robot dq (deg/s)')
    axs[0, 1].plot(dq_log[:, 0], dq_log[:, 1])
    axs[0, 1].plot(humjstates_log[:, 0], humjstates_log[:, 3],'--')
    axs[0, 1].grid()

    axs[1, 1].plot(dq_log[:, 0], dq_log[:, 2])
    axs[1, 1].plot(humjstates_log[:, 0], humjstates_log[:, 4],'--')
    axs[1, 1].set_xlabel('time (s)')
    axs[1, 1].grid()
    plt.show(block=False)

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].set_title('Human q (deg)')
    axs[0, 0].plot(humjstates_log[:, 0], humjstates_log[:, 1])
    axs[0, 0].plot(q_log[:, 0], qdes_log[:, 0])
    axs[0, 0].plot(q_log[:, 0], deg(q0[0]) * np.ones(q_log.shape))
    axs[0, 0].grid()

    axs[1, 0].plot(humjstates_log[:, 0], humjstates_log[:, 2])
    axs[1, 0].plot(q_log[:, 0], qdes_log[:, 1])
    axs[1, 0].plot(q_log[:, 0], deg(q0[1]) * np.ones(q_log.shape))
    axs[1, 0].set_xlabel('time (s)')
    axs[1, 0].grid()

    axs[0, 1].set_title('Human dq (deg/s)')
    axs[0, 1].plot(humjstates_log[:, 0], humjstates_log[:, 3])
    axs[0, 1].plot(q_log[:, 0], dqdes_log[:, 0])
    axs[0, 1].grid()

    axs[1, 1].plot(humjstates_log[:, 0], humjstates_log[:, 4])
    axs[1, 1].plot(q_log[:, 0], dqdes_log[:, 1])
    axs[1, 1].set_xlabel('time (s)')
    axs[1, 1].grid()
    plt.show(block=True)
# endof plots

# FINALLY!!!
