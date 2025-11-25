# %%
"""
Variable Duration Pendulum Swing Up
===================================

Objectives
----------

- Demonstrate how to make the simulation duration variable.
- Show how to use the NumPy backend which solves the problem without needing
  just-in-time C compilation.
- Demonstrate plotting the constraint violations as subplots.
- Show how to access the problem's attributes and methods inside the objective
  and gradient functions to help simplify constructing the objective and
  gradient values.

Introduction
------------

Given a simple pendulum that is driven by a torque about its joint axis, swing
the pendulum from hanging down to standing up in a minimal amount of time using
minimal input energy with a bounded torque magnitude.

Notes
-----

There is a mechanically identiclal system in the examples-gallery/beginner
folder that uses a fixed duration.

"""
import os
import numpy as np
import sympy as sm
import sympy.physics.mechanics as me
from opty import Problem
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# %%
# Start with defining the fixed duration and number of nodes.
target_angle = np.pi
num_nodes = 501

# %%
# Symbolic equations of motion
m, g, d, t, h = sm.symbols('m, g, d, t, h', real=True)
theta, omega, T, Tdt = sm.symbols('theta, omega, T, Tdt', cls=sm.Function)
delay = sm.symbols('delay', cls=sm.Function)

state_symbols = (theta(t), omega(t), T(t), Tdt(t))
constant_symbols = (m, g, d)

eom = sm.Matrix([
    theta(t).diff() - omega(t),
    m*d**2*omega(t).diff() + m*g*d*sm.sin(theta(t)) - (T(t) - delay(t)),
    # The derivative of T is needed.
    Tdt(t) - T(t).diff()
])
sm.pprint(eom)

# %%
# Specify the known system parameters.
par_map = {
    m: 1.0,
    g: 9.81,
    d: 1.0,
}


# %%
# Specify the objective function and it's gradient. In this case, make the
# problem instance the first argument and it is available to use inside the
# these functions. This shows how to use ``.parse_free()``,
# ``.extract_values()``, and ``.fill_free()`` to manage the numerical vectors.


def obj(prob, free):
    """Minimize the sum of the squares of the control torque."""
    return np.sum(free[2*num_nodes:3*num_nodes]**2) * free[-1]


def obj_grad(prob, free):
    grad = np.zeros_like(free)
    grad[2*num_nodes:3*num_nodes] = 2 * free[2*num_nodes:3*num_nodes] * free[-1]
    grad[-1] = np.sum(free[2*num_nodes:3*num_nodes]**2)
    return grad


# %%
# Specify the symbolic instance constraints, i.e. initial and end conditions
# using node numbers 0 to N - 1
instance_constraints = (theta(0*h),
                        theta((num_nodes - 1)*h) - target_angle,
                        omega(0*h),
                        omega((num_nodes - 1)*h))

delta_T = 0.5 # The delay before delay should start. 180 sec in Nevilles problem


def delay_traj(free):
    # number of initial nodes to be filled with zeros
    zero_time = min(int(np.floor(delta_T / free[-1])), num_nodes)
    s1 = list(np.zeros(zero_time))
    s2 = [free[2*num_nodes + i] for i in range(num_nodes - zero_time)]
    return np.array(s1 + s2)


def delaydt_traj(free):
    zero_time = min(int(np.floor(delta_T / free[-1])), num_nodes)
    s1 = list(np.zeros(zero_time))
    s2 = [free[3*num_nodes + i] for i in range(num_nodes - zero_time)]
    return np.array(s1 + s2)


# %%
# Create an optimization problem. If the backend is set to ``numpy``, no C
# compiler is needed and the problem can be solved using pure Python code.
# There is a large performance loss but for simple problems performance may not
# be a concern.
prob = Problem(obj, obj_grad, eom, state_symbols, num_nodes, h,
               known_parameter_map=par_map,
               instance_constraints=instance_constraints,
               time_symbol=t,
               bounds={T(t): (-2.0, 2.0), h: (0.0, 0.5)},
               known_trajectory_map={
                   delay(t): delay_traj,
                   delay(t).diff(t): delaydt_traj
                },
)

# %%
# Use existing solution if available else pick a reasonable initial guess and
# solve the problem. Use approximately zero as an initial guess to avoid
# divide-by-zero, and solve the problem.
initial_guess = np.full(prob.num_free, 1e-10)
solution, info = prob.solve(initial_guess)
print(info['status_msg'])
print('first obj val:', info['obj_val'])
_ = prob.plot_trajectories(solution)

# %%
# Plot the optimal state and input trajectories.
_ = prob.plot_trajectories(solution)

# %%
# Plot the constraint violations.
_ = prob.plot_constraint_violations(solution, subplots=True)

# %%
# objectiva value
_ = prob.plot_objective_value()

# %%
# Animate the pendulum swing up.
interval_value = solution[-1]
time = prob.time_vector(solution=solution)
angle = solution[:num_nodes]

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-2, 2), ylim=(-2, 2))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_template = 'time = {:0.1f}s'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)


# sphinx_gallery_thumbnail_number = 3
def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text


def animate(i):
    x = [0, par_map[d]*np.sin(angle[i])]
    y = [0, -par_map[d]*np.cos(angle[i])]

    line.set_data(x, y)
    time_text.set_text(time_template.format(i*interval_value))
    return line, time_text


ani = animation.FuncAnimation(fig, animate, range(0, num_nodes, 4),
                              interval=int(interval_value*1000*4),
                              blit=True, init_func=init)

plt.show()
