import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.setrecursionlimit(500000)
sns.set_theme()

##############
## Settings ##
##############

## Sim Variables ##

Bx = 1 #100# 0.001 #0.98
n0 = 0.
v = 1./3.
t = 1./2.

dB0_fac = 0.1
dB0 = 8. * t**2 / (135. * v) * dB0_fac

dB0 = 0.04

t0 = 0.0001
dt = 0.2
tmax = 0.01

figsize=(10, 5)

## Initialization settings ##

# Lattice Vectors
G = np.array([
    [- np.sqrt(3.) / 2., -0.5],
    [0., 1.],
    [np.sqrt(3.) / 2., -0.5]
])

limit_radius = 10. # if > 0 -> caps initialization at this radius around 0
eta_const_val = (t + np.sqrt(t**2 - 15 * v * dB0)) / 15 * v
eta_const_val = 0.01608
add_eta = 0.
eps = 3. * 3.14

## Coordinate Variables ##

lim = 50 # x & y will go [-lim, lim]
M = 100 # The number of values between the interval in x and y
theta_N = 500

## Plot settings ##

frame_time = 0 # time between frames of simulation steps
frames = 1
conturf_cbar_N = 100 # how many colors are displayed in color bar

###################
## Sim Functions ##
###################

def amp_abs_sq_sum(eta_i):

    global etas

    sum_ = np.zeros(etas[0].shape, dtype=complex)
    for eta_j in range(G.shape[0]):

        is_eta_i = int(eta_i == eta_j)
        sum_ += (2. - is_eta_i) * etas[eta_j] * np.conj(etas[eta_j])

    return sum_

def g_sq_hat_fnc(eta_i):

    global G, kxm, kym, kx2, ky2

    ret = kx2 + ky2
    ret += 2. * G[eta_i, 0] * kxm
    ret += 2. * G[eta_i, 1] * kym

    return ret**2

def n_hat(eta_i):

    global etas, D, C, B

    poss_eta_is = set([i for i in range(G.shape[0])])
    other_etas = list(poss_eta_is.difference({eta_i}))

    eta_conj1 = np.conj(etas[other_etas[0]])
    eta_conj2 = np.conj(etas[other_etas[1]])

    n = 3. * D * amp_abs_sq_sum(eta_i) * etas[eta_i]
    n += 2. * C * eta_conj1 * eta_conj2
    n = np.fft.fft2(n)

    return -1. * n * np.linalg.norm(G[eta_i])**2

def lagr_hat(eta_i):

    global etas, A, B, g_sq_hat
    lagr = A * g_sq_hat[eta_i] + B

    return -1. * lagr * np.linalg.norm(G[eta_i])**2

def eta_routine(eta_i, dt):

    global etas

    lagr = lagr_hat(eta_i)
    n = n_hat(eta_i)

    exp_lagr = np.exp(lagr * dt)

    n_eta = exp_lagr * np.fft.fft2(etas[eta_i])
    n_eta += ((exp_lagr - 1.) / lagr) * n

    return np.fft.ifft2(n_eta)

def run_one_step(dt):

    global etas, G

    n_etas = np.zeros(etas.shape, dtype=complex)

    for eta_i in range(G.shape[0]):
        n_etas[eta_i,:,:] = eta_routine(eta_i, dt)

    etas = n_etas

def get_surf_en(thetas):

    global etas, xm, ym, G, A

    # grid spacing to get tolerance for
    # selecting relevant fields
    # equal spacing assumed
    h = np.diff(xm[0])[0]
    h_r = lambda r, h: 2 * np.arcsin(h / (2. * r))

    # get relevant angles
    angles = np.arctan2(ym, xm)
    ang_small_0 = angles < 0 # for -y --> negative angles --> correct
    angles[ang_small_0] = 2. * np.pi - np.abs(angles[ang_small_0])

    radii = np.sqrt(xm**2 + ym**2)

    int_sum = np.zeros(thetas.shape)

    for theta_i, theta in enumerate(thetas):

        rel_angles = np.logical_and(
            angles <= theta + h,
            angles >= theta - h
        )

        for eta_i in range(G.shape[0]):

            xy = np.real(np.array([
                radii[rel_angles],
                etas[eta_i][rel_angles]
            ]))

            xy = xy[:,xy[0].argsort()]

            deta = np.gradient(xy[1])
            d2eta = np.gradient(deta)

            curv = d2eta * (1. + deta**2)**(-1.5)

            int_p1 = 8 * A * (G[eta_i, 0] * np.cos(theta) + G[eta_i, 1] * np.sin(theta))**2
            int_p1 = np.trapz(int_p1 * deta**2, xy[0])

            int_p2 = np.trapz(4 * A * curv**2 * deta**4)

            int_sum[theta_i] += int_p1 + int_p2

    return int_sum

def get_stiffness(surf_en, thetas):
    return surf_en + np.gradient(np.gradient(surf_en))

####################
## Initialization ##
####################

## Constants ##

A = Bx
B = dB0 - 2. * t * n0 + 3. * v * n0**2
C = -t - 3. * n0
D = v

## Real Coords ##

x = np.linspace(-lim, lim, M)
y = x
xm, ym = np.meshgrid(x, y)

## Freq. Coords ##

kx = np.fft.fftfreq(len(x), np.diff(x)[0])
ky = kx # assuming even spacing

kxm, kym = np.meshgrid(kx, ky)

kx2 = np.power(kxm, 2)
ky2 = np.power(kym, 2)

## Amplitudes ##

etas = np.ones((G.shape[0], xm.shape[0], xm.shape[1]), dtype=complex)

def tanhmin(rads, eps):
    return .5 * (1. + np.tanh(-3. * rads / eps))

if (limit_radius > 0. ):

    rad = np.sqrt(xm**2 + ym**2) - limit_radius
    tan = tanhmin(rad, eps)

    for i in range(G.shape[0]):
        etas[i,:,:] *= tan

etas += add_eta

g_sq_hat = np.zeros(etas.shape, dtype=complex)
for eta_i in range(G.shape[0]):
    g_sq_hat[eta_i] = g_sq_hat_fnc(eta_i)

##########
## Plot ##
##########

fig = plt.figure()
ax = plt.subplot(131)
ax.set_aspect("equal")

ax2 = plt.subplot(132, projection="polar")
ax2.set_aspect("equal")
ax3 = plt.subplot(133, projection="polar")
ax3.set_aspect("equal")

div = make_axes_locatable(ax)
cax = div.append_axes("right", "5%", "5%")

time = t0
frame_i = 1

def plot(frame):

    global ax, etas, cax, dt, frame_i, dt, Bx, dB0, n0, v, t, time, ax2, ax3, theta_N

    eta_sum = np.zeros(etas[0].shape, dtype=float)
    for eta_i in range(G.shape[0]):
        eta_sum += np.real(etas[eta_i] * np.conj(etas[eta_i]))

    thetas = np.linspace(0, 2 * np.pi, theta_N)
    surf_en = get_surf_en(thetas)
    stiff = get_stiffness(surf_en, thetas)

    surf_en = surf_en / np.max(surf_en)
    stiff = stiff / np.max(stiff)

    ax.cla()
    ax2.cla()
    ax3.cla()

    area = np.sum((eta_sum > 0).astype(int))

    gen_title = r"$\sum_m|\eta_m|^2$"
    cont = ax.contourf(xm, ym, eta_sum, conturf_cbar_N)
    ax.set_title(f"{area}\n{gen_title}\ntime = {time:.4f}, i = {frame_i}, dt={dt:.4f}\n$B^x={Bx:.4f}, \Delta B^0={dB0:.4f}, n_0={n0:.4f}, v={v:.4f}, t={t:.4f}$")
    fig.colorbar(cont, cax=cax)

    ax2.plot(thetas, surf_en)
    ax2.set_title("Surface Energy")
    ax2.set_xticks([])
    ax2.set_yticklabels([])

    ax3.plot(thetas, stiff)
    ax3.set_title("Stiffness")
    ax3.set_xticks([])
    ax3.set_yticklabels([])

    time += dt
    frame_i += 1

    run_one_step(dt)

ani = FuncAnimation(plt.gcf(), plot, interval=frame_time, frames=frames)
plt.show()
