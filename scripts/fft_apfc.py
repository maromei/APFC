import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys

sys.setrecursionlimit(5000)
sns.set_theme()

##############
## Settings ##
##############

## Sim Variables ##

Bx = 0.98
n0 = 0.
v = 1./3.
t = 1./2.

dB0_fac = 10
dB0 = 8. * t**2 / (135. * v) * dB0_fac

dB0 = - 0.01

t0 = 0.0001
dt = 0.5
tmax = 0.01

figsize=(15, 10)

## Initialization settings ##

# Lattice Vectors
G = np.array([
    [- np.sqrt(3.) / 2., -0.5],
    [0., 1.],
    [np.sqrt(3.) / 2., -0.5]
])

init_type = "const" # random, gauss, wave, wave-gauss, const,  // seed_collection
limit_radius = 16. # if > 0 -> caps initialization at this radius around 0

eta_const_val = (t + np.sqrt(t**2 - 15 * v * dB0)) / 15 * v

print(eta_const_val)

add_eta = 0.

seed_colection = np.array([
    [0, 1],
    [1, 0],
    [-1, -1]
])

wave_f = np.array([1, 10, 0.2]) # plain wave frequencies in wave/wave-gauss init
another_factor = 100
wave_im_offset = np.array([ # offset for im part of imaginary part in wave/wave-gauss
    G[0] * another_factor * 1 / wave_f[0],
    G[1] * another_factor * 1 / wave_f[1],
    G[2] * another_factor * 1 / wave_f[2]
])
wave_im_scal = 0. # scalar to multiply offset in imaginary part in wave/wave-gauss

gauss_sigma = np.array([10, 10, 10]) # for "gauss"/"gauss-wave" initialization sigma
gauss_offsets = 20. * np.array([
    G[0], G[1], G[2]
])
gauss_im_offset = np.array([ # offset for im part of imaginary part in gauss/wave-gauss
    [0., 0., 0.,],
    [0., 0., 0.,],
    [0., 0., 0.,]
])
gauss_im_scal = 0. # scalar to multiply offset in imaginary part in gauss/wave-gauss

eps = 3. * np.pi # interface width if capped init.

## Coordinate Variables ##

lim = 200 # x & y will go [-lim, lim]
M = 501 # The number of values between the interval in x and y

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

def random_amplitude(scal_fac, shape):
    """Generate Random Amplitudes

    Creates a complex numpy array with the given shape.
    Each entry will be chosen form a uniform random interval of [0, 1].

    Parameters
    ----------

    scal_fac : float
        A scalar value to scale the random value.
    shape : Iterable[2]
        The shape of the array

    Returns
    -------
    Random complex numpy array
    """

    return scal_fac * np.array([
        [
            complex(np.random.uniform(), np.random.uniform())
            for _ in range(shape[0])
        ] for _ in range(shape[1])
    ])

def norm_dist(offset, sigma):
    """Normal Distribution in shape of global meshgrid x

    Parameters
    ----------
    offset : np.array[2]
        Mean value / center offset
    sigma : float
        variance
    """

    global xm, ym

    prefac = 1. / (sigma * 2. * np.pi)
    expon = -1 * ( (xm - offset[0])**2 + (ym - offset[1])**2) / (2. * sigma**2)

    return prefac * np.exp(expon)

def wave(k, w, offset_x=None, offset_y=None):
    """Generate pure plain sin wave in shape of global meshgrid x

    Paraemters
    ----------
    k : np.array[2]
        direction of wave
    w : float
        frequency
    offset_x : None or np.array[xm.shape]
        offset in x for wave
    offset_y : None or np.array[ym.shape]
        offset in y for wave
    """
    global xm, ym

    if offset_x is None:
        offset_x = np.zeros(xm.shape)
    if offset_y is None:
        offset_y = np.zeros(ym.shape)

    return np.sin(w * k[0] * (xm + offset_x) + w * k[1] * (ym + offset_y))

def tanhmin(rads, eps):
    return .5 * (1. + np.tanh(-3. * rads / eps))

# First index is the eta, other 2 indeces are x and y coordinates
# etas[eta_i, x, y]
etas = np.zeros((G.shape[0], xm.shape[0], xm.shape[1]), dtype=complex)
if init_type == "random":

    for eta_i in range(G.shape[0]):
        etas[eta_i,:,:] = random_amplitude(1., (etas.shape[1], etas.shape[2]))

elif init_type == "wave-gauss":

    for eta_i in range(G.shape[0]):
        etas[eta_i,:,:] = norm_dist(gauss_offsets[eta_i], gauss_sigma[eta_i])
        etas[eta_i,:,:] += complex(0, 1) * gauss_im_scal * norm_dist(gauss_im_offset[eta_i], gauss_sigma[eta_i])
        etas[eta_i,:,:] += wave(G[eta_i], wave_f[eta_i])
        etas[eta_i,:,:] += complex(0, 1) * wave_im_scal * wave(G[eta_i], wave_f[eta_i])

elif init_type == "wave":

    for eta_i in range(G.shape[0]):
        etas[eta_i,:,:] = wave(G[eta_i], wave_f[eta_i])
        etas[eta_i,:,:] += complex(0, 1) * wave_im_scal * wave(G[eta_i], wave_f[eta_i], wave_im_offset[eta_i, 0], wave_im_offset[eta_i, 1])

elif init_type == "gauss":

    for eta_i in range(G.shape[0]):
        etas[eta_i,:,:] = norm_dist(gauss_offsets[eta_i], gauss_sigma[eta_i])
        etas[eta_i,:,:] += complex(0, 1) * gauss_im_scal * norm_dist(gauss_im_offset[eta_i], gauss_sigma[eta_i])

elif init_type == "const":

    etas = np.ones(etas.shape, dtype=complex) * eta_const_val

elif init_type == "seed_collection":

    n_etas = np.zeros(etas.shape, dtype=complex)
    for i, place in enumerate(seed_colection):
        for eta_i in range(G.shape[0]):

            n_etas[eta_i,:,:] = wave(G[eta_i], wave_f[eta_i])
            n_etas[eta_i,:,:] += complex(0, 1) * wave_im_scal * wave(G[eta_i], wave_f[eta_i])

            rad = np.sqrt((xm - place[0])**2 + (ym - place[1])**2) - limit_radius
            tan = tanhmin(rad, eps)

            n_etas[eta_i,:,:] *= tan

        etas += n_etas


if (limit_radius > 0. and init_type != "seed_collection"):

    rad = np.sqrt(xm**2 + ym**2) - limit_radius
    tan = tanhmin(rad, eps)

    for i in range(G.shape[0]):
        etas[i,:,:] *= tan

etas += add_eta

g_sq_hat = np.zeros(etas.shape, dtype=complex)
for eta_i in range(G.shape[0]):
    g_sq_hat[eta_i] = g_sq_hat_fnc(eta_i)

####################
## Plot Functions ##
####################

def plot_window(frame):

    global t, time, dt, etas, xm, ym, fig, ax, frame_i, cax, conturf_cbar_N
    global Bx, dB0, n0, v

    run_one_step(dt)

    eta_sum = np.abs(etas[0])
    for eta_i in range(1, G.shape[0]):
        eta_sum += np.abs(etas[eta_i])

    ax.cla()

    cont = ax.contourf(xm, ym, eta_sum, conturf_cbar_N)
    ax.grid('on')
    ax.set_title(f"time = {time:.4f}, i = {frame_i}, dt={dt:.4f}\n$B^x={Bx:.4f}, \Delta B^0={dB0:.4f}, n_0={n0:.4f}, v={v:.4f}, t={t:.4f}$")
    fig.colorbar(cont, cax=cax)
    plt.tight_layout()

    time += dt
    frame_i += 1

def plot_window2(frame):

    global t, time, dt, etas, xm, ym, fig, ax, frame_i, cax, conturf_cbar_N
    global Bx, dB0, n0, v

    run_one_step(dt)

    eta_sum = np.zeros(etas[0].shape, dtype=complex)
    for eta_i in range(G.shape[0]):
        gdotxy = G[eta_i][0] * xm + G[eta_i][1] * ym
        eta_prod = etas[eta_i] * np.exp(complex(0, 1) * gdotxy)
        eta_sum += eta_prod + etas[eta_i] * np.exp(complex(0, -1) * gdotxy)

    eta_sum = np.abs(eta_sum)

    ax.cla()

    cont = ax.contourf(xm, ym, eta_sum, conturf_cbar_N)
    ax.grid('on')
    ax.set_title(f"time = {time:.4f}, i = {frame_i}, dt={dt:.4f}\n$B^x={Bx:.4f}, \Delta B^0={dB0:.4f}, n_0={n0:.4f}, v={v:.4f}, t={t:.4f}$")
    fig.colorbar(cont, cax=cax)
    plt.tight_layout()

    time += dt
    frame_i += 1

def plot_window_w_surface(frame):

    global t, time, dt, etas, xm, ym, fig, axs, frame_i, cax, conturf_cbar_N, cax_log
    global Bx, dB0, n0, v

    run_one_step(dt)

    #eta_sum = np.zeros(etas[0].shape, dtype=complex)
    #for eta_i in range(G.shape[0]):
    #    gdotxy = G[eta_i][0] * xm + G[eta_i][1] * ym
    #    eta_prod = etas[eta_i] * np.exp(complex(0, 1) * gdotxy)
    #    eta_sum += eta_prod + np.conj(etas[eta_i]) * np.exp(complex(0, -1) * gdotxy)
    #eta_sum = np.real(eta_sum * np.conj(eta_sum))

    eta_sum = np.zeros(etas[0].shape, dtype=complex)
    for eta_i in range(G.shape[0]):
        eta_sum += etas[eta_i] * np.conj(etas[eta_i])
    eta_sum = np.real(eta_sum).astype(float)

    """eta_sum = np.zeros(etas[0].shape, dtype=float)
    for eta_i in range(G.shape[0]):
        eta_sum += np.real(etas[eta_i] * np.conj(etas[eta_i]))"""

    thetas = np.linspace(0, 2 * np.pi, 200)

    surf_x = calc_surf_en(0, thetas)
    #surf_y = calc_surf_en(1, thetas)

    stiff_x = get_stiffness(surf_x, thetas)
    #stiff_y = get_stiffness(surf_y, thetas)

    for fig_arr in axs:
        for fig_ in fig_arr:
            fig_.cla()

    gen_title = r"$|\sum_m \eta_m \exp^{i G_m \dot r} + c.c.|^2$"
    gen_title_log = r"$\log \left(|\sum_m \eta_m \exp^{i G_m \dot r} + c.c.|^2 \right)$"

    axs[0][1].plot(thetas, surf_x / np.max(surf_x))
    axs[0][1].set_title("Surface Energy x-Axis")
    axs[0][1].set_xticks([])
    axs[0][1].set_yticklabels([])

    #axs[1][1].plot(thetas, surf_y / np.max(surf_y))
    #axs[1][1].set_title("Surface Energy y-Axis")
    #axs[1][1].set_xticks([])
    #axs[1][1].set_yticklabels([])

    axs[0][2].plot(thetas, stiff_x / np.max(stiff_x))
    axs[0][2].set_title("Stiffness x-Axis")
    axs[0][2].set_xticks([])
    axs[0][2].set_yticklabels([])

    #axs[1][2].plot(thetas, stiff_y / np.max(stiff_y))
    #axs[1][2].set_title("Stiffness y-Axis")
    #axs[1][2].set_xticks([])
    #axs[1][2].set_yticklabels([])

    cont = axs[0][0].contourf(xm, ym, eta_sum, conturf_cbar_N)
    axs[0][0].grid('on')
    #axs[0][0].set_title(f"{gen_title}\ntime = {time:.4f}, i = {frame_i}, dt={dt:.4f}\n$B^x={Bx:.4f}, \Delta B^0={dB0:.4f}, n_0={n0:.4f}, v={v:.4f}, t={t:.4f}$")
    fig.colorbar(cont, cax=cax)

    cont_log = axs[1][0].contourf(
        xm, ym, eta_sum,
        np.logspace(np.log10(eta_sum.min()), np.log10(eta_sum.max()), conturf_cbar_N),
        locator=matplotlib.ticker.LogLocator()
    )
    axs[1][0].grid('on')
    axs[1][0].set_title(f"time = {time:.4f}, i = {frame_i}, dt={dt:.4f}\n$B^x={Bx:.4f}, \Delta B^0={dB0:.4f}, n_0={n0:.4f}, v={v:.4f}, t={t:.4f}$")
    fig.colorbar(cont_log, cax=cax_log)

    #plt.tight_layout()

    time += dt
    frame_i += 1

    plt.savefig("/home/max/projects/apfc/tmp/aaaa.png")

def plot():

    global t, time, dt, etas, xm, ym, fig, ax, frame_i, cax, conturf_cbar_N
    global Bx, dB0, n0, v

    eta_sum = np.abs(etas[0])
    for eta_i in range(1, G.shape[0]):
        eta_sum += np.abs(etas[eta_i])

    ax.cla()

    cont = ax.contourf(xm, ym, eta_sum, conturf_cbar_N)
    ax.grid('on')
    ax.set_title(f"time = {time:.4f}, i = {frame_i}, dt={dt:.4f}\n$B^x={Bx:.4f}, \Delta B^0={dB0:.4f}, n_0={n0:.4f}, v={v:.4f}, t={t:.4f}$")
    fig.colorbar(cont, cax=cax)
    plt.tight_layout()

    time += dt
    frame_i += 1

#####################
## SURFACE EN TEST ##
#####################

def calc_surf_en2(axis, thetas):

    global etas, xm, ym, G

    angles = np.arctan2(ym, xm)
    ang_small_0 = angles < 0
    angles[ang_small_0] = np.pi + np.abs(angles[ang_small_0])

    radii = np.sqrt(xm**2 + ym**2)

    tol = 5e-2

    int_sum = np.zeros(thetas.shape)

    xmn = xm / radii
    ymn = ym / radii

    for angle in thetas:

        rel_fields = np.logical_and(angles <= angle + tol, angles >= angle - tol)

        xy = np.real(np.array([
            radii[rel_fields],
            etas[eta_i][rel_fields],
            xmn[rel_fields],
            ymn[rel_fields]
        ]))
        xy = xy[:,xy[0].argsort()]

        x = xy[0]
        y = xy[1]

        deta = np.gradient(y)
        d2eta = np.gradient(deta)

        curv = d2eta * (1. + deta**2)**-1.5

        int_p2 = 4 * A * curv**2 * deta**4
        int_p2 = np.trapz(int_p2)

        int_prefac = 8. * A * ( xy[2] * np.cos(angle) + xy[3] * np.sin(angle))**2
        int_p1 = np.trapz(int_prefac * deta**2)

        int_sum += int_p1 + int_p2

    return int_sum

def calc_surf_en3(axis, thetas):

    global etas, xm, ym, G

    angles = np.arctan2(ym, xm)
    ang_small_0 = angles < 0
    angles[ang_small_0] = 2. * np.pi - np.abs(angles[ang_small_0])

    radii = np.sqrt(xm**2 + ym**2)

    G_ang = np.array([complex(e[0], e[1]) for e in G])

    tol = 5e-2

    int_sum = np.zeros(thetas.shape)

    for eta_i in range(G.shape[0]):

        angle = np.angle(G_ang[eta_i])
        if angle < 0:
            angle = np.pi - np.abs(angle)

        rel_fields = np.logical_and(angles <= angle + tol, angles >= angle - tol)

        xy = np.array([
            radii[rel_fields],
            etas[eta_i][rel_fields]
        ])
        xy = xy[:,xy[0].argsort()]

        x = np.real(xy[0])
        y = np.real(xy[1])

        deta = np.gradient(y)
        d2eta = np.gradient(deta)

        curv = d2eta * (1. + deta**2)**-1.5

        int_p2 = 4 * A * curv**2 * deta**4
        int_p2 = np.trapz(int_p2)

        int_p1 = np.trapz(deta**2)
        int_prefac = 8. * A * (G[eta_i, 0] * np.cos(thetas) + G[eta_i, 1] * np.sin(thetas))**2

        int_sum += int_prefac * int_p1 + int_p2

    return int_sum

def calc_surf_en_old_used(axis, thetas):

    global xm, ym, etas, A, G

    ##################
    ## get x values ##
    ##################

    x = None
    axis_val = None

    if axis == 1:
        axis_val = np.argmin(np.abs(xm[0,:])) # value closest to 0
        x = ym[:,axis_val]
    else:
        axis_val = np.argmin(np.abs(ym[:,0])) # value closest to 0
        x = xm[axis_val,:]

    dx = np.gradient(x)
    d2x = np.gradient(dx)

    ###########################
    ## Build necessairy etas ##
    ###########################

    etas_flat = np.zeros((G.shape[0], etas[0].shape[1]), dtype=float)

    if axis == 1:
        for eta_i in range(G.shape[0]):
            etas_flat[eta_i] = np.real(etas[eta_i, :, axis_val])
    else:
        for eta_i in range(G.shape[0]):
            etas_flat[eta_i] = np.real(etas[eta_i, axis_val, :])

    detas = np.zeros((G.shape[0], etas[0].shape[1]), dtype=float)
    for eta_i in range(G.shape[0]):
        detas[eta_i] = np.gradient(etas_flat[eta_i])

    d2etas = np.zeros((G.shape[0], etas[0].shape[1]), dtype=float)
    for eta_i in range(G.shape[0]):
        d2etas[eta_i] = np.gradient(detas[eta_i])

    ####################
    ## calc curvature ##
    ####################

    curvatures = np.ones((G.shape[0], dx.shape[0]))
    for eta_i in range(G.shape[0]):
        curv_abs = d2etas[eta_i]
        curv_pow = ( 1 + detas[eta_i]**2 )**1.5
        curvatures[eta_i] = curv_abs / curv_pow

    #######################
    ## 2nd integral part ##
    #######################

    int_p2 = np.zeros(detas.shape[1])
    for eta_i in range(G.shape[0]):
        int_p2 = np.add(int_p2, curvatures[eta_i]**2 * detas[eta_i]**4)
    int_p2 *= 4. * A

    int_p2 = np.trapz(int_p2, x)

    #######################
    ## 1st integral part ##
    #######################

    int_p1 = np.zeros(G.shape[0])
    for eta_i in range(G.shape[0]):
        int_p1[eta_i] = np.trapz(detas[eta_i]**2, x)

    ##############
    ## Assemble ##
    ##############

    return get_surface_energy(thetas, int_p1, int_p2)

def calc_surf_en(axis, thetas):

    global etas, xm, ym, G, A

    int_sum = np.zeros(thetas.shape)

    h = np.diff(xm[0])[0]

    for theta_i, theta in enumerate(thetas):

        rot_xm = np.cos(theta) * xm - np.sin(theta) * ym
        rot_ym = np.sin(theta) * xm + np.cos(theta) * ym

        rel_fields = np.logical_and(
            np.abs(rot_ym) < h,
            rot_xm > 0
        )

        radii = np.sqrt(xm**2 + ym**2)

        for eta_i in range(G.shape[0]):

            xy = np.real(np.array([
                radii[rel_fields],
                etas[eta_i][rel_fields]
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

def get_surface_energy(thetas, int_p1, int_p2):

    global G, A

    p1 = np.zeros(thetas.shape)
    for eta_i in range(G.shape[0]):
        p1 += 8. * A * (G[eta_i, 0] * np.cos(thetas) + G[eta_i, 1] * np.sin(thetas))**2 * int_p1[eta_i]

    return p1 + int_p2

def get_stiffness(surf_en, thetas):
    dx = np.diff(thetas)[0]
    return surf_en + np.gradient(np.gradient(surf_en))

################
## Run / Plot ##
################

nstep = int(round(tmax / dt))

time = t0
frame_i = 1

"""
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect("equal")

div = make_axes_locatable(ax)
cax = div.append_axes("right", "5%", "5%")
"""

fig = plt.figure(figsize=figsize)

axs = [
    [
        plt.subplot(231),
        plt.subplot(232, projection="polar"),
        plt.subplot(233, projection="polar")
    ],
    [
        plt.subplot(234),
        plt.subplot(235, projection="polar"),
        plt.subplot(236, projection="polar")
    ],
]

for fig_arr in axs:
    for fig_ in fig_arr:
        fig_.set_aspect("equal")

div = make_axes_locatable(axs[0][0])
cax = div.append_axes("right", "5%", "5%")

div_log = make_axes_locatable(axs[1][0])
cax_log = div_log.append_axes("right", "5%", "5%")

ani = FuncAnimation(plt.gcf(), plot_window_w_surface, interval=frame_time, frames=frames)
plt.show()

#fig.savefig(f"data/{init_type}Bx{Bx:.4f}dB0{dB0:.4f}n0{n0:.4f}v{v:.4f}t{t:.4f}time{time:.4f}i{frame_i}.svg")
#fig.savefig(f"data/{init_type}Bx{Bx:.4f}dB0{dB0:.4f}n0{n0:.4f}v{v:.4f}t{t:.4f}time{time:.4f}i{frame_i}.png")
