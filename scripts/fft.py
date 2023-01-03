import numpy as np
import seaborn as sns
import sys
import json
import scipy

sys.setrecursionlimit(5000)
sns.set_theme()

##############
## Settings ##
##############

base_path = "/home/max/projects/apfc/data/tmp2"
write_every_i = 100

## Sim Variables ##

Bx = 0.98
n0 = 0.
v = 1./3.
t = 1./2.

dB0_fac = 0.95
dB0 = 8. * t**2 / (135. * v) * dB0_fac

dB0 = 0.01

t0 = 0.0001
dt = 0.5
numT = 7000

## Initialization settings ##

# Lattice Vectors
G = np.array([
    [- np.sqrt(3.) / 2., -0.5],
    [0., 1.],
    [np.sqrt(3.) / 2., -0.5]
])

limit_radius = 16. # if > 0 -> caps initialization at this radius around 0

#eta_const_val = (t + np.sqrt(t**2 - 15 * v * dB0)) / 15 * v
eta_const_val = 4. * t / (45. * v)

print(eta_const_val, dB0)

eps = 3. * np.pi # interface width if capped init.

## Coordinate Variables ##

lim = 400 # x & y will go [-lim, lim]
M = 1001 # The number of values between the interval in x and y

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


def tanhmin(rads, eps):
    return .5 * (1. + np.tanh(-3. * rads / eps))

# First index is the eta, other 2 indeces are x and y coordinates
# etas[eta_i, x, y]
etas = np.zeros((G.shape[0], xm.shape[0], xm.shape[1]), dtype=complex)
etas = np.ones(etas.shape, dtype=complex) * eta_const_val

rad = np.sqrt(xm**2 + ym**2) - limit_radius
tan = tanhmin(rad, eps)

for i in range(G.shape[0]):
    etas[i,:,:] *= tan

g_sq_hat = np.zeros(etas.shape, dtype=complex)
for eta_i in range(G.shape[0]):
    g_sq_hat[eta_i] = g_sq_hat_fnc(eta_i)

################
## Run / Plot ##
################

def write_config():

    global base_path
    global Bx, n0, v, t, dB0
    global M, lim, dt, limit_radius
    global eta_const_val, eps
    global A, B, C, D
    global G

    conf = {
        "Bx": Bx,
        "n0": n0,
        "v": v,
        "t": t,
        "dB0": dB0,
        "numPts": M,
        "xlim": lim,
        "dt": dt,
        "initRadius": limit_radius,
        "initEta": eta_const_val,
        "interfaceWidth": eps,
        "A": A,
        "B": B,
        "C": C,
        "D": D,
        "G": [
            [i for i in e] for e in G
        ]
    }

    with open(f"{base_path}/config.json", "w+") as f:
        json.dump(conf, f)

def write_etas():

    global etas, base_path, G

    for eta_i in range(G.shape[0]):

        lst = etas[eta_i].flatten()
        lst = np.real(lst).astype(str)

        out_str = ",".join(lst.tolist())
        out_str += ",\n"

        with open(f"{base_path}/out_{eta_i}.txt", "a+") as f:
            f.write(out_str)

def reset_files():

    global base_path, G

    for eta_i in range(G.shape[0]):
        with open(f"{base_path}/out_{eta_i}.txt", "w") as f:
            f.write("")

reset_files()
write_config()

#write_etas()
#write_etas()

for i in range(numT):

    run_one_step(dt)

    if i % write_every_i == 0:
        write_etas()
        perc = i / numT * 100
        sys.stdout.write(f"Progress: {perc:.4f}%\r")
        sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()
