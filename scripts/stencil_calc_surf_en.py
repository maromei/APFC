import numpy as np
import os
import json
import scipy
import sys

import matplotlib.pyplot as plt

base_path = "/home/max/projects/apfc/data/stencil"

def get_stiffness(surf, dt):
    return surf + np.gradient(np.gradient(surf, dt), dt)

dims = (501, 501)

show_time = 1
ignore_last = True

def read_complex(line, dim1, dim2):

    split = line.split("),(")
    split = [s.replace("(", "").replace(")", "") for s in split[:]]
    arr = []
    for ss in split:
        s = ss.split(",")
        arr.append(complex(float(s[0]), float(s[1])))
    arr = np.array(arr)
    arr = arr.reshape((dim1, dim2))

    return arr

def read_eta(eta_path, dim1, dim2):

    eta = []

    with open(eta_path, "r") as f:

        for line in f:
            if line == "\n" or line == "":
                break
            s = read_complex(line, dim1, dim2)
            eta.append(s)

    return eta

def write_res(path, arr):

    with open(path, "w") as f:
        for i in range(arr.shape[0]):

            fstr = arr[i].astype(str).tolist()
            f.write(",".join(fstr))

all_files_and_folders = os.listdir(base_path)
thetas_strs = [
    f for f in all_files_and_folders
    if os.path.isdir(f"{base_path}/{f}")
]
thetas = np.array(thetas_strs)
thetas = thetas.astype(float)
thetas = np.sort(thetas)

thetas_strs = np.array([f"{t:.2f}" for t in thetas])

if ignore_last:
    thetas = thetas[:-1]
    thetas_strs = thetas_strs[:-1]

# read 1 eta for dimensions
eta_test = read_eta(f"{base_path}/{thetas_strs[0]}/out_0.txt", dims[0], dims[1])

surf_en = np.zeros((len(eta_test), thetas.shape[0]), dtype=complex)
stiff = np.zeros((len(eta_test), thetas.shape[0]), dtype=complex)

for theta_i, theta in enumerate(thetas):

    with open(f"{base_path}/{thetas_strs[theta_i]}/config.json", "r") as f:
        config = json.load(f)

    etas = [
        read_eta(f"{base_path}/{thetas_strs[theta_i]}/out_0.txt", dims[0], dims[1]),
        read_eta(f"{base_path}/{thetas_strs[theta_i]}/out_1.txt", dims[0], dims[1]),
        read_eta(f"{base_path}/{thetas_strs[theta_i]}/out_2.txt", dims[0], dims[1]),
    ]

    rows = etas[0][0].shape[0]
    cols = etas[0][0].shape[1]

    half_i = rows // 2

    x_ = np.linspace(-config["xlim"], config["xlim"], config["numPts"])
    xm, ym = np.meshgrid(x_, x_)

    xvals = xm[0]
    dx = np.diff(xvals)[0]

    G = np.array(config["G"])

    # iterate over time in etas
    for time_i in range(len(etas[0])):

        int_sum = 0.

        for eta_i in range(len(etas)):

            x = etas[eta_i][time_i][half_i,:]

            deta = np.gradient(x, dx)
            d2eta = np.gradient(deta, dx)

            curv = d2eta * (1. + deta**2)**(-1.5)

            int_p1 = 8 * config["A"] * (G[eta_i, 0] * np.cos(theta) + G[eta_i, 1] * np.sin(theta))**2
            int_p1 += scipy.integrate.simpson(int_p1 * deta**2, xvals)

            int_p2 = scipy.integrate.simpson(4 * config["A"] * curv**2 * deta**4, xvals)

            int_sum += int_p1 + int_p2

        surf_en[time_i,theta_i] = int_sum

    perc = (theta_i + 1) / thetas.shape[0] * 100
    sys.stdout.write(f"Progress: {perc:.4f}%\r")

dtheta = np.diff(thetas)[0]
for time_i in range(len(etas[0])):
    stiff[time_i] = get_stiffness(surf_en[time_i], dtheta)

plt_surf_en = np.real(np.abs(surf_en[show_time,:])).astype(float)
plt_stiff = np.real(np.abs(surf_en[show_time,:])).astype(float)

fig = plt.figure(figsize=(12, 6))

ax = plt.subplot(121, projection="polar")
ax.set_aspect("equal")
ax.plot(thetas, plt_surf_en)
ax.set_title("Surface Energy")

ax = plt.subplot(122, projection="polar")
ax.set_aspect("equal")
ax.plot(thetas, plt_stiff)
ax.set_title("Stiffness")

np.save("/home/max/projects/apfc/data/stencil_surf_en.npy", surf_en)
np.save("/home/max/projects/apfc/data/stencil_stiff.npy", stiff)
np.save("/home/max/projects/apfc/data/stencil_surf_en_thetas.npy", thetas)

plt.show()
