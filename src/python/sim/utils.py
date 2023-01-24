import os

import numpy as np


def make_path_arg_absolute(path: str):

    if path[0] == "/":  # is already absolute
        return path

    if path[0] == ".":  # asssumes starting with "./path/to/file"
        path = path[2:]

    if path[-1] == "/":  # if folder is passed remove trailing "/"
        path = path[:-1]

    return "/".join([os.getcwd(), path])


def build_sim_info_str(config, index, theta=None):

    theta_str = ""
    if theta is not None:
        theta_str = f"\n$\\theta = {theta:.4f}$\n"

    txt = f"""\
        \\begin{{center}}
        sim iteration: {index} \\vspace{{0.5em}}
        {theta_str}
        $B_x = {config['Bx']:.4f}, n_0 = {config['n0']:.4f}$
        $v = {config['v']:.4f}, t = {config['t']:.4f}$
        $\\Delta B_0 = {config['dB0']:.4f}$
        $\\mathrm{{d}}t = {config['dt']:.4f}$ \\vspace{{0.5em}}
        initial Radius: {config['initRadius']:.4f}
        initial Eta in solid: {config['initEta']:.4f}
        interface width: {config['interfaceWidth']:.4f}
        domain: $[-{config['xlim']}, {config['xlim']}]^2$
        points: {config['numPts']} x {config['numPts']}
        \\end{{center}}
    """

    txt = "".join(map(str.lstrip, txt.splitlines(1)))

    return txt


def get_thetas(config, use_div=True, endpoint=True):

    if use_div:
        thetas = np.linspace(
            0,
            2.0 * np.pi / config["theta_div"],
            config["theta_count"],
            endpoint=endpoint,
        )
    else:
        thetas = np.linspace(0, 2.0 * np.pi, config["theta_count"], endpoint=endpoint)

    return thetas


def create_float_scientific_string(val: float):

    val_str = f"{val:.2e}"
    val_str = val_str.split("e")
    val_str = f"{val_str[0]} \\cdot 10^{{{val_str[1]}}}"

    return val_str
