import os
import json

import numpy as np


def make_path_arg_absolute(path: str) -> str:
    """
    Takes in a path and transforms it into an absolute one based
    on the current working directory.

    Args:
        path (str): path

    Returns:
        str: absolute path
    """

    if path[0] == "/":  # is already absolute
        return path

    if path[0] == ".":  # asssumes starting with "./path/to/file"
        path = path[2:]

    if path[-1] == "/":  # if folder is passed remove trailing "/"
        path = path[:-1]

    return "/".join([os.getcwd(), path])


def build_sim_info_str(
    config: dict, index: int, theta: float | None = None, add_info: str = ""
) -> str:
    """
    Builds a latex string with additional information.
    The purpose is to include this string into plots to give context.

    Args:
        config (dict): The config dictionary
        index (int): Which index of the simulation is displayed
        theta (float | None, optional): Angle. Defaults to None.
        add_info (str, optional): additional info to dislay.
            It will be appended at the end. Defaults to "".

    Returns:
        str: formatted latex string
    """

    theta_str = ""
    if theta is not None:
        theta_str = f"\n$\\theta = {theta:.4f}$\n"

    is_1d = config["numPtsY"] <= 1

    txt = f"""\
        \\begin{{center}}
        sim iteration: {index} \\vspace{{0.5em}}
        {theta_str}
        $B^x = {config['Bx']:.4f}, n_0 = {config['n0']:.4f}$
        $v = {config['v']:.4f}, t = {config['t']:.4f}$
        $\\Delta B^0 = {config['dB0']:.4f}$
        $\\mathrm{{d}}t = {config['dt']:.4f}$ \\vspace{{0.5em}}
        initial Radius: {config['initRadius']:.4f}
        initial Eta in solid: {config['initEta']:.4f}
        interface width: {config['interfaceWidth']:.4f}
        domain: $[-{config['xlim']}, {config['xlim']}]{'' if is_1d else '^2'}$
        points: {config['numPtsX']} x {config['numPtsY']}
        {add_info}
        \\end{{center}}
    """

    txt = "".join(map(str.lstrip, txt.splitlines(1)))

    return txt


def get_thetas(config: dict, use_div: bool = True, endpoint: bool = True) -> np.array:
    """
    Creates an array of thetas based on the config.
    This function is supposed to supply a consistent way to get the same
    angles.

    Args:
        config (dict): config object. Explicitely used keys are:
            `thetaCount` and `thetaDiv` (if `use_div=True`).
        use_div (bool, optional): Whether to divide the interval
            into `thetaDiv` parts. Where `thetaDiv`. Defaults to True.
        endpoint (bool, optional): Whether to include the last
            value in the range. Defaults to True.

    Returns:
        np.array: The equaly spaced thetas
    """

    if use_div:
        thetas = np.linspace(
            0,
            2.0 * np.pi / config["thetaDiv"],
            config["thetaCount"],
            endpoint=endpoint,
        )
    else:
        thetas = np.linspace(0, 2.0 * np.pi, config["thetaCount"], endpoint=endpoint)

    return thetas


def create_float_scientific_string(val: float) -> str:
    """
    Takes in a value and generates a scientific notation for it.
    2 digits infront of the decimal place will be shown.

    Args:
        val (float): input value

    Returns:
        str: formatted string
    """

    val_str = f"{val:.2e}"
    val_str = val_str.split("e")
    val_str = f"{val_str[0]} \\cdot 10^{{{val_str[1]}}}"

    return val_str


def get_config(sim_path: str) -> dict:
    """
    Reads the config from the sim_path.

    Args:
        sim_path (str): directory where the :code:`config.json` is stored.

    Returns:
        dict: config
    """

    sim_path = make_path_arg_absolute(sim_path)
    config_path = f"{sim_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    return config
