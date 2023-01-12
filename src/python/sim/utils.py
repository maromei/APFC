import os


def make_path_arg_absolute(path: str):

    if path[0] == "/":  # is already absolute
        return path

    if path[0] == ".":  # asssumes starting with "./path/to/file"
        path = path[2:]

    if path[-1] == "/":  # if folder is passed remove trailing "/"
        path = path[:-1]

    return "/".join([os.getcwd(), path])


def build_sim_info_str(config, index):

    txt = f"""\
        \\begin{{center}}
        sim iteration: {index} \\vspace{{0.5em}}
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
