import os


def make_path_arg_absolute(path: str):

    if path[0] == "/":  # is already absolute
        return path

    if path[0] == ".":  # asssumes starting with "./path/to/file"
        path = path[2:]

    if path[-1] == "/":  # if folder is passed remove trailing "/"
        path = path[:-1]

    return "/".join([os.getcwd(), path])
