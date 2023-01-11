import numpy as np


def read_eta_line_str(eta_str: str, dim_x: int, dim_y: int, dtype=complex) -> np.array:

    eta = np.array(eta_str.split(","), dtype=dtype)
    eta = eta.reshape((dim_x, dim_y))

    return eta


def read_eta_last_file(
    file_path: str, dim_x: int, dim_y: int, dtype=complex
) -> tuple[np.ndarray, int]:

    last_line = ""
    line_i = 0
    with open(file_path, "r") as f:

        for line in f:
            if line.strip() == "":
                break

            last_line = line
            line_i += 1

    if last_line == "":
        eta = np.zeros((dim_x, dim_y))
    else:
        eta = read_eta_line_str(last_line, dim_x, dim_y, dtype=dtype)

    return eta, line_i


def read_eta_at_line(
    file_path: str, index: int, dim_x: int, dim_y: int, dtype=complex
) -> np.ndarray:

    if index < 0:

        line_count = count_lines(file_path)
        index = line_count - np.abs(index)

    with open(file_path, "r") as f:

        for line_i, line in enumerate(f):

            if line.strip() == "":
                return None

            if line_i == index:
                break

    if line_i < index:
        return None

    eta = read_eta_line_str(line, dim_x, dim_y, dtype=dtype)

    return eta


def read_all_etas_at_line(
    sim_dir: str, index: int, dim_x: int, dim_y: int, eta_count: int, dtype=complex
) -> np.ndarray:

    etas = np.zeros((eta_count, dim_x, dim_y), dtype=dtype)

    for eta_i in range(eta_count):

        eta_path = f"{sim_dir}/out_{eta_i}.txt"

        # the read_eta_at_line function
        # raises the StopIteration Exception
        eta = read_eta_at_line(eta_path, index, dim_x, dim_y, dtype)

        if eta is None:
            return None

        etas[eta_i, :, :] += eta

    return etas


def count_lines(path):

    with open(path, "r") as f:
        for line_i, _ in enumerate(f):
            pass

    return line_i + 1


class EtaIterator:
    def __init__(
        self, sim_dir: str, dim_x: int, dim_y: int, eta_count: int, dtype=complex
    ):

        self.sim_dir = sim_dir
        self.dtype = dtype
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.eta_count = eta_count

    def __iter__(self):

        self.current_index = 0
        return self

    def __next__(self):

        etas = read_all_etas_at_line(
            self.sim_dir,
            self.current_index,
            self.dim_x,
            self.dim_y,
            self.eta_count,
            self.dtype,
        )

        if etas is None:
            raise StopIteration

        self.current_index += 1

        return etas

    def count_lines(self):

        path = f"{self.sim_dir}/out_0.txt"
        with open(path, "r") as f:
            for line_i, _ in enumerate(f):
                pass

        return line_i + 1
