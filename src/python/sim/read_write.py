import numpy as np


def read_eta_cplx_line_str(eta_str: str, dim_x: int, dim_y: int) -> np.array:

    eta = np.array(eta_str.split(","), dtype=complex)
    eta = eta.reshape((dim_x, dim_y))

    return eta


def read_eta_cplx_last_file(
    file_path: str, dim_x: int, dim_y: int
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
        eta = read_eta_cplx_line_str(last_line, dim_x, dim_y)

    return eta, line_i
