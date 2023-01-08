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
