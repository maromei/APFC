import numpy as np


def str_to_arr(eta_str: str, dim_x: int, dim_y: int, dtype=float) -> np.array:
    """
    Converts one string read from a file to a 2d array.

    Args:
        eta_str (str): The line to read
        dim_x (int): output array x dimensions
        dim_y (int): output array y dimensions
        dtype (_type_, optional): type of data. Defaults to float.

    Returns:
        np.array: The created array
    """

    eta = np.array(eta_str.split(","), dtype=dtype)
    eta = eta.reshape((dim_x, dim_y))

    if dim_y <= 1:
        eta = eta.flatten()

    return eta


def read_last_line_to_array(
    file_path: str, dim_x: int, dim_y: int, dtype=float
) -> tuple[np.ndarray, int]:
    """
    Reads a files last line and converts it to an array.
    Uses :py:meth:`manage.read_write.str_to_arr` for the conversion.

    Args:
        file_path (str): file path
        dim_x (int): output array x dimensions
        dim_y (int): output array y dimensions
        dtype (_type_, optional): data type. Defaults to float.

    Returns:
        tuple[np.ndarray, int]: The created array, and the line index of the
            last line.
    """

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
        eta = str_to_arr(last_line, dim_x, dim_y, dtype=dtype)

    return eta, line_i


def read_arr_at_line(
    file_path: str, index: int, dim_x: int, dim_y: int, dtype=float
) -> np.ndarray:
    """
    Reads a file at the specified line index and returns the read
    line as an array.

    Args:
        file_path (str): file path to read
        index (int): at which index should be read
        dim_x (int): output array x dimensions
        dim_y (int): output array y dimensions
        dtype (_type_, optional): Data type. Defaults to float.

    Returns:
        np.ndarray: the read array
    """

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

    eta = str_to_arr(line, dim_x, dim_y, dtype=dtype)

    return eta


def read_all_etas_at_line(
    sim_dir: str, index: int, dim_x: int, dim_y: int, eta_count: int, dtype=float
) -> np.ndarray:
    """
    This function reads all etas at one specific line in a file.

    Args:
        sim_dir (str): The sim path. It should contain all "out_{i}.txt" files.
        index (int): at which index should be read
        dim_x (int): output array x dimensions
        dim_y (int): output array y dimensions
        eta_count (int): How many files should be looked at.
        dtype (_type_, optional): Data type. Defaults to float.

    Returns:
        np.ndarray: Array with shape (eta_count, dim_x, dim_y) where
            the first index identifies which eta was read.
    """

    if dim_y <= 1:
        etas = np.zeros((eta_count, dim_x), dtype=dtype)
    else:
        etas = np.zeros((eta_count, dim_x, dim_y), dtype=dtype)

    for eta_i in range(eta_count):

        eta_path = f"{sim_dir}/out_{eta_i}.txt"

        # the read_arr_at_line function
        # raises the StopIteration Exception
        eta = read_arr_at_line(eta_path, index, dim_x, dim_y, dtype)

        if eta is None:
            return None

        etas[eta_i] = eta

    return etas


def count_lines(path: str) -> int:
    """
    Runs through a file and checks how many lines there are

    Args:
        path (str): file path to check

    Returns:
        int: line count
    """

    with open(path, "r") as f:
        for line_i, _ in enumerate(f):
            pass

    return line_i + 1


class EtaIterator:
    """
    This class' only purpose is to create an iterator for the different
    timesteps in a sim.
    """

    def __init__(
        self,
        sim_dir: str,
        dim_x: int,
        dim_y: int,
        eta_count: int,
        dtype=float,
        include_n0: bool = False,
    ):
        """
        Initiliazes the class. It just sets the parameters

        Args:
            sim_dir (str): Where the files are located. It should contain
                all "out_{i}.txt" files and if applicable the "n0.txt" file.
            dim_x (int): x dimension of data
            dim_y (int): y dimension of data
            eta_count (int): how many "out_{i}.txt" files there are
            dtype (_type_, optional): type of data. Defaults to float.
            include_n0 (bool, optional): Should n0 be included. Defaults to False.
        """

        self.sim_dir = sim_dir
        self.dtype = dtype
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.eta_count = eta_count
        self.include_n0 = include_n0

    def __iter__(self):
        """
        Initializes the Iterator.
        sets `self.current_index = 0`.

        Returns:
            EtaIterator: self
        """

        self.current_index = 0
        return self

    def __next__(self) -> tuple[np.ndarray] | np.ndarray:
        """
        Gets the next etas and n0 if applicable.

        Raises:
            StopIteration: If no new eta or n0 can be read.

        Returns:
            tuple[np.ndarray] | np.ndarray:
                An array with shape (eta_count, dim_x, dim_y).
                If `include_n0=True` a tuple will be returned where the
                second entry is the n0 array with shape (dim_x, dim_y).
        """

        etas = read_all_etas_at_line(
            self.sim_dir,
            self.current_index,
            self.dim_x,
            self.dim_y,
            self.eta_count,
            self.dtype,
        )

        if self.include_n0:
            n0 = read_arr_at_line(
                f"{self.sim_dir}/n0.txt",
                self.current_index,
                self.dim_x,
                self.dim_y,
                self.dtype,
            )

        if etas is None:
            raise StopIteration

        self.current_index += 1

        if self.include_n0:
            return etas, n0
        else:
            return etas

    def count_lines(self) -> int:
        """
        Counts how many etas can be read.
        Uses the "out_0.txt" for reading.

        Returns:
            int: line count
        """

        path = f"{self.sim_dir}/out_0.txt"
        with open(path, "r") as f:
            for line_i, _ in enumerate(f):
                pass

        return line_i + 1
