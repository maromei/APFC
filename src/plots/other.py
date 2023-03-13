import numpy as np
import matplotlib.pyplot as plt

from manage.utils import build_sim_info_str


def plot_info(
    config: dict,
    index: int,
    axis: plt.Axes,
    theta: float | None = None,
    add_info: str = "",
):
    """
    Plots an info text box.

    Args:
        config (dict): The config dictionary
        index (int): Which index of the simulation is displayed
        axis (plt.Axes): axis to plot on
        theta (float | None, optional): Angle. Defaults to None.
        add_info (str, optional): additional info to dislay.
            It will be appended at the end. Defaults to "".
    """

    info_str = build_sim_info_str(config, index, theta, add_info)

    axis.axis("off")
    axis.text(
        0.5, 0.5, info_str, verticalalignment="center", horizontalalignment="center"
    )
