"""
Plotting functions.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_multiple(data_x, *args, **kwargs):
    """
    Plot multiple data curves against a same x-axis on mulitple subplots.

    Arguments:
        datax (darray): the data point on the x-axis.
        *args: each entry of args is a list containing multiple sets of data
            and parameters that will be plotted in the same subplot.
            An entry should follow the format `(data_1, param_1, ...)`,
            where each of the `data_i` is a numpy array, and each of the
            `param_i` is a `dict` of the parameters for ploting `data_i` against
            `data_x`. Alternatively, an entry can simply be an numpy array. In
            this case, only one curve will be plotted in the corresponding
            subplot.

    Keyword Arguments:
        figw (float): the figure width.
        figh (float): the figure height.
        xlabel (str): the label of the x-axis.

        The additional keyword arguments will propagate into the private
        plotting method `_plot`, and eventually into the `pyplot.plot` method.
    """

    def _plot(axe, data_x, data_y, **kwargs):
        """
        Arguments:
            axe (matplotlib.Axes.axe): the axe of the subplot.
            data_x (darray): the data point along the x-axis.
            data_y (darray): the data point along the y-axis.

        Keyword Arguments:
            xlim (tuple): a tuple-like with two entries of limits of the x-axis.
            ylim (tuple): a tuple-like with two entries of limits of the y-axis.
            spike (bool): specify if `data_y` is a spike sequence.
            ylabel (str): the label of the y-axis.
            ds_rate (int): the downsample rate of the data.

            The additional keyword arguments will propagate into the
            `pyplot.plot` method. For example, one could use `label` to add a
            legend to a curve.
        """
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        spike = kwargs.pop("spike", False)
        ylabel = kwargs.pop("ylabel", None)
        ds_rate = kwargs.pop("ds_rate", None)

        if spike:
            ylim = [0, 1.2]
            ylabel = ylabel or "Spike Train"
            axe.yaxis.set_ticklabels([" "])

        if ds_rate is not None:
            data_x = data_x[::ds_rate]
            data_y = data_y[::ds_rate]

        axe.plot(data_x, data_y, **kwargs)

        if xlim:
            axe.set_xlim(xlim)
        if ylim:
            axe.set_ylim(ylim)
        if ylabel:
            axe.set_ylabel(ylabel)

    figw = kwargs.pop("figw", 5)
    figh = kwargs.pop("figh", 2)
    xlabel = kwargs.pop("xlabel", "Time, [s]")

    num = len(args)

    fig, axes = plt.subplots(num, 1, figsize=(figw, num * figh))

    if not hasattr(axes, "__len__"):
        axes = [axes]

    for i, (dataset, axe) in enumerate(zip(args, axes)):
        axe.grid()
        if i < num - 1:
            axe.xaxis.set_ticklabels([])

        if isinstance(dataset, np.ndarray):
            param_list = [{}]
            data_list = [dataset]
        else:
            param_list = dataset[1::2]
            data_list = dataset[0::2]

        has_legend = False
        for data_y, subkwargs in zip(data_list, param_list):
            for key, val in kwargs.items():
                if not key in subkwargs:
                    subkwargs[key] = val
            has_legend = has_legend or ("label" in subkwargs)
            _plot(axe, data_x, data_y, **subkwargs)
        if has_legend:
            axe.legend()

    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()

    return fig, axes


def yyaxis(ax: plt.Axes, c: "color" = "red") -> plt.Axes:
    """Create A second axis with colored spine/ticks/label

    Note:
        This method will only make the twinx look like the color in
        MATLAB's :code:`yyaxis` function. However, unlike in MATLAB,
        it will not set the linestyle and linecolor of the lines that
        are plotted after twinx creation.

    Arguments:
        ax: the main axis to generate a twinx from
        c: color of the twinx, see https://matplotlib.org/stable/gallery/color/color_demo.html
            for color specifications accepted by matplotlib.
    """
    ax2 = ax.twinx()
    ax2.spines["right"].set_color(c)
    ax2.tick_params(axis="y", colors=c)
    ax2.yaxis.label.set_color(c)
    return ax2
