"""
Plotting functions.
"""
from cycler import cycler
from matplotlib import ticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import typing as tp
import numpy as np
from .logger import NeuralUtilityError, NeuralUtilityWarning
from warnings import warn


def plot_multiple(
    data_x: np.ndarray, *args, **kwargs
) -> tp.Tuple[plt.Figure, np.ndarray]:
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
    axes = np.atleast_1d(axes)

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


def plot_spikes(
    spikes: np.ndarray,
    t: np.ndarray = None,
    ax: plt.Axes = None,
    markersize: int = None,
    color_func: tp.Callable = lambda n: "k",
) -> plt.Axes:
    """
    Plot Spikes in raster format

    Arguments:
        spikes: the spike states in binary format, where 1 stands for a spike.
            The shape of the spikes should either be (N_times, ) or (N_trials, N_times)
        t: time axes for the spikes, use arange if not provided
        ax: which axis to plot into, create one if not provided
        markersize: size of raster
        color_func: function that maps a index (1:N_trials) to a color.
            The returned color value can be anything accepted by the `color=` keyword argument
            in the `matplotlib.pyplot.plot` function call.

    Returns:
        ax: the axis that the raster is plotted into
    """
    spikes = np.atleast_2d(spikes)
    if spikes.ndim != 2:
        raise NeuralUtilityError(
            f"matrix need to be of ndim 2, (channels x time), got ndim={spikes.ndim}"
        )
    neu_idx, t_idx = np.nonzero(spikes)
    if t is None:
        t = np.arange(spikes.shape[1])

    try:
        color_cycler = cycler(color=[color_func(n) for n in range(spikes.shape[0])])
    except Exception as e:
        err = NeuralUtilityWarning(
            f"plot_spike keyword argument color_func error, default to all black, {e}"
        )
        warn(err)
        color_cycler = cycler(color=["k"])

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot()
        ax.set_prop_cycle(custom_cycler)
        # plt.plot(t[t_idx], neu_idx, "|", c="k", markersize=markersize)
        # ax = plt.gca()
    else:
        ax.plot(t[t_idx], neu_idx, "|", c="k", markersize=markersize)
    ax.set_xlim([t.min(), t.max()])
    return ax


def plot_mat(
    mat: np.ndarray,
    t: np.ndarray = None,
    ax: plt.Axes = None,
    cax=None,
    vmin: float = None,
    vmax: float = None,
    cbar_kw: dict = None,
    cmap: tp.Any = None,
) -> tp.Union[tp.Tuple[plt.Axes, tp.Any], plt.Axes]:
    """
    Plot Matrix with formatted time axes

    Arguments:
        mat: the matrix to be plotted, it should of shape (N, Time)
        t: time axes for the spikes, use arange if not provided
        ax: which axis to plot into, create one if not provided
        cax: which axis to plot colorbar into
            - if instance of axis, plot into that axis
            - if is True, steal axis from `ax`
        vmin: minimum value for the imshow
        vmax: maximum value for the imshow
        cbar_kw: keyword arguments to be passed into the colorbar creation
        cmap: colormap to use

    Returns:
        ax: the axis that the raster is plotted into
        cbar: colorbar object
            - only returned if cax is `True` or a `plt.Axes` instance
    """
    mat = np.atleast_2d(mat)
    if mat.ndim != 2:
        raise NeuralUtilityError(
            f"matrix need to be of ndim 1 (N_time),or ndim 2 (N_trials x N_times), got ndim={mat.ndim}"
        )
    if t is None:
        t = np.arange(mat.shape[1])
    dt = t[1] - t[0]

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        return "{:.1f}".format(dt * x)

    if ax is None:
        cim = plt.imshow(
            mat,
            aspect="auto",
            interpolation="none",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax = plt.gca()
        ax.xaxis.set_major_formatter(major_formatter)
    else:
        cim = ax.imshow(
            mat,
            aspect="auto",
            interpolation="none",
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.xaxis.set_major_formatter(major_formatter)

    if cax is not None:
        if cbar_kw is None:
            cbar_kw = {}
        if not isinstance(cax, plt.Axes):
            cbar = plt.colorbar(cim, ax=ax, **cbar_kw)
        else:
            cbar = plt.colorbar(cim, cax, **cbar_kw)
        return ax, cbar
    else:
        return ax
