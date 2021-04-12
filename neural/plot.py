"""
Plotting functions.
"""
import typing as tp
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
import numpy as np
from . import errors as err

COLOR_NORMS = tp.Union['none', 'log', 'discrete', 'power', ]

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


def plot_spikes(
    spikes: np.ndarray,
    dt: float = None,
    t: np.ndarray = None,
    ax: plt.Axes = None,
    markersize: int = None,
    color: tp.Union[str, tp.Any] = "k",
) -> plt.Axes:
    """
    Plot Spikes in raster format
    Arguments:
        spikes: the spike states in binary format, where 1 stands for a spike.
            The shape of the spikes should either be (N_times, ) or (N_trials, N_times)
        dt: time resolution of the time axis.
        t: time axes for the spikes, use arange if not provided

        .. note::

            If `t` is specified, it is assumed to have the same
            length as `mat.shape[1]`, which is used to find the x coordinate of
            the spiking values of the data. If `t` is
            not specified, the time-axis is formated by resolution `dt`.
            `dt` is assumed to be 1 if not specified.

        ax: which axis to plot into, create one if not provided
        markersize: size of raster
        color: color for the raster. Any acceptable type of `matplotlib.pyplot.plot`'s
            color argument is accepted.
    Returns:
        ax: the axis that the raster is plotted into
    """
    spikes = np.atleast_2d(spikes)
    if spikes.ndim != 2:
        raise err.NeuralPlotError(
            f"matrix need to be of ndim 2, (channels x time), got ndim={spikes.ndim}"
        )

    if t is not None:
        if len(t) != spikes.shape[1]:
            raise err.NeuralPlotError(
                "Time vector 't' does not have the same shape as the matrix."
                f" Expected length {spikes.shape[1]} but got {len(t)}"
            )
    else:
        if dt is None:
            dt = 1.0
        else:
            if not np.isscalar(dt):
                raise err.NeuralPlotError("dt must be a scalar value.")
        t = np.arange(spikes.shape[1]) * dt

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot()

    neu_idx, t_idx = np.nonzero(spikes)

    try:
        ax.plot(t[t_idx], neu_idx, "|", c=color, markersize=markersize)
    except ValueError as e:
        raise err.NeuralPlotError(
            "Raster plot failed, likely an issue with color or markersize setting"
        ) from e
    except IndexError as e:
        raise err.NeuralPlotError(
            "Raster plot failed, likely an issue with spikes and time vector mismatch"
        ) from e
    except Exception as e:
        raise err.NeuralPlotError("Raster plot failed due to unknown error") from e
    ax.set_xlim([t.min(), t.max()])
    return ax


def plot_mat(
    mat: np.ndarray,
    dt: float = None,
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
        dt: time resolution of the time axis.
        t: time axes for the spikes, use arange if not provided.

        .. note::

            If `t` is specified, it is assumed to have the same
            length as `mat.shape[1]`. Consequently, the x-axis will be formatted
            to take the corresponding values from `t` based on index. If `t` is
            not specified, the time-axis is formated by resolution `dt`.
            If neither are specified, `dt` is assumed to be 1.

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

    Example:
        >>> dt, dur, start, stop = 1e-4, 2, 0.5, 1.0
        >>> t = np.arange(0, dur, dt)
        >>> amps = np.arange(0, 100, 10)
        >>> wav = utils.generate_stimulus('step', dt, dur, (start, stop), amps)
        >>> ax,cbar = plot_mat(wav, t=t, cax=True, vmin=10, vmax=100, cbar_kw={'label':'test'}, cmap=plt.cm.gnuplot)
        >>> ax, = plot_mat(wav, t=t, cax=False, vmin=10, vmax=100, cbar_kw={'label':'test'}, cmap=plt.cm.gnuplot)
    """
    mat = np.atleast_2d(mat)
    if mat.ndim != 2:
        raise err.NeuralPlotError(
            "matrix need to be of ndim 1 (N_time),or ndim 2 (N_trials x N_times),"
            f" got ndim={mat.ndim}"
        )
    if t is not None:
        if len(t) != mat.shape[1]:
            raise err.NeuralPlotError(
                "Time vector 't' does not have the same shape as the matrix."
                f" Expected length {mat.shape[1]} but got {len(t)}"
            )

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return "{:.1f}".format(np.interp(x, np.arange(len(t)), t))

    else:
        if dt is None:
            dt = 1

        @ticker.FuncFormatter
        def major_formatter(x, pos):
            return "{:.1f}".format(dt * x)

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot()

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

    if cax:
        if cbar_kw is None:
            cbar_kw = {}
        if not isinstance(cax, plt.Axes):
            cbar = plt.colorbar(cim, ax=ax, **cbar_kw)
        else:
            cbar = plt.colorbar(cim, cax, **cbar_kw)
        return ax, cbar
    else:
        return (ax,)


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
    ax2.tick_params(axis="y", colors=c, which='both')
    ax2.yaxis.label.set_color(c)
    return ax2
