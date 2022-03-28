"""
Plotting functions.
"""
import typing as tp
import numpy as np
import matplotlib.pyplot as plt
from .. import errors as err

def plot_multiple(
    data_x: np.ndarray,
    *args,
    figw: float = 5,
    figh: float = 2,
    xlabel: str = "Time, [s]",
    axes: tp.Iterable[plt.Axes] = None,
    **subplots_kw,
) -> tp.Tuple[plt.Figure, np.ndarray]:
    """
    Plot multiple data curves against a same x-axis on mulitple subplots.

    Arguments:
        datax: the data point on the x-axis.
        *args: each entry of args is a list containing multiple sets of data
          and parameters that will be plotted in the same subplot.
          An entry should follow the format `(data_1, param_1, ...)`,
          where each of the `data_i` is a numpy array, and each of the
          `param_i` is a `dict` of the parameters for ploting `data_i` against
          `data_x`. Alternatively, an entry can simply be an numpy array. In
          this case, only one curve will be plotted in the corresponding
          subplot.

    Keyword Arguments:
        figw: the figure width.
        figh: the figure height.
        xlabel: the label of the x-axis.
        axes: axes into which the data will be plotted. Must have same length as :code:`*args`.
          If not specified, new figure and axes are created.
        subplots_kw: any arguments for :code:`matplotlib.pyplot.subplots` call.

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

    num = len(args)

    if axes is not None:
        assert len(axes) == num, "axes must have same length as args"
        fig = None
    else:
        fig, axes = plt.subplots(num, 1, figsize=(figw, num * figh), **subplots_kw)

    for i, (dataset, axe) in enumerate(zip(args, axes)):
        if isinstance(dataset, np.ndarray):
            param_list = [{}]
            data_list = [dataset]
        else:
            param_list = dataset[1::2]
            data_list = dataset[0::2]

        has_legend = False
        for data_y, subkwargs in zip(data_list, param_list):
            has_legend = has_legend or ("label" in subkwargs)
            _plot(axe, data_x, data_y, **subkwargs)
        if has_legend:
            axe.legend()
        axe.grid()
    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()

    return fig, axes


def plot_spikes(
    spikes: np.ndarray,
    dt: float = None,
    t: np.ndarray = None,
    ax: plt.Axes = None,
    color: tp.Union[str, tp.Callable] = None,
    **scatter_kwargs
) -> plt.Axes:
    """Plot Spikes in raster format

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
        color: color for raster
          - `str`: if color is a string, it is assumed to be the same color for all rasters
          - `function(t,channel)`: if color is a function, it is assumed to be a 2 argument
            function that maps time and channel of spike times to some value. The color
            can then be controlled by specifying `cmap` argument of the scatter plot.

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
    lw = scatter_kwargs.pop('linewidth', 1)
    try:
        ax.scatter(
            t[t_idx],
            neu_idx,
            marker="|",
            c=color(t[t_idx], neu_idx) if callable(color) else color,
            linewidth=lw,
            **scatter_kwargs
        )
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
    dy: float = None,
    t: np.ndarray = None,
    y: np.ndarray = None,
    ax: plt.Axes = None,
    cax=None,
    vmin: float = None,
    vmax: float = None,
    cbar_kw: dict = None,
    **pcolormesh_kwargs,
) -> tp.Union[tp.Tuple[plt.Axes, tp.Any], plt.Axes]:
    """
    Plot Matrix with formatted time axes

    Arguments:
        mat: the matrix to be plotted, it should of shape (N, Time)
        dt: time resolution of the time axis.
        dy: resolution of the Y-axis
        t: time axes for the matrix, use arange if not provided.

        .. note::

            If `t` is specified, it is assumed to have the same
            length as `mat.shape[1]`. Consequently, the x-axis will be formatted
            to take the corresponding values from `t` based on index. If `t` is
            not specified, the time-axis is formated by resolution `dt`.
            If neither are specified, `dt` is assumed to be 1.

        y: spatial axes of the matrix, use arange if not provided.

        .. note::

            If `y` is specified, it is assumed to have the same
            length as `mat.shape[0]`. Consequently, the y-axis will be formatted
            to take the corresponding values from `y` based on index. If `y` is
            not specified, the time-axis is formated by resolution `dy`.
            If neither are specified, `dy` is assumed to be 1.

        ax: which axis to plot into, create one if not provided
        cax: which axis to plot colorbar into
            - if instance of axis, plot into that axis
            - if is True, steal axis from `ax`
        vmin: minimum value for the imshow
        vmax: maximum value for the imshow
        cbar_kw: keyword arguments to be passed into the colorbar creation

    Keyword Arguments:
        **pcolormesh_kwargs: Keyword Arguments to be passed into the :py:func:`matplotlib.pyplot.pcolormesh` function.

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
    else:
        if dt is None:
            dt = 1
        t = np.arange(mat.shape[1]) * dt

    if y is not None:
        if len(y) != mat.shape[0]:
            raise err.NeuralPlotError(
                "Spatial vector 'y' does not have the same shape as the matrix."
                f" Expected length {mat.shape[0]} but got {len(y)}"
            )
    else:
        if dy is None:
            dy = 1
        y = np.arange(mat.shape[0]) * dy

    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot()

    shading = pcolormesh_kwargs.pop("shading", "auto")
    cim = ax.pcolormesh(
        t, y, mat, vmin=vmin, vmax=vmax, shading=shading, **pcolormesh_kwargs
    )

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
    ax2.tick_params(axis="y", colors=c, which="both")
    ax2.yaxis.label.set_color(c)
    return ax2
