import pytest
import numpy as np
import matplotlib.pyplot as plt
from neural import plot  # pylint:disable=import-error
import neural.errors as err  # pylint:disable=import-error

DT = 1e-4
DUR = 1.0
T = np.arange(0, DUR, DT)


@pytest.fixture
def matrix_data():
    return np.random.rand(100, len(T))


@pytest.fixture
def spike_data():
    spikes = np.random.rand(100, len(T)) < 0.5
    return spikes.astype(float)


def test_plot_mat(matrix_data):
    ax, cbar = plot.plot_mat(matrix_data, dt=DT, cax=True)
    assert isinstance(ax, plt.Axes)
    (ax,) = plot.plot_mat(matrix_data, dt=DT, cax=False)
    assert isinstance(ax, plt.Axes)
    (ax,) = plot.plot_mat(matrix_data, t=T, cax=False)
    assert isinstance(ax, plt.Axes)

    fig, ax = plt.subplots(1, 1)
    (ax2,) = plot.plot_mat(matrix_data, t=T, cax=False, ax=ax)
    assert ax == ax2

    fig, axes = plt.subplots(1, 2)
    ax, cbar = plot.plot_mat(matrix_data, t=T, ax=axes[0], cax=axes[1])
    assert ax == axes[0]

    fig, axes = plt.subplots(1, 2)
    ax, cbar = plot.plot_mat(
        matrix_data, t=T, ax=axes[0], cax=axes[1], cbar_kw={"orientation": "horizontal"}
    )
    assert ax == axes[0]

    with pytest.raises(
        err.NeuralPlotError, match=r"Time vector .* does not have the same shape"
    ):
        ax, cbar = plot.plot_mat(matrix_data, t=[0], cax=True)

    with pytest.raises(
        err.NeuralPlotError, match=r"Time vector .* does not have the same shape"
    ):
        ax, cbar = plot.plot_mat(matrix_data.T, t=T, cax=True)


def test_plot_spikes(spike_data):
    ax = plot.plot_spikes(spike_data)
    assert isinstance(ax, plt.Axes)

    ax = plot.plot_spikes(spike_data, t=T)
    assert isinstance(ax, plt.Axes)

    ax = plot.plot_spikes(spike_data, dt=DT)
    assert isinstance(ax, plt.Axes)

    with pytest.raises(
        err.NeuralPlotError, match=r"Time vector .* does not have the same shape"
    ):
        ax = plot.plot_spikes(spike_data.T, t=[0])

    with pytest.raises(err.NeuralPlotError, match=r"dt must be a scalar value"):
        ax = plot.plot_spikes(spike_data.T, dt=[DT])

    with pytest.raises(err.NeuralPlotError, match=r"dt must be a scalar value"):
        ax = plot.plot_spikes(spike_data.T, dt=T)
