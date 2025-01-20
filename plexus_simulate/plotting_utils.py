import numpy as np
import matplotlib.pyplot as plt


def plot_calcium_traces(gcamp_signal: np.ndarray,
                        plot_color: str,
                        fs: int):
    """
    Plot calcium traces.

    Parameters
    ----------
    gcamp_signal : np.ndarray
        The calcium signal.
    plot_color : str
        The color of the plot.
    fs : int
        The sampling frequency.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(6, 6))
    sig_max = np.amax(gcamp_signal) * 0.5
    time = np.arange(gcamp_signal.shape[1]) / fs
    for c, cell in enumerate(gcamp_signal):
        cell_scaled = (cell / sig_max) + c
        plt.plot(time, cell_scaled, color='k')
        plt.fill_between(time, c, cell_scaled, color=plot_color, alpha=.4)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Number')
    plt.show()


def plot_calcium_traces_and_save(gcamp_signal: np.ndarray,
                        plot_color: str,
                        fs: int,
                        save_path: str):
    """
    Plot calcium traces and save the plot.

    Parameters
    ----------
    gcamp_signal : np.ndarray
        The calcium signal.
    plot_color : str
        The color of the plot.
    fs : int
        The sampling frequency.
    save_path : str
        The path to save the plot.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(3, 3))
    sig_max = 0.4
    time = np.arange(gcamp_signal.shape[1]) / fs
    for c, cell in enumerate(gcamp_signal):
        cell_scaled = (cell / sig_max) + c
        plt.plot(time, cell_scaled, color='k')
        plt.fill_between(time, c, cell_scaled, color=plot_color, alpha=.4)
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Number')
    plt.tight_layout()
    plt.savefig(save_path, dpi=800)
    plt.show()