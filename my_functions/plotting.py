import numpy as np
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=False)

from my_functions.processing import apply_fancy_filter

def correlogram_plot(data, N, L, fs, lc=None, hc=None):
    """
    Compute all pairwise combinations of correlation coefficients for a multi-dimensional array.
    Plot them as a color mesh where x-axis is time, y-axis is pair numbers, z-axis is the CC value
    :param data: N-dimensional array
    :param N: Moving window length
    :param L: Moving window step
    :param fs: Sampling rate of original array
    :param lc: Lower cut-off of the band-pass filter
    :param hc: Higher cut-off of the band-pass filter
    :return: Displays a figure.
    """
    # How many times the window will shift?
    shifts = int(np.floor((len(data)-N) / L) + 1)

    fdata = np.array(data)

    if hc is not None:
        fdata = apply_fancy_filter(fdata, fs, lc, hc, ftype='bandpass')
    elif lc is not None:
        fdata = apply_fancy_filter(fdata, fs, lc)

    cols = int(shifts + 1)
    rows = int(fdata.shape[1] * (fdata.shape[1] - 1) * 0.5)
    CC = np.zeros((rows,cols))
    for i in range(shifts):
        s = i * L
        e = s + N
        cc = np.triu(np.corrcoef(fdata[s:e], rowvar=False), 1) # Extract the upper triangle of the correlation matrix
        cc = cc.flatten()
        cc = cc[np.nonzero(cc)]
        if len(cc) != rows: print('Warning! Missing value.')
        CC[:,i] = cc
        if len(fdata) - e < L:
            s = e - L
            e = len(fdata)
            cc = np.triu(np.corrcoef(fdata[s:e], rowvar=False), 1)
            cc = cc.flatten()
            cc = cc[np.nonzero(cc)]
            if len(cc) != rows: print('Warning! Missing value.')
            CC[:,i+1] = cc

    # Plot surface
    x = np.linspace(0, len(fdata)/fs, CC.shape[1])
    y = np.linspace(1, CC.shape[0], CC.shape[0])
    f, ax = plt.subplots(figsize=(8, 4))
    # If there are only two channels:
    if CC.shape[0] == 1:
        ax.plot(x, CC[0])
        ax.set_ylabel('Corr')
        ax.set_xlabel('Time [s]')
        ax.set_title('Between two channels')
    # If there are 3 or more channels:
    else:
        cax = ax.pcolormesh(x, y, CC, cmap='viridis')
        ax.set_ylabel('Channel pairs')
        ax.set_xlabel('Time [s]')
        ax.set_title('Correlogram')
        f.colorbar(cax, ticks=[-1, -0.5, 0, 0.5, 1])


##
def simple_plot(x, y=None, xlab='x-axis', ylab='y-axis', ttl='Title', grd=True):
    """
    Function for easy plotting jobs.
    :param x: x data
    :param y: y data
    :param xlab: x label
    :param ylab: y label
    :param ttl: title
    :param grd: grid flag
    :return: Figure handle
    """
    f = plt.figure()
    if y is None:
        plt.plot(x)
    else:
        plt.plot(x, y)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(ttl)
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(grd)
    return f


##
def fancy_plot(traces, time=None, title='temp'):
    if time is None:
        time = np.linspace(1, len(traces), len(traces))

    if traces.ndim < 2:
        trace = [{'type': 'scattergl',
                  'x': time,
                  'y': traces,
                  'name': 1
                  }]
    else:
        trace = [{'type': 'scattergl',
                  'x': time,
                  'y': col,
                  'name': ind + 1
                  } for ind, col in enumerate(traces.T)]

    layout = go.Layout(
        width=1000,
        height=750,
        title=title,
        xaxis=dict(
            title="time(s)"
        ),
        yaxis=dict(
            title="uV"
        )
    )

    fig = dict(data=trace, layout=layout)

    plotly.offline.plot(fig, filename=title + '.html')

##
def fancy_plot_other(traces, time=None, title='temp'):
    if time is None:
        time = np.linspace(1, len(traces), len(traces))

    trace = []
    if traces.ndim < 2:
        trace = [go.Scattergl(
            x=time,
            y=traces
        )]
    else:
        for col in traces.T:
            trace.append(go.Scattergl(
                x=time,
                y=col
            )
            )

    layout = go.Layout(
        width=1000,
        height=750,
        title=title,
        xaxis=dict(
            title="time(s)"
        ),
        yaxis=dict(
            title="uV"
        )
    )

    fig = dict(data=trace, layout=layout)

    plotly.offline.plot(fig, filename='WebGL_line')

##
def fancy_subplot(*argv):

    row = 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
    col = 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2
    num = len(argv)

    plot_titles = ['Plot {}'.format(i+1) for i in range(num)]
    plot_titles = tuple(plot_titles)

    fig = plotly.tools.make_subplots(rows=row[num], cols=col[num], subplot_titles=plot_titles)

    all_traces = []
    for arg in argv:
        trace = []
        if arg.ndim < 2:
            trace = [{'type': 'scatter',
                      'x': np.linspace(1, len(arg), len(arg)),
                      'y': arg,
                      'name': 1
                      }]
        else:
            trace = [{'type': 'scatter',
                      'x': np.linspace(1, len(arg), len(arg)),
                      'y': col,
                      'name': ind + 1
                      } for ind, col in enumerate(arg.T)]
        all_traces.append(trace)

    row = 1, 1, 2, 2, 3, 3, 4, 4, 5, 5
    col = 1, 2, 1, 2, 1, 2, 1, 2, 1, 2
    for i in range(num):
        fig.append_trace(all_traces[i], row[i], col[i])

    fig['layout'].update(height=600, width=600, title='Multiple Subplots with Titles')

    #fig = dict(data=trace, layout=layout)

    plotly.offline.plot(fig, filename='subplots')