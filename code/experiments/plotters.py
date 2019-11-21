from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['axes.linewidth'] = 5
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['grid.linewidth'] = 5
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geopandas as gpd
import pandas as pd
import scipy.spatial as sp
from scipy import special
from scipy.stats.mstats import mquantiles
import numpy as np
from ..mvn_mixture.diag_mvn_mixture import DiagMVNMixture



def plot_2d_samples(x, y, z, name='samples_2d.pdf'):
    x=np.unique(x)
    y=np.unique(y)
    X,Y = np.meshgrid(x,y)

    Z=z.reshape(y.shape[0],x.shape[0])

    fig = plt.pcolormesh(X,Y,Z, cmap=cm.jet)
    ax = plt.gca()
    plt.axis('off')
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(name)

    return fig

def plot_samples(x, y, name='samples.pdf', xlabel=r'$x$', ylabel=r'$y$',
        labels=[], linewidth=0, markersize=4, fig=None, marker='o',
        plot_legend = True, default_colours = False, y_tick_labels=True):
    """
    Plot a number of y's against a common x.
    """
    assert x.shape[0] == y.shape[1]

    # Colormap
    if not default_colours:
        viridis = cm.viridis 
        norm = Normalize(vmin = np.log2(1), vmax=np.log2(max(len(labels), 1)))

    # Do the plot
    if fig is None:
        fig = plt.figure(figsize=(7,7))
    for s in range(y.shape[0]):
        y_s = y[s,:]
        if default_colours:
            plt.plot(x, y_s,
                marker=marker, markersize=markersize, linewidth=linewidth)
        else:
            plt.plot(x, y_s, c=viridis(norm(np.log2(s+1))),
                marker=marker, markersize=markersize, linewidth=linewidth)

    # Format the plot
    plt.xlabel(xlabel, fontsize=40)
    plt.ylabel(ylabel, fontsize=40)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 30)

    if (plot_legend) and not default_colours:
        handles = [mpatches.Patch(color=viridis(norm(np.log2(idx+1))), 
                        label=l) for idx, l in enumerate(labels)]
        plt.legend(handles=handles, fontsize=25)

    # Disable y ticks if required
    if not y_tick_labels:
        ax.set_yticklabels([])
    # Save
    plt.savefig(name)
    return fig

def plot_sequence(x, y, name='sequence.pdf', fig=None, label=None):
    assert len(x) == len(y)
    fontsize = 40
    ticksize = 15

    if fig is None:
        fig = plt.figure(figsize=(7,7))
    else:
        plt.figure(fig.number) 
    plt.plot(x, y, label=label)
    plt.xlabel('Width', fontsize=fontsize)
    plt.ylabel('Squared MMD', fontsize=fontsize)
    plt.gca().set_ylim(bottom=0)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(loc='upper right', fontsize=25)
    #plt.grid(True)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.savefig(name)
    return fig

def plot_likelihood(likelihood, extent, name='likelihood.pdf', close=False, 
        reparam=False, sigma=None, mu=None):
    fontsize = 40
    ticksize = 15
    fig = plt.figure(figsize=(7,7))
    # tripcolor looks nicer than imshow
    if (not sigma is None) and (not mu is None):
        plt.tripcolor(sigma.flatten(), mu.flatten(), likelihood.flatten())
    else:
        plt.imshow(likelihood, extent = extent, origin='lower')
    plt.xlim([extent[0], extent[1]])
    plt.ylim([extent[2], extent[3]])
    plt.gca().set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
    #plt.colorbar()
    if reparam:
        plt.ylabel(r'$\mu/\sigma$', fontsize=fontsize)
        plt.xlabel(r'$\sigma$', fontsize=fontsize)
    else:
        plt.ylabel(r'$\mu$', fontsize=fontsize)
        plt.xlabel(r'$\sigma^2$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.savefig(name, bbox_inches='tight')
    if close:
        plt.close()

def plot_posterior_samples(samples, name='posterior_samples.pdf', close=False,
        marker='kx'):
    plt.plot(samples[:,1], samples[:,0], marker, zorder=100)
    plt.savefig(name, bbox_inches='tight')
    if close:
        plt.close()

def plot_gp(mus, sigmas, test_x, test_y=None, train_x=None, train_y=None, 
        fig=None, name='gp.pdf', color='b', num_samples = 10000):
    """
    Plot regression training data, testing data, mean and quantiles of a 
    mixture of m equal weighted Gaussian processes.

    Args:
        mus (nparray): (n, m) array, where n is the number of test points and
            m is the number of mixture components.
        sigmas (nparray): (n, m) array, where n is the number of test points and
            m is the number of mixture components.
        train_x (nparray): (N, 1) training x data
        train_y (nparray): (N, 1) training y data
        test_x (nparray): (n, 1) testing x data
        test_y (nparray): (n, 1) testing y data
        fig : matplotlib figure
        name (str): filename to save plot as
        color (str): color to plot.
        num_samples (str): number of samples to take in quantile calculation
    """
    if fig is None:
        fig = plt.figure(figsize=(7,7))

    # Plot the quantiles, we can just sample from the INDEPENDENT MVN then find 
    # empirically
    mixture = DiagMVNMixture(mus, sigmas)
    quantiles = mixture.quantiles(prob=[0.125, 0.25, 0.375, 0.625, .750, .875])

    # Plot the mean
    mean = mixture.mean()
    plt.plot(test_x, mean, 'k')

    for q in range(0, int(quantiles.shape[0]/2)):
        plt.fill_between(test_x.flatten(), quantiles[q,:], quantiles[-(q+1),:],
                alpha=0.3, color=color)

    if not (test_y is None):
        plt.plot(test_x, test_y, 'o', markersize=4, color='r')
        plt.plot(train_x, train_y, 'o', markersize=8, color='tab:orange')
    #plt.ylim([-3,3])

    plt.axis('off')
    plt.savefig(name, bbox_inches='tight')
    plt.close()

    return fig

def plot_gp_errors(depth_list, errors, ylabel, name='gp_error.pdf'):
    fontsize = 40
    ticksize = 15
    fig = plt.figure(figsize=(7,7))
    for row in range(errors.shape[0]):
        plt.plot(depth_list, errors[row, :])
    plt.gca().set_ylim(top = np.amax(errors[:,0]))
    plt.gca().set_ylim(bottom = np.amin(errors))
    plt.legend(['MLE, $\mu=0$', 'MAP, $\mu=0$', 'MLE', 'MAP', 'Marginalised'],
            prop={'size': ticksize })
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel('Depth $L$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.savefig(name, bbox_inches='tight')
    plt.close()

def plot_gdf(input_gdf, name, extent, ax=None, world=None, close=False,
        heatmap = False):
    #http://data.daff.gov.au/anrdl/metadata_files/pa_nsaasr9nnd_02211a04.xml 
    if world is None:
        world = gpd.read_file('code/experiments/data/aust_cd66states.shp')
    if ax is None:
        # Plot Australia
        ax = world.plot(color=None, edgecolor='black', figsize=(7,7),
                zorder=10, facecolor='none')
        # Plot a white background
        rect = gpd.GeoSeries([Polygon(\
                [(extent[0],extent[2]), (extent[1],extent[2]), 
                    (extent[1],extent[3]), (extent[0],extent[3])])])
        rect = gpd.GeoDataFrame({'geometry': rect, 'rect':[1]})

        res_difference = gpd.overlay(rect, world, how='difference')
        res_difference.plot(facecolor='white', ax=ax, zorder=3)

        # Set the boundary
        ax.set_xlim([extent[0], extent[1]])
        ax.set_ylim([extent[2], extent[3]])

    # Different plot settings if heatmap or data points
    if heatmap:
        markersize = 100
        linewidth = 0
        marker = 's'
        zorder = 2
        legend = False
        cax = None
    else:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        markersize = 10
        linewidth = 1
        marker = 'o'
        zorder = 4
        legend = True

    input_gdf.plot(ax=ax, column='value', markersize=markersize, legend=legend,
            cax=cax, marker=marker,
            zorder = zorder, edgecolor='k', linewidth=linewidth, cmap='viridis')
    plt.savefig(name, bbox_inches='tight')
    if close:
        plt.close()

    return ax, world

def plot_proposal(mh, fname, extent):
    pdf = mh.get_proposal()
    plt.imshow(pdf, origin='lower', extent=extent)
    plt.gca().set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2])))
    plt.savefig(fname)
    plt.close()



