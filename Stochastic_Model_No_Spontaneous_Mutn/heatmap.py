# Created on 13/01/2025
# By Abheepsa Nanda
# Code to create a heatmap with x and y being two factors out of gamma, number of initial serotypes,
# recombination rate and mutation rate and the colourmapped z axis being either average number of serotypes left or 

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.colors as mcolors
import pandas as pd
#import os


def heatmap(response_variable, num_seros,m, data, row_labels, col_labels, ax=None,
            cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    data[data == -1] = np.nan
    if ax is None:
        ax = plt.gca()

    #if cbar_kw is None:
    #    cbar_kw = {}
    print(data)
    print(np.nanmax(data))
    if response_variable==0:
        boundaries = np.arange(0,num_seros+0.2, 0.2)
    elif response_variable==1:
        boundaries = np.arange(0,np.nanmax(data)+100, 100)
    # Define a colormap (you can use a built-in one like 'viridis' or 'plasma')
    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='grey')
    
    # Normalize the colors according to the boundaries
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    # Plot the heatmap
    im = ax.imshow(data,  cmap=cmap, norm=norm, **kwargs)
    ax.set_title('Serotypes = '+str(num_seros)+', m = '+str(m))
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, boundaries = boundaries)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
    ax.set_xlabel('Sigma')
    ax.set_ylabel('Gamma')
    ax.xaxis.set_label_position('top')
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=( "white", "black"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

num_muts = 2
num_seros = 10
init_muts = [0,1]
gamma = [0.1, 0.5, 1.0]
num_iter = 100
sigma = [[1e-13,1e-13],[1e-12,1e-12], [1e-11,1e-11]]
mut = [[0.02,0.05], [0.04,0.05], [0.05,0.05]]

response_variable = 0 #0 for serotypes left, 1 for fixation_times
bar_lab = ["Serotypes left", "Fixation time (x50)"]

fig, ax = plt.subplots(1,3, figsize = (14,3))

for m in range(len(mut)):
    gridres = np.zeros((len(sigma), len(gamma)), dtype = float)
    for i in range(len(gamma)):
        for j in range(len(sigma)):        
            if response_variable==0:
                instance = 'serosend_'+'numm_2_sero_'+str(num_seros)+'_initm_'+str(init_muts)+'_g_'+str(gamma[i])+'_sig_'+str(sigma[j])+'_m_'+str(mut[m])
                arr = pd.read_csv('Summary/Seros_end/'+instance+'.csv', delimiter = ',').to_numpy()
            elif response_variable==1:
                arr = np.loadtxt('Summary/Fixation_times/tfix_sero_'+str(num_seros)+'_g_'+str(gamma[i])+'_sig_'+str(sigma[j])+'_m_'+str(m)+'.csv', delimiter = ',')
            #arr[arr<=0] = NaN
            counter = np.count_nonzero(np.isnan(arr))
            if counter >25:
                print(counter)
                print(arr)
                gridres[i,j] = -1
            else:
                val_mean = np.nanmean(arr)
                print(arr)
                gridres[i,j] = val_mean
    im, cbar = heatmap(response_variable, num_seros,mut[m], gridres, gamma, np.array(sigma)[:,0], ax=ax[m], cbarlabel=bar_lab[response_variable])
    texts = annotate_heatmap(im, valfmt="{x:.2f} ", threshold = 1)

plt.savefig('heatmaptimefix_sero_'+str(num_seros)+'_initmuts='+str(init_muts)+'.png')
plt.show()
print(gridres)
