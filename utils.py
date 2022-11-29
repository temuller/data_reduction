import numpy as np
import matplotlib.pyplot as plt
from ccdproc import CCDData

import warnings
from astropy.utils.exceptions import AstropyWarning

def plot_image(data):
    """Plots a 2D image.
    
    Parameters
    ----------
    data: ndarray
        Image data.
        
    Returns
    -------
    ax: `~.axes.Axes`
        Plot axis.
    """
    m, s = np.nanmean(data), np.nanstd(data)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(data, interpolation='nearest',
               cmap='gray',
               vmin=m-s, vmax=m+s,
               origin='lower')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)    
    return ax

def obs_plots(observations, obstype):
    """Plots all images of a given ``obstype``.
    
    Parameters
    ----------
    observations: `~ImageFileCollection`
        Table-like object with images information.
    obstype: str
        Type of Image. E.g. ``BIAS``, ``FLAT``.
    """    
    for filename in observations.files_filtered(include_path=True, obstype=obstype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyWarning)
            ccd = CCDData.read(filename, hdu=1, unit=u.electron)
            plot_image(ccd.data.T)
