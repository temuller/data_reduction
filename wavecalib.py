import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from astropy.stats import sigma_clip
from scipy.optimize import minimize, curve_fit

import emcee
import corner

################
# ARC spectrum #
################

def gaussian(x, *params):
    """Simple Gaussian function for fitting.
    """
    amp, x0, sigma = params
    return amp * np.exp(-(x - x0)**2 / 2 / sigma**2)


def fit_gauss2peaks(arc_disp, arc_profile, peak_ids, plot_diag=False):
    """Fits Gaussian functions to a lamp.

    Parameters
    ----------
    arc_disp: array
        Dispersion axis of the lamp (e.g. columns in an image).
    arc_profile: array
        Profile of the lamp (e.g. intensity/amplitude in an image).
    peak_ids: array-like
        Indeces of the peaks in the lamp.

    Returns
    -------
    amplitudes: array
        Amplitudes of the peaks/spectral lines.
    centers: array
        Centers of the peaks/spectral lines.
    sigmas: array
        Standard deviations of the peaks/spectral lines.
    """
    amplitudes, centers, sigmas = [], [], []
    sigma = 1.0  # initial guess
    width = 4  # width of the lines to fit
    for i in peak_ids:
        center = arc_disp[i]
        amplitude = arc_profile[i]

        guess = (amplitude, center, sigma)
        bounds = ((0, center - 2, 0), (np.inf, center + 2, np.inf))

        # indices to bound the profile of each line
        i_min = int(center - width)
        i_max = int(center + width)

        try:
            popt, pcov = curve_fit(gaussian,
                                   arc_disp[i_min:i_max],
                                   arc_profile[i_min:i_max],
                                   p0=guess, bounds=bounds)
            amp, center, sigma = popt

            if plot_diag:
                # diagnostic plot with the fit result
                x_mod = np.linspace(center - 3 * sigma, center + 3 * sigma, 1000)
                y_mod = gaussian(x_mod, *popt)
                mask = (arc_disp >= x_mod.min()) & (arc_disp <= x_mod.max())

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(x_mod, y_mod, label='Gaussian fit')
                ax.plot(arc_disp[mask], arc_profile[mask])
                ax.set_ylabel('Intensity', fontsize=16)
                ax.set_xlabel('Dispersion axis (pixels)', fontsize=16)
                ax.legend()
                plt.show()
        except RuntimeError:
            # curve_fit failed to converge...skip
            continue

        amplitudes.append(amp)
        centers.append(center)
        sigmas.append(sigma)

    amplitudes = np.array(amplitudes)
    centers = np.array(centers)
    sigmas = np.array(sigmas)

    # filter lamp_lines to keep only lines that were fit
    fit_mask = np.isfinite(centers)
    amplitudes = amplitudes[fit_mask]
    centers = centers[fit_mask]
    sigmas = sigmas[fit_mask]

    return amplitudes, centers, sigmas


def find_arc_peaks(data, plot_solution=False, plot_diag=False):
    """Fits Gaussian functions to a lamp.

    Parameters
    ----------
    data: `~astropy.nddata.CCDData`-like, array-like
        Image data.
    optimize: bool, default ``True``
        If ``True``, the peak centers are found by fitting Gaussians.
    plot_solution: bool, default ``False``
        If ``True``, the lamp with the solution is plotted.

    Returns
    -------
    arc_pixels: array
        Centers of the peaks in pixels.
    arc_peaks: array
        Peaks intensity:
    arc_sigmas: array
        Standard deviations of the Gaussian fits.
    """
    ny, nx = data.shape
    cy, cx = ny // 2, nx // 2

    arc_disp = np.arange(nx)
    arc_profile = data[cy][::-1]  # the axis is inverted
    arc_profile -= arc_profile.min()

    # initial peak estimation
    prominence = 100  # minimum line intensity
    peak_ids = find_peaks(arc_profile, prominence=prominence)[0]

    # peak estimation with gaussian fitting
    arc_peaks, arc_pixels, arc_sigmas = fit_gauss2peaks(arc_disp, arc_profile, peak_ids, plot_diag)

    # saturation mask / maximum line intensity
    sat_mask = arc_peaks < 64000
    arc_pixels = arc_pixels[sat_mask]
    arc_peaks = arc_peaks[sat_mask]

    if plot_solution:
        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(arc_profile)
        ax.scatter(arc_disp[peak_ids], arc_profile[peak_ids], marker='*', color='g', label='Initial Peaks')
        ax.scatter(arc_pixels, arc_peaks, marker='*', color='r', label='Optimised Peaks (Gaussian Fit)')

        ax.set_ylabel('Intensity', fontsize=16)
        ax.set_xlabel('Dispersion axis (pixels)', fontsize=16)
        ax.legend()
        plt.show()

    return arc_pixels, arc_peaks, arc_sigmas


#######################
# Wavelength Solution #
#######################

def find_nearest(array1, array2):
    """Finds the nearest values between two arrays.

    Finds the values in ``array2`` nearest to ``array1``,
    without repetition.
    **NOTE:* The arrays are assumed to be sorted. In addition,
    each input array is assumed to not have repeated values.

    Parameters
    ----------
    array1: array
        First array.
    array2: array
        Second array.

    Returns
    -------
    ids1: list
        Indices of the values in ``array1``.
    ids2: list
        Indices of the values in ``array2``.
    """
    ids1, ids2 = [], []
    if len(array1) < len(array2):
        short_array = array1
        long_array = array2
    else:
        long_array = array1
        short_array = array2

    i_min = 0
    for i, val in enumerate(short_array):
        imin_temp = np.argmin(np.abs(val - long_array[i_min:]))
        i_min = np.where(long_array == long_array[i_min:][imin_temp])[0][0]
        if len(array1) < len(array2):
            ids1.append(i)
            ids2.append(i_min)
        else:
            ids1.append(i_min)
            ids2.append(i)

    return ids1, ids2

def wavelength_function(params, x, model='legendre'):
    """Function to fit for the wavelength solution.
    
    Parameters
    ----------
    params: array-like
        Whatever parameters the function accepts.
    x: array
        x-coordinate values.
    model: str
        Model to use: ``legendre`` or ``chebyshev``
        
    Returns
    -------
    y_model: array
        Model evaluated at ``x``.
    """
    if model=='legendre':
        y_model = np.polynomial.legendre.legval(x, params)
    elif model=='chebyshev':
        y_model = np.polynomial.chebyshev.chebval(x, params)
    
    return y_model
    
    
# Quick solution

def chi_sq(params, arc_pixels, lamp_wave, model='legendre'):
    """Chi squared for the wavelength solution.

    Parameters
    ----------
    params: array-like
        Parameters for the wavelength-solution function.
    arc_pixels: array
        Dispersion axis in pixels.
    lamp_wave: array
        Wavelengths of a lamp.

    Returns
    -------
    chi: float
        Chi squared value.
    """
    model_wave = wavelength_function(params, arc_pixels, model)
    ids_lamp, ids_model = find_nearest(lamp_wave, model_wave)

    M, N = len(lamp_wave[ids_lamp]), len(params)
    residual = model_wave[ids_model] - lamp_wave[ids_lamp]

    # reduced chi square
    chi = np.sum(residual ** 2) / (M - N)

    return chi


def quick_wavelength_solution(arc_pixels, lamp_wave, params=None, model='legendre', k=3, niter=3, plot_solution=False,
                              data=None):
    """Finds a wavelength solution with a simple fit.

    Parameters
    ----------
    params: array-like
        Parameters for the wavelength-solution function.
    arc_pixels: array
        Dispersion axis in pixels.
    lamp_wave: array
        Wavelengths of a lamp.
    params: array-like, default ``None``
        Initial guess for the parameters for the
        wavelength-solution function.
    arc_sigmas: array, default ``None``
        Sigmas used for ``find_nearest``.

    data: `~astropy.nddata.CCDData`-like, array-like
        Image data for plotting purposes only.

    Returns
    -------
    params: array-like
        Parameters for the wavelength-solution function.
    """
    if params is None:
        params = [4000] + (k - 1) * [0]

    arc_pixels0 = np.copy(arc_pixels)
    if arc_sigmas is None:
        arc_sigmas0 = None
    else:
        arc_sigmas0 = np.copy(arc_sigmas)

    if niter > 0:
        for i in range(niter):
            results = minimize(chi_sq, params, args=(arc_pixels0, lamp_wave, model), method='Powell')
            params = results.x

            calibrated_wave = wavelength_function(params, arc_pixels0, model)

            ids_lamp, ids_calwave = find_nearest(lamp_wave, calibrated_wave)
            residuals = calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]

            # outliers removal
            mask = ~sigma_clip(residuals, sigma=2.0).mask
            arc_pixels0 = arc_pixels0[ids_calwave][mask]
            if arc_sigmas0 is not None:
                arc_sigmas0 = arc_sigmas0[ids_calwave][mask]

    results = minimize(chi_sq, params, args=(arc_pixels0, lamp_wave, model), method='Powell')
    params = results.x

    outliers_mask = np.array([False if pixel in arc_pixels0 else True for pixel in arc_pixels])
    if plot_solution:
        check_solution(params, arc_pixels, lamp_wave, outliers_mask, data, model)

    return params


# MCMC solution

def log_likelihood(params, arc_pixels, lamp_wave, arc_sigmas, model):
    """Logarithm of the likelihood for the wavelength solution.
    """
    model_wave = wavelength_function(params, arc_pixels, model)
    ids_lamp, ids_model = find_nearest(lamp_wave, model_wave, arc_sigmas)

    M, N = len(lamp_wave[ids_lamp]), len(params)
    residual = model_wave[ids_model] - lamp_wave[ids_lamp]
    if arc_sigmas is None:
        std = 1
    else:
        std = arc_sigmas[ids_model]

    ll = -0.5 * np.sum(residual ** 2 / std ** 2) / (M - N)
    
    return ll

def log_prior(params):
    """Priors for the fitting.
    """
    for p in params:
        if -1e4 < p < 1e4:
            continue
        else:
            return -np.inf
    return 0.0
    
def log_probability(params, arc_pixels, lamp_wave, arc_sigmas, model):
    """Posterior function.
    """
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, arc_pixels, lamp_wave, arc_sigmas, model)

def optimised_wavelength_solution(params, arc_pixels, lamp_wave, arc_sigmas=None, model='legendre'):

    pos = params + 1e-4 * np.random.randn(32, len(params))
    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(arc_pixels, lamp_wave, arc_sigmas, model)
    )
    sampler.run_mcmc(pos, 3000, progress=True)
    flat_samples = sampler.get_chain(discard=500, thin=30, flat=True)
    
    # plotting
    labels = [f"p{i}" for i in range(len(params))]
    corner.corner(flat_samples, labels=labels)
    plt.show()
    
    params = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        params.append(mcmc[1])
        
    print(params)

    return params

# Checking output

def check_solution(params, arc_pixels, lamp_wave, mask=None, data=None, model='legendre'):
    """Shows the residuals of the wavelength solution.

    Parameters
    ----------
    params: array-like
        Parameters for the wavelength-solution function.
    arc_pixels: array
        Dispersion axis in pixels.
    lamp_wave: array
        Wavelengths of a lamp.
    mask: bool array, default ``None``
        Mask of outliers.
    data: `~astropy.nddata.CCDData`-like, array-like
        Image data for plotting purposes only.
    """
    calibrated_wave = wavelength_function(params, arc_pixels, model)
    ids_lamp, ids_calwave = find_nearest(lamp_wave, calibrated_wave)

    residuals = calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]
    mean, std = residuals.mean(), residuals.std()

    if data is not None:
        ny, nx = data.shape
        cy, cx = ny // 2, nx // 2
        arc_disp = np.arange(nx)
        arc_profile = data[cy][::-1]
        arc_wave = wavelength_function(params, arc_disp, model)

        # plot ARC lines
        fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True,
                                 height_ratios=[3, 4, 2], gridspec_kw=dict(hspace=0))

        axes[0].plot(arc_wave, arc_profile)
        axes[0].set_ylabel('Intensity', fontsize=16)
        for wave in calibrated_wave:
            axes[0].axvline(wave, ls='--', alpha=0.2, color='k')
        i = 1
    else:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                 height_ratios=[4, 2], gridspec_kw=dict(hspace=0))
        i = 0
    # plot pixels vs wavelengths
    axes[i].scatter(calibrated_wave[ids_calwave], arc_pixels[ids_calwave], marker='*', c='r')

    # fit
    arc_model = np.arange(arc_pixels[ids_calwave].min(), arc_pixels[ids_calwave].max(), 10)
    wave_model = wavelength_function(params, arc_model, model)
    axes[i].plot(wave_model, arc_model, lw=2, c='g', label=f'fit ({model} k={len(params) - 1})', zorder=0)

    # plot wavelength solution residuals
    axes[i + 1].scatter(calibrated_wave[ids_calwave], residuals, marker='*', c='r')

    if mask is not None:
        masked_calibrated_wave = wavelength_function(params, arc_pixels[~mask], model)
        ids_lamp, ids_calwave = find_nearest(lamp_wave, masked_calibrated_wave)
        masked_res = masked_calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]
        mean, std = masked_res.mean(), masked_res.std()

        masked_calibrated_wave = wavelength_function(params, arc_pixels[mask], model)
        ids_lamp, ids_calwave = find_nearest(lamp_wave, masked_calibrated_wave)
        masked_res = masked_calibrated_wave[ids_calwave] - lamp_wave[ids_lamp]

        # plot outliers
        axes[i].scatter(masked_calibrated_wave[ids_calwave], arc_pixels[mask][ids_calwave], marker='x', c='b',
                        label='outliers')
        axes[i].legend(fontsize=14)
        axes[i + 1].scatter(masked_calibrated_wave[ids_calwave], masked_res, marker='x', c='b')

    axes[i + 1].axhline(mean, c='k')
    axes[i + 1].axhline(mean + std, c='k', ls='--')
    axes[i + 1].axhline(mean - std, c='k', ls='--')
    axes[i + 1].set_ylabel(r'Residual ($\AA$)', fontsize=16)
    axes[i + 1].set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
    axes[i + 1].set_ylim(mean - 3 * std, mean + 3 * std)
    plt.show()

    print(f'Residual mean: {mean:.1f} +/- {std:.1f} angstroms')