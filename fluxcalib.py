import pandas as pd
import matplotlib.pyplot as plt

def get_calspec(plot_spec=False):
    """Retrieves a calibrated spectrum.
    
    Returns
    -------
    calspec: `~DataFrame`
        Calibrated spectrum (wavelength, flux, flur error, delta lambda).
    """
    filename = 'https://ftp.eso.org/pub/usg/standards/ctiostan/fhr8634.dat'
    calspec = pd.read_csv(filename, names=['wave', 'flux', 'eflux', 'dlam'], delim_whitespace=True)
    
    if plot_spec:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(calspec['wave'], calspec['flux'], lw=2)
        ax.set_xlabel('Wavelength ($\AA$)', fontsize=16)
        ax.set_ylabel(r'$F_{\lambda}$', fontsize=16)
        plt.show()
        
    return calspec


