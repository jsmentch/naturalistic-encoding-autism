# convenience functions to apply hrf to features
# jsm 3/4/21
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import gamma
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feat_in", type=str, help="input feature file")
    parser.add_argument("output_dir", type=str, help="path to dir to save output")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-f", "--file", action='store_true', help="run on a file")
    group.add_argument("-d", "--directory", action='store_true', help="run on all .npy files in a directory")
    #parser.set_defaults(file=True, directory=False)
    args = parser.parse_args()
    if args.file:
        basename = Path(args.feat_in).stem
        feat=np.load(args.feat_in)
        feat_hrf=apply_optimal_hrf_10hz(feat)
        feat_out = resample_1hz(feat_hrf)
        np.save(args.output_dir+'/'+basename+'_hrf.npy', feat_out)
    elif args.directory:
        print(args.feat_in + "*.npy")
        file_list = glob.glob(args.feat_in + "*.npy")
        for f in file_list:
            basename = Path(f).stem
            feat=np.load(f)
            feat_hrf=apply_optimal_hrf_10hz(feat)
            feat_out = resample_1hz(feat_hrf)
            np.save(args.output_dir+'/'+basename+'_hrf.npy', feat_out)
            
def apply_canonical_hrf_10hz(feat_in,hz):
    #applies optimal hrf from merlin study to a 10hz feature 
    t=np.arange(round(32*hz))
    optimal_hrf=spm_hrf_compat(t,
                   peak_delay=round(6*hz),
                   under_delay=round(16*hz),
                   peak_disp=round(1*hz),
                   under_disp=round(1*hz),
                   p_u_ratio = 6,
                   normalize=False,)
    n_to_remove = optimal_hrf.shape[0]-1 #how much to trim from end after convolution
    for m in np.arange(feat_in.shape[1]):
        feat_in[:,m]=np.convolve(feat_in[:,m], optimal_hrf)[:-n_to_remove]
    return feat_in


def apply_optimal_hrf_10hz(feat_in,hz):
    #applies optimal hrf from merlin study to a 10hz feature 
    t=np.arange(round(32*hz))
    optimal_hrf=spm_hrf_compat(t,
                   peak_delay=round(6*hz),
                   under_delay=round(18*hz),
                   peak_disp=round(1.4*hz),
                   under_disp=round(1.5*hz),
                   p_u_ratio = 4,
                   normalize=False,)
    n_to_remove = optimal_hrf.shape[0]-1 #how much to trim from end after convolution
    for m in np.arange(feat_in.shape[1]):
        feat_in[:,m]=np.convolve(feat_in[:,m], optimal_hrf)[:-n_to_remove]
    return feat_in


def apply_optimal_hrf_10hz_NEW(feat_in, hz):
    # Applies optimal HRF to a 10Hz feature
    t = np.arange(round(32 * hz))
    optimal_hrf = spm_hrf_compat(
        t,
        peak_delay=round(6 * hz),
        under_delay=round(18 * hz),
        peak_disp=round(1.4 * hz),
        under_disp=round(1.5 * hz),
        p_u_ratio=4,
        normalize=False,
    )
    
    n_to_remove = optimal_hrf.shape[0] - 1  # How much to trim from end after convolution
    
    # Create a new array to hold the convolved result
    feat_out = np.zeros_like(feat_in, dtype=float)  # Ensure it's float
    
    for m in np.arange(feat_in.shape[1]):
        # Perform the convolution and store in a separate array
        feat_out[:,m] = np.convolve(feat_in[:,m], optimal_hrf, mode='full')[:-n_to_remove]
    
    return feat_out


def apply_alternative_hrf_10hz_NEW(feat_in, hz):
    # Applies optimal HRF to a 10Hz feature
    t = np.arange(round(32 * hz))
    optimal_hrf = spm_hrf_compat(
        t,
        peak_delay=round(5 * hz),
        under_delay=round(12 * hz),
        peak_disp=round(1.4 * hz),
        under_disp=round(1.5 * hz),
        p_u_ratio=4,
        normalize=False,
    )
    
    n_to_remove = optimal_hrf.shape[0] - 1  # How much to trim from end after convolution
    
    # Create a new array to hold the convolved result
    feat_out = np.zeros_like(feat_in, dtype=float)  # Ensure it's float
    
    for m in np.arange(feat_in.shape[1]):
        # Perform the convolution and store in a separate array
        feat_out[:,m] = np.convolve(feat_in[:,m], optimal_hrf, mode='full')[:-n_to_remove]
    
    return feat_out




def resample_1hz(feat_in):
    # input: 2d array [time,feat]
    # output: 2d array [time,feat] time downsampled by factor of 10 eg from 10 hz to 1 hz]
    feat_in = feat_in[::10,:] 
    return feat_in

# SPM's HRF
def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio = 6,
                   normalize=True,
                  ):
    """ SPM HRF function from sum of two gamma PDFs

    This function is designed to be partially compatible with SPMs `spm_hrf.m`
    function.

    The SPN HRF is a *peak* gamma PDF (with location `peak_delay` and dispersion
    `peak_disp`), minus an *undershoot* gamma PDF (with location `under_delay`
    and dispersion `under_disp`, and divided by the `p_u_ratio`).

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF
    peak_delay : float, optional
        delay of peak
    peak_disp : float, optional
        width (dispersion) of peak
    under_delay : float, optional
        delay of undershoot
    under_disp : float, optional
        width (dispersion) of undershoot
    p_u_ratio : float, optional
        peak to undershoot ratio.  Undershoot divided by this value before
        subtracting from peak.
    normalize : {True, False}, optional
        If True, divide HRF values by their sum before returning.  SPM does this
        by default.

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    See ``spm_hrf.m`` in the SPM distribution.
    """
    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape)
    pos_t = t[t > 0]
    peak = gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)

if __name__ == "__main__":
    main()