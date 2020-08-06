import disper
import sys
import numpy as np
import glob
import os
import time
import sys
import humanize
from datetime import datetime, timedelta
from mpi4py import MPI
from mpisetup import totalrank, rank, log, comm

def DMs(DMstart,DMend,dDM):
    """
    Calculate the number of DMs searched between DMstart and DMend, with spacing dDM * DM.
    Required:
    DMstart   - Starting Dispersion Measure in pc cm-3
    DMend     - Ending Dispersion Measure in pc cm-3
    dDM       - DM spacing in pc cm-3
    """

    #NDMs = np.log10(float(DMend)/float(DMstart))/np.log10(1.0+dDM)
    NDMs = (DMend-DMstart)/dDM

    return int(np.round(NDMs))


def delay2(freq, dm):
    """
    Calculate the relative delay due to dispersion over a given frequency
    range in Hz for a particular dispersion measure in pc cm^-3.  Return
    the dispersive delay in seconds.  Same as delay, but w.r.t to highest frequency.
    ***Used to simulate a dispersed pulse.***
    Required:
    freq - 1-D array of frequencies in MHz
    dm   - Dispersion Measure in pc cm-3
    """
    # Dispersion constant in MHz^2 s / pc cm^-3
    _D = 4.148808e3
    # Delay in s
    tDelay = dm*_D*((1/freq)**2 - (1/freq.max())**2)

    return tDelay


def Threshold(ts, thresh, clip=3, niter=1):
    """
    Wrapper to scipy threshold a given time series using Scipy's threshold function (in
    scipy.stats.stats).  First it calculates the mean and rms of the given time series.  It then
    makes the time series in terms of SNR.  If a given SNR value is less than the threshold, it is
    set to "-1".  Returns a SNR array with values less than thresh = -1, all other values = SNR.
    Also returns the mean and rms of the timeseries.
    Required:
    ts   -  input time series.
    Options:
    thresh  -  Time series signal-to-noise ratio threshold.  default = 5.
    clip    -  Clipping SNR threshold for values to leave out of a mean/rms calculation.  default = 3.
    niter   -  Number of iterations in mean/rms calculation.  default = 1.
    Usage:
    >>sn, mean, rms = Threshold(ts, *options*)
    """
    #  Calculate, robustly, the mean and rms of the time series.  Any values greater than 3sigma are left
    #  out of the calculation.  This keeps the mean and rms free from sturation due to large deviations.

    mean = np.mean(ts)
    std  = np.std(ts)
    #print mean,std

    if niter > 0:
        for i in range(niter):
            ones = np.where((ts-mean)/std < clip)[0]  # only keep the values less than 3sigma
            mean = np.mean(ts[ones])
            std  = np.std(ts[ones])
    SNR = (ts-mean)/std
    # Often getting "invalid value encountered in less" here:
    with np.errstate(invalid='raise'):
        SNR[SNR<thresh]=-1
    #SNR = st.threshold((ts-mean)/std, threshmin=thresh, newval=-1)

    return SNR, mean, std

def Decimate_ts(ts, ndown=2):
    """
    Takes a 1-D timeseries and decimates it by a factore of ndown, default = 2.
    Code adapted from analysis.binarray module:
      http://www.astro.ucla.edu/~ianc/python/_modules/analysis.html#binarray
    from Ian's Python Code (http://www.astro.ucla.edu/~ianc/python/index.html)

    Optimized for time series' with length = multiple of 2.  Will handle others, though.
    Required:

    ts  -  input time series
    Options:

    ndown  -  Factor by which to decimate time series. Default = 2.
              if ndown = 1, returns ts
    """

    if ndown==1:
       return ts

    ncols = len(ts)
    n_rep = ncols / ndown
    ts_ds = np.array([ts[i::ndown][0:n_rep] for i in range(ndown)]).mean(0)

    return ts_ds


class OutputSource():

      pulse = None  # Pulse Number
      SNR   = None  # SNR of pulse
      DM    = None  # DM (pc/cm3) of pulse
      time  = None  # Time at which pulse ocurred
      dtau  = None  # Temporal resolution of time series
      dnu   = None  # Spectral resolution
      nu    = None  # Central Observing Frequency
      mean  = None  # Mean in the time series
      rms   = None  # RMS in the time series

      formatter = "{0.pulse:07d}    {0.SNR:10.6f}     {0.DM:10.4f}     {0.time:10.6f} "+\
                 "     {0.dtau:10.6f}     {0.dnu:.4f}     {0.nu:.4f}    {0.mean:.5f}"+\
                 "    {0.rms:0.5f}\n "

      def __str__(self):
          return self.formatter.format(self)

def savitzky_golay(y, window_size, order, deriv=0):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter
    This implementation is based on [1]_.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techhniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute
        (default = 0 means only smoothing)
    Returns
    -------
    y_smooth : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    >>> t = np.linspace(-4, 4, 500)
    >>> y = np.exp(-t ** 2)
    >>> np.random.seed(0)
    >>> y_noisy = y + np.random.normal(0, 0.05, t.shape)
    >>> y_smooth = savitzky_golay(y, window_size=31, order=4)
    >>> print np.rms(y_noisy - y)
    >>> print np.rms(y_smooth - y)
    References
    ----------
    .. [1] http://www.scipy.org/Cookbook/SavitzkyGolay
    .. [2] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [3] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")

    if window_size % 2 != 1 or window_size < 1:
        #raise TypeError("window_size size must be a positive odd number")
        window_size += 1

    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order + 1)

    half_window = (window_size - 1) // 2

    # precompute coefficients
    b = np.mat([[k ** i for i in order_range]
                for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv]

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(y, m, mode='valid')

def snr(x):return (x-x.mean())/x.std()

def bpf(x, windows = 40):
    bp = savitzky_golay(x,windows,1)
    x2 = x / bp
    mask = np.where(snr(x2)>1)[0]
    mask2= np.zeros(x.shape[0])
    mask2[mask] = 1.
    y = np.ma.array(x, mask = mask2)
    bp = savitzky_golay(y,windows,1)
    fit = np.ma.polyfit(np.arange(len(y)),y,4)
    p = np.poly1d(fit)(np.arange(len(y)))[mask]
    bp = x
    bp[mask] = np.poly1d(fit)(np.arange(len(y)))[mask]
    bp = savitzky_golay(bp,windows,2)
    return bp

def fold(t,period,T0=0):
    time = np.arange(len(t))
    epoch = np.floor( 1.*(time - T0)/period )
    phase = 1.*(time - T0)/period - epoch
    foldt = t[np.argsort(phase)]
    return Decimate_ts(foldt, 1.*len(t)/period )


def RFImask(spr):#sp has shape[:,:]
    x = np.where(abs(spr.mean(1))>np.sort(spr.mean(1))[spr.shape[0]/2]+np.sort(spr.mean(1))[spr.shape[0]/2]-np.sort(spr.mean(1))[1])
    y = np.where(abs(spr.mean(0))>np.sort(spr.mean(0))[spr.shape[1]/2]+np.sort(spr.mean(0))[spr.shape[1]/2]-np.sort(spr.mean(0))[1])
    return [x[0],y[0]]

def massagesp(spectrometer, windows_x=43,windows_y=100):
    bp = bpf(spectrometer.mean(0),windows_x)
    spectrometer /= bp
    bl = bpf(spectrometer.mean(1),windows_y)
    spectrometer = (spectrometer.T - bl).T
    mask  = np.array ( RFImask(spectrometer) )
    mask2 = np.zeros((spectrometer.shape))
    mask2[mask[0],:] = 1.
    mask2[:,mask[1]] = 1.
    temp_spec = np.ma.array(spectrometer, mask = mask2 )
    mean = temp_spec.mean()
    spectrometer[mask[0],:] = mean
    spectrometer[:,mask[1]] = mean
    spectrometer -= mean
    return spectrometer

def progress(t_start, fraction, message):
    if fraction > 0:
        elapsed = (datetime.now() - t_start).total_seconds()
        remaining = timedelta(seconds = (elapsed / fraction) - elapsed)
        log("%s (%0.1f%% complete, %s remaining)" %
            (message, 100*fraction, humanize.naturaldelta(remaining)))

if __name__ == '__main__':
    # TODO: make selecting the freq range more robust by auto-scaling
    # it to the number of bins in the file.
    fcl = 200  #/4
    fch = 3700 #/4

    # TODO: make selecting tuning more robust by detecting which is 
    # actually lower. The HDF file (like the DRX as well, apparently) 
    # can contain them in either order.
    pol = 1

    DMstart = 300 # Initial DM trial
    DMend   = 400 # Final DM trial

    maxpw = 5 # Maximum pulse width to search (seconds)
    thresh= 5.0 #SNR cut off

    import h5py
    import humanize
    h5f = h5py.File(sys.argv[1], 'r')['Observation1']
    num_time_bins = h5f['time'].shape[0]
    tInt = h5f['time'][1] - h5f['time'][0]  # Temporal resolution
    log("%s time bins, %.9f sec each" % (humanize.intcomma(num_time_bins), tInt))

    # These are time offsets rather than independent filenames
    time_bins_per_file = min(3000, num_time_bins / totalrank)
    fn = range(0, num_time_bins, time_bins_per_file)
    fpp = len(fn) / totalrank # Files per process
    numberofFiles = fpp * totalrank

    npws = int(np.round(np.log2(maxpw/tInt)))+1 

    spectarray = np.zeros((fpp, time_bins_per_file, fch-fcl))

    h5t = h5f['Tuning%d' % (pol+1)]
    freq = h5t['freq'][fcl:fch]
    freq /= 10**6
    cent_freq = np.median(freq)
    BW = freq.max()-freq.min()

    # Announce the setup for the computation
    if rank == 0:
        log("Using frequency buckets %i:%i" % (fcl, fch))
        log("Tuning %d, central freq %f, BW %f, %d buckets" % (pol, cent_freq, BW, freq.shape[0]))
        log("%i files, %i per processor * %i processors = %i" % 
                (len(fn), fpp, totalrank, numberofFiles))
        log("Max pulse width %0.3fs" % maxpw)
        log("spectarray will have dimensions %s" % str(spectarray.shape))
        log("PHASE1: Process spectrograms")

    outname = 'spectarray%02i.npy' % rank
    #combine spectrogram and remove background
    for i in range(fpp):
        findex = rank*fpp + i
        time_bin_start = fn[findex]
        log("Loading #%i (%i of %i): %d" % (findex, i, fpp, time_bin_start))
        spx = h5t['XX'][time_bin_start : (time_bin_start + time_bins_per_file), fcl:fch]
        spy = h5t['YY'][time_bin_start : (time_bin_start + time_bins_per_file), fcl:fch]
        sp = (spx + spy) / 2
        spectarray[i,:sp.shape[0],:] = massagesp( sp, 10, 50 )
    log("Writing %s" % outname)
    np.save(outname, spectarray)
    #sys.exit()

    # Or we can pick up the saved spectarrays
    """
    log("Loading %s" % outname)
    spectarray = np.load(outname)
    """

    txtsize=np.zeros((npws,2),dtype=np.int32) #fileno = txtsize[ranki,0], pulse number = txtsize[ranki,1],ranki is the decimated order of 2
    txtsize[:,0]=1 #fileno start from 1

    # Calculate which DMs to test
    DM = DMstart
    DMtrials = DMstart
    if rank == 0:
        log("Approaching phase 2, calculating which DMs to try...")
        while DM < DMend:
            if DM < 1000:
                dDM = 0.1
            elif DM >= 1000:
                dDM = 1.
            DM += dDM
            DMtrials = np.append(DMtrials,DM)
        log("Will test %i DMs in range %.1f:%.1f" % (len(DMtrials), DMstart, DMend))

    DMtrials = comm.bcast(DMtrials, root=0)
    
    if rank == 0:
        log("PHASE 2: DM search")

    t_start = datetime.now()
    for DM in DMtrials:
        if rank == 0:
            progress(t_start, (DM-DMstart)/(DMend-DMstart), "DM trial %f" % DM)
        tb=np.round((delay2(freq,DM)/tInt)).astype(np.int32)

        ts=np.zeros((tb.max()+numberofFiles*time_bins_per_file))
        for freqbin in range(len(freq)):
            for i in range(fpp):
                ts[tb.max()-tb[freqbin] + (rank*fpp+i)*time_bins_per_file :tb.max()-tb[freqbin] + (rank*fpp+i+1)*time_bins_per_file ] += spectarray[i,:,freqbin]

        tstotal=ts*0#initiate a 4 hour blank time series
        comm.Allreduce(ts,tstotal,op=MPI.SUM)#merge the 4 hour timeseries from all processor
        tstotal = tstotal[tb.max():len(tstotal)-tb.max()]#cut the dispersed time lag


        '''
        # save the time series around the Pulsar's DM
        if rank == 0:
            if np.abs(DM - 10.922) <= dDM:
                print 'DM=',DM
                np.save('ts_pol%.1i_DMx100_%.6i' % (pol,DM*100),tstotal)
        sys.exit()
        '''

        #search for signal with decimated timeseries
        if rank<npws:#timeseries is ready for signal search
            ranki=rank
            filename = "ppc_SNR_pol_%.1i_td_%.2i_no_%.05i.txt" % (pol,ranki,txtsize[ranki,0])
            outfile = open(filename,'a')
            ndown = 2**ranki #decimate the time series
            sn,mean,rms = Threshold(Decimate_ts(tstotal,ndown),thresh,niter=0)
            ones = np.where(sn!=-1)[0]
            for one in ones:# Now record all pulses above threshold
                pulse = OutputSource()
                txtsize[ranki,1] += 1
                if txtsize[ranki,1] % 100 == 0:
                    log("Reached %d pulses" % txtsize[ranki,1])
                pulse.pulse = txtsize[ranki,1]
                pulse.SNR = sn[one]
                pulse.DM = DM
                pulse.time = one*tInt*ndown
                pulse.dtau = tInt*ndown
                pulse.dnu = freq[1]-freq[0]
                pulse.nu = cent_freq
                pulse.mean = mean
                pulse.rms = rms
                outfile.write(pulse.formatter.format(pulse)[:-1])
                if txtsize[ranki,1] >200000*txtsize[ranki,0]:
                    outfile.close()
                    txtsize[ranki,0]+=1
                    filename = "ppc_SNR_pol_%.1i_td_%.2i_no_%.05d.txt" % (pol,ranki,txtsize[ranki,0])
                    log("Previous pulse file reached limit, recording to %s" % filename)
                    outfile = open(filename,'a')
