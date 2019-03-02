# this module will be imported in the into your flowgraph
import time
import math
import numpy
import random
import ephem
import xmlrpclib
import copy
import os
import signal



#
# FFT size
#
FFTSIZE=2048

#
# Number of virtual channels
#
NCHAN=3

#
# Size of the median filter
#
MSIZE=9


def get_fftsize():
    return FFTSIZE

#
# IMPORTANT: We use this to keep track of which virtual (switched) channel we're looking at
#
frq_ndx = 0

def do_freq_schedule(p,freqs):
    global schedule_counter
    global frq_ndx

    nv = frq_ndx + 1
    nv = nv % NCHAN
    old_frq_ndx = frq_ndx
    frq_ndx = nv
    
    return(freqs[frq_ndx])

#
# Store the averaged/integrated FFT results (post-excision) here
#
avg_fft = [[0.0]*FFTSIZE]*NCHAN

#
# Store the RAW FFT
#
raw_fft = [[-120.0]*FFTSIZE]*NCHAN

#
# An event counter for impulse events that are getting
#  (partially, at least) suppressed
#
impulse_events=[0]*NCHAN


#
# Statistical counters, etc
#
exceeded_ocount = [[0]*FFTSIZE]*NCHAN
exceeded_mtime = [[0.0]*FFTSIZE]*NCHAN
exceeded_delta = [[0.0]*FFTSIZE]*NCHAN

#
# For fuzzing during impulse-blanking
#
fuzz_buffer_01 = [0.1]*FFTSIZE
fuzz_buffer_04 = [0.4]*FFTSIZE

#
# Cached alpha value for excised FFT
#
ealpha = -200.0
#
# Evaluate signal (as fft), and perform spectral excision
#  and impulse removal, calculate current total-power estimate
#  after all that has been done.
#
# Keep stats on these events
#
#
# We select/process buffers based on frq_ndx, so in effect, we're
#  maintaining NCHAN parallel "state machines" that keep track of
#  "stuff" happening with the incoming signals.
#
# It handles both types of excision, and keeps track of stats
#
#
# **************************************************************
# This function is the "meat" of what I think is an important and
#   fresh approach to processing riometer RF data, for environemnts
#   that will increasingly be very "crappy".
#
# The approach has been field tested and found to work very well, to
#  the extent that in a situation where a conventional analog riometer
#  produces "garbage" out, this approach produces quite-usable data.
#
# *****************************************************************
#
last_eval_ndx = 0
eval_hold_off = -1

def signal_evaluator(infft,prefix,prate,swrate):
    global avg_fft
    global exceeded_ocount
    global exceeded_mtime
    global exceeded_delta
    global ealpha
    global fuzz_buffer_01
    global fuzz_buffer_04
    global frq_ndx
    global last_eval_ndx
    global eval_hold_off
    global raw_fft
    global gated


    lndx = frq_ndx

    #
    # Handle buffer init/resize
    #
    if (len(avg_fft[lndx]) != len(infft)):
        avg_fft = [list(infft)]*NCHAN
    #
    # Skip some of the initial values coming in, since there's
    #  unavoidable flow-graph latency that will render the first few
    #  FFT frames we get somewhat ambiguous with respect to exactly
    #  which frequency setting (frq_ndx) they belong to.  In *addition* to
    #  that issue, the tuner produces glitches across the frequency transition.
    #
    # We do this whenever the frequency index changes
    #
    #
    # The ignore time, in seconds
    ignoretime = 0.285

    #
    # Map this into counts, since we get called at prate Hz (more or less)
    #
    ignorecount = float(prate)*ignoretime
    ignorecount = int(round(ignorecount))

    #
    # Detect frequency change
    #
    if (lndx != last_eval_ndx):
        last_eval_ndx = lndx
        eval_hold_off = ignorecount

    if (eval_hold_off > 0):
        eval_hold_off = eval_hold_off - 1
        return None

    #
    # We capture the "raw" FFT input here.
    # It is used for display AND antenna-fault detection
    #
    raw_fft[lndx] = list(infft)
    
    #
    # We don't do anything further if input has been GATED by
    # The strong-impulse detector function
    #
    if (gated):
        return

    #
    # (re)init fuzz buffers either
    #  because they are too small, or we want to randomly
    #  refresh them.
    # The random() functions aren't particularly speedy, so we
    #   only want to call them 'sometimes'.  So, we arrange for there
    #   to be a probability of 0.333 that we'll refresh the fuzz buffers
    #
    if ((len(fuzz_buffer_01) < len(infft)) or (random.randint(0,3) == 0)):
        #
        # Such pythonic, wow
        #
        fuzz_buffer_01 = [(x*0)+random.uniform(-0.1,0.1) for x in range(len(infft))]
        fuzz_buffer_04 = [(x*0)+random.uniform(-0.4,0.4) for x in range(len(infft))]
    #
    # We build a dict of quantized-to-one-dB
    #   FFT bin values
    #
    # We count the occurence, and use the largest two
    #   as the "mode" of the dataset
    #
    dbdict = {}
    for v in infft:
        key = int(v)
        if key in dbdict:
            dbdict[key] += 1
        else:
            dbdict[key] = 1

    #
    # Not quite ready
    #
    if (infft[0] == -220.0 and infft[1] == -220.0):
        return None

    #
    # Sort the list by values
    # Returning the "keys" which are the quantized dB values
    #
    slist = sorted(dbdict, key=dbdict.__getitem__)

    #
    # We estimate the "mode" by taking the top two values
    #  (top two most-frequent quantized dB values), we
    #  slightly bias the most-frequent value
    #
    mode = slist[len(slist)-1]*1.3
    mode += slist[len(slist)-2]*0.7
    mode /= 2.0
    mode = float(mode)
    
    #
    # Find reasonable minimum
    #
    # Lop-off the edge roll-off
    #
    parta=infft[int(FFTSIZE*0.17):int(FFTSIZE/2.05)]
    partb=infft[int(FFTSIZE/0.95):int(FFTSIZE*0.83)]
    minny = sorted(parta+partb)
    minny = sum(minny[0:10])
    minny /= 10.0
    
    #
    # Fold minny and mode together, with a bias towards
    #  "mode"
    #
    #
    # The basic strategy is to try to come up with some estimate for the
    #   notional "noise floor", since that's what we're measuring--the slowly
    #   varying "noise floor".  Anything that exceeds this significantly is likely
    #   not "noise floor" but something else, and we can use this estimate to
    #   excise those artifacts prior to further processing.
    #
    mode = (mode*1.4) + (minny*0.6)
    mode /= 2.0
    
    #
    # Setup for mode-based excision
    #
    outfft = [0.0]*len(infft)
    indx = 0

    #
    # Re-size stats buffers if necessary
    #
    if (len(exceeded_ocount[0]) != len(infft)):
        exceeded_ocount = [[0]*len(infft)]*NCHAN
        exceeded_mtime = [[0.0]*len(infft)]*NCHAN
        exceeded_delta = [[0.0]*len(infft)]*NCHAN

    #
    # Anything that exceeds the mode estimate by 2.2dB or more,
    #  we "smooth".
    #
    now = time.time()
    indx = 0
    for v in infft:
        if (v-mode >= 2.2):
            outfft[indx] = mode+fuzz_buffer_04[indx]

            #
            # Manage statistics for recording
            #
            if ((now - exceeded_mtime[lndx][indx]) >= 5.0):
                exceeded_delta[lndx][indx] = now - exceeded_mtime[lndx][indx]
                exceeded_mtime[lndx][indx] = now
                exceeded_ocount[lndx][indx] += 1
        else:
            outfft[indx] = v
        indx += 1


    #
    # Set up "reasonable" IIR filter parameters
    #
    alpha = 1.0-(math.pow(math.e,-2*(1.0/prate)))
    alpha *= MSIZE
    beta = 1.0-alpha
    
    #
    # The excised FFT is in "outfft"
    # We use it to contribue to the avg_fft for this channel
    #
    # Do single-pole IIR filter
    #
    
    #
    # We median-filter the excised FFF, and then use it to update avg_fft
    #
    filtered = median_filter(outfft,lndx,MSIZE)
    
    #
    # The median filter only returns an output once every MSIZE cycles
    #
    if (filtered != None):
        new_fft = numpy.multiply(outfft,[alpha]*len(outfft))
        avg_fft[lndx] = numpy.add(new_fft, numpy.multiply(avg_fft[lndx],[beta]*len(outfft)))

        #
        # Compute the total power across avg_fft
        #
        pvect = numpy.multiply(avg_fft[lndx], [0.1]*len(avg_fft[lndx]))
        pvect = numpy.power([10.0]*len(avg_fft[lndx]),pvect)
        pwr = numpy.sum(pvect)

        #
        # Other subsystems need to know current value of pwr
        #
        set_current_pwr(pwr,lndx)

    return None

#
# UI instances for getting raw FFT
#
# We fuzz the buffer in the case where the desired display channel is the current
#  actual RF channel.  This is purely for UI pleasantness reasons.
#
def get_raw_fft_ui(p, which):
    global fuzz_buffer_04

    return numpy.add(get_raw_fft(which),fuzz_buffer_04)

def get_raw_fft(which):
    global raw_fft

    return raw_fft[which]

#
# Get the (excised) avg fft buffer for this channel
#
# Again, do some fuzzing for the UI if the desired channel
#   and current RF channel aren't the same.
#
def get_avg_fft_ui(p,which):
    global fuzz_buffer_01

    retval = numpy.add(get_avg_fft(which),[15.0]*FFTSIZE)

    return numpy.add(retval,fuzz_buffer_01)

def get_avg_fft(which):
    global avg_fft

    return (avg_fft[which])

def get_exceeded_ocount(which):
    global exceeded_ocount

    return (exceeded_ocount[which])

def get_exceeded_delta(which):
    global exceeded_delta

    return(exceeded_delta[which])

#
# We use pyephem to tell us what our LMST is--for logging purposes
#
def cur_sidereal(longitude):
    longstr = "%02d" % int(longitude)
    longstr = longstr + ":"
    longitude = abs(longitude)
    frac = longitude - int(longitude)
    frac *= 60
    mins = int(frac)
    longstr += "%02d" % mins
    longstr += ":00"
    x = ephem.Observer()
    x.date = ephem.now()
    x.long = longstr
    jdate = ephem.julian_date(x)
    tokens=str(x.sidereal_time()).split(":")
    hours=int(tokens[0])
    minutes=int(tokens[1])
    seconds=int(float(tokens[2]))
    sidt = "%02d,%02d,%02d" % (hours, minutes, seconds)
    return (sidt)
#
# For keeping track of logging at regular intervals
#
last_time = time.time()
seconds = 0
fast_data_pHz = [[0.0,0.0]*300]*NCHAN
fast_data_ndx = [0]*NCHAN

def do_pHz_data(pacer,prate):
    global frq_ndx
    global fast_data_pHz
    global fast_data_ndx

    lndx = frq_ndx

    fast_data_pHz[lndx][fast_data_ndx[lndx]] = [get_current_pwr(lndx),get_refval(lndx)]
    fast_data_ndx[lndx] += 1
    
    if (fast_data_ndx[lndx] >= prate):
        fast_data_ndx[lndx] = 0

def get_exit_required(p):
    
    if (os.path.exists("stop_riometer")):
        os.kill(os.getpid(),signal.SIGTERM)
        return True
    elif (os.path.exists("restart_riometer")):
        try:
            os.remove("restart_riometer")
            os.kill(os.getpid(),signal.SIGTERM)
        except:
            pass
    return False

#
# Log data items
#
def logging(p,prefix,freq,bw,prate,longitude,frqlist):

    global seconds
    global last_time
    global fast_data_pHz
    global fast_data_ndx


    #
    # Update the seconds counter
    #
    if ((time.time() - last_time) >= 1.0):
        last_time = time.time()
        seconds += 1

        #
        # Pickup system time
        #
        ltp = time.gmtime(time.time())


        #
        # Power estimates
        #
        #
        # Every second
        #
        if (seconds != 0 and (seconds % 1 == 0)):
            for n in range(NCHAN):
                hdr_format = "%02d,%02d,%02d,%s,%d,%d"
                hdr_contents = (ltp.tm_hour, ltp.tm_min, ltp.tm_sec, cur_sidereal(longitude), frqlist[n], bw)
                hdr = hdr_format % hdr_contents
                #
                # Do recording of powers/temps
                #
                mtemp = tsky_model(frqlist[n])
                handle_pwr_recording(get_current_pwr(n), get_refval(0), get_Tsky(n), hdr, ltp, prefix, fast_data_pHz[n][0:fast_data_ndx[n]],n,prate,mtemp)
        #
        # Spectral
        #
        # Every 60 seconds
        #
        if (seconds != 0 and (seconds % 60 == 0)):
            for n in range(NCHAN):
                handle_spec_recording(get_raw_fft(n),"spec-raw-%d-" % n, ltp, hdr, prefix)
                handle_spec_recording(get_avg_fft(n), "spec-excised-%d-" % n, ltp, hdr, prefix)
                handle_spec_recording(get_peakhold(n), "spec-peak-%d-" % n, ltp, hdr, prefix)
                handle_spec_recording(get_exceeded_ocount(n), "spec-ecounts-%d-" % n, ltp, hdr, prefix)
                handle_spec_recording(get_exceeded_delta(n), "spec-edeltas-%d-" % n, ltp, hdr, prefix)

#
# A buffer for the averaged reference FFT
#
# We don't actually use the NCHAN channels in this, because
#   the reference is tuned to half-way between the two channels, and
#   stays there.
#
avg_rfft = [[-220.0]*FFTSIZE]*NCHAN

def get_ref_fft(which):
    global avg_rfft

    return avg_rfft[0]

def get_ref_fft_ui(p,which):
    return get_ref_fft(which)
#
# Process the reference-side FFT
#
# We decided to have the reference NOT track the frequency alternations
#   It is tuned in between the two frequencies
#   SOOO, while there are two reference buffers, we only ever use one of them
#
#

def do_reference(ref_fft,prate):
    global avg_rfft

    lndx = 0

    alpha = 1.0-math.pow(math.e,-2*(0.5/prate))
    beta = 1.0-alpha

    #
    # Handle re-size
    #
    if (len(avg_rfft[lndx]) != len(ref_fft)):
        avg_rfft = [ref_fft]*NCHAN
    #
    # Single-pole IIR filter it
    #
    tfft = numpy.multiply([alpha]*len(ref_fft),ref_fft)
    avg_rfft[lndx] = numpy.add(tfft,numpy.multiply([beta]*len(avg_rfft[lndx]),avg_rfft[lndx]))

    #
    # Compute reference level from FFT data
    #
    pvect = numpy.multiply(ref_fft, [0.1]*len(ref_fft))
    pvect = numpy.power([10.0]*len(ref_fft),pvect)
    b = beta*get_refval(lndx)
    set_refval((alpha*numpy.sum(pvect))+b,lndx)

    return None

#
# A buffer for the converted-to-kelvin stripchart display(s)
#
stripchart = [[0.0]*FFTSIZE]*NCHAN

#
# This gets called from the flow-graph to update the stripchart
#   buffer at 1sec intervals -- one for each of NCHAN channels.
#
#
def chart_Tsky(pace,siz,which):
    global stripchart

    if (len(stripchart[0]) != siz):
        stripchart = [[0.0]*siz]*NCHAN

    #
    # Shift chart, place new value
    #
    s = stripchart[which]
    stripchart[which] = [get_Tsky(which)]+s[0:len(s)-1]
    return (stripchart[which])

#
# We regularly get called by the flow-graph code to
#  compute a Tsky estimate--one estimate for each of NCHAN
#  channels.
#
# The estimator is based on the user providing a number of guessed-in-advance
#  parameters, including:
#
#     Tsys for both REF and SKY channels
#     Tref  -- the (hopefully calibrated) effective noise temperature of the reference source
#
# Once we have those, then it's just a matter of calculating the ratio between the measured values for
#  both SKY (NCHAN of them), and REF, then applying that to a bit of algebra to determine the apparent
#  value for Tsky for all NCHANs.
#
# This is actually remarkably similar to the process that is used to calculate noise figures when taking
#  power measurements from amplifiers.
#
tscount=0
def estimate_Tsky(pace,reftemp,tsys,tsys_ref):
    global tscount


    tscount += 1
    #
    # We know that our pacer is running pretty fast, and we don't need
    #   to do this all THAT often
    #
    if ((tscount % 5) != 0):
        return None

    #
    # Need to find Sky temp that satisfies:
    #
    # ratio = (Tsky+Tsys)/(Tsys_ref+reftemp)
    #
    #
    # X = (Tsys_ref+reftemp)
    # ratio*X = (Tsky+Tsys)
    # (ratio*X)-Tsys = Tsky
    #
    # We know the ratio from actual measurements,
    #  and all the other parameters except Tsky
    #  are available as fixed parameters (guesstimates from
    #  user input).
    #
    #
    X = tsys_ref+reftemp

    #
    # OK, having gotten the prelminary bits of algebra
    #   out of the way, do both channel estimates
    #
    for n in range(NCHAN):
        rv = get_refval(n)
        if (rv == 0):
            rv = 1.0e-15

        pwr = get_current_pwr(n)

        #
        # Last bit of algebra, which yields the Tsky
        #   estimate.
        #
        tsky = ((pwr/rv)*X)-tsys

        #
        # Blend current Tsky for this channel with new value
        #  COULD single-pole IIR here, but this seems to be
        #  "good enough".
        #
        cts = get_Tsky(n)
        set_Tsky((tsky+cts)/2.0,n)

    return (None)


#
# This needs to be revisited, since it doesn't know about multi-channels, etc.
#
def annotate(prefix, reftemp, tsys, freq, bw, gain, itsys_ref, lnagain, notes):
    if (False):
        fn = prefix+"annotation-"
        ltp = time.gmtime(time.time())

        fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
        fn += ".csv"

        fp = open(fn, "a")
        fp.write("%02d,%02d,%02d," % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
        fp.write("%.1f,%.1f," % (reftemp, tsys))
        fp.write("%.1f,%.1f,%.1f," % (freq, bw, gain))
        fp.write("%.1f,%.1f" % (itsys_ref, lnagain))
        fp.write("%s\n" % notes)
        fp.close()


#
# Because this little model is so damned useful
#
# We use it for automatically annotating the Tsky
#   stripchart display with a constant "thick" line at
#   this level.  Helps to elucidate whether the instrument
#   is, a least crudely, "working".
#
#
def tsky_model(freq):
    a = ((freq/1.0e6)/39.0)
    a = math.pow(a,-2.55)
    a *= 9120.0
    return a

#
# A different strip-chart that shows 24 hours worth of data, updated once per
#   minute
#
DAILY_MINUTES=1440
minutes_chart = [[0.0]*DAILY_MINUTES]*NCHAN
mtsky = [0]*NCHAN
mcounter = [0]*NCHAN
def minute_data(p,which):
    global minutes_chart
    global mtsky
    global mcounter

    mtsky[which] += get_Tsky(which)
    mcounter[which] += 1
    if ((mcounter[which] % 60) == 0):
        mtsky[which] /= 60.0
        m = minutes_chart[which]
        minutes_chart[which] = [mtsky[which]]+m[0:DAILY_MINUTES-1]
        mtsky[which] = 0.0

    return minutes_chart[which]

sky_metrics = [1.0]*NCHAN
qmcounter = 0
SECONDS = 60
#
# We use this to keep a (internal for now) metric about the average value
#  of the estimated apparent sky temperature, compared to what it "should" be.
#
#
def update_sky_metrics(p,freqs):
    global minutes_chart
    global sky_metrics

    #
    # Get current TOD, in minutes
    #
    t = int(time.time() / 60.0)

    #
    # If we're on a daily boundary
    #
    if ((t % DAILY_MINUTES) in [0,1]):
        for n in range(NCHAN):
            sky_metrics[n] = tsky_model(freqs[n])/ (numpy.sum(minutes_chart[n])/DAILY_MINUTES)

#
# Perhaps expose this in the UI?
#
def get_sky_metrics(p):
    global sky_metrics
    
    return sky_metrics


def impulses(p,which):
    return impulse_events[which]

#
# Keep track of peak-hold data, both for logging, and display
#
peakhold = [[0.0]*FFTSIZE]*NCHAN

#
# For auto-peak-reset on startup (in ticks)
#
auto_rst=100
auto_init=False

def handle_peak_hold(infft,ticks):
    global auto_rst
    global peakhold
    global auto_init
    global frq_ndx

    lndx = frq_ndx

    #
    # Setup inital auto_rst value
    #
    if (auto_init == False):
        auto_rst = 10*ticks
        auto_init = True

    #
    # Resize if necessary
    #
    nfft = list(infft)
    if (len(peakhold[lndx]) < len(infft)):
        peakhold = [nfft]*NCHAN

    #
    # Time for auto peak-hold reset (done on startup as a convenience)
    #
    auto_rst -= 1
    if (auto_rst == 0):
        for n in range(NCHAN):
            peakhold[n] = list(get_raw_fft(n))

    #
    # Do the peak-hold math
    #
    for i in range(0,len(peakhold[lndx])):
        if (infft[i] > peakhold[lndx][i]):
            peakhold[lndx][i] = infft[i]

    return None

#
# The UI peakhold getter--has support for a reset button
#
def get_peakhold_ui(p,which,rst):

    if (rst):
        set_peakhold(which,get_raw_fft(which))

    return get_peakhold(which)

def get_peakhold(which):
    global peakhold

    return(peakhold[which])

def set_peakhold(which,fft):
    global peakhold

    peakhold[which] = fft

#
# Write data files with all total-power related data
#
def handle_pwr_recording(pwr,rv,tsky,hdr,ltp,prefix,fdata,which,prate,mtemp):
    #
    # Record data in a daily file
    #
    fn = prefix+"rio-%d-" % which
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"

    fp = open(fn, "a")
    fp.write (hdr+",")
    rv = rv + 1.0e-15
    #
    # We write an extended record, with model sky temp every 10 seconds
    #
    if (int(time.time()) % 10 == 0):
        fp.write ("%.3e,%.3e,%.3e,%.3e,%.2f,%d\n" % (pwr, rv, pwr-rv, pwr/rv, tsky, mtemp/2.0))
    else:
        fp.write("%.3e,%.3e,%.3e,%.3e,%.2f\n" % (pwr, rv, pwr-rv, pwr/rv, tsky))
    fp.close()

    #
    # Record fast data as well
    #
    fn = prefix+"fast-%dHz-%d-" % (prate,which)
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"
    fp = open(fn, "a")
    fp.write(hdr+",")
    for v in fdata:
        fp.write ("(%.7f,%.7f)," % (v[0], v[1]) )
    fp.write("\n")
    fp.close()


#
# Write spectral data
#
def handle_spec_recording(fft,variant,ltp,hdr,prefix):
    #
    # Spectral-type data
    #
    fn = prefix+variant+"-"
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"

    fp = open(fn, "a")
    fp.write(hdr+",")
    for v in fft:
        fp.write("%.2f," % v)
    fp.write("\n")
    fp.close()


#
# The current ratio between SKY and REF
#
current_ratio = 1.0

#
# Current pwr value
#
current_pwr = [0.0]*NCHAN

#
# A place to hold an L=PMEDIAN median filter for each channel
#
PMEDIAN=17
current_pwr_filter = [[-1.0]*PMEDIAN]*NCHAN
def set_current_pwr(p,which):
    global current_pwr

    if (current_pwr_filter[which][0] < 0):
        current_pwr_filter[which] = [p]*PMEDIAN
    
    #
    # Do the shift
    #
    current_pwr_filter[which] = [p]+current_pwr_filter[which][0:PMEDIAN-1]
    
    #
    # Median filter
    #
    current_pwr[which] = numpy.median(current_pwr_filter[which])

def get_current_pwr(which):
    global current_pwr

    return (current_pwr[which])

#
# Current reference value
#
refval = [1.0]*NCHAN

def get_refval(which):
    global refval

    return(refval[0])

def set_refval(r,which):
    global refval

    refval[0] = r

#
# The current Tsky estimate
#
Tsky = [100.0]*NCHAN

def set_Tsky(t,which):
    global Tsky

    Tsky[which] = t

def get_Tsky(which):
    global Tsky

    return(Tsky[which])

#
# A dictionary for fault-related things like states for the state machine,
#   blood for the blood god, etc.
#
fault_dict = {"IDLE" : 0, "MEASURING": 1, "FAULTED" : 2, "INTERVAL" : 30,
    "NOISE" : 0, "ANTENNA_FLED" : 1}
fault_state = fault_dict["IDLE"]

def get_measurement_state(p):
    if fault_state == fault_dict["IDLE"]:
        return "IDLE"
    if fault_state == fault_dict["MEASURING"]:
        return "MEASURING"
    if fault_state == fault_dict["FAULTED"]:
        return "FAULTED"
    else:
        return "UNKNOWN"

smoothed_raw_power = 0.0
last_raw_power = -1.0
measure_counter = fault_dict["INTERVAL"]

def do_fault_schedule(tp,relayport,finterval):
    global smoothed_raw_power
    global last_raw_power
    global fault_state
    global fault_dict
    global measure_counter

    t = int(time.time())

    #
    # Power estimate comes from raw total-power estimator in flow-graph
    #
    smoothed_raw_power = tp
    if (last_raw_power < 0.0):
        last_raw_power = smoothed_raw_power

    #
    # Transiton from IDLE to MEASURING every user-defined minutes
    #
    if (fault_state == fault_dict["IDLE"] and (t % finterval) in [0,1,2,3]):
        fault_state = fault_dict["MEASURING"]
        last_raw_power = smoothed_raw_power
        measure_counter = fault_dict["INTERVAL"]

        #
        # Turn diagnostic noise source ON
        #
        try:
            relay_event(fault_dict["NOISE"],1,relayport)
        except:
            pass

    if (fault_state == fault_dict["MEASURING"]):
        measure_counter -= 1
        if (measure_counter <= 0):
            fault_state = fault_dict["IDLE"]

            #
            # Try turning diagnostic noise source OFF
            #
            try:
                relay_event(fault_dict["NOISE"], 0, relayport)
            except:
                pass

            #
            # Look for sudden increase in received power level
            #  If the antenna/feedline are working correctly, there'll
            #  be very little power reflected at the directional coupler
            #  back towards the receiver port.
            #
            if (smoothed_raw_power/last_raw_power > 3.0):
                #
                # Try turning on the antenna fault LED
                #
                try:
                    relay_event(fault_dict["ANTENNA_FLED"], 1, relayport)
                    antenna_fault(True)
                except:
                    pass
            else:
                #
                # Extinguish the antenna fault LED
                #
                try:
                    relay_event(fault_dict["ANTENNA_FLED"], 0, relayport)
                    antenna_fault(False)
                except:
                    pass
    #
    # Look for sudden drop in power--also a possible antenna fault
    #
    handle_normal_power(smoothed_raw_power,0,0)
    if (get_fault(0) == True):
        relay_event(fault_dict["ANTENNA_FLED"], 1, relayport)
    else:
        relay_event(fault_dict["ANTENNA_FLED"], 0, relayport)

    return None


#
# For auto normal_power setting (in seconds)
#
auto_normal = 45

#
# Expected power level -- for fault detection
#
normal_power=[1.0]*NCHAN

#
# We use this on a regular basis to try to intuit issues with the antenna.
#
# It is necessarily heuristic, and simply looks for a sudden *drop* in
#  antenna input.
#
# Not perhaps that reliable.
#
def handle_normal_power(pwr,renormal,which):
    global auto_normal
    global normal_power

    #
    # Set "normal power level" after "auto_normal" has timed out
    #
    auto_normal -= 1
    if (auto_normal == 0):
        normal_power[which] = pwr
        antenna_fault(False)

    #
    # Reset possible fault
    #
    if (renormal):
        normal_power[which] = pwr
        antenna_fault(False)

    #
    # Try to detect a significant drop in antenna power, and
    #   declare a probable antenna fault
    #
    # We try to make the fault state "sticky"
    #
    if (pwr <= (normal_power[which]/4.0)):
        antenna_fault(True)
    elif (get_fault(0) == False):
        antenna_fault(False)
#
# Keep track of antenna-fault state
#
astate = False
def antenna_fault(state):
    global astate
    afault = state

def get_fault(p):
    global astate
    return astate

#
# Pass XMLRPC into the (external) relay control server
#
def relay_event(bit,value,rport):
    try:
        xmls = xmlrpclib.Server("http://localhost:%d/" % rport)
        xmls.set_bit(bit,value)

    except:
        pass

medians = None
fndxs = [0]*NCHAN
cur_flen = 0

def median_filter(fft,which,flen):
    global medians
    global fndxs
    global cur_flen
    
    #
    # Build medians array when necessary
    #
    if (cur_flen != flen):
        medians = [[[0.0]*FFTSIZE]*flen]*NCHAN
        curf_flen = flen
    
    #
    # Determine which filter we're heading into
    #
    filt = medians[which]
    
    #
    # Stuff it into the right location
    #
    filt[fndxs[which]] = fft
    
    fndxs[which] += 1
    
    #
    # Time to sort the list, and return the median
    #  value
    #
    if (fndxs[which] >= flen):
        fndxs[which] = 0
        npa = numpy.array(filt)
        out = numpy.median(npa,axis=0)
        out = list(out)
        return out
        
    else:
        return None

#
# A gating function on total-power value
#
# Maintain last TP value--VERY raw, from early
#  in the flow-graph
#
last_tp_value = 0.0 
last_tp_is_valid = 200 
gate_counter = 0
gated=False
gate_timer=0
def do_gating(tp):
    global last_tp_value
    global last_tp_is_valid
    global gate_counter
    global gated
    global gate_timer

    alpha = 0.01
    beta = 1.0 - alpha
    
    #
    # This prevents us from being gated forever
    #
    if (gate_timer < 50):
        if (last_tp_is_valid <= 0):
            if (tp > (12.0 * last_tp_value)):
                
                #
                # We only count an initial gating
                #
                if (gated == False):
                    gate_counter += 1
                gated=True
                gate_timer +=1
                return (True,gate_counter)
    else:
        last_tp_value = tp
    #
    # Just update last_tp_value, applying smoothing as we go
    #
    last_tp_value = (tp*alpha) + (beta*last_tp_value)
    gate_timer = 0
    gated = False
    
    #
    # We need a bit of time to determine what the average value actually
    #  is
    #
    last_tp_is_valid -= 1
    
    return (False,gate_counter)

#
# Used to clean directory of "old" data
#
import fnmatch
def clean(direct,retention):
    #
    # Filename prefixes we care about
    #
    fileprefixes = ["rio", "spec", "fast", "annotation"]
    
    #
    # Suffix
    #
    suffix = ".csv"
    
    #
    #
    # Handle old data
    #
    DAY=86400
    
    #
    # Establish parameters for removal
    #
    end = int(time.time()) - (DAY*retention)
    
    #
    # Walk tree, based on pattern
    #
    for dirName, subdirList, fileList in os.walk(direct):
        for fname in fileList:
            for fp in fileprefixes:
                filepat = "%s*%s" % (fp, suffix)
                if fnmatch.fnmatch(fname, filepat):
                    actualfn = os.path.join(dirName,fname)
                    try:
                        stat = os.stat(actualfn)
                        if (stat.st_mtime < end):
                            os.remove(actualfn)
                    except:
                        pass
