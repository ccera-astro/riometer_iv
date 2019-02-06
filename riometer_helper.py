# this module will be imported in the into your flowgraph
import time
import math
import numpy
import random

#
# Store the averaged/integrated FFT results (post-excision) here
#
avg_fft = [0.0]*128

#
# Keep track of the most recent post-excision, pre-integration
#  output, for impulse blanking.
#
last_out = [0.0]*128

#
# For keeping track of logging at regular intervals
#
last_time = time.time()
seconds = 0

#
# An event counter for impulse events that are getting
#  (partially, at least) suppressed
#
impulse_events = 0

#
# A counter/timer for hold-off for impulse detection
#
impulse_count = 0


#
# Statistical counters, etc
#
exceeded_ocount = [0]*128
exceeded_mtime = [0.0]*128
exceeded_delta = [0.0]*128

#
# Cached alpha value for excised FFT
#
ealpha = -200.0
#
# This function used to be just about doing excisions, but it has morphed into
#  much, much more
#
# It handles logging
# It handles both types of excision, and keeps track of stats
# It handles peak-hold computations
# It handles declaring an antenna fault
#
# It handles some rapid-reponse UI stuff as well
#
# It does call-outs to other functions for a lot of the above
#
def signal_evaluator(infft,prefix,rst,thresh,duration,freq,bw,renormal,prate):
    global avg_fft
    global last_time
    global seconds
    global last_out
    global impulse_count
    global impulse_events
    global exceeded_ocount
    global exceeded_mtime
    global exceeded_delta
    global ealpha

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
        return ([-80.0]*len(infft))
    
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
    mode = slist[len(slist)-1]*1.2
    mode += slist[len(slist)-2]*0.8
    mode /= 2.0
    
    mode = float(mode)
    
    
    #
    # Setup for mode-based excision
    #
    outfft = [0.0]*len(infft)
    indx = 0
    
    if (len(exceeded_ocount) != len(infft)):
        exceeded_ocount = [0]*len(infft)
        exceeded_mtime = [0.0]*len(infft)
        exceeded_delta = [0.0]*len(infft)
    
    #
    # Anything that exceeds the mode estimate by 1.5dB or more,
    #  we "smooth".
    #
    now = time.time()
    for v in infft:
        if (v-mode >= 1.5):
            outfft[indx] = mode+random.uniform(-0.4,0.4)
            
            #
            # "Fresh" excision event
            #
            if (now - exceeded_mtime[indx] >= 5.0):
                exceeded_delta[indx] = now - exceeded_mtime[indx]
                exceeded_mtime[indx] = now
                exceeded_ocount[indx] += 1
        else:
            outfft[indx] = v
        indx += 1
    
    #
    # Handle buffer init/resize
    #
    if (len(avg_fft) != len(outfft)):
        avg_fft = list(outfft)
        last_out = list(outfft)
    
    #
    # Try to detect impulse noise, and reject it
    # If we aren't in the middle of a impulse event holdoff period,
    #    check for impulse (value exceeds threshold)
    #
    exceeded_bins = 0
    if (impulse_count <= 0):
        for i in range(0,len(outfft)):
            if (abs(outfft[i]-last_out[i]) > thresh):
                exceeded_bins += 1
        #
        # If more than 25% of the bins are in "exceeded" state
        #   declare an impulse-noise event
        #
        if (exceeded_bins >= len(outfft)/4.0):
            impulse_count = duration
            impulse_events += 1
    #
    # Update averages, etc only if we aren't in an impulse-noise
    #    blanking interval
    #
    if (ealpha < 0.0):
		ealpha = 1.0-(math.pow(math.e,-2*(1.0/prate)))
    beta = 1.0-ealpha
    if (impulse_count <= 0):
        
        #
        # Do single-pole IIR filter
        #
        new_fft = numpy.multiply(outfft,[ealpha]*len(outfft))
        avg_fft = numpy.add(new_fft, numpy.multiply(avg_fft,[beta]*len(outfft)))
    
    #
    # Always record the last excised FFT buffer
    #
    last_out = list(outfft)
    
    if (impulse_count > 0):
        impulse_count -= 1

    #
    # For display purposes only, bump the excised value up by 15dB
    #  so that it constrasts with the input enough to be clear
    #
    outfft = numpy.add(avg_fft, [15.0]*len(avg_fft))
    
    #
    # Compute the total power across avg_fft
    #
    pvect = numpy.multiply(avg_fft, [0.1]*len(avg_fft))
    pvect = numpy.power([10.0]*len(avg_fft),pvect)
    pwr = numpy.sum(pvect)
    
    #
    # Other things need to know
    #
    set_current_pwr(pwr)
    
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
        
        hdr_format = "%02d,%02d,%02d,%d,%d,"
        hdr_contents = (ltp.tm_hour, ltp.tm_min, ltp.tm_sec, freq, bw)
        hdr = hdr_format % hdr_contents

        #
        # Power estimates
        #
        #
        # Every second
        #
        if (seconds != 0 and (seconds % 1 == 0)):
            #
            # Time to check on possible antenna faults
            #
            handle_normal_power(pwr,renormal)
            
            #
            # Compute current_ratio
            #
            rv = get_refval()
            if (rv == 0.0):
                rv += 1.0e-15
            
            #
            # Do recording of powers/temps
            #
            handle_pwr_recording(pwr, rv, hdr, ltp, prefix)
        #
        # Spectral
        #
        # Every 60 seconds
        #
        if (seconds != 0 and (seconds % 60 == 0)):
            
            handle_spec_recording(infft,"spec-raw", ltp, hdr, prefix)
            handle_spec_recording(avg_fft, "spec-excised", ltp, hdr, prefix)
            handle_spec_recording(get_peakhold(), "spec-peak", ltp, hdr, prefix)
            handle_spec_recording(exceeded_ocount, "spec-ecounts", ltp, hdr, prefix)
            handle_spec_recording(exceeded_delta, "spec-edeltas", ltp, hdr, prefix)
            
                
    return (outfft)

#
# A buffer for the averaged reference FFT
#
avg_rfft = [-220.0]*128

#
# Cached alpha value
#
ralpha = -200.0

#
# Process the reference-side FFT
#
def do_reference(ref_fft,prate):
    global avg_rfft
    global ralpha
    
    if (ralpha < 0.0):
        ralpha = 1.0-math.pow(math.e,-2*(0.5/prate))
    beta = 1.0-ralpha
    
    #
    # Handle re-size
    #
    if (len(avg_rfft) != len(ref_fft)):
        avg_rfft = list(ref_fft)
    
    #
    # Single-pole IIR filter it
    #
    tfft = numpy.multiply([ralpha]*len(ref_fft),ref_fft)
    avg_rfft = numpy.add(tfft,numpy.multiply([beta]*len(avg_rfft),avg_rfft))
    
    #
    # Compute reference level from FFT data
    #
    pvect = numpy.multiply(ref_fft, [0.1]*len(ref_fft))
    pvect = numpy.power([10.0]*len(ref_fft),pvect)
    set_refval((ralpha*numpy.sum(pvect))+(beta*refval))
    return avg_rfft

#
# A buffer for the converted-to-kelvin stripchart display
#
stripchart = [0.0]*128

def chart_Tsky(pace,siz):
    global stripchart
    
    if (len(stripchart) != siz):
        stripchart = [0.0]*siz

    #
    # Shift chart, place new value
    #
    stripchart = [get_Tsky()]+stripchart[0:len(stripchart)-1]

    return (stripchart)

#
# Pretty much what the name suggests
#
def estimate_Tsky(pace,reftemp,tsys,tsys_ref,lnagain,estimate,commit):
    
    #
    # Need to find Sky temp that satisfies:
    #
    # ratio = (Tsky+Tsys)/(Tsys_ref+reftemp)
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
    
    rv = get_refval()
    if (rv == 0):
        rv += 1.0e-15
    
    pwr = get_current_pwr()
    
    tsky = ((pwr/rv)*X)-tsys
    
    #
    # We divide the apparent Tsky by any LNA gain that may be behind it,
    #  but not in-common with the REF side of the house.
    #
    tsky /= math.pow(10.0, lnagain/10.0)
    
    #
    # This can be used to "tweak" the sky-side estimate, using an
    #  external calibrator source.  User enters CAL noise value, in dB ENR.
    #
    # This only really works well if the other parameters have been "dialed in"
    #   fairly well.
    #
    if (estimate > 0.0 and commit != 0):
        #
        # Convert from dB ENR to temperature
        #
        estimate = math.pow(10.0,estimate/10.0)
        estimate *= 297.0
        set_skyratio(estimate/tsky)

    #
    # Adjust by tweakage ratio
    #
    tsky = tsky * get_skyratio()
    set_Tsky(tsky)  
    return (get_Tsky())

def annotate(prefix, reftemp, tsys, freq, bw, gain, itsys_ref, lnagain, notes):
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
# We use it for automatically scaling the Sky temperature display
#  (Assuming, of course, that all the other algebra is correct)
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
minutes_chart = [0.0]*DAILY_MINUTES
mtsky = -1
mcounter = 0
def minute_data(p):
    global minutes_chart
    global mtsky
    global mcounter
    
    mtsky += get_Tsky()
    mcounter += 1
    if ((mcounter % 60) == 0):
        mtsky /= 60.0
        minutes_chart = [mtsky]+minutes_chart[0:DAILY_MINUTES-1]
        mtsky = 0.0

    return minutes_chart

def impulses(p):
    return impulse_events

astate = False   
def antenna_fault(state):
    global astate
    afault = state

def get_fault(p):
    global astate
    return astate

#
# Keep track of peak-hold data, both for logging, and display
#
peakhold = [0.0]*128

#
# For auto-peak-reset on startup (in ticks)
#
auto_rst = 100
auto_init = False

def handle_peak_hold(fft,rst,ticks):
    global auto_rst
    global peakhold
    global auto_init
    
    #
    # Setup inital auto_rst value
    #
    if (auto_init == False):
		auto_rst = 10*ticks
		auto_init = True

    #
    # Resize if necessary
    #
    if (len(peakhold) < len(fft)):
        peakhold = list(fft)
        
    #
    # If they pushed the "Reset Peak Hold" button in the UI, OR
    #   The auto-rst variable has dropped through zero
    #
    auto_rst -= 1
    if (rst or (auto_rst == 0)):
        peakhold = list(fft)
    
    #
    # Do the peak-hold math
    #
    for i in range(0,len(peakhold)):
        if (fft[i] > peakhold[i]):
            peakhold[i] = fft[i]
    
    return (peakhold)

def get_peakhold():
    global peakhold
    
    return(peakhold)

#
# For auto normal_power setting (in seconds)
#
auto_normal = 30

#
# Expected power level -- for fault detection
#
normal_power = -1.0

def handle_normal_power(pwr,renormal):
    global auto_normal
    global normal_power
    
    #
    # Set "normal power level" after "auto_normal" has timed out
    #
    auto_normal -= 1
    if (auto_normal == 0):
        normal_power = pwr
        antenna_fault(False)

    #
    # Reset possible fault
    #
    if (renormal):
        normal_power = pwr
        antenna_fault(False)
    
    #
    # Try to detect a significant drop in antenna power, and
    #   declare a probably antenna fault
    #
    if (pwr < (normal_power/3.5)):
        antenna_fault(True)
    else:
        antenna_fault(False)

def handle_pwr_recording(pwr,rv,hdr,ltp,prefix):
    #
    # Record data in a daily file
    #
    fn = prefix+"rio-"
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"

    fp = open(fn, "a")
    fp.write (hdr)
    fp.write ("%.7f,%.7f,%.7f,%e,%.1f\n" % (pwr, rv, pwr-rv, pwr/rv, get_Tsky()))
    fp.close()

def handle_spec_recording(fft,variant,ltp,hdr,prefix):
    #
    # Spectral-type data
    #
    fn = prefix+variant+"-"
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"
    
    fp = open(fn, "a")
    fp.write(hdr)
    for v in fft:
        fp.write("%.3f," % v)
    fp.write("\n")
    fp.close()
    

#
# The current ratio between SKY and REF
#
current_ratio = 1.0

#
# Current pwr value
#
current_pwr = 0.0
def set_current_pwr(p):
    global current_pwr
    
    current_pwr = p

def get_current_pwr():
    global current_pwr
    
    return (current_pwr)
    
#
# Current reference value
#
refval = -1.0

def get_refval():
    global refval
    
    return(refval)

def set_refval(r):
    global refval
    
    refval = r

#
# The current Tsky estimate
#
Tsky = 100.0

def set_Tsky(t):
    global Tsky
    
    Tsky = t

def get_Tsky():
    global Tsky
    
    return(Tsky)


#
# We can use this to "force" a known calibration point using the UI
#
# For example, if the user plugs in a 10,000K noise source, and
#  Tsky is only reading 5000, this can be used to instantaneously
#  calculate an adjustment factor
#
skyratio = 1.0

def get_skyratio():
    global skyratio
    
    return (skyratio)

def set_skyratio(r):
    global skyratio
    
    skyratio = r

