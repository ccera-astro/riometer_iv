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
# Keep track of peak-hold data, both for logging, and display
#
peakhold = [0.0]*128

#
# Current reference value
#
refval = -1.0

#
# For keeping track of logging at regular intervals
#
last_time = time.time()
seconds = 0

#
# The current ratio between SKY and REF
#
current_ratio = 1.0

#
# The current Tsky estimate
#
Tsky = 100.0

#
# An event counter for impulse events that are getting
#  (partially, at least) suppressed
#
impulse_events = 0

#
# A counter/timer for hold-off for impulse detection
#
impulse_count = 0


def modified_fft(infft,prefix,refscale,rst,rst2,thresh,duration):
    global avg_fft
    global refval
    global last_time
    global seconds
    global current_ratio
    global Tsky
    global peakhold
    global last_out
    global impulse_count
    global impulse_events

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
    
    #
    # Anything that exceeds the mode estimate by 1.5dB or more,
    #  we "smooth".
    #
    for v in infft:
        if (v-mode >= 1.5):
            outfft[indx] = mode+random.uniform(-0.4,0.4)
        else:
            outfft[indx] = v
        indx += 1
    
    #
    # Handle buffer init/resize
    #
    if (len(avg_fft) != len(outfft)):
        avg_fft = list(outfft)
        peakhold = list(infft)
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
    if (impulse_count <= 0):
        alpha = 0.025
        beta = 1.0-alpha
        
        #
        # Do single-pole IIR filter
        #
        new_fft = numpy.multiply(outfft,[alpha]*len(outfft))
        avg_fft = numpy.add(new_fft, numpy.multiply(avg_fft,[beta]*len(outfft)))
    
    #
    # Always record the last excised FFT buffer
    #
    last_out = list(outfft)
    
    if (impulse_count > 0):
        impulse_count -= 1
    
    #
    # If they pushed the "Reset Peak Hold" button in the UI
    #
    if (rst):
        peakhold = list(infft)

    #
    # Do the peak-hold math
    #
    for i in range(0,len(peakhold)):
        if (infft[i] > peakhold[i]):
            peakhold[i] = infft[i]
    
    #
    # For display purposes only, bump the excised value up by 15dB
    #  so that it constrasts with the input enough to be clear
    #
    outfft = numpy.add(avg_fft, [15.0]*len(avg_fft))
    
    #
    # Update the seconds counter
    #
    if ((time.time() - last_time) >= 1.0):
        last_time = time.time()
        seconds += 1

        #
        # Power estimates
        #
        #
        # Every second
        #
        if (seconds != 0 and (seconds % 1 == 0)):
            
            #
            # Compute the total power across avg_fft
            #
            pvect = numpy.multiply(avg_fft, [0.1]*len(avg_fft))
            pvect = numpy.power([10.0]*len(avg_fft),pvect)
            pwr = numpy.sum(pvect)
            
            #
            # Pickup system time
            #
            ltp = time.gmtime(time.time())

            #
            # Record data in a daily file
            #
            fn = prefix+"rio-"
            fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
            fn += ".csv"

            fp = open(fn, "a")
            rv = refval * refscale
            rv = 1.0e-15 + rv
            fp.write ("%02d,%02d,%02d," % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
            fp.write ("%.7f,%.7f,%.7f,%e,%.1f\n" % (pwr, refval, pwr-rv, pwr/rv, Tsky))
            current_ratio = pwr/rv
            fp.close()
        
        #
        # Spectral
        #
        # Every 60 seconds
        #
        if (seconds != 0 and (seconds % 60 == 0)):
            fn = prefix+"spec-raw-"
            fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
            fn += ".csv"
            
            fp = open(fn, "a")
            fp.write("%02d,%02d,%02d," % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
            for v in infft:
                fp.write("%.2f," % v)
            fp.write("\n")
            fp.close()
            
            fn = prefix+"spec-excised-"
            fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
            fn += ".csv"
            
            fp = open(fn, "a")
            for v in avg_fft:
                fp.write("%.2f," % v)
            fp.write("\n")
            fp.close()
            
            fn = prefix+"spec-peak-"
            fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
            fn += ".csv"
            
            fp = open(fn, "a")
            fp.write("%02d,%02d,%02d," % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
            for v in peakhold:
                fp.write("%.2f," % v)
            fp.write("\n")
            fp.close()
              
    return (outfft)
    

#
# A buffer for the averaged reference FFT
#
avg_rfft = [-220.0]*128

#
# Process the reference-side FFT
#
def do_reference(ref_fft):
    global refval
    global avg_rfft
    
    alpha = 0.1
    beta = 1.0-alpha
    
    #
    # Handle re-size
    #
    if (len(avg_rfft) != len(ref_fft)):
        avg_rfft = list(ref_fft)
    
    #
    # Single-pole IIR filter it
    #
    tfft = numpy.multiply([alpha]*len(ref_fft),ref_fft)
    avg_rfft = numpy.add(tfft,numpy.multiply([beta]*len(avg_rfft),avg_rfft))
    
    #
    # Compute reference level from FFT data
    #
    pvect = numpy.multiply(ref_fft, [0.1]*len(ref_fft))
    pvect = numpy.power([10.0]*len(ref_fft),pvect)
    refval = (alpha*numpy.sum(pvect))+(beta*refval)
    return avg_rfft

#
# A buffer for the converted-to-kelvin stripchart display
#
stripchart = [0.0]*128

#
# We can use this to "force" a known calibration point using the UI
#
# For example, if the user plugs in a 10,000K noise source, and
#  Tsky is only reading 5000, this can be used to instantaneously
#  calculate an adjustment factor
#
skyratio = 1.0

def power_ratio(pace,siz,reftemp,tsys,tsys_ref,lnagain,estimate,commit):
    global stripchart
    global current_ratio
    global Tsky
    global skyratio
    
    if (len(stripchart) != siz):
        stripchart = [0.0]*siz
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
    Tsky = (current_ratio*X)-tsys
    
    #
    # We divide the apparent Tsky by any LNA gain that may be behind it,
    #  but not in-common with the REF side of the house.
    #
    Tsky /= math.pow(10.0, lnagain/10.0)
    
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
        skyratio = estimate/Tsky
    
    #
    # Adjust by tweakage ratio
    #
    Tsky = Tsky * skyratio

    #
    # Shift the "stripchart" buffer
    #
    for i in range(len(stripchart)-1,0,-1):
        stripchart[i] = stripchart[i-1]
    #
    # Plonk the current Tsky value into the 0th position
    #
    stripchart[0] = Tsky
    return (stripchart)
    
def do_peak(pfft):
    global peakhold
    
    if (len(peakhold) != len(pfft)):
        peakhold = [-140.0]*len(pfft)
    return (peakhold)

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

DAILY_MINUTES=1440
minutes_chart = [0.0]*DAILY_MINUTES
mtsky = -1
mcounter = 0
def minute_data(p):
    global minutes_chart
    global Tsky
    global mtsky
    global mcounter
    
    mtsky += Tsky
    mcounter += 1
    if ((mcounter % 60) == 0):
        mtsky /= 60.0
        minutes_chart = [mtsky]+minutes_chart[0:DAILY_MINUTES-1]
        mtsky = 0.0

    return minutes_chart

def impulses(p):
    return impulse_events
