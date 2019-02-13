# this module will be imported in the into your flowgraph
import time
import math
import numpy
import random
import ephem
import xmlrpclib

FFTSIZE=2048

def get_fftsize():
    return FFTSIZE

# 
# IMPORTANT: We use this to keep track of which virtual (switched) channel we're looking at
#
frq_ndx = 0

schedule_counter = 0
def do_freq_schedule(p,interval,freqs,xmlport):
    global schedule_counter
    global frq_ndx
    
    schedule_counter += 1
    if (schedule_counter >= interval):
        nv = frq_ndx + 1
        nv = nv % NCHAN
        old_frq_ndx = frq_ndx
        frq_ndx = nv
        schedule_counter = 0
        try:
            rpcHandle = xmlrpclib.Server("http://localhost:%d/" % xmlport)
            rpcHandle.set_ifreq(freqs[frq_ndx])
        except:
            frq_ndx = old_frq_ndx
            

#
# Number of virtual channels
#
NCHAN=2

#
# Store the averaged/integrated FFT results (post-excision) here
#
avg_fft = [[0.0]*FFTSIZE]*NCHAN

#
# Store the RAW FFT
#
raw_fft = [[-120.0]*FFTSIZE]*NCHAN


#
# Keep track of the most recent post-excision, pre-integration
#  output, for impulse blanking.
#
last_out = [[0.0]*FFTSIZE]*NCHAN


#
# An event counter for impulse events that are getting
#  (partially, at least) suppressed
#
impulse_events=[0]*NCHAN

#
# A counter/timer for hold-off for impulse detection
#
impulse_count=[0]*NCHAN

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
# The "which" input is used to return the relevant buffer, but
#  the real-time *processing* is based on the frq_ndx variable
#
#
# It handles both types of excision, and keeps track of stats
#
last_eval_ndx = 0
eval_hold_off = -1
def signal_evaluator(infft,prefix,thresh,duration,prate):
    global avg_fft
    global last_out
    global impulse_count
    global impulse_events
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
    
    lndx = frq_ndx

    #
    # Handle buffer init/resize
    #
    if (len(avg_fft[lndx]) != len(infft)):
        avg_fft[0] = list(infft)
        avg_fft[1] = list(infft)
        last_out[1] = list(infft)
        last_out[0] = list(infft)
    
    #
    # Skip some of the initial values coming in, since there's
    #  unavoidable FG latency that will render the first few
    #  FFT frames we get somewhat ambiguous with respect to exactly
    #  which frequency setting they belong to.
    #
    #
    if (lndx != last_eval_ndx):
        last_eval_ndx = lndx
        eval_hold_off = int(prate/4)+1

    if (eval_hold_off > 0):
        eval_hold_off = eval_hold_off - 1
        return None
    
    raw_fft[lndx] = list(infft)
    
    #
    # We cache an alpha value, just so we don't have to do this
    #   gnarly math at prate
    #
    if (ealpha < 0.0):
        ealpha = 1.0-(math.pow(math.e,-2*(1.0/prate)))
    beta = 1.0-ealpha
    
    #
    # (re)init fuzz buffers either
    #  because they are too small, or we want to randomly
    #  refresh them.
    # The random() functions aren't particularly speedy, so we
    #   only want to call them 'sometimes'.  So, we arrange for there
    #   to be a probability of 0.2 that we'll refresh the fuzz buffers
    #
    if ((len(fuzz_buffer_01) < len(infft)) or (random.randint(0,5) == 0)):
        fuzz_buffer_01 = [0.0]*len(infft)
        fuzz_buffer_04 = [0.0]*len(infft)
        
        for i in range(0,len(infft)):
            fuzz_buffer_01[i] = random.uniform(-0.1,0.1)
            fuzz_buffer_04[i] = random.uniform(-0.4,0.4)

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
    # Re-size stats buffers if necessary
    #
    if (len(exceeded_ocount[0]) != len(infft)):
        exceeded_ocount = [[0]*len(infft)]*NCHAN
        exceeded_mtime = [[0.0]*len(infft)]*NCHAN
        exceeded_delta = [[0.0]*len(infft)]*NCHAN
    
    #
    # Anything that exceeds the mode estimate by 1.5dB or more,
    #  we "smooth".
    #
    now = time.time()
    for v in infft:
        if (v-mode >= 1.5):
            outfft[indx] = mode+fuzz_buffer_04[indx]
            
            #
            # "Fresh" excision event
            #
            if ((now - exceeded_mtime[lndx][indx]) >= 5.0):
                exceeded_delta[lndx][indx] = now - exceeded_mtime[lndx][indx]
                exceeded_mtime[lndx][indx] = now
                exceeded_ocount[lndx][indx] += 1
        else:
            outfft[indx] = v
        indx += 1
    
    #
    # Try to detect impulse noise, and reject it
    # If we aren't in the middle of an impulse event holdoff period,
    #    check for impulse (value exceeds threshold)
    #
    exceeded_bins = 0
    if (impulse_count[lndx] <= 0):
        for i in range(0,len(outfft)):
            if (abs(outfft[i]-last_out[lndx][i]) > thresh):
                exceeded_bins += 1
        #
        # If more than 25% of the bins are in "exceeded" state
        #   declare an impulse-noise event
        #
        if (exceeded_bins >= (len(outfft)/4.0)):
            impulse_count[lndx] = duration
            impulse_events[lndx] += 1
    #
    # Update averages, etc only if we aren't in an impulse-noise
    #    blanking interval
    #
    
    if (impulse_count[lndx] <= 0):
        #
        # Do single-pole IIR filter
        #
        new_fft = numpy.multiply(outfft,[ealpha]*len(outfft))
        avg_fft[lndx] = numpy.add(new_fft, numpy.multiply(avg_fft[lndx],[beta]*len(outfft)))
    
    #
    # This will cause us to frob on avg_fft with P = 0.3
    #
    elif (random.randint(0,3) == 0):
        
        #
        # First, fold-in fuzz buffer
        new_fft = numpy.add(avg_fft[lndx], fuzz_buffer_01)
        avg_fft[lndx] = numpy.add(new_fft, avg_fft[lndx])
        avg_fft[lndx] = numpy.divide(avg_fft[lndx], [2.0]*len(avg_fft[lndx]))
        
        #
        # Then fold-in a tiny contribution from current data
        #
        diff = numpy.sub(avg_fft[lndx], last_out[lndx])
        diff = numpy.add(avg_fft[lndx], numpy.multiply(diff,[0.0025]*len(diff)))
        avg_fft[lndx] = numpy.divide(diff, [2.0]*len(avg_fft[lndx]))
    
    #
    # Always record the last excised FFT buffer
    #
    last_out[lndx] = list(outfft)
    
    #
    # Decrement impulse holdoff counter if necessary
    #
    if (impulse_count[lndx] > 0):
        impulse_count[lndx] -= 1


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

def get_raw_fft_ui(p, which):
    global fuzz_buffer_04
    
    if (which != frq_ndx):
        return numpy.add(get_raw_fft(which),fuzz_buffer_04)
    else:
        return get_raw_fft(which)

def get_raw_fft(which):
    global raw_fft
    
    return raw_fft[which]

def get_avg_fft_ui(p,which):
    global fuzz_buffer_01
    
    retval = numpy.add(get_avg_fft(which),[15.0]*FFTSIZE)
    
    if (which != frq_ndx):
        return numpy.add(retval,fuzz_buffer_01)
    else:
        return retval

def get_avg_fft(which):
    global avg_fft
    
    return (avg_fft[which])

def get_exceeded_ocount(which):
    global exceeded_ocount
    
    return (exceeded_ocount[which])

def get_exceeded_delta(which):
    global exceeded_delta
    
    return(exceeded_delta[which])

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
fast_data_pHz = [[0.0,0.0]*100]*NCHAN
fast_data_ndx = [0]*NCHAN

def logging(infft,renormal,prefix,freq,bw,prate,longitude,frqlist):

    global seconds
    global last_time
    global fast_data_pHz
    global fast_data_ndx
    global frq_ndx
    
    lndx = frq_ndx
    
    
    #
    # Resize fast_data_pHz if necessary
    #
    if (len(fast_data_pHz[0]) != prate):
        fast_data_pHz[0] = [[0.0,0.0]]*prate
        fast_data_pHz[1] = [[0.0,0.0]]*prate
    #
    # We record fast data in a buffer, and that buffer only gets logged once per second
    #
    fast_data_pHz[lndx][fast_data_ndx[lndx] % prate] = [get_current_pwr(lndx), get_refval(lndx)]
    
    #
    # It heads off to infinity
    #
    fast_data_ndx[lndx] += 1
    
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
        
        hdr_format = "%02d,%02d,%02d,%s,%d,%d,"
        hdr_contents = (ltp.tm_hour, ltp.tm_min, ltp.tm_sec, cur_sidereal(longitude), freq, bw)
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
            handle_normal_power(get_current_pwr(lndx),renormal,lndx)
            
            #
            # Implement a median filter for SKY data
            #
            
            #
            # First extract SKY values from fast_data_pHz
            #
            sortedp = [x[0] for x in fast_data_pHz[lndx]]
            
            #
            # Then the REF values
            #
            sortedr = [x[1] for x in fast_data_pHz[lndx]]
            
            #
            # Then sort
            #
            sortedp = sorted(sortedp)
            sortedr = sorted(sortedr)
            
            #
            # Pull the middle 30%
            #
            ls = int(len(sortedp)/2)
            mid = ls
            start = int(mid-(ls/6))
            end =  (mid+(ls/6))

            p = numpy.sum(sortedp[start:end])
            p /= ((start-end)+1)
            
            r = numpy.sum(sortedr[start:end])
            r /= ((start-end)+1)
            
            #
            # Do recording of powers/temps
            #
            handle_pwr_recording(p, r, hdr, ltp, prefix, fast_data_pHz[lndx],lndx)
        #
        # Spectral
        #
        # Every 60 seconds
        #
        if (seconds != 0 and (seconds % 60 == 0)):
            handle_spec_recording(infft,"spec-raw", ltp, hdr, prefix)
            handle_spec_recording(get_avg_fft(lndx), "spec-excised", ltp, hdr, prefix)
            handle_spec_recording(get_peakhold(lndx), "spec-peak", ltp, hdr, prefix)
            handle_spec_recording(get_exceeded_ocount(lndx), "spec-ecounts", ltp, hdr, prefix)
            handle_spec_recording(get_exceeded_delta(lndx), "spec-edeltas", ltp, hdr, prefix)
#
# A buffer for the averaged reference FFT
#
avg_rfft = [[-220.0]*FFTSIZE]*NCHAN

#
# Cached alpha value
#
ralpha = -200.0

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
last_ref_ndx = 0
ref_hold_off = -1
def do_reference(ref_fft,prate):
    global avg_rfft
    global ralpha
    global frq_ndx
    global last_ref_ndx
    global ref_hold_off
    
    lndx = 0
   
    
    if (ralpha < 0.0):
        ralpha = 1.0-math.pow(math.e,-2*(0.5/prate))
    beta = 1.0-ralpha
    
    #
    # Handle re-size
    #
    if (len(avg_rfft[lndx]) != len(ref_fft)):
        avg_rfft[0] = list(ref_fft)
        avg_rfft[1] = list(ref_fft)
    
    #
    # Single-pole IIR filter it
    #
    tfft = numpy.multiply([ralpha]*len(ref_fft),ref_fft)
    avg_rfft[lndx] = numpy.add(tfft,numpy.multiply([beta]*len(avg_rfft[lndx]),avg_rfft[lndx]))
    
    #
    # Compute reference level from FFT data
    #
    pvect = numpy.multiply(ref_fft, [0.1]*len(ref_fft))
    pvect = numpy.power([10.0]*len(ref_fft),pvect)
    b = beta*get_refval(lndx)
    set_refval((ralpha*numpy.sum(pvect))+b,lndx)
    
    return None

#
# A buffer for the converted-to-kelvin stripchart display
#
stripchart = [[0.0]*FFTSIZE]*2

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
# Pretty much what the name suggests
#
def estimate_Tsky(pace,reftemp,tsys,tsys_ref):
    global frq_ndx
    
    lndx = frq_ndx
    
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
    
    rv = get_refval(lndx)
    if (rv == 0):
        rv = 1.0e-15
    
    pwr = get_current_pwr(lndx)
    
    tsky = ((pwr/rv)*X)-tsys
    set_Tsky(tsky,lndx)  
    
    return (None)
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
minutes_chart = [[0.0]*DAILY_MINUTES]*NCHAN
mtsky = -1
mcounter = 0
def minute_data(p,which):
    global minutes_chart
    global mtsky
    global mcounter
    
    mtsky += get_Tsky(which)
    mcounter += 1
    if ((mcounter % 60) == 0):
        mtsky /= 60.0
        m = minutes_chart[which]
        minutes_chart[which] = [mtsky]+m[0:DAILY_MINUTES-1]
        mtsky = 0.0

    return minutes_chart[which]

def impulses(p,which):
    return impulse_events[which]

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
peakhold = [[0.0]*FFTSIZE]*2

#
# For auto-peak-reset on startup (in ticks)
#
auto_rst=100
auto_init=False

import copy
def handle_peak_hold(fft,ticks):
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
    if (len(peakhold[lndx]) < len(fft)):
        peakhold[lndx] = fft
        
    #
    # If they pushed the "Reset Peak Hold" button in the UI, OR
    #   The auto-rst variable has dropped through zero
    #
    auto_rst -= 1
    if (auto_rst == 0):
        peakhold[0] = copy.deepcopy(get_raw_fft(0))
        peakhold[1] = copy.deepcopy(get_raw_fft(1))
    
    #
    # Do the peak-hold math
    #
    for i in range(0,len(peakhold[lndx])):
        if (fft[i] > peakhold[lndx][i]):
            peakhold[lndx][i] = fft[i]
    
    return None

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
# For auto normal_power setting (in seconds)
#
auto_normal = 30

#
# Expected power level -- for fault detection
#
normal_power=[1.0]*NCHAN

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
    #   declare a probably antenna fault
    #
    if (pwr < (normal_power[which]/3.5)):
        antenna_fault(True)
    else:
        antenna_fault(False)

def handle_pwr_recording(pwr,rv,hdr,ltp,prefix,fdata,which):
    #
    # Record data in a daily file
    #
    fn = prefix+"rio-"
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"

    fp = open(fn, "a")
    fp.write (hdr)
    rv = rv + 1.0e12
    fp.write ("%.7f,%.7f,%.7f,%e,%.1f\n" % (pwr, rv, pwr-rv, pwr/rv, get_Tsky(which)))
    fp.close()
    
    #
    # Record fast data as well
    #
    fn = prefix+"fast-%dHz" % len(fdata)
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"
    fp = open(fn, "a")
    fp.write(hdr)
    for v in fdata:
        fp.write ("(%.7f,%.7f)," % (v[0], v[1]) )
    fp.write("\n")
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
current_pwr = [0.0]*NCHAN
def set_current_pwr(p,which):
    global current_pwr
    
    current_pwr[which] = p

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
