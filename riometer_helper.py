# this module will be imported in the into your flowgraph
import time
import math
import numpy
import random
avg_fft = [0.0]*128
peakhold = [0.0]*128
refval = -1.0
last_time = time.time()
seconds = 0
current_ratio = 1.0
Tsky = 100.0

def modified_fft(infft,prefix,refscale,rst):
    global avg_fft
    global refval
    global last_time
    global seconds
    global current_ratio
    global Tsky
    global peakhold
    dbdict = {}
    for v in infft:
        key = "%d" % v
        if key in dbdict:
            dbdict[key] += 1
        else:
            dbdict[key] = 1
    
    if (infft[0] == -220.0 and infft[1] == -220.0):
        return ([-80.0]*len(infft))
    
    maxkey = "??"
    maxval = 0
    for k in dbdict:
        if dbdict[k] > maxval:
            maxval = dbdict[k]
            maxkey = k
    
    mode = float(maxkey)
    
    outfft = [0.0]*len(infft)
    
    indx = 0
    for v in infft:
        if (v-mode >= 1.0):
            outfft[indx] = mode+random.uniform(-0.4,0.4)
        else:
            outfft[indx] = v
        indx += 1
    
    
    if (len(avg_fft) != len(outfft)):
        avg_fft = list(outfft)
        peakhold = list(infft)
    
    alpha = 0.05
    beta = 1.0-alpha
    new_fft = numpy.multiply(outfft,[alpha]*len(outfft))
    avg_fft = numpy.add(new_fft, numpy.multiply(avg_fft,[beta]*len(outfft)))
    
    if (rst):
        for i in range(0,len(peakhold)):
            peakhold[i] = infft[i]

    for i in range(0,len(peakhold)):
        if (infft[i] > peakhold[i]):
            peakhold[i] = infft[i]
    
    #
    # For display purposes only
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
        # Every two seconds
        #
        if (seconds != 0 and (seconds % 1 == 0)):
            pvect = numpy.multiply(avg_fft, [0.1]*len(avg_fft))
            pvect = numpy.power([10.0]*len(avg_fft),pvect)
            pwr = numpy.sum(pvect)
            
            ltp = time.gmtime(time.time())

            fn = prefix+"rio-"
            fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
            fn += ".csv"

            fp = open(fn, "a")
            rv = refval * refscale
            rv = 1.0e-15 + rv
            fp.write ("%02d,%02d,%02d" % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
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
              
    return (outfft)
    

avg_rfft = [-220.0]*128
def do_reference(ref_fft):
    global refval
    global avg_rfft
    
    alpha = 0.1
    beta = 1.0-alpha
    
    if (len(avg_rfft) != len(ref_fft)):
        avg_rfft = list(ref_fft)
    
    tfft = numpy.multiply([alpha]*len(ref_fft),ref_fft)
    avg_rfft = numpy.add(tfft,numpy.multiply([beta]*len(avg_rfft),avg_rfft))
    
    pvect = numpy.multiply(ref_fft, [0.1]*len(ref_fft))
    pvect = numpy.power([10.0]*len(ref_fft),pvect)
    refval = (alpha*numpy.sum(pvect))+(beta*refval)
    return avg_rfft

stripchart = [0.0]*128
def power_ratio(pace,siz,reftemp,tsys):
    global stripchart
    global current_ratio
    global Tsky
    
    if (len(stripchart) != siz):
        stripchart = [0.0]*siz
    
    #
    # Need to find Sky temp that satisfies:
    #
    # ratio = (Tsky+Tsys)/(Tsys+reftemp)
    #
    X = tsys+reftemp
    Tsky = (current_ratio*X)-tsys
    
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

def annotate(prefix, reftemp, tsys, freq, bw, gain, notes):
    fn = prefix+"annotation-"
    ltp = time.gmtime(time.time())
    
    fn += "%04d%02d%02d" % (ltp.tm_year, ltp.tm_mon, ltp.tm_mday)
    fn += ".csv"
    
    fp = open(fn, "a")
    fp.write("%02d,%02d,%02d," % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
    fp.write("%.1f,%.1f," % (reftemp, tsys))
    fp.write("%.1f,%.1f,%.1f," % (freq, bw, gain))
    fp.write("%s\n" % notes)
    fp.close()

    
    
