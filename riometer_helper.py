# this module will be imported in the into your flowgraph
import time
import math
import numpy
import random
avg_fft = [0.0]*128
refval = -1.0
last_time = time.time()
seconds = 0
def modified_fft(infft,prefix,refscale):
    global avg_fft
    global refval
    global last_time
    global seconds
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
        if (abs(v-mode) > 1.0):
            outfft[indx] = mode+random.uniform(-0.25,0.25)
        else:
            outfft[indx] = v
        indx += 1
    
    
    if (len(avg_fft) != len(outfft)):
        avg_fft = list(outfft)
    
    new_fft = numpy.multiply(outfft,[0.2]*len(outfft))
    avg_fft = numpy.add(new_fft, numpy.multiply(avg_fft,[0.8]*len(outfft)))
    
    #
    # For display purposes only
    #
    outfft = numpy.add(avg_fft, [20.0]*len(avg_fft))
    
    if ((time.time() - last_time) >= 1.0):
        last_time = time.time()
        seconds += 1

        #
        # Power estimates
        #
        if (seconds != 0 and (seconds % 2 == 0)):
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
            fp.write ("%02d,%02d,%02d," % (ltp.tm_hour, ltp.tm_min, ltp.tm_sec))
            fp.write ("%.7f,%.7f,%.7f,%e\n" % (pwr, refval, pwr-rv, pwr/rv))
            fp.close()
            
        #
        # Spectral
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
    
            
def stash_ref(rv):
    global refval
    
    refval = rv
