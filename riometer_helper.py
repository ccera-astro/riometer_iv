# this module will be imported in the into your flowgraph
import time
import math
import numpy
import random
avg_fft = [0.0]*128
refval = -1.0
last_time = time.time()
seconds = 0
current_ratio = 1.0
Tsky = 100.0
def modified_fft(infft,prefix,refscale):
    global avg_fft
    global refval
    global last_time
    global seconds
    global current_ratio
    global Tsky
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
        if (v-mode >= 2.0):
            outfft[indx] = mode+random.uniform(-0.3,0.3)
        else:
            outfft[indx] = v
        indx += 1
    
    
    if (len(avg_fft) != len(outfft)):
        avg_fft = list(outfft)
    
    new_fft = numpy.multiply(outfft,[0.1]*len(outfft))
    avg_fft = numpy.add(new_fft, numpy.multiply(avg_fft,[0.9]*len(outfft)))
    
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
              
    return (outfft)
    
            
def stash_ref(rv):
    global refval
    
    refval = rv

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
	
	
