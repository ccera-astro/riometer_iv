#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  relay_server.py
#  
#  Copyright 2019 Marcus D. Leech <mleech@localhost.localdomain>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  


import sys
import os
import time
import math
from pylibftdi import BitBangDevice
import SimpleXMLRPCServer

bits=[0]*8
def set_bit(b,v):
    global bits
    global bbd
    
    bits[b%8] = v % 2
    msk = 0
    for x in range(8):
        if (bits[x] == 1):
            msk |= 1<<x
    bbd.port = msk

        

def main():
    global bbd
    
    bbd = None
    #
    # Create an FTDI BigBangDevice instance, based on serial number passed in
    #
    bbd = BitBangDevice(device_index=0)
    
    #
    # Set direction regiser
    #
    bbd.direction = 0xFF
    
    #
    # Set initial port value to 0
    #
    bbd.port = 0
    
    #
    # That worked, setup XMLRPC server
    #
    server = SimpleXMLRPCServer.SimpleXMLRPCServer(('localhost', int(sys.argv[1])), logRequests=False, allow_none=True)
    server.register_introspection_functions()
    server.register_function(set_bit, 'set_bit')
    server.serve_forever()
    return 0

if __name__ == '__main__':
    main()

