# Test Code for binRep: It show how the an 32 bit floating point number is represented in a binary string.

import tensorflow as tf
import random
import numpy as np
import ctypes

# The following code comes from:
# https://stackoverflow.com/questions/16444726/binary-representation-of-float-in-python-bits-not-hex
#
def binRep(num):
        binNum = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
        print("bits: " + binNum.rjust(32,"0"))
        mantissa = "1" + binNum[-23:]
        print("sig (bin): " + mantissa.rjust(24))
        mantInt = int(mantissa,2)/2**23
        print("sig (float): " + str(mantInt))
        base = int(binNum[-31:-23],2)-127
        print("base:" + str(base))
        sign = 1-2*("1"==binNum[-32:-31].rjust(1,"0"))
        print("sign:" + str(sign))
        print("recreate:" + str(sign*mantInt*(2**base)))

        
# Test...
binRep(0.01171875)
binRep(0.013671875)
binRep(0.015625)
binRep(0.01953125)
binRep(0.0234375)
binRep(0.02734375)
binRep(0.03125)
binRep(0.0390625)
binRep(0.046875)
binRep(0.0546875)
binRep(0.0625)
binRep(3.0)

