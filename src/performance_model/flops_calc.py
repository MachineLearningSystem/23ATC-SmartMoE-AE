
import os
import sys

from models import GPT
from devices import A100,V100
from utils import GBstr, TFLOPSstr, Gstr, SECstr

assert len(sys.argv) == 6

m = GPT(sys.argv[1])
gbs = int(sys.argv[2])
device_name = sys.argv[3]
dtype = sys.argv[4]
nodes = int(sys.argv[5])

if device_name == 'A100':
    device = A100('my_a100')
elif device_name == 'V100':
    device = V100('my_v100')
else:
    assert False, f"device {device_name} not found."

flops = m.flops(gbs)
p50_exec_time = flops / (0.5 * device.peak(dtype) * nodes)

print("50% utils: " + SECstr(p50_exec_time))
