
from models import GPT
from parallelisms import DataParallel, Topo, ExpertParallel, PipelineParallel
from utils import *

topo = Topo("nico", 16, bw_alltoall=float('6e9'), bw_allreduce=float('3e9'))
m = GPT('GPT-1.3B')

ddp = DataParallel("ddp", m, topo)

flops = m.flops(256)
exec_time = ddp.exec_time(256)
print(SECstr(exec_time))
print(TFLOPSstr(flops / exec_time / topo.world_size))