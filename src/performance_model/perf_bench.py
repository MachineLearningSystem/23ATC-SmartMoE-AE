import sys
import json
from models import FFN_MoE
from devices import A100
from parallelisms import Topo

topo_sh16 = Topo("sh-lab_2nodes", 16, bw_sendrecv_intra=float('40e9'), bw_sendrecv_inter=float('12e9'), bw_allreduce_intra=float('100e9'), bw_allreduce_inter=float('8e9'), device_name='A100')

topo_sh8 = Topo("sh-lab_1nodes", 8, bw_sendrecv_intra=float('50e9'), bw_sendrecv_inter=float('12e9'), bw_allreduce_intra=float('100e9'), bw_allreduce_inter=float('8e9'), device_name='A100')

log = sys.argv[1]
table_prefix = sys.argv[2]
world_size = int(sys.argv[3])
batch_size = int(sys.argv[4]) * 1024


def estimate(iter, layer_idx):
    global table_prefix,world_size,batch_size
    hidden_size = 2048
    tot_experts = 32
    alpha = 2
    top_k = 2
    dtype = 'fp16'

    device = A100()

    m = FFN_MoE('moe', hidden_size, alpha, tot_experts, top_k, dtype)
    
    return m.exec_time(batch_size, device, topo=topo_sh8 if world_size == 8 else topo_sh16, method='fastermoe', gate='file', gate_file=table_prefix, gate_iter=iter, gate_layer_idx=layer_idx)

def naive_estimate():
    global table_prefix,world_size,batch_size
    hidden_size = 2048
    tot_experts = 32
    alpha = 2
    top_k = 2
    dtype = 'fp16'

    device = A100()

    m = FFN_MoE('moe', hidden_size, alpha, tot_experts, top_k, dtype)
    
    return m.exec_time(batch_size, device, topo=topo_sh8 if world_size == 8 else topo_sh16, method='fastermoe', gate='gshard')

if __name__ == '__main__':
    x = []
    y = []
    with open(log, "r") as f:
        for line in f.readlines():
            if line[:6] == 'test d':
                data = line.split(' ')
                iter = int(data[12])
                layer = int(data[14].split(':')[0])
                if iter < 100:
                    continue
                round = int(data[16])
                #if iter != 4901 or layer != 23 or world_size != 16:
                #    continue
                vals = data[17:22]
                t = {}
                for val in vals:
                    name, t0 = val.split('=')
                    t0 = float(t0)
                    t[name] = t0
                if round == 0:
                    t_rounds = []
                t_rounds.append(t)
                if round != 3:
                    continue

                t_real = min([t['FastMoE']*1000 for t in t_rounds])
                t_est = estimate(iter, layer) * 1000
                x.append(t_est)
                y.append(t_real)
                
    
    d = {'real':y,'est':x}
    output = f"/home/zms/test/jupy/perf_model_{world_size//8}nodes_mbs{batch_size//1024}.log"
    with open(output, "w") as f:
        print(naive_estimate()*1000, file=f)
        print(json.dumps(d), file=f)