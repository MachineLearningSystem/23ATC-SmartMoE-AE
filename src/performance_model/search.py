import sys
from models import GPT
from parallelisms import DataParallel, Topo, ExpertParallel, PipelineParallel
from utils import GBstr, TFLOPSstr, Gstr, SECstr, parse_config

models = [
#    'GPT-760M',
#    'GPT-760M-16MoE',
#    'GPT-760M-32MoE',
    'GPT-1.3B',
    'GPT-1.3B-16MoE',
#    'GPT-1.3B-32MoE',
#    'GPT-2.7B',
#    'GPT-2.7B-16MoE',
#    'GPT-2.7B-32MoE',
#    'GPT-6.7B',
#    'GPT-6.7B-16MoE',
#    'GPT-6.7B-32MoE',
    ]


def search_moe(m, topo, gate, global_batch_size, output):
    out = open(output, "w")

    min_exec_time = float('+inf')
    min_spec = ""

    world_size = topo.world_size
    for stages in range(1,world_size//4+1):
        if world_size % stages != 0 or m.num_layers % stages != 0:
            continue
        
        inner_size = world_size // stages

        for ep_size in range(1, inner_size + 1):
            if inner_size % ep_size != 0 or ep_size > m.tot_experts:
                continue

            local_batch_size=1
        
            dp_size = inner_size // ep_size

            def inner_ep(name, model, topo):
                return ExpertParallel(name, model, topo, ep_size=ep_size, dp_size=dp_size)

            plan = PipelineParallel('pp', m, topo, inner_ep, stages)
            mem = plan.mem_usage
            if mem / 1e9 > topo.mem_cap() / 2.0:
                continue
            micro_batch_size = min(global_batch_size, inner_size*local_batch_size)
            exec_time = plan.exec_time(global_batch_size, micro_batch_size, 'fastermoe', gate)
        
            spec = f"{global_batch_size} {micro_batch_size//inner_size} {stages} {inner_size} {ep_size} {dp_size}"

            print(f"{spec} "+GBstr(mem).split(' ')[0]+" "+SECstr(exec_time).split(' ')[0], file=out)

            if exec_time < min_exec_time:
                min_exec_time = exec_time
                min_spec = spec

    print(f"chosen: {min_spec}", file=out)


if __name__ == "__main__":
    cluster_config = sys.argv[1]
    dense_config = sys.argv[2]
    sparse_config = sys.argv[3]
    global_batch_size = int(sys.argv[4])
    output = sys.argv[5]

    cluster_config = parse_config(cluster_config)
    dense_config = parse_config(dense_config)
    sparse_config = parse_config(sparse_config)
    
    topo = Topo(cluster_config)
    model = GPT(dense_config, sparse_config)
    gate = sparse_config['GATE']

    search_moe(model, topo, gate, global_batch_size, output)