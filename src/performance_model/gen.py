
from models import GPT
from parallelisms import DataParallel, Topo, ExpertParallel, PipelineParallel
from utils import GBstr, TFLOPSstr, Gstr, SECstr

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



topo = Topo("nico", 16, bw_alltoall=float('6e9'), bw_allreduce=float('1.5e9'))
global_batch_size = 64
for name in models:
    m = GPT(name)
    print(m.name, GBstr(m.tot_mem), GBstr(m.dense_mem))
    if not m.is_dense:
        plan = ExpertParallel('ep', m, topo, ep_size=16, dp_size=1)
        plan.report_mem_usage()

        exec_time_naive = plan.exec_time(global_batch_size, 'naive')
        exec_time_fastermoe = plan.exec_time(global_batch_size, 'fastermoe')

        print("naive estimate: " + SECstr(exec_time_naive))
        print("fastermoe estimate: " + SECstr(exec_time_fastermoe))

        edp_plan = ExpertParallel('ep', m, topo, ep_size=8, dp_size=2)
        edp_plan.report_mem_usage()

        edp_exec_time_naive = edp_plan.exec_time(global_batch_size, 'naive')
        edp_exec_time_fastermoe = edp_plan.exec_time(global_batch_size, 'fastermoe')

        print("edp naive estimate: " + SECstr(edp_exec_time_naive))
        print("edp fastermoe estimate: " + SECstr(edp_exec_time_fastermoe))

        stages = 2
        def inner_ep(name, model, topo):
            return ExpertParallel(name, model, topo, ep_size=16//stages, dp_size=1)
        
        def inner_ep_d2(name, model, topo):
            return ExpertParallel(name, model, topo, ep_size=16//stages//2, dp_size=2)

        pp_plan = PipelineParallel('pp', m, topo, inner_ep, stages)
        pp_plan.report_mem_usage()
        micro_batch_size = (16 // stages) * 1
        pp_exec_time_naive = pp_plan.exec_time(global_batch_size, micro_batch_size, 'naive')
        pp_exec_time_fastermoe = pp_plan.exec_time(global_batch_size, micro_batch_size, 'fastermoe')
        
        print("pipeline naive estimate: " + SECstr(pp_exec_time_naive))
        print("pipeline fastermoe estimate: " + SECstr(pp_exec_time_fastermoe))
        
        stages = 2
        pp_edp_plan = PipelineParallel('pp', m, topo, inner_ep_d2, stages)
        pp_edp_plan.report_mem_usage()
        micro_batch_size = (16 // stages) * 1
        pp_edp_exec_time_naive = pp_edp_plan.exec_time(global_batch_size, micro_batch_size, 'naive')
        pp_edp_exec_time_fastermoe = pp_edp_plan.exec_time(global_batch_size, micro_batch_size, 'fastermoe')
        
        print("pipeline+edp naive estimate: " + SECstr(pp_edp_exec_time_naive))
        print("pipeline+edp fastermoe estimate: " + SECstr(pp_edp_exec_time_fastermoe))

    else:
        plan = DataParallel('dp', m, topo)
        exec_time = plan.exec_time(global_batch_size)
        print(SECstr(exec_time)) 

# ddp = DataParallel("ddp", m, topo)

# flops = m.flops(256)
# print(TFLOPSstr(flops))
# exec_time = ddp.exec_time(256)
# print(exec_time)
# print(TFLOPSstr(flops / exec_time / 16))