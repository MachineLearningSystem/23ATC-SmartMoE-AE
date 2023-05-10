
from utils import dtype_size, GBstr
from devices import V100,A100

class Topo():

    def __init__(self, configs):
        print(configs)
        self.name = configs['cluster_name']
        self.world_size = int(configs['NNODES']) * 8

        self.bw_allreduce_inter = float(configs['FMOE_FASTER_GLBPLC_NETBW_Bcast'])
        self.bw_allreduce_intra = float(configs['FMOE_FASTER_GLBPLC_NETBW_Bcast'])
        
        self.bw_sendrecv_inter = float(configs['FMOE_FASTER_GLBPLC_NETBW'])
        self.bw_sendrecv_intra = float(configs['FMOE_FASTER_GLBPLC_NETBW'])

        if configs['FMOE_FASTER_GLBPLC_GPUTP'] == "112e12":
            self.device = V100()
        elif configs['FMOE_FASTER_GLBPLC_GPUTP'] == "314e12":
            self.device = A100()

    def mem_cap(self):
        return self.device.mem_cap()

    def __repr__(self):
        return f"{self.name}_nodes{self.world_size}"

def split_topo(global_topo : Topo, cnt):
    assert global_topo.world_size % cnt == 0, "Split topo error"
    import copy
    sub_topo = copy.deepcopy(global_topo)
    sub_topo.world_size = global_topo.world_size // cnt
    
    sub_topo.bw_allreduce_inter = global_topo.bw_allreduce_inter
    sub_topo.bw_allreduce_intra = global_topo.bw_allreduce_intra
    
    sub_topo.bw_sendrecv_inter = global_topo.bw_sendrecv_inter
    sub_topo.bw_sendrecv_intra = global_topo.bw_sendrecv_intra
    
    sub_topo.device = global_topo.device

    return sub_topo

def AllReduce_exec_time(msg_size, topo: Topo):
    bw = topo.bw_allreduce_inter if topo.world_size > 8 else topo.bw_allreduce_intra
    return msg_size * 2 * (topo.world_size - 1) / topo.world_size / bw

class DataParallel():

    def __init__(self, name, model, topo):
        self.name = name
        self.model = model
        self.topo = topo
        self.dtype = model.dtype
        self.tot_dense_params_mem = self.model.dense_params * dtype_size(self.dtype)

    def sync_exec_time(self):
        return AllReduce_exec_time(self.tot_dense_params_mem, self.topo)
    
    def pipeline_sync_exec_time(self, local_layers):
        params_mem = self.model.pipeline_dense_params(local_layers) * dtype_size(self.dtype)
        return AllReduce_exec_time(params_mem, self.topo)

    def pipeline_mem_usage(self, local_layers):
        return self.mem_usage * local_layers / self.model.num_layers
    
    def fwd_bwd_exec_time(self, batch_size):
        micro_batch_size = batch_size / self.topo.world_size
        return self.model.exec_time(micro_batch_size, self.topo.device)

    def pipeline_fwd_bwd_exec_time(self, local_layers, batch_size, *args):
        micro_batch_size = batch_size / self.topo.world_size
        return self.model.pipeline_exec_time(local_layers, micro_batch_size, self.topo.device)

    def exec_time(self, batch_size, *args):
        return self.fwd_bwd_exec_time(batch_size) + self.sync_exec_time()

class ExpertParallel():

    def __init__(self, name, model, topo, ep_size=None, dp_size=None):
        assert not model.is_dense, "Only MoE models use Expert Parallel."
        assert ep_size * dp_size == topo.world_size, "world size mismatch in Expert Parallel."
        self.name = name
        self.model = model
        self.topo = topo
        self.ep_size = ep_size
        self.dp_size = dp_size
        self.dtype = model.dtype
        
        self.ep_topo = split_topo(topo, self.dp_size)
        self.dp_topo = split_topo(topo, self.ep_size)

        self.sync_params_mem = self.model.dense_params * dtype_size(self.dtype)
        if self.dp_size > 1:
            self.sync_params_mem += self.model.sparse_params * self.dp_size / self.ep_size * dtype_size(self.dtype) 

    def sync_exec_time(self):
        return AllReduce_exec_time(self.sync_params_mem, self.dp_topo)

    def pipeline_sync_exec_time(self, local_layers):
        params_mem = self.model.pipeline_dense_params(local_layers) * dtype_size(self.dtype)
        if self.dp_size > 1:
            params_mem += self.model.pipeline_sparse_params(local_layers) / self.ep_size * dtype_size(self.dtype) 
        return AllReduce_exec_time(params_mem, self.dp_topo)

    def fwd_bwd_exec_time(self, batch_size, method, gate):
        micro_batch_size = batch_size / self.topo.world_size
        return self.model.exec_time(micro_batch_size, self.topo.device, topo=self.ep_topo, method=method, gate=gate)
    
    def pipeline_fwd_bwd_exec_time(self, local_layers, batch_size, method, gate):
        micro_batch_size = batch_size / self.topo.world_size
        return self.model.pipeline_exec_time(local_layers, micro_batch_size, self.topo.device, topo=self.ep_topo, method=method, gate=gate)
    
    def exec_time(self, batch_size, method, gate):
        return self.fwd_bwd_exec_time(batch_size, method, gate) + self.sync_exec_time()

    @property
    def mem_usage(self):
        mem = self.model.dense_mem + self.model.sparse_mem / self.ep_size
        return mem

    def pipeline_mem_usage(self, local_layers):
        return self.mem_usage * local_layers / self.model.num_layers

    def report_mem_usage(self):
        print(f"EP mem per device " + GBstr(self.mem_usage))

class PipelineParallel():

    def __init__(self, name, model, topo, inner_parallel, pipeline_stages):
        self.name = name
        self.model = model
        self.topo = topo
        self.inner_parallel = inner_parallel
        self.pipeline_stages = pipeline_stages
        self.dtype = model.dtype

        self.pipeline_topo = split_topo(topo, pipeline_stages)

        assert model.num_layers % pipeline_stages == 0, "num layers for pipeline error."
        self.local_layers = model.num_layers // pipeline_stages

        self.inner = self.inner_parallel(name + "_inner", model, self.pipeline_topo)
    
    def sync_exec_time(self):
        return self.inner.pipeline_sync_exec_time(self.local_layers)

    def exec_time(self, global_batch_size, micro_batch_size, method=None, gate=None):
        num_micro_batch = global_batch_size // micro_batch_size
        single_fwd_bwd_time = self.inner.pipeline_fwd_bwd_exec_time(self.local_layers, micro_batch_size, method, gate)
        fwd_bwd_time = (num_micro_batch + self.pipeline_stages - 1) * single_fwd_bwd_time

        local_batch_size = micro_batch_size // self.pipeline_topo.world_size
        single_pipe_send_time = self.inner.model.io_mem(local_batch_size) / self.topo.bw_sendrecv_inter * num_micro_batch
        send_recv_time = single_pipe_send_time * (2 * self.pipeline_stages - 2)
        print(send_recv_time)
        # print(fwd_bwd_time, self.sync_exec_time(), send_recv_time)

        return fwd_bwd_time + self.sync_exec_time() + send_recv_time

    @property
    def mem_usage(self):
        return self.inner.pipeline_mem_usage(self.local_layers)
    
    def report_mem_usage(self):
        print(f"PP mem per device " + GBstr(self.mem_usage))