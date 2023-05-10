from devices import V100
from utils import dtype_size, dtype_storage_size
import numpy as np
import json

class Linear():

    def __init__(self, name, in_feat, out_feat, dtype):
        self.name = name
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.dtype = dtype

    @property
    def params(self):
        return self.in_feat * self.out_feat

    def flops(self, batch_size):
        return 3 * 2 * batch_size * self.in_feat * self.out_feat
    
    def exec_time(self, batch_size, device):
        result = self.flops(batch_size) / device.matmul_perf(self.dtype)
        # print(f"batch_size={batch_size} in={self.in_feat} out={self.out_feat} exec={result}")
        return result

class Attn_OP():

    def __init__(self, name, hidden_size, dtype):
        self.name = name
        self.hidden_size = hidden_size
        self.dtype = dtype

    def flops(self, batch_size, seq_len):
        batch_size /= seq_len
        return 3 * 4 * batch_size * (seq_len**2) * self.hidden_size

    def exec_time(self, batch_size, seq_len, device):
        return self.flops(batch_size, seq_len) / device.matmul_perf(self.dtype)

class Attn():

    def __init__(self, name, hidden_size, dtype):
        self.name = name
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.qkv = Linear("attn_qkv", self.hidden_size, 3 * self.hidden_size, self.dtype)
        self.linear = Linear("attn_linear", self.hidden_size, self.hidden_size, self.dtype)
        self.attn_op = Attn_OP("attn_op", self.hidden_size, self.dtype)

    @property
    def params(self):
        return self.qkv.params + self.linear.params

    def flops(self, batch_size, seq_len):
        return self.qkv.flops(batch_size) + self.linear.flops(batch_size) + self.attn_op.flops(batch_size, seq_len)
    
    def exec_time(self, batch_size, seq_len, device):
        result = self.qkv.exec_time(batch_size, device) + self.linear.exec_time(batch_size, device) + self.attn_op.exec_time(batch_size, seq_len, device)
        result *= 1.7
        # print("attn time", result)
        return result

class FFN():

    def __init__(self, name, hidden_size, alpha, dtype):
        self.name = name
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.dtype = dtype
        self.linear = Linear("ffn_linear", self.hidden_size, self.hidden_size * self.alpha, self.dtype)

    @property
    def params(self):
        return 2 * self.linear.params 

    def flops(self, batch_size):
        return 2 * self.linear.flops(batch_size)

    def exec_time(self, batch_size, device, **kwargs):
        result = 2 * self.linear.exec_time(batch_size, device)
        # print("ffn time", batch_size, result)
        return result

def get_routing_table(micro_batch_size, tot_experts, world_size, gate=None, gate_file=None, gate_iter=None, gate_layer_idx=None):
    table = np.zeros([tot_experts, world_size], dtype=float)

    assert tot_experts % world_size == 0, "expert place error"
    num_experts = tot_experts // world_size
    expert_mapping = [idx // num_experts for idx in range(tot_experts)]

    # naive estimation: evenly dispatch to each expert
    # per_expert_tokens = [micro_batch_size / tot_experts for _ in range(tot_experts)]

    if gate == "gshard":
        cap = 1.2
        ratio = (1 - cap / tot_experts) / (tot_experts - 2)

        per_expert_tokens = [ratio * micro_batch_size for _ in range(tot_experts)]
        per_expert_tokens[0] = cap / tot_experts * micro_batch_size
        per_expert_tokens[-1] = cap / tot_experts * micro_batch_size
        
        for i in range(tot_experts):
            for j in range(world_size):
                table[expert_mapping[i]][j] += per_expert_tokens[i]

    elif gate == "naive":
        ratio = 0.7
        # top-2 estimation
        per_expert_tokens = [ratio * micro_batch_size / tot_experts for _ in range(tot_experts)]
        per_expert_tokens[0] += (1 - ratio) / 2 * micro_batch_size
        per_expert_tokens[-1] += (1 - ratio) / 2 * micro_batch_size
        
        for i in range(tot_experts):
            for j in range(world_size):
                table[expert_mapping[i]][j] += per_expert_tokens[i]
    elif gate == "file":
        
        rank = 0
        filename = gate_file + f"_iter{gate_iter}_rank{rank}.log"

        with open(filename, "r") as f:
            lines = f.readlines()
            line = lines[gate_layer_idx]
            _, log_table = line.split(':')
            microbatch_table = json.loads(log_table)[0]
            total = sum(microbatch_table)
            per_expert_tokens = [microbatch_table[e] / total * micro_batch_size for e in range(tot_experts)]

            for i in range(tot_experts):
                for j in range(world_size):
                    table[i][j] += per_expert_tokens[i]    
    else:
        assert False, f"gate {gate} not found."


    
    return table

class FFN_MoE():

    def __init__(self, name, hidden_size, alpha, tot_experts, top_k, dtype):
        self.name = name
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.tot_experts = tot_experts
        self.top_k = top_k
        self.dtype = dtype
        self.linear = Linear("ffn_linear", self.hidden_size, self.hidden_size * self.alpha, self.dtype)

    @property
    def params(self):
        return 2 * self.linear.params * self.tot_experts 

    def flops(self, batch_size):
        return 2 * self.linear.flops(batch_size) * self.top_k

    def exec_time(self, batch_size, device, topo=None, method=None, gate=None, **kwargs):
        assert topo is not None
        if method == 'naive':
            result = 2 * self.linear.exec_time(batch_size, device) * self.top_k
        elif method == 'fastermoe':
            routing_table = get_routing_table(self.top_k * batch_size, self.tot_experts, topo.world_size, gate=gate, **kwargs)

            num_experts = self.tot_experts // topo.world_size

            bw_alltoall_net = topo.bw_sendrecv_inter if topo.world_size > 8 else topo.bw_sendrecv_intra
            bw_mm = device.matmul_perf(self.dtype)

            data_size = dtype_size(self.dtype)
            alpha = self.alpha
            d_model = self.hidden_size

            alphaH2 = alpha * (d_model ** 2)
            send_feature_time = d_model * data_size / bw_alltoall_net
            comp_time = 2 * 2 * alphaH2 / bw_mm # MxNxK matmul need O(2MNK) ops, 2 matmul of [1,h]x[h,ah]
    
            mm_size = 4096
            def tokens_comp_time(tokens):
                if tokens <= 0:
                    return 0
                # 1 for fwd, 2 for bwd, 1+2=3
                return 3 * comp_time * tokens * (min(5, mm_size/tokens) if tokens < mm_size else 1) + 0.0007

            def tokens_send_time(tokens):
                # 2 times(scatter & gather) for fwd, 2 times for bwd, 2+2=4
                return 4 * send_feature_time * tokens
            
            bw_size = 262144 * 32
            def tokens_inter_send_time(tokens):
                if tokens <= 0:
                    return 0
                send_feature_time = d_model * data_size / topo.bw_sendrecv_inter #* min(20, max(1, bw_size / tokens / d_model / data_size))
                # 2 times(scatter & gather) for fwd, 2 times for bwd, 2+2=4
                return 4 * send_feature_time * tokens + 0.0007
            
            def tokens_intra_send_time(tokens):
                if tokens <= 0:
                    return 0
                send_feature_time = d_model * data_size / topo.bw_sendrecv_intra #* min(20, max(1, bw_size / tokens / d_model / data_size))
                # 2 times(scatter & gather) for fwd, 2 times for bwd, 2+2=4
                return 4 * send_feature_time * tokens

            per_expert_count = routing_table.sum(1)
            expert_idx = per_expert_count.argsort() # min to max
            # print(per_expert_count)

            def expert_send_count(e_idx):
                assert per_expert_count.shape[0] == routing_table.shape[1]
                return per_expert_count[e_idx] - routing_table[e_idx][e_idx]
            
            def expert_inter_send_time(idx):
                t = 0
                for i in range(topo.world_size):
                    if i // 8 != idx // 8:
                        cnt = 0
                        for j in range(idx * num_experts, (idx + 1) * num_experts):
                            cnt += routing_table[j][i]
                        #print("inter", idx, cnt)
                        t += tokens_inter_send_time(cnt)
                return t
            
            def expert_intra_send_time(idx):
                t = 0
                for i in range(topo.world_size):
                    if i // 8 == idx // 8 and i != idx:
                        cnt = 0
                        for j in range(idx * num_experts, (idx + 1) * num_experts):
                            cnt += routing_table[j][i]
                        t += tokens_intra_send_time(cnt)
                return t

            fastmoe_time = 0
            for i in range(topo.world_size):
                cur = expert_inter_send_time(i) + expert_intra_send_time(i)
                for j in range(num_experts):
                    cnt = 0
                    for k in range(topo.world_size):
                        cnt += routing_table[j + i * num_experts][k]
                    cur += tokens_comp_time(cnt)

                fastmoe_time = max(fastmoe_time, cur)
            # fastmoe_time = tokens_comp_time(per_expert_count[expert_idx[-1]]) + tokens_send_time(expert_send_count(expert_idx[-1])) 

            result = fastmoe_time
        else:
            assert False, f"Estimation method {method} not found."
            
        return result

class GPT():

    def __init__(self, dense_configs, sparse_configs):

        self.name = 'GPT'

        self.vocab_size = 53227
        self.dtype = "fp16"

        self.num_layers = int(dense_configs['NUM_LAYERS'])
        self.hidden_size = int(dense_configs['HIDDEN_SIZE'])
        self.num_attn_heads = int(dense_configs['NUM_ATTN_HEADS'])
        self.tot_experts = int(sparse_configs['TOT_EXPERTS'])
        self.seq_len = int(dense_configs['SEQ_LEN'])
        self.top_k = 2

        self.emb = Linear("emb", self.vocab_size, self.hidden_size, self.dtype)
        self.attn = Attn("attn", self.hidden_size, self.dtype)
        if self.is_dense:
            self.ffn = FFN("ffn", self.hidden_size, self.ffn_alpha, self.dtype)
        else:
            self.ffn = FFN_MoE("ffn_moe", self.hidden_size, self.ffn_alpha, self.tot_experts, self.top_k, self.dtype)

    @property
    def is_dense(self):
        return self.tot_experts == -1

    @property
    def tot_mem(self):
        return self.tot_params * dtype_storage_size(self.dtype)

    @property
    def dense_mem(self):
        return self.dense_params * dtype_storage_size(self.dtype)

    @property
    def sparse_mem(self):
        return self.tot_mem - self.dense_mem

    def io_mem(self, batch_size):
        return self.seq_len * batch_size * self.hidden_size * dtype_size(self.dtype)

    @property
    def tot_params(self):
        return self.emb.params + self.num_layers * self.layer_params

    @property
    def sparse_params(self):
        return self.tot_params - self.dense_params

    @property
    def dense_params(self):
        if self.is_dense:
            return self.tot_params
        else:
            return self.emb.params + self.num_layers * self.attn.params 

    def pipeline_dense_params(self, local_layers):
        if self.is_dense:
            return self.layer_params * local_layers
        else:
            return self.attn.params * local_layers

    def pipeline_sparse_params(self, local_layers):
        if self.is_dense:
            return 0
        else:
            return self.ffn.params * local_layers

    @property
    def layer_params(self):
        return self.attn.params + self.ffn.params

    @property
    def ffn_alpha(self):
        if self.is_dense:
            return 4
        else:
            if self.name[:4] == 'Alpa':
                return 8
            assert 4 % self.top_k == 0
            return 4 // self.top_k

    def flops(self, batch_size):
        batch_size = batch_size * self.seq_len
        return self.num_layers * self.layer_flops(batch_size, self.seq_len)

    def layer_flops(self, batch_size, seq_len):
        return self.attn.flops(batch_size, seq_len) + self.ffn.flops(batch_size)

    def exec_time(self, batch_size, device, topo=None, method=None, gate=None):
        assert self.is_dense or (topo is not None), "MoE models need cluster topology to estimate exec. time"
        assert self.is_dense or (gate is not None), "MoE models need gate type to estimate exec. time"
        batch_size = batch_size * self.seq_len
        return self.num_layers * self.layer_exec_time(batch_size, self.seq_len, device, topo=topo, method=method, gate=gate)

    def pipeline_exec_time(self, local_layers, batch_size, device, topo=None, method=None, gate=None):
        assert self.is_dense or (topo is not None), "MoE models need cluster topology to estimate exec. time"
        assert self.is_dense or (gate is not None), "MoE models need gate type to estimate exec. time"
        batch_size = batch_size * self.seq_len
        return local_layers * self.layer_exec_time(batch_size, self.seq_len, device, topo=topo, method=method, gate=gate)

    def layer_exec_time(self, batch_size, seq_len, device, topo=None, method=None, gate=None):
        attn_time = self.attn.exec_time(batch_size, seq_len, device)
        ffn_time = self.ffn.exec_time(batch_size, device, topo=topo, method=method, gate=gate)
        # print(f"attn_time {attn_time:.3f} ffn_time {ffn_time:.3f}")
        return attn_time + ffn_time