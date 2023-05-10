
def parse_config(config_path):
    configs = {}
    with open(config_path, "r") as f:
        lines = f.readlines()
        for l in lines:
            if len(l) > 6 and l[:6] == 'export':
                k, v= l.split(' ')[1].split('=')
                configs[k] = v.strip("\n\"")

    return configs

def dtype_size(dtype):
    if dtype == "fp16":
        return 2
    else:
        assert False, f"dtype {dtype} not found."

def dtype_storage_size(dtype):
    if dtype == "fp16":
        # one fp16 params need 20byte for storage.
        # ref to: Zero-Infinity Chapter 3
        return 20
    else:
        assert False, f"dtype {dtype} not found."

def Gstr(x):
    return f"{x/1e9:.2f} G"

def GBstr(x):
    return f"{x/1e9:.2f} GB"

def TFLOPSstr(x):
    return f"{x/1e12:.2f} TFLOPS"

def SECstr(x):
    return f"{x:.3f} seconds"
