
class V100():

    def __init__(self, name='V100'):
        self.name = name

    def peak(self, dtype):
        if dtype == "fp16":
            return float('112e12')
        else:
            assert False, f"{self.name} peak dtype {dtype} not found."

    def matmul_perf(self, dtype):
        if dtype == "fp16":
            return float('112e12')
        else:
            assert False, f"{self.name} matmul dtype {dtype} not found."

    def mem_cap(self):
        return 32 # GB

class A100():
    
    def __init__(self, name='A100'):
        self.name = name

    def peak(self, dtype):
        if dtype == "fp16":
            return float('312e12')
        else:
            assert False, f"{self.name} peak dtype {dtype} not found."


    def matmul_perf(self, dtype):
        if dtype == "fp16":
            return float('280e12')
        else:
            assert False, f"{self.name} matmul dtype {dtype} not found."

    def mem_cap(self):
        return 80 # GB