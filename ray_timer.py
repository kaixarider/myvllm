import ray
import os
from enum import Enum

class metricstype(Enum):
    kv_cache="kv_cache"
    prefill_gemm="prefill_gemm"
    prefill_attention="prefill_attention"
    decode_gemm="decode_gemm"
    decode_attention="decode_attention"
    model_weight="model_weight"
    generation_throughput="generation_throughput"
    prompt_throughput="prompt_throughput"
    max_iteration="max_iteration"
    start="start"
    record="record"
    block_size="block_size"
    sample="sample"
@ray.remote
class Timer:
    def __init__(self,worker_name):
        self.worker_name=worker_name
        self.sample={}
        self.kv_cache=0
        self.record={}
        self.prefill_gemm=0
        self.prefill_attention=0
        self.decode_gemm=0
        self.decode_attention=0
        self.model_weight=0
        self.block_size=0
        self.generation_throughput=0
        self.prompt_throughput=0
        self.start=True
        self.max_iteration=0
        self.block_size=0
        pid = os.getpid()  # 获取当前进程号
        print(f"Worker {self.worker_name} starts timer. PID: {pid}")

    def add_value(self,value:float,tp:metricstype):
        if tp==metricstype.prefill_gemm:
            self.prefill_gemm+=value
        elif tp==metricstype.prefill_attention:
            self.prefill_attention+=value
        elif tp==metricstype.decode_gemm:
            self.decode_gemm+=value
        elif tp==metricstype.decode_attention:
            self.decode_attention+=value
        else:
            raise Exception("add unexpected type!")
    def max_value(self,value,tp:metricstype):
        if tp==metricstype.kv_cache:
            self.kv_cache=max(value,self.kv_cache)
        elif tp==metricstype.model_weight:
            self.model_weight=max(value,self.model_weight)
        elif tp==metricstype.generation_throughput:
            self.generation_throughput=max(value,self.generation_throughput)
        elif tp==metricstype.prompt_throughput:
            self.prompt_throughput=max(value,self.prompt_throughput)
        elif tp==metricstype.max_iteration:
            self.max_iteration=max(self.max_iteration,value)
        else:
            raise Exception("max unexpected type!")
    def set_value(self,value,tp:metricstype):
        if tp==metricstype.kv_cache:
            self.kv_cache=value
        elif tp==metricstype.model_weight:
            self.model_weight=value
        elif tp==metricstype.generation_throughput:
            self.generation_throughput=value
        elif tp==metricstype.prompt_throughput:
            self.prompt_throughput=value
        elif tp==metricstype.prefill_gemm:
            self.prefill_gemm=value
        elif tp==metricstype.prefill_attention:
            self.prefill_attention=value
        elif tp==metricstype.decode_gemm:
            self.decode_gemm=value
        elif tp==metricstype.decode_attention:
            self.decode_attention=value
        elif tp==metricstype.max_iteration:
            self.max_iteration=value
        elif tp==metricstype.block_size:
            self.block_size=value
        elif tp==metricstype.sample:
            self.sample=value
        else:
            raise Exception("set unexpected type!")
    def get_value(self,tp:metricstype):
        if tp==metricstype.kv_cache:
            return self.kv_cache
        elif tp==metricstype.model_weight:
            return self.model_weight
        elif tp==metricstype.generation_throughput:
            return self.generation_throughput
        elif tp==metricstype.prompt_throughput:
            return self.prompt_throughput
        elif tp==metricstype.prefill_gemm:
            return self.prefill_gemm
        elif tp==metricstype.prefill_attention:
            return self.prefill_attention
        elif tp==metricstype.decode_gemm:
            return self.decode_gemm
        elif tp==metricstype.decode_attention:
            return self.decode_attention
        elif tp==metricstype.start:
            return self.start
        elif tp==metricstype.record:
            return self.record
        elif tp==metricstype.block_size:
            return self.block_size
        elif tp==metricstype.max_iteration:
            return self.max_iteration
        elif tp==metricstype.sample:
            return self.sample
        else:
            raise Exception("get unexpected type!")
    def set_start_decode(self,start:bool):
        self.start=start
    
    def set_record_value(self,id:int,key:str,value:float):
        if id not in self.record:
            self.record[id]={"prefill":0,"decode":0}
        else:
            self.record[id][key]+=value
    def set_sample_value(self,id:int,value:float):
        if id not in self.record:
            self.sample[id]=0
        else:
            self.sample[id]+=value
    def get_self(self):
        return self
    def show_name(self):
        return self.worker_name