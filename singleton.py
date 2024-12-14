import enum
from enum import Enum
import ray
from typing import Union,Dict
import torch
class record_operator:
    #metrics
    prefill_time:list=[]
    decode_time:float=0
    sample_time:float=0
    prefill_attention:float=0
    decode_attention:float=0
    prefill_gemm:float=0
    decode_gemm:float=0
    prefill_schedule:float=0
    decode_schedule:float=0
    
    #parameter
    batchsize:int=0
    output_length:int=0
    model_weight:int=0 #GB

class metricstype(Enum):
    prefill_time="prefill_time"
    decode_time="decode_time"
    prefill_attention="prefill_attention"
    decode_attention="decode_attention"
    prefill_gemm="prefill_gemm"
    decode_gemm="decode_gemm"
    prefill_schedule="prefill_schedule"
    decode_schedule="decode_schedule"
    sample="sample"   
class parametertype(Enum):
    batchsize="batchsize"
    model_weight="model_weight"
    input_length="input_length"
    output_length="output_length"
@ray.remote
class raytimer:
    def __init__(self):
        #metrics
        self.prefill_time:list=[]
        self.decode_time:float=0
        self.prefill_attention:float=0
        self.decode_attention:float=0
        self.prefill_gemm:float=0
        self.decode_gemm:float=0
        self.prefill_schedule:float=0
        self.decode_schedule:float=0
        self.sample:float=0
        #parameter
        self.batchsize:int=0
        self.output_length:int=0
        self.input_length:int=0
        self.model_weight:int=0 #GB
    def add_value(self,type:metricstype,value:Union[int,float]):
        match type:
            case metricstype.prefill_time:
                self.prefill_time+=value
            case metricstype.decode_time:
                self.decode_time+=value
            case metricstype.prefill_attention:
                self.prefill_attention+=value
            case metricstype.decode_attention:
                self.decode_attention+=value
            case metricstype.prefill_gemm:
                self.prefill_gemm+=value
            case metricstype.decode_gemm:
                self.decode_gemm+=value
            case metricstype.prefill_schedule:
                self.prefill_schedule+=value
            case metricstype.decode_schedule:
                self.decode_schedule+=value
            case metricstype.sample:
                self.sample+=value
            case _:
                print(f"warning!type is {type.name} no such type in add_value")
    def set_value(self,type:parametertype,value:Union[int,float]):
        match type:
            case parametertype.batchsize:
                self.batchsize=value
            case parametertype.input_length:
                self.input_length=value
            case parametertype.output_length:
                self.output_length=value
            case parametertype.model_weight:
                self.model_weight=value
            case _:
                print(f"warning!type is {type.name}no such type in set_value")

    def append_prefill(self,prefill_data:dict):
        self.prefill_time.append(prefill_data)
        
    def get_value(self,type:metricstype|parametertype):
        match type:
            case metricstype.prefill_attention:
                return self.prefill_attention
            case metricstype.prefill_gemm:
                return self.prefill_gemm
            case metricstype.decode_attention:
                return self.decode_attention
            case metricstype.decode_gemm:
                return self.decode_gemm
            case metricstype.decode_schedule:
                return self.decode_schedule
            case metricstype.prefill_schedule:
                return self.prefill_schedule
            case metricstype.decode_time:
                return self.decode_time
            case metricstype.prefill_time:
                return self.prefill_time
            case metricstype.sample:
                return self.sample
            case parametertype.model_weight:
                return self.model_weight
            case parametertype.input_length:
                return self.input_length
            case parametertype.output_length:
                return self.output_length
            case parametertype.batchsize:
                return self.batchsize
    
class samebatch:
    input_length:int=0
    batchsize:int=0
    tpot:Dict={}