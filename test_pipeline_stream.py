import torch
import time
import torch.cuda.profiler as profiler
#import pyprof
#pyprof.init()

x = torch.rand((60000,10000))
xx = torch.rand((13000,13000)).cuda()
y = torch.empty((60000,10000)).cuda()
x = x.pin_memory()
print(x.device)
yy = torch.matmul(xx,xx)

load_stream = torch.cuda.Stream()
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

with torch.cuda.stream(load_stream):
    y.copy_(x, non_blocking=True) #0.19s

yy = torch.matmul(xx,xx) #0.18s
torch.cuda.synchronize()
end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(temp_time)