import torch
import time
import torch.cuda.profiler as profiler
#import pyprof
#pyprof.init()

chunk_size = 10 #1, 10, 100
total_len = 10000
chunk_num = int(total_len/chunk_size)

y_size = 10000

x = [torch.rand((chunk_size, y_size)) for i in range(chunk_num)]
y_chunk = torch.empty((chunk_size, y_size)).cuda()
x_total = torch.rand((total_len, y_size))
y = torch.empty((total_len, y_size)).cuda()
x = [xx.pin_memory() for xx in x]
x_total = x_total.pin_memory()
#print(x.device)


torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

y.copy_(x_total)

end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(f"Load time:{temp_time}")

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for c in range(chunk_num):
    y[c*chunk_size:(c+1)*chunk_size].copy_(x[c])

end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(f"Looped load time:{temp_time}")


torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for c in range(chunk_num):
    y_chunk.copy_(x[c])

end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(f"Looped load time (no idx):{temp_time}")


load_streams = [torch.cuda.Stream() for i in range(chunk_num)]
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for c in range(chunk_num):
    with torch.cuda.stream(load_streams[c]):
        y[c*chunk_size:(c+1)*chunk_size].copy_(x[c],non_blocking=True)

torch.cuda.synchronize()
end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(f"Streamed load time:{temp_time}")
