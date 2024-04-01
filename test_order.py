import torch
import time
import torch.cuda.profiler as profiler
import threading
import queue



path = "x.pt"
x = torch.rand((5000,10000))
cpu_buf = torch.rand((50000,10000))
cpu_tensor = torch.rand((50000,10000))
xx = torch.rand((13000,13000)).cuda()
y = torch.empty((50000,10000)).cuda()
path = "/home/jiayi3/x.pt"
torch.save(x,path)
cpu_buf = cpu_buf.pin_memory()

time.sleep(5)


torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

#x = torch.load(path) #0.12-0.15s
cpu_buf.copy_(cpu_tensor) #0.08-0.09s
y.copy_(cpu_buf) #0.16s


torch.cuda.synchronize() #sync + join

end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(temp_time)