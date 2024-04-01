import torch
import time
import torch.cuda.profiler as profiler
import threading
import queue
load_cache_stream = torch.cuda.Stream()


def disk_load_worker(path, cpu_buf, dst):
    with torch.cuda.stream(load_cache_stream):
        x = torch.load(path)
        cpu_buf.copy_(x)
        dst.copy_(cpu_buf, non_blocking=True)
    print("load finish")

def disk_load_worker_naive(path, cpu_buf, dst):
    x = torch.load(path)
    cpu_buf.copy_(x)
    dst.copy_(cpu_buf)
    print("load finish")
    
#The following compute and load cannot be overlapped

x = torch.rand((5000,10000))
xx = torch.rand((13000,13000)).cuda()
y = torch.empty((5000,10000)).cuda()
path = "x.pt"
torch.save(x,path)
x = x.pin_memory()
print(x.device)
yy = torch.matmul(xx,xx)
time.sleep(5)

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

#t1 = threading.Thread(target=disk_load_worker, args=(path,x,y))
#t1 = threading.Thread(target=disk_load_worker_naive, args=(path,x,y))
#t1.start()

disk_load_worker(path,x,y) #0.16-0.21
#disk_load_worker_naive(path,x,y) #0.17-0.2
#yy = torch.matmul(xx,xx) #0.18s

#t1.join()
torch.cuda.synchronize() #sync + join

end.record()
torch.cuda.synchronize()
temp_time = start.elapsed_time(end)
print(temp_time)