import torch

#FIXME(Jiayi): we should support xxx -> xxx, not just xxx -> gpu

#rootdir = "/home/fsuser/hanchen/transformers_fuse/kv_cache"
rootdir = "/dataheart/jiayi3/transformers_fuse/kv_cache"
tiers = ['gpu', 'cpu', 'cpu_pin', 'nfs']
cpu_hash = {}
cpu_pin_hash = {}
gpu_hash = {} #key, [size (GB), content]
nfs_hash = {}

#HACK(Jiayi): need to be fixed
# for we do not know the orignal devie when saved
original_device = {}

import hashlib

def hash_string(input_string):
    # Use SHA-256 hashing algorithm
    hash_object = hashlib.sha256(input_string.encode())  # Encode the string to bytes
    hash_hex = hash_object.hexdigest()  # Get the hexadecimal representation of the hash
    return hash_hex



#specific computing device
def estimated_time(text_hash, tier):
    if (tier == 'gpu'):
        if (text_hash in gpu_hash):
            return gpu_hash[text_hash][0] / 100
        else:
            return -1
    elif(tier == 'cpu'):
        if (text_hash in cpu_hash):
            return cpu_hash[text_hash][0] / 10
        else:
            return -1
    elif(tier == 'nfs'):
        if (text_hash in nfs_hash):
            return nfs_hash[text_hash][0] / 1
        else:
            return -1
        
    return -1


def get_predicted_loading_time(text_hash):
    for i in tiers:
        time_tier = estimated_time(text_hash, i)
        if (time_tier != -1):
            return (time_tier, i)    
    return (-1, -1)

#[Hanchen] need to add lock for concurrent write read 
def fetch_kv(text_hash, tier):
    if (tier == 'gpu'):
        if (text_hash in gpu_hash):
            return gpu_hash[text_hash][1]
        else:
            return -1
    elif(tier == 'cpu'):
        if (text_hash in cpu_hash):
            return cpu_hash[text_hash][1]#.to(original_device[text_hash])
        else:
            return -1
    elif (tier == 'cpu_pin'):
        if (text_hash in cpu_pin_hash):
            return cpu_pin_hash[text_hash][1]#.to(original_device[text_hash])
        else:
            return -1
    elif(tier == 'nfs'):
        if (text_hash in nfs_hash):
            temp = torch.load(nfs_hash[text_hash][1]).to(original_device[text_hash])
            # print(original_device[text_hash])
            return temp
        else:
            return -1
    return -1

cpu_total = 40
cpu_available = 0
disk_total = 200
disk_available = 0
#Hanchen Need to add logic
def decide_tier_to_add(kv):
    kv_bytes = sum(tensor.element_size() * tensor.nelement() for tensor in kv)
    kv_occupied_GB = kv_bytes / (1024 ** 3)

    #Get GPU gb
    GPU_available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    gpu_available = GPU_available_memory / (1024 ** 3)

    #get CPU gb
    global cpu_available

    #get disk db
    global disk_available

    #determining the logic
    #return "nfs"
    return "cpu"

def add_kv(text_hash, kv, tier=None):
    global cpu_available, disk_available
    if tier is None:
        tier = decide_tier_to_add(kv)
    memory_occupied_bytes = sum(tensor.element_size() * tensor.nelement() for tensor in kv)
    memory_occupied_GB = memory_occupied_bytes / (1024 ** 3)
    if(tier == 'gpu'):
        gpu_hash[text_hash] = [memory_occupied_GB]
        gpu_hash[text_hash].append(torch.clone(kv))

    elif(tier == 'cpu'):
        cpu_hash[text_hash] = [memory_occupied_GB]
        cpu_hash[text_hash].append(kv.to('cpu'))
        cpu_available += memory_occupied_GB
    elif(tier=='cpu_pin'):
        cpu_pin_hash[text_hash] = [memory_occupied_GB]
        cpu_pin_hash[text_hash].append(kv.to('cpu').pin_memory())
    elif (tier == 'nfs'):
        nfs_hash[text_hash] = [memory_occupied_GB]
        torch.save(kv, f"{rootdir}/{str(text_hash)}")
        nfs_hash[text_hash].append(f"{rootdir}/{str(text_hash)}")
        disk_available += memory_occupied_GB
    original_device[text_hash] = kv.device

def add_kv_layer(data, text, layer, tier=None):
    text_hashed = hash_string(text + str(layer))
    # print("adding hash is: ", text_hashed)
    add_kv(text_hashed, data, tier=tier)

def fetch_kv_layer(text, layer, mask=None, tier=None):
    text_hashed = hash_string(text + str(layer))
    # print("fetching hash is ", text_hashed)
    #time, tier = get_predicted_loading_time(text_hashed)
    if tier is None or tier==-1:
        tier = 'cpu'
    #import pdb
    #pdb.set_trace()
    #[ADD] decide whether to fetch
    if (True):
        if mask:
            # FIXME(Jiayi): This is still not efficient enough
            # We are still loading everything
            # Better indexing on the disk?
            kv = fetch_kv(text_hashed, tier)[:,:,mask,:]
        else:
            kv = fetch_kv(text_hashed, tier)
        return kv
    
    return -1