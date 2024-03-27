import torch

rootdir = "/home/fsuser/hanchen/transformers_fuse/kv_cache"
tiers = ['gpu', 'cpu', 'nfs']
cpu_hash = {}
gpu_hash = {} #key, [size (GB), content]
nfs_hash = {}

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
            return cpu_hash[text_hash][1].cuda()
        else:
            return -1
    elif(tier == 'nfs'):
        if (text_hash in nfs_hash):
            temp = torch.load(nfs_hash[text_hash][1])
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
    return "nfs"

def add_kv(text_hash, kv):
    global cpu_available, disk_available
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

    elif (tier == 'nfs'):
        nfs_hash[text_hash] = [memory_occupied_GB]
        torch.save(kv, f"{rootdir}/{str(text_hash)}")
        nfs_hash[text_hash].append(f"{rootdir}/{str(text_hash)}")
        disk_available += memory_occupied_GB

def add_kv_layer(data, text, layer):
    text_hashed = hash_string(text + str(layer))
    # print("adding hash is: ", text_hashed)
    add_kv(text_hashed, data)

def fetch_kv_layer(text, layer, mask, indices_list):
    text_hashed = hash_string(text + str(layer))
    print("fetching hash is ", text_hashed)
    time, tier = get_predicted_loading_time(text_hashed)
    
    #[ADD] decide whether to fetch
    if (True):
        return fetch_kv(text_hashed, tier)[:,:,indices_list,:]
    
    return -1