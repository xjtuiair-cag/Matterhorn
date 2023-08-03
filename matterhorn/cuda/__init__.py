stdp = None
try:
    from matterhorn_cuda import stdp
except:
    raise ImportError("Please install materhorn_cuda first!")