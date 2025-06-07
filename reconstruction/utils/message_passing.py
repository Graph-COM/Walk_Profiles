import torch

def message_passing(x, edge_index, num_nodes, degree=None):
    # x: [*, N], edge_index: [2, E]
    if edge_index.size(0) == 0:
        # in case of no edge index
        return torch.zeros_like(x)
    sources = edge_index[0]
    targets = edge_index[1]
    x_s = x[..., sources]
    if degree is not None:
        x_s = x_s / (degree[None, None, sources] + 1e-8)

    try:
        from torch_scatter import scatter_add
        x_t = scatter_add(x_s, targets, dim=-1, dim_size=num_nodes)
    except:
        # a manual scatter_add of 3-d tensor
        #targets = targets[None][None].tile([1, x_s.size(1), 1])
        x_t = torch.zeros(*x_s.shape[:-1], num_nodes, dtype=x_s.dtype, device=x_s.device)
        expanded_targets = targets.unsqueeze(0).unsqueeze(0).expand(x_s.shape[0], x_s.shape[1], -1)
        #x_t = torch.scatter_add(torch.zeros([x_s.size(0), x_s.size(1), num_nodes]), -1, targets, x_s) # in case torch_scatter is not available
        x_t = x_t.scatter_add_(-1, expanded_targets, x_s)
    #if degree is not None:
    #x_t = x_t / (degree.unsqueeze(0).unsqueeze(0)+1e-8)
    #x_t = scatter_mean(x_s, targets, dim=-1, dim_size=num_nodes)
    return x_t



def bidirectional_message_passing(x, edge_index, num_nodes, degree=None, q=None):
    # x: [*, N], edge_index: [2, E]
    if edge_index.size(0) == 0:
        # in case of no edge index
        temp = torch.zeros_like(x)
        return temp, temp
    sources = edge_index[0]
    targets = edge_index[1]
    x_s = x[..., sources]
    x_t = x[..., targets]
    if degree is not None:
        x_s = x_s / (degree[None, None, sources] + 1e-8)
        x_t = x_t / (degree[None, None, targets] + 1e-8)
    if q is not None:
        x_s = torch.exp(1j*2*torch.pi*q) * x_s
        x_t = torch.exp(-1j * 2 * torch.pi * q) * x_t

    try:
        from torch_scatter import scatter_add
        x_st, x_ts = scatter_add(x_s, targets, dim=-1, dim_size=num_nodes), scatter_add(x_t, sources, dim=-1, dim_size=num_nodes)
    except:
        #sources = sources[None][None].tile([x_t.size(0), x_t.size(1), 1])
        #targets = targets[None][None].tile([x_s.size(0), x_s.size(1), 1])
        #x_st = torch.scatter_add(torch.zeros([x_s.size(0), x_s.size(1), num_nodes])*(0*1j), -1, targets, x_s) # in case torch_scatter is not available
        #x_ts = torch.scatter_add(torch.zeros([x_t.size(0), x_t.size(1), num_nodes])*(0*1j), -1, sources, x_t) # in case torch_scatter is not available
        x_st = torch.zeros(*x_s.shape[:-1], num_nodes, dtype=x_s.dtype, device=x_s.device)
        x_ts = torch.zeros(*x_t.shape[:-1], num_nodes, dtype=x_t.dtype, device=x_t.device)
        expanded_targets = targets.unsqueeze(0).unsqueeze(0).expand(x_s.shape[0], x_s.shape[1], -1)
        expanded_sources = sources.unsqueeze(0).unsqueeze(0).expand(x_t.shape[0], x_t.shape[1], -1)
        x_st = x_st.scatter_add_(-1, expanded_targets, x_s)
        x_ts = x_ts.scatter_add_(-1, expanded_sources, x_t)
    #if degree is not None:
    #x_t = x_t / (degree.unsqueeze(0).unsqueeze(0)+1e-8)
    #x_t = scatter_mean(x_s, targets, dim=-1, dim_size=num_nodes)
    return x_st, x_ts
