import torch

def walk_profile_check_spd(wp, spd, walk_length):
    wp_flat = torch.cat([w for w in wp], dim=0)
    N = spd.shape[0]
    for i in range(N):
        for j in range(N):
            if spd[i, j] > walk_length:
                assert torch.linalg.norm(wp_flat[:, i, j]) < 1e-8
            if torch.linalg.norm(wp_flat[:, i, j]) < 1e-8:
                assert spd[i, j] > walk_length

def walk_profile_check_reachablility(wp, node_pair_idx):
    wp_flat = torch.cat([w for w in wp], dim=0)
    for idx in node_pair_idx.transpose(0, 1):
        i, j = idx[0], idx[1]
        assert torch.linalg.norm(wp_flat[:, i, j]) >= 1
