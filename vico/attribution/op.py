import torch

def batch_min_max_matrix_multiplication(A, B):
    """
    Compute the element-wise minimum across the last dimension of tensors A and B,
    and then take the maximum over the k dimension.

    Args:
        A (torch.Tensor): Tensor of shape [B, m, k].
        B (torch.Tensor): Tensor of shape [B, k, n].

    Returns:
        torch.Tensor: Tensor of shape [B, m, n] containing the maximum values.
    """
    
    # Expand A and B to [B, m, 1, k] and [B, 1, n, k] respectively to prepare for broadcasting
    A_expanded = A.unsqueeze(2)  # Shape becomes [B, m, 1, k]
    B_expanded = B.permute(0, 2, 1).unsqueeze(1)  # Shape becomes [B, 1, n, k]
    
    # Compute the element-wise minimum across the last dimension (k)
    min_vals = torch.min(A_expanded, B_expanded)  # Shape becomes [B, m, n, k]
    
    # Take the maximum over the k dimension (dimension 3)
    max_vals = torch.max(min_vals, dim=3).values  # Shape becomes [B, m, n]
    
    return max_vals

def batch_soft_min_max_matrix_multiplication(A, B, temperature=0.001):
    # Inputs:
    # A: tensor of shape [B, m, k]
    # B: tensor of shape [B, k, n]
    
    # Expand A and B to [B, m, 1, k] and [B, 1, n, k] respectively to prepare for broadcasting
    A_expanded = A.unsqueeze(2)  # Shape becomes [B, m, 1, k]
    B_expanded = B.permute(0, 2, 1).unsqueeze(1)  # Shape becomes [B, 1, n, k]
    
    size = A_expanded.shape[1], B_expanded.shape[2], A_expanded.shape[3]
    # expand A and B to [m, n, k]
    A_expanded = A_expanded.expand(-1, *size)
    B_expanded = B_expanded.expand(-1, *size)
    
    # Compute the soft element-wise minimum across the last dimension (k)
    neg_A = -A_expanded / temperature
    neg_B = -B_expanded / temperature
    soft_min_vals = -torch.logsumexp(torch.stack([neg_A, neg_B]), dim=0) * temperature  # Shape becomes [B, m, n, k]
    
    # Compute the soft element-wise maximum using logsumexp
    soft_max_vals = torch.logsumexp(soft_min_vals / temperature, dim=3) * temperature  # Shape becomes [B, m, n]
    
    return soft_max_vals