from typing import Tuple
import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = (  # shape: (batch_size, seqlen, n_local_heads, head_dim // 2)
        query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    )
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)   # shape: (batch_size, seqlen, n_local_kv_heads, head_dim // 2)
    
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # First, compute the trigonometric values in the second and fourth columns in
    # slide 22 (linked above).
    theta_tensor = torch.arange(0, head_dim, 2.0, device=device) / head_dim
    theta_tensor = torch.pow(theta, -theta_tensor)
    freqs_cis = torch.zeros((seqlen, head_dim // 2, 2), device=device) 
    seqidx = torch.arange(seqlen, device=device) 
    seq_theta = torch.outer(seqidx, theta_tensor)  # shape: (seqlen, head_dim // 2)

    freqs_cis[:, :, 0] = torch.cos(seq_theta)
    freqs_cis[:, :, 1] = torch.sin(seq_theta)
    
    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    query_real = query_real.transpose(1, 2) # shape: (batch_size, n_local_heads, seqlen, head_dim // 2)
    query_imag = query_imag.transpose(1, 2) # shape: (batch_size, n_local_heads, seqlen, head_dim // 2)
    key_real = key_real.transpose(1, 2) # shape: (batch_size, n_local_kv_heads, seqlen, head_dim // 2)
    key_imag = key_imag.transpose(1, 2) # shape: (batch_size, n_local_kv_heads, seqlen, head_dim // 2) 

    q_item1, q_item2 = (
        query_real * freqs_cis[:, :, 0] - query_imag * freqs_cis[:, :, 1],
        query_real * freqs_cis[:, :, 1] + query_imag * freqs_cis[:, :, 0],
    )
    k_item1, k_item2 = (
        key_real * freqs_cis[:, :, 0] - key_imag * freqs_cis[:, :, 1],
        key_real * freqs_cis[:, :, 1] + key_imag * freqs_cis[:, :, 0],
    )

    # Finally, reshape the results back to the original shape.
    query_out = torch.stack((q_item1, q_item2), dim=-1).reshape(query.shape)
    key_out = torch.stack((k_item1, k_item2), dim=-1).reshape(key.shape)

    query_out = query_out.transpose(1, 2)
    key_out = key_out.transpose(1, 2)
    
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
