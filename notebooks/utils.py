import random
import torch


def iter_batches(batch_size, train_in, train_out, shuffle=False):
    if shuffle:
        pass
    
    cur = 0
    batch_in, batch_out = train_in[cur:cur+batch_size], train_out[cur:cur+batch_size]
    yield batch_in, batch_out
    
    while batch_in.shape[0] == batch_size:
        cur += batch_size
        batch_in, batch_out = train_in[cur:cur+batch_size], train_out[cur:cur+batch_size]
        if batch_in.numel() > 0:
            yield batch_in, batch_out
            
            
def merge_and_shuffle(true_X, fake_X):
    true_len, fake_len = true_X.shape[0], fake_X.shape[0]
    Y = torch.tensor([1]*true_len + [0]*fake_len)

    shuffle_inds = list(range(true_len+fake_len))
    random.shuffle(shuffle_inds)
    
    both = torch.cat([true_X, fake_X])
    return both[shuffle_inds], Y[shuffle_inds]