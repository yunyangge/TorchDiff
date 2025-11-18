
import torch
from einops import rearrange

test_num = 64
samples = torch.arange(0, 64)
samples = samples.unsqueeze(0).unsqueeze(-1)

def skiparse_1d(x, group=False):
    if not group:
        return rearrange(x, 'b (n p) c -> (b p) n c', p=4)
    else:
        return rearrange(x, 'b (n p q) c -> (b p) (n q) c', p=4, q=4)

def reverse_skiparse_1d(x, group=False):
    if not group:
        return rearrange(x, '(b p) n c -> b (n p) c', p=4)
    else:
        return rearrange(x, '(b p) (n q) c -> b (n p q) c', p=4, q=4)
    

def skiparse_2d(x, group=False):
    if not group:
        return rearrange(x, 'b (t h p w q) c -> (b p q) (t h w) c', p=2, q=2, t=1, h=4, w=4)
    else:
        return rearrange(x, 'b (t h p1 p2 w q1 q2) c -> (b p1 q1) (t h p2 w q2) c', p1=2, q1=2, p2=2, q2=2, t=1, h=2, w=2)
    
def reverse_skiparse_2d(x, group=False):
    if not group:
        return rearrange(x, '(b p q) (t h w) c -> b (t h p w q) c', p=2, q=2, t=1, h=4, w=4)
    else:
        return rearrange(x, '(b p1 q1) (t h p2 w q2) c -> b (t h p1 p2 w q1 q2) c', p1=2, q1=2, p2=2, q2=2, t=1, h=2, w=2)
    
# skiparse_1d_samples = skiparse_1d(samples)
# reverse_skiparse_1d_samples = reverse_skiparse_1d(skiparse_1d_samples)
# print(samples)
# print(skiparse_1d_samples)
# print(reverse_skiparse_1d_samples)
    
# skiparse_1d_samples = skiparse_1d(samples, group=True)
# reverse_skiparse_1d_samples = reverse_skiparse_1d(skiparse_1d_samples, group=True)
# print(samples)
# print(skiparse_1d_samples)
# print(reverse_skiparse_1d_samples)

# skiparse_2d_samples = skiparse_2d(samples)
# reverse_skiparse_2d_samples = reverse_skiparse_2d(skiparse_2d_samples)
# print(samples)
# print(skiparse_2d_samples)
# print(reverse_skiparse_2d_samples)

skiparse_2d_samples = skiparse_2d(samples, group=True)
reverse_skiparse_2d_samples = reverse_skiparse_2d(skiparse_2d_samples, group=True)
print(samples)
print(skiparse_2d_samples)
print(reverse_skiparse_2d_samples)
