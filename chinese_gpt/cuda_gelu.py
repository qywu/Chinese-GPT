"""
CUDA implementation for BERT's GELU
"""
import cupy
import torch

# CUDA Kernel
geluForward = '''
extern "C"
__global__ void geluForward(float *input, float *output, const int total)
{
    // for 1-d contiguous arry
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid >= total)
      return;

    output[tid] = input[tid] * 0.5 * (1.0 + erff(input[tid] * 0.7071067811865475));
}
'''

geluBackward = '''
extern "C"
__global__ void geluBackward(float *grad_input, const float* grad_output, float *x, int total)
{
    // for 1-d contiguous arry
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if(tid >= total)
      return;

    grad_input[tid] = (0.5 * (erff(0.707107 * x[tid]) + 1.0) + 
                      0.398942 * powf(2.71828, -0.5 * x[tid] * x[tid]) *
                      x[tid]) * grad_output[tid];
}
'''


# Cupy Helper
class Stream:
    ptr = torch.cuda.current_stream().cuda_stream


@cupy.util.memoize(for_each_device=True)
def cunnex(strFunction):
	return cupy.cuda.compile_with_cache(globals()[strFunction]).get_function(strFunction)


class GELU(torch.autograd.Function):
    """
    The Function is forced to work under fp32
    """
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        assert x.is_contiguous()
        self.saved = x.float()
        output = self.saved.new(*self.saved.shape)
        n = x.numel()
        cunnex('geluForward')(
            grid=((n + 1023) // 1024, 1, 1),
            block=(1024, 1, 1),
            args=[self.saved.data_ptr(), output.data_ptr(), n],
            stream=Stream
        )
        return output.type(x.type())

    def backward(self, grad_output):
        assert grad_output.is_contiguous()
        grad_input = grad_output.new(*grad_output.shape).float()
        n = grad_output.numel()
        
        cunnex('geluBackward')(
            grid=((n + 1023) // 512, 1, 1),
            block=(1024, 1, 1),
            args=[grad_input.data_ptr(), grad_output.float().data_ptr(), self.saved.data_ptr(), n],
            stream=Stream
        )
        return grad_input.type(grad_output.type())  