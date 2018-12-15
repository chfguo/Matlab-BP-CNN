function c = copyToGPU(x)
%COPYTOGPU opies the numeric data X to the GPU.
%   
try 
    c = gpuArray(x);
catch 
    c = x;
end
end

