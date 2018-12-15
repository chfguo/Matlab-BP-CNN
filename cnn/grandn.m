function c = grandn(varargin)
%GRANDN return randn() arrays on GPU
%   
try
    c = gpuArray.randn(varargin{:});
catch
    c = randn(varargin{:});
end

end

