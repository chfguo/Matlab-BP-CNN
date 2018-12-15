function c = gzeros(varargin)
%GZEROS return zeros array on GPU
%   
try
    c = gpuArray.zeros(varargin{:});
catch
    c = zeros(varargin{:});
end

end

