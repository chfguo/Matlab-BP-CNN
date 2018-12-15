function c = gones(varargin)
%GONES return ones array on GPU
%  
try
    c = gpuArray.ones(varargin{:});
catch
    c = ones(varargin{:});
end

end

