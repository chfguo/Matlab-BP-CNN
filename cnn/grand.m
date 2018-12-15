function c = grand(varargin)
%GRAND return rand() arrays on GPU
%   
try
    c = gpuArray.rand(varargin{:});
catch
    c = rand(varargin{:});
end


end

