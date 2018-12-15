function a = acti_relu_prime(x)
%ACTI_RELU_PRIME 0 for x < 0, 1 for >= 0
%   
a = zeros(size(x));
a(x>0) = 1;

end

