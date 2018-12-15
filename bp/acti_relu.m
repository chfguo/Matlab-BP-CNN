function a = acti_relu(x)
%ACTI_RELU f = max(0,x)
%   
a = x;
a(x<0) = 0;
end

