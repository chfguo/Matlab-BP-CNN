function a = softmax(z)
%SOFTMAX exp(z)/sum(exp(z))
%   

f = exp(z);
s = sum(f);
a = zeros(size(z));
num = size(f,2);
for in = 1:num
    a(:,in) = f(:,in)./s(in);
end

end

