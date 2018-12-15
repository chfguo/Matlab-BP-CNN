function d = sigmoid_prime(z)
% sigmoid激活函数的导数
f = sigmoid(z);
d = f.*(1-f);
end

