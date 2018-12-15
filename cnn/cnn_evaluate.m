function error = cnn_evaluate(cnn,x_valid,y_valid)
%CNN_FEEDFORWARD CNN feedforward
%   
num = floor(numel(y_valid)/cnn.mini_size);
y = zeros(1,cnn.mini_size*num);

for in = 1:num
cnn = cnn_feedforward(cnn,x_valid(:,:,(in-1)*cnn.mini_size+1:cnn.mini_size*in));   
[~,yp] = max(cnn.a{end});
yp = yp-1;
y((in-1)*cnn.mini_size+1:cnn.mini_size*in) = gather(yp);
error = sum(y == y_valid(1:cnn.mini_size*num))/(cnn.mini_size*num);
end

