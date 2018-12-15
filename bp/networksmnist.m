% A BP network for MNIST data
% Author: Guo chengfeng
% Version 1.0
% date: 2018-3-15


%神经网络的层次
ntrain = length(y_train);
arch = [784,30,10];
nlayer = length(arch);
mini_batch_size = 100;
max_epochs = 5000;
zeta = 2;
% mini = 20, zeta = 6 eps= 5000. 可以达到0.946
%神经网络参数初始化
weight = cell(1,nlayer);
bias = cell(1,nlayer);
nabla_weight = cell(1,nlayer);
nabla_bias = cell(1,nlayer);
a = cell(1,nlayer);
z = cell(1,nlayer);

for in = 2:nlayer
    weight{in} = rand(arch(in),arch(in-1))-0.5;
    bias{in} = rand(arch(in),1)-0.5;
    nabla_weight{in} = rand(arch(in),arch(in-1));
    nabla_bias{in} = rand(arch(in),1);
end
for in = 1:nlayer
     a{in} = zeros(arch(in),mini_batch_size);
     z{in} = zeros(arch(in),mini_batch_size);
end
accuracy = zeros(1,max_epochs);


for ip = 1:max_epochs
    
    pos = randi(ntrain-mini_batch_size);
    x = x_train(:,pos+1:pos+mini_batch_size);
    y = y_train(:,pos+1:pos+mini_batch_size);
    %正向计算
    a{1} = x;
    [a,z]=feedforward(@acti_relu,@acti_sigmoid,weight,bias,nlayer,mini_batch_size,a,z);
    [weight,bias] = SGD(@acti_relu_prime,@acti_sigmoid_prime,weight,bias,...
        nabla_weight,nabla_bias,nlayer,mini_batch_size,zeta,a,z,y);
    accuracy(ip) = evaluatemnist(@acti_relu,@acti_sigmoid,x_valid,y_valid,weight,bias,nlayer);
    plot(accuracy);
    title(['Accuracy:',num2str(accuracy(ip))]);
    getframe;
end



