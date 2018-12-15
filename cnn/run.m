cnn.eta = 1;
cnn.lambda = 5;

ntrain = size(training_data_label,2);
mini_batch_size = 100;
cnn.ntrain = ntrain;
accuracy = zeros(1,50);
cnn.layer = {
    % input layer: 'input', mini_size, [height,width] of image
    {'input',mini_batch_size,[28,28]};
    % convlution layer: 'conv', kernel_number, [height,width] of kernel
    {'conv',20,[9,9]}; 
    % pooling layer: 'pool', pooling_type, [height,width] of pooling area
    {'pool','mean',[2,2]};
    % flatten layer: 'flat', a layer for pre-allocated memory
    {'flat'};
    % full connect layer: 'full', neuron number
    {'full',100};
    {'full',100};
    % output layer: 'output', neuron number
    {'output',10};
    };

tic
cnn = cnn_initialize(cnn);
max_iter = 20000;
ik = 0;

disp('采用有放回的训练方法')
disp(['每次计算随机选取 ',num2str(mini_batch_size), ' 个样本进行计算'])
disp(['共计算 ',num2str(max_iter),' 次,相当于全部数据迭代 ', ...
num2str(max_iter/(size(training_data,3)/mini_batch_size)),'次'])



for in = 1:max_iter
    pos = randi(ntrain-mini_batch_size);
    x = training_data(:,:,pos+1:pos+mini_batch_size);
    y = training_data_label(:,pos+1:pos+mini_batch_size);
    cnn = cnn_feedforward(cnn,x);
    cnn = cnn_backpropagation2(cnn,y);
    if mod(in,100) == 0
        disp('迭代次数：',num2str(in));
    end
    if mod(in,1000) == 0
        ik = ik + 1;
        accuracy(ik) = cnn_evaluate(cnn,validation_data,validation_data_label)*100;
        disp(['validtion accuracy: ',num2str(accuracy(ik)), '%']);
    end
end  

toc

plot(accuracy,'r','LineWidth',1.5)
grid on
title(['1+2网络结构，测试数据最高识别率为: ，',num2str(max(accuracy)),'%']);
xlabel('迭代次数:  X1000');
ylabel('精度');

cnn_evaluate(cnn,test_data,test_data_label)