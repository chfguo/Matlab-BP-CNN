function cnn = cnn_feedforward(cnn,x)
%CNN_FEEDFORWARD CNN feedforward
%   
num = numel(cnn.layer);
for in = 1:num

switch cnn.layer{in}{1}
    case 'input'
        cnn.a{in} = x;
     case 'conv'
         kernel_num = cnn.layer{in}{2};
         for ik = 1:kernel_num
             cnn.z{in}(:,:,:,ik) = convn(cnn.a{in-1},...
                 cnn.weights{in}(:,:,ik),'valid')+cnn.biases{in}(ik);
         end
         cnn.a{in} = relu(cnn.z{in});
    
     case 'pool'
         
         ker_h = cnn.layer{in}{3}(1);
         ker_w = cnn.layer{in}{3}(2);
         kernel = ones(ker_h,ker_w)/ker_h/ker_w;
         
         tmp = convn(cnn.a{in-1},kernel,'valid');
         cnn.a{in} = tmp(1:ker_h:end,1:ker_w:end,:,:);

     case 'flat'
        [height,width,mini_size,kernel_num] = size(cnn.a{in-1});
        for ik = 1:mini_size
            cnn.a{in}(:,ik) = reshape(cnn.a{in-1}(:,:,ik,:),[height*width*kernel_num,1]);
        end
     case 'full'
         cnn.z{in}= bsxfun(@plus,cnn.weights{in}*cnn.a{in-1},cnn.biases{in});
         cnn.a{in} = sigmoid(cnn.z{in});
     case 'output'
         cnn.z{in}= bsxfun(@plus,cnn.weights{in}*cnn.a{in-1},cnn.biases{in});
         cnn.a{in} = softmax(cnn.z{in});
end

end

end

