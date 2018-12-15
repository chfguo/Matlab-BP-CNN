function [a,z] = feedforward(hidenActiFcn,outputActiFcn,weight,bias,nlayer,mini_batch_size,a,z)
%FEEDFORWARD Return the output of the network
%

for in = 2:nlayer-1
    w = weight{in};
    b = bias{in};
    ix = a{in-1};
    %Ð¡¼¼ÇÉ£¬
    iz = bsxfun(@plus,w*ix,b);
    a{in} = hidenActiFcn(iz);
    z{in} = iz;
end

w = weight{nlayer};
b = bias{nlayer};
ix = a{nlayer-1};
iz = bsxfun(@plus,w*ix,b);
a{nlayer} = outputActiFcn(iz);
z{nlayer} = iz;

end

