function [a,z] = bp_feedforward(hidenActiFcn,outputActiFcn,weight,bias,nlayer,a,z)
%BP_FEEDFORWARD 此处显示有关此函数的摘要
%   此处显示详细说明
for in = 2:nlayer-1
    w = weight{in};
    b = bias{in};
    ix = a{in-1};
    %小技巧，
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

