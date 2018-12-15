function showdigit(x,y,index)
    if length(size(x))==3
        digit = squeeze(x(:,:,index));
    else
        digit = reshape(x(:,index),28,28);
    end
    digit = 1-digit;
    [nz,nx] = size(y);
    if nz == 1 || nx == 1
        num = y(index);
    else
        [~,num] = max(y(:,index));
        num = num-1;
    end

    imshow(digit);
    title(['This is ',num2str(num)]);
end