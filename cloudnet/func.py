from cloudnet.cuda import gnp






def im2col(input,fh,fw,stride=1,padding=0):
    n,c,h,w = input.shape

    out_h = (h + 2*padding - fh)//stride + 1
    out_w = (w + 2*padding - fw)//stride + 1

    img = gnp.pad(input, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    print((n,c,fh,fw,out_h,out_w))
    col = gnp.zeros((n,c,fh,fw,out_h,out_w))

    for y in range(fh):
        y_max = y + stride*out_h
        for x in range(fh):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    return col.transpose(0, 4, 5, 1, 2, 3).reshape(n*out_h*out_w, -1)



def col2im(col,input_shape,fh,fw,stride=1,padding=0):
    n,c,h,w = input_shape
    out_h = (h + 2*padding - fh)//stride + 1
    out_w = (w + 2*padding - fw)//stride + 1

    img = gnp.zeros((n,c,h,w))
    col = col.reshape(n,out_h,out_w,c,fh,fw).transpose(0, 3, 4, 5, 1, 2)

    for y in range(fh):
        y_max = y + stride*out_h
        for x in range(fw):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
    
    return img[:, :, padding:h + padding, padding:w + padding]




def cross_entropy(y,t):
    return -gnp.sum(gnp.log(y[gnp.arange(y.shape[0]),t] + 1e-6))/y.shape[0]


