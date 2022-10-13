from turtle import forward

from sklearn.preprocessing import scale
from cloudnet.cuda import gnp
from cloudnet.func import *



class Conv2D():

    def __init__(self,f,kernel_size,stride,padding):
        self.type = "conv2d"
        self.f = f
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


        # 反向傳播時用
        self.x = None
        self.c_w=None
        self.c_=None
        self.x_shape = None

    def init(self,shape,dtype):
        self.shape = shape
        self.filters = gnp.random.randn(*shape)
        self.filters = self.filters.reshape(1,self.filters.shape[0],self.filters.shape[1],self.filters.shape[2])


    def forward(self,input):
        n,c,h,w = input.shape
        f,c,kh,kw = self.filters.shape
        

        out_h = int((h + 2*self.padding - kh)/self.stride) + 1
        out_w = int((w + 2*self.padding - kw)/self.stride) + 1

        col = im2col(input,kh,kw,self.stride,self.padding)
        col_w = self.filters.reshape(f,-1).T

        #TODO 加入bias
        out = gnp.dot(col,col_w) 
        out = out.reshape(n,out_h,out_w,-1).transpose(0,3,1,2) 

        self.x = input
        self.c_w = col_w
        self.c = col
        self.x_shape = input.shape

        return out

    def backward(self,dout):


        f,c,kh,kw = self.filters.shape
        
        dout = dout.transpose(0,2,3,1).reshape(-1,f)

        self.db = gnp.sum(dout,axis=0)
        self.dw = gnp.dot(self.c_.T,dout)
        self.dw = self.dw.transpose(1,0).reshape(f,c,kh,kw)

        dcol = gnp.dot(dout,self.c_w.T)
        dx = col2im(dcol,self.x_shape[2],self.x_shape[3],self.stride,self.padding)

        return dx



class MaxPool2D():

    def __init__(self,pool_h,pool_w,stride=1,padding=0):
        self.type = "maxpool2d"
        self.h = pool_h
        self.w = pool_w
        self.stride = stride
        self.padding = padding


    def init(self,*a):
        pass

    def forward(self,input):
        n,c,h,w = input.shape
        out_h = (h - self.h)/self.stride + 1
        out_w = (w - self.w)/self.stride + 1
        col = im2col(input,self.h,self.w,self.stride,self.padding)
        col = col.reshape(-1,self.h*self.w)
        argmax = gnp.argmax(col,axis=1)
        out = gnp.max(col,axis=1)
        out = out.reshape(n,out_h,out_w,c).transpose(0,3,1,2)
        self.x = input
        self.argmax = argmax
        return out

    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)
        dmax = gnp.zeros((dout.size, self.h * self.w))
        dmax[gnp.arange(self.argmax.size), self.argmax.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape+(self.h * self.w))
        dcol = dmax.reshape(dmax.shape[0]*dmax.shape[1]*dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.h, self.w,self.stride,self.padding)
        return dx

class flatten():
        
    def __init__(self):
        self.type = "flatten"
        self.x = None

    def init(self,*a):
        pass

    def forward(self,input):
        self.x = input
        return input.reshape(input.shape[0],-1)

    def backward(self,dout):
        return dout.reshape(self.x.shape)


class Dense():
        
    def __init__(self,f):
        self.type = "dense"
        self.f = f

        self.x = None
        self.weight = None

    def init(self,shape,dtype):
        self.shape = shape
        #TODO 下面這行完全不確定
        #TODO line133
        self.weight = gnp.random.randn(*shape)

    def forward(self,input):
        self.x = input
        return gnp.dot(input,self.weight)

    def backward(self,dout):
        dx = gnp.dot(dout,self.weight.T)
        self.dw = gnp.dot(self.x.T,dout)
        self.db = gnp.sum(dout,axis=0)
        self.x = None
        return dx
                


                








        






            

