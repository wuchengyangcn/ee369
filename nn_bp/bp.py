import struct
import numpy
import time
import random
class nn:
    train=[];
    trainlabel=[];
    test=[];
    testlabel=[];
    #number of nodes for each layer
    layers=[];
    w=[];
    b=[];
    epoch=0;
    batchsize=0;
    def __init__(self,hidden,epoch=500,batchsize=500):
        self.epoch=epoch;
        self.batchsize=batchsize;
        data=['train','t10k'];
        self.train,self.trainlabel=self.load(data[0]);
        self.test,self.testlabel=self.load(data[1]);
        self.layers.append(numpy.size(self.train,1));
        #add hidden layers
        for temp in hidden:self.layers.append(temp);
        self.layers.append(numpy.max(self.trainlabel)\
                                    -numpy.min(self.trainlabel)+1);
        #initialize each w and b
        for temp in range(1,len(self.layers)):
            self.w.append(0.01*numpy.random.randn(\
                self.layers[temp-1],self.layers[temp]));
            self.b.append(numpy.zeros(self.layers[temp]));
        self.fit(self.epoch,self.batchsize);
        
    def load(self,filename):
        images,num=self.loadimage(filename+'-images.idx3-ubyte');
        labels=self.loadlabel(filename+'-labels.idx1-ubyte',num);
        image=numpy.array(images>0).astype(int);
        return image,labels;
        
    def loadimage(self,filename):
        data=open(filename,'rb');
        temp=data.read();
        head=struct.unpack_from('>iiii',temp,0);
        index=struct.calcsize('>iiii');
        num=head[1];
        col=head[2];
        row=head[3];
        images=struct.unpack_from('>'+str(num*col*row)+'B',temp,index);
        data.close(); 
        return numpy.reshape(images,[num,col*row]),num;
    
    def loadlabel(self,filename,num):
        data=open(filename,'rb');
        temp=data.read();
        struct.unpack_from('>ii',temp,0);
        index=struct.calcsize('>ii');
        labels=struct.unpack_from('>'+str(num)+'B',temp,index);
        data.close();
        return numpy.reshape(labels,[num]);

    def update(self,batch):
        dx=[];
        db=[];
        for temp in range(0,len(self.layers)-1):
            dx.append(numpy.zeros(numpy.shape(self.w[temp])));
            db.append(numpy.zeros(numpy.shape(self.b[temp])));
        for temp in batch:
            outputs=self.output(temp);
            error=self.loss(outputs[-1],temp);
            error*=self.dsigmoid(outputs[-1]);
            cache=error.copy();
            d=[error];
            for layer in range(len(self.layers)-2,0,-1):
                error=numpy.dot(error,self.w[layer].T)\
                *self.dsigmoid(outputs[layer]);
                d.insert(0,error);
            for layer in range(0,len(self.layers)-1):
                self.w[layer]-=self.lrate(cache)*outputs[layer][:,None]\
                *d[layer][:,None].T;
                self.b[layer]-=self.lrate(cache)*d[layer];
    
    def output(self,sample):
        data=self.train[sample];
        outputs=[data];
        for layer in range(0,len(self.layers)-1):
            data=self.sigmoid(numpy.dot(data, self.w[layer])+self.b[layer]);
            outputs.append(data);
        return outputs;

    def loss(self,result,sample):
        error=[];
        for temp in result:error.append(temp);
        error[self.trainlabel[sample]]-=1;
        return error;

    def sigmoid(self,data):
        temp=numpy.array(data);
        return 1/(1+numpy.exp(-temp));
    
    def dsigmoid(self,data):
        temp=numpy.array(data);
        return (1-temp)*temp;

    def lrate(self,cache):
        loss=numpy.sum(cache*cache);
        return 0.5*(numpy.sqrt(loss)+1e-5);
    
    def fit(self,epoch,batchsize):
        start=time.time();
        for temp in range(0,epoch):
            batch=random.sample(range(len(self.trainlabel)),batchsize);
            self.update(batch);
        end=time.time();
        print('train time: ',end-start,'s');
        temp=self.result();
        accuracy=str(round(temp*100,2))+'%';
        total=str(round((end-start),2))+'s';
        outfile=open('bp.txt','a');
        outfile.write('epoch:'+str(self.epoch)+' ');
        outfile.write('batchsize:'+str(self.batchsize)+' ');
        outfile.write('time:'+total+' ');
        outfile.write('accuracy:'+accuracy+'\n');
        outfile.close();
            
    def forward(self,data):
        for temp in range(0,len(self.layers)-1):
            data=self.sigmoid(numpy.dot(data,self.w[temp])+self.b[temp]);
        return data;
    
    def result(self,number=10000):
        right=0.0;
        for temp in range(0,number):
            result=self.forward(self.test[temp]);
            if(numpy.argmax(result)==self.testlabel[temp]):right+=1;
        print('accuracy: ',right/number);
        return right/number;

network=nn([80]);
