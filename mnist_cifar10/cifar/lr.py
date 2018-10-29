import time
import numpy
start=time.time();
trainfeature=numpy.loadtxt('train.txt',dtype='int');
testfeature=numpy.loadtxt('test.txt',dtype='int');
trainlabel=trainfeature[:,0];
testlabel=testfeature[:,0];
numpy.delete(trainfeature,0,axis=0);
numpy.delete(testfeature,0,axis=0);
end=time.time();
print(str(end-start)+'\ts');

num_of_iter=80;
w=[];
b=[];
for i in range(0,10):
    tempw=numpy.zeros(trainfeature.shape[1]);
    tempb=0;
    templabel=[];
    for j in range(0,trainfeature.shape[0]):templabel.append(int(trainlabel[j]==i));
    for j in range(0,num_of_iter):
        if j<20:alpha=0.01;
        elif j<30:alpha=0.001;
        elif j<45:alpha=0.0001;
        elif j<60:alpha=0.00001;
        elif j<80:alpha=0.000001;
        for k in range(0,trainfeature.shape[0]):
            current=trainfeature[k];
            result=numpy.dot(tempw,current)+tempb;
            error=templabel[k]-1/(1+numpy.exp(-result));
            tempw+=alpha*error*current;
            tempb+=alpha*error;
    w.append(tempw);
    b.append(tempb);
end=time.time();
print(str(end-start)+'\ts');

sample=10000;
result=[];
for i in testfeature[:sample]:
    temp=[];
    for j in range(0,10):
        templabel=numpy.dot(w[j],i)+b[j];
        temp.append(1/(1+numpy.exp(-templabel)));
    temp=numpy.array(temp);
    result.append(numpy.argmax(temp));
right=0;
for i in range(0,len(result)):
    if result[i]==testlabel[i]:
        right+=1;
outfile=open('lr.txt','a');
outfile.write(str(num_of_iter)+'\t'+str(right)+'\t'+str(sample)+'\n');
outfile.close();
end=time.time();
print(str(end-start)+'\ts');