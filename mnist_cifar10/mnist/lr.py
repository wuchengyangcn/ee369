import time
import numpy
start=time.time();
files=['train.txt','test.txt'];
infile=open(files[0]);
label=[];
feature=[];
for i in infile:
    temp=i.split();
    label.append(int(i[0]));
    vector=[];
    for j in temp[1:]:
        vector.append(int(j));
    feature.append(vector);
infile.close();
feature=numpy.array(feature);

infile=open(files[1]);
testlabel=[];
testfeature=[];
for i in infile:
    temp=i.split();
    testlabel.append(int(i[0]));
    vector=[];
    for j in temp[1:]:
        vector.append(int(j));
    testfeature.append(vector);
infile.close();
testfeature=numpy.array(testfeature);
end=time.time();
print(str(end-start)+'\ts');

num_of_iter=30;
w=[];
b=[];
for i in range(0,10):
    tempw=numpy.zeros(feature.shape[1]);
    tempb=0;
    templabel=[];
    for j in range(0,feature.shape[0]):templabel.append(int(label[j]==i));
    for j in range(0,num_of_iter):
        if j<15:alpha=0.01;
        elif j<20:alpha=0.001;
        elif j<30:alpha=0.0001;
        for k in range(0,feature.shape[0]):
            current=feature[k];
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