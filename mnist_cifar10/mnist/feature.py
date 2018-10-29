import struct
import numpy
import time
start=time.time();
files=['train-images.idx3-ubyte','train-labels.idx1-ubyte',\
       't10k-images.idx3-ubyte','t10k-labels.idx1-ubyte'];
#data=pickle.load(open(files[0],'rb'));
data=open(files[0],'rb');
temp=data.read();
head=struct.unpack_from('>iiii',temp,0);
index=struct.calcsize('>iiii');
num=head[1];
col=head[2];
row=head[3];
images=struct.unpack_from('>'+str(num*col*row)+'B',temp,index);
images=numpy.reshape(images,[num,col*row]);
data.close();
data=open(files[1],'rb');
temp=data.read();
head=struct.unpack_from('>ii',temp,0);
index=struct.calcsize('>ii');
labels=struct.unpack_from('>'+str(num)+'B',temp,index);
labels=numpy.reshape(labels,[num]);
data.close();
outfile=open('train.txt','w');
for i in range(0,num):
    outfile.write(str(labels[i])+'\t');
    for j in images[i]:
        outfile.write(str(int(j>200))+'\t');
    outfile.write('\n');
outfile.close();
end=time.time();
print(str(end-start)+'\ts');

data=open(files[2],'rb');
temp=data.read();
head=struct.unpack_from('>iiii',temp,0);
index=struct.calcsize('>iiii');
tnum=head[1];
tcol=head[2];
trow=head[3];
timages=struct.unpack_from('>'+str(tnum*tcol*trow)+'B',temp,index);
timages=numpy.reshape(timages,[tnum,tcol*trow]);
data.close();
data=open(files[3],'rb');
temp=data.read();
head=struct.unpack_from('>ii',temp,0);
index=struct.calcsize('>ii');
tlabels=struct.unpack_from('>'+str(tnum)+'B',temp,index);
tlabels=numpy.reshape(tlabels,[tnum]);
data.close();
outfile=open('test.txt','w');
for i in range(0,tnum):
    outfile.write(str(tlabels[i])+'\t');
    for j in timages[i]:
        outfile.write(str(int(j>200))+'\t');
    outfile.write('\n');
outfile.close();
end=time.time();
print(str(end-start)+'\ts');