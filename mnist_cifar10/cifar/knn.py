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

sample=10000;
for threshold in range(14,15,1):
    result=[];
    for i in testfeature[:sample]:
        distance=[];
        for j in trainfeature:distance.append(numpy.sum(numpy.square(i-j)));
        vote=[0]*10;
        knn=numpy.argpartition(distance,threshold)[:threshold];
        for j in knn:
            vote[trainlabel[j]]+=1;
        result.append(numpy.argmax(vote));
    right=0;
    for i in range(0,len(result)):
        if result[i]==testlabel[i]:
            right+=1;
    outfile=open('knn.txt','a');
    outfile.write(str(threshold)+'\t'+str(right)+'\t'+str(sample)+'\n');
    outfile.close();
    end=time.time();
    print(str(end-start)+'\ts');