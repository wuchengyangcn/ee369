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

sample=10000;
for threshold in range(5,6,1):
    result=[];
    for i in testfeature[:sample]:
        distance=[];
        for j in feature:
            distance.append(numpy.sum(numpy.square(i-j)));
        vote=[0]*10;
        knn=numpy.argpartition(distance,threshold)[:threshold];
        for j in knn:
            vote[label[j]]+=1;
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