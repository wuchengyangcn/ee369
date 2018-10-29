import time
import pickle
import numpy
from PIL import Image
from cyvlfeat import sift,kmeans
start=time.time();
files=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch'];
trainlabels=[];
trainfeatures=[];
for i in range(0,5):
    with open(files[i],'rb') as fo:
        dict=pickle.load(fo,encoding='bytes');
    fo.close();
    for j in dict[b'labels']:
        trainlabels.append(j);
    for j in dict[b'data']:
        trainfeatures.append(j);
trainfeatures=numpy.array(trainfeatures);
trainlabels=numpy.array(trainlabels);

testlabels=[];
testfeatures=[];
with open(files[5],'rb') as fo:
    dict=pickle.load(fo,encoding='bytes');
fo.close();
for j in dict[b'labels']:
    testlabels.append(j);
for j in dict[b'data']:
    testfeatures.append(j);
testfeatures=numpy.array(testfeatures);
testlabels=numpy.array(testlabels);

trainvectors=[];
for i in trainfeatures:
    a=numpy.array(Image.fromarray(i.reshape((32,32,3),order='F'))\
                  .transpose(Image.TRANSPOSE).convert('L'));
    trainvectors.append(sift.sift(a,compute_descriptor='True')[1]);

bag=[];
for i in trainvectors:
    for j in i:
        bag.append(j);
bag=numpy.array(bag);
bag=bag.astype(numpy.float32);
num_of_words=8;
words=numpy.array(kmeans.kmeans(bag,num_centers=num_of_words));
trainwords=[];

testvectors=[];
for i in testfeatures:
    a=numpy.array(Image.fromarray(i.reshape((32,32,3),order='F'))\
                  .transpose(Image.TRANSPOSE).convert('L'));
    testvectors.append(sift.sift(a,compute_descriptor='True')[1]);
testwords=[];

for i in trainvectors:
    result=[0]*num_of_words;
    for k1 in range(0,i.shape[0]):
        target=0;
        distance=numpy.sum(numpy.square(i[k1]-words[0]));
        for k2 in range(1,num_of_words):
            if numpy.sum(numpy.square(i[k1]-words[k2]))<distance:
                distance=numpy.sum(numpy.square(i[k1]-words[k2]));
                target=k2;
        result[target]+=1;
    trainwords.append(result);
trainwords=numpy.array(trainwords);
trainwords=numpy.insert(trainwords,0,trainlabels,axis=1);

for i in testvectors:
    result=[0]*num_of_words;
    for k1 in range(0,i.shape[0]):
        target=0;
        distance=numpy.sum(numpy.square(i[k1]-words[0]));
        for k2 in range(1,num_of_words):
            if numpy.sum(numpy.square(i[k1]-words[k2]))<distance:
                distance=numpy.sum(numpy.square(i[k1]-words[k2]));
                target=k2;
        result[target]+=1;
    testwords.append(result);
testwords=numpy.array(testwords);
testwords=numpy.insert(testwords,0,testlabels,axis=1);

numpy.savetxt('train.txt',trainwords,fmt='%s');
numpy.savetxt('test.txt',testwords,fmt='%s');
end=time.time();
print(str(end-start),'\ts');