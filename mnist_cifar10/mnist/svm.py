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
feature=numpy.array(feature[:100]);

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

alpha=[];
b=[];
trainlabel=[];
gamma=-0.00127;
margin=0.001;
c=10;
sigma=1e-5;
num_of_iter=5;
kernel=[];
for k1 in range(0,feature.shape[0]):
    temp=[];
    for k2 in range(0,feature.shape[0]):
        temp.append(numpy.exp(numpy.sum(numpy.square(feature[k2]-feature[k1]))*gamma));
    kernel.append(temp);
end=time.time();
print(str(end-start)+'\ts');

for i in range(0,10):
    tempalpha=numpy.zeros(feature.shape[0]);
    tempb=0;
    templabel=[];
    for j in range(0,feature.shape[0]):templabel.append(float(label[j]==i)*2-1);
    error=[];
    for k1 in range(0,feature.shape[0]):
        fx=tempb;
        for k2 in range(0,feature.shape[0]):
            fx+=tempalpha[k2]*templabel[k2]*kernel[k1][k2];
        error.append(fx-templabel[k1]);
    error=numpy.array(error);
    a1=numpy.argmin(error);
    a2=numpy.argmax(error);
    error1=error[a1];
    error2=error[a2];
    alpha1=tempalpha[a1];
    alpha2=tempalpha[a2];
    if templabel[a1]!=templabel[a2]:
        low=max(0,alpha2-alpha1);
        high=min(c,c+alpha2-alpha1);
    else:
        low=max(0,alpha2+alpha1-c);
        high=min(c,alpha1+alpha2);
        if low!=high:
            eta=kernel[a1][a2]*2-2;
            if eta<0:
                temp=tempalpha[a2]-templabel[a2]*(error1-error2)/eta;
                tempalpha[a2]=min(max(temp,low),high);
                if abs(tempalpha[a2]-alpha2)>sigma:
                    tempalpha[a1]+=(alpha2-tempalpha[a2])*\
                    templabel[a1]*templabel[a2];
                    b1=tempb-error1-templabel[a1]*(tempalpha[a1]-alpha1)\
                    -templabel[a2]*(tempalpha[a2]-alpha2)*kernel[a1][a2];
                    b2=tempb-error2-templabel[a2]*(tempalpha[a2]-alpha2)\
                    -templabel[a1]*(tempalpha[a1]-alpha1)*kernel[a1][a2];
                    if tempalpha[a1]>0 and tempalpha[a1]<c:tempb=b1;
                    elif tempalpha[a2]>0 and tempalpha[a2]<c:tempb=b2;
                    else:tempb=0.5*(b1+b2);
    total_iter=0;
    while (total_iter<num_of_iter):
        flag=1;
        for j in range(0,feature.shape[0]):
            fx=tempb;
            for k in range(0,feature.shape[0]):
                fx+=tempalpha[k]*templabel[k]*kernel[j][k];
            error1=fx-templabel[j];
            if (error1*templabel[j]<-margin and tempalpha[j]<c) or\
            (error1*templabel[j]>margin and tempalpha[j]>0):
                a1=j;
                error=[];
                for k1 in range(0,feature.shape[0]):
                    fx=tempb;
                    for k2 in range(0,feature.shape[0]):
                        fx+=tempalpha[k2]*templabel[k2]*kernel[k1][k2];
                    error.append(fx-templabel[k1]);
                maximum=0;
                for k in range(0,feature.shape[0]):
                    if abs(error[k]-error1)>maximum:
                        a2=k;
                        maximum=abs(error[k]-error1);
                error2=error[a2];
                alpha1=tempalpha[a1];
                alpha2=tempalpha[a2];
                if templabel[a1]!=templabel[a2]:
                    low=max(0,alpha2-alpha1);
                    high=min(c,c+alpha2-alpha1);
                else:
                    low=max(0,alpha2+alpha1-c);
                    high=min(c,alpha1+alpha2);
                if low!=high:
                    eta=kernel[a1][a2]*2-2;
                    if eta<0:
                        temp=tempalpha[a2]-templabel[a2]*(error1-error2)/eta;
                        tempalpha[a2]=min(max(temp,low),high);
                        if abs(tempalpha[a2]-alpha2)>sigma:
                            tempalpha[a1]+=(alpha2-tempalpha[a2])*\
                            templabel[a1]*templabel[a2];
                            b1=tempb-error1-templabel[a1]*(tempalpha[a1]-alpha1)\
                                -templabel[a2]*(tempalpha[a2]-alpha2)*kernel[a1][a2];
                            b2=tempb-error2-templabel[a2]*(tempalpha[a2]-alpha2)\
                                -templabel[a1]*(tempalpha[a1]-alpha1)*kernel[a1][a2];
                            if tempalpha[a1]>0 and tempalpha[a1]<c:tempb=b1;
                            elif tempalpha[a2]>0 and tempalpha[a2]<c:tempb=b2;
                            else:tempb=0.5*(b1+b2);
                            flag=0;
        if flag:total_iter+=1;
    end=time.time();
    print(str(end-start)+'\ts');
    alpha.append(tempalpha);
    b.append(tempb);
    trainlabel.append(templabel);

sample=1000;
result=[];
for i in testfeature[:sample]:
    temp=[];
    for j in range(0,10):
        temp.append(b[j]);
        for k in range(0,feature.shape[0]):
            temp[j]+=alpha[j][k]*trainlabel[j][k]*numpy.exp(numpy.sum(\
                       numpy.square(i-feature[k]))*gamma);
    result.append(numpy.argmax(temp));
right=0;
for i in range(0,len(result)):
    if result[i]==testlabel[i]:
        right+=1;
outfile=open('svm.txt','a');
outfile.write(str(gamma)+'\t'+str(margin)+'\t'+str(c)+'\t'+
              str(right)+'\t'+str(sample)+'\n');
outfile.close();
end=time.time();
print(str(end-start)+'\ts');