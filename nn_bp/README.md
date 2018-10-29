# contents for each file
|file|content|
|-|-|
|train-images.idx3-ubyte|train image|
|train-labels.idx1-ubyte|train label|
|t10k-images.idx3-ubyte|test image|
|t10k-labels.idx1-ubyte|test label|
|bp.py|defines the *Back Propagation Neural Network* model, including loading data, training the network, testing data and recording results|
|bp.txt|saves training time and testing accuracy|

------

# definition for class **nn** in bp.py 
    nn(hidden,epoch=500,batchsize=500)
**hidden** denotes the structure for hidden layers. For example:

    network=nn([80]);
denotes one hidden layer with 80 nodes.

    network=nn([40,60]);
denotes two hidden layers each with 40 and 60 nodes.

    network=nn([90,50,70]);

denotes three hidden layers each with 90, 50, and 70 nodes.

------
# recording results
    epoch:500 batchsize:500 time:69.36s accuracy:96.26%
|element|description|
|-|-|
|epoch|iterations for training|
|batchsize|data size for each epoch|
|time|training time|
|accuracy|testing accuracy|