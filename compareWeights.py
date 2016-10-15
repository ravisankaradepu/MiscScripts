// Python code to compare weights from two different CNN's, to check the CNN's when trained with same random seed
// net1 should be initialized to 0 else it goes into infinite loop
import caffe
net1 = caffe.Net('examples/mnist/lenet_train_test.prototxt','examples/mnist/lenet1_iter_5.caffemodel',caffe.TEST)
a=net1.params['conv1'][0].data
net1=0
net1 = caffe.Net('examples/mnist/lenet_train_test.prototxt','examples/mnist/lenet2_iter_5.caffemodel',caffe.TEST)
b=net1.params['conv1'][0].data
a==b
