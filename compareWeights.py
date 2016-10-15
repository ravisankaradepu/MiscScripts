// Python code to compare weights from two different CNN's, to check the CNN's when trained with same random seed
import caffe
net1 = caffe.Net('examples/mnist/lenet_train_test.prototxt','examples/mnist/lenet1_iter_1.caffemodel',caffe.TEST)
net1.params['conv1'][0].data
net2 = caffe.Net('examples/mnist/lenet_train_test.prototxt','examples/mnist/lenet2_iter_1.caffemodel',caffe.TEST)
net1 == net2
