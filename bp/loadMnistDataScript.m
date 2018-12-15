load('../data/mnist.mat');
x_train = reshape(training_data,784,50000);
y_train = training_data_label;

x_valid = reshape(validation_data,784,10000);
y_valid = validation_data_label;

x_test = reshape(test_data,784,10000);
y_test = test_data_label;