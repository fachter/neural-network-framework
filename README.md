# Neural Network Framework

A project done at the university JMU Würzburg by [fachter](https://github.com/fachter) together with [marja-w](https://github.com/marja-w) and [mamadfrhi](https://github.com/mamadfrhi)

The package is being used by the remaining implementation of that project in another repository 
called [gesture-detection-neural-network](https://github.com/fachter/gesture-detection-neural-network)
and further usage of this package can be seen in the implementation

### Project by Felix Achter, Marja Wahl, Mohammad Farrahi
This repository is designated to creating, training, and using a neural network. There are five main tasks you can
tackle:
- **Data Preprocessing**: get the data ready for being trained on
- **Creating the Network**: create your network by using information of your training data
- **Training the Network**: train the resulting network by optionally choosing number of iterations, learning rate, and 
optionally optimizer
- **Evaluation**: evaluate the trained network with metrics like accuracy and F1 score
- **Predicting**: use the final network for predicting on unseen data

## Data Preprocessing
### Scaling
 	You don't have to scale your data in advance. 
    Creating a new NeuralNetwork instance, the Standard Scaler is set as the default scaler.
    You can read more about creating a NeuralNetwork instance in section "Creating the Network".
    
The [feature_scaling.py](netneural/network/feature_scaling.py) file provides two classes for scaling your data: the StandardScaler and
the NormalScaler class. You can scale your data, before instantiating a neural network, by creating an instance of either
class, fitting the scaler to your data, and then transform your data. The data can be transformed back by using the
inverse_transform() function.

```
from netneural.network.feature_scaling import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(X)  # X being the training data
X_scaled = standard_scaler.transform(X)
X = standard_scaler.inverse_transform(X_scaled)  # returns original X
``` 

Don't forget to set the right scaler, if you did not use the default StandardScaler, when creating a NeuralNetwork 
instance:

```
from netneural.network.feature_scaling import NormalScaler

normal_scaler = NormalScaler()
normal_scaler.fit(X)  # X being the training data
X_scaled = normal_scaler.transform(X)

nn = NeuralNetwork(shape, scaler='normal')
```

### Encoding
    You don't have to encode your target values in advance.
    If no encoder is handed over when instantiating a NeuralNetwork, the data is encoded, if necessary.
    You can read more about creating a NeuralNetwork instance in section "Creating the Network".

The [one_hot_encoder.py](netneural/network/one_hot_encoder.py) can be used for creating your own OneHotEncoder instance and encoding
multiclass labels into one hot encoded label lists.

```
from netneural.network.one_hot_encoder import OneHotEncoder

encoder = OneHotEncoder()
y_one_hot = encoder.encode(y)  # y being the target values with multiple labels
unique_labels = encoder.unique_labels  # get unique labels found by encoder
```

Don't forget to pass the created encoder to a NeuralNetwork instance, if you encode the labels in advance:

```
nn = NeuralNetwork(shape, encoder=encoder)
```

### Training Test Split
In order to properly analyse your network's training, you will need to split the data into a training and a test set.
You can do that by the provided function `train_test_split()`in [data_loader.py](netneural/session/data_loader.py):

```
# take only the first four outputs, because no validation set is produced
X_train, y_train, X_test, y_test = train_test_split(X, y, train_per=0.8, test_per=0.2, randomized=True)[:4]
```

You can also divide your data into training, test, and validation set:

```
X_train, y_train, X_test, y_test, X_val, y_val = train_test_split(X, y, train_per=0.8, test_per=0.1, valid_per=0.1)
```

### Principal Component Analysis (PCA)
Additionally, you can perform PCA using methods provided in [pca.py](netneural/pca/pca.py). 

If you do not know anything about PCA you can simply reduce your
feature count by specifying how much variance of data you want to cover with the remaining features:

```
from netneural.pca.pca import PCA

pca = PCA()
X_pca = pca.pca(X, var_per=0.99)
```

If you want to directly specify the number of desired features, you can also do so:

```
X_pca_10 = pca.pca(X, number_pcs=10)
```

You can also specify both, where `number_pcs` would act like a maximum for the returned constructed features:

```
X_pca = pca.pca(X, var_per=0.99, number_pcs=10)  # returns 10 principal components, if more are needed for covering 99% of variance
```

You can also directly work with eigenvectors and eigenvalues:

```
eigenvectors, eigenvalues = pca.get_eigenvectors_and_values(X)

# gets matrix with 'rank' of every attribute in every principal component
ordered_attributes_matrix = pca.analyze_eigenvectors(eigenvectors, attributes)  # attributes is list of original attributes

# gets a list of the original attributes, ordered by its position in each eigenvector
ordered_attributes = pca.compute_order_attributes(ordered_attributes_matrix)

# returns the number of principal components needed for covering the input percentage
n_pc = pca.explained_variance(eigenvalues, 0.99, plots=True)  # if plots is True, plots variance covered per n principal components

X_pca = pca.get_n_dimensions(X, n_pc, eigenvectors)
```

## Creating the Network

For creating the network, you need to create a new instance of the NeuralNetwork class, provided in [nn.py](netneural/network/nn.py).
You have to set several parameters when instantiating.

- **Activation Function**: you can choose between sigmoid, tanh, and relu activation
- **Architecture**: you will need to provide the number of neurons per layer
- **Weigths**: you don't have to specifiy weights, but you can do so by passing a list of lists, each containing weights
for each layer

```
from netneural.network.nn import NeuralNetwork

shape = (input_features, 2, 2, output_classes)  # variables store the number of input features/output classes
nn = NeuralNetwork(shape, activation_function='sigmoid')  # uses randomly assigned weights
```

If you want to train a neural network for regression, you will have to set the `regression` parameter to true:

```
nn = NeuralNetwork(shape, regression=True)
```

## Training the Network
Training the network requires several hyperparameters you have to set. The `train()` function of the NeuralNetwork class
takes severable variables in order to define them:

- **Iterations**: choose how many epochs the neural network will train for
- **Learning Rate**: decide on the learning rate for training
- **Test Data**: you have the choice of handing over a test data set in order to receive feedback about the evaluation of
the network on this data set 
- **Optimizer**: you can choose between the default gradient descent or the Adam optimizer
- **Cost function**: the network decides on the cost function on its own, whether your last layer has one or multiple nodes,
it uses the Mean Squared Error function or Cross Entropy (with Softmax)
- **Batch Size**: if a batch size is set, the model trains batches of the training data of this size every iteration, it 
updates its weights after training each batch, and it sees each batch during one iteration. Each iteration, the training 
examples are randomly assigned to a batch. If the batch size is not given, the model will normally train on all the training
data per one iteration
- **Plotting**: if the `plots` variable is set to True, the function will plot the error, accuracy, and F1 score on training
and eventually test data set after training
- **Encode**: if you did not encode your data yet (multiclass classification) and did not hand over an encoder during the
creation of the neural network, you need to set this parameter to true, if your multiclass labels are not encoded yet

The function returns the training history on either the training data set or the test data set, which can later be used
for analyzing and visualizing.

```
f1_history, acc_history, err_history = nn.train(X, y, 100, 0.1)  # trains with default settings (optimizer=gradient descent, no test data, no plots)
```
```
f1_history, acc_history, err_history = nn.train(X, y, 100, 0.1, X_test, y_test, optimizer='adam', batch_size=500, plots=True)  # trains while evaluating on test set, uses Adam optimizer, uses batches, plots results
```

For the format of data sets applies:
- the data matrix (X) should be formatted data points x features
- the target values (y) should be formatted class x data points (one row for binary classification)

When training a regression neural network, the only metric output afterwards is the history of the Mean Squared Error (MSE):

```
error_history = nn.train(X, y, 100, 0.1)
```

### Training Sessions
The functions provided in the [session_util.py](netneural/session/nn_session_util.py) can be used for saving and loading a trained network.
Along with the network the function `save_session()` also takes other parameters, like the unique labels, PCA object, 
training history, training data, iterations, and learning rate, in order to easily reenact the documented session.

```
from netneural.session.nn_session_util import save_session

learning_rate = 0.1
iterations = 100
f1_history, acc_history, err_history = nn.train(X, y, iterations, learning_rate)  # we train a neural network

# save_session stores a .pkl file with a name consisting of current time and the final f1 score
save_session(nn, nn.encoder.unique_labels, pca, f1_history[-1])
```

After the file is stored, it can be used to again load the neural network:

```
from netneural.session.nn_session_util import load_from_config

config_file = "config_90_f1_2022-03-25 14-56-20.json"  # path to the config file
nn, pca, f1_history, learning_rate, X_train, y_train, X_test, y_test = load_session_from_config(config_file)
```

You can also use these functions for a regression neural network and if you did not use PCA:

```
save_session(nn, learning_rate, X, y, X_test, y_test, iterations, err_history)  # saves file in 'config_file'

# returns None for values that weren't set, like pca and f1_history
nn, pca = load_from_config(config_file)
```

## Evaluation
### Metrics
For evaluating a trained model intrinsic metrics can be calculated using the methods provided in [metrics.py](netneural/network/metrics.py).
These include accuracy and the F1 score, which essentially compare the predictions of the model `h` to the ground truth `y`.
The output differs, depending if binary or multiclass classification is demanded.

```
from netneural.network.metrics import get_accuracy, get_f1_score

accuracy_per_class, accuracy_mc = get_accuracy(h_multiclass, y_multiclass)  # additionally returns accuracy per class
accuracy = get_accuracy(h, y)  

f1_score_per_class, f1_score_mc = get_f1_score(h_multiclass, y_multiclass)
f1_score = get_f1_score(h, y)
```

For the format of the input values it is required that the rows stand for the number of classes.

The error can be calculated using the `get_error()` function of the NeuralNetwork class:

```
errors = nn.get_error(h,y)  # returns error for every data point in a list
error = errors.mean()  # final error
```

When using a regression neural network, you don't have to take the mean over the error. We use the MSE, which will already
output the mean over the errors:

```
error = nn.get_error(h,y)  # for nn with regression=True
```

## Plotting
For making plotting a little bit easier, two functions are introduced in the [plot_lib.py](netneural/network/plot_lib.py) script. One that 
plots a line and one function that creates a scatter plot.

```
from netneural.network.plot_lib import plot_line, plot_scatter

plot_line(list_to_plot, x_label, y_label, title)  # plots a line
plot_scatter(x_data, y_data, colors, x_label, y_label, title):  # creates scatter plot
```

## Predicting
Finally, we can use the NeuralNetwork instance for predicting values. This can be done using the `predict()` provided in
the NeuralNetwork class:

```
predictions = nn.predict(input_values)
```

The input values of this function need to be a `numpy array`, also if it is only one value.

