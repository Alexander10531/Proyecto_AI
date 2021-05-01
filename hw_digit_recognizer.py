import numpy as np
import csv 

class HWDigitRecognizer:

  def __init__(self, train_filename, test_filename):
    self.obtainData(train_filename, test_filename)

  def obtainData(self, train_filename, test_filename):
    trainFile = open(train_filename, 'r')
    trainFile = csv.reader(trainFile, delimiter=',')
    trainTest = open(test_filename, 'r')
    trainTest = csv.reader(trainTest, delimiter=',')
    yTrain = [] 
    yTest = []
    xTrain = []
    xTest = []

    for row in trainFile:
      if row[0] != 'label':        
        yTrain.append(self.createVector(row[0]))        
        del(row[0])
        row = np.asarray(row)
        xTrain.append(np.asarray(row.astype(int))/255)

    for row in trainTest:
      if row[0] != 'label':
        yTest.append(self.createVector(row[0]))
        del(row[0])
        row = np.asarray(row)
        xTest.append(np.asarray(row.astype(int))/255)
    
    xTrain = np.asarray(xTrain).T
    xTest = np.asarray(xTest).T
    yTrain = np.asarray(yTrain).T
    yTest = np.asarray(yTest).T
    
    self.X_train = xTrain
    self.X_test = xTest
    self.Y_train = yTrain
    self.Y_test = yTest

  def createVector(self, value):
    value = int(value)
    vectorLabel = np.zeros(10)
    vectorLabel[value] = 1
    return vectorLabel

  def get_datasets(self):

    d = { "X_train": self.X_train,
    "X_test": self.X_test,
    "Y_train": self.Y_train,
    "Y_test": self.Y_test
    }
    return d

  def train_model(self):

    costs = []
    num_iterations = 3000
    learning_rate = 0.05
    derivate_fs = self.get_datasets()
    xTrain = derivate_fs['X_train']
    yTrain = derivate_fs['Y_train']
    layer = [xTrain.shape[0], 100, 100, 10]
    parms = self.initializate_fParameters(layer)
    
    for i in range(0, num_iterations):
      AL, caches = self.lForward(parms, xTrain)
      cost = self.compute_cost(AL, yTrain)
      grads = self.L_model_backward(AL, yTrain, caches)
      parms = self.update_parameters(grads, parms, learning_rate)

      if i % 100 == 0:
        print("Cost after iteration %i: %f" % (i, cost))
      if i % 100 == 0:
        costs.append(cost)

    predictions = self.predict_model(derivate_fs["X_test"], self.Y_test, parms)
    params_dict = {}
    params_dict["model_params"] = parms
    params_dict["layer_dims"] = layer
    params_dict["learning_rate"] = learning_rate
    params_dict["num_iterations"] = num_iterations
    params_dict["costs"] = costs

    file_name = 'all_params.dict'
    out_file = open(file_name, 'wb')
    pickle.dump(params_dict, out_file)
    out_file.close()
    return parms, costs

  def L_model_backward(self, AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dZ = AL - Y
    dAL = -(Y/AL)+(1-Y)/(1-AL)
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = \
    self.linear_activation_backward(dAL, current_cache, "softmax", dZ)

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = \
        self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, "relu", dZ)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

  def initializate_fParameters(self, layer_dims):
    np.random.seed(5)
    parameters = {}

    for l in range(1, len(layer_dims)):
      parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
      parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
      assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
      assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
    
    return parameters

  def propagationForward(X, parameters):

    cache = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches.append(cache)

    A_prev = A
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    AL, cache = linear_activation_forward(A_prev, W, b, "softmax")
    caches.append(cache)
    assert(AL.shape == (W.shape[0], X.shape[1]))

    return AL, caches

  def relu(self, Z):
    A = np.maximum(0, Z)
    if(A.shape == Z.shape):
      return A
  
  def lForward(self, params, X):
    A = X 
    caches = []

    for l in range(1, len(params) // 2):
      b = params['b' + str(l)]
      w = params['W' + str(l)]
      A, cache = self.exec_activation_function(A, 'relu', w, b)
      caches.append(cache)

    w = params["W" + str(len(params) // 2)]
    b = params["b" + str(len(params) // 2)]
    AL, cache = self.exec_activation_function(A, 'softmax', w, b)
    caches.append(cache)

    return AL, caches

  def exec_activation_function(self, A_prev, activation, w, b):
    
    Z = w.dot(A_prev) + b
    if activation == "softmax":
      A = self.softmax(Z)
    else:
      A = self.relu(Z)

    cache = (Z, A_prev, w)
    return A, cache

  def relu(self, Z):
    A = np.maximum(0, Z)
    return A

  def softmax(self, x):

    e = np.exp(x-np.max(x))
    soft = e/e.sum(axis=0, keepdims=True)
    return soft

  def compute_cost(self, AL, Y):

    m = Y.shape[1]

    cost = np.mean(Y*np.log(AL + np.exp(-8)))
    cost = np.squeeze(cost)
    return cost

  def linear_activation_backward(self, dA, cache, activation, dZ):
    Z, A_prev, W = cache

    if activation == "relu":
      dZ = np.array(dA, copy=True)
      dZ[Z <= 0] = 0
      dA_prev, dW, db = self.linear_backward(dZ, cache)
    elif activation == "softmax":
        s = self.softmax(Z)
        dA_prev, dW, db = self.linear_backward(dZ, cache)

    return dA_prev, dW, db

  def linear_backward(self, dZ, cache):

    Z, A_prev, W = cache
    m = A_prev.shape[1]

    dW = 1/m * dZ.dot(A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = (W.T).dot(dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)

    return dA_prev, dW, db

  def update_parameters(self, grads, parameters, learning_rate):

    L = len(parameters) // 2

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters
  
  def predict_model(self, X, y, parameters):

    m = X.shape[1]
    n = len(parameters) // 2 
    p = np.zeros((1, m))

    probas, caches = self.lForward(parameters, X)

    predictions = probas.T
    for i in range(0, predictions.shape[0]):
        arr = predictions[i]
        mx = self.get_max(arr)
        p[0, i] = mx[0]

    if len(y) != 0:
        print("Accuracy: " + str(np.sum((p == y)/m)))

    return p

  def get_max(self, a):
    mx = np.max(a)
    idx = np.where(mx == a)
    return idx