import numpy as np
from scipy.stats import mode

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        dists[i, j] = np.sqrt(np.sum((X[i]-self.X_train[j])**2))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      dists[i,:] = np.sqrt(np.sum((self.X_train - X[i])**2,axis=1))   
      # ,: 이러한 방식으로 저장되니까 self.X_train의 원소하나하나 X[i]랑 빼져서
      # 하나씩 저장된다.
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists
    #print(dists.shape) => 500,5000

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #print(self.X_train.shape) => (5000, 3072) #Train
    #print(X.shape) #=> (500, 3072) #Test
    #print(self.X_train.T.shape) => 3072,5000
    test_sum = np.sum(np.square(X), axis=1)
                                       #row 방향으로 더한다.
    #print(test_sum.shape) # (500,)
    train_sum = np.sum(np.square(self.X_train), axis=1)
                                       #row 방향으로 더한다.
    #print(train_sum.shape) # (5000,)
    #np.dot이라는 것이 있다. 이것은 로와 행을 곱함. 즉 행렬의 곱셈연산임
    multi = np.dot(X, self.X_train.T) # (500,3072)X(3072,5000)
    #print(inner.shape) # (500,5000)
                    #test_sum은 500, 이다.
    dists = np.sqrt(test_sum.reshape(num_test,1) + train_sum -2 * multi)
                    #500,1                    5000,           500,5000
    # 500, 과 500,5000 끼리는 broadcast가 되지 않는다. 그래서 500,을 500,1로 reshape을 해준 것  
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
      closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
      #np.argsort는 큰 순서대로 인덱스로 순위를 매겨준다.
      #dists는 (500,5000) 임   
      '''
      >>> a=np.array([[2,4,8,1],[2,3,1,2]])
      >>> np.argsort(a)
          크기 순으로 sort하는데 배열의 인덱스를 표시해준다.
          array([[3, 0, 1, 2],
                 [2, 0, 3, 1]])
      '''
      indexs = np.argsort(dists[i])
      indexs = indexs[:k]
      # distance가 가장 작은 순으로 k만큼 일단 뽑는다.  
      [closest_y.append(self.y_train[idx]) for idx in indexs]  
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################
      #print('>',closest_y)
      # scipy.stats.mode => 빈도수가 높은 원소를 뽑아주고 빈도수가 같은 경우 작은 원소를 뽑아준다.
      # return은 원소 array 와 count array
      #print(mode(closest_y))
      #k개 만큼 뽑아놓은 거리가 가장 작은 것들 중 빈도수가 높은 것을 뽑아서 y_pred 값으로 지정한다.
      y_pred[i] = mode(closest_y)[0][0]
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred
