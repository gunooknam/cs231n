import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero
            #결과는 float형으로 채워진다.
  # compute the loss and the gradient
  num_classes = W.shape[1] #10
  num_train = X.shape[0]   #500
  loss = 0.0
  #print('shape',W.shape, X.shape)
  # W = 3073,10
  # X = 500,3073
  # XW = (500,10) 나온다. 각 (1,10)은 각각의 class 들의 score
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]] #정답의 점수
    for j in range(num_classes):
      if j == y[i]: # 정답의 인덱스와 현재 점수를 뽑으려는 인덱스가 같으면 계산안함
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1  그렇지 않으면 정답이 아닌 스코어 - 정답 스코어 + 1(safety margin) 더해져서  
      if margin > 0:
        loss += margin # 마진이 0이상이면 더해진다. 
        dW[:, j] += X[i]
        dW[:, y[i]] += -X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train # 마지막에 평균 내려고
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW = dW/num_train + reg * W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  #scores.shape 500,10
  correct_class_score = scores[np.arange(num_train), list(y)]
                                                    # y가 0~9니까 이걸 리스트로 만들고 그걸 인덱스로  넣으니까 scores의 행에서 인덱스가 넣어져서 하나 뽑힌다. 대신에 list y는 score의 column과 일치해야함
  # y.shape = (500, )
  # column으로 num_train은 500개 뽑고
  
  #correct_class_score 이 500,이니까 이걸 reshape
  #score가 500,10이니까 correct_class_score이 broadcast
  margins = np.maximum(0,scores - correct_class_score.reshape(-1,1) + 1)
  margins[np.arange(num_train), y] = 0
  #correct_class_score부븐 0으로 처리
  loss = np.sum(margins)
 
  loss /= num_train #평균
  loss += 0.5 * reg * np.sum(W*W) #L2 Regularization
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  X_mask = np.zeros((num_train, num_classes))
  X_mask[margins > 0] = 1
  X_mask[range(num_train), list(y)] = 0
  X_mask[range(num_train), list(y)] = -np.sum(X_mask, axis=1)  
  
  dW = (X.T).dot(X_mask)
  dW = dW/num_train + reg*W
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
