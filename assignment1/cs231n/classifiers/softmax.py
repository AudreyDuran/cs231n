import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classe = W.shape[1]
  loss = 0.0

  for i in range(num_train): #pour chaque image de l'ensemble d'entrainement
    scores = X[i].dot(W)
    scores -= max(scores)

    correct_class_score = scores[y[i]] #y[i]=c
    e_syi = np.exp(correct_class_score)
    e_sj = np.sum(np.exp(scores))

    loss -= np.log(e_syi/e_sj)

    for k in range(num_classe): #pour chaque classe
      dW[:, k] += ((np.exp(scores[k])/e_sj) - (k == y[i])) * X[i].T

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  dim = dW.shape[0]
  num_classe = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W)
  # ajouter le - max a la fin
  indexes=np.arange(num_train)
  #c = correct class score
  c = S[indexes, y]

  e_syi = np.exp(c)
  e_sj = np.sum(np.exp(S), axis=1)
  Li = - np.log(e_syi/e_sj)
  loss = np.sum(Li) / num_train + reg * np.sum(W * W)


  M = np.exp(S)/(np.repeat(e_sj, num_classe).reshape(num_train, num_classe)) #(500,10)
  M[indexes, y] -= 1 #bonnes classes
  dW = X.T.dot(M)

  dW = dW/num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

