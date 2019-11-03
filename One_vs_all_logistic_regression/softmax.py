import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################



# wenyan
# regularize = reg / (2*m) * np.sum(np.multiply(theta, theta))
# K = theta.shape[1]

# for i in range(m):
#   for k in range(K):
#     J + = (y[i] == k) * np.log(np.exp(np.dot(theta[:,k], X[i,:]))

# J = -1.0/m *J
# J += regularize





  K = theta.shape[1]
  P = np.zeros([m, K])  
  for k in range(K):
      for i in range(m):
          P[i, k] = np.exp(np.dot(theta[:,k], X[i,:]))
  

 # change to column vector
  P = P / (P.max(axis=1)[:,np.newaxis])
  P = P / (P.sum(axis=1)[:,np.newaxis])
  

  for k in range(K):
      for i in range(m):
          J += -1./m * (y[i] == k) * np.log(P[i, k])
          grad[:, k] += -1./m * (X[i, :] * ((y[i] == k) - P[i, k]))
  
  for k in range(K):
      for d in range(dim):
          J += reg/2./m * theta[d, k]**2
          grad[d, k] += reg/m * theta[d, k]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################
  K = theta.shape[1]
  P = np.exp(X.dot(theta))
  P = P / (P.max(axis=1)[:,np.newaxis])
  P = P / (P.sum(axis=1)[:,np.newaxis])
  I = (y[:,np.newaxis] == np.arange(K))
  J = -1./m * (I * np.log(P)).sum() + reg/2./m * (theta**2).sum()
  grad += -1./m * X.T.dot(I - P) + reg/m * theta

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
