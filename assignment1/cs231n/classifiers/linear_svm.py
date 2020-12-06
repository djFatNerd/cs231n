from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    # compute the loss and the gradient
    num_classes = W.shape[1] # C
    num_train = X.shape[0]   # N
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]: # correct class itself, skip
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

                # ************************
                # this step is the key, very important, check derivation
                # analytical gradient  
                dW[:, j] += X[i].T
                dW[:, y[i]] -= X[i].T
 
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # ***************
    # gradient also need to be divide by num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather than first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # regularization term also contributes to the gradient dW
    dW += 2 * reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    # take the y'th element at each row, which is the correct score for training example X'i
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)
    # convert correct_class_score to a column vector for subtraction
    # correct_class_scores = correct_class_scores.reshape(-1, 1)

    # calculate margins -> if > 0, keep, else set to 0
    margins = np.maximum(0, scores - correct_class_scores + 1)

    # set correct class loss to 0
    # margins[np.arange(num_train), y] = 0

    # overall loss is the summation of the margins
    loss = np.sum(margins)

    # note we don't want to include loss for the correct class, for each example we contributed 1 to the total loss
    # so subtracted by 1 * N at the end will eliminate this effect
    loss -= num_train

    # divide by num_train
    loss /= num_train

    # add the regularization term
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # ******** unresolved
    # ***************************************************************
    # ***************************************************************
    # I have no idea how to do this
    # gradient compute

    # compute input mask         N X C
    valid_margin_mask = np.zeros(margins.shape)

    # set a positive mask if margin is ppsitive
    valid_margin_mask[margins > 0] = 1
    # subtract in correct class (-s_y) aka valid_margin_count
    valid_margin_mask[np.arange(num_train), y] -= np.sum(valid_margin_mask, axis = 1)
    # ? ? ?
    # scores = X.dot(W), and valid_margin_mask is a function of margin,
    # which is a function of scores
    # we get dW by multiplying X.T and valid_margin_mask
    dW = X.T.dot(valid_margin_mask)
    
    dW /= num_train

    # add regularization gradient
    dW += 2 * reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW