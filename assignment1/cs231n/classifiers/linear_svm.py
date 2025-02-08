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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1] #(3073,10)이므로 10을 가져옴
    num_train = X.shape[0] #49000을 가져옴
    loss = 0.0
    #반복문을 통해 X 데이터를 하나씩 계산하는 방법을 사용
    for i in range(num_train):
        scores = X[i].dot(W) #w와x를 행렬곱연산
        '''
        X와 w를 행렬곱하면 49000x10의 행렬이 나오게 된다.
        그럼 이중 실제 정답 클래스의 점수 값을 알고자 하는 것이
        아래 코드이다.
        '''
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
              loss += margin
              dW[:,j] += X[i]
              dW[:,y[i]] -= X[i]
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dW /= num_train
    dW += 2*reg*W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0] #입력 이미지 X의 개수를 구함
    num_classes = W.shape[1] #클래스의 개수를 구함
    scores = X.dot(W) # X*W의 값을 구함

    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1) #구한 scores에서 실제 정답 클래스의 점수가 얼마인지를 계산
    '''
    .reshape(-1,1) 설명:
    우선 reshape는 배열의 모양을 바꿔주는 역할을 한다.
    reshape(n,m)은 n열 m행으로 바꾸라는 뜻이다.
    여기서 -1인 자동으로 알아서 뒤의 행에 맞게 설정하라는 뜻이기에
    reshape(-1,1)은 1행 짜리 배열을 2차원 배열 형식으로 바꾸는 뜻이다.
    ex) [1,2,3] -> [1,] 이런식으로 변환함
                   [2,]
                   [3,]
    '''
    
    #softmax 손실함수 계산을 위해 마진을 계산한다.
    margins = np.maximum(0, scores - correct_class_scores + 1)
    margins[np.arange(num_train), y] = 0  # 정답 레이블은 신경쓰지 않으므로 0으로 설정
    '''
    이러면 margin은 (N,C)의 형태를 가진다. (데이터의 개수, 클래스의 개수)
    ex) [[1,2,3]
        [1,2,3]] 이미지 1개당 계산된 각 클래스별 score값을 가진다.
    '''
    

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

    #손실함수 계산
    loss = np.sum(margins) / num_train # margin값들을 구한다음 N으로 나누어 평균 손실값 계산
    loss += reg * np.sum(W * W)  # 정규화 reg 추가
    print(loss)
    #기울기 값 계산
    binary = margins > 0  # binary는 (N,C) 크기의 margin이 0보다 큰지 아닌지를 구분하는 T/F mask를 생성하게 된다
    #binary도 N,C의 크기를 가지게 된다.
    row_sum = np.sum(binary, axis=1) # 이제 각 행렬값에서 margin이 0보다 큰 score 값의 개수를 구하게 된다.
    #그러면 row_sum = [1,2,3] 이런식으로 구성이 된다. 이는 1번째 행은 0보다 큰게 1개, 2번째 행은 2개... 이런 의미를 가진다.
    #즉, 각 샘플별 0보다 큰 score 값을 가지는 클래스의 개수를 의미한다.
    binary[np.arange(num_train), y] = -row_sum  # 정답 클래스는 기울기를 조정해야하므로 뺴준다.

    dW = X.T.dot(binary) / num_train  # Shape: (D, C)
    dW += 2 * reg * W  # Add regularization gradient


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
