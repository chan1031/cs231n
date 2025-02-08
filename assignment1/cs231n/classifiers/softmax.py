from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]

    for i in range(num_train):
      scores = X[i].dot(W) 
      '''
      넘겨져 오는 X 값은 X_dev로 (500,3073)으로 구성되어있음
      그럼 지중 [i]번째 행렬을 가져오므로 [1,3073]의 행렬과 [3073,10]의 행렬을 행렬곱함
      그렇게 되면 scores는 [1,10]의 행렬이됨
      '''
      scores -= np.max(scores)
      #exp연산을 하기 전에 값이 너무 큰 경우 overflow문제가 발생할 수 있으므로 최대값과 빼줌
      #이렇게 되면 score 값은 0이하의 값을 가지게 되므로 overflow가 발생하지 않음
      
      #소프트 맥스 확률을 계산
      exp_scores = np.exp(scores)
      softmax_probs = exp_scores / np.sum(exp_scores)  # Softmax probabilities
      '''
      이는 softmax 공식을 사용해서 각 클래스에 속할 확률을 구하는 코드임
      exp_scroes는 scores값에 exp를 적용한 것임
      예를들면 exp_scoers = [e^1.2, e^3.2, e^6.2] = [0.123,23.12, 25.23]이런식으로 구성이 된다.
      그러면 softmax_porbs는 exp_scores 와 exp_scores를 구한 값으로 나누어 주므로
      [0.5,0.2,0.3]이런식의 확률 값으로 표현이 됨
      '''

      #손실값 계산
      loss += -np.log(softmax_probs[y[i]]) #크로스엔트로피 손실을 구한다. 이때 -log을 쓰기 때문에 확률 값이 높게 나오면 손실값이 적어지는 형태이다.

      #그라디언트 계산
      for j in range(num_class):  # Loop over classes
        if j == y[i]:
          dW[:, j] += (softmax_probs[j] - 1) * X[i]
        else:
          dW[:, j] += softmax_probs[j] * X[i]
      '''
      dw 계산
      그라디언트의 계산 결과는 손실함수 L을 W로 미분한 결과값이다.
      이는 직접 계산해보면 위에처럼 정답 클래스, 정답 클래스가 아닌 경우로 구분되어진다.
      '''
    
    # Average over batch 
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


#벡터화된 버전은 반복문을 사용하지 않고 행렬 연산으로만 구하는 것을 의미함
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    # Step 1: Compute class scores (N, C)
    scores = X.dot(W)  # (N, D) x (D, C) = (N, C)
    
    # Step 2: Numeric stability trick (subtract max for each row)
    scores -= np.max(scores, axis=1, keepdims=True)  #axis=1이면 행 방향으로의 최대값을 계산한다. keepdims는 원본의 차원을 동일하게 유지할지를 의미한다.
    #이렇게 되면 각 행에서 최대값을 구해서 빼주게 된다.

    # Step 3: Compute softmax probabilities
    exp_scores = np.exp(scores)  # (N, C)
    softmax_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

    # Step 4: Compute loss (Cross-Entropy Loss)
    N = X.shape[0]  # 전체 이미지의 개수
    correct_class_probs = softmax_probs[np.arange(N), y]  # 정답 클래스의 확률들만 가져오게 만듬
    '''
    np.arrange(N)은 0~N까지의 인덱스를 생성하는 함수이다.
    즉, softmax_probs[[0,1,2,3...N], [N까지의 정답 클래스]]이므로 2차원 행렬 형태가 될 것이다.
    '''
    loss = -np.sum(np.log(correct_class_probs)) / N  # Compute average loss

    # Step 5: Compute gradient
    softmax_probs[np.arange(N), y] -= 1  #정답 클래스일 경우 1을 빼줌
    dW = X.T.dot(softmax_probs) / N  
    '''
    Naive 버전의 그라디언트 계산을
    vector화된 버전으로 바꾼것이다.
    '''

    # Step 6: Regularization
    loss += reg * np.sum(W * W)  # L2 Regularization term
    dW += 2 * reg * W  # Regularization gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

