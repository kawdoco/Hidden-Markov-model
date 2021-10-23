# Class HiddenMarkovModel

**HiddenMarkovModel implements hidden Markov models with Gaussian mixtures as**
**distributions on top of TensorFlow 2.0**

### Installation

```
pip install --upgrade git+https://gitlab.com/kesmarag/hmm-gmm-tf2
```

```python
HiddenMarkovModel(p0, tp, em_w, em_mu, em_var)
```

```
Args:
  p0: 1D numpy array
    Determines the probability of the first hidden variable
    in the Markov chain for each hidden state.
    e.g. np.array([0.5, 0.25, 0.25]) (3 hidden states)
  tp: 2D numpy array
    Determines the transition probabilities for moving from one hidden state to each
    other. The (i,j) element of the matrix denotes the probability of
    transiting from i-th state to the j-th state.
    e.g. np.array([[0.80, 0.15, 0.05],
                   [0.20, 0.55, 0.25],
                   [0.15, 0.15, 0.70]])
    (3 hidden states)
  em_w: 2D numpy array
    Contains the weights of the Gaussian mixtures.
    Each line correspond to a hidden state.
    e.g. np.array([[0.8, 0.2],
                   [0.5, 0.5],
                   [0.1, 0.9]])
    (3 hidden states, 2 Gaussian mixtures)
  em_mu: 3D numpy array
    Determines the mean value vector for each component
    of the emission distributions.
    The first dimension refers to the hidden states whereas the
    second one refer to the mixtures.
    e.g. np.array([[[2.2, 1.3], [1.2, 0.2]],    1st hidden state
                   [[1.3, 5.0], [4.3, -2.3]],   2nd hidden state
                   [[0.0, 1.2], [0.4, -2.0]]])  3rd hidden state
    (3 hidden states, 2 Gaussian mixtures)
  em_var: 3D numpy array
    Determines the variance vector for each component of the
    emission distributions.
    e.g. np.array([[[2.2, 1.3], [1.2, 0.2]],    1st hidden state
                    [[1.3, 5.0], [4.3, -2.3]],   2nd hidden state
                    [[0.0, 1.2], [0.4, -2.0]]])  3rd hidden state
    (3 hidden states, 2 Gaussian mixtures)
```


## log_posterior
```python
HiddenMarkovModel.log_posterior(self, data)
```
```
Log probability density function.

Args:
  data: 3D numpy array
    The first dimension refers to each component of the batch.
    The second dimension refers to each specific time interval.
    The third dimension refers to the values of the observed data.

Returns:
  1D numpy array with the values of the log-probability function with respect to the observations.
```

## viterbi_algorithm
```python
HiddenMarkovModel.viterbi_algorithm(self, data)
```
```
Performs the viterbi algorithm for calculating the most probable
hidden state path of some batch data.

Args:
  data: 3D numpy array
    The first dimension refers to each component of the batch.
    The second dimension refers to each specific time interval.
    The third dimension refers to the values of the observed data.

Returns:
  2D numpy array with the most probable hidden state paths.
    The first dimension refers to each component of the batch.
    The second dimension the order of the hidden states.
    (0, 1, ..., K-1), where K is the total number of hidden states.
```

## fit
```python
HiddenMarkovModel.fit(self, data, max_iter=100, min_var=0.01, verbose=False)
```
```
This method re-adapts the model parameters with respect to a batch of
observations, using the Expectation-Maximization (E-M) algorithm.

Args:
  data: 3D numpy array
    The first dimension refers to each component of the batch.
    The second dimension refers to each specific time step.
    The third dimension refers to the values of the observed data.
  max_iter: positive integer number
    The maximum number of iterations.
  min_var: non-negative real value
    The minimum acceptance variance. We use this restriction
    in order to prevent overfitting of the model.

Returns:
  1D numpy array with the log-posterior probability densities for each training iteration.
```

## generate
```python
HiddenMarkovModel.generate(self, length, num_series=1, p=0.2)
```
```
Generates a batch of time series using an importance sampling like approach.

Args:
  length: positive integer
    The length of each time series.
  num_series: positive integer (default 1)
    The number of the time series.
  p: real value between 0.0 and 1.0 (default 0.2)
    The importance sampling parameter.
    At each iteration:
  k[A] Draw X and calculate p(X)
      if p(X) > p(X_{q-1}) then
        accept X as X_q
      else
        draw r from [0,1] using the uniform distribution.
        if r > p then
          accept the best of the rejected ones.
        else
          go to [A]

Returns:
  3D numpy array with the drawn time series.
  2D numpy array with the corresponding hidden states.
```

## kl_divergence
```python
HiddenMarkovModel.kl_divergence(self, other, data)
```
```
Estimates the value of the Kullback-Leibler divergence (KLD)
between the model and another model with respect to some data.
```


## Example

```python
import numpy as np
from kesmarag.hmm import HiddenMarkovModel, new_left_to_right_hmm, store_hmm, restore_hmm, toy_example
```


```python
dataset = toy_example()
```

This helper function creates a test dataset with a single two dimensional time series with 700 samples.

    The first 200 samples corresponds to a Gaussian mixture with 

        w1 = 0.6, w2=0.4
        mu1 = [0.5, 1], mu2 = [2, 1]
        var1 = [1, 1], var2=[1.2, 1]

    the next 300 corresponds to a Gaussian mixture with

        w1 = 0.6, w2=0.4
        mu1 = [2, 5], mu2 = [4, 5]
        var1 = [0.8, 1], var2=[0.8, 1]

    and the last 200 corresponds to a Gaussian mixture with

        w1 = 0.6, w2=0.4
        mu1 = [4, 1], mu2 = [6, 5]
        var1 = [1, 1], var2=[0.8, 1.2]


```python
print(dataset.shape)
```

    (1, 700, 2)


```python
model = new_left_to_right_hmm(states=3, mixtures=2, data=dataset)
```


```python
model.fit(dataset, verbose=True)
```

    epoch:   0 , ln[p(X|λ)] = -3094.3748904062295
    epoch:   1 , ln[p(X|λ)] = -2391.3602228316568
    epoch:   2 , ln[p(X|λ)] = -2320.1563724302564
    epoch:   3 , ln[p(X|λ)] = -2284.996645965759
    epoch:   4 , ln[p(X|λ)] = -2269.0055909790053
    epoch:   5 , ln[p(X|λ)] = -2266.1395773469876
    epoch:   6 , ln[p(X|λ)] = -2264.4267494952455
    epoch:   7 , ln[p(X|λ)] = -2263.156612481979
    epoch:   8 , ln[p(X|λ)] = -2262.2725752851293
    epoch:   9 , ln[p(X|λ)] = -2261.612564557431
    epoch:  10 , ln[p(X|λ)] = -2261.102826808333
    epoch:  11 , ln[p(X|λ)] = -2260.7189908960695
    epoch:  12 , ln[p(X|λ)] = -2260.437608729253
    epoch:  13 , ln[p(X|λ)] = -2260.231860238426
    epoch:  14 , ln[p(X|λ)] = -2260.0784163526014
    epoch:  15 , ln[p(X|λ)] = -2259.960659542152
    epoch:  16 , ln[p(X|λ)] = -2259.8679640963023
    epoch:  17 , ln[p(X|λ)] = -2259.793721328861
    epoch:  18 , ln[p(X|λ)] = -2259.733658260372
    epoch:  19 , ln[p(X|λ)] = -2259.684791553708
    epoch:  20 , ln[p(X|λ)] = -2259.6448728507144
    epoch:  21 , ln[p(X|λ)] = -2259.6121181368353
    epoch:  22 , ln[p(X|λ)] = -2259.5850765029527





    [-3094.3748904062295,
     -2391.3602228316568,
     -2320.1563724302564,
     -2284.996645965759,
     -2269.0055909790053,
     -2266.1395773469876,
     -2264.4267494952455,
     -2263.156612481979,
     -2262.2725752851293,
     -2261.612564557431,
     -2261.102826808333,
     -2260.7189908960695,
     -2260.437608729253,
     -2260.231860238426,
     -2260.0784163526014,
     -2259.960659542152,
     -2259.8679640963023,
     -2259.793721328861,
     -2259.733658260372,
     -2259.684791553708,
     -2259.6448728507144,
     -2259.6121181368353,
     -2259.5850765029527]




```python
print(model)
```

    ### [kesmarag.hmm.HiddenMarkovModel] ###

    === Prior probabilities ================

    [1. 0. 0.]

    === Transition probabilities ===========

    [[0.995    0.005    0.      ]
     [0.       0.996666 0.003334]
     [0.       0.       1.      ]]

    === Emission distributions =============

    *** Hidden state #1 ***

    --- Mixture #1 ---
    weight : 0.779990073797613
    mean_values : [0.553266 1.155844]
    variances : [1.000249 0.967666]

    --- Mixture #2 ---
    weight : 0.22000992620238702
    mean_values : [2.598735 0.633391]
    variances : [1.234133 0.916872]

    *** Hidden state #2 ***

    --- Mixture #1 ---
    weight : 0.5188217626642593
    mean_values : [2.514082 5.076246]
    variances : [1.211327 0.903328]

    --- Mixture #2 ---
    weight : 0.4811782373357407
    mean_values : [3.080913 5.039015]
    variances : [1.327171 1.152902]

    *** Hidden state #3 ***

    --- Mixture #1 ---
    weight : 0.5700082256217439
    mean_values : [4.03977  1.118112]
    variances : [0.97422 1.00621]

    --- Mixture #2 ---
    weight : 0.429991774378256
    mean_values : [6.162698 5.064422]
    variances : [0.753987 1.278449]



```python
store_hmm(model, 'test_model.npz')
```


```python
load_model = restore_hmm('test_model.npz')
```


```python
gen_data = model.generate(700, 10, 0.05)
```

    0 -2129.992044055025
    1 -2316.443344656749
    2 -2252.206072731434
    3 -2219.667047368621
    4 -2206.6760352374367
    5 -2190.952289092368
    6 -2180.0268345326112
    7 -2353.7153702977475
    8 -2327.955163192414
    9 -2227.4471755146196



```python
print(gen_data)
```

    (array([[[-0.158655,  0.117973],
            [ 4.638243,  0.249049],
            [ 0.160007,  1.079808],
            ...,
            [ 4.671152,  4.18109 ],
            [ 2.121958,  3.747366],
            [ 2.572435,  6.352445]],

           [[-0.158655,  0.117973],
            [-1.379849,  0.998761],
            [-0.209945,  0.947926],
            ...,
            [ 3.93909 ,  1.383347],
            [ 5.356786,  1.57808 ],
            [ 5.0488  ,  5.586755]],

           [[-0.158655,  0.117973],
            [ 1.334   ,  0.979797],
            [ 3.708721,  1.321735],
            ...,
            [ 3.819756,  0.78794 ],
            [ 6.53362 ,  4.177215],
            [ 7.410012,  6.30113 ]],

           ...,

           [[-0.158655,  0.117973],
            [-0.152573,  0.612675],
            [-0.917723, -0.632936],
            ...,
            [ 4.110186, -0.027864],
            [ 2.82694 ,  0.65438 ],
            [ 6.825696,  5.27543 ]],

           [[-0.158655,  0.117973],
            [ 3.141896,  0.560984],
            [ 2.552211, -0.223568],
            ...,
            [ 4.41791 , -0.430231],
            [ 2.525892, -0.64211 ],
            [ 5.52568 ,  6.313566]],

           [[-0.158655,  0.117973],
            [ 0.845694,  2.436781],
            [ 1.564802, -0.652546],
            ...,
            [ 2.33009 ,  0.932121],
            [ 7.095326,  6.339674],
            [ 3.748988,  2.25159 ]]]), array([[0., 0., 0., ..., 1., 1., 1.],
           [0., 0., 0., ..., 2., 2., 2.],
           [0., 0., 0., ..., 2., 2., 2.],
           ...,
           [0., 0., 0., ..., 2., 2., 2.],
           [0., 0., 0., ..., 2., 2., 2.],
           [0., 0., 0., ..., 2., 2., 2.]]))

