# EM-HMM
Implemented EM to train an HMM. Used the first 900 observations as a single training sequence, and the last 100 as a single development sequence. 

## Main codes:

### Forward
```python

def forward(model, data, args):
    from scipy.stats import multivariate_normal
    from math import log
    alphas = np.zeros((len(data),args.cluster_num))
    log_likelihood = 0.0
    #TODO: Calculate and return forward probabilities (normalized at each timestep; see next line) and log_likelihood
    #NOTE: To avoid numerical problems, calculate the sum of alpha[t] at each step, normalize alpha[t] by that value, and increment log_likelihood by the log of the value you normalized by. This will prevent the probabilities from going to 0, and the scaling will be cancelled out in train_model when you normalize (you don't need to do anything different than what's in the notes). This was discussed in class on April 3rd.
    # raise NotImplementedError
    initials, transitions, mus, sigmas = extract_parameters(model)
    K = args.cluster_num
    N = len(data)
    sum_of_alphas = np.zeros(N)
    emissions = np.zeros((N, K))

    for t in range(N):
        for k in range(K):
            # distinguished args tied
            if not args.tied:
                emissions[t, k] = multivariate_normal.pdf(data[t], mus[k], sigmas[k])
            else:
                emissions[t, k] = multivariate_normal.pdf(data[t], mus[k], sigmas)

    for t in range(N):
        # the first element when t == 0
        if t == 0:  # alphas[0, k]
            for i in range(K):
                alphas[t, i] = initials[i] * emissions[0, i]
        else:
            for i in range(K):
                for j in range(K):
                    alphas[t, i] += alphas[t-1, j] * transitions[j, i]
                # alphas[n] = alphas[n-1] * transitions * emissions[n].T
                alphas[t, i] *= emissions[t, i]
        sum_of_alphas[t] = sum(alphas[t, :])
        alphas[t, :] /= sum_of_alphas[t]
        log_likelihood += log(sum_of_alphas[t])
    return alphas, log_likelihood

```
### Backward

```python
def backward(model, data, args):
    from scipy.stats import multivariate_normal
    betas = np.zeros((len(data),args.cluster_num))
    #TODO: Calculate and return backward probabilities (normalized like in forward before)
    initials, transitions, mus, sigmas = extract_parameters(model)
    N = len(data)
    K = args.cluster_num
    emissions = np.zeros((N, K))

    for n in range(N):
        for k in range(K):
            # distinguished args tied
            if not args.tied:   
                emissions[n, k] = multivariate_normal.pdf(data[n], mus[k], sigmas[k])
            else:
                emissions[n, k] = multivariate_normal.pdf(data[n], mus[k], sigmas)

    for t in range(N-1, -1, -1):
        # the first element when t == N-1
        if t == N-1:    # alphas[N-1, k]
            for i in range(K):
                betas[t, i] = 1
        else:
            for i in range(K):
                for j in range(K):
                    betas[t, i] += betas[t+1, j] * transitions[i, j] * emissions[t+1, j]
        betas[t, :] /= np.sum(betas[t, :])

    # raise NotImplementedError
    return betas
```

### Training
```python
while current_iter < args.iterations:
        alphas, _ = forward(model, train_xs, args)
        betas = backward(model, train_xs, args)
        """ E step """
        for t in range(N):
            for i in range(K):
                gammas[t, i] = alphas[t, i] * betas[t, i]
            gammas[t, :] /= sum(gammas[t, :])

        for n in range(N):
            for k in range(K):
                if not args.tied:
                    emissions[n, k] = multivariate_normal.pdf(train_xs[n], mus[k], sigmas[k])
                else:
                    emissions[n, k] = multivariate_normal.pdf(train_xs[n], mus[k], sigmas)

        for t in range(1, N):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = np.dot(alphas[t-1, i], betas[t, j].T) * transitions[i, j] * emissions[t, j]
            xi[t, :, :] /= np.sum(xi[t, :, :])
        """ M step: calculate the new mus and sigmas for each gaussian by applying above xi """
        sum_of_gammas = np.sum(gammas, axis=0)
        for k in range(K):
            initials[k] = gammas[0, k]
            for j in range(K):
                # update Transision matrix
                transitions[k, j] = np.sum(xi[:, k, j]) / np.sum(gammas[:, k])

            weight_sum = 0
            for n in range(N):
                weight_sum += (gammas[n, k] * train_xs[n])
            # update mus
            mus[k] = weight_sum / sum_of_gammas[k]

        # update sigmas different with arg tied
        if not args.tied:
            for k in range(K):
                weighted_sum = np.zeros((2, 2))
                for n in range(N):
                    weighted_sum += (gammas[n, k] * np.outer(train_xs[n]-mus[k], train_xs[n]-mus[k]))
                sigmas[k] = weighted_sum / sum_of_gammas[k]
        else:
            sigmas = np.zeros(sigmas.shape)
            for k in range(K):
                for n in range(N):
                    sigmas += (gammas[n, k] * np. outer(train_xs[n]-mus[k], train_xs[n]-mus[k]))
            sigmas[:] = sigmas / N

        ## likelihood computation for plotting
        current_model = [initials, transitions, mus, sigmas]
        # current_log_likelihood = np.sum(np.log(np.sum(P_Z_given_X, axis = 1)))
        current_log_likelihood = average_log_likelihood(current_model, train_xs, args)
        log_likelihoods.append(current_log_likelihood)

        current_iter += 1

        if not args.nodev:
            ll_dev = average_log_likelihood(current_model, dev_xs, args)
            print("iter %s dev log_likelihood: %s" % (str(current_iter), str(ll_dev)))
            if ll_dev > best_ll:
                best_ll = ll_dev
                best_model = current_model
                best_iter = current_iter
        print("iter %s train log_likelihood: %s" % (str(current_iter), str(current_log_likelihood)))


```


## 1.What I did?
    1.1 Implemented the model initilization
    1.2 Implemented both tied and full covariance
    1.3 Implemented the logic of E and M step
    1.4 Implemented the logic of forward and backward
    1.5 Declared and implemented a function for computing Prob given the whole data matrix
    1.6 Implemented logic using dev data to choose best model
    1.7 Run different set of hyperparameters to choose best K(number of clusters) and I(number of iterations) for the data given
    1.8 Wrote a function to plot the trend of average likelihood based on different input
    1.9 Wrote a function to visualize the cluster

## 2.Compared with original non-sequence

Better. The best log likelihood of HMM could reach -3.7. While the non-sequence only -4.31

## 3.Best num of states (datas shown below)

    From the log likelihood running on dev data, the best K is 5 and the I is 22.
    Average log likelihood when K = 5: Train LL: -3.7228302479806064	Dev LL: -3.709709886498374

## 4.Tied and full covariance

Tied: (2,2) all used the same tied covariance
Full: (args.cluster_num,2,2) k different covariance

	I tried tied and full on the data and get quiet different results. Full cov are much better than the tied that the tied's corresponding ll even decrease with the iteration increases. Because there are two different gaussian model for each cluster. They don't need to be with total same cov. So the full cov will better help to train the data.
	    [Tied:]
	      best iterations: 10
	      Train LL: -4.3973604221966
	      Dev LL: -4.419237979207335
	      Initials: 1.0 | 0.0 | 0.0
	      Transitions: 0.06796666462417814 0.2543992331495606 0.679048679952909 | 0.40907535622740343 0.028675091741660188 0.5585417930546622 | 0.12913809025272815 0.7319158563699996 0.13855216256616776
	      Mus: -4.490292072296143 -0.7727795839309692 | 0.7569531798362732 -3.2173733711242676 | -0.03431331738829613 1.7204155921936035
	      Sigma: 3.4570930491496688 0.3648431369193479 0.3648431369193479 1.6972277344249995
	    [Full]
	      best iterations: 9
	      Train LL: -3.998961889620625
	      Dev LL: -4.044441410167979
	      Initials: 1.9360876256934118e-76 | 1.0 | 0.0
	      Transitions: 0.033332541729351725 0.5445371205691123 0.4195922665858741 | 0.41632807990405685 4.523187483273485e-27 0.5836719200959151 | 0.9999958933011637 1.5192606444174447e-06 9.392860242604773e-24
	      Mus: 0.18539197742938995 -2.794691324234009 | -3.90271258354187 0.024534499272704124 | 0.07384778559207916 1.9276814460754395
	      Sigmas: 7.122908590414413 -0.14455999054449772 -0.14455999054449772 3.249535768019475 | 5.5543410967785345 2.3743385564636106 2.3743385564636106 1.2014391507318603 | 1.0765094010074774 0.11424378486511712 0.11424378486511712 1.0633718099301348


## 5.Logic for dev data

5.1 Inside the train_model function, I only focus on the best iteration. I kept track of each iteration and compare the average likelihood finally return the best one.

5.2 What's more, I also iterate K (number of clusters) and find the best hyperparameters set using different data. Here I run 30 iteration for each K and record the best ll on dev and final ll on training data. Every K run by 10 times to choose the best to avoid the influences of initialization. 

	Average log likelihood when K = 2: Train LL: -4.331358290053831		Dev LL: -4.425244581508012
	Average log likelihood when K = 3: Train LL: -3.9989623458797072	Dev LL: -4.044436700309244
	Average log likelihood when K = 4: Train LL: -3.7292294388793743	Dev LL: -3.7099826154884337
	Average log likelihood when K = 5: Train LL: -3.7228302479806064	Dev LL: -3.709709886498374
	Average log likelihood when K = 6: Train LL: -3.715170765930907		Dev LL: -3.7149759050458813
	Average log likelihood when K = 7: Train LL: -3.7041004915401663	Dev LL: -3.7530755154399738
	Average log likelihood when K = 10: Train LL: -3.704849905140207	Dev LL: -3.740708758613173

From the log likelihood running on dev data, the best K is 5 and the I is 22.


## 6.Likelihood changing with iterations

I run 30 iterations based on gaussian_smoketest_clusters.txt and get the result as follows. I also plotted them out in the graph. We can see that only after 35 iterations, the log likelihood changes very little after that. Because after rough 35 iterations, the data already converged.

	best iterations: 35
	Train LL: -4.333276723427157
	Dev LL: -4.444069546118861
	Initials: 6.273402101914838e-91 | 1.0
	Transitions: 0.28085631727355 0.7191367270839577 | 0.957569513985722 0.0398491462953853
	Mus: -1.6273572444915771 1.0996357202529907 | 0.23399387300014496 -2.8481454849243164
	Sigmas: 6.808788522156473 2.929013064111468 2.929013064111468 2.021901872156217 | 7.065985273962603 -0.061952384515201415 -0.061952384515201415 3.1385384291110254

	iter 1 dev log_likelihood: -4.775873864239315
	iter 1 train log_likelihood: -4.77194805671639
	iter 2 dev log_likelihood: -4.759851948437602
	iter 2 train log_likelihood: -4.743156036861762
	iter 3 dev log_likelihood: -4.756474829724014
	iter 3 train log_likelihood: -4.7318429762048195
	iter 4 dev log_likelihood: -4.754871906160093
	iter 4 train log_likelihood: -4.723505146234699
	iter 5 dev log_likelihood: -4.753691682065679
	iter 5 train log_likelihood: -4.715227365050867
	iter 6 dev log_likelihood: -4.7526518444072705
	iter 6 train log_likelihood: -4.705828417398861
	iter 7 dev log_likelihood: -4.75104046339296
	iter 7 train log_likelihood: -4.694036840493648
	iter 8 dev log_likelihood: -4.7465461960533775
	iter 8 train log_likelihood: -4.677211092732966
	iter 9 dev log_likelihood: -4.732217044386876
	iter 9 train log_likelihood: -4.64903350863819
	iter 10 dev log_likelihood: -4.693048513482815
	iter 10 train log_likelihood: -4.597446981383891
	iter 11 dev log_likelihood: -4.625254157260241
	iter 11 train log_likelihood: -4.519547302332696
	iter 12 dev log_likelihood: -4.576123675905664
	iter 12 train log_likelihood: -4.460829312187511
	iter 13 dev log_likelihood: -4.566019437690257
	iter 13 train log_likelihood: -4.445860026622655
	iter 14 dev log_likelihood: -4.564878713918133
	iter 14 train log_likelihood: -4.44349414054086
	iter 15 dev log_likelihood: -4.565007133459366
	iter 15 train log_likelihood: -4.442660509316287
	iter 16 dev log_likelihood: -4.565457979489788
	iter 16 train log_likelihood: -4.442029997146123
	iter 17 dev log_likelihood: -4.565920395728872
	iter 17 train log_likelihood: -4.441417061187635
	iter 18 dev log_likelihood: -4.566259230818609
	iter 18 train log_likelihood: -4.44077269143238
	iter 19 dev log_likelihood: -4.566400378874664
	iter 19 train log_likelihood: -4.44007087892365
	iter 20 dev log_likelihood: -4.5662848122976305
	iter 20 train log_likelihood: -4.43928918876655
	iter 21 dev log_likelihood: -4.5658511437571345
	iter 21 train log_likelihood: -4.438404375053876
	iter 22 dev log_likelihood: -4.565028237686182
	iter 22 train log_likelihood: -4.43739132862564
	iter 23 dev log_likelihood: -4.5637324262218275
	iter 23 train log_likelihood: -4.436222038885511
	iter 24 dev log_likelihood: -4.561864013718502
	iter 24 train log_likelihood: -4.434862048682379
	iter 25 dev log_likelihood: -4.559295976765579
	iter 25 train log_likelihood: -4.433260827300157
	iter 26 dev log_likelihood: -4.555837405165956
	iter 26 train log_likelihood: -4.4313305198028905
	iter 27 dev log_likelihood: -4.551152845431904
	iter 27 train log_likelihood: -4.428902640651494
	iter 28 dev log_likelihood: -4.544604923309437
	iter 28 train log_likelihood: -4.4256393713465005
	iter 29 dev log_likelihood: -4.534996003964877
	iter 29 train log_likelihood: -4.420847976355753
	iter 30 dev log_likelihood: -4.520378838767149
	iter 30 train log_likelihood: -4.413131285032901
	iter 31 dev log_likelihood: -4.499007410086839
	iter 31 train log_likelihood: -4.400024953549397
	iter 32 dev log_likelihood: -4.472713770409835
	iter 32 train log_likelihood: -4.379001542790551
	iter 33 dev log_likelihood: -4.4510914280608045
	iter 33 train log_likelihood: -4.354611009912839
	iter 34 dev log_likelihood: -4.443664415058445
	iter 34 train log_likelihood: -4.339368779562594
	iter 35 dev log_likelihood: -4.4436590492425765
	iter 35 train log_likelihood: -4.334687412542192
	iter 36 dev log_likelihood: -4.443937852948241
	iter 36 train log_likelihood: -4.333608330316279
	iter 37 dev log_likelihood: -4.444018434805081
	iter 37 train log_likelihood: -4.333353480828559
	iter 38 dev log_likelihood: -4.444045465129044
	iter 38 train log_likelihood: -4.333293448314042
	iter 39 dev log_likelihood: -4.44405734426838
	iter 39 train log_likelihood: -4.333279883828872
	iter 40 dev log_likelihood: -4.444063368743342
	iter 40 train log_likelihood: -4.333277071879572
	iter 41 dev log_likelihood: -4.444066384104095
	iter 41 train log_likelihood: -4.33327661091904
	iter 42 dev log_likelihood: -4.444068070512921
	iter 42 train log_likelihood: -4.33327660211752
	iter 43 dev log_likelihood: -4.444068775290524
	iter 43 train log_likelihood: -4.333276649030138
	iter 44 dev log_likelihood: -4.444069214340737
	iter 44 train log_likelihood: -4.333276683804839
	iter 45 dev log_likelihood: -4.44406942298218
	iter 45 train log_likelihood: -4.33327670361212
	iter 46 dev log_likelihood: -4.444069431433417
	iter 46 train log_likelihood: -4.333276713931034
	iter 47 dev log_likelihood: -4.44406943885494
	iter 47 train log_likelihood: -4.333276719054701
	iter 48 dev log_likelihood: -4.444069508567622
	iter 48 train log_likelihood: -4.33327672155297
	iter 49 dev log_likelihood: -4.444069516526327
	iter 49 train log_likelihood: -4.333276722813024
	iter 50 dev log_likelihood: -4.444069546118861
	iter 50 train log_likelihood: -4.333276723427157



