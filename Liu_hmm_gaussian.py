#!/usr/bin/env python3
import numpy as np
if not __file__.endswith('_hmm_gaussian.py'):
    print('ERROR: This file is not named correctly! Please name it as Lastname_hmm_gaussian.py (replacing Lastname with your last name)!')
    exit(1)

DATA_PATH = "." #TODO: if doing development somewhere other than the cycle server (not recommended), then change this to the directory where your data file is (points.dat)

def parse_data(args):
    num = float
    dtype = np.float32
    data = []
    with open(args.data_file, 'r') as f:
        for line in f:
            data.append([num(t) for t in line.split()])
    dev_cutoff = int(.9*len(data))
    train_xs = np.asarray(data[:dev_cutoff],dtype=dtype)
    dev_xs = np.asarray(data[dev_cutoff:],dtype=dtype) if not args.nodev else None
    return train_xs, dev_xs

def init_model(args):
    if args.cluster_num:
        mus = np.zeros((args.cluster_num,2))
        if not args.tied:
            sigmas = np.zeros((args.cluster_num,2,2))
        else:
            sigmas = np.zeros((2,2))
        transitions = np.zeros((args.cluster_num,args.cluster_num)) #transitions[i][j] = probability of moving from cluster i to cluster j
        initials = np.zeros(args.cluster_num) #probability for starting in each state
        #TODO: randomly initialize clusters (mus, sigmas, initials, and transitions)
        K = args.cluster_num
        mus = np.random.rand(K, 2)
        if not args.tied:
            for k in range(K):
                sigmas[k] = np.identity(2)
        else:
            sigmas = np.identity(2)

        transitions = np.ones((K, K)) / K
        for i in range(K):
            initials[i] = np.random.random()
            for j in range(K):
                transitions[i, j] = np.random.random()
            transitions[i, :] /= np.sum(transitions[i, :], axis=0)
        initials /= initials.sum()
        # raise NotImplementedError #remove when random initialization is implemented

    else:
        mus = []
        sigmas = []
        transitions = []
        initials = []
        with open(args.clusters_file,'r') as f:
            for line in f:
                #each line is a cluster, and looks like this:
                #initial mu_1 mu_2 sigma_0_0 sigma_0_1 sigma_1_0 sigma_1_1 transition_this_to_0 transition_this_to_1 ... transition_this_to_K-1
                vals = list(map(float,line.split()))
                initials.append(vals[0])
                mus.append(vals[1:3])
                sigmas.append([vals[3:5],vals[5:7]])
                transitions.append(vals[7:])
        initials = np.asarray(initials)
        transitions = np.asarray(transitions)
        mus = np.asarray(mus)
        sigmas = np.asarray(sigmas)
        args.cluster_num = len(initials)

    #TODO: Do whatever you want to pack mus, sigmas, initals, and transitions into the model variable (just a tuple, or a class, etc.)
    model = [initials, transitions, mus, sigmas]
    # raise NotImplementedError #remove when model initialization is implemented
    return model

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

def train_model(model, train_xs, dev_xs, args):
    from scipy.stats import multivariate_normal
    #TODO: train the model, respecting args (note that dev_xs is None if args.nodev is True)
    # raise NotImplementedError #remove when model training is implemented
    initials, transitions, mus, sigmas = extract_parameters(model)
    N = len(train_xs)
    K = args.cluster_num
    emissions = np.zeros((N, K))
    gammas = np.zeros((N, K))
    xi = np.zeros((N, K, K))

    ### log_likelihoods of every iterations
    log_likelihoods = []
    # for dev data to choose best model
    best_model = model
    best_ll = float("-inf")
    best_iter = 0
    current_iter = 0

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


    model = [initials, transitions, mus, sigmas]
    # plot_log_likelihood(log_likelihoods)
    # demo_2d(train_xs, mus, sigmas, log_likelihoods)
    if not args.nodev:
        print("best iterations:", str(best_iter))
        return best_model

    return model

def average_log_likelihood(model, data, args):
    #TODO: implement average LL calculation (log likelihood of the data, divided by the length of the data)
    #NOTE: yes, this is very simple, because you did most of the work in the forward function above
    alphas, log_likelihood = forward(model, data, args)
    ## likelihood computation for plotting
    ll = 1.0 / len(data) * log_likelihood
    return ll

def extract_parameters(model):
    #TODO: Extract initials, transitions, mus, and sigmas from the model and return them (same type and shape as in init_model)
    # raise NotImplementedError #remove when parameter extraction is implemented
    return model[0], model[1], model[2], model[3]
def plot_log_likelihood(log_likelihoods):
    import pylab as plt
    # print("log_likelihoods: ", log_likelihoods)
    plt.plot(log_likelihoods)
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()

def demo_2d(data, mus, sigmas, log_likelihoods):
    import pylab as plt    
    from matplotlib.patches import Ellipse
    
    def plot_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
    
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(abs(vals))
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
        ax.add_artist(ellip)
        return ellip    
    
    def show(X, mu, cov):

        plt.cla()
        K = len(mu) # number of clusters
        colors = ['b', 'k', 'g', 'c', 'm', 'y', 'r']
        plt.plot(X.T[0], X.T[1], 'm*')
        for k in range(K):
          plot_ellipse(mu[k], cov[k],  alpha=0.6, color = colors[k % len(colors)])  

    
    fig = plt.figure(figsize = (13, 6))
    fig.add_subplot(121)
    show(data, mus, sigmas)
    fig.add_subplot(122)
    plt.plot(np.array(log_likelihoods))
    plt.title('Log Likelihood vs iteration plot')
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    plt.show()

def main():
    import argparse
    import os
    print('Gaussian') #Do not change, and do not print anything before this.
    parser = argparse.ArgumentParser(description='Use EM to fit a set of points')
    init_group = parser.add_mutually_exclusive_group(required=True)
    init_group.add_argument('--cluster_num', type=int, help='Randomly initialize this many clusters.')
    init_group.add_argument('--clusters_file', type=str, help='Initialize clusters from this file.')
    parser.add_argument('--nodev', action='store_true', help='If provided, no dev data will be used.')
    parser.add_argument('--data_file', type=str, default=os.path.join(DATA_PATH, 'points.dat'), help='Data file.')
    parser.add_argument('--print_params', action='store_true', help='If provided, learned parameters will also be printed.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of EM iterations to perform')
    parser.add_argument('--tied',action='store_true',help='If provided, use a single covariance matrix for all clusters.')
    args = parser.parse_args()
    if args.tied and args.clusters_file:
        print('You don\'t have to (and should not) implement tied covariances when initializing from a file. Don\'t provide --tied and --clusters_file together.')
        exit(1)

    train_xs, dev_xs = parse_data(args)
    model = init_model(args)
    model = train_model(model, train_xs, dev_xs, args)
    nll_train = average_log_likelihood(model, train_xs, args)
    print('Train LL: {}'.format(nll_train))
    if not args.nodev:
        nll_dev = average_log_likelihood(model, dev_xs, args)
        print('Dev LL: {}'.format(nll_dev))
    initials, transitions, mus, sigmas = extract_parameters(model)
    if args.print_params:
        def intersperse(s):
            return lambda a: s.join(map(str,a))
        print('Initials: {}'.format(intersperse(' | ')(np.nditer(initials))))
        print('Transitions: {}'.format(intersperse(' | ')(map(intersperse(' '),transitions))))
        print('Mus: {}'.format(intersperse(' | ')(map(intersperse(' '),mus))))
        if args.tied:
            print('Sigma: {}'.format(intersperse(' ')(np.nditer(sigmas))))
        else:
            print('Sigmas: {}'.format(intersperse(' | ')(map(intersperse(' '),map(lambda s: np.nditer(s),sigmas)))))

if __name__ == '__main__':
    main()