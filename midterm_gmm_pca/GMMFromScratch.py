from numpy import array, argmax, hstack
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy.random import multivariate_normal as mvn_random
from scipy.stats import multivariate_normal
from numpy.random import normal, uniform, seed

# NUM_CLUSTERS = 3
# NUM_SAMLPES = 1000
# NUM_ITERATIONS = 20

class Component:
    def __init__(self, mixture_prop, mean, variance):
        self.mixture_prop = mixture_prop
        self.mean = mean
        self.variance = variance

# def generate_gmm_dataset(gmm_params, sample_size):
#     def get_random_component(gmm_params):
#         '''
#             returns component with prob
#             proportional to mixture_prop
#         '''
#         r = uniform()
#         for c in gmm_params:
#             r -= c.mixture_prop
#             if r <= 0:
#                 return c

#     dataset = []
#     for _ in range(sample_size):
#         comp = get_random_component(gmm_params)
#         dataset += [normal(comp.mean, comp.variance)]
#     return array(dataset)

# gmm = [
#     Component(.25, [-3, 3], [[1, 0], [0, 1]]),
#     Component(.50, [0, 0], [[1, 0], [0, 1]]),
#     Component(.25, [3, 3], [[1, 0], [0, 1]])
# ]


# data = generate_gmm_dataset(gmm, NUM_SAMLPES)
# data = hstack([data[:,0,0].reshape(-1,1), data[:,1,1].reshape(-1,1)])

class GMMFromScratch:
    def __init__(self, k, dataset):
        self.params = []

        kmeans = KMeans(k, init='k-means++', random_state=42).fit(dataset)
        
        for j in range(k):
            p_cj = sum([1 if kmeans.labels_[i] == j else 0 for i in range(len(dataset))]) / len(dataset)
            mean_j = sum([dataset[i] for i in range(len(dataset)) if kmeans.labels_[i] == j]) / sum([1 if kmeans.labels_[i] == j else 0 for i in range(len(dataset))])
            var_j = sum([(dataset[i] - mean_j).reshape(-1, 1) * (dataset[i] - mean_j).reshape(1, -1) for i in range(len(dataset)) if kmeans.labels_[i] == j]) / sum([1 if kmeans.labels_[i] == j else 0 for i in range(len(dataset))])
            
            self.params.append(Component(p_cj, mean_j, var_j))
    
    def compute_gmm(self, k, dataset, probs):
        '''
            Compute P(C_j), mean_j, var_j
            Here mean_j is a vector and var_j is a matrix
        '''
        gmm_params = []

        for j in range(k):
            p_cj = sum([probs[i][j] for i in range(len(dataset))]) / len(dataset)
            mean_j = sum([probs[i][j] * dataset[i] for i in range(len(dataset))]) / sum([probs[i][j] for i in range(len(dataset))])
            var_j = sum([probs[i][j] * (dataset[i] - mean_j).reshape(-1, 1) * (dataset[i] - mean_j).reshape(1, -1) for i in range(len(dataset))]) / sum([probs[i][j] for i in range(len(dataset))])

            gmm_params.append(Component(p_cj, mean_j, var_j))

        return gmm_params


    def compute_probs(self, k, dataset, gmm_params):
        '''
            For all x_i in dataset, compute P(C_j | X_i) = P(X_i | C_j)P(C_j) / P(X_i) for all C_j
            return the list of lists of all P(C_j | X_i) for all x_i in dataset.
        '''
        probs = []

        for i in range(len(dataset)):
            p_cj_xi = []
            for j in range(k):
                p_cj_xi += [gmm_params[j].mixture_prop * multivariate_normal.pdf(dataset[i], gmm_params[j].mean, gmm_params[j].variance)]
            p_cj_xi = p_cj_xi / sum(p_cj_xi)
            probs.append(p_cj_xi)

        return probs
    
    def expectation_maximization(self, k, dataset, iterations):
        for _ in range(iterations):
            # expectation step
            probs = self.compute_probs(k, dataset, self.params)

            # maximization step
            self.params = self.compute_gmm(k, dataset, probs)

            # print(_)

        return probs, self.params
    
# model = GMMFromScratch(3, data)
# probs, gmm_params = model.expectation_maximization(NUM_CLUSTERS, data, NUM_ITERATIONS)

# labels = [argmax(array(p)) for p in probs] # create a hard assignment
# size = 50 * array(probs).max(1) ** 2 # emphasizes the difference in probability

# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=size)
# plt.title('GMM with {} clusters and {} samples'.format(NUM_CLUSTERS, NUM_SAMLPES))
# plt.show()