import pandas as pd
from numpy import array, argmax, unique
from numpy.random import seed
from GMMFromScratch import GMMFromScratch
from PCAFromScratch import PCAFromScratch
import matplotlib.pyplot as plt

seed(42)

data = pd.read_csv('vowel_train.txt')
true_labels = array(data['y'])
input = array(data[[f'x.{i}' for i in range(1, 11)]])
input = PCAFromScratch(3, input).transformation()

model = GMMFromScratch(11, input)
probs, gmm_params = model.expectation_maximization(11, input, 100)
gmm_labels = [argmax(array(p)) for p in probs]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
class_to_color_map = {
    1: "red",
    2: "blue",
    3: "green",
    4: "yellow",
    5: "orange",
    6: "purple",
    7: "pink",
    8: "brown",
    9: "gray",
    10: "olive",
    11: "cyan",
}
ax.scatter(input[:,0], input[:,1], input[:, 2], c=[class_to_color_map[label+1] for label in gmm_labels])
plt.show()

# test_data = pd.read_csv('vowel_test.txt')
# test_labels = array(test_data['y'])
# test_input = array(test_data[[f'x.{i}' for i in range(1, 11)]])

# probs = model.compute_probs(11, test_input, model.params)
# gmm_labels = [argmax(array(p)) for p in probs]

# disagreement distance
# def disagree_dist(p, c):
#     dist = 0
#     for i in range(len(p)):
#         for j in range(i+1, len(p)):
#             if (p[i]==p[j] and c[i]!=c[j]) or (p[i]!=p[j] and c[i]==c[j]):
#                 dist += 1
#     return dist

# print(disagree_dist(test_labels, gmm_labels))