from numpy import mean, std, cov, array
from numpy.linalg import eig

class PCAFromScratch:
    def __init__(self, k, data):
        self.k = k
        self.data = data

    def standardize(self, data):
        return (data - mean(data, axis=0)) / std(data, axis=0)

    def cov_mat(self, data):
        return cov(data)
    
    def sorted_eigen_vectors(self, mat):
        eig_vals, eig_vecs = eig(mat)

        # sort by eigenvalues
        eig_val_vec_pairs = list(zip(eig_vals, eig_vecs))
        eig_val_vec_pairs.sort(key=lambda x: x[0], reverse=True)

        # keep just the vectors
        self.components = array([p[1] for p in eig_val_vec_pairs])
        return self.components

    def transformation(self):
        std_data = self.standardize(self.data)
        cov_mat = self.cov_mat(std_data.T)
        eig_vecs = self.sorted_eigen_vectors(cov_mat)
        projection = std_data.dot(eig_vecs[:self.k, :].T)
        return projection

if __name__ == '__main__':
    X = array([[1,1], [2,2], [3,3], [4,4]])
    pca = PCAFromScratch(1, X)
    print(pca.transformation())