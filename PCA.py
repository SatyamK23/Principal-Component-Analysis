import numpy as np
import matplotlib.pyplot as plt

'''
Step 1: Generated  Random Data
Step 2: Calculate Mean and subtract that from data point
Step 3: Generate Covariance Matrix
Step 4: Calculate Eigenvectors and Eigenvalues of COvariance Matrix
Step 5: Bigger of EIgen vector is the principal Component

'''
def mean_shift_data(data, mean):
	return (data.T + mean).T

def generate_data():
    data = np.random.multivariate_normal(mean, cov, 5000).T
    return data

def generate_positive_semidefinite_matrix():
	A = np.random.rand(2,2)
	return np.dot(A,A.transpose()) 

#TO check whether matrix is positive semi-definite or not
def is_pos_definate(data):
	return np.all(np.linalg.eigvals(A) > 0)

'''
Co-variance Matrix of sample data is positive definite
https://stats.stackexchange.com/questions/52976/is-a-sample-covariance-matrix-always-symmetric-and-positive-definite
len(data) is the sample size
'''
def get_covariance_matrix(data):
	return (1/len(data))*generate_positive_semidefinite_matrix()

def get_eigenvector_matrix(data):
	eig_val, eig_vec = np.linalg.eig(data)
	return eig_val, eig_vec

def get_transformed_data(data, transformation):
    return np.matmul(transformation, data)

def pca_using_cov_matrix(data):
    data = mean_shift_data(data, -data.mean(axis=1))
    cov = get_covariance_matrix(data)
    eig_val, eig_vec_matrix = get_eigenvector_matrix(cov)
    new_data = get_transformed_data(data, eig_vec_matrix)
    return new_data, eig_val, eig_vec_matrix

def plot_data_after_pca(data, new_data, eig_vecs, title=""):
    x = data[0, :]
    y = data[1, :]
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    for temp_axis in eig_vecs:
        start, end = data.mean(axis=1), data.mean(axis=1) + new_data.std(axis=1).mean()*temp_axis
        ax.annotate('',end, start, arrowprops=dict(facecolor='black', width=1.0))
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    title = title + "\nCovariance Matrix is " + str(cov)
    plt.suptitle(title)
    plt.show()


if __name__=='__main__':
    mean = [0, 0]
    cov = generate_positive_semidefinite_matrix()
    data = generate_data()
    x = data[0, :]
    y = data[1, :]
    plt.scatter(x, y)
    plt.suptitle("Original Data")
    plt.show()
    new_data, eig_vals, eig_vecs = pca_using_cov_matrix(data)
    plot_data_after_pca(data, new_data, eig_vecs, "PCA using Covariance Matrix")


