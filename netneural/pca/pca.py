from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# https://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues

def get_training_data_gestures(input_folder=None):
    frames = pd.read_csv("../data/data_merge/output_csv/rotate_felix_gt.csv")
    swipe_right = pd.read_csv("../data/data_merge/output_csv/swipe_right_mamad_gt.csv")
    swipe_right = swipe_right.append(pd.read_csv("../data/data_merge/output_csv/swipe_right_marja_gt.csv"),
                                     ignore_index=True)
    swipe_right['ground_truth'] = swipe_right['ground_truth'].fillna('swipe_right')
    frames = frames.append(swipe_right, ignore_index=True)
    frames = frames.append(pd.read_csv("../data/data_merge/output_csv/swipe_left_mamad_gt.csv"), ignore_index=True)
    frames = frames.append(pd.read_csv("../data/data_merge/output_csv/swipe_left_marja_gt.csv"), ignore_index=True)

    return frames


def normalize_data(data):
    return data - np.mean(data, axis=0)


class PCA:

    def __init__(self, n=None, eigenvectors=None):
        self.eigenvectors = eigenvectors
        self.eigenvalues = None
        self.n = n

    def pca(self, data, number_pcs=None, var_per=None):
        """
        Perform PCA on given data.
        :param data: basis data for the PCA
        :param number_pcs: desired number of principal components to return
        :param var_per: desired percentage of variance to cover with pcs
        :return: the desired number of pcs or the number of pcs that cover the desired percentage of variance.
        Returns all pcs if no variable was given, returns desired number of pcs if it is lower then the number of pcs that
        would cover the desired variance percentage and vice versa.
        """
        self.eigenvectors, self.eigenvalues = self.get_eigenvectors_and_values(data)
        if var_per is not None:
            final_number_pcs = self.explained_variance(self.eigenvalues, var_per)
            if number_pcs is not None:
                final_number_pcs = min(final_number_pcs, number_pcs)
        else:
            if number_pcs is not None:
                final_number_pcs = number_pcs
            else:
                final_number_pcs = data.shape[1]
        self.n = final_number_pcs
        return self.get_n_dimensions(data, self.n, self.eigenvectors)

    def transform_data(self, data):
        """
        Transform given data using the previously stored eigenvectors
        :param data: data to transform
        :return: transformed data
        """
        return self.get_n_dimensions(data, self.n, self.eigenvectors)

    @staticmethod
    def get_eigenvectors_and_values(data):
        """
        Compute eigenvalues and eigenvectors for the covariance matrix of the input data
        :param data: input data
        :return: eigenvectors and eigenvalues
        """
        # DATA SCALING
        # normalize data or use standard scaler
        # scaled_data = normalize_data(data)
        # scaler = StandardScaler()
        # scaler.fit(data)
        # scaled_data = scaler.transform(data)
        scaled_data = data  # data usually already scaled after preprocessing

        # COVARIANCE MATRIX
        # covariance: reports how much two random variables vary
        cov_mat = np.cov(scaled_data, rowvar=False)

        # EIGENVALUES AND EIGENVECTORS
        # eigenvalues: higher -> higher variability in data
        # eigenvector: orthogonal to each other, each represents a principal axis
        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)

        # SORT
        # the first column in rearranged eigenvector matrix is component capturing highest variability
        # sort the eigenvalues in descending order
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalues = eigen_values[sorted_index]
        # similarly sort the eigenvectors
        sorted_eigenvectors = eigen_vectors[:, sorted_index]
        return sorted_eigenvectors, sorted_eigenvalues

    @staticmethod
    def get_n_dimensions(data, n, eigenvectors):
        """
        Get the first n principal components
        :param data: input data
        :param n: the number of desired dimensions
        :param eigenvectors: needed for transforming data
        :return: n dimensional data
        """
        # select the first n eigenvectors
        eigenvector_subset = eigenvectors[:, 0:n]

        # TRANSFORM
        # reduce data from higher dimension to lower dimension
        n_dimensional_data = np.dot(eigenvector_subset.transpose(), data.transpose()).transpose()
        return n_dimensional_data

    @staticmethod
    def explained_variance(eigenvalues, variance_per=1.0, plot=False):
        """
        Computes how many principal components are needed for covering the wanted percentage of variance.
        Possibly plots the explained variance (sum of eigenvalues) against the number n of principal components.
        :param plot: default False, set True for a plot of the values
        :param variance_per: the percentage of variance the user wants to cover with principal components, default value is
            100 %, what will return the number of all principal components
        :param eigenvalues: needed for computing the percentage of explained variance one eigenvector resembles
        :return: the number of principal components needed for covering given percentage of variance
        """
        explained_variances = []
        for i in range(len(eigenvalues)):
            explained_variances.append(eigenvalues[i] / np.sum(eigenvalues))

        n_pcs = list()
        explained_variance = list()
        for n_pc in range(len(eigenvalues)):
            n_pcs.append(n_pc + 1)
            explained_variance.append(np.sum(explained_variances[:n_pc + 1]))

        if plot:
            plt.plot(explained_variance)
            plt.ylabel('explained variance')
            plt.xlabel('n')
            plt.title('Explained Variance using n Principal Components')
            plt.grid(visible=True)
            plt.show()

            # zoom in
            plt.plot(explained_variance, markevery=93)
            plt.ylabel('explained variance')
            plt.xlabel('n')
            plt.title('Explained Variance using n Principal Components')
            plt.grid(visible=True)
            plt.ylim((0.99, 1.005))
            plt.xlim((60, 110))
            plt.show()

        max = 0
        for var in explained_variance:
            if variance_per < var:
                break
            if var > max:
                max = var

        return explained_variance.index(max)

    @staticmethod
    def analyze_eigenvector(eigenvector, data_attributes: List[str]):
        """
        Analyze the importance of each attribute according to specific eigenvector
        :param eigenvector: distributes importance among attributes
        :param data_attributes: list of attributes
        :return: list of attributes ordered according to eigenvector values
        """
        sorted_index = np.argsort(-1 * eigenvector)
        return [data_attributes[i] for i in sorted_index]

    def analyze_eigenvectors(self, eigenvectors, data_attributes: List[str]):
        """
        Analyze the importance of each attribute according to the list of eigenvectors
        :param eigenvectors: distributes importance among attributes
        :param data_attributes: list of attributes
        :return: matrix eigenvectors x attributes where each attribute is ranked in every eigenvector according to percentage in eigenvector
        """
        sorted_attributes = list()
        for eigenvector in eigenvectors:
            sorted_attributes.append(self.analyze_eigenvector(eigenvector, data_attributes))
        sorted_attributes = np.array(sorted_attributes)
        matrix = pd.DataFrame()
        for attribute in data_attributes:
            matrix[attribute] = np.apply_along_axis(lambda x: list(x).index(attribute), 1, sorted_attributes)
        return matrix

    @staticmethod
    def compute_order_attributes(ordered_attributes_matrix: pd.DataFrame):
        """
        Sort the attributes according to their averaged values in matrix
        :param ordered_attributes_matrix: holds a Series for every attribute with its ranks in every principal component
        :return: attributes ordered according to their weighted average of ranks
        """
        weights = np.arange(1, ordered_attributes_matrix.shape[0] + 1, 1)[::-1]
        return ordered_attributes_matrix.apply(lambda x: np.average(x, weights=weights), axis=0).sort_values()
