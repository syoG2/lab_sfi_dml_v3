# -*-coding:utf-8-*-

import copy
from enum import IntEnum
from math import log

import numpy
import pyclustering.core.xmeans_wrapper as wrapper
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.encoder import type_encoding
from pyclustering.cluster.kmeans import kmeans
from pyclustering.core.metric_wrapper import metric_wrapper
from pyclustering.core.wrapper import ccore_library
from pyclustering.utils import distance_metric, type_metric


class splitting_type(IntEnum):

    BAYESIAN_INFORMATION_CRITERION = 0
    MINIMUM_NOISELESS_DESCRIPTION_LENGTH = 1


class adjusted_xmeans:
    def __init__(
        self,
        data,
        c,
        initial_centers=None,
        kmax=20,
        tolerance=0.001,
        criterion=splitting_type.BAYESIAN_INFORMATION_CRITERION,
        ccore=False,
        **kwargs
    ):

        self.__c = c
        self.__pointer_data = numpy.array(data)
        self.__clusters = []
        self.__random_state = kwargs.get("random_state", None)
        self.__metric = copy.copy(
            kwargs.get("metric", distance_metric(type_metric.EUCLIDEAN_SQUARE))
        )

        if initial_centers is not None:
            self.__centers = numpy.array(initial_centers)
        else:
            self.__centers = kmeans_plusplus_initializer(
                data, 2, random_state=self.__random_state
            ).initialize()

        self.__kmax = kmax
        self.__tolerance = tolerance
        self.__criterion = criterion
        self.__total_wce = 0.0
        self.__repeat = kwargs.get("repeat", 1)
        self.__alpha = kwargs.get("alpha", 0.9)
        self.__beta = kwargs.get("beta", 0.9)

        self.__ccore = ccore and self.__metric.get_type() != type_metric.USER_DEFINED
        if self.__ccore is True:
            self.__ccore = ccore_library.workable()

        self.__verify_arguments()

    def process(self):

        if self.__ccore is True:
            self.__process_by_ccore()

        else:
            self.__process_by_python()

        return self

    def __process_by_ccore(self):

        ccore_metric = metric_wrapper.create_instance(self.__metric)

        result = wrapper.xmeans(
            self.__pointer_data,
            self.__centers,
            self.__kmax,
            self.__tolerance,
            self.__criterion,
            self.__alpha,
            self.__beta,
            self.__repeat,
            self.__random_state,
            ccore_metric.get_pointer(),
        )

        self.__clusters = result[0]
        self.__centers = result[1]
        self.__total_wce = result[2][0]

    def __process_by_python(self):

        self.__clusters = []
        while len(self.__centers) <= self.__kmax:
            current_cluster_number = len(self.__centers)

            self.__clusters, self.__centers, _ = self.__improve_parameters(
                self.__centers
            )
            allocated_centers = self.__improve_structure(
                self.__clusters, self.__centers
            )

            if current_cluster_number == len(allocated_centers):
                break
            else:
                self.__centers = allocated_centers

        self.__clusters, self.__centers, self.__total_wce = self.__improve_parameters(
            self.__centers
        )

    def predict(self, points):

        np_points = numpy.array(points)
        if len(self.__clusters) == 0:
            return []

        self.__metric.enable_numpy_usage()

        np_centers = numpy.array(self.__centers)
        differences = numpy.zeros((len(np_points), len(np_centers)))
        for index_point in range(len(np_points)):
            differences[index_point] = self.__metric(np_points[index_point], np_centers)

        self.__metric.disable_numpy_usage()

        return numpy.argmin(differences, axis=1)

    def get_clusters(self):
        return self.__clusters

    def get_centers(self):
        return self.__centers

    def get_cluster_encoding(self):
        return type_encoding.CLUSTER_INDEX_LIST_SEPARATION

    def get_total_wce(self):
        return self.__total_wce

    def __search_optimal_parameters(self, local_data):
        optimal_wce, optimal_centers, optimal_clusters = float("+inf"), None, None

        for repeat in range(self.__repeat):
            candidates = 5
            if len(local_data) < candidates:
                candidates = len(local_data)

            local_centers = kmeans_plusplus_initializer(
                local_data, 2, candidates, random_state=repeat
            ).initialize()

            kmeans_instance = kmeans(
                local_data,
                local_centers,
                tolerance=self.__tolerance,
                ccore=False,
                metric=self.__metric,
            )
            kmeans_instance.process()

            local_wce = kmeans_instance.get_total_wce()
            if local_wce < optimal_wce:
                optimal_centers = kmeans_instance.get_centers()
                optimal_clusters = kmeans_instance.get_clusters()
                optimal_wce = local_wce

        return optimal_clusters, optimal_centers, optimal_wce

    def __improve_parameters(self, centers, available_indexes=None):

        if available_indexes and len(available_indexes) == 1:
            index_center = available_indexes[0]
            return [available_indexes], self.__pointer_data[index_center], 0.0

        local_data = self.__pointer_data
        if available_indexes:
            local_data = [self.__pointer_data[i] for i in available_indexes]

        local_centers = centers
        if centers is None:
            clusters, local_centers, local_wce = self.__search_optimal_parameters(
                local_data
            )
        else:
            kmeans_instance = kmeans(
                local_data,
                local_centers,
                tolerance=self.__tolerance,
                ccore=False,
                metric=self.__metric,
            ).process()

            local_wce = kmeans_instance.get_total_wce()
            local_centers = kmeans_instance.get_centers()
            clusters = kmeans_instance.get_clusters()

        if available_indexes:
            clusters = self.__local_to_global_clusters(clusters, available_indexes)

        return clusters, local_centers, local_wce

    def __local_to_global_clusters(self, local_clusters, available_indexes):

        clusters = []
        for local_cluster in local_clusters:
            current_cluster = []
            for index_point in local_cluster:
                current_cluster.append(available_indexes[index_point])

            clusters.append(current_cluster)

        return clusters

    def __improve_structure(self, clusters, centers):

        allocated_centers = []
        amount_free_centers = self.__kmax - len(centers)

        for index_cluster in range(len(clusters)):
            (
                parent_child_clusters,
                parent_child_centers,
                _,
            ) = self.__improve_parameters(None, clusters[index_cluster])

            if len(parent_child_clusters) > 1:
                parent_scores = self.__splitting_criterion(
                    [clusters[index_cluster]], [centers[index_cluster]]
                )
                child_scores = self.__splitting_criterion(
                    [parent_child_clusters[0], parent_child_clusters[1]],
                    parent_child_centers,
                )

                split_require = False

                if self.__criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION:
                    if parent_scores < child_scores:
                        split_require = True

                elif (
                    self.__criterion
                    == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH
                ):
                    if parent_scores > child_scores:
                        split_require = True

                if (split_require is True) and (amount_free_centers > 0):
                    allocated_centers.append(parent_child_centers[0])
                    allocated_centers.append(parent_child_centers[1])

                    amount_free_centers -= 1
                else:
                    allocated_centers.append(centers[index_cluster])

            else:
                allocated_centers.append(centers[index_cluster])

        return allocated_centers

    def __splitting_criterion(self, clusters, centers):

        if self.__criterion == splitting_type.BAYESIAN_INFORMATION_CRITERION:
            return self.__bayesian_information_criterion(clusters, centers)

        elif self.__criterion == splitting_type.MINIMUM_NOISELESS_DESCRIPTION_LENGTH:
            return self.__minimum_noiseless_description_length(clusters, centers)

        else:
            assert 0

    def __minimum_noiseless_description_length(self, clusters, centers):

        score = float("inf")

        W = 0.0
        K = len(clusters)
        N = 0.0

        sigma_square = 0.0

        alpha = self.__alpha
        alpha_square = alpha * alpha
        beta = self.__beta

        for index_cluster in range(0, len(clusters), 1):
            Ni = len(clusters[index_cluster])
            if Ni == 0:
                return float("inf")

            Wi = 0.0
            for index_object in clusters[index_cluster]:
                Wi += self.__metric(
                    self.__pointer_data[index_object], centers[index_cluster]
                )

            sigma_square += Wi
            W += Wi / Ni
            N += Ni

        if N - K > 0:
            sigma_square /= N - K
            sigma = sigma_square ** 0.5

            Kw = (1.0 - K / N) * sigma_square
            Ksa = (2.0 * alpha * sigma / (N ** 0.5)) * (
                alpha_square * sigma_square / N + W - Kw / 2.0
            ) ** 0.5
            UQa = W - Kw + 2.0 * alpha_square * sigma_square / N + Ksa

            score = (
                sigma_square * K / N
                + UQa
                + sigma_square * beta * ((2.0 * K) ** 0.5) / N
            )

        return score

    def __bayesian_information_criterion(self, clusters, centers):

        scores = [float("inf")] * len(clusters)
        dimension = len(self.__pointer_data[0])

        sigma_sqrt = 0.0
        K = len(clusters)
        N = 0.0

        for index_cluster in range(0, len(clusters), 1):
            for index_object in clusters[index_cluster]:
                sigma_sqrt += self.__metric(
                    self.__pointer_data[index_object], centers[index_cluster]
                )

            N += len(clusters[index_cluster])

        if N - K > 0:
            sigma_sqrt /= N - K
            p = (K - 1) + dimension * K + 1

            sigma_multiplier = 0.0
            if sigma_sqrt <= 0.0:
                sigma_multiplier = float("-inf")
            else:
                sigma_multiplier = dimension * 0.5 * log(sigma_sqrt)

            for index_cluster in range(0, len(clusters), 1):
                n = len(clusters[index_cluster])

                L = (
                    n * log(n)
                    - n * log(N)
                    - n * 0.5 * log(2.0 * numpy.pi)
                    - n * sigma_multiplier
                    - (n - K) * 0.5
                )

                scores[index_cluster] = L - p * 0.5 * log(N) * self.__c

        return sum(scores)

    def __verify_arguments(self):

        if len(self.__pointer_data) == 0:
            raise ValueError(
                "Input data is empty (size: '%d')." % len(self.__pointer_data)
            )

        if len(self.__centers) == 0:
            raise ValueError(
                "Initial centers are empty (size: '%d')." % len(self.__pointer_data)
            )

        if self.__tolerance < 0:
            raise ValueError(
                "Tolerance (current value: '%d') should be greater or equal to 0."
                % self.__tolerance
            )

        if self.__repeat <= 0:
            raise ValueError(
                "Repeat (current value: '%d') should be greater than 0." % self.__repeat
            )

        if self.__alpha < 0.0 or self.__alpha > 1.0:
            raise ValueError(
                "Parameter for the probabilistic bound Q(alpha) should in the following range [0, 1] "
                "(current value: '%f')." % self.__alpha
            )

        if self.__beta < 0.0 or self.__beta > 1.0:
            raise ValueError(
                "Parameter for the probabilistic bound Q(beta) should in the following range [0, 1] "
                "(current value: '%f')." % self.__beta
            )

