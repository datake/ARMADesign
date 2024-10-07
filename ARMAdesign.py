from itertools import count
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import linear_sum_assignment
import time
import tqdm
import random
from localreg import *
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
import sys
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
from scipy.optimize import minimize, Bounds
from operator import itemgetter
import argparse
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

NUMBER_OF_GRID_TILES_X = 9
NUMBER_OF_GRID_TILES_Y= 9
NUMBER_OF_TIME_STEPS = 20
MAX_MANHATTAN_DISTANCE = 2
def manhattan_distance(p1,p2):
    return np.abs(p1[0]-p2[0]) + np.abs(p1[1]-p2[1])


NUMBER_OF_ORDERS = 100

LOWER_BOUND_WAITING_TIME = 0
UPPER_BOUND_WAITING_TIME = 5
MEAN_WAITING_TIME = 2.5
STANDARD_DEVIATION_WAITING_TIME = 2

######## waiting time
waiting_time_sampler = stats.truncnorm((LOWER_BOUND_WAITING_TIME - MEAN_WAITING_TIME) / STANDARD_DEVIATION_WAITING_TIME, (UPPER_BOUND_WAITING_TIME - MEAN_WAITING_TIME) / STANDARD_DEVIATION_WAITING_TIME, loc=MEAN_WAITING_TIME, scale=STANDARD_DEVIATION_WAITING_TIME)
waiting_times = waiting_time_sampler.rvs(NUMBER_OF_ORDERS)

PROBABILITY_FIRST_GAUSSIAN = 1./3
PROBABILITY_SECOND_GAUSSIAN = 2./3

MEAN_FIRST_GAUSSIAN = [3,3,5]
MEAN_SECOND_GAUSSIAN = [6,6,15]

STANDARD_DEVIATION_FIRST_GAUSSIAN = [2,2,3]
STANDARD_DEVIATION_SECOND_GAUSSIAN = [2,2,3]

LOWER_LIMITS_BY_DIMENSION = [0 for _ in range(3)]
UPPER_LIMITS_BY_DIMENSION = [NUMBER_OF_GRID_TILES_X - 1, NUMBER_OF_GRID_TILES_Y - 1, NUMBER_OF_TIME_STEPS - 1]

# generate samples (size, 3), truncated normal distribution for 3 components
class TruncatedMultivariateNormalInteger():
    def __init__(self, normals):
        self._normals = []
        for [lower, upper, mean, standard_deviation] in normals:
            X = stats.truncnorm(
    (lower - mean) / standard_deviation, (upper - mean) / standard_deviation, loc=mean, scale=standard_deviation)
            self._normals.append(X)
    # size equals 3 (e.g., 3 independent truncated normals per mixture component) in our example
    def rvs(self, size):
        return np.array([[normal.rvs(size=1) for normal in self._normals] for _ in range(size)])

# generate mixture model (size) from weighted truncated normal distribution
class MixtureModel():
    def __init__(self, submodels, weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.submodels = submodels
        self.weights = weights/np.sum(weights)

    def rvs(self, size):
        rvs = []
        for i in range(size):
            random_model = np.random.choice(range(len(self.submodels)), p=self.weights)
            rvs.append(self.submodels[random_model].rvs(size=1))
        return np.round(np.squeeze(np.array(rvs))).astype(int)

# the first component: Gaussian
first_truncated_multivariate_normal = TruncatedMultivariateNormalInteger([[LOWER_LIMITS_BY_DIMENSION[i], UPPER_LIMITS_BY_DIMENSION[i], MEAN_FIRST_GAUSSIAN[i], STANDARD_DEVIATION_FIRST_GAUSSIAN[i]] for i in range(3)])
# the second component: Gaussian
second_truncated_multivariate_normal = TruncatedMultivariateNormalInteger([[LOWER_LIMITS_BY_DIMENSION[i], UPPER_LIMITS_BY_DIMENSION[i], MEAN_SECOND_GAUSSIAN[i], STANDARD_DEVIATION_SECOND_GAUSSIAN[i]] for i in range(3)])
# mix them by weights
mixture_gaussian_model = MixtureModel([first_truncated_multivariate_normal, second_truncated_multivariate_normal],[1./3,2./3])
def spawn_uniformly_x_y_location():
    return [np.random.choice(range(NUMBER_OF_GRID_TILES_X)), np.random.choice(range(NUMBER_OF_GRID_TILES_Y))]

DISCOUNT_FACTOR = 0.9

def discounted_reward_mdp(gamma, T, R):
    total_gamma = 0
    discounted_gamma = 1
    for _ in range(T):
        total_gamma += discounted_gamma
        discounted_gamma *= gamma
    return total_gamma * R / T


time_ = 2

def policy_evaluation(state_transactions, V, N, starting_index, method):
    if V is None:
        V = np.zeros(np.array(UPPER_LIMITS_BY_DIMENSION) - np.array(LOWER_LIMITS_BY_DIMENSION) + [1,1,1]) # [9,9,20], i.e., state: x,y,time
    if N is None:
        N = np.zeros(np.array(UPPER_LIMITS_BY_DIMENSION) - np.array(LOWER_LIMITS_BY_DIMENSION) + [1,1,1])
    for t in range(NUMBER_OF_TIME_STEPS, -1, -1):
        for state, action, reward, next_state in state_transactions[starting_index:]:
            if state[time_] == t:
                N[tuple(state)] += 1
                delta_t = 1
                if action[0] == 1:
                    delta_t += manhattan_distance(state[:2], action[1]) + manhattan_distance(action[1], action[2])
                future_value = 0
                if next_state[time_] < NUMBER_OF_TIME_STEPS:
                    future_value = np.power(DISCOUNT_FACTOR, delta_t) * V[tuple(next_state)]
                if method == 'mdp':
                    modified_reward = discounted_reward_mdp(DISCOUNT_FACTOR, delta_t, reward)
                elif method == 'myopic':
                    modified_reward = reward
                V[tuple(state)] += 1./(N[tuple(state)]) * (future_value + modified_reward - V[tuple(state)])
    return V, N

def value_iteration(theta, q):
    '''
    max_a -\sum_{k=1}^q c_k A_t A_{t-k}
    '''
    M21, M1, M2 = theta[0], theta[1], theta[2]
    c1 = M21+M1
    c2 = M2
    Value = np.random.rand(2, 2) # initialize value functions: [A_{t-1}, A_{t-2}] -> R, where index=0 if A_t=1
    gamma = 0.9
    st = np.matrix([[1, 1], [1, -1], [-1, 1], [-1, -1]]) # (A_t-1, A_t-2)
    # do the value iteration, Value: A_{t-1} * A_{t-2}, max_a [r(s,a,s^')+\gamma Value(s^\prime)]
    Policy = np.ones((2, 2), dtype=int)  # [2, 2] -> {0, 1} to action
    for i in range(100):
        Value_ = deepcopy(Value) # checking the convergence
        temp1 = np.matmul(np.array([-c1, -c2]), st.transpose()).reshape(2, 2) + gamma * Value[(0, 0, 0, 0), (0, 0, 1, 1)].reshape(2, 2)  # A_t = 1
        temp2 = -np.matmul(np.array([-c1, -c2]), st.transpose()).reshape(2, 2) + gamma * Value[(1, 1, 1, 1), (0, 0, 1, 1)].reshape(2, 2)  # A_t = -1
        Value = np.maximum(temp1, temp2)
        # print(np.sum((Value_ - Value) ** 2)) # converge very fast
    print(Value)
    temp1 = np.matmul(np.array([-c1, -c2]), st.transpose()).reshape(2, 2) + gamma * Value[(0, 0, 0, 0), (0, 0, 1, 1)].reshape(2, 2)  # A_t = 1
    temp2 = -np.matmul(np.array([-c1, -c2]), st.transpose()).reshape(2, 2) + gamma * Value[(1, 1, 1, 1), (0, 0, 1, 1)].reshape(2, 2)  # A_t = -1
    Policy[temp1 < temp2] = 0 # 0: take A_t = -1, i.e., action 0 strategy
    return Policy # [2, 2]

def VI_decision(V_, theta, St):
    Action_previous = St # [A_{t-2}, A_{t-1}] different from the definition in value_iteration
    at2 = 1 if Action_previous[0] == 0 else int(Action_previous[0])-1  # a_{t-2}: 0:action-1(index1), 1:action1(index0)
    at1 = 1 if Action_previous[1] == 0 else int(Action_previous[1])-1 # a_{t-1}
    return V_[at1, at2]

ratios_of_served_orders = []

def real_time_order_dispatch_algorithm(num_drivers=50):
    NUM_EPISODES = 5000
    BENCHMARK_RUNS = 50
    NUM_INDEPENDENT_RUNS = NUM_EPISODES - BENCHMARK_RUNS

    number_of_drivers_list = [num_drivers]

    method_list = ['mdp']
    measurement_keypoints = ['Total Revenue','ratio served', 'average distance to driver']

    stored_mdp_V_functions = []


    benchmark_data = np.zeros((len(number_of_drivers_list), len(method_list), len(measurement_keypoints), BENCHMARK_RUNS))


    for number_of_drivers_ind, number_of_drivers in enumerate(number_of_drivers_list): # we fix the number of drivers
        for method_ind, method in enumerate(method_list):

            print('number_of_drivers: {}, method: {}'.format(number_of_drivers, method), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
            transition_data = []

            if method in ['mdp', 'myopic']:
                # Initialize the value and the state counter
                V, N = policy_evaluation(transition_data, None, None, 0, method)
                starting_index = 0

            for episode in tqdm.tqdm(range(NUM_EPISODES)):
                order_driver_distances = []
                revenue_all = []
                number_of_aviable_drivers = []
                number_of_call_orders = []

                if episode >= NUM_INDEPENDENT_RUNS and method in ['mdp', 'myopic']:
                    V, N = policy_evaluation(transition_data, V, N, starting_index, method) # evaluate the value function based on historical
                    starting_index = len(transition)


                destinations = []
                for _ in range(NUMBER_OF_ORDERS):
                    destinations.append(spawn_uniformly_x_y_location())

                orders = list(map(list, zip([False] * NUMBER_OF_ORDERS, mixture_gaussian_model.rvs(NUMBER_OF_ORDERS),
                                            np.round(waiting_times).astype(int), destinations,
                                            range(NUMBER_OF_ORDERS))))
                drivers = []
                for i in range(number_of_drivers):
                    drivers.append([0, spawn_uniformly_x_y_location(), i])

                for t in range(NUMBER_OF_TIME_STEPS):
                    print(drivers)
                    active_orders = [order for order in orders if
                                     (order[0] == False) and (order[1][2] <= t) and (order[1][2] + order[2] >= t)]
                    available_drivers = [driver for driver in drivers if driver[0] <= t]
                    number_of_aviable_drivers.append(len(active_orders))
                    number_of_call_orders.append(len(available_drivers))

                    allowed_match = np.ones((len(active_orders), len(available_drivers)), dtype=bool)
                    for order_count, active_order in enumerate(active_orders):
                        for driver_count, available_driver in enumerate(available_drivers):
                            if manhattan_distance(available_driver[1], active_order[1][:2]) > MAX_MANHATTAN_DISTANCE:
                                allowed_match[order_count, driver_count] = False # make sense

                    if method in ['mdp', 'myopic']:
                        advantage_function = np.zeros((len(active_orders), len(available_drivers)))
                        for order_count, active_order in enumerate(active_orders):
                            for driver_count, available_driver in enumerate(available_drivers):
                                if (allowed_match[order_count, driver_count]):
                                    delta_t = 1 + manhattan_distance(available_driver[1], active_order[1][:2]) + manhattan_distance(active_order[1][:2], active_order[3])
                                    reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER * manhattan_distance(active_order[1][:2], active_order[3])
                                    future_value = 0.
                                    if t + delta_t < NUMBER_OF_TIME_STEPS:
                                        discount = DISCOUNT_FACTOR
                                        if method == 'myopic':
                                            discount = 0.
                                        future_value = np.power(discount, delta_t) * V[active_order[3][0], active_order[3][1], t + delta_t]
                                    current_value = V[available_driver[1][0], available_driver[1][1], t]
                                    modified_reward = reward
                                    if method == 'mdp':
                                        modified_reward = discounted_reward_mdp(DISCOUNT_FACTOR, delta_t, reward)
                                    advantage_function[order_count, driver_count] = future_value + modified_reward - current_value

                    if episode >= NUM_INDEPENDENT_RUNS and method in ['mdp', 'myopic']:
                        penalized_advantage_matrix = advantage_function # fixing the advantage function in the evaluation
                        for i in range(len(active_orders)):
                            for j in range(len(available_drivers)):
                                if not allowed_match[i, j]:
                                    penalized_advantage_matrix[i, j] = - 100 * NUMBER_OF_ORDERS
                        row_ind, col_ind = linear_sum_assignment(-penalized_advantage_matrix)

                    else:
                        distance_matrix = -np.ones(
                            (len(active_orders), len(available_drivers))) * 100 * NUMBER_OF_ORDERS
                        for i in range(len(active_orders)):
                            for j in range(len(available_drivers)):
                                if allowed_match[i, j]:
                                    distance_matrix[i, j] = -manhattan_distance(available_drivers[j][1],
                                                                                active_orders[i][1][:2])
                        row_ind, col_ind = linear_sum_assignment(-distance_matrix) # combination optimization for assignment: the closest distance in total

                    # collect all the matched order and driver
                    matched_order_ind = []
                    matched_driver_ind = []

                    for i in range(len(row_ind)):
                        if row_ind[i] < len(active_orders) and col_ind[i] < len(available_drivers) and allowed_match[row_ind[i], col_ind[i]]:
                            matched_order_ind.append(row_ind[i])
                            matched_driver_ind.append(col_ind[i])

                    revenue_temp = 0
                    for i in range(len(matched_order_ind)):
                        if allowed_match[matched_order_ind[i]][matched_driver_ind[i]]:
                            matched_order = active_orders[matched_order_ind[i]]
                            matched_driver = available_drivers[matched_driver_ind[i]]
                            matched_order[0] = True # indicator function: has been served

                            order_driver_distance = manhattan_distance(matched_driver[1], matched_order[1][:2])

                            assert (order_driver_distance <= 2)

                            order_driver_distances.append(order_driver_distance)

                            delta_t = 1 + manhattan_distance(matched_driver[1], matched_order[1][:2]) + manhattan_distance(matched_order[1][:2], matched_order[3])
                            matched_driver[0] = t + delta_t

                            matched_driver[1] = matched_order[1][:2]
                            reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER * manhattan_distance(matched_order[1][:2], matched_order[3])

                            ######## revenue based on the order distance exactly
                            revenue_temp += manhattan_distance(matched_order[1][:2], matched_order[3])

                            # transition: [[x, y, t], [1, (x, y), destination], reward, [x_, y_, t+delta_t(available time)]]
                            transition = [[matched_driver[1][0], matched_driver[1][1], t],
                                          [1, matched_order[1][:2], matched_order[3]], reward,
                                          [matched_order[3][0], matched_order[3][1], t + delta_t]]
                            transition_data.append(transition.copy())

                    # Set transition data for unmatched drivers: this is because not all drivers can be matched after the combination optimization
                    for i, unmatched_driver in enumerate(available_drivers):
                        if i not in matched_driver_ind: # t=0, still available
                            transition = [[unmatched_driver[1][0], unmatched_driver[1][1], t], [0], 0,
                                          [unmatched_driver[1][0], unmatched_driver[1][1], t + 1]]
                            # transition data is all the order and driver data with different flags mathched or not
                            transition_data.append(transition.copy())

                if episode >= NUM_INDEPENDENT_RUNS:
                    number_of_served_orders = 0
                    for i in range(len(orders)):
                        number_of_served_orders += orders[i][0] # if 1 then true
                    ratio_served = float(number_of_served_orders) / NUMBER_OF_ORDERS
                    benchmark_data[number_of_drivers_ind, method_ind, :,  episode - NUM_INDEPENDENT_RUNS] = [np.sum(np.array(revenue_all)), ratio_served, np.mean(np.array(order_driver_distances))]
                    if method == 'mdp' and episode == NUM_EPISODES - 1:
                        # Used for visualising value functions
                        stored_mdp_V_functions.append(V.copy())
                        np.savez('Value_function_vary_order_driver_{}.npz'.format(number_of_drivers), V.copy())
        return benchmark_data, transition_data



############################ Estimation ####################
def phi_basis(X):
    nx = np.shape(X)[1]
    phi_vector = []
    for i in range(nx):
        phi_vector.append([X[:, i], X[:, i] ** 2, X[:, i] ** 3])
    phi_vector = np.vstack((phi_vector)).T
    return phi_vector

## estimation for NMDP
def Q_est(data, treatment):
    data = data[data['A'] == treatment]
    N = len(np.unique(data['n'].values))
    T = len(np.unique(data['T'].values))
    revenue = np.array(data['revenue'].values).reshape((N, T))
    cumulative_revenue = np.sum(revenue, axis=1)
    S = np.array(data[data['T'] == 0][['orders', 'drivers']])
    phi_S = phi_basis(S)
    phi_S_with_intercept = np.hstack((np.ones(N).reshape(-1, 1), phi_S))
    beta_a = np.linalg.inv(
        phi_S_with_intercept.T.dot(phi_S_with_intercept) + np.identity(np.shape(phi_S_with_intercept)[1]) * 1e-5).dot(
        phi_S_with_intercept.T).dot(cumulative_revenue)
    Q_value = phi_S_with_intercept.dot(beta_a)
    TD_error = (cumulative_revenue - Q_value) ** 2
    return TD_error, beta_a, Q_value

## estimation for TMDP
def Q_est_TMDP(data, treatment, prob_s1):
    data['probS1'] = prob_s1
    data = data[data['A'] == treatment]
    N = len(np.unique(data['n'].values))
    T = len(np.unique(data['T'].values))
    revenue = np.array(data['revenue'].values).reshape((N, T))
    cumulative_revenue = np.repeat(np.sum(revenue, axis=1), T).reshape(N, T) + revenue - np.cumsum(revenue, 1)

    S = np.array(data[['orders', 'drivers']])

    time_h = data['T'].values.reshape(-1, 1)
    indicator_morning = np.array(time_h == 2) + 0
    indicator_night = np.array(time_h == 15) + 0

    indicator_last_time = 1 - np.array(time_h == (T - 1))

    phi_S = phi_basis(S)

    phi_S_with_intercept = np.hstack((np.ones(N * T).reshape(-1, 1), time_h, indicator_morning, indicator_night, phi_S))

    inverse_design = np.linalg.inv(
        phi_S_with_intercept.T.dot(phi_S_with_intercept) + np.identity(np.shape(phi_S_with_intercept)[1]) * 1e-5).dot(
        phi_S_with_intercept.T)

    beta_a = inverse_design.dot(cumulative_revenue.reshape(-1, 1))  #

    Next_all_covariates = indicator_last_time * np.vstack(
        (np.delete(phi_S_with_intercept, 0, axis=0), np.zeros(np.shape(phi_S_with_intercept)[1]).reshape(1, -1)))

    Q_value_difference = (Next_all_covariates - phi_S_with_intercept).dot(beta_a)

    TD_error = revenue.reshape(-1, 1) - Q_value_difference

    density_ratio = phi_S_with_intercept.dot(inverse_design).dot(data['probS1'])

    density_ratio = np.array(density_ratio < 0) * (1 / T) + np.array(density_ratio > 0) * density_ratio

    TD_error_density = (TD_error ** 2) * density_ratio.reshape(-1, 1)

    TD_error_final = np.sum(TD_error_density.reshape(N, T), axis=1)

    return TD_error_final, beta_a, Q_value_difference, density_ratio


# compute the ATE estimator based on three-dimensional observations
def Q_eta_est_poly(data, treatment):
    # extract dataset by the design policy
    data_left = data[data['A'] == treatment]
    # observation: [available order, available drivers, revenue]
    revenue = data_left['revenue'].values
    S = np.array(data_left[['orders', 'drivers']])
    Next_S = np.array(data_left[['ordersNext', 'driversNext']])
    phi_S = phi_basis(S)
    phi_next_S = phi_basis(Next_S)
    revenue_c = revenue - np.mean(revenue)
    phi_S_c = phi_S - np.mean(phi_S, axis=0)
    phi_next_S_c = phi_next_S - np.mean(phi_next_S, axis=0)
    diff_phi_S_c = phi_S_c - phi_next_S_c
    beta_a = np.linalg.inv(diff_phi_S_c.T.dot(phi_S_c)).dot(phi_S_c.T).dot(revenue_c)
    Q_diff_vec = diff_phi_S_c.dot(beta_a)
    eta_est = np.mean(revenue - Q_diff_vec)
    TD_error = revenue - Q_diff_vec - eta_est
    return np.round(eta_est, 3), TD_error, beta_a


# compute the HT estimate of Swithback design based on Eq.4 in their paper
def Switch_estimate(data, switch_m=2):
    series_new = data['revenue'].values
    action = data['A'].values
    action[action == 0] = -1
    N_sample = len(series_new)
    if switch_m == 0:
        switch_n = int(N_sample / 1)  # UR design
    else:
        switch_n = int(N_sample / switch_m)
    P_den = np.ones(N_sample) * 1 / 4
    P_den[:2 * switch_m + 1] = 1 / 2
    P_den[(switch_n - 1) * switch_m - 1:] = 1 / 2
    for kk in range(3, switch_n - 1):
        P_den[kk * switch_m - 1:kk * switch_m + 1] = 1 / 2
    num_p = np.zeros(N_sample)  # nominator with 1
    num_n = np.zeros(N_sample)  # nominator with -1
    # p=m
    for kk in range(switch_m, N_sample + 1):
        if int(action[kk - switch_m:kk].sum()) == switch_m:
            num_p[kk - 1] = 1
        elif int(action[kk - switch_m:kk].sum()) == -switch_m:
            num_n[kk - 1] = 1
        else:
            continue
    ATE_estimator = series_new * (num_p / P_den - num_n / P_den)
    ATE_estimator = ATE_estimator[switch_m:].sum() / (N_sample - switch_m)
    return np.round(ATE_estimator, 3)


def optimize_ARIMA(order_list, data, exog):
    results = []
    for order in order_list:
        try:
            model = SARIMAX(data, order=order, trend='c', exog=exog).fit(disp=-1, maxiter=200)
        except:
            continue
        results.append([order, model.aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True) # lower AIC is better
    if EvaluationOrder:
        print(result_df)
    return result_df

def armax_selection(y, p_max=5, q_max=5, d_max=2, exog=None):
    order_list = []
    for d in range(d_max+1):
        for p in range(p_max+1):
            for q in range(q_max+1):
                order = tuple([p, d, q])
                order_list.append(order)
    result_df = optimize_ARIMA(order_list, data=y, exog=exog)
    return result_df

def varmax_selection(y, p_max=5, q_max=5,  exog=None):
    order_list = []
    for p in range(p_max+1):
        for q in range(q_max+1):
            order = tuple([p, q])
            order_list.append(order)
    results = []
    for order in order_list:
        try:
            model = sm.tsa.VARMAX(y, order=order, trend='c', exog=exog).fit(disp=-1, maxiter=200)
        except:
            continue
        results.append([order, model.aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, q)', 'AIC']
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)  # lower AIC is better
    if EvaluationOrder:
        print(result_df)
    return result_df


def Matmul(mat1, sigma, mat2=None):
    if mat2 is None:
        return np.matmul(mat1, sigma)
    else:
        return np.matmul(np.matmul(mat1, sigma), mat2)

def metric_armax(data):
    # ['n', 'T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']
    orders = np.array(data['orders'])[1:]
    drivers = np.array(data['drivers'])[1:]
    series = np.array(data['revenue'])[:-1] * REWARD_FOR_DISTANCE_PARAMETER  # should drift the reward
    action = np.array(data['A'])[1:]
    action[action == 0.0] = -1.0
    VectorY = pd.DataFrame({'orders': orders, 'drivers':drivers, 'revenue':series ,'action': action})

    if opt.q == 0 and opt.p ==0:
        result = varmax_selection(VectorY[['orders', 'drivers', 'revenue']], p_max=opt.order, q_max=opt.order, exog=action)
        p, q = result['(p, q)'][0]
    else:
        p, q = opt.p, opt.q
    print('p: {}, q: {}'.format(p, q))
    model = sm.tsa.VARMAX(VectorY[['orders', 'drivers', 'revenue']], order=(p, q), trend='c', exog=action).fit(disp=-1, maxiter=200)

    ar = model.coefficient_matrices_var # [p, 3, 3]
    ma = model.coefficient_matrices_vma # [q, 3, 3]
    beta = np.zeros(3)
    beta[0], beta[1], beta[2] = model.params['beta.x1.orders'], model.params['beta.x1.drivers'], model.params['beta.x1.revenue']
    Sigma = np.zeros((3,3))
    Sigma[0,0], Sigma[1,1], Sigma[2,2] = model.params['sqrt.var.orders'], model.params['sqrt.var.drivers'], model.params['sqrt.var.revenue']
    Sigma[0,1], Sigma[1,0]=model.params['sqrt.cov.orders.drivers'], model.params['sqrt.cov.orders.drivers']
    Sigma[0,2], Sigma[2,0]=model.params['sqrt.cov.orders.revenue'], model.params['sqrt.cov.orders.revenue']
    Sigma[1,2], Sigma[2,1]=model.params['sqrt.cov.drivers.revenue'], model.params['sqrt.cov.drivers.revenue']

    Dim = 3

    if p == 0:
        a1, a2, a3, a4, a5 = np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 1:
        a1, a2, a3, a4, a5 = ar[0, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 2:
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 3:
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], ar[2, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 4:
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], ar[2, :, :], ar[3, :, :], np.zeros((Dim, Dim))
    else:  # p==2
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], ar[2, :, :], ar[3, :, :], ar[4, :, :]
    if q == 0:
        theta1, theta2, theta3, theta4, theta5 = np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif q == 1:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((3, Dim))
    elif q == 2:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif q == 3:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], ma[2, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif q == 4:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], ma[2, :, :], ma[3, :, :], np.zeros((Dim, Dim))
    else:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], ma[2, :, :], ma[3, :, :], ma[4, :, :]

    e = np.zeros(Dim)
    e[Dim-1] = 1.0
    A = a1 + a2 + a3 + a4 + a5

    temp5 = Matmul(theta5, Sigma, theta4) + Matmul(theta4, Sigma, theta3) + Matmul(theta3, Sigma, theta2) + Matmul(theta2, Sigma, theta1) + Matmul(theta1, Sigma)
    temp4 = Matmul(theta5, Sigma, theta3) + Matmul(theta4, Sigma, theta2) + Matmul(theta3, Sigma, theta1) + Matmul(theta2, Sigma)
    temp3 = Matmul(theta5, Sigma, theta2) + Matmul(theta4, Sigma, theta1) + Matmul(theta3, Sigma)
    temp2 = Matmul(theta5, Sigma, theta1) + Matmul(theta4, Sigma)
    temp1 = Matmul(theta5, Sigma)
    sum_theta_temp = temp5 + temp4 + temp3 + temp2 + temp1
    sum_theta_temp_minus = -temp5 + temp4 - temp3 + temp2 - temp1
    I_A = np.linalg.inv(np.eye(Dim) - A)
    eI_A = np.matmul(e.reshape(1, -1), I_A) # 1*3
    sum_theta = np.matmul(np.matmul(eI_A, sum_theta_temp), eI_A.reshape(-1, 1))[0,0] # only scalar
    sum_theta_minus = np.matmul(np.matmul(eI_A, sum_theta_temp_minus), eI_A.reshape(-1, 1))[0,0] # only scalar
    ATE_estimator = 2 * np.matmul(e, np.matmul(I_A, beta)) # 2e^t (I-a)^-1 beta

    M21 = np.matmul(np.matmul(eI_A, np.matmul(np.matmul(theta2, Sigma), theta1)), eI_A.reshape(-1, 1))[0,0]
    M1 = np.matmul(np.matmul(eI_A, np.matmul(theta1, Sigma)), eI_A.reshape(-1, 1))[0,0]
    M2 = np.matmul(np.matmul(eI_A, np.matmul(theta2, Sigma)), eI_A.reshape(-1, 1))[0, 0]
    model_info = (np.round(ATE_estimator, 3), np.round(sum_theta, 3),np.round(sum_theta_minus, 3), p, q, M21, M1, M2) # c1=M21+M1, c2=M2

    return model_info

def Markov(data):
    def polynomial(x, coeffs):
        return np.polyval(coeffs, x)

    def objective(x, *args):
        return polynomial(x, *args)

    orders = np.array(data['orders'])[1:]
    drivers = np.array(data['drivers'])[1:]
    series = np.array(data['revenue'])[:-1] * REWARD_FOR_DISTANCE_PARAMETER  # !!! should drift
    action = np.array(data['A'])[1:]
    action[action == 0.0] = -1.0

    VectorY = pd.DataFrame({'orders': orders, 'drivers': drivers, 'revenue': series, 'action': action})
    if opt.p==0 and opt.q==0: # select the optimal one
        result = varmax_selection(VectorY[['orders', 'drivers', 'revenue']], p_max=opt.order, q_max=opt.order, exog=action)
        p, q = result['(p, q)'][0]
    else:
        p, q = opt.p, opt.q # consistent with our strategy in metric_armax
    model = sm.tsa.VARMAX(VectorY[['orders', 'drivers', 'revenue']], order=(p, q), trend='c', exog=action).fit(disp=-1, maxiter=200)

    ar = model.coefficient_matrices_var  # [2, 3, 3]
    ma = model.coefficient_matrices_vma  # [3, 3, 3]
    beta = np.zeros(3)
    beta[0], beta[1], beta[2] = model.params['beta.x1.orders'], model.params['beta.x1.drivers'], model.params['beta.x1.revenue']
    Sigma = np.zeros((3, 3))
    Sigma[0, 0], Sigma[1, 1], Sigma[2, 2] = model.params['sqrt.var.orders'], model.params['sqrt.var.drivers'], model.params['sqrt.var.revenue']
    Sigma[0, 1], Sigma[1, 0] = model.params['sqrt.cov.orders.drivers'], model.params['sqrt.cov.orders.drivers']
    Sigma[0, 2], Sigma[2, 0] = model.params['sqrt.cov.orders.revenue'], model.params['sqrt.cov.orders.revenue']
    Sigma[1, 2], Sigma[2, 1] = model.params['sqrt.cov.drivers.revenue'], model.params['sqrt.cov.drivers.revenue']

    Dim = 3

    if p == 0:
        a1, a2, a3, a4, a5 = np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 1:
        a1, a2, a3, a4, a5 = ar[0, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 2:
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 3:
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], ar[2, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif p == 4:
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], ar[2, :, :], ar[3, :, :], np.zeros((Dim, Dim))
    else:  # p==2
        a1, a2, a3, a4, a5 = ar[0, :, :], ar[1, :, :], ar[2, :, :], ar[3, :, :], ar[4, :, :]
    if q == 0:
        theta1, theta2, theta3, theta4, theta5 = np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif q == 1:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((3, Dim))
    elif q == 2:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif q == 3:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], ma[2, :, :], np.zeros((Dim, Dim)), np.zeros((Dim, Dim))
    elif q == 4:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], ma[2, :, :], ma[3, :, :], np.zeros((Dim, Dim))
    else:
        theta1, theta2, theta3, theta4, theta5 = ma[0, :, :], ma[1, :, :], ma[2, :, :], ma[3, :, :], ma[4, :, :]

    temp5 = Matmul(theta5, Sigma, theta4) + Matmul(theta4, Sigma, theta3) + Matmul(theta3, Sigma, theta2) + Matmul(theta2, Sigma, theta1) + Matmul(theta1, Sigma)
    temp4 = Matmul(theta5, Sigma, theta3) + Matmul(theta4, Sigma, theta2) + Matmul(theta3, Sigma, theta1) + Matmul(theta2, Sigma)
    temp3 = Matmul(theta5, Sigma, theta2) + Matmul(theta4, Sigma, theta1) + Matmul(theta3, Sigma)
    temp2 = Matmul(theta5, Sigma, theta1) + Matmul(theta4, Sigma)
    temp1 = Matmul(theta5, Sigma)

    e = np.zeros(Dim)
    e[Dim-1] = 1.0
    A = a1 + a2 + a3 + a4 + a5
    coeffs = np.zeros(opt.order)
    I_A = np.linalg.inv(np.eye(3) - A)
    eI_A = np.matmul(e.reshape(1, -1), I_A)  # 1*3

    if opt.order == 2:
        coeffs[0] = np.matmul(np.matmul(eI_A, np.matmul(theta2, Sigma)), eI_A.reshape(-1, 1))[0, 0]  # the second order
        first_temp = np.matmul(theta1, Sigma) + np.matmul(np.matmul(theta2, Sigma), theta1)
        coeffs[1] = np.matmul(np.matmul(eI_A, first_temp), eI_A.reshape(-1, 1))[0, 0]  # the first order
    else: #opt.order == 5
        coeffs[0] = np.matmul(np.matmul(eI_A, temp1), eI_A.reshape(-1, 1))[0, 0] # the highest order
        coeffs[1] = np.matmul(np.matmul(eI_A, temp2), eI_A.reshape(-1, 1))[0, 0]
        coeffs[2] = np.matmul(np.matmul(eI_A, temp3), eI_A.reshape(-1, 1))[0, 0]
        coeffs[3] = np.matmul(np.matmul(eI_A, temp4), eI_A.reshape(-1, 1))[0, 0]
        coeffs[4] = np.matmul(np.matmul(eI_A, temp5), eI_A.reshape(-1, 1))[0, 0]

    initial_guess = np.array([0.0])
    bounds = Bounds(-1.0, 1.0)
    optim = minimize(objective, initial_guess, args=(np.append(coeffs, 0.0),), bounds=bounds)  # from higher order to lower order
    alpha_estimate = (optim.x + 1) / 2.0

    return alpha_estimate

def normfun(x, mu, sigma):
    pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return pdf


def Sigma_S_est(S, Next_S, A, TD_error1, TD_error0, pre_S, prob_or):
    nS = np.shape(S)[1]

    density_s_all = []
    density_s = 1

    S_for_1_all = S[np.array(np.where(A == 1))[0, :], :]
    S_for_0_all = S[np.array(np.where(A == 0))[0, :], :]

    ###### calculate sigmas ####################
    sigma_1 = np.sqrt(TD_error1)
    sigma_0 = np.sqrt(TD_error0)

    if type(pre_S) == str:
        if pre_S == 'fitted':
            np_sigma_1 = localreg(S_for_1_all, np.array(sigma_1), x0=S, radius=6)
            np_sigma_0 = localreg(S_for_0_all, np.array(sigma_0), x0=S, radius=6)

            prob_update = np_sigma_1 / (np_sigma_1 + np_sigma_0)

            invalid_probs = np.array(prob_update < 0) + np.array(prob_update > 1) + 0

            prob_update = prob_update * (1 - invalid_probs) + invalid_probs * prob_or
    else:
        np_sigma_1 = localreg(S_for_1_all, np.array(sigma_1), x0=np.array(pre_S).reshape(1, -1), radius=2)
        np_sigma_0 = localreg(S_for_0_all, np.array(sigma_0), x0=np.array(pre_S).reshape(1, -1), radius=2)

        prob_update = prob_or

        if np_sigma_1 > 0 and np_sigma_0 > 0:
            prob_update = np_sigma_1 / (np_sigma_1 + np_sigma_0)

    return prob_update, sigma_1, sigma_0


def real_time_order_dispatch_algorithm_revised(allocation, NUM_EPISODES=500, alpha_estimate=None, simu_i=0, VI=None, carryeffect=None):
    method_list = ['distance', 'mdp', 'myopic']
    transition_data = []
    data_all = []
    prob_or = 0.5
    ind = 0

    for episode in range(NUM_EPISODES):

        # for each day: initialize the first 2 actions uniformly
        if VI is not None and allocation == 8: # MDP design
            (V_, theta) = VI
            q = len(V_.shape)
            if q == 1:
                action_previous = 1  # [0, 1]
            else:
                action_previous = np.zeros(2, dtype=int)
                if episode < NUM_EPISODES / 2:
                    action_previous[0] = 1
                    action_previous[1] = 1

        same_seeds(simu_i*NUM_EPISODES+episode)
        number_of_drivers = 50

        if allocation == 7: # markov optimal
            ind = episode < (NUM_EPISODES / 2)  # equal intialization

        if allocation == 10 and carryeffect is not None: # switchback design: pre-specify all the action
            N_sample = NUM_EPISODES * NUMBER_OF_TIME_STEPS # days * times in each day
            switch_m = carryeffect
            if switch_m == 0:
                switch_n = int(N_sample / 1) # UR design
            else:
                switch_n = int(N_sample / switch_m)
            action_ = np.random.choice(2, switch_n-2) # 1, 0
            action = np.ones(N_sample)
            for kk in range(switch_n-2):
                if kk == 0:
                    action[:2*switch_m] = action_[kk]
                elif kk == switch_n-3:
                    action[(switch_n-2) * switch_m:] = action_[kk]
                else:
                    action[(kk+1)*switch_m:(kk+2)*switch_m] = action_[kk] # Theorem 2 in Bojinov's paper
            action[action == 0] = -1 # action=1 or -1

        destinations = []
        for _ in range(NUMBER_OF_ORDERS):
            # destination is drawn uniformly randomly from the grid
            destinations.append(spawn_uniformly_x_y_location())

        # in orders first entry is boolean corresponding to whether it is served.
        orders = list(map(list, zip([False] * NUMBER_OF_ORDERS, mixture_gaussian_model.rvs(NUMBER_OF_ORDERS),
                                    np.round(waiting_times).astype(int), destinations, range(NUMBER_OF_ORDERS))))
        ##### order: [status, [x, y, startingtime], waitingtime, [x_destination, y_destination], index]
        drivers = []
        for i in range(number_of_drivers):
            drivers.append([0, spawn_uniformly_x_y_location(), i])

        active_orders_next = [order for order in orders if (order[0] == False) and (order[1][2] <= 0) and (order[1][2] + order[2] >= 0)]
        available_drivers_next = [driver for driver in drivers if driver[0] <= 0]

        for t in range(NUMBER_OF_TIME_STEPS):
            # obtain active orders
            active_orders = active_orders_next
            available_drivers = available_drivers_next

            allowed_match = np.ones((len(active_orders), len(available_drivers)), dtype=bool)
            for order_count, active_order in enumerate(active_orders):
                for driver_count, available_driver in enumerate(available_drivers):
                    # only consider drivers whose manhattan distance is slower than 2
                    if manhattan_distance(available_driver[1], active_order[1][:2]) > MAX_MANHATTAN_DISTANCE:
                        allowed_match[order_count, driver_count] = False
            # print(allowed_match)

            ###### allocation methods ######
            if allocation == 3: # AD
                prob = allocation
                ind = 0 if episode % 2 == 0 else 1
                method = method_list[int(ind)] # mdp -> distance

            elif allocation == 2: # AT
                prob = allocation
                ind = t % 2
                method = method_list[int(ind)] # distance, mdp, distance, mdp, .....,

            elif allocation == 4:  # TMDP method
                if episode > int(NUM_EPISODES / 2) and t == 0:
                    data_for_update = pd.DataFrame(np.vstack((data_all)))
                    data_for_update.columns = ['n', 'T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext','driversNext']

                    eta1, TD_error1, beta1 = Q_eta_est_poly(data_for_update, 1)
                    eta0, TD_error0, beta0 = Q_eta_est_poly(data_for_update, 0)

                    #####################
                    prob = np.sqrt(np.mean(TD_error1 ** 2)) / (np.sqrt(np.mean(TD_error1 ** 2)) + np.sqrt(np.mean(TD_error0 ** 2)))
                    prob_or = prob

                    ind = np.random.binomial(1, prob, 1)  # 0.5 for random
                    method = method_list[int(ind)]
                elif episode > int(NUM_EPISODES / 2) and t > 0:
                    method = method_list[int(ind)]  # follow the method of t=0
                else:
                    prob = 0.5
                    ind = episode < NUM_EPISODES / 4  # fixed
                    method = method_list[int(ind)]

            elif allocation == 5:  # NMDP method
                if episode > int(NUM_EPISODES / 2) and t == 0:
                    data_for_update = pd.DataFrame(np.vstack((data_all)))
                    data_for_update.columns = ['n', 'T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']
                    TD_error1, beta_1, Q_value1 = Q_est(data_for_update, 1)
                    TD_error0, beta_0, Q_value0 = Q_est(data_for_update, 0)
                    #####################
                    S = np.array(data_for_update[data_for_update['T'] == 0][['orders', 'drivers']])
                    Next_S = np.array(data_for_update[data_for_update['T'] == 0][['ordersNext', 'driversNext']])
                    A = data_for_update[data_for_update['T'] == 0]['A'].values
                    pre_S = np.array([[len(active_orders), len(available_drivers)]])
                    #####################
                    prob, sigma_1, sigma_0 = Sigma_S_est(S, Next_S, A, TD_error1, TD_error0, pre_S, prob_or)
                    prob_or = prob

                    ind = np.random.binomial(1, prob, 1)  # 0.5 for random
                    method = method_list[int(ind)]
                elif episode > int(NUM_EPISODES / 2) and t > 0:
                    if ind == 1:
                        prob = 1
                    else:
                        prob = 0
                    method = method_list[int(ind)]
                else:
                    if t == 0:
                        prob = 0.5
                    else:
                        prob = float(ind)
                    ind = episode < NUM_EPISODES / 4  # fixed
                    method = method_list[int(ind)]

            elif allocation == 6:  # epsilon greedy
                if episode > int(NUM_EPISODES / 2):
                    data_for_update = pd.DataFrame(np.vstack((data_all)))
                    data_for_update.columns = ['n', 'T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext','driversNext']
                    eta1, TD_error1, beta1 = Q_eta_est_poly(data_for_update, 1)
                    eta0, TD_error0, beta0 = Q_eta_est_poly(data_for_update, 0)
                    #####################
                    S = np.array(data_for_update[['orders', 'drivers']])
                    Next_S = np.array(data_for_update[['ordersNext', 'driversNext']])
                    A = data_for_update['A'].values
                    pre_S = np.array([[len(active_orders), len(available_drivers)]])
                    #####################
                    pre_S_basis = phi_basis(pre_S)
                    Q1 = pre_S_basis.dot(beta1)
                    Q0 = pre_S_basis.dot(beta0)
                    epsilon = 0.1
                    greedy_prob = np.random.uniform()
                    if greedy_prob < epsilon:
                        prob = 0.5
                        ind = np.random.binomial(1, prob, 1)  # 0.5 for random
                        method = method_list[int(ind)]
                    else:
                        prob = 1
                        ind = 1 * (Q1 >= Q0) + 0 * (Q1 < Q0)
                        method = method_list[int(ind)]
                    prob_or = prob
                else:
                    prob = 0.5
                    ind = np.random.binomial(1, prob, 1)
                    method = method_list[int(ind)]

            elif allocation == 7: # markov policy
                prob = allocation
                if ind==0:
                    ind = np.random.binomial(1, 1-alpha_estimate, 1) # P(0|0)=alpha
                else:
                    ind = np.random.binomial(1, alpha_estimate, 1) # P(1|1)=alpha
                method = method_list[int(ind)]

            elif allocation == 8: # MDP policy
                prob = allocation
                if t > 1:
                    ind = VI_decision(V_, theta, action_previous) # action previous: (1,) q=1, (2,) q=2
                    # update action_previous
                    if q == 1:
                        action_previous = ind
                    else: # q=2,
                        action_previous[0] = action_previous[1] # A_t-2
                        action_previous[1] = ind # A_t-1
                else:
                    ind = action_previous[t] # t=0: distance, t=1: MDP allocation
                method = method_list[int(ind)]

            elif allocation == 10: # switchdesign
                prob = allocation
                ind_ = action[episode*NUMBER_OF_TIME_STEPS+t] # 1 or -1
                ind = 0 if ind_==-1 else ind_
                method = method_list[int(ind)]  # mdp -> distance

            else: # UR, always 1, always 0
                prob = allocation
                ind = np.random.binomial(1, prob, 1)  # 0.5 for random, 0: distance, 1: mdp
                method = method_list[int(ind)]


            # computation of advantage function based on the saved value function
            if method in ['mdp', 'myopic']:
                # Could also initialize with - infinity.
                advantage_function = np.zeros((len(active_orders), len(available_drivers)))
                for order_count, active_order in enumerate(active_orders):
                    for driver_count, available_driver in enumerate(available_drivers):
                        if (allowed_match[order_count, driver_count]):
                            # the pickup time
                            delta_t = 1 + manhattan_distance(available_driver[1],
                                                             active_order[1][:2]) + manhattan_distance(
                                active_order[1][:2], active_order[3])
                            reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER * manhattan_distance(
                                active_order[1][:2], active_order[3])
                            # If the completion time is later than the last time step, we just stop set the future value to zero
                            future_value = 0.
                            if t + delta_t < NUMBER_OF_TIME_STEPS:
                                discount = DISCOUNT_FACTOR
                                if method == 'myopic':
                                    discount = 0.
                                future_value = np.power(discount, delta_t) * V[active_order[3][0], active_order[3][1], t + delta_t]
                            current_value = V[available_driver[1][0], available_driver[1][1], t]
                            modified_reward = reward
                            if method == 'mdp':
                                modified_reward = discounted_reward_mdp(DISCOUNT_FACTOR, delta_t, reward)
                            advantage_function[order_count, driver_count] = future_value - current_value + modified_reward

            row_ind = []
            col_ind = []

            # The initial independent runs should use the 'distance' policy to find the matching.
            # Later runs could either use 'mdp', 'myopic' or 'distance' policy
            if method in ['mdp', 'myopic']:
                penalized_advantage_matrix = advantage_function
                for i in range(len(active_orders)):
                    for j in range(len(available_drivers)):
                        if not allowed_match[i, j]:
                            penalized_advantage_matrix[i, j] = - 100 * NUMBER_OF_ORDERS
                row_ind, col_ind = linear_sum_assignment(-penalized_advantage_matrix)

            else:
                # Use distance matrix to compute assignment
                distance_matrix = -np.ones((len(active_orders), len(available_drivers))) * 100 * NUMBER_OF_ORDERS
                for i in range(len(active_orders)):
                    for j in range(len(available_drivers)):
                        if allowed_match[i, j]:
                            distance_matrix[i, j] = -manhattan_distance(available_drivers[j][1],
                                                                        active_orders[i][1][:2])
                row_ind, col_ind = linear_sum_assignment(-distance_matrix)

            matched_order_ind = []
            matched_driver_ind = []

            for i in range(len(row_ind)):
                if row_ind[i] < len(active_orders) and col_ind[i] < len(available_drivers) and allowed_match[
                    row_ind[i], col_ind[i]]:
                    matched_order_ind.append(row_ind[i])
                    matched_driver_ind.append(col_ind[i])

            # print(f"Matched orders in iteration {t}")
            revenue_temp = 0
            for i in range(len(matched_order_ind)):
                if allowed_match[matched_order_ind[i]][matched_driver_ind[i]]:
                    matched_order = active_orders[matched_order_ind[i]]
                    matched_driver = available_drivers[matched_driver_ind[i]]

                    matched_order[0] = True

                    order_driver_distance = manhattan_distance(matched_driver[1], matched_order[1][:2])

                    # continue to run the code only when the assertion is satisfied. Stop and return an error otherwise
                    assert (order_driver_distance <= 2)

                    # order_driver_distances.append(order_driver_distance)

                    delta_t = 1 + manhattan_distance(matched_driver[1], matched_order[1][:2]) + manhattan_distance(matched_order[1][:2], matched_order[3])
                    matched_driver[0] = t + delta_t


                    # Append to transition data.
                    reward = BASE_REWARD_PER_TRIP + REWARD_FOR_DISTANCE_PARAMETER * manhattan_distance(matched_order[1][:2], matched_order[3])
                    revenue_temp = revenue_temp + manhattan_distance(matched_order[1][:2], matched_order[3])

                    transition = [[matched_driver[1][0], matched_driver[1][1], t],
                                  [1, matched_order[1][:2], matched_order[3]], reward,
                                  [matched_order[3][0], matched_order[3][1], t + delta_t]]
                    transition_data.append(transition.copy())
                    matched_driver[1][:2] = matched_order[3]  # this line should be put at the end of the transition. However, we dont use it as our observation is a whole perspective


            # Set transition data for unmatched drivers
            for i, unmatched_driver in enumerate(available_drivers):
                if i not in matched_driver_ind:
                    transition = [[unmatched_driver[1][0], unmatched_driver[1][1], t], [0], 0,
                                  [unmatched_driver[1][0], unmatched_driver[1][1], t + 1]]
                    transition_data.append(transition.copy())

            active_orders_next = [order for order in orders if
                                  (order[0] == False) and (order[1][2] <= t + 1) and (order[1][2] + order[2] >= t + 1)]
            available_drivers_next = [driver for driver in drivers if driver[0] <= t + 1] # only available for the time

            #### for each t, compute the all revenue
            data_temp = [[episode, t, len(active_orders), len(available_drivers), int(ind), float(prob), revenue_temp,
                          len(active_orders_next), len(available_drivers_next)]]

            data_all.append(data_temp)

    data_final = pd.DataFrame(np.vstack((data_all)))
    data_final.columns = ['n', 'T', 'orders', 'drivers', 'A', 'Prob', 'revenue', 'ordersNext', 'driversNext']

    return data_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_dri', type=int, default=50, help='number of drivers, default=50')
    parser.add_argument('--num_sim', type=int, default=30, help='number of simulation, default=200 for ARMAX, default=30 for VARMAX')
    parser.add_argument('--num_epi', type=int, default=50, help='number of episodes/days, default=50')
    parser.add_argument('--num_epi_order', type=int, default=0, help='number of episodes to determine the order, e.g., 500')
    parser.add_argument('--num_epi_ate', type=int, default=0, help='number of episodes to evaluate the true ATE, e.g., 10000')
    parser.add_argument('--p', type=int, default=2, help='if p not = 0, set the order, default=2')
    parser.add_argument('--q', type=int, default=2, help='if q not = 0, set the order, default=2')
    parser.add_argument('--order', type=int, default=2, help='p_max/q_max: default=2')
    parser.add_argument('--num', type=int, default=1, help='number of runs for the use of file names')
    opt = parser.parse_args()
    print(opt)

    BASE_REWARD_PER_TRIP = 1
    REWARD_FOR_DISTANCE_PARAMETER = 1
    num_drivers = opt.num_dri
    simu = opt.num_sim
    NUM_EPISODES = opt.num_epi
    gamma = 0.99
    CollectHistoricalData = False
    EvaluationTrueATE = True if opt.num_epi_ate > 0 else False
    EvaluationOrder = True if opt.num_epi_order > 0 else False

    #### create value function #######
    if CollectHistoricalData:
        benchmark_data, transition_data = real_time_order_dispatch_algorithm(num_drivers)  # generate the value function, make sure that the settings are the same
        sys.exit()
    #### compute ATE estimator
    print('Loading data......')
    Value = np.load('Value_function_vary_order_driver_{}.npz'.format(num_drivers))  # load value function
    V = Value['arr_0'] # (9,9,20)

    #################### step 1: evaluate the true ATE
    True_ate = 2.24
    if EvaluationTrueATE:
        # 1: mdp, 0: distance
        print('Working on the true ATE............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        dd1 = real_time_order_dispatch_algorithm_revised(1, NUM_EPISODES=opt.num_epi_ate)
        dd0 = real_time_order_dispatch_algorithm_revised(0, NUM_EPISODES=opt.num_epi_ate)
        dd_combine = pd.concat([dd0, dd1], axis=0)
        ATE_emprical_true = np.mean(dd1['revenue']) - np.mean(dd0['revenue'])
        print('MDP reward: {:.5f}, distance reward: {:.5f}, True ATE is {:.5f}'.format(np.mean(dd1['revenue']), np.mean(dd0['revenue']), ATE_emprical_true))
        print('Finished the True ATE: ', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        sys.exit()
    else:
        ATE_emprical_true = True_ate * REWARD_FOR_DISTANCE_PARAMETER

    #################### step 2: apply desin policy, compute ATE estimator based on offline data
    Metric = 'ARMAX'
    columns_name = ['Method', "ATE_estimator", 'sum_theta',  'sum_theta_minus', 'p', 'q', 'M21', 'M1', 'M2']

    # select the optimal order
    if EvaluationOrder:
        assert opt.q == 0 and opt.p == 0 # to select the optimal order in VARMAX / ARMAX
        eval_episodes = opt.num_epi_order
        print('Working on {} episodes to determine the orders!'.format(eval_episodes))
        print('Collecting data on action=1', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        dd1 = real_time_order_dispatch_algorithm_revised(1, NUM_EPISODES=eval_episodes)
        print('Collecting data on action=0', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        dd0 = real_time_order_dispatch_algorithm_revised(0, NUM_EPISODES=eval_episodes)
        dd_combine = pd.concat([dd0, dd1], axis=0)
        print('Fit the model!!!', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        ATE_order = metric_armax(dd_combine) # select the best model
        ATE_order = pd.DataFrame(np.array(ATE_order).reshape(1, -1))
        ATE_order.columns = columns_name[1:]
        pd.set_option('display.max_columns', None)
        print(ATE_order)
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        sys.exit()


    ####################### Step 3: compare different designs

    ### Method 1: Switch design with different carry effect
    # Method 1.1: Switch 2
    ATE_Switch2 = []
    print('Working on Switch2 design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(10, NUM_EPISODES, simu_i=i, carryeffect=2)
        ATE_temp = metric_armax(dd_temp)
        ATE_Switch2.append(ATE_temp)
    # Method 1.2: Switch 5
    ATE_Switch5 = []
    print('Working on Switch2 design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(10, NUM_EPISODES, simu_i=i, carryeffect=5)
        ATE_temp = metric_armax(dd_temp)
        ATE_Switch5.append(ATE_temp)
    # Method 1.3: Switch 10
    ATE_Switch10 = []
    print('Working on Switch2 design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(10, NUM_EPISODES, simu_i=i, carryeffect=10)
        ATE_temp = metric_armax(dd_temp)
        ATE_Switch10.append(ATE_temp)


    ### Method 2: MDP optimal (ours)
    ATE_MDP = []
    print('Working on Optimal MDP Design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    dd_temp_estimate = real_time_order_dispatch_algorithm_revised(3, 50) # the result is same for 50 and 500
    print('Estimating collected dataset !')
    ATE_temp_estimate = metric_armax(dd_temp_estimate)
    print('Finishied Estimating collected dataset !')
    action = dd_temp_estimate['A']
    q_ = 2
    theta = np.array(ATE_temp_estimate[-3:])  # M21, M1, M2
    V_ = value_iteration(theta, q_)  # action: 0, 1
    print('------------------', V_)

    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        ATE_temp = metric_armax(dd_temp)
        ATE_MDP.append(ATE_temp)

    ### Method 3: Markov (ours)
    ATE_Markov = []
    print('To initialize the optimal alpha')
    dd_temp_estimate = real_time_order_dispatch_algorithm_revised(3, 50)
    alpha_estimate = Markov(dd_temp_estimate)  # 1.0: deterministic poliy
    print('the optimal alpha is: ', alpha_estimate[0])
    alpha_estimate = alpha_estimate[0]  # intialization， alpha depends on theta, which depends on action, but can be viewed as constant when ATE is small (b is small)
    print('Working on Optimal Markov design, with alpha: {:.3f}............'.format(alpha_estimate),
          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(7, NUM_EPISODES, alpha_estimate=alpha_estimate, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_Markov.append(ATE_temp)


    ### Method 4: AD
    ATE_AD = []
    print('Working on AD design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(3, NUM_EPISODES, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_AD.append(ATE_temp)

    ### Method 5: UR
    ATE_UR = []
    print('Working on UR design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(0.5, NUM_EPISODES, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_UR.append(ATE_temp)

    ### Method 6: AT
    ATE_AT = []
    print('Working on AT design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(2, NUM_EPISODES, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_AT.append(ATE_temp)

    ### Method 7: epsilon_greedy
    ATE_greedy = []
    print('Working on Greedy design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(6, NUM_EPISODES, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_greedy.append(ATE_temp)

    ### Method 8: TMDP
    ATE_TMDP = []
    print('Working on TMDP design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(4, NUM_EPISODES, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_TMDP.append(ATE_temp)

    ### Method 9: NMDP
    ATE_NMDP = []
    print('Working on NMDP design............', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    for i in tqdm.tqdm(range(simu)):
        time_start = time.time()
        dd_temp = real_time_order_dispatch_algorithm_revised(5, NUM_EPISODES, simu_i=i)
        ATE_temp = metric_armax(dd_temp)
        ATE_NMDP.append(ATE_temp)

    ################## Step 4: save the results
    keys = ['ATE_Switch2','ATE_Switch5','ATE_Switch10','ATE_MDP', 'ATE_Markov','ATE_AD','ATE_UR','ATE_AT','ATE_greedy','ATE_TMDP','ATE_NMDP']
    index_keys = []
    for i in keys:
        for k in range(simu):
            index_keys.append(i)

    ATE_all = pd.concat([pd.DataFrame(ATE_Switch2), pd.DataFrame(ATE_Switch5), pd.DataFrame(ATE_Switch10), pd.DataFrame(ATE_MDP),pd.DataFrame(ATE_Markov),
                         pd.DataFrame(ATE_AD), pd.DataFrame(ATE_UR), pd.DataFrame(ATE_AT),
                         pd.DataFrame(ATE_greedy), pd.DataFrame(ATE_TMDP), pd.DataFrame(ATE_NMDP)], axis=0)
    ATE_all = ATE_all.reset_index()
    ATE_all = ATE_all.iloc[:,1:] # remove the index
    ATE_all = pd.concat([pd.DataFrame(index_keys), ATE_all], axis=1)
    ATE_all.columns = columns_name
    ATE_pd = ATE_all[['Method', 'ATE_estimator']].copy()
    ATE_pd['MSE'] = (ATE_pd['ATE_estimator'] - ATE_emprical_true) ** 2
    ATEs_MSE_ave = ATE_pd.groupby('Method')['MSE'].mean().sort_values(ascending=True)
    print(ATEs_MSE_ave)
    print(opt)
    print('alpha: ', alpha_estimate, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

    if opt.p == 0 and opt.q == 0:
        ATE_all.to_excel('ARMAdesign_dri{}_epi{}_sim{}_num{}.xlsx'.format(num_drivers, NUM_EPISODES, simu, opt.num), index=False, header=True)
    else:
        print('Determine the order !!!!') # we directly determine the order of ARMAX
        ATE_all.to_excel('ARMAdesign_dri{}_epi{}_sim{}_num{}_p{}q{}.xlsx'.format(num_drivers, NUM_EPISODES, simu, opt.num, opt.p, opt.q), index=False, header=True)


