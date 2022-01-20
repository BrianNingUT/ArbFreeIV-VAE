import numpy as np
from scipy.linalg import expm
from helpers import GetPRfromKF, ImpVolModel, BSplineCV
from DensityEstimation import DensityEstimation
from scipy.stats import wasserstein_distance
import scipy.optimize as optimize
from scipy.interpolate import splev
import matplotlib.pyplot as plt
import ctmc as CTMC


def param_separator(params, NY):
    """
    Separates the params into its corresponding parts
    """
    pi = np.ones(NY)/NY
    mu = params[: NY]
    sigma = params[NY:2 * NY]
    lambdas = params[2 * NY: 3 * NY]
    cur_A = np.zeros((NY, NY))

    for i in range(NY):
        if i == 0:
            cur_A[i, 0] = -lambdas[i]
            cur_A[i, 1] = lambdas[i]/2
            cur_A[i, -1] = lambdas[i]/2
        elif i == NY - 1:
            cur_A[-1, -2] = lambdas[-1]/2
            cur_A[i, 0] = lambdas[i]/2
            cur_A[-1, -1] = -lambdas[-1]
        else:
            cur_A[i, i - 1] = lambdas[i] / 2
            cur_A[i, i] = -lambdas[i]
            cur_A[i, i + 1] = lambdas[i] / 2

    return pi, mu, sigma, cur_A


class FitCTMC:
    def __init__(self):
        self.pi0 = None
        self.mu0 = None
        self.sigma0 = None
        self.A_list = None
        self.m0 = None
        self.prev_EY = None
        self.model = None
        self.tau = None
        self.KFs = None
        self.NY = None
        self.xs = None
        self.sorted_KFs = None
        self.was_p = None
        self.PRs = None
        self.cond_data_density = None

    def Objective(self, params):
        """
        Objective function fed into optimizer
        """
        
       # Separates parameters and init model from C code then obtain prices
        if len(self.tau) == 1:
            pi, mu, sigma, cur_A = param_separator(params, self.NY)
            new_A_list = [cur_A]
            self.model.UpdateParameters(np.array(self.tau), pi, np.array(new_A_list), np.expand_dims(mu,0), np.expand_dims(sigma,0))
            cur_EY = self.model.mean()
            m = -np.log(cur_EY)
                
            kf_adj = np.exp(-m) 
            lu_int = m
            PR_model = (kf_adj ** -1) * self.model.price(1, self.KFs*kf_adj)

        else:
            pi, mu, sigma, cur_A = param_separator(params, self.NY)
            new_A_list = self.A_list.copy()
            new_A_list.append(cur_A)
            self.model.UpdateParameters(np.array(self.tau), pi, np.array(new_A_list), np.vstack([self.mu0, mu]),
                                   np.vstack([self.sigma0, sigma]))
            cur_EY = self.model.mean()
            m = -np.log(cur_EY) + np.log(self.prev_EY[-1])
                
            kf_adj = np.exp(-np.sum(self.m0) - m)
            lu_int = np.sum(self.m0) + m
            PR_model = (kf_adj ** -1) * self.model.price(1, self.KFs*kf_adj)

        # Compute the density then take the wassertein distance wrt data density, if statement for telling C code 
        # how detailed to do integration
        if self.tau[-1] <= 1:
            model_density = np.multiply(1 / (self.xs), np.maximum(self.model.pdf(np.log(1), np.log(self.xs)-lu_int,100, True), 0))
        else:
            model_density = np.multiply(1 / (self.xs), np.maximum(self.model.pdf(np.log(1), np.log(self.xs)-lu_int,100, False), 0))
            
        # Compute the conditional density (conditioned on the range of xs given
        cond_model_density = model_density / (np.average(model_density) * (self.sorted_KFs[-1] - self.sorted_KFs[0]))

        if np.any(np.isnan(model_density)):
            return np.NaN

        # Calc the squared loss from differences between model prices and actual then add wasserstein penalty
        IV_error = np.sqrt(sum((PR_model - self.PRs) ** 2))

        was_dis = wasserstein_distance(u_values=self.xs, v_values=self.xs, u_weights=self.cond_data_density,
                                       v_weights=cond_model_density)

        J = IV_error + self.was_p * was_dis + np.abs(m)

        return J
    
    def ordering(self, params):
        """
        Impose some ordering to the states
        """
        pi, mu, sigma, cur_A = param_separator(params, self.NY)
        cons = []
        for i in range(1, self.NY):
            cons.append(mu[i] - mu[i-1])
        return np.array(cons)
    
    def find_best_rstart(self, num_s, pi=None):
        """
        Used to randomly sample over parameter space and find the best objective value to start optimization at
        """
        best_y = np.inf
        best_x0 = None
        rand_starts_l = np.random.uniform(size=(num_s, self.NY))
        rand_starts_m = np.random.uniform(size=(num_s, self.NY))*0.8 - 0.4 
        rand_starts_s = np.random.uniform(size=(num_s, self.NY))*0.2 + 0.01
        for i in range(num_s):
            x0_rs = np.concatenate([rand_starts_m[i,:], rand_starts_s[i,:], rand_starts_l[i,:]], axis=0)

            y = self.Objective(x0_rs)
            if y < best_y:
                best_y = y
                best_x0 = x0_rs
        x0 = best_x0
        
        if x0 is None:
            print([num_s, self.A_list, self.mu0, self.sigma0, self.pi0, pi])
            x0 = self.find_best_rstart(num_s, pi)
            
        return x0

    def FitCTMC(
            self,
            KFs: np.ndarray,
            tau: np.ndarray,
            IV: np.ndarray,
            A_list: list,
            mu0: np.ndarray,
            sigma0: np.ndarray,
            pi0: np.ndarray,
            m0: np.ndarray,
            pre_mean_Y: np.ndarray,
            NY: int,
            was_p: float,
            x0: np.ndarray,
            num_points: int = 100,
            tol: float = None,
            init_rs = 0,
            fail_attempts = 5,
    ) -> dict:
        """
        Main fitting function, given the KFs obtained from the data and prior fitted pi/mu/sigma, determine the
        best pi/mu/sigma for the current maturity options.

        :param KFs: The KFs calculated form the data
        :param tau: The list of all tte until current maturity
        :param IV: The list of IVs of the current maturity options
        :param A_list: The array of initial transition matrices
        :param pi0: Fitted pi's for all previous maturities
        :param mu0: Fitted mu's for all previous maturities
        :param sigma0: Fitted sigma's for all previous maturities
        :param m0: Fitted overall mean for previous maturities
        :param pre_mean_Y: Previous values for E(exp(Y_tau))
        :param NY: Number of states at each maturity
        :param c_pen: Penalty for having a non-zero complex part of solution to mu (very rare)
        :param was_p: Penalty for the Wasserstein distance
        :param x0: Initial starting point during optimization, must be of length NY * 3 - 2
        :param num_points: Number of points used to evaluate densities
        :param tol: Tolerence used in minimizer (default is 1e-9)
        :param init_rs: Number of initial random starts used to compute best optimizer starting loc
        :param fail_attempts: Number of attempts used to obtain losses within a good bound before picking best one
        :return: A dictionary summarizing results
        """
        self.pi0 = pi0
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.A_list = A_list
        self.tau = tau
        self.KFs = KFs
        self.NY = NY
        self.was_p = was_p
        self.m0 = m0
        self.prev_EY = pre_mean_Y

        # Fit B-splines and calc conditional density
        sorted_idx = np.argsort(KFs)
        sorted_KFs = KFs[sorted_idx]
        sorted_IVs = IV[sorted_idx]
        tck = BSplineCV(sorted_KFs, sorted_IVs, 10)
        xs = np.linspace(sorted_KFs[0], sorted_KFs[-1], num_points)
        data_density = DensityEstimation(tck,xs,tau[-1])
        cond_data_density = data_density/(np.average(data_density)*(sorted_KFs[-1] - sorted_KFs[0]))
        if np.any(cond_data_density < 0):
            cond_data_density = np.abs(cond_data_density)

        self.cond_data_density = cond_data_density
        
        # Calc the call price ratios based on given KFs from data
        PRs = np.zeros(len(KFs))
        for i in range(len(PRs)):
            PRs[i] = GetPRfromKF(KFs[i], IV[i], tau[-1])
        self.PRs = PRs

        # Create a default model instance
        self.model = CTMC.Model(np.array(tau), np.zeros(self.NY), np.zeros((len(tau),self.NY,self.NY)),
                                np.zeros((1, self.NY)), np.zeros((1, self.NY)))
        self.sorted_KFs = sorted_KFs
        self.xs = xs
            
        # Calc ObjF at all points in domain randomly
        if init_rs > 0:
            x0 = self.find_best_rstart(init_rs, pi0)

        # Add some reasonable bounds to avoid divergence or odd behaviours
        lb = np.zeros((3*NY, 1))
        ub = np.zeros((3*NY, 1))
        lb[: NY] = -2
        lb[NY: 2 * NY] = 0.01
        lb[2 * NY: 3 * NY] = 1e-5
        ub[: NY] = 2
        ub[NY: 2 * NY] = 10
        ub[2 * NY: 3 * NY] = 5

        constraints = [
            {
                'type': 'ineq',
                'fun': self.ordering
            }
        ]
        # Ensures initial value is within bounds
        delta = 1e-8
        for i in range(len(x0)):
            if x0[i] <= lb[i][0]:
                x0[i] = lb[i][0] + delta
            if x0[i] >= ub[i][0]:
                x0[i] = ub[i][0] - delta

        # This block is used to compute some values to ensure fits are good enough otherwise restart with random points
        best_result = None
        best_loss = np.inf
        success = False
        for i in range(fail_attempts):

            # Optimization using predefined constraints and objective function
            if tol is not None:
                result = optimize.minimize(fun=self.Objective, x0=x0, bounds=np.hstack((lb, ub)), tol=tol,
                                           constraints=constraints, options={'eps': 1.4901161193847656e-09})
            else:
                result = optimize.minimize(fun=self.Objective, x0=x0, bounds=np.hstack((lb, ub)),
                                           constraints=constraints, options={'eps': 1.4901161193847656e-09})

            results = {}

            if len(tau) == 1:
                pi1, mu1, sigma1, cur_A = param_separator(result.x, NY)
                self.model.UpdateParameters(np.array(tau), pi1, np.array([cur_A]), np.expand_dims(mu1,0), np.expand_dims(sigma1,0))
                cur_EY = self.model.mean()
                m = -np.log(cur_EY)
                kf_adj = np.exp(-m)
                lu_int = m

                results['mu'] = np.array([mu1])  # estimated mu (including prior tte's)
                results['sigma'] = np.array([sigma1])  # estimated sigma (including prior tte's)
                results['A'] = [cur_A]  # estimated list of generator matrices
                results['m'] = np.array([m])
                results['EYs'] = np.array([cur_EY])

            else:
                pi1, mu1, sigma1, cur_A = param_separator(result.x, NY)
                new_A_list = A_list.copy()
                new_A_list.append(cur_A)
                self.model.UpdateParameters(np.array(tau), pi1, np.array(new_A_list), np.vstack([mu0,mu1]), np.vstack([sigma0,sigma1]))
                cur_EY = self.model.mean()
                m = -np.log(cur_EY) + np.log(self.prev_EY[-1])
                kf_adj = np.exp(-np.sum(self.m0) - m)
                lu_int = np.sum(self.m0) +m

                results['mu'] = np.vstack((mu0, mu1))  # estimated mu (including prior tte's)
                results['sigma'] = np.vstack((sigma0, sigma1))  # estimated sigma (including prior tte's)
                results['A'] = new_A_list  # estimated list of generator matrices
                results['m'] = np.append(self.m0, m)
                results['EYs'] = np.append(self.prev_EY, cur_EY)

            PR_model = (kf_adj ** -1) * self.model.price(1, KFs*kf_adj)
            if self.tau[-1] <= 1:
                model_density = np.multiply(1 / (xs), np.maximum(self.model.pdf(np.log(1), np.log(xs)-lu_int,100, True), 0))
            else:
                model_density = np.multiply(1 / (xs), np.maximum(self.model.pdf(np.log(1), np.log(xs)-lu_int,100, False), 0))

            PR_model_full = (kf_adj ** -1) * self.model.price(1, xs*kf_adj)
            IV_model = np.zeros(num_points)
            for j in range(num_points):
                IV_model[j] = ImpVolModel(xs[j], tau[-1], PR_model_full[j])

            # Calculate some statistics for the result and save to dictionary
            IV_error = np.sqrt(sum((PR_model - PRs) ** 2))
            spline_IVs = splev(x=xs, tck=tck, der=0)
            
            # Compute the final density then take the wassertein distance wrt data density
            cond_model_density = model_density / (np.average(model_density) * (sorted_KFs[-1] - sorted_KFs[0]))
            was_dis = wasserstein_distance(u_values=xs, v_values=xs, u_weights=cond_data_density, v_weights=cond_model_density)

            # Saves the diagnosis variates for current result
            results['pi'] = pi1  # estimated pi (include prior tte's)
            results['IV_model'] = IV_model  # IV from model evaluated num_points number of times being min and max KFs
            results['IV_data'] = IV  # IV from the actual data (exactly 5 points)
            results['KFs'] = xs  # KFs used to evaluate IV_model (1 x num_points)
            results['data_density'] = data_density  # density of price ratio using fitted splines on original data (eval at KFs)
            results['model_density'] = model_density  # density of price ratio based on fitted model (eval at KFs)
            results['spline'] = tck  # tuple of representing fitted splines
            results['was_dis'] = was_dis  # wasserstein distance between the conditional data and model density
            results['spline_IV'] = spline_IVs  # IVs evaluated at all KFs using spline approx. (1 x num_points)
            results['sorted_IV'] = sorted_IVs  # original data IV to match sorted KFs (1x5)
            results['sorted_KFs'] = sorted_KFs  # sorted KFs (ascending) (1x5)
            results['fit_error'] = IV_error  # squared error of difference between model and data c/(s_0 e^-rf)
            results['was_error'] = was_dis * was_p
            results['tau'] = tau  # list of tte being used in the model
            results['PR_model'] = PR_model  # The price ratios determined from the model
            results['rcons'] = self.model.mean() - np.exp(-np.sum(results['m']))

            #######
            # rerun with random starts if fit_error is too high adjusted for tau
            # Setting for success condition as a factor of TTM
            err_lim = 0.0001 * (self.tau[-1]/(1/12))
            #######
            
            if (IV_error > err_lim) or (was_dis * was_p > err_lim):
                
                total_err = IV_error + was_dis * was_p
                if total_err < best_loss:
                    best_loss = total_err
                    best_result = results
            else:
                success = True
                break

        # Outputs summary of best results
        if not success:
            print("Random limit Reached. Copying best result instead")
            results = best_result

        return results