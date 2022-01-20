import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import norm
from typing import List, Union
from scipy.interpolate import splrep, splev
from scipy.linalg import expm
import ctmc
import torch
from VAE_fit import VAE
import json
import os
import pickle
from multiprocessing import Pool

DEFAULT_TAU_NAMES = ["1M", "2M", "3M", "6M", "9M", "1Y", "3Y", "5Y"]
DEFAULT_TAU = np.array([0.08333333, 0.16666667, 0.25      , 0.5       , 0.75      ,
       1.        , 3.        , 5.        ])
DEFAULT_BOUNDS = np.array([
    [-5, -0.5],
    [-0.5, 0.5],
    [-4, 2],
    [-2, 1]
])
DEFAULT_DELTAS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

def GetPRfromKF(KF: Union[np.ndarray, float], sigma: float, T: float) -> Union[np.ndarray, float]:
    """
    Returns the call price ratio (c/(S_0 e^(-rf))) from the data given the IV.

    :param KF: List of KFs to evaluate the call price ratio at
    :param sigma: The IV given by the data
    :param T: The time to maturity
    :returns: Array of the call price ratio evaluated at each KF
    """
    d1 = (-np.log(KF) + (sigma**2) * T / 2) / (sigma * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))
    return norm.cdf(d1) - KF * norm.cdf(d2)

def ImpVolModel(KF: float, T: float, PR: float, sig_0: float = 0.1, a = None, b = None) -> float:
    """
    Calculates the IV given a specific call price ratio at a particular KF and maturity

    :param KF: The specific KF to evaluate it at
    :param T: Time to maturity
    :param PR: The call price ratio
    :return: The implied volatility
    """
    
    def PriceError(vol):
        if PR == 0:
            PR_temp = 1e-12
        else:
            PR_temp = PR
        return GetPRfromKF(KF, vol, T) - PR_temp

    return optimize.root_scalar(PriceError, x0=0.1, x1=0.2).root


def BSplineCV (KFs: np.ndarray, IVs: np.ndarray, num_cv_evals: int = 10) -> tuple:
    """
    Fits a standard degree 3 B-spline to the data with smoothness set to 0

    :param KFs: The KFs (x-val) of the data points
    :param IVs: The IVs (y-val) of the data points
    :param num_cv_evals: The number of points in the smoothness range to test
    """
    num_points = len(IVs)
    if np.std(IVs) == 0:
        weight = 1
    else:
        weight = 1/np.std(IVs)

    #Fits the full dataset with the best smoothness value
    tck = splrep(x=KFs, y=IVs, w=np.ones(num_points) * weight, s=0.0, full_output=False)

    return tck

def gen_surface(taus, deltas, pi, A, mu, sig, ms, kf_arr):
    """
    Generate IV surface given model parameters (only allows original fixed TTM)
    """
    
    r_f = 0
    num_t = len(taus)
    num_d = len(deltas)
    IV_arr = np.zeros((num_t,num_d))

    for i in range(num_t):
        model = ctmc.Model(
            taus[:i+1],
            pi, 
            A[:i+1,:,:], 
            mu[:i+1,:], 
            sig[:i+1,:]
        )

        if i == 0:
            kf_adj = np.exp(-ms[i]) 
        else:
            kf_adj = np.exp(-np.sum(ms[:i+1]))
                
        PR_model_full = (kf_adj ** -1) * model.price(1, kf_arr[i,:]*kf_adj)
                
        PR_model_full = np.clip(PR_model_full, a_min=1e-5, a_max=None)
        
        for j in range(num_d):
            IV_arr[i,j] = ImpVolModel(kf_arr[i,j], taus[i], PR_model_full[j])

    
    return np.array(IV_arr)


def construct_params(tensor_data, NY, NM, calc_mu = True, inc_pi=True):
    """
    Reconstructs CTMC parameters based on given tensor of parameters
    """
    
    
    num_days = tensor_data.size(0)

    pi = (torch.ones(tensor_data.size(0), NY)/NY).numpy()
    mu = tensor_data[:,:NY*NM].view(-1, NM, NY).numpy()
    sig = torch.exp(tensor_data[:, NY*NM: 2*NY*NM]).view(-1, NM, NY).numpy()
    lam = torch.exp(tensor_data[:, 2*NY*NM :]).view(-1, NM, NY).numpy()
    
    all_mat_list = []
    for j in range(num_days):
        day_mat_list = []
        for k in range(NM):
            cur_A = np.zeros((NY, NY))
            lambdas = lam[j,k,:]
            
            if calc_mu and not inc_pi:
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
            else:
                for i in range(NY):
                    if i == 0:
                        cur_A[i, 0] = -lambdas[i]
                        cur_A[i, 1] = lambdas[i]
                    elif i == NY - 1:
                        cur_A[-1, -2] = lambdas[-1]
                        cur_A[-1, -1] = -lambdas[-1]
                    else:
                        cur_A[i, i - 1] = lambdas[i] / 2
                        cur_A[i, i] = -lambdas[i]
                        cur_A[i, i + 1] = lambdas[i] / 2
            day_mat_list.append(cur_A)
        all_mat_list.append(day_mat_list)
    A_list = np.array(all_mat_list)
    
    return pi, mu, sig, A_list, lam

def check_kf_arr(kf_arr):
    """
    Checks to see if surface is valid (ordering is correct)
    """
    
    # Check if any values are nan
    if np.isnan(kf_arr).any():
        return False
    
    # Checks rows
    for i in range(kf_arr.shape[0]):
        for j in range(1, kf_arr.shape[1]):
            if kf_arr[i,j] > kf_arr[i,j-1]:
                return False
    # Checks Columns
    cols = [0,1]
    for j in cols:
        for i in range(1, kf_arr.shape[0]):
            if kf_arr[i,j] < kf_arr[i-1,j]:
                return False
            
    cols = [3,4]
    for j in cols:
        for i in range(1, kf_arr.shape[0]):
            if kf_arr[i,j] > kf_arr[i-1,j]:
                return False
    return True

def load_net(net_dir, NY, NM, norm_mean, norm_std):
    """
    Loads the VAE using outputted network files
    """
    
    with open(net_dir + "param_sum.txt") as json_file: 
        params = json.load(json_file) 

    latent_size = params['latent_size']
    hidden_dims = params['hidden_dim']
    

    abs_idx = None
    vae = VAE(NY*NM*3, latent_size, 1, hidden_dims, norm_mean=norm_mean, norm_std=norm_std)
    vae.encoder.load_state_dict(torch.load(net_dir + "encoder"))
    vae.fc_mu.load_state_dict(torch.load(net_dir + "fc_mu"))
    vae.fc_var.load_state_dict(torch.load(net_dir + "fc_var"))
    vae.decoder_input.load_state_dict(torch.load(net_dir + "decoder_input"))
    vae.decoder.load_state_dict(torch.load(net_dir + "decoder"))
    vae.final_layer.load_state_dict(torch.load(net_dir + "final_layer"))

    vae = vae.float()
    vae.eval()
    return vae

def sample_latent_based(data_tensor, vae, num_s):    
    """
    Samples from latent space by uniformly sampling from historical posterior
    """
    # uniform sample from historical data, then sample from encoding
    idx = torch.randint(data_tensor.size(0), (1,num_s))
    sample = torch.squeeze(data_tensor[idx,:])
    mu, log_var = vae.encode(sample)
    sample = vae.reparameterize(mu, log_var)
    return vae.decode(sample)

def scatter_data(data1, data2, x_lab, y_lab, output_dir, save = False, tau_idx=list(range(8)), graph_bounds = DEFAULT_BOUNDS, file_name="scatter2.png"):
    """
    Creates 2-D scatter plots of two parameters using fitted data.
    """
    
    tau_names = DEFAULT_TAU_NAMES
    matplotlib.rcParams.update({'xtick.labelsize' : 14, 'ytick.labelsize' : 14})
    fig = plt.figure(figsize = (10,10))
    
    min_x = (data1[:,:-1,:].min(), data1[:,:-1,:].max())
    min_y = (data2[:,:-1,:].min(), data2[:,:-1,:].max())
    for i, k in enumerate(tau_idx):
        plt.subplot(3,3, i + 1)
        plt.scatter(data1[:,k,2], data2[:,k,2], color = 'b',label = 'State 2', s=8, alpha = 0.5)
        plt.scatter(data1[:,k,1], data2[:,k,1], color = 'g',label = 'State 1', s=8, alpha = 0.5)
        plt.scatter(data1[:,k,0], data2[:,k,0], color = 'r',label = 'State 0', s=8, alpha = 0.5)
        plt.ylabel(y_lab, fontsize = 14)
        plt.xlabel(x_lab, fontsize = 14)
        plt.legend()
        plt.title("Maturity: " + tau_names[k], fontsize = 14)
        #print(min(min_y_lim[0], min_y[0]))
        if k < 7:
            plt.ylim(graph_bounds[0,0],graph_bounds[0,1])
            plt.xlim(graph_bounds[1,0],graph_bounds[1,1])
        else:
            plt.ylim(graph_bounds[2,0], graph_bounds[2,1])
            plt.xlim(graph_bounds[3,0], graph_bounds[3,1])
        plt.legend()
    plt.tight_layout()
    plt.show()
    
    if save:
        fig.savefig(output_dir + file_name)


def compute_mean(tau, pi, A, mu, sig):
    """
    Helper for the gen_curves function
    """
    
    NM = len(tau)
    prod = np.matmul(pi, expm(tau[0] * (A[0] + np.diag(mu[0]))))
    for i in range(1, NM):
        prod = np.matmul(prod, expm((tau[i] - tau[i-1])*(A[i] + np.diag(mu[i]))))
    return np.sum(prod)


def compute_m(tau, pi, A, mu, sig, prev_mean):
    """
    Helper for the gen_curves function
    """
    mean = compute_mean(tau, pi, A, mu, sig)
        
    if prev_mean is None:
        return -np.log(mean), mean
    else:
        return -np.log(mean) + np.log(prev_mean), mean


def gen_curves(gen_data, file_path="", plot=True):
    """
    Generates all parameters/values needed to produce an IV surface. No actual computations done
    except to compute relevant values for the drift.
    """
    
    num_gen_days = len(gen_data)
    tau = DEFAULT_TAU

    pis, mus, sigs, As, lams = construct_params(gen_data.detach(), 3, 8, calc_mu = True, inc_pi=False)

    m_arr = []
    mean_arr = []
    
    for i in range(num_gen_days):
        m_list = []
        mean_list = []
        for j in range(len(tau)):
            if len(mean_list) == 0:
                last_mean = None
            else:
                last_mean = mean_list[-1]
            m, mean = compute_m(tau[:j+1], pis[i], As[i,:j+1,:,:], mus[i,:j+1,:], sigs[i,:j+1,:], last_mean)
            m_list.append(m)
            mean_list.append(mean)
            
        m_arr.append(m_list)
        mean_arr.append(mean_list)
 
    
    return tau, pis, mus, As, sigs, lams, np.array(m_arr), np.array(mean_arr)

def find_KFs(model, tau, r_f, delta, kf_adj, t_idx):
    """
    Find the appropriate KF value given a single delta
    """
    
    
    def Objective(kf):
        model_price = (kf_adj ** -1) * model.price(1, kf*kf_adj)
        
        if model_price < 0:
            model_price = 1e-4
        
        iv = ImpVolModel(kf, tau, model_price)
        LHS = norm.ppf(np.exp(r_f * tau) * delta)
        RHS = (-np.log(kf) + 0.5 * tau * iv **2)/(iv * np.sqrt(tau))
        return LHS - RHS
    
    try:
        x0, r = optimize.brentq(Objective, a= adj_kf_days[t_idx,0], b= adj_kf_days[t_idx,1],full_output=True)
    except:
        x0, r = optimize.newton(Objective, x0 = 0.99, x1 = 1.01, full_output=True, disp=False)

        if not r.converged:
            x0, r = optimize.newton(Objective, x0 = 1.4, x1 = 1.5, full_output=True, disp=False)

        if not r.converged:
            x0, r = optimize.newton(Objective, x0 = 0.5, x1 = 0.6, full_output=True ,disp=False)
    
    if r.converged:
        return x0
    else:
        print("ERROR NO CONVERGENCE!!!!!!")
        print(x0)
        print(r)
        return x0

def get_KFs(taus, deltas, pi, A, mu, sig, ms):
    """
    Compute grid of KF values for grid of Deltas
    """
    
    r_f = 0
    num_t = len(taus)
    num_d = len(deltas)
    KF_arr = np.zeros((num_t,num_d))
    
    for i in range(num_t):
        model = ctmc.Model(
            taus[:i+1],
            pi, 
            A[:i+1,:,:], 
            mu[:i+1,:], 
            sig[:i+1,:]
        )
        
        if i == 0:
            kf_adj = np.exp(-ms[i]) 
        else:
            kf_adj = np.exp(-np.sum(ms[:i+1]))

        # Compute for kfs based adjustment
        for j in range(num_d):
            KF_arr[i,j] = find_KFs(model, 
                                   taus[i],
                                   r_f, 
                                   deltas[j],
                                   kf_adj,
                                   i
                                  )
        
    return np.array(KF_arr)

def get_init_values(mu, sigma, A, pi = None):
    """
    Creates array of initial parameter values (x0) based on given data

    """
    if pi is None:
        return np.hstack([mu, sigma, -np.diag(A)])
    else:
        return np.hstack([mu, sigma, -np.diag(A), pi[:-1]])

def convert_to_deltas (ID, sample, taus=DEFAULT_TAU, deltas=DEFAULT_DELTAS):
    """
    Converts a strike-tte grid of iv to a delta-tte grid, uses some precomputed bounds 
    for easy optimizations
    """
    # Create some bounds for Delta-KF conversion via root finder
    with open("Data/kf_days.pickle", 'rb') as handle:
        kf_days = pickle.load(handle)[ID + '_Max']

    adj_kf_days = kf_days
    adj_kf_days[:,0] = adj_kf_days[:,0] - 0.5 * (1 - adj_kf_days[:,0])
    adj_kf_days[:,1] = adj_kf_days[:,1] + 0.5 * (adj_kf_days[:,1] - 1)

    # Compute kf_arr for each sample
    _, pi, mu, A, sig, _, m_arr, mean_arr = gen_curves(sample, "", plot=False)

    inputs = list(zip(
        np.tile(taus, (sample.size(0), 1)),
        np.tile(deltas, (sample.size(0), 1)),
        pi, A, mu, sig, 
        m_arr
    ))

    pool = Pool(os.cpu_count())
    all_kf_arr = np.array(pool.starmap(get_KFs, inputs, chunksize=1))

    # Removes any generated arrays that does not pass sanity check
    failed_kfs = []
    valid_idx = []

    for i in range(sample.size(0)):
        if not check_kf_arr(all_kf_arr[i]):
            failed_kfs.append(all_kf_arr[i])
        else:
            valid_idx.append(i)

    num_good_samples = len(valid_idx)

    # Compute kf_arr for each sample
    inputs_iv = list(zip(
        np.tile(taus, (num_good_samples, 1)),
        np.tile(deltas, (num_good_samples, 1)),
        pi[valid_idx], A[valid_idx], mu[valid_idx], sig[valid_idx], m_arr[valid_idx], all_kf_arr[valid_idx],
    ))

    pool = Pool(os.cpu_count())
    all_iv_arr = np.array(pool.starmap(gen_surface, inputs_iv, chunksize=1))
    
    return all_iv_arr
