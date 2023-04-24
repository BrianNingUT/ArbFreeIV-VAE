import torch
import numpy as np
import pandas as pd
import pickle
import json
import ctmc
import os
from copy import deepcopy as dc
from latent_space import sample_latent_based, get_data_latent
from helpers import gen_surface, create_data_tensor, extract_data, get_KFs
from loading_and_graphics import load_net, gen_curves, compute_m
from portfolio_val import *
from multiprocessing import Pool


def compute_portfolio_val (port_grid, kf_grid, params, plus_day = False):
    """
    Params must be of num_param_sets x params
    Returns Price Ratios (Price/Forward Spot Price)
    
    """
    
    set_num = params.size(0)
    tau, pi, mu, A, sig, _, m_arr, mean_arr = gen_curves(params)
    
    
    pf_price = []
    for i in range(set_num):
        prices = []
        for j in range(8):
            m = ctmc.Model(
                tau[:j+1] + (1/365) * plus_day, # Adds a day to all maturities if selected
                pi[i], 
                A[i][:j+1,:,:], 
                mu[i][:j+1,:], 
                sig[i][:j+1,:]
            )
            
            if np.isnan(m_arr).any():
                model_price =  np.Nan
            else:
                kf_adj = np.exp(-np.sum(m_arr[i][:j+1]))

                model_price = (kf_adj ** -1) * m.price(1, kf_grid[j,:]*kf_adj)
                
            prices.append(model_price)
            
        prices = np.array(prices)
        pf_price.append(np.sum(prices*port_grid))
        
    return np.array(pf_price)

def transform_raw(tensor_data):
    new_data = tensor_data.detach().clone()
    new_data[:,3*8:] = torch.exp(tensor_data[:,3*8:])
    return new_data

# Used to generate new surface using same parameters but different kf grid

def construct_params_no_trans(tensor_data, NY, NM, calc_mu = True, inc_pi=True):
    """
    Reconstructs CTMC parameters based on given tensor of parameters
    """
    num_days = tensor_data.size(0)
    
    pi = (torch.ones(tensor_data.size(0), NY)/NY).numpy()
    mu = tensor_data[:,:NY*NM].view(-1, NM, NY).numpy()
    sig = tensor_data[:, NY*NM: 2*NY*NM].view(-1, NM, NY).numpy()
    lam = tensor_data[:, 2*NY*NM :].view(-1, NM, NY).numpy()
    
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

def gen_curves_no_trans(gen_data, file_path="", plot=True):
    """
    Generates all parameters/values needed to produce an IV surface. No actual computations done
    except to compute relevant values for the drift.
    """
    
    num_gen_days = len(gen_data)
    tau = np.array([0.08333333, 0.16666667, 0.25      , 0.5       , 0.75      ,
       1.        , 3.        , 5.        ])

    pis, mus, sigs, As, lams = construct_params_no_trans(gen_data.detach(), 3, 8, calc_mu = True, inc_pi=False)

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

def convert_param_to_x0(param, NY, NM):
    all_x0 = []
    for i in range(NM):
        new_x0 = np.hstack([param[i*NY:i*NY+NY],param[i*NY + NM*NY:i*NY+NY+ NM*NY],param[i*NY+ NM*NY*2:i*NY+NY+ NM*NY*2]])
        all_x0.append(new_x0)
    return np.array(all_x0)

def convert_surf(surf):
    base_ivs = np.array([
        surf[0,0],
        surf[0,2],
        surf[3,0],
        surf[3,2],
        surf[5,0],
        surf[5,2],
        surf[7,0],
        surf[7,1],
        surf[7,2],
    ])
    return base_ivs

def compute_delta_samples(date_idx, save_dir, net_path, param_path):
    ID = 'AUD'
    save_path = net_path
    results_sum_dir = param_path
    NY=3;NM=8

    # initialize network and load parameters parameters

    with open(save_path + "param_sum.txt") as json_file: 
        params = json.load(json_file) 

    latent_size = params['latent_size']
    hidden_dims = params['hidden_dim']

    taus = np.array([0.08333333, 0.16666667, 0.25, 0.5, 0.75,1., 3., 5.])
    deltas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    # Loads surfaces for testing set
    with open("all_cur_train_valid_days_new.pickle", 'rb') as handle:
        all_days = pickle.load(handle)
    with open('IV_data_' + ID + '.pickle', 'rb') as handle:
        raw_data = pickle.load(handle)

    # Creates normalized data
    full_train_data = []
    train_date_list = []

    for file in sorted(all_days['train']):
        if file.endswith(".pickle"):
            try:
                with open(results_sum_dir + file, 'rb') as handle:
                    data_info = pickle.load(handle)
                full_train_data.append(data_info)
                train_date_list.append(pd.Timestamp(file[:10]))
            except:
                continue;

    mu, sig, a, pis, lam = extract_data(full_train_data)

    train_sum_data = {'mu': mu, 'sig': sig, 'pi':pis, 'lam':lam}
    train_data_tensor = create_data_tensor(train_sum_data)           # REMEMBER TO CHANGE IF MODIFYING THE TRAINING DATA
    original_train_data_tensor = dc(train_data_tensor)
    data_mean = torch.mean(train_data_tensor, dim=0)
    data_std = torch.std(train_data_tensor, dim=0)
    train_data_tensor = (train_data_tensor - data_mean)/data_std


    full_test_data = []
    test_date_list = []

    for file in sorted(all_days['valid_days']):
        if file.endswith(".pickle"):
            try:
                with open(results_sum_dir + file, 'rb') as handle:
                    data_info = pickle.load(handle)
                full_test_data.append(data_info)
                test_date_list.append(pd.Timestamp(file[:10]))
            except:
                continue;

    mu, sig, a, pis, lam = extract_data(full_test_data)

    test_sum_data = {'mu': mu, 'sig': sig, 'pi':pis, 'lam':lam}
    test_data_tensor = create_data_tensor(test_sum_data)           # REMEMBER TO CHANGE IF MODIFYING THE TRAINING DATA
    original_test_data_tensor = dc(test_data_tensor)
    test_data_tensor = (test_data_tensor - data_mean)/data_std


    vae = load_net(save_path, NY, NM, data_mean, data_std)
    vae.eval()
    
    
    mu, log_var = vae.encode(train_data_tensor)
    mu_diff = torch.diff(mu, n=1, dim=0)
    # fit mv normal then sample 1,000 points then decode and renormalize for parameters
    # This effectively assumes your day 0 iv surface to be the decoded origin
    # We append the origin as the last point to decode
    mean = np.mean(mu_diff.detach().numpy(), axis=0)
    cov = np.cov(mu_diff.detach().numpy(), rowvar=0)

    sample = torch.from_numpy(np.random.multivariate_normal(mean, cov, 1000)).float()

    # Center around day 500's encoding mean instead
    center_idx = date_idx
    mu, log_var = vae.encode(train_data_tensor)
    sample = sample + mu[center_idx]
    decoded_sample = vae.decode(sample)
    #decoded_sample = decoded_sample * data_std + data_mean
    
    ### Computes average kf for 25 and 75 deltas
    av_kf = np.zeros((8,2))
    for day in full_train_data:
        kfs = []
        if len(day) == 8:
            for i in range(8):
                kf_temp = day[i]['sorted_KFs']
                kfs.append([kf_temp[1], kf_temp[3]])
        else:
            for i in range(8):
                kf_temp = day[i*2+1]['sorted_KFs']
                kfs.append([kf_temp[1], kf_temp[3]])
        kfs = np.array(kfs)
        av_kf += kfs
    av_kf = av_kf/len(full_train_data)
    kfs = av_kf
    ttms = np.array(["1M", "2M", "3M", "6M", "9M", "1Y", "3Y", "5Y"])
    strike_grid = kfs

    base_params = transform_raw(original_train_data_tensor[center_idx,:].unsqueeze(0)).detach().numpy()
    
    tau, pi, mu, A, sig, _, m_arr, mean_arr = gen_curves_no_trans(torch.from_numpy(base_params))

    # Add at-the-money to strike-grid

    new_strike_grid = np.zeros([8,3])
    new_strike_grid[:,0] =  strike_grid[:,0]
    new_strike_grid[:,2] =  strike_grid[:,1]
    new_strike_grid[:,1] = 1

    surf = []
    for i in range(1):
        surf.append(gen_surface(taus, deltas, pi[i], A[i], mu[i], sig[i], m_arr[i], new_strike_grid))

    surf = np.squeeze(np.array(surf))
    
    # Translation of entire surface up and down by some average volatility

    # First compute average volatility in training set
    daily_mean_vol = []
    for day in full_train_data:
        all_vol = []
        if len(day) == 8:
            for i in range(8):
                all_vol.append(day[i]['sorted_IV'])
        else:
            for i in range(8):
                all_vol.append(day[i*2+1]['sorted_IV'])
        all_vol = np.array(all_vol)
        daily_mean_vol.append(np.mean(all_vol))
    av_vol = np.mean(daily_mean_vol)

    # Now copy and shift entire surface by +/- 2 and 3 sigmas
    baseline_surfs = np.tile(surf, (9, 1, 1))

    for i in range(9):
        if i < 4:
            baseline_surfs[i,:,:] = baseline_surfs[i,:,:] - np.std(daily_mean_vol) * (4 - i)
        elif i > 4:
            baseline_surfs[i,:,:] = baseline_surfs[i,:,:] + np.std(daily_mean_vol) * (i - 4)
            
    baseline_surfs = np.clip(baseline_surfs, 0.001, np.inf) # clip negative surface values
            
    ############
    #Use portfolio solver to contruct delta and gamma neutral portfolio
    ############
    # Construct basic portfolio 
    baseline_port = {}
    baseline_port['K'] = np.array([
        strike_grid[0,0],
        strike_grid[0,1],
        strike_grid[3,0],
        strike_grid[3,1],
        strike_grid[5,0],
        strike_grid[5,1],
        strike_grid[7,0],
        1,
        strike_grid[7,1],
    ])
    baseline_port['t'] = np.array([
        taus[0],
        taus[0],
        taus[3],
        taus[3],
        taus[5],
        taus[5],
        taus[7],
        taus[7],
        taus[7],
    ])
    baseline_port['t'] = baseline_port['t'] + 1/365 # Add an extra day 

    baseline_port['b_coeff'] = np.array([1, -1, -1, 1, 1, -1, 1, -2, 1])
    #baseline_port['b_coeff'] = np.array([1, -1, -1, 1])


    base_ivs = np.array([
        surf[0,0],
        surf[0,2],
        surf[3,0],
        surf[3,2],
        surf[5,0],
        surf[5,2],
        surf[7,0],
        surf[7,1],
        surf[7,2],
    ])

    types = [True,True,True,True,True,True,True,True,True]

    a = opt_port(baseline_port, 1, base_ivs, types, np.array([-3, 2, 2]))
    coeff = np.insert(a, 0, 1)
    baseline_val = port_val(baseline_port, 1, base_ivs, types)
    
    # Construct portfolio for computation
    #new_grid = np.insert(strike_grid, 1, np.ones(8), axis = 1)
    new_portfolio_grid = np.vstack([
        np.array([1, 0, -1])* coeff[0],
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([-1, 0, 1])* coeff[1],
        np.array([0, 0, 0]),
        np.array([1, 0, -1])* coeff[2],
        np.array([0, 0, 0]),
        np.array([1, -2, 1])* coeff[3],
    ])
    
    pf_all = compute_portfolio_val(new_portfolio_grid, new_strike_grid , decoded_sample)
    pf_base = compute_portfolio_val(new_portfolio_grid, new_strike_grid , original_train_data_tensor[center_idx,:].unsqueeze(0), plus_day=True)
    pf_actual = compute_portfolio_val(new_portfolio_grid, new_strike_grid , original_train_data_tensor[center_idx+1,:].unsqueeze(0))
    
    new_baseline_port = dc(baseline_port)
    new_baseline_port['t'] = np.array([
        taus[0],
        taus[0],
        taus[3],
        taus[3],
        taus[5],
        taus[5],
        taus[7],
        taus[7],
        taus[7],
    ])
    
    # Compute portfolio value for other baseline cases
    port_vals = []
    for i in range(9):
        a, _, _ = port_val(new_baseline_port, 1, convert_surf(baseline_surfs[i]), types)
        port_vals.append(a)
        
    results = {}
    results['sample_val'] = pf_all
    results['line_val'] = port_vals
    results['line_deltas'] = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    results['baseline_port'] = baseline_port
    results['new_baseline_port'] = new_baseline_port
    results['baseline_val'] = baseline_val
    results['baseline_surfs'] = baseline_surfs
    results['actual_pf_val'] = pf_actual

    date = str(train_date_list[center_idx].to_pydatetime().date())

    with open(save_dir + date + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def compute_delta_samples_fixed(date_idx, save_dir, net_path, param_path, base_port=None):
    ID = 'AUD'
    save_path = net_path
    results_sum_dir = param_path
    NY=3;NM=8

    # initialize network and load parameters parameters

    with open(save_path + "param_sum.txt") as json_file: 
        params = json.load(json_file) 

    latent_size = params['latent_size']
    hidden_dims = params['hidden_dim']

    taus = np.array([0.08333333, 0.16666667, 0.25, 0.5, 0.75,1., 3., 5.])
    deltas = np.array([0.1, 0.25, 0.5, 0.75, 0.9])

    # Loads surfaces for testing set
    with open("all_cur_train_valid_days_new.pickle", 'rb') as handle:
        all_days = pickle.load(handle)
    with open('IV_data_' + ID + '.pickle', 'rb') as handle:
        raw_data = pickle.load(handle)

    # Creates normalized data
    full_train_data = []
    train_date_list = []

    for file in sorted(all_days['train']):
        if file.endswith(".pickle"):
            try:
                with open(results_sum_dir + file, 'rb') as handle:
                    data_info = pickle.load(handle)
                full_train_data.append(data_info)
                train_date_list.append(pd.Timestamp(file[:10]))
            except:
                continue;

    mu, sig, a, pis, lam = extract_data(full_train_data)

    train_sum_data = {'mu': mu, 'sig': sig, 'pi':pis, 'lam':lam}
    train_data_tensor = create_data_tensor(train_sum_data)           # REMEMBER TO CHANGE IF MODIFYING THE TRAINING DATA
    original_train_data_tensor = dc(train_data_tensor)
    data_mean = torch.mean(train_data_tensor, dim=0)
    data_std = torch.std(train_data_tensor, dim=0)
    train_data_tensor = (train_data_tensor - data_mean)/data_std


    full_test_data = []
    test_date_list = []

    for file in sorted(all_days['valid_days']):
        if file.endswith(".pickle"):
            try:
                with open(results_sum_dir + file, 'rb') as handle:
                    data_info = pickle.load(handle)
                full_test_data.append(data_info)
                test_date_list.append(pd.Timestamp(file[:10]))
            except:
                continue;

    mu, sig, a, pis, lam = extract_data(full_test_data)

    test_sum_data = {'mu': mu, 'sig': sig, 'pi':pis, 'lam':lam}
    test_data_tensor = create_data_tensor(test_sum_data)           # REMEMBER TO CHANGE IF MODIFYING THE TRAINING DATA
    original_test_data_tensor = dc(test_data_tensor)
    test_data_tensor = (test_data_tensor - data_mean)/data_std


    vae = load_net(save_path, NY, NM, data_mean, data_std)
    vae.eval()
    
    
    mu, log_var = vae.encode(train_data_tensor)
    mu_diff = torch.diff(mu, n=1, dim=0)
    # fit mv normal then sample 1,000 points then decode and renormalize for parameters
    # This effectively assumes your day 0 iv surface to be the decoded origin
    # We append the origin as the last point to decode
    mean = np.mean(mu_diff.detach().numpy(), axis=0)
    cov = np.cov(mu_diff.detach().numpy(), rowvar=0)

    sample = torch.from_numpy(np.random.multivariate_normal(mean, cov, 1000)).float()

    # Center around day 500's encoding mean instead
    center_idx = date_idx
    mu, log_var = vae.encode(train_data_tensor)
    sample = sample + mu[center_idx]
    decoded_sample = vae.decode(sample)
    #decoded_sample = decoded_sample * data_std + data_mean
    
    ### Computes average kf for 25 and 75 deltas
    av_kf = np.zeros((8,2))
    for day in full_train_data:
        kfs = []
        if len(day) == 8:
            for i in range(8):
                kf_temp = day[i]['sorted_KFs']
                kfs.append([kf_temp[1], kf_temp[3]])
        else:
            for i in range(8):
                kf_temp = day[i*2+1]['sorted_KFs']
                kfs.append([kf_temp[1], kf_temp[3]])
        kfs = np.array(kfs)
        av_kf += kfs
    av_kf = av_kf/len(full_train_data)
    kfs = av_kf
    ttms = np.array(["1M", "2M", "3M", "6M", "9M", "1Y", "3Y", "5Y"])
    strike_grid = kfs

    base_params = transform_raw(original_train_data_tensor[center_idx,:].unsqueeze(0)).detach().numpy()
    
    tau, pi, mu, A, sig, _, m_arr, mean_arr = gen_curves_no_trans(torch.from_numpy(base_params))

    # Add at-the-money to strike-grid

    new_strike_grid = np.zeros([8,3])
    new_strike_grid[:,0] =  strike_grid[:,0]
    new_strike_grid[:,2] =  strike_grid[:,1]
    new_strike_grid[:,1] = 1

    surf = []
    for i in range(1):
        surf.append(gen_surface(taus, deltas, pi[i], A[i], mu[i], sig[i], m_arr[i], new_strike_grid))

    surf = np.squeeze(np.array(surf))
    
    # Translation of entire surface up and down by some average volatility

    # First compute average volatility in training set
    daily_mean_vol = []
    for day in full_train_data:
        all_vol = []
        if len(day) == 8:
            for i in range(8):
                all_vol.append(day[i]['sorted_IV'])
        else:
            for i in range(8):
                all_vol.append(day[i*2+1]['sorted_IV'])
        all_vol = np.array(all_vol)
        daily_mean_vol.append(np.mean(all_vol))
    av_vol = np.mean(daily_mean_vol)

    # Now copy and shift entire surface by +/- 2 and 3 sigmas
    baseline_surfs = np.tile(surf, (9, 1, 1))

    for i in range(9):
        if i < 4:
            baseline_surfs[i,:,:] = baseline_surfs[i,:,:] - np.std(daily_mean_vol) * (4 - i)
        elif i > 4:
            baseline_surfs[i,:,:] = baseline_surfs[i,:,:] + np.std(daily_mean_vol) * (i - 4)
            
    baseline_surfs = np.clip(baseline_surfs, 0.001, np.inf) # clip negative surface values
    
    if base_port is None:

        ############
        #Use portfolio solver to contruct delta and gamma neutral portfolio
        ############
        # Construct basic portfolio 
        baseline_port = {}
        baseline_port['K'] = np.array([
            strike_grid[0,0],
            strike_grid[0,1],
            strike_grid[3,0],
            strike_grid[3,1],
            strike_grid[5,0],
            strike_grid[5,1],
            strike_grid[7,0],
            1,
            strike_grid[7,1],
        ])
        baseline_port['t'] = np.array([
            taus[0],
            taus[0],
            taus[3],
            taus[3],
            taus[5],
            taus[5],
            taus[7],
            taus[7],
            taus[7],
        ])
        baseline_port['t'] = baseline_port['t'] + 1/365 # Add an extra day 

        baseline_port['b_coeff'] = np.array([1, -1, -1, 1, 1, -1, 1, -2, 1])
        #baseline_port['b_coeff'] = np.array([1, -1, -1, 1])


        base_ivs = np.array([
            surf[0,0],
            surf[0,2],
            surf[3,0],
            surf[3,2],
            surf[5,0],
            surf[5,2],
            surf[7,0],
            surf[7,1],
            surf[7,2],
        ])

        types = [True,True,True,True,True,True,True,True,True]

        a = opt_port(baseline_port, 1, base_ivs, types, np.array([-3, 2, 2]))
        coeff = np.insert(a, 0, 1)
        baseline_port['sum_coeffs'] = coeff
    else:
        baseline_port = base_port
    
    baseline_val = port_val(baseline_port, 1, base_ivs, types)
    coeff = baseline_port['sum_coeffs']
    
    # Construct portfolio for computation
    #new_grid = np.insert(strike_grid, 1, np.ones(8), axis = 1)
    new_portfolio_grid = np.vstack([
        np.array([1, 0, -1])* coeff[0],
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([-1, 0, 1])* coeff[1],
        np.array([0, 0, 0]),
        np.array([1, 0, -1])* coeff[2],
        np.array([0, 0, 0]),
        np.array([1, -2, 1])* coeff[3],
    ])
    
    pf_all = compute_portfolio_val(new_portfolio_grid, new_strike_grid , decoded_sample)
    pf_base = compute_portfolio_val(new_portfolio_grid, new_strike_grid , original_train_data_tensor[center_idx,:].unsqueeze(0), plus_day=True)
    pf_actual = compute_portfolio_val(new_portfolio_grid, new_strike_grid , original_train_data_tensor[center_idx+1,:].unsqueeze(0))
    
    new_baseline_port = dc(baseline_port)
    new_baseline_port['t'] = np.array([
        taus[0],
        taus[0],
        taus[3],
        taus[3],
        taus[5],
        taus[5],
        taus[7],
        taus[7],
        taus[7],
    ])
    
    # Compute portfolio value for other baseline cases
    port_vals = []
    for i in range(9):
        a, _, _ = port_val(new_baseline_port, 1, convert_surf(baseline_surfs[i]), types)
        port_vals.append(a)
        
    results = {}
    results['sample_val'] = pf_all
    results['line_val'] = port_vals
    results['line_deltas'] = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    results['baseline_port'] = baseline_port
    results['new_baseline_port'] = new_baseline_port
    results['baseline_val'] = baseline_val
    results['baseline_surfs'] = baseline_surfs
    results['actual_pf_val'] = pf_actual

    date = str(train_date_list[center_idx].to_pydatetime().date())

    with open(save_dir + date + '.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return baseline_port
    
    
if __name__ == '__main__':
    output_dir = os.getenv("SCRATCH") + '/train_outputs/'
    net_dir = 'network/'
    param_path = "AUD_outputs_0.3_drift/"

    with open("all_cur_train_valid_days_new.pickle", 'rb') as handle:
        all_days = pickle.load(handle)

    num_test_days = len(all_days['valid_days'])
    #num_test_days = 5
    
    # Do first day to get base_port for all other days
    port = compute_delta_samples_fixed(0, output_dir, net_dir, param_path)
    
    inputs = list(zip(
        list(range(1 , num_test_days)),
        [output_dir]*(num_test_days-1),
        [net_dir]*(num_test_days-1), 
        [param_path]*(num_test_days-1),
        [port]*(num_test_days-1)
    ))
    pool = Pool(os.cpu_count())
    pool.starmap(compute_delta_samples_fixed, inputs, chunksize=1)
    