import numpy as np
from scipy.interpolate import splev
from scipy import stats


def DensityEstimation(tck: tuple, KFs: np.ndarray, tau) -> np.ndarray:
    """
    Computes the density of the asset price ratio from the fitted spline approximation by taking the second
    derivative of the call price ratio, refer to documentation for details.

    :param tck: The tuple representing the fitted B-splines (direct output from function)
    :param KFs: The KFs which the density function will be evaluated at
    :param tau: The time to maturity of the option
    :return: The density values evaluated at the KFs given
    """

    # The evaluations of \sigma, \sigma', and \sigma''
    sigma = splev(x=KFs, tck=tck, der=0)
    sigma_p1 = splev(x=KFs, tck=tck, der=1)
    sigma_p2 = splev(x=KFs, tck=tck, der=2)

    # Values of d1 and d2
    d1 = np.divide((- np.log(KFs) + sigma**2 * tau / 2), sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # Values of d1' and d2'
    d1_p1 = - (np.multiply(KFs, sigma) * np.sqrt(tau)) ** (-1) \
            + np.divide(np.multiply(np.log(KFs), sigma_p1), sigma**2 * np.sqrt(tau)) \
            + sigma_p1 * np.sqrt(tau)/2
    d2_p1 = d1_p1 - sigma_p1*np.sqrt(tau)

    # Values of d1'' and d2''
    d1_p2 = np.divide((sigma + 2*np.multiply(KFs, sigma_p1)),(np.multiply(KFs**2,sigma**2) * np.sqrt(tau))) \
           + np.divide((np.multiply(np.multiply(np.log(KFs),sigma),sigma_p2) + 2 * np.multiply(np.log(KFs), sigma_p1)), sigma**3 * np.sqrt(tau)) \
           + sigma_p2 * np.sqrt(tau) / 2

    d2_p2 = d1_p2 - sigma_p2 * np.sqrt(tau)

    phi_d1 = stats.norm.pdf(d1)
    phi_d2 = stats.norm.pdf(d2)

    t1 = -np.multiply(np.multiply(d1,phi_d1),d1_p1**2) + np.multiply(phi_d1,d1_p2)
    t2 = 2 * np.multiply(phi_d2,d2_p1) - np.multiply(np.multiply(KFs,phi_d2),np.multiply(d2,d2_p1**2) - d2_p2)

    return t1 - t2

def LogDensityEstimation(tck: tuple, log_KFs: np.ndarray, tau) -> np.ndarray:
    """
    Computes the log density defined as e^x * f(e^x) of the asset price ratio from the fitted spline approximation by taking the second
    derivative of the call price ratio, refer to documentation for details.

    :param tck: The tuple representing the fitted B-splines (direct output from function)
    :param KFs: The KFs which the density function will be evaluated at
    :param tau: The time to maturity of the option
    :return: The density values evaluated at the KFs given
    """
    
    # Transforms logKFs into KFs
    KFs = np.exp(log_KFs)

    # The evaluations of \sigma, \sigma', and \sigma''
    sigma = splev(x=KFs, tck=tck, der=0)
    sigma_p1 = splev(x=KFs, tck=tck, der=1)
    sigma_p2 = splev(x=KFs, tck=tck, der=2)

    # Values of d1 and d2
    d1 = np.divide((- np.log(KFs) + sigma**2 * tau / 2), sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    # Values of d1' and d2'
    d1_p1 = - (np.multiply(KFs, sigma) * np.sqrt(tau)) ** (-1) \
            + np.divide(np.multiply(np.log(KFs), sigma_p1), sigma**2 * np.sqrt(tau)) \
            + sigma_p1 * np.sqrt(tau)/2
    d2_p1 = d1_p1 - sigma_p1*np.sqrt(tau)

    # Values of d1'' and d2''
    d1_p2 = np.divide((sigma + 2*np.multiply(KFs, sigma_p1)),(np.multiply(KFs**2,sigma**2) * np.sqrt(tau))) \
           + np.divide((np.multiply(np.multiply(np.log(KFs),sigma),sigma_p2) + 2 * np.multiply(np.log(KFs), sigma_p1)), sigma**3 * np.sqrt(tau)) \
           + sigma_p2 * np.sqrt(tau) / 2

    d2_p2 = d1_p2 - sigma_p2 * np.sqrt(tau)

    phi_d1 = stats.norm.pdf(d1)
    phi_d2 = stats.norm.pdf(d2)

    t1 = -np.multiply(np.multiply(d1,phi_d1),d1_p1**2) + np.multiply(phi_d1,d1_p2)
    t2 = 2 * np.multiply(phi_d2,d2_p1) - np.multiply(np.multiply(KFs,phi_d2),np.multiply(d2,d2_p1**2) - d2_p2)

    return np.multiply(np.exp(log_KFs), t1 - t2)
