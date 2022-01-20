import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad

class Model:
    def __init__(self, tau, pi, A, mu, sigma):
        self.tau = tau
        self.pi = pi
        self.A = A
        self.mu = mu
        self.sigma = sigma

    def UpdateParameters(self, tau, pi, A, mu, sigma):
        self.tau = tau
        self.pi = pi
        self.A = A
        self.mu = mu
        self.sigma = sigma

    def mean(self):
        """
        Python code for computing mean of X_t, used in case C-code fails
        """
        mat = expm(self.tau[0] * (self.A[0] + np.diag(self.mu[0])))
        for i in range(1, len(self.A)):
            mat = np.matmul(mat, expm((self.tau[i] - self.tau[i - 1]) * (self.A[i] + np.diag(self.mu[i]))))

        return np.sum(np.matmul(self.pi, mat))

    def psi(self, idx, z):
        """
        Computes the Psi matrix for arbitrary maturity idx
        """
        B = self.A[idx]
        B = B + np.diag(1j * (self.mu[idx,:] - 0.5 * self.sigma[idx,:]**2) * z - 0.5 * (self.sigma[idx,:] * z) ** 2)
        return B

    def CharFunc(self, z):
        """
        Characteristic function of the process X_t
        """
        tau_psi = self.tau[0] * self.psi(0, z)
        exp_psi_t = np.matmul(self.pi, expm(tau_psi))
        for i in range(1, len(self.tau)):
            tau_psi = (self.tau[i] - self.tau[i-1]) * self.psi(i,z)
            exp_psi_t = np.matmul(exp_psi_t, expm(tau_psi))

        return np.matmul(exp_psi_t, np.ones(len(self.pi)))

    def pdf_integrand(self, z, x):
        """
        Integrand to be integrated when computing pdf
        """
        return np.real(np.exp(-1j * z * x) * self.CharFunc(z))

    def price_integrand(self, z, logSK):
        """
        Integrand to be integrated when computing price
        """
        omega = z + 1.5 * 1j
        return np.real(np.exp(-1j * omega * logSK) * self.CharFunc(-omega) / (omega*omega - 1j * omega))

    def pdf(self, x0, xs, *args):
        if isinstance(xs, float):
            return quad(self.pdf_integrand, -np.inf, np.inf, args=(xs))[0] / 6.283185307179586
        else:
            result = []
            for i in range(len(xs)):
                cur_r = quad(self.pdf_integrand, -np.inf, np.inf, args=(xs[i]))[0] / 6.283185307179586
                result.append(cur_r)
            return np.array(result)

    def price(self, S, Ks, *args):
        if isinstance(Ks, float):
            return -(Ks/6.283185307179586) * quad(self.price_integrand, -np.inf, np.inf, args=(np.log(S/Ks)))[0]
        else:
            result = []
            for i in range(len(Ks)):
                cur_r = -(Ks[i]/6.283185307179586) * quad(self.price_integrand, -np.inf, np.inf, args=(np.log(S/Ks[i])))[0]
                result.append(cur_r)
            return np.array(result)