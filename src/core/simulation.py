import numpy as np
import pandas as pd
from numba import jit
from hmmlearn.hmm import GaussianHMM

class RegimeSimulator:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.params = {}
        
    def fit(self):
        """Train HMM to find Bull/Bear states"""
        X = self.returns.values
        # 2 Components: Bull vs Bear
        model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X)
        
        means = model.means_.sum(axis=1)
        bull_idx = np.argmax(means)
        bear_idx = np.argmin(means)
        
        self.params = {
            "mus": np.array([model.means_[bear_idx], model.means_[bull_idx]]),
            "covs": np.array([model.covars_[bear_idx], model.covars_[bull_idx]]),
            "trans": np.array([model.transmat_[bear_idx], model.transmat_[bull_idx]]),
            "start_state": 1 if model.predict(X)[-1] == bull_idx else 0
        }
        
    def run(self, weights_dict: dict, n_sims=1000, days=252):
        if not self.params: self.fit()
        
        w_vec = np.array([weights_dict.get(c, 0.0) for c in self.returns.columns])
        
        final_values = _fast_mc(
            n_sims, days, self.params['trans'], 
            self.params['mus'], self.params['covs'], w_vec
        )
        
        return {
            "expected_return": float(np.mean(final_values) - 1.0),
            "var_95": float(np.percentile(final_values, 5) - 1.0),
            "win_rate": float(np.mean(final_values > 1.0))
        }

@jit(nopython=True)
def _fast_mc(n_sims, days, trans_mat, mus, covs, weights):
    results = np.zeros(n_sims)
    
    # Pre-calculate Portfolio Params for each state
    # State 0 (Bear)
    p_mu_0 = np.dot(weights, mus[0])
    p_var_0 = np.dot(weights, np.dot(covs[0], weights))
    p_std_0 = np.sqrt(p_var_0)
    
    # State 1 (Bull)
    p_mu_1 = np.dot(weights, mus[1])
    p_var_1 = np.dot(weights, np.dot(covs[1], weights))
    p_std_1 = np.sqrt(p_var_1)
    
    mu_vec = np.array([p_mu_0, p_mu_1])
    std_vec = np.array([p_std_0, p_std_1])
    
    for i in range(n_sims):
        val = 1.0
        state = 0 if np.random.random() < 0.5 else 1 
        
        for t in range(days):
            if np.random.random() < trans_mat[state, 0]:
                state = 0
            else:
                state = 1
            r = np.random.normal(mu_vec[state], std_vec[state])
            val *= (1 + r)
        results[i] = val
        
    return results