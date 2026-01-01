import numpy as np
import pandas as pd
import cvxpy as cp
from typing import Dict

class SparseOptimizer:
    @staticmethod
    def solve(returns: pd.DataFrame, tau: float) -> Dict[str, float]:
        """
        Solves: Minimize Risk - Return + Tau * |Weights|
        """
        tickers = returns.columns.tolist()
        n = len(tickers)
        
        # Annualize inputs
        mu = returns.mean().values * 252
        Sigma = returns.cov().values * 252
        
        # Variables
        w = cp.Variable(n)
        gamma = cp.Parameter(nonneg=True)
        gamma.value = tau
        
        # Objective
        risk = cp.quad_form(w, Sigma)
        ret = mu.T @ w
        l1_penalty = cp.norm(w, 1)
        
        # Minimize Negative Utility
        prob = cp.Problem(cp.Minimize(risk - ret + gamma * l1_penalty), 
                          [cp.sum(w) == 1, w >= 0])
        
        try:
            prob.solve()
        except cp.SolverError:
            return {t: 1.0/n for t in tickers}
            
        if w.value is None:
            return {t: 1.0/n for t in tickers}

        # Thresholding
        clean_weights = np.round(w.value, 4)
        clean_weights[clean_weights < 0.01] = 0.0
        
        if clean_weights.sum() > 0:
            clean_weights /= clean_weights.sum()
            
        return dict(zip(tickers, clean_weights))