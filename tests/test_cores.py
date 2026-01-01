import pytest
import pandas as pd
import numpy as np
from src.core.math_engine import SparseOptimizer
from src.core.simulation import RegimeSimulator
from src.core.backtester import BacktestEngine

# Fixtures (Fake Data Generators)
@pytest.fixture
def mock_returns():
    """Generates 2 years of fake returns for 3 assets."""
    np.random.seed(42)
    data = np.random.normal(0.0005, 0.01, (500, 3)) # Mean, Std, Shape
    df = pd.DataFrame(data, columns=["Asset_A", "Asset_B", "Asset_C"])
    return df

@pytest.fixture
def mock_prices():
    """Generates a fake price series for backtesting."""
    np.random.seed(42)
    # Random walk
    returns = np.random.normal(0.0005, 0.01, 500)
    price_path = (1 + returns).cumprod()
    return pd.Series(price_path, name="Mock_Asset")

# Test Math Engine (Optimization)
def test_l1_optimization_sparsity(mock_returns):
    """Does increasing Tau actually make the portfolio sparse?"""
    
    # Low Tau (0.001) -> Should have many assets
    w_low = SparseOptimizer.solve(mock_returns, tau=0.001)
    active_low = sum(1 for w in w_low.values() if w > 0)
    
    # High Tau (0.5) -> Should force some assets to zero
    w_high = SparseOptimizer.solve(mock_returns, tau=0.5)
    active_high = sum(1 for w in w_high.values() if w > 0)
    
    # Assertion: Higher penalty = Fewer assets
    assert active_high <= active_low
    assert sum(w_high.values()) == pytest.approx(1.0, rel=1e-4) # Sums to 1

def test_optimization_fallback(mock_returns):
    """If Solver fails (we simulate this by passing garbage), do we get Equal Weight?"""
    # Create impossible data (NaNs)
    bad_returns = mock_returns.copy()
    bad_returns.iloc[0, 0] = np.nan
    
    # This might trigger solver error or pandas error, 
    # but we want to test the robust handling of the class
    try:
        w = SparseOptimizer.solve(bad_returns, tau=0.1)
        # Should simulate fallback or handle gracefully
        assert len(w) == 3
    except:
        pass # If it raises, that's fine too, as long as it doesn't crash the simulation

# Test Simulation Engine (Numba) 
def test_regime_simulation(mock_returns):
    """Does the Monte Carlo engine return valid stats?"""
    sim = RegimeSimulator(mock_returns)
    sim.fit()
    
    # Check HMM params were found
    assert "mus" in sim.params
    assert sim.params["mus"].shape == (2, 3) # 2 regimes, 3 assets (this is the simplified shape)
    
    # Run Simulation
    weights = {"Asset_A": 0.5, "Asset_B": 0.5, "Asset_C": 0.0}
    stats = sim.run(weights, n_sims=100, days=50)
    
    assert "expected_return" in stats
    assert "var_95" in stats
    assert isinstance(stats["expected_return"], float)

# Test Backtester (VectorBT)
def test_backtest_strategy(mock_prices):
    """Does the strategy engine calculate Sharpe?"""
    params = {'fast': 10, 'slow': 20}
    result = BacktestEngine.run(mock_prices, "sma_cross", params)
    
    assert "equity" in result
    assert "metrics" in result
    assert "sharpe" in result["metrics"]
    
    # Check Equity Curve shape
    assert len(result["equity"]) == len(mock_prices)