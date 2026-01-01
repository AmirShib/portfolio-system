import vectorbt as vbt
import pandas as pd
import json
import hashlib

class BacktestEngine:
    @staticmethod
    def generate_key(asset, strategy, params):
        p_str = json.dumps(params, sort_keys=True)
        raw = f"{asset}:{strategy}:{p_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def run(price_series: pd.Series, strategy: str, params: dict):
        if strategy == "sma_cross":
            fast = vbt.MA.run(price_series, params['fast'])
            slow = vbt.MA.run(price_series, params['slow'])
            entries = fast.ma_crossed_above(slow)
            exits = fast.ma_crossed_below(slow)
        elif strategy == "rsi":
            rsi = vbt.RSI.run(price_series, window=params['window'])
            entries = rsi.rsi_crossed_below(params['buy'])
            exits = rsi.rsi_crossed_above(params['sell'])
            
        pf = vbt.Portfolio.from_signals(price_series, entries, exits, init_cash=1.0, fees=0.001, freq='1D')
        
        return {
            "equity": pf.value().to_dict(),
            "metrics": {
                "total_return": pf.total_return(),
                "sharpe": pf.sharpe_ratio(),
                "max_drawdown": pf.max_drawdown(),
                "win_rate": pf.stats()['Win Rate [%]'] / 100
            }
        }