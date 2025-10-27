import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from cvxopt import matrix, solvers
import warnings
from config import settings as config
from src.utils.logger import logger
solvers.options['show_progress'] = False
warnings.filterwarnings('ignore', category=UserWarning)

def _calculate_long_term_delta(all_returns_df, ff_df, market_proxy_permno=84398):
    market_returns = all_returns_df[all_returns_df['permno'] == market_proxy_permno].set_index('date')['total_return']
    ff_series = ff_df.set_index('date')['RF']
    aligned_market, aligned_rf = market_returns.align(ff_series, join='inner')
    
    if aligned_market.empty or len(aligned_market) < 2:
        return 2.5

    market_excess_returns = aligned_market - aligned_rf
    market_excess_return_annualized = market_excess_returns.mean() * 12
    market_variance_annualized = aligned_market.var() * 12
    
    if market_variance_annualized <= 0:
        return 2.5

    long_term_delta = market_excess_return_annualized / market_variance_annualized
    
    if not np.isfinite(long_term_delta) or long_term_delta <= 0:
        long_term_delta = 2.5
    
    return long_term_delta

class BlackLittermanPortfolio:
    def __init__(self, all_returns_df, ff_df, expense_ratios, lookback_months, tau, market_proxy_permno):
        self.all_returns_df = all_returns_df
        self.ff_df = ff_df
        self.expense_ratios = expense_ratios
        self.lookback_months = lookback_months
        self.tau = tau
        self.market_proxy_permno = market_proxy_permno
        self.default_delta = _calculate_long_term_delta(all_returns_df, ff_df, market_proxy_permno)

    def _get_current_universe(self, analysis_date):
        start_date = analysis_date - pd.DateOffset(months=self.lookback_months)
        recent_data = self.all_returns_df[
            (self.all_returns_df['date'] >= start_date) & 
            (self.all_returns_df['date'] <= analysis_date)
        ]
        permno_counts = recent_data.groupby('permno')['date'].nunique()
        valid_permnos = permno_counts[permno_counts >= self.lookback_months].index.tolist()
        current_returns_df = recent_data[recent_data['permno'].isin(valid_permnos)]
        returns_pivot = current_returns_df.pivot_table(
            index='date', columns='permno', values='total_return'
        ).fillna(0)
        return sorted(valid_permnos), returns_pivot

    def _calculate_inputs(self, returns_pivot, analysis_date):
        cov_estimator = LedoitWolf()
        cov_estimator.fit(returns_pivot)
        Sigma = cov_estimator.covariance_ * 12

        if np.trace(Sigma) < 1e-8:
            return None, None, None

        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals[eigvals < 1e-8] = 1e-8
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        lookback_start = analysis_date - pd.DateOffset(months=self.lookback_months)
        relevant_ff = self.ff_df[
            (self.ff_df['date'] >= lookback_start) & (self.ff_df['date'] <= analysis_date)
        ]
        
        if self.market_proxy_permno in returns_pivot.columns and not relevant_ff.empty:
            market_returns = returns_pivot[self.market_proxy_permno]
            ff_series = relevant_ff.set_index('date')['RF']
            aligned_market, aligned_rf = market_returns.align(ff_series, join='inner')

            if aligned_market.empty or len(aligned_market) < 2:
                delta = self.default_delta
            else:
                market_excess_returns = aligned_market - aligned_rf
                market_excess_return_annualized = market_excess_returns.mean() * 12
                market_variance_annualized = aligned_market.var() * 12
                
                if market_variance_annualized > 0:
                    delta = market_excess_return_annualized / market_variance_annualized
                else:
                    delta = self.default_delta
        else:
            delta = self.default_delta

        if not np.isfinite(delta) or delta <= 0:
            delta = self.default_delta

        num_assets = len(returns_pivot.columns)
        W_mkt = np.ones(num_assets) / num_assets
        return Sigma, delta, W_mkt

    def get_black_litterman_portfolio(self, analysis_date, P, Q, Omega, pre_calculated_inputs=None, max_weight=0.25, previous_weights=None):
        current_permnos, returns_pivot = self._get_current_universe(analysis_date)

        if pre_calculated_inputs:
            Sigma, delta, W_mkt = pre_calculated_inputs
        else:
            Sigma, delta, W_mkt = self._calculate_inputs(returns_pivot, analysis_date)
        
        if Sigma is None or delta is None or W_mkt is None:
            weights = np.ones(len(current_permnos)) / len(current_permnos)
            return pd.Series(weights, index=current_permnos), None

        Pi = delta * np.dot(Sigma, W_mkt)
        Sigma_post = Sigma
        
        if P.size > 0:
            try:
                tau_Sigma_inv = np.linalg.inv(self.tau * Sigma)
                Omega_inv = np.linalg.inv(Omega)
                M_inv = tau_Sigma_inv + np.dot(np.dot(P.T, Omega_inv), P)
                M = np.linalg.inv(M_inv)
                term1 = np.dot(tau_Sigma_inv, Pi)
                term2 = np.dot(np.dot(P.T, Omega_inv), Q.flatten())
                mu_bl = np.dot(M, term1 + term2)
                Sigma_post = Sigma + M
                if not np.all(np.isfinite(mu_bl)):
                    mu_bl = Pi
                    Sigma_post = Sigma
            except np.linalg.LinAlgError:
                mu_bl = Pi
                Sigma_post = Sigma
        else:
            mu_bl = Pi
            
        expenses_arr = np.array([self.expense_ratios.get(permno, {}).get('expense_ratio', 0.0) for permno in current_permnos])
        mu_bl_arr = np.asarray(mu_bl, dtype=float)
        mu_bl_net = mu_bl_arr - (expenses_arr / 12.0)
        n = len(current_permnos)

        use_turnover = config.USE_TURNOVER_CONSTRAINT and previous_weights is not None and not previous_weights.empty
        if use_turnover:
            P_opt = matrix(np.block([[delta * Sigma_post, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]]))
            q_opt = matrix(np.hstack([-mu_bl_net, np.zeros(n)]))
            w_old, _ = previous_weights.align(pd.Series(index=current_permnos), join='right', fill_value=0)
            w_old = w_old.values
            G_list, h_list = [], []
            G_list.append(np.block([[-np.identity(n), np.zeros((n, n))], [np.identity(n), np.zeros((n, n))]]))
            h_list.append(np.hstack([np.zeros(n), np.full(n, max_weight)]))
            G_list.append(np.block([np.zeros((n, n)), -np.identity(n)]))
            h_list.append(np.zeros(n))
            G_list.append(np.block([[np.identity(n), -np.identity(n)], [-np.identity(n), -np.identity(n)]]))
            h_list.append(np.hstack([w_old, -w_old]))
            G_list.append(np.hstack([np.zeros(n), np.ones(n)]))
            h_list.append(config.MAX_TURNOVER)
            G = matrix(np.vstack(G_list))
            h = matrix(np.hstack(h_list))
            A = matrix(np.hstack([np.ones(n), np.zeros(n)])).T
            b = matrix(1.0)
        else:
            P_opt = matrix(delta * Sigma_post)
            q_opt = matrix(-mu_bl_net)
            G = matrix(np.vstack([-np.identity(n), np.identity(n)]))
            h = matrix(np.hstack([np.zeros(n), np.full(n, max_weight)]))
            A = matrix(1.0, (1, n))
            b = matrix(1.0)

        try:
            solution = solvers.qp(P_opt, q_opt, G, h, A, b, solver='qp')
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] CVXOPT solution status: {solution['status']}")
            result_x = np.array(solution['x']).flatten()
            weights = result_x[:n]
        except ValueError as e:
            weights = np.ones(n) / n

        weights[weights < 1e-5] = 0
        weights /= weights.sum()
        portfolio = pd.Series(weights, index=current_permnos)
        return portfolio, mu_bl_net
