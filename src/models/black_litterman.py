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
    logger.info("[_calculate_long_term_delta] Function entry.")
    logger.info(f"[_calculate_long_term_delta] Input: all_returns_df shape={all_returns_df.shape}, ff_df shape={ff_df.shape}, market_proxy_permno={market_proxy_permno}")
    market_returns = all_returns_df[all_returns_df['permno'] == market_proxy_permno].set_index('date')['total_return']
    ff_series = ff_df.set_index('date')['RF']
    logger.info(f"[_calculate_long_term_delta] market_returns shape={market_returns.shape}, ff_series shape={ff_series.shape}")
    aligned_market, aligned_rf = market_returns.align(ff_series, join='inner')
    logger.info(f"[_calculate_long_term_delta] aligned_market shape={aligned_market.shape}, aligned_rf shape={aligned_rf.shape}")
    
    if aligned_market.empty or len(aligned_market) < 2:
        logger.warning("Insufficient data for long-term delta calculation (less than 2 after alignment), using default value 2.5")
        logger.info("[_calculate_long_term_delta] Function exit (data insufficient).")
        return 2.5

    market_excess_returns = aligned_market - aligned_rf
    market_excess_return_annualized = market_excess_returns.mean() * 12
    market_variance_annualized = aligned_market.var() * 12
    logger.info(f"[_calculate_long_term_delta] market_excess_return_annualized={market_excess_return_annualized:.4f}, market_variance_annualized={market_variance_annualized:.4f}")
    
    if market_variance_annualized <= 0:
        logger.warning(f"Market return variance is zero or negative, cannot calculate valid long-term delta, using default value 2.5")
        logger.info("[_calculate_long_term_delta] Function exit (variance invalid).")
        return 2.5

    long_term_delta = market_excess_return_annualized / market_variance_annualized
    
    if not np.isfinite(long_term_delta) or long_term_delta <= 0:
        logger.warning(f"Cannot calculate valid long-term delta (result {long_term_delta:.2f}), using default value 2.5")
        long_term_delta = 2.5
    else:
        logger.info(f"Calculated default delta: {long_term_delta:.2f}")
    logger.info("[_calculate_long_term_delta] Function exit.")
    return long_term_delta

def get_current_universe(all_returns_df, analysis_date, lookback_months):
    logger.info("[BlackLittermanPortfolio._get_current_universe] Function entry.")
    logger.info(f"[BlackLittermanPortfolio._get_current_universe] Input: analysis_date={analysis_date}")
    start_date = analysis_date - pd.DateOffset(months=lookback_months)
    recent_data = all_returns_df[
        (all_returns_df['date'] >= start_date) & 
        (all_returns_df['date'] <= analysis_date)
    ]
    logger.info(f"[BlackLittermanPortfolio._get_current_universe] recent_data shape={recent_data.shape}")
    permno_counts = recent_data.groupby('permno')['date'].nunique()
    valid_permnos = permno_counts[permno_counts >= lookback_months].index.tolist()
    logger.info(f"[BlackLittermanPortfolio._get_current_universe] valid_permnos count={len(valid_permnos)}")
    current_returns_df = recent_data[recent_data['permno'].isin(valid_permnos)]
    returns_pivot = current_returns_df.pivot_table(
        index='date', columns='permno', values='total_return'
    ).fillna(0)
    logger.info(f"[BlackLittermanPortfolio._get_current_universe] returns_pivot shape={returns_pivot.shape}")
    logger.info("[BlackLittermanPortfolio._get_current_universe] Function exit.")
    return sorted(valid_permnos), returns_pivot

class BlackLittermanPortfolio:
    def __init__(self, all_returns_df, ff_df, expense_ratios, lookback_months, tau, market_proxy_permno):
        logger.info("[BlackLittermanPortfolio.__init__] Function entry.")
        # logger.info(f"[BlackLittermanPortfolio.__init__] Input: all_returns_df shape={all_returns_df.shape}, full_monthly_df shape={full_monthly_df.shape}, ff_df shape={ff_df.shape}, lookback_months={lookback_months}, tau={tau}, market_proxy_permno={market_proxy_permno}")
        self.all_returns_df = all_returns_df
        self.ff_df = ff_df
        self.expense_ratios = expense_ratios
        self.lookback_months = lookback_months
        self.tau = tau
        self.market_proxy_permno = market_proxy_permno
        self.default_delta = _calculate_long_term_delta(all_returns_df, ff_df, market_proxy_permno)
        logger.info(f"[BlackLittermanPortfolio.__init__] default_delta={self.default_delta}")
        logger.info("[BlackLittermanPortfolio.__init__] Function exit.")

    def _calculate_inputs(self, returns_pivot, analysis_date):
        logger.info("[BlackLittermanPortfolio._calculate_inputs] Function entry.")
        logger.info(f"[BlackLittermanPortfolio._calculate_inputs] Input: returns_pivot shape={returns_pivot.shape}, analysis_date={analysis_date}")
        cov_estimator = LedoitWolf()
        cov_estimator.fit(returns_pivot)
        Sigma = cov_estimator.covariance_ * 12
        logger.info(f"[BlackLittermanPortfolio._calculate_inputs] Initial Sigma shape={Sigma.shape}")

        if np.trace(Sigma) < 1e-8:
            logger.warning("Covariance matrix is close to zero, optimization not possible.")
            logger.info("[BlackLittermanPortfolio._calculate_inputs] Function exit (Sigma is None).")
            return None, None, None

        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals[eigvals < 1e-8] = 1e-8
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T
        logger.info(f"[BlackLittermanPortfolio._calculate_inputs] Adjusted Sigma shape={Sigma.shape}")

        lookback_start = analysis_date - pd.DateOffset(months=self.lookback_months)
        relevant_ff = self.ff_df[
            (self.ff_df['date'] >= lookback_start) & (self.ff_df['date'] <= analysis_date)
        ]
        logger.info(f"[BlackLittermanPortfolio._calculate_inputs] relevant_ff shape={relevant_ff.shape}")
        
        if self.market_proxy_permno in returns_pivot.columns and not relevant_ff.empty:
            market_returns = returns_pivot[self.market_proxy_permno]
            ff_series = relevant_ff.set_index('date')['RF']
            aligned_market, aligned_rf = market_returns.align(ff_series, join='inner')
            logger.info(f"[BlackLittermanPortfolio._calculate_inputs] aligned_market shape={aligned_market.shape}")

            if aligned_market.empty or len(aligned_market) < 2:
                logger.warning(f"Insufficient data for delta calculation (less than 2 after alignment), using default value {self.default_delta:.2f}.")
                delta = self.default_delta
            else:
                market_excess_returns = aligned_market - aligned_rf
                market_excess_return_annualized = market_excess_returns.mean() * 12
                market_variance_annualized = aligned_market.var() * 12
                
                if market_variance_annualized > 0:
                    delta = market_excess_return_annualized / market_variance_annualized
                else:
                    logger.warning(f"Market return variance is zero or invalid, using default delta value {self.default_delta:.2f}.")
                    delta = self.default_delta
        else:
            delta = self.default_delta

        if not np.isfinite(delta) or delta <= 0:
            logger.warning(f"Calculated delta is invalid ({delta:.2f}), using default value {self.default_delta:.2f}.")
            delta = self.default_delta
        logger.info(f"[BlackLittermanPortfolio._calculate_inputs] Calculated delta={delta}")

        num_assets = len(returns_pivot.columns)
        W_mkt = np.ones(num_assets) / num_assets
        logger.info(f"[BlackLittermanPortfolio._calculate_inputs] W_mkt shape={W_mkt.shape}")
        logger.info("[BlackLittermanPortfolio._calculate_inputs] Function exit.")
        return Sigma, delta, W_mkt

    def get_black_litterman_portfolio(self, analysis_date, P, Q, Omega, pre_calculated_inputs=None, max_weight=0.25, previous_weights=None):
        logger.info("[BlackLittermanPortfolio.get_black_litterman_portfolio] Function entry.")
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Input: analysis_date={analysis_date}, P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
        current_permnos, returns_pivot = get_current_universe(self.all_returns_df, analysis_date, lookback_months=self.lookback_months)
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] current_permnos count={len(current_permnos)}, returns_pivot shape={returns_pivot.shape}")

        if pre_calculated_inputs:
            Sigma, delta, W_mkt = pre_calculated_inputs
            logger.info("[BlackLittermanPortfolio.get_black_litterman_portfolio] Using pre-calculated inputs.")
        else:
            Sigma, delta, W_mkt = self._calculate_inputs(returns_pivot, analysis_date)
            logger.info("[BlackLittermanPortfolio.get_black_litterman_portfolio] Calculated inputs.")
        
        if Sigma is None or delta is None or W_mkt is None: # Check all three
            weights = np.ones(len(current_permnos)) / len(current_permnos)
            logger.warning("[BlackLittermanPortfolio.get_black_litterman_portfolio] Inputs (Sigma, delta, or W_mkt) are None, returning equal weights.")
            logger.info("[BlackLittermanPortfolio.get_black_litterman_portfolio] Function exit (Inputs are None).")
            return pd.Series(weights, index=current_permnos), None
        
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Sigma shape={Sigma.shape}, W_mkt shape={W_mkt.shape}")

        Pi = delta * np.dot(Sigma, W_mkt)
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Pi shape={Pi.shape}")
        Sigma_post = Sigma
        
        if P.size > 0:
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Processing views: P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
            try:
                tau_Sigma_inv = np.linalg.inv(self.tau * Sigma)
                Omega_inv = np.linalg.inv(Omega)
                M_inv = tau_Sigma_inv + np.dot(np.dot(P.T, Omega_inv), P)
                M = np.linalg.inv(M_inv)
                term1 = np.dot(tau_Sigma_inv, Pi)
                term2 = np.dot(np.dot(P.T, Omega_inv), Q.flatten())
                mu_bl = np.dot(M, term1 + term2)
                Sigma_post = Sigma + M
                logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] mu_bl shape={mu_bl.shape}, Sigma_post shape={Sigma_post.shape}")
                if not np.all(np.isfinite(mu_bl)):
                    logger.warning("mu_bl contains invalid values, using equilibrium returns.")
                    mu_bl = Pi
                    Sigma_post = Sigma
            except np.linalg.LinAlgError:
                logger.warning("Singular matrix encountered during BL calculation, using equilibrium returns.")
                mu_bl = Pi
                Sigma_post = Sigma
        else:
            mu_bl = Pi
            logger.info("No ML views provided, using equilibrium returns.")
            
        # 비용 차감을 제거 - 실제 수익률 계산에서만 비용을 차감하도록 변경
        # expenses = np.array([self.expense_ratios.get(permno, {}).get('expense_ratio', 0) for permno in current_permnos])
        # expenses_series = pd.Series(expenses, index=current_permnos)
        # expenses_aligned = expenses_series.reindex(current_permnos).fillna(0)
        
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Using gross expected returns (costs will be deducted during actual return calculation)")
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Before explicit conversion: mu_bl type={type(mu_bl)}, mu_bl shape={getattr(mu_bl, 'shape', 'N/A')}")

        # Ensure mu_bl is numpy array of float type
        mu_bl_arr = np.asarray(mu_bl, dtype=float)
        # expenses_arr = np.asarray(expenses_aligned.values, dtype=float)

        mu_bl_net = mu_bl_arr  # 비용을 차감하지 않음
        # mu_bl_net = mu_bl_arr - (expenses_arr / 12.0)
        n = len(current_permnos)
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] mu_bl_net shape={mu_bl_net.shape}, n={n}")

        use_turnover = config.USE_TURNOVER_CONSTRAINT and previous_weights is not None and not previous_weights.empty
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] use_turnover={use_turnover}")
        if use_turnover:
            logger.info(f"Applying turnover constraint (max: {config.MAX_TURNOVER:.0%})")
            P_opt = matrix(np.block([[delta * Sigma_post, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]]))
            q_opt = matrix(np.hstack([-mu_bl_net, np.zeros(n)]))
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] P_opt shape={P_opt.size}, q_opt shape={q_opt.size}")

            w_old, _ = previous_weights.align(pd.Series(index=current_permnos), join='right', fill_value=0)
            w_old = w_old.values
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] w_old shape={w_old.shape}")

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
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] G shape={G.size}, h shape={h.size}")
            
            A = matrix(np.hstack([np.ones(n), np.zeros(n)])).T
            b = matrix(1.0)
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] A shape={A.size}, b shape={b.size}")

        else: # If there is no turnover constraint
            P_opt = matrix(delta * Sigma_post)
            q_opt = matrix(-mu_bl_net)
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] P_opt shape={P_opt.size}, q_opt shape={q_opt.size}")
            
            G = matrix(np.vstack([-np.identity(n), np.identity(n)]))
            h = matrix(np.hstack([np.zeros(n), np.full(n, max_weight)]))
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] G shape={G.size}, h shape={h.size}")

            A = matrix(1.0, (1, n))
            b = matrix(1.0)
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] A shape={A.size}, b shape={b.size}")

        try:
            solution = solvers.qp(P_opt, q_opt, G, h, A, b, solver='qp')
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] CVXOPT solution status: {solution['status']}")
            result_x = np.array(solution['x']).flatten()
            weights = result_x[:n]
            logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Optimization successful. weights shape={weights.shape}")
        except ValueError as e:
            logger.error(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] CVXOPT Error {e}, replacing with equal weights.")
            weights = np.ones(n) / n

        weights[weights < 1e-5] = 0
        weights /= weights.sum()
        portfolio = pd.Series(weights, index=current_permnos)
        logger.info(f"[BlackLittermanPortfolio.get_black_litterman_portfolio] Final portfolio weights shape={portfolio.shape}")
        
        logger.info("Portfolio construction complete.")
        active_portfolio = portfolio[portfolio > 0]
        assets_log_str = ', '.join([f'{idx} {val:.4f}' for idx, val in active_portfolio.round(4).items()])
        log_str = f"{len(active_portfolio)} assets: {assets_log_str}"
        logger.info(log_str)
        logger.info("[BlackLittermanPortfolio.get_black_litterman_portfolio] Function exit.")
        return portfolio, mu_bl_net