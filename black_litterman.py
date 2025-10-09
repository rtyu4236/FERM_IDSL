import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf
from cvxopt import matrix, solvers
import warnings
import config
from logger_setup import logger
solvers.options['show_progress'] = False
warnings.filterwarnings('ignore', category=UserWarning)

def _calculate_long_term_delta(all_returns_df, ff_df, market_proxy_ticker='SPY'):
    """
    장기적, 데이터 기반의 위험 회피 계수(delta) 계산

    - Pi = delta * Sigma * W_mkt 공식에 사용되는 핵심 파라미터
    - (시장 평균 초과 수익률) / (시장 분산) 공식을 사용해 과거 데이터로부터 추정
    - 투자자의 위험 단위당 역사적 보상 수준을 반영

    Args:
        all_returns_df (pd.DataFrame): 모든 자산의 월별 수익률 데이터
        ff_df (pd.DataFrame): Fama-French 요인 데이터 (무위험 수익률 포함)
        market_proxy_ticker (str): 시장 대표 티커

    Returns:
        float: 계산된 delta, 유효하지 않을 경우 기본값 2.5 반환
    """
    market_returns = all_returns_df[all_returns_df['TICKER'] == market_proxy_ticker].set_index('date')['retx']
    ff_series = ff_df.set_index('date')['RF']
    aligned_market, aligned_rf = market_returns.align(ff_series, join='inner')
    
    if aligned_market.empty or len(aligned_market) < 2:
        logger.warning("장기 delta 계산 데이터 부족 (정렬 후 2개 미만), 기본값 2.5 사용")
        return 2.5

    market_excess_returns = aligned_market - aligned_rf
    market_excess_return_annualized = market_excess_returns.mean() * 12
    market_variance_annualized = aligned_market.var() * 12
    
    if market_variance_annualized <= 0:
        logger.warning(f"시장 수익률 분산 0 또는 음수, 유효한 장기 delta 계산 불가, 기본값 2.5 사용")
        return 2.5

    long_term_delta = market_excess_return_annualized / market_variance_annualized
    
    if not np.isfinite(long_term_delta) or long_term_delta <= 0:
        logger.warning(f"유효한 장기 delta 계산 불가 (결과 {long_term_delta:.2f}), 기본값 2.5 사용")
        long_term_delta = 2.5
    else:
        logger.info(f"계산된 기본 delta: {long_term_delta:.2f}")
        
    return long_term_delta

class BlackLittermanPortfolio:
    """Black-Litterman 포트폴리오 최적화 수행.

    시장의 균형 수익률과 투자자의 주관적 '뷰'를 결합하여
    전통적 평균-분산 최적화의 한계를 극복하는 것을 목표로 함.
    """
    
    def __init__(self, all_returns_df, ff_df, expense_ratios, lookback_months, tau, market_proxy_ticker, asset_groups, group_constraints):
        self.all_returns_df = all_returns_df
        self.ff_df = ff_df
        self.expense_ratios = expense_ratios
        self.lookback_months = lookback_months
        self.tau = tau
        self.market_proxy_ticker = market_proxy_ticker
        self.asset_groups = asset_groups
        self.group_constraints = group_constraints
        self.default_delta = _calculate_long_term_delta(all_returns_df, ff_df, market_proxy_ticker)

    def _get_current_universe(self, analysis_date):
        """분석 날짜 기준, 투자 가능한 자산 유니버스 결정.
        
        lookback_months 기간 동안 데이터가 충분한 자산만 필터링하여
        모델 안정성을 확보하고 비유동성 자산으로 인한 왜곡을 방지.
        """
        start_date = analysis_date - pd.DateOffset(months=self.lookback_months)
        recent_data = self.all_returns_df[
            (self.all_returns_df['date'] >= start_date) & 
            (self.all_returns_df['date'] <= analysis_date)
        ]
        ticker_counts = recent_data.groupby('TICKER')['date'].nunique()
        valid_tickers = ticker_counts[ticker_counts >= self.lookback_months].index.tolist()
        current_returns_df = recent_data[recent_data['TICKER'].isin(valid_tickers)]
        returns_pivot = current_returns_df.pivot_table(
            index='date', columns='TICKER', values='retx'
        ).fillna(0)
        return sorted(valid_tickers), returns_pivot

    def _calculate_inputs(self, returns_pivot, analysis_date):
        """
        Black-Litterman 모델의 핵심 입력값 계산
        - Sigma: Ledoit-Wolf 공분산, 극단값 문제 완화
        - delta: 동적 위험 회피 계수, 시변성 반영
        - W_mkt: 시장 균형 가중치, 여기서는 동일 가중치 가정
        """
        cov_estimator = LedoitWolf()
        cov_estimator.fit(returns_pivot)
        Sigma = cov_estimator.covariance_ * 12

        if np.trace(Sigma) < 1e-8:
            logger.warning("공분산 행렬 0에 가까워 최적화 불가")
            return None, None, None

        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals[eigvals < 1e-8] = 1e-8
        Sigma = eigvecs @ np.diag(eigvals) @ eigvecs.T

        lookback_start = analysis_date - pd.DateOffset(months=self.lookback_months)
        relevant_ff = self.ff_df[
            (self.ff_df['date'] >= lookback_start) & (self.ff_df['date'] <= analysis_date)
        ]
        
        if self.market_proxy_ticker in returns_pivot.columns and not relevant_ff.empty:
            market_returns = returns_pivot[self.market_proxy_ticker]
            ff_series = relevant_ff.set_index('date')['RF']
            aligned_market, aligned_rf = market_returns.align(ff_series, join='inner')

            if aligned_market.empty or len(aligned_market) < 2:
                logger.warning(f"Delta 계산 데이터 부족 (정렬 후 2개 미만), 기본값 {self.default_delta:.2f} 사용")
                delta = self.default_delta
            else:
                market_excess_returns = aligned_market - aligned_rf
                market_excess_return_annualized = market_excess_returns.mean() * 12
                market_variance_annualized = aligned_market.var() * 12
                
                if market_variance_annualized > 0:
                    delta = market_excess_return_annualized / market_variance_annualized
                else:
                    logger.warning(f"시장 수익률 분산 0 또는 유효하지 않음, 기본 delta값 {self.default_delta:.2f} 사용")
                    delta = self.default_delta
        else:
            delta = self.default_delta

        if not np.isfinite(delta) or delta <= 0:
            logger.warning(f"계산된 delta 유효하지 않음 ({delta:.2f}), 기본값 {self.default_delta:.2f} 사용")
            delta = self.default_delta

        num_assets = len(returns_pivot.columns)
        W_mkt = np.ones(num_assets) / num_assets
        return Sigma, delta, W_mkt

    def get_black_litterman_portfolio(self, analysis_date, P, Q, Omega, pre_calculated_inputs=None, max_weight=0.25, previous_weights=None):
        """Black-Litterman 공식을 기반으로 최종 포트폴리오 가중치 계산.

        사후 기대 수익률(mu_bl)과 사후 공분산(Sigma_post)을 도출한 후,
        cvxopt를 사용한 2차 계획법으로 제약조건 하에서 최적 가중치를 결정.
        """
        current_tickers, returns_pivot = self._get_current_universe(analysis_date)
        logger.info("포트폴리오 구성 시작")
        logger.info(f"현재투자종목: {len(current_tickers)}개, {', '.join(current_tickers)}")

        if pre_calculated_inputs:
            Sigma, delta, W_mkt = pre_calculated_inputs
        else:
            Sigma, delta, W_mkt = self._calculate_inputs(returns_pivot, analysis_date)
        
        if Sigma is None:
            weights = np.ones(len(current_tickers)) / len(current_tickers)
            return pd.Series(weights, index=current_tickers), None
        
        logger.error(f"DEBUG: Sigma shape={Sigma.shape}, W_mkt shape={W_mkt.shape}")

        Pi = delta * np.dot(Sigma, W_mkt)
        logger.error(f"DEBUG: Pi shape={Pi.shape}")
        Sigma_post = Sigma
        
        if P.size > 0:
            logger.error(f"DEBUG: Received P shape={P.shape}, Q shape={Q.shape}, Omega shape={Omega.shape}")
            try:
                tau_Sigma_inv = np.linalg.inv(self.tau * Sigma)
                Omega_inv = np.linalg.inv(Omega)
                M_inv = tau_Sigma_inv + np.dot(np.dot(P.T, Omega_inv), P)
                M = np.linalg.inv(M_inv)
                term1 = np.dot(tau_Sigma_inv, Pi)
                term2 = np.dot(np.dot(P.T, Omega_inv), Q.flatten())
                mu_bl = np.dot(M, term1 + term2)
                Sigma_post = Sigma + M
                logger.error(f"DEBUG: mu_bl shape={mu_bl.shape}, Sigma_post shape={Sigma_post.shape}")
                if not np.all(np.isfinite(mu_bl)):
                    logger.warning("mu_bl에 유효하지 않은 값 포함, 균형 수익률 사용")
                    mu_bl = Pi
                    Sigma_post = Sigma
            except np.linalg.LinAlgError:
                logger.warning("BL 계산 중 특이 행렬 발생, 균형 수익률 사용")
                mu_bl = Pi
                Sigma_post = Sigma
        else:
            mu_bl = Pi
            logger.info("제공된 ML 뷰 없음, 균형 수익률 사용")
            
        expenses = np.array([self.expense_ratios.get(ticker, 0) for ticker in current_tickers])
        mu_bl_net = mu_bl - expenses
        n = len(current_tickers)

        use_turnover = config.USE_TURNOVER_CONSTRAINT and previous_weights is not None and not previous_weights.empty
        if use_turnover:
            logger.info(f"거래 회전율 제약 조건 적용 (최대: {config.MAX_TURNOVER:.0%})")
            P_opt = matrix(np.block([[delta * Sigma_post, np.zeros((n, n))], [np.zeros((n, n)), np.zeros((n, n))]]))
            q_opt = matrix(np.hstack([-mu_bl_net, np.zeros(n)]))

            w_old, _ = previous_weights.align(pd.Series(index=current_tickers), join='right', fill_value=0)
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
            
            for group_name, constraints in self.group_constraints.items():
                group_tickers = self.asset_groups.get(group_name, [])
                group_row = np.array([1.0 if ticker in group_tickers else 0.0 for ticker in current_tickers])
                if group_row.sum() > 0:
                    G_list.append(np.hstack([group_row, np.zeros(n)]))
                    h_list.append(constraints['max'])
                    G_list.append(np.hstack([-group_row, np.zeros(n)]))
                    h_list.append(-constraints['min'])

            G = matrix(np.vstack(G_list))
            h = matrix(np.hstack(h_list))
            
            A = matrix(np.hstack([np.ones(n), np.zeros(n)])).T
            b = matrix(1.0)

        else: # 거래 회전율 제약이 없는 경우
            P_opt = matrix(delta * Sigma_post)
            q_opt = matrix(-mu_bl_net)
            
            G_individual = np.vstack([-np.identity(n), np.identity(n)])
            h_individual = np.hstack([np.zeros(n), np.full(n, max_weight)])
            
            group_G_rows, group_h_rows = [], []
            for group_name, constraints in self.group_constraints.items():
                group_tickers = self.asset_groups.get(group_name, [])
                group_row = np.array([1.0 if ticker in group_tickers else 0.0 for ticker in current_tickers])
                if group_row.sum() > 0:
                    group_G_rows.append(group_row)
                    group_h_rows.append(constraints['max'])
                    group_G_rows.append(-group_row)
                    group_h_rows.append(-constraints['min'])

            if group_G_rows:
                G = matrix(np.vstack([G_individual, np.vstack(group_G_rows)]))
                h = matrix(np.hstack([h_individual, np.array(group_h_rows)]))
            else:
                G = matrix(G_individual)
                h = matrix(h_individual)

            A = matrix(1.0, (1, n))
            b = matrix(1.0)

        try:
            solution = solvers.qp(P_opt, q_opt, G, h, A, b, solver='qp')
            result_x = np.array(solution['x']).flatten()
            weights = result_x[:n]
        except ValueError as e:
            logger.error(f"CVXOPT 오류 {e}, 동일 가중치로 대체")
            weights = np.ones(n) / n

        logger.error(f"DEBUG: final weights shape={weights.shape}")
        weights[weights < 1e-5] = 0
        weights /= weights.sum()
        portfolio = pd.Series(weights, index=current_tickers)
        
        logger.info("포트폴리오 구성 완료")
        active_portfolio = portfolio[portfolio > 0]
        assets_log_str = ', '.join([f'{idx} {val:.4f}' for idx, val in active_portfolio.round(4).items()])
        log_str = f"{len(active_portfolio)}개 종목: {assets_log_str}"
        logger.info(log_str)
        
        return portfolio, mu_bl_net
