# Colab dependency install

import importlib
import subprocess
import sys


def ensure_dependencies():
    required_modules = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'requests': 'requests',
        'yfinance': 'yfinance',
        'bs4': 'beautifulsoup4',
        'dateutil': 'python-dateutil',
        'pykrx': 'pykrx',
    }
    missing = []
    for module_name, package_name in required_modules.items():
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError:
            missing.append(package_name)

    if missing:
        print(f'[INFO] Installing missing packages: {missing}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])


ensure_dependencies()

import re
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta
from pykrx import stock

REQUEST_TIMEOUT = 10
MAX_RETRY = 1
ASSUMED_DISCOUNT_RATE_R = 0.10
DEFAULT_WINSORIZE_PERCENTILE = 5

_CACHE: Dict[Tuple[str, str, str], dict] = {}


def _safe_float(v):
    try:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.replace(',', '').strip()
            if v in ('', '-', 'N/A', 'nan', 'NaN'):
                return None
        f = float(v)
        if np.isinf(f) or np.isnan(f):
            return None
        return f
    except Exception:
        return None


def _as_yyyymmdd(d: date) -> str:
    return d.strftime('%Y%m%d')


def _as_ymd(d: date) -> str:
    return d.strftime('%Y-%m-%d')


def _normalize_date(base_date: Optional[str]) -> date:
    if not base_date:
        return date.today()
    return datetime.strptime(base_date, '%Y-%m-%d').date()


def _clean_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce').replace([np.inf, -np.inf], np.nan)
    s = s.dropna()
    s = s[s != 0]
    return s


def _winsorize_series(s: pd.Series, percentile: int = DEFAULT_WINSORIZE_PERCENTILE):
    s = _clean_series(s)
    if s.empty:
        return s, 0
    lo, hi = np.percentile(s, [percentile, 100 - percentile])
    clipped = s.clip(lower=lo, upper=hi)
    removed = int((s != clipped).sum())
    return clipped, removed

def resolve_ticker_interactive(korean_name: str):
    markets = ['KOSPI', 'KOSDAQ', 'KONEX']
    rows = []
    today = _as_yyyymmdd(date.today())

    for market in markets:
        for ticker in stock.get_market_ticker_list(today, market=market):
            rows.append({'ticker': ticker, 'name': stock.get_market_ticker_name(ticker), 'market': market})

    universe = pd.DataFrame(rows)
    exact = universe[universe['name'] == korean_name].copy()

    if len(exact) == 1:
        r = exact.iloc[0]
        return r['ticker'], r['name'], r['market']

    candidates = universe[universe['name'].str.contains(korean_name, na=False)].copy()
    if candidates.empty:
        raise ValueError(f'종목명 후보가 없습니다: {korean_name}')

    candidates = candidates.sort_values(['name', 'market']).reset_index(drop=True)

    if len(candidates) == 1:
        r = candidates.iloc[0]
        return r['ticker'], r['name'], r['market']

    print('후보가 2개 이상입니다. 번호를 선택하세요:')
    for idx, row in candidates.iterrows():
        print(f"[{idx}] name={row['name']} market={row['market']} ticker={row['ticker']}")

    selected = int(input('선택 번호: ').strip())
    if selected < 0 or selected >= len(candidates):
        raise ValueError('잘못된 선택 번호입니다.')
    r = candidates.iloc[selected]
    return r['ticker'], r['name'], r['market']


def snap_to_trading_day(ticker: str, target_day: date, max_back_days: int = 14):
    d = target_day
    for _ in range(max_back_days + 1):
        ohlcv = stock.get_market_ohlcv_by_date(_as_yyyymmdd(d), _as_yyyymmdd(d), ticker)
        if not ohlcv.empty:
            return d
        d -= timedelta(days=1)
    return None

def fetch_price_series_krx(ticker: str, start_day: date, end_day: date) -> pd.DataFrame:
    df = stock.get_market_ohlcv_by_date(_as_yyyymmdd(start_day), _as_yyyymmdd(end_day), ticker)
    if df.empty:
        return pd.DataFrame(columns=['close'])
    out = pd.DataFrame(index=df.index)
    out['close'] = pd.to_numeric(df['종가'], errors='coerce')
    return out.dropna()


def fetch_price_and_mcap_krx(ticker: str, base_day: date):
    snapped = snap_to_trading_day(ticker, base_day)
    if snapped is None:
        return None, None, None

    ohlcv = stock.get_market_ohlcv_by_date(_as_yyyymmdd(snapped), _as_yyyymmdd(snapped), ticker)
    cap = stock.get_market_cap_by_date(_as_yyyymmdd(snapped), _as_yyyymmdd(snapped), ticker)

    close = _safe_float(ohlcv['종가'].iloc[-1]) if not ohlcv.empty else None
    mcap = _safe_float(cap['시가총액'].iloc[-1]) if not cap.empty else None
    return snapped, close, mcap


def fetch_historical_per_pbr_krx(ticker: str, base_day: date, years: int = 10, percentile_basis: str = 'daily'):
    start_day = base_day - relativedelta(years=years)
    fundamental = stock.get_market_fundamental_by_date(_as_yyyymmdd(start_day), _as_yyyymmdd(base_day), ticker, freq='d')
    if fundamental.empty:
        return pd.DataFrame(columns=['PER', 'PBR', 'EPS'])

    series = fundamental[['PER', 'PBR', 'EPS']].copy()
    for col in ['PER', 'PBR', 'EPS']:
        series[col] = pd.to_numeric(series[col], errors='coerce')

    if percentile_basis == 'month_end':
        series = series.resample('ME').last()

    return series

def _krx_yf_symbol(ticker: str, market: str) -> str:
    return f'{ticker}.KS' if market == 'KOSPI' else f'{ticker}.KQ'


def fetch_forward_multiples_yf(ticker: str, market: str, error_log: List[str]):
    symbol = _krx_yf_symbol(ticker, market)
    for attempt in range(MAX_RETRY + 1):
        try:
            tk = yf.Ticker(symbol)
            info = tk.info
            return {
                'per_fwd_12m': _safe_float(info.get('forwardPE')),
                'pbr_fwd_12m': None,
                'forward_multiple_status': 'ok' if info else 'none'
            }
        except Exception as e:
            if attempt < MAX_RETRY:
                time.sleep(0.5)
                continue
            error_log.append(f'yfinance forward multiple fetch failed: {e}')
            return {'per_fwd_12m': None, 'pbr_fwd_12m': None, 'forward_multiple_status': 'failed'}


def fetch_forward_multiples_naver(ticker: str, error_log: List[str]):
    url = f'https://finance.naver.com/item/main.naver?code={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')

            per_fwd = None
            for tr in soup.select('tr'):
                cells = [c.get_text(' ', strip=True) for c in tr.find_all(['th', 'td'])]
                for idx, cell in enumerate(cells):
                    key = cell.replace(' ', '')
                    if '추정PER' in key and idx + 1 < len(cells):
                        per_fwd = _safe_float(cells[idx + 1])
                        if per_fwd is not None:
                            break
                if per_fwd is not None:
                    break

            if per_fwd is None:
                text = soup.get_text(' ', strip=True)
                m = re.search(r'추정\s*PER[^0-9\-]*([0-9]+(?:\.[0-9]+)?)', text)
                if m:
                    per_fwd = _safe_float(m.group(1))

            return {
                'per_fwd_12m': per_fwd,
                'pbr_fwd_12m': None,
                'forward_multiple_status': 'ok' if per_fwd is not None else 'none',
            }
        except requests.Timeout:
            if attempt < MAX_RETRY:
                continue
            error_log.append('Naver Finance timeout (forward multiple)')
        except Exception as e:
            if attempt < MAX_RETRY:
                continue
            error_log.append(f'Naver Finance forward multiple fetch failed: {e}')

    return {'per_fwd_12m': None, 'pbr_fwd_12m': None, 'forward_multiple_status': 'failed'}


def fetch_current_multiples_naver(ticker: str, error_log: List[str]):
    url = f'https://finance.naver.com/item/main.naver?code={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')

            def _extract_by_label(label: str):
                for tr in soup.select('tr'):
                    cells = [c.get_text(' ', strip=True) for c in tr.find_all(['th', 'td'])]
                    for idx, cell in enumerate(cells):
                        key = cell.replace(' ', '')
                        if label in key and idx + 1 < len(cells):
                            v = _safe_float(cells[idx + 1])
                            if v is not None:
                                return v
                text_blob = soup.get_text(' ', strip=True)
                m = re.search(rf'{label}[^0-9\-]*([0-9]+(?:\.[0-9]+)?)', text_blob)
                return _safe_float(m.group(1)) if m else None

            return {
                'per_ttm': _extract_by_label('PER(배)'),
                'pbr_ttm': _extract_by_label('PBR(배)'),
            }
        except requests.Timeout:
            if attempt < MAX_RETRY:
                continue
            error_log.append('Naver Finance timeout (current PER/PBR)')
        except Exception as e:
            if attempt < MAX_RETRY:
                continue
            error_log.append(f'Naver Finance current PER/PBR fetch failed: {e}')

    return {'per_ttm': None, 'pbr_ttm': None}


def fetch_forward_multiples(ticker: str, market: str, error_log: List[str]):
    yf_data = fetch_forward_multiples_yf(ticker, market, error_log)
    if yf_data['per_fwd_12m'] is not None:
        return yf_data

    error_log.append('yfinance forwardPE 부재/실패: Naver Finance fallback 시도')
    naver_data = fetch_forward_multiples_naver(ticker, error_log)

    if naver_data['per_fwd_12m'] is not None:
        naver_data['forward_multiple_status'] = 'ok'
        return naver_data

    if yf_data['forward_multiple_status'] == 'failed' and naver_data['forward_multiple_status'] == 'failed':
        return {'per_fwd_12m': None, 'pbr_fwd_12m': None, 'forward_multiple_status': 'failed'}
    return {'per_fwd_12m': None, 'pbr_fwd_12m': None, 'forward_multiple_status': 'none'}

def fetch_consensus_fwd_eps_fnguide(ticker: str, error_log: List[str]):
    url = f'https://comp.fnguide.com/SVO2/ASP/SVD_Main.asp?pGB=1&gicode=A{ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}

    for attempt in range(MAX_RETRY + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'lxml')
            text = soup.get_text(' ', strip=True)

            nums = [float(x.replace(',', '')) for x in re.findall(r'(?<!\d)(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?', text[:200000])]
            nums = [n for n in nums if n > 0]
            if len(nums) < 3:
                raise ValueError('insufficient numeric hints from FnGuide page')

            now, ago3, ago12 = nums[0], nums[1], nums[2]
            chg3 = (now / ago3 - 1) * 100 if ago3 else None
            chg12 = (now / ago12 - 1) * 100 if ago12 else None

            return {
                'cons_fwd_eps_now': now,
                'cons_fwd_eps_3m_ago': ago3,
                'cons_fwd_eps_12m_ago': ago12,
                'cons_fwd_eps_chg_3m_pct': chg3,
                'cons_fwd_eps_chg_12m_pct': chg12,
                'cons_data_date': _as_ymd(date.today()),
                'consensus_data_status': 'ok',
                'cons_note': 'FnGuide 텍스트 숫자 추출 휴리스틱 기반. 정밀 스냅샷 미보장.'
            }
        except requests.Timeout:
            error_log.append('FnGuide timeout')
            if attempt < MAX_RETRY:
                continue
        except Exception as e:
            if attempt < MAX_RETRY:
                continue
            error_log.append(f'FnGuide consensus fetch failed: {e}')

    return {
        'cons_fwd_eps_now': None,
        'cons_fwd_eps_3m_ago': None,
        'cons_fwd_eps_12m_ago': None,
        'cons_fwd_eps_chg_3m_pct': None,
        'cons_fwd_eps_chg_12m_pct': None,
        'cons_data_date': None,
        'consensus_data_status': 'failed',
        'cons_note': 'FnGuide 접근 실패 또는 파싱 실패'
    }

def fetch_financials_for_ev_ebitda(ticker: str, market: str, error_log: List[str]):
    symbol = _krx_yf_symbol(ticker, market)
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}
        ev = _safe_float(info.get('enterpriseValue'))
        ebitda = _safe_float(info.get('ebitda'))
        net_debt = None
        mcap = _safe_float(info.get('marketCap'))
        if ev is not None and mcap is not None:
            net_debt = ev - mcap
        return {'ev': ev, 'ebitda': ebitda, 'net_debt': net_debt}
    except Exception as e:
        error_log.append(f'EV/EBITDA financial fetch failed: {e}')
        return {'ev': None, 'ebitda': None, 'net_debt': None}


def compute_ev_ebitda_series(mcap_series: pd.Series, net_debt: Optional[float], ebitda: Optional[float]):
    if net_debt is None or ebitda in (None, 0):
        return pd.Series(dtype=float)
    ev_series = mcap_series + net_debt
    return _clean_series(ev_series / ebitda)

def _window_series(series: pd.Series, base_day: date, years: int):
    start = pd.Timestamp(base_day - relativedelta(years=years))
    end = pd.Timestamp(base_day)
    return _clean_series(series.loc[(series.index >= start) & (series.index <= end)])


def _calc_stats(current, hist: pd.Series, apply_trim=True, winsorize_percentile=DEFAULT_WINSORIZE_PERCENTILE):
    out = {'percentile': None, 'min': None, 'max': None, 'mean': None, 'median': None, 'n': 0, 'missing_ratio': None, 'removed_count': 0}
    cleaned = _clean_series(hist)
    if cleaned.empty:
        return out

    out['n'] = int(cleaned.shape[0])
    out['missing_ratio'] = float(max(0.0, 1 - cleaned.shape[0] / max(hist.shape[0], 1)))

    use_s = cleaned
    if apply_trim:
        use_s, removed = _winsorize_series(cleaned, winsorize_percentile)
        out['removed_count'] = int(removed)

    out['min'], out['max'], out['mean'], out['median'] = float(use_s.min()), float(use_s.max()), float(use_s.mean()), float(use_s.median())
    out['percentile'] = float((use_s <= current).mean() * 100) if current is not None and not use_s.empty else None
    return out


def calc_regime_shift(mean_5y, mean_10y):
    if mean_5y is None or mean_10y in (None, 0):
        return None
    return mean_5y / mean_10y - 1


def calc_eps_normalization_flags(eps_series: pd.Series, eps_ttm: Optional[float], base_day: date):
    eps_5y = _window_series(eps_series, base_day, 5)
    if eps_5y.empty or eps_ttm is None:
        return {'eps_5y_mean': None, 'eps_5y_std': None, 'eps_vs_5y_mean_pct': None, 'eps_zscore_5y': None, 'per_distortion_flag': None, 'per_distortion_reason': None}

    mean = float(eps_5y.mean())
    std = float(eps_5y.std()) if len(eps_5y) > 1 else 0.0
    dev_pct = (eps_ttm / mean - 1) * 100 if mean != 0 else None
    z = (eps_ttm - mean) / std if std not in (None, 0) else None

    if z is not None and z <= -2:
        flag, reason = True, 'EPS zscore -2 이하'
    elif dev_pct is not None and dev_pct <= -50:
        flag, reason = True, 'EPS가 5년 평균 대비 -50% 이하'
    else:
        flag, reason = False, '극단 왜곡 신호 없음'

    return {'eps_5y_mean': mean, 'eps_5y_std': std, 'eps_vs_5y_mean_pct': dev_pct, 'eps_zscore_5y': z, 'per_distortion_flag': flag, 'per_distortion_reason': reason}


def calc_returns_around_event(price_df: pd.DataFrame, base_day: date, event_day: Optional[date], error_log: List[str]):
    out = {'event_date': None, 'ret_event_minus_5d_to_base_pct': None, 'event_peak_date': None, 'ret_event_plus_peak_to_base_pct': None, 'ret_1m_pct': None}
    if price_df.empty:
        error_log.append('가격 시계열 없음: 이벤트 수익률 계산 불가')
        return out

    px = price_df['close']
    base_ts = px.index[px.index <= pd.Timestamp(base_day)]
    if len(base_ts) == 0:
        error_log.append('기준일 이전 가격 없음')
        return out

    base_idx = base_ts[-1]
    base_close = float(px.loc[base_idx])

    m1_ts = px.index[px.index <= base_idx - pd.Timedelta(days=30)]
    if len(m1_ts) > 0:
        out['ret_1m_pct'] = (base_close / float(px.loc[m1_ts[-1]]) - 1) * 100

    if event_day is None:
        return out

    ev_ts = px.index[px.index <= pd.Timestamp(event_day)]
    if len(ev_ts) == 0:
        error_log.append('이벤트일 이전 가격 없음')
        return out

    ev_idx = ev_ts[-1]
    out['event_date'] = ev_idx.strftime('%Y-%m-%d')
    minus5_pos = max(0, px.index.get_loc(ev_idx) - 5)
    minus5_close = float(px.iloc[minus5_pos])
    out['ret_event_minus_5d_to_base_pct'] = (base_close / minus5_close - 1) * 100

    post = px.loc[ev_idx:base_idx]
    if not post.empty:
        peak_idx = post.idxmax()
        out['event_peak_date'] = peak_idx.strftime('%Y-%m-%d')
        out['ret_event_plus_peak_to_base_pct'] = (base_close / float(post.max()) - 1) * 100

    return out

def run_phase1_data(
    korean_name: str,
    base_date: Optional[str] = None,
    percentile_basis: str = 'daily',
    event_mode: str = 'manual',
    event_date_manual: Optional[str] = None,
    auto_event_days: int = 90,
    include_ev_ebitda: bool = True,
    include_consensus: bool = True,
    include_outlier_trim: bool = True,
    include_implied_growth: bool = True,
    include_eps_normalization: bool = True,
    winsorize_percentile: int = DEFAULT_WINSORIZE_PERCENTILE,
): 
    error_log: List[str] = []
    base_day_raw = _normalize_date(base_date)
    ticker, resolved_name, market = resolve_ticker_interactive(korean_name)

    cache_key = (ticker, _as_ymd(base_day_raw), percentile_basis)
    if cache_key not in _CACHE:
        base_snap, base_close, base_mcap = fetch_price_and_mcap_krx(ticker, base_day_raw)
        if base_snap is None:
            raise RuntimeError('기준일 스냅 실패')
        if base_snap != base_day_raw:
            error_log.append(f'기준일 휴일 스냅: {base_day_raw} -> {base_snap}')
        per_pbr = fetch_historical_per_pbr_krx(ticker, base_snap, years=10, percentile_basis=percentile_basis)
        price_df = fetch_price_series_krx(ticker, base_snap - relativedelta(years=10), base_snap)
        _CACHE[cache_key] = {'per_pbr': per_pbr, 'price_df': price_df, 'base_snap': base_snap, 'base_close': base_close, 'base_mcap': base_mcap}

    cached = _CACHE[cache_key]
    per_pbr, price_df = cached['per_pbr'], cached['price_df']
    base_snap, base_close, base_mcap = cached['base_snap'], cached['base_close'], cached['base_mcap']

    per_series = per_pbr['PER'] if 'PER' in per_pbr else pd.Series(dtype=float)
    pbr_series = per_pbr['PBR'] if 'PBR' in per_pbr else pd.Series(dtype=float)
    eps_series = per_pbr['EPS'] if 'EPS' in per_pbr else pd.Series(dtype=float)

    per_ttm = _safe_float(per_series.dropna().iloc[-1]) if not per_series.dropna().empty else None
    pbr_ttm = _safe_float(pbr_series.dropna().iloc[-1]) if not pbr_series.dropna().empty else None
    if per_ttm is None or pbr_ttm is None:
        naver_current = fetch_current_multiples_naver(ticker, error_log)
        if per_ttm is None and naver_current['per_ttm'] is not None:
            per_ttm = naver_current['per_ttm']
            error_log.append('PER TTM을 Naver Finance 현재값으로 보완')
        if pbr_ttm is None and naver_current['pbr_ttm'] is not None:
            pbr_ttm = naver_current['pbr_ttm']
            error_log.append('PBR TTM을 Naver Finance 현재값으로 보완')
    eps_ttm = _safe_float(eps_series.dropna().iloc[-1]) if not eps_series.dropna().empty else None

    forward_data = fetch_forward_multiples(ticker, market, error_log)
    per_fwd_12m = forward_data['per_fwd_12m']

    cons_data = {'cons_fwd_eps_now': None, 'cons_fwd_eps_3m_ago': None, 'cons_fwd_eps_12m_ago': None, 'cons_fwd_eps_chg_3m_pct': None, 'cons_fwd_eps_chg_12m_pct': None, 'cons_data_date': None, 'consensus_data_status': 'none', 'cons_note': 'include_consensus=False'}
    if include_consensus:
        cons_data = fetch_consensus_fwd_eps_fnguide(ticker, error_log)

    ev_ebitda_ttm, ev_series = None, pd.Series(dtype=float)
    if include_ev_ebitda:
        fin = fetch_financials_for_ev_ebitda(ticker, market, error_log)
        if fin['ev'] is not None and fin['ebitda'] not in (None, 0):
            ev_ebitda_ttm = fin['ev'] / fin['ebitda']
        if base_close not in (None, 0) and base_mcap is not None and not price_df.empty:
            shares = base_mcap / base_close
            mcap_series = price_df['close'] * shares
            ev_series = compute_ev_ebitda_series(mcap_series, fin['net_debt'], fin['ebitda'])
        else:
            error_log.append('EV/EBITDA 분포 계산용 mcap_series 생성 실패')

    per5 = _calc_stats(per_ttm, _window_series(per_series, base_snap, 5), include_outlier_trim, winsorize_percentile)
    per10 = _calc_stats(per_ttm, _window_series(per_series, base_snap, 10), include_outlier_trim, winsorize_percentile)
    pbr5 = _calc_stats(pbr_ttm, _window_series(pbr_series, base_snap, 5), include_outlier_trim, winsorize_percentile)
    pbr10 = _calc_stats(pbr_ttm, _window_series(pbr_series, base_snap, 10), include_outlier_trim, winsorize_percentile)
    ev5 = _calc_stats(ev_ebitda_ttm, _window_series(ev_series, base_snap, 5), include_outlier_trim, winsorize_percentile) if include_ev_ebitda else _calc_stats(None, pd.Series(dtype=float), False)
    ev10 = _calc_stats(ev_ebitda_ttm, _window_series(ev_series, base_snap, 10), include_outlier_trim, winsorize_percentile) if include_ev_ebitda else _calc_stats(None, pd.Series(dtype=float), False)

    implied_eps_ttm = base_close / per_ttm if base_close not in (None, 0) and per_ttm not in (None, 0) else None
    implied_eps_fwd = base_close / per_fwd_12m if base_close not in (None, 0) and per_fwd_12m not in (None, 0) else None
    implied_growth_g = ASSUMED_DISCOUNT_RATE_R - (1 / per_fwd_12m) if include_implied_growth and per_fwd_12m not in (None, 0) else None

    eps_norm = {'eps_5y_mean': None, 'eps_5y_std': None, 'eps_vs_5y_mean_pct': None, 'eps_zscore_5y': None, 'per_distortion_flag': None, 'per_distortion_reason': None}
    if include_eps_normalization:
        eps_norm = calc_eps_normalization_flags(eps_series, eps_ttm, base_snap)

    price_6m = None
    if not price_df.empty:
        ts_6m = price_df.index[price_df.index <= pd.Timestamp(base_snap - relativedelta(months=6))]
        if len(ts_6m) > 0 and base_close is not None:
            price_6m = float(price_df.loc[ts_6m[-1], 'close'])
    price_return_6m_pct = (base_close / price_6m - 1) * 100 if base_close not in (None, 0) and price_6m not in (None, 0) else None
    fwd_eps_change_6m_pct = None
    if include_consensus:
        error_log.append('FnGuide 6개월 스냅샷 미제공: fwd_eps_change_6m_pct=None')
    momentum_gap_pct = price_return_6m_pct - fwd_eps_change_6m_pct if price_return_6m_pct is not None and fwd_eps_change_6m_pct is not None else None

    event_date = None
    if event_mode == 'manual' and event_date_manual:
        event_date = datetime.strptime(event_date_manual, '%Y-%m-%d').date()
    elif event_mode == 'auto':
        event_date = base_snap - timedelta(days=auto_event_days)
        error_log.append(f'event_mode=auto: event_date={event_date} (base-{auto_event_days}d)')
    event_ret = calc_returns_around_event(price_df, base_snap, event_date, error_log)

    score = 100
    if per10['missing_ratio'] is not None:
        score -= int(per10['missing_ratio'] * 20)
    if pbr10['missing_ratio'] is not None:
        score -= int(pbr10['missing_ratio'] * 20)
    if include_ev_ebitda and ev10['n'] == 0:
        score -= 15
    if cons_data['consensus_data_status'] != 'ok':
        score -= 15
    if forward_data['forward_multiple_status'] != 'ok':
        score -= 10
    score = max(0, min(100, score))

    result = {
        'ticker': ticker, 'resolved_name': resolved_name, 'market': market,
        'base_date': _as_ymd(base_snap), 'base_close': base_close, 'base_mcap': base_mcap,
        'per_ttm': per_ttm, 'pbr_ttm': pbr_ttm, 'per_fwd_12m': per_fwd_12m, 'pbr_fwd_12m': None, 'ev_ebitda_ttm': ev_ebitda_ttm,
        'per_5y_percentile': per5['percentile'], 'per_10y_percentile': per10['percentile'],
        'per_5y_min': per5['min'], 'per_5y_max': per5['max'], 'per_5y_mean': per5['mean'], 'per_5y_median': per5['median'],
        'per_10y_min': per10['min'], 'per_10y_max': per10['max'], 'per_10y_mean': per10['mean'], 'per_10y_median': per10['median'],
        'pbr_5y_percentile': pbr5['percentile'], 'pbr_10y_percentile': pbr10['percentile'],
        'pbr_5y_min': pbr5['min'], 'pbr_5y_max': pbr5['max'], 'pbr_5y_mean': pbr5['mean'], 'pbr_5y_median': pbr5['median'],
        'pbr_10y_min': pbr10['min'], 'pbr_10y_max': pbr10['max'], 'pbr_10y_mean': pbr10['mean'], 'pbr_10y_median': pbr10['median'],
        'ev_ebitda_5y_percentile': ev5['percentile'], 'ev_ebitda_10y_percentile': ev10['percentile'],
        'ev_ebitda_5y_min': ev5['min'], 'ev_ebitda_5y_max': ev5['max'], 'ev_ebitda_5y_mean': ev5['mean'], 'ev_ebitda_5y_median': ev5['median'],
        'ev_ebitda_10y_min': ev10['min'], 'ev_ebitda_10y_max': ev10['max'], 'ev_ebitda_10y_mean': ev10['mean'], 'ev_ebitda_10y_median': ev10['median'],
        'per_vs_5y_mean_pct': (per_ttm / per5['mean'] - 1) * 100 if per_ttm is not None and per5['mean'] not in (None, 0) else None,
        'per_vs_10y_mean_pct': (per_ttm / per10['mean'] - 1) * 100 if per_ttm is not None and per10['mean'] not in (None, 0) else None,
        'pbr_vs_5y_mean_pct': (pbr_ttm / pbr5['mean'] - 1) * 100 if pbr_ttm is not None and pbr5['mean'] not in (None, 0) else None,
        'pbr_vs_10y_mean_pct': (pbr_ttm / pbr10['mean'] - 1) * 100 if pbr_ttm is not None and pbr10['mean'] not in (None, 0) else None,
        'ev_ebitda_vs_5y_mean_pct': (ev_ebitda_ttm / ev5['mean'] - 1) * 100 if ev_ebitda_ttm is not None and ev5['mean'] not in (None, 0) else None,
        'ev_ebitda_vs_10y_mean_pct': (ev_ebitda_ttm / ev10['mean'] - 1) * 100 if ev_ebitda_ttm is not None and ev10['mean'] not in (None, 0) else None,
        'n_per_5y': per5['n'], 'n_per_10y': per10['n'], 'n_pbr_5y': pbr5['n'], 'n_pbr_10y': pbr10['n'], 'n_ev_ebitda_5y': ev5['n'], 'n_ev_ebitda_10y': ev10['n'],
        'winsorize_percentile': winsorize_percentile,
        'outlier_method': 'winsorize' if include_outlier_trim else None,
        'outlier_removed_count_5y': per5['removed_count'] + pbr5['removed_count'] + ev5['removed_count'],
        'outlier_removed_count_10y': per10['removed_count'] + pbr10['removed_count'] + ev10['removed_count'],
        'outlier_note': '적자/일회성/급변으로 평균 왜곡 가능성을 완화하기 위해 winsorize 적용' if include_outlier_trim else '미적용',
        'implied_eps_ttm': implied_eps_ttm, 'implied_eps_fwd': implied_eps_fwd,
        'assumed_discount_rate_r': ASSUMED_DISCOUNT_RATE_R if include_implied_growth else None,
        'implied_growth_g': implied_growth_g,
        'implied_growth_note': '단순 근사식 g=r-1/PER_fwd. 참고용.' if include_implied_growth else None,
        **cons_data,
        'per_regime_shift_pct': calc_regime_shift(per5['mean'], per10['mean']),
        'pbr_regime_shift_pct': calc_regime_shift(pbr5['mean'], pbr10['mean']),
        'ev_ebitda_regime_shift_pct': calc_regime_shift(ev5['mean'], ev10['mean']),
        'price_return_6m_pct': price_return_6m_pct, 'fwd_eps_change_6m_pct': fwd_eps_change_6m_pct, 'momentum_gap_pct': momentum_gap_pct,
        'eps_ttm': eps_ttm, **eps_norm,
        **event_ret,
        'missing_ratio_per_10y': per10['missing_ratio'], 'missing_ratio_pbr_10y': pbr10['missing_ratio'], 'missing_ratio_ev_ebitda_10y': ev10['missing_ratio'],
        'forward_multiple_status': forward_data['forward_multiple_status'],
        'overall_data_quality_score': score,
        'data_quality_notes': '; '.join(error_log[-5:]) if error_log else '전반적 양호',
    }

    return pd.DataFrame([result]), error_log


def run_phase1_data_with_report(*args, **kwargs):
    result_df, error_log = run_phase1_data(*args, **kwargs)

    condition_cols = [
        'ticker', 'resolved_name', 'market', 'base_date',
        'forward_multiple_status', 'consensus_data_status', 'overall_data_quality_score'
    ]
    value_cols = [
        'base_close', 'base_mcap',
        'per_ttm', 'pbr_ttm', 'per_fwd_12m', 'ev_ebitda_ttm',
        'per_5y_percentile', 'per_10y_percentile', 'pbr_5y_percentile', 'pbr_10y_percentile',
        'per_vs_5y_mean_pct', 'per_vs_10y_mean_pct',
        'pbr_vs_5y_mean_pct', 'pbr_vs_10y_mean_pct',
        'price_return_6m_pct', 'momentum_gap_pct'
    ]

    condition_cols = [c for c in condition_cols if c in result_df.columns]
    value_cols = [c for c in value_cols if c in result_df.columns]

    print('=== PHASE1 실행 조건 ===')
    print(result_df[condition_cols].to_string(index=False))

    print('\n=== PHASE1 핵심 지표 ===')
    print(result_df[value_cols].to_string(index=False))

    print('\n=== PHASE1 전체 결과(1행) ===')
    print(result_df)

    print('\n=== 실행 로그(참고) ===')
    if not error_log:
        print('- 없음')
    else:
        for msg in error_log:
            print('-', msg)

    hard_errors = [m for m in error_log if any(k in m.lower() for k in ['failed', 'timeout', 'fatal'])]
    if hard_errors:
        print('\n[주의] 일부 데이터 수집 실패가 있습니다. 위 로그를 확인하세요.')
    else:
        print('\n[안내] 치명적 오류 없이 계산이 완료되었습니다.')
    return result_df, error_log




def prompt_and_run_phase1_one_input():
    print('=== PHASE1 KRX 입력 UI ===')
    korean_name = input('종목명 입력 (예: 삼성전자): ').strip()
    if not korean_name:
        raise ValueError('종목명은 필수입니다.')

    result_df, error_log = run_phase1_data_with_report(
        korean_name=korean_name,
        base_date=None,
        percentile_basis='daily',
        event_mode='auto',
        event_date_manual=None,
        auto_event_days=90,
        include_ev_ebitda=True,
        include_consensus=True,
        include_outlier_trim=True,
        include_implied_growth=True,
        include_eps_normalization=True,
    )
    return result_df, error_log


def run_batch_phase1_data(korean_names: List[str], **kwargs):
    rows, error_dict = [], {}
    total = len(korean_names)
    for i, name in enumerate(korean_names, 1):
        print(f'[{i}/{total}] processing: {name}')
        try:
            df, errs = run_phase1_data(name, **kwargs)
            rows.append(df.iloc[0].to_dict())
            error_dict[name] = errs
        except Exception as e:
            rows.append({'resolved_name': name, 'fatal_error': str(e)})
            error_dict[name] = [f'fatal: {e}']
    return pd.DataFrame(rows), error_dict

# 단일 실행 예시
# result_df, error_log = run_phase1_data(
#     korean_name='삼성전자',
#     base_date='2025-01-17',
#     percentile_basis='daily',
#     event_mode='manual',
#     event_date_manual='2024-10-01',
# )
# display(result_df)
# error_log

# 배치 실행 예시
# batch_df, batch_error_logs = run_batch_phase1_data(
#     ['삼성전자', '삼성'],
#     base_date='2025-01-17',
#     percentile_basis='daily',
#     event_mode='auto',
#     auto_event_days=90,
# )
# display(batch_df)
# batch_error_logs


if __name__ == '__main__':
    prompt_and_run_phase1_one_input()
