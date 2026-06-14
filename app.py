import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta, date
from scipy.stats import norm

OPTIONABLE_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "META", "GOOGL", "AMD", "NFLX", "JPM", "BAC", "GS", "XOM", "CVX", "SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "SLV", "USO"]

INSUFFICIENT_LIQUIDITY_MESSAGE = ("Not enough liquid option quotes are available for this selection. " "Please choose a more actively traded strike, maturity, or option type.")

def get_last_price(ticker):

    try:
        spot = ticker.fast_info["lastPrice"]
    except Exception:
        try:
            history = ticker.history(period="5d")
            spot = history["Close"].dropna().iloc[-1] if not history.empty else np.nan
        except Exception:
            spot = np.nan

    try:
        spot = float(spot)
    except (TypeError, ValueError):
        return np.nan

    return spot if np.isfinite(spot) and spot > 0 else np.nan

def find_data(ticker,option):

    dfs = []
    maturities = ticker.options
    today = datetime.today()

    if len(maturities) == 0:
        return pd.DataFrame(columns=["strike", "bid", "ask", "currency", "option_type", "maturity", "mid"])

    for maturity in maturities:

        maturity_datetime = datetime.strptime(maturity, "%Y-%m-%d")

        if maturity_datetime <= today + timedelta(days=1):
            
            continue

        try:
            chain = ticker.option_chain(maturity)
        except Exception:
            continue

        if option == 'Calls':

            df = pd.DataFrame(chain.calls)
            df["option_type"] = "Calls"
        
        elif option == 'Puts':

            df = pd.DataFrame(chain.puts)
            df["option_type"] = "Puts"

        elif option == 'Calls & Puts':

            df_calls = pd.DataFrame(chain.calls)
            df_calls["option_type"] = "Calls"
            df_puts = pd.DataFrame(chain.puts)
            df_puts["option_type"] = "Puts"
            df = pd.concat([df_calls, df_puts], ignore_index=True)

        if df.empty:
            continue

        base_columns = ["strike", "bid", "ask", "currency", "option_type"]
        optional_columns = ["lastPrice", "volume", "openInterest"]
        available_optional_columns = [col for col in optional_columns if col in df.columns]
        df = df[base_columns + available_optional_columns]
        df["maturity"] = maturity
        dfs.append(df)
    
    if len(dfs) == 0:
        return pd.DataFrame(columns=["strike", "bid", "ask", "currency", "option_type", "maturity", "mid"])

    data = pd.concat(dfs, ignore_index=True)
    data["mid"] = (data["bid"] + data["ask"])/2
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["strike", "bid", "ask", "mid"])
    data = data[data["mid"] > 0]
    
    return data

def risk_free_rate(data):

    if data.empty:
        return np.nan
    
    currency = data["currency"].iloc[0]
    rf = np.nan
    
    if currency == "USD":

        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR"
        df = pd.read_csv(url)
        df = df[df["SOFR"] != "."]
        rf = float(df["SOFR"].iloc[-1]) / 100
    
    if currency == "EUR":
        
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=ECBESTRVOLWGTTRMDMNRT"
        df = pd.read_csv(url)
        df = df[df["ECBESTRVOLWGTTRMDMNRT"] != "."]
        rf = float(df["ECBESTRVOLWGTTRMDMNRT"].iloc[-1]) / 100
    
    if currency == "GBP":
        url = "https://www.bankofengland.co.uk/boeapps/database/fromshowcolumns.asp?CSVF=TT&DAT=RNG&FD=1&FM=Jan&FY=2024&TD=31&TM=Dec&TY=2030&FNY=&Filter=N&FromSeries=1&ToSeries=50&SeriesCodes=IUDSOIA&UsingCodes=Y&VPD=Y"
        df = pd.read_csv(url)
        df = df[df["IUDSOIA"] != "."]
        rf = float(df["IUDSOIA"].iloc[-1]) / 100
    
    return rf

def pricer_bs(strike,maturity,IV,spot,rf,option_type,barrier=None,dividend_yield=0.0,cash_payout=1.0,extra_strike=None,lower=None,upper=None):

    nan_result = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    if isinstance(maturity, date) and not isinstance(maturity, datetime):
        maturity = datetime.combine(maturity, datetime.min.time())

    today = datetime.today()
    T = (maturity - today).days / 365.0

    if (T <= 0 or IV <= 0 or spot <= 0 or strike <= 0 or not np.isfinite(rf) or not np.isfinite(dividend_yield)):
        return nan_result

    option_type = option_type.strip()

    def vanilla_price(S, K, tau, sigma, r, q, opt):
        if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return np.nan

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if opt == "Calls":
            return S * np.exp(-q * tau) * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

        if opt == "Puts":
            return K * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)

        return np.nan

    def digital_price(S, K, tau, sigma, r, q, opt, Q):
        if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return np.nan

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if opt == "Digital Calls Cash or Nothing":
            return Q * np.exp(-r * tau) * norm.cdf(d2)

        if opt == "Digital Puts Cash or Nothing":
            return Q * np.exp(-r * tau) * norm.cdf(-d2)

        if opt == "Digital Calls Asset or Nothing":
            return S * np.exp(-q * tau) * norm.cdf(d1)

        if opt == "Digital Puts Asset or Nothing":
            return S * np.exp(-q * tau) * norm.cdf(-d1)

        return np.nan

    def gap_price(S, K_trigger, K_payoff, tau, sigma, r, q, opt):
        if K_payoff is None or K_payoff <= 0:
            return np.nan

        d1 = (np.log(S / K_trigger) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if opt == "Gap Calls":
            return S * np.exp(-q * tau) * norm.cdf(d1) - K_payoff * np.exp(-r * tau) * norm.cdf(d2)

        if opt == "Gap Puts":
            return K_payoff * np.exp(-r * tau) * norm.cdf(-d2) - S * np.exp(-q * tau) * norm.cdf(-d1)

        return np.nan

    def capped_price(S, K, cap_or_floor, tau, sigma, r, q, opt):
        if cap_or_floor is None or cap_or_floor <= 0:
            return np.nan

        if opt == "Capped Calls":
            cap = cap_or_floor
            if cap <= K:
                return np.nan
            return vanilla_price(S, K, tau, sigma, r, q, "Calls") - vanilla_price(S, cap, tau, sigma, r, q, "Calls")

        if opt == "Capped Puts":
            floor = cap_or_floor
            if floor >= K:
                return np.nan
            return vanilla_price(S, K, tau, sigma, r, q, "Puts") - vanilla_price(S, floor, tau, sigma, r, q, "Puts")

        return np.nan

    def barrier_price(S, K, H, tau, sigma, r, q, opt):
        if tau <= 0 or sigma <= 0 or S <= 0 or K <= 0 or H is None or H <= 0:
            return np.nan

        is_call = "Calls" in opt
        is_up = "Up" in opt
        is_down = "Down" in opt
        is_in = "In" in opt

        vanilla_opt = "Calls" if is_call else "Puts"
        vanilla = vanilla_price(S, K, tau, sigma, r, q, vanilla_opt)

        if is_down and S <= H:
            return vanilla if is_in else 0.0

        if is_up and S >= H:
            return vanilla if is_in else 0.0

        phi = 1 if is_call else -1
        eta = 1 if is_down else -1

        sqrt_tau = np.sqrt(tau)
        mu = (r - q - 0.5 * sigma**2) / sigma**2

        d1 = (np.log(S / K) / (sigma * sqrt_tau)) + (mu + 1) * sigma * sqrt_tau
        d2 = d1 - sigma * sqrt_tau

        h1 = (np.log(S / H) / (sigma * sqrt_tau)) + (mu + 1) * sigma * sqrt_tau
        h2 = h1 - sigma * sqrt_tau

        y1 = (np.log(H**2 / (S * K)) / (sigma * sqrt_tau)) + (mu + 1) * sigma * sqrt_tau
        y2 = y1 - sigma * sqrt_tau

        y3 = (np.log(H / S) / (sigma * sqrt_tau)) + (mu + 1) * sigma * sqrt_tau
        y4 = y3 - sigma * sqrt_tau

        disc_q = np.exp(-q * tau)
        disc_r = np.exp(-r * tau)

        A = phi * S * disc_q * norm.cdf(phi * d1) - phi * K * disc_r * norm.cdf(phi * d2)
        B = phi * S * disc_q * norm.cdf(phi * h1) - phi * K * disc_r * norm.cdf(phi * h2)

        C = (phi * S * disc_q * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y1) - phi * K * disc_r * (H / S) ** (2 * mu) * norm.cdf(eta * y2))
        D = (phi * S * disc_q * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y3) - phi * K * disc_r * (H / S) ** (2 * mu) * norm.cdf(eta * y4))

        if opt == "Calls Down and In":
            return C if K > H else A - B + D

        if opt == "Calls Down and Out":
            return A - C if K > H else B - D

        if opt == "Calls Up and In":
            return A if K > H else B - C + D

        if opt == "Calls Up and Out":
            return 0.0 if K > H else A - B + C - D

        if opt == "Puts Down and In":
            return B - C + D if K > H else A

        if opt == "Puts Down and Out":
            return A - B + C - D if K > H else 0.0

        if opt == "Puts Up and In":
            return A - B + D if K > H else C

        if opt == "Puts Up and Out":
            return B - D if K > H else A - C

        return np.nan

    def price_only(S, K, tau, sigma, r, q):
        if option_type in ["Calls", "Puts"]:
            return vanilla_price(S, K, tau, sigma, r, q, option_type)

        if option_type in ["Digital Calls Cash or Nothing","Digital Puts Cash or Nothing","Digital Calls Asset or Nothing","Digital Puts Asset or Nothing"]:
            return digital_price(S, K, tau, sigma, r, q, option_type, cash_payout)

        if option_type in ["Calls Up and In","Calls Up and Out","Calls Down and In","Calls Down and Out","Puts Up and In","Puts Up and Out","Puts Down and In","Puts Down and Out"]:
            return barrier_price(S, K, barrier, tau, sigma, r, q, option_type)

        if option_type in ["Gap Calls", "Gap Puts"]:
            return gap_price(S, K, extra_strike, tau, sigma, r, q, option_type)

        if option_type in ["Capped Calls", "Capped Puts"]:
            return capped_price(S, K, extra_strike, tau, sigma, r, q, option_type)

        return np.nan

    price = price_only(spot, strike, T, IV, rf, dividend_yield)

    if not np.isfinite(price):
        return nan_result

    if option_type in ["Calls", "Puts"]:
        d1 = (np.log(spot / strike) + (rf - dividend_yield + 0.5 * IV**2) * T) / (IV * np.sqrt(T))
        d2 = d1 - IV * np.sqrt(T)

        gamma = np.exp(-dividend_yield * T) * norm.pdf(d1) / (spot * IV * np.sqrt(T))
        vega = spot * np.exp(-dividend_yield * T) * norm.pdf(d1) * np.sqrt(T)
        volga = vega * d1 * d2 / IV
        vanna = -np.exp(-dividend_yield * T) * norm.pdf(d1) * d2 / IV

        if option_type == "Calls":
            delta = np.exp(-dividend_yield * T) * norm.cdf(d1)
            theta = (-spot * np.exp(-dividend_yield * T) * norm.pdf(d1) * IV / (2 * np.sqrt(T)) - rf * strike * np.exp(-rf * T) * norm.cdf(d2) + dividend_yield * spot * np.exp(-dividend_yield * T) * norm.cdf(d1)) / 365.0
            rho = strike * T * np.exp(-rf * T) * norm.cdf(d2)

        else:
            delta = np.exp(-dividend_yield * T) * (norm.cdf(d1) - 1)
            theta = (-spot * np.exp(-dividend_yield * T) * norm.pdf(d1) * IV / (2 * np.sqrt(T)) + rf * strike * np.exp(-rf * T) * norm.cdf(-d2) - dividend_yield * spot * np.exp(-dividend_yield * T) * norm.cdf(-d1)) / 365.0
            rho = -strike * T * np.exp(-rf * T) * norm.cdf(-d2)

        return price, delta, gamma, vega, theta, rho, volga, vanna

    hS = max(spot * SPOT_BUMP_RELATIVE, 1e-4)
    hV = max(min(max(IV * VOL_BUMP_RELATIVE, VOL_BUMP_ABSOLUTE), IV * 0.50), 1e-4)
    hR = RATE_BUMP_ABSOLUTE
    hT = 1 / 365.0

    p_S_up = price_only(spot + hS, strike, T, IV, rf, dividend_yield)
    p_S_down = price_only(max(spot - hS, 1e-8), strike, T, IV, rf, dividend_yield)

    delta = (p_S_up - p_S_down) / (2 * hS)
    gamma = (p_S_up - 2 * price + p_S_down) / (hS**2)

    p_v_up = price_only(spot, strike, T, IV + hV, rf, dividend_yield)
    p_v_down = price_only(spot, strike, T, max(IV - hV, 1e-8), rf, dividend_yield)

    vega = (p_v_up - p_v_down) / (2 * hV)
    volga = (p_v_up - 2 * price + p_v_down) / (hV**2)

    p_r_up = price_only(spot, strike, T, IV, rf + hR, dividend_yield)
    p_r_down = price_only(spot, strike, T, IV, rf - hR, dividend_yield)

    rho = (p_r_up - p_r_down) / (2 * hR)

    if T > hT:
        p_tomorrow = price_only(spot, strike, T - hT, IV, rf, dividend_yield)
        theta = p_tomorrow - price
    else:
        theta = np.nan

    p_up_up = price_only(spot + hS, strike, T, IV + hV, rf, dividend_yield)
    p_up_down = price_only(spot + hS, strike, T, max(IV - hV, 1e-8), rf, dividend_yield)
    p_down_up = price_only(max(spot - hS, 1e-8), strike, T, IV + hV, rf, dividend_yield)
    p_down_down = price_only(max(spot - hS, 1e-8), strike, T, max(IV - hV, 1e-8), rf, dividend_yield)

    vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hV)

    return price, delta, gamma, vega, theta, rho, volga, vanna

def pricer_mc(strike,maturity,IV,spot,rf,option_type,barrier=None,dividend_yield=0.0,cash_payout=1.0,extra_strike=None,lower=None,upper=None,price_only=False,plot=True,num_simulations=200_000,seed=42,steps_per_year=252):
    
    nan_result = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    if isinstance(maturity, date) and not isinstance(maturity, datetime):
        maturity = datetime.combine(maturity, datetime.min.time())

    today = datetime.today()
    T = (maturity - today).days / 365.0

    if (T <= 0 or IV <= 0 or spot <= 0 or strike <= 0 or not np.isfinite(rf) or not np.isfinite(dividend_yield)):
        return np.nan if price_only else nan_result

    option_type = option_type.strip()

    barrier_types = ["Calls Up and In","Calls Up and Out","Calls Down and In","Calls Down and Out","Puts Up and In","Puts Up and Out","Puts Down and In","Puts Down and Out",]

    valid_option_types = ["Calls","Puts","Digital Calls Cash or Nothing","Digital Puts Cash or Nothing","Digital Calls Asset or Nothing","Digital Puts Asset or Nothing","Gap Calls","Gap Puts","Capped Calls","Capped Puts"] + barrier_types

    if option_type not in valid_option_types:
        return np.nan if price_only else nan_result

    def to_tau(input_date):
        if input_date is None:
            return None

        if isinstance(input_date, date) and not isinstance(input_date, datetime):
            input_date = datetime.combine(input_date, datetime.min.time())

        return (input_date - today).days / 365.0

    def simulate_paths(S0, tau, sigma, r, q, rng_seed):
        n_steps = max(1, int(np.ceil(tau * steps_per_year)))
        dt = tau / n_steps

        rng = np.random.default_rng(rng_seed)
        Z = rng.standard_normal((num_simulations, n_steps))

        log_returns = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

        S_paths = S0 * np.exp(np.cumsum(log_returns, axis=1))
        S_paths = np.column_stack((np.full(num_simulations, S0), S_paths))

        return S_paths

    def event_index(tau_event, tau_total, n_steps, allow_zero=False):
        if tau_event is None:
            return None

        if allow_zero:
            if tau_event < 0 or tau_event >= tau_total:
                return None
        else:
            if tau_event <= 0 or tau_event >= tau_total:
                return None

        idx = int(round((tau_event / tau_total) * n_steps))
        idx = max(0 if allow_zero else 1, min(idx, n_steps))

        return idx

    def mc_price_core(S0, K, tau, sigma, r, q, return_paths=False):
        if tau <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
            return (np.nan, None) if return_paths else np.nan

        S_paths = simulate_paths(S0, tau, sigma, r, q, seed)
        n_steps = S_paths.shape[1] - 1

        S_T = S_paths[:, -1]
        discount = np.exp(-r * tau)

        call_payoff = np.maximum(S_T - K, 0.0)
        put_payoff = np.maximum(K - S_T, 0.0)

        if option_type == "Calls":
            payoffs = call_payoff

        elif option_type == "Puts":
            payoffs = put_payoff

        elif option_type == "Digital Calls Cash or Nothing":
            payoffs = cash_payout * (S_T > K)

        elif option_type == "Digital Puts Cash or Nothing":
            payoffs = cash_payout * (S_T < K)

        elif option_type == "Digital Calls Asset or Nothing":
            payoffs = S_T * (S_T > K)

        elif option_type == "Digital Puts Asset or Nothing":
            payoffs = S_T * (S_T < K)

        elif option_type in barrier_types:
            if barrier is None or barrier <= 0:
                price = np.nan
                return (price, S_paths) if return_paths else price

            is_call = "Calls" in option_type
            is_up = "Up" in option_type
            is_in = "In" in option_type

            vanilla_payoff = call_payoff if is_call else put_payoff

            if is_up:
                barrier_touched = np.max(S_paths, axis=1) >= barrier
            else:
                barrier_touched = np.min(S_paths, axis=1) <= barrier

            if is_in:
                payoffs = np.where(barrier_touched, vanilla_payoff, 0.0)
            else:
                payoffs = np.where(~barrier_touched, vanilla_payoff, 0.0)

        elif option_type == "Gap Calls":
            if extra_strike is None or extra_strike <= 0:
                price = np.nan
                return (price, S_paths) if return_paths else price

            payoffs = np.where(S_T > K, S_T - extra_strike, 0.0)

        elif option_type == "Gap Puts":
            if extra_strike is None or extra_strike <= 0:
                price = np.nan
                return (price, S_paths) if return_paths else price

            payoffs = np.where(S_T < K, extra_strike - S_T, 0.0)

        elif option_type == "Capped Calls":
            cap = extra_strike

            if cap is None or cap <= K:
                price = np.nan
                return (price, S_paths) if return_paths else price

            payoffs = np.minimum(np.maximum(S_T - K, 0.0), cap - K)

        elif option_type == "Capped Puts":
            floor = extra_strike

            if floor is None or floor >= K or floor <= 0:
                price = np.nan
                return (price, S_paths) if return_paths else price

            payoffs = np.minimum(np.maximum(K - S_T, 0.0), K - floor)

        else:
            return (np.nan, S_paths) if return_paths else np.nan

        payoffs = np.asarray(payoffs, dtype=float)
        price = discount * np.mean(payoffs)

        return (price, S_paths) if return_paths else price

    if price_only:
        return mc_price_core(spot, strike, T, IV, rf, dividend_yield, return_paths=False)

    price, S_paths = mc_price_core(spot, strike, T, IV, rf, dividend_yield, return_paths=True)

    if not np.isfinite(price):
        return nan_result

    hS = max(spot * SPOT_BUMP_RELATIVE, 1e-4)
    hV = max(min(max(IV * VOL_BUMP_RELATIVE, VOL_BUMP_ABSOLUTE), IV * 0.50), 1e-4)
    hR = RATE_BUMP_ABSOLUTE
    hT = 1 / 365.0

    p_S_up = mc_price_core(spot + hS, strike, T, IV, rf, dividend_yield, return_paths=False)
    p_S_down = mc_price_core(max(spot - hS, 1e-8), strike, T, IV, rf, dividend_yield, return_paths=False)

    delta = (p_S_up - p_S_down) / (2 * hS)
    gamma = (p_S_up - 2 * price + p_S_down) / (hS**2)

    p_v_up = mc_price_core(spot, strike, T, IV + hV, rf, dividend_yield, return_paths=False)
    p_v_down = mc_price_core(spot, strike, T, max(IV - hV, 1e-8), rf, dividend_yield, return_paths=False)

    vega = (p_v_up - p_v_down) / (2 * hV)
    volga = (p_v_up - 2 * price + p_v_down) / (hV**2)

    p_r_up = mc_price_core(spot, strike, T, IV, rf + hR, dividend_yield, return_paths=False)
    p_r_down = mc_price_core(spot, strike, T, IV, rf - hR, dividend_yield, return_paths=False)

    rho = (p_r_up - p_r_down) / (2 * hR)

    if T > hT:
        p_tomorrow = mc_price_core(spot, strike, T - hT, IV, rf, dividend_yield, return_paths=False)
        theta = p_tomorrow - price
    else:
        theta = np.nan

    p_up_up = mc_price_core(spot + hS, strike, T, IV + hV, rf, dividend_yield, return_paths=False)
    p_up_down = mc_price_core(spot + hS,strike,T,max(IV - hV, 1e-8),rf,dividend_yield,return_paths=False)
    p_down_up = mc_price_core(max(spot - hS, 1e-8),strike,T,IV + hV,rf,dividend_yield,return_paths=False)
    p_down_down = mc_price_core(max(spot - hS, 1e-8),strike,T,max(IV - hV, 1e-8),rf,dividend_yield,return_paths=False)

    vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hV)

    if plot and S_paths is not None:
        plt.style.use("dark_background")

        n_paths_plot = 250
        n_steps = S_paths.shape[1] - 1
        time_grid = np.arange(n_steps + 1)

        n_paths_plot = min(n_paths_plot, num_simulations)

        plot_rng = np.random.default_rng(seed + 1)
        selected_paths = plot_rng.choice(num_simulations, size=n_paths_plot, replace=False)

        fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
        ax.set_facecolor("#0e1117")

        for i in selected_paths:
            ax.plot(time_grid,S_paths[i],color="#00d4ff",linewidth=0.8,alpha=0.18)

        mean_path = np.mean(S_paths, axis=0)

        ax.plot(time_grid,mean_path,color="white",linewidth=2.5,label="Mean path")

        ax.axhline(spot,color="#d0d0d0",linestyle="--",linewidth=1,alpha=0.5,label=f"Initial spot: {spot:.2f}")

        ax.axhline(strike,color="#ff4d4d",linestyle="--",linewidth=1.5,alpha=0.8,label=f"Strike: {strike:.2f}")

        if barrier is not None:
            ax.axhline(barrier,color="#ffcc00",linestyle="--",linewidth=1.8,alpha=0.9,label=f"Barrier: {barrier:.2f}")

        if lower is not None:
            ax.axhline(lower,color="#66ff99",linestyle="--",linewidth=1.4,alpha=0.8,label=f"Lower: {lower:.2f}")

        if upper is not None:
            ax.axhline(upper,color="#cc99ff",linestyle="--",linewidth=1.4,alpha=0.8,label=f"Upper: {upper:.2f}")

        ax.set_title(f"Monte Carlo simulated underlying paths\n" f"{option_type} | Maturity {maturity.date()}",fontsize=16,fontweight="bold",color="white",pad=18)

        ax.set_xlabel("Simulated trading days", fontsize=12, color="#d0d0d0")
        ax.set_ylabel("Simulated underlying price", fontsize=12, color="#d0d0d0")

        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="white")

        ax.tick_params(axis="x", colors="#d0d0d0")
        ax.tick_params(axis="y", colors="#d0d0d0")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")

        ax.legend(loc="upper left",facecolor="#0e1117",edgecolor="#555555",fontsize=10)

        ax.text(0.98,0.95,f"Monte Carlo option price: {price:.4f}",transform=ax.transAxes,ha="right",va="top",fontsize=12,fontweight="bold",color="white",zorder=30,bbox=dict(boxstyle="round,pad=0.45",facecolor="#1c1f26",edgecolor="#00d4ff",linewidth=1.2,alpha=0.95))

        plt.tight_layout()
        plt.show()

    return price, delta, gamma, vega, theta, rho, volga, vanna

def newton_raphson(ticker,data,initial_IV,spot,rf,max_iter=100,dividend_yield=0.0):
    
    IV = initial_IV
    mid = data["mid"]
    option_type = data["option_type"]
    strike = data["strike"]
    maturity = datetime.strptime(data["maturity"], "%Y-%m-%d")

    price, delta, gamma, vega, theta, rho, volga, vanna = pricer_bs(strike, maturity, IV, spot, rf, option_type, dividend_yield=dividend_yield)

    for _ in range(max_iter):

        if not np.isfinite(price) or not np.isfinite(vega) or not np.isfinite(IV):
            return np.nan

        if abs(price - mid) <= 1e-5:
            return IV

        if abs(vega) < 1e-8:
            return np.nan

        IV = IV - (price - mid) / vega

        if IV <= 0:
            return np.nan

        price, delta, gamma, vega, theta, rho, volga, vanna = pricer_bs(strike, maturity, IV, spot, rf, option_type, dividend_yield=dividend_yield)

    return np.nan

GREEK_NAMES = ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Volga", "Vanna"]

VANILLA_TYPES = ["Calls", "Puts"]

EXOTIC_OPTION_TYPES = ["Calls", "Puts","Digital Calls Cash or Nothing", "Digital Puts Cash or Nothing","Digital Calls Asset or Nothing", "Digital Puts Asset or Nothing","Calls Up and In", "Calls Up and Out", "Calls Down and In", "Calls Down and Out","Puts Up and In", "Puts Up and Out", "Puts Down and In", "Puts Down and Out","Gap Calls", "Gap Puts", "Capped Calls", "Capped Puts"]

BARRIER_TYPES = ["Calls Up and In", "Calls Up and Out", "Calls Down and In", "Calls Down and Out","Puts Up and In", "Puts Up and Out", "Puts Down and In", "Puts Down and Out"]

DIGITAL_TYPES = ["Digital Calls Cash or Nothing", "Digital Puts Cash or Nothing","Digital Calls Asset or Nothing", "Digital Puts Asset or Nothing"]

EXTRA_STRIKE_TYPES = ["Gap Calls", "Gap Puts", "Capped Calls", "Capped Puts"]

def option_requires_pricing_strike(option_type):

    return True

def get_effective_pricing_strike(option_type, spot, reference_strike):

    return reference_strike

def get_strike_input_label(option_type):

    return "Strike"

def get_available_maturities(symbol):

    ticker = yf.Ticker(symbol)
    today = datetime.today()
    maturities = []

    for maturity in ticker.options:
        maturity_datetime = datetime.strptime(maturity, "%Y-%m-%d")

        if maturity_datetime > today + timedelta(days=1):
            maturities.append(maturity)

    return maturities

def load_option_data(symbol, option_type):

    ticker = yf.Ticker(symbol)
    return find_data(ticker, option_type)

def load_spot_price(symbol):

    ticker = yf.Ticker(symbol)
    return get_last_price(ticker)

def get_dividend_yield_from_yahoo(symbol, spot):

    ticker = yf.Ticker(symbol)
    dividend_yield = np.nan
    annual_dividend = np.nan
    method = "No dividend data found on Yahoo Finance"

    try:
        dividends = ticker.dividends

        if dividends is not None and len(dividends) > 0:
            dividends = dividends.dropna()
            dividends.index = pd.to_datetime(dividends.index)
            one_year_ago = pd.Timestamp.today(tz=dividends.index.tz) - pd.DateOffset(years=1)
            recent_dividends = dividends[dividends.index >= one_year_ago]

            if len(recent_dividends) > 0 and np.isfinite(spot) and spot > 0:
                annual_dividend = float(recent_dividends.sum())
                dividend_yield = annual_dividend / spot
                method = "Trailing 12-month dividends divided by spot"
    except Exception:
        pass

    if not np.isfinite(dividend_yield):
        try:
            info = ticker.info
            raw_dividend_yield = info.get("dividendYield", np.nan)

            if raw_dividend_yield is not None and np.isfinite(raw_dividend_yield):
                dividend_yield = float(raw_dividend_yield)
                if dividend_yield > 1:
                    dividend_yield = dividend_yield / 100
                method = "Yahoo Finance dividendYield field"
        except Exception:
            pass

    if not np.isfinite(dividend_yield):
        dividend_yield = 0.0

    return dividend_yield, annual_dividend, method

def safe_float(value, default=np.nan):

    try:
        value = float(value)
    except (TypeError, ValueError):
        return default

    return value if np.isfinite(value) else default

def format_strike(strike):

    return f"{strike:.2f}".rstrip("0").rstrip(".")

def format_percent(x):

    if not np.isfinite(x):
        return "N/A"

    return f"{x * 100:.2f}%"

def format_price(x, currency=""):

    if not np.isfinite(x):
        return "N/A"

    suffix = f" {currency}" if isinstance(currency, str) and len(currency) > 0 else ""
    return f"{x:,.4f}{suffix}"

def result_tuple_to_series(result):

    return pd.Series(result, index=GREEK_NAMES, dtype="float64")

def build_single_result_dataframe(method_name, result):

    return pd.DataFrame({"Metric": GREEK_NAMES, method_name: result}).set_index("Metric")

def build_comparison_dataframe(bs_result, mc_result):

    df = pd.DataFrame({"Black-Scholes": result_tuple_to_series(bs_result),"Monte Carlo": result_tuple_to_series(mc_result)})

    df["Absolute difference"] = df["Monte Carlo"] - df["Black-Scholes"]
    df["Relative difference"] = np.where(df["Black-Scholes"].abs() > 1e-12,df["Absolute difference"] / df["Black-Scholes"],np.nan)

    return df

def display_result_metrics(result, currency=""):

    price, delta, gamma, vega, theta, rho, volga, vanna = result

    metric_cols = st.columns(4)
    metric_cols[0].metric("Price", format_price(price, currency))
    metric_cols[1].metric("Delta", f"{delta:.6f}" if np.isfinite(delta) else "N/A")
    metric_cols[2].metric("Gamma", f"{gamma:.6f}" if np.isfinite(gamma) else "N/A")
    metric_cols[3].metric("Vega", f"{vega:.6f}" if np.isfinite(vega) else "N/A")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Theta / day", f"{theta:.6f}" if np.isfinite(theta) else "N/A")
    metric_cols[1].metric("Rho", f"{rho:.6f}" if np.isfinite(rho) else "N/A")
    metric_cols[2].metric("Volga", f"{volga:.6f}" if np.isfinite(volga) else "N/A")
    metric_cols[3].metric("Vanna", f"{vanna:.6f}" if np.isfinite(vanna) else "N/A")

def create_price_comparison_figure(comparison_df):

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    prices = comparison_df.loc["Price", ["Black-Scholes", "Monte Carlo"]]
    ax.bar(prices.index, prices.values, color=["#00d4ff", "#ffcc00"], alpha=0.85)

    ax.set_title("Option Price Comparison", fontsize=16, fontweight="bold", color="white", pad=16)
    ax.set_ylabel("Option price", fontsize=12, color="#d0d0d0")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25, color="white")
    ax.tick_params(axis="x", colors="#d0d0d0")
    ax.tick_params(axis="y", colors="#d0d0d0")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    for index, value in enumerate(prices.values):
        ax.annotate(f"{value:.4f}", xy=(index, value), xytext=(0, 8), textcoords="offset points", ha="center", color="white", fontsize=10)

    plt.tight_layout()
    return fig

def create_greeks_comparison_figure(comparison_df):

    plt.style.use("dark_background")

    greeks = ["Delta", "Gamma", "Vega", "Theta", "Rho", "Volga", "Vanna"]
    plot_data = comparison_df.loc[greeks, ["Black-Scholes", "Monte Carlo"]].copy()

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    x = np.arange(len(greeks))
    width = 0.38

    ax.bar(x - width / 2, plot_data["Black-Scholes"].values, width, label="Black-Scholes", color="#00d4ff", alpha=0.85)
    ax.bar(x + width / 2, plot_data["Monte Carlo"].values, width, label="Monte Carlo", color="#ffcc00", alpha=0.85)

    ax.set_title("Greeks Comparison", fontsize=16, fontweight="bold", color="white", pad=16)
    ax.set_xticks(x)
    ax.set_xticklabels(greeks, rotation=0, color="#d0d0d0")
    ax.set_ylabel("Greek value", fontsize=12, color="#d0d0d0")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25, color="white")
    ax.tick_params(axis="y", colors="#d0d0d0")
    ax.legend(facecolor="#0e1117", edgecolor="#555555")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    plt.tight_layout()
    return fig

def create_payoff_figure(spot, strike, option_type, price=None, lower=None, upper=None, barrier=None, extra_strike=None, cash_payout=1.0):

    plt.style.use("dark_background")

    x_min = max(0.01, spot * 0.4)
    x_max = spot * 1.8

    if barrier is not None and np.isfinite(barrier):
        x_min = max(0.01, min(x_min, barrier * 0.7))
        x_max = max(x_max, barrier * 1.3)

    if lower is not None and np.isfinite(lower):
        x_min = max(0.01, min(x_min, lower * 0.7))

    if upper is not None and np.isfinite(upper):
        x_max = max(x_max, upper * 1.3)

    S_T = np.linspace(x_min, x_max, 400)

    vanilla_reference = None
    payoff_label = "Payoff at maturity"

    if option_type == "Calls":
        payoff = np.maximum(S_T - strike, 0.0)
    elif option_type == "Puts":
        payoff = np.maximum(strike - S_T, 0.0)
    elif option_type == "Digital Calls Cash or Nothing":
        payoff = cash_payout * (S_T > strike)
    elif option_type == "Digital Puts Cash or Nothing":
        payoff = cash_payout * (S_T < strike)
    elif option_type == "Digital Calls Asset or Nothing":
        payoff = S_T * (S_T > strike)
    elif option_type == "Digital Puts Asset or Nothing":
        payoff = S_T * (S_T < strike)
    elif option_type in BARRIER_TYPES:
        if barrier is None or not np.isfinite(barrier) or barrier <= 0:
            payoff = np.full_like(S_T, np.nan)
        else:
            is_call = "Calls" in option_type
            is_up = "Up" in option_type
            is_in = "In" in option_type

            vanilla_reference = np.maximum(S_T - strike, 0.0) if is_call else np.maximum(strike - S_T, 0.0)

            # A one-dimensional payoff diagram cannot fully represent a path-dependent barrier.
            # This terminal-axis proxy makes the barrier visible on the chart: down barriers are
            # treated as touched for S_T <= H, up barriers as touched for S_T >= H.
            barrier_touched_proxy = S_T >= barrier if is_up else S_T <= barrier

            if is_in:
                payoff = np.where(barrier_touched_proxy, vanilla_reference, 0.0)
            else:
                payoff = np.where(~barrier_touched_proxy, vanilla_reference, 0.0)

            payoff_label = "Barrier-adjusted payoff"
    elif option_type == "Gap Calls":
        payoff = np.where(S_T > strike, S_T - extra_strike, 0.0) if extra_strike else np.full_like(S_T, np.nan)
    elif option_type == "Gap Puts":
        payoff = np.where(S_T < strike, extra_strike - S_T, 0.0) if extra_strike else np.full_like(S_T, np.nan)
    elif option_type == "Capped Calls":
        payoff = np.minimum(np.maximum(S_T - strike, 0.0), extra_strike - strike) if extra_strike else np.full_like(S_T, np.nan)
    elif option_type == "Capped Puts":
        payoff = np.minimum(np.maximum(strike - S_T, 0.0), strike - extra_strike) if extra_strike else np.full_like(S_T, np.nan)
    elif option_type == "Range Digital Cash or Nothing":
        payoff = cash_payout * ((S_T > lower) & (S_T < upper)) if lower and upper else np.full_like(S_T, np.nan)
    elif option_type == "Supershare":
        payoff = S_T * ((S_T > lower) & (S_T < upper)) if lower and upper else np.full_like(S_T, np.nan)
    else:
        payoff = np.maximum(S_T - strike, 0.0) if "Calls" in option_type else np.maximum(strike - S_T, 0.0)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    ax.plot(S_T, payoff, color="#00d4ff", linewidth=2.5, label=payoff_label)
    ax.fill_between(S_T, payoff, alpha=0.12, color="#00d4ff")

    if vanilla_reference is not None:
        ax.plot(S_T, vanilla_reference, color="#d0d0d0", linestyle=":", linewidth=1.6, alpha=0.75, label="Vanilla reference payoff")

    ax.axvline(spot, color="#d0d0d0", linestyle="--", linewidth=1.2, alpha=0.75, label=f"Spot: {spot:.2f}")
    if option_requires_pricing_strike(option_type):
        ax.axvline(strike, color="#ff4d4d", linestyle="--", linewidth=1.4, alpha=0.85, label=f"Strike: {strike:.2f}")

    if barrier is not None and np.isfinite(barrier):
        ax.axvline(barrier, color="#ffcc00", linestyle="--", linewidth=1.4, alpha=0.85, label=f"Barrier: {barrier:.2f}")

    if lower is not None and np.isfinite(lower):
        ax.axvline(lower, color="#66ff99", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Lower: {lower:.2f}")

    if upper is not None and np.isfinite(upper):
        ax.axvline(upper, color="#cc99ff", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Upper: {upper:.2f}")

    if price is not None and np.isfinite(price):
        ax.axhline(price, color="white", linestyle=":", linewidth=1.4, alpha=0.75, label=f"Premium: {price:.4f}")

    ax.set_title(f"Payoff Diagram - {option_type}", fontsize=16, fontweight="bold", color="white", pad=16)
    ax.set_xlabel("Underlying price at maturity", fontsize=12, color="#d0d0d0")
    ax.set_ylabel("Payoff", fontsize=12, color="#d0d0d0")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="white")
    ax.tick_params(axis="x", colors="#d0d0d0")
    ax.tick_params(axis="y", colors="#d0d0d0")
    ax.legend(facecolor="#0e1117", edgecolor="#555555")


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    plt.tight_layout()
    return fig

def simulate_paths_for_display(spot, maturity, IV, rf, dividend_yield, num_simulations=2_000, seed=42, steps_per_year=252):

    if isinstance(maturity, date) and not isinstance(maturity, datetime):
        maturity = datetime.combine(maturity, datetime.min.time())

    T = (maturity - datetime.today()).days / 365.0

    if T <= 0 or IV <= 0 or spot <= 0:
        return None

    n_steps = max(1, int(np.ceil(T * steps_per_year)))
    dt = T / n_steps
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((num_simulations, n_steps))
    log_returns = (rf - dividend_yield - 0.5 * IV**2) * dt + IV * np.sqrt(dt) * Z
    paths = spot * np.exp(np.cumsum(log_returns, axis=1))
    paths = np.column_stack((np.full(num_simulations, spot), paths))

    return paths

def create_mc_paths_figure(spot, strike, maturity, IV, rf, dividend_yield, option_type, barrier=None, lower=None, upper=None, num_simulations=2_000, seed=42, steps_per_year=252):

    S_paths = simulate_paths_for_display(spot, maturity, IV, rf, dividend_yield, num_simulations=num_simulations, seed=seed, steps_per_year=steps_per_year)

    if S_paths is None:
        return None

    plt.style.use("dark_background")

    n_paths_plot = min(250, S_paths.shape[0])
    n_steps = S_paths.shape[1] - 1
    time_grid = np.arange(n_steps + 1)

    rng = np.random.default_rng(seed + 1)
    selected_paths = rng.choice(S_paths.shape[0], size=n_paths_plot, replace=False)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    for i in selected_paths:
        ax.plot(time_grid, S_paths[i], color="#00d4ff", linewidth=0.8, alpha=0.18)

    mean_path = np.mean(S_paths, axis=0)
    percentile_5 = np.percentile(S_paths, 5, axis=0)
    percentile_95 = np.percentile(S_paths, 95, axis=0)

    ax.plot(time_grid, mean_path, color="white", linewidth=2.5, label="Mean path")
    ax.plot(time_grid, percentile_5, color="#d0d0d0", linewidth=1.2, linestyle="--", alpha=0.65, label="5th / 95th percentiles")
    ax.plot(time_grid, percentile_95, color="#d0d0d0", linewidth=1.2, linestyle="--", alpha=0.65)

    ax.axhline(spot, color="#d0d0d0", linestyle=":", linewidth=1, alpha=0.5, label=f"Initial spot: {spot:.2f}")
    if option_requires_pricing_strike(option_type):
        ax.axhline(strike, color="#ff4d4d", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Strike: {strike:.2f}")

    if barrier is not None and np.isfinite(barrier):
        ax.axhline(barrier, color="#ffcc00", linestyle="--", linewidth=1.5, alpha=0.85, label=f"Barrier: {barrier:.2f}")

    if lower is not None and np.isfinite(lower):
        ax.axhline(lower, color="#66ff99", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Lower: {lower:.2f}")

    if upper is not None and np.isfinite(upper):
        ax.axhline(upper, color="#cc99ff", linestyle="--", linewidth=1.2, alpha=0.85, label=f"Upper: {upper:.2f}")

    ax.set_title(f"Monte Carlo Simulated Paths\n{option_type} | Maturity {maturity.date()}", fontsize=16, fontweight="bold", color="white", pad=16)
    ax.set_xlabel("Simulated trading days", fontsize=12, color="#d0d0d0")
    ax.set_ylabel("Underlying price", fontsize=12, color="#d0d0d0")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25, color="white")
    ax.tick_params(axis="x", colors="#d0d0d0")
    ax.tick_params(axis="y", colors="#d0d0d0")
    ax.legend(facecolor="#0e1117", edgecolor="#555555", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")

    plt.tight_layout()
    return fig

def get_extra_inputs(option_type, spot, strike, maturity):

    barrier = None
    cash_payout = 1.0
    extra_strike = None
    lower = None
    upper = None

    if option_type in BARRIER_TYPES:
        barrier = st.number_input("Barrier", min_value=0.0001, value=float(spot * 1.1 if "Up" in option_type else spot * 0.9), step=1.0)

    if option_type in DIGITAL_TYPES:
        cash_payout = st.number_input("Cash payout", min_value=0.0001, value=1.0, step=0.5)

    if option_type in EXTRA_STRIKE_TYPES:
        label = "Payoff strike / cap / floor / alpha"
        default_extra = strike * 1.1 if option_type in ["Capped Calls", "Gap Calls"] else strike * 0.9
        if option_type in ["Forward Start Calls", "Forward Start Puts"]:
            default_extra = 1.0
            label = "Alpha: future strike = alpha × S_start"
        extra_strike = st.number_input(label, min_value=0.0001, value=float(default_extra), step=0.5)

    return barrier, cash_payout, extra_strike, lower, upper

def get_manual_volatility_input(option_type, spot, strike, barrier, extra_strike, base_iv):

    volatility_model = st.radio(
        "Volatility model",
        ["Manual constant volatility", "Manual smile-adjusted proxy"],
        horizontal=True,
        help="Manual constant volatility uses the base IV directly. Manual smile-adjusted proxy combines manually entered IVs at the relevant levels for the selected exotic option.",
    )

    implied_volatility = float(base_iv)
    volatility_formula = "Manual input"
    volatility_diagnostics = pd.DataFrame()

    if volatility_model != "Manual smile-adjusted proxy":
        return implied_volatility, volatility_model, volatility_formula, volatility_diagnostics

    st.caption("Enter manual IVs for the key levels used by the smile-adjusted proxy.")

    vol_cols = st.columns(3)
    strike_iv = vol_cols[0].number_input("Strike IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
    atm_iv = vol_cols[1].number_input("ATM / spot IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")

    diagnostics = [
        {"Reference point": "Pricing strike", "Level": float(strike), "Vanilla IV": float(strike_iv), "Method": "Manual input"},
        {"Reference point": "ATM / spot", "Level": float(spot), "Vanilla IV": float(atm_iv), "Method": "Manual input"},
    ]

    if option_type in BARRIER_TYPES:
        default_barrier_iv = float(base_iv)
        barrier_iv = vol_cols[2].number_input("Barrier IV", min_value=0.0001, value=default_barrier_iv, step=0.01, format="%.4f")
        diagnostics.append({"Reference point": "Barrier", "Level": float(barrier), "Vanilla IV": float(barrier_iv), "Method": "Manual input"})
        implied_volatility = weighted_average_available([(0.50, strike_iv), (0.30, barrier_iv), (0.20, atm_iv)])
        volatility_formula = "50% strike IV + 30% barrier IV + 20% ATM IV"

    elif option_type in ["Gap Calls", "Gap Puts"]:
        payoff_iv = vol_cols[2].number_input("Payoff strike IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
        diagnostics.append({"Reference point": "Payoff strike", "Level": float(extra_strike), "Vanilla IV": float(payoff_iv), "Method": "Manual input"})
        implied_volatility = weighted_average_available([(0.50, strike_iv), (0.50, payoff_iv)])
        volatility_formula = "50% trigger strike IV + 50% payoff strike IV"

    elif option_type in ["Capped Calls", "Capped Puts"]:
        cap_floor_iv = vol_cols[2].number_input("Cap / floor IV", min_value=0.0001, value=float(base_iv), step=0.01, format="%.4f")
        diagnostics.append({"Reference point": "Cap / floor", "Level": float(extra_strike), "Vanilla IV": float(cap_floor_iv), "Method": "Manual input"})
        implied_volatility = weighted_average_available([(0.50, strike_iv), (0.50, cap_floor_iv)])
        volatility_formula = "50% strike IV + 50% cap/floor IV"

    elif option_type in DIGITAL_TYPES:
        implied_volatility = float(strike_iv)
        volatility_formula = "Manual strike IV for digital payoff"

    else:
        implied_volatility = float(strike_iv)
        volatility_formula = "Manual strike IV"

    if not np.isfinite(implied_volatility) or implied_volatility <= 0:
        implied_volatility = float(base_iv)
        volatility_formula = "Fallback to base manual IV"

    volatility_diagnostics = pd.DataFrame(diagnostics)

    if not volatility_diagnostics.empty:
        with st.expander("Manual volatility diagnostics"):
            display_df = volatility_diagnostics.copy()
            display_df["Level"] = display_df["Level"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A")
            display_df["Vanilla IV"] = display_df["Vanilla IV"].map(format_percent)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    return float(implied_volatility), volatility_model, volatility_formula, volatility_diagnostics


def get_manual_parameters():

    option_type = st.selectbox("Option type", EXOTIC_OPTION_TYPES, index=0)

    input_cols = st.columns(3)
    spot = input_cols[0].number_input("Spot price", min_value=0.0001, value=100.0, step=1.0)

    strike = input_cols[1].number_input("Strike", min_value=0.0001, value=100.0, step=1.0)

    maturity_date = input_cols[2].date_input("Maturity", value=datetime.today().date() + timedelta(days=90), min_value=datetime.today().date() + timedelta(days=1))

    input_cols = st.columns(3)
    base_iv = input_cols[0].number_input("Base volatility / IV", min_value=0.0001, value=0.20, step=0.01, format="%.4f")
    rf = input_cols[1].number_input("Risk-free rate", value=0.04, step=0.005, format="%.4f")
    dividend_yield = input_cols[2].number_input("Dividend yield", min_value=0.0, value=0.00, step=0.005, format="%.4f")

    maturity = datetime.combine(maturity_date, datetime.min.time())
    barrier, cash_payout, extra_strike, lower, upper = get_extra_inputs(option_type, spot, strike, maturity)
    IV, volatility_mode, volatility_formula, volatility_diagnostics = get_manual_volatility_input(option_type, spot, strike, barrier, extra_strike, base_iv)

    currency = ""
    market_mid = np.nan
    annual_dividend = np.nan
    dividend_method = "Manual input"

    return {"source": "Manual inputs","symbol": "Manual input","option_type": option_type,"spot": float(spot),"strike": float(strike),"vol_reference_strike": np.nan,"maturity": maturity,"IV": float(IV),"rf": float(rf),"dividend_yield": float(dividend_yield),"currency": currency,"market_mid": market_mid,"annual_dividend": annual_dividend,"dividend_method": dividend_method,"volatility_mode": volatility_mode,"volatility_formula": volatility_formula,"volatility_diagnostics": volatility_diagnostics,"barrier": barrier,"cash_payout": cash_payout,"extra_strike": extra_strike,"lower": lower,"upper": upper}

def infer_quote_option_type(option_type):

    if "Puts" in option_type:
        return "Puts"

    return "Calls"

def get_quote_implied_volatility(ticker, quote_row, spot, rf, dividend_yield):

    # Deliberately do not use Yahoo Finance's `impliedVolatility` field.
    # Every vanilla IV used by the app is solved from the bid/ask mid via Newton-Raphson.
    implied_volatility = newton_raphson(ticker, quote_row, 0.2, spot, rf, dividend_yield=dividend_yield)

    if np.isfinite(implied_volatility) and implied_volatility > 0:
        return float(implied_volatility), "Newton-Raphson from vanilla mid"

    return np.nan, "Unavailable from Newton-Raphson"

def build_iv_points_for_maturity(data, maturity_string, ticker, spot, rf, dividend_yield):

    maturity_data = data[data["maturity"] == maturity_string].copy()
    points = []

    for _, row in maturity_data.iterrows():
        strike = safe_float(row.get("strike", np.nan))
        if not np.isfinite(strike) or strike <= 0:
            continue

        implied_volatility, source = get_quote_implied_volatility(ticker, row, spot, rf, dividend_yield)
        if np.isfinite(implied_volatility) and implied_volatility > 0:
            points.append({"strike": float(strike), "iv": float(implied_volatility), "source": source})

    if len(points) == 0:
        return pd.DataFrame(columns=["strike", "iv", "source"])

    iv_points = pd.DataFrame(points)
    iv_points = iv_points.dropna(subset=["strike", "iv"])
    iv_points = iv_points[(iv_points["strike"] > 0) & (iv_points["iv"] > 0)]
    iv_points = iv_points.sort_values("strike").drop_duplicates(subset=["strike"], keep="first")

    return iv_points

def interpolate_iv_at_strike(iv_points, target_strike):

    target_strike = safe_float(target_strike)
    if not np.isfinite(target_strike) or target_strike <= 0 or iv_points.empty:
        return np.nan, "Unavailable"

    strikes = iv_points["strike"].to_numpy(dtype=float)
    vols = iv_points["iv"].to_numpy(dtype=float)

    if len(strikes) == 1:
        return float(vols[0]), "Only one usable vanilla IV available"

    exact_match_index = np.where(np.isclose(strikes, target_strike, rtol=0.0, atol=1e-10))[0]
    if len(exact_match_index) > 0:
        return float(vols[exact_match_index[0]]), "Exact listed strike"

    if target_strike < strikes[0]:
        return float(vols[0]), "Nearest listed strike below available range"

    if target_strike > strikes[-1]:
        return float(vols[-1]), "Nearest listed strike above available range"

    interpolated_iv = np.interp(target_strike, strikes, vols)
    return float(interpolated_iv), "Linear interpolation between listed strikes"

def weighted_average_available(weighted_vols):

    clean_items = [(weight, vol) for weight, vol in weighted_vols if np.isfinite(vol) and weight > 0]
    total_weight = sum(weight for weight, _ in clean_items)

    if total_weight <= 0:
        return np.nan

    return sum(weight * vol for weight, vol in clean_items) / total_weight

def calculate_smile_adjusted_volatility(option_type, strike, spot, barrier, extra_strike, iv_points, base_iv):

    diagnostics = []

    def get_point(label, level):
        iv, source = interpolate_iv_at_strike(iv_points, level)
        diagnostics.append({
            "Reference point": label,
            "Level": level,
            "Vanilla IV": iv,
            "Method": source,
        })
        return iv

    strike_iv = get_point("Pricing strike", strike)
    atm_iv = get_point("ATM / spot", spot)

    if not np.isfinite(strike_iv):
        strike_iv = base_iv

    if not np.isfinite(atm_iv):
        atm_iv = base_iv

    effective_iv = strike_iv
    formula_label = "Vanilla IV at pricing strike"

    if option_type in BARRIER_TYPES:
        barrier_iv = get_point("Barrier", barrier)
        effective_iv = weighted_average_available([
            (0.50, strike_iv),
            (0.30, barrier_iv),
            (0.20, atm_iv),
        ])
        formula_label = "50% strike IV + 30% barrier IV + 20% ATM IV"

    elif option_type in ["Gap Calls", "Gap Puts"]:
        payoff_iv = get_point("Payoff strike", extra_strike)
        effective_iv = weighted_average_available([
            (0.50, strike_iv),
            (0.50, payoff_iv),
        ])
        formula_label = "50% trigger strike IV + 50% payoff strike IV"

    elif option_type in ["Capped Calls", "Capped Puts"]:
        cap_floor_iv = get_point("Cap / floor", extra_strike)
        effective_iv = weighted_average_available([
            (0.50, strike_iv),
            (0.50, cap_floor_iv),
        ])
        formula_label = "50% strike IV + 50% cap/floor IV"

    elif option_type in DIGITAL_TYPES:
        effective_iv = strike_iv
        formula_label = "Vanilla IV at digital strike"

    if not np.isfinite(effective_iv):
        effective_iv = base_iv
        formula_label = "Fallback to selected vanilla IV"

    diagnostics_df = pd.DataFrame(diagnostics)
    if not diagnostics_df.empty:
        diagnostics_df["Level"] = diagnostics_df["Level"].astype(float)
        diagnostics_df["Vanilla IV"] = diagnostics_df["Vanilla IV"].astype(float)

    return float(effective_iv), formula_label, diagnostics_df

FIXED_MONTE_CARLO_SEED = 42
FIXED_STEPS_PER_YEAR = 252
MONTE_CARLO_BATCH_SIZE = 50_000

SPOT_BUMP_RELATIVE = 0.01
VOL_BUMP_ABSOLUTE = 0.02
VOL_BUMP_RELATIVE = 0.10
RATE_BUMP_ABSOLUTE = 0.001

def get_yahoo_parameters():

    with st.sidebar:
        st.subheader("Underlying")
        ticker_source = st.radio("Ticker source", ["Liquid ticker universe", "Custom ticker"], horizontal=False)

        if ticker_source == "Liquid ticker universe":
            default_index = OPTIONABLE_TICKERS.index("AAPL") if "AAPL" in OPTIONABLE_TICKERS else 0
            symbol = st.selectbox("Ticker", OPTIONABLE_TICKERS, index=default_index)
        else:
            symbol = st.text_input("Yahoo Finance ticker", value="AAPL")

        symbol = symbol.strip().upper()

    if not symbol:
        st.info("Enter a ticker to begin.")
        return None

    option_type = st.selectbox("Option type", EXOTIC_OPTION_TYPES, index=0)

    volatility_mode = st.radio("Volatility input",["Single vanilla IV", "Smile-adjusted proxy"],horizontal=True,help="Single vanilla IV uses one listed option. Smile-adjusted proxy combines several vanilla IVs from the selected maturity. It is not a full local volatility model.",)

    quote_option_type = infer_quote_option_type(option_type)
    ticker = yf.Ticker(symbol)

    with st.spinner("Loading Yahoo Finance market data..."):
        data = load_option_data(symbol, quote_option_type)

    if data.empty:
        st.warning(f"No usable option quotes are available for {symbol} on Yahoo Finance after bid/ask filtering.")
        return None

    spot = load_spot_price(symbol)

    if not np.isfinite(spot):
        st.warning(f"Unable to retrieve a valid equity spot price for {symbol}.")
        return None

    currency = data["currency"].dropna().iloc[0] if "currency" in data.columns and not data["currency"].dropna().empty else ""
    available_maturities = sorted(data["maturity"].dropna().unique())

    snapshot_cols = st.columns(3)
    snapshot_cols[0].metric("Equity spot", format_price(spot, currency))
    snapshot_cols[1].metric("Ticker", symbol)
    snapshot_cols[2].metric("Currency", currency if currency else "N/A")

    st.subheader("Option setup")
    selection_cols = st.columns(2)
    maturity_string = selection_cols[0].selectbox("Maturity", available_maturities)
    strikes_for_maturity = sorted(data.loc[data["maturity"] == maturity_string, "strike"].dropna().unique())

    reference_strike_label = get_strike_input_label(option_type)
    reference_strike = selection_cols[1].selectbox(reference_strike_label, strikes_for_maturity, format_func=format_strike)
    strike = get_effective_pricing_strike(option_type, float(spot), float(reference_strike))

    selected_rows = data[(data["maturity"] == maturity_string) & (data["strike"] == reference_strike)].copy()

    if selected_rows.empty:
        st.warning("No quote was found for this reference strike and maturity selection.")
        return None

    selected_quote = selected_rows.iloc[0]
    maturity = datetime.strptime(maturity_string, "%Y-%m-%d")

    with st.spinner("Preparing rates and dividends..."):
        rf = risk_free_rate(data)
        dividend_yield, annual_dividend, dividend_method = get_dividend_yield_from_yahoo(symbol, spot)

    if not np.isfinite(rf):
        rf = 0.04

    barrier, cash_payout, extra_strike, lower, upper = get_extra_inputs(option_type, spot, strike, maturity)

    with st.spinner("Building volatility input..."):
        base_iv, base_iv_source = get_quote_implied_volatility(ticker, selected_quote, spot, rf, dividend_yield)

        if not np.isfinite(base_iv):
            base_iv = 0.20
            base_iv_source = "Fallback default"

        volatility_formula = base_iv_source
        volatility_diagnostics = pd.DataFrame()
        implied_volatility = base_iv

        if volatility_mode == "Smile-adjusted proxy":
            iv_points = build_iv_points_for_maturity(data, maturity_string, ticker, spot, rf, dividend_yield)
            implied_volatility, volatility_formula, volatility_diagnostics = calculate_smile_adjusted_volatility(option_type,strike,spot,barrier,extra_strike,iv_points,base_iv)

    input_cols = st.columns(3)
    input_cols[0].metric("Volatility input", format_percent(implied_volatility))
    input_cols[1].metric("Risk-free rate", format_percent(rf))
    input_cols[2].metric("Dividend yield", format_percent(dividend_yield))

    st.caption(f"Volatility method: {volatility_mode}. {volatility_formula}.")

    if volatility_mode == "Smile-adjusted proxy" and not volatility_diagnostics.empty:
        with st.expander("Smile-adjusted volatility diagnostics"):
            display_df = volatility_diagnostics.copy()
            display_df["Level"] = display_df["Level"].map(lambda x: f"{x:.4f}" if np.isfinite(x) else "N/A")
            display_df["Vanilla IV"] = display_df["Vanilla IV"].map(format_percent)
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    return {"source": "Yahoo Finance","symbol": symbol,"option_type": option_type,"spot": float(spot),"strike": float(strike),"vol_reference_strike": float(reference_strike),"maturity": maturity,"IV": float(implied_volatility),"rf": float(rf),"dividend_yield": float(dividend_yield),"currency": currency,"market_mid": np.nan,"annual_dividend": annual_dividend,"dividend_method": dividend_method,"volatility_mode": volatility_mode,"volatility_formula": volatility_formula,"volatility_diagnostics": volatility_diagnostics,"barrier": barrier,"cash_payout": cash_payout,"extra_strike": extra_strike,"lower": lower,"upper": upper}


def calculate_black_scholes(parameters):

    return pricer_bs(parameters["strike"],parameters["maturity"],parameters["IV"],parameters["spot"],parameters["rf"],parameters["option_type"],barrier=parameters["barrier"],dividend_yield=parameters["dividend_yield"],cash_payout=parameters["cash_payout"],extra_strike=parameters["extra_strike"],lower=parameters["lower"],upper=parameters["upper"])


def _monte_carlo_price_batched(parameters, num_simulations, seed=FIXED_MONTE_CARLO_SEED, steps_per_year=FIXED_STEPS_PER_YEAR, batch_size=MONTE_CARLO_BATCH_SIZE):

    total_simulations = int(num_simulations)
    if total_simulations <= 0:
        return np.nan

    batch_size = int(min(batch_size, total_simulations))
    remaining_simulations = total_simulations
    weighted_price_sum = 0.0
    completed_simulations = 0
    batch_index = 0

    while remaining_simulations > 0:
        current_batch_size = min(batch_size, remaining_simulations)

        batch_price = pricer_mc(parameters["strike"],parameters["maturity"],parameters["IV"],parameters["spot"],parameters["rf"],parameters["option_type"],barrier=parameters["barrier"],dividend_yield=parameters["dividend_yield"],cash_payout=parameters["cash_payout"],extra_strike=parameters["extra_strike"],lower=parameters["lower"],upper=parameters["upper"],price_only=True,plot=False,num_simulations=current_batch_size,seed=seed + batch_index,steps_per_year=steps_per_year)

        if not np.isfinite(batch_price):
            return np.nan

        weighted_price_sum += batch_price * current_batch_size
        completed_simulations += current_batch_size
        remaining_simulations -= current_batch_size
        batch_index += 1

    return weighted_price_sum / completed_simulations if completed_simulations > 0 else np.nan

def _copy_parameters_with_updates(parameters, **updates):

    updated_parameters = dict(parameters)
    updated_parameters.update(updates)
    return updated_parameters

def calculate_monte_carlo(parameters, num_simulations, seed=FIXED_MONTE_CARLO_SEED, steps_per_year=FIXED_STEPS_PER_YEAR):

    price = _monte_carlo_price_batched(parameters, num_simulations, seed=seed, steps_per_year=steps_per_year)

    if not np.isfinite(price):
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    greek_simulations = int(num_simulations)
    greek_base = price

    hS = max(parameters["spot"] * SPOT_BUMP_RELATIVE, 1e-4)
    hV = max(min(max(parameters["IV"] * VOL_BUMP_RELATIVE, VOL_BUMP_ABSOLUTE), parameters["IV"] * 0.50), 1e-4)
    hR = RATE_BUMP_ABSOLUTE

    p_S_up = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, spot=parameters["spot"] + hS), greek_simulations, seed=seed, steps_per_year=steps_per_year)
    p_S_down = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, spot=max(parameters["spot"] - hS, 1e-8)), greek_simulations, seed=seed, steps_per_year=steps_per_year)

    delta = (p_S_up - p_S_down) / (2 * hS) if np.isfinite(p_S_up) and np.isfinite(p_S_down) else np.nan
    gamma = (p_S_up - 2 * greek_base + p_S_down) / (hS**2) if np.isfinite(p_S_up) and np.isfinite(p_S_down) and np.isfinite(greek_base) else np.nan

    p_v_up = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, IV=parameters["IV"] + hV), greek_simulations, seed=seed, steps_per_year=steps_per_year)
    p_v_down = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, IV=max(parameters["IV"] - hV, 1e-8)), greek_simulations, seed=seed, steps_per_year=steps_per_year)

    vega = (p_v_up - p_v_down) / (2 * hV) if np.isfinite(p_v_up) and np.isfinite(p_v_down) else np.nan
    volga = (p_v_up - 2 * greek_base + p_v_down) / (hV**2) if np.isfinite(p_v_up) and np.isfinite(p_v_down) and np.isfinite(greek_base) else np.nan

    p_r_up = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, rf=parameters["rf"] + hR), greek_simulations, seed=seed, steps_per_year=steps_per_year)
    p_r_down = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, rf=parameters["rf"] - hR), greek_simulations, seed=seed, steps_per_year=steps_per_year)

    rho = (p_r_up - p_r_down) / (2 * hR) if np.isfinite(p_r_up) and np.isfinite(p_r_down) else np.nan

    theta = np.nan
    maturity = parameters["maturity"]
    if isinstance(maturity, datetime) and (maturity - datetime.today()).days > 1:
        tomorrow_parameters = _copy_parameters_with_updates(parameters, maturity=maturity - timedelta(days=1))
        p_tomorrow = _monte_carlo_price_batched(tomorrow_parameters, greek_simulations, seed=seed, steps_per_year=steps_per_year)
        theta = p_tomorrow - greek_base if np.isfinite(p_tomorrow) and np.isfinite(greek_base) else np.nan

    p_up_up = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, spot=parameters["spot"] + hS, IV=parameters["IV"] + hV), greek_simulations, seed=seed, steps_per_year=steps_per_year)
    p_up_down = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, spot=parameters["spot"] + hS, IV=max(parameters["IV"] - hV, 1e-8)), greek_simulations, seed=seed, steps_per_year=steps_per_year)
    p_down_up = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, spot=max(parameters["spot"] - hS, 1e-8), IV=parameters["IV"] + hV), greek_simulations, seed=seed, steps_per_year=steps_per_year)
    p_down_down = _monte_carlo_price_batched(_copy_parameters_with_updates(parameters, spot=max(parameters["spot"] - hS, 1e-8), IV=max(parameters["IV"] - hV, 1e-8)), greek_simulations, seed=seed, steps_per_year=steps_per_year)

    vanna = np.nan
    if all(np.isfinite(value) for value in [p_up_up, p_up_down, p_down_up, p_down_down]):
        vanna = (p_up_up - p_up_down - p_down_up + p_down_down) / (4 * hS * hV)

    return (price, delta, gamma, vega, theta, rho, volga, vanna)

def display_pricing_inputs(parameters):

    input_df = pd.DataFrame({
        "Input": ["Source", "Symbol", "Option type", "Spot", "Strike", "Maturity", "Volatility input", "Volatility method", "Risk-free rate", "Dividend yield"],
        "Value": [parameters["source"],parameters["symbol"],parameters["option_type"],f"{parameters['spot']:.4f}",f"{parameters['strike']:.4f}",parameters["maturity"].strftime("%Y-%m-%d"),format_percent(parameters["IV"]),parameters.get("volatility_mode", "Single volatility"),format_percent(parameters["rf"]),format_percent(parameters["dividend_yield"])]})

    st.dataframe(input_df, use_container_width=True, hide_index=True)

def build_price_summary_dataframe(results):

    rows = []

    for method_name, result in results.items():
        row = {"Model": method_name,"Price": result[0],"Delta": result[1],"Gamma": result[2],"Vega": result[3],"Theta / day": result[4],"Rho": result[5],"Volga": result[6],"Vanna": result[7]}
        rows.append(row)

    return pd.DataFrame(rows)

def create_greeks_result_figure(results):

    plt.style.use("dark_background")
    greek_labels = [("Delta", 1),("Gamma", 2),("Vega", 3),("Theta / day", 4),("Rho", 5),("Volga", 6),("Vanna", 7)]

    n_cols = 4
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7.5), facecolor="#0e1117")
    axes = axes.flatten()

    method_names = list(results.keys())

    for ax, (greek_name, value_index) in zip(axes, greek_labels):
        ax.set_facecolor("#0e1117")

        values = [results[method][value_index] for method in method_names]
        x = np.arange(len(method_names))
        ax.bar(x, values, alpha=0.85)

        ax.axhline(0, color="white", linewidth=0.8, alpha=0.45)
        ax.set_title(greek_name, fontsize=13, fontweight="bold", color="white", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(method_names, rotation=0, color="#d0d0d0", fontsize=9)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.55, alpha=0.25, color="white")
        ax.tick_params(axis="y", colors="#d0d0d0", labelsize=9)

        finite_values = [value for value in values if np.isfinite(value)]
        if finite_values:
            min_value = min(finite_values)
            max_value = max(finite_values)
            if min_value == max_value:
                padding = max(abs(max_value) * 0.25, 1e-4)
                ax.set_ylim(min_value - padding, max_value + padding)
            else:
                value_range = max_value - min_value
                padding = max(value_range * 0.25, 1e-4)
                ax.set_ylim(min_value - padding, max_value + padding)

        for index, value in enumerate(values):
            if np.isfinite(value):
                ax.annotate(f"{value:.4g}",xy=(index, value),xytext=(0, 6 if value >= 0 else -12),textcoords="offset points",ha="center",va="bottom" if value >= 0 else "top",color="white",fontsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")

    for ax in axes[len(greek_labels):]:
        ax.axis("off")
        ax.set_facecolor("#0e1117")

    fig.suptitle("Risk Sensitivities by Greek", fontsize=17, fontweight="bold", color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

def display_results_dashboard(parameters, results, num_simulations, selected_methods):

    st.subheader("Pricing Results")

    price_cols = st.columns(3 if len(results) == 2 else 2)
    for col, (method_name, result) in zip(price_cols, results.items()):
        col.metric(f"{method_name} Price", format_price(result[0], parameters["currency"]))

    if len(results) == 2:
        bs_price = results["Black-Scholes"][0]
        mc_price = results["Monte Carlo"][0]
        difference = mc_price - bs_price if np.isfinite(bs_price) and np.isfinite(mc_price) else np.nan
        price_cols[2].metric("Monte Carlo − Black-Scholes", f"{difference:.6f}" if np.isfinite(difference) else "N/A")

    summary_df = build_price_summary_dataframe(results)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    if len(results) == 2:
        st.subheader("Risk Sensitivities")
        st.pyplot(create_greeks_result_figure(results), use_container_width=True)

    st.subheader("Payoff and Path Visualisation")

    first_valid_price = next((result[0] for result in results.values() if np.isfinite(result[0])), np.nan)
    payoff_fig = create_payoff_figure(parameters["spot"],parameters["strike"],parameters["option_type"],price=first_valid_price,lower=parameters["lower"],upper=parameters["upper"],barrier=parameters["barrier"],extra_strike=parameters["extra_strike"],cash_payout=parameters["cash_payout"])

    if "Monte Carlo" in selected_methods:
        visual_cols = st.columns(2)
        visual_cols[0].pyplot(payoff_fig, use_container_width=True)

        mc_paths_fig = create_mc_paths_figure(parameters["spot"],parameters["strike"],parameters["maturity"],parameters["IV"],parameters["rf"],parameters["dividend_yield"],parameters["option_type"],barrier=parameters["barrier"],lower=parameters["lower"],upper=parameters["upper"],num_simulations=min(2_000, int(num_simulations)),seed=FIXED_MONTE_CARLO_SEED,steps_per_year=FIXED_STEPS_PER_YEAR)
        if mc_paths_fig is not None:
            visual_cols[1].pyplot(mc_paths_fig, use_container_width=True)
        else:
            visual_cols[1].warning("Monte Carlo path visualisation is not available for the current setup.")

    else:
        st.pyplot(payoff_fig, use_container_width=True)


def streamlit_app():

    st.set_page_config(page_title="Option Pricer", layout="wide")

    st.title("Option Pricer")

    with st.sidebar:
        st.header("Settings")
        data_source = st.radio("Input source", ["Yahoo Finance", "Manual inputs"], horizontal=False)

        pricing_choice = st.radio("Pricing method",["Black-Scholes", "Monte Carlo", "Both"],index=2,horizontal=False)

        if pricing_choice == "Black-Scholes":
            selected_methods = ["Black-Scholes"]
        elif pricing_choice == "Monte Carlo":
            selected_methods = ["Monte Carlo"]
        else:
            selected_methods = ["Black-Scholes", "Monte Carlo"]

        if "Monte Carlo" in selected_methods:
            st.divider()
            num_simulations = st.slider("Monte Carlo simulations",min_value=100_000,max_value=1_000_000,value=500_000,step=50_000,format="%d")
        else:
            num_simulations = 500_000

    if data_source == "Yahoo Finance":
        parameters = get_yahoo_parameters()
    else:
        parameters = get_manual_parameters()

    if parameters is None:
        return

    if not all(np.isfinite(parameters[key]) for key in ["spot", "strike", "IV", "rf", "dividend_yield"]):
        st.warning("At least one required pricing input is not valid.")
        return

    st.divider()
    run_pricing = st.button("Run pricing", type="primary", use_container_width=True)

    if not run_pricing:
        return

    results = {}

    if "Black-Scholes" in selected_methods:
        with st.spinner("Running Black-Scholes pricing..."):
            results["Black-Scholes"] = calculate_black_scholes(parameters)

    if "Monte Carlo" in selected_methods:
        with st.spinner("Running Monte Carlo simulation..."):
            results["Monte Carlo"] = calculate_monte_carlo(parameters, int(num_simulations))

    display_results_dashboard(parameters, results, num_simulations, selected_methods)

if __name__ == "__main__":
    streamlit_app()
