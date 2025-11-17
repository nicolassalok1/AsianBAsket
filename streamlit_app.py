import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import norm
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False)
def get_option_expiries(ticker: str):
    tk = yf.Ticker(ticker)
    return tk.options or []


@st.cache_data(show_spinner=False)
def get_option_surface_from_yf(ticker: str, expiry: str):
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)

    frames = []
    for frame in [chain.calls, chain.puts]:
        tmp = frame[["strike", "impliedVolatility"]].rename(
            columns={"strike": "K", "impliedVolatility": "iv"}
        )
        # La maturité T sera imposée plus tard (via T commun) dans ui_basket_surface;
        # on met ici une valeur neutre par défaut.
        tmp["T"] = 0.0
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["K", "iv"])
    return df


@st.cache_data(show_spinner=False)
def get_spot_and_hist_vol(ticker: str, period: str = "6mo", interval: str = "1d"):
    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError("Aucune donnée téléchargée.")
    close = data["Close"]
    spot = float(close.iloc[-1])
    log_returns = np.log(close / close.shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(252))
    hist_df = data.reset_index()
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    return spot, sigma, hist_df


def build_grid(
    df: pd.DataFrame,
    spot: float,
    n_k: int = 200,
    n_t: int = 200,
    k_span: float = 100.0,
    t_min: float = 0.0,
    t_max: float = 2.0,
):
    k_min = spot - k_span
    k_max = spot + k_span

    k_vals = np.linspace(k_min, k_max, n_k)
    t_vals = np.linspace(t_min, t_max, n_t)

    df = df.copy()
    df = df[(df["K"] >= k_min) & (df["K"] <= k_max)]
    df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

    if df.empty:
        raise ValueError("Aucun point dans le domaine de la grille après filtrage.")

    df["K_idx"] = np.searchsorted(k_vals, df["K"], side="left")
    df["T_idx"] = np.searchsorted(t_vals, df["T"], side="left")

    df["K_idx"] = df["K_idx"].clip(0, n_k - 1)
    df["T_idx"] = df["T_idx"].clip(0, n_t - 1)

    grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

    iv_grid = np.full((n_t, n_k), np.nan, dtype=float)

    for _, row in grouped.iterrows():
        ti = int(row["T_idx"])
        ki = int(row["K_idx"])
        iv_grid[ti, ki] = row["iv"]

    k_grid, t_grid = np.meshgrid(k_vals, t_vals)
    return k_grid, t_grid, iv_grid


def make_iv_surface_figure(k_grid, t_grid, iv_grid, title_suffix=""):
    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(1, 2, 1, projection="3d")

    iv_flat = iv_grid[~np.isnan(iv_grid)]
    if iv_flat.size == 0:
        raise ValueError("La grille iv_grid ne contient aucune valeur non-NaN.")
    iv_mean = iv_flat.mean()
    iv_grid_filled = np.where(np.isnan(iv_grid), iv_mean, iv_grid)

    surf = ax3d.plot_surface(
        k_grid,
        t_grid,
        iv_grid_filled,
        rstride=1,
        cstride=1,
        linewidth=0.2,
        antialiased=True,
        cmap="viridis",
    )

    ax3d.set_xlabel("Strike K")
    ax3d.set_ylabel("Maturité T (années)")
    ax3d.set_zlabel("Implied vol")
    ax3d.set_title(f"Surface 3D de volatilité implicite{title_suffix}")

    fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax3d, label="iv")

    ax2d = fig.add_subplot(1, 2, 2)
    im = ax2d.imshow(
        iv_grid_filled,
        extent=[k_grid.min(), k_grid.max(), t_grid.min(), t_grid.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    ax2d.set_xlabel("Strike K")
    ax2d.set_ylabel("Maturité T (années)")
    ax2d.set_title(f"Heatmap IV{title_suffix}")
    fig.colorbar(im, ax=ax2d, label="iv")

    plt.tight_layout()
    return fig


def btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps):
    delta_t = maturity / steps
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1.0 / up
    prob = (np.exp(rate * delta_t) - down) / (up - down)

    spot_paths = [spot]
    avg_paths = [spot]
    strike_paths = [strike]

    for _ in range(steps):
        spot_paths = [s * up for s in spot_paths] + [s * down for s in spot_paths]
        avg_paths = avg_paths + avg_paths
        strike_paths = strike_paths + strike_paths
        for index in range(len(avg_paths)):
            avg_paths[index] = avg_paths[index] + spot_paths[index]

    avg_paths = np.array(avg_paths) / (steps + 1)
    spot_paths = np.array(spot_paths)
    strike_paths = np.array(strike_paths)

    if strike_type == "fixed":
        if option_type == "C":
            payoff = np.maximum(avg_paths - strike_paths, 0.0)
        else:
            payoff = np.maximum(strike_paths - avg_paths, 0.0)
    else:
        if option_type == "C":
            payoff = np.maximum(spot_paths - avg_paths, 0.0)
        else:
            payoff = np.maximum(avg_paths - spot_paths, 0.0)

    option_price = payoff.copy()
    for _ in range(steps):
        length = len(option_price) // 2
        option_price = prob * option_price[:length] + (1 - prob) * option_price[length:]

    return float(option_price[0])


def hw_btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps, m_points):
    n_steps = steps
    delta_t = maturity / n_steps
    up = np.exp(sigma * np.sqrt(delta_t))
    down = 1.0 / up
    prob = (np.exp(rate * delta_t) - down) / (up - down)

    avg_grid = []
    strike_vec = np.array([strike] * m_points)

    for j_index in range(n_steps + 1):
        path_up_then_down = np.array(
            [spot * up**j * down**0 for j in range(n_steps - j_index)]
            + [spot * up**(n_steps - j_index) * down**j for j in range(j_index + 1)]
        )
        avg_max = path_up_then_down.mean()

        path_down_then_up = np.array(
            [spot * down**j * up**0 for j in range(j_index + 1)]
            + [spot * down**j_index * up**(j + 1) for j in range(n_steps - j_index)]
        )
        avg_min = path_down_then_up.mean()

        diff = avg_max - avg_min
        avg_vals = [avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)]
        avg_grid.append(avg_vals)

    avg_grid = np.round(avg_grid, 4)

    payoff = []
    for j_index in range(n_steps + 1):
        avg_vals = np.array(avg_grid[j_index])
        spot_vals = np.array([spot * up**(n_steps - j_index) * down**j_index] * m_points)

        if strike_type == "fixed":
            if option_type == "C":
                pay = np.maximum(avg_vals - strike_vec, 0.0)
            else:
                pay = np.maximum(strike_vec - avg_vals, 0.0)
        else:
            if option_type == "C":
                pay = np.maximum(spot_vals - avg_vals, 0.0)
            else:
                pay = np.maximum(avg_vals - spot_vals, 0.0)

        payoff.append(pay)

    payoff = np.round(np.array(payoff), 4)

    for n_index in range(n_steps - 1, -1, -1):
        avg_backward = []
        payoff_backward = []

        for j_index in range(n_index + 1):
            path_up_then_down = np.array(
                [spot * up**j * down**0 for j in range(n_index - j_index)]
                + [spot * up**(n_index - j_index) * down**j for j in range(j_index + 1)]
            )
            avg_max = path_up_then_down.mean()

            path_down_then_up = np.array(
                [spot * down**j * up**0 for j in range(j_index + 1)]
                + [spot * down**j_index * up**(j + 1) for j in range(n_index - j_index)]
            )
            avg_min = path_down_then_up.mean()

            diff = avg_max - avg_min
            avg_vals = np.array(
                [avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)]
            )
            avg_backward.append(avg_vals)

        avg_backward = np.round(np.array(avg_backward), 4)

        payoff_new = []
        for j_index in range(n_index + 1):
            avg_vals = avg_backward[j_index]
            pay_vals = np.zeros_like(avg_vals)

            avg_up = np.array(avg_grid[j_index])
            avg_down = np.array(avg_grid[j_index + 1])
            pay_up = payoff[j_index]
            pay_down = payoff[j_index + 1]

            for k_index, avg_k in enumerate(avg_vals):
                if avg_k <= avg_up[0]:
                    fu = pay_up[0]
                elif avg_k >= avg_up[-1]:
                    fu = pay_up[-1]
                else:
                    idx = np.searchsorted(avg_up, avg_k) - 1
                    x0, x1 = avg_up[idx], avg_up[idx + 1]
                    y0, y1 = pay_up[idx], pay_up[idx + 1]
                    fu = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                if avg_k <= avg_down[0]:
                    fd = pay_down[0]
                elif avg_k >= avg_down[-1]:
                    fd = pay_down[-1]
                else:
                    idx = np.searchsorted(avg_down, avg_k) - 1
                    x0, x1 = avg_down[idx], avg_down[idx + 1]
                    y0, y1 = pay_down[idx], pay_down[idx + 1]
                    fd = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                pay_vals[k_index] = (prob * fu + (1 - prob) * fd) * np.exp(-rate * delta_t)

            payoff_backward.append(pay_vals)

        avg_grid = avg_backward
        payoff = np.round(np.array(payoff_backward), 4)

    option_price = payoff[0].mean()
    return float(option_price)


def bs_option_price(time, spot, strike, maturity, rate, sigma, option_kind):
    tau = maturity - time
    if tau <= 0:
        if option_kind == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    if option_kind == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
    else:
        price = strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return float(price)


def compute_asian_price(
    strike_type: str,
    option_type: str,
    model: str,
    spot: float,
    strike: float,
    rate: float,
    sigma: float,
    maturity: float,
    steps: int,
    m_points: int | None,
):
    if model == "BTM naïf":
        return btm_asian(
            strike_type=strike_type,
            option_type=option_type,
            spot=spot,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
            steps=int(steps),
        )
    m_points_val = int(m_points) if m_points is not None else 10
    return hw_btm_asian(
        strike_type=strike_type,
        option_type=option_type,
        spot=spot,
        strike=strike,
        rate=rate,
        sigma=sigma,
        maturity=maturity,
        steps=int(steps),
        m_points=m_points_val,
    )


def ui_basket_surface(ticker, period, interval, spot_common, maturity_common, rate_common, hist_df):
    st.header("Surface de volatilité implicite (module Basket)")
    st.markdown(
        "Les paramètres et actions sont disponibles dans la sidebar. "
        "Le fichier CSV requis doit contenir les colonnes `K`, `T` et `iv`."
    )

    if hist_df is None:
        hist_df = pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Spot commun S0 = {spot_common:.4f}")
        st.info(f"T commun = {maturity_common:.4f} années")
        st.info(f"Taux sans risque commun r = {rate_common:.4f}")
        k_span = st.number_input(
            "Étendue en strike autour de S0",
            value=100.0,
            min_value=1.0,
            key="basket_k_span",
        )
    with col2:
        grid_k = st.number_input(
            "Points de grille en K",
            value=100,
            min_value=20,
            max_value=400,
            step=10,
            key="basket_grid_k",
        )
        grid_t = st.number_input(
            "Points de grille en T",
            value=100,
            min_value=20,
            max_value=400,
            step=10,
            key="basket_grid_t",
        )

    option_surface = None
    expiries = get_option_expiries(ticker)
    if expiries:
        try:
            # Choix de l'échéance dont la date est la plus proche de T commun
            expiry_dates = [pd.to_datetime(e).date() for e in expiries]
            target_date = (
                pd.Timestamp.utcnow() + pd.to_timedelta(maturity_common * 365.0, unit="D")
            ).date()
            idx_min = int(np.argmin([abs((d - target_date).days) for d in expiry_dates]))
            expiry_chosen = expiries[idx_min]
            option_surface = get_option_surface_from_yf(ticker, expiry_chosen)
            # On force la colonne T à la maturité commune
            option_surface["T"] = float(maturity_common)
        except Exception as exc:
            st.error(f"Erreur lors de la récupération des options yfinance: {exc}")

    if option_surface is not None and not option_surface.empty:
        st.subheader("Surface IV (données options yfinance)")
        st.caption(
            f"Échantillon: {len(option_surface)} lignes, ticker {ticker}, T commun = {maturity_common:.4f} ans"
        )
        required_cols = {"K", "T", "iv"}
        if not required_cols.issubset(option_surface.columns):
            missing = required_cols - set(option_surface.columns)
            st.error(f"Colonnes manquantes dans la surface: {missing}")
            return
        try:
            k_grid, t_grid, iv_grid = build_grid(
                option_surface,
                spot=spot_common,
                n_k=int(grid_k),
                n_t=int(grid_t),
                k_span=k_span,
                t_min=0.0,
                t_max=maturity_common,
            )
            fig = make_iv_surface_figure(
                k_grid,
                t_grid,
                iv_grid,
                title_suffix=f" (S0={spot_common})",
            )
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Erreur lors de la construction de la surface: {exc}")


def _render_asian_heatmaps_for_model(
    model,
    s_vals,
    k_vals,
    sigma,
    maturity,
    steps,
    m_points,
    strike_common,
    rate_common,
):
    heatmaps = {}
    for opt_label, opt_code in [("Call", "C"), ("Put", "P")]:
        for stype in ["fixed", "floating"]:
            grid = np.zeros((len(s_vals), len(k_vals)))
            for i, s_ in enumerate(s_vals):
                for j, _ in enumerate(k_vals):
                    grid[i, j] = compute_asian_price(
                        strike_type=stype,
                        option_type=opt_code,
                        model=model,
                        spot=float(s_),
                        strike=float(strike_common),
                        rate=rate_common,
                        sigma=sigma,
                        maturity=maturity,
                        steps=int(steps),
                        m_points=m_points,
                    )
            heatmaps[(opt_label, stype)] = grid

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    plots = [
        ("Call", "fixed"),
        ("Call", "floating"),
        ("Put", "fixed"),
        ("Put", "floating"),
    ]
    for ax, (opt_label, stype) in zip(axes, plots):
        grid = heatmaps[(opt_label, stype)]
        im = ax.imshow(
            grid,
            extent=[k_vals.min(), k_vals.max(), s_vals.min(), s_vals.max()],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xlabel("K")
        ax.set_ylabel("S0")
        ax.set_title(f"{opt_label} asiatique - strike {stype}")
        fig.colorbar(im, ax=ax, label="Prix")

    plt.tight_layout()
    st.pyplot(fig)


def ui_asian_options(
    ticker,
    period,
    interval,
    spot_default,
    sigma_common,
    hist_df,
    maturity_common,
    strike_common,
    rate_common,
):
    st.header("Options asiatiques (module Asian)")

    if spot_default is None:
        st.warning("Aucun téléchargement yfinance : utilisez le spot commun.")
        spot_default = 57830.0
    if sigma_common is None:
        sigma_common = 0.05
    if hist_df is None:
        hist_df = pd.DataFrame()

    col1, col2 = st.columns(2)
    with col1:
        spot_common = st.session_state.get("common_spot", spot_default)
        strike_common_local = st.session_state.get("common_strike", strike_common)
        st.info(f"Spot commun S0 = {spot_common:.4f}")
        st.info(f"Strike commun K = {strike_common_local:.4f}")
        st.info(f"Taux sans risque commun r = {rate_common:.4f}")
    with col2:
        sigma = sigma_common
        st.info(f"Volatilité commune σ = {sigma:.4f}")
        maturity = maturity_common
        st.info(f"T commun = {maturity:.4f} années")
        steps = st.number_input(
            "Nombre de pas N",
            value=10,
            min_value=1,
            max_value=60,
            step=1,
            key="asian_steps",
        )

    st.subheader("Heatmaps prix asiatique (S0 vs K)")
    col_s, col_k = st.columns(2)
    with col_s:
        s_center = st.session_state.get("common_spot", spot_default)
        default_s_min = st.session_state.get("asian_s_min", max(0.01, s_center - 20.0))
        default_s_max = st.session_state.get("asian_s_max", s_center + 20.0)
        s_min = st.number_input(
            "S0 min",
            value=float(default_s_min),
            min_value=0.01,
            step=1.0,
            key="asian_s_min",
        )
        s_max = st.number_input(
            "S0 max",
            value=float(default_s_max),
            min_value=s_min + 1.0,
            step=1.0,
            key="asian_s_max",
        )
        st.caption(f"Domaine S0 utilisé: [{s_min:.2f}, {s_max:.2f}] pas 1")
    with col_k:
        k_center = st.session_state.get("common_strike", strike_common)
        default_k_min = st.session_state.get("asian_k_min", max(0.01, k_center - 20.0))
        default_k_max = st.session_state.get("asian_k_max", k_center + 20.0)
        k_min = st.number_input(
            "K min",
            value=float(default_k_min),
            min_value=0.01,
            step=1.0,
            key="asian_k_min",
        )
        k_max = st.number_input(
            "K max",
            value=float(default_k_max),
            min_value=k_min + 1.0,
            step=1.0,
            key="asian_k_max",
        )
        st.caption(f"Domaine K utilisé: [{k_min:.2f}, {k_max:.2f}] pas 1")

    s_vals = np.arange(s_min, s_max + 1.0, 1.0, dtype=float)
    k_vals = np.arange(k_min, k_max + 1.0, 1.0, dtype=float)

    tab_btm, tab_hw = st.tabs(["BTM naïf", "Hull-White (HW_BTM)"])

    with tab_btm:
        _render_asian_heatmaps_for_model(
            model="BTM naïf",
            s_vals=s_vals,
            k_vals=k_vals,
            sigma=sigma,
            maturity=maturity,
            steps=steps,
            m_points=None,
            strike_common=strike_common_local,
            rate_common=rate_common,
        )

    with tab_hw:
        m_points = st.number_input(
            "Nombre de points de moyenne M (Hull-White)",
            value=10,
            min_value=2,
            max_value=200,
            step=1,
            key="asian_m_points_hw",
        )
        _render_asian_heatmaps_for_model(
            model="Hull-White (HW_BTM)",
            s_vals=s_vals,
            k_vals=k_vals,
            sigma=sigma,
            maturity=maturity,
            steps=steps,
            m_points=m_points,
            strike_common=strike_common_local,
            rate_common=rate_common,
        )


def main():
    st.set_page_config(page_title="Basket + Asian", layout="wide")
    st.title("Application Streamlit : Basket + Asian")

    with st.sidebar:
        st.subheader("Recherche yfinance (commune Basket/Asian)")
        ticker = st.text_input("Ticker", value="AAPL", key="common_ticker")
        # Période et intervalle de prix fixés
        period = "2y"
        interval = "1d"
        fetch_data = st.button("Télécharger / actualiser les données yfinance", key="common_download")

        # Téléchargement yfinance, au premier affichage, au changement de ticker,
        # ou quand l'utilisateur clique sur le bouton.
        yf_key = st.session_state.get("yf_key")
        curr_key = (ticker, period, interval)
        should_fetch = fetch_data or yf_key is None or yf_key != curr_key

        if should_fetch:
            try:
                spot_yf, sigma_yf, hist_yf = get_spot_and_hist_vol(
                    ticker, period=period, interval=interval
                )
                st.session_state["yf_key"] = curr_key
                st.session_state["yf_spot"] = float(spot_yf)
                st.session_state["yf_sigma"] = float(sigma_yf)
                st.session_state["yf_hist_df"] = hist_yf

                # Met à jour automatiquement les bornes S0/K des heatmaps asiatiques (+/- 20)
                s_center = float(st.session_state.get("common_spot", spot_yf))
                k_center = float(st.session_state.get("common_strike", spot_yf))
                st.session_state["asian_s_min"] = max(0.01, s_center - 20.0)
                st.session_state["asian_s_max"] = s_center + 20.0
                st.session_state["asian_k_min"] = max(0.01, k_center - 20.0)
                st.session_state["asian_k_max"] = k_center + 20.0
            except Exception as exc:
                st.warning(f"Impossible de récupérer les prix pour {ticker}: {exc}")

        spot_seed = st.session_state.get("common_spot", st.session_state.get("yf_spot", 100.0))
        spot_common = st.number_input(
            "Spot commun S0 (pris pour les deux onglets)",
            value=spot_seed,
            min_value=0.01,
            key="common_spot",
        )
        maturity_common = st.number_input(
            "T commun (années, utilisé partout)",
            value=1.0,
            min_value=0.01,
            key="common_maturity",
        )
        strike_seed = st.session_state.get("common_strike", st.session_state.get("yf_spot", 100.0))
        strike_common = st.number_input(
            "Strike commun K (utilisé partout)",
            value=strike_seed,
            min_value=0.01,
            key="common_strike",
        )
        rate_common = st.number_input(
            "Taux sans risque commun r",
            value=0.01,
            step=0.001,
            format="%.4f",
            key="common_rate",
        )
        sigma_seed = st.session_state.get("yf_sigma", 0.2)
        sigma_common = st.number_input(
            "Volatilité commune σ",
            value=float(sigma_seed),
            min_value=0.0001,
            key="common_sigma",
        )
    # Récupère les données yfinance mises en cache dans la session
    hist_df = st.session_state.get("yf_hist_df", pd.DataFrame())
    spot_default = st.session_state.get("yf_spot", spot_common)
    sigma_default = st.session_state.get("yf_sigma", None)
    sigma_common = st.session_state.get(
        "common_sigma",
        sigma_default if sigma_default is not None else 0.2,
    )

    if not hist_df.empty:
        st.subheader("Courbe des prix de clôture (yfinance)")
        st.line_chart(hist_df.set_index("Date")["Close"])

    tab_basket, tab_asian = st.tabs(["Surface IV (Basket)", "Options asiatiques"])

    with tab_basket:
        ui_basket_surface(
            ticker=ticker,
            period=period,
            interval=interval,
            spot_common=spot_common,
            maturity_common=maturity_common,
            rate_common=rate_common,
            hist_df=hist_df,
        )
    with tab_asian:
        ui_asian_options(
            ticker=ticker,
            period=period,
            interval=interval,
            spot_default=spot_default,
            sigma_common=sigma_common,
            hist_df=hist_df,
            maturity_common=maturity_common,
            strike_common=strike_common,
            rate_common=rate_common,
        )


if __name__ == "__main__":
    main()
