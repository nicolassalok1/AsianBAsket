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
    expiry_dt = pd.to_datetime(expiry)
    now_ts = pd.Timestamp.utcnow().normalize()
    tau_years = max((expiry_dt - now_ts).total_seconds() / (365.0 * 24 * 3600), 0.0)

    frames = []
    for frame in [chain.calls, chain.puts]:
        tmp = frame[["strike", "impliedVolatility"]].rename(
            columns={"strike": "K", "impliedVolatility": "iv"}
        )
        tmp["T"] = tau_years
        frames.append(tmp)
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["K", "iv"])
    return df


@st.cache_data(show_spinner=False)
def get_spot_and_hist_vol(ticker: str, period: str = "6mo", interval: str = "1d"):
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError("Aucune donnée téléchargée.")
    close = data["Close"]
    spot = float(close.iloc[-1])
    log_returns = np.log(close / close.shift(1)).dropna()
    sigma = float(log_returns.std() * np.sqrt(252))
    return spot, sigma, data.tail(10).reset_index()


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


def ui_basket_surface():
    st.header("Surface de volatilité implicite (module Basket)")
    st.markdown(
        "Les paramètres et actions sont disponibles dans la sidebar. "
        "Le fichier CSV requis doit contenir les colonnes `K`, `T` et `iv`."
    )

    with st.sidebar:
        st.subheader("Paramètres Basket")
        ticker = st.text_input("Ticker Yahoo Finance", value="AAPL", key="basket_ticker")
        expiries = get_option_expiries(ticker)
        if not expiries:
            st.warning("Aucune échéance d'options récupérée pour ce ticker.")
        expiry = st.selectbox(
            "Échéance (options)",
            options=expiries if expiries else ["N/A"],
            index=0,
            key="basket_expiry",
        )
        period = st.selectbox(
            "Période",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
            index=3,
            key="basket_period",
        )
        interval = st.selectbox(
            "Intervalle",
            ["1d", "1wk", "1mo"],
            index=0,
            key="basket_interval",
        )
        download_requested = st.button("Télécharger le CSV du ticker", key="basket_download")

        try:
            spot_default, _, _ = get_spot_and_hist_vol(ticker, period="1mo", interval="1d")
        except Exception:
            spot_default = 100.0
        spot = st.number_input("Spot S0 (auto yfinance)", value=spot_default, min_value=0.01, key="basket_spot")
        k_span = st.number_input(
            "Étendue en strike autour de S0",
            value=100.0,
            min_value=1.0,
            key="basket_k_span",
        )
        max_maturity = st.number_input(
            "Maturité maximale T_max (années)",
            value=2.0,
            min_value=0.1,
            key="basket_tmax",
        )
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
        plot_requested = st.button("Construire la surface IV", key="basket_plot")

    if download_requested:
        with st.spinner("Téléchargement en cours..."):
            data_downloaded = yf.download(ticker, period=period, interval=interval)
        if data_downloaded.empty:
            st.error("Aucune donnée téléchargée. Vérifiez le ticker ou modifiez la période/intervalle.")
        else:
            data_reset = data_downloaded.reset_index()
            csv_bytes = data_reset.to_csv(index=False).encode("utf-8")
            st.success(f"Données téléchargées pour {ticker}.")
            st.download_button(
                label="Télécharger le CSV",
                data=csv_bytes,
                file_name=f"{ticker}_data.csv",
                mime="text/csv",
            )
            st.dataframe(data_reset.tail(10))

    if plot_requested:
        try:
            data_frame = get_option_surface_from_yf(ticker, expiry)
        except Exception as exc:
            st.error(f"Erreur lors de la récupération des options yfinance: {exc}")
            return

        required_cols = {"K", "T", "iv"}
        if not required_cols.issubset(data_frame.columns):
            missing = required_cols - set(data_frame.columns)
            st.error(f"Colonnes manquantes dans le CSV: {missing}")
            return

        try:
            k_grid, t_grid, iv_grid = build_grid(
                data_frame,
                spot=spot,
                n_k=int(grid_k),
                n_t=int(grid_t),
                k_span=k_span,
                t_min=0.0,
                t_max=max_maturity,
            )
            fig = make_iv_surface_figure(
                k_grid,
                t_grid,
                iv_grid,
                title_suffix=f" (S0={spot})",
            )
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"Erreur lors de la construction de la surface: {exc}")


def ui_asian_options():
    st.header("Options asiatiques (module Asian)")

    with st.sidebar:
        st.subheader("Paramètres Asian")
        ticker = st.text_input("Ticker Yahoo Finance", value="AAPL", key="asian_ticker")
        try:
            spot_default, sigma_default, hist_tail = get_spot_and_hist_vol(
                ticker, period="6mo", interval="1d"
            )
        except Exception:
            spot_default, sigma_default, hist_tail = 57830.0, 0.05, pd.DataFrame()
        model = st.selectbox(
            "Schéma binomial",
            ["BTM naïf", "Hull-White (HW_BTM)"],
            key="asian_model",
        )
        option_label = st.selectbox("Type d'option", ["Call", "Put"], key="asian_option_label")
        strike_type_label = st.selectbox(
            "Type de strike asiatique", ["fixed", "floating"], key="asian_strike_type"
        )

        spot = st.number_input("Spot S0 (auto yfinance)", value=spot_default, min_value=0.01, key="asian_spot")
        strike = st.number_input("Strike K", value=spot_default, min_value=0.01, key="asian_strike")
        rate = st.number_input("Taux sans risque r", value=0.01, key="asian_rate")

        sigma = st.number_input(
            "Volatilité σ (hist. yfinance)", value=sigma_default, min_value=0.0001, key="asian_sigma"
        )
        maturity = st.number_input(
            "Maturité T (années)", value=1.0, min_value=0.01, key="asian_maturity"
        )
        max_steps = 15 if model == "BTM naïf" else 60
        steps = st.number_input(
            "Nombre de pas N",
            value=10,
            min_value=1,
            max_value=max_steps,
            step=1,
            key="asian_steps",
        )

        m_points = None
        if model == "Hull-White (HW_BTM)":
            m_points = st.number_input(
                "Nombre de points de moyenne M",
                value=10,
                min_value=2,
                max_value=200,
                step=1,
                key="asian_m_points",
            )

        show_bs = st.checkbox(
            "Afficher le prix européen Black-Scholes correspondant",
            value=True,
            key="asian_show_bs",
        )
        compute_requested = st.button("Calculer le prix asiatique", key="asian_compute")

    if hist_tail is not None and not hist_tail.empty:
        st.caption("Aperçu des dernières observations de clôture (yfinance)")
        st.dataframe(hist_tail)

    if compute_requested:
        option_type = "C" if option_label == "Call" else "P"
        with st.spinner("Calcul en cours..."):
            try:
                if model == "BTM naïf":
                    price = btm_asian(
                        strike_type=strike_type_label,
                        option_type=option_type,
                        spot=spot,
                        strike=strike,
                        rate=rate,
                        sigma=sigma,
                        maturity=maturity,
                        steps=int(steps),
                    )
                else:
                    price = hw_btm_asian(
                        strike_type=strike_type_label,
                        option_type=option_type,
                        spot=spot,
                        strike=strike,
                        rate=rate,
                        sigma=sigma,
                        maturity=maturity,
                        steps=int(steps),
                        m_points=int(m_points),
                    )
            except Exception as exc:
                st.error(f"Erreur lors du calcul asiatique: {exc}")
                return

        st.success(f"Prix de l'option asiatique: {price:.6f}")

        if show_bs and strike_type_label == "fixed":
            option_kind = "call" if option_label == "Call" else "put"
            try:
                euro_price = bs_option_price(
                    time=0.0,
                    spot=spot,
                    strike=strike,
                    maturity=maturity,
                    rate=rate,
                    sigma=sigma,
                    option_kind=option_kind,
                )
                st.info(f"Prix européen Black-Scholes (même K, T): {euro_price:.6f}")
            except Exception as exc:
                st.error(f"Erreur lors du calcul Black-Scholes: {exc}")


def main():
    st.set_page_config(page_title="Basket + Asian", layout="wide")
    st.title("Application Streamlit : Basket + Asian")

    tab_basket, tab_asian = st.tabs(["Surface IV (Basket)", "Options asiatiques"])

    with tab_basket:
        ui_basket_surface()
    with tab_asian:
        ui_asian_options()


if __name__ == "__main__":
    main()
