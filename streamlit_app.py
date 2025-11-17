import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import stats
from scipy.stats import norm
import streamlit as st
import yfinance as yf
import io
import tensorflow as tf
import os
import subprocess
import sys
from pathlib import Path


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


def fetch_closing_prices(tickers, period="1mo", interval="1d"):
    if isinstance(tickers, str):
        tickers = [tickers]
    # Nettoie les variables d'impersonation problématiques
    for var in ["YF_IMPERSONATE", "YF_SCRAPER_IMPERSONATE"]:
        try:
            os.environ.pop(var, None)
        except Exception:
            pass
    try:
        yf.set_config(proxy=None)
    except Exception:
        pass

    data = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if data.empty:
        raise RuntimeError(f"Aucune donnée récupérée pour {tickers} sur {period}.")

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
        elif "Close" in data.columns:
            prices = data[["Close"]].copy()
        else:
            raise RuntimeError("Colonnes de prix introuvables dans les données yfinance.")
        prices.columns = tickers

    prices = prices.reset_index()
    return prices


def compute_corr_from_prices(prices_df: pd.DataFrame):
    price_cols = [c for c in prices_df.columns if c.lower() != "date"]
    returns = np.log(prices_df[price_cols] / prices_df[price_cols].shift(1)).dropna(how="any")
    if returns.empty:
        raise RuntimeError("Pas assez de données pour calculer la corrélation.")
    return returns.corr()


class BasketOption:
    def __init__(self, weights, prices, volatility, corr, strike, maturity, rate):
        self.weights = weights
        self.vol = volatility
        self.strike = strike
        self.mat = maturity
        self.rate = rate
        self.corr = corr
        self.prices = prices

    def get_mc(self, m_paths: int = 10000):
        b_ts = stats.multivariate_normal(np.zeros(len(self.weights)), cov=self.corr).rvs(size=m_paths)
        s_ts = self.prices * np.exp((self.rate - 0.5 * self.vol**2) * self.mat + self.vol * b_ts)
        if len(self.weights) > 1:
            payoffs = (np.sum(self.weights * s_ts, axis=1) - self.strike).clip(0)
        else:
            payoffs = np.maximum(s_ts - self.strike, np.zeros(m_paths))
        return float(np.exp(-self.rate * self.mat) * np.mean(payoffs))

    def get_bs_price(self):
        d1 = (np.log(self.prices / self.strike) + (self.rate + 0.5 * self.vol**2) * self.mat) / (
            self.vol * np.sqrt(self.mat)
        )
        d2 = d1 - self.vol * np.sqrt(self.mat)
        bs_price = stats.norm.cdf(d1) * self.prices - stats.norm.cdf(d2) * self.strike * np.exp(
            -self.rate * self.mat
        )
        return float(bs_price)


class DataGen:
    def __init__(self, n_assets: int, n_samples: int):
        if n_samples <= 0:
            raise ValueError("n_samples needs to be positive")
        if n_assets <= 0:
            raise ValueError("n_assets needs to be positive")
        self.n_assets = n_assets
        self.n_samples = n_samples

    def generate(self, corr, strike_price: float, base_price: float, method: str = "bs"):
        mats = np.random.uniform(0.2, 1.1, size=self.n_samples)
        vols = np.random.uniform(0.01, 1.0, size=self.n_samples)
        rates = np.random.uniform(0.02, 0.1, size=self.n_samples)

        strikes = np.random.randn(self.n_samples) + strike_price
        prices = np.random.randn(self.n_samples) + base_price

        if self.n_assets > 1:
            weights = np.random.rand(self.n_samples * self.n_assets).reshape((self.n_samples, self.n_assets))
            weights /= np.sum(weights, axis=1)[:, np.newaxis]
        else:
            weights = np.ones((self.n_samples, self.n_assets))

        labels = []
        for i in range(self.n_samples):
            basket = BasketOption(
                weights[i],
                prices[i],
                vols[i],
                corr,
                strikes[i],
                mats[i],
                rates[i],
            )
            if method == "bs":
                labels.append(basket.get_bs_price())
            else:
                labels.append(basket.get_mc())

        data = pd.DataFrame(
            {
                "S/K": prices / strikes,
                "Maturity": mats,
                "Volatility": vols,
                "Rate": rates,
                "Labels": labels,
                "Prices": prices,
                "Strikes": strikes,
            }
        )
        for i in range(self.n_assets):
            data[f"Weight_{i}"] = weights[:, i]
        return data


def simulate_dataset_notebook(n_assets: int, n_samples: int, method: str, corr: np.ndarray, base_price: float, base_strike: float):
    generator = DataGen(n_assets=n_assets, n_samples=n_samples)
    return generator.generate(corr=corr, strike_price=base_strike, base_price=base_price, method=method)


@st.cache_data(show_spinner=False)
def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


def split_data_nn(data: pd.DataFrame, split_ratio: float = 0.7):
    feature_cols = ["S/K", "Maturity", "Volatility", "Rate"]
    target_col = "Labels"
    train = data.iloc[: int(split_ratio * len(data)), :]
    test = data.iloc[int(split_ratio * len(data)) :, :]
    x_train, y_train = train[feature_cols], train[target_col]
    x_test, y_test = test[feature_cols], test[target_col]
    return x_train, y_train, x_test, y_test


def build_model_nn(input_dim: int) -> tf.keras.Model:
    inp = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(32, activation="relu")(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    out = tf.keras.layers.Dense(1, activation="relu")(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        loss="mean_squared_error",
        optimizer="adam",
        metrics=["mean_squared_error"],
    )
    return model


def price_basket_nn(model: tf.keras.Model, S: float, K: float, maturity: float, volatility: float, rate: float) -> float:
    S_over_K = S / K
    x = np.array([[S_over_K, maturity, volatility, rate]], dtype=float)
    return float(model.predict(x, verbose=0)[0, 0])


def plot_heatmap_nn(model: tf.keras.Model, data: pd.DataFrame, spot_ref: float | None = None, strike_ref: float | None = None, maturity_fixed: float = 1.0):
    """
    Heatmap comme dans le notebook : grille S (vertical) x K (horizontal),
    T fixé à 1 an, sigma/rate = médianes, prédiction directe sur S/K.
    """
    df = data.copy()
    if "Prices" not in df.columns and spot_ref is not None:
        df["Prices"] = spot_ref
    if "Strikes" not in df.columns and strike_ref is not None:
        df["Strikes"] = strike_ref

    if not {"Prices", "Strikes"}.issubset(df.columns):
        raise ValueError("Colonnes Prices et Strikes requises pour reproduire la heatmap du notebook.")

    s_min, s_max = df["Prices"].quantile([0.01, 0.99])
    k_min, k_max = df["Strikes"].quantile([0.01, 0.99])
    n_S, n_K = 50, 50
    s_vals = np.linspace(s_min, s_max, n_S)
    k_vals = np.linspace(k_min, k_max, n_K)

    K_grid, S_grid = np.meshgrid(k_vals, s_vals)
    s_over_k_grid = S_grid / K_grid

    sigma_ref = float(df["Volatility"].median())
    rate_ref = float(df["Rate"].median())

    X = np.stack(
        [
            s_over_k_grid.ravel(),
            np.full(s_over_k_grid.size, maturity_fixed),
            np.full(s_over_k_grid.size, sigma_ref),
            np.full(s_over_k_grid.size, rate_ref),
        ],
        axis=1,
    )
    prices_grid = model.predict(X, verbose=0).reshape(n_S, n_K)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        prices_grid,
        origin="lower",
        extent=[k_vals.min(), k_vals.max(), s_vals.min(), s_vals.max()],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Spot S")
    ax.set_title("Heatmap du prix NN en fonction de S et K (T=1 an)")
    fig.colorbar(im, ax=ax, label="Prix NN")
    plt.tight_layout()
    return fig


def build_grid(
    df: pd.DataFrame,
    spot: float,
    n_k: int = 200,
    n_t: int = 200,
    k_min: float | None = None,
    k_max: float | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    k_span: float | None = None,
    margin_frac: float = 0.02,
):
    """Version notebook : bornes K/T déduites des données avec marge, option k_span autour du spot."""
    if k_min is None or k_max is None:
        if k_span is not None:
            k_min = spot - k_span
            k_max = spot + k_span
        else:
            data_k_min = float(df["K"].min())
            data_k_max = float(df["K"].max())
            delta_k = data_k_max - data_k_min
            pad = delta_k * margin_frac
            k_min = data_k_min - pad
            k_max = data_k_max + pad

    if t_min is None:
        t_min = float(df["T"].min())
    if t_max is None:
        t_max = float(df["T"].max())

    if k_min >= k_max:
        raise ValueError("k_min doit être inférieur à k_max.")
    if t_min >= t_max:
        raise ValueError("t_min doit être inférieur à t_max.")

    k_vals = np.linspace(k_min, k_max, n_k)
    t_vals = np.linspace(t_min, t_max, n_t)

    df = df[(df["K"] >= k_min) & (df["K"] <= k_max)].copy()
    df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

    if df.empty:
        raise ValueError("Aucun point n'appartient au domaine défini par la grille.")

    df["K_idx"] = np.searchsorted(k_vals, df["K"], side="left").clip(0, n_k - 1)
    df["T_idx"] = np.searchsorted(t_vals, df["T"], side="left").clip(0, n_t - 1)

    grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

    iv_grid = np.full((n_t, n_k), np.nan, dtype=float)
    for _, row in grouped.iterrows():
        iv_grid[int(row["T_idx"]), int(row["K_idx"])] = row["iv"]

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


def ui_basket_surface(spot_common, maturity_common, rate_common, strike_common):
    st.header("Basket – Pricing NN + corrélation (3 actifs)")

    # Saisie des tickers (corrélation)
    if "basket_tickers" not in st.session_state:
        st.session_state["basket_tickers"] = ["AAPL", "SPY", "MSFT"]

    min_assets, max_assets = 2, 10
    with st.container():
        st.subheader("Sélection des assets (2 à 10)")
        btn_col_add, btn_col_remove = st.columns(2)
        with btn_col_add:
            if st.button("Ajouter un asset", disabled=len(st.session_state["basket_tickers"]) >= max_assets):
                st.session_state["basket_tickers"].append(f"TICKER{len(st.session_state['basket_tickers']) + 1}")
        with btn_col_remove:
            if st.button("Retirer un asset", disabled=len(st.session_state["basket_tickers"]) <= min_assets):
                st.session_state["basket_tickers"].pop()

        # Champs de saisie dynamiques (3 par ligne)
        tickers = []
        for i, default_tk in enumerate(st.session_state["basket_tickers"]):
            if i % 3 == 0:
                cols = st.columns(3)
            col = cols[i % 3]
            with col:
                tick = st.text_input(f"Ticker {i + 1}", value=default_tk, key=f"corr_tk_dynamic_{i}")
                tickers.append(tick.strip() or default_tk)
        # Met à jour la session avec les saisies (borne le nombre)
        tickers = tickers[:max_assets]
        if len(tickers) < min_assets:
            tickers += ["SPY"] * (min_assets - len(tickers))
        st.session_state["basket_tickers"] = tickers

    period = st.selectbox("Période yfinance", ["1mo", "3mo", "6mo", "1y"], index=0, key="corr_period")
    interval = st.selectbox("Intervalle", ["1d", "1h"], index=0, key="corr_interval")

    st.caption("Le calcul de corrélation utilise les prix de clôture ajustés téléchargés via yfinance. En cas d'échec, une matrice de corrélation inventée sera utilisée.")
    # Génère closing_prices.csv via le script dédié (avec les tickers saisis)
    regen_csv = st.button("Regénérer closing_prices.csv", key="btn_regen_closing")
    try:
        if regen_csv or not Path("closing_prices.csv").exists():
            cmd = [sys.executable, "fetch_closing_prices.py", "--tickers", *tickers, "--output", "closing_prices.csv"]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            st.info(f"closing_prices.csv généré via le script ({res.stdout.strip()})")
    except Exception as exc:
        st.warning(f"Impossible d'exécuter fetch_closing_prices.py : {exc}")

    corr_df = None
    try:
        # Charge la matrice de corrélation depuis closing_prices.csv généré ci-dessus
        closing_path = Path("closing_prices.csv")
        prices = pd.read_csv(closing_path)
        corr_df = compute_corr_from_prices(prices)
        st.success(f"Corrélation calculée à partir de {closing_path.name}")
        st.dataframe(corr_df)
    except Exception as exc:
        st.warning(f"Impossible de calculer la corrélation depuis closing_prices.csv : {exc}")
        corr_df = pd.DataFrame(
            [
                [1.0, 0.6, 0.4],
                [0.6, 1.0, 0.7],
                [0.4, 0.7, 1.0],
            ],
            columns=tickers,
            index=tickers,
        )
        st.info("Utilisation d'une matrice de corrélation inventée pour la suite des calculs.")
        st.dataframe(corr_df)

    st.subheader("Dataset Basket pour NN")
    st.subheader("Dataset Basket pour NN")
    st.caption("Dataset généré automatiquement via DataGen (comme dans le notebook).")
    n_samples = st.slider("Taille du dataset simulé", 1000, 20000, 10000, 1000)
    method = st.selectbox("Méthode de pricing pour les labels", ["bs", "mc"], index=0)

    df = simulate_dataset_notebook(
        n_assets=len(tickers),
        n_samples=int(n_samples),
        method=method,
        corr=corr_df.values,
        base_price=float(spot_common),
        base_strike=float(strike_common),
    )

    st.write("Aperçu :", df.head())
    st.write("Shape :", df.shape)

    split_ratio = st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05)
    epochs = st.slider("Epochs d'entraînement", 5, 200, 20, 5)

    x_train, y_train, x_test, y_test = split_data_nn(df, split_ratio=split_ratio)
    Path("data").mkdir(parents=True, exist_ok=True)
    pd.concat([x_train, y_train], axis=1).to_csv("data/train.csv", index=False)
    pd.concat([x_test, y_test], axis=1).to_csv("data/test.csv", index=False)
    st.info("train.csv et test.csv régénérés pour la surface IV.")

    st.write(f"Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")

    train_button = st.button("Entraîner le modèle NN", key="btn_train_nn")
    if not train_button:
        st.info("Clique sur 'Entraîner le modèle NN' pour lancer l'apprentissage.")
        return

    tf.keras.backend.clear_session()
    model = build_model_nn(input_dim=x_train.shape[1])
    train_logs: list[str] = []
    log_box = st.empty()

    class StreamlitLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            msg = f"Epoch {epoch + 1}/{epochs} - loss: {logs.get('loss', float('nan')):.4f} - mse: {logs.get('mean_squared_error', float('nan')):.4f}"
            if "val_loss" in logs or "val_mean_squared_error" in logs:
                msg += f" - val_loss: {logs.get('val_loss', float('nan')):.4f} - val_mse: {logs.get('val_mean_squared_error', float('nan')):.4f}"
            train_logs.append(msg)
            log_box.text("\n".join(train_logs))

    with st.spinner("Entraînement du NN en cours…"):
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            verbose=0,
            callbacks=[StreamlitLogger()],
        )
    st.success("Entraînement terminé.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Courbes MSE")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history.history["mean_squared_error"], label="train")
        ax.plot(history.history["val_mean_squared_error"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

    with col2:
        st.subheader("Heatmap prix NN (S vs K)")
        try:
            with st.spinner("Calcul de la heatmap…"):
                heatmap_fig = plot_heatmap_nn(
                   model=model,
                    data=df,
                    spot_ref=float(spot_common),
                    strike_ref=float(strike_common),
                    maturity_fixed=1.0,
                )
            st.pyplot(heatmap_fig)
        except Exception as exc:
            st.warning(f"Impossible d'afficher la heatmap : {exc}")

    st.subheader("Surface IV (Strike, Maturité)")
    try:
        with st.spinner("Calcul de la surface IV…"):
            iv_df = df.copy()
            if "Strikes" in iv_df.columns:
                iv_df["K"] = iv_df["Strikes"]
            else:
                iv_df["K"] = spot_common / iv_df["S/K"].replace(0.0, np.nan)
            iv_df = iv_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["K", "Maturity", "Volatility"])

            if iv_df.empty:
                raise ValueError("Pas de données IV exploitables (S/K nuls ou manquants).")

            spot_ref_for_grid = float(iv_df["Prices"].mean()) if "Prices" in iv_df.columns else float(spot_common)

            grid_k, grid_t, grid_iv = build_grid(
                df=iv_df.rename(columns={"Maturity": "T", "Volatility": "iv"}),
                spot=spot_ref_for_grid,
            )
            iv_fig = make_iv_surface_figure(grid_k, grid_t, grid_iv, title_suffix=" (dataset NN)")
        st.pyplot(iv_fig)
    except Exception as exc:
        st.warning(f"Impossible d'afficher la surface IV : {exc}")


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

    tab_basket, tab_asian = st.tabs(["Basket (NN pricing)", "Options asiatiques"])

    with tab_basket:
        ui_basket_surface(
            spot_common=spot_common,
            maturity_common=maturity_common,
            rate_common=rate_common,
            strike_common=strike_common,
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
