import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import norm
import streamlit as st


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
        "Chargez un fichier CSV contenant les colonnes `K`, `T` et `iv` "
        "(par exemple `Basket/data/train.csv`)."
    )

    uploaded = st.file_uploader("Fichier CSV", type=["csv"])

    col_left, col_right = st.columns(2)
    with col_left:
        spot = st.number_input("Spot S0", value=100.0, min_value=0.01)
        k_span = st.number_input("Étendue en strike autour de S0", value=100.0, min_value=1.0)
    with col_right:
        max_maturity = st.number_input("Maturité maximale T_max (années)", value=2.0, min_value=0.1)
        grid_k = st.number_input("Points de grille en K", value=100, min_value=20, max_value=400, step=10)
        grid_t = st.number_input("Points de grille en T", value=100, min_value=20, max_value=400, step=10)

    if uploaded is not None:
        try:
            data_frame = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Erreur de lecture du CSV: {exc}")
            return

        required_cols = {"K", "T", "iv"}
        if not required_cols.issubset(data_frame.columns):
            missing = required_cols - set(data_frame.columns)
            st.error(f"Colonnes manquantes dans le CSV: {missing}")
            return

        if st.button("Tracer la surface IV"):
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

    col_model, col_type = st.columns(2)
    with col_model:
        model = st.selectbox(
            "Schéma binomial",
            ["BTM naïf", "Hull-White (HW_BTM)"],
        )
    with col_type:
        option_label = st.selectbox("Type d'option", ["Call", "Put"])

    strike_type_label = st.selectbox("Type de strike asiatique", ["fixed", "floating"])

    col_spot, col_strike, col_rate = st.columns(3)
    with col_spot:
        spot = st.number_input("Spot S0", value=57830.0, min_value=0.01)
    with col_strike:
        strike = st.number_input("Strike K", value=58000.0, min_value=0.01)
    with col_rate:
        rate = st.number_input("Taux sans risque r", value=0.01)

    col_sigma, col_maturity, col_steps = st.columns(3)
    with col_sigma:
        sigma = st.number_input("Volatilité σ", value=0.05, min_value=0.0001)
    with col_maturity:
        maturity = st.number_input("Maturité T (années)", value=1.0, min_value=0.01)
    with col_steps:
        max_steps = 15 if model == "BTM naïf" else 60
        steps = st.number_input(
            "Nombre de pas N",
            value=10,
            min_value=1,
            max_value=max_steps,
            step=1,
        )

    m_points = None
    if model == "Hull-White (HW_BTM)":
        m_points = st.number_input(
            "Nombre de points de moyenne M",
            value=10,
            min_value=2,
            max_value=200,
            step=1,
        )

    show_bs = st.checkbox("Afficher le prix européen Black-Scholes correspondant", value=True)

    if st.button("Calculer le prix asiatique"):
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

    section = st.sidebar.radio(
        "Choisissez un module",
        ["Surface IV (Basket)", "Options asiatiques"],
    )

    if section == "Surface IV (Basket)":
        ui_basket_surface()
    else:
        ui_asian_options()


if __name__ == "__main__":
    main()

