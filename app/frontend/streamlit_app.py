from __future__ import annotations

from datetime import datetime
import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st


API_DEFAULT = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Sales Forecasting System")
st.caption("Weekly sales forecasting powered by FastAPI and Prophet")

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    api_url = st.text_input(
        "API URL",
        value=API_DEFAULT,
    ).rstrip("/")

    store_id = st.number_input(
        "Store ID",
        min_value=1,
        value=1,
        step=1,
    )

    dept_id = st.number_input(
        "Department ID",
        min_value=1,
        value=1,
        step=1,
    )

    periods = st.slider(
        "Forecast Periods (Weeks)",
        min_value=1,
        max_value=52,
        value=12,
    )

    selected_model = st.selectbox(
        "Model",
        ["prophet"],
    )

    st.divider()

    if st.button("Health Check", use_container_width=True):
        try:
            response = requests.get(
                f"{api_url}/",
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()

                st.success(f"API: {data['status']}")

                if "models_loaded" in data:
                    st.info(
                        "Models: "
                        + ", ".join(data["models_loaded"])
                    )
            else:
                st.error(
                    f"API returned HTTP {response.status_code}"
                )

        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

    if st.button("Refresh Models", use_container_width=True):
        try:
            response = requests.get(
                f"{api_url}/models",
                timeout=10,
            )

            if response.status_code == 200:
                models = response.json().get(
                    "available_models",
                    [],
                )

                st.success(
                    "Available models: "
                    + ", ".join(models)
                )
            else:
                st.error(
                    f"API returned HTTP {response.status_code}"
                )

        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")


# -------------------------------------------------------------------
# Main forecast
# -------------------------------------------------------------------

st.header("Sales Forecast")

if st.button(
    "🚀 Generate Forecast",
    type="primary",
    use_container_width=True,
):

    payload = {
        "store_id": int(store_id),
        "department_id": int(dept_id),
        "periods": int(periods),
        "model": selected_model,
        "include_uncertainty": False,
    }

    with st.spinner("Generating forecast..."):

        try:
            response = requests.post(
                f"{api_url}/forecast",
                json=payload,
                timeout=180,
            )

            if response.status_code != 200:
                try:
                    error_detail = response.json().get(
                        "detail",
                        response.text,
                    )
                except Exception:
                    error_detail = response.text

                st.error(
                    f"Forecast failed "
                    f"(HTTP {response.status_code}): "
                    f"{error_detail}"
                )

                st.stop()

            data = response.json()

            forecast_df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(data["dates"]),
                    "Sales": pd.to_numeric(
                        data["predictions"],
                        errors="coerce",
                    ),
                }
            )

            if forecast_df.empty:
                st.error("API returned an empty forecast.")
                st.stop()

            st.session_state.forecast_data = forecast_df
            st.session_state.forecast_metadata = data

            st.success(
                f"Forecast generated for "
                f"Store {data['store_id']} / "
                f"Department {data['department_id']}"
            )

        except requests.RequestException as exc:
            st.error(f"API connection failed: {exc}")

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")


# -------------------------------------------------------------------
# Results
# -------------------------------------------------------------------

if "forecast_data" in st.session_state:

    df = st.session_state.forecast_data
    metadata = st.session_state.forecast_metadata

    st.divider()

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    total_sales = df["Sales"].sum()
    average_sales = df["Sales"].mean()
    max_sales = df["Sales"].max()
    min_sales = df["Sales"].min()

    with col1:
        st.metric(
            "Total Forecast",
            f"${total_sales:,.0f}",
        )

    with col2:
        st.metric(
            "Average Weekly",
            f"${average_sales:,.0f}",
        )

    with col3:
        st.metric(
            "Highest Week",
            f"${max_sales:,.0f}",
        )

    with col4:
        st.metric(
            "Lowest Week",
            f"${min_sales:,.0f}",
        )

    # Trend
    if len(df) > 1:
        first_value = df["Sales"].iloc[0]
        last_value = df["Sales"].iloc[-1]

        if first_value != 0:
            percentage_change = (
                (last_value - first_value)
                / abs(first_value)
                * 100
            )
        else:
            percentage_change = 0.0

        if percentage_change > 0:
            st.success(
                f"Forecast trend: upward "
                f"({percentage_change:+.1f}%)"
            )
        elif percentage_change < 0:
            st.warning(
                f"Forecast trend: downward "
                f"({percentage_change:+.1f}%)"
            )
        else:
            st.info("Forecast trend: stable")

    # Forecast chart
    st.subheader("Forecast")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Sales"],
            mode="lines+markers",
            name="Forecast",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="Weekly Sales ($)",
        hovermode="x unified",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Bar chart
    st.subheader("Weekly Forecast")

    bar_fig = go.Figure()

    bar_fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Sales"],
            name="Weekly Sales",
        )
    )

    bar_fig.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="Weekly Sales ($)",
    )

    st.plotly_chart(
        bar_fig,
        use_container_width=True,
    )

    # Data
    with st.expander("View Forecast Data"):

        display_df = df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime(
            "%Y-%m-%d"
        )
        display_df["Sales"] = display_df["Sales"].map(
            lambda value: f"${value:,.2f}"
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        csv = df.to_csv(index=False)

        st.download_button(
            "Download Forecast CSV",
            data=csv,
            file_name=(
                f"forecast_store_{int(store_id)}_"
                f"dept_{int(dept_id)}_"
                f"{datetime.now().strftime('%Y%m%d')}.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    # Metadata
    with st.expander("Forecast Metadata"):

        st.json(
            {
                "store_id": metadata["store_id"],
                "department_id": metadata["department_id"],
                "model": metadata["model"],
                "periods": metadata["periods"],
                "generated_at": metadata.get(
                    "generated_at",
                    datetime.now().isoformat(),
                ),
            }
        )

else:

    st.info(
        "Configure a store and department, then click "
        "**Generate Forecast**."
    )
