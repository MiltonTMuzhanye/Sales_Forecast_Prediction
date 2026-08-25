import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📊",
    layout="wide"
)

st.title(" Sales Forecasting System")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    api_url = st.text_input("API URL", "http://localhost:8000")
    
    store_id = st.number_input("Store ID", min_value=1, value=1)
    dept_id = st.number_input("Department ID", min_value=1, value=1)
    
    periods = st.slider("Forecast Periods (Weeks)", min_value=4, max_value=52, value=12)
    
    model_options = ["prophet", "xgboost", "lightgbm"]
    selected_model = st.selectbox("Select Model", model_options)
    
    if st.button("Refresh Models"):
        try:
            response = requests.get(f"{api_url}/models")
            if response.status_code == 200:
                models = response.json()['available_models']
                st.success(f"Available: {', '.join(models)}")
        except Exception as e:
            st.error(f"Error: {e}")
    
    if st.button("Health Check"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                data = response.json()
                st.success(f"Status: {data['status']}")
                st.info(f"Models: {', '.join(data['models_loaded'])}")
        except Exception as e:
            st.error(f"Error: {e}")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.header(" Sales Forecast")
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            try:
                payload = {
                    "store_id": int(store_id),
                    "department_id": int(dept_id),
                    "periods": int(periods),
                    "model": selected_model,
                    "include_uncertainty": True
                }
                
                response = requests.post(f"{api_url}/forecast", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    forecast_df = pd.DataFrame({
                        'Date': data['dates'],
                        'Sales': data['predictions']
                    })
                    forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                    
                    st.session_state.forecast_data = forecast_df
                    st.success("Forecast generated!")
                    
            except Exception as e:
                st.error(f"Error: {e}")

    if 'forecast_data' in st.session_state:
        df = st.session_state.forecast_data
        
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=("Sales Forecast", "Weekly Sales Trend"),
                           vertical_spacing=0.15)
        
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['Sales'],
                      mode='lines+markers', name='Forecast',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['Sales'],
                  name='Weekly Sales',
                  marker_color='rgba(0,150,255,0.6)'),
            row=2, col=1
        )
        
        fig.update_layout(height=600, showlegend=True)
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
        fig.update_yaxes(title_text="Sales ($)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("View Data"):
            st.dataframe(df.style.format({'Sales': '${:,.2f}'}))
            
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, 
                             file_name=f"forecast_{datetime.now().strftime('%Y%m%d')}.csv")

with col2:
    st.header(" Metrics")
    
    if 'forecast_data' in st.session_state:
        df = st.session_state.forecast_data
        
        st.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
        st.metric("Average Weekly", f"${df['Sales'].mean():,.0f}")
        st.metric("Max Weekly", f"${df['Sales'].max():,.0f}")
        st.metric("Min Weekly", f"${df['Sales'].min():,.0f}")
        
        if len(df) > 1:
            trend = df['Sales'].iloc[-1] - df['Sales'].iloc[0]
            if trend > 0:
                st.success(f" Upward: +{trend/df['Sales'].iloc[0]*100:.1f}%")
            else:
                st.error(f" Downward: {trend/df['Sales'].iloc[0]*100:.1f}%")