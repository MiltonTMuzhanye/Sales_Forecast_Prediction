import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from pathlib import Path
import requests
import sys
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.append('..')

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #F18F01;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    .forecast-card {
        background-color: #fff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

class ForecastingDashboard:
    """Streamlit dashboard for sales forecasting"""
    
    def __init__(self):
        self.api_url = "http://localhost:8000"
        self.models_dir = Path("../models")
        
    def load_data(self):
        """Load sample data for demonstration"""
        try:
            # Load features data
            features_path = Path("../data/features/time_series_features.parquet")
            if features_path.exists():
                self.features_df = pd.read_parquet(features_path)
                self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])
            else:
                self.features_df = None
            
            # Load predictions if available
            predictions_path = self.models_dir / "predictions_store_1_dept_1.csv"
            if predictions_path.exists():
                self.predictions_df = pd.read_csv(predictions_path)
                self.predictions_df['Date'] = pd.to_datetime(self.predictions_df['Date'])
            else:
                self.predictions_df = None
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            self.features_df = None
            self.predictions_df = None
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        with st.sidebar:
            st.title("📊 Forecast Controls")
            
            # Store selection
            st.subheader("Store Selection")
            if self.features_df is not None:
                stores = sorted(self.features_df['Store'].unique())
                selected_store = st.selectbox("Select Store", stores, index=0)
                
                # Department selection
                depts = sorted(
                    self.features_df[self.features_df['Store'] == selected_store]['Dept'].unique()
                )
                selected_dept = st.selectbox("Select Department", depts, index=0)
            else:
                selected_store = st.number_input("Store ID", min_value=1, value=1)
                selected_dept = st.number_input("Department ID", min_value=1, value=1)
            
            # Forecast horizon
            st.subheader("Forecast Settings")
            forecast_horizon = st.slider(
                "Forecast Horizon (weeks)",
                min_value=4,
                max_value=52,
                value=12,
                step=4
            )
            
            # Model selection
            model_type = st.selectbox(
                "Forecast Model",
                ["Prophet", "ARIMA", "SARIMA", "Ensemble"],
                index=0
            )
            
            # Additional options
            st.subheader("Options")
            include_confidence = st.checkbox("Show Confidence Intervals", value=True)
            show_historical = st.checkbox("Show Historical Data", value=True)
            
            # Generate forecast button
            generate_forecast = st.button(
                "🚀 Generate Forecast",
                type="primary",
                use_container_width=True
            )
            
            # Refresh data button
            refresh_data = st.button(
                "🔄 Refresh Data",
                use_container_width=True
            )
            
            return {
                'store_id': int(selected_store),
                'dept_id': int(selected_dept),
                'forecast_horizon': forecast_horizon,
                'model_type': model_type.lower(),
                'include_confidence': include_confidence,
                'show_historical': show_historical,
                'generate_forecast': generate_forecast,
                'refresh_data': refresh_data
            }
    
    def render_header(self):
        """Render the main header"""
        st.markdown('<h1 class="main-header">📈 Sales Forecasting & Demand Planning</h1>', unsafe_allow_html=True)
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stores", "45" if self.features_df is not None else "N/A")
        
        with col2:
            st.metric("Total Departments", "81" if self.features_df is not None else "N/A")
        
        with col3:
            st.metric("Data Updated", datetime.now().strftime("%Y-%m-%d"))
        
        with col4:
            if self.predictions_df is not None:
                best_mape = min([
                    self.predictions_df[f'{model}_Prediction'].corr(self.predictions_df['Actual'])
                    for model in ['Prophet', 'ARIMA', 'SARIMA']
                    if f'{model}_Prediction' in self.predictions_df.columns
                ])
                st.metric("Best Model MAPE", f"{best_mape:.1f}%")
            else:
                st.metric("Model Status", "Not Trained")
    
    def render_historical_view(self, store_id, dept_id):
        """Render historical sales view"""
        st.markdown('<h2 class="sub-header">📋 Historical Sales</h2>', unsafe_allow_html=True)
        
        if self.features_df is not None:
            # Filter data
            mask = (self.features_df['Store'] == store_id) & (self.features_df['Dept'] == dept_id)
            historical_data = self.features_df[mask].copy().sort_values('Date')
            
            if len(historical_data) > 0:
                # Create time series plot
                fig = go.Figure()
                
                # Add actual sales
                fig.add_trace(go.Scatter(
                    x=historical_data['Date'],
                    y=historical_data['Weekly_Sales'],
                    mode='lines',
                    name='Actual Sales',
                    line=dict(color='#2E86AB', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(46, 134, 171, 0.1)'
                ))
                
                # Add holiday markers
                if 'IsHoliday' in historical_data.columns:
                    holidays = historical_data[historical_data['IsHoliday'] == 1]
                    if len(holidays) > 0:
                        fig.add_trace(go.Scatter(
                            x=holidays['Date'],
                            y=holidays['Weekly_Sales'],
                            mode='markers',
                            name='Holiday',
                            marker=dict(color='#F18F01', size=10, symbol='star'),
                            hovertemplate='Holiday Week<br>Sales: $%{y:,.0f}<extra></extra>'
                        ))
                
                # Update layout
                fig.update_layout(
                    title=f'Historical Weekly Sales - Store {store_id}, Department {dept_id}',
                    xaxis_title='Date',
                    yaxis_title='Weekly Sales ($)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Average Sales",
                        f"${historical_data['Weekly_Sales'].mean():,.0f}",
                        delta=f"{historical_data['Weekly_Sales'].pct_change().mean()*100:.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Total Sales",
                        f"${historical_data['Weekly_Sales'].sum():,.0f}"
                    )
                
                with col3:
                    st.metric(
                        "Best Week",
                        f"${historical_data['Weekly_Sales'].max():,.0f}",
                        delta_date=historical_data.loc[historical_data['Weekly_Sales'].idxmax(), 'Date'].strftime('%Y-%m-%d')
                    )
                
                with col4:
                    holiday_sales = historical_data[historical_data['IsHoliday'] == 1]['Weekly_Sales'].mean() if 'IsHoliday' in historical_data.columns else 0
                    non_holiday_sales = historical_data[historical_data['IsHoliday'] == 0]['Weekly_Sales'].mean() if 'IsHoliday' in historical_data.columns else historical_data['Weekly_Sales'].mean()
                    if non_holiday_sales > 0:
                        holiday_boost = ((holiday_sales - non_holiday_sales) / non_holiday_sales) * 100
                        st.metric(
                            "Holiday Boost",
                            f"{holiday_boost:.1f}%"
                        )
                    else:
                        st.metric("Holiday Boost", "N/A")
            
            else:
                st.warning(f"No historical data found for Store {store_id}, Department {dept_id}")
        else:
            st.info("Historical data not loaded. Run the data pipeline first.")
    
    def render_forecast_view(self, store_id, dept_id, forecast_horizon, model_type, include_confidence):
        """Render forecast view"""
        st.markdown('<h2 class="sub-header">🔮 Sales Forecast</h2>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Forecast Chart", "📊 Forecast Table", "📋 Model Comparison"])
        
        with tab1:
            self._render_forecast_chart(store_id, dept_id, forecast_horizon, model_type, include_confidence)
        
        with tab2:
            self._render_forecast_table(store_id, dept_id, forecast_horizon, model_type)
        
        with tab3:
            self._render_model_comparison(store_id, dept_id)
    
    def _render_forecast_chart(self, store_id, dept_id, forecast_horizon, model_type, include_confidence):
        """Render forecast chart"""
        try:
            # Call API to get forecast
            forecast_data = self._get_forecast_from_api(
                store_id, dept_id, forecast_horizon, model_type, include_confidence
            )
            
            if forecast_data:
                # Convert to DataFrame
                forecast_df = pd.DataFrame(forecast_data['forecasts'])
                forecast_df['date'] = pd.to_datetime(forecast_df['date'])
                
                # Get historical data for context
                if self.features_df is not None:
                    mask = (self.features_df['Store'] == store_id) & (self.features_df['Dept'] == dept_id)
                    historical_data = self.features_df[mask].copy().sort_values('Date')
                    
                    # Create figure
                    fig = go.Figure()
                    
                    # Add historical data (last 26 weeks for context)
                    if len(historical_data) > 0:
                        recent_history = historical_data.tail(26)
                        fig.add_trace(go.Scatter(
                            x=recent_history['Date'],
                            y=recent_history['Weekly_Sales'],
                            mode='lines',
                            name='Historical Sales',
                            line=dict(color='#2E86AB', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(46, 134, 171, 0.1)'
                        ))
                    
                    # Add forecast
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['forecast'],
                        mode='lines',
                        name='Forecast',
                        line=dict(color='#F18F01', width=3, dash='dash')
                    ))
                    
                    # Add confidence intervals
                    if include_confidence and 'confidence_interval' in forecast_df.columns:
                        forecast_df['lower'] = forecast_df['confidence_interval'].apply(lambda x: x['lower_bound'] if x else None)
                        forecast_df['upper'] = forecast_df['confidence_interval'].apply(lambda x: x['upper_bound'] if x else None)
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                            y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(241, 143, 1, 0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo='skip',
                            showlegend=True,
                            name='Confidence Interval'
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f'{forecast_horizon}-Week Sales Forecast - {model_type.title()} Model',
                        xaxis_title='Date',
                        yaxis_title='Weekly Sales ($)',
                        hovermode='x unified',
                        template='plotly_white',
                        height=500,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show forecast metrics
                    self._render_forecast_metrics(forecast_df, forecast_data.get('metrics', {}))
                else:
                    st.warning("Historical data not available for context")
            else:
                st.error("Failed to generate forecast. Please check if the API is running.")
        
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
            st.info("Make sure the API server is running: `python src/api/app.py`")
    
    def _render_forecast_table(self, store_id, dept_id, forecast_horizon, model_type):
        """Render forecast table"""
        try:
            # Call API to get forecast
            forecast_data = self._get_forecast_from_api(
                store_id, dept_id, forecast_horizon, model_type, True
            )
            
            if forecast_data:
                forecast_df = pd.DataFrame(forecast_data['forecasts'])
                
                # Format the table
                display_df = forecast_df.copy()
                display_df['date'] = pd.to_datetime(display_df['date']).dt.strftime('%Y-%m-%d')
                display_df['forecast'] = display_df['forecast'].apply(lambda x: f"${x:,.0f}")
                
                if 'confidence_interval' in display_df.columns:
                    display_df['confidence_interval'] = display_df['confidence_interval'].apply(
                        lambda x: f"${x['lower_bound']:,.0f} - ${x['upper_bound']:,.0f}" if x else "N/A"
                    )
                
                display_df.columns = ['Date', 'Forecast', 'Week Ahead', 'Confidence Interval']
                display_df = display_df[['Date', 'Week Ahead', 'Forecast', 'Confidence Interval']]
                
                # Display table
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast",
                    data=csv,
                    file_name=f"forecast_store_{store_id}_dept_{dept_id}_{model_type}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No forecast data available")
        
        except Exception as e:
            st.error(f"Error loading forecast table: {e}")
    
    def _render_model_comparison(self, store_id, dept_id):
        """Render model comparison"""
        if self.predictions_df is not None:
            # Calculate metrics for each model
            models = []
            metrics_data = []
            
            for model in ['Prophet', 'ARIMA', 'SARIMA']:
                pred_col = f'{model}_Prediction'
                if pred_col in self.predictions_df.columns:
                    actual = self.predictions_df['Actual']
                    predicted = self.predictions_df[pred_col]
                    
                    # Calculate metrics
                    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
                    mae = np.mean(np.abs(actual - predicted))
                    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                    bias = np.mean(predicted - actual)
                    
                    models.append(model)
                    metrics_data.append({
                        'Model': model,
                        'MAPE (%)': mape,
                        'MAE ($)': mae,
                        'RMSE ($)': rmse,
                        'Bias ($)': bias
                    })
            
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)
                
                # Display metrics table
                st.dataframe(
                    metrics_df.style.format({
                        'MAPE (%)': '{:.1f}',
                        'MAE ($)': '{:,.0f}',
                        'RMSE ($)': '{:,.0f}',
                        'Bias ($)': '{:,.0f}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Create comparison chart
                fig = go.Figure()
                
                for i, model in enumerate(models):
                    pred_col = f'{model}_Prediction'
                    fig.add_trace(go.Scatter(
                        x=self.predictions_df['Date'],
                        y=self.predictions_df[pred_col],
                        mode='lines',
                        name=model,
                        line=dict(width=2)
                    ))
                
                # Add actuals
                fig.add_trace(go.Scatter(
                    x=self.predictions_df['Date'],
                    y=self.predictions_df['Actual'],
                    mode='lines',
                    name='Actual',
                    line=dict(color='black', width=3)
                ))
                
                fig.update_layout(
                    title='Model Comparison - Predictions vs Actual',
                    xaxis_title='Date',
                    yaxis_title='Weekly Sales ($)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No model predictions available for comparison")
        else:
            st.info("Model predictions not loaded. Train models first.")
    
    def _render_forecast_metrics(self, forecast_df, metrics):
        """Render forecast metrics"""
        st.markdown("#### 📊 Forecast Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Forecasted",
                f"${metrics.get('total_forecast', forecast_df['forecast'].sum()):,.0f}",
                delta=f"{len(forecast_df)} weeks"
            )
        
        with col2:
            st.metric(
                "Average Weekly",
                f"${metrics.get('mean_forecast', forecast_df['forecast'].mean()):,.0f}"
            )
        
        with col3:
            growth = ((forecast_df['forecast'].iloc[-4:].mean() - forecast_df['forecast'].iloc[:4].mean()) / 
                     forecast_df['forecast'].iloc[:4].mean() * 100)
            st.metric(
                "Projected Growth",
                f"{growth:.1f}%"
            )
        
        with col4:
            if 'confidence_interval' in forecast_df.columns:
                forecast_df['range'] = forecast_df['confidence_interval'].apply(
                    lambda x: x['upper_bound'] - x['lower_bound'] if x else 0
                )
                avg_range = forecast_df['range'].mean()
                st.metric(
                    "Avg Uncertainty",
                    f"${avg_range:,.0f}"
                )
            else:
                st.metric("Uncertainty", "N/A")
    
    def render_error_analysis(self, store_id, dept_id):
        """Render error analysis view"""
        st.markdown('<h2 class="sub-header">📉 Error Analysis</h2>', unsafe_allow_html=True)
        
        if self.predictions_df is not None:
            # Calculate errors for Prophet (assuming it's the best model)
            if 'Prophet_Prediction' in self.predictions_df.columns:
                actual = self.predictions_df['Actual']
                predicted = self.predictions_df['Prophet_Prediction']
                errors = predicted - actual
                pct_errors = (errors / actual) * 100
                
                # Create tabs for different analyses
                tab1, tab2, tab3 = st.tabs(["Error Distribution", "Temporal Analysis", "Root Cause"])
                
                with tab1:
                    # Error distribution
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.histogram(
                            x=errors,
                            nbins=30,
                            title='Error Distribution',
                            labels={'x': 'Forecast Error ($)', 'y': 'Frequency'},
                            color_discrete_sequence=['#2E86AB']
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="red")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Error statistics
                        st.metric("Mean Error", f"${errors.mean():,.0f}")
                        st.metric("Std Deviation", f"${errors.std():,.0f}")
                        st.metric("MAPE", f"{np.mean(np.abs(pct_errors)):.1f}%")
                        st.metric("Direction Accuracy", 
                                 f"{(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))).mean()*100:.1f}%")
                
                with tab2:
                    # Temporal error analysis
                    self.predictions_df['Error'] = errors
                    self.predictions_df['Month'] = self.predictions_df['Date'].dt.month
                    
                    monthly_error = self.predictions_df.groupby('Month')['Error'].mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=monthly_error.index,
                        y=monthly_error.values,
                        name='Mean Error',
                        marker_color='#F18F01'
                    ))
                    
                    fig.update_layout(
                        title='Monthly Error Pattern',
                        xaxis_title='Month',
                        yaxis_title='Mean Error ($)',
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    # Root cause analysis
                    st.info("Root cause analysis would identify patterns in forecast errors")
                    
                    # Example insights
                    insights = []
                    
                    # Check for bias
                    if errors.mean() > actual.mean() * 0.1:
                        insights.append("⚠️ Model shows systematic over-forecasting bias")
                    
                    # Check holiday performance
                    if 'IsHoliday' in self.predictions_df.columns:
                        holiday_errors = self.predictions_df[self.predictions_df['IsHoliday'] == 1]['Error']
                        if len(holiday_errors) > 0 and abs(holiday_errors.mean()) > errors.mean() * 1.5:
                            insights.append("🎯 Holiday forecasts less accurate - consider separate model")
                    
                    # Check error trend
                    if len(errors) > 10:
                        error_trend = np.polyfit(range(len(errors)), errors, 1)[0]
                        if abs(error_trend) > errors.std() * 0.1:
                            insights.append("📈 Error shows time trend - model may need retraining")
                    
                    for insight in insights:
                        st.write(f"- {insight}")
            else:
                st.info("Prophet predictions not available for error analysis")
        else:
            st.info("Error analysis requires model predictions. Train models first.")
    
    def render_business_impact(self, store_id, dept_id):
        """Render business impact analysis"""
        st.markdown('<h2 class="sub-header">💰 Business Impact</h2>', unsafe_allow_html=True)
        
        # Create simulation controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            service_level = st.slider(
                "Service Level Target",
                min_value=0.85,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Desired probability of not having stockouts"
            )
        
        with col2:
            holding_cost = st.slider(
                "Holding Cost Rate",
                min_value=0.1,
                max_value=0.5,
                value=0.25,
                step=0.05,
                help="Annual cost to hold inventory as % of value"
            )
        
        with col3:
            stockout_cost = st.slider(
                "Stockout Cost Multiplier",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                help="Multiplier for lost sales during stockouts"
            )
        
        # Simulate impact
        if self.predictions_df is not None and 'Prophet_Prediction' in self.predictions_df.columns:
            # Calculate forecast error statistics
            actual = self.predictions_df['Actual']
            predicted = self.predictions_df['Prophet_Prediction']
            errors = predicted - actual
            
            mean_error = errors.mean()
            std_error = errors.std()
            mape = np.mean(np.abs(errors / actual)) * 100
            
            # Simple safety stock calculation
            from scipy import stats
            z_score = stats.norm.ppf(service_level)
            lead_time = 2  # weeks
            
            safety_stock_current = z_score * std_error * np.sqrt(lead_time)
            safety_stock_improved = safety_stock_current * 0.8  # Assume 20% improvement
            
            # Calculate costs
            avg_weekly_sales = actual.mean()
            holding_cost_current = safety_stock_current * holding_cost / 52
            holding_cost_improved = safety_stock_improved * holding_cost / 52
            
            # Display results
            st.markdown("#### Inventory Optimization Impact")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Current Safety Stock",
                    f"${safety_stock_current:,.0f}",
                    delta="Based on current forecast accuracy"
                )
            
            with col2:
                st.metric(
                    "Potential Reduction",
                    f"${safety_stock_current - safety_stock_improved:,.0f}",
                    delta="With 20% accuracy improvement"
                )
            
            with col3:
                annual_savings = (holding_cost_current - holding_cost_improved) * 52
                st.metric(
                    "Annual Savings Potential",
                    f"${annual_savings:,.0f}"
                )
            
            # Recommendations
            st.markdown("#### 🎯 Recommendations")
            
            recommendations = [
                f"**Safety Stock**: Maintain ${safety_stock_current:,.0f} safety stock at {service_level:.0%} service level",
                f"**Accuracy Target**: Reduce MAPE from {mape:.1f}% to {mape*0.8:.1f}% for inventory savings",
                f"**Monitoring**: Track forecast accuracy weekly, retrain monthly",
                f"**Action**: Implement forecast-based reorder points"
            ]
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
        else:
            st.info("Business impact analysis requires forecast error data")
    
    def _get_forecast_from_api(self, store_id, dept_id, periods, model_type, include_confidence):
        """Get forecast from API"""
        try:
            # In a real implementation, this would call the API
            # For demo purposes, we'll create sample data
            
            # Check if API is available
            try:
                response = requests.get(f"{self.api_url}/health", timeout=2)
                if response.status_code == 200:
                    # API is available, make real request
                    payload = {
                        "store_id": store_id,
                        "dept_id": dept_id,
                        "periods": periods,
                        "model_type": model_type,
                        "include_confidence": include_confidence
                    }
                    
                    response = requests.post(
                        f"{self.api_url}/forecast/generate",
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        return response.json()
            except:
                pass  # API not available, use sample data
            
            # Generate sample forecast data
            base_date = datetime.now() + timedelta(days=7)
            forecasts = []
            
            # Get baseline from historical data if available
            if self.features_df is not None:
                mask = (self.features_df['Store'] == store_id) & (self.features_df['Dept'] == dept_id)
                historical = self.features_df[mask]
                if len(historical) > 0:
                    baseline = historical['Weekly_Sales'].mean()
                else:
                    baseline = 20000
            else:
                baseline = 20000
            
            # Add seasonal pattern
            for i in range(periods):
                date = base_date + timedelta(days=7*i)
                
                # Simple seasonal pattern
                month = date.month
                if month in [11, 12]:  # Holiday season
                    forecast = baseline * 1.3
                elif month in [6, 7, 8]:  # Summer
                    forecast = baseline * 0.9
                else:
                    forecast = baseline
                
                # Add some randomness
                forecast = forecast * np.random.uniform(0.95, 1.05)
                
                forecast_item = {
                    "date": date.isoformat(),
                    "forecast": float(forecast),
                    "horizon": i + 1
                }
                
                if include_confidence:
                    forecast_item["confidence_interval"] = {
                        "lower_bound": float(forecast * 0.85),
                        "upper_bound": float(forecast * 1.15)
                    }
                
                forecasts.append(forecast_item)
            
            metrics = {
                "total_forecast": sum(f["forecast"] for f in forecasts),
                "mean_forecast": np.mean([f["forecast"] for f in forecasts]),
                "forecast_periods": periods
            }
            
            return {
                "store_id": store_id,
                "dept_id": dept_id,
                "model_type": model_type,
                "forecast_horizon": periods,
                "forecasts": forecasts,
                "metrics": metrics
            }
            
        except Exception as e:
            st.error(f"Error calling API: {e}")
            return None
    
    def run(self):
        """Run the dashboard"""
        # Load data
        self.load_data()
        
        # Render header
        self.render_header()
        
        # Render sidebar and get controls
        controls = self.render_sidebar()
        
        # Store selection in session state for persistence
        if 'store_id' not in st.session_state:
            st.session_state.store_id = controls['store_id']
            st.session_state.dept_id = controls['dept_id']
        
        # Update if controls changed
        if (controls['store_id'] != st.session_state.store_id or 
            controls['dept_id'] != st.session_state.dept_id):
            st.session_state.store_id = controls['store_id']
            st.session_state.dept_id = controls['dept_id']
            st.rerun()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Historical View", 
            "🔮 Forecast", 
            "📉 Error Analysis", 
            "💰 Business Impact"
        ])
        
        with tab1:
            self.render_historical_view(
                st.session_state.store_id,
                st.session_state.dept_id
            )
        
        with tab2:
            if controls['generate_forecast'] or st.session_state.get('auto_generate', False):
                self.render_forecast_view(
                    st.session_state.store_id,
                    st.session_state.dept_id,
                    controls['forecast_horizon'],
                    controls['model_type'],
                    controls['include_confidence']
                )
                st.session_state.auto_generate = True
            else:
                st.info("Click 'Generate Forecast' in the sidebar to see predictions")
        
        with tab3:
            self.render_error_analysis(
                st.session_state.store_id,
                st.session_state.dept_id
            )
        
        with tab4:
            self.render_business_impact(
                st.session_state.store_id,
                st.session_state.dept_id
            )
        
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666;'>
            <p>Sales Forecasting Dashboard v1.0 | Built with ❤️ using Streamlit</p>
            <p>Data updated: {}</p>
            </div>
            """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            unsafe_allow_html=True
        )

# Run the dashboard
if __name__ == "__main__":
    dashboard = ForecastingDashboard()
    dashboard.run()