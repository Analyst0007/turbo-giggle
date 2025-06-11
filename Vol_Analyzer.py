# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 22:50:51 2025

@author: Hemal
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


# Main VolatilityAnalyzer class
class VolatilityAnalyzer:
    def __init__(self, symbol, period="1mo", interval="1h"):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.data = None
        self.volatility_by_hour = None

    def fetch_data(self):
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period, interval=self.interval)
            if self.data.empty:
                raise ValueError(f"No data found for {self.symbol}")
            self.data['Hour'] = self.data.index.hour
            return True
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return False

    def calculate_volatility(self, method='returns'):
        if self.data is None:
            return

        if method == 'returns':
            self.data['Returns'] = self.data['Close'].pct_change()
            self.volatility_by_hour = self.data.groupby('Hour')['Returns'].agg(['std', 'mean', 'count'])
            self.volatility_by_hour.columns = ['Volatility', 'Avg_Return', 'Sample_Size']
        elif method == 'range':
            self.data['Range_Pct'] = ((self.data['High'] - self.data['Low']) / self.data['Close']) * 100
            self.volatility_by_hour = self.data.groupby('Hour')['Range_Pct'].agg(['mean', 'std', 'count'])
            self.volatility_by_hour.columns = ['Volatility', 'Volatility_Std', 'Sample_Size']

        self.volatility_by_hour = self.volatility_by_hour.dropna()
        self.volatility_by_hour = self.volatility_by_hour.sort_values('Volatility', ascending=False)

    def get_extreme_volatility_hours(self, top_n=3):
        if self.volatility_by_hour is None:
            return None, None
        return (self.volatility_by_hour.head(top_n), self.volatility_by_hour.tail(top_n))

    def plot_volatility(self):
        if self.volatility_by_hour is None:
            st.warning("No volatility data to plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'Volatility Analysis for {self.symbol}', fontsize=16, fontweight='bold')

        hours = self.volatility_by_hour.index
        volatilities = self.volatility_by_hour['Volatility']
        bars = axes[0, 0].bar(hours, volatilities, color='steelblue', alpha=0.7)
        axes[0, 0].set_title('Volatility by Hour')
        axes[0, 0].set_xticks(hours)

        max_idx = volatilities.idxmax()
        min_idx = volatilities.idxmin()
        bars[list(hours).index(max_idx)].set_color('red')
        bars[list(hours).index(min_idx)].set_color('green')

        axes[0, 1].plot(hours, volatilities, marker='o', linewidth=2)
        axes[0, 1].set_title('Volatility Trend')

        axes[1, 0].bar(hours, self.volatility_by_hour['Sample_Size'], color='orange', alpha=0.7)
        axes[1, 0].set_title('Sample Size by Hour')

        vol_matrix = volatilities.values.reshape(1, -1)
        im = axes[1, 1].imshow(vol_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[1, 1].set_title('Volatility Heatmap')
        axes[1, 1].set_xticks(range(len(hours)))
        axes[1, 1].set_xticklabels(hours)
        axes[1, 1].set_yticks([])
        plt.colorbar(im, ax=axes[1, 1], orientation='horizontal', pad=0.1)

        plt.tight_layout()
        st.pyplot(fig)

    def export_results(self):
        if self.volatility_by_hour is not None:
            export_data = self.volatility_by_hour.copy()
            export_data['Hour_12H'] = [datetime.strptime(str(h), '%H').strftime('%I %p') for h in export_data.index]
            return export_data
        else:
            return None


# Streamlit Interface
st.set_page_config(page_title="Hourly Volatility Analyzer", layout="wide")
st.title("📈 Hourly Volatility Analyzer")

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL)", "AAPL")
    period = st.selectbox("Select Period", ['5d', '1mo', '3mo', '6mo', '1y'])
    interval = st.selectbox("Select Interval", ['1h', '30m', '15m'])
    method = st.radio("Volatility Method", ['returns', 'range'])
    top_n = st.slider("Top N Volatile Hours", 1, 6, 3)

if st.button("Run Analysis"):
    analyzer = VolatilityAnalyzer(symbol, period, interval)
    if analyzer.fetch_data():
        analyzer.calculate_volatility(method=method)
        st.subheader(f"📊 Volatility Table for {symbol}")
        st.dataframe(analyzer.volatility_by_hour)

        high, low = analyzer.get_extreme_volatility_hours(top_n)
        if high is not None:
            st.write(f"🔥 Top {top_n} Highest Volatility Hours:")
            st.dataframe(high.style.highlight_max(axis=0, color='lightcoral'))

            st.write(f"😴 Top {top_n} Lowest Volatility Hours:")
            st.dataframe(low.style.highlight_min(axis=0, color='lightgreen'))

            st.markdown("## 📉 Volatility Plots")
            analyzer.plot_volatility()

            csv_data = analyzer.export_results()
            if csv_data is not None:
                st.download_button("📥 Download CSV", data=csv_data.to_csv().encode('utf-8'),
                                   file_name=f"{symbol}_volatility.csv", mime='text/csv')
