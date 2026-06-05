# ── Smart Urban Energy Efficiency System v2 ──────────────────────────────────
# Optimized deployment version — loads compressed parquet files
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Smart Urban EE System — Bangalore",
    page_icon="⚡", layout="wide")

st.markdown("""
<style>
    .main-title {
        font-size:2.2rem; font-weight:800;
        color:#1a1a2e; text-align:center; padding:10px;
    }
    .sub-title {
        font-size:1rem; color:#666;
        text-align:center; margin-bottom:20px;
    }
    .alert-critical {
        background-color:#ffe0e0;
        border-left:5px solid #c0392b;
        padding:10px 15px; border-radius:5px; margin:5px 0;
    }
    .alert-high {
        background-color:#fff3e0;
        border-left:5px solid #e67e22;
        padding:10px 15px; border-radius:5px; margin:5px 0;
    }
    .alert-normal {
        background-color:#e8f5e9;
        border-left:5px solid #2ecc71;
        padding:10px 15px; border-radius:5px; margin:5px 0;
    }
</style>""", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    df_elec      = pd.read_parquet(os.path.join(BASE,"Deploy","electricity.parquet"))
    df_water     = pd.read_parquet(os.path.join(BASE,"Deploy","water.parquet"))
    elec_report  = pd.read_parquet(os.path.join(BASE,"Deploy","elec_report.parquet"))
    water_report = pd.read_parquet(os.path.join(BASE,"Deploy","water_report.parquet"))
    df_elec['timestamp']  = pd.to_datetime(df_elec['timestamp'])
    df_water['timestamp'] = pd.to_datetime(df_water['timestamp'])
    return df_elec, df_water, elec_report, water_report

df_elec, df_water, elec_report, water_report = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/"
    "Flag_of_Karnataka.svg/320px-Flag_of_Karnataka.svg.png",
    width=120)
st.sidebar.title("⚡ Smart Urban EE")
st.sidebar.markdown("**Bangalore Urban Zones**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",[
    "🏠 Overview","🔍 Zone Monitor",
    "🧪 Manual Check","📋 Reports","📊 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Mini Project — BE")
st.sidebar.markdown("**City:** Bangalore, India")
st.sidebar.markdown("**Data:** 2022")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">⚡ Smart Urban Energy Efficiency System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Bangalore Urban Zone Monitoring — 2022</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🏙️ Zones",          "8")
    c2.metric("🗺️ Wards",          "24")
    c3.metric("🔲 Blocks",         "96")
    c4.metric("⚡ Elec Anomalies",  f"{df_elec['combined_anomaly'].sum():,}")
    c5.metric("💧 Water Anomalies", f"{df_water['combined_anomaly'].sum():,}")
    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("⚡ Electricity Anomaly Rate by Zone")
        ez = (df_elec.groupby('zone')['combined_anomaly']
              .mean().mul(100).reset_index())
        ez.columns = ['Zone','Anomaly Rate (%)']
        fig = px.bar(ez, x='Zone', y='Anomaly Rate (%)',
                     color='Anomaly Rate (%)',
                     color_continuous_scale='Reds', text_auto='.2f')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💧 Water Anomaly Rate by Zone")
        wz = (df_water.groupby('zone')['combined_anomaly']
              .mean().mul(100).reset_index())
        wz.columns = ['Zone','Anomaly Rate (%)']
        fig = px.bar(wz, x='Zone', y='Anomaly Rate (%)',
                     color='Anomaly Rate (%)',
                     color_continuous_scale='Blues', text_auto='.2f')
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    col3,col4 = st.columns(2)
    with col3:
        st.subheader("⚡ Electricity Severity Breakdown")
        se = (df_elec[df_elec['combined_anomaly']==1]
              ['severity'].value_counts().reset_index())
        se.columns = ['Severity','Count']
        fig = px.pie(se, names='Severity', values='Count',
                     color='Severity',
                     color_discrete_map={'Critical':'#c0392b','High':'#e67e22',
                                         'Medium':'#f1c40f','Low':'#2ecc71'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("💧 Water Severity Breakdown")
        sw = (df_water[df_water['combined_anomaly']==1]
              ['severity'].value_counts().reset_index())
        sw.columns = ['Severity','Count']
        fig = px.pie(sw, names='Severity', values='Count',
                     color='Severity',
                     color_discrete_map={'Critical':'#c0392b','High':'#e67e22',
                                         'Medium':'#f1c40f','Low':'#2ecc71'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ZONE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Zone Monitor":
    st.title("🔍 Zone Level Anomaly Monitor")
    st.markdown("Select a location to view its consumption and anomalies.")
    st.markdown("---")
    col1,col2,col3,col4 = st.columns(4)
    with col1:
        resource = st.selectbox("Resource",["⚡ Electricity","💧 Water"])
    with col2:
        zone  = st.selectbox("Zone", sorted(df_elec['zone'].unique()))
    with col3:
        ward  = st.selectbox("Ward",["Ward 1","Ward 2","Ward 3"])
    with col4:
        block = st.selectbox("Block",["Block A","Block B","Block C","Block D"])
    df    = df_elec if resource=="⚡ Electricity" else df_water
    v_col = 'electricity_kwh' if resource=="⚡ Electricity" else 'water_kl'
    loc_id= f"{zone} | {ward} | {block}"
    loc_df= df[df['location_id']==loc_id].copy()
    st.markdown("---")
    col5,col6 = st.columns(2)
    with col5:
        start = st.date_input("Start Date", value=pd.to_datetime("2022-01-01"))
    with col6:
        end   = st.date_input("End Date",   value=pd.to_datetime("2022-01-07"))
    mask   = ((loc_df['timestamp']>=pd.to_datetime(start)) &
              (loc_df['timestamp']<=pd.to_datetime(end)))
    sample = loc_df[mask]
    if sample.empty:
        st.warning("No data found for selected date range.")
    else:
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("📍 Location",      f"{zone} → {ward} → {block}")
        k2.metric("📊 Total Readings", f"{len(sample):,}")
        k3.metric("⚠️ Anomalies",      f"{sample['combined_anomaly'].sum():,}")
        k4.metric("🔴 Critical",       f"{(sample['severity']=='Critical').sum():,}")
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample['timestamp'], y=sample[v_col],
            mode='lines', name='Consumption',
            line=dict(color='steelblue',width=1.5)))
        anom = sample[sample['combined_anomaly']==1]
        fig.add_trace(go.Scatter(
            x=anom['timestamp'], y=anom[v_col],
            mode='markers', name='Anomaly',
            marker=dict(color='red',size=8,symbol='x')))
        fig.update_layout(
            title=f"{resource} Consumption — {loc_id}",
            xaxis_title='Timestamp',
            yaxis_title='kWh' if resource=="⚡ Electricity" else 'kL',
            height=400)
        st.plotly_chart(fig, use_container_width=True)
        if anom.empty:
            st.success("✅ No anomalies in selected date range!")
        else:
            st.markdown("### ⚠️ Detected Anomalies")
            st.dataframe(
                anom[['timestamp','severity','confidence',
                      'suggested_cause','deviation_pct']]
                .sort_values('severity').reset_index(drop=True),
                use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MANUAL CHECK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧪 Manual Check":
    st.title("🧪 Manual Reading Check")
    st.markdown("Enter a meter reading to instantly check if it is normal or anomalous.")
    st.markdown("---")
    col1,col2,col3 = st.columns(3)
    with col1:
        m_resource = st.selectbox("Resource",["⚡ Electricity","💧 Water"])
    with col2:
        m_zone = st.selectbox("Zone", sorted(df_elec['zone'].unique()))
    with col3:
        m_ward = st.selectbox("Ward",["Ward 1","Ward 2","Ward 3"])
    col4,col5 = st.columns(2)
    with col4:
        m_block = st.selectbox("Block",["Block A","Block B","Block C","Block D"])
    with col5:
        m_hour  = st.slider("Hour of Day", 0, 23, 12)
    m_loc  = f"{m_zone} | {m_ward} | {m_block}"
    df_ref = df_elec if m_resource=="⚡ Electricity" else df_water
    v_col  = 'electricity_kwh' if m_resource=="⚡ Electricity" else 'water_kl'
    unit   = "kWh" if m_resource=="⚡ Electricity" else "kL"
    ref      = df_ref[(df_ref['location_id']==m_loc)&(df_ref['hour']==m_hour)]
    ref_mean = ref[v_col].mean()
    ref_std  = ref[v_col].std()
    ref_min  = ref[v_col].min()
    ref_max  = ref[v_col].max()
    st.markdown("---")
    st.markdown(f"### 📍 Reference Stats for {m_loc} at {m_hour}:00 hrs")
    r1,r2,r3,r4 = st.columns(4)
    r1.metric("📊 Normal Mean", f"{ref_mean:.1f} {unit}")
    r2.metric("📉 Normal Min",  f"{ref_min:.1f} {unit}")
    r3.metric("📈 Normal Max",  f"{ref_max:.1f} {unit}")
    r4.metric("📐 Std Dev",     f"{ref_std:.1f} {unit}")
    st.markdown("---")
    m_value = st.number_input(
        f"Enter meter reading ({unit})",
        min_value=0.0, max_value=10000.0,
        value=float(round(ref_mean,1)), step=0.1)
    if st.button("🔍 Check Reading", type="primary"):
        zscore     = abs((m_value-ref_mean)/ref_std) if ref_std>0 else 0
        dev_pct    = ((m_value-ref_mean)/ref_mean*100) if ref_mean>0 else 0
        is_anomaly = zscore > 2.5
        st.markdown("---")
        st.markdown("## 🔎 Result")
        if not is_anomaly:
            st.markdown(
                f'<div class="alert-normal"><h3>✅ NORMAL</h3>'
                f'<p>Reading of <b>{m_value} {unit}</b> is within '
                f'expected range for {m_loc} at {m_hour}:00 hrs.</p>'
                f'<p>Z-Score: {zscore:.2f} | Deviation: {dev_pct:.1f}%</p>'
                f'</div>', unsafe_allow_html=True)
        else:
            sev   = ("🔴 CRITICAL" if abs(dev_pct)>150
                     else "🟠 HIGH" if abs(dev_pct)>80 else "🟡 MEDIUM")
            color = ("alert-critical" if abs(dev_pct)>80 else "alert-high")
            if m_resource=="⚡ Electricity":
                cause = ("Illegal tapping / Faulty equipment" if dev_pct>150
                         else "Streetlight malfunction" if dev_pct>80 and m_hour>=22
                         else "Sustained overload" if dev_pct>80
                         else "Power outage / Sensor fault")
            else:
                cause = ("Pipeline burst / Major leakage" if dev_pct>150
                         else "Tap left open / Tank overflow" if dev_pct>80 and 5<=m_hour<8
                         else "Pipeline leakage" if dev_pct>80
                         else "Supply disruption / Meter fault")
            st.markdown(
                f'<div class="{color}"><h3>⚠️ ANOMALY DETECTED — {sev}</h3>'
                f'<p><b>Location:</b> {m_loc}</p>'
                f'<p><b>Reading:</b> {m_value} {unit} '
                f'(Normal: {ref_mean:.1f} {unit})</p>'
                f'<p><b>Deviation:</b> {dev_pct:.1f}% | '
                f'Z-Score: {zscore:.2f}</p>'
                f'<p><b>Suggested Cause:</b> {cause}</p>'
                f'<p><b>Action:</b> Send technician to {m_loc} immediately.</p>'
                f'</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — REPORTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Reports":
    st.title("📋 Anomaly Reports")
    st.markdown("Zone → Ward → Block level anomaly summary.")
    st.markdown("---")
    tab1,tab2 = st.tabs(["⚡ Electricity Report","💧 Water Report"])
    with tab1:
        st.subheader("⚡ Electricity Anomaly Report")
        st.dataframe(elec_report, use_container_width=True)
        st.download_button("📥 Download",
            elec_report.to_csv(index=False),
            "electricity_report.csv","text/csv")
    with tab2:
        st.subheader("💧 Water Anomaly Report")
        st.dataframe(water_report, use_container_width=True)
        st.download_button("📥 Download",
            water_report.to_csv(index=False),
            "water_report.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Comparison")
    st.markdown("Z-Score vs Isolation Forest detection results.")
    st.markdown("---")
    metrics = ['Accuracy','Precision','Recall','F1 Score']
    def get_scores(df, pred_col):
        return [
            accuracy_score(df['is_anomaly'],df[pred_col])*100,
            precision_score(df['is_anomaly'],df[pred_col])*100,
            recall_score(df['is_anomaly'],df[pred_col])*100,
            f1_score(df['is_anomaly'],df[pred_col])*100]
    ze = get_scores(df_elec, 'zscore_anomaly')
    ie = get_scores(df_elec, 'if_anomaly')
    zw = get_scores(df_water,'zscore_anomaly')
    iw = get_scores(df_water,'if_anomaly')
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("⚡ Electricity")
        st.dataframe(pd.DataFrame({
            'Metric':metrics,
            'Z-Score':[f"{v:.2f}%" for v in ze],
            'Isolation Forest':[f"{v:.2f}%" for v in ie]}),
            use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Z-Score',x=metrics,y=ze,
                             marker_color='steelblue'))
        fig.add_trace(go.Bar(name='Isolation Forest',x=metrics,y=ie,
                             marker_color='tomato'))
        fig.update_layout(barmode='group',height=350,
                          yaxis=dict(range=[80,100]))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("💧 Water")
        st.dataframe(pd.DataFrame({
            'Metric':metrics,
            'Z-Score':[f"{v:.2f}%" for v in zw],
            'Isolation Forest':[f"{v:.2f}%" for v in iw]}),
            use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='Z-Score',x=metrics,y=zw,
                              marker_color='steelblue'))
        fig2.add_trace(go.Bar(name='Isolation Forest',x=metrics,y=iw,
                              marker_color='tomato'))
        fig2.update_layout(barmode='group',height=350,
                           yaxis=dict(range=[80,100]))
        st.plotly_chart(fig2, use_container_width=True)