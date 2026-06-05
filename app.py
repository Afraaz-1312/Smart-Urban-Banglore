# ── Smart Urban Energy Efficiency System ─────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
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

# ── Data Generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_data():
    np.random.seed(42)
    START_DATE   = pd.Timestamp("2022-01-01")
    END_DATE = pd.Timestamp("2022-01-31")
    ANOMALY_RATE = 0.08

    ZONES = {
        "Whitefield":      {"type":"mixed",       "base_e":380,"base_w":260},
        "Koramangala":     {"type":"commercial",  "base_e":420,"base_w":240},
        "Indiranagar":     {"type":"commercial",  "base_e":400,"base_w":230},
        "Hebbal":          {"type":"residential", "base_e":300,"base_w":290},
        "Jayanagar":       {"type":"residential", "base_e":290,"base_w":300},
        "Rajajinagar":     {"type":"mixed",       "base_e":350,"base_w":270},
        "Electronic City": {"type":"industrial",  "base_e":460,"base_w":200},
        "Yeshwanthpur":    {"type":"industrial",  "base_e":440,"base_w":210},
    }
    WARDS  = ["Ward 1","Ward 2","Ward 3"]
    BLOCKS = ["Block A","Block B","Block C","Block D"]
    BLOCK_PROFILE = {
        "Block A":{"e_mult":1.10,"w_mult":0.95,"profile":"commercial"},
        "Block B":{"e_mult":0.95,"w_mult":1.10,"profile":"residential"},
        "Block C":{"e_mult":1.05,"w_mult":1.00,"profile":"mixed"},
        "Block D":{"e_mult":0.90,"w_mult":1.05,"profile":"residential"},
    }
    WARD_MULT = {"Ward 1":1.08,"Ward 2":1.00,"Ward 3":0.94}

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='h')
    records_e, records_w = [], []

    for zone_name, zone_info in ZONES.items():
        for ward in WARDS:
            for block in BLOCKS:
                location_id = f"{zone_name} | {ward} | {block}"
                profile     = BLOCK_PROFILE[block]["profile"]
                base_e = (zone_info["base_e"] * WARD_MULT[ward]
                          * BLOCK_PROFILE[block]["e_mult"]
                          * np.random.uniform(0.93,1.07))
                base_w = (zone_info["base_w"] * WARD_MULT[ward]
                          * BLOCK_PROFILE[block]["w_mult"]
                          * np.random.uniform(0.93,1.07))

                for ts in dates:
                    hour = ts.hour; dow = ts.dayofweek
                    month = ts.month; is_weekend = int(dow>=5)

                    if profile=="commercial":
                        hf_e = (1.35 if 9<=hour<13 else 1.25 if 13<=hour<18
                                else 1.00 if 6<=hour<9 else 0.90 if 18<=hour<22
                                else 0.60 if 22<=hour<24 else 0.40)
                    elif profile=="residential":
                        hf_e = (1.20 if 6<=hour<9 else 1.45 if 18<=hour<23
                                else 0.85 if 9<=hour<18 else 0.55)
                    else:
                        hf_e = (1.25 if 6<=hour<9 else 1.10 if 9<=hour<18
                                else 1.35 if 18<=hour<22 else 0.60)

                    we = (0.80 if (is_weekend and profile=="commercial")
                          else 1.10 if (is_weekend and profile=="residential")
                          else 1.0)
                    se = {1:0.90,2:0.92,3:1.05,4:1.15,5:1.20,6:1.10,
                          7:0.95,8:0.93,9:0.97,10:1.00,11:0.95,12:0.88}[month]
                    e_val = max(8, base_e*hf_e*we*se
                                + np.random.normal(0, base_e*hf_e*we*se*0.07))

                    hf_w = (1.45 if 5<=hour<8 else 1.10 if 8<=hour<11
                            else 0.85 if 11<=hour<17 else 1.35 if 17<=hour<20
                            else 1.00 if 20<=hour<23 else 0.50)
                    sw = {1:0.90,2:0.92,3:1.05,4:1.12,5:1.18,6:0.98,
                          7:0.88,8:0.85,9:0.90,10:0.95,11:0.92,12:0.88}[month]
                    w_val = max(5, base_w*hf_w*sw
                                + np.random.normal(0, base_w*hf_w*sw*0.08))

                    is_ae=0; cause_e="Normal"
                    is_aw=0; cause_w="Normal"

                    if np.random.random() < ANOMALY_RATE:
                        at = np.random.choice(
                            ["spike_high","spike_low","sustained_high"],
                            p=[0.50,0.25,0.25])
                        if at=="spike_high":
                            e_val *= np.random.uniform(2.2,3.5)
                            cause_e = np.random.choice([
                                "Faulty equipment","Illegal tapping",
                                "Industrial overconsumption",
                                "Streetlight malfunction"])
                        elif at=="spike_low":
                            e_val *= np.random.uniform(0.10,0.30)
                            cause_e = "Power outage / sensor fault"
                        else:
                            e_val *= np.random.uniform(1.5,1.9)
                            cause_e = "Sustained overload"
                        is_ae = 1

                    if np.random.random() < ANOMALY_RATE:
                        at = np.random.choice(
                            ["spike_high","sustained_high","spike_low"],
                            p=[0.45,0.35,0.20])
                        if at=="spike_high":
                            w_val *= np.random.uniform(2.0,3.2)
                            cause_w = np.random.choice([
                                "Pipeline burst","Tank overflow",
                                "Irrigation overuse","Tap left open"])
                        elif at=="sustained_high":
                            w_val *= np.random.uniform(1.5,1.85)
                            cause_w = "Gradual leakage / continuous flow"
                        else:
                            w_val *= np.random.uniform(0.08,0.25)
                            cause_w = "Supply disruption / meter fault"
                        is_aw = 1

                    row = dict(timestamp=ts, zone=zone_name, ward=ward,
                               block=block, location_id=location_id,
                               zone_type=zone_info["type"],
                               block_profile=profile,
                               hour=hour, day_of_week=dow, month=month,
                               is_weekend=is_weekend)
                    records_e.append({**row,
                        "electricity_kwh":round(e_val,2),
                        "is_anomaly":is_ae, "anomaly_cause":cause_e})
                    records_w.append({**row,
                        "water_kl":round(w_val,2),
                        "is_anomaly":is_aw, "anomaly_cause":cause_w})

    df_e = pd.DataFrame(records_e)
    df_w = pd.DataFrame(records_w)
    return df_e, df_w

# ── Feature Engineering ───────────────────────────────────────────────────────
@st.cache_data
def add_features(df, value_col):
    df = df.copy().sort_values(['location_id','timestamp']).reset_index(drop=True)
    df['rolling_mean_24h'] = (df.groupby('location_id')[value_col]
                               .transform(lambda x: x.rolling(24,min_periods=1).mean()))
    df['rolling_std_24h']  = (df.groupby('location_id')[value_col]
                               .transform(lambda x: x.rolling(24,min_periods=1).std().fillna(0)))
    df['deviation']        = df[value_col] - df['rolling_mean_24h']
    df['deviation_pct']    = (df['deviation']
                               / df['rolling_mean_24h'].replace(0,1)) * 100
    df['is_night']         = ((df['hour']>=22)|(df['hour']<5)).astype(int)
    df['is_morning_peak']  = ((df['hour']>=5) &(df['hour']<9)).astype(int)
    df['is_evening_peak']  = ((df['hour']>=17)&(df['hour']<22)).astype(int)
    return df

# ── Z-Score ───────────────────────────────────────────────────────────────────
@st.cache_data
def apply_zscore(df, value_col):
    df = df.copy()
    df['zscore'] = (df.groupby('location_id')[value_col]
                     .transform(lambda x: (x-x.mean())/x.std())).fillna(0)
    df['zscore_anomaly']   = (df['zscore'].abs()>2.5).astype(int)
    df['zscore_direction'] = 'Normal'
    df.loc[df['zscore'] >  2.5,'zscore_direction'] = 'Abnormally High'
    df.loc[df['zscore'] < -2.5,'zscore_direction'] = 'Abnormally Low'
    return df

# ── Isolation Forest ──────────────────────────────────────────────────────────
@st.cache_data
def apply_iforest(df, value_col):
    FEATURES = ['hour','day_of_week','month','is_weekend',
                'is_night','is_morning_peak','is_evening_peak',
                'rolling_mean_24h','rolling_std_24h',
                'deviation','deviation_pct']
    results = []
    for loc in df['location_id'].unique():
        loc_df = df[df['location_id']==loc].copy()
        X = loc_df[[value_col]+FEATURES].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        preds = IsolationForest(
            n_estimators=30, contamination=0.08,
            random_state=42, n_jobs=-1).fit_predict(X_scaled)
        loc_df['if_anomaly'] = (preds==-1).astype(int)
        results.append(loc_df)
    return pd.concat(results).sort_values(
        ['location_id','timestamp']).reset_index(drop=True)

# ── Combined + Severity + Cause ───────────────────────────────────────────────
@st.cache_data
def finalize(df, value_col):
    df = df.copy()
    df['combined_anomaly'] = ((df['zscore_anomaly']==1) &
                               (df['if_anomaly']==1)).astype(int)
    df['confidence'] = 'Normal'
    df.loc[(df['zscore_anomaly']==1)&(df['if_anomaly']==1),'confidence'] = 'High'
    df.loc[(df['zscore_anomaly']==0)&(df['if_anomaly']==1),'confidence'] = 'Medium'
    df.loc[(df['zscore_anomaly']==1)&(df['if_anomaly']==0),'confidence'] = 'Low'

    df['severity'] = 'Normal'
    df.loc[(df['combined_anomaly']==1)&(df['deviation_pct'].abs()>=150),'severity'] = 'Critical'
    df.loc[(df['combined_anomaly']==1)&(df['deviation_pct'].abs()>=80)
           &(df['deviation_pct'].abs()<150),'severity'] = 'High'
    df.loc[(df['combined_anomaly']==1)&(df['deviation_pct'].abs()>=40)
           &(df['deviation_pct'].abs()<80),'severity']  = 'Medium'
    df.loc[(df['combined_anomaly']==1)&(df['deviation_pct'].abs()<40),'severity'] = 'Low'

    df['suggested_cause'] = 'Normal'
    if value_col == 'electricity_kwh':
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']>150),
               'suggested_cause'] = 'Illegal tapping / Faulty equipment'
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']>80)
               &(df['is_night']==1),
               'suggested_cause'] = 'Streetlight malfunction / Faulty equipment'
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']>80)
               &(df['is_evening_peak']==1),
               'suggested_cause'] = 'Sustained overload / Industrial overconsumption'
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']<-60),
               'suggested_cause'] = 'Power outage / Sensor fault'
    else:
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']>150),
               'suggested_cause'] = 'Pipeline burst / Major leakage'
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']>80)
               &(df['is_morning_peak']==1),
               'suggested_cause'] = 'Tap left open / Tank overflow'
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']>80)
               &(df['is_night']==1),
               'suggested_cause'] = 'Pipeline leakage / Continuous flow'
        df.loc[(df['combined_anomaly']==1)&(df['deviation_pct']<-60),
               'suggested_cause'] = 'Supply disruption / Meter fault'
    return df

# ── Generate Report ───────────────────────────────────────────────────────────
@st.cache_data
def generate_report(df):
    anomalies = df[df['combined_anomaly']==1].copy()
    return (anomalies.groupby(['zone','ward','block'])
            .agg(total_anomalies =('combined_anomaly','count'),
                 critical_count  =('severity', lambda x:(x=='Critical').sum()),
                 high_count      =('severity', lambda x:(x=='High').sum()),
                 medium_count    =('severity', lambda x:(x=='Medium').sum()),
                 low_count       =('severity', lambda x:(x=='Low').sum()),
                 avg_deviation   =('deviation_pct','mean'),
                 top_cause       =('suggested_cause',
                                   lambda x: x.value_counts().index[0]
                                   if len(x.value_counts())>0 else 'Normal'))
            .reset_index()
            .sort_values('critical_count', ascending=False))

# ── Load Everything ───────────────────────────────────────────────────────────
with st.spinner("⏳ Initializing Smart Urban EE System... Please wait (2-3 mins on first load)"):
    df_e_raw, df_w_raw     = generate_data()
    df_e_feat              = add_features(df_e_raw,  'electricity_kwh')
    df_w_feat              = add_features(df_w_raw,  'water_kl')
    df_e_z                 = apply_zscore(df_e_feat, 'electricity_kwh')
    df_w_z                 = apply_zscore(df_w_feat, 'water_kl')
    df_elec                = apply_iforest(df_e_z,   'electricity_kwh')
    df_water               = apply_iforest(df_w_z,   'water_kl')
    df_elec                = finalize(df_elec,  'electricity_kwh')
    df_water               = finalize(df_water, 'water_kl')
    elec_report            = generate_report(df_elec)
    water_report           = generate_report(df_water)

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
st.sidebar.markdown("**Data:** 2022–2023")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-title">⚡ Smart Urban Energy Efficiency System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Bangalore Urban Zone Monitoring — 2022 to 2023</p>',
                unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🏙️ Zones",         "8")
    c2.metric("🗺️ Wards",         "24")
    c3.metric("🔲 Blocks",        "96")
    c4.metric("⚡ Elec Anomalies", f"{df_elec['combined_anomaly'].sum():,}")
    c5.metric("💧 Water Anomalies",f"{df_water['combined_anomaly'].sum():,}")
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
