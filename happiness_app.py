import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="World Happiness Predictor", layout="wide")

@st.cache_data
def load_and_train():
    df = pd.read_csv("World-happiness-report-2024.csv")
    df.rename(columns={
        'Ladder score': 'hapiness',
        'Regional indicator': 'region',
        'Log GDP per capita': 'GDP',
        'Social support': 'SS',
        'Healthy life expectancy': 'HLE',
        'Freedom to make life choices': 'FLMC',
        'Perceptions of corruption': 'corr',
        'Country name': 'Country',
    }, inplace=True)

    conti_vars = ['GDP', 'SS', 'HLE', 'FLMC', 'corr']
    df = df[['Country', 'region', 'hapiness'] + conti_vars].dropna()

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[conti_vars] = scaler.fit_transform(df[conti_vars])

    df_encoded = pd.get_dummies(df_scaled, columns=['region'], drop_first=True, dtype=int)
    region_cols = [c for c in df_encoded.columns if c.startswith('region_')]
    all_predictors = conti_vars + region_cols

    X = df_encoded[all_predictors]
    Y = df_encoded['hapiness']

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = Ridge(alpha=7.087370814634009)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    global_avg = {v: float(df[v].mean()) for v in conti_vars}

    return df, df_encoded, model, scaler, all_predictors, conti_vars, region_cols, r2, rmse, global_avg


df, df_encoded, model, scaler, all_predictors, conti_vars, region_cols, R2, RMSE, global_avg = load_and_train()

regions_list = sorted([c.replace("region_", "").replace("_", " ") for c in region_cols])
REFERENCE_REGION = "Central and Eastern Europe"
if REFERENCE_REGION not in regions_list:
    regions_list = [REFERENCE_REGION] + regions_list

country_list = ["Manual Entry"] + sorted(df['Country'].tolist())


def predict_happiness(gdp_raw, ss_raw, hle_raw, flmc_raw, corr_raw, region_name):
    raw = np.array([[gdp_raw, ss_raw, hle_raw, flmc_raw, corr_raw]])
    scaled = scaler.transform(raw)[0]
    row = {p: 0 for p in all_predictors}
    row['GDP'], row['SS'], row['HLE'], row['FLMC'], row['corr'] = scaled
    region_col = f"region_{region_name.replace(' ', '_')}"
    if region_col in row:
        row[region_col] = 1
    X_input = pd.DataFrame([row])[all_predictors]
    pred = float(np.clip(model.predict(X_input)[0], 1.0, 10.0))
    return pred


with st.sidebar:
    st.title("Inputs")

    selected_country = st.selectbox("Load a country", country_list)
    if selected_country != "Manual Entry":
        r = df[df['Country'] == selected_country].iloc[0]
        d_gdp, d_ss, d_hle, d_flmc, d_corr = float(r['GDP']), float(r['SS']), float(r['HLE']), float(r['FLMC']), float(r['corr'])
        d_reg = r['region']
    else:
        d_gdp  = float(df['GDP'].mean())
        d_ss   = float(df['SS'].mean())
        d_hle  = float(df['HLE'].mean())
        d_flmc = float(df['FLMC'].mean())
        d_corr = float(df['corr'].mean())
        d_reg  = "Western Europe"

    gdp_val  = st.slider("Economic Strength (GDP)",    0.0, 2.5, round(d_gdp,  3), 0.001)
    hle_val  = st.slider("Healthcare (Life Expectancy)",0.0, 1.2, round(d_hle,  3), 0.001)
    ss_val   = st.slider("Social Safety Net",           0.0, 1.6, round(d_ss,   3), 0.001)
    flmc_val = st.slider("Freedom of Choice",           0.0, 1.0, round(d_flmc, 3), 0.001)
    corr_val = st.slider("Government Integrity",        0.0, 1.0, round(d_corr, 3), 0.001)

    region_val = st.selectbox("Region", regions_list,
                               index=regions_list.index(d_reg) if d_reg in regions_list else 0)

    st.divider()
    st.metric("Model R²",  f"{R2*100:.1f}%")
    st.metric("RMSE",      f"{RMSE:.3f}")
    st.caption("Ridge Regression | alpha = 7.09 | 80/20 split")


score = predict_happiness(gdp_val, ss_val, hle_val, flmc_val, corr_val, region_val)

if score >= 7.0:
    score_color = "#2e7d32"
elif score >= 5.5:
    score_color = "#f57f17"
else:
    score_color = "#c62828"

st.title("World Happiness Predictor")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown(f"""
    <div style="background:#f5f5f5; border-radius:12px; padding:30px; text-align:center;">
        <div style="font-size:72px; font-weight:700; color:{score_color};">{score:.2f}</div>
        <div style="font-size:18px; color:#555;">out of 10</div>
        <div style="font-size:13px; color:#999; margin-top:8px;">
            Range: {max(1.0, score-RMSE):.2f} – {min(10.0, score+RMSE):.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 40}, 'suffix': '/10'},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': score_color, 'thickness': 0.25},
            'steps': [
                {'range': [0, 4],   'color': '#ffcdd2'},
                {'range': [4, 5.5], 'color': '#ffe0b2'},
                {'range': [5.5, 7], 'color': '#fff9c4'},
                {'range': [7, 10],  'color': '#c8e6c9'},
            ],
            'threshold': {
                'line': {'color': 'black', 'width': 2},
                'thickness': 0.75,
                'value': float(df['hapiness'].mean())
            }
        },
        title={'text': f"Region: {region_val}<br><span style='font-size:12px;color:#888'>Black line = global avg ({df['hapiness'].mean():.2f})</span>"}
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

coef_dict = dict(zip(all_predictors, model.coef_))
scaled_vals = scaler.transform(np.array([[gdp_val, ss_val, hle_val, flmc_val, corr_val]]))[0]
factor_values = dict(zip(conti_vars, scaled_vals))

labels_map = {'GDP': 'GDP', 'SS': 'Social Support', 'HLE': 'Life Expectancy', 'FLMC': 'Freedom', 'corr': 'Low Corruption'}

contributions = {labels_map[v]: coef_dict.get(v, 0) * factor_values[v] for v in conti_vars}
contribs_df = pd.DataFrame({'Factor': list(contributions.keys()), 'Contribution': list(contributions.values())}).sort_values('Contribution', ascending=True)

col3, col4 = st.columns(2)

with col3:
    st.subheader("Factor Contributions")
    colors = ['#f44336' if v < 0 else '#4CAF50' for v in contribs_df['Contribution']]
    fig_contrib = go.Figure(go.Bar(
        x=contribs_df['Contribution'],
        y=contribs_df['Factor'],
        orientation='h',
        marker_color=colors,
        text=[f"{v:+.3f}" for v in contribs_df['Contribution']],
        textposition='outside',
    ))
    fig_contrib.update_layout(
        height=320, margin=dict(l=10, r=60, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(zeroline=True, zerolinecolor='#aaa', zerolinewidth=1.5),
        xaxis_title="Contribution to score",
    )
    st.plotly_chart(fig_contrib, use_container_width=True)

with col4:
    st.subheader("Feature Importance (Ridge Coefficients)")
    feat_df = pd.DataFrame({
        'Feature': [labels_map[v] for v in conti_vars],
        'Coefficient': [abs(coef_dict.get(v, 0)) for v in conti_vars]
    }).sort_values('Coefficient', ascending=True)

    fig_feat = go.Figure(go.Bar(
        x=feat_df['Coefficient'],
        y=feat_df['Feature'],
        orientation='h',
        marker_color='#2196F3',
        text=[f"{v:.3f}" for v in feat_df['Coefficient']],
        textposition='outside',
    ))
    fig_feat.update_layout(
        height=320, margin=dict(l=10, r=60, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Absolute Coefficient",
    )
    st.plotly_chart(fig_feat, use_container_width=True)

st.divider()

col5, col6 = st.columns(2)

with col5:
    st.subheader("Your Inputs vs Global Average (Radar)")
    radar_labels = [labels_map[v] for v in conti_vars]
    user_values = [gdp_val, ss_val, hle_val, flmc_val, corr_val]
    avg_values  = [global_avg[v] for v in conti_vars]
    maxes       = [2.5, 1.6, 1.2, 1.0, 1.0]
    user_norm   = [v / m for v, m in zip(user_values, maxes)]
    avg_norm    = [v / m for v, m in zip(avg_values, maxes)]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_norm + [user_norm[0]],
        theta=radar_labels + [radar_labels[0]],
        fill='toself', name='Your Inputs',
        line_color='#2196F3', fillcolor='rgba(33,150,243,0.15)',
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_norm + [avg_norm[0]],
        theta=radar_labels + [radar_labels[0]],
        fill='toself', name='Global Average',
        line_color='#FF9800', fillcolor='rgba(255,152,0,0.1)', line_dash='dot',
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center'),
        height=360, margin=dict(l=30, r=30, t=30, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col6:
    st.subheader("Closest Real Countries")
    df_cmp = df.copy()
    df_cmp['distance'] = abs(df_cmp['hapiness'] - score)
    closest = df_cmp.nsmallest(8, 'distance')[['Country', 'region', 'hapiness']].rename(
        columns={'hapiness': 'Happiness Score', 'region': 'Region'}
    ).reset_index(drop=True)
    closest.index += 1
    closest['Happiness Score'] = closest['Happiness Score'].round(3)
    st.dataframe(closest, use_container_width=True, height=320)
