import pandas as pd
import numpy as np
import os
import datetime
import base64
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# =================================================================
# I. CẤU HÌNH TRANG & GIAO DIỆN (CORPORATE TERMINAL STYLE)
# =================================================================
st.set_page_config(page_title="Yuanta Stock Gems Elite", layout="wide", initial_sidebar_state="expanded")

# Hàm mã hóa ảnh sang Base64 để hiển thị trong HTML
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return ""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

# Tải chuỗi base64 của logo
LOGO_PATH = "logo-ysvn.png"
logo_base64 = get_base64_of_bin_file(LOGO_PATH)
logo_html = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""

st.markdown("""
    <style>
    .main { background-color: #041C32; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    
    /* Hero Banner Corporate Design - Phối hợp Xanh, Cam, Trắng */
    .hero-banner {
        background: linear-gradient(90deg, #001E3C 0%, #034EA2 100%);
        padding: 35px 30px;
        border-radius: 15px;
        margin-bottom: 30px;
        color: white;
        border-bottom: 5px solid #F26522; /* Điểm nhấn màu Cam */
        box-shadow: 0 15px 35px rgba(0,0,0,0.4);
    }
    
    .hero-subtitle {
        text-transform: uppercase; 
        letter-spacing: 2px; 
        font-size: 0.85rem; 
        color: #F26522; /* Màu Cam cho Subtitle */
        font-weight: 700;
        margin-bottom: 10px;
    }

    .hero-main-title {
        font-weight: 700; 
        font-size: 2.6rem; 
        margin: 0;
        color: #FFFFFF; /* Màu Trắng cho Title */
    }

    /* Logo Styling */
    .company-logo {
        height: 50px;
        margin-bottom: 15px;
        display: block;
    }

    /* Timestamp Box */
    .sync-timestamp {
        background: rgba(255,255,255,0.15); 
        padding: 5px 12px; 
        border-radius: 8px; 
        color: #FFFFFF; 
        font-size: 0.95rem; 
        border: 1px solid rgba(255,255,255,0.2);
        font-family: 'Courier New', Courier, monospace;
    }
    
    /* Profile Card for Tab 4 */
    .stock-profile-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 25px;
        margin-bottom: 25px;
        border-left: 6px solid #F26522;
    }
    .profile-ticker { font-size: 2.2rem; font-weight: 800; color: #FFFFFF; margin: 0; }
    .profile-name { font-size: 1.1rem; color: #94A3B8; margin-bottom: 10px; font-weight: 600; }
    .profile-meta { font-size: 0.9rem; color: #CBD5E1; display: flex; gap: 20px; }
    .profile-meta b { color: #F26522; }

    /* Custom Table Styling for Scorecard */
    .scorecard-table {
        width: 100%;
        border-collapse: collapse;
        margin: 15px 0;
        background: rgba(255,255,255,0.01);
        border-radius: 10px;
        overflow: hidden;
    }
    .scorecard-table th {
        background: rgba(3, 78, 162, 0.4);
        color: #FFFFFF;
        padding: 14px;
        text-align: left;
        font-size: 0.85rem;
        text-transform: uppercase;
        border-bottom: 2px solid #034EA2;
    }
    .scorecard-table td {
        padding: 16px 14px;
        color: #FFFFFF;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 1.05rem;
        font-weight: 500;
    }
    .scorecard-table tr:hover {
        background: rgba(255,255,255,0.03);
    }

    /* KPI Card Style */
    .kpi-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
        padding: 22px;
        border-radius: 15px;
        border-left: 5px solid #034EA2;
        margin-bottom: 15px;
        min-height: 160px;
    }
    .kpi-label { font-size: 0.75rem; color: #94A3B8; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { font-size: 1.8rem; font-weight: 800; color: #FFFFFF; margin: 10px 0; }
    
    /* Typography & Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] {
        height: 55px; background-color: rgba(255,255,255,0.03);
        border-radius: 10px 10px 0 0; padding: 10px 35px; color: #94A3B8; font-weight: 600;
        text-transform: uppercase; font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] { background-color: #034EA2 !important; color: white !important; }
    
    /* Data Sidebar Styling */
    .sidebar-header { background: #034EA2; padding: 15px; border-radius: 10px; color: white; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# II. ENGINE: XỬ LÝ DỮ LIỆU ĐỘNG (CORE LOGIC)
# =================================================================
PATH_MERGED = "data_cache/df_merged.parquet"
PATH_STATS = "data_cache/df_market_stats_historical.parquet"

@st.cache_data(ttl=3600) # Lưu cache trong 1 giờ để tăng tốc độ truy cập công khai
def load_and_standardize_data():
    """Tải dữ liệu và chuẩn hóa 100% sang đơn vị NGHÌN TỶ VND."""
    if not os.path.exists(PATH_MERGED):
        st.error(f"⚠️ Thiếu file dữ liệu: {PATH_MERGED}")
        st.stop()
    
    df = pd.read_parquet(PATH_MERGED)
    
    # Ép kiểu dữ liệu sớm để tối ưu bộ nhớ
    df['Nam'] = pd.to_numeric(df['Nam'], errors='coerce').fillna(0).astype(np.int32)
    df['Q_int'] = df['Quy'].str.extract('(\d+)').astype(float).fillna(0).astype(np.int8)
    
    # Lọc ưu tiên báo cáo Hợp nhất (HN) để tránh trùng lặp mã CP
    df = df.sort_values(['MaCoPhieu', 'Nam', 'Q_int', 'LoaiBaoCao'], ascending=[True, True, True, False])
    df = df.drop_duplicates(subset=['MaCoPhieu', 'Nam', 'Q_int'], keep='first')

    # Quy đổi đơn vị: Triệu VND -> NGHÌN TỶ VND
    financial_cols = ['DoanhThuThuan', 'LoiNhuanTruocThue', 'LoiNhuanSauThue', 'VonHoa']
    for col in financial_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) / 1000000.0
            
    return df

def calculate_growth_metrics(df):
    """Tính toán bứt phá Like-for-Like đa chiều (Logic Map-Back siêu bền bỉ)."""
    if df.empty: return df
    df_res = df.copy()
    
    # Tạo key duy nhất để map dữ liệu quá khứ nhanh chóng
    df_res['lookup_key'] = df_res['MaCoPhieu'] + "_" + df_res['Nam'].astype(str) + "_" + df_res['Q_int'].astype(str)
    
    metrics = ['DoanhThuThuan', 'LoiNhuanTruocThue', 'LoiNhuanSauThue']
    
    for m in metrics:
        # 1. Giá trị lũy kế YTD
        df_res[f'{m}_YTD_Val'] = df_res.groupby(['MaCoPhieu', 'Nam'])[m].cumsum()
        
        # 2. Map giá trị cùng kỳ năm trước (YoY)
        df_res['prev_year_key'] = df_res['MaCoPhieu'] + "_" + (df_res['Nam'] - 1).astype(str) + "_" + df_res['Q_int'].astype(str)
        val_map_yoy = df_res.set_index('lookup_key')[m].to_dict()
        df_res['val_prev_yoy'] = df_res['prev_year_key'].map(val_map_yoy)
        df_res[f'{m}_YoY_Pct'] = (df_res[m] - df_res['val_prev_yoy']) / df_res['val_prev_yoy'].abs()
        
        # 3. Map giá trị quý trước (QoQ)
        # Logic tính quý trước (Nếu Q1 thì lùi về Q4 năm trước)
        df_res['prev_q_year'] = np.where(df_res['Q_int'] == 1, df_res['Nam'] - 1, df_res['Nam'])
        df_res['prev_q_idx'] = np.where(df_res['Q_int'] == 1, 4, df_res['Q_int'] - 1)
        df_res['prev_q_key'] = df_res['MaCoPhieu'] + "_" + df_res['prev_q_year'].astype(str) + "_" + df_res['prev_q_idx'].astype(str)
        
        val_map_qoq = df_res.set_index('lookup_key')[m].to_dict()
        df_res['val_prev_qoq'] = df_res['prev_q_key'].map(val_map_qoq)
        df_res[f'{m}_QoQ_Pct'] = (df_res[m] - df_res['val_prev_qoq']) / df_res['val_prev_qoq'].abs()
        
        # 4. Map YTD cùng kỳ (YTD Growth)
        ytd_map = df_res.set_index('lookup_key')[f'{m}_YTD_Val'].to_dict()
        df_res['val_ytd_prev'] = df_res['prev_year_key'].map(ytd_map)
        df_res[f'{m}_YTD_Pct'] = (df_res[f'{m}_YTD_Val'] - df_res['val_ytd_prev']) / df_res['val_ytd_prev'].abs()

    # Dọn dẹp các cột phụ
    drop_cols = ['lookup_key', 'prev_year_key', 'val_prev_yoy', 'prev_q_year', 'prev_q_idx', 'prev_q_key', 'val_prev_qoq', 'val_ytd_prev']
    return df_res.drop(columns=[c for c in drop_cols if c in df_res.columns])



# =================================================================
# III. RENDER LAYER: TAB 1 - TOÀN THỊ TRƯỜNG
# =================================================================

def render_market_trend_chart(df, year, quarter, metrics_map):
    """Hàm xử lý và hiển thị biểu đồ xu hướng (CELL 9)."""
    q_i = int(quarter.replace('Q', ''))
    st.divider()
    st.markdown("### BIỂU ĐỒ XU HƯỚNG TĂNG TRƯỞNG LỊCH SỬ")
    
    col_t1, col_t2 = st.columns(2)
    with col_t1: 
        m_sel = st.selectbox("Chọn chỉ tiêu phân tích:", list(metrics_map.keys()), format_func=lambda x: metrics_map[x][0], key='m_trend')
    with col_t2: 
        c_sel = st.selectbox("Loại so sánh:", ['YoY', 'QoQ', 'YTD'], key='c_trend')

    df_clean = df[(df['Nam'] < 2025) | ((df['Nam'] == 2025) & (df['Q_int'] <= 3))].copy()
    comp_name_map = {'YoY': 'YoY_Growth', 'QoQ': 'QoQ_Growth', 'YTD': 'YTD_Growth'}
    comp_label_map = {'YoY': 'CÙNG KỲ NĂM TRƯỚC (YoY)', 'QoQ': 'QUÝ TRƯỚC (QoQ)', 'YTD': 'LŨY KẾ ĐẦU NĂM (YTD)'}
    target_col = comp_name_map.get(c_sel)
    display_comp = comp_label_map.get(c_sel)
    display_name = metrics_map[m_sel][0]

    df_filtered = df_clean[(df_clean['Nam'] < year) | ((df_clean['Nam'] == year) & (df_clean['Q_int'] <= q_i))].copy()
    periods_df = df_filtered[['Nam', 'Quy', 'Q_int']].drop_duplicates().sort_values(['Nam', 'Q_int']).tail(12)
    target_periods = (periods_df['Nam'].astype(str) + " " + periods_df['Quy']).tolist()

    plot_data_list = []
    groups = [g for g in df_filtered['NhomPhanTich'].unique() if g is not None]
    for g in groups + ['Toàn thị trường']:
        subset = df_filtered if g == 'Toàn thị trường' else df_filtered[df_filtered['NhomPhanTich'] == g]
        res = []
        for _, row in periods_df.iterrows():
            y_p, q_s, qi_p = row['Nam'], row['Quy'], row['Q_int']
            curr_d = subset[(subset['Nam'] == y_p) & (subset['Q_int'] == qi_p)]
            stocks = curr_d['MaCoPhieu'].unique()
            prev_y = subset[(subset['Nam'] == y_p-1) & (subset['Q_int'] == qi_p) & (subset['MaCoPhieu'].isin(stocks))]
            yoy_v = (curr_d[m_sel].sum() - prev_y[m_sel].sum()) / abs(prev_y[m_sel].sum()) if not prev_y.empty and prev_y[m_sel].sum() != 0 else np.nan
            py_q, pq_q = (y_p-1, 4) if qi_p == 1 else (y_p, qi_p-1)
            prev_q = subset[(subset['Nam'] == py_q) & (subset['Q_int'] == pq_q) & (subset['MaCoPhieu'].isin(stocks))]
            qoq_v = (curr_d[m_sel].sum() - prev_q[m_sel].sum()) / abs(prev_q[m_sel].sum()) if not prev_q.empty and prev_q[m_sel].sum() != 0 else np.nan
            ytd_v_curr = curr_d[f'{m_sel}_YTD_Val'].sum()
            ytd_v_prev = prev_y[f'{m_sel}_YTD_Val'].sum()
            ytd_g = (ytd_v_curr - ytd_v_prev) / abs(ytd_v_prev) if not prev_y.empty and ytd_v_prev != 0 else np.nan
            res.append({'Period': f"{y_p} {q_s}", 'Group': g, 'YoY_Growth': yoy_v, 'QoQ_Growth': qoq_v, 'YTD_Growth': ytd_g})
        plot_data_list.append(pd.DataFrame(res))
    
    df_plot = pd.concat(plot_data_list)
    color_map = {'Toàn thị trường': '#FFFFFF', 'Ngân hàng': '#F26522', 'Tài chính': '#0091FF', 'Phi tài chính': '#10B981', 'Khác': '#60A5FA'}
    
    # Điều chỉnh column_widths: Tăng không gian cho bảng từ 0.35 lên 0.45
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], specs=[[{"type": "scatter"}, {"type": "table"}]], horizontal_spacing=0.06)
    last_p = target_periods[-1]
    summary_data = []

    for group in df_plot['Group'].unique():
        df_sub = df_plot[df_plot['Group'] == group]
        fig.add_trace(go.Scatter(x=df_sub['Period'], y=df_sub[target_col], name=group, mode='markers+lines',
            line=dict(width=4 if group=='Toàn thị trường' else 2, color=color_map.get(group, '#94A3B8'), shape='spline'),
            marker=dict(size=8 if group=='Toàn thị trường' else 6),
            hovertemplate=f"<b>{group}</b><br>{c_sel}: %{{y:.2%}}<extra></extra>"), row=1, col=1)
        
        row_l = df_sub[df_sub['Period'] == last_p]
        summary_data.append({'Nhóm': group, 
                             'YoY': f"{row_l['YoY_Growth'].values[0]:+.2%}" if not pd.isna(row_l['YoY_Growth'].values[0]) else "N/A", 
                             'QoQ': f"{row_l['QoQ_Growth'].values[0]:+.2%}" if not pd.isna(row_l['QoQ_Growth'].values[0]) else "N/A", 
                             'YTD': f"{row_l['YTD_Growth'].values[0]:+.2%}" if not pd.isna(row_l['YTD_Growth'].values[0]) else "N/A", 
                             'color': color_map.get(group, '#94A3B8')})

    summary_data = sorted(summary_data, key=lambda x: 0 if x['Nhóm'] == 'Toàn thị trường' else 1)
    
    # Nâng cấp bảng: Tăng cỡ chữ (size) và chiều cao hàng (height)
    fig.add_trace(go.Table(
        header=dict(
            values=["NHÓM", "YoY", "QoQ", "YTD"], 
            fill_color='#1E293B', 
            align='center', 
            font=dict(color='white', size=14), 
            height=45
        ),
        cells=dict(
            values=[
                [i['Nhóm'] for i in summary_data], 
                [i['YoY'] for i in summary_data], 
                [i['QoQ'] for i in summary_data], 
                [i['YTD'] for i in summary_data]
            ],
            fill_color='#0F172A', 
            align='center', 
            font=dict(color=[[i['color'] for i in summary_data], 'white', 'white', 'white'], size=13), 
            height=40
        )
    ), row=1, col=2)

    fig.update_layout(
        plot_bgcolor='#041C32', paper_bgcolor='#041C32', font=dict(family="Be Vietnam Pro", color="#E2E8F0"),
        title=dict(text=f"<b>{display_name}</b> <span style='font-size:12px; color:#94A3B8;'>| TĂNG TRƯỞNG {display_comp}</span>", font=dict(size=18, color="#FFFFFF"), x=0.05, y=0.96),
        xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(gridcolor='rgba(255, 255, 255, 0.08)', zerolinecolor='rgba(255, 255, 255, 0.15)', tickformat='.0%'),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.35, bgcolor='rgba(0,0,0,0)'), showlegend=True, margin=dict(l=50, r=50, t=100, b=50)
    )
    st.plotly_chart(fig, use_container_width=True)


def render_tab_market(df, year, quarter):
    """Tab 1: Toàn thị trường - Bố cục tối ưu 3 cột cho chỉ tiêu chính."""
    q_i = int(quarter.replace('Q', ''))
    
    # TRUY XUẤT ĐỘ PHỦ
    total_listed, total_listed_cap = 1, 1 
    if os.path.exists(PATH_STATS):
        df_stats = pd.read_parquet(PATH_STATS)
        st_p = df_stats[(df_stats['Nam'] == year) & (df_stats['Quy'] == quarter)]
        if not st_p.empty:
            total_listed = st_p['Tổng số công ty niêm yết'].values[0]
            total_listed_cap = st_p['Tổng vốn hóa toàn thị trường'].values[0] / 1000000.0

    df_curr = df[(df['Nam'] == year) & (df['Q_int'] == q_i)]
    reported = df_curr[(df_curr['DoanhThuThuan'] != 0) | (df_curr['LoiNhuanSauThue'] != 0)].copy()
    reported_cap = reported['VonHoa'].sum()
    curr_codes = reported['MaCoPhieu'].unique()

    # --- 3 KPI CHI TIÊU NỔI BẬT (3 CỘT) ---
    metrics_map = {'DoanhThuThuan': ('DOANH THU THUẦN', '#034EA2'), 'LoiNhuanTruocThue': ('LỢI NHUẬN TRƯỚC THUẾ', '#F26522'), 'LoiNhuanSauThue': ('LỢI NHUẬN SAU THUẾ', '#10B981')}
    df_prev_y = df[(df['Nam'] == year - 1) & (df['Q_int'] == q_i)]
    p_y, p_q = (year-1, 4) if q_i == 1 else (year, q_i-1)
    df_prev_q = df[(df['Nam'] == p_y) & (df['Q_int'] == p_q)]

    c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
    cols = [c_kpi1, c_kpi2, c_kpi3]

    for idx, (m_key, (m_label, m_color)) in enumerate(metrics_map.items()):
        v_q = reported[m_key].sum()
        v_ytd = reported[f'{m_key}_YTD_Val'].sum()
        
        yoy_prev_sum = df_prev_y[df_prev_y['MaCoPhieu'].isin(curr_codes)][m_key].sum()
        yoy = (v_q - yoy_prev_sum) / abs(yoy_prev_sum) if yoy_prev_sum != 0 else 0
        
        mom_prev_sum = df_prev_q[df_prev_q['MaCoPhieu'].isin(curr_codes)][m_key].sum()
        mom = (v_q - mom_prev_sum) / abs(mom_prev_sum) if mom_prev_sum != 0 else 0
        
        v_ytd_p = df_prev_y[df_prev_y['MaCoPhieu'].isin(curr_codes)][f'{m_key}_YTD_Val'].sum()
        ytd_pct = (v_ytd - v_ytd_p) / abs(v_ytd_p) if v_ytd_p != 0 else 0

        with cols[idx]:
            st.markdown(f"""<div class="kpi-card" style="border-left-color:{m_color}"><div class="kpi-label">{m_label}</div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div><div class="kpi-value">{v_q:,.1f} <span style="font-size:0.9rem; font-weight:400; color:#94A3B8;">nghìn tỷ</span></div>
                    <div class="kpi-sub">Lũy kế YTD: <b>{v_ytd:,.1f} nghìn tỷ</b></div></div>
                    <div style="text-align:right;"><div class="growth-tag" style="color:{'#10B981' if yoy>=0 else '#EF4444'}">{yoy:+.2%} YoY</div>
                    <div style="color:#94A3B8; font-size:0.85rem; margin-top:5px;">{mom:+.2%} MoM | {ytd_pct:+.2%} YTD%</div></div>
                </div></div>""", unsafe_allow_html=True)

    # --- ĐỘ PHỦ DỮ LIỆU (Nằm phía sau 3 box chỉ tiêu) ---
    st.markdown(f"""
        <div style="background: rgba(255,255,255,0.02); padding: 10px 20px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.05); margin-top: 10px; display: flex; justify-content: space-between; font-size: 0.85rem; color: #94A3B8;">
            <span>DN Công bố: <b style="color:white;">{len(reported):,} / {total_listed:,} ({len(reported)/total_listed:.1%})</b></span>
            <span>Vốn hóa Công bố: <b style="color:white;">{reported_cap:,.0f} / {total_listed_cap:,.0f}k tỷ ({reported_cap/total_listed_cap:.1%})</b></span>
        </div>
    """, unsafe_allow_html=True)

    # --- GỌI HÀM BIỂU ĐỒ XU HƯỚNG ---
    render_market_trend_chart(df, year, quarter, metrics_map)


# =================================================================
# IV. RENDER LAYER: TAB 2 - PHÂN TÍCH NGÀNH
# =================================================================
def render_tab_industry(df, year, quarter):
    """Tab 2: Phân tích ngành - Xếp hạng & Top 10 (Elite UI)."""
    q_i = int(quarter.replace('Q', ''))
    #st.markdown(f"## HIỆU SUẤT PHÂN NGÀNH {quarter}/{year}")
    
    # 1. Bộ lọc nhanh
    c1, c2 = st.columns(2)
    metric_map_ind = {'DoanhThuThuan': 'DOANH THU THUẦN', 'LoiNhuanSauThue': 'LỢI NHUẬN SAU THUẾ', 'LoiNhuanTruocThue': 'LỢI NHUẬN TRƯỚC THUẾ'}
    with c1: m_k = st.selectbox("🎯 Chỉ tiêu phân tích:", list(metric_map_ind.keys()), format_func=lambda x: metric_map_ind[x], key='ind_m')
    with c2: c_t = st.selectbox("📈 Loại tăng trưởng:", ['YoY', 'QoQ', 'YTD'], key='ind_c')
    
    # 2. Logic tính toán Like-for-Like (CELL 10)
    ind_results = []
    industries = df['Phân ngành - ICB L2'].dropna().unique()
    for ind in industries:
        sub = df[df['Phân ngành - ICB L2'] == ind]
        curr = sub[(sub['Nam'] == year) & (sub['Q_int'] == q_i)]
        curr_stocks = curr['MaCoPhieu'].unique()
        
        # YoY
        prev_y_data = sub[(sub['Nam'] == year-1) & (sub['Q_int'] == q_i) & (sub['MaCoPhieu'].isin(curr_stocks))]
        yoy = (curr[m_k].sum() - prev_y_data[m_k].sum()) / abs(prev_y_data[m_k].sum()) if not prev_y_data.empty and prev_y_data[m_k].sum() != 0 else np.nan
        
        # QoQ
        p_y, p_q = (year-1, 4) if q_i == 1 else (year, q_i-1)
        prev_q_data = sub[(sub['Nam'] == p_y) & (sub['Q_int'] == p_q) & (sub['MaCoPhieu'].isin(curr_stocks))]
        qoq = (curr[m_k].sum() - prev_q_data[m_k].sum()) / abs(prev_q_data[m_k].sum()) if not prev_q_data.empty and prev_q_data[m_k].sum() != 0 else np.nan

        # YTD
        curr_ytd = curr[f'{m_k}_YTD_Val'].sum()
        prev_ytd_data = prev_y_data[f'{m_k}_YTD_Val'].sum()
        ytd_growth = (curr_ytd - prev_ytd_data) / abs(prev_ytd_data) if not prev_y_data.empty and prev_ytd_data != 0 else np.nan
        
        val_to_plot = yoy if c_t == 'YoY' else qoq if c_t == 'QoQ' else ytd_growth
        ind_results.append({'Ngành': ind, 'Growth': val_to_plot, 'YoY': yoy, 'QoQ': qoq, 'YTD': ytd_growth})
    
    # Tính Toàn thị trường
    m_curr = df[(df['Nam'] == year) & (df['Q_int'] == q_i)]
    m_stocks = m_curr['MaCoPhieu'].unique()
    m_prev_y = df[(df['Nam'] == year-1) & (df['Q_int'] == q_i) & (df['MaCoPhieu'].isin(m_stocks))]
    p_y, p_q = (year-1, 4) if q_i == 1 else (year, q_i-1)
    m_prev_q = df[(df['Nam'] == p_y) & (df['Q_int'] == p_q) & (df['MaCoPhieu'].isin(m_stocks))]
    
    m_yoy = (m_curr[m_k].sum() - m_prev_y[m_k].sum()) / abs(m_prev_y[m_k].sum())
    m_qoq = (m_curr[m_k].sum() - m_prev_q[m_k].sum()) / abs(m_prev_q[m_k].sum())
    m_ytd = (m_curr[f'{m_k}_YTD_Val'].sum() - m_prev_y[f'{m_k}_YTD_Val'].sum()) / abs(m_prev_y[f'{m_k}_YTD_Val'].sum())
    m_val = m_yoy if c_t == 'YoY' else m_qoq if c_t == 'QoQ' else m_ytd

    # 3. Trực quan hóa Biểu đồ (CELL 10 Neon Glow)
    df_p = pd.DataFrame(ind_results).dropna(subset=['Growth']).sort_values('Growth', ascending=True)
    df_p = pd.concat([df_p, pd.DataFrame([{'Ngành': 'TOÀN THỊ TRƯỜNG', 'Growth': m_val, 'YoY': m_yoy, 'QoQ': m_qoq, 'YTD': m_ytd}])]).reset_index(drop=True)
    
    colors = ['#FFFFFF' if n == 'TOÀN THỊ TRƯỜNG' else ('#034EA2' if v >= 0 else '#FF3D00') for n, v in zip(df_p['Ngành'], df_p['Growth'])]

    fig_ind = go.Figure(go.Bar(
        y=df_p['Ngành'], x=df_p['Growth'], orientation='h', 
        marker=dict(color=colors, line=dict(width=2.5, color=colors), cornerradius=15),
        text=df_p['Growth'].apply(lambda x: f"<b>{x:+.1%}</b>"), textposition='outside',
        textfont=dict(color=colors)
    ))
    fig_ind.update_layout(
        title=f"<b>XẾP HẠNG TĂNG TRƯỞNG NGÀNH: {metric_map_ind[m_k]}</b>", 
        plot_bgcolor='#041C32', paper_bgcolor='#041C32', font=dict(color="white"), 
        height=700, xaxis_tickformat='.0%', margin=dict(l=160, r=60, t=100, b=50)
    )
    st.plotly_chart(fig_ind, use_container_width=True)

    # 4. Bảng tổng hợp hiệu suất
    st.markdown(f"#### TỔNG HỢP HIỆU SUẤT NGÀNH TẠI {quarter}/{year}")
    df_table = df_p.sort_values('Growth', ascending=False).copy()
    for col in ['YoY', 'QoQ', 'YTD']:
        df_table[col] = df_table[col].apply(lambda x: f"{x:+.1%}" if not pd.isna(x) else "N/A")
    st.dataframe(df_table[['Ngành', 'YoY', 'QoQ', 'YTD']], use_container_width=True, hide_index=True)

    # 5. Top 10 Cổ phiếu từng ngành (Grid 4 cột)
    st.divider()
    st.markdown(f"### TOP 10 DẪN ĐẦU THEO QUY MÔ GIÁ TRỊ")
    df_curr_stocks = df[(df['Nam'] == year) & (df['Q_int'] == q_i)].copy()
    sorted_inds = sorted(industries)
    
    for i in range(0, len(sorted_inds), 4):
        cols = st.columns(4)
        for j in range(4):
            if i + j < len(sorted_inds):
                ind_name = sorted_inds[i+j]
                top10 = df_curr_stocks[df_curr_stocks['Phân ngành - ICB L2'] == ind_name].nlargest(10, m_k)
                if not top10.empty:
                    with cols[j]:
                        with st.expander(f"NGÀNH {ind_name}"):
                            st.plotly_chart(
                                go.Figure(go.Bar(x=top10['MaCoPhieu'], y=top10[m_k]*1000, marker_color='#F26522'))
                                .update_layout(height=250, margin=dict(t=20, b=0, l=0, r=0), xaxis_title=None, yaxis_visible=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white')), 
                                use_container_width=True
                            )


# =================================================================
# V. RENDER LAYER: TAB 3 - TOP CP (FIXED & SYNCED)
# =================================================================

def render_tab_top_cp(df, year, quarter):
    """
    Phân tích cổ phiếu tinh hoa với giao diện chuyên nghiệp:
    - Bong bóng kích thước lớn, nổi bật hơn.
    - Hiển thị Ticker trực tiếp bên trong bong bóng.
    - Bảng xếp hạng hiển thị Top 100 mã.
    """
    #st.markdown(f"## PHÂN TÍCH TĂNG TRƯỞNG CHIẾN LƯỢC {quarter}/{year}")
    
    # 1. Bộ lọc tương tác
    c1, c2, c3 = st.columns(3)
    metric_options = {'DoanhThuThuan': 'Doanh thu thuần', 'LoiNhuanSauThue': 'Lợi nhuận sau thuế', 'LoiNhuanTruocThue': 'Lợi nhuận trước thuế'}
    with c1: m_top = st.selectbox("🎯 Chỉ tiêu trọng tâm:", list(metric_options.keys()), format_func=lambda x: metric_options[x], key='top_m_elite', index=0)
    with c2: cap_group = st.selectbox("⚖️ Nhóm vốn hóa:", ['Tất cả', 'Big', 'Mid', 'Small'], key='top_cap_elite', index=1)
    with c3: 
        industry_list_raw = sorted(df['Phân ngành - ICB L2'].dropna().unique().tolist())
        ind_choice = st.selectbox("🏭 Lọc theo ngành:", ['Toàn thị trường'] + industry_list_raw, key='top_ind_elite', index=0)
    
    # 2. Xử lý dữ liệu đồng bộ
    q_i = int(quarter.replace('Q', ''))
    df_curr = df[(df['Nam'] == int(year)) & (df['Q_int'] == q_i)].copy()
    
    if ind_choice != 'Toàn thị trường': df_curr = df_curr[df_curr['Phân ngành - ICB L2'] == ind_choice]
    if cap_group == 'Big': df_curr = df_curr[df_curr['VonHoa'] >= 10.0]
    elif cap_group == 'Mid': df_curr = df_curr[(df_curr['VonHoa'] >= 1.0) & (df_curr['VonHoa'] < 10.0)]
    elif cap_group == 'Small': df_curr = df_curr[df_curr['VonHoa'] < 1.0]

    if df_curr.empty:
        st.warning(f"⚠️ Không tìm thấy dữ liệu phù hợp với bộ lọc tại kỳ {quarter}/{year}.")
        return

    # 3. Triple View Bubble Charts (Cấu hình bong bóng lớn)
    plots_cfg = [
        {'x': m_top, 'y': f'{m_top}_QoQ_Pct', 'title': 'MOMENTUM TĂNG TRƯỞNG NGẮN HẠN (QoQ)', 'x_type': 'Quý'},
        {'x': m_top, 'y': f'{m_top}_YoY_Pct', 'title': 'SỨC MẠNH TĂNG TRƯỞNG DÀI HẠN (YoY)', 'x_type': 'Quý'},
        {'x': f'{m_top}_YTD_Val', 'y': f'{m_top}_YTD_Pct', 'title': 'HIỆU SUẤT TỔNG THỂ NĂM (YTD)', 'x_type': 'Lũy kế'}
    ]

    vibrant_colors = px.colors.qualitative.Prism + px.colors.qualitative.Bold

    for cfg in plots_cfg:
        df_plot_sub = df_curr.copy()
        df_plot_sub = df_plot_sub.dropna(subset=[cfg['x'], cfg['y']]).replace([np.inf, -np.inf], np.nan)
        df_plot_sub = df_plot_sub[df_plot_sub[cfg['y']].notna()] 
        
        if df_plot_sub.empty:
            st.info(f"ℹ️ {cfg['title']}: Không tìm thấy mã bứt phá có dữ liệu lịch sử.")
            continue
            
        df_top_bubble = df_plot_sub.sort_values(cfg['x'], ascending=False).head(15)
        
        # LOGIC TĂNG KÍCH THƯỚC BONG BÓNG: 
        # Giảm Denominator (mẫu số) trong sizeref để bong bóng to hơn đáng kể
        plot_size = df_top_bubble['VonHoa'].fillna(0)
        max_cap = plot_size.max()
        # safe_sizeref thấp hơn = bong bóng to hơn. Chuyển từ 2.2/(55**2) sang 1.5/(45**2)
        target_sizeref = 1.5 * max_cap / (45**2) if max_cap > 0 else 1

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_top_bubble[cfg['x']] * 1000, 
            y=df_top_bubble[cfg['y']], 
            mode='markers+text', 
            text=df_top_bubble['MaCoPhieu'],
            textposition="middle center", 
            cliponaxis=False,
            marker=dict(
                size=plot_size, sizemode='area', sizeref=target_sizeref, sizemin=22, # Sizemin tăng lên để nổi bật
                color=vibrant_colors[:len(df_top_bubble)], 
                line=dict(width=2, color='rgba(255,255,255,0.9)'), # Viền trắng rõ nét hơn
                opacity=0.9
            ),
            textfont=dict(family="Arial Black", size=11, color="white"), # Font text to hơn
            customdata=np.stack((df_top_bubble['MaCoPhieu'], df_top_bubble['TenCongTy'], df_top_bubble['Phân ngành - ICB L2'], plot_size * 1000), axis=-1),
            hovertemplate=(
                "<span style='font-size:16px; font-weight:bold; color:white;'>%{customdata[0]}</span><br>" +
                "<i>%{customdata[1]}</i><br><br>" +
                "Ngành: %{customdata[2]}<br>" +
                "Vốn hóa: %{customdata[3]:,.0f} tỷ<br>" +
                f"Giá trị {cfg['x_type']}: %{{x:,.0f}} tỷ<br>" +
                "Tăng trưởng: %{y:+.1%}<extra></extra>"
            )
        ))
        
        fig.add_hline(y=df_top_bubble[cfg['y']].mean(), line_dash="dot", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(
            plot_bgcolor='#041C32', paper_bgcolor='#041C32', font=dict(family="Be Vietnam Pro", color="#94A3B8"),
            title=dict(text=f"<b>{cfg['title']}</b>", x=0.02, y=0.95, font=dict(color="white", size=18)),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', title=f"Quy mô {cfg['x_type']} (Tỷ VND)"),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickformat='.0%', title="Hiệu suất (%)"),
            margin=dict(l=50, r=50, t=100, b=50), height=550, showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Bảng xếp hạng chi tiết TOP 100
    st.markdown("### 📋 TOP 100 DOANH NGHIỆP DẪN ĐẦU")
    df_summary_base = df_curr.sort_values(m_top, ascending=False).head(100) # Mở rộng lên Top 100
    summary = df_summary_base[['MaCoPhieu', 'TenCongTy', 'Phân ngành - ICB L2', 'VonHoa', 
                              m_top, f'{m_top}_YoY_Pct', f'{m_top}_QoQ_Pct', f'{m_top}_YTD_Pct']].copy()
    summary.columns = ['Mã CP', 'Doanh Nghiệp', 'Ngành', 'Vốn hóa', 'Giá trị Quý', '%YoY', '%MoM', '%YTD']
    for col in ['Vốn hóa', 'Giá trị Quý']: summary[col] = (summary[col] * 1000).map('{:,.0f} tỷ'.format)
    for col in ['%YoY', '%MoM', '%YTD']: summary[col] = summary[col].map('{:+.1%}'.format)
    
    st.dataframe(summary, use_container_width=True, hide_index=True, height=500) # Thêm chiều cao cho bảng
# =================================================================
# VI. RENDER LAYER: TAB 4 - DỮ LIỆU CHI TIẾT (FIXED SCORECARD RENDER)
# =================================================================

def render_tab_data(df):
    """Tab 4: Soi chi tiết mã cổ phiếu qua 3 Zone chuyên sâu - FIXED HTML Table."""
    col_sel1, col_sel2 = st.columns([1.5, 3])
    with col_sel1:
        ticker = st.selectbox("🔍 NHẬP MÃ CỔ PHIẾU CẦN PHÂN TÍCH:", sorted(df['MaCoPhieu'].unique()), key='data_t')
    
    df_s = df[df['MaCoPhieu'] == ticker.upper()].sort_values(['Nam', 'Q_int']).copy()
    if df_s.empty: return
    row = df_s.iloc[-1]

    # ZONE 1: HỒ SƠ DOANH NGHIỆP
    st.markdown(f"""
        <div class="stock-profile-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <h1 class="profile-ticker">{ticker}</h1>
                    <p class="profile-name">{row['TenCongTy']}</p>
                    <div class="profile-meta">
                        <span>Lĩnh vực: <b>{row['Phân ngành - ICB L2']}</b></span>
                        <span>Vốn hóa: <b>{row['VonHoa']*1000:,.0f} tỷ VND</b></span>
                    </div>
                </div>
                <div style="text-align: right;">
                    <p style="color: #94A3B8; font-size: 0.8rem; margin: 0;">Kỳ báo cáo gần nhất</p>
                    <h3 style="color: #F26522; margin: 0;">Quý {row['Q_int']}/{row['Nam']}</h3>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
  
    
    fmt_bil = lambda x: f"{x*1000:,.0f} tỷ"
    fmt_pct = lambda x: f"{x:+.1%}" if not pd.isna(x) and x != 0 else "N/A"
    
    # Chuẩn bị dữ liệu cho bảng
    score_rows = [
        {'label': 'Doanh thu thuần / Thu nhập lãi', 'val': row['DoanhThuThuan'], 'ytd': row['DoanhThuThuan_YTD_Val'], 'yoy': row['DoanhThuThuan_YoY_Pct'], 'mom': row['DoanhThuThuan_QoQ_Pct']},
        {'label': 'Lợi nhuận trước thuế', 'val': row['LoiNhuanTruocThue'], 'ytd': row['LoiNhuanTruocThue_YTD_Val'], 'yoy': row['LoiNhuanTruocThue_YoY_Pct'], 'mom': row['LoiNhuanTruocThue_QoQ_Pct']},
        {'label': 'Lợi nhuận sau thuế (LNST)', 'val': row['LoiNhuanSauThue'], 'ytd': row['LoiNhuanSauThue_YTD_Val'], 'yoy': row['LoiNhuanSauThue_YoY_Pct'], 'mom': row['LoiNhuanSauThue_QoQ_Pct']}
    ]

    # Xây dựng chuỗi HTML sạch (KHÔNG CÓ KHOẢNG TRẮNG ĐẦU DÒNG) để tránh lỗi render plain text
    html_rows = ""
    for item in score_rows:
        yoy_col = "#10B981" if (item['yoy'] or 0) >= 0 else "#EF4444"
        mom_col = "#10B981" if (item['mom'] or 0) >= 0 else "#EF4444"
        html_rows += f'<tr><td style="color: #94A3B8;">{item["label"]}</td><td style="text-align:right">{fmt_bil(item["val"])}</td><td style="text-align:right; color: #CBD5E1">{fmt_bil(item["ytd"])}</td><td style="text-align:right; color: {yoy_col}">{fmt_pct(item["yoy"])}</td><td style="text-align:right; color: {mom_col}">{fmt_pct(item["mom"])}</td></tr>'

    html_scorecard = f'<table class="scorecard-table"><thead><tr><th>Chỉ tiêu Tài chính</th><th style="text-align:right">Giá trị Quý</th><th style="text-align:right">Lũy kế YTD</th><th style="text-align:right">% YoY</th><th style="text-align:right">% QoQ</th></tr></thead><tbody>{html_rows}</tbody></table>'
    
    st.markdown(html_scorecard, unsafe_allow_html=True)

    # ZONE 3: BIỂU ĐỒ XU HƯỚNG LỊCH SỬ

    df_h = df_s[(df_s['Nam'] > 2022) | ((df_s['Nam'] == 2022) & (df_s['Q_int'] >= 1))].copy()
    df_h['Period'] = df_h['Nam'].astype(str) + " " + df_h['Quy']
    
    charts_cfg = [('DoanhThuThuan', 'DOANH THU THUẦN', '#034EA2', '#F26522'), ('LoiNhuanSauThue', 'LỢI NHUẬN SAU THUẾ', '#10B981', '#EF4444')]
    c_chart1, c_chart2 = st.columns(2)
    cols_chart = [c_chart1, c_chart2]

    for i, (m_k, m_n, c_b, c_l) in enumerate(charts_cfg):
        with cols_chart[i]:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df_h['Period'], y=df_h[m_k]*1000, name=m_n, marker_color=c_b, opacity=0.7), secondary_y=False)
            fig.add_trace(go.Scatter(x=df_h['Period'], y=df_h[f'{m_k}_YoY_Pct'], name="% YoY", line=dict(color=c_l, width=4), mode='markers+lines'), secondary_y=True)
            fig.update_layout(title=f"<b>Xu hướng {m_n}</b>", height=400, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=50, b=0, l=0, r=0), showlegend=False)
            fig.update_yaxes(title_text="Tỷ VND", secondary_y=False, showgrid=False)
            fig.update_yaxes(tickformat='.0%', secondary_y=True, showgrid=False)
            st.plotly_chart(fig, use_container_width=True)

    # Historical Table
    st.markdown("###### 📋 DỮ LIỆU CHUỖI THỜI GIAN CHI TIẾT (Tỷ VND)")
    cols_map = {'Nam':'Năm', 'Quy':'Quý', 'DoanhThuThuan':'Doanh thu', 'LoiNhuanSauThue':'LN sau thuế', 'DoanhThuThuan_YoY_Pct':'% YoY DT', 'LoiNhuanSauThue_YoY_Pct':'% YoY LN'}
    df_disp = df_h[list(cols_map.keys())].copy()
    for m in ['DoanhThuThuan', 'LoiNhuanSauThue']: df_disp[m] = (df_disp[m]*1000).map('{:,.0f}'.format)
    for p in ['DoanhThuThuan_YoY_Pct', 'LoiNhuanSauThue_YoY_Pct']: df_disp[p] = df_disp[p].apply(fmt_pct)
    st.dataframe(df_disp.rename(columns=cols_map).sort_values(['Năm', 'Quý'], ascending=False), use_container_width=True, hide_index=True)

# =================================================================
# VII. MAIN EXECUTION
# =================================================================
def render_footer():
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align: center; color: #94A3B8; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 20px; margin-bottom: 20px;">
            Hệ thống được phát triển bởi Trung tâm Phân tích - Yuanta Securities Vietnam <br>
            Dữ liệu nguồn: VSTDataFeed | © {datetime.datetime.now().year} Terminal v1.0 Alpha
        </div>
    """, unsafe_allow_html=True)

def main():
    df_raw = load_and_standardize_data()
    df_proc = calculate_growth_metrics(df_raw)
    
    # 1. TỰ ĐỘNG XÁC ĐỊNH KỲ MỚI NHẤT TRONG DỮ LIỆU
    latest_year = int(df_proc['Nam'].max())
    latest_q_int = int(df_proc[df_proc['Nam'] == latest_year]['Q_int'].max())
    latest_q_str = f"Q{latest_q_int}"
    
    st.sidebar.markdown(f"""<div class="sidebar-header"><h3 style="margin:0; font-size:1.1rem;">BỘ LỌC CHIẾN LƯỢC</h3></div>""", unsafe_allow_html=True)
    
    unique_years = sorted(df_proc['Nam'].unique(), reverse=True)
    year_default_idx = unique_years.index(latest_year)
    sel_y = st.sidebar.selectbox("Năm báo cáo:", unique_years, index=year_default_idx)
    
    q_options = ['Q1', 'Q2', 'Q3', 'Q4']
    q_default_idx = q_options.index(latest_q_str)
    sel_q = st.sidebar.selectbox("Quý báo cáo:", q_options, index=q_default_idx)
    
    now = datetime.datetime.now().strftime('%H:%M - %d/%m/%Y')

    st.markdown(f"""
        <div class="hero-banner">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p class="hero-subtitle">Yuanta Research Department</p>
                    <h1 class="hero-main-title">PHÂN TÍCH HIỆU SUẤT DOANH NGHIỆP {sel_q}/{sel_y}</h1>
                </div>
                <div style="text-align: right; display: flex; flex-direction: column; align-items: flex-end;">
                    <img src="{logo_html}" class="company-logo" onerror="this.style.display='none'">
                    <p style="margin: 0 0 10px 0; opacity: 0.7; font-size: 0.8rem;">Đồng bộ lần cuối</p>
                    <span class="sync-timestamp">{now}</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    t = st.tabs(["TOÀN THỊ TRƯỜNG", "NGÀNH", "TOP CỔ PHIẾU", "DỮ LIỆU CHI TIẾT"])
    with t[0]: render_tab_market(df_proc, sel_y, sel_q)
    with t[1]: render_tab_industry(df_proc, sel_y, sel_q)
    with t[2]: render_tab_top_cp(df_proc, sel_y, sel_q)
    with t[3]: render_tab_data(df_proc)
    
    render_footer()

if __name__ == "__main__":
    main()