import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tarfile
import os
import tempfile
import numpy as np
import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

# ==========================================
# 1. Configuration & Caching
# ==========================================

# 페이지 설정 (가장 먼저 호출)
st.set_page_config(
    page_title="CM SAF R Toolbox (Python Ver)",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 연산자 정의
OPERATOR_GROUPS = [
    "Hourly statistics", "Daily statistics", "Monthly statistics",
    "Seasonal statistics", "Annual statistics", "Temporal operators",
    "Climate Analysis"
]

OPERATORS = {
    "Hourly statistics": {"Hourly mean": "hourmean", "Hourly sum": "hoursum"},
    "Daily statistics": {"Diurnal means": "daymean", "Diurnal sums": "daysum", "Diurnal maxima": "daymax",
                         "Diurnal minima": "daymin"},
    "Monthly statistics": {"Monthly means": "monmean", "Monthly sums": "monsum", "Monthly anomalies": "mon.anomaly"},
    "Climate Analysis": {"Absolute map": "absolute_map", "Anomaly map": "anomaly_map",
                         "Time Series Plot": "time_series_plot"}
}

# 색상맵 옵션
COLORMAPS = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'jet', 'Spectral']


# ==========================================
# 2. Data Utilities (Cached)
# ==========================================

def identify_lat_lon_names(ds):
    """데이터셋에서 위경도 변수명 식별"""
    lat_name = None
    lon_name = None
    lat_candidates = ['lat', 'latitude', 'xlqc']
    lon_candidates = ['lon', 'longitude', 'xlgc']

    iterator = ds.variables if isinstance(ds, xr.Dataset) else ds.coords

    # 1. attrs 기반 검색
    for var_name in iterator:
        try:
            attrs = ds[var_name].attrs
            std_name = attrs.get('standard_name', '').lower()
            units = attrs.get('units', '').lower()
            if std_name == 'latitude' or 'degrees_north' in units: lat_name = var_name
            if std_name == 'longitude' or 'degrees_east' in units: lon_name = var_name
        except:
            continue

    # 2. 이름 기반 검색
    if not lat_name:
        for var_name in iterator:
            if str(var_name).lower() in lat_candidates: lat_name = var_name; break
    if not lon_name:
        for var_name in iterator:
            if str(var_name).lower() in lon_candidates: lon_name = var_name; break

    return lat_name, lon_name


def extract_tar(file_obj, extract_path):
    """TAR 파일 압축 해제"""
    try:
        with tarfile.open(fileobj=file_obj, mode='r') as tar:
            tar.extractall(path=extract_path)
            return [f for f in os.listdir(extract_path) if f.endswith('.nc')]
    except Exception as e:
        st.error(f"Error extracting tar file: {e}")
        return []


@st.cache_resource(show_spinner=False)
def load_dataset(file_path):
    """NetCDF 파일 로드 (캐싱 적용)"""
    try:
        ds = xr.open_dataset(file_path, decode_times=True)
        lat_name, lon_name = identify_lat_lon_names(ds)
        coords_to_set = [n for n in [lat_name, lon_name] if n and n in ds.data_vars]
        if coords_to_set:
            ds = ds.set_coords(coords_to_set)
        return ds
    except Exception as e:
        return None


@st.cache_data(show_spinner=False)
def calculate_statistics(file_path, var_name, operator_code):
    """통계 연산 수행 (데이터가 아닌 파일 경로를 키로 캐싱)"""
    ds = load_dataset(file_path)
    if ds is None: return None

    da = ds[var_name]

    if operator_code == "monmean":
        return da.resample(time="1MS").mean(dim="time")
    elif operator_code == "monsum":
        return da.resample(time="1MS").sum(dim="time")
    elif operator_code == "daymean":
        return da.resample(time="1D").mean(dim="time")
    elif operator_code == "mon.anomaly":
        climatology = da.groupby("time.month").mean("time")
        return da.groupby("time.month") - climatology
    return da


# ==========================================
# 3. Plotting Utilities
# ==========================================

def plot_map(data_array, time_step=0, cmap='viridis', title="Map Plot"):
    # 데이터 슬라이싱
    data_slice = data_array.isel(time=time_step) if 'time' in data_array.dims and data_array.sizes[
        'time'] > time_step else data_array.isel(time=0) if 'time' in data_array.dims else data_array

    lat_name, lon_name = identify_lat_lon_names(data_array)

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.5)
    ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')

    plot_kwargs = {
        'ax': ax, 'transform': ccrs.PlateCarree(),
        'cbar_kwargs': {'label': data_array.attrs.get('units', ''), 'shrink': 0.8},
        'cmap': cmap
    }
    if lat_name and lon_name:
        plot_kwargs.update({'x': lon_name, 'y': lat_name})

    try:
        data_slice.plot(**plot_kwargs)
        time_str = str(data_slice.time.values)[:10] if 'time' in data_slice.coords else ''
        ax.set_title(f"{title} | {time_str}", fontsize=14, fontweight='bold')
    except Exception as e:
        st.error(f"Plotting Error: {e}")

    return fig


def plot_timeseries(data_array, title="Time Series"):
    lat_name, lon_name = identify_lat_lon_names(data_array)
    dims_to_mean = [d for d in [lat_name, lon_name, 'lat', 'lon'] if d in data_array.dims]

    ts = data_array.mean(dim=dims_to_mean) if dims_to_mean else data_array

    fig, ax = plt.subplots(figsize=(12, 5))
    ts.plot.line(ax=ax, color='#1f77b4', linewidth=2)
    ax.set_title(f"Spatially Averaged Time Series: {title}", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_ylabel(data_array.attrs.get('units', 'Value'))
    return fig


# ==========================================
# 4. Main Application
# ==========================================

def main():
    # --- Sidebar Navigation ---
    with st.sidebar:
        st.title("☁️ CM SAF Toolbox")
        st.markdown("Cloud & Radiation Analysis")

        step = st.radio(
            "Workflow Step",
            ["1. Prepare Data", "2. Analyze Data", "3. Visualize Results"],
            index=0
        )

        st.divider()
        st.caption("Based on CM SAF R Toolbox")
        st.caption("Powered by Streamlit & xarray")

    # --- Session State Init ---
    if 'nc_file_path' not in st.session_state: st.session_state.nc_file_path = None
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None

    # --- STEP 1: PREPARE ---
    if step == "1. Prepare Data":
        st.title("📂 Data Preparation")
        st.markdown("분석할 **NetCDF**(.nc) 또는 **Tar**(.tar) 파일을 업로드하세요.")

        uploaded_file = st.file_uploader("Upload File", type=['tar', 'nc'], help="최대 200MB 권장")

        if uploaded_file:
            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # TAR 파일 처리
            if uploaded_file.name.endswith('.tar'):
                with st.status("Extracting tar file...", expanded=True) as status:
                    nc_files = extract_tar(uploaded_file, temp_dir)
                    if nc_files:
                        status.update(label="Extraction Complete!", state="complete", expanded=False)
                        st.success(f"📦 {len(nc_files)} files extracted.")
                        selected_file = st.selectbox("Select a NetCDF file to load", nc_files)
                        st.session_state.nc_file_path = os.path.join(temp_dir, selected_file)
                    else:
                        status.update(label="Extraction Failed", state="error")
                        st.error("No .nc files found.")
            else:
                st.session_state.nc_file_path = file_path

            # 데이터셋 로드 및 미리보기
            if st.session_state.nc_file_path:
                ds = load_dataset(st.session_state.nc_file_path)
                if ds:
                    st.success(f"✅ Loaded: {os.path.basename(st.session_state.nc_file_path)}")
                    with st.expander("🔍 View Dataset Metadata", expanded=True):
                        st.write(ds)
                        st.caption(f"Dimensions: {dict(ds.dims)}")

    # --- STEP 2: ANALYZE ---
    elif step == "2. Analyze Data":
        st.title("⚡ Data Analysis")

        if not st.session_state.nc_file_path:
            st.warning("⚠️ 'Prepare Data' 단계에서 파일을 먼저 업로드해주세요.")
            st.stop()

        ds = load_dataset(st.session_state.nc_file_path)
        if not ds: st.stop()

        col1, col2 = st.columns([1, 2], gap="large")

        with col1:
            st.subheader("⚙️ Settings")
            with st.container(border=True):
                selected_var = st.selectbox("Target Variable", list(ds.data_vars))
                op_group = st.selectbox("Operator Group", OPERATOR_GROUPS)

                op_list = list(OPERATORS.get(op_group, {}).keys())
                if op_list:
                    op_name = st.selectbox("Operator", op_list)
                    op_code = OPERATORS[op_group][op_name]
                else:
                    st.error("No operators available")
                    op_code = None

                if st.button("Run Calculation ▶️", type="primary", use_container_width=True):
                    if op_code:
                        with st.spinner("Processing..."):
                            # 캐싱된 함수 호출
                            result = calculate_statistics(st.session_state.nc_file_path, selected_var, op_code)
                            if result is not None:
                                st.session_state.processed_data = result
                                st.session_state.processed_data.name = f"{selected_var}_{op_code}"
                                st.rerun()  # 결과 갱신을 위해 리런

        with col2:
            st.subheader("📊 Result Preview")
            if st.session_state.processed_data is not None:
                res = st.session_state.processed_data

                # 요약 메트릭 표시
                m1, m2, m3 = st.columns(3)
                m1.metric("Min Value", f"{res.min().values:.2f}")
                m2.metric("Mean Value", f"{res.mean().values:.2f}")
                m3.metric("Max Value", f"{res.max().values:.2f}")

                st.success("Calculation completed successfully.")
                with st.expander("See raw data structure"):
                    st.write(res)
            else:
                st.info("👈 왼쪽 패널에서 연산을 실행하면 결과가 이곳에 표시됩니다.")

    # --- STEP 3: VISUALIZE ---
    elif step == "3. Visualize Results":
        st.title("🎨 Visualization")

        data = st.session_state.processed_data
        if data is None:
            st.warning("⚠️ 분석된 데이터가 없습니다. 'Analyze Data' 단계에서 연산을 먼저 수행해주세요.")
            st.stop()

        # 탭 구성을 통한 시각화 분리
        tab1, tab2, tab3 = st.tabs(["🌍 2D Map", "📈 Time Series", "🔍 Interactive Explorer"])

        with tab1:
            col_opt, col_plot = st.columns([1, 3])
            with col_opt:
                st.markdown("#### Map Options")
                cmap = st.selectbox("Colormap", COLORMAPS, index=0)

                time_idx = 0
                if 'time' in data.dims and data.sizes['time'] > 1:
                    time_len = data.sizes['time']
                    time_idx = st.slider("Time Step", 0, time_len - 1, 0, format="Idx %d")
                    st.caption(f"Date: {str(data.time.values[time_idx])[:10]}")

            with col_plot:
                fig = plot_map(data, time_idx, cmap, title=data.name)
                st.pyplot(fig)

        with tab2:
            st.markdown("#### Spatial Average Time Series")
            if 'time' in data.dims and data.sizes['time'] > 1:
                fig_ts = plot_timeseries(data, title=data.name)
                st.pyplot(fig_ts)
            else:
                st.info("시계열 차원을 가진 데이터가 아닙니다 (Time dimension size <= 1).")

        with tab3:
            st.markdown("#### PyGWalker Interactive Analysis")
            st.caption("Tableau-like drag-and-drop interface")

            if st.checkbox("Load PyGWalker (May take resources)", value=False):
                with st.spinner("Preparing interactive data..."):
                    try:
                        # 데이터프레임 변환 최적화 (너무 크면 샘플링)
                        df = data.to_dataframe(name='value').reset_index()
                        # if len(df) > 100000:
                        #     st.warning(f"Large dataset ({len(df)} rows). Using top 50,000 for performance.")
                        #     df = df.head(50000)

                        renderer = StreamlitRenderer(df, spec="./gw_config.json", spec_io_mode="RW")
                        renderer.explorer()
                    except Exception as e:
                        st.error(f"Error launching PyGWalker: {e}")

        # 다운로드 버튼 (하단 고정)
        st.divider()
        try:
            nc_bytes = data.to_netcdf()
            st.download_button(
                label="📥 Download Result (.nc)",
                data=nc_bytes,
                file_name=f"{data.name}_result.nc",
                mime="application/x-netcdf",
                type="primary"
            )
        except:
            pass


if __name__ == "__main__":
    main()