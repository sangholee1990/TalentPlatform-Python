import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import tarfile
import os
import tempfile
import numpy as np

# ==========================================
# 1. Configuration (설정 및 연산자 정의)
# ==========================================

# 연산자 그룹 정의
OPERATOR_GROUPS = [
    "Hourly statistics", "Daily statistics", "Monthly statistics",
    "Seasonal statistics", "Annual statistics", "Temporal operators",
    "Climate Analysis"
]

# 연산자 매핑 (R 패키지 함수명에 대응하는 코드)
OPERATORS = {
    "Hourly statistics": {
        "Hourly mean": "hourmean",
        "Hourly sum": "hoursum"
    },
    "Daily statistics": {
        "Diurnal means": "daymean",
        "Diurnal sums": "daysum",
        "Diurnal maxima": "daymax",
        "Diurnal minima": "daymin"
    },
    "Monthly statistics": {
        "Monthly means": "monmean",
        "Monthly sums": "monsum",
        "Monthly anomalies": "mon.anomaly"
    },
    "Climate Analysis": {
        "Absolute map": "absolute_map",
        "Anomaly map": "anomaly_map",
        "Time Series Plot": "time_series_plot"
    }
}


# ==========================================
# 2. Data Utilities (데이터 처리 로직)
# ==========================================

def extract_tar(file_obj, extract_path):
    """
    업로드된 tar 파일을 지정된 경로에 압축 해제하고
    내부의 .nc (NetCDF) 파일 목록을 반환합니다.
    """
    try:
        with tarfile.open(fileobj=file_obj, mode='r') as tar:
            tar.extractall(path=extract_path)
            # 압축 해제된 파일 중 .nc 파일만 리스트업
            nc_files = [f for f in os.listdir(extract_path) if f.endswith('.nc')]
            return nc_files
    except Exception as e:
        st.error(f"Error extracting tar file: {e}")
        return []


def load_dataset(file_path):
    """
    NetCDF 파일을 xarray Dataset으로 로드합니다.
    """
    try:
        # decode_times=True는 날짜/시간 정보를 자동으로 datetime 객체로 변환합니다.
        # ds = xr.open_dataset(file_path, decode_times=True)
        ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        st.error(f"Error loading NetCDF file: {e}")
        return None


def apply_operator(ds, var_name, operator_code):
    """
    선택된 변수에 대해 연산자(Operator)를 적용합니다.
    R 패키지(cmsafops)의 기능을 xarray로 대체 구현한 것입니다.
    """
    da = ds[var_name]

    # 연산 로직 구현
    if operator_code == "monmean":
        # 월별 평균 (Monthly Mean)
        # 1MS: Month Start 빈도
        return da.resample(time="1MS").mean(dim="time")

    elif operator_code == "monsum":
        # 월별 합계 (Monthly Sum)
        return da.resample(time="1MS").sum(dim="time")

    elif operator_code == "daymean":
        # 일별 평균 (Daily Mean)
        return da.resample(time="1D").mean(dim="time")

    elif operator_code == "mon.anomaly":
        # 월별 편차 (Monthly Anomaly)
        # 1. 월별 기후값(Climatology) 계산
        climatology = da.groupby("time.month").mean("time")
        # 2. 원본 데이터에서 기후값을 뺌
        return da.groupby("time.month") - climatology

    # 추가적인 연산자 구현 가능 (예: hourmean, yearmean 등)

    # 구현되지 않은 연산자의 경우 원본 데이터 반환 (또는 에러 처리)
    return da


# ==========================================
# 3. Plotting Utilities (시각화 로직)
# ==========================================

def plot_map(data_array, time_step=0, title="Map Plot"):
    """
    2D 지도 시각화 (Cartopy + Matplotlib)
    """
    # 시간 차원이 있다면 특정 시간 단계(time_step)를 선택
    if 'time' in data_array.dims:
        # 시간이 1개만 있거나 time_step 인덱스가 유효한지 확인
        if data_array.sizes['time'] > time_step:
            data_slice = data_array.isel(time=time_step)
        else:
            data_slice = data_array.isel(time=0)
    else:
        data_slice = data_array

    # 그래프 생성
    fig = plt.figure(figsize=(10, 6))

    # 투영법 설정 (PlateCarree: 일반적인 위경도 도법)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 해안선 및 국경 추가
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    # 데이터 플로팅
    # xarray의 plot 메서드는 cartopy와 잘 연동됩니다.
    p = data_slice.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cbar_kwargs={'label': data_array.attrs.get('units', '')},
        cmap='viridis'  # 컬러맵 설정
    )

    # 제목 설정 (변수명 + 시간)
    time_str = str(data_slice.time.values)[:10] if 'time' in data_slice.coords else ''
    ax.set_title(f"{title}  {time_str}")

    return fig


def plot_timeseries(data_array, title="Time Series"):
    """
    1D 시계열 시각화 (공간 평균)
    """
    # 위도(lat)/경도(lon) 차원이 있다면 공간 평균을 수행하여 1D 시계열로 변환
    dims_to_mean = []
    if 'lat' in data_array.dims: dims_to_mean.append('lat')
    if 'lon' in data_array.dims: dims_to_mean.append('lon')

    if dims_to_mean:
        ts = data_array.mean(dim=dims_to_mean)
    else:
        ts = data_array

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 4))
    ts.plot.line(ax=ax, hue=None, marker='o', markersize=3)

    ax.set_title(f"Time Series (Spatial Mean): {title}")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    return fig


# ==========================================
# 4. Main Application Logic (앱 실행 로직)
# ==========================================

def main():
    # 페이지 기본 설정
    st.set_page_config(page_title="CM SAF R Toolbox (Python Ver)", layout="wide")

    # 세션 상태(Session State) 초기화
    # Streamlit은 매번 코드를 다시 실행하므로, 단계(Step)나 데이터 경로를 기억하기 위해 사용
    if 'step' not in st.session_state:
        st.session_state.step = 'Prepare'
    if 'nc_file_path' not in st.session_state:
        st.session_state.nc_file_path = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None

    # --- 사이드바 (네비게이션) ---
    with st.sidebar:
        st.title("☁️ CM SAF Toolbox")

        # 단계별 버튼 (누르면 해당 단계로 session_state 변경)
        if st.button("1. Prepare", use_container_width=True):
            st.session_state.step = 'Prepare'
        if st.button("2. Analyze", use_container_width=True):
            st.session_state.step = 'Analyze'
        if st.button("3. Visualize", use_container_width=True):
            st.session_state.step = 'Visualize'

        st.markdown("---")
        st.info("Python Streamlit Port of\nCM SAF R Toolbox")

    # --- 1. PREPARE 단계 ---
    if st.session_state.step == 'Prepare':
        st.header("1. Prepare Data")
        st.markdown("Tar 파일(.tar) 또는 NetCDF 파일(.nc)을 업로드하여 분석을 준비합니다.")

        uploaded_file = st.file_uploader("Choose a file", type=['tar', 'nc'])

        if uploaded_file:
            # 임시 디렉토리를 생성하여 파일 처리 (안전한 파일 핸들링)
            # 주의: 실제 서비스에서는 영구 저장소나 S3 등을 고려해야 합니다.
            # 여기서는 간소화를 위해 tempfile을 사용하며, 앱이 재시작되면 파일이 사라질 수 있습니다.

            # Streamlit Cloud 등에서는 파일 경로 유지가 까다로울 수 있으므로
            # session_state에 바이트 데이터 자체를 저장하거나 캐싱 전략이 필요할 수 있으나,
            # 로컬 실행을 가정하고 temp 파일 경로를 사용합니다.

            temp_dir = tempfile.mkdtemp()
            file_path = os.path.join(temp_dir, uploaded_file.name)

            # 파일 저장
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Tar 파일 처리
            if uploaded_file.name.endswith('.tar'):
                st.spinner("Extracting tar file...")
                nc_files = extract_tar(uploaded_file, temp_dir)

                if nc_files:
                    st.success(f"Extracted {len(nc_files)} NetCDF files.")
                    selected_file = st.selectbox("Select file to process", nc_files)
                    full_path = os.path.join(temp_dir, selected_file)
                    st.session_state.nc_file_path = full_path
                else:
                    st.warning("No .nc files found in the tar archive.")

            # NC 파일 처리
            else:
                st.session_state.nc_file_path = file_path
                st.success("NetCDF file loaded successfully.")

            # 파일 로드 및 미리보기
            if st.session_state.nc_file_path and os.path.exists(st.session_state.nc_file_path):
                ds = load_dataset(st.session_state.nc_file_path)
                if ds:
                    st.write("### Dataset Overview")
                    st.write(ds)  # xarray dataset의 요약 정보 출력
                else:
                    st.error("Failed to read the NetCDF file.")

    # --- 2. ANALYZE 단계 ---
    elif st.session_state.step == 'Analyze':
        st.header("2. Analyze Data")

        if not st.session_state.nc_file_path or not os.path.exists(st.session_state.nc_file_path):
            st.warning("⚠️ 먼저 [Prepare] 단계에서 파일을 업로드하고 선택해주세요.")
        else:
            # 데이터 로드
            ds = load_dataset(st.session_state.nc_file_path)

            if ds:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.subheader("Settings")
                    # 변수(Variable) 선택
                    # data_vars만 추출 (좌표계 변수 제외)
                    vars_list = list(ds.data_vars)
                    selected_var = st.selectbox("Select Variable", vars_list)

                    # 연산자 그룹 선택
                    op_group = st.selectbox("Select Operator Group", OPERATOR_GROUPS)

                    # 세부 연산자 선택
                    if op_group in OPERATORS:
                        op_name = st.selectbox("Select Operator", list(OPERATORS[op_group].keys()))
                        op_code = OPERATORS[op_group][op_name]
                    else:
                        op_name = "No operators available"
                        op_code = None

                    # 적용 버튼
                    if st.button("Apply Operator", type="primary"):
                        if op_code:
                            with st.spinner("Calculating..."):
                                # 연산 수행
                                result_da = apply_operator(ds, selected_var, op_code)
                                # 결과 저장 (메모리에 xarray DataArray로 저장)
                                st.session_state.processed_data = result_da
                                st.success("Calculation Complete!")
                        else:
                            st.warning("This operator is not yet implemented.")

                with col2:
                    st.subheader("Analysis Result Info")
                    if st.session_state.processed_data is not None:
                        st.info("결과 데이터가 생성되었습니다.")
                        st.write(st.session_state.processed_data)
                        st.markdown("👉 **[Visualize]** 탭으로 이동하여 결과를 확인하세요.")
                    else:
                        st.write("좌측에서 연산을 선택하고 실행해주세요.")

    # --- 3. VISUALIZE 단계 ---
    elif st.session_state.step == 'Visualize':
        st.header("3. Visualize Results")

        data_to_plot = st.session_state.processed_data

        if data_to_plot is None:
            st.warning("⚠️ 먼저 [Analyze] 단계에서 연산을 수행하여 결과를 생성해주세요.")
        else:
            # 시각화 옵션 선택
            st.subheader("Plot Options")
            col_opt1, col_opt2 = st.columns(2)

            with col_opt1:
                plot_type = st.radio("Plot Type", ["Map (2D)", "Time Series (1D)"], horizontal=True)

            # Map(2D) 선택 시 시간 슬라이더 표시
            if plot_type == "Map (2D)":
                if 'time' in data_to_plot.dims and data_to_plot.sizes['time'] > 1:
                    time_len = data_to_plot.sizes['time']
                    # 슬라이더로 시간 인덱스 선택
                    time_idx = st.slider(
                        "Select Time Step",
                        0, time_len - 1, 0,
                        format=f"Index %d"
                    )
                    # 선택된 시간 표시
                    selected_time = str(data_to_plot.time.values[time_idx])
                    st.caption(f"Selected Time: {selected_time}")
                else:
                    time_idx = 0

                # 지도 그리기
                fig = plot_map(data_to_plot, time_idx, title=data_to_plot.name)
                st.pyplot(fig)

            elif plot_type == "Time Series (1D)":
                # 시계열 그리기
                fig = plot_timeseries(data_to_plot, title=data_to_plot.name)
                st.pyplot(fig)

            # 다운로드 버튼 (결과 데이터를 NetCDF로 저장)
            st.markdown("---")
            # 메모리상의 데이터를 bytes로 변환하여 다운로드 제공
            # xarray -> netcdf bytes
            try:
                nc_bytes = data_to_plot.to_netcdf()
                st.download_button(
                    label="📥 Download Result (.nc)",
                    data=nc_bytes,
                    file_name="result.nc",
                    mime="application/x-netcdf"
                )
            except Exception as e:
                st.error(f"다운로드 준비 중 오류 발생: {e}")


if __name__ == "__main__":
    main()