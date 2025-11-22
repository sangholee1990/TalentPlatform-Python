import streamlit as st
import pandas as pd
import os
from datetime import date

# ==============================================================================
# 1. Configuration & Global Variables (global.R 및 ui.R의 상단부 반영)
# ==============================================================================

# R 코드의 operatorGroups 및 operatorGroupsPrepare를 Python Dict로 정의
OPERATOR_GROUPS_PREPARE = {
    "Diurnal means": "daymean",
    "Diurnal averages": "dayavg",
    "Diurnal sums": "daysum",
    "Monthly means": "monmean",
    "Monthly averages": "monavg",
    "Monthly sums": "monsum",
    "Time selection mean": "timselmean",
    "Time selection sum": "timselsum"
}

OPERATOR_GROUPS_ANALYZE = {
    "Hourly statistics": ["hourmean", "hoursum"],
    "Daily statistics": ["daymean", "daysum", "daymax", "daymin"],
    "Temporal operators": ["timmean", "timmax", "timmin", "trend", "cmsaf.mk.test"],
    "Selection": ["sellonlatbox", "selpoint", "selperiod", "selyear"],
    "Climate Analysis": ["absolute_map", "anomaly_map", "climatology_map", "warming_stripes_plot"],
    "Compare Data": ["cmsaf.diff.absolute", "cmsaf.scatter", "cmsaf.time.series", "cmsaf.stats"]
}

MONTHS_LIST = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# R 코드의 isRunningLocally 로직을 Streamlit에서는 간략화
IS_RUNNING_LOCALLY = True  # 실제 환경에 따라 변경 가능

# R 코드의 설명 문자열
DESCRIPTION_STRING = """
The CM SAF R TOOLBOX 3.6.0 -- 'Funny, It Worked Last Time...'

The intention of the CM SAF R Toolbox is to help you using
CM SAF NetCDF formatted climate data

This includes:
1. **Preparation** of data files.
2. **Analysis** and manipulation of prepared data.
3. **Visualization** of the results.

To begin, choose a .tar file or a .nc file in the prepare section or jump
right in and analyze or visualize a .nc file.

- Steffen Kothe - 2025-10-20 -
"""


# ==============================================================================
# 2. Layout (R Shiny의 fluidPage, sidebarLayout, mainPanel 반영)
# ==============================================================================

def setup_page():
    st.set_page_config(layout="wide")

    # Streamlit의 사이드바 (R Shiny의 sidebarPanel)
    with st.sidebar:
        # R_Toolbox_Logo.png 이미지는 로컬 환경에 있어야 함
        # st.image("R_Toolbox_Logo.png")
        st.markdown("<h1 style='text-align: center;'>CM SAF Toolbox</h1>", unsafe_allow_html=True)

        st.markdown("---")

        # R Shiny의 actionButton들을 Streamlit의 버튼과 상태 변수로 구현
        st.session_state.page = st.radio(
            "Navigation",
            options=["Home", "Prepare", "Analyze", "Visualize", "EXIT"],
            index=0,
            key="navigation_radio"
        )

        st.markdown("---")

        # 다운로더 및 기타 숨겨진 요소 (간소화)
        if st.session_state.page in ["Analyze", "Visualize"] and st.session_state.get('output_file'):
            if IS_RUNNING_LOCALLY:
                st.download_button(
                    label="Download the session files.",
                    data="Simulated session data...",
                    file_name="session_files.tar",
                    mime="application/x-tar"
                )
            st.button("View or edit the user directory.")

    # 메인 콘텐츠 영역 (R Shiny의 mainPanel)
    st.markdown(
        """
        <style>
        .reportview-container .main {
            padding-top: 0;
        }
        .stButton button {
            width: 100%;
            font-size: 30px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.page == "Home":
        st.header("Welcome to the CM SAF R Toolbox")
        st.markdown(f"```\n{DESCRIPTION_STRING}\n```")

    elif st.session_state.page == "Prepare":
        render_prepare_section()

    elif st.session_state.page == "Analyze":
        render_analyze_section()

    elif st.session_state.page == "Visualize":
        render_visualize_section()

    elif st.session_state.page == "EXIT":
        st.error("Application Exit - (In a real deployment, this would stop the Shiny app).")


# ==============================================================================
# 3. Prepare Section (R Shiny의 panel_prepareGo 및 하위 패널 반영)
# ==============================================================================

def render_prepare_section():
    st.header("1. Prepare")
    st.write("Please select a file or enter a URL to start the preparation process.")

    # 파일 선택 및 URL 입력 (R Shiny의 tarFile/ncFile/ncURL 버튼)
    file_type = st.radio(
        "Input Type",
        options=["Upload .tar file", "Upload .nc file", "Enter .nc URL"],
        key="prepare_input_type",
        horizontal=True
    )

    if file_type == "Upload .tar file":
        tar_file = st.file_uploader("Choose a .tar-file...", type=["tar"])
        if tar_file:
            st.session_state.prepare_file = tar_file
            handle_file_upload(tar_file, "tar")

    elif file_type == "Upload .nc file":
        nc_file = st.file_uploader("Choose .nc-files...", type=["nc"])
        if nc_file:
            st.session_state.prepare_file = nc_file
            handle_file_upload(nc_file, "nc")

    elif file_type == "Enter .nc URL":
        st.session_state.nc_url_text = st.text_input(
            "Please enter a URL to a NetCDF (.nc) file",
            placeholder="Enter URL..."
        )
        if st.button("Connect to URL") and st.session_state.nc_url_text:
            handle_url_connect(st.session_state.nc_url_text)

    # --- 파일 선택 후 로직 (panel_prepareInput1, panel_prepareInput2 등) ---

    if 'file_info' in st.session_state and st.session_state.file_info:
        st.subheader("2. Date and Spatial Selection")
        info = st.session_state.file_info

        # 날짜 범위 선택 (R Shiny의 dateRange_ui)
        start_date = info.get('date_from', date(2000, 1, 1))
        end_date = info.get('date_to', date(2020, 1, 1))

        date_range = st.date_input(
            "Select a date range to prepare.",
            value=(start_date, end_date),
            min_value=start_date,
            max_value=end_date
        )

        # 변수 선택 (R Shiny의 variable_ui)
        selected_var = st.selectbox(
            "Choose a variable:",
            options=info.get('variables', ['Variable 1', 'Variable 2'])
        )

        # 위도/경도 범위 선택 (R Shiny의 lonRange_ui, latRange_ui)
        min_lon, max_lon = info.get('lon_range', (-180.0, 180.0))
        min_lat, max_lat = info.get('lat_range', (-90.0, 90.0))

        lon_range = st.slider(
            "Select a longitude range.",
            min_value=min_lon,
            max_value=max_lon,
            value=(min_lon, max_lon),
            step=0.05
        )

        lat_range = st.slider(
            "Select a latitude range.",
            min_value=min_lat,
            max_value=max_lat,
            value=(min_lat, max_lat),
            step=0.05
        )

        # 집계 옵션 (R Shiny의 checkboxInput_aggregate, operatorGroupsPrepare)
        aggregate = st.checkbox("Do you want to aggregate the data before analysis?")
        if aggregate:
            st.selectbox(
                "Select an operator for aggregation",
                options=list(OPERATOR_GROUPS_PREPARE.keys()),
                key="prepare_aggregation_operator"
            )

        # 출력 포맷 (R Shiny의 selectInput("outputFormat"))
        st.selectbox(
            "Select output format",
            options=["NetCDF4", "NetCDF3"],
            index=0,
            key="prepare_output_format"
        )

        if st.button("Create output file!"):
            # R 코드의 creatingOutputfile 로직을 실행
            st.success("Output file generation initiated (simulated).")
            st.session_state.output_file = f"{selected_var}_prepared.nc"


def handle_file_upload(uploaded_file, file_type):
    # 파일 정보 추출 로직 (R 코드의 extractDateRange, getUserOptions 시뮬레이션)
    st.success(f"File uploaded: {uploaded_file.name}")

    # 실제 환경에서는 여기서 파일 내용을 읽고 메타데이터를 추출해야 합니다.
    # 여기서는 단순한 더미 데이터로 세션 상태를 업데이트합니다.
    st.session_state.file_info = {
        'date_from': date(2010, 1, 1),
        'date_to': date(2020, 12, 31),
        'lon_range': (-180.0, 180.0),
        'lat_range': (-90.0, 90.0),
        'variables': ['tsa', 'clt', 'rsds', 'sst']
    }


def handle_url_connect(url):
    st.info(f"Attempting to connect to URL: {url} (Simulated)")
    # 실제 환경에서는 httr::HEAD, ncdf4::nc_open 등을 사용하여 유효성 검사 수행

    if "valid" in url.lower():
        st.success("Valid NetCDF (.nc) URL")
        st.session_state.file_info = {
            'date_from': date(2015, 6, 1),
            'date_to': date(2023, 6, 30),
            'lon_range': (-50.0, 50.0),
            'lat_range': (-45.0, 45.0),
            'variables': ['precipitation', 'temperature']
        }
        st.session_state.connected_url = url
    else:
        st.error("URL does not appear to lead to a NetCDF (.nc) file.")
        st.session_state.file_info = None


# ==============================================================================
# 4. Analyze Section (R Shiny의 panel_analyzeGo 및 panel_analyze 반영)
# ==============================================================================

def render_analyze_section():
    st.header("2. Analyze")

    # --------------------------------
    # 파일 선택 로직 (panel_analyzeGo)
    # --------------------------------
    analyze_file_path = st.session_state.get('output_file', None)

    if analyze_file_path:
        st.subheader("Use Prepared File")
        st.info(f"We prepared the following .nc file for you: **{analyze_file_path}**")
        if st.button("Analyze this file!", key="use_output_analyze"):
            st.session_state.current_analyze_file = analyze_file_path
            st.session_state.analyze_file_loaded = True
            # Streamlit은 상태 관리를 위해 전체 앱을 다시 실행하므로,
            # analyze_file_loaded 상태를 사용하여 UI를 업데이트합니다.

        st.markdown("---")
        st.subheader("OR Select Another File")

    # 파일 업로더/선택기 (R Shiny의 ncFileLocal_analyze/ncFileRemote_analyze)
    uploaded_analyze_file = st.file_uploader("Choose a .nc-file for analysis...", type=["nc"], key="analyze_uploader")
    if uploaded_analyze_file:
        st.session_state.current_analyze_file = uploaded_analyze_file.name
        st.session_state.analyze_file_loaded = True
        st.success(f"File loaded: {uploaded_analyze_file.name}")

    # --------------------------------
    # 분석 옵션 로직 (panel_analyze)
    # --------------------------------
    if st.session_state.get('analyze_file_loaded'):
        st.subheader("3. Apply Operator")

        col1, col2 = st.columns([1, 2])

        with col1:
            # 변수 선택 (R Shiny의 usedVariable)
            st.selectbox(
                "Please choose a variable",
                options=st.session_state.file_info.get('variables', ['tsa']),
                key="analyze_variable"
            )

            # 오퍼레이터 그룹 선택 (R Shiny의 operatorGroup)
            operator_group = st.selectbox(
                "Select a group of operators",
                options=list(OPERATOR_GROUPS_ANALYZE.keys()),
                key="operator_group_select"
            )

            # 오퍼레이터 선택 (R Shiny의 operatorInput)
            operator_choices = OPERATOR_GROUPS_ANALYZE.get(operator_group, [])
            selected_operator = st.selectbox(
                "Select an operator",
                options=operator_choices,
                key="selected_operator"
            )

            # --- 동적 옵션 (R Shiny의 shinyjs::hidden 로직) ---
            if selected_operator == "sellonlatbox" or operator_group == "Climate Analysis":
                st.subheader("Region Selection")
                st.slider("Longitude min/max", -180.0, 180.0, (-10.0, 10.0), key="lon_region_analyze")
                st.slider("Latitude min/max", -90.0, 90.0, (40.0, 60.0), key="lat_region_analyze")

            if selected_operator in ["cmsaf.addc", "cmsaf.subc", "cmsaf.mulc", "cmsaf.divc"]:
                st.number_input("Enter a constant", value=1.0, key="analyze_constant_input")

            # 출력 포맷
            st.selectbox("Select output format", options=["NetCDF4", "NetCDF3"], key="analyze_output_format")

            # 후속 작업 선택 (R Shiny의 applyAnother/instantlyVisualize)
            st.checkbox("Do you want to apply another operator afterwards?", key="apply_another")
            if not st.session_state.apply_another and selected_operator not in ["selpoint.multi",
                                                                                "cmsaf.adjust.two.files"]:
                st.checkbox("Do you want to visualize the results right away?", value=True, key="instant_visualize")

            if st.button("Apply operator", type="primary"):
                # R 코드의 applyOperator 로직 실행
                st.success(f"Operator **{selected_operator}** applied (simulated).")
                st.session_state.output_file = f"result_{selected_operator}.nc"
                if st.session_state.instant_visualize:
                    st.session_state.navigation_radio = "Visualize"
                    st.rerun()

        with col2:
            st.subheader("Short File Information")
            st.code(
                f"File: {st.session_state.current_analyze_file}\nTime steps: 120\nVariables: {st.session_state.file_info.get('variables')}")

            st.subheader("Operator Group Info")
            if operator_group == "Temporal operators":
                st.info("Calculate statistics over all timesteps of a file.")
            elif operator_group == "Compare Data":
                st.info("Compare data of two input files (e.g., spatial vs. station data).")

            if 'operator_history' not in st.session_state:
                st.session_state.operator_history = pd.DataFrame(columns=["Operator", "Value", "Output File"])

            st.subheader("List of applied operators")
            st.dataframe(st.session_state.operator_history, hide_index=True)


# ==============================================================================
# 5. Visualize Section (R Shiny의 panel_visualizeGo 및 visualizePage 반영)
# ==============================================================================

def render_visualize_section():
    st.header("3. Visualize")

    # --------------------------------
    # 파일 선택 로직 (panel_visualizeGo)
    # --------------------------------
    visualize_file_path = st.session_state.get('output_file', None)

    if visualize_file_path:
        st.subheader("Use Prepared File")
        st.info(f"We prepared the following .nc file for you: **{visualize_file_path}**")
        if st.button("Visualize this file!", key="use_output_visualize"):
            st.session_state.current_visualize_file = visualize_file_path
            st.session_state.visualize_file_loaded = True

        st.markdown("---")
        st.subheader("OR Select Another File")

    # 파일 업로더/선택기
    uploaded_visualize_file = st.file_uploader("Choose a .nc-file for visualization...", type=["nc"],
                                               key="visualize_uploader")
    if uploaded_visualize_file:
        st.session_state.current_visualize_file = uploaded_visualize_file.name
        st.session_state.visualize_file_loaded = True
        st.success(f"File loaded: {uploaded_visualize_file.name}")

    # --------------------------------
    # 시각화 옵션 및 플롯 로직 (visualizePage)
    # --------------------------------
    if st.session_state.get('visualize_file_loaded'):
        st.subheader("Visualizer Options")

        file_to_plot = st.session_state.current_visualize_file
        st.markdown(f"**File in view:** `{file_to_plot}`")

        # R Shiny의 tabsetPanel을 Streamlit의 탭으로 구현
        tab_plot, tab_stats, tab_summary, tab_about = st.tabs(["Plot", "Statistics", "File Summary", "About"])

        with tab_plot:
            col_options, col_plot = st.columns([1, 3])

            with col_options:
                st.subheader("Plot Settings")

                # 타임스텝 선택 (R Shiny의 timestep_visualize)
                st.selectbox("Select Time Step", options=[f"Time Step {i + 1}" for i in range(10)],
                             key="visualize_timestep")

                # 색상 바 및 범위 설정 (R Shiny의 PAL, num_brk, num_rmin, num_rmax)
                st.select_slider("Colorbar", options=["Viridis", "Plasma", "Blue-Red"], key="visualize_colorbar")
                st.number_input("Number of Colors", min_value=2, max_value=128, value=32, step=1, key="num_brk")

                # 제목 및 캡션
                st.text_input("Title", value=st.session_state.file_info.get('variables', ['Variable'])[0],
                              key="plot_title")
                st.text_input("Scale Caption", value="Data [Unit]", key="scale_caption")
                st.slider("Font Size", min_value=5, max_value=25, value=12, step=1, key="font_size_2d")

            with col_plot:
                st.subheader("Plot Output (2D Map or 1D Timeseries)")
                # 실제 Plotly/Matplotlib/Cartopy를 사용하여 플롯을 생성하고 st.pyplot 또는 st.plotly_chart로 표시해야 함
                st.warning("Map/Plot rendering simulation. Use plotting libraries for actual output.")
                st.empty()  # 이 자리에 실제 플롯이 들어갑니다.

                # 줌 기능 (R Shiny의 previewSpatialCoveragePlot_vis)
                if st.checkbox("Show Zoom"):
                    st.info("Interactive map to select region for zoom (Simulated).")
                    st.empty()  # 이 자리에 상호작용 가능한 미리보기 플롯이 들어갑니다.

        with tab_stats:
            st.subheader("Statistics")
            st.code("Mean: 15.5\nMedian: 15.0\nStandard deviation: 2.1\n...\n")

        with tab_summary:
            st.subheader("File Summary")
            st.code(
                f"File: {file_to_plot}\n\nShort File Info (cmsafops::ncinfo(nc))\n\nDetailed File Info (cmsafops::ncinfo(nc, 'm'))")

        with tab_about:
            st.subheader("The CM SAF Visualizer")
            st.write(
                "The CM SAF Visualizer is part of the CM SAF R Toolbox. This tool helps you to visualize 1D-timeseries and 2D-maps.")
            st.markdown("[CM SAF R Toolbox URL](https://www.cmsaf.eu/R_toolbox)")


# ==============================================================================
# 6. Main App Execution
# ==============================================================================

if __name__ == "__main__":
    setup_page()