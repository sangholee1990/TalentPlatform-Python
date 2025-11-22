import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
from datetime import date

# ==============================================================================
# 1. Configuration & Global Variables
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
    "Daily statistics": ["dayavg", "daymax", "daymean"],
    "Temporal operators": ["timavg", "timmax", "timmean", "trend"],
    "Selection": ["sellonlatbox", "selpoint", "selperiod", "selyear"],
    "Climate Analysis": ["absolute_map", "anomaly_map", "climatology_map", "warming_stripes_plot"],
    "Compare Data": ["cmsaf.diff.absolute", "cmsaf.scatter", "cmsaf.time.series", "cmsaf.stats"]
}

MONTHS_LIST = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# R 코드의 설명 문자열
DESCRIPTION_STRING = """
The CM SAF R TOOLBOX 3.6.0 -- 'Funny, It Worked Last Time...'

The intention of the CM SAF R Toolbox is to help you using
CM SAF NetCDF formatted climate data

This includes:
  1. Preparation of data files.
  2. Analysis and manipulation of prepared data.
  3. Visualization of the results.
"""

# 가상의 파일 정보 (R Shiny의 getUserOptions 결과와 유사)
MOCK_FILE_INFO = {
    'date_from': date(2010, 1, 1),
    'date_to': date(2020, 12, 31),
    'lon_range': (-180.0, 180.0),
    'lat_range': (-90.0, 90.0),
    'variables': ['tsa', 'clt', 'rsds', 'sst']
}
MOCK_FILE_PATH = "prepared_output.nc"

# ==============================================================================
# 2. UI Layout (R Shiny의 ui.R 구조 반영)
# ==============================================================================

# Dash 앱 초기화
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)

# --------------------------
# A. 사이드바 레이아웃 정의 (R Shiny의 sidebarPanel)
# --------------------------
SIDEBAR = dbc.Col(
    [
        # 로고 및 홈 버튼 (Streamlit 예시에서와 같이 이미지 경로 필요)
        html.Div(
            html.Img(src="/assets/R_Toolbox_Logo.png", style={"width": "100%", "padding": "10px", "cursor": "pointer"}),
            id="home-image-div"
        ),
        dbc.Button("Prepare", id="btn-prepare", className="btn btn-primary btn-lg btn-block",
                   style={"fontSize": "30px", "marginBottom": "10px"}),
        dbc.Button("Analyze", id="btn-analyze", className="btn btn-primary btn-lg btn-block",
                   style={"fontSize": "30px", "marginBottom": "10px"}),
        dbc.Button("Visualize", id="btn-visualize", className="btn btn-primary btn-lg btn-block",
                   style={"fontSize": "30px", "marginBottom": "10px"}),
        dbc.Button("EXIT", id="btn-exit", className="btn btn-danger btn-lg btn-block",
                   style={"fontSize": "30px", "marginBottom": "10px"}),

        html.Hr(),
        # 다운로드 및 설정 요소 (초기에는 숨김 상태)
        html.Div([
            dbc.Button("Download the session files.", id="btn-download-session", n_clicks=0, color="secondary",
                       className="mt-3 w-100")
        ], id="downloader-div", style={'display': 'none'}),
        dbc.Button("View or edit the user directory.", id="btn-modify-user-dir", n_clicks=0, color="info",
                   className="mt-2 w-100"),
    ],
    width=3,
    style={"padding": "1rem", "height": "100vh", "position": "sticky", "top": 0}
)

# --------------------------
# B. 메인 콘텐츠 레이아웃 정의 (R Shiny의 mainPanel)
# --------------------------

# 1. 홈 패널
HOME_CONTENT = html.Div(
    id="panel-home",
    children=[
        html.Pre(DESCRIPTION_STRING)
    ]
)

# 2. 준비 섹션 (Prepare)
PREPARE_CONTENT = html.Div(
    id="panel-prepare",
    children=[
        # 파일 선택 단계 (panel_prepareGo 대체)
        html.Div(id="panel-prepare-go", children=[
            html.H2("Prepare"),
            html.P("Please select a TAR file (.tar), a NetCDF file (.nc) or a NetCDF URL to start the preparation."),
            html.Div([
                dbc.Button("Choose a .tar-file...", id="btn-tar-local", n_clicks=0, className="me-2"),
                html.Span("or", className="mx-2"),
                dbc.Button("Choose .nc-files...", id="btn-nc-local", n_clicks=0, className="me-2"),
                html.Span("or", className="mx-2"),
                dbc.Button("Enter .nc file URL...", id="btn-nc-url", n_clicks=0),
            ], className="mb-4")
        ]),

        # 날짜/영역 선택 단계 (panel_prepareInput2 대체)
        html.Div(id="panel-prepare-input-details", style={'display': 'none'}, children=[
            html.H3("2. Define Extraction Details"),
            dcc.DatePickerRange(
                id='date-range-picker',
                start_date=MOCK_FILE_INFO['date_from'],
                end_date=MOCK_FILE_INFO['date_to'],
                min_date_allowed=MOCK_FILE_INFO['date_from'],
                max_date_allowed=MOCK_FILE_INFO['date_to'],
                display_format='YYYY-MM-DD',
                style={'marginBottom': '15px'}
            ),
            dcc.Dropdown(
                id='variable-select-prepare',
                options=[{'label': v, 'value': v} for v in MOCK_FILE_INFO['variables']],
                value=MOCK_FILE_INFO['variables'][0],
                clearable=False
            ),
            html.H4("Spatial Range"),
            dcc.RangeSlider(
                id='lon-range-slider',
                min=MOCK_FILE_INFO['lon_range'][0],
                max=MOCK_FILE_INFO['lon_range'][1],
                step=0.05,
                value=MOCK_FILE_INFO['lon_range'],
                marks={i: f'{i}°' for i in range(-180, 181, 45)}
            ),
            html.Div(id='lon-range-output', style={'marginBottom': '10px'}),

            dcc.RangeSlider(
                id='lat-range-slider',
                min=MOCK_FILE_INFO['lat_range'][0],
                max=MOCK_FILE_INFO['lat_range'][1],
                step=0.05,
                value=MOCK_FILE_INFO['lat_range'],
                marks={i: f'{i}°' for i in range(-90, 91, 30)}
            ),
            html.Div(id='lat-range-output', style={'marginBottom': '15px'}),

            dbc.Select(
                id="output-format-select",
                options=[{"label": "NetCDF4", "value": 4}, {"label": "NetCDF3", "value": 3}],
                value=4,
                className="mb-3"
            ),
            dbc.Button("Create output file!", id="btn-create-output", color="success", size="lg")
        ])
    ]
)

# 3. 분석 섹션 (Analyze)
ANALYZE_CONTENT = html.Div(
    id="panel-analyze",
    children=[
        html.H2("Analyze"),
        html.Div(id='analyze-file-status', children=[
            html.P(f"Currently analyzing: {MOCK_FILE_PATH}")
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id='analyze-variable-select',
                    options=[{'label': v, 'value': v} for v in MOCK_FILE_INFO['variables']],
                    value=MOCK_FILE_INFO['variables'][0],
                    clearable=False
                ),
                dbc.Select(
                    id="operator-group-select",
                    options=[{"label": g, "value": g} for g in OPERATOR_GROUPS_ANALYZE.keys()],
                    value="Daily statistics",
                    className="mt-3"
                ),
                dcc.Dropdown(
                    id="operator-select",
                    options=[{"label": o, "value": o} for o in OPERATOR_GROUPS_ANALYZE['Daily statistics']],
                    value=OPERATOR_GROUPS_ANALYZE['Daily statistics'][0],
                    clearable=False
                ),
                # 동적 옵션 영역 (R Shiny의 숨겨진 div 및 conditionalPanel 역할)
                html.Div(id='analyze-dynamic-options', className="mt-3", children=[
                    # 예시: constant 입력
                    dbc.InputGroup([
                        dbc.InputGroupText("Constant Value"),
                        dbc.Input(id="constant-input", type="number", value=1.0)
                    ], id="constant-group", style={'display': 'none'})
                ]),

                html.Hr(),
                dbc.Select(
                    id="analyze-output-format-select",
                    options=[{"label": "NetCDF4", "value": 4}, {"label": "NetCDF3", "value": 3}],
                    value=4,
                    className="mb-3"
                ),
                dbc.Checklist(
                    options=[
                        {"label": "Apply another operator afterwards?", "value": 1},
                        {"label": "Visualize the results right away?", "value": 2, "disabled": False, 'value': True},
                    ],
                    value=[2],
                    id="analyze-post-options",
                    inline=False,
                    className="mb-4"
                ),
                dbc.Button("Apply operator", id="btn-apply-operator", color="success", size="lg", className="w-100")

            ], width=5),
            dbc.Col([
                html.H3("File Information"),
                html.Pre(f"Short Info for: {MOCK_FILE_PATH}", id="nc-short-info"),
                html.H4("Applied Operators History"),
                # 데이터프레임 (R Shiny의 tableOutput)
                dbc.Table.from_dataframe(
                    pd.DataFrame(columns=["Operator", "Output File"]),
                    striped=True, bordered=True, hover=True, id='applied-operators-table'
                ),
            ], width=7)
        ])
    ],
    style={'display': 'none'}  # 초기에는 숨김
)

# 4. 시각화 섹션 (Visualize)
VISUALIZE_CONTENT = html.Div(
    id="panel-visualize",
    children=[
        html.H2("Visualize"),
        dbc.Tabs(id="visualize-tabs", active_tab="tab-plot", children=[
            dbc.Tab(label="Plot", tab_id="tab-plot", children=[
                dbc.Row([
                    dbc.Col([
                        html.H4("Plot Options"),
                        dbc.InputGroup([dbc.InputGroupText("Time Step"),
                                        dbc.Input(id="vis-timestep", value="2015-07-01", type="text")]),
                        dbc.Select(id="vis-projection", options=[{"label": "Rectangular", "value": "rect"},
                                                                 {"label": "Orthographic", "value": "ortho"}],
                                   value="rect"),
                        dbc.InputGroup([dbc.InputGroupText("Title"),
                                        dbc.Input(id="vis-title", value="Variable Name", type="text")]),
                        dbc.InputGroup(
                            [dbc.InputGroupText("Colors"), dbc.Input(id="vis-num-colors", value=32, type="number")]),
                        # Dash에서는 색상 피커나 팔레트 이미지를 위한 추가 컴포넌트(예: dash-color-picker)가 필요
                        dbc.Button("Download Image", id="btn-download-plot", className="mt-3 w-100")
                    ], width=4),
                    dbc.Col([
                        html.H4("Main Plot"),
                        # Plotly 그래프 컴포넌트
                        dcc.Graph(id='main-data-plot', figure={
                            'layout': {'title': 'Data Visualization (Simulated)'}
                        }),
                    ], width=8)
                ])
            ]),
            dbc.Tab(label="Statistics", tab_id="tab-stats", children=[
                html.H4("Summary Statistics"),
                html.Pre("Mean: ...\nMedian: ...", id="vis-stats")
            ]),
            dbc.Tab(label="File Summary", tab_id="tab-summary", children=[
                html.H4("File Details"),
                html.Pre(f"File Info for: {MOCK_FILE_PATH}", id="vis-file-summary")
            ])
        ])
    ],
    style={'display': 'none'}  # 초기에는 숨김
)

# --------------------------
# C. 앱 레이아웃 설정
# --------------------------
app.layout = dbc.Container(
    [
        # R Shiny의 reactive value 저장을 위한 dcc.Store
        dcc.Store(id='page-state', data={'page': 'Home', 'output_file': MOCK_FILE_PATH, 'file_info': MOCK_FILE_INFO,
                                         'operation_count': 0}),
        dcc.Store(id='operation-history-store', data=[]),

        dbc.Row([
            SIDEBAR,
            dbc.Col(
                html.Div(id="page-content",
                         children=[HOME_CONTENT, PREPARE_CONTENT, ANALYZE_CONTENT, VISUALIZE_CONTENT]),
                width=9,
                style={"padding": "1rem"}
            )
        ])
    ],
    fluid=True,
    style={"height": "100vh"}
)


# ==============================================================================
# 3. Callbacks (R Shiny의 server.R 로직 반영)
# ==============================================================================

# --------------------------
# A. 메인 페이지 전환 콜백 (R Shiny의 observeEvent(input$prepare/analyze/visualize))
# --------------------------
@app.callback(
    [
        Output('panel-home', 'style'),
        Output('panel-prepare', 'style'),
        Output('panel-analyze', 'style'),
        Output('panel-visualize', 'style'),
        Output('page-state', 'data'),
    ],
    [
        Input('btn-prepare', 'n_clicks'),
        Input('btn-analyze', 'n_clicks'),
        Input('btn-visualize', 'n_clicks'),
        Input('home-image-div', 'n_clicks'),
    ],
    [State('page-state', 'data')]
)
def render_page_content(prep_clicks, analyze_clicks, viz_clicks, home_clicks, current_state):
    # n_clicks를 사용하여 어떤 버튼이 클릭되었는지 확인 (R Shiny의 observeEvent와 유사)
    ctx = dash.callback_context
    if not ctx.triggered:
        # 초기 로드 시
        triggered_id = 'panel-home'
    else:
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    styles = {'display': 'none'}

    if triggered_id in ['btn-prepare', 'btn-tar-local', 'btn-nc-local', 'btn-nc-url']:
        current_state['page'] = 'Prepare'
    elif triggered_id == 'btn-analyze':
        current_state['page'] = 'Analyze'
    elif triggered_id == 'btn-visualize':
        current_state['page'] = 'Visualize'
    elif triggered_id == 'home-image-div':
        current_state['page'] = 'Home'
    else:
        # 기본 페이지 유지
        pass

    # 스타일 업데이트
    home_style = styles.copy()
    prepare_style = styles.copy()
    analyze_style = styles.copy()
    visualize_style = styles.copy()

    if current_state['page'] == 'Home':
        home_style['display'] = 'block'
    elif current_state['page'] == 'Prepare':
        prepare_style['display'] = 'block'
    elif current_state['page'] == 'Analyze':
        analyze_style['display'] = 'block'
    elif current_state['page'] == 'Visualize':
        visualize_style['display'] = 'block'

    return home_style, prepare_style, analyze_style, visualize_style, current_state


# --------------------------
# B. 준비 섹션의 단계 전환 콜백 (R Shiny의 untarAndUnzip/createOutput 후 전환)
# --------------------------
@app.callback(
    Output('panel-prepare-input-details', 'style'),
    [Input('btn-tar-local', 'n_clicks'), Input('btn-nc-local', 'n_clicks'), Input('btn-nc-url', 'n_clicks')]
)
def toggle_prepare_details(tar_clicks, nc_clicks, url_clicks):
    if tar_clicks > 0 or nc_clicks > 0 or url_clicks > 0:
        # 파일/URL 선택 시 세부 정보 입력 필드 표시
        return {'display': 'block'}
    return {'display': 'none'}


# --------------------------
# C. 분석 섹션 오퍼레이터 동적 옵션 콜백 (R Shiny의 observeEvent(operatorInput_value()))
# --------------------------
@app.callback(
    [
        Output('operator-select', 'options'),
        Output('operator-select', 'value'),
        Output('analyze-dynamic-options', 'children'),
    ],
    [Input('operator-group-select', 'value')]
)
def update_analyze_options(selected_group):
    options = OPERATOR_GROUPS_ANALYZE.get(selected_group, [])
    options_dash = [{'label': o, 'value': o} for o in options]
    default_value = options[0] if options else None

    # 동적 옵션: 예시로 Mathematical operators를 선택하면 constant input 표시
    dynamic_options = []
    if selected_group == "Mathematical operators":
        dynamic_options.append(
            dbc.InputGroup([
                dbc.InputGroupText("Constant Value"),
                dbc.Input(id="constant-input", type="number", value=1.0)
            ])
        )

    return options_dash, default_value, dynamic_options


# --------------------------
# D. 오퍼레이터 적용 콜백 (R Shiny의 observeEvent(input$applyOperator))
# --------------------------
@app.callback(
    [
        Output('operation-history-store', 'data'),
        Output('analyze-file-status', 'children'),
        Output('btn-apply-operator', 'n_clicks'),  # 버튼 클릭 횟수 초기화
    ],
    [Input('btn-apply-operator', 'n_clicks')],
    [
        State('analyze-variable-select', 'value'),
        State('operator-select', 'value'),
        State('page-state', 'data'),
        State('operation-history-store', 'data'),
    ],
    prevent_initial_call=True
)
def apply_operator(n_clicks, variable, operator, page_state, history):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    # 가상의 출력 파일 생성
    current_time = pd.Timestamp.now().strftime("%H%M%S")
    new_output_file = f"{variable}_{operator}_{current_time}.nc"

    # 이력 업데이트
    new_row = {"Operator": operator, "Output File": new_output_file}
    history.append(new_row)

    # 페이지 상태 (output_file) 업데이트
    page_state['output_file'] = new_output_file
    page_state['operation_count'] = page_state.get('operation_count', 0) + 1

    # Analyze 파일 상태 메시지 업데이트
    status_message = html.P(f"Successfully applied {operator}. New file: {new_output_file}")

    # Visualize 즉시 전환 로직 (R Shiny의 instantlyVisualize 체크박스)
    post_options = dash.callback_context.states_list[0]['id']  # State로 체크박스 값 가져오는 로직 생략

    # 여기서는 간단히 항상 visualize로 전환
    # if 2 in post_options_value: # 'Visualize the results right away'가 선택된 경우
    #     page_state['page'] = 'Visualize'

    # 버튼 클릭 횟수 초기화 및 리턴
    return history, status_message, 0


# --------------------------
# E. 이력 테이블 업데이트 콜백
# --------------------------
@app.callback(
    Output('applied-operators-table', 'children'),
    [Input('operation-history-store', 'data')]
)
def update_history_table(data):
    if not data:
        return dbc.Table.from_dataframe(pd.DataFrame(columns=["Operator", "Output File"]), striped=True, bordered=True,
                                        hover=True)

    df = pd.DataFrame(data)
    return dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)


if __name__ == '__main__':
    # Streamlit과 달리 Dash는 서버 실행만으로 충분
    app.run_server(debug=True, port=9050)