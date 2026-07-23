import sys
import os
import json
import numpy as np
import xarray as xr

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QStackedWidget, QMessageBox, QFrame, QSplitter, QFileDialog, QScrollArea)
from PyQt6.QtCore import Qt, QUrl, QSize
from PyQt6.QtGui import QIcon
from PyQt6.QtWebEngineWidgets import QWebEngineView

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

import io
import contextlib

with contextlib.redirect_stdout(io.StringIO()):
    from qfluentwidgets import (SwitchButton, MSFluentWindow, NavigationItemPosition, setTheme, Theme, setThemeColor,
                                SubtitleLabel, setFont, ComboBox, PushButton, TextEdit,
                                SpinBox, Pivot, SegmentedWidget, LineEdit, TitleLabel, 
                                StrongBodyLabel, CardWidget, FluentIcon, ToolButton, PasswordLineEdit, Slider)

from nmsc_climate_toolbox import nct


class PreprocessInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("PreprocessInterface")
        
        self.files = []
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        title = TitleLabel("전처리 (Preprocessing)")
        main_layout.addWidget(title)
        
        # 1. 상단 탭 분리용 세그먼트 (SegmentedWidget)
        self.segment = SegmentedWidget(self)
        main_layout.addWidget(self.segment, 0, Qt.AlignmentFlag.AlignLeft)
        
        # 2. 탭 내용을 보여줄 StackedWidget
        self.stack = QStackedWidget(self)
        main_layout.addWidget(self.stack)
        
        # ----------------------------------------------------
        # Page 1: 입력자료 탭 (Input Data)
        # ----------------------------------------------------
        page_input = QWidget()
        page_input_layout = QVBoxLayout(page_input)
        page_input_layout.setContentsMargins(0, 10, 0, 0)
        
        splitter_in = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE (Inputs) ---
        scroll_left_in = QScrollArea()
        scroll_left_in.setWidgetResizable(True)
        scroll_left_in.setFrameShape(QFrame.Shape.NoFrame)
        scroll_left_in.setStyleSheet("QScrollArea { background-color: transparent; }")
        
        left_widget_in = QWidget()
        v_left_in = QVBoxLayout(left_widget_in)
        v_left_in.setContentsMargins(0, 0, 10, 0)
        
        card_in = CardWidget()
        v_input = QVBoxLayout(card_in)
        v_input.setContentsMargins(20, 20, 20, 20)
        
        v_input.addWidget(StrongBodyLabel("1. 입력자료 및 변수 선택"))
        ctrl_layout = QHBoxLayout()
        self.file_combo = ComboBox()
        self.file_combo.addItems(self.files)
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        ctrl_layout.addWidget(self.file_combo, 2)
        btn_browse = PushButton("파일 열기")
        btn_browse.clicked.connect(self.browse_file)
        ctrl_layout.addWidget(btn_browse, 1)
        
        self.var_combo = ComboBox()
        ctrl_layout.addWidget(self.var_combo, 1)
        v_input.addLayout(ctrl_layout)
        
        v_input.addWidget(StrongBodyLabel("2. 분석 기간 설정 (Time Range)"))
        time_layout = QHBoxLayout()
        self.txt_start_date = LineEdit()
        self.txt_start_date.setPlaceholderText("시작일 (예: 2015-01-01)")
        self.txt_end_date = LineEdit()
        self.txt_end_date.setPlaceholderText("종료일 (예: 2023-12-31)")
        time_layout.addWidget(StrongBodyLabel("시작일:"))
        time_layout.addWidget(self.txt_start_date)
        time_layout.addWidget(StrongBodyLabel("종료일:"))
        time_layout.addWidget(self.txt_end_date)
        v_input.addLayout(time_layout)
        
        v_input.addWidget(StrongBodyLabel("3. 위도/경도 범위 제한 (Lat/Lon Bounds)"))
        bbox_layout = QHBoxLayout()
        self.txt_min_lon = LineEdit(); self.txt_min_lon.setPlaceholderText("Min Lon (-180~180)")
        self.txt_max_lon = LineEdit(); self.txt_max_lon.setPlaceholderText("Max Lon (-180~180)")
        bbox_layout.addWidget(StrongBodyLabel("경도(Lon):"))
        bbox_layout.addWidget(self.txt_min_lon)
        bbox_layout.addWidget(StrongBodyLabel("~"))
        bbox_layout.addWidget(self.txt_max_lon)
        v_input.addLayout(bbox_layout)
        
        bbox_layout2 = QHBoxLayout()
        self.txt_min_lat = LineEdit(); self.txt_min_lat.setPlaceholderText("Min Lat (-90~90)")
        self.txt_max_lat = LineEdit(); self.txt_max_lat.setPlaceholderText("Max Lat (-90~90)")
        bbox_layout2.addWidget(StrongBodyLabel("위도(Lat):"))
        bbox_layout2.addWidget(self.txt_min_lat)
        bbox_layout2.addWidget(StrongBodyLabel("~"))
        bbox_layout2.addWidget(self.txt_max_lat)
        v_input.addLayout(bbox_layout2)
        
        v_input.addSpacing(10)
        self.btn_apply_input = PushButton("🚀 설정 적용 및 시각화 업데이트")
        self.btn_apply_input.clicked.connect(self.on_apply_settings)
        v_input.addWidget(self.btn_apply_input)
        
        v_left_in.addWidget(card_in)
        v_left_in.addStretch(1)
        scroll_left_in.setWidget(left_widget_in)
        
        # --- RIGHT SIDE (Display) ---
        right_widget_in = CardWidget()
        v_right_in = QVBoxLayout(right_widget_in)
        v_right_in.setContentsMargins(20, 20, 20, 20)
        v_right_in.addWidget(SubtitleLabel("입력 데이터셋 요약 정보 (Attributes)"))
        
        self.txt_overview = TextEdit()
        self.txt_overview.setReadOnly(True)
        font = self.txt_overview.font()
        font.setFamily("Consolas")
        font.setPointSize(10)
        self.txt_overview.setFont(font)
        v_right_in.addWidget(self.txt_overview)
        
        splitter_in.addWidget(scroll_left_in)
        splitter_in.addWidget(right_widget_in)
        splitter_in.setStretchFactor(0, 1)
        splitter_in.setStretchFactor(1, 1)
        
        page_input_layout.addWidget(splitter_in)
        
        # ----------------------------------------------------
        # Page 2: 검증자료 탭 (Validation Data)
        # ----------------------------------------------------
        page_valid = QWidget()
        page_valid_layout = QVBoxLayout(page_valid)
        page_valid_layout.setContentsMargins(0, 10, 0, 0)
        
        splitter_val = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE (Inputs) ---
        scroll_left_val = QScrollArea()
        scroll_left_val.setWidgetResizable(True)
        scroll_left_val.setFrameShape(QFrame.Shape.NoFrame)
        scroll_left_val.setStyleSheet("QScrollArea { background-color: transparent; }")
        
        left_widget_val = QWidget()
        v_left_val = QVBoxLayout(left_widget_val)
        v_left_val.setContentsMargins(0, 0, 10, 0)
        
        card_valid = CardWidget()
        v_valid = QVBoxLayout(card_valid)
        v_valid.setContentsMargins(20, 20, 20, 20)
        
        v_valid.addWidget(StrongBodyLabel("검증자료 및 변수 선택"))
        v_ctrl_val = QHBoxLayout()
        self.valid_file_combo = ComboBox()
        self.valid_file_combo.addItems(self.files)
        self.valid_file_combo.currentIndexChanged.connect(self.on_valid_file_changed)
        v_ctrl_val.addWidget(self.valid_file_combo, 2)
        btn_browse_valid = PushButton("파일 열기")
        btn_browse_valid.clicked.connect(self.browse_valid_file)
        v_ctrl_val.addWidget(btn_browse_valid, 1)
        
        self.valid_var_combo = ComboBox()
        v_ctrl_val.addWidget(self.valid_var_combo, 1)
        v_valid.addLayout(v_ctrl_val)
        
        v_valid.addSpacing(10)
        self.btn_apply_valid = PushButton("🚀 설정 적용 및 시각화 업데이트")
        self.btn_apply_valid.clicked.connect(self.on_apply_settings)
        v_valid.addWidget(self.btn_apply_valid)
        
        v_left_val.addWidget(card_valid)
        v_left_val.addStretch(1)
        scroll_left_val.setWidget(left_widget_val)
        
        # --- RIGHT SIDE (Display) ---
        right_widget_val = CardWidget()
        v_right_val = QVBoxLayout(right_widget_val)
        v_right_val.setContentsMargins(20, 20, 20, 20)
        v_right_val.addWidget(SubtitleLabel("검증 데이터셋 요약 정보 (Attributes)"))
        
        self.txt_valid_overview = TextEdit()
        self.txt_valid_overview.setReadOnly(True)
        self.txt_valid_overview.setFont(font)
        v_right_val.addWidget(self.txt_valid_overview)
        
        splitter_val.addWidget(scroll_left_val)
        splitter_val.addWidget(right_widget_val)
        splitter_val.setStretchFactor(0, 1)
        splitter_val.setStretchFactor(1, 1)
        
        page_valid_layout.addWidget(splitter_val)
        
        # Add Pages to Stack
        self.stack.addWidget(page_input)
        self.stack.addWidget(page_valid)
        
        self.segment.addItem("input", "입력자료")
        self.segment.addItem("valid", "검증자료")
        self.segment.currentItemChanged.connect(
            lambda k: self.stack.setCurrentIndex(0 if k == 'input' else 1)
        )
        self.segment.setCurrentItem("input")
        
        if self.files:
            self.on_file_changed()

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "NetCDF Files (*.nc);;All Files (*)")
        if filepath:
            if filepath not in self.files:
                self.files.append(filepath)
                self.file_combo.addItem(filepath)
                self.valid_file_combo.addItem(filepath)
            self.file_combo.setCurrentText(filepath)

    def browse_valid_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Validation File", "", "NetCDF Files (*.nc);;CSV Files (*.csv);;All Files (*)")
        if filepath:
            if filepath not in self.files:
                self.files.append(filepath)
                self.file_combo.addItem(filepath)
                self.valid_file_combo.addItem(filepath)
            self.valid_file_combo.setCurrentText(filepath)

    def on_file_changed(self):
        filepath = self.file_combo.currentText()
        if not filepath: return
        try:
            ds = nct.open(filepath)
            self.window().ds = ds
            data_vars = [var for var in ds.data_vars if 'bnds' not in var and 'bounds' not in var]
            
            self.var_combo.blockSignals(True)
            self.var_combo.clear()
            self.var_combo.addItems(data_vars)
            self.var_combo.blockSignals(False)
            
            # Setup Lat/Lon and Time LineEdits Based on data defaults
            if 'time' in ds.dims:
                t_vals = ds['time'].values
                if len(t_vals) > 0:
                    start_date = str(t_vals[0])[:10]
                    end_date = str(t_vals[-1])[:10]
                    self.txt_start_date.setText(start_date)
                    self.txt_end_date.setText(end_date)
            
            lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
            lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
            
            if lon_name in ds.coords:
                min_lo = float(ds[lon_name].min().values)
                max_lo = float(ds[lon_name].max().values)
                self.txt_min_lon.setText(f"{min_lo:.2f}")
                self.txt_max_lon.setText(f"{max_lo:.2f}")
                
            if lat_name in ds.coords:
                min_la = float(ds[lat_name].min().values)
                max_la = float(ds[lat_name].max().values)
                self.txt_min_lat.setText(f"{min_la:.2f}")
                self.txt_max_lat.setText(f"{max_la:.2f}")
            
            self.update_overview()
            self.on_apply_settings()
        except Exception as e:
            pass

    def on_valid_file_changed(self):
        filepath = self.valid_file_combo.currentText()
        if not filepath: return
        try:
            vds = nct.open(filepath)
            self.window().valid_ds = vds
            data_vars = [var for var in vds.data_vars if 'bnds' not in var and 'bounds' not in var]
            
            self.valid_var_combo.blockSignals(True)
            self.valid_var_combo.clear()
            self.valid_var_combo.addItems(data_vars)
            self.valid_var_combo.blockSignals(False)
            
            self.update_valid_overview()
            self.on_apply_settings()
        except Exception as e:
            pass

    def update_overview(self):
        ds = self.window().ds
        if ds is None: return
        filepath = self.file_combo.currentText()
        content = f"<b>파일 경로:</b> {filepath}<br>"
        content += f"<b>포맷:</b> NetCDF<br><br>"
        content += "<b>Coordinates (차원):</b><br>" + str(list(ds.coords)) + "<br><br>"
        content += "<b>Data Variables (변수):</b><br>" + str(list(ds.data_vars)) + "<br><br>"
        content += "<b>Global Attributes:</b><br>" + str(ds.attrs)
        self.txt_overview.setHtml(content)

    def update_valid_overview(self):
        vds = self.window().valid_ds
        if vds is None: return
        filepath = self.valid_file_combo.currentText()
        content = f"<b>검증 파일 경로:</b> {filepath}<br>"
        content += f"<b>포맷:</b> NetCDF/CSV<br><br>"
        content += "<b>Coordinates (차원):</b><br>" + str(list(vds.coords)) + "<br><br>"
        content += "<b>Data Variables (변수):</b><br>" + str(list(vds.data_vars)) + "<br><br>"
        content += "<b>Global Attributes:</b><br>" + str(vds.attrs)
        self.txt_valid_overview.setHtml(content)

    def on_apply_settings(self):
        w = self.window()
        w.selected_var = self.var_combo.currentText()
        w.selected_valid_var = self.valid_var_combo.currentText()
        
        ds = w.ds
        if ds is not None:
            # Slicing the dataset
            lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
            lat_name = 'lat' if 'lat' in ds.coords else 'latitude'
            
            try:
                slc = {}
                # Time Slicing
                t_start = self.txt_start_date.text().strip()
                t_end = self.txt_end_date.text().strip()
                if 'time' in ds.dims and t_start and t_end:
                    slc['time'] = slice(t_start, t_end)
                
                # Spatial Slicing
                min_lat = float(self.txt_min_lat.text() or -90)
                max_lat = float(self.txt_max_lat.text() or 90)
                min_lon = float(self.txt_min_lon.text() or -180)
                max_lon = float(self.txt_max_lon.text() or 180)
                
                if ds[lat_name].values[0] > ds[lat_name].values[-1]:
                    slc[lat_name] = slice(max_lat, min_lat)
                else:
                    slc[lat_name] = slice(min_lat, max_lat)
                    
                if ds[lon_name].values[0] > ds[lon_name].values[-1]:
                    slc[lon_name] = slice(max_lon, min_lon)
                else:
                    slc[lon_name] = slice(min_lon, max_lon)
                
                w.processed_ds = ds.sel(**slc)
                w.selected_time_idx = 0 # Default map to the first time slice after filtering
            except Exception as e:
                QMessageBox.warning(self, "설정 적용 오류", f"슬라이싱 중 오류가 발생했습니다:\\n{str(e)}\\n(원본 데이터를 그대로 사용합니다)")
                w.processed_ds = ds
        else:
            w.processed_ds = None
            
        # Call redraw
        if hasattr(w, 'visualize_interface'):
            QMessageBox.information(self, "적용 완료", "설정이 성공적으로 적용되었습니다.\n[시각화] 탭으로 이동하면 그래프가 자동으로 그려집니다.")


class CalculateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("CalculateInterface")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        title = TitleLabel("산출 (Calculation)")
        main_layout.addWidget(title)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: transparent; }")
        
        scroll_widget = QWidget()
        v_scroll = QVBoxLayout(scroll_widget)
        v_scroll.setContentsMargins(0,0,0,0)
        
        # 기후값 산출 옵션 카드
        card_cli = CardWidget()
        v_cli = QVBoxLayout(card_cli)
        v_cli.setContentsMargins(20, 20, 20, 20)
        v_cli.addWidget(SubtitleLabel("1. 기후 평년값 및 편차 산출 (Climatology & Anomaly)"))
        
        h_cli = QHBoxLayout()
        self.sw_cli = SwitchButton()
        self.sw_cli.setOnText("평년값 산출 켜기")
        self.sw_cli.setOffText("평년값 산출 끄기")
        h_cli.addWidget(self.sw_cli)
        
        self.txt_cli_start = LineEdit()
        self.txt_cli_start.setPlaceholderText("기준 시작년도 (예: 1991)")
        self.txt_cli_end = LineEdit()
        self.txt_cli_end.setPlaceholderText("기준 종료년도 (예: 2020)")
        h_cli.addWidget(StrongBodyLabel("기준 연도:"))
        h_cli.addWidget(self.txt_cli_start)
        h_cli.addWidget(StrongBodyLabel("~"))
        h_cli.addWidget(self.txt_cli_end)
        h_cli.addStretch(1)
        v_cli.addLayout(h_cli)
        
        h_ano = QHBoxLayout()
        self.sw_ano = SwitchButton()
        self.sw_ano.setOnText("편차(Anomaly) 산출 켜기")
        self.sw_ano.setOffText("편차(Anomaly) 산출 끄기")
        h_ano.addWidget(self.sw_ano)
        h_ano.addStretch(1)
        v_cli.addLayout(h_ano)
        
        v_scroll.addWidget(card_cli)
        
        # 시공간 평균 옵션 카드
        card_mean = CardWidget()
        v_mean = QVBoxLayout(card_mean)
        v_mean.setContentsMargins(20, 20, 20, 20)
        v_mean.addWidget(SubtitleLabel("2. 시공간 평균 산출 (Spatial & Temporal Mean)"))
        
        h_spa = QHBoxLayout()
        h_spa.addWidget(StrongBodyLabel("공간 평균 (Spatial Mean):"))
        self.cb_spatial = ComboBox()
        self.cb_spatial.addItems(["적용 안함 (None)", "전구 평균 (Global Mean)", "위도별 평균 (Zonal Mean)", "경도별 평균 (Meridional Mean)"])
        h_spa.addWidget(self.cb_spatial, 1)
        v_mean.addLayout(h_spa)
        
        h_time = QHBoxLayout()
        h_time.addWidget(StrongBodyLabel("시간 평균 (Temporal Mean):"))
        self.cb_temporal = ComboBox()
        self.cb_temporal.addItems(["적용 안함 (None)", "일 평균 (Daily)", "월 평균 (Monthly)", "연 평균 (Yearly)", "계절 평균 (Seasonal)"])
        h_time.addWidget(self.cb_temporal, 1)
        v_mean.addLayout(h_time)
        
        v_scroll.addWidget(card_mean)
        
        # 실행 버튼
        v_scroll.addSpacing(20)
        self.btn_execute = PushButton("🚀 기후 데이터 산출 실행 (Execute Calculation)")
        self.btn_execute.setMinimumHeight(50)
        self.btn_execute.clicked.connect(self.execute_calculation)
        v_scroll.addWidget(self.btn_execute)
        
        v_scroll.addStretch(1)
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

    def execute_calculation(self):
        w = self.window()
        if w.processed_ds is None:
            QMessageBox.warning(self, "경고", "먼저 [전처리] 탭에서 입력자료를 설정하고 적용해주세요.")
            return
            
        var_name = w.selected_var
        if not var_name:
            QMessageBox.warning(self, "경고", "분석할 변수가 선택되지 않았습니다.")
            return
            
        try:
            # 1. 툴박스 가중치 적용 (면적 가중치 등)
            try:
                ds = nct.weg(w.processed_ds)
            except Exception:
                ds = w.processed_ds
            
            # 2. 기후 평년값 산출 (Climatology)
            if self.sw_cli.isChecked():
                # Extract years from line edits if needed for filtering
                # For now just use toolbox cli
                ds = nct.cli(ds, var_name)
            
            # 3. 편차 산출 (Anomaly)
            if self.sw_ano.isChecked():
                ds = nct.ano(ds, var_name)
                
            # 4. 공간 평균 (Spatial Mean)
            spa_opt = self.cb_spatial.currentIndex()
            if spa_opt == 1: # Global
                ds = nct.spaMean(ds, var_name)
            elif spa_opt == 2: # Zonal
                # Typically mean across longitude
                if 'lon' in ds.dims:
                    ds = ds.mean(dim='lon', keep_attrs=True)
                elif 'longitude' in ds.dims:
                    ds = ds.mean(dim='longitude', keep_attrs=True)
            elif spa_opt == 3: # Meridional
                if 'lat' in ds.dims:
                    ds = ds.mean(dim='lat', keep_attrs=True)
                elif 'latitude' in ds.dims:
                    ds = ds.mean(dim='latitude', keep_attrs=True)
                    
            # 5. 시간 평균 (Temporal Mean)
            time_opt = self.cb_temporal.currentIndex()
            if time_opt > 0 and 'time' in ds.dims:
                if time_opt == 1: # Daily
                    ds = ds.resample(time='1D').mean(keep_attrs=True)
                elif time_opt == 2: # Monthly
                    ds = ds.resample(time='1ME').mean(keep_attrs=True)
                elif time_opt == 3: # Yearly
                    ds = ds.resample(time='1YE').mean(keep_attrs=True)
                elif time_opt == 4: # Seasonal
                    ds = ds.resample(time='QS-DEC').mean(keep_attrs=True)

            w.calculated_ds = ds
            
            # Switch to Visualization Tab to show result
            QMessageBox.information(self, "산출 완료", "기후 데이터 산출이 완료되었습니다.\\n[시각화] 탭에서 결과를 확인하세요.")
            
        except Exception as e:
            QMessageBox.critical(self, "오류", f"산출 중 오류 발생:\\n{str(e)}")



import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PyQt6.QtCore import QThread, pyqtSignal

class MapPlotThread(QThread):
    # Emits JSON string: {image_b64, extent:[minLon,minLat,maxLon,maxLat], vmin, vmax, var_name, buoy_points}
    plot_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, ds, var_name, time_idx, vds, vvar, vmin, vmax, cmap, bounds):
        super().__init__()
        self.ds = ds
        self.var_name = var_name
        self.time_idx = time_idx
        self.vds = vds
        self.vvar = vvar
        self.vmin = vmin
        self.vmax = vmax
        self.cmap = cmap
        self.bounds = bounds

    def run(self):
        try:
            import json as _json
            from mpl_toolkits.basemap import Basemap

            # ── 1. 위경도 로드 ─────────────────────────────────────────────
            lat_var = None
            lon_var = None
            for v in ['lat', 'latitude']:
                if v in self.ds.variables or v in self.ds.coords:
                    lat_var = v; break
            for v in ['lon', 'longitude']:
                if v in self.ds.variables or v in self.ds.coords:
                    lon_var = v; break

            if not lat_var or not lon_var:
                import os, netCDF4 as nc
                latlon_path = r"C:\SYSTEMS\PROG\PYTHON\climate_extremes\gk2a_sst\cfg\gk2a_ami_ea020lc_latlon.nc"
                if os.path.exists(latlon_path):
                    with nc.Dataset(latlon_path) as ds_ll:
                        lat_mesh = np.array(ds_ll.variables['lat'][:])
                        lon_mesh = np.array(ds_ll.variables['lon'][:])
                else:
                    self.error_occurred.emit("위경도 차원을 찾을 수 없습니다.")
                    return
            else:
                lat = self.ds[lat_var].values
                lon = self.ds[lon_var].values
                if lat.ndim == 1:
                    lon_mesh, lat_mesh = np.meshgrid(lon, lat)
                else:
                    lon_mesh, lat_mesh = lon, lat

            lon_mesh = np.nan_to_num(np.array(lon_mesh, dtype=float), nan=0.0)
            lat_mesh = np.nan_to_num(np.array(lat_mesh, dtype=float), nan=0.0)

            # ── 2. 데이터 로드 & 켈빈 변환 ─────────────────────────────────
            if 'time' in self.ds.dims:
                data = self.ds[self.var_name].isel(time=self.time_idx).values
            else:
                data = self.ds[self.var_name].values
            if np.nanmin(data) > 200:
                data = data - 273.15

            # ── 3. 지리 범위 계산 (OpenLayers ImageStatic extent용) ────────
            valid_mask = ~np.isnan(lon_mesh) & ~np.isnan(lat_mesh)
            min_lon = float(np.nanmin(lon_mesh[valid_mask]))
            max_lon = float(np.nanmax(lon_mesh[valid_mask]))
            min_lat = float(np.nanmin(lat_mesh[valid_mask]))
            max_lat = float(np.nanmax(lat_mesh[valid_mask]))
            extent = [min_lon, min_lat, max_lon, max_lat]

            # ── 4. Basemap 렌더링 (cyl 투영 → EPSG:4326 호환) ─────────────
            fig = plt.figure(figsize=(10, 8), dpi=150)
            ax = fig.add_subplot(111)

            m = Basemap(
                projection='cyl', resolution='i',
                llcrnrlat=min_lat, urcrnrlat=max_lat,
                llcrnrlon=min_lon, urcrnrlon=max_lon
            )

            x, y = m(lon_mesh, lat_mesh)
            x = np.where(x > 1e20, np.nan, x)
            y = np.where(y > 1e20, np.nan, y)

            try: vmin_val = float(self.vmin) if self.vmin else float(np.nanmin(data))
            except: vmin_val = float(np.nanmin(data))
            try: vmax_val = float(self.vmax) if self.vmax else float(np.nanmax(data))
            except: vmax_val = float(np.nanmax(data))

            dlev = max((vmax_val - vmin_val) / 9, 0.5)
            levels = np.arange(vmin_val, vmax_val + 1e-6, dlev)

            data_ma = np.ma.masked_invalid(data)

            pcm = m.pcolormesh(x, y, data_ma,
                               cmap=self.cmap if self.cmap else 'jet',
                               shading='auto', vmin=vmin_val, vmax=vmax_val)

            m.drawcoastlines(color='k', linewidth=0.8)
            m.drawcountries(color='gray', linewidth=0.5)
            m.fillcontinents(color='lightgray', lake_color='white')
            m.drawparallels(np.arange(-90, 91, 10), labels=[1,0,0,0], fontsize=9, fmt='%d')
            m.drawmeridians(np.arange(-180, 181, 10), labels=[0,0,0,1], fontsize=9, fmt='%d')

            # 가우시안 필터 등온선 (plot_sst_monthly_ea.py 동일)
            weight = np.ones_like(data_ma.data)
            weight[data_ma.mask] = 0.0
            data_zeroed = data_ma.filled(0.0)
            smoothed = gaussian_filter(data_zeroed, sigma=10)
            sw = gaussian_filter(weight, sigma=10)
            with np.errstate(invalid='ignore', divide='ignore'):
                smoothed_c = smoothed / sw
            smoothed_ma = np.ma.masked_array(smoothed_c, mask=data_ma.mask)

            try:
                c = m.contour(x, y, smoothed_ma, levels=levels,
                              colors='black', linewidths=1.0, alpha=0.8)
                labels = plt.clabel(c, inline=True, fontsize=8, fmt='%d', colors='black')
                for lbl in labels: lbl.set_rotation(0)
            except Exception:
                pass

            cbar = plt.colorbar(pcm, shrink=0.8, extend='both')
            cbar.set_label(f'{self.var_name}', fontsize=10, fontweight='bold')
            plt.title(f'NMSC Climate Toolbox  -  {self.var_name}', fontsize=12, fontweight='bold')
            plt.tight_layout()

            # 이미지 저장 (no axes, tight bbox) — 축/여백 제거해서 OpenLayers와 정밀 정합
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0)
            plt.close()
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')

            # ── 5. 부이 포인트 수집 ──────────────────────────────────────
            buoy_points = []
            if self.vds is not None and self.vvar:
                try:
                    v_lat_dim = next((v for v in ['lat','latitude'] if v in self.vds.coords), None)
                    v_lon_dim = next((v for v in ['lon','longitude'] if v in self.vds.coords), None)
                    if v_lat_dim and v_lon_dim:
                        for st in self.vds['station_id'].values:
                            v_lat = float(self.vds[v_lat_dim].sel(station_id=st).values)
                            v_lon = float(self.vds[v_lon_dim].sel(station_id=st).values)
                            st_name = str(self.vds['station_name'].sel(station_id=st).values) if 'station_name' in self.vds else str(st)
                            buoy_points.append({'lon': v_lon, 'lat': v_lat, 'name': st_name})
                except Exception:
                    pass

            result = _json.dumps({
                'image_b64': img_b64,
                'extent': extent,
                'vmin': vmin_val,
                'vmax': vmax_val,
                'var_name': self.var_name,
                'buoy_points': buoy_points
            })
            self.plot_finished.emit(result)

        except Exception as e:
            import traceback
            self.error_occurred.emit(traceback.format_exc())

class VisualizeInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("VisualizeInterface")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        title = TitleLabel("시각화 (Visualization)")
        main_layout.addWidget(title)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE (Options) ---
        left_widget = CardWidget()
        v_left = QVBoxLayout(left_widget)
        v_left.setContentsMargins(20, 20, 20, 20)
        
        v_left.addWidget(SubtitleLabel("시각화 옵션 설정"))
        
        h_cmap = QHBoxLayout()
        h_cmap.addWidget(StrongBodyLabel("컬러맵 (Color Map):"))
        self.cb_cmap = ComboBox()
        self.cb_cmap.addItems(['RdYlBu_r', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'bwr', 'jet'])
        h_cmap.addWidget(self.cb_cmap, 1)
        v_left.addLayout(h_cmap)
        
        v_left.addWidget(StrongBodyLabel("값 표시 범위 (Value Range)"))
        h_range = QHBoxLayout()
        self.txt_vmin = LineEdit()
        self.txt_vmin.setPlaceholderText("Min (비우면 자동)")
        self.txt_vmax = LineEdit()
        self.txt_vmax.setPlaceholderText("Max (비우면 자동)")
        h_range.addWidget(self.txt_vmin)
        h_range.addWidget(StrongBodyLabel("~"))
        h_range.addWidget(self.txt_vmax)
        v_left.addLayout(h_range)
        
        v_left.addSpacing(10)
        self.btn_refresh = PushButton("🔄 시각화 화면 갱신 (Refresh)")
        self.btn_refresh.setMinimumHeight(40)
        self.btn_refresh.clicked.connect(self.refresh_current_plot)
        v_left.addWidget(self.btn_refresh)
        v_left.addStretch(1)
        
        splitter.addWidget(left_widget)
        
        # --- RIGHT SIDE (Plots) ---
        right_widget = QWidget()
        v_right = QVBoxLayout(right_widget)
        v_right.setContentsMargins(10, 0, 0, 0)
        
        self.pivot = Pivot()
        v_right.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)
        
        self.stack = QStackedWidget()
        v_right.addWidget(self.stack)
        
        page_map = QWidget()
        self.map_canvas_layout = QVBoxLayout(page_map)
        self.map_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(page_map)
        
        page_trend = QWidget()
        self.trend_canvas_layout = QVBoxLayout(page_trend)
        self.trend_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(page_trend)
        
        page_comp = QWidget()
        self.valid_canvas_layout = QVBoxLayout(page_comp)
        self.valid_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(page_comp)
        
        self.pivot.addItem("map", "지도 맵", lambda: self.on_tab_changed(0))
        self.pivot.addItem("trend", "시계열 트렌드", lambda: self.on_tab_changed(1))
        self.pivot.addItem("comp", "검증 산점도", lambda: self.on_tab_changed(2))
        self.pivot.setCurrentItem("map")
        
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        
        main_layout.addWidget(splitter)

    def on_tab_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        self.refresh_current_plot()

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_current_plot()

    def refresh_current_plot(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            self.plot_map()
        elif idx == 1:
            self.plot_trend()
        elif idx == 2:
            self.plot_valid()

    def get_ds(self):
        w = self.window()
        if hasattr(w, 'calculated_ds') and w.calculated_ds is not None:
            return w.calculated_ds
        elif hasattr(w, 'processed_ds') and w.processed_ds is not None:
            return w.processed_ds
        return None

    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.clear_layout(item.layout())

    def plot_trend(self):
        ds = self.get_ds()
        var_name = self.window().selected_var
        if ds is None or not var_name:
            return
            
        time_dim = 'time' if 'time' in ds.dims else None
        if not time_dim: return
        
        try:
            import json
            import pandas as pd
            import numpy as np
            
            # Check for lat/lon dims properly
            has_spatial = False
            lat_dim = 'lat' if 'lat' in ds.dims else ('latitude' if 'latitude' in ds.dims else None)
            lon_dim = 'lon' if 'lon' in ds.dims else ('longitude' if 'longitude' in ds.dims else None)
            
            if lat_dim and lon_dim:
                has_spatial = True
                
            if has_spatial:
                ts = ds[var_name].mean(dim=[lat_dim, lon_dim], skipna=True)
            else:
                ts = ds[var_name]
                
            df = ts.to_dataframe().reset_index()
            
            if len(df) == 0:
                return
                
            # Convert time to timestamp in milliseconds
            df['timestamp'] = pd.to_datetime(df[time_dim]).astype('int64') // 10**6
            
            # Replace NaNs with None for JSON serialization
            df[var_name] = df[var_name].replace({np.nan: None})
            
            data_list = df[['timestamp', var_name]].values.tolist()
            data_json = json.dumps(data_list)
            
            hc_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://code.highcharts.com/highcharts.js"></script>
                <script src="https://code.highcharts.com/modules/exporting.js"></script>
                <style>
                    body, html {{ margin: 0; padding: 0; height: 100%; }}
                    #container {{ width: 100%; height: 100%; }}
                </style>
            </head>
            <body>
                <div id="container"></div>
                <script>
                    Highcharts.chart('container', {{
                        chart: {{ type: 'line', zoomType: 'x' }},
                        title: {{ text: '{var_name} 시계열 트렌드' }},
                        xAxis: {{ type: 'datetime', title: {{ text: '시간' }} }},
                        yAxis: {{ title: {{ text: '{var_name}' }} }},
                        tooltip: {{ xDateFormat: '%Y-%m-%d %H:%M:%S', shared: true }},
                        series: [{{ name: '{var_name}', data: {data_json}, color: '#0078d4' }}],
                        credits: {{ enabled: false }}
                    }});
                </script>
            </body>
            </html>
            """
            
            self.clear_layout(self.trend_canvas_layout)
            view = QWebEngineView()
            view.setHtml(hc_html)
            self.trend_canvas_layout.addWidget(view)
            
        except Exception as e:
            print("Trend plot error:", e)

    def plot_valid(self):
        ds = self.get_ds()
        vds = self.window().valid_ds
        var_name = self.window().selected_var
        vvar = self.window().selected_valid_var
        time_idx = self.window().selected_time_idx
        
        if ds is None or vds is None or not var_name or not vvar:
            return
            
        try:
            import json
            import numpy as np
            
            lat_dim = 'lat' if 'lat' in ds.dims else ('latitude' if 'latitude' in ds.dims else None)
            lon_dim = 'lon' if 'lon' in ds.dims else ('longitude' if 'longitude' in ds.dims else None)
            v_lat_dim = 'lat' if 'lat' in vds.coords else ('latitude' if 'latitude' in vds.coords else None)
            v_lon_dim = 'lon' if 'lon' in vds.coords else ('longitude' if 'longitude' in vds.coords else None)
            
            if not lat_dim or not v_lat_dim:
                return
                
            scatter_data = []
            x_vals = []
            y_vals = []
            
            for st in vds['station_id'].values:
                try:
                    lat_v = float(vds[v_lat_dim].sel(station_id=st).values)
                    lon_v = float(vds[v_lon_dim].sel(station_id=st).values)
                    st_name = str(vds['station_name'].sel(station_id=st).values)
                    
                    if 'time' in ds.dims:
                        d_val = float(ds[var_name].isel(time=time_idx).sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)
                    else:
                        d_val = float(ds[var_name].sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)
                        
                    if 'time' in vds.dims:
                        v_val = float(vds[vvar].isel(time=time_idx).sel(station_id=st).values)
                    else:
                        v_val = float(vds[vvar].sel(station_id=st).values)
                        
                    if not np.isnan(d_val) and not np.isnan(v_val):
                        scatter_data.append({'x': d_val, 'y': v_val, 'name': st_name})
                        x_vals.append(d_val)
                        y_vals.append(v_val)
                except:
                    continue
                    
            if not scatter_data: return
            
            min_v = min(min(x_vals), min(y_vals))
            max_v = max(max(x_vals), max(y_vals))
            
            # Extend line a bit
            padding = (max_v - min_v) * 0.1
            if padding == 0: padding = 1
            line_min = min_v - padding
            line_max = max_v + padding
            
            scatter_json = json.dumps(scatter_data)
            
            hc_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://code.highcharts.com/highcharts.js"></script>
                <script src="https://code.highcharts.com/modules/exporting.js"></script>
                <style>
                    body, html {{ margin: 0; padding: 0; height: 100%; }}
                    #container {{ width: 100%; height: 100%; }}
                </style>
            </head>
            <body>
                <div id="container"></div>
                <script>
                    Highcharts.chart('container', {{
                        chart: {{ type: 'scatter', zoomType: 'xy' }},
                        title: {{ text: '{var_name} vs {vvar} 검증 산점도' }},
                        xAxis: {{ title: {{ text: '위성 산출물 ({var_name})' }}, crosshair: true }},
                        yAxis: {{ title: {{ text: '검증 자료 ({vvar})' }}, crosshair: true }},
                        tooltip: {{
                            useHTML: true,
                            headerFormat: '<b>{{point.key}}</b><br>',
                            pointFormat: '위성: {{point.x}}<br>부이: {{point.y}}'
                        }},
                        series: [{{
                            name: '관측소 데이터',
                            data: {scatter_json},
                            color: 'rgba(223, 83, 83, .5)',
                            marker: {{ radius: 5 }}
                        }}, {{
                            type: 'line',
                            name: 'Y = X',
                            data: [[{line_min}, {line_min}], [{line_max}, {line_max}]],
                            marker: {{ enabled: false }},
                            states: {{ hover: {{ lineWidth: 0 }} }},
                            enableMouseTracking: false,
                            color: 'black',
                            dashStyle: 'Dash'
                        }}],
                        credits: {{ enabled: false }}
                    }});
                </script>
            </body>
            </html>
            """
            
            self.clear_layout(self.valid_canvas_layout)
            view = QWebEngineView()
            view.setHtml(hc_html)
            self.valid_canvas_layout.addWidget(view)
        except Exception as e:
            print("Valid plot error:", e)

    def plot_map(self):
        ds = self.get_ds()
        var_name = self.window().selected_var
        if ds is None or not var_name:
            self.clear_layout(self.map_canvas_layout)
            self.map_canvas_layout.addWidget(BodyLabel("데이터를 먼저 불러오세요."))
            return

        time_idx = self.window().selected_time_idx
        vds = self.window().valid_ds
        vvar = self.window().selected_valid_var
        cmap = self.cb_cmap.currentText()
        vmin_txt = self.txt_vmin.text().strip()
        vmax_txt = self.txt_vmax.text().strip()

        self.clear_layout(self.map_canvas_layout)
        loading_label = TitleLabel("지도를 그리는 중입니다... (최대 15초 소요)")
        self.map_canvas_layout.addWidget(loading_label)

        self.map_thread = MapPlotThread(
            ds, var_name, time_idx, vds, vvar,
            vmin_txt, vmax_txt, cmap, self.window().bounds
        )
        self.map_thread.plot_finished.connect(self.on_map_finished)
        self.map_thread.error_occurred.connect(self.on_map_error)
        self.map_thread.start()

    def on_map_finished(self, result_json):
        self.clear_layout(self.map_canvas_layout)
        try:
            import json
            result = json.loads(result_json)
            img_b64 = result['image_b64']
            extent = result['extent']   # [minLon, minLat, maxLon, maxLat]
            vmin = result['vmin']
            vmax = result['vmax']
            var_name = result['var_name']
            buoy_points = result.get('buoy_points', [])
            buoy_json = json.dumps(buoy_points)
            img_src = f"data:image/png;base64,{img_b64}"

            center_lon = (extent[0] + extent[2]) / 2
            center_lat = (extent[1] + extent[3]) / 2

            ol_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/ol@v8.2.0/ol.css">
<script src="https://cdn.jsdelivr.net/npm/ol@v8.2.0/dist/ol.js"></script>
<style>
  html, body {{ margin:0; padding:0; height:100%; background:#1a1a2e; font-family:sans-serif; }}
  #map {{ width:100%; height:100%; }}
  #info-panel {{
    position:absolute; top:10px; left:10px; z-index:100;
    background:rgba(0,0,0,0.75); color:white;
    padding:8px 14px; border-radius:8px; font-size:13px; pointer-events:none;
  }}
  #tooltip {{
    position:absolute; background:rgba(0,0,0,0.8); color:white;
    padding:6px 10px; border-radius:5px; font-size:12px;
    display:none; pointer-events:none; white-space:nowrap; z-index:200;
  }}
  #legend {{
    position:absolute; bottom:20px; right:14px; z-index:100;
    background:rgba(255,255,255,0.93); padding:10px 14px;
    border-radius:8px; font-size:12px; min-width:60px; text-align:center;
    box-shadow:0 2px 8px rgba(0,0,0,0.3);
  }}
  .cb-bar {{
    width:20px; height:140px;
    background:linear-gradient(to top,#313695,#4575b4,#74add1,#abd9e9,#e0f3f8,#ffffbf,#fee090,#fdae61,#f46d43,#d73027,#a50026);
    display:inline-block; margin-right:6px; border:1px solid #ccc; vertical-align:middle;
  }}
  .cb-labels {{display:inline-block; height:140px; position:relative; vertical-align:middle;}}
  .cb-labels span {{position:absolute; right:0; font-size:11px; font-weight:bold;}}
</style>
</head>
<body>
<div id="map"></div>
<div id="info-panel">🗺 {var_name} | 범위: {vmin:.1f} ~ {vmax:.1f}</div>
<div id="tooltip"></div>
<div id="legend">
  <div style="font-size:11px;font-weight:bold;margin-bottom:4px;">{var_name}</div>
  <div style="display:flex;align-items:center;">
    <div class="cb-bar"></div>
    <div class="cb-labels">
      <span style="top:0">{vmax:.1f}</span>
      <span style="top:50%;transform:translateY(-50%)">{(vmin+vmax)/2:.1f}</span>
      <span style="bottom:0">{vmin:.1f}</span>
    </div>
  </div>
</div>
<script>
const extent = {extent};
const imgSrc = "{img_src}";
const buoyPoints = {buoy_json};

// 투영 (EPSG:4326)
const proj4326 = new ol.proj.Projection({{
  code:'EPSG:4326', units:'degrees', axisOrientation:'neu'
}});

// 데이터 이미지 레이어 (Basemap 렌더링 결과물을 지리 좌표에 정확히 오버레이)
const imageLayer = new ol.layer.Image({{
  source: new ol.source.ImageStatic({{
    url: imgSrc,
    projection: proj4326,
    imageExtent: extent,
  }}),
  opacity: 0.85
}});

// 부이 벡터 레이어
const vectorSource = new ol.source.Vector();
buoyPoints.forEach(pt => {{
  const feature = new ol.Feature({{
    geometry: new ol.geom.Point([pt.lon, pt.lat]),
    name: pt.name
  }});
  vectorSource.addFeature(feature);
}});
const vectorLayer = new ol.layer.Vector({{
  source: vectorSource,
  style: new ol.style.Style({{
    image: new ol.style.Circle({{
      radius: 7,
      fill: new ol.style.Fill({{color:'rgba(255,60,60,0.9)'}}),
      stroke: new ol.style.Stroke({{color:'white', width:2}})
    }})
  }})
}});

// OSM 베이스 타일
const tileLayer = new ol.layer.Tile({{
  source: new ol.source.OSM(),
  opacity: 0.3
}});

const map = new ol.Map({{
  target:'map',
  layers:[tileLayer, imageLayer, vectorLayer],
  view: new ol.View({{
    projection:'EPSG:4326',
    center:[{center_lon}, {center_lat}],
    zoom:4
  }})
}});
map.getView().fit(extent, {{padding:[30,30,30,30], maxZoom:10}});

// 툴팁
const tooltip = document.getElementById('tooltip');
map.on('pointermove', evt => {{
  const feature = map.forEachFeatureAtPixel(evt.pixel, f => f);
  if (feature) {{
    tooltip.style.display='block';
    tooltip.style.left = evt.pixel[0]+12+'px';
    tooltip.style.top  = evt.pixel[1]+12+'px';
    tooltip.innerHTML = `<b>🔴 부이 관측소</b><br>${{feature.get('name')}}`;
  }} else {{
    tooltip.style.display='none';
  }}
}});
map.getViewport().addEventListener('mouseout', () => {{ tooltip.style.display='none'; }});
</script>
</body>
</html>"""

            view = QWebEngineView()
            view.setHtml(ol_html)
            self.map_canvas_layout.addWidget(view)

        except Exception as e:
            import traceback
            self.map_canvas_layout.addWidget(StrongBodyLabel(f"이미지 표출 오류:\n{traceback.format_exc()}"))



    def on_map_error(self, err_msg):
        self.clear_layout(self.map_canvas_layout)
        self.map_canvas_layout.addWidget(StrongBodyLabel(f"지도 렌더링 오류:\n{err_msg}"))



    def plot_valid(self):
        ds = self.get_ds()
        vds = self.window().valid_ds
        var_name = self.window().selected_var
        vvar = self.window().selected_valid_var
        time_idx = self.window().selected_time_idx
        
        if ds is None or vds is None or not var_name or not vvar:
            return
            
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            import pandas as pd
            
            lat_dim = 'lat' if 'lat' in ds.dims else ('latitude' if 'latitude' in ds.dims else None)
            lon_dim = 'lon' if 'lon' in ds.dims else ('longitude' if 'longitude' in ds.dims else None)
            v_lat_dim = 'lat' if 'lat' in vds.coords else ('latitude' if 'latitude' in vds.coords else None)
            v_lon_dim = 'lon' if 'lon' in vds.coords else ('longitude' if 'longitude' in vds.coords else None)
            
            if not lat_dim or not v_lat_dim:
                return
                
            y_vals = []
            x_vals = []
            
            for st in vds['station_id'].values:
                try:
                    lat_v = float(vds[v_lat_dim].sel(station_id=st).values)
                    lon_v = float(vds[v_lon_dim].sel(station_id=st).values)
                    
                    if 'time' in ds.dims:
                        d_val = float(ds[var_name].isel(time=time_idx).sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)
                    else:
                        d_val = float(ds[var_name].sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)
                        
                    if 'time' in vds.dims:
                        v_val = float(vds[vvar].isel(time=time_idx).sel(station_id=st).values)
                    else:
                        v_val = float(vds[vvar].sel(station_id=st).values)
                        
                    x_vals.append(d_val)
                    y_vals.append(v_val)
                except:
                    continue
                    
            if not x_vals: return
            
            fig = plt.figure(figsize=(6, 6), dpi=100)
            plt.scatter(x_vals, y_vals, alpha=0.7)
            
            min_v = min(min(x_vals), min(y_vals))
            max_v = max(max(x_vals), max(y_vals))
            plt.plot([min_v, max_v], [min_v, max_v], 'r--')
            
            plt.title(f"Scatter Plot: {var_name} vs {vvar}")
            plt.xlabel(f"Satellite ({var_name})")
            plt.ylabel(f"Validation ({vvar})")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            plt.close(fig)
            
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            html = f'<div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; background:white;"><img src="data:image/png;base64,{img_b64}" style="max-width:100%; max-height:100%; object-fit:contain;"></div>'
            
            self.clear_layout(self.valid_canvas_layout)
            view = QWebEngineView()
            view.setHtml(html)
            self.valid_canvas_layout.addWidget(view)
        except Exception as e:
            pass




from qfluentwidgets import TextEdit, ScrollArea, SegmentedWidget
from PyQt6.QtCore import QThread, pyqtSignal

class AIGeneratorThread(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, engine, prompt, api_key="", model_id="", hf_token=""):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        self.api_key = api_key
        self.model_id = model_id
        self.hf_token = hf_token

    def run(self):
        try:
            if self.engine == "gemini":
                if not self.api_key:
                    self.error_occurred.emit("Gemini API 키가 입력되지 않았습니다.")
                    return
                from google import genai
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content_stream(
                    model='gemini-2.5-flash',
                    contents=self.prompt,
                )
                for chunk in response:
                    self.chunk_received.emit(chunk.text)
            elif self.engine == "gemma":
                model_id = self.model_id if self.model_id else "Qwen/Qwen2.5-7B-Instruct"
                if not self.hf_token:
                    self.error_occurred.emit("HF Access Token이 입력되지 않았습니다.")
                    return
                
                try:
                    from huggingface_hub import InferenceClient
                except ImportError:
                    self.error_occurred.emit("huggingface_hub 패키지가 설치되지 않았습니다.")
                    return
                    
                self.chunk_received.emit("[System] HuggingFace 클라우드 서버(API)를 통해 답변을 생성 중입니다...\n\n")
                
                client = InferenceClient(model=model_id, token=self.hf_token)
                
                messages = [{"role": "user", "content": self.prompt}]
                for chunk in client.chat_completion(messages=messages, max_tokens=512, stream=True):
                    content = chunk.choices[0].delta.content
                    if content:
                        self.chunk_received.emit(content)
                    
        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self.finished.emit()

class AIAssistantInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("AIAssistantInterface")
        self.llm_thread = None
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        title = TitleLabel("AI 어시스턴트 (AI Chat)")
        main_layout.addWidget(title)
        
        # --- Config Area ---
        config_card = CardWidget()
        v_config = QVBoxLayout(config_card)
        
        h_engine = QHBoxLayout()
        h_engine.addWidget(StrongBodyLabel("AI 엔진 선택:"))
        self.combo_engine = ComboBox()
        self.combo_engine.addItems(["Gemini (Cloud)", "Gemma (Local Transformers)"])
        self.combo_engine.currentTextChanged.connect(self.on_engine_changed)
        h_engine.addWidget(self.combo_engine, 1)
        v_config.addLayout(h_engine)
        
        self.stack_config = QStackedWidget()
        
        # Gemini config
        page_gemini = QWidget()
        h_gemini = QHBoxLayout(page_gemini)
        h_gemini.setContentsMargins(0,0,0,0)
        h_gemini.addWidget(StrongBodyLabel("Gemini API Key:"))
        self.txt_api_key = LineEdit()
        self.txt_api_key.setPlaceholderText("AIzaSy...")
        self.txt_api_key.setEchoMode(LineEdit.EchoMode.Password)
        h_gemini.addWidget(self.txt_api_key, 1)
        self.stack_config.addWidget(page_gemini)
        
        # Gemma config
        page_gemma = QWidget()
        v_gemma = QVBoxLayout(page_gemma)
        v_gemma.setContentsMargins(0,0,0,0)
        
        h_gemma_1 = QHBoxLayout()
        h_gemma_1.addWidget(StrongBodyLabel("HuggingFace Model ID:"))
        self.txt_model_id = LineEdit()
        self.txt_model_id.setText("Qwen/Qwen2.5-7B-Instruct")
        h_gemma_1.addWidget(self.txt_model_id, 1)
        v_gemma.addLayout(h_gemma_1)
        
        h_gemma_2 = QHBoxLayout()
        h_gemma_2.addWidget(StrongBodyLabel("HF Access Token (Gemma 필수):"))
        self.txt_hf_token = LineEdit()
        self.txt_hf_token.setText("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")
        self.txt_hf_token.setEchoMode(LineEdit.EchoMode.Password)
        h_gemma_2.addWidget(self.txt_hf_token, 1)
        v_gemma.addLayout(h_gemma_2)
        
        self.stack_config.addWidget(page_gemma)
        
        v_config.addWidget(self.stack_config)
        main_layout.addWidget(config_card)
        
        # --- Chat Area ---
        self.chat_area = TextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setPlaceholderText("여기에 AI와의 대화 내용이 표시됩니다...")
        main_layout.addWidget(self.chat_area, 1)
        
        # --- Input Area ---
        h_input = QHBoxLayout()
        self.txt_prompt = TextEdit()
        self.txt_prompt.setFixedHeight(80)
        self.txt_prompt.setPlaceholderText("기후 데이터에 관해 AI에게 질문해보세요...")
        h_input.addWidget(self.txt_prompt, 1)
        
        self.btn_send = PushButton("전송 🚀")
        self.btn_send.setMinimumHeight(80)
        self.btn_send.clicked.connect(self.send_message)
        h_input.addWidget(self.btn_send)
        
        main_layout.addLayout(h_input)

    def on_engine_changed(self, text):
        if "Gemini" in text:
            self.stack_config.setCurrentIndex(0)
        else:
            self.stack_config.setCurrentIndex(1)
            
    def send_message(self):
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt: return
        
        engine_text = self.combo_engine.currentText()
        engine = "gemini" if "Gemini" in engine_text else "gemma"
        api_key = self.txt_api_key.text().strip()
        model_id = self.txt_model_id.text().strip()
        hf_token = self.txt_hf_token.text().strip() if hasattr(self, 'txt_hf_token') else ""
        
        # 사용자 질문 (우측 정렬, 파란색)
        you_html = f"<div align='right' style='color:#0078d4; margin-bottom:10px;'><b>[나]</b><br>{prompt}</div><br>"
        self.chat_area.append(you_html)
        self.txt_prompt.clear()
        
        # AI 답변 시작 (좌측 정렬)
        ai_html = f"<div align='left' style='margin-bottom:10px;'><b>[AI]</b><br></div>"
        self.chat_area.append(ai_html)
        
        # Move cursor to end to prepare for chunk insertion inside the AI div block
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_area.setTextCursor(cursor)
        
        self.btn_send.setEnabled(False)
        self.txt_prompt.setEnabled(False)
        
        self.llm_thread = AIGeneratorThread(engine, prompt, api_key, model_id, hf_token)
        self.llm_thread.chunk_received.connect(self.on_chunk)
        self.llm_thread.error_occurred.connect(self.on_error)
        self.llm_thread.finished.connect(self.on_finished)
        self.llm_thread.start()
        
    def on_chunk(self, text):
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.chat_area.setTextCursor(cursor)
        
    def on_error(self, error_msg):
        self.chat_area.append(f"\n<font color='red'>[Error] {error_msg}</font>")
        
    def on_finished(self):
        self.chat_area.append("\n")
        self.btn_send.setEnabled(True)
        self.txt_prompt.setEnabled(True)
        self.txt_prompt.setFocus()

class NMSCFluentApp(MSFluentWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMSC Climate Toolbox")
        self.resize(1200, 800)
        
        # Shared State
        self.ds = None
        self.processed_ds = None
        self.calculated_ds = None
        self.valid_ds = None
        self.selected_var = ""
        self.selected_valid_var = ""
        self.selected_time_idx = 0
        self.bounds = {'min_lon': -180, 'max_lon': 180, 'min_lat': -90, 'max_lat': 90}
        
        # Interfaces
        self.preprocess_interface = PreprocessInterface(self)
        self.calculate_interface = CalculateInterface(self)
        self.visualize_interface = VisualizeInterface(self)
        self.ai_interface = AIAssistantInterface(self)
        
        self.init_navigation()

    def init_navigation(self):
        self.addSubInterface(self.preprocess_interface, FluentIcon.DOCUMENT, "전처리")
        self.addSubInterface(self.calculate_interface, FluentIcon.PIE_SINGLE, "산출")
        self.addSubInterface(self.visualize_interface, FluentIcon.PHOTO, "시각화")
        
        self.addSubInterface(self.ai_interface, FluentIcon.CHAT, "AI 어시스턴트", position=NavigationItemPosition.BOTTOM)
        
        # Theme color setting
        setThemeColor('#0078d4')

if __name__ == "__main__":
    # Force light mode globally before app starts
    os.environ["QT_QPA_PLATFORM"] = "windows:darkmode=0"
    
    app = QApplication(sys.argv)
    setTheme(Theme.LIGHT)
    window = NMSCFluentApp()
    window.show()
    sys.exit(app.exec())
