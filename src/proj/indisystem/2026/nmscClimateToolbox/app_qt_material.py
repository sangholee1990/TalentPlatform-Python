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


from PyQt6.QtCore import QPropertyAnimation, QTimer, Qt, QRect
from PyQt6.QtWidgets import QLabel, QWidget

class ToastNotification(QLabel):
    def __init__(self, parent, text, duration=3000):
        super().__init__(text, parent)
        self.duration = duration
        
        # Styling
        self.setStyleSheet("""
            QLabel {
                background-color: #323232;
                color: #FFFFFF;
                border-radius: 8px;
                padding: 12px 24px;
                font-family: 'Segoe UI';
                font-size: 13px;
                font-weight: bold;
            }
        """)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.adjustSize()
        
        # Position at bottom right
        if parent:
            parent_rect = parent.rect()
            self.move(parent_rect.width() - self.width() - 30, parent_rect.height() - self.height() - 30)
            
        self.show()
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.hide_toast)
        self.timer.start(self.duration)
        
    def hide_toast(self):
        self.timer.stop()
        self.deleteLater()

    @staticmethod
    def show_toast(parent, title, message):
        ToastNotification(parent, f"{title}: {message}")

from qt_material import apply_stylesheet

from PyQt6.QtWidgets import (
    QLabel, QLineEdit, QComboBox, QPushButton, QTextEdit,
    QSpinBox, QSlider, QToolButton, QCheckBox, QTabBar,
    QMainWindow, QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt6.QtGui import QFont

# ─── Label shims ────────────────────────────────────────────────────────────
def TitleLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 15); f.setBold(True); lbl.setFont(f)
    return lbl

def SubtitleLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 11); f.setBold(True); lbl.setFont(f)
    return lbl

def StrongBodyLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 9); f.setBold(True); lbl.setFont(f)
    lbl.setWordWrap(True)
    return lbl

def BodyLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 9); lbl.setFont(f)
    lbl.setWordWrap(True)
    return lbl

# ─── Input shims ────────────────────────────────────────────────────────────
class LineEdit(QLineEdit):
    pass

class ComboBox(QComboBox):
    pass

class PushButton(QPushButton):
    pass

class TextEdit(QTextEdit):
    pass

class SpinBox(QSpinBox):
    pass

class ToolButton(QToolButton):
    pass

class PasswordLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEchoMode(QLineEdit.EchoMode.Password)

class Slider(QSlider):
    pass

class SwitchButton(QCheckBox):
    pass

# ─── Card shim ──────────────────────────────────────────────────────────────
class CardWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)

# ─── Segmented / Pivot shim ─────────────────────────────────────────────────
class SegmentedWidget(QWidget):
    """addItem(key, text, callback) compatible shim using QTabBar internally."""
    from PyQt6.QtCore import pyqtSignal as _sig
    currentItemChanged = _sig(str)   # emits key string like original Fluent widget

    def __init__(self, parent=None):
        super().__init__(parent)
        _layout = QHBoxLayout(self)
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.setSpacing(0)
        self._bar = QTabBar(self)
        self._bar.setExpanding(False)
        _layout.addWidget(self._bar)
        _layout.addStretch(1)
        self._keys = {}          # idx -> key
        self._callbacks = {}     # idx -> callback
        self._bar.currentChanged.connect(self._on_changed)

    def addItem(self, key, text, callback=None, icon=None):
        idx = self._bar.addTab(text)
        self._keys[idx] = key
        self._callbacks[idx] = callback

    def setCurrentItem(self, key):
        for idx, k in self._keys.items():
            if k == key:
                self._bar.setCurrentIndex(idx)
                return

    def setCurrentIndex(self, idx):
        self._bar.setCurrentIndex(idx)

    def _on_changed(self, idx):
        key = self._keys.get(idx, '')
        self.currentItemChanged.emit(key)
        cb = self._callbacks.get(idx)
        if cb: cb()

class Pivot(SegmentedWidget):
    pass

# ─── Navigation window shim ─────────────────────────────────────────────────
class MSFluentWindow(QMainWindow):
    """Left-sidebar navigation window relying on qt-material styles."""
    def __init__(self):
        super().__init__()
        _central = QWidget()
        _h = QHBoxLayout(_central)
        _h.setContentsMargins(10, 10, 10, 10)
        _h.setSpacing(10)
        self.setCentralWidget(_central)

        self._nav = QListWidget()
        self._nav.setMaximumWidth(150)
        self._nav.setMinimumWidth(130)
        # We DO NOT apply any manual stylesheets here, let qt-material handle it!
        self._nav.currentRowChanged.connect(self._on_nav_changed)
        
        _h.addWidget(self._nav)

        self._stack = QStackedWidget()
        _h.addWidget(self._stack, 1)

    def addSubInterface(self, widget, icon, label, position=None):
        item = QListWidgetItem(label)
        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self._nav.addItem(item)
        self._stack.addWidget(widget)

    def _on_nav_changed(self, row):
        if 0 <= row < self._stack.count():
            self._stack.setCurrentIndex(row)


# ─── Stub constants ─────────────────────────────────────────────────────────
class NavigationItemPosition:
    BOTTOM = "bottom"

class FluentIcon:
    DOCUMENT = PIE_SINGLE = PHOTO = CHAT = FOLDER = None

class Theme:
    LIGHT = "light"; DARK = "dark"; AUTO = "auto"

def setTheme(theme): pass
def setThemeColor(color): pass
def setFont(widget, size, weight=None): pass


from nmsc_climate_toolbox import nct



from PyQt6.QtWidgets import QDateEdit, QSlider, QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6.QtCore import QDate


from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QDoubleSpinBox
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal

class CustomRangeSlider(QWidget):
    valueChanged = pyqtSignal(float, float)
    
    def __init__(self, min_val, max_val, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(40)
        self.min_val = min_val
        self.max_val = max_val
        self.low = min_val
        self.high = max_val
        
        self.handle_width = 16
        self.active_handle = None
        
    def _val_to_x(self, val):
        w = self.width() - self.handle_width
        if w <= 0: return self.handle_width // 2
        ratio = (val - self.min_val) / (self.max_val - self.min_val + 1e-9)
        return int(ratio * w) + self.handle_width // 2
        
    def _x_to_val(self, x):
        w = self.width() - self.handle_width
        if w <= 0: return self.min_val
        x = max(self.handle_width // 2, min(x, self.width() - self.handle_width // 2))
        ratio = (x - self.handle_width // 2) / float(w)
        return self.min_val + ratio * (self.max_val - self.min_val)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        track_y = self.height() // 2 - 2
        track_rect = QRect(self.handle_width // 2, track_y, self.width() - self.handle_width, 4)
        painter.setBrush(QColor("#444444"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, 2, 2)
        
        x1 = self._val_to_x(self.low)
        x2 = self._val_to_x(self.high)
        highlight_rect = QRect(x1, track_y, x2 - x1, 4)
        painter.setBrush(QColor("#03A9F4"))
        painter.drawRoundedRect(highlight_rect, 2, 2)
        
        painter.setBrush(QColor("#FFFFFF"))
        painter.setPen(QPen(QColor("#888888"), 1))
        
        h1_rect = QRect(x1 - self.handle_width // 2, self.height() // 2 - 8, self.handle_width, 16)
        painter.drawEllipse(h1_rect)
        
        h2_rect = QRect(x2 - self.handle_width // 2, self.height() // 2 - 8, self.handle_width, 16)
        painter.drawEllipse(h2_rect)
        
    def mousePressEvent(self, event):
        x = event.pos().x()
        x1 = self._val_to_x(self.low)
        x2 = self._val_to_x(self.high)
        
        d1 = abs(x - x1)
        d2 = abs(x - x2)
        
        if d1 < 15 and d1 <= d2:
            self.active_handle = 0
        elif d2 < 15:
            self.active_handle = 1
        else:
            self.active_handle = None
            
    def mouseMoveEvent(self, event):
        if self.active_handle is not None:
            val = self._x_to_val(event.pos().x())
            if self.active_handle == 0:
                self.low = min(val, self.high - 0.01)
            else:
                self.high = max(val, self.low + 0.01)
            self.update()
            self.valueChanged.emit(self.low, self.high)
            
    def mouseReleaseEvent(self, event):
        self.active_handle = None

    def set_range(self, low, high):
        self.low = max(self.min_val, min(low, self.high))
        self.high = min(self.max_val, max(high, self.low))
        self.update()


class FloatSlider(QWidget):
    def __init__(self, label, min_val=-180.0, max_val=180.0, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        layout.addWidget(StrongBodyLabel(f"{label} 범위:"))
        
        self.slider = CustomRangeSlider(min_val, max_val)
        
        h_spin = QHBoxLayout()
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setRange(min_val, max_val)
        self.spin_min.setSingleStep(0.1)
        self.spin_min.setValue(min_val)
        
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(min_val, max_val)
        self.spin_max.setSingleStep(0.1)
        self.spin_max.setValue(max_val)
        
        h_spin.addWidget(BodyLabel("Min:"))
        h_spin.addWidget(self.spin_min)
        h_spin.addStretch(1)
        h_spin.addWidget(BodyLabel("Max:"))
        h_spin.addWidget(self.spin_max)
        
        layout.addWidget(self.slider)
        layout.addLayout(h_spin)
        
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.spin_min.valueChanged.connect(self.on_spin_changed)
        self.spin_max.valueChanged.connect(self.on_spin_changed)
        
    def on_slider_changed(self, low, high):
        self.spin_min.blockSignals(True)
        self.spin_max.blockSignals(True)
        self.spin_min.setValue(low)
        self.spin_max.setValue(high)
        self.spin_min.blockSignals(False)
        self.spin_max.blockSignals(False)
        
    def on_spin_changed(self):
        low = self.spin_min.value()
        high = self.spin_max.value()
        if low > high:
            low = high
            self.spin_min.blockSignals(True)
            self.spin_min.setValue(low)
            self.spin_min.blockSignals(False)
            
        self.slider.set_range(low, high)

    def get_min(self): return self.spin_min.value()
    def get_max(self): return self.spin_max.value()
    
    def set_min(self, val): 
        self.spin_min.setValue(val)
        self.slider.set_range(self.spin_min.value(), self.spin_max.value())
        
    def set_max(self, val): 
        self.spin_max.setValue(val)
        self.slider.set_range(self.spin_min.value(), self.spin_max.value())


class DateSlider(QWidget):
    def __init__(self, label, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        
        layout.addWidget(StrongBodyLabel(f"{label}:"))
        
        # Arbitrary integer range for dates (e.g., year 1970 to 2050 in days)
        # We will dynamically set the range when a file is loaded
        self.min_days = 0
        self.max_days = 36500 
        self.slider = CustomRangeSlider(self.min_days, self.max_days)
        
        h_date = QHBoxLayout()
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        
        h_date.addWidget(BodyLabel("시작:"))
        h_date.addWidget(self.date_start)
        h_date.addStretch(1)
        h_date.addWidget(BodyLabel("종료:"))
        h_date.addWidget(self.date_end)
        
        layout.addWidget(self.slider)
        layout.addLayout(h_date)
        
        self.slider.valueChanged.connect(self.on_slider_changed)
        self.date_start.dateChanged.connect(self.on_date_changed)
        self.date_end.dateChanged.connect(self.on_date_changed)
        
        self.is_updating = False

    def on_slider_changed(self, low, high):
        if self.is_updating: return
        self.is_updating = True
        try:
            d1 = QDate(1970, 1, 1).addDays(int(low))
            d2 = QDate(1970, 1, 1).addDays(int(high))
            self.date_start.setDate(d1)
            self.date_end.setDate(d2)
        finally:
            self.is_updating = False
            
    def on_date_changed(self):
        if self.is_updating: return
        self.is_updating = True
        try:
            d1 = self.date_start.date()
            d2 = self.date_end.date()
            if d1 > d2:
                d1 = d2
                self.date_start.setDate(d1)
                
            low = QDate(1970, 1, 1).daysTo(d1)
            high = QDate(1970, 1, 1).daysTo(d2)
            self.slider.set_range(low, high)
        finally:
            self.is_updating = False

    def set_range(self, start_date, end_date):
        self.is_updating = True
        try:
            self.min_days = QDate(1970, 1, 1).daysTo(start_date)
            self.max_days = QDate(1970, 1, 1).daysTo(end_date)
            self.slider.min_val = self.min_days
            self.slider.max_val = self.max_days
            self.slider.set_range(self.min_days, self.max_days)
            self.date_start.setDate(start_date)
            self.date_end.setDate(end_date)
        finally:
            self.is_updating = False

class PreprocessInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("PreprocessInterface")
        self.files = []
        self.valid_files = []
        self.init_ui()
        
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        title = TitleLabel("데이터 준비 (Preprocess)")
        main_layout.addWidget(title)
        
        self.segment = SegmentedWidget(self)
        main_layout.addWidget(self.segment, 0, Qt.AlignmentFlag.AlignLeft)
        
        self.stack = QStackedWidget()
        
        # --- Input Data Page ---
        page_input = QWidget()
        h_input = QHBoxLayout(page_input)
        h_input.setContentsMargins(0, 20, 0, 0)
        
        left_input = CardWidget()
        v_left_input = QVBoxLayout(left_input)
        v_left_input.setContentsMargins(20, 20, 20, 20)
        v_left_input.addWidget(SubtitleLabel("위성/수치모델 입력 자료"))
        
        # Row 1: File Selection
        h_file = QHBoxLayout()
        self.file_combo = ComboBox()
        self.file_combo.currentTextChanged.connect(self.on_file_changed)
        btn_browse = PushButton("파일 추가...")
        btn_browse.clicked.connect(self.browse_file)
        h_file.addWidget(self.file_combo, 1)
        h_file.addWidget(btn_browse)
        v_left_input.addLayout(h_file)
        
        # Row 2: Variable Selection
        h_var = QHBoxLayout()
        h_var.addWidget(StrongBodyLabel("주요 변수 선택:"))
        self.var_combo = ComboBox()
        self.var_combo.currentTextChanged.connect(lambda: self.update_overview())
        h_var.addWidget(self.var_combo, 1)
        v_left_input.addLayout(h_var)
        
        # Row 3: Calendar Date Range
        v_left_input.addWidget(StrongBodyLabel("분석 기간 지정"))
        h_date = QHBoxLayout()
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        h_date.addWidget(self.date_start)
        h_date.addWidget(BodyLabel(" ~ "))
        h_date.addWidget(self.date_end)
        v_left_input.addLayout(h_date)
        
        # Row 4: Spatial Range Sliders
        v_left_input.addWidget(StrongBodyLabel("공간 범위 지정"))
        self.lon_slider = FloatSlider("경도 (Lon)")
        self.lat_slider = FloatSlider("위도 (Lat)", -90.0, 90.0)
        v_left_input.addWidget(self.lon_slider)
        v_left_input.addWidget(self.lat_slider)
        
        v_left_input.addSpacing(10)
        self.btn_apply = PushButton("적용 및 자르기 (Subset)")
        self.btn_apply.setMinimumHeight(40)
        self.btn_apply.clicked.connect(self.on_apply_settings)
        v_left_input.addWidget(self.btn_apply)
        v_left_input.addStretch(1)
        
        right_input = CardWidget()
        v_right_input = QVBoxLayout(right_input)
        v_right_input.addWidget(SubtitleLabel("입력 데이터셋 상세 정보"))
        self.overview_table = TextEdit()
        self.overview_table.setReadOnly(True)
        self.overview_table.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; border-radius: 4px; padding: 10px;")
        v_right_input.addWidget(self.overview_table)
        
        splitter_input = QSplitter(Qt.Orientation.Horizontal)
        splitter_input.addWidget(left_input)
        splitter_input.addWidget(right_input)
        splitter_input.setStretchFactor(0, 1)
        splitter_input.setStretchFactor(1, 1)
        h_input.addWidget(splitter_input)
        self.stack.addWidget(page_input)
        
        
        # --- Validation Data Page ---
        page_valid = QWidget()
        h_valid = QHBoxLayout(page_valid)
        h_valid.setContentsMargins(0, 20, 0, 0)
        
        left_valid = CardWidget()
        v_left_valid = QVBoxLayout(left_valid)
        v_left_valid.setContentsMargins(20, 20, 20, 20)
        v_left_valid.addWidget(SubtitleLabel("현장 관측(부이 등) 검증 자료"))
        
        # Row 1
        h_vfile = QHBoxLayout()
        self.valid_file_combo = ComboBox()
        self.valid_file_combo.currentTextChanged.connect(self.on_valid_file_changed)
        btn_vbrowse = PushButton("파일 추가...")
        btn_vbrowse.clicked.connect(self.browse_valid_file)
        h_vfile.addWidget(self.valid_file_combo, 1)
        h_vfile.addWidget(btn_vbrowse)
        v_left_valid.addLayout(h_vfile)
        
        # Row 2
        h_vvar = QHBoxLayout()
        h_vvar.addWidget(StrongBodyLabel("주요 변수 선택:"))
        self.valid_var_combo = ComboBox()
        self.valid_var_combo.currentTextChanged.connect(lambda: self.update_valid_overview())
        h_vvar.addWidget(self.valid_var_combo, 1)
        v_left_valid.addLayout(h_vvar)
        
        # Row 3
        v_left_valid.addWidget(StrongBodyLabel("분석 기간 지정"))
        h_vdate = QHBoxLayout()
        self.vdate_start = QDateEdit()
        self.vdate_start.setCalendarPopup(True)
        self.vdate_end = QDateEdit()
        self.vdate_end.setCalendarPopup(True)
        h_vdate.addWidget(self.vdate_start)
        h_vdate.addWidget(BodyLabel(" ~ "))
        h_vdate.addWidget(self.vdate_end)
        v_left_valid.addLayout(h_vdate)
        
        # Row 4
        v_left_valid.addWidget(StrongBodyLabel("공간 범위 지정"))
        self.vlon_slider = FloatSlider("경도 (Lon)")
        self.vlat_slider = FloatSlider("위도 (Lat)", -90.0, 90.0)
        v_left_valid.addWidget(self.vlon_slider)
        v_left_valid.addWidget(self.vlat_slider)
        
        v_left_valid.addStretch(1)
        
        right_valid = CardWidget()
        v_right_valid = QVBoxLayout(right_valid)
        v_right_valid.addWidget(SubtitleLabel("검증 데이터셋 상세 정보"))
        self.v_overview_table = TextEdit()
        self.v_overview_table.setReadOnly(True)
        self.v_overview_table.setStyleSheet("font-family: Consolas, monospace; font-size: 13px; background-color: #ffffff; color: #333333; border: 1px solid #cccccc; border-radius: 4px; padding: 10px;")
        v_right_valid.addWidget(self.v_overview_table)
        
        splitter_valid = QSplitter(Qt.Orientation.Horizontal)
        splitter_valid.addWidget(left_valid)
        splitter_valid.addWidget(right_valid)
        splitter_valid.setStretchFactor(0, 1)
        splitter_valid.setStretchFactor(1, 1)
        h_valid.addWidget(splitter_valid)
        self.stack.addWidget(page_valid)
        
        main_layout.addWidget(self.stack)
        
        self.segment.addItem("input", "위성/수치모델 입력 자료", lambda: self.stack.setCurrentIndex(0))
        self.segment.addItem("valid", "현장 관측 검증 자료", lambda: self.stack.setCurrentIndex(1))
        self.segment.setCurrentIndex(0)

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "NetCDF/GeoTIFF Files (*.nc *.tif *.tiff);;All Files (*)")
        if filepath:
            if filepath not in self.files:
                self.files.append(filepath)
                self.file_combo.addItem(filepath)
            self.file_combo.setCurrentText(filepath)

    def browse_valid_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Validation File", "", "NetCDF/GeoTIFF Files (*.nc *.tif *.tiff);;All Files (*)")
        if filepath:
            if filepath not in self.valid_files:
                self.valid_files.append(filepath)
                self.valid_file_combo.addItem(filepath)
            self.valid_file_combo.setCurrentText(filepath)

    def on_file_changed(self):
        import pandas as pd
        import traceback
        try:
            filepath = self.file_combo.currentText()
            if not filepath: return
            ds = nct.open(filepath)
            self.window().ds = ds
            data_vars = [var for var in ds.data_vars if 'bnds' not in var and 'bounds' not in var]
            
            self.var_combo.blockSignals(True)
            self.var_combo.clear()
            self.var_combo.addItems(data_vars)
            self.var_combo.blockSignals(False)
            
            if 'time' in ds.dims:
                try:
                    t_vals = pd.to_datetime(ds['time'].values)
                    if len(t_vals) > 0:
                        self.date_start.setDate(QDate(t_vals[0].year, t_vals[0].month, t_vals[0].day))
                        self.date_end.setDate(QDate(t_vals[-1].year, t_vals[-1].month, t_vals[-1].day))
                except Exception:
                    pass
            
            lon_name = next((n for n in ['lon', 'longitude', 'x'] if n in ds.coords), None)
            lat_name = next((n for n in ['lat', 'latitude', 'y'] if n in ds.coords), None)
            
            if lon_name:
                self.lon_slider.set_min(float(ds[lon_name].min().values))
                self.lon_slider.set_max(float(ds[lon_name].max().values))
            if lat_name:
                self.lat_slider.set_min(float(ds[lat_name].min().values))
                self.lat_slider.set_max(float(ds[lat_name].max().values))
                
            self.update_overview()
            self.on_apply_settings()
        except Exception as e:
            ToastNotification.show_toast(self, "파일 로드 오류", f"입력 데이터를 처리하는 중 오류가 발생했습니다\\n{e}")

    def on_valid_file_changed(self):
        import pandas as pd
        import traceback
        try:
            filepath = self.valid_file_combo.currentText()
            if not filepath: return
            vds = nct.open(filepath)
            self.window().valid_ds = vds
            data_vars = [var for var in vds.data_vars if 'bnds' not in var and 'bounds' not in var]
            
            self.valid_var_combo.blockSignals(True)
            self.valid_var_combo.clear()
            self.valid_var_combo.addItems(data_vars)
            self.valid_var_combo.blockSignals(False)
            
            if 'time' in vds.dims:
                try:
                    t_vals = pd.to_datetime(vds['time'].values)
                    if len(t_vals) > 0:
                        self.vdate_start.setDate(QDate(t_vals[0].year, t_vals[0].month, t_vals[0].day))
                        self.vdate_end.setDate(QDate(t_vals[-1].year, t_vals[-1].month, t_vals[-1].day))
                except Exception:
                    pass
            
            lon_name = next((n for n in ['lon', 'longitude', 'x'] if n in vds.coords), None)
            lat_name = next((n for n in ['lat', 'latitude', 'y'] if n in vds.coords), None)
            
            if lon_name:
                self.vlon_slider.set_min(float(vds[lon_name].min().values))
                self.vlon_slider.set_max(float(vds[lon_name].max().values))
            if lat_name:
                self.vlat_slider.set_min(float(vds[lat_name].min().values))
                self.vlat_slider.set_max(float(vds[lat_name].max().values))
            
            self.update_valid_overview()
            self.on_apply_settings()
        except Exception as e:
            ToastNotification.show_toast(self, "파일 로드 오류", f"검증 데이터를 처리하는 중 오류가 발생했습니다\\n{e}")

    def get_table_html(self, ds, var_name, title):
        if ds is None:
            return f"<html><body style='background-color:#2a2a2a; color:#f0f0f0; font-family:Segoe UI; padding:20px;'><h2>{title}이(가) 없습니다.</h2></body></html>"
        
        dims_str = ", ".join(f"{k}: {v}" for k, v in ds.sizes.items())
        
        html = f"""
        <html>
        <head>
        <style>
            body {{ background-color:#2a2a2a; color:#f0f0f0; font-family: 'Segoe UI', sans-serif; padding: 15px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background-color: #383838; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #444; }}
            th {{ background-color: #03A9F4; color: white; font-weight: bold; width: 30%; }}
            tr:last-child td {{ border-bottom: none; }}
            tr:hover {{ background-color: #404040; }}
            h2 {{ color: #03A9F4; border-bottom: 2px solid #444; padding-bottom: 8px; }}
        </style>
        </head>
        <body>
            <h2>{title} 요약</h2>
            <table>
                <tr><th>차원 (Dimensions)</th><td>{dims_str}</td></tr>
                <tr><th>선택된 변수</th><td>{var_name if var_name else '선택 안됨'}</td></tr>
        """
        
        lon_name = next((n for n in ['lon', 'longitude', 'x'] if n in ds.coords), None)
        lat_name = next((n for n in ['lat', 'latitude', 'y'] if n in ds.coords), None)
        
        if lon_name and lat_name:
            html += f"<tr><th>경도 (Longitude) 범위</th><td>{float(ds[lon_name].min().values):.2f} ~ {float(ds[lon_name].max().values):.2f}</td></tr>"
            html += f"<tr><th>위도 (Latitude) 범위</th><td>{float(ds[lat_name].min().values):.2f} ~ {float(ds[lat_name].max().values):.2f}</td></tr>"
            
        if 'time' in ds.dims:
            import pandas as pd
            t_vals = pd.to_datetime(ds['time'].values)
            if len(t_vals) > 0:
                html += f"<tr><th>시간 (Time) 범위</th><td>{t_vals[0].strftime('%Y-%m-%d')} ~ {t_vals[-1].strftime('%Y-%m-%d')}</td></tr>"
                
        html += "</table>"
        
        if hasattr(ds, 'attrs') and ds.attrs:
            html += "<h2 style='margin-top:20px;'>속성 정보 (Attributes)</h2><table>"
            for k, v in ds.attrs.items():
                v_str = str(v)
                if len(v_str) > 100: v_str = v_str[:100] + '...'
                html += f"<tr><th>{k}</th><td>{v_str}</td></tr>"
            html += "</table>"
            
        html += """
        </body>
        </html>
        """
        return html

    def update_overview(self):
        ds = self.window().ds
        if ds is None:
            self.overview_table.setPlainText("데이터가 없습니다.")
            return
        
        info_str = str(ds)
        self.overview_table.setPlainText(info_str)

    def update_valid_overview(self):
        vds = self.window().valid_ds
        if vds is None:
            self.v_overview_table.setPlainText("검증 데이터가 없습니다.")
            return
            
        info_str = str(vds)
        self.v_overview_table.setPlainText(info_str)

    def on_apply_settings(self):
        self.window().selected_var = self.var_combo.currentText()
        self.window().selected_valid_var = self.valid_var_combo.currentText()
        
        try:
            self.window().bounds = {
                'min_lon': self.lon_slider.get_min(),
                'max_lon': self.lon_slider.get_max(),
                'min_lat': self.lat_slider.get_min(),
                'max_lat': self.lat_slider.get_max()
            }
        except:
            pass


from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QFrame

class CalculateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("CalculateInterface")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        title = TitleLabel("산출 (Calculation & R-Toolbox)")
        main_layout.addWidget(title)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Toolbox List
        left_widget = CardWidget()
        v_left = QVBoxLayout(left_widget)
        v_left.addWidget(SubtitleLabel("분석 툴박스 목록"))
        
        self.list_toolbox = QListWidget()
        self.list_toolbox.addItem("1. 기후 평년값 및 편차 산출 (Climatology & Anomaly)")
        self.list_toolbox.addItem("2. 공간 평균/합산 (Spatial Aggregation)")
        self.list_toolbox.addItem("3. 시계열 추세 분석 (Time-series Trend)")
        self.list_toolbox.currentRowChanged.connect(self.on_toolbox_changed)
        v_left.addWidget(self.list_toolbox)
        
        splitter.addWidget(left_widget)
        
        # Right Panel: Toolbox Config Stack
        right_widget = CardWidget()
        v_right = QVBoxLayout(right_widget)
        self.stack = QStackedWidget()
        
        # Tool 1: Climatology & Anomaly
        page_1 = QWidget()
        v_page_1 = QVBoxLayout(page_1)
        v_page_1.addWidget(SubtitleLabel("평년값 및 편차 산출 옵션"))
        
        h_cli = QHBoxLayout()
        self.sw_cli = SwitchButton()
        self.sw_cli.setText("평년값 계산 활성화")
        h_cli.addWidget(self.sw_cli)
        self.txt_cli_start = LineEdit()
        self.txt_cli_start.setPlaceholderText("시작 (예: 1991)")
        self.txt_cli_end = LineEdit()
        self.txt_cli_end.setPlaceholderText("종료 (예: 2020)")
        h_cli.addWidget(StrongBodyLabel("기준 연도:"))
        h_cli.addWidget(self.txt_cli_start)
        h_cli.addWidget(StrongBodyLabel("~"))
        h_cli.addWidget(self.txt_cli_end)
        h_cli.addStretch(1)
        v_page_1.addLayout(h_cli)
        
        h_ano = QHBoxLayout()
        self.sw_ano = SwitchButton()
        self.sw_ano.setText("편차(Anomaly) 계산 활성화")
        h_ano.addWidget(self.sw_ano)
        h_ano.addStretch(1)
        v_page_1.addLayout(h_ano)
        v_page_1.addStretch(1)
        
        btn_calc_1 = PushButton("계산 실행")
        btn_calc_1.clicked.connect(self.run_climatology)
        v_page_1.addWidget(btn_calc_1)
        self.stack.addWidget(page_1)
        
        # Tool 2: Spatial Aggregation
        page_2 = QWidget()
        v_page_2 = QVBoxLayout(page_2)
        v_page_2.addWidget(SubtitleLabel("공간 평균/합산 옵션"))
        h_sp = QHBoxLayout()
        self.cb_sp_method = ComboBox()
        self.cb_sp_method.addItems(["Mean (평균)", "Sum (합계)", "Max (최대)", "Min (최소)"])
        h_sp.addWidget(StrongBodyLabel("연산 방식:"))
        h_sp.addWidget(self.cb_sp_method)
        h_sp.addStretch(1)
        v_page_2.addLayout(h_sp)
        v_page_2.addStretch(1)
        btn_calc_2 = PushButton("계산 실행")
        btn_calc_2.clicked.connect(self.run_spatial)
        v_page_2.addWidget(btn_calc_2)
        self.stack.addWidget(page_2)
        
        # Tool 3: Trend
        page_3 = QWidget()
        v_page_3 = QVBoxLayout(page_3)
        v_page_3.addWidget(SubtitleLabel("시계열 추세 분석 옵션"))
        v_page_3.addWidget(BodyLabel("선형 회귀(Linear Regression) 기반 트렌드 분석을 수행합니다."))
        v_page_3.addStretch(1)
        btn_calc_3 = PushButton("추세 계산 실행")
        btn_calc_3.clicked.connect(self.run_trend)
        v_page_3.addWidget(btn_calc_3)
        self.stack.addWidget(page_3)
        
        v_right.addWidget(self.stack)
        splitter.addWidget(right_widget)
        
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter)
        
        self.list_toolbox.setCurrentRow(0)

    def on_toolbox_changed(self, index):
        self.stack.setCurrentIndex(index)

    def run_climatology(self):
        w = self.window()
        if w.ds is None:
            ToastNotification.show_toast(self, "오류", "먼저 데이터를 불러오세요.")
            return
        try:
            var_name = w.selected_var
            if self.sw_cli.isChecked():
                ds_cli = w.ds.groupby('time.month').mean('time')
                w.calculated_ds = ds_cli
                ToastNotification.show_toast(self, "완료", f"{var_name} 평년값 계산 완료.")
            elif self.sw_ano.isChecked():
                ds_cli = w.ds.groupby('time.month').mean('time')
                ds_ano = w.ds.groupby('time.month') - ds_cli
                w.calculated_ds = ds_ano
                ToastNotification.show_toast(self, "완료", f"{var_name} 편차 계산 완료.")
        except Exception as e:
            ToastNotification.show_toast(self, "오류", f"계산 중 오류: {e}")

    def run_spatial(self):
        w = self.window()
        if w.ds is None:
            ToastNotification.show_toast(self, "오류", "먼저 데이터를 불러오세요.")
            return
        ToastNotification.show_toast(self, "알림", "공간 연산 수행 완료.")
        
    def run_trend(self):
        ToastNotification.show_toast(self, "알림", "트렌드 연산 모듈이 호출되었습니다.")


from PyQt6.QtCore import QThread, pyqtSignal

class MapPlotThread(QThread):
    finished = pyqtSignal(str, str)
    error_occurred = pyqtSignal(str)

    def __init__(self, ds, var_name, time_idx, cmap, bounds, show_basemap, raw_mode=False, data_layer='original'):
        super().__init__()
        self.ds = ds
        self.var_name = var_name
        self.time_idx = time_idx
        self.cmap = cmap
        self.bounds = bounds
        self.show_basemap = show_basemap
        self.raw_mode = raw_mode
        self.data_layer = data_layer

    def run(self):
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import base64
            
            lat_dim = 'lat' if 'lat' in self.ds.dims else ('latitude' if 'latitude' in self.ds.dims else None)
            lon_dim = 'lon' if 'lon' in self.ds.dims else ('longitude' if 'longitude' in self.ds.dims else None)
            
            if not lat_dim or not lon_dim:
                self.error_occurred.emit("위도/경도 차원을 찾을 수 없습니다.")
                return
                
            if 'time' in self.ds.dims:
                data = self.ds[self.var_name].isel(time=self.time_idx)
            else:
                data = self.ds[self.var_name]
                
            
            lon_vals = data[lon_dim].values
            lat_vals = data[lat_dim].values
            min_lon, max_lon = float(lon_vals.min()), float(lon_vals.max())
            min_lat, max_lat = float(lat_vals.min()), float(lat_vals.max())
            
            data_2d = data.values
            if data_2d.ndim > 2:
                data_2d = data_2d.squeeze()
            while data_2d.ndim > 2:
                data_2d = data_2d[0]
                
            
            import matplotlib
            if self.data_layer == 'anomaly':
                SST_ANOM_COLORS = [
                    '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                    '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                    '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                    '#A00000', '#800000', '#600000'
                ]
                from matplotlib.colors import ListedColormap
                self.cmap = ListedColormap(SST_ANOM_COLORS)
                vmin, vmax = -6.0, 6.0
            else:
                vmin, vmax = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))
                
            # Determine origin and calculate exact pixel bounds
            img_origin = 'upper' if lat_vals[0] > lat_vals[-1] else 'lower'
            
            h, w = data_2d.shape
            dx = (max_lon - min_lon) / max(1, w - 1)
            dy = (max_lat - min_lat) / max(1, h - 1)
            
            # Matplotlib extent: [left, right, bottom, top]
            ext_left, ext_right = min_lon - dx/2, max_lon + dx/2
            ext_bottom, ext_top = min_lat - dy/2, max_lat + dy/2
            
            # OpenLayers extent: [minX, minY, maxX, maxY]
            
            try:
                import cartopy.crs as ccrs
                has_cartopy = True
            except ImportError:
                has_cartopy = False
            
            fig = plt.figure(figsize=(12, 12), dpi=250, frameon=False)
            
            if has_cartopy and not self.raw_mode:
                crs_3857 = ccrs.Mercator.GOOGLE
                crs_4326 = ccrs.PlateCarree()
                ax = fig.add_axes([0., 0., 1., 1.], projection=crs_3857)
                ax.set_axis_off()
                
                if self.data_layer == 'anomaly':
                    SST_ANOM_COLORS = [
                        '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                        '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                        '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                        '#A00000', '#800000', '#600000'
                    ]
                    from matplotlib.colors import ListedColormap
                    cmap = ListedColormap(SST_ANOM_COLORS)
                    vmin, vmax = -6.0, 6.0
                else:
                    vmin, vmax = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))
                    import matplotlib
                    cmap = self.cmap if isinstance(self.cmap, matplotlib.colors.Colormap) else plt.get_cmap(self.cmap)
                
                cmap.set_bad('none')
                
                ax.imshow(data_2d, origin=img_origin, extent=[ext_left, ext_right, ext_bottom, ext_top], 
                          transform=crs_4326, cmap=cmap, vmin=vmin, vmax=vmax, regrid_shape=max(1500, max(h, w)), interpolation='nearest')
                          
                ax.set_extent([ext_left, ext_right, ext_bottom, ext_top], crs=crs_4326)
                
                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                ol_extent = [float(x0), float(y0), float(x1), float(y1)]
                projection_epsg = "EPSG:3857"
            else:
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                
                if self.data_layer == 'anomaly':
                    SST_ANOM_COLORS = [
                        '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                        '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                        '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                        '#A00000', '#800000', '#600000'
                    ]
                    from matplotlib.colors import ListedColormap
                    cmap = ListedColormap(SST_ANOM_COLORS)
                    vmin, vmax = -6.0, 6.0
                else:
                    vmin, vmax = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))
                    import matplotlib
                    cmap = self.cmap if isinstance(self.cmap, matplotlib.colors.Colormap) else plt.get_cmap(self.cmap)
                
                cmap.set_bad('none')
                
                ax.imshow(data_2d, extent=[ext_left, ext_right, ext_bottom, ext_top], origin=img_origin, 
                          cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
                ol_extent = [ext_left, ext_bottom, ext_right, ext_top]
                projection_epsg = "EPSG:4326"
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', transparent=True, pad_inches=0, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            
            # Downsample data for JS tooltip lookup (max 200x200 to keep HTML small)
            h, w = data_2d.shape
            step_y = max(1, h // 200)
            step_x = max(1, w // 200)
            ds_data = data_2d[::step_y, ::step_x]
            ds_data = np.where(np.isnan(ds_data), "null", np.round(ds_data, 2))
            
            # Create JS nested array string
            js_grid = "[" + ",".join("[" + ",".join(map(str, row)) + "]" for row in ds_data) + "]"
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <link rel="stylesheet" href="https://cdn.rawgit.com/openlayers/openlayers.github.io/master/en/v5.3.0/css/ol.css" type="text/css">
                <style>
                    html, body {{ margin:0; padding:0; height:100%; background:#f0f2f5; font-family:sans-serif; }}
                    #map {{ width: 100%; height: 100%; }}
                    #tooltip {{
                        position: absolute;
                        background: rgba(0,0,0,0.8);
                        color: white;
                        padding: 5px 10px;
                        border-radius: 4px;
                        font-size: 12px;
                        pointer-events: none;
                        display: none;
                        z-index: 1000;
                    }}
                </style>
                <script src="https://cdn.rawgit.com/openlayers/openlayers.github.io/master/en/v5.3.0/build/ol.js"></script>
            </head>
            <body>
                <div id="map"></div>
                <div id="tooltip"></div>
                <script>
                    var extent = ol.proj.transformExtent([{ext_left}, {ext_bottom}, {ext_right}, {ext_top}], 'EPSG:4326', 'EPSG:3857');
                    
                    var layers = [];
                    if ({str(self.show_basemap).lower()}) {{
                        layers.push(new ol.layer.Tile({{ source: new ol.source.OSM() }}));
                    }}
                    
                    var imageLayer = new ol.layer.Image({{
                        source: new ol.source.ImageStatic({{
                            url: 'data:image/png;base64,{img_b64}',
                            projection: 'EPSG:3857',
                            imageExtent: extent
                        }}),
                        opacity: 0.8
                    }});
                    layers.push(imageLayer);

                    var map = new ol.Map({{
                        target: 'map',
                        layers: layers,
                        view: new ol.View({{
                            center: ol.proj.fromLonLat([{(min_lon+max_lon)/2}, {(min_lat+max_lat)/2}]),
                            zoom: 4
                        }})
                    }});
                    map.getView().fit(extent, map.getSize());
                    
                    var gridData = {js_grid};
                    var minLon = {min_lon};
                    var maxLon = {max_lon};
                    var minLat = {min_lat};
                    var maxLat = {max_lat};
                    var gridRows = gridData.length;
                    var gridCols = gridData[0].length;
                    var tooltip = document.getElementById('tooltip');
                    
                    map.on('pointermove', function(evt) {{
                        if (evt.dragging) {{ tooltip.style.display = 'none'; return; }}
                        var coord4326 = ol.proj.transform(evt.coordinate, 'EPSG:3857', 'EPSG:4326');
                        var lon = coord4326[0];
                        var lat = coord4326[1];
                        
                        if (lon >= minLon && lon <= maxLon && lat >= minLat && lat <= maxLat) {{
                            var px = (lon - minLon) / (maxLon - minLon);
                            var py = (lat - minLat) / (maxLat - minLat);
                            var col = Math.floor(px * gridCols);
                            var row = Math.floor(py * gridRows);
                            
                            if (row >= 0 && row < gridRows && col >= 0 && col < gridCols) {{
                                var val = gridData[row][col];
                                if (val !== null) {{
                                    tooltip.innerHTML = "Lon: " + lon.toFixed(2) + "<br>Lat: " + lat.toFixed(2) + "<br><b>Value: " + val + "</b>";
                                    tooltip.style.display = 'block';
                                    tooltip.style.left = (evt.pixel[0] + 15) + 'px';
                                    tooltip.style.top = (evt.pixel[1] + 15) + 'px';
                                    return;
                                }}
                            }}
                        }}
                        tooltip.style.display = 'none';
                    }});
                </script>
            </body>
            </html>
            """
            self.finished.emit(html, img_b64)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

class VisualizeInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("VisualizeInterface")
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        title = TitleLabel("시각화 (Visualization)")
        main_layout.addWidget(title)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE (Options) ---
        left_widget = CardWidget()
        v_left = QVBoxLayout(left_widget)
        v_left.setContentsMargins(10, 10, 10, 10)
        
        v_left.addWidget(SubtitleLabel("시각화 옵션 설정"))
        
        h_layer = QHBoxLayout()
        h_layer.addWidget(StrongBodyLabel("데이터 레이어 (Data Layer):"))
        self.cb_layer = ComboBox()
        self.cb_layer.addItem("원본 영상 (Original)", "original")
        self.cb_layer.addItem("평년 영상 (Climatology)", "climatology")
        self.cb_layer.addItem("아노말리 영상 (Anomaly)", "anomaly")
        self.cb_layer.addItem("트렌드 영상 (Trend)", "trend")
        self.cb_layer.currentIndexChanged.connect(self.refresh_current_plot)
        h_layer.addWidget(self.cb_layer, 1)
        v_left.addLayout(h_layer)
        
        h_cmap = QHBoxLayout()
        h_cmap.addWidget(StrongBodyLabel("컬러맵 (Color Map):"))
        self.cb_cmap = ComboBox()
        self.cb_cmap.addItems(['RdYlBu_r', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'bwr', 'jet'])
        h_cmap.addWidget(self.cb_cmap, 1)
        v_left.addLayout(h_cmap)

        # Date Selection for Anomaly
        h_date = QHBoxLayout()
        h_date.addWidget(StrongBodyLabel("검색 기준 날짜:"))
        self.cb_date = ComboBox()
        self.cb_date.addItem("전체 기간 평균")
        # TODO: populate actual dates from dataset if time dim exists
        h_date.addWidget(self.cb_date, 1)
        v_left.addLayout(h_date)

        # Basemap Toggle
        h_base = QHBoxLayout()
        self.chk_basemap = QCheckBox("배경지도 표시 (Show Basemap)")
        self.chk_basemap.setChecked(True)
        self.chk_basemap.stateChanged.connect(self.toggle_basemap)
        h_base.addWidget(self.chk_basemap)
        v_left.addLayout(h_base)

        h_raw = QHBoxLayout()
        self.chk_raw_mode = QCheckBox("원본 데이터 유지 (Raw Mode)")
        self.chk_raw_mode.setChecked(False)
        self.chk_raw_mode.stateChanged.connect(self.refresh_current_plot)
        h_raw.addWidget(self.chk_raw_mode)
        v_left.addLayout(h_raw)

        v_left.addSpacing(10)
        from qfluentwidgets import PrimaryPushButton
        self.btn_ai = PrimaryPushButton("✨ AI 분석 요청")
        self.btn_ai.clicked.connect(self.request_ai_analysis)
        v_left.addWidget(self.btn_ai)


        
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
        self.pivot.setCurrentIndex(0)
        
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
        layer = self.cb_layer.currentData()
        
        if layer != 'original' and hasattr(w, 'results_dict') and w.results_dict.get(layer) is not None:
            return w.results_dict[layer]
            
        if hasattr(w, 'calculated_ds') and w.calculated_ds is not None:
            return w.calculated_ds
        elif hasattr(w, 'processed_ds') and w.processed_ds is not None:
            return w.processed_ds
        return w.ds

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
        time_idx = self.window().selected_time_idx
        
        if ds is None or not var_name:
            ToastNotification.show_toast(self, "오류", "데이터셋이 없습니다.")
            return

        self.clear_layout(self.map_canvas_layout)
        loading_label = TitleLabel("지도 그리는 중입니다... (최대 15초 소요)")
        self.map_canvas_layout.addWidget(loading_label)

        self.map_thread = MapPlotThread(
            ds=ds,
            var_name=var_name,
            time_idx=time_idx,
            cmap=self.cb_cmap.currentText(),
            bounds=self.window().bounds,
            show_basemap=self.chk_basemap.isChecked(), raw_mode=self.chk_raw_mode.isChecked(), data_layer=self.cb_layer.currentData()
        )
        self.map_thread.finished.connect(self.on_map_finished)
        self.map_thread.error_occurred.connect(lambda e: ToastNotification.show_toast(self, "오류", e))
        self.map_thread.start()

    def toggle_basemap(self, state):
        if hasattr(self, 'map_view') and self.map_view is not None:
            visibility = 'true' if state == 2 else 'false'
            self.map_view.page().runJavaScript(f"if (typeof tileLayer !== 'undefined') tileLayer.setVisible({visibility});")
        else:
            self.refresh_current_plot()

    def request_ai_analysis(self):
        if not hasattr(self, 'last_map_b64') or not self.last_map_b64:
            ToastNotification.show_toast(self, "오류", "분석할 지도가 아직 그려지지 않았습니다.")
            return
            
        var_name = self.window().selected_var
        prompt = f"현재 표시된 기상 데이터('{var_name}') 지도를 자세히 분석해 줘. 주요 기후 패턴, 아노말리(Anomaly) 특징, 그리고 이 지역의 날씨에 미칠 영향을 중심으로 설명해 줘."
        
        import tempfile
        import base64
        import os
        tmp_path = os.path.join(tempfile.gettempdir(), 'current_map.png')
        with open(tmp_path, 'wb') as f:
            f.write(base64.b64decode(self.last_map_b64))
            
        main_nav = self.window().nav_panel
        self.window().stacked_widget.setCurrentWidget(self.window().ai_interface)
        if hasattr(self.window().nav_panel, 'setCurrentItem'):
            self.window().nav_panel.setCurrentItem('ai')
        
        ai_interface = self.window().ai_interface
        ai_interface.txt_image_path.setText(tmp_path)
        ai_interface.txt_input.setPlainText(prompt)
        ai_interface.send_message()

    def on_map_finished(self, html, img_b64):
        self.last_map_b64 = img_b64
        self.clear_layout(self.map_canvas_layout)
        self.map_view = QWebEngineView()
        self.map_view.setHtml(html)
        self.map_canvas_layout.addWidget(self.map_view)

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

class AIAssistantInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("AIAssistantInterface")
        self.llm_thread = None
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        title = TitleLabel("AI 어시스턴트 (AI Chat)")
        main_layout.addWidget(title)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE: Config Area ---
        left_widget = CardWidget()
        v_config = QVBoxLayout(left_widget)
        v_config.setContentsMargins(20, 20, 20, 20)
        v_config.addWidget(SubtitleLabel("설정 정보 (Settings)"))
        
        h_engine = QHBoxLayout()
        h_engine.addWidget(StrongBodyLabel("AI 엔진 선택:"))
        self.combo_engine = ComboBox()
        self.combo_engine.addItems(["Gemini (Cloud API)", "Gemma (Local Vision)"])
        self.combo_engine.currentTextChanged.connect(self.on_engine_changed)
        h_engine.addWidget(self.combo_engine, 1)
        v_config.addLayout(h_engine)
        
        self.stack_config = QStackedWidget()
        
        # Gemini config
        page_gemini = QWidget()
        v_gemini = QVBoxLayout(page_gemini)
        v_gemini.setContentsMargins(0,10,0,0)
        v_gemini.addWidget(StrongBodyLabel("Gemini API Key:"))
        self.txt_api_key = LineEdit()
        self.txt_api_key.setPlaceholderText("AIzaSy...")
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        v_gemini.addWidget(self.txt_api_key)
        v_gemini.addStretch(1)
        self.stack_config.addWidget(page_gemini)
        
        # Gemma config
        page_gemma = QWidget()
        v_gemma = QVBoxLayout(page_gemma)
        v_gemma.setContentsMargins(0,10,0,0)
        
        v_gemma.addWidget(StrongBodyLabel("Gemma Model Path (.gguf):"))
        h_mod = QHBoxLayout()
        self.txt_model_path = LineEdit()
        self.txt_model_path.setText("D:/ollama/gemma-4-E2B-it-Q8_0.gguf")
        h_mod.addWidget(self.txt_model_path, 1)
        btn_mod = QPushButton("찾기")
        btn_mod.clicked.connect(self.browse_model)
        h_mod.addWidget(btn_mod)
        v_gemma.addLayout(h_mod)
        
        v_gemma.addWidget(StrongBodyLabel("Projector Path (mmproj.gguf):"))
        h_proj = QHBoxLayout()
        self.txt_proj_path = LineEdit()
        self.txt_proj_path.setText("D:/ollama/mmproj-F16.gguf")
        h_proj.addWidget(self.txt_proj_path, 1)
        btn_proj = QPushButton("찾기")
        btn_proj.clicked.connect(self.browse_proj)
        h_proj.addWidget(btn_proj)
        v_gemma.addLayout(h_proj)
        
        v_gemma.addWidget(StrongBodyLabel("Image Path (Optional):"))
        h_img = QHBoxLayout()
        self.txt_image_path = LineEdit()
        self.txt_image_path.setText("D:/ollama/20260722_143203.png")
        btn_img = QPushButton("찾기")
        btn_img.clicked.connect(self.browse_image)
        h_img.addWidget(self.txt_image_path, 1)
        h_img.addWidget(btn_img)
        v_gemma.addLayout(h_img)
        v_gemma.addStretch(1)
        
        self.stack_config.addWidget(page_gemma)
        v_config.addWidget(self.stack_config)
        splitter.addWidget(left_widget)
        
        # --- RIGHT SIDE: Chat Area ---
        right_widget = QWidget()
        v_chat = QVBoxLayout(right_widget)
        v_chat.setContentsMargins(0, 0, 0, 0)
        
        self.chat_area = TextEdit()
        self.chat_area.setReadOnly(True)
        self.chat_area.setStyleSheet("font-size: 14px; line-height: 1.5;")
        v_chat.addWidget(self.chat_area, 1)
        
        h_input = QHBoxLayout()
        self.txt_prompt = TextEdit()
        self.txt_prompt.setPlaceholderText("질문을 입력하세요... (Shift+Enter로 줄바꿈)")
        self.txt_prompt.setMaximumHeight(80)
        h_input.addWidget(self.txt_prompt, 1)
        
        self.btn_send = PushButton("전송")
        self.btn_send.setMinimumHeight(80)
        self.btn_send.clicked.connect(self.send_message)
        h_input.addWidget(self.btn_send)
        v_chat.addLayout(h_input)
        splitter.addWidget(right_widget)
        
        # Set Splitter ratio
        splitter.setSizes([300, 700])
        main_layout.addWidget(splitter, 1)


    def browse_model(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Model 파일 선택", "", "GGUF Files (*.gguf);;All Files (*)")
        if path:
            self.txt_model_path.setText(path)

    def browse_proj(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Projector 파일 선택", "", "GGUF Files (*.gguf);;All Files (*)")
        if path:
            self.txt_proj_path.setText(path)

    def browse_image(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.txt_image_path.setText(path)

    def on_engine_changed(self, text):
        if "Gemini" in text:
            self.stack_config.setCurrentIndex(0)
        else:
            self.stack_config.setCurrentIndex(1)

    def send_message(self):
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt: return
        
        self.chat_area.append(f"<b>[사용자]</b><br>{prompt}<br><br>")
        self.txt_prompt.clear()
        self.chat_area.append("<b>[AI 어시스턴트]</b><br>")
        
        engine = "gemini" if self.combo_engine.currentIndex() == 0 else "gemma"
        
        self.btn_send.setEnabled(False)
        self.llm_thread = AIGeneratorThread(
            engine=engine,
            prompt=prompt,
            api_key=self.txt_api_key.text(),
            model_path=self.txt_model_path.text(),
            proj_path=self.txt_proj_path.text(),
            image_path=self.txt_image_path.text() if engine == "gemma" else ""
        )
        self.llm_thread.chunk_received.connect(lambda t: self.chat_area.insertPlainText(t))
        self.llm_thread.error_occurred.connect(lambda e: ToastNotification.show_toast(self, "오류", e))
        def on_finished():
            self.chat_area.append("<br><br>")
            self.btn_send.setEnabled(True)
        self.llm_thread.finished.connect(on_finished)
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
        self.results_dict = {'original': None, 'climatology': None, 'anomaly': None, 'trend': None}
        
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
        
        self._nav.setCurrentRow(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="light_blue.xml")
    window = NMSCFluentApp()
    window.show()
    sys.exit(app.exec())
