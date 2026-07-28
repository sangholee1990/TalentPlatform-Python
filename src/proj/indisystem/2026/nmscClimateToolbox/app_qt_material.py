import sys
import resources_rc
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
    f = QFont("Segoe UI", 15);
    f.setBold(True);
    lbl.setFont(f)
    return lbl


def SubtitleLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 11);
    f.setBold(True);
    lbl.setFont(f)
    return lbl


def StrongBodyLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 9);
    f.setBold(True);
    lbl.setFont(f)
    lbl.setWordWrap(True)
    return lbl


def BodyLabel(text, parent=None):
    lbl = QLabel(text, parent)
    f = QFont("Segoe UI", 9);
    lbl.setFont(f)
    lbl.setWordWrap(True)
    return lbl


# ─── Input shims ────────────────────────────────────────────────────────────
class LineEdit(QLineEdit):
    pass


class ComboBox(QComboBox):
    pass


class CheckableComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        from PyQt6.QtGui import QStandardItemModel, QStandardItem
        from PyQt6.QtCore import Qt
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QStandardItemModel(self))
        self.QStandardItem = QStandardItem
        self.Qt = Qt

    def handleItemPressed(self, index):
        item = self.model().itemFromIndex(index)
        if item.checkState() == self.Qt.CheckState.Checked:
            item.setCheckState(self.Qt.CheckState.Unchecked)
        else:
            item.setCheckState(self.Qt.CheckState.Checked)

    def addCheckableItem(self, text, data=None, checked=True):
        item = self.QStandardItem(text)
        item.setCheckState(self.Qt.CheckState.Checked if checked else self.Qt.CheckState.Unchecked)
        item.setFlags(self.Qt.ItemFlag.ItemIsUserCheckable | self.Qt.ItemFlag.ItemIsEnabled | self.Qt.ItemFlag.ItemIsSelectable)
        if data is not None:
            item.setData(data)
        self.model().appendRow(item)

    def getCheckedItems(self):
        checked = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item.checkState() == self.Qt.CheckState.Checked:
                checked.append(item.data() if item.data() is not None else item.text())
        return checked


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
    currentItemChanged = _sig(str)  # emits key string like original Fluent widget

    def __init__(self, parent=None):
        super().__init__(parent)
        _layout = QHBoxLayout(self)
        _layout.setContentsMargins(0, 0, 0, 0)
        _layout.setSpacing(0)
        self._bar = QTabBar(self)
        self._bar.setExpanding(False)
        _layout.addWidget(self._bar)
        _layout.addStretch(1)
        self._keys = {}  # idx -> key
        self._callbacks = {}  # idx -> callback
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
        # self._nav.setMaximumWidth(80)
        # self._nav.setMinimumWidth(80)
        self._nav.setMaximumWidth(100)
        self._nav.setMinimumWidth(100)

        self._nav.setStyleSheet("""
                    QListWidget::item {
                        color: #3c3c3c;
                    }
                    QListWidget::item:selected {
                        background-color: rgba(41, 121, 255, 0.15); 
                        color: #2979ff;
                        font-weight: bold;
                    }
                    QListWidget::item:hover:!selected {
                        background-color: #e6e6e6;
                    }
                """)

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
    LIGHT = "light";
    DARK = "dark";
    AUTO = "auto"


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
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(StrongBodyLabel(f"{label} 범위"))

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

        h_spin.addWidget(BodyLabel("최소"))
        h_spin.addWidget(self.spin_min)
        h_spin.addStretch(1)
        h_spin.addWidget(BodyLabel("최대"))
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
        layout.setContentsMargins(0, 0, 0, 0)

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
        self.date_start.setDisplayFormat("yyyy-MM")
        self.date_end.setDisplayFormat("yyyy-MM")

        h_date.addWidget(self.date_start)
        h_date.addStretch(1)
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
            
            default_start = end_date.addYears(-1)
            if default_start < start_date:
                default_start = start_date
            default_start_days = QDate(1970, 1, 1).daysTo(default_start)
            
            self.slider.set_range(default_start_days, self.max_days)
            self.date_start.setDate(default_start)
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

    def set_all_stations(self, state_bool):
        try:
            model = self.valid_station_combo.model()
            from PyQt6.QtCore import Qt
            new_state = Qt.CheckState.Checked if state_bool else Qt.CheckState.Unchecked
            for i in range(model.rowCount()):
                model.item(i).setCheckState(new_state)
        except Exception as e:
            print("Set all stations error:", e)

    def toggle_all_stations(self):
        try:
            model = self.valid_station_combo.model()
            from PyQt6.QtCore import Qt
            if model.rowCount() > 0:
                first_state = model.item(0).checkState()
                new_state = Qt.CheckState.Unchecked if first_state == Qt.CheckState.Checked else Qt.CheckState.Checked
                for i in range(model.rowCount()):
                    model.item(i).setCheckState(new_state)
        except Exception as e:
            print("Toggle stations error:", e)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # title = TitleLabel("데이터 준비 (Preprocess)")
        # main_layout.addWidget(title)

        self.segment = SegmentedWidget(self)
        main_layout.addWidget(self.segment, 0, Qt.AlignmentFlag.AlignLeft)

        self.stack = QStackedWidget()

        # --- Input Data Page ---
        page_input = QWidget()
        h_input = QHBoxLayout(page_input)
        # h_input.setContentsMargins(0, 10, 0, 0)
        h_input.setContentsMargins(0, 0, 0, 0)

        left_input = CardWidget()
        v_left_input = QVBoxLayout(left_input)
        # v_left_input.setContentsMargins(20, 20, 20, 20)
        # v_left_input.addWidget(SubtitleLabel("자료 경로"))
        # h_left_input = QHBoxLayout(left_input)
        # h_left_input.setContentsMargins(20, 20, 20, 20)
        # h_left_input.addWidget(SubtitleLabel("입력 자료"))

        # Row 1: File Selection
        h_row1 = QHBoxLayout()
        h_row1.addWidget(StrongBodyLabel("자료 설정"))
        self.file_combo = ComboBox()
        self.file_combo.currentTextChanged.connect(self.on_file_changed)
        btn_browse = PushButton("찾기")
        btn_browse.clicked.connect(self.browse_file)

        h_row1.addWidget(self.file_combo, 1)
        h_row1.addWidget(btn_browse)
        v_left_input.addLayout(h_row1)

        # Row 2: Variable Selection
        h_row2 = QHBoxLayout()
        h_row2.addWidget(StrongBodyLabel("세부 속성"))
        self.var_combo = ComboBox()
        self.var_combo.currentTextChanged.connect(lambda _: self.update_overview())

        h_row2.addWidget(self.var_combo, 1)
        v_left_input.addLayout(h_row2)

        # Row 3: Calendar Date Range
        h_row3 = QHBoxLayout()
        h_row3.addWidget(StrongBodyLabel("분석 기간"))
        self.date_start = QDateEdit()
        self.date_start.setCalendarPopup(True)
        self.date_end = QDateEdit()
        self.date_end.setCalendarPopup(True)
        self.date_start.setDisplayFormat("yyyy-MM-dd")
        self.date_end.setDisplayFormat("yyyy-MM-dd")

        h_row3.addWidget(self.date_start, 1)
        h_row3.addWidget(BodyLabel(" ~ "))
        h_row3.addWidget(self.date_end, 1)
        # h_row3.addStretch(1)
        v_left_input.addLayout(h_row3)

        # Row 4: Spatial Range Sliders
        # v_left_input.addWidget(StrongBodyLabel("공간 범위 지정"))
        self.lon_slider = FloatSlider("경도")
        v_left_input.addWidget(self.lon_slider)

        self.lat_slider = FloatSlider("위도", -90.0, 90.0)
        v_left_input.addWidget(self.lat_slider)

        v_left_input.addSpacing(10)
        self.btn_apply = PushButton("적용")
        self.btn_apply.setMinimumHeight(40)
        self.btn_apply.clicked.connect(self.on_apply_settings)
        v_left_input.addWidget(self.btn_apply)
        v_left_input.addStretch(1)

        right_input = CardWidget()
        v_right_input = QVBoxLayout(right_input)
        v_right_input.addWidget(SubtitleLabel("상세 정보"))
        self.overview_table = TextEdit()
        self.overview_table.setReadOnly(True)
        self.overview_table.setStyleSheet(
            "font-family: 'Pretendard GOV', 'Pretendard', 'Malgun Gothic', sans-serif; "
            "font-size: 13px; background-color: #ffffff; color: #333333; "
            "border: 1px solid #cccccc; border-radius: 4px; padding: 10px;"
        )
        v_right_input.addWidget(self.overview_table)

        splitter_input = QSplitter(Qt.Orientation.Horizontal)
        splitter_input.addWidget(left_input)
        splitter_input.addWidget(right_input)
        splitter_input.setSizes([1000, 1000])
        splitter_input.setStretchFactor(0, 1)
        splitter_input.setStretchFactor(1, 1)
        h_input.addWidget(splitter_input)
        self.stack.addWidget(page_input)

        # --- Validation Data Page ---
        page_valid = QWidget()
        h_valid = QHBoxLayout(page_valid)
        h_valid.setContentsMargins(0, 0, 0, 0)

        left_valid = CardWidget()
        v_left_valid = QVBoxLayout(left_valid)
        # v_left_valid.setContentsMargins(20, 20, 20, 20)
        # v_left_valid.addWidget(SubtitleLabel("자료 설정"))
        # v_left_valid.addSpacing(10)

        # Row 1
        h_row1_v = QHBoxLayout()
        h_row1_v.addWidget(StrongBodyLabel("자료 설정"))
        self.valid_file_combo = ComboBox()
        self.valid_file_combo.currentTextChanged.connect(self.on_valid_file_changed)
        btn_vbrowse = PushButton("찾기")
        btn_vbrowse.clicked.connect(self.browse_valid_file)

        h_row1_v.addWidget(self.valid_file_combo, 1)
        h_row1_v.addWidget(btn_vbrowse)
        v_left_valid.addLayout(h_row1_v)

        # Row 2
        h_row2_v = QHBoxLayout()
        h_row2_v.addWidget(StrongBodyLabel("세부 속성"))
        self.valid_var_combo = ComboBox()
        self.valid_var_combo.currentTextChanged.connect(lambda _: self.update_valid_overview())

        h_row2_v.addWidget(self.valid_var_combo, 1)
        v_left_valid.addLayout(h_row2_v)

        # Row 3
        h_row3_v = QHBoxLayout()
        h_row3_v.addWidget(StrongBodyLabel("분석 기간"))
        self.vdate_start = QDateEdit()
        self.vdate_start.setCalendarPopup(True)
        self.vdate_end = QDateEdit()
        self.vdate_end.setCalendarPopup(True)
        self.vdate_start.setDisplayFormat("yyyy-MM-dd")
        self.vdate_end.setDisplayFormat("yyyy-MM-dd")

        h_row3_v.addWidget(self.vdate_start, 1)
        h_row3_v.addWidget(BodyLabel(" ~ "))
        h_row3_v.addWidget(self.vdate_end, 1)
        v_left_valid.addLayout(h_row3_v)

        # Row 4
        # v_left_valid.addSpacing(10)
        self.vlon_slider = FloatSlider("경도")
        v_left_valid.addWidget(self.vlon_slider)

        self.vlat_slider = FloatSlider("위도", -90.0, 90.0)
        v_left_valid.addWidget(self.vlat_slider)

        v_left_valid.addSpacing(10)
        self.btn_valid_apply = PushButton("적용")
        self.btn_valid_apply.setMinimumHeight(40)
        self.btn_valid_apply.clicked.connect(self.on_apply_settings)
        v_left_valid.addWidget(self.btn_valid_apply)

        v_left_valid.addStretch(1)

        right_valid = CardWidget()
        v_right_valid = QVBoxLayout(right_valid)
        v_right_valid.addWidget(SubtitleLabel("상세 정보"))
        self.v_overview_table = TextEdit()
        self.v_overview_table.setReadOnly(True)
        # 공공 폰트(Pretendard GOV) 적용
        self.v_overview_table.setStyleSheet(
            "font-family: 'Pretendard GOV', 'Pretendard', 'Malgun Gothic', sans-serif; "
            "font-size: 13px; background-color: #ffffff; color: #333333; "
            "border: 1px solid #cccccc; border-radius: 4px; padding: 10px;"
        )
        v_right_valid.addWidget(self.v_overview_table)

        splitter_valid = QSplitter(Qt.Orientation.Horizontal)
        splitter_valid.addWidget(left_valid)
        splitter_valid.addWidget(right_valid)
        splitter_valid.setSizes([1000, 1000])
        splitter_valid.setStretchFactor(0, 1)
        splitter_valid.setStretchFactor(1, 1)
        h_valid.addWidget(splitter_valid)
        self.stack.addWidget(page_valid)

        main_layout.addWidget(self.stack)

        self.segment.addItem("input", "입력 자료", lambda: self.stack.setCurrentIndex(0))
        self.segment.addItem("valid", "검증 자료", lambda: self.stack.setCurrentIndex(1))
        self.segment.setCurrentIndex(0)

    def browse_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Data File", "",
                                                  "NetCDF/GeoTIFF Files (*.nc *.tif *.tiff);;All Files (*)")
        if filepath:
            if filepath not in self.files:
                self.files.append(filepath)
                self.file_combo.addItem(filepath)
            self.file_combo.setCurrentText(filepath)

    def browse_valid_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Validation File", "",
                                                  "NetCDF/GeoTIFF Files (*.nc *.tif *.tiff);;All Files (*)")
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
            if 'station_id' in vds.dims:
                vds = vds.rename({'station_id': 'station_code'})
            self.window().valid_ds = vds
            data_vars = [var for var in vds.data_vars if 'bnds' not in var and 'bounds' not in var]

            self.valid_var_combo.blockSignals(True)
            self.valid_var_combo.clear()
            self.valid_var_combo.addItems(data_vars)
            self.valid_var_combo.blockSignals(False)

            if 'station_code' in vds.dims:
                try:
                    target_combo = self.window().visualize_interface.valid_station_combo
                    target_combo.clear()
                    for st in vds['station_code'].values:
                        st_name = str(vds['station_name'].sel(station_code=st).values) if 'station_name' in vds else str(st)
                        if "전체" in str(st) or "전체" in st_name or "All" in str(st) or "All" in st_name:
                            continue
                        is_target = (str(st) == "22105")
                        target_combo.addCheckableItem(f"{st} ({st_name})", data=st, checked=is_target)
                except AttributeError:
                    pass

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
        # 현재 열려있는 탭 확인 (0: 입력 자료, 1: 검증 자료)
        current_tab = self.stack.currentIndex()

        # 1. 입력 자료 탭에서 적용을 눌렀을 때의 예외 처리
        if current_tab == 0:
            if self.window().ds is None:
                ToastNotification.show_toast(self, "경고", "입력 자료를 먼저 불러와 주세요.")
                return
            if not self.var_combo.currentText():
                ToastNotification.show_toast(self, "경고", "입력 자료의 세부 속성을 선택해 주세요.")
                return

            try:
                # 입력 자료의 공간 범위 적용
                self.window().bounds = {
                    'min_lon': self.lon_slider.get_min(),
                    'max_lon': self.lon_slider.get_max(),
                    'min_lat': self.lat_slider.get_min(),
                    'max_lat': self.lat_slider.get_max()
                }
            except Exception as e:
                pass

        # 2. 검증 자료 탭에서 적용을 눌렀을 때의 예외 처리
        elif current_tab == 1:
            if self.window().valid_ds is None:
                ToastNotification.show_toast(self, "경고", "검증 자료를 먼저 불러와 주세요.")
                return
            if not self.valid_var_combo.currentText():
                ToastNotification.show_toast(self, "경고", "검증 자료의 세부 속성을 선택해 주세요.")
                return

            try:
                # 검증 자료의 공간 범위 적용
                self.window().bounds = {
                    'min_lon': self.vlon_slider.get_min(),
                    'max_lon': self.vlon_slider.get_max(),
                    'min_lat': self.vlat_slider.get_min(),
                    'max_lat': self.vlat_slider.get_max()
                }
            except Exception as e:
                pass

        # 공통 적용 사항 저장
        self.window().selected_var = self.var_combo.currentText()
        self.window().selected_valid_var = self.valid_var_combo.currentText()

        # 적용 성공 안내 메시지
        ToastNotification.show_toast(self, "알림", "설정이 성공적으로 적용되었습니다.")


from PyQt6.QtWidgets import QListWidget, QListWidgetItem, QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QFrame


class CalculateInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("CalculateInterface")
        self.init_ui()

    def set_all_stations(self, state_bool):
        try:
            model = self.valid_station_combo.model()
            from PyQt6.QtCore import Qt
            new_state = Qt.CheckState.Checked if state_bool else Qt.CheckState.Unchecked
            for i in range(model.rowCount()):
                model.item(i).setCheckState(new_state)
        except Exception as e:
            print("Set all stations error:", e)

    def toggle_all_stations(self):
        try:
            model = self.valid_station_combo.model()
            from PyQt6.QtCore import Qt
            if model.rowCount() > 0:
                first_state = model.item(0).checkState()
                new_state = Qt.CheckState.Unchecked if first_state == Qt.CheckState.Checked else Qt.CheckState.Checked
                for i in range(model.rowCount()):
                    model.item(i).setCheckState(new_state)
        except Exception as e:
            print("Toggle stations error:", e)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # title = TitleLabel("산출")
        # main_layout.addWidget(title)

        # =========================================================
        # 1. 좌측 메뉴 대신 상단 탭(SegmentedWidget) 사용
        self.segment = SegmentedWidget(self)
        main_layout.addWidget(self.segment, 0, Qt.AlignmentFlag.AlignLeft)
        # main_layout.addSpacing(10)

        # 2. 하단 설정 영역 (기존 우측 패널)
        content_widget = CardWidget()
        v_content = QVBoxLayout(content_widget)
        self.stack = QStackedWidget()
        # =========================================================

        # Tool 1: Climatology & Anomaly
        page_1 = QWidget()
        v_page_1 = QVBoxLayout(page_1)
        v_page_1.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_page_1.addWidget(SubtitleLabel("평년값 및 편차 산출 옵션"))
        v_page_1.addSpacing(10)

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
        v_page_1.addLayout(h_cli)

        h_ano = QHBoxLayout()
        self.sw_ano = SwitchButton()
        self.sw_ano.setText("편차(Anomaly) 계산 활성화")
        h_ano.addWidget(self.sw_ano)
        v_page_1.addLayout(h_ano)

        h_month = QHBoxLayout()
        self.cb_target_month = ComboBox()
        self.cb_target_month.addItem("전체 월 (All Months)")
        for m in range(1, 13):
            self.cb_target_month.addItem(f"{m}월 (Month {m})")
        h_month.addWidget(StrongBodyLabel("대상 월 선택 (선택):"))
        h_month.addWidget(self.cb_target_month, 1)
        v_page_1.addLayout(h_month)

        v_page_1.addSpacing(10)
        btn_calc_1 = PushButton("계산 실행")
        btn_calc_1.clicked.connect(self.run_climatology)
        v_page_1.addWidget(btn_calc_1)

        self.stack.addWidget(page_1)

        # Tool 2: Spatial Aggregation
        page_2 = QWidget()
        v_page_2 = QVBoxLayout(page_2)
        v_page_2.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_page_2.addWidget(SubtitleLabel("공간 평균/합산 옵션"))
        v_page_2.addSpacing(10)

        h_sp = QHBoxLayout()
        self.cb_sp_method = ComboBox()
        self.cb_sp_method.addItems(["Mean (평균)", "Sum (합계)", "Max (최대)", "Min (최소)"])
        h_sp.addWidget(StrongBodyLabel("연산 방식:"))
        h_sp.addWidget(self.cb_sp_method)
        v_page_2.addLayout(h_sp)

        v_page_2.addSpacing(10)
        btn_calc_2 = PushButton("계산 실행")
        btn_calc_2.clicked.connect(self.run_spatial)
        v_page_2.addWidget(btn_calc_2)

        self.stack.addWidget(page_2)

        # Tool 3: Trend
        page_3 = QWidget()
        v_page_3 = QVBoxLayout(page_3)
        v_page_3.setAlignment(Qt.AlignmentFlag.AlignTop)
        v_page_3.addWidget(SubtitleLabel("시계열 추세 분석 옵션"))
        v_page_3.addSpacing(10)
        v_page_3.addWidget(BodyLabel("선형 회귀(Linear Regression) 기반 트렌드 분석을 수행합니다."))

        v_page_3.addSpacing(10)
        btn_calc_3 = PushButton("추세 계산 실행")
        btn_calc_3.clicked.connect(self.run_trend)
        v_page_3.addWidget(btn_calc_3)

        self.stack.addWidget(page_3)

        # =========================================================
        # 3. 레이아웃에 스택 추가 및 탭 아이템 등록
        v_content.addWidget(self.stack)
        main_layout.addWidget(content_widget, 1)

        # SegmentedWidget에 탭 아이템 추가 (클릭 시 스택 인덱스 변경)
        self.segment.addItem("clim", "기후 평년값 및 편차 산출", lambda: self.stack.setCurrentIndex(0))
        self.segment.addItem("spatial", "공간 평균/합산", lambda: self.stack.setCurrentIndex(1))
        self.segment.addItem("trend", "시계열 추세 분석", lambda: self.stack.setCurrentIndex(2))

        self.segment.setCurrentIndex(0)

    def on_toolbox_changed(self, index):
        self.stack.setCurrentIndex(index)

    def run_climatology(self):
        w = self.window()
        if w.ds is None:
            ToastNotification.show_toast(self, "오류", "먼저 데이터를 불러오세요.")
            return
        try:
            var_name = w.selected_var
            layer_to_set = None
            msg = ""
            if 'time' not in w.ds.dims:
                ToastNotification.show_toast(self, '오류', '시간(time) 차원이 없어 평년값/편차를 계산할 수 없습니다.')
                return

            # 사용자 요청: 각 월마다 평균하여 평년값을 구하고 개별 월평균값에서 빼야함
            ds_monthly = w.ds.resample(time='1MS').mean('time')

            if self.sw_cli.isChecked():
                ds_cli = ds_monthly.groupby('time.month').mean('time')
                w.calculated_ds = ds_cli
                if not hasattr(w, 'results_dict'): w.results_dict = {}
                w.results_dict['climatology'] = ds_cli
                layer_to_set = 'climatology'
                msg = f"{var_name} 평년값 계산이 완료되었습니다.\n[시각화 탭 > 평년 영상] 에서 미리보기 및 분석이 가능합니다.\n\n지금 바로 시각화 탭으로 이동하시겠습니까?"
            elif self.sw_ano.isChecked():
                ds_cli = ds_monthly.groupby('time.month').mean('time')
                ds_ano = ds_monthly.groupby('time.month') - ds_cli

                # Fix: groupby('time.month') scrambles the chronological order (groups by month).
                # We MUST sort it back by time so the timeline slider works correctly!
                if 'time' in ds_ano.coords:
                    ds_ano = ds_ano.sortby('time')

                w.calculated_ds = ds_ano
                if not hasattr(w, 'results_dict'): w.results_dict = {}
                w.results_dict['anomaly'] = ds_ano
                layer_to_set = 'anomaly'
                msg = f"{var_name} 편차(아노말리) 계산이 완료되었습니다.\n[시각화 탭 > 아노말리 영상] 에서 미리보기 및 분석이 가능합니다.\n\n지금 바로 시각화 탭으로 이동하시겠습니까?"

            if layer_to_set:
                from PyQt6.QtWidgets import QMessageBox
                reply = QMessageBox.question(self, '계산 완료 및 미리보기', msg,
                                             QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                             QMessageBox.StandardButton.Yes)

                if reply == QMessageBox.StandardButton.Yes:
                    if hasattr(w, 'nav_panel'):
                        w.nav_panel.setCurrentItem('viz')
                        w.stacked_widget.setCurrentWidget(w.viz_interface)

                    # Set the combobox layer to trigger the preview
                    cb = w.viz_interface.cb_layer
                    idx = cb.findData(layer_to_set)
                    if idx >= 0:
                        cb.setCurrentIndex(idx)

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
            elif 'month' in self.ds.dims:
                idx = self.time_idx % len(self.ds['month']) if len(self.ds['month']) > 0 else 0
                data = self.ds[self.var_name].isel(month=idx)
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

            # Determine origin
            if lat_vals[0] > lat_vals[-1]:
                data_2d = data_2d[::-1, :]
                img_origin = 'lower'
                ext_bottom, ext_top = min_lat, max_lat
            else:
                img_origin = 'lower'
                ext_bottom, ext_top = min_lat, max_lat

            ext_left, ext_right = min_lon, max_lon
            h, w = data_2d.shape

            try:
                import cartopy.crs as ccrs
                has_cartopy = True
            except ImportError:
                has_cartopy = False

            fig = plt.figure(figsize=(12, 12), dpi=300, frameon=False)

            if has_cartopy and not self.raw_mode:
                crs_3857 = ccrs.Mercator.GOOGLE
                crs_4326 = ccrs.PlateCarree()
                ax = fig.add_axes([0., 0., 1., 1.], projection=crs_3857)
                ax.set_axis_off()

                if self.bounds:
                    vmin, vmax = self.bounds
                else:
                    vmin, vmax = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))

                import matplotlib
                import matplotlib.colors as mcolors
                if self.cmap == 'SST_ANOM (custom)':
                    SST_ANOM_COLORS = [
                        '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                        '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                        '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                        '#A00000', '#800000', '#600000'
                    ]
                    cmap = mcolors.ListedColormap(SST_ANOM_COLORS)
                else:
                    cmap = self.cmap if isinstance(self.cmap, mcolors.Colormap) else plt.get_cmap(self.cmap)
                cmap.set_bad('none')

                ax.imshow(data_2d, origin=img_origin, extent=[ext_left, ext_right, ext_bottom, ext_top],
                          transform=crs_4326, cmap=cmap, vmin=vmin, vmax=vmax, regrid_shape=max(1500, max(h, w)),
                          interpolation='nearest')

                ax.set_extent([ext_left, ext_right, ext_bottom, ext_top], crs=crs_4326)

                x0, x1 = ax.get_xlim()
                y0, y1 = ax.get_ylim()
                ol_extent = [float(x0), float(y0), float(x1), float(y1)]
                projection_epsg = "EPSG:3857"
            else:
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)

                if self.bounds:
                    vmin, vmax = self.bounds
                else:
                    vmin, vmax = float(np.nanmin(data_2d)), float(np.nanmax(data_2d))

                import matplotlib
                import matplotlib.colors as mcolors
                if self.cmap == 'SST_ANOM (custom)':
                    SST_ANOM_COLORS = [
                        '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                        '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                        '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                        '#A00000', '#800000', '#600000'
                    ]
                    cmap = mcolors.ListedColormap(SST_ANOM_COLORS)
                else:
                    cmap = self.cmap if isinstance(self.cmap, mcolors.Colormap) else plt.get_cmap(self.cmap)
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

            js_data = "[" + ",".join("[" + ",".join(map(str, row)) + "]" for row in ds_data) + "]"

            import json
            payload = json.dumps({
                "image": "data:image/png;base64," + img_b64,
                "extent": ol_extent,
                "projection": projection_epsg,
                "data_grid": js_data,
                "data_shape": [h, w],
                "data_step": [step_y, step_x],
                "data_bounds": [ext_left, ext_right, ext_bottom, ext_top]
            })

            self.finished.emit(payload, "")

        except Exception as e:
            self.error_occurred.emit(str(e))


class VisualizeInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("VisualizeInterface")
        self.init_ui()

    def set_all_stations(self, state_bool):
        try:
            model = self.valid_station_combo.model()
            from PyQt6.QtCore import Qt
            new_state = Qt.CheckState.Checked if state_bool else Qt.CheckState.Unchecked
            for i in range(model.rowCount()):
                model.item(i).setCheckState(new_state)
        except Exception as e:
            print("Set all stations error:", e)

    def toggle_all_stations(self):
        try:
            model = self.valid_station_combo.model()
            from PyQt6.QtCore import Qt
            if model.rowCount() > 0:
                first_state = model.item(0).checkState()
                new_state = Qt.CheckState.Unchecked if first_state == Qt.CheckState.Checked else Qt.CheckState.Checked
                for i in range(model.rowCount()):
                    model.item(i).setCheckState(new_state)
        except Exception as e:
            print("Toggle stations error:", e)

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # title = TitleLabel("시각화")
        # main_layout.addWidget(title)

        self.pivot = Pivot()
        main_layout.addWidget(self.pivot, 0, Qt.AlignmentFlag.AlignLeft)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT SIDE (Options) ---
        left_widget = CardWidget()
        v_left = QVBoxLayout(left_widget)
        v_left.setContentsMargins(10, 10, 10, 10)

        # v_left.addWidget(SubtitleLabel("옵션 설정"))

        h_layer = QHBoxLayout()
        h_layer.addWidget(StrongBodyLabel("데이터"))
        self.cb_layer = ComboBox()
        self.cb_layer.addItem("원본 영상", "original")
        self.cb_layer.addItem("평년 영상", "climatology")
        self.cb_layer.addItem("아노말리 영상", "anomaly")
        # self.cb_layer.addItem("트렌드 영상", "trend")
        self.cb_layer.currentIndexChanged.connect(self.on_layer_changed)
        h_layer.addWidget(self.cb_layer, 1)
        v_left.addLayout(h_layer)

        self.w_cmap = QWidget()
        h_cmap = QHBoxLayout(self.w_cmap)
        h_cmap.setContentsMargins(0, 0, 0, 0)
        h_cmap.addWidget(StrongBodyLabel("색상바"))
        self.cb_cmap = ComboBox()
        self.cb_cmap.addItems(
            ['RdYlBu_r', 'viridis', 'plasma', 'inferno', 'magma', 'coolwarm', 'bwr', 'jet', 'SST_ANOM (custom)'])
        h_cmap.addWidget(self.cb_cmap, 1)
        v_left.addWidget(self.w_cmap)

        # Date Selection for Anomaly
        self.w_date = QWidget()
        h_date = QHBoxLayout(self.w_date)
        h_date.setContentsMargins(0, 0, 0, 0)
        h_date.addWidget(StrongBodyLabel("검색 기준 날짜"))
        self.cb_date = ComboBox()
        self.cb_date.addItem("전체 기간 평균")
        self.cb_date.currentIndexChanged.connect(self.on_date_changed)
        h_date.addWidget(self.cb_date, 1)
        v_left.addWidget(self.w_date)

        # Basemap Toggle
        self.w_base = QWidget()
        h_base = QHBoxLayout(self.w_base)
        h_base.setContentsMargins(0, 0, 0, 0)
        self.chk_basemap = QCheckBox("배경지도 표시")
        self.chk_basemap.setChecked(True)
        self.chk_basemap.stateChanged.connect(self.toggle_basemap)
        h_base.addWidget(self.chk_basemap)
        v_left.addWidget(self.w_base)

        self.w_raw = QWidget()
        h_raw = QHBoxLayout(self.w_raw)
        h_raw.setContentsMargins(0, 0, 0, 0)
        self.chk_raw_mode = QCheckBox("원본 데이터 유지")
        self.chk_raw_mode.setChecked(False)
        self.chk_raw_mode.stateChanged.connect(self.refresh_current_plot)
        h_raw.addWidget(self.chk_raw_mode)
        v_left.addWidget(self.w_raw)

        # DateSlider for Vis
        v_left.addSpacing(10)
        self.date_slider_vis = DateSlider("검색 기간")
        v_left.addWidget(self.date_slider_vis)

        # Station Multi-Select
        h_vst = QHBoxLayout()
        h_vst.addWidget(StrongBodyLabel("검증 지점"))
        self.valid_station_combo = CheckableComboBox()
        h_vst.addWidget(self.valid_station_combo, 1)
        v_vst = QVBoxLayout()
        v_vst.setContentsMargins(0, 0, 0, 0)
        v_vst.addLayout(h_vst)

        self.w_station = QWidget()
        self.w_station.setLayout(v_vst)
        v_left.addWidget(self.w_station)

        self.w_range = QWidget()
        h_range = QHBoxLayout(self.w_range)
        h_range.setContentsMargins(0, 0, 0, 0)
        h_range.addWidget(StrongBodyLabel("값범위"))
        self.txt_vmin = LineEdit()
        self.txt_vmin.setPlaceholderText("최소값")
        self.txt_vmax = LineEdit()
        self.txt_vmax.setPlaceholderText("최대값")
        h_range.addWidget(self.txt_vmin)
        h_range.addWidget(StrongBodyLabel("~"))
        h_range.addWidget(self.txt_vmax)
        v_left.addWidget(self.w_range)

        self.btn_download = PushButton("영상 다운로드")
        self.btn_download.clicked.connect(self.download_image)
        v_left.addWidget(self.btn_download)

        v_left.addStretch(1)

        from qfluentwidgets import PrimaryPushButton
        self.btn_ai = PrimaryPushButton("AI 분석 요청")
        self.btn_ai.setMinimumHeight(40)
        self.btn_ai.clicked.connect(self.request_ai_analysis)
        v_left.addWidget(self.btn_ai)

        v_left.addSpacing(5)

        self.btn_refresh = PushButton("적용")
        self.btn_refresh.setMinimumHeight(40)
        self.btn_refresh.clicked.connect(self.refresh_current_plot)
        v_left.addWidget(self.btn_refresh)

        splitter.addWidget(left_widget)

        # --- RIGHT SIDE (Plots) ---
        right_widget = QWidget()
        right_widget.setStyleSheet("background-color: white; border-radius: 8px;")
        v_right = QVBoxLayout(right_widget)
        v_right.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        v_right.addWidget(self.stack)

        page_map = QWidget()
        self.map_canvas_layout = QVBoxLayout(page_map)
        self.map_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.stack.addWidget(page_map)

        page_trend = QWidget()
        self.trend_canvas_layout = QVBoxLayout(page_trend)
        self.trend_canvas_layout.setContentsMargins(0, 0, 0, 0)

        hc_html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <script src="qrc:///js/highcharts.js"></script>
            <script src="qrc:///js/exporting.js"></script>
            <style>
                @font-face {
                    font-family: 'Pretendard';
                    src: url("qrc:///font/Pretendard-1.3.9/public/variable/PretendardGOVVariable.ttf") format('truetype');
                }
                body, html { 
                    margin: 0; padding: 0; height: 100%; overflow: hidden; 
                    background-color: transparent; font-family: 'Pretendard', sans-serif; 
                }
                #container { width: 100%; height: 100%; }
            </style>
        </head>
        <body>
            <div id="container"></div>
            <script>
                Highcharts.setOptions({
                    chart: { style: { fontFamily: 'Pretendard, sans-serif' } }
                });
                var chart = null;
                function updateChart(options) {
                    options.chart = options.chart || {};
                    options.chart.renderTo = 'container';
                    chart = new Highcharts.Chart(options);
                }
            </script>
        </body>
        </html>
        '''

        self.trend_view = QWebEngineView()
        self.trend_view.setHtml(hc_html_template)
        self.trend_canvas_layout.addWidget(self.trend_view)

        self.stack.addWidget(page_trend)

        page_comp = QWidget()
        self.valid_canvas_layout = QVBoxLayout(page_comp)
        self.valid_canvas_layout.setContentsMargins(0, 0, 0, 0)

        self.valid_view = QWebEngineView()
        self.valid_view.setHtml(hc_html_template)
        self.valid_canvas_layout.addWidget(self.valid_view)

        self.stack.addWidget(page_comp)

        page_image = QWidget()
        self.image_canvas_layout = QVBoxLayout(page_image)
        self.image_canvas_layout.setContentsMargins(0, 0, 0, 0)

        h_center = QHBoxLayout()

        from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene
        from PyQt6.QtGui import QPainter
        self.image_view = QGraphicsView()
        self.image_scene = QGraphicsScene()
        self.image_view.setScene(self.image_scene)
        self.image_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.image_view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.image_view.setStyleSheet("QGraphicsView { border: none; background-color: transparent; }")

        h_center.addWidget(self.image_view, 1)
        self.image_canvas_layout.addLayout(h_center)

        # Add Timeline Slider and Toolbar at bottom
        h_timeline = QHBoxLayout()
        h_timeline.setContentsMargins(5, 5, 5, 5)

        self.lbl_timeline = StrongBodyLabel("시간 선택 안됨")
        from qfluentwidgets import Slider
        self.slider_timeline = Slider(Qt.Orientation.Horizontal)
        self.slider_timeline.setMinimum(0)
        self.slider_timeline.setMaximum(0)
        self.slider_timeline.valueChanged.connect(self.on_image_timeline_changed)

        btn_fit = PushButton("화면 맞춤")
        btn_zoom_out = PushButton("- 축소")
        btn_zoom_in = PushButton("+ 확대")

        h_timeline.addWidget(self.lbl_timeline)
        h_timeline.addWidget(self.slider_timeline, 1)
        h_timeline.addWidget(btn_zoom_out)
        h_timeline.addWidget(btn_zoom_in)
        h_timeline.addWidget(btn_fit)

        self.image_canvas_layout.addLayout(h_timeline)

        self.stack.addWidget(page_image)

        btn_zoom_in.clicked.connect(self.zoom_in_image)
        btn_zoom_out.clicked.connect(self.zoom_out_image)
        btn_fit.clicked.connect(self.fit_image)

        self.pivot.addItem("image", "이미지 영상", lambda: self.on_tab_changed(3))
        # self.pivot.addItem("map", "지도 맵", lambda: self.on_tab_changed(0))
        self.pivot.addItem("trend", "검증 시계열", lambda: self.on_tab_changed(1))
        self.pivot.addItem("comp", "검증 산점도", lambda: self.on_tab_changed(2))

        self.pivot.setCurrentItem('image')
        self.on_tab_changed(3)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([2000, 8000])
        # splitter.setStretchFactor(0, 3)
        # splitter.setStretchFactor(1, 7)
        # splitter.setSizes([3000, 7000])
        # splitter.setStretchFactor(0, 4)
        # splitter.setStretchFactor(1, 6)
        # splitter.setSizes([4000, 6000])

        main_layout.addWidget(splitter, 1)

    def on_tab_changed(self, idx):
        self.stack.setCurrentIndex(idx)

        # UI Visibility per tab
        # 0: Map (Layer, Cmap, Date, Basemap, Raw Mode)
        # 1, 2, 3: Trend, Comp, Image (Layer, Range)
        is_map = (idx == 0)

        self.w_cmap.setVisible(is_map or idx == 3)
        if hasattr(self, 'w_date'):
            self.w_date.setVisible(is_map)
        self.w_base.setVisible(is_map)
        self.w_raw.setVisible(is_map)
        if hasattr(self, 'w_range'):
            self.w_range.setVisible(idx == 3)
        if hasattr(self, 'btn_download'):
            self.btn_download.setVisible(not is_map)
        if hasattr(self, 'w_station'):
            self.w_station.setVisible(idx in [1, 2])
        if hasattr(self, 'date_slider_vis'):
            self.date_slider_vis.setVisible(idx in [1, 2])

        current_data = self.cb_layer.currentData()
        self.cb_layer.blockSignals(True)
        self.cb_layer.clear()
        self.cb_layer.addItem("기본 영상", "original")
        
        if idx not in [1, 2]:
            self.cb_layer.addItem("평년 영상", "climatology")
            self.cb_layer.addItem("아노말리 영상", "anomaly")
            
        idx_to_set = self.cb_layer.findData(current_data)
        if idx_to_set >= 0:
            self.cb_layer.setCurrentIndex(idx_to_set)
        else:
            self.cb_layer.setCurrentIndex(0)
            
        self.cb_layer.blockSignals(False)

        self.refresh_current_plot()

    def showEvent(self, event):
        super().showEvent(event)
        self.populate_vis_ui()
        self.refresh_current_plot()

    def populate_vis_ui(self):
        w = self.window()
        # Populate stations
        if hasattr(w, 'valid_ds') and w.valid_ds is not None:
            vds = w.valid_ds
            if 'station_code' in vds.dims:
                # Only populate if empty to avoid losing user selection
                if self.valid_station_combo.model().rowCount() == 0:
                    from PyQt6.QtCore import Qt
                    for st in vds['station_code'].values:
                        st_name = str(
                            vds['station_name'].sel(station_code=st).values) if 'station_name' in vds else str(st)
                        
                        # Skip '전체' or 'All' stations
                        if "전체" in str(st) or "전체" in st_name or "All" in str(st) or "All" in st_name:
                            continue
                            
                        # Default select Donghae (22105)
                        is_target = (str(st) == "22105")
                        self.valid_station_combo.addCheckableItem(f"{st} ({st_name})", data=st, checked=is_target)

        # Populate dates
        ds = self.get_ds()
        if ds is not None and 'time' in ds.dims:
            import pandas as pd
            times = pd.to_datetime(ds['time'].values)
            if len(times) > 0:
                from PyQt6.QtCore import QDate
                start_q = QDate(times[0].year, times[0].month, times[0].day)
                end_q = QDate(times[-1].year, times[-1].month, times[-1].day)
                # set range only if not set yet to avoid override
                # We can just unconditionally set it since CustomRangeSlider handles bounds
                self.date_slider_vis.set_range(start_q, end_q)

    def update_dates(self):
        """Update cb_date based on current layer's dataset time dimension."""
        self.cb_date.blockSignals(True)
        self.cb_date.clear()

        ds = self.get_ds()
        if ds is not None and 'time' in ds.dims:
            import pandas as pd
            times = pd.to_datetime(ds['time'].values)
            for i, t in enumerate(times):
                self.cb_date.addItem(t.strftime('%Y-%m-%d %H:%M:%S'), userData=i)
        else:
            self.cb_date.addItem("단일 시점 (Time N/A)", userData=0)

        # Select current time index if valid
        w = self.window()
        if hasattr(w, 'selected_time_idx') and w.selected_time_idx < self.cb_date.count():
            self.cb_date.setCurrentIndex(w.selected_time_idx)
        else:
            if hasattr(w, 'selected_time_idx'):
                w.selected_time_idx = 0
            self.cb_date.setCurrentIndex(0)

        self.cb_date.blockSignals(False)

    def on_date_changed(self, index):
        if index < 0: return
        w = self.window()
        w.selected_time_idx = index
        self.refresh_current_plot()

    def on_layer_changed(self, index):
        layer = self.cb_layer.currentData()
        if layer == 'anomaly':
            self.cb_cmap.setCurrentText('SST_ANOM (custom)')
            if hasattr(self, 'txt_vmin'):
                self.txt_vmin.setText('-6.0')
                self.txt_vmax.setText('6.0')
        elif layer in ['original', 'climatology']:
            self.cb_cmap.setCurrentText('jet')
            if hasattr(self, 'txt_vmin'):
                self.txt_vmin.setText('0')
                self.txt_vmax.setText('36')

        self.update_dates()

        self.refresh_current_plot()

    def refresh_current_plot(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            self.plot_map()
        elif idx == 1:
            self.plot_trend()
        elif idx == 2:
            self.plot_valid()
        elif idx == 3:
            self.plot_static_image()

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
        vds = self.window().valid_ds
        var_name = self.window().selected_var
        vvar = self.window().selected_valid_var

        if ds is None or not var_name:
            return

        time_dim = 'time' if 'time' in ds.dims else None
        if not time_dim: return

        try:
            import json
            import pandas as pd
            import numpy as np
            from PyQt6.QtCore import QDate

            # Apply date filter
            try:
                start_date = self.date_slider_vis.date_start.date()
                end_date = self.date_slider_vis.date_end.date()
                st_ts = pd.Timestamp(start_date.year(), start_date.month(), start_date.day())
                en_ts = pd.Timestamp(end_date.year(), end_date.month(), end_date.day())

                if 'time' in ds.dims:
                    ds = ds.sel(time=slice(st_ts, en_ts))
                if vds is not None and 'time' in vds.dims:
                    vds = vds.sel(time=slice(st_ts, en_ts))
            except Exception as e:
                print("Date filter error in trend:", e)

            lat_dim = 'lat' if 'lat' in ds.dims else ('latitude' if 'latitude' in ds.dims else None)
            lon_dim = 'lon' if 'lon' in ds.dims else ('longitude' if 'longitude' in ds.dims else None)

            if vds is not None and vvar and lat_dim and lon_dim:
                v_lat_dim = 'lat' if 'lat' in vds.coords else ('latitude' if 'latitude' in vds.coords else None)
                v_lon_dim = 'lon' if 'lon' in vds.coords else ('longitude' if 'longitude' in vds.coords else None)

                if v_lat_dim and v_lon_dim and 'station_code' in vds.dims:
                    sat_time_series = {}
                    valid_time_series = {}

                    times = pd.to_datetime(ds[time_dim].values)
                    timestamps = (times.astype('int64') // 10 ** 6).tolist()

                    try:
                        selected_stations = self.valid_station_combo.getCheckedItems()
                    except:
                        selected_stations = vds['station_code'].values.tolist()

                    if not selected_stations:
                        self.trend_view.page().runJavaScript(
                            f"updateChart({json.dumps({'title': {'text': '선택된 관측소가 없습니다'}})});")
                        return

                    if len(times) == 0:
                        self.trend_view.page().runJavaScript(
                            f"updateChart({json.dumps({'title': {'text': '선택된 기간에 데이터가 없습니다'}})});")
                        return

                    for t_idx in range(len(times)):
                        ds_t = ds[var_name].isel(**{time_dim: t_idx})

                        vds_t = None
                        if 'time' in vds.dims:
                            try:
                                vds_t = vds[vvar].isel(**{time_dim: t_idx})
                            except:
                                vds_t = vds[vvar].isel(**{time_dim: -1})
                        else:
                            vds_t = vds[vvar]

                        for st in vds['station_code'].values:
                            if st not in selected_stations and str(st) not in selected_stations:
                                continue
                            try:
                                lat_v = float(vds[v_lat_dim].sel(station_code=st).values)
                                lon_v = float(vds[v_lon_dim].sel(station_code=st).values)

                                s_val = float(ds_t.sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)
                                v_val = float(vds_t.sel(station_code=st).values)

                                st_str = str(st)
                                if st_str not in sat_time_series:
                                    sat_time_series[st_str] = []
                                    valid_time_series[st_str] = []

                                if not np.isnan(s_val):
                                    sat_time_series[st_str].append([timestamps[t_idx], s_val])
                                if not np.isnan(v_val):
                                    valid_time_series[st_str].append([timestamps[t_idx], v_val])
                            except:
                                continue

                    def replace_nan(ts):
                        return [[x[0], None] if x[1] is None or np.isnan(x[1]) else x for x in ts]

                    def calc_trendline(ts):
                        from scipy.stats import linregress
                        clean_ts = [p for p in ts if p[1] is not None and not np.isnan(p[1])]
                        if len(clean_ts) > 1:
                            x_vals = [p[0] for p in clean_ts]
                            y_vals = [p[1] for p in clean_ts]
                            slope, intercept, r_value, _, _ = linregress(x_vals, y_vals)
                            min_x, max_x = min(x_vals), max(x_vals)
                            return [[min_x, min_x * slope + intercept], [max_x, max_x * slope + intercept]], r_value
                        return [], 0.0

                    series_data = []
                    for st_str in sat_time_series:
                        try:
                            st_name_val = str(vds['station_name'].sel(station_code=int(st_str) if st_str.isdigit() else st_str).values) if 'station_name' in vds else st_str
                        except:
                            st_name_val = st_str
                        st_label = f"{st_str} ({st_name_val})"
                        
                        s_ts = sat_time_series[st_str]
                        v_ts = valid_time_series[st_str]
                        
                        s_trend, s_r = calc_trendline(s_ts)
                        v_trend, v_r = calc_trendline(v_ts)
                        
                        series_data.append({'name': f'위성 - {st_label}', 'data': replace_nan(s_ts)})
                        series_data.append({'name': f'관측 - {st_label}', 'data': replace_nan(v_ts)})
                        if s_trend: series_data.append({'name': f'위성 추세 - {st_label} (R={s_r:.2f})', 'data': replace_nan(s_trend), 'marker': {'enabled': False}, 'dashStyle': 'Dash', 'linkedTo': ':previous'})
                        if v_trend: series_data.append({'name': f'관측 추세 - {st_label} (R={v_r:.2f})', 'data': replace_nan(v_trend), 'marker': {'enabled': False}, 'dashStyle': 'Dash', 'linkedTo': ':previous'})

                    options = {
                        'chart': {'type': 'line', 'zoomType': 'x'},
                        'title': {'text': f"{var_name} vs 관측소 시계열 검증 트렌드"},
                        'subtitle': {'text': "선택된 관측소별 비교"},
                        'xAxis': {'type': 'datetime', 'crosshair': True},
                        'yAxis': {'title': {'text': 'Value'}},
                        'tooltip': {'shared': True, 'valueDecimals': 2},
                        'series': series_data,
                        'credits': {'enabled': False}
                    }

                    js_code = f"updateChart({json.dumps(options)});"
                    self.trend_view.page().runJavaScript(js_code)
                    return

            if lat_dim and lon_dim:
                ts = ds[var_name].mean(dim=[lat_dim, lon_dim], skipna=True)
            else:
                ts = ds[var_name]

            df = ts.to_dataframe().reset_index()
            if len(df) == 0:
                self.trend_view.page().runJavaScript(
                    f"updateChart({json.dumps({'title': {'text': '선택된 기간에 데이터가 없습니다'}})});")
                return

            df['timestamp'] = pd.to_datetime(df[time_dim]).astype('int64') // 10 ** 6
            df[var_name] = df[var_name].replace({np.nan: None})
            data_list = df[['timestamp', var_name]].values.tolist()

            from scipy.stats import linregress
            single_trend = []
            single_r = 0.0
            clean_ts = [p for p in data_list if p[1] is not None and not np.isnan(p[1])]
            if len(clean_ts) > 1:
                x_vals = [p[0] for p in clean_ts]
                y_vals = [p[1] for p in clean_ts]
                slope, intercept, r_value, _, _ = linregress(x_vals, y_vals)
                min_x, max_x = min(x_vals), max(x_vals)
                single_trend = [[min_x, min_x * slope + intercept], [max_x, max_x * slope + intercept]]
                single_r = r_value

            series_data = [{'name': var_name, 'data': data_list, 'color': 'rgba(54, 162, 235, 1)'}]
            if single_trend:
                series_data.append({'name': f'추세선 (R={single_r:.2f})', 'type': 'line', 'data': single_trend,
                                    'color': 'rgba(54, 162, 235, 0.5)', 'dashStyle': 'Dash',
                                    'marker': {'enabled': False}})

            options = {
                'chart': {'type': 'line', 'zoomType': 'x'},
                'title': {'text': f"{var_name} 시계열 트렌드 (공간 평균)"},
                'xAxis': {'type': 'datetime', 'crosshair': True},
                'yAxis': {'title': {'text': f'{var_name}'}},
                'tooltip': {'shared': True, 'valueDecimals': 2},
                'series': series_data,
                'credits': {'enabled': False}
            }
            js_code = f"updateChart({json.dumps(options)});"
            self.trend_view.page().runJavaScript(js_code)

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
            import pandas as pd
            import numpy as np
            from scipy.stats import linregress

            # Apply date filter
            try:
                start_date = self.date_slider_vis.date_start.date()
                end_date = self.date_slider_vis.date_end.date()
                st_ts = pd.Timestamp(start_date.year(), start_date.month(), start_date.day())
                en_ts = pd.Timestamp(end_date.year(), end_date.month(), end_date.day())

                if 'time' in ds.dims:
                    ds = ds.sel(time=slice(st_ts, en_ts))
                if 'time' in vds.dims:
                    vds = vds.sel(time=slice(st_ts, en_ts))
            except Exception as e:
                print("Date filter error in valid:", e)

            lat_dim = 'lat' if 'lat' in ds.dims else ('latitude' if 'latitude' in ds.dims else None)
            lon_dim = 'lon' if 'lon' in ds.dims else ('longitude' if 'longitude' in ds.dims else None)
            v_lat_dim = 'lat' if 'lat' in vds.coords else ('latitude' if 'latitude' in vds.coords else None)
            v_lon_dim = 'lon' if 'lon' in vds.coords else ('longitude' if 'longitude' in vds.coords else None)

            if not lat_dim or not v_lat_dim:
                return

            scatter_data = []
            x_vals = []
            y_vals = []

            try:
                selected_stations = self.valid_station_combo.getCheckedItems()
            except:
                selected_stations = vds['station_code'].values.tolist()

            if not selected_stations:
                self.valid_view.page().runJavaScript(
                    f"updateChart({json.dumps({'title': {'text': '선택된 관측소가 없습니다'}})});")
                return

            if 'time' in ds.dims and len(ds['time']) == 0:
                self.valid_view.page().runJavaScript(
                    f"updateChart({json.dumps({'title': {'text': '선택된 기간에 데이터가 없습니다'}})});")
                return

            # When time is filtered, we shouldn't use time_idx if it's out of bounds.
            # Usually scatter plot aggregates over all times in the slice, or just the currently selected time?
            # Let's aggregate over ALL times in the filtered slice!
            times_len = len(ds['time']) if 'time' in ds.dims else 1

            for st in vds['station_code'].values:
                if st not in selected_stations and str(st) not in selected_stations:
                    continue
                try:
                    lat_v = float(vds[v_lat_dim].sel(station_code=st).values)
                    lon_v = float(vds[v_lon_dim].sel(station_code=st).values)
                    st_name = str(vds['station_name'].sel(station_code=st).values) if 'station_name' in vds else str(st)

                    for t_idx in range(times_len):
                        if 'time' in ds.dims:
                            d_val = float(ds[var_name].isel(time=t_idx).sel({lat_dim: lat_v, lon_dim: lon_v},
                                                                            method='nearest').values)
                        else:
                            d_val = float(ds[var_name].sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)

                        if 'time' in vds.dims:
                            try:
                                v_val = float(vds[vvar].isel(time=t_idx).sel(station_code=st).values)
                            except:
                                v_val = float(vds[vvar].isel(time=-1).sel(station_code=st).values)
                        else:
                            v_val = float(vds[vvar].sel(station_code=st).values)

                        if not np.isnan(d_val) and not np.isnan(v_val):
                            scatter_data.append({'x': d_val, 'y': v_val, 'name': f"{st} ({st_name})"})
                            x_vals.append(d_val)
                            y_vals.append(v_val)
                except:
                    continue

            if not scatter_data:
                self.valid_view.page().runJavaScript(f"updateChart({json.dumps({'title': {'text': '데이터가 없습니다'}})});")
                return

            slope, intercept, r_value, p_value, std_err = linregress(x_vals, y_vals)
            min_x, max_x = min(x_vals), max(x_vals)
            trend_line = [
                [min_x, min_x * slope + intercept],
                [max_x, max_x * slope + intercept]
            ]

            options = {
                'chart': {'type': 'scatter', 'zoomType': 'xy'},
                'title': {'text': f"{var_name} vs {vvar} (R={r_value:.2f})"},
                'xAxis': {'title': {'text': f"위성 ({var_name})"}},
                'yAxis': {'title': {'text': f"관측소 ({vvar})"}},
                'tooltip': {
                    'pointFormat': '{point.name}<br/>위성: {point.x:.2f}<br/>관측: {point.y:.2f}',
                    'valueDecimals': 2
                },
                'series': [
                    {'name': '관측소 비교', 'data': scatter_data, 'color': 'rgba(54, 162, 235, 0.5)'},
                    {'name': '추세선', 'type': 'line', 'data': trend_line, 'color': 'red', 'marker': {'enabled': False}}
                ],
                'credits': {'enabled': False}
            }

            js_code = f"updateChart({json.dumps(options)});"
            self.valid_view.page().runJavaScript(js_code)

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
            show_basemap=self.chk_basemap.isChecked(), raw_mode=self.chk_raw_mode.isChecked(),
            data_layer=self.cb_layer.currentData()
        )
        self.map_thread.finished.connect(self.on_map_finished)
        self.map_thread.error_occurred.connect(lambda e: ToastNotification.show_toast(self, "오류", e))
        self.map_thread.start()

    def toggle_basemap(self, state):
        if hasattr(self, 'map_view') and self.map_view is not None:
            visibility = 'true' if state == 2 else 'false'
            self.map_view.page().runJavaScript(
                f"if (typeof tileLayer !== 'undefined') tileLayer.setVisible({visibility});")
        else:
            self.refresh_current_plot()

    def download_image(self):
        if not hasattr(self, 'current_img_b64') or not self.current_img_b64:
            ToastNotification.show_toast(self, "알림", "저장할 이미지가 없습니다.")
            return
        from PyQt6.QtWidgets import QFileDialog
        import base64
        filepath, _ = QFileDialog.getSaveFileName(self, "영상 다운로드", "static_image.png", "PNG Files (*.png)")
        if filepath:
            try:
                with open(filepath, 'wb') as f:
                    f.write(base64.b64decode(self.current_img_b64))
                ToastNotification.show_toast(self, "성공", "이미지가 성공적으로 저장되었습니다.")
            except Exception as e:
                ToastNotification.show_toast(self, "오류", f"저장 실패: {e}")

    def request_ai_analysis(self):
        import tempfile
        import base64
        import os
        from PyQt6.QtGui import QPixmap

        var_name = getattr(self.window(), 'selected_var', None)
        if not var_name: var_name = "기상 자료"

        current_idx = self.stack.currentIndex()
        tmp_path = os.path.join(tempfile.gettempdir(), 'current_analysis.png')

        prompt = ""
        try:
            if current_idx == 0:
                if not hasattr(self, 'last_map_b64') or not self.last_map_b64:
                    from qfluentwidgets import InfoBar, InfoBarPosition
                    InfoBar.error(title="오류", content="분석할 지도가 아직 그려지지 않았습니다.", parent=self,
                                  position=InfoBarPosition.TOP)
                    return
                prompt = f"현재 표시된 기상 데이터('{var_name}') 공간 지도를 자세히 분석해 줘. 주요 기후 패턴, 특징, 그리고 이 지역의 날씨에 미칠 영향을 중심으로 설명해 줘."
                with open(tmp_path, 'wb') as f:
                    f.write(base64.b64decode(self.last_map_b64))

            elif current_idx == 1:
                pixmap = self.trend_view.grab()
                pixmap.save(tmp_path, "PNG")
                prompt = f"현재 표시된 '{var_name}' 시계열 트렌드 차트를 분석해 줘. 시간에 따른 값의 변화 추이와 주기성, 그리고 주요 변동폭을 중심으로 설명해 줘."

            elif current_idx == 2:
                pixmap = self.valid_view.grab()
                pixmap.save(tmp_path, "PNG")
                prompt = f"현재 표시된 '{var_name}' 위성 및 관측소 검증 산점도 차트를 분석해 줘. 두 데이터 간의 상관관계(R값 추이)와 추세선의 기울기를 중심으로 정확도를 평가해 줘."

            elif current_idx == 3:
                pixmap = self.image_view.grab()
                pixmap.save(tmp_path, "PNG")
                prompt = f"현재 표시된 '{var_name}' 정적 이미지를 분석해 줘. 전반적인 기상 패턴과 눈에 띄는 특이점을 설명해 줘."

        except Exception as e:
            from qfluentwidgets import InfoBar, InfoBarPosition
            InfoBar.error(title="오류", content=f"이미지 캡처 중 오류가 발생했습니다: {e}", parent=self, position=InfoBarPosition.TOP)
            return

        main_nav = self.window().nav_panel
        self.window().stacked_widget.setCurrentWidget(self.window().ai_interface)
        if hasattr(self.window().nav_panel, 'setCurrentItem'):
            self.window().nav_panel.setCurrentItem('ai')

        ai_interface = self.window().ai_interface
        ai_interface.txt_image_path.setText(tmp_path)
        ai_interface.txt_prompt.setPlainText(prompt)
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

            try:
                selected_stations = self.window().input_panel.valid_station_combo.getCheckedItems()
            except:
                selected_stations = vds['station_code'].values.tolist()

            if not selected_stations:
                self.trend_view.page().runJavaScript(
                    f"updateChart({json.dumps({'title': {'text': '선택된 관측소가 없습니다'}})});")
                return

            for st in vds['station_code'].values:
                if st not in selected_stations and str(st) not in selected_stations:
                    continue
                try:
                    lat_v = float(vds[v_lat_dim].sel(station_code=st).values)
                    lon_v = float(vds[v_lon_dim].sel(station_code=st).values)
                    st_name = str(vds['station_name'].sel(station_code=st).values)

                    if 'time' in ds.dims:
                        d_val = float(ds[var_name].isel(time=time_idx).sel({lat_dim: lat_v, lon_dim: lon_v},
                                                                           method='nearest').values)
                    else:
                        d_val = float(ds[var_name].sel({lat_dim: lat_v, lon_dim: lon_v}, method='nearest').values)

                    if 'time' in vds.dims:
                        v_val = float(vds[vvar].isel(time=time_idx).sel(station_code=st).values)
                    else:
                        v_val = float(vds[vvar].sel(station_code=st).values)

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

            options = {
                'chart': {'type': 'scatter', 'zoomType': 'xy'},
                'title': {'text': f"{var_name} vs {vvar} 검증 산점도"},
                'xAxis': {'title': {'text': f"위성 산출물 ({var_name})"}, 'crosshair': True},
                'yAxis': {'title': {'text': f"검증 자료 ({vvar})"}, 'crosshair': True},
                'tooltip': {
                    'useHTML': True,
                    'headerFormat': '<b>{point.key}</b><br>',
                    'pointFormat': '위성: {point.x}<br>부이: {point.y}'
                },
                'series': [{
                    'name': '관측소 데이터',
                    'data': scatter_data,
                    'color': 'rgba(223, 83, 83, .5)',
                    'marker': {'radius': 5}
                }, {
                    'type': 'line',
                    'name': 'Y = X',
                    'data': [[line_min, line_min], [line_max, line_max]],
                    'marker': {'enabled': False},
                    'states': {'hover': {'lineWidth': 0}},
                    'enableMouseTracking': False,
                    'color': 'black',
                    'dashStyle': 'Dash'
                }],
                'credits': {'enabled': False}
            }

            js_code = f"updateChart({json.dumps(options)});"
            self.valid_view.page().runJavaScript(js_code)
        except Exception as e:
            print("Valid plot error:", e)

    def on_image_timeline_changed(self, value):
        w = self.window()
        w.selected_time_idx = value

        layer = self.cb_layer.currentData()
        if layer == 'climatology':
            self.lbl_timeline.setText(f"시간 {value + 1}월")
        else:
            ds = getattr(w, 'processed_ds', None)
            if ds is None: ds = getattr(w, 'ds', None)

            if ds is not None and 'time' in ds.dims:
                try:
                    import pandas as pd
                    start_date = self.date_slider_vis.date_start.date()
                    end_date = self.date_slider_vis.date_end.date()
                    st_ts = pd.Timestamp(start_date.year(), start_date.month(), start_date.day())
                    en_ts = pd.Timestamp(end_date.year(), end_date.month(), end_date.day())
                    ds = ds.sel(time=slice(st_ts, en_ts))

                    if value < len(ds.time):
                        time_val = ds.time.values[value]
                        ts = pd.to_datetime(str(time_val))
                        self.lbl_timeline.setText(f"시간 {ts.strftime('%Y-%m')}")
                    else:
                        self.lbl_timeline.setText(f"시간 index {value}")
                except Exception as e:
                    self.lbl_timeline.setText(f"시간 index {value}")
            else:
                self.lbl_timeline.setText(f"시간 index {value}")

        if not hasattr(self, 'timeline_timer'):
            from PyQt6.QtCore import QTimer
            self.timeline_timer = QTimer()
            self.timeline_timer.setSingleShot(True)
            self.timeline_timer.timeout.connect(self.plot_static_image)

        # Debounce the rendering
        self.timeline_timer.start(500)

    def plot_static_image(self):
        w = self.window()
        ds = getattr(w, 'processed_ds', None)
        if ds is None: ds = getattr(w, 'ds', None)

        var_name = w.selected_var
        layer = self.cb_layer.currentData()

        if ds is None or not var_name:
            ToastNotification.show_toast(self, "오류", "데이터가 없습니다.")
            return

        import pandas as pd
        try:
            start_date = self.date_slider_vis.date_start.date()
            end_date = self.date_slider_vis.date_end.date()
            st_ts = pd.Timestamp(start_date.year(), start_date.month(), start_date.day())
            en_ts = pd.Timestamp(end_date.year(), end_date.month(), end_date.day())
            if 'time' in ds.dims:
                ds = ds.sel(time=slice(st_ts, en_ts))
        except Exception as e:
            print("Date filter error:", e)

        # Update timeline bounds silently
        if 'time' in ds.dims:
            t_len = len(ds.time)
            if t_len == 0:
                ToastNotification.show_toast(self, "오류", "선택한 기간 내에 데이터가 없습니다.")
                return

            self.slider_timeline.blockSignals(True)
            max_val = 11 if layer == 'climatology' else t_len - 1
            self.slider_timeline.setMaximum(max_val)

            # Clamp index if needed
            if getattr(w, 'selected_time_idx', 0) > max_val:
                w.selected_time_idx = max_val
            elif getattr(w, 'selected_time_idx', 0) < 0:
                w.selected_time_idx = 0

            self.slider_timeline.setValue(w.selected_time_idx)
            self.slider_timeline.blockSignals(False)

            if layer == 'climatology':
                self.lbl_timeline.setText(f"시간 {w.selected_time_idx + 1}월")
            else:
                try:
                    time_val = ds.time.values[w.selected_time_idx]
                    ts = pd.to_datetime(str(time_val))
                    self.lbl_timeline.setText(f"시간 {ts.strftime('%Y-%m')}")
                except:
                    self.lbl_timeline.setText(f"시간 index {w.selected_time_idx}")
        else:
            self.slider_timeline.blockSignals(True)
            self.slider_timeline.setMaximum(0)
            self.slider_timeline.blockSignals(False)
            self.lbl_timeline.setText("시간 차원 없음")

        time_idx = getattr(w, 'selected_time_idx', 0)

        self.image_scene.clear()
        # self.image_scene.addText("이미지 그리는 중... (최대 15초 소요)")
        ToastNotification.show_toast(self, "알림", "이미지 그리는 중... (최대 15초 소요)")

        try:
            vmin = float(self.txt_vmin.text()) if hasattr(self, 'txt_vmin') and self.txt_vmin.text() else None
        except ValueError:
            vmin = None
        try:
            vmax = float(self.txt_vmax.text()) if hasattr(self, 'txt_vmax') and self.txt_vmax.text() else None
        except ValueError:
            vmax = None

        self.static_thread = StaticImageThread(
            ds=ds,
            var_name=var_name,
            time_idx=time_idx,
            data_layer=layer,
            bounds=(vmin, vmax) if (vmin is not None and vmax is not None) else None,
            cmap_name=self.cb_cmap.currentText()
        )
        self.static_thread.finished.connect(self.on_static_image_finished)
        self.static_thread.error_occurred.connect(lambda e: ToastNotification.show_toast(self, "오류", e))
        self.static_thread.start()

    def zoom_in_image(self):
        self.image_view.scale(1.2, 1.2)

    def zoom_out_image(self):
        self.image_view.scale(0.8, 0.8)

    def fit_image(self):
        if hasattr(self, 'image_pixmap_item') and self.image_pixmap_item:
            self.image_view.fitInView(self.image_pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def on_static_image_finished(self, img_b64):
        self.current_img_b64 = img_b64
        import base64
        from PyQt6.QtGui import QPixmap
        pixmap = QPixmap()
        pixmap.loadFromData(base64.b64decode(img_b64))

        self.image_scene.clear()
        self.image_pixmap_item = self.image_scene.addPixmap(pixmap)

        # Delay fit_image slightly to ensure layout is updated
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self.fit_image)


class StaticImageThread(QThread):
    finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, ds, var_name, time_idx, data_layer, bounds=None, cmap_name='jet'):
        super().__init__()
        self.ds = ds
        self.var_name = var_name
        self.time_idx = time_idx
        self.data_layer = data_layer
        self.bounds = bounds
        self.cmap_name = cmap_name

    def run(self):
        try:
            import numpy as np
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            from mpl_toolkits.basemap import Basemap
            from scipy.ndimage import gaussian_filter, median_filter
            import matplotlib.patheffects as pe
            import io
            import base64

            lat_dim = 'lat' if 'lat' in self.ds.dims else ('latitude' if 'latitude' in self.ds.dims else None)
            lon_dim = 'lon' if 'lon' in self.ds.dims else ('longitude' if 'longitude' in self.ds.dims else None)

            if not lat_dim or not lon_dim:
                self.error_occurred.emit("위도/경도 차원 없음.")
                return

            if 'time' in self.ds.dims:
                if self.data_layer == 'climatology':
                    target_month = self.time_idx + 1
                    ds_month = self.ds.isel(time=(self.ds.time.dt.month == target_month))
                    data = ds_month[self.var_name].mean(dim='time')
                elif self.data_layer == 'anomaly':
                    t_slice = self.ds.isel(time=self.time_idx)
                    target_month = int(self.ds.time[self.time_idx].dt.month)
                    clim = self.ds.isel(time=(self.ds.time.dt.month == target_month))[self.var_name].mean(dim='time')
                    data = t_slice[self.var_name] - clim
                else:
                    data = self.ds[self.var_name].isel(time=self.time_idx)
            elif 'month' in self.ds.dims:
                idx = self.time_idx % len(self.ds['month']) if len(self.ds['month']) > 0 else 0
                data = self.ds[self.var_name].isel(month=idx)
            else:
                data = self.ds[self.var_name]

            lat = data[lat_dim].values
            lon = data[lon_dim].values

            data_2d = data.values
            if data_2d.ndim > 2: data_2d = data_2d.squeeze()
            while data_2d.ndim > 2: data_2d = data_2d[0]

            # Use same projection as reference scripts
            fig = plt.figure(figsize=(2400 / 300, 2000 / 300))
            ax = fig.add_subplot(111)

            m = Basemap(
                projection='lcc', resolution='i',
                lat_0=38, lon_0=126,
                llcrnrlat=11.308528, urcrnrlat=53.303712,
                llcrnrlon=101.395259, urcrnrlon=175.188166,
                lat_1=30, lat_2=60, ax=ax
            )

            # To handle origin difference (some datasets are reversed in lat)
            if lat[0] > lat[-1]:
                # Reverse lat and data to ensure monotonically increasing for contour/pcolormesh
                lat = lat[::-1]
                data_2d = data_2d[::-1, :]

            lon_grid, lat_grid = np.meshgrid(lon, lat)
            x, y = m(lon_grid, lat_grid)

            if self.data_layer == 'anomaly':
                # Anomaly styling
                SST_ANOM_COLORS = [
                    '#FF66FF', '#FF33CC', '#CC33CC', '#9933CC', '#6633CC', '#3333CC', '#0033CC',
                    '#0066CC', '#3399FF', '#66CCFF', '#99FFFF', '#CCFFFF', '#FFFFCC', '#FFFF99',
                    '#FFFF33', '#FFCC33', '#FF9933', '#FF6633', '#FF3333', '#FF0000', '#CC0000',
                    '#A00000', '#800000', '#600000'
                ]
                VMIN = self.bounds[0] if self.bounds else -6.0
                VMAX = self.bounds[1] if self.bounds else 6.0
                DLEV = (VMAX - VMIN) / 24.0 if self.bounds else 0.5
                LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)
                if self.cmap_name == 'SST_ANOM (custom)':
                    cmap = mcolors.ListedColormap(SST_ANOM_COLORS)
                    norm = mcolors.BoundaryNorm(LEVELS, cmap.N)
                    pcm = m.pcolormesh(x, y, data_2d, cmap=cmap, norm=norm, shading='auto')
                else:
                    pcm = m.pcolormesh(x, y, data_2d, cmap=self.cmap_name, shading='auto', vmin=VMIN, vmax=VMAX)

                # median filter for contours
                size_val_ano = 15
                smoothed_data_ano = median_filter(data_2d, size=size_val_ano)
                smoothed_ma = np.ma.masked_where(np.isnan(data_2d), smoothed_data_ano)

                line_levels = np.arange(VMIN, VMAX + 1e-6, 2.0)
                line_levels = line_levels[line_levels != 0.0]
                line_levels = line_levels.astype(int)

                cs = m.contour(x, y, smoothed_ma, levels=line_levels, colors='k', linewidths=1.0, alpha=0.8)
                cs0 = m.contour(x, y, smoothed_ma, levels=[0.0], colors='k', linewidths=0.5, alpha=0.5)

                candidates = []
                paths = []
                if hasattr(cs, 'collections'):
                    for collection in cs.collections:
                        paths.extend(collection.get_paths())
                else:
                    paths = cs.get_paths()
                for path in paths:
                    for poly in path.to_polygons():
                        line_length = len(poly)
                        if line_length > 30:
                            mid_idx = line_length // 2
                            candidates.append({
                                'length': line_length,
                                'coord': (poly[mid_idx][0], poly[mid_idx][1])
                            })
                candidates.sort(key=lambda x: x['length'], reverse=True)
                MAX_LABELS = 15
                manual_locs = [cand['coord'] for cand in candidates[:MAX_LABELS]]

                labels = plt.clabel(cs, inline=True, fontsize=8, fmt='%d', colors='black', manual=manual_locs)
                for label in labels:
                    label.set_rotation(0)
                    label.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])

                cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, extend='both', spacing='proportional')
                cbar.set_ticks(LEVELS)
                cbar.set_label('SST Anomaly (C)', fontsize=10, fontname='DejaVu Sans', fontweight='bold')
                plt.title(f'GK2A SST Anomaly ({self.var_name})', fontsize=12, fontweight='bold', fontname='DejaVu Sans')
            else:
                # Original/Climatology styling
                VMIN = self.bounds[0] if self.bounds else 0.0
                VMAX = self.bounds[1] if self.bounds else 36.0
                DLEV = (VMAX - VMIN) / 9.0 if self.bounds else 4.0
                LEVELS = np.arange(VMIN, VMAX + 1e-6, DLEV)

                pcm = m.pcolormesh(x, y, data_2d, cmap=self.cmap_name, shading='auto', vmin=VMIN, vmax=VMAX)

                # median filter for contours
                size_val = 15
                smoothed_data = median_filter(data_2d, size=size_val)
                smoothed_ma = np.ma.masked_where(np.isnan(data_2d), smoothed_data)

                c = m.contour(x, y, smoothed_ma, levels=LEVELS, colors='black', linewidths=1.0, alpha=0.8)
                
                candidates = []
                paths = []
                if hasattr(c, 'collections'):
                    for collection in c.collections:
                        paths.extend(collection.get_paths())
                else:
                    paths = c.get_paths()
                for path in paths:
                    for poly in path.to_polygons():
                        line_length = len(poly)
                        if line_length > 30:
                            mid_idx = line_length // 2
                            candidates.append({
                                'length': line_length,
                                'coord': (poly[mid_idx][0], poly[mid_idx][1])
                            })
                candidates.sort(key=lambda x: x['length'], reverse=True)
                MAX_LABELS = 15
                manual_locs = [cand['coord'] for cand in candidates[:MAX_LABELS]]

                labels = plt.clabel(c, inline=True, fontsize=8, fmt='%d', colors='black', manual=manual_locs)
                for label in labels:
                    label.set_rotation(0)
                    label.set_path_effects([pe.withStroke(linewidth=2.0, foreground='white')])

                cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, extend='both', spacing='proportional')
                cbar.set_label('Sea Surface Temperature (C)', fontsize=10, fontname='DejaVu Sans', fontweight='bold')
                plt.title(f'Monthly Mean GK2A Sea Surface Temperature ({self.var_name})', fontsize=12,
                          fontweight='bold', fontname='DejaVu Sans')

            m.drawcoastlines(color='k', linewidth=0.5)
            m.drawcountries(color='gray', linewidth=0.5)
            m.fillcontinents(color='lightgray', lake_color='white')
            m.drawparallels(np.arange(-10, 61, 10), labels=[1, 0, 0, 0], fontsize=10, fontname='DejaVu Sans', fmt='%d',
                            fontweight='bold')
            m.drawmeridians(np.arange(50, 181, 10), labels=[0, 0, 0, 1], fontsize=10, fontname='DejaVu Sans', fmt='%d',
                            fontweight='bold')

            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, transparent=False)
            plt.close(fig)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')

            self.finished.emit(img_b64)

        except Exception as e:
            self.error_occurred.emit(str(e))


class AIGeneratorThread(QThread):
    chunk_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, engine, prompt, api_key, model_path, proj_path, image_path, chat_history=None):
        super().__init__()
        self.engine = engine
        self.prompt = prompt
        self.api_key = api_key
        self.model_path = model_path
        self.proj_path = proj_path
        self.image_path = image_path
        self.chat_history = chat_history or []

    def run(self):
        try:
            import time
            if self.engine == 'gemini':
                if not self.api_key:
                    self.error_occurred.emit("Gemini API Key가 입력되지 않았습니다.")
                    return
                try:
                    import google.generativeai as genai
                except ImportError:
                    self.error_occurred.emit("google-generativeai 라이브러리가 설치되어 있지 않습니다.")
                    return

                genai.configure(api_key=self.api_key)

                # Format history for Gemini
                formatted_history = []
                for msg in self.chat_history:
                    role = "user" if msg['role'] == "user" else "model"
                    formatted_history.append({"role": role, "parts": [msg['text']]})

                model = genai.GenerativeModel(self.model_path if self.model_path else 'gemini-1.5-flash')

                contents = [self.prompt]
                if self.image_path:
                    import os
                    if os.path.exists(self.image_path):
                        try:
                            from PIL import Image
                            img = Image.open(self.image_path)
                            contents.insert(0, img)
                        except Exception as e:
                            self.error_occurred.emit(f"이미지 열람 오류: {e}")
                    else:
                        self.error_occurred.emit(f"첨부된 이미지를 찾을 수 없습니다: {self.image_path}")

                if formatted_history:
                    chat = model.start_chat(history=formatted_history)
                    response = chat.send_message(contents, stream=True)
                else:
                    response = model.generate_content(contents, stream=True)

                for chunk in response:
                    if chunk.text:
                        self.chunk_received.emit(chunk.text)

            elif self.engine == 'gemma':
                if not self.model_path:
                    self.error_occurred.emit("Gemma Model Path가 입력되지 않았습니다.")
                    return
                try:
                    from llama_cpp import Llama
                except ImportError:
                    self.error_occurred.emit("llama-cpp-python 라이브러리가 설치되어 있지 않습니다.")
                    return

                try:
                    chat_handler = None
                    if self.image_path and self.proj_path:
                        try:
                            from llama_cpp.llama_chat_format import Llava15ChatHandler
                            chat_handler = Llava15ChatHandler(clip_model_path=self.proj_path)
                        except ImportError:
                            pass

                    llm = Llama(
                        model_path=self.model_path,
                        chat_handler=chat_handler,
                        n_ctx=2048,
                        n_threads=4,
                        n_gpu_layers=0,
                        verbose=False
                    )

                    messages = []
                    for msg in self.chat_history[-5:]:
                        role = "user" if msg['role'] == "user" else "assistant"
                        messages.append({"role": role, "content": msg['text']})
                    
                    if self.image_path and chat_handler:
                        messages.append({
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"file://{self.image_path}"}},
                                {"type": "text", "text": self.prompt}
                            ]
                        })
                    else:
                        messages.append({"role": "user", "content": self.prompt})

                    response = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=-1,
                        temperature=0.5,
                        stream=True
                    )

                    for chunk in response:
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            if 'content' in delta:
                                self.chunk_received.emit(delta['content'])

                    for chunk in response:
                        text = chunk['choices'][0]['text']
                        self.chunk_received.emit(text)
                except Exception as model_err:
                    self.error_occurred.emit(f"로컬 모델 구동 실패: {model_err}")

            self.chunk_received.emit("\n")

        except Exception as e:
            self.error_occurred.emit(f"AI 분석 중 예상치 못한 오류 발생: {str(e)}")


class AIAssistantInterface(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("AIAssistantInterface")
        self.llm_thread = None
        self.chat_history = []

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        # title = TitleLabel("AI 연계")
        # main_layout.addWidget(title)

        self.segment = SegmentedWidget(self)
        self.segment.addItem("online", "온라인", lambda: self.stack.setCurrentIndex(0))
        self.segment.addItem("offline", "오프라인", lambda: self.stack.setCurrentIndex(1))
        main_layout.addWidget(self.segment, 0, Qt.AlignmentFlag.AlignLeft)
        # main_layout.addSpacing(10)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT SIDE: Config Area ---
        left_widget = CardWidget()
        v_config = QVBoxLayout(left_widget)
        # v_config.setContentsMargins(20, 20, 20, 20)
        # v_config.addWidget(SubtitleLabel("설정 정보 (Settings)"))

        self.stack = QStackedWidget()

        # Online config (Gemini)
        page_online = QWidget()
        v_online = QVBoxLayout(page_online)
        v_online.setContentsMargins(0, 10, 0, 0)

        v_online.addWidget(StrongBodyLabel("Gemini 버전"))
        self.combo_gemini_ver = ComboBox()
        self.combo_gemini_ver.addItems(["gemini-1.5-flash", "gemini-1.5-pro"])
        v_online.addWidget(self.combo_gemini_ver)

        v_online.addWidget(StrongBodyLabel("API키"))
        self.txt_api_key = LineEdit()
        self.txt_api_key.setPlaceholderText("AIzaSy...")
        self.txt_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        v_online.addWidget(self.txt_api_key)
        v_online.addStretch(1)
        self.stack.addWidget(page_online)

        # Offline config (Local LLM)
        page_offline = QWidget()
        v_offline = QVBoxLayout(page_offline)
        v_offline.setContentsMargins(0, 10, 0, 0)

        v_offline.addWidget(StrongBodyLabel("VLM 대규모언어모델"))
        h_mod = QHBoxLayout()
        self.txt_model_path = LineEdit()
        self.txt_model_path.setText("gemma-4-E2B-it-Q8_0.gguf")
        h_mod.addWidget(self.txt_model_path, 1)
        btn_mod = QPushButton("찾기")
        btn_mod.clicked.connect(self.browse_model)
        h_mod.addWidget(btn_mod)
        v_offline.addLayout(h_mod)

        v_offline.addWidget(StrongBodyLabel("VLM 시각언어모델"))
        h_proj = QHBoxLayout()
        self.txt_proj_path = LineEdit()
        self.txt_proj_path.setText("mmproj-F16.gguf")
        h_proj.addWidget(self.txt_proj_path, 1)
        btn_proj = QPushButton("찾기")
        btn_proj.clicked.connect(self.browse_proj)
        h_proj.addWidget(btn_proj)
        v_offline.addLayout(h_proj)

        v_offline.addWidget(StrongBodyLabel("첨부 이미지"))
        h_img = QHBoxLayout()
        self.txt_image_path = LineEdit()
        self.txt_image_path.setText("20260722_143203.png")
        btn_img = QPushButton("찾기")
        btn_img.clicked.connect(self.browse_image)
        h_img.addWidget(self.txt_image_path, 1)
        h_img.addWidget(btn_img)
        v_offline.addLayout(h_img)
        v_offline.addStretch(1)

        self.stack.addWidget(page_offline)
        v_config.addWidget(self.stack)

        self.segment.setCurrentItem("online")
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
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([2000, 8000])
        # splitter.setStretchFactor(0, 3)
        # splitter.setStretchFactor(1, 7)
        # splitter.setSizes([3000, 7000])
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
            self.stack.setCurrentIndex(0)
        else:
            self.stack.setCurrentIndex(1)

    def simple_markdown_to_html(self, text):
        import re
        # Bold
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        # Lists (simple)
        text = re.sub(r'^\s*\*\s+(.*)$', r'<li>\1</li>', text, flags=re.MULTILINE)
        # Line breaks
        text = text.replace('\n', '<br>')
        return text

    def send_message(self):
        prompt = self.txt_prompt.toPlainText().strip()
        if not prompt: return

        # History 갱신
        self.chat_history.append({'role': 'user', 'text': prompt})
        self.chat_history.append({'role': 'ai', 'text': ''})

        self.txt_prompt.clear()

        # 1틱 초기 렌더링을 위해 전체 채팅 UI 업데이트 강제 호출
        self.update_full_chat_ui()

        # QTextCursor로 정확한 스트리밍 시작점 캡처
        cursor = self.chat_area.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.ai_start_pos = cursor.position()

        engine = "gemini" if self.stack.currentIndex() == 0 else "gemma"

        self.btn_send.setEnabled(False)
        self.llm_thread = AIGeneratorThread(
            engine=engine,
            prompt=prompt,
            api_key=self.txt_api_key.text(),
            model_path=self.txt_model_path.text(),
            proj_path=self.txt_proj_path.text(),
            image_path=self.txt_image_path.text(),
            chat_history=self.chat_history[:-2]
        )
        self.llm_thread.chunk_received.connect(self.on_chunk)
        self.llm_thread.error_occurred.connect(self.on_error)
        self.llm_thread.finished.connect(self.on_finished)
        self.llm_thread.start()

    def on_chunk(self, text):
        from PyQt6.QtGui import QTextCursor
        # History 업데이트
        if self.chat_history and self.chat_history[-1]['role'] == 'ai':
            self.chat_history[-1]['text'] += text

        # 글자를 끝에 삽입 (드래그나 선택이 풀리지 않도록 insertText 사용)
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)

        # 자동 스크롤
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_error(self, error_msg):
        self.chat_area.append(f"<br><font color='red'>[Error] {error_msg}</font><br>")

    def update_full_chat_ui(self):
        """Rebuilds the entire chat HTML to ensure flawless styling."""
        html = "<div style='font-family: sans-serif; font-size: 14px; line-height: 1.5;'>"
        for msg in self.chat_history:
            if msg['role'] == 'user':
                text = msg['text'].replace('\n', '<br>')
                html += f"<div align='right' style='color: #0078D4; margin-bottom: 15px;'><b>[사용자]</b><br>{text}</div>"
            else:
                text = self.simple_markdown_to_html(msg['text'])
                html += f"<div align='left' style='color: #000000; margin-bottom: 15px;'><b>[AI 연계]</b><br>{text}</div>"
        html += "</div>"

        # Save scroll position state before resetting HTML
        scrollbar = self.chat_area.verticalScrollBar()
        is_at_bottom = (scrollbar.value() >= scrollbar.maximum() - 10)

        self.chat_area.setHtml(html)

        if is_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def on_finished(self):
        # 스트리밍 완료 시 전체 채팅 영역의 HTML을 새로고침하여 스타일 중첩 방지
        if self.chat_history and self.chat_history[-1]['role'] == 'ai':
            self.update_full_chat_ui()

            # 자동 스크롤
            scrollbar = self.chat_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        self.btn_send.setEnabled(True)
        self.txt_prompt.setEnabled(True)
        self.txt_prompt.setFocus()


class NMSCFluentApp(MSFluentWindow):
    def __init__(self):
        super().__init__()
        # self.setWindowTitle("NMSC Climate Toolbox")
        self.setWindowTitle("국가기상위성센터 분석툴박스")
        self.resize(1000, 700)
        try:
            import os
            from PyQt6.QtGui import QIcon
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon/태극-문양_단독.png")
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
            else:
                from qfluentwidgets import FluentIcon
                self.setWindowIcon(FluentIcon.GLOBE.icon())
        except Exception as e:
            print("Icon loading error:", e)

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

        # Try to automatically load BUOY validation data
        try:
            import os
            import xarray as xr
            val_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'doc', 'Validation_BUOY_SST_2D.nc')
            if os.path.exists(val_path):
                self.valid_ds = xr.open_dataset(val_path)
                if 'station_id' in self.valid_ds.dims:
                    self.valid_ds = self.valid_ds.rename({'station_id': 'station_code'})
                self.selected_valid_var = 'sst'
        except Exception as e:
            print(f"Failed to load default validation data: {e}")

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

        self.addSubInterface(self.ai_interface, FluentIcon.CHAT, "AI 연계", position=NavigationItemPosition.BOTTOM)

        self._nav.setCurrentRow(0)


if __name__ == "__main__":
    import sys
    import logging
    import traceback
    import os
    from datetime import datetime

    # # Create log directory if not exists
    # log_dir = "logs"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    #
    # log_filename = os.path.join(log_dir, f"error_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    # logging.basicConfig(
    #     filename=log_filename,
    #     level=logging.ERROR,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )


    def global_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        print("Uncaught exception:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback)


    sys.excepthook = global_exception_handler

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="light_blue.xml")
    window = NMSCFluentApp()
    window.show()
    sys.exit(app.exec())
