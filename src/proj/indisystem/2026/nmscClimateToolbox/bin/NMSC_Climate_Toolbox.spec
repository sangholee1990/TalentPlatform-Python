# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import copy_metadata

datas = [('C:\\SYSTEMS\\PROG\\PYTHON\\TalentPlatform-Python\\src\\proj\\indisystem\\2026\\nmscClimateToolbox\\icon', 'icon'), ('C:\\SYSTEMS\\PROG\\PYTHON\\TalentPlatform-Python\\src\\proj\\indisystem\\2026\\nmscClimateToolbox\\nmsc_climate_toolbox.py', '.'), ('C:\\SYSTEMS\\PROG\\PYTHON\\TalentPlatform-Python\\src\\proj\\indisystem\\2026\\nmscClimateToolbox\\doc\\Validation_BUOY_SST_2D.nc', 'doc'), ('C:\\Users\\HP\\anaconda3\\envs\\py311\\Lib\\site-packages\\mpl_toolkits\\basemap_data', 'mpl_toolkits\\basemap_data')]
binaries = []
datas += collect_data_files('qt_material')
datas += collect_data_files('cartopy')
datas += collect_data_files('rasterio')
datas += collect_data_files('pyproj')
datas += collect_data_files('rioxarray')
datas += collect_data_files('xarray')
datas += collect_data_files('llama_cpp')
datas += copy_metadata('xarray')
datas += copy_metadata('rioxarray')
datas += copy_metadata('rasterio')
binaries += collect_dynamic_libs('llama_cpp')


a = Analysis(
    ['C:\\SYSTEMS\\PROG\\PYTHON\\TalentPlatform-Python\\src\\proj\\indisystem\\2026\\nmscClimateToolbox\\nmsc_climate_toolbox_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['numpy', 'pandas', 'scipy', 'scipy.stats', 'scipy.ndimage', 'matplotlib', 'seaborn', 'xarray', 'rioxarray', 'rasterio', 'rasterio.sample', 'rasterio.vrt', 'rasterio._features', 'netCDF4', 'h5netcdf', 'cftime', 'dask', 'cartopy', 'mpl_toolkits.basemap', 'qt_material', 'jinja2', 'PIL', 'google.generativeai', 'llama_cpp', 'llama_cpp.llama_chat_format', 'resources_rc', 'nmsc_climate_toolbox'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'PySide6', 'torch', 'torchvision', 'tensorflow', 'xgboost', 'transformers', 'sklearn', 'nltk', 'uvicorn', 'fastapi', 'llama_cpp.server', 'numba', 'llvmlite', 'sympy', 'sqlalchemy', 'statsmodels', 'openpyxl', 'pyarrow', 'distributed', 'tkinter', 'IPython', 'jupyter', 'notebook', 'pytest'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NMSC_Climate_Toolbox',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\SYSTEMS\\PROG\\PYTHON\\TalentPlatform-Python\\src\\proj\\indisystem\\2026\\nmscClimateToolbox\\icon\\icon.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NMSC_Climate_Toolbox',
)
