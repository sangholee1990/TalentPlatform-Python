#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_complete_two_cogs.py (OpenLayers WebGL 최적화 버전)
- nmsc_climate_toolbox 모듈을 활용하여 월합성장 및 월간 기후값(평년장) 산출
- Scale: x100 (Int16, 0.01°C 정밀도 유지, Nodata=-9999)
- Interleave: BAND (다중 밴드에서 단일 월 조회 시 HTTP Range Request 최소화)
- Compression: DEFLATE + Predictor=2 (geotiff.js 네이티브 고속 디코딩)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from nmsc_climate_toolbox import NMSCClimateToolbox

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def parse_yyyymm(val):
    """'201501', '2015-01', '2015.01' 등의 문자열을 (year, month)로 파싱"""
    s = str(val).strip().replace('-', '').replace('.', '').replace('/', '')
    if len(s) >= 6:
        return int(s[:4]), int(s[4:6])
    raise ValueError(f"Invalid date format: {val}. Expected YYYYMM.")


def quantize_to_int16(arr):
    """
    유효 수온(-2.0 ~ 40.0°C)만 100을 곱해 Int16(-200 ~ 4000)으로 변환하고,
    결측치 및 육지는 언더플로우 없이 -9999로 고정.
    """
    valid_mask = np.isfinite(arr) & (arr >= -2.0) & (arr <= 40.0)
    return np.where(valid_mask, np.round(arr * 100.0), -9999).astype(np.int16)


def get_web_cog_profile():
    """OpenLayers (geotiff.js) 최적화 COG 프로파일 반환"""
    cog_p = cog_profiles.get('deflate')
    cog_p.update({
        'blockxsize': 256,
        'blockysize': 256,
        'predictor': 2,
        'interleave': 'band'
    })
    return cog_p


def convert_4326_to_int16_cog(src_float_tif, dst_cog_path):
    """EPSG:4326 Float32 임시 파일을 Int16 경량 COG로 변환"""
    cur_dir = os.path.dirname(dst_cog_path)
    pid = os.getpid()
    tmp_int_tif = os.path.join(cur_dir, f"tmp_int4326_{pid}_" + os.path.basename(dst_cog_path))

    with rasterio.open(src_float_tif) as src:
        kwargs = src.meta.copy()
        kwargs.update({'dtype': 'int16', 'nodata': -9999})
        with rasterio.open(tmp_int_tif, 'w', **kwargs) as dst:
            dst.scales = [0.01] * src.count
            dst.offsets = [0.0] * src.count
            dst.update_tags(**src.tags())
            for b in range(1, src.count + 1):
                arr = src.read(b)
                dst.write(quantize_to_int16(arr), b)
                if src.descriptions and len(src.descriptions) >= b and src.descriptions[b - 1]:
                    dst.set_band_description(b, src.descriptions[b - 1])

    cog_translate(
        tmp_int_tif,
        dst_cog_path,
        get_web_cog_profile(),
        in_memory=True,
        quiet=True,
        overview_level=5,
        overview_resampling='bilinear',
        web_optimized=True
    )
    if os.path.exists(tmp_int_tif):
        os.remove(tmp_int_tif)


def reproject_to_3857_cog(src_tif, dst_cog_path, resolution=5000):
    """EPSG:4326 Float32 GeoTIFF를 EPSG:3857로 보간 후 Int16 경량 COG로 변환"""
    cur_dir = os.path.dirname(dst_cog_path)
    pid = os.getpid()
    tmp_tif = os.path.join(cur_dir, f"tmp_warp_{pid}_" + os.path.basename(dst_cog_path))

    print(f"[*] Reprojecting {os.path.basename(src_tif)} to EPSG:3857 (res={resolution}m, Int16 x100)...")
    t0 = time.time()
    with rasterio.open(src_tif) as src:
        dst_crs = 'EPSG:3857'
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': -9999,
            'dtype': 'int16'
        })

        with rasterio.open(tmp_tif, 'w', **kwargs) as dst:
            dst.scales = [0.01] * src.count
            dst.offsets = [0.0] * src.count
            dst.update_tags(**src.tags())
            for b in range(1, src.count + 1):
                dest_arr = np.full((height, width), -9999.0, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, b),
                    destination=dest_arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=-9999.0,
                    dst_nodata=-9999.0
                )

                dest_int = quantize_to_int16(dest_arr)
                dst.write(dest_int, b)

                if src.descriptions and len(src.descriptions) >= b and src.descriptions[b - 1]:
                    desc_ascii = src.descriptions[b - 1].encode('ascii', 'ignore').decode('ascii')
                    dst.set_band_description(b, ' '.join(desc_ascii.split()))

    print(f"    Warp complete in {time.time() - t0:.2f}s. Converting to Web-Optimized COG...")

    cog_translate(
        tmp_tif,
        dst_cog_path,
        get_web_cog_profile(),
        in_memory=True,
        quiet=True,
        overview_level=5,
        overview_resampling='bilinear',
        web_optimized=True
    )

    if os.path.exists(tmp_tif):
        os.remove(tmp_tif)

    sz_mb = os.path.getsize(dst_cog_path) / (1024 * 1024)
    print(f"[+] Created EPSG:3857 COG (Int16, BAND-interleaved): {os.path.basename(dst_cog_path)} ({sz_mb:.2f} MB)")


def build_complete_two_cogs(
    start_ym='201501',
    end_ym='202312',
    crs='EPSG:3857',
    nc_path=None,
    out_dir=None,
    var_name='SST_L3_monthly'
):
    """
    nmsc_climate_toolbox 모듈을 이용하여 2종의 COG를 생성합니다:
    1. 월합성장 COG (multi_merge_le3_cdr-sst01m_ea020lc_{YYYYMM}010000.tif)
    2. 월간 기후값(평년장) COG (multi_merge_le3_cdr-sst01m-clim_ea020lc_{YYYYMM}010000.tif)
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # 출력 경로 설정
    if not out_dir:
        sub_viewer = os.path.join(cur_dir, "cog_web_viewer")
        out_dir = sub_viewer if os.path.isdir(sub_viewer) else cur_dir
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # NetCDF 입력 경로 탐색
    if not nc_path:
        default_nc = os.path.join(cur_dir, "doc", "L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled.nc")
        if os.path.exists(default_nc):
            nc_path = default_nc
        else:
            nc_path = r"C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\doc\L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled.nc"

    if not os.path.exists(nc_path):
        print(f"[ERROR] NetCDF file not found: {nc_path}")
        sys.exit(1)

    print(f"[*] Opening NetCDF via NMSCClimateToolbox: {os.path.basename(nc_path)}")
    ds = NMSCClimateToolbox.open(nc_path)

    # GeoTIFF 상하 반전 방지: 위도가 남->북으로 되어 있는 경우 북->남으로 정렬
    if ds.lat[0] < ds.lat[-1]:
        ds = ds.sortby('lat', ascending=False)

    lat_min = float(ds.lat.min().values)
    lat_max = float(ds.lat.max().values)
    lon_min = float(ds.lon.min().values)
    lon_max = float(ds.lon.max().values)
    h, w = len(ds.lat), len(ds.lon)

    sy, sm = parse_yyyymm(start_ym)
    ey, em = parse_yyyymm(end_ym)
    start_str = f"{sy:04d}-{sm:02d}-01"
    end_str = f"{ey:04d}-{em:02d}-31"

    # 대상 기간 필터링
    ds_filtered = ds.sel(time=slice(start_str, end_str))

    # NMSCClimateToolbox 모듈을 이용한 월합성장(resampled) 및 월간 기후값(cli) 산출
    print(f"[*] Computing Monthly Composite & Climatology via NMSCClimateToolbox ({sy:04d}.{sm:02d} ~ {ey:04d}.{em:02d})...")
    ds_resampled, ds_cli, ds_ano = NMSCClimateToolbox.process_climate_data(
        ds_filtered[[var_name]],
        freq='1MS',
        op='평균',
        start_yr=str(sy),
        end_yr=str(ey)
    )

    # 1. 월간 기후값 (12-Month Climatology)
    raw_clim = ds_cli[var_name].values.astype(np.float32)
    clim_12 = np.where(
        np.isfinite(raw_clim) & (raw_clim >= -2.0) & (raw_clim <= 40.0),
        raw_clim, -9999.0
    ).astype(np.float32)

    descriptions_clim = [
        f"{m:02d}" for m in range(1, 13)
    ]

    # 2. 월합성장 (Monthly Composite)
    raw_monthly = ds_resampled[var_name].values.astype(np.float32)
    monthly_stack = np.where(
        np.isfinite(raw_monthly) & (raw_monthly >= -2.0) & (raw_monthly <= 40.0),
        raw_monthly, -9999.0
    ).astype(np.float32)

    total_monthly_bands = len(ds_resampled.time)
    final_dates = [pd.to_datetime(t).strftime('%Y-%m') for t in ds_resampled.time.values]
    descriptions_monthly = [
        d for d in final_dates
    ]

    last_ym = final_dates[-1].replace('-', '')
    date_stamp = f"{last_ym}010000"

    out_monthly_final = os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m_ea020lc_{date_stamp}.tif")
    out_clim_final = os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m-clim_ea020lc_{date_stamp}.tif")

    # 3. 임시 Float32 GeoTIFF 생성 (Warp 시 Bilinear 보간 품질 확보용)
    transform_4326 = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    meta_base = {
        'driver': 'GTiff', 'height': h, 'width': w,
        'crs': 'EPSG:4326', 'transform': transform_4326,
        'dtype': 'float32', 'nodata': -9999.0
    }

    pid = os.getpid()
    tmp_monthly_4326 = os.path.join(out_dir, f"tmp_m_4326_{date_stamp}_{pid}.tif")
    tmp_clim_4326 = os.path.join(out_dir, f"tmp_c_4326_{date_stamp}_{pid}.tif")

    meta_m = {**meta_base, 'count': total_monthly_bands}
    with rasterio.open(tmp_monthly_4326, 'w', **meta_m) as dst:
        dst.write(monthly_stack)
        for b_idx, desc in enumerate(descriptions_monthly, 1):
            dst.set_band_description(b_idx, desc)

    meta_c = {**meta_base, 'count': 12}
    with rasterio.open(tmp_clim_4326, 'w', **meta_c) as dst:
        dst.write(clim_12)
        for b_idx, desc in enumerate(descriptions_clim, 1):
            dst.set_band_description(b_idx, desc)

    # 4. 최종 Int16 COG 생성
    if crs in ('EPSG:4326', 'both'):
        out_m_4326 = out_monthly_final if crs == 'EPSG:4326' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m_ea020lc_{date_stamp}_4326.tif")
        out_c_4326 = out_clim_final if crs == 'EPSG:4326' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m-clim_ea020lc_{date_stamp}_4326.tif")
        convert_4326_to_int16_cog(tmp_monthly_4326, out_m_4326)
        convert_4326_to_int16_cog(tmp_clim_4326, out_c_4326)

    if crs in ('EPSG:3857', 'both'):
        out_m_3857 = out_monthly_final if crs == 'EPSG:3857' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m_ea020lc_{date_stamp}_3857.tif")
        out_c_3857 = out_clim_final if crs == 'EPSG:3857' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m-clim_ea020lc_{date_stamp}_3857.tif")
        reproject_to_3857_cog(tmp_monthly_4326, out_m_3857, resolution=5000)
        reproject_to_3857_cog(tmp_clim_4326, out_c_3857, resolution=5000)

    # 임시 파일 정리
    for f in (tmp_monthly_4326, tmp_clim_4326):
        if os.path.exists(f):
            os.remove(f)

    print("================================================================================")
    print("  [SUCCESS] 2-COG System Successfully Created via NMSCClimateToolbox!")
    print(f"  - Date Range:         {final_dates[0]} ~ {final_dates[-1]} ({total_monthly_bands} bands)")
    print(f"  - Target CRS:         {crs}")
    print(f"  - Extent:             Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")
    print(f"  1. Monthly Composite: {out_monthly_final}")
    print(f"  2. Climatology:       {out_clim_final}")
    print("================================================================================")


if __name__ == '__main__':
    # ArgumentParser 없이 기본 파라미터로 즉시 실행
    build_complete_two_cogs(
        start_ym='201501',
        end_ym='202312',
        crs='EPSG:3857'
    )
