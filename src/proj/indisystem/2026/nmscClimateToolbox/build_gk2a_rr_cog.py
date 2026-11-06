#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_gk2a_rr_cog.py (OpenLayers WebGL 최적화 버전)
- NMSC 위성자료 다운로드 URL에서 GK2A AMI LE2 (RR: 강수강도, ADPS: 에어로졸, FF: 산불) NetCDF 수집 및 개별 시각별 COG 생성
- nmsc_climate_toolbox 모듈(NMSCClimateToolbox.open)을 활용하여 NetCDF 로드
- Scale: x100 (Int16, 0.01 정밀도 유지, Nodata=-9999)
- Interleave: BAND
- Compression: DEFLATE + Predictor=2 (geotiff.js 네이티브 고속 디코딩)
"""

import os
import sys
import ssl
import time
import warnings
import urllib.request
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles

from nmsc_climate_toolbox import NMSCClimateToolbox

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def parse_datetime_str(val):
    """'202611060000', '2026.11.06 00:00', '2026-11-06 02:00' 등의 문자열을 Timestamp로 변환"""
    s = str(val).strip().replace('-', '').replace('.', '').replace('/', '').replace(':', '').replace(' ', '')
    if len(s) == 12:
        return pd.to_datetime(s, format='%Y%m%d%H%M')
    if len(s) == 8:
        return pd.to_datetime(s, format='%Y%m%d')
    return pd.to_datetime(val)


def quantize_to_int16(arr, min_val=0.0, max_val=100.0):
    """
    유효 범위(min_val ~ max_val)만 100을 곱해 Int16으로 변환하고,
    결측치 및 관측 영역 외 픽셀은 언더플로우 없이 -9999로 고정.
    """
    valid_mask = np.isfinite(arr) & (arr >= min_val) & (arr <= max_val)
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


def download_nmsc_nc(file_key, obs_time, save_dir, timeout=30):
    """
    NMSC 위성자료 다운로드 API로부터 지정된 관측시각(YYYYMMDDHHMM)의 NetCDF 파일을 다운로드.
    이미 유효한 로컬 파일이 있으면 재사용하고, 서버에 파일이 없으면 None을 반환.
    """
    os.makedirs(save_dir, exist_ok=True)
    safe_key = file_key.lower().replace(':', '_')
    local_path = os.path.join(save_dir, f"{safe_key}_{obs_time}.nc")

    if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
        with open(local_path, 'rb') as f:
            magic = f.read(4)
        if magic in (b'\x89HDF', b'CDF\x01', b'CDF\x02'):
            return local_path

    url = (
        "https://nmsc.kma.go.kr/homepage/html/satellite/viewer/selectImgDown.do"
        f"?fileKey={file_key}&observationTime={obs_time}&type=NC"
    )
    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            data = resp.read()
    except Exception as e:
        print(f"[WARN] {obs_time} 다운로드 실패 ({e})")
        return None

    if len(data) < 4 or data[:4] not in (b'\x89HDF', b'CDF\x01', b'CDF\x02'):
        print(f"[WARN] {obs_time}: 서버에 해당 시각의 NC 첨부파일이 존재하지 않아 건너뜁니다.")
        return None

    with open(local_path, 'wb') as f:
        f.write(data)

    return local_path


def extract_gk2a_georef(ds):
    """
    GK2A NetCDF의 gk2a_imager_projection 변수 속성으로부터 CRS 및 Affine Transform 추출
    """
    if 'gk2a_imager_projection' in ds.variables:
        attrs = ds['gk2a_imager_projection'].attrs
    else:
        attrs = ds.attrs

    lat_1 = float(attrs.get('standard_parallel1', 30.0))
    lat_2 = float(attrs.get('standard_parallel2', 60.0))
    lat_0 = float(attrs.get('origin_latitude', 38.0))
    lon_0 = float(attrs.get('central_meridian', 126.0))
    x_0 = float(attrs.get('false_easting', 0.0))
    y_0 = float(attrs.get('false_northing', 0.0))

    proj4_str = (
        f"+proj=lcc +lat_1={lat_1} +lat_2={lat_2} +lat_0={lat_0} +lon_0={lon_0} "
        f"+x_0={x_0} +y_0={y_0} +datum=WGS84 +units=m +no_defs"
    )
    src_crs = CRS.from_proj4(proj4_str)

    width = int(attrs.get('image_width', ds.sizes.get('dim_x', 3000)))
    height = int(attrs.get('image_height', ds.sizes.get('dim_y', 2600)))
    pixel_size = float(attrs.get('pixel_size', 2000.0))
    half_px = pixel_size / 2.0

    # GK2A upper_left / lower_right 좌표는 픽셀 중심(center) 기준이므로 외곽 경계는 반 픽셀 확장
    ul_x = float(attrs.get('upper_left_easting', -2999000.0))
    ul_y = float(attrs.get('upper_left_northing', 2599000.0))
    lr_x = float(attrs.get('lower_right_easting', 2999000.0))
    lr_y = float(attrs.get('lower_right_northing', -2599000.0))

    west = ul_x - half_px
    south = lr_y - half_px
    east = lr_x + half_px
    north = ul_y + half_px

    transform = from_bounds(west, south, east, north, width, height)
    bounds = (west, south, east, north)
    return src_crs, transform, width, height, bounds


def extract_product_array(ds, var_name):
    """
    산출물 종류(RR, ADPS, FF)에 맞춰 유효 물리량 배열(Float32) 및 유효 범위(min_val, max_val) 반환.
    - RR   : 0.0 ~ 100.0 (mm/h)
    - ADPS : 1.0 ~ 5.99 (1~2: ASH, 3.xx~4.xx: DUST, 5.xx: DAY HAZE / 0: 기타, 6: 청정 제외)
    - FF   : 1.0 = Forest Fire (FF==1), 1.5 = Industrial Heat Detection (DQF_FF==10), 그 외 -9999.0
    """
    v_upper = var_name.upper()
    if v_upper == 'ADPS':
        raw_arr = ds['ADPS'].values.astype(np.float32)
        min_val, max_val = 1.0, 5.99
        clean_arr = np.where(
            np.isfinite(raw_arr) & (raw_arr >= min_val) & (raw_arr <= max_val),
            raw_arr, -9999.0
        ).astype(np.float32)
        return clean_arr, min_val, max_val, Resampling.nearest, 'average'

    if v_upper == 'FF':
        ff_arr = ds['FF'].values.astype(np.float32)
        clean_arr = np.full(ff_arr.shape, -9999.0, dtype=np.float32)
        if 'DQF_FF' in ds.variables:
            dqf_arr = ds['DQF_FF'].values
            # Industrial Heat Detection (DQF_FF == 10) -> 1.5
            clean_arr = np.where(dqf_arr == 10, 1.5, clean_arr)
        # Forest Fire (FF == 1) -> 1.0
        clean_arr = np.where(ff_arr == 1, 1.0, clean_arr).astype(np.float32)
        return clean_arr, 0.5, 1.5, Resampling.nearest, 'average'

    # 기본: RR (강수강도, 0.0 ~ 100.0 mm/h)
    raw_arr = ds[var_name].values.astype(np.float32)
    min_val, max_val = 0.0, 100.0
    clean_arr = np.where(
        np.isfinite(raw_arr) & (raw_arr >= min_val) & (raw_arr <= max_val),
        raw_arr, -9999.0
    ).astype(np.float32)
    return clean_arr, min_val, max_val, Resampling.bilinear, 'bilinear'


def reproject_to_target_cog(
    src_tif,
    dst_cog_path,
    dst_crs='EPSG:3857',
    resolution=None,
    min_val=0.0,
    max_val=100.0,
    warp_resampling=Resampling.bilinear,
    overview_resampling='bilinear'
):
    """원본 투영(LCC 등) Float32 GeoTIFF를 대상 좌표계(EPSG:3857 또는 EPSG:4326)로 보간 후 Int16 경량 COG로 변환"""
    cur_dir = os.path.dirname(dst_cog_path)
    pid = os.getpid()
    tmp_tif = os.path.join(cur_dir, f"tmp_warp_{pid}_" + os.path.basename(dst_cog_path))

    t0 = time.time()
    with rasterio.open(src_tif) as src:
        calc_kwargs = {'resolution': resolution} if resolution else {}
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, **calc_kwargs
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
                    resampling=warp_resampling,
                    src_nodata=-9999.0,
                    dst_nodata=-9999.0
                )

                dest_int = quantize_to_int16(dest_arr, min_val=min_val, max_val=max_val)
                dst.write(dest_int, b)

                if src.descriptions and len(src.descriptions) >= b and src.descriptions[b - 1]:
                    desc_ascii = src.descriptions[b - 1].encode('ascii', 'ignore').decode('ascii')
                    dst.set_band_description(b, ' '.join(desc_ascii.split()))

    cog_translate(
        tmp_tif,
        dst_cog_path,
        get_web_cog_profile(),
        in_memory=True,
        quiet=True,
        overview_level=5,
        overview_resampling=overview_resampling,
        web_optimized=True
    )

    if os.path.exists(tmp_tif):
        os.remove(tmp_tif)

    sz_mb = os.path.getsize(dst_cog_path) / (1024 * 1024)
    print(f"[+] Created {dst_crs} COG: {os.path.basename(dst_cog_path)} ({sz_mb:.2f} MB, {time.time() - t0:.2f}s)")


def build_gk2a_cog_series(
    start_dt='2026.11.06 00:00',
    end_dt='2026.11.06 02:00',
    step_minutes=10,
    file_key='GK2A:AMI:LE2:RR:NC:EA:020:LC',
    var_name='RR',
    out_prefix='gk2a_ami_le2_rr_ea020lc',
    crs='EPSG:3857',
    resolution=2000,
    nc_dir=None,
    out_dir=None,
    overwrite=False
):
    """
    지정된 기간의 GK2A 위성 산출물(RR, ADPS, FF 등) NetCDF를 다운로드하고
    nmsc_climate_toolbox로 읽어 각 관측 시각마다 개별 WebGL 최적화 COG 파일을 생성합니다.
    """
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    if not out_dir:
        sub_viewer = os.path.join(cur_dir, "cog_web_viewer")
        out_dir = sub_viewer if os.path.isdir(sub_viewer) else cur_dir
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not nc_dir:
        nc_dir = os.path.join(cur_dir, "doc", f"gk2a_{var_name.lower()}_nc")
    nc_dir = os.path.abspath(nc_dir)
    os.makedirs(nc_dir, exist_ok=True)

    t_start = parse_datetime_str(start_dt)
    t_end = parse_datetime_str(end_dt)
    time_index = pd.date_range(start=t_start, end=t_end, freq=f"{int(step_minutes)}min")

    print("--------------------------------------------------------------------------------")
    print(f"[*] Product: {var_name} ({file_key})")
    print(f"[*] Target Time Range: {t_start.strftime('%Y-%m-%d %H:%M')} ~ {t_end.strftime('%Y-%m-%d %H:%M')} ({len(time_index)} steps)")

    created_files = []
    src_crs = None
    src_transform = None
    width = height = None
    src_bounds = None
    pid = os.getpid()

    for dt in time_index:
        obs_time = dt.strftime('%Y%m%d%H%M')
        desc_str = dt.strftime('%Y-%m-%d %H:%M')
        out_cog_final = os.path.join(out_dir, f"{out_prefix}_{obs_time}.tif")

        if not overwrite and os.path.exists(out_cog_final) and os.path.getsize(out_cog_final) > 1024:
            print(f"[=] Skip existing COG: {os.path.basename(out_cog_final)}")
            created_files.append(out_cog_final)
            continue

        nc_path = download_nmsc_nc(file_key=file_key, obs_time=obs_time, save_dir=nc_dir)
        if not nc_path:
            continue

        print(f"[*] Opening NetCDF via NMSCClimateToolbox: {os.path.basename(nc_path)} ({desc_str})")
        ds = NMSCClimateToolbox.open(nc_path)

        if src_crs is None:
            src_crs, src_transform, width, height, src_bounds = extract_gk2a_georef(ds)

        clean_arr, min_val, max_val, warp_resampling, ovr_resampling = extract_product_array(ds, var_name)
        ds.close()

        # 임시 원본 투영(LCC) 단일 밴드 Float32 GeoTIFF 생성
        tmp_src_tif = os.path.join(out_dir, f"tmp_{out_prefix}_lcc_{obs_time}_{pid}.tif")
        meta_src = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'crs': src_crs,
            'transform': src_transform,
            'dtype': 'float32',
            'nodata': -9999.0
        }

        with rasterio.open(tmp_src_tif, 'w', **meta_src) as dst:
            dst.write(clean_arr, 1)
            dst.set_band_description(1, desc_str)

        if crs in ('EPSG:4326', 'both'):
            out_4326 = out_cog_final if crs == 'EPSG:4326' else os.path.join(out_dir, f"{out_prefix}_{obs_time}_4326.tif")
            reproject_to_target_cog(
                tmp_src_tif, out_4326, dst_crs='EPSG:4326', resolution=None,
                min_val=min_val, max_val=max_val, warp_resampling=warp_resampling, overview_resampling=ovr_resampling
            )
            created_files.append(out_4326)

        if crs in ('EPSG:3857', 'both'):
            out_3857 = out_cog_final if crs == 'EPSG:3857' else os.path.join(out_dir, f"{out_prefix}_{obs_time}_3857.tif")
            reproject_to_target_cog(
                tmp_src_tif, out_3857, dst_crs='EPSG:3857', resolution=resolution,
                min_val=min_val, max_val=max_val, warp_resampling=warp_resampling, overview_resampling=ovr_resampling
            )
            created_files.append(out_3857)

        if os.path.exists(tmp_src_tif):
            os.remove(tmp_src_tif)

    if not created_files:
        print(f"[WARN] {var_name}: 생성된 COG 파일이 없습니다.")
        return []

    print("================================================================================")
    print(f"  [SUCCESS] GK2A {var_name} Individual COGs Ready! ({len(created_files)} files)")
    print(f"  - First File: {created_files[0]}")
    print(f"  - Last File:  {created_files[-1]}")
    print("================================================================================")
    return created_files


if __name__ == '__main__':
    # ArgumentParser 없이 3종 산출물(RR, ADPS, FF) 기본 파라미터로 즉시 실행
    # 1. 강수강도 (RR): 2026.11.06 00:00 ~ 02:00 (10분 간격)
    build_gk2a_cog_series(
        start_dt='2026.11.06 00:00',
        end_dt='2026.11.06 02:00',
        step_minutes=10,
        file_key='GK2A:AMI:LE2:RR:NC:EA:020:LC',
        var_name='RR',
        out_prefix='gk2a_ami_le2_rr_ea020lc',
        crs='EPSG:3857'
    )

    # 2. 에어로졸 (ADPS): 2026.11.06 00:00 ~ 02:00 (10분 간격)
    build_gk2a_cog_series(
        start_dt='2026.11.06 00:00',
        end_dt='2026.11.06 02:00',
        step_minutes=10,
        file_key='GK2A:AMI:LE2:ADPS:NC:EA:020:LC',
        var_name='ADPS',
        out_prefix='gk2a_ami_le2_adps_ea020lc',
        crs='EPSG:3857'
    )

    # 3. 산불 (FF): 2025.03.24 06:00 ~ 10:00 (10분 간격)
    build_gk2a_cog_series(
        start_dt='2025.03.24 06:00',
        end_dt='2025.03.24 10:00',
        step_minutes=10,
        file_key='GK2A:AMI:LE2:FF:NC:EA:020:LC',
        var_name='FF',
        out_prefix='gk2a_ami_le2_ff_ea020lc',
        crs='EPSG:3857'
    )
