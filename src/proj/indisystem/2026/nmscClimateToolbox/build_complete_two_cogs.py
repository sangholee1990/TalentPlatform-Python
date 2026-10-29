#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_complete_two_cogs.py
--------------------------
주어진 날짜 범위(start_date ~ end_date)에 따라:
1. 월합성장 COG (gk2a_ami_le3_ecv-sst01m_ea020lc_{YYYYMM}010000.tif)
2. 월평년장 COG (gk2a_ami_le3_ecv-sst01m-clim_ea020lc_{YYYYMM}010000.tif)

핵심 원칙:
- 모의 합성장(인위적 random/warming 노이즈)은 완전히 배제.
- 해당 날짜에 관측 데이터가 없을 경우:
  * 기본 동작: 제외(exclude)하여 순수 관측 날짜 밴드만 생산.
  * 옵션(--missing climatology): 결측/부재 월에 대해 모의 합성장이 아닌 순수 '해당 월의 평년값'으로 채움.
- 영역(Bounds):
  lat_min = 15.0, lat_max = 49.95, lon_min = 105.0, lon_max = 149.95
- 기본 출력 CRS: EPSG:3857 (웹 지도 정합용) 또는 EPSG:4326, both 지원.
"""

import os
import sys
import time
import argparse
import datetime
import numpy as np
import netCDF4 as nc
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)


def parse_yyyymm(val):
    """'201501', '2015-01', '2015.01' 등의 문자열을 (year, month)로 파싱"""
    s = str(val).strip().replace('-', '').replace('.', '').replace('/', '')
    if len(s) >= 6:
        return int(s[:4]), int(s[4:6])
    raise ValueError(f"Invalid date format: {val}. Expected YYYYMM.")


def generate_month_list(start_ym, end_ym):
    """start_ym 부터 end_ym 까지 연속된 (year, month, 'YYYYMM') 리스트 반환"""
    sy, sm = start_ym
    ey, em = end_ym
    cur = datetime.date(sy, sm, 1)
    end = datetime.date(ey, em, 1)

    result = []
    while cur <= end:
        result.append((cur.year, cur.month, f"{cur.year}{cur.month:02d}"))
        # 다음 달 1일로 이동
        if cur.month == 12:
            cur = datetime.date(cur.year + 1, 1, 1)
        else:
            cur = datetime.date(cur.year, cur.month + 1, 1)
    return result


def reproject_to_3857_cog(src_tif, dst_cog_path, resolution=5000):
    """EPSG:4326 GeoTIFF를 EPSG:3857로 Warp 후 COG로 변환"""
    cur_dir = os.path.dirname(dst_cog_path)
    pid = os.getpid()
    tmp_tif = os.path.join(cur_dir, f"tmp_warp_{pid}_" + os.path.basename(dst_cog_path))

    print(f"[*] Reprojecting {os.path.basename(src_tif)} to EPSG:3857 (res={resolution}m)...")
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
            'nodata': -9999.0,
            'dtype': 'float32'
        })

        with rasterio.open(tmp_tif, 'w', **kwargs) as dst:
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
                # 육지 및 결측치(-9999 미만, 40 초과)는 엄격히 -9999.0으로 마스킹
                dest_clean = np.where(
                    np.isfinite(dest_arr) & (dest_arr >= -2.0) & (dest_arr <= 40.0),
                    dest_arr,
                    -9999.0
                ).astype(np.float32)
                dst.write(dest_clean, b)

                if src.descriptions and len(src.descriptions) >= b and src.descriptions[b-1]:
                    desc_ascii = src.descriptions[b-1].encode('ascii', 'ignore').decode('ascii')
                    dst.set_band_description(b, ' '.join(desc_ascii.split()))

    print(f"    Warp complete in {time.time() - t0:.2f}s. Now converting to COG with 3 overview levels...")
    cog_p = cog_profiles.get('deflate')
    cog_p.update({
        'blockxsize': 256,
        'blockysize': 256,
        'predictor': 3
    })

    cog_translate(
        tmp_tif,
        dst_cog_path,
        cog_p,
        in_memory=True,
        quiet=True,
        overview_level=3,
        overview_resampling='average',
        web_optimized=True
    )

    if os.path.exists(tmp_tif):
        os.remove(tmp_tif)

    sz_mb = os.path.getsize(dst_cog_path) / (1024 * 1024)
    print(f"[+] Created EPSG:3857 COG (256x256, 3-level overviews): {os.path.basename(dst_cog_path)} ({sz_mb:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Build Monthly Composite and Climatology COGs without synthetic simulation."
    )
    parser.add_argument('--start', '-s', default='201501',
                        help="시작 연월 (YYYYMM, 기본값: 201501)")
    parser.add_argument('--end', '-e', default='202312',
                        help="종료 연월 (YYYYMM, 기본값: 202312)")
    parser.add_argument('--missing', choices=['exclude', 'climatology'], default='exclude',
                        help="관측 자료에 해당 날짜가 없을 경우 처리: 'exclude'(제외, 기본값) 또는 'climatology'(월 평년값으로 대체)")
    parser.add_argument('--crs', choices=['EPSG:3857', 'EPSG:4326', 'both'], default='EPSG:3857',
                        help="출력 COG 좌표계 (기본값: EPSG:3857)")
    parser.add_argument('--nc-path', default=None,
                        help="입력 NetCDF 파일 경로 (지정하지 않을 경우 doc 폴더 기본 파일 사용)")
    parser.add_argument('--out-dir', default=None,
                        help="출력 디렉터리 경로 (기본값: 스크립트 실행 디렉터리)")
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(cur_dir)

    # 출력 디렉터리 결정 (기본: cog_web_viewer가 있으면 우선 사용, 없으면 cur_dir)
    out_dir = args.out_dir
    if not out_dir:
        sub_viewer = os.path.join(cur_dir, "cog_web_viewer")
        if os.path.isdir(sub_viewer):
            out_dir = sub_viewer
        else:
            out_dir = cur_dir
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # NetCDF 파일 자동 검색 (스크립트 위치와 무관하게 doc 폴더 내 파일 자동 감지)
    nc_path = args.nc_path
    if not nc_path:
        filename = "L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled.nc"
        candidates = [
            os.path.join(cur_dir, "doc", filename),
            os.path.join(root_dir, "doc", filename),
            os.path.join(cur_dir, "..", "doc", filename),
            os.path.join(cur_dir, filename),
            r"C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\doc\L3_CDR_Monthly_201501_202312_Final_Combinded_gapfilled.nc"
        ]
        for c in candidates:
            if os.path.exists(c):
                nc_path = os.path.abspath(c)
                break
        if not nc_path:
            nc_path = candidates[0]

    if not os.path.exists(nc_path):
        print(f"[ERROR] NetCDF file not found: {nc_path}")
        sys.exit(1)

    start_ym = parse_yyyymm(args.start)
    end_ym = parse_yyyymm(args.end)
    requested_months = generate_month_list(start_ym, end_ym)

    print(f"[*] Reading NetCDF: {nc_path}")
    ds = nc.Dataset(nc_path, 'r')

    # 위경도 범위 확인
    # 글로벌 속성 또는 기본값
    lat_min = float(getattr(ds, 'geospatial_lat_min', 15.0))
    lat_max = float(getattr(ds, 'geospatial_lat_max', 49.95))
    lon_min = float(getattr(ds, 'geospatial_lon_min', 105.0))
    lon_max = float(getattr(ds, 'geospatial_lon_max', 149.95))

    lats = ds.variables['lat'][:]
    lons = ds.variables['lon'][:]
    time_var = ds.variables['time']
    time_vals = time_var[:]
    sst_all = ds.variables['SST_L3_monthly'][:] # (T, 700, 900)
    ds.close()

    # NetCDF 시간 좌표 변환 (days since 2015-01-01 00:00:00)
    base_date = datetime.datetime(2015, 1, 1)
    obs_map = {} # {'YYYYMM': array_index}
    for idx, d_val in enumerate(time_vals):
        dt = base_date + datetime.timedelta(days=int(d_val))
        obs_map[f"{dt.year}{dt.month:02d}"] = idx

    print(f"[*] Available observation dates in NetCDF: {len(obs_map)} months ({min(obs_map.keys())} ~ {max(obs_map.keys())})")
    print(f"[*] Requested date range: {args.start} ~ {args.end} (Total {len(requested_months)} months)")
    print(f"[*] Extent: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")

    # GeoTIFF raster orientation: North-up (Row 0 = North, Row H-1 = South)
    if lats[0] < lats[-1]:
        sst_flip = np.flip(sst_all, axis=1)
    else:
        sst_flip = sst_all

    h, w = sst_flip.shape[1], sst_flip.shape[2]

    # 물리적 유효 범위 마스킹 (-2.0 ~ 40.0 °C)
    if np.ma.is_masked(sst_flip):
        sst_flip = sst_flip.filled(np.nan)
    sst_valid = np.where((sst_flip >= -2.0) & (sst_flip <= 40.0), sst_flip, np.nan)

    # ==============================================================================
    # 1. 12-Month Climatology (월평년장, 12개 밴드) 계산
    # ==============================================================================
    # 실제 관측 기간의 데이터만 기반으로 각 월(1월~12월) 평균 평년값을 산출
    print("[*] Computing 12-Month Climatology from pure observation data...")
    clim_12 = np.full((12, h, w), -9999.0, dtype=np.float32)
    descriptions_clim = []
    month_names = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

    for m in range(12):
        # 관측 데이터에서 월(m+1)에 해당하는 모든 슬라이스 수집
        m_indices = [idx for ym, idx in obs_map.items() if int(ym[4:6]) == (m + 1)]
        if m_indices:
            m_stack = sst_valid[m_indices]
            with np.errstate(divide='ignore', invalid='ignore'):
                clim_mean = np.nanmean(m_stack, axis=0)
            clim_clean = np.where(
                np.isfinite(clim_mean) & (clim_mean >= -2.0) & (clim_mean <= 40.0),
                clim_mean,
                -9999.0
            ).astype(np.float32)
        else:
            clim_clean = np.full((h, w), -9999.0, dtype=np.float32)

        clim_12[m] = clim_clean
        descriptions_clim.append(f"Climatology {month_names[m]} Mean SST (2015-2023)")

        valid_px = clim_clean[clim_clean != -9999.0]
        if len(valid_px) > 0:
            print(f"    - {month_names[m]}월 평년값: 유효격자 {len(valid_px):,}, "
                  f"최저={valid_px.min():.2f}°C, 평균={valid_px.mean():.2f}°C, 최고={valid_px.max():.2f}°C")

    # ==============================================================================
    # 2. 월합성장 (Monthly Composite) 밴드 구성 (모의 합성장 완전 배제)
    # ==============================================================================
    final_monthly_list = []
    descriptions_monthly = []
    final_dates = []

    for y, m, ym in requested_months:
        if ym in obs_map:
            # 1) 관측 자료가 존재하는 경우 -> 순수 관측 슬라이스 사용
            obs_idx = obs_map[ym]
            raw_slice = sst_valid[obs_idx]
            clean_slice = np.where(
                np.isfinite(raw_slice) & (raw_slice >= -2.0) & (raw_slice <= 40.0),
                raw_slice,
                -9999.0
            ).astype(np.float32)
            final_monthly_list.append(clean_slice)
            descriptions_monthly.append(f"SST Monthly Composite {y}.{m:02d}")
            final_dates.append(ym)
        else:
            # 2) 관측 자료가 없는 경우 -> 모의 합성장은 절대 생성하지 않음!
            if args.missing == 'exclude':
                # 사용자의 지시: 해당 날짜가 없을 경우 제외!
                print(f"    [SKIP] Date {ym} not in observation data -> Excluded.")
                continue
            elif args.missing == 'climatology':
                # 사용자의 지시: 월마다 평년값으로 채움 (임의 노이즈/모의 합성장 배제)
                print(f"    [CLIM] Date {ym} not in observation data -> Using Climatology Month {m:02d}.")
                clim_slice = clim_12[m - 1].copy()
                final_monthly_list.append(clim_slice)
                descriptions_monthly.append(f"SST Monthly Composite {y}.{m:02d} (Climatology Fill)")
                final_dates.append(ym)

    if not final_monthly_list:
        print("[ERROR] No valid monthly bands to write! Check your date range.")
        sys.exit(1)

    monthly_stack = np.stack(final_monthly_list, axis=0)
    total_monthly_bands = len(final_monthly_list)
    last_ym = final_dates[-1]

    print(f"[+] Assembled Monthly Composite: {total_monthly_bands} bands ({final_dates[0]} ~ {last_ym})")

    # ==============================================================================
    # 3. 파일명 결정 (요청된 명명 규칙)
    #    gk2a_ami_le3_ecv-sst01m_ea020lc_{YYYYMM}010000.tif
    #    gk2a_ami_le3_ecv-sst01m-clim_ea020lc_{YYYYMM}010000.tif
    # ==============================================================================
    date_stamp = f"{last_ym}010000"
    name_monthly = f"multi_merge_le3_cdr-sst01m_ea020lc_{date_stamp}.tif"
    name_clim = f"multi_merge_le3_cdr-sst01m-clim_ea020lc_{date_stamp}.tif"

    out_monthly_final = os.path.join(out_dir, name_monthly)
    out_clim_final = os.path.join(out_dir, name_clim)

    # GeoTIFF 기본 메타데이터 (EPSG:4326)
    transform_4326 = from_bounds(lon_min, lat_min, lon_max, lat_max, w, h)
    meta_base = {
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'crs': 'EPSG:4326',
        'transform': transform_4326,
        'dtype': 'float32',
        'nodata': -9999.0
    }

    cog_profile = cog_profiles.get('deflate')
    cog_profile.update({
        'blockxsize': 256,
        'blockysize': 256,
        'predictor': 3
    })

    # 임시 TIFF 파일 경로 (고유 PID 부여로 동시 실행 또는 파일 락 충돌 방지)
    pid = os.getpid()
    tmp_monthly_4326 = os.path.join(out_dir, f"tmp_m_4326_{date_stamp}_{pid}.tif")
    tmp_clim_4326 = os.path.join(out_dir, f"tmp_c_4326_{date_stamp}_{pid}.tif")

    # 3-1. EPSG:4326 중간 GeoTIFF 쓰기
    print(f"[*] Writing base raster for Monthly Composite ({total_monthly_bands} bands)...")
    meta_m = meta_base.copy()
    meta_m['count'] = total_monthly_bands
    with rasterio.open(tmp_monthly_4326, 'w', **meta_m) as dst:
        dst.write(monthly_stack)
        for b_idx, desc in enumerate(descriptions_monthly, 1):
            dst.set_band_description(b_idx, desc)

    print(f"[*] Writing base raster for Climatology (12 bands)...")
    meta_c = meta_base.copy()
    meta_c['count'] = 12
    with rasterio.open(tmp_clim_4326, 'w', **meta_c) as dst:
        dst.write(clim_12)
        for b_idx, desc in enumerate(descriptions_clim, 1):
            dst.set_band_description(b_idx, desc)

    # 3-2. 좌표계별 최종 COG 출력
    if args.crs in ('EPSG:4326', 'both'):
        out_m_4326 = out_monthly_final if args.crs == 'EPSG:4326' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m_ea020lc_{date_stamp}_4326.tif")
        out_c_4326 = out_clim_final if args.crs == 'EPSG:4326' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m-clim_ea020lc_{date_stamp}_4326.tif")

        print(f"[*] Translating to EPSG:4326 COG: {os.path.basename(out_m_4326)}...")
        cog_translate(tmp_monthly_4326, out_m_4326, cog_profile, in_memory=True, quiet=True, overview_level=3, overview_resampling='average', web_optimized=True)
        print(f"[*] Translating to EPSG:4326 COG: {os.path.basename(out_c_4326)}...")
        cog_translate(tmp_clim_4326, out_c_4326, cog_profile, in_memory=True, quiet=True, overview_level=3, overview_resampling='average', web_optimized=True)

    if args.crs in ('EPSG:3857', 'both'):
        out_m_3857 = out_monthly_final if args.crs == 'EPSG:3857' else os.path.join(out_dir, f"multi_mergei_le3_cdr-sst01m_ea020lc_{date_stamp}_3857.tif")
        out_c_3857 = out_clim_final if args.crs == 'EPSG:3857' else os.path.join(out_dir, f"multi_merge_le3_cdr-sst01m-clim_ea020lc_{date_stamp}_3857.tif")

        reproject_to_3857_cog(tmp_monthly_4326, out_m_3857, resolution=5000)
        reproject_to_3857_cog(tmp_clim_4326, out_c_3857, resolution=5000)

    # 임시 파일 정리
    if os.path.exists(tmp_monthly_4326):
        os.remove(tmp_monthly_4326)
    if os.path.exists(tmp_clim_4326):
        os.remove(tmp_clim_4326)

    print("================================================================================")
    print("  [SUCCESS] 2-COG System Successfully Created!")
    print(f"  - Date Range:         {final_dates[0]} ~ {last_ym} ({total_monthly_bands} bands)")
    print(f"  - Missing Handling:   {args.missing} (No synthetic noise applied)")
    print(f"  - Target CRS:         {args.crs}")
    print(f"  - Extent:             Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}]")
    print(f"  1. Monthly Composite: {out_monthly_final}")
    print(f"  2. Climatology:       {out_clim_final}")
    print("================================================================================")


if __name__ == '__main__':
    main()

