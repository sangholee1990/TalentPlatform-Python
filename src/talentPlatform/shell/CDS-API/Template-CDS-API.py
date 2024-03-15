import cdsapi

c = cdsapi.Client()

c.retrieve(
    'derived-near-surface-meteorological-variables',
    {
        'version': '2.1',
        'format': 'tgz',
        'variable': [
            'grid_point_altitude', 'near_surface_air_temperature', 'near_surface_specific_humidity',
            'near_surface_wind_speed', 'rainfall_flux', 'snowfall_flux',
            'surface_air_pressure', 'surface_downwelling_longwave_radiation', 'surface_downwelling_shortwave_radiation',
        ],
        'reference_dataset': 'cru',
        'year': [
            '1983',
        ],
        'month': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10', '11', '12',
        ],
    },
    '/data3/dxinyu/WFDE5/download_1983.tar.gz')
