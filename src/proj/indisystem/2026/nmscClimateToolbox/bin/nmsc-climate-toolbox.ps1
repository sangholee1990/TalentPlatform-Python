# Settings
$CondaExe = "C:\Users\HP\anaconda3\envs\py311"
$CondaPs = "C:\Users\HP\anaconda3\shell\condabin\conda-hook.ps1"
$IsccExe = "C:\Program Files\Inno Setup 7\ISCC.exe"
$ProjectRoot = "C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox"
$AppName = "NMSC_Climate_Toolbox"
$IconFile = Join-Path $ProjectRoot "icon\icon.ico"

$PyinstallerExe = Join-Path $CondaExe "Scripts\pyinstaller.exe"
$PythonFile = Join-Path $ProjectRoot "nmsc_climate_toolbox_gui.py"
$IssFile = Join-Path $ProjectRoot "bin\nmsc-climate-toolbox.iss"
$DistPath = Join-Path $ProjectRoot "bin\dist"
$PkgPath = Join-Path $ProjectRoot "bin\pkg"
$ErrorActionPreference = "Stop"

Write-Host "===================================================" -ForegroundColor Blue
Write-Host " PowerShell Compile" -ForegroundColor Blue
Write-Host "===================================================" -ForegroundColor Blue
Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Compiling..." -ForegroundColor Blue

# Conda environment
. "$CondaPs"
conda activate py311

# Stop running process before compiling if locked
$ActualProcessName = $AppName
$RunningApp = Get-Process -Name $ActualProcessName -ErrorAction SilentlyContinue
if ($RunningApp) {
    Write-Host "Stopping running app..." -ForegroundColor Blue
    try { $RunningApp | Stop-Process -Force -ErrorAction Stop } catch { Write-Host "Warning: could not stop process - $($_.Exception.Message)" -ForegroundColor Yellow }
    Start-Sleep -Seconds 1
}

# Build semicolon-safe argument strings using char(59)
$s = [char]59
$addDataIcon = "$ProjectRoot\icon${s}icon"
$addDataToolbox = "$ProjectRoot\nmsc_climate_toolbox.py${s}."
$addDataVal = "$ProjectRoot\doc\Validation_BUOY_SST_2D.nc${s}doc"
$addDataBasemap = "$CondaExe\Lib\site-packages\mpl_toolkits\basemap_data${s}mpl_toolkits\basemap_data"

$pyArgs = @(
    '--clean', '-y', '--noconsole', '--onedir',
    '--icon', $IconFile,
    '--specpath', "$ProjectRoot\bin",
    '--distpath', $DistPath,
    '--workpath', "$ProjectRoot\bin\build",
    '--add-data', $addDataIcon,
    '--add-data', $addDataToolbox,
    '--add-data', $addDataVal,
    '--add-data', $addDataBasemap,
    '--collect-data', 'qt_material',
    '--collect-data', 'cartopy',
    '--collect-data', 'rasterio',
    '--collect-data', 'pyproj',
    '--collect-data', 'rioxarray',
    '--collect-data', 'xarray',
    '--collect-binaries', 'llama_cpp',
    '--collect-data', 'llama_cpp',
    '--copy-metadata', 'xarray',
    '--copy-metadata', 'rioxarray',
    '--copy-metadata', 'rasterio',
    '--hidden-import', 'numpy',
    '--hidden-import', 'pandas',
    '--hidden-import', 'scipy',
    '--hidden-import', 'scipy.stats',
    '--hidden-import', 'scipy.ndimage',
    '--hidden-import', 'matplotlib',
    '--hidden-import', 'seaborn',
    '--hidden-import', 'xarray',
    '--hidden-import', 'rioxarray',
    '--hidden-import', 'rasterio',
    '--hidden-import', 'rasterio.sample',
    '--hidden-import', 'rasterio.vrt',
    '--hidden-import', 'rasterio._features',
    '--hidden-import', 'netCDF4',
    '--hidden-import', 'h5netcdf',
    '--hidden-import', 'cftime',
    '--hidden-import', 'dask',
    '--hidden-import', 'cartopy',
    '--hidden-import', 'mpl_toolkits.basemap',
    '--hidden-import', 'qt_material',
    '--hidden-import', 'jinja2',
    '--hidden-import', 'PIL',
    '--hidden-import', 'google.generativeai',
    '--hidden-import', 'llama_cpp',
    '--hidden-import', 'llama_cpp.llama_chat_format',
    '--hidden-import', 'resources_rc',
    '--hidden-import', 'nmsc_climate_toolbox',
    '--exclude-module', 'PyQt5',
    '--exclude-module', 'PySide6',
    '--exclude-module', 'torch',
    '--exclude-module', 'torchvision',
    '--exclude-module', 'tensorflow',
    '--exclude-module', 'xgboost',
    '--exclude-module', 'transformers',
    '--exclude-module', 'sklearn',
    '--exclude-module', 'nltk',
    '--exclude-module', 'uvicorn',
    '--exclude-module', 'fastapi',
    '--exclude-module', 'llama_cpp.server',
    '--exclude-module', 'numba',
    '--exclude-module', 'llvmlite',
    '--exclude-module', 'sympy',
    '--exclude-module', 'sqlalchemy',
    '--exclude-module', 'statsmodels',
    '--exclude-module', 'openpyxl',
    '--exclude-module', 'pyarrow',
    '--exclude-module', 'distributed',
    '--exclude-module', 'tkinter',
    '--exclude-module', 'IPython',
    '--exclude-module', 'jupyter',
    '--exclude-module', 'notebook',
    '--exclude-module', 'pytest',
    '--name', $AppName, $PythonFile
)
& "$PyinstallerExe" @pyArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Compile FAILED" -ForegroundColor Red
    exit 1
}
Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Compile OK: $DistPath" -ForegroundColor Green

Write-Host "===================================================" -ForegroundColor Blue
Write-Host " PowerShell Packaging" -ForegroundColor Blue
Write-Host "===================================================" -ForegroundColor Blue
Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Generating installer script..." -ForegroundColor Blue

$IssContent = @"
#define MyAppName "NMSC Climate Toolbox"
#define MyAppVersion "1.0"
#define MyAppPublisher "nmsc & indisystem"
#define MyAppExePath "$AppName"
#define MyAppExeName "$AppName.exe"
#define MySourceDir "$DistPath"

[Setup]
AppId={{D890C234-E567-4F9A-81BC-DEF012345680}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir="$PkgPath"
OutputBaseFilename="$AppName"
SetupIconFile=$IconFile
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#MySourceDir}\{#MyAppExePath}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MySourceDir}\{#MyAppExePath}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "$IconFile"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
"@

$IssContent | Out-File -FilePath $IssFile -Encoding utf8
Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Installer script written" -ForegroundColor Green

$RunningApp = Get-Process -Name $ActualProcessName -ErrorAction SilentlyContinue
if ($RunningApp) {
    Write-Host "Stopping running app..." -ForegroundColor Blue
    try { $RunningApp | Stop-Process -Force -ErrorAction Stop } catch { Write-Host "Warning: could not stop process - $($_.Exception.Message)" -ForegroundColor Yellow }
    Start-Sleep -Seconds 1
}

if (Get-Process "Compil32" -ErrorAction SilentlyContinue) {
    Write-Host "Stopping Inno Setup..." -ForegroundColor Blue
    try { Stop-Process -Name "Compil32" -Force -ErrorAction Stop } catch { Write-Host "Warning: could not stop Compil32 - $($_.Exception.Message)" -ForegroundColor Yellow }
}

$PkgExe = Join-Path $PkgPath "$AppName.exe"
if (Test-Path $PkgExe) {
    $lockedBy = Get-Process | Where-Object { $_.Modules.FileName -eq $PkgExe } -ErrorAction SilentlyContinue
    if ($lockedBy) {
        Write-Host "Stopping process holding pkg exe..." -ForegroundColor Blue
        $lockedBy | Stop-Process -Force
        Start-Sleep -Seconds 1
    }
}

Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Packaging..." -ForegroundColor Blue

if (-not (Test-Path $IsccExe)) {
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] FAILED: $IsccExe not found" -ForegroundColor Red
    exit 1
}

& $IsccExe "$IssFile"

if ($LASTEXITCODE -ne 0) {
    Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Packaging FAILED, exit code $LASTEXITCODE" -ForegroundColor Red
    exit 1
}

Write-Host "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Packaging OK: $IssFile" -ForegroundColor Green
exit 0
