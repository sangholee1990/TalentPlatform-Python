#define MyAppName "NMSC Climate Toolbox"
#define MyAppVersion "1.0"
#define MyAppPublisher "nmsc & indisystem"
#define MyAppExePath "NMSC_Climate_Toolbox"
#define MyAppExeName "NMSC_Climate_Toolbox.exe"
#define MySourceDir "C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\bin\dist"

[Setup]
AppId={{D890C234-E567-4F9A-81BC-DEF012345680}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
Compression=lzma2/ultra64
SolidCompression=yes
OutputDir="C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\bin\pkg"
OutputBaseFilename="NMSC_Climate_Toolbox"
SetupIconFile=C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\icon\icon.ico
PrivilegesRequired=admin
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
Source: "{#MySourceDir}\{#MyAppExePath}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MySourceDir}\{#MyAppExePath}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "C:\SYSTEMS\PROG\PYTHON\TalentPlatform-Python\src\proj\indisystem\2026\nmscClimateToolbox\icon\icon.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\icon.ico"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
