@echo off
setlocal EnableExtensions EnableDelayedExpansion

if /I not "%~1"=="--inner" (
  set "ZLC_ACTION=build + program"
  if /I "%~1"=="--diagnose" set "ZLC_ACTION=hardware diagnose"
  if /I "%~1"=="--build-only" set "ZLC_ACTION=build"
  if /I "%~1"=="--program-only" set "ZLC_ACTION=program"
  if /I "%~1"=="--flash" set "ZLC_ACTION=program SPI flash (persists across power cycle)"
  call "%~f0" --inner %*
  set "ZLC_STATUS=!ERRORLEVEL!"
  if "!ZLC_STATUS!"=="0" (
    if "%~1"=="--help" exit /b 0
    if "%~1"=="/?" exit /b 0
    echo.
    echo ZLC !ZLC_ACTION! completed successfully.
    echo You can close this window, or press any key to exit.
    if "%ZLC_NO_PAUSE%"=="" pause
  ) else (
    echo.
    echo ZLC !ZLC_ACTION! failed with code !ZLC_STATUS!.
    echo Keep this window open and read the messages above.
    if "%ZLC_NO_PAUSE%"=="" pause
  )
  exit /b !ZLC_STATUS!
)
shift /1

rem This launcher lives in bin\ -- one folder for everything a human clicks --
rem while what it drives is zlc_pulse's own fpga tree: its RTL, its board
rem config, its build.  The path is derived from the repository root rather
rem than from %~dp0, so moving the launcher does not move the board.
set "FPGA_DIR=%~dp0..\packages\zlc_pulse\fpga\"
for %%I in ("%~dp0..") do set "ZLC_HOME=%%~fI"
for %%I in ("%FPGA_DIR%..") do set "PULSE_ROOT=%%~fI"
set "STREAMER_DIR=%FPGA_DIR%pulse_streamer"
set "ZLC_REPO_ROOT=%ZLC_HOME%"
set "ZLC_CREATE_TCL=create_project.tcl"
set "ZLC_PROGRAM_TCL=program_fpga.tcl"
set "ZLC_PROJ_SUB=ps"
set "ZLC_TOP=zlc_pulse_streamer_top"

set "MODE=build_program"
set "ZLC_OPTION_OK="
if "%~1"=="" set "ZLC_OPTION_OK=1"
if "%~1"=="--help" goto zlc_help
if "%~1"=="/?" goto zlc_help
if /I "%~1"=="--diagnose" (set "MODE=diagnose"& set "ZLC_OPTION_OK=1")
if /I "%~1"=="--build-only" (set "MODE=build"& set "ZLC_OPTION_OK=1")
if /I "%~1"=="--program-only" (set "MODE=program"& set "ZLC_OPTION_OK=1")
if /I "%~1"=="--flash" (set "MODE=flash"& set "ZLC_OPTION_OK=1")
rem --force-build rebuilds even if the sources are unchanged (default mode otherwise
rem skips the build and programs the existing bitstream when nothing changed).
if /I "%~1"=="--force-build" (set "ZLC_FORCE_BUILD=1"& set "ZLC_OPTION_OK=1")
if not defined ZLC_OPTION_OK (
  echo Unknown option: %~1
  echo.
  goto zlc_help
)

call "%FPGA_DIR%_resolve_tools.bat" vivado
if errorlevel 1 exit /b 1
call :zlc_default_paths
call :zlc_verify_sources
if errorlevel 1 exit /b 1

call "%FPGA_DIR%_resolve_tools.bat" python "%ZLC_HOME%"
if errorlevel 1 exit /b 1
call :zlc_require_product
if errorlevel 1 exit /b 1
set "ZLC_CFG_JSON=%PULSE_ROOT%\fpga\board_config\streamer_config.json"
call :zlc_require_config
if errorlevel 1 exit /b 1
call :zlc_resolve_part
if errorlevel 1 exit /b 1

if /I "%MODE%"=="diagnose" (
  call :zlc_run_tcl "diagnose_hw_target.tcl"
  exit /b !ERRORLEVEL!
)

rem --flash: write the EXISTING bitstream into the board's SPI configuration flash so the FPGA
rem auto-boots from flash and the program SURVIVES a power cycle (program_fpga.tcl only loads the
rem volatile config SRAM).  Needs a built .bit + the MODE jumper set to SPI/QSPI boot.
if /I "%MODE%"=="flash" (
  echo ZLC FPGA pulse streamer: program SPI flash ^(persists across power cycle^)
  call :zlc_run_tcl "program_flash.tcl"
  exit /b !ERRORLEVEL!
)

if /I "%MODE%"=="program" goto zlc_program

rem Only a real build/check may derive source files from streamer_config.json.
rem Diagnose/program/flash consume the already-built frozen artifact and must be read-only.
call :zlc_emit_geom
if errorlevel 1 exit /b 1
call :zlc_print_capacity_estimate
if errorlevel 1 exit /b 1


rem Skip the slow synth+implementation when a bitstream and its routed reports already match
rem every recorded source/artifact input.  Default mode still PROGRAMS that qualified bit;
rem --build-only stops after proving the bitstream current.
call :zlc_check_prebuilt
if not defined ZLC_FORCE_BUILD if defined ZLC_PREBUILT (
  echo ZLC bitstream is up to date ^(sources unchanged since last build^) -- skipping build.
  echo ZLC   bit: !ZLC_BIT!
  echo ZLC   ^(force a rebuild with: build_and_program.bat --force-build^)
  if /I "%MODE%"=="build_program" goto zlc_program
  exit /b 0
)

echo ZLC FPGA pulse streamer: build FINAL bitstream (1-tick FIFO prefetch + autonomous scan, JTAG-to-AXI)
call :zlc_run_tcl "!ZLC_CREATE_TCL!"
if errorlevel 1 exit /b 1
call :zlc_save_src_hash
if errorlevel 1 exit /b 1

if /I "%MODE%"=="build_program" goto zlc_program
if /I "%MODE%"=="build" exit /b 0

:zlc_program
echo ZLC FPGA pulse streamer: program FINAL bitstream
call :zlc_run_tcl "!ZLC_PROGRAM_TCL!"
exit /b %ERRORLEVEL%

:zlc_help
echo Build/program the FINAL ZLC FPGA pulse streamer (one clean design, no variants).
echo Control path: JTAG-to-AXI master -^> AXI BRAM controller -^> edge/scan BRAMs + bus loader.
echo Engine: 1-tick (20 ns) FIFO prefetch + streamed autonomous ping-pong scan.
echo.
echo Usage:
echo   bin\build_and_program.bat              Build if needed, then PROGRAM volatile FPGA
echo   bin\build_and_program.bat --force-build Rebuild, then PROGRAM volatile FPGA
echo   bin\build_and_program.bat --build-only Build only
echo   bin\build_and_program.bat --program-only Program existing bitstream (VOLATILE: lost on power-off)
echo   bin\build_and_program.bat --flash      Program the SPI flash so the program SURVIVES a power cycle
echo   bin\build_and_program.bat --diagnose   List Vivado hw targets/devices
echo.
echo --program-only loads the VOLATILE FPGA config (lost when the board powers off).  --flash writes
echo the bitstream into the board's SPI configuration flash so the FPGA AUTO-BOOTS from it on every
echo power-up -- run it ONCE and the program persists across reboots (no rebuild needed).  Requires the
echo board MODE jumper on SPI/QSPI boot; set ZLC_PS_CFGMEM_PART if your flash differs from the default
echo (run 'get_cfgmem_parts' in Vivado to list valid names).
echo.
echo The default mode SKIPS the (slow) synth+impl only when the Vivado build, target part,
echo engine/top HDL, create tcl, board XDC, streamer_config and generated geometry all match,
echo and the bitstream plus routed reports match their saved receipt; it then PROGRAMS that bit.
echo --force-build forces a rebuild before programming.  The qualified build receipt
echo is fpga\build\ps\.zlc_src_hash.
echo.
echo Real build XDC:
echo   fpga\board_config\board.xdc
echo   This is the default 62-output board pin map ^(see fpga\board_config\README.md^).
echo   For a different board, replace board.xdc or set:
echo   set ZLC_PS_XDC=C:\path\to\board.xdc
echo.
echo Optional:
echo   set ZLC_PS_VIVADO_BIN=C:\Xilinx\Vivado\2019.2\bin\vivado.bat
echo   set ZLC_PS_PROJECT_DIR=%%CD%%\fpga\build\ps
exit /b 0

:zlc_verify_sources
set "ZLC_DEFAULT_XDC=%PULSE_ROOT%\fpga\board_config\board.xdc"
if not defined ZLC_PS_XDC set "ZLC_PS_XDC=%ZLC_DEFAULT_XDC%"
set "ZLC_SELECTED_XDC=%ZLC_PS_XDC%"
if not exist "%STREAMER_DIR%\zlc_edge_streamer.v" (
  echo ERROR: missing FINAL engine HDL: %STREAMER_DIR%\zlc_edge_streamer.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\zlc_pulse_streamer_top.v" (
  echo ERROR: missing FINAL top HDL: %STREAMER_DIR%\zlc_pulse_streamer_top.v
  exit /b 2
)
if not exist "%STREAMER_DIR%\create_project.tcl" (
  echo ERROR: missing FINAL build Tcl: %STREAMER_DIR%\create_project.tcl
  exit /b 2
)
findstr /C:"zlc_edge_streamer.v" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not read the FINAL engine HDL.
  exit /b 2
)
findstr /C:"module zlc_pulse_streamer_top" "%STREAMER_DIR%\zlc_pulse_streamer_top.v" >nul || (
  echo ERROR: FINAL top module name is wrong.
  exit /b 2
)
findstr /C:"module zlc_edge_streamer" "%STREAMER_DIR%\zlc_edge_streamer.v" >nul || (
  echo ERROR: FINAL engine module name is wrong.
  exit /b 2
)
findstr /C:"create_ip -name jtag_axi" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not create the JTAG-to-AXI master IP.
  exit /b 2
)
findstr /C:"create_ip -name axi_bram_ctrl" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not create the AXI BRAM controller IP.
  exit /b 2
)
findstr /C:"blk_mem_gen_edge_tick" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not create the 3 parallel edge BRAMs.
  exit /b 2
)
findstr /C:"zlc_force_latency2" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl does not force the edge-BRAM read latency to 2.
  exit /b 2
)
findstr /C:"zlc_safe_project_dir" "%STREAMER_DIR%\create_project.tcl" >nul || (
  echo ERROR: create_project.tcl is missing the Vivado path-length guard.
  exit /b 2
)
if not exist "!ZLC_SELECTED_XDC!" (
  echo ERROR: missing board XDC: !ZLC_SELECTED_XDC!
  echo Put your board pin map at fpga\board_config\board.xdc -- see its README -- or set ZLC_PS_XDC.
  exit /b 2
)
findstr /C:"[get_ports trig]" "!ZLC_SELECTED_XDC!" >nul || (
  echo ERROR: selected XDC does not define the trig output.
  exit /b 2
)
findstr /C:"<PIN_CH" "!ZLC_SELECTED_XDC!" >nul && (
  echo ERROR: selected XDC still contains PIN_CH placeholders: !ZLC_SELECTED_XDC!
  exit /b 2
)
echo ZLC FINAL source contract: channels=62 num_slots=4 control=JTAG-to-AXI (jtag_axi+axi_bram_ctrl+5 BRAMs, forced edge latency 2)
echo ZLC FINAL XDC: !ZLC_SELECTED_XDC!
exit /b 0

:zlc_require_product
rem Hardware projection must come from the one installed distribution. Running
rem from a neutral directory prevents this checkout's bootstrap or a neighboring
rem standalone layer from masking a missing or stale product install.
pushd "%TEMP%"
%ZLC_PY_CMD% -m zou_lab_control check
set "ZLC_PRODUCT_STATUS=!ERRORLEVEL!"
popd
if not "!ZLC_PRODUCT_STATUS!"=="0" (
  echo ERROR: the selected Python does not own one valid zou-lab-control product.
  echo        Run bin\install_requirements.bat, or set ZLC_PY_CMD to the
  echo        interpreter where the root product is installed.
  exit /b !ZLC_PRODUCT_STATUS!
)
exit /b 0

:zlc_require_config
if not exist "%ZLC_CFG_JSON%" (
  echo ERROR: canonical FPGA config is missing: %ZLC_CFG_JSON%
  exit /b 1
)
pushd "%TEMP%"
%ZLC_PY_CMD% -c "import dataclasses,json,pathlib,sys;import zou_lab_control;from zlc_pulse.fpga import load_streamer_config;from zlc_pulse.wire import StreamerParams;p=pathlib.Path(sys.argv[1]).resolve();pairs=lambda x:dict(x) if len(x)==len(dict(x)) else (_ for _ in ()).throw(ValueError('duplicate key in streamer_config.json'));raw=json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('non-finite JSON constant '+x)));top={'_README','_field_docs','fpga_part','clock_hz','target_pct','params','board'};expected={f.name for f in dataclasses.fields(StreamerParams)}|{'slot_mul_width'};assert isinstance(raw,dict) and set(raw)==top,'streamer_config.json fields are not exact';assert isinstance(raw['params'],dict) and set(raw['params'])==expected,'streamer_config.json params fields are not exact';assert isinstance(raw['board'],dict),'streamer_config.json board must be an object';assert isinstance(raw['fpga_part'],str) and raw['fpga_part'].strip(),'fpga_part must be non-empty text';cfg=load_streamer_config(p);assert cfg['source'] is not None and pathlib.Path(cfg['source']).resolve()==p,'build config fell back from the requested file';assert not cfg['warnings'],'; '.join(cfg['warnings']);print(cfg['fpga_part'])" "%ZLC_CFG_JSON%"
set "ZLC_CONFIG_STATUS=%ERRORLEVEL%"
popd
if not "%ZLC_CONFIG_STATUS%"=="0" (
  echo ERROR: streamer_config.json is invalid; build/program/flash/diagnose refuse fallback geometry.
  exit /b 1
)
exit /b 0

:zlc_resolve_part
rem Single source: take the synthesis part from fpga\board_config\streamer_config.json
rem (unless ZLC_PS_FPGA_PART is already set) and export it so create_project.tcl targets
rem the configured board.  Python was resolved once at the build boundary above.
if not "%ZLC_PS_FPGA_PART%"=="" goto :zlc_resolve_part_done
if not exist "%ZLC_CFG_JSON%" (
  echo ERROR: canonical FPGA config is missing: %ZLC_CFG_JSON%
  exit /b 1
)
rem Through a file, not for /f: cmd re-parses the command inside for /f and
rem eats the quotes around an interpreter given as a full path.
set "ZLC_PART_FILE=%TEMP%\zlc_fpga_part.txt"
%ZLC_PY_CMD% -c "import json;print(json.load(open(r'%ZLC_CFG_JSON%'))['fpga_part'])" > "%ZLC_PART_FILE%"
if errorlevel 1 (
  del "%ZLC_PART_FILE%" >nul 2>&1
  echo ERROR: could not read fpga_part from canonical FPGA config.
  exit /b 1
)
if exist "%ZLC_PART_FILE%" set /p ZLC_PS_FPGA_PART=<"%ZLC_PART_FILE%"
del "%ZLC_PART_FILE%" >nul 2>&1
:zlc_resolve_part_done
if "%ZLC_PS_FPGA_PART%"=="" (
  echo ERROR: canonical FPGA config did not provide fpga_part.
  exit /b 1
)
echo ZLC synthesis part: %ZLC_PS_FPGA_PART% (from streamer_config.json / env)
exit /b 0

:zlc_emit_geom
rem Single source: from streamer_config.json generate (1) the Vivado geometry tcl (BRAM-IP sizes)
rem create_project.tcl sources via ZLC_PS_GEOM_TCL, and (2) regenerate zlc_geometry.vh (the RTL
rem parameter defaults + LAYOUT_FINGERPRINT the .v `include) IN PLACE, so editing the config changes
rem the SYNTHESIZED bitstream (IP depths + every RTL geometry param + the connect fingerprint) with
rem no hand edits.  This routine is called only for a real build/check and fails closed: synthesis
rem must never proceed with a stale header or a partially generated geometry file.
pushd "%TEMP%"
rem NOT >nul 2>nul.  The generic "failed to derive FPGA geometry" below is
rem worth nothing on its own -- the reason is whatever Python printed, and
rem throwing it away is the same mistake that hid "python is not recognized"
rem behind an empty source hash for months.
%ZLC_PY_CMD% -m zou_lab_control fpga --config "%ZLC_CFG_JSON%" --emit-geometry-vh "%STREAMER_DIR%\zlc_geometry.vh" >nul
if errorlevel 1 goto zlc_emit_geom_fail
if "%ZLC_PS_GEOM_TCL%"=="" (
  set "ZLC_GEOM_OUT=%ZLC_PS_BUILD_ROOT%\geom.tcl"
  %ZLC_PY_CMD% -m zou_lab_control fpga --config "%ZLC_CFG_JSON%" --emit-geom-tcl "!ZLC_GEOM_OUT!" >nul
  if errorlevel 1 goto zlc_emit_geom_fail
  set "ZLC_PS_GEOM_TCL=!ZLC_GEOM_OUT!"
)
popd
echo ZLC RTL geometry header regenerated from config: %STREAMER_DIR%\zlc_geometry.vh
if not "%ZLC_PS_GEOM_TCL%"=="" echo ZLC geometry tcl: %ZLC_PS_GEOM_TCL% (from streamer_config.json)
exit /b 0

:zlc_emit_geom_fail
echo.
echo ERROR: failed to derive FPGA geometry from streamer_config.json.
echo   config: %ZLC_CFG_JSON%
echo   header: %STREAMER_DIR%\zlc_geometry.vh
echo   geom tcl: %ZLC_PS_BUILD_ROOT%\geom.tcl
echo   The reason is printed above, from zlc_pulse.fpga.
exit /b 1

:zlc_compute_src_hash
rem Hash every bit-relevant input plus the complete Vivado build identity and
rem the resolved target part.  program_fpga.tcl is deliberately absent: it can
rem load an existing bit but cannot change one.  Every listed input is required;
rem a missing file disables reuse instead of being silently omitted.
rem
rem Each file is identified by its path RELATIVE to the repository root, so WHICH files went in is
rem part of the hash without WHERE the checkout sits being part of it.  Absolute paths made the
rem cache non-relocatable: moving this tree changed every path, so a bitstream built from
rem byte-for-byte identical sources was reported out of date and the honest answer to "what
rem changed?" was "nothing did".
set "ZLC_SRC_HASH="
set "ZLC_HASH_GEOM="
if defined ZLC_PS_GEOM_TCL if exist "%ZLC_PS_GEOM_TCL%" set "ZLC_HASH_GEOM=%ZLC_PS_GEOM_TCL%"
rem Hash EVERY synthesized HDL create_project.tcl reads -- INCLUDING zlc_uart_bridge.v.  Omitting it
rem meant a UART-bridge edit did not invalidate the build cache, so the bat "skipped build" and
rem re-programmed a stale bitstream (the byte-mux fix silently never made it onto the board).
rem NOT through `for /f ('...')`: cmd strips the outer quotes of a piped command line, so a
rem quoted interpreter followed by a quoted -c script came back as `'python" -c "import' is not
rem recognized` -- on stderr, which the loop swallowed.  The hash was therefore ALWAYS empty:
rem every run rebuilt from scratch and no build was ever recorded, silently, for as long as the
rem interpreter path has been quoted.  A plain redirect parses the command line the normal way.
set "ZLC_HASH_TMP=%ZLC_PS_BUILD_ROOT%\.zlc_src_hash.tmp"
set "ZLC_VIVADO_ID_TMP=%ZLC_PS_BUILD_ROOT%\.zlc_vivado_id.tmp"
del "%ZLC_HASH_TMP%" "%ZLC_VIVADO_ID_TMP%" >nul 2>nul
pushd "!ZLC_PS_BUILD_ROOT!" || exit /b 1
call "%ZLC_PS_VIVADO_BIN%" -version > "%ZLC_VIVADO_ID_TMP%" 2>&1
set "ZLC_VIVADO_VERSION_STATUS=!ERRORLEVEL!"
popd
if not "!ZLC_VIVADO_VERSION_STATUS!"=="0" (
  echo ERROR: could not read the selected Vivado build identity.
  del "%ZLC_VIVADO_ID_TMP%" >nul 2>nul
  exit /b 1
)
%ZLC_PY_CMD% -c "import hashlib,os,sys;r,part,top,tool=sys.argv[1:5];paths=sys.argv[5:];missing=[p for p in paths if not p or not os.path.isfile(p)];assert not missing,'missing bitstream input(s): '+repr(missing);h=hashlib.sha256();h.update(b'tool\0');h.update(open(tool,'rb').read());h.update(b'part\0'+part.encode());h.update(b'top\0'+top.encode());[(h.update(os.path.relpath(p,r).replace(chr(92),chr(47)).encode()),h.update(b'\0'),h.update(open(p,'rb').read())) for p in paths];print(h.hexdigest())" "%ZLC_HOME%" "%ZLC_PS_FPGA_PART%" "%ZLC_TOP%" "%ZLC_VIVADO_ID_TMP%" "%STREAMER_DIR%\zlc_edge_streamer.v" "%STREAMER_DIR%\zlc_uart_bridge.v" "%STREAMER_DIR%\zlc_pulse_streamer_top.v" "%STREAMER_DIR%\zlc_geometry.vh" "%STREAMER_DIR%\!ZLC_CREATE_TCL!" "!ZLC_SELECTED_XDC!" "%PULSE_ROOT%\fpga\board_config\streamer_config.json" "!ZLC_HASH_GEOM!" > "%ZLC_HASH_TMP%"
set "ZLC_HASH_STATUS=%ERRORLEVEL%"
del "%ZLC_VIVADO_ID_TMP%" >nul 2>nul
if not "%ZLC_HASH_STATUS%"=="0" (
  del "%ZLC_HASH_TMP%" >nul 2>nul
  exit /b 1
)
if exist "%ZLC_HASH_TMP%" set /p ZLC_SRC_HASH=<"%ZLC_HASH_TMP%"
del "%ZLC_HASH_TMP%" >nul 2>nul
if not defined ZLC_SRC_HASH exit /b 1
exit /b 0

:zlc_compute_artifact_hash
rem Bind the skip decision to the qualified outputs, not merely to source age.
rem Replacing a bit, deleting a report, or editing routed evidence invalidates
rem the receipt and forces a fresh implementation.
set "ZLC_ARTIFACT_HASH="
set "ZLC_ARTIFACT_TMP=%ZLC_PS_BUILD_ROOT%\.zlc_artifact_hash.tmp"
del "%ZLC_ARTIFACT_TMP%" >nul 2>nul
%ZLC_PY_CMD% -c "import hashlib,os,sys;r=sys.argv[1];paths=sys.argv[2:];missing=[p for p in paths if not os.path.isfile(p)];assert not missing,'missing qualified artifact(s): '+repr(missing);h=hashlib.sha256();[(h.update(os.path.relpath(p,r).replace(chr(92),chr(47)).encode()),h.update(b'\0'),h.update(open(p,'rb').read())) for p in paths];print(h.hexdigest())" "%ZLC_PS_PROJECT_DIR%" "%ZLC_BIT%" "%ZLC_TIMING_RPT%" "%ZLC_BUS_SKEW_RPT%" "%ZLC_UTIL_RPT%" > "%ZLC_ARTIFACT_TMP%"
if errorlevel 1 (
  del "%ZLC_ARTIFACT_TMP%" >nul 2>nul
  exit /b 1
)
if exist "%ZLC_ARTIFACT_TMP%" set /p ZLC_ARTIFACT_HASH=<"%ZLC_ARTIFACT_TMP%"
del "%ZLC_ARTIFACT_TMP%" >nul 2>nul
if not defined ZLC_ARTIFACT_HASH exit /b 1
exit /b 0

:zlc_check_prebuilt
rem Set ZLC_PREBUILT=1 iff the bitstream exists AND the stored source hash matches the current
rem sources (i.e. nothing that affects the .bit changed since it was built).
set "ZLC_PREBUILT="
set "ZLC_HASHFILE=%ZLC_PS_PROJECT_DIR%\.zlc_src_hash"
if not exist "%ZLC_HASHFILE%" exit /b 0
call :zlc_compute_src_hash
if errorlevel 1 exit /b 0
call :zlc_compute_artifact_hash
if errorlevel 1 exit /b 0
set "ZLC_STORED_HASH="
set /p ZLC_STORED_HASH=<"%ZLC_HASHFILE%"
if "%ZLC_STORED_HASH%"=="%ZLC_SRC_HASH%:%ZLC_ARTIFACT_HASH%" set "ZLC_PREBUILT=1"
exit /b 0

:zlc_save_src_hash
rem Record the current source hash next to the freshly built bitstream so the next default-mode
rem run can skip the build when nothing changed.
call :zlc_compute_src_hash
if errorlevel 1 (
  echo ERROR: the bitstream inputs could not be hashed; refusing an unqualified build receipt.
  exit /b 1
)
call :zlc_compute_artifact_hash
if errorlevel 1 (
  echo ERROR: bitstream or routed reports are missing; refusing an incomplete build receipt.
  exit /b 1
)
> "%ZLC_PS_PROJECT_DIR%\.zlc_src_hash" echo %ZLC_SRC_HASH%:%ZLC_ARTIFACT_HASH%
exit /b 0

:zlc_print_capacity_estimate
set "ZLC_EST_PART=%ZLC_PS_FPGA_PART%"
if "%ZLC_EST_PART%"=="" set "ZLC_EST_PART=xc7a35tfgg484-2"
pushd "%TEMP%"
%ZLC_PY_CMD% -m zou_lab_control fpga --part "%ZLC_EST_PART%"
popd
exit /b 0

:zlc_default_paths
if defined ZLC_PS_BUILD_ROOT if "!ZLC_PS_BUILD_ROOT: =!"=="" set "ZLC_PS_BUILD_ROOT="
if defined ZLC_PS_PROJECT_DIR if "!ZLC_PS_PROJECT_DIR: =!"=="" set "ZLC_PS_PROJECT_DIR="
if defined ZLC_PS_LOG_DIR if "!ZLC_PS_LOG_DIR: =!"=="" set "ZLC_PS_LOG_DIR="
if not defined ZLC_PS_BUILD_ROOT set "ZLC_PS_BUILD_ROOT=%FPGA_DIR%build"
set "ZLC_PS_PROJECT_ROOT=!ZLC_PS_BUILD_ROOT!"
if not exist "!ZLC_PS_BUILD_ROOT!\" mkdir "!ZLC_PS_BUILD_ROOT!" >nul 2>nul
rem Stop cloud-sync (Dropbox) from syncing -- and thus file-locking -- the Vivado build artifacts:
rem a held IP-cache handle makes create_project's project-dir delete fail with "permission denied".
rem Best-effort NTFS "com.dropbox.ignored" stream on the build root; harmless when the tree is not
rem synced.  This is why you never need to set ZLC_PS_BUILD_ROOT to dodge the lock -- just run this bat.
>"!ZLC_PS_BUILD_ROOT!:com.dropbox.ignored" echo 1 2>nul
rem In-repo build (fpga\build\ps).  The SHORT subdir "ps" keeps Vivado's deep
rem run/.Xil temp path under the Windows MAX_PATH limit without leaving fpga/.
if not defined ZLC_PS_PROJECT_DIR set "ZLC_PS_PROJECT_DIR=%ZLC_PS_BUILD_ROOT%\!ZLC_PROJ_SUB!"
if not defined ZLC_PS_LOG_DIR set "ZLC_PS_LOG_DIR=%ZLC_PS_BUILD_ROOT%\logs"
set "ZLC_IMPL_DIR=!ZLC_PS_PROJECT_DIR!\!ZLC_PROJ_SUB!.runs\impl_1"
set "ZLC_BIT=!ZLC_IMPL_DIR!\!ZLC_TOP!.bit"
set "ZLC_TIMING_RPT=!ZLC_IMPL_DIR!\!ZLC_TOP!_timing_summary_routed.rpt"
set "ZLC_BUS_SKEW_RPT=!ZLC_IMPL_DIR!\!ZLC_TOP!_bus_skew_routed.rpt"
set "ZLC_UTIL_RPT=!ZLC_IMPL_DIR!\!ZLC_TOP!_utilization_routed.rpt"
echo ZLC build root: %ZLC_PS_BUILD_ROOT%
exit /b 0

:zlc_run_tcl
set "TCL_NAME=%~1"
set "TCL_STEM=%~n1"
set "DIRECT_TCL=%STREAMER_DIR%\%TCL_NAME%"
if not exist "%DIRECT_TCL%" (
  echo Missing Tcl script: %DIRECT_TCL%
  exit /b 2
)
if not exist "%ZLC_PS_LOG_DIR%" mkdir "%ZLC_PS_LOG_DIR%" >nul 2>nul

echo ZLC direct Vivado path: %DIRECT_TCL%
echo ZLC project dir: !ZLC_PS_PROJECT_DIR!
pushd "!ZLC_PS_BUILD_ROOT!" || exit /b 1
call "%ZLC_PS_VIVADO_BIN%" -mode batch -journal "!ZLC_PS_LOG_DIR!\!TCL_STEM!.jou" -log "!ZLC_PS_LOG_DIR!\!TCL_STEM!.log" -source "%DIRECT_TCL%"
set "ZLC_TCL_STATUS=!ERRORLEVEL!"
popd
exit /b !ZLC_TCL_STATUS!
