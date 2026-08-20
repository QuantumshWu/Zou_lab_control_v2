# Program the FINAL pulse-streamer bitstream and leave the JTAG-to-AXI master
# discoverable as a hw_axi core (the host then drives it with axi_session.py).
proc path_env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return [file normalize $::env($name)] }
    return $default
}
proc raw_env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
    return $default
}
proc zlc_default_project_root {script_dir} {
    if {[info exists ::env(ZLC_PS_PROJECT_ROOT)] && $::env(ZLC_PS_PROJECT_ROOT) ne ""} {
        return [file normalize $::env(ZLC_PS_PROJECT_ROOT)]
    }
    return [file normalize [file join $script_dir .. build]]
}

set script_dir [file normalize [file dirname [info script]]]
set project_root [zlc_default_project_root $script_dir]
set project_dir [path_env_or ZLC_PS_PROJECT_DIR [file join $project_root ps]]
set top zlc_pulse_streamer_top
set default_bit_path [file join $project_dir ps.runs impl_1 ${top}.bit]
set default_ltx_path [file join $project_dir ps.runs impl_1 ${top}.ltx]
set bit_path [path_env_or ZLC_PS_VIVADO_BIT [path_env_or ZLC_PS_BIT $default_bit_path]]
set ltx_path [path_env_or ZLC_PS_VIVADO_LTX [path_env_or ZLC_PS_LTX $default_ltx_path]]
set hw_server_url [raw_env_or ZLC_PS_HW_SERVER_URL [raw_env_or ZLC_HW_SERVER_URL ""]]
set expected_part [raw_env_or ZLC_PS_FPGA_PART ""]
if {$expected_part eq ""} { error "ZLC_PS_FPGA_PART is required; validate streamer_config.json before programming" }

puts "ZLC program_fpga contract: CHANNEL_COUNT=62 NUM_SLOTS=4 control=JTAG-to-AXI (final BRAM tables + streaming)"
puts "ZLC program_fpga project_dir: $project_dir"
puts "ZLC program_fpga bitstream: $bit_path"

if {![file exists $bit_path]} { error "Bitstream not found: $bit_path. Build it first (build_and_program.bat --build-only)." }

if {[llength [info commands load_features]]} { catch {load_features labtools} }
if {[llength [info commands open_hw_manager]]} {
    open_hw_manager
} elseif {[llength [info commands open_hw]]} {
    open_hw
}
if {![llength [info commands connect_hw_server]]} {
    error "Vivado hardware Tcl commands are unavailable. Install/enable Vivado LabTools."
}
if {$hw_server_url ne ""} {
    connect_hw_server -url $hw_server_url
} elseif {[catch {connect_hw_server} zlc_connect_error]} {
    puts "connect_hw_server failed: $zlc_connect_error"
    connect_hw_server
}
catch {refresh_hw_server}
set zlc_targets {}
if {[catch {set zlc_targets [get_hw_targets]} zlc_target_error]} {
    puts "get_hw_targets failed after refresh: $zlc_target_error"
    set zlc_targets {}
}
puts "Available hardware targets: $zlc_targets"
if {[llength $zlc_targets] != 1} {
    error "Expected exactly one Vivado hardware target, found [llength $zlc_targets]: $zlc_targets"
}
set zlc_target $zlc_targets
current_hw_target $zlc_target
if {[catch {open_hw_target $zlc_target} zlc_open_target_error]} {
    puts "open_hw_target failed: $zlc_open_target_error"
    catch {close_hw_target}
    if {[catch {open_hw_target -jtag_mode on $zlc_target} zlc_open_target_jtag_error]} {
        error "No FPGA device could be opened on '$zlc_target'. Check power/JTAG. Last error: $zlc_open_target_jtag_error"
    }
}
set zlc_devices [get_hw_devices]
if {[llength $zlc_devices] != 1} {
    error "Expected exactly one FPGA device, found [llength $zlc_devices]: $zlc_devices"
}
set device $zlc_devices
set actual_part [get_property PART $device]
if {$actual_part ne $expected_part} {
    error "FPGA part mismatch: streamer_config.json requires '$expected_part', hardware reports '$actual_part'"
}

set_property PROGRAM.FILE $bit_path $device
if {[file exists $ltx_path]} {
    set_property PROBES.FILE $ltx_path $device
    set_property FULL_PROBES.FILE $ltx_path $device
    puts "ZLC program_fpga probes: $ltx_path"
} else {
    puts "ZLC program_fpga: no .ltx at $ltx_path (continuing; hw_axi may still auto-detect)."
}
program_hw_devices $device
refresh_hw_device $device

set zlc_axi [get_hw_axis]
if {[llength $zlc_axi] != 1} {
    error "Programmed device must expose exactly one JTAG-to-AXI core, found [llength $zlc_axi]: $zlc_axi"
}
puts "Programmed $device"
puts "Bitstream: $bit_path"
puts "JTAG-to-AXI cores: $zlc_axi"
