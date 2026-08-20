proc env_or {name default} {
    if {[info exists ::env($name)]} { return $::env($name) }
    return $default
}

set hw_server_url [env_or ZLC_PS_HW_SERVER_URL [env_or ZLC_HW_SERVER_URL ""]]
set expected_part [env_or ZLC_PS_FPGA_PART ""]
if {$expected_part eq ""} { error "ZLC_PS_FPGA_PART is required; validate streamer_config.json before hardware diagnosis" }

if {[llength [info commands load_features]]} { catch {load_features labtools} }
if {[llength [info commands open_hw_manager]]} {
    open_hw_manager
} elseif {[llength [info commands open_hw]]} {
    open_hw
}
if {![llength [info commands connect_hw_server]]} {
    error "Vivado hardware Tcl commands are unavailable. Install/enable Vivado LabTools or set ZLC_PS_VIVADO_BIN to a Vivado with Hardware Manager support."
}

if {$hw_server_url ne ""} {
    connect_hw_server -url $hw_server_url
} elseif {[catch {connect_hw_server} zlc_connect_error]} {
    puts "connect_hw_server failed: $zlc_connect_error"
    connect_hw_server
}

catch {refresh_hw_server}
set targets {}
if {[catch {set targets [get_hw_targets]} err]} {
    puts "get_hw_targets failed after refresh: $err"
    set targets {}
}
puts "ZLC hardware targets: $targets"
set zlc_targets $targets
if {[llength $zlc_targets] != 1} {
    error "Expected exactly one Vivado hardware target, found [llength $zlc_targets]: $zlc_targets"
}
set target $zlc_targets
puts "ZLC opening target: $target"
current_hw_target $target
if {[catch {open_hw_target $target} err]} {
    puts "open_hw_target failed: $err"
    catch {close_hw_target}
    if {[catch {open_hw_target -jtag_mode on $target} jtag_err]} {
        error "Could not open the sole hardware target '$target': $jtag_err"
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
set line "  NAME=[get_property NAME $device]"
foreach prop {PART IDCODE PROGRAM.FILE PROBES.FILE} {
    if {![catch {get_property $prop $device} value]} { append line " $prop=$value" }
}
puts $line
