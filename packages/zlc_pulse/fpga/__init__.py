"""The pulse streamer's FPGA assets: the frozen RTL and its simulation
benches, the Vivado build/program scripts and the board configuration.

Only what the Verilog and the Vivado flow read lives here.  The wire contract
the host speaks to this RTL (BRAM image layout, capacity solver, transports)
is owned by ``zlc_pulse.transport`` and the modules beside it.
"""
