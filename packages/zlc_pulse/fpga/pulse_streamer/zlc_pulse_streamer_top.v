`timescale 1ns / 1ps
// SINGLE GEOMETRY SOURCE: every parameter default below comes from zlc_geometry.vh, which is
// AUTO-GENERATED from fpga/board_config/streamer_config.json by wire.emit_geometry_vh during a
// separately approved recovery build.  Normal experiment startup never regenerates or programs
// hardware; no .v carries a hand-typed geometry literal or LAYOUT_FINGERPRINT.
`include "zlc_geometry.vh"
// =============================================================================
// zlc_pulse_streamer_top -- FINAL board top for the period-table streamer.
//
// One clean design (no variants).  JTAG-to-AXI control; the period-table rows
// and the scan window live in BLOCK RAM, the loop table and the per-signal
// delays in registers.  The host preloads two scan chunks and its sole
// observer refills released banks through the frozen mailbox while this FPGA
// remains the owner of every scan-point transition.
//
// Control path (all behind ONE proven axi_bram_ctrl, so AXI handshakes are the
// vendor IP -- only a SIMPLE combinational write decoder is custom):
//   jtag_axi_0 -> axi_bram_ctrl_0 -> {bram_addr_a, bram_we_a, ...} -> decoder,
//   by word-address region (bases == host.wire.region_bases, single source):
//     R_CTRL  regfile: scalars + COMMAND/STATUS mailbox + LOOP_TABLE_COUNT + BANK_SIZE
//             + SLOT_COUNT + CURSOR(read-back) + BANK_READY(host-written)
//     R_ROWS  period-row BRAM: ROW_WORDS 32-bit words per row on port A, one
//             whole row (ROW_PORTB_BITS) per engine read on port B
//     R_SCAN  scan BRAM (one slot vector per point), 2*BANK_SIZE deep (ping-pong)
//     R_LOOP  loop-table registers: two words per entry (first|last<<16, count)
//     R_DELAY per-signal delay registers (TTL channels, then DAC buses)
//
// SCAN BANKS: the engine plays scan point 0..N-1 through two banks and exposes
// scan_cursor plus BANK_READY/BANK*_CHUNK.  Prepare loads the first two chunks;
// the sole host observer refills each released bank without driving per-point
// timing.  A late or missing chunk holds the engine and raises UNDERFLOW, which
// invalidates the run.
//
// 1-TICK: the build tcl forces both BRAMs to READ_LATENCY_B = 2 so the engine's
// RD_LAT=2 prefetch pipeline is deterministic and back-to-back 20 ns rows play
// one per clock (see zlc_period_streamer.v and the xsim benches).
//
// Geometry localparams are computed by the SAME formulas as host.wire.region_bases
// (locked by the wire tests); the create-project tcl derives the BRAM IP geometry
// from host.wire too.
// =============================================================================

module zlc_pulse_streamer_top #(
    // Geometry defaults are macros from the generated zlc_geometry.vh (config-derived) -- see the
    // header include above; SCAN_COUNT_WIDTH is an intrinsic 32-bit counter width, not a config knob.
    parameter integer CHANNEL_COUNT = `ZLC_CHANNEL_COUNT,
    parameter integer ROW_ADDR_WIDTH = `ZLC_ROW_ADDR_WIDTH,
    parameter integer ROW_WORDS = `ZLC_ROW_WORDS,             // image words per row (power of two)
    parameter integer BANK_SIZE = `ZLC_BANK_SIZE,           // power of two; scan ping-pong bank
    parameter integer SCAN_ADDR_WIDTH = `ZLC_SCAN_ADDR_WIDTH, // = clog2(2*BANK_SIZE), wire.scan_addr_width
    parameter integer SCAN_COUNT_WIDTH = 32,                // unique-row count/cursor width; independent of bank depth
    parameter integer TICK_WIDTH = `ZLC_TICK_WIDTH,
    parameter integer NUM_SLOTS = `ZLC_NUM_SLOTS,
    parameter integer SLOT_SEL_WIDTH = `ZLC_SLOT_SEL_WIDTH,
    parameter integer BUS_COUNT = `ZLC_BUS_COUNT,
    parameter integer BUS_WIDTH = `ZLC_BUS_WIDTH,
    parameter integer MAX_LOOPS = `ZLC_MAX_LOOPS,
    parameter integer LOOP_INDEX_WIDTH = `ZLC_LOOP_INDEX_WIDTH,
    parameter integer LOOP_DEPTH = `ZLC_LOOP_DEPTH,
    parameter integer EVT_FIFO_DEPTH = `ZLC_EVT_FIFO_DEPTH,     // TTL delay event FIFO depth (per channel)
    parameter integer BUS_EVT_FIFO_DEPTH = `ZLC_BUS_EVT_FIFO_DEPTH, // per-BUS action FIFO depth
    // Host<->bitstream compatibility fingerprint exposed on CTRL word 63 -- wire.build_fingerprint of
    // THIS build's geometry (all StreamerParams geometry fields folded with LAYOUT_STRUCT_VERSION).  The
    // macro carries the config-derived value; the host connect-check verifies it.
    parameter integer LAYOUT_FINGERPRINT = `ZLC_LAYOUT_FINGERPRINT
)(
    input  wire clk,
    // UART fast-control side-channel (assign to FT2232 ch-B pins, or an external USB-UART on 2 spare
    // pins, in board.xdc).  Writes the SAME region_bases map as JTAG-to-AXI -- a byte-identical
    // transport swap for ~82 ms program apply / ~sub-ms scan-step vs ~1 s over Vivado-Tcl JTAG.
    input  wire uart_rx,
    output wire uart_tx,
    output wire [1:0] led,
    output wire cooling, output wire shutter_420, output wire repump, output wire probe,
    output wire pushout, output wire state_pre, output wire trig, output wire coil,
    output wire grey_cooling, output wire trap, output wire UV, output wire emCCD,
    output wire microwave, output wire address,
    output wire GND1, output wire pgc_1D, output wire push_shutter, output wire single_cooling_shutter,
    output wire cooling_pgc, output wire sweep_trig, output wire push_freq_switch,
    output wire pgc_1D_freq_switch, output wire GND11,
    output wire cooling_shutter, output wire GND12, output wire repump_shutter, output wire GND13,
    output wire probe_shutter, output wire GND14, output wire bias, output wire GND15,
    output wire [9:0] da_dipole, output wire da_clk0,
    output wire [9:0] da_bias_y, output wire da_clk1,
    output wire [9:0] da_bias_x, output wire da_clk2,
    output wire [9:0] da_bias_z, output wire da_clk3
);

    // Physical lane identity includes DAC data and clocks. Row masks do not.
    localparam integer TTL_CHANNEL_COUNT = CHANNEL_COUNT - BUS_COUNT * (BUS_WIDTH + 1);
    localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;       // 128
    // Port-B widths DERIVED from the geometry (== wire.build_ip_sizes): the row port is the
    // row's words, the scan port the full slot vector.  Never a bare literal.
    localparam integer ROW_PORTB_BITS = ROW_WORDS * 32;          // 128
    localparam integer SCAN_PORTB_BITS = SLOT_BITS;              // 128 = 4x32
    localparam integer SCAN_WORDS = SCAN_PORTB_BITS / 32;        // 4
    localparam integer MAX_ROWS = (1 << ROW_ADDR_WIDTH);
    localparam integer SCAN_DEPTH = 2 * BANK_SIZE;
    localparam integer LOOP_WORDS = 2;                           // == wire.StreamerParams.loop_words
    // The engine's row vector (== wire.StreamerParams.row_bits); the BRAM port pads it to ROW_PORTB_BITS.
    localparam integer BUS_ACTION_BITS = 2 + SLOT_SEL_WIDTH + BUS_WIDTH;
    localparam integer ROW_BITS = TICK_WIDTH + SLOT_SEL_WIDTH + TTL_CHANNEL_COUNT + BUS_COUNT * BUS_ACTION_BITS;

    // --- word-address region bases (== host.wire.region_bases) ---------------
    localparam integer R_CTRL_BASE = 0;
    localparam integer R_CTRL_WORDS = 64;
    localparam integer R_ROWS_BASE  = R_CTRL_BASE + R_CTRL_WORDS;
    localparam integer R_SCAN_BASE  = R_ROWS_BASE + MAX_ROWS * ROW_WORDS;
    localparam integer R_LOOP_BASE  = R_SCAN_BASE + SCAN_DEPTH * SCAN_WORDS;
    // DELAY register region: ONE 32-bit word per delay-eligible signal (channels then buses),
    // the event-scheduler delay in ticks.  128 words of headroom regardless of channel count
    // so the layout is stable across configs.
    localparam integer R_DELAY_BASE  = R_LOOP_BASE + MAX_LOOPS * LOOP_WORDS;
    localparam integer R_DELAY_WORDS = `ZLC_DELAY_REG_WORDS;   // >= TTL_CHANNEL_COUNT + BUS_COUNT
    localparam integer R_TOTAL_WORDS = R_DELAY_BASE + R_DELAY_WORDS;

    // CTRL regfile word offsets (== host.wire.CtrlWords).
    localparam integer C_COMMAND = 1;   // bit0 LOAD bit1 FIRE bit2 RESET bit3 SAFE
    localparam integer C_STATUS = 2;    // bit0 LOADED bit1 RUNNING bit2 DONE bit3 ENGINE_ERROR bit4 UNDERFLOW bit5 LINK_ERROR
    localparam integer C_PROG_COUNT = 3;
    localparam integer C_SCAN_COUNT = 4;
    localparam integer C_SCAN_ENABLE = 5;
    localparam integer C_RUN_REPEAT_COUNT = 6;
    localparam integer C_LOOP_TABLE_COUNT = 7;
    localparam integer C_BANK_SIZE = 13;
    localparam integer C_SLOT_COUNT = 14;
    localparam integer C_CURSOR = 15;       // engine -> host (cumulative row-visit ordinal)
    localparam integer C_BANK_READY = 16;   // host -> engine (bit b: bank b loaded)
    localparam integer C_BANK0_CHUNK = 17;  // host -> engine: sweep chunk resident in bank 0
    localparam integer C_BANK1_CHUNK = 18;  // host -> engine: sweep chunk resident in bank 1
    localparam integer C_SCAN_REPEAT_COUNT = 19;
    // One enable bit per DAC latch clock, not per physical lane.
    localparam integer C_CLK_ENABLE = C_SCAN_REPEAT_COUNT + 1;                  // 20, low BUS_COUNT bits
    localparam integer C_COMMAND_ID = 22, C_ACK_ID = 23, C_ACK_STATUS = 24, C_ACK_CURSOR = 25;
    reg [31:0] ack_id = 0, ack_status = 0, ack_cursor = 0;

    // engine outputs
    wire [TTL_CHANNEL_COUNT-1:0] out;
    wire [BUS_COUNT*BUS_WIDTH-1:0] zlc_bus_out;
    wire zlc_running, zlc_done, zlc_underflow, zlc_overflow, zlc_physical_active;
    wire [SCAN_COUNT_WIDTH-1:0] zlc_cursor;
    reg eng_reset = 1'b1, eng_start = 1'b0;

    // --- delays: BOTH TTL channels AND DAC buses use the 32b/word R_DELAY register region,
    // driving the per-signal event scheduler (long delays; see zlc_period_streamer).
    localparam integer TTL_DELAY_WIDTH = 32;
    localparam integer DELAY_REG_COUNT = TTL_CHANNEL_COUNT + BUS_COUNT;
    reg  [31:0] delay_reg [0:DELAY_REG_COUNT-1];
    integer dri;
    initial for (dri = 0; dri < DELAY_REG_COUNT; dri = dri + 1) delay_reg[dri] = 32'b0;
    wire [TTL_CHANNEL_COUNT*TTL_DELAY_WIDTH-1:0] delay_ticks_w;
    wire [BUS_COUNT*TTL_DELAY_WIDTH-1:0] bus_delay_ticks_w;

    // --- loop table: registers written through R_LOOP (word 0 = first | last << 16, word 1 = count)
    reg [ROW_ADDR_WIDTH-1:0] loop_first_reg [0:MAX_LOOPS-1];
    reg [ROW_ADDR_WIDTH-1:0] loop_last_reg [0:MAX_LOOPS-1];
    reg [31:0] loop_count_reg [0:MAX_LOOPS-1];
    integer lri;
    initial for (lri = 0; lri < MAX_LOOPS; lri = lri + 1) begin
        loop_first_reg[lri] = {ROW_ADDR_WIDTH{1'b0}}; loop_last_reg[lri] = {ROW_ADDR_WIDTH{1'b0}};
        loop_count_reg[lri] = 32'b0;
    end
    wire [MAX_LOOPS*ROW_ADDR_WIDTH-1:0] loop_first_w;
    wire [MAX_LOOPS*ROW_ADDR_WIDTH-1:0] loop_last_w;
    wire [MAX_LOOPS*32-1:0] loop_count_w;

    // TTLs, DAC data and DAC clocks have separate physical output owners.
    //
    // DAC LATCH PHASE (critical -- do NOT change back to plain `clk`): the clk pins are
    // the parallel-DAC latch strobes (da_clk0..3, wired here via the GUI clk button).  The
    // 40 DAC DATA bits (da_bias_*/da_dipole = zlc_bus_out) are registered on `posedge clk`,
    // so the parallel word CHANGES on the rising edge.  If the strobe were plain `clk` the
    // DAC would latch on that SAME rising edge -- coincident with the data transition AND
    // with the ~30 TTL outputs all switching at a period boundary -- so a value change is
    // captured half-old/half-new = a sporadic THIRD code (the "third DA value between two
    // edge periods" bug; a long HOLD gap only masked it by moving the DAC step off the
    // busy edge).  Driving the strobe as ~clk moves the DAC latch to the clk FALLING edge =
    // the intended centre of the 20 ns fabric data eye. `tb_da_ttl_align.v` proves
    // the RTL launch/latch ordering; actual board-level setup/hold margin remains
    // an instrumented hardware-acceptance item because the DAC I/O delays are not
    // specified by this repository.
    wire [BUS_COUNT-1:0] bus_clk_enable;
    wire [BUS_COUNT-1:0] bus_clk_final;
    wire [TTL_CHANNEL_COUNT-1:0] out_final = eng_reset ? {TTL_CHANNEL_COUNT{1'b0}} : out;
    localparam [BUS_WIDTH-1:0] BUS_SAFE_CODE = {1'b1, {(BUS_WIDTH-1){1'b0}}};
    wire [BUS_COUNT*BUS_WIDTH-1:0] bus_safe_pack = {BUS_COUNT{BUS_SAFE_CODE}};
    wire [BUS_COUNT*BUS_WIDTH-1:0] bus_out_final =
        eng_reset ? bus_safe_pack : zlc_bus_out;

    // --- JTAG-to-AXI master -> FULL AXI4 -> AXI BRAM controller ---------------
    // Full AXI4 (not Lite) so the host issues INCR burst writes (up to 256 words per
    // transaction) -> ~100x faster BRAM upload.  ID width 1; the extra burst sidebands
    // (awid/awlen/awsize/awburst/awlock/awcache/wlast/bid/... + read mirror) are wired
    // master<->slave 1:1.  m_axi_awqos/arqos are driven by the master but axi_bram_ctrl
    // has no qos/region/user ports, so those two wires are intentionally left dangling.
    wire axi_clk = clk;
    wire axi_resetn = 1'b1;
    wire [0:0]  m_axi_awid;    wire [7:0] m_axi_awlen;   wire [2:0] m_axi_awsize;
    wire [1:0]  m_axi_awburst; wire [0:0] m_axi_awlock;  wire [3:0] m_axi_awcache;  wire [3:0] m_axi_awqos;
    wire [31:0] m_axi_awaddr;  wire [2:0] m_axi_awprot;  wire m_axi_awvalid; wire m_axi_awready;
    wire [31:0] m_axi_wdata;   wire [3:0] m_axi_wstrb;   wire m_axi_wlast;   wire m_axi_wvalid;  wire m_axi_wready;
    wire [0:0]  m_axi_bid;     wire [1:0] m_axi_bresp;   wire m_axi_bvalid;        wire m_axi_bready;
    wire [0:0]  m_axi_arid;    wire [7:0] m_axi_arlen;   wire [2:0] m_axi_arsize;
    wire [1:0]  m_axi_arburst; wire [0:0] m_axi_arlock;  wire [3:0] m_axi_arcache;  wire [3:0] m_axi_arqos;
    wire [31:0] m_axi_araddr;  wire [2:0] m_axi_arprot;  wire m_axi_arvalid; wire m_axi_arready;
    wire [0:0]  m_axi_rid;     wire [31:0] m_axi_rdata;  wire [1:0] m_axi_rresp;   wire m_axi_rlast; wire m_axi_rvalid;  wire m_axi_rready;

    wire        bram_clka, bram_rsta, bram_ena;
    wire [3:0]  bram_wea;
    wire [31:0] bram_addra;          // byte address from axi_bram_ctrl
    wire [31:0] bram_dina;
    reg  [31:0] bram_douta;          // read mux back to AXI

    // --- UART fast-control bridge + write-side MUX (before the region decode) ----------------
    // The bridge decodes serial frames into (u_word_addr, u_wdata, u_we) writes to the SAME flat
    // word map.  It and JTAG-AXI are never used simultaneously (JTAG = bring-up/ILA, UART = runtime),
    // so a priority mux (UART wins when u_active) is correct + free; a UART write is byte-for-byte a
    // JTAG write to the same word (only the operands are re-sourced, decode/timing unchanged).
    wire [29:0] u_word_addr; wire [31:0] u_wdata; wire u_we, u_active, u_protocol_error;
    wire [5:0]  u_rd_word;   reg [31:0] u_rd_data;
    wire u_cmd_valid, u_cmd_reply_ready;
    wire [3:0] u_cmd_code;
    wire [7:0] u_cmd_seq;
    wire [31:0] u_cmd_id, u_cmd_runs, u_cmd_scans;
    reg u_cmd_reply_valid = 1'b0;
    reg [7:0] u_cmd_reply_seq = 0;
    reg  [3:0]  uart_por = 4'h0;                            // power-on reset, independent of eng_reset
    always @(posedge clk) if (uart_por != 4'hF) uart_por <= uart_por + 1'b1;
    wire uart_rst = (uart_por != 4'hF);
    zlc_uart_bridge #(.CLK_HZ(50_000_000), .BAUD(`ZLC_UART_BAUD), .ADDRESS_WORDS(R_TOTAL_WORDS)) zlc_uart_i (
        .clk(clk), .rst(uart_rst), .uart_rx(uart_rx), .uart_tx(uart_tx),
        .u_word_addr(u_word_addr), .u_wdata(u_wdata), .u_we(u_we), .u_active(u_active), .u_error(u_protocol_error),
        .u_rd_word(u_rd_word), .u_rd_data(u_rd_data),
        .u_cmd_valid(u_cmd_valid), .u_cmd_code(u_cmd_code), .u_cmd_seq(u_cmd_seq),
        .u_cmd_id(u_cmd_id), .u_cmd_runs(u_cmd_runs), .u_cmd_scans(u_cmd_scans),
        .u_cmd_reply_valid(u_cmd_reply_valid), .u_cmd_reply_ready(u_cmd_reply_ready),
        .u_cmd_reply_seq(u_cmd_reply_seq), .u_cmd_reply_id(ack_id),
        .u_cmd_reply_status(ack_status), .u_cmd_reply_cursor(ack_cursor)
    );
    wire        uart_sel  = u_active;
    wire [29:0] word_addr = uart_sel ? u_word_addr : bram_addra[31:2];
    wire [31:0] wdata_mux = uart_sel ? u_wdata     : bram_dina;
    wire        ena_mux   = uart_sel ? u_we        : bram_ena;
    wire [3:0]  wea_mux   = uart_sel ? (u_we ? 4'hF : 4'h0) : bram_wea;
    wire        wr        = |wea_mux;

    // region selects (combinational decode of the word address)
    wire sel_ctrl  = (word_addr >= R_CTRL_BASE)  && (word_addr < R_ROWS_BASE);
    wire sel_rows  = (word_addr >= R_ROWS_BASE)  && (word_addr < R_SCAN_BASE);
    wire sel_scan  = (word_addr >= R_SCAN_BASE)  && (word_addr < R_LOOP_BASE);
    wire sel_loop  = (word_addr >= R_LOOP_BASE)  && (word_addr < R_DELAY_BASE);
    wire sel_delay = (word_addr >= R_DELAY_BASE) && (word_addr < R_TOTAL_WORDS);
    wire [29:0] rows_word_off  = word_addr - R_ROWS_BASE[29:0];
    wire [29:0] scan_word_off  = word_addr - R_SCAN_BASE[29:0];
    wire [29:0] loop_word_off  = word_addr - R_LOOP_BASE[29:0];
    wire [29:0] delay_word_off = word_addr - R_DELAY_BASE[29:0];

    // --- CTRL regfile ---------------------------------------------------------
    reg [31:0] ctrl_reg [0:R_CTRL_WORDS-1];
    integer ci;
    initial begin for (ci = 0; ci < R_CTRL_WORDS; ci = ci + 1) ctrl_reg[ci] = 32'b0; end

    // assemble the DENSE delay-tick busses and the loop table from their registers
    genvar dw;
    assign bus_clk_enable = ctrl_reg[C_CLK_ENABLE][BUS_COUNT-1:0];
    generate
        for (dw = 0; dw < TTL_CHANNEL_COUNT; dw = dw + 1) begin : zlc_delay_reg_pack_gen
            assign delay_ticks_w[dw*TTL_DELAY_WIDTH +: TTL_DELAY_WIDTH] = delay_reg[dw];
        end
        // per-bus DAC delays ride the SAME R_DELAY region, just after the channels.
        for (dw = 0; dw < BUS_COUNT; dw = dw + 1) begin : zlc_bus_delay_reg_pack_gen
            assign bus_delay_ticks_w[dw*TTL_DELAY_WIDTH +: TTL_DELAY_WIDTH] = delay_reg[TTL_CHANNEL_COUNT + dw];
        end
        for (dw = 0; dw < MAX_LOOPS; dw = dw + 1) begin : zlc_loop_pack_gen
            assign loop_first_w[dw*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH] = loop_first_reg[dw];
            assign loop_last_w[dw*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH] = loop_last_reg[dw];
            assign loop_count_w[dw*32 +: 32] = loop_count_reg[dw];
        end
    endgenerate

    // PARKING THE DACs ON SAFE (do NOT remove this window).
    //
    // The four DAC buses are EXTERNAL PARALLEL converters: the data pins
    // only reach the converter on a rising latch strobe (da_clk0..3, the
    // clk-enabled channels; see the DAC LATCH PHASE note above).
    // `eng_reset` puts the safe code on the data pins -- bus_out_final,
    // above -- and used to hold every strobe low from the same clock
    // onwards.  So the safe code was PRESENTED and never LATCHED, and the
    // converters went on driving the last code of the run: a stopped
    // sequence left coils and modulators energised at their last edge or
    // mid-ramp value while the TTL lines went low and the log said
    // outputs=SAFE.  A finite program reaching DONE parked correctly, and
    // only because its drain keeps zlc_physical_active high long enough
    // for the strobe to run -- which is the same mechanism, by accident.
    //
    // So the strobe goes on running for a few clocks after SAFE, with the
    // safe code already on the pins, and stops once it has certainly been
    // taken.  It is NOT left free-running: stopping it while the board is
    // idle is what the gate was added for, and only the clk-enabled
    // channels are driven during the window -- every other channel is low,
    // as SAFE requires.  The counter is armed while the engine runs, so
    // the window opens on the transition into SAFE however SAFE is
    // reached, and it starts armed so that configuration itself parks the
    // converters at 0 V rather than at whatever they powered up holding.
    localparam [3:0] SAFE_LATCH_TICKS = 4'd4;
    reg [3:0] safe_latch_left = SAFE_LATCH_TICKS;
    always @(posedge clk) begin
        if (!eng_reset) begin
            safe_latch_left <= SAFE_LATCH_TICKS;
        end else if (safe_latch_left != 4'd0) begin
            safe_latch_left <= safe_latch_left - 4'd1;
        end
    end
    wire zlc_safe_latching = eng_reset && (safe_latch_left != 4'd0);

    genvar cmx;
    generate
        for (cmx = 0; cmx < BUS_COUNT; cmx = cmx + 1) begin : zlc_bus_clock_gen
            assign bus_clk_final[cmx] = bus_clk_enable[cmx]
                && (zlc_safe_latching || (!eng_reset && zlc_physical_active)) ? ~clk : 1'b0;
        end
    endgenerate

    // loader/engine-driven write-backs (separate from AXI host writes)
    reg ldr_status_we;
    reg [31:0] ldr_status_val;

    always @(posedge clk) begin
        if (ena_mux && wr && sel_ctrl) ctrl_reg[word_addr[5:0]] <= wdata_mux;
        if (u_cmd_valid && u_cmd_code == 4'd2) begin
            ctrl_reg[C_RUN_REPEAT_COUNT] <= u_cmd_runs;
            ctrl_reg[C_SCAN_REPEAT_COUNT] <= u_cmd_scans;
        end
        if (ena_mux && wr && sel_delay && (delay_word_off < DELAY_REG_COUNT))
            delay_reg[delay_word_off[6:0]] <= wdata_mux;
        if (ena_mux && wr && sel_loop && (loop_word_off < MAX_LOOPS * LOOP_WORDS)) begin
            if (loop_word_off[0]) begin
                loop_count_reg[loop_word_off[LOOP_INDEX_WIDTH:1]] <= wdata_mux;
            end else begin
                loop_first_reg[loop_word_off[LOOP_INDEX_WIDTH:1]] <= wdata_mux[ROW_ADDR_WIDTH-1:0];
                loop_last_reg[loop_word_off[LOOP_INDEX_WIDTH:1]] <= wdata_mux[16 +: ROW_ADDR_WIDTH];
            end
        end
        if (ldr_status_we) ctrl_reg[C_STATUS] <= ldr_status_val;
        ctrl_reg[C_CURSOR] <= zlc_cursor;        // engine cursor visible to host
    end

    // --- read mux back to AXI -------------------------------------------------
    // CTRL word 63 reads back the GEOMETRY FINGERPRINT (LAYOUT_FINGERPRINT = ZLC_LAYOUT_FINGERPRINT
    // from the generated zlc_geometry.vh = this build's wire.build_fingerprint; writes land in
    // ctrl_reg[63] but are never read back).  The host (build_fingerprint of ITS OWN config)
    // verifies it BEFORE writing anything layout-dependent, so a host packing for one geometry can
    // NEVER silently mis-drive a bitstream built for another.  An OLD bitstream returns its own
    // (different) fingerprint or ctrl_reg[63]=0 here, so a mismatched host refuses it.
    localparam integer C_LAYOUT_ID = 63;
    localparam [31:0] ZLC_LAYOUT_ID = LAYOUT_FINGERPRINT[31:0];   // geometry fingerprint (wire.build_fingerprint)
    always @(*) begin
        if (sel_ctrl) bram_douta = (word_addr[5:0] == C_LAYOUT_ID[5:0])
                                   ? ZLC_LAYOUT_ID : command_readback(word_addr[5:0], ctrl_reg[word_addr[5:0]], ack_id, ack_status, ack_cursor);
        else bram_douta = 32'b0;
    end

    // UART read tap: COMBINATIONAL, byte-identical to the AXI read mux above (same hardwired LAYOUT_ID
    // readback).  MUST NOT be registered: the bridge sets u_rd_word with a NON-BLOCKING assign in D_READ
    // (so u_rd_word is valid only in the NEXT state, D_RLAT) and latches u_rd_data into wbuf THAT SAME
    // D_RLAT cycle.  A registered tap adds a second cycle of latency, so the bridge would capture the
    // PREVIOUS word's value -> every UART read returns stale data (observed on hardware).
    always @(*)
        u_rd_data = (u_rd_word == C_LAYOUT_ID[5:0]) ? ZLC_LAYOUT_ID : command_readback(u_rd_word, ctrl_reg[u_rd_word], ack_id, ack_status, ack_cursor);

    function [31:0] command_readback;
        input [5:0] word_index;
        input [31:0] ordinary_value, completed_id, completed_status, completed_cursor;
        begin
            case (word_index)
                C_ACK_ID: command_readback = completed_id;
                C_ACK_STATUS: command_readback = completed_status;
                C_ACK_CURSOR: command_readback = completed_cursor;
                default: command_readback = ordinary_value;
            endcase
        end
    endfunction

    // --- ROW BRAM (port A 32b write, port B one whole row; MAX_ROWS deep) ---------
    // Forced READ_LATENCY_B = 2 by the build tcl; engine RD_LAT must match.
    wire [ROW_PORTB_BITS-1:0] row_rdata_w;
    wire [ROW_ADDR_WIDTH-1:0] row_raddr;
    blk_mem_gen_rows zlc_rows_i (
        .clka(axi_clk), .ena(ena_mux && sel_rows), .wea(wea_mux),
        .addra(rows_word_off[($clog2(MAX_ROWS*ROW_WORDS))-1:0]), .dina(wdata_mux), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web({(ROW_PORTB_BITS/8){1'b0}}),
        .addrb(row_raddr), .dinb({ROW_PORTB_BITS{1'b0}}), .doutb(row_rdata_w)
    );

    // --- SCAN BRAM (port A 32b write, port B 128b read; 2*BANK_SIZE deep) ------
    wire [SCAN_PORTB_BITS-1:0] scan_rdata_w;
    wire [SCAN_ADDR_WIDTH-1:0] scan_raddr;
    blk_mem_gen_scan zlc_scan_bram_i (
        .clka(axi_clk), .ena(ena_mux && sel_scan), .wea(wea_mux),
        .addra(scan_word_off[($clog2(SCAN_DEPTH*SCAN_WORDS))-1:0]), .dina(wdata_mux), .douta(),
        .clkb(axi_clk), .enb(1'b1), .web({(SCAN_PORTB_BITS/8){1'b0}}),
        .addrb(scan_raddr), .dinb({SCAN_PORTB_BITS{1'b0}}), .doutb(scan_rdata_w)
    );

    // --- control FSM ---------------------------------------------------------
    // LOAD marks the uploaded image resident and acknowledges LOADED: the engine reads
    // the row and scan BRAMs and the loop/delay registers directly, so there is nothing
    // to copy.  FIRE resets only runtime, lets the engine's arm refill its prefetch from
    // the resident image, and starts it.  SAFE can interrupt any state and preserves a
    // fully loaded program.  Rising-edge-detected commands.
    localparam CMD_LOAD = 4'b0001, CMD_FIRE = 4'b0010, CMD_RESET = 4'b0100, CMD_SAFE = 4'b1000;
    // STATUS bit map MUST match host.wire: LOADED=1 RUNNING=2 DONE=4
    // ENGINE_ERROR=8 UNDERFLOW=16 LINK_ERROR=32.  Underflow is bit4 (NOT bit3) so a transient
    // streaming STALL is never confused with the host's fatal ERROR bit.
    localparam [4:0] ST_LOADED = 5'd1, ST_RUNNING = 5'd2, ST_DONE = 5'd4,
                     ST_ERROR = 5'd8, ST_UNDERFLOW = 5'd16;
    localparam [5:0] ST_LINK_ERROR = 6'd32;

    reg protocol_error = 1'b0;
    // FSM-owned "engine is in its RUNNING/DONE-tracking phase" flag.  The DONE/
    // UNDERFLOW refresh is gated on THIS (not on ctrl_reg[C_STATUS], which a separate
    // block writes back one cycle late): a command clears it atomically here, so a
    // SAFE/RESET/LOAD cannot be bounced back to RUNNING by a stale-STATUS re-read.
    reg status_running = 1'b0;

    localparam [3:0] L_IDLE=0, L_LOAD=1, L_FIRE=5, L_SAFE=7, L_ARM=8, L_START_ACK=9;
    // A Fire holds the engine in reset long enough for its arm to flush and refill
    // its prefetch FIFOs from the resident image at least once (the engine flushes
    // every 2^ARM_PERIOD_BITS clocks and refills within FIFO_DEPTH + RD_LAT + 2).
    // These are fabric clocks, not a host polling gap.
    localparam integer ENGINE_ARM_PERIOD = 64;
    localparam integer ENGINE_ARM_CYCLES = 2 * ENGINE_ARM_PERIOD + 16;
    reg [3:0] lstate = L_IDLE;
    reg resident_valid = 1'b0;
    reg pending_command = 1'b0, pending_uart = 1'b0, ack_valid = 1'b0;
    reg [31:0] pending_id = 0;
    reg [7:0] pending_seq = 0;
    reg [7:0] command_wait = 0;
    reg [3:0] cmd_seen;
    initial begin cmd_seen=0; end

    wire [3:0] cmd_now = ctrl_reg[C_COMMAND][3:0];
    wire [3:0] cmd_edge = cmd_now & ~cmd_seen;
    wire command_request = u_cmd_valid || (|cmd_edge);
    wire [3:0] command_code = u_cmd_valid ? u_cmd_code : cmd_edge;
    wire [31:0] command_id = u_cmd_valid ? u_cmd_id : ctrl_reg[C_COMMAND_ID];

    task complete_command;
        input [31:0] result_status;
        input [31:0] result_cursor;
        begin
            ack_id <= pending_id; ack_status <= result_status; ack_cursor <= result_cursor;
            ack_valid <= 1'b1; pending_command <= 1'b0;
            if (pending_uart) begin
                u_cmd_reply_seq <= pending_seq;
                u_cmd_reply_valid <= 1'b1;
            end
        end
    endtask

    always @(posedge clk) begin
        ldr_status_we <= 1'b0;
        eng_start <= 1'b0;
        cmd_seen <= cmd_now;
        if (u_cmd_reply_valid && u_cmd_reply_ready) u_cmd_reply_valid <= 1'b0;
        if (u_protocol_error) protocol_error <= 1'b1;
        if (command_request && ack_valid && command_id == ack_id) begin
            // An ACK may be lost; the same execution ID never executes twice.
            if (u_cmd_valid) begin u_cmd_reply_seq <= u_cmd_seq; u_cmd_reply_valid <= 1'b1; end
        end else if (command_request && pending_command && command_id == pending_id) begin
            // A retry while a command is pending observes its original completion.
            if (u_cmd_valid) begin pending_seq <= u_cmd_seq; pending_uart <= 1'b1; end
        end else if (command_request && (command_code == CMD_SAFE || command_code == CMD_RESET)) begin
            // SAFE wins in every state.
            eng_reset <= 1'b1; status_running <= 1'b0; protocol_error <= 1'b0;
            pending_id <= command_id; pending_seq <= u_cmd_seq; pending_uart <= u_cmd_valid;
            pending_command <= 1'b1; command_wait <= 8'd4; lstate <= L_SAFE;
            if (command_code == CMD_RESET) resident_valid <= 1'b0;
        end else if (command_request && lstate == L_IDLE) begin
            pending_id <= command_id; pending_seq <= u_cmd_seq; pending_uart <= u_cmd_valid;
            pending_command <= 1'b1;
            if (command_code == CMD_LOAD) begin
                eng_reset <= 1'b1; status_running <= 1'b0; protocol_error <= 1'b0;
                resident_valid <= 1'b0;
                command_wait <= 8'd4; lstate <= L_LOAD;
            end else if (command_code == CMD_FIRE && resident_valid && !status_running && ctrl_reg[C_PROG_COUNT] != 0) begin
                eng_reset <= 1'b1; status_running <= 1'b0; protocol_error <= 1'b0;
                command_wait <= ENGINE_ARM_CYCLES; lstate <= L_ARM;
            end else begin
                ack_id <= command_id; ack_status <= {27'b0, ST_ERROR}; ack_cursor <= zlc_cursor;
                ack_valid <= 1'b1; pending_command <= 1'b0;
                if (u_cmd_valid) begin u_cmd_reply_seq <= u_cmd_seq; u_cmd_reply_valid <= 1'b1; end
            end
        end else begin
        case (lstate)
        L_IDLE: begin end
        L_SAFE: begin
            if (command_wait != 0) command_wait <= command_wait - 1'b1;
            else begin
                ldr_status_we <= 1'b1; ldr_status_val <= 0;
                complete_command(0, 0); lstate <= L_IDLE;
            end
        end
        L_LOAD: begin
            // The image is already in the BRAMs/registers; a few clocks let the last
            // host write settle, then the program is resident.
            if (command_wait != 0) command_wait <= command_wait - 1'b1;
            else begin
                ldr_status_we <= 1'b1; ldr_status_val <= {27'b0, ST_LOADED}; resident_valid <= 1'b1;
                complete_command({27'b0, ST_LOADED}, 0); lstate <= L_IDLE;
            end
        end
        L_ARM: begin
            if (command_wait != 0) command_wait <= command_wait - 1'b1;
            else lstate <= L_FIRE;
        end
        L_FIRE: begin
            eng_reset <= 1'b0;
            eng_start <= 1'b1;
            status_running <= 1'b1;
            ldr_status_we <= 1'b1; ldr_status_val <= {27'b0, ST_RUNNING};
            lstate <= L_START_ACK;
        end
        L_START_ACK: begin
            if (zlc_underflow || zlc_overflow) begin
                complete_command({27'b0, ST_ERROR} | (zlc_underflow ? {27'b0, ST_UNDERFLOW} : 32'b0), zlc_cursor);
                lstate <= L_IDLE;
            end else if (zlc_running || zlc_done) begin
                complete_command({27'b0, ST_RUNNING}, 0); lstate <= L_IDLE;
            end
        end
        default: lstate <= L_IDLE;
        endcase
        end
        // Surface DONE / UNDERFLOW while running -- but ONLY when idle and NOT
        // handling a command this cycle.  This block runs after the case and shares
        // ldr_status_val with it, so if it fired unconditionally it would OVERWRITE a
        // command-driven STATUS write (SAFE/RESET clear, LOAD's LOADED) every cycle,
        // re-asserting RUNNING forever -> the host could never clear RUNNING and the
        // next CMD_LOAD's LOADED would never stick (observed as STATUS stuck at 0x2).
        // Gating on (idle && no command edge) lets SAFE/RESET/LOAD/FIRE win their
        // cycle, while still tracking done/underflow on the quiescent run cycles.
        if ((lstate == L_IDLE) && !command_request && status_running) begin
            ldr_status_we <= 1'b1;
            ldr_status_val <= {26'b0, ((zlc_done ? 6'b0 : {1'b0, ST_RUNNING})
                              | (zlc_done ? {1'b0, ST_DONE} : 6'b0)
                              | (zlc_overflow ? {1'b0, ST_ERROR} : 6'b0)
                              | (zlc_underflow ? {1'b0, ST_UNDERFLOW} : 6'b0)
                              | (protocol_error ? ST_LINK_ERROR : 6'b0))};
            if (zlc_done) status_running <= 1'b0;   // DONE latched; stop re-asserting STATUS
        end else if ((lstate == L_IDLE) && !command_request && protocol_error) begin
            ldr_status_we <= 1'b1;
            ldr_status_val <= ctrl_reg[C_STATUS] | ST_LINK_ERROR;
        end
    end

    // --- the FINAL period-table engine ------------------------------------------
    zlc_period_streamer #(
        .CHANNEL_COUNT(TTL_CHANNEL_COUNT), .ROW_ADDR_WIDTH(ROW_ADDR_WIDTH),
        .SCAN_ADDR_WIDTH(SCAN_ADDR_WIDTH), .SCAN_COUNT_WIDTH(SCAN_COUNT_WIDTH), .BANK_SIZE(BANK_SIZE),
        .TICK_WIDTH(TICK_WIDTH), .NUM_SLOTS(NUM_SLOTS), .SLOT_SEL_WIDTH(SLOT_SEL_WIDTH),
        .BUS_COUNT(BUS_COUNT), .BUS_WIDTH(BUS_WIDTH),
        .MAX_LOOPS(MAX_LOOPS), .LOOP_INDEX_WIDTH(LOOP_INDEX_WIDTH), .LOOP_DEPTH(LOOP_DEPTH),
        // EVT_DEPTH = per-channel delay event FIFO depth (in-flight edges).  MUST match
        // evt_fifo_depth in fpga/board_config/streamer_config.json -- the host
        // validator rejects programs that would overflow this depth.
        .EVT_DEPTH(EVT_FIFO_DEPTH),
        .BUS_EVT_DEPTH(BUS_EVT_FIFO_DEPTH),
        // RD_LAT = the configured BRAM latency.  The registered address plus
        // generated memory/core output stages make issue->data RD_LAT+2 cycles.
        .RD_LAT(2), .ARM_PERIOD_BITS(6)
    ) zlc_engine_i (
        .clk(axi_clk), .reset(eng_reset), .start(eng_start),
        .prog_count(ctrl_reg[C_PROG_COUNT][ROW_ADDR_WIDTH:0]),
        .run_repeat_count(ctrl_reg[C_RUN_REPEAT_COUNT]),
        .scan_enable(ctrl_reg[C_SCAN_ENABLE][0]),
        .scan_count(ctrl_reg[C_SCAN_COUNT][SCAN_COUNT_WIDTH-1:0]),
        .scan_repeat_count(ctrl_reg[C_SCAN_REPEAT_COUNT]),
        .loop_table_count(ctrl_reg[C_LOOP_TABLE_COUNT][LOOP_INDEX_WIDTH:0]),
        .loop_first_flat(loop_first_w), .loop_last_flat(loop_last_w), .loop_count_flat(loop_count_w),
        .row_raddr(row_raddr), .row_rdata(row_rdata_w[ROW_BITS-1:0]),
        .scan_raddr(scan_raddr), .scan_rdata(scan_rdata_w),
        .bank_ready(ctrl_reg[C_BANK_READY][1:0]),
        .bank_chunk0(ctrl_reg[C_BANK0_CHUNK][SCAN_COUNT_WIDTH-1:0]),
        .bank_chunk1(ctrl_reg[C_BANK1_CHUNK][SCAN_COUNT_WIDTH-1:0]),
        .scan_cursor(zlc_cursor), .underflow(zlc_underflow),
        // OUTPUT delay event scheduler -- per-channel / per-bus delay tick counts (the engine
        // queues each output's toggles against g_time and pops them d ticks later).
        .bus_delay_ticks(bus_delay_ticks_w),
        .delay_ticks(delay_ticks_w),
        .out(out), .bus_out(zlc_bus_out), .running(zlc_running), .done(zlc_done),
        .overflow(zlc_overflow), .physical_active(zlc_physical_active)
    );

    // ---- JTAG-to-AXI + AXI BRAM controller IP --------------------------------
    jtag_axi_0 zlc_jtag_axi_i (
        .aclk(axi_clk), .aresetn(axi_resetn),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(m_axi_awlock), .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot),
        .m_axi_awqos(m_axi_awqos), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid), .m_axi_bresp(m_axi_bresp), .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(m_axi_arlock), .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
        .m_axi_arqos(m_axi_arqos), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid), .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready)
    );
    // axi_bram_ctrl in full AXI4: same wires, plus the burst sidebands.  It has no
    // qos/region/user ports, so m_axi_awqos/m_axi_arqos are NOT connected here (the
    // master drives them; they simply have no slave load).  The external BRAM port
    // (bram_*) is identical to before -- burst beats just increment bram_addra.
    axi_bram_ctrl_0 zlc_bram_ctrl_i (
        .s_axi_aclk(axi_clk), .s_axi_aresetn(axi_resetn),
        .s_axi_awid(m_axi_awid), .s_axi_awaddr(m_axi_awaddr),
        .s_axi_awlen(m_axi_awlen), .s_axi_awsize(m_axi_awsize), .s_axi_awburst(m_axi_awburst),
        .s_axi_awlock(m_axi_awlock), .s_axi_awcache(m_axi_awcache), .s_axi_awprot(m_axi_awprot),
        .s_axi_awvalid(m_axi_awvalid), .s_axi_awready(m_axi_awready),
        .s_axi_wdata(m_axi_wdata), .s_axi_wstrb(m_axi_wstrb), .s_axi_wlast(m_axi_wlast),
        .s_axi_wvalid(m_axi_wvalid), .s_axi_wready(m_axi_wready),
        .s_axi_bid(m_axi_bid), .s_axi_bresp(m_axi_bresp), .s_axi_bvalid(m_axi_bvalid), .s_axi_bready(m_axi_bready),
        .s_axi_arid(m_axi_arid), .s_axi_araddr(m_axi_araddr),
        .s_axi_arlen(m_axi_arlen), .s_axi_arsize(m_axi_arsize), .s_axi_arburst(m_axi_arburst),
        .s_axi_arlock(m_axi_arlock), .s_axi_arcache(m_axi_arcache), .s_axi_arprot(m_axi_arprot),
        .s_axi_arvalid(m_axi_arvalid), .s_axi_arready(m_axi_arready),
        .s_axi_rid(m_axi_rid), .s_axi_rdata(m_axi_rdata), .s_axi_rresp(m_axi_rresp),
        .s_axi_rlast(m_axi_rlast), .s_axi_rvalid(m_axi_rvalid), .s_axi_rready(m_axi_rready),
        .bram_rst_a(bram_rsta), .bram_clk_a(bram_clka), .bram_en_a(bram_ena),
        .bram_we_a(bram_wea), .bram_addr_a(bram_addra),
        .bram_wrdata_a(bram_dina), .bram_rddata_a(bram_douta)
    );

    // ---- LEDs + physical board map (identical to the validated board XDC) -----
    assign led[0] = zlc_running;
    assign led[1] = |out;
    assign cooling = out_final[0]; assign shutter_420 = out_final[1]; assign repump = out_final[2]; assign probe = out_final[3];
    assign pushout = out_final[4]; assign state_pre = out_final[5]; assign trig = out_final[6]; assign coil = out_final[7];
    assign grey_cooling = out_final[8]; assign trap = out_final[9]; assign UV = out_final[10]; assign emCCD = out_final[11];
    assign microwave = out_final[12]; assign address = out_final[13];
    assign cooling_shutter = out_final[14]; assign repump_shutter = out_final[15]; assign probe_shutter = out_final[16];
    assign bias = out_final[17];
    assign da_dipole[0] = bus_out_final[0]; assign da_dipole[1] = bus_out_final[1];
    assign da_dipole[2] = bus_out_final[2]; assign da_dipole[3] = bus_out_final[3];
    assign da_dipole[4] = bus_out_final[4]; assign da_dipole[5] = bus_out_final[5];
    assign da_dipole[6] = bus_out_final[6]; assign da_dipole[7] = bus_out_final[7];
    assign da_dipole[8] = bus_out_final[8]; assign da_dipole[9] = bus_out_final[9];
    assign da_clk0 = bus_clk_final[0];
    assign da_bias_y[0] = bus_out_final[10]; assign da_bias_y[1] = bus_out_final[11];
    assign da_bias_y[2] = bus_out_final[12]; assign da_bias_y[3] = bus_out_final[13];
    assign da_bias_y[4] = bus_out_final[14]; assign da_bias_y[5] = bus_out_final[15];
    assign da_bias_y[6] = bus_out_final[16]; assign da_bias_y[7] = bus_out_final[17];
    assign da_bias_y[8] = bus_out_final[18]; assign da_bias_y[9] = bus_out_final[19];
    assign da_clk1 = bus_clk_final[1];
    assign da_bias_x[0] = bus_out_final[20]; assign da_bias_x[1] = bus_out_final[21];
    assign da_bias_x[2] = bus_out_final[22]; assign da_bias_x[3] = bus_out_final[23];
    assign da_bias_x[4] = bus_out_final[24]; assign da_bias_x[5] = bus_out_final[25];
    assign da_bias_x[6] = bus_out_final[26]; assign da_bias_x[7] = bus_out_final[27];
    assign da_bias_x[8] = bus_out_final[28]; assign da_bias_x[9] = bus_out_final[29];
    assign da_clk2 = bus_clk_final[2];
    assign da_bias_z[0] = bus_out_final[30]; assign da_bias_z[1] = bus_out_final[31];
    assign da_bias_z[2] = bus_out_final[32]; assign da_bias_z[3] = bus_out_final[33];
    assign da_bias_z[4] = bus_out_final[34]; assign da_bias_z[5] = bus_out_final[35];
    assign da_bias_z[6] = bus_out_final[36]; assign da_bias_z[7] = bus_out_final[37];
    assign da_bias_z[8] = bus_out_final[38]; assign da_bias_z[9] = bus_out_final[39];
    assign da_clk3 = bus_clk_final[3];
    assign pgc_1D = out_final[18];
    assign push_shutter = out_final[19]; assign single_cooling_shutter = out_final[20];
    assign cooling_pgc = out_final[21]; assign sweep_trig = out_final[22];
    assign push_freq_switch = out_final[23]; assign pgc_1D_freq_switch = out_final[24];
    assign GND1 = 1'b0;
    assign GND11 = 1'b0; assign GND12 = 1'b0; assign GND13 = 1'b0; assign GND14 = 1'b0;
    assign GND15 = 1'b0;
endmodule
