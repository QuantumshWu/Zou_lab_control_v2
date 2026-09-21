`timescale 1ns / 1ps
// Geometry parameter defaults come from zlc_geometry.vh (AUTO-GENERATED from
// fpga/board_config/streamer_config.json).  The top overrides them at the instance; these
// defaults are for standalone / testbench elaboration and the single-source contract tests.
`include "zlc_geometry.vh"
// =============================================================================
// zlc_period_streamer -- FINAL period-table pulse streamer engine.
//
// The program is a PERIOD TABLE in block RAM: one row per authored period holding
//   * how long the row lasts -- a literal tick count, or which scan slot supplies it,
//   * the TTL levels held for the whole row (one bit per TTL channel),
//   * one action per DAC bus, applied when the row is entered: hold, edge to a
//     value, or ramp from the carried level to a value over the row's whole
//     duration.  The value is a literal code or read from a scan slot.
// Brackets are a separate LOOP TABLE of (first row, last row, count), outermost
// first.  A body is stored once however many times it plays.
//
// Two prefetchers feed one executor:
//   * the ROW WALKER issues row reads in play order.  It follows the loop table
//     with a small stack (LOOP_DEPTH levels) and wraps to row 0 after the last
//     row, tagging that read FRAME_START.  Rows land in a FIFO_DEPTH-deep FIFO.
//     Every stack level is evaluated in parallel each issue (which entries start
//     here, which levels end here, which of those still has iterations left), so
//     a row that opens and closes several nested brackets costs one clock.
//   * the SCAN PREFETCHER reads scan-point slot vectors in point order through
//     the two-bank ping-pong window (bank = chunk parity ^ base, base flips by
//     the chunk-count parity at every wrap) into a VF_DEPTH-deep vector FIFO,
//     one point ahead of what the executor needs, and only from a bank the
//     host has marked ready and loaded with that chunk.
//   * the EXECUTOR counts each row's ticks down and, at a FRAME_START row,
//     decides the seam exactly like the affine engine did: another Run repeat
//     on the same point, the next scan point, the next sweep (wrap), or done.
//     A new point's vector is popped at that seam, so the seam costs nothing
//     and every row -- one tick long or a minute long -- plays gaplessly.
// Neither prefetcher depends on the executor's timing: a one-tick row simply
// pops one FIFO entry per clock, and the read pipeline (RD_LAT+2 clocks) is
// covered by the FIFO_DEPTH resident entries (LUTRAM rings; FIFO_DEPTH is a
// power of two so their pointers wrap for free).
//
// While reset is held (LOAD/SAFE) both prefetchers are flushed and refilled
// every 2^ARM_PERIOD_BITS clocks, so at FIRE they hold the program the host
// finished uploading.  The top waits longer than one period before it
// releases reset.
//
// OUTPUT DELAY -- TTL channels queue value-change events; each DAC bus queues resolved action
//   descriptors and replays its ramp stepper after one shared delay.  Both implement
//   out_delayed[t] = out_undelayed[t-d], but their storage contracts differ: TTL scales with
//   toggles in flight (<= EVT_DEPTH), DAC with actions in flight per bus (<= BUS_EVT_DEPTH).
//   Neither scales with delay length.  The 32-bit delay field is host-capped at (1<<31)-1 ticks
//   (~42.9 s at 20 ns).  d=0 is passthrough; d=1 is one register.
// =============================================================================

module zlc_period_streamer #(
    // Geometry defaults are macros from the generated zlc_geometry.vh (config-derived); the top
    // overrides them at the instance.  SCAN_COUNT_WIDTH is an intrinsic 32-bit counter width.
    parameter integer CHANNEL_COUNT = `ZLC_NUM_DELAY_CH,
    parameter integer ROW_ADDR_WIDTH = `ZLC_ROW_ADDR_WIDTH,
    parameter integer SCAN_ADDR_WIDTH = `ZLC_SCAN_ADDR_WIDTH,   // = clog2(2*BANK_SIZE)
    parameter integer SCAN_COUNT_WIDTH = 32,    // unique-row count/cursor width; independent of bank depth
    parameter integer BANK_SIZE = `ZLC_BANK_SIZE,               // power of two; points per ping-pong bank
    parameter integer TICK_WIDTH = `ZLC_TICK_WIDTH,
    parameter integer NUM_SLOTS = `ZLC_NUM_SLOTS,
    parameter integer SLOT_SEL_WIDTH = `ZLC_SLOT_SEL_WIDTH,     // 0 = literal, k = slot k-1
    parameter integer BUS_COUNT = `ZLC_BUS_COUNT,
    parameter integer BUS_WIDTH = `ZLC_BUS_WIDTH,
    parameter integer MAX_LOOPS = `ZLC_MAX_LOOPS,
    parameter integer LOOP_INDEX_WIDTH = `ZLC_LOOP_INDEX_WIDTH,
    parameter integer LOOP_DEPTH = `ZLC_LOOP_DEPTH,
    // IDLE/SAFE DAC code.  The DAC driver is bipolar OFFSET-BINARY: code 0 = NEGATIVE full
    // scale, code 2^(B-1) (=512 for 10 bits) = true 0 V.  Every "rest" value of a bus --
    // power-up, reset/CMD_SAFE, FIRE re-init, the delayed-read gate before the ring fills,
    // and after done -- uses THIS mid-scale code so an idle DAC outputs 0 V, not -FS.
    parameter integer BUS_SAFE_VALUE = (1 << (BUS_WIDTH - 1)),
    parameter integer RD_LAT = 2,               // row/scan BRAM read latency (forced by the build)
    // The generated IP model exposes data RD_LAT+2 cycles after issue: registered
    // address plus the configured memory/core output stages.  FIFO_DEPTH must be a
    // power of two (ring pointers) and at least RD_LAT + 4 (pipeline coverage).
    parameter integer FIFO_DEPTH = 8,
    parameter integer ARM_PERIOD_BITS = 6,      // flush+refill every 2^N clocks while reset holds
    // ----- TTL EVENT-SCHEDULER delay geometry ------------------------------------------
    // TTL channel delays are NOT bounded by a fixed delay-line depth: each channel schedules its
    // output TOGGLES (time + level) in a small event FIFO against a free-running global
    // tick counter, so the storage scales with the number of IN-FLIGHT TOGGLES (validated
    // <= EVT_DEPTH by the host), not with the delay length.  TTL_DELAY_WIDTH is the 32b
    // register field; the HOST caps one delay at a conservative default of (1<<31)-1 ticks
    // (~42.9 s at 20 ns; streamer_config.json ttl_delay_max_ticks).  GTIME_WIDTH bounds one
    // RUN (48b = ~65 days).
    parameter integer TTL_DELAY_WIDTH = 32,
    parameter integer EVT_DEPTH = `ZLC_EVT_FIFO_DEPTH,
    // DAC delay ACTION-descriptor FIFO depth: each BUS has ONE FIFO holding its RESOLVED
    // edge/ramp actions in flight for the delayed re-player, so the depth scales with
    // ACTIONS IN FLIGHT (host-validated <= BUS_EVT_DEPTH), NOT value-changes.
    parameter integer BUS_EVT_DEPTH = `ZLC_BUS_EVT_FIFO_DEPTH,
    parameter integer GTIME_WIDTH = 48
)(
    input  wire clk,
    input  wire reset,
    input  wire start,

    // held program scalars (top regfile)
    input  wire [ROW_ADDR_WIDTH:0] prog_count,       // rows in the table
    input  wire [31:0] run_repeat_count,             // complete Pulse executions per row; 0 = infinite
    input  wire scan_enable,
    input  wire [SCAN_COUNT_WIDTH-1:0] scan_count,   // unique rows in one sweep
    input  wire [31:0] scan_repeat_count,            // complete sweeps; 0 = infinite

    // loop table (top registers): entry i = rows loop_first[i]..loop_last[i], loop_count[i] times
    input  wire [LOOP_INDEX_WIDTH:0] loop_table_count,
    input  wire [MAX_LOOPS*ROW_ADDR_WIDTH-1:0] loop_first_flat,
    input  wire [MAX_LOOPS*ROW_ADDR_WIDTH-1:0] loop_last_flat,
    input  wire [MAX_LOOPS*32-1:0] loop_count_flat,

    // row BRAM read port (latency RD_LAT, one whole row per access)
    output reg  [ROW_ADDR_WIDTH-1:0] row_raddr,
    input  wire [TICK_WIDTH+SLOT_SEL_WIDTH+CHANNEL_COUNT+BUS_COUNT*(2+SLOT_SEL_WIDTH+BUS_WIDTH)-1:0] row_rdata,

    // scan BRAM read port (2-bank window; latency RD_LAT)
    output reg  [SCAN_ADDR_WIDTH-1:0] scan_raddr,
    input  wire [NUM_SLOTS*TICK_WIDTH-1:0] scan_rdata,

    // streaming handshake with the host
    input  wire [1:0] bank_ready,               // bit b: bank b loaded
    input  wire [SCAN_COUNT_WIDTH-1:0] bank_chunk0,  // chunk index resident in bank 0
    input  wire [SCAN_COUNT_WIDTH-1:0] bank_chunk1,  // chunk index resident in bank 1
    output reg  [SCAN_COUNT_WIDTH-1:0] scan_cursor,  // cumulative row-visit ordinal; Run repeats do not increment
    output reg  underflow,                      // a point was needed before its bank was ready

    // PHYSICAL per-bus DAC DELAY -- INSTRUCTION-LEVEL (see the g_busseg generate below): the delay
    // captures each RESOLVED ACTION the engine applies (one descriptor per ramp/edge, NOT one event
    // per Bresenham step) and a delayed re-player re-runs it d ticks later -> bus_out[t] =
    // bus_undelayed[t - d_bus].  The bits of a bus SHARE one d so the DAC value shifts coherently.
    //   * bus_delay_ticks : per-bus delay d in ticks (0 = no delay = passthrough)
    input  wire [BUS_COUNT*TTL_DELAY_WIDTH-1:0] bus_delay_ticks,

    // PHYSICAL per-channel OUTPUT DELAY -- EVENT-SCHEDULED (see the g_evtfifo generate below).
    // out_delayed[t] = out_undelayed[t - d], 0 before fire.  When the channel's undelayed bit
    // toggles at global tick t the engine queues {t + d_ch - 1, new_level} in that channel's
    // EVT_DEPTH-deep event FIFO; the free-running g_time pops it so the level appears exactly at
    // t + d_ch.  The host folds negative authored delays into a global shift G.
    //   * delay_ticks : per-channel delay d in ticks (0 = no delay), one 32b slice per channel
    input  wire [CHANNEL_COUNT*TTL_DELAY_WIDTH-1:0] delay_ticks,

    output wire [CHANNEL_COUNT-1:0] out,
    output wire [BUS_COUNT*BUS_WIDTH-1:0] bus_out,
    output reg  running = 1'b0,
    output reg  done = 1'b0,
    output reg  overflow = 1'b0,
    output wire physical_active
);

    // ----- row layout (== host wire.pack_row) --------------------------------
    localparam integer BUS_ACTION_BITS = 2 + SLOT_SEL_WIDTH + BUS_WIDTH;
    localparam integer O_DUR  = 0;
    localparam integer O_DSEL = O_DUR + TICK_WIDTH;
    localparam integer O_MASK = O_DSEL + SLOT_SEL_WIDTH;
    localparam integer O_BUS  = O_MASK + CHANNEL_COUNT;
    localparam integer ROW_BITS = O_BUS + BUS_COUNT * BUS_ACTION_BITS;
    localparam integer A_VALUE = 0;
    localparam integer A_SEL   = A_VALUE + BUS_WIDTH;
    localparam integer A_MODE  = A_SEL + SLOT_SEL_WIDTH;
    localparam [1:0] BUS_MODE_HOLD = 2'd0, BUS_MODE_EDGE = 2'd1, BUS_MODE_RAMP = 2'd2;
    localparam integer SLOT_BITS = NUM_SLOTS * TICK_WIDTH;
    localparam integer BANK_BITS = $clog2(BANK_SIZE);
    localparam integer PIPE = RD_LAT + 2;      // issue -> data-valid latency measured with real IP model
    localparam integer VF_DEPTH = FIFO_DEPTH;  // scan vectors resident ahead of the executor
    localparam integer FIFO_CNT_W = $clog2(FIFO_DEPTH + 1);
    localparam integer FIFO_PTR_W = $clog2(FIFO_DEPTH);

    // ----- executor state -----------------------------------------------------
    reg [CHANNEL_COUNT-1:0] state_mask = {CHANNEL_COUNT{1'b0}};
    reg [TICK_WIDTH-1:0] left = {TICK_WIDTH{1'b0}};          // ticks left in the current row
    reg [TICK_WIDTH-1:0] frame_tick = {TICK_WIDTH{1'b0}};    // ticks since the frame started (diagnostic)
    reg [31:0] run_repeat_count_active = 32'd1;
    reg [31:0] runs_remaining = 32'd1;
    reg scan_enable_active = 1'b0;
    reg [SLOT_BITS-1:0] slot_active = {SLOT_BITS{1'b0}};
    reg [SCAN_COUNT_WIDTH-1:0] active_scan_count = {SCAN_COUNT_WIDTH{1'b0}};
    reg [SCAN_COUNT_WIDTH-1:0] scan_point_index = {SCAN_COUNT_WIDTH{1'b0}};
    reg [31:0] scans_remaining = 32'd1;

    // ----- row prefetch (walker + FIFO) ---------------------------------------
    // A LUTRAM ring: rows land at rf_wr, the executor reads the head at rf_rd.
    (* ram_style = "distributed" *) reg [ROW_BITS:0] rf_mem [0:FIFO_DEPTH-1];   // {FRAME_START, row}
    reg [FIFO_PTR_W-1:0] rf_rd = {FIFO_PTR_W{1'b0}}, rf_wr = {FIFO_PTR_W{1'b0}};
    reg [FIFO_CNT_W-1:0] rf_nv = {FIFO_CNT_W{1'b0}};
    wire [ROW_BITS:0] rf_head = rf_mem[rf_rd];
    wire [ROW_BITS-1:0] rf_head_row = rf_head[ROW_BITS-1:0];
    wire rf_head_fs = rf_head[ROW_BITS];
    reg [PIPE-1:0] pend = {PIPE{1'b0}};           // in-flight row reads, one bit per pipeline stage
    reg [PIPE-1:0] pend_fs = {PIPE{1'b0}};        // their FRAME_START tags
    reg [ROW_ADDR_WIDTH:0] fetch_row = {(ROW_ADDR_WIDTH+1){1'b0}};   // next row to issue
    reg fetch_fs = 1'b1;                                             // it starts a frame
    reg [LOOP_INDEX_WIDTH:0] next_loop = {(LOOP_INDEX_WIDTH+1){1'b0}};   // next table entry to push
    reg [$clog2(LOOP_DEPTH+1)-1:0] stk_n = 0;      // loops on the stack
    reg [LOOP_DEPTH*LOOP_INDEX_WIDTH-1:0] stk_idx = {(LOOP_DEPTH*LOOP_INDEX_WIDTH){1'b0}};
    reg [LOOP_DEPTH*32-1:0] stk_rem = {(LOOP_DEPTH*32){1'b0}};

    // ----- scan prefetch (point walker + vector FIFO) -------------------------
    (* ram_style = "distributed" *) reg [SLOT_BITS-1:0] vf_mem [0:VF_DEPTH-1];
    reg [FIFO_PTR_W-1:0] vf_rd = {FIFO_PTR_W{1'b0}}, vf_wr = {FIFO_PTR_W{1'b0}};
    reg [FIFO_CNT_W-1:0] vf_nv = {FIFO_CNT_W{1'b0}};
    wire [SLOT_BITS-1:0] vf_head = vf_mem[vf_rd];
    reg [PIPE-1:0] vpend = {PIPE{1'b0}};
    reg [SCAN_COUNT_WIDTH-1:0] pf_point = {SCAN_COUNT_WIDTH{1'b0}};   // next point to issue
    reg pf_base = 1'b0;                                              // bank parity of the sweep being fetched

    // ----- arm (reset held): periodic flush + refill ---------------------------
    reg arm_kicked = 1'b0;
    reg [ARM_PERIOD_BITS-1:0] arm_timer = {ARM_PERIOD_BITS{1'b0}};

    // ----- bus runtime --------------------------------------------------------
    reg [BUS_WIDTH-1:0] bus_value_active [0:BUS_COUNT-1];
    integer bus_pu;
    reg bus_ramp_active [0:BUS_COUNT-1];
    reg bus_ramp_dir_up [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_left [0:BUS_COUNT-1];      // stepping ticks left in the ramp
    reg [BUS_WIDTH-1:0] bus_ramp_target [0:BUS_COUNT-1];
    // Bresenham ramp stepping: value(k) = vstart +/- floor(k*delta/span).  Per tick the
    // value moves by step (= delta/span, >1 for STEEP ramps) plus 1 on a remainder-
    // accumulator carry, so ANY slope tracks the ideal line exactly and lands on the
    // target on the last tick (no 1-LSB/tick crawl + end snap).
    reg [BUS_WIDTH:0] bus_ramp_step [0:BUS_COUNT-1];
    reg [BUS_WIDTH:0] bus_ramp_rem [0:BUS_COUNT-1];
    // The d/span divmod for a STEEP ramp is DEFERRED from the apply to the first
    // stepping tick (>= 1 full cycle later): the divider then reads REGISTERED operands
    // (rem temporarily holds d), keeping the row-decode mux off its timing path.
    reg bus_ramp_steep [0:BUS_COUNT-1];
    reg [TICK_WIDTH-1:0] bus_ramp_denom [0:BUS_COUNT-1];
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_ramp_accum [0:BUS_COUNT-1];

    initial begin
        for (bus_pu = 0; bus_pu < BUS_COUNT; bus_pu = bus_pu + 1) begin
            bus_value_active[bus_pu] = BUS_SAFE_VALUE[BUS_WIDTH-1:0];
            bus_ramp_active[bus_pu] = 1'b0;
            bus_ramp_dir_up[bus_pu] = 1'b0;
            bus_ramp_left[bus_pu] = {TICK_WIDTH{1'b0}};
            bus_ramp_target[bus_pu] = {BUS_WIDTH{1'b0}};
            bus_ramp_step[bus_pu] = {(BUS_WIDTH+1){1'b0}};
            bus_ramp_rem[bus_pu] = {(BUS_WIDTH+1){1'b0}};
            bus_ramp_steep[bus_pu] = 1'b0;
            bus_ramp_denom[bus_pu] = {TICK_WIDTH{1'b0}};
            bus_ramp_accum[bus_pu] = {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
        end
    end

    // Exact reciprocal table for the only variable division in the real-time path.
    // A STEEP ramp has 0 < span < delta < 2^BUS_WIDTH.  For S=2^(2*BUS_WIDTH),
    // magic(span)=ceil(S/span) gives
    //
    //   floor(delta * magic(span) / S) == floor(delta / span)
    //
    // over that complete input domain: the reciprocal error is < delta/S and
    // delta*span < S, so it cannot cross the next integer boundary.  The
    // remainder is then delta - quotient*span.  Vivado maps the read port to a
    // compact distributed ROM and the two products per bus to DSP48s.
    //
    // ONE lookup serves every bus and the delayed re-players too: a steep ramp's
    // divmod runs on the first stepping tick of its row, every ramp entered on a
    // tick shares that row's span, and the descriptor a delayed bus captures is
    // completed with the same quotient/remainder as it is pushed.  The magic of
    // the row entered LAST cycle is therefore the magic every divmod needs; it is
    // registered at the row load (ramp_magic_q).
    localparam integer RAMP_RECIP_BITS = 2 * BUS_WIDTH;
    localparam integer RAMP_RECIP_SIZE = (1 << BUS_WIDTH);
    localparam integer RAMP_RECIP_SCALE = (1 << RAMP_RECIP_BITS);
    (* rom_style = "distributed" *) reg [RAMP_RECIP_BITS:0] ramp_reciprocal [0:RAMP_RECIP_SIZE-1];
    integer ramp_recip_i;
    initial begin
        ramp_reciprocal[0] = {(RAMP_RECIP_BITS+1){1'b0}};
        for (ramp_recip_i = 1; ramp_recip_i < RAMP_RECIP_SIZE; ramp_recip_i = ramp_recip_i + 1)
            ramp_reciprocal[ramp_recip_i] = (RAMP_RECIP_SCALE + ramp_recip_i - 1) / ramp_recip_i;
    end
    reg [RAMP_RECIP_BITS:0] ramp_magic_q = {(RAMP_RECIP_BITS+1){1'b0}};

    // Quotient + remainder of delta/span for a STEEP ramp.  The reciprocal-ROM
    // identity above is exact for every legal BUS_WIDTH-bit operand; this is not
    // a fixed-point approximation.  Gentle ramps skip the divider entirely
    // (step=0, rem=delta, the historic 0/1-step-per-tick behaviour).
    function [2*BUS_WIDTH+1:0] zlc_bus_ramp_divmod;
        input [BUS_WIDTH:0] num;
        input [BUS_WIDTH:0] den;
        input [RAMP_RECIP_BITS:0] magic;
        (* use_dsp = "yes" *) reg [BUS_WIDTH+RAMP_RECIP_BITS:0] scaled_num;
        reg [BUS_WIDTH:0] q;
        (* use_dsp = "yes" *) reg [2*BUS_WIDTH:0] scaled_den;
        reg [BUS_WIDTH:0] r;
        begin
            scaled_num = num[BUS_WIDTH-1:0] * magic;
            q = scaled_num[BUS_WIDTH+RAMP_RECIP_BITS:RAMP_RECIP_BITS];
            scaled_den = q * den[BUS_WIDTH-1:0];
            r = num - scaled_den[BUS_WIDTH:0];
            zlc_bus_ramp_divmod = {q, r};
        end
    endfunction
    // Every bus's deferred divmod, from REGISTERED operands (rem parks d, denom the span)
    // and the shared magic: read by the live stepper on the divmod tick and by the
    // descriptor push of the same tick.
    wire [2*BUS_WIDTH+1:0] bus_qr_w [0:BUS_COUNT-1];
    genvar gqr;
    generate
    for (gqr = 0; gqr < BUS_COUNT; gqr = gqr + 1) begin : g_qr
        assign bus_qr_w[gqr] = zlc_bus_ramp_divmod(bus_ramp_rem[gqr], bus_ramp_denom[gqr][BUS_WIDTH:0], ramp_magic_q);
    end
    endgenerate

    // ----- per-bus ACTION-DESCRIPTOR delay capture (raised by zlc_bus_apply_action) --------------
    // DAC delay is INSTRUCTION-LEVEL: each RESOLVED action the engine applies is captured as ONE
    // descriptor and RE-RUN d ticks later by a per-bus delayed player (g_busseg below), so buffer
    // depth = actions-in-flight, INDEPENDENT of ramp density.  The descriptor is in the DELAYED
    // time base (emit = g_time + d = the action's shifted start) and carries the ramp RESOLVED:
    // a steep ramp's step/remainder are taken from the shared divmod as the descriptor is
    // pushed (one tick after the apply, the divmod tick), so the re-player has no divider.
    reg                   bus_seg_push  [0:BUS_COUNT-1];   // 1-cycle strobe: bus i applied an action
    reg [GTIME_WIDTH-1:0] bus_seg_emit  [0:BUS_COUNT-1];   // g_time+d at apply (delayed base)
    reg [BUS_WIDTH-1:0]   bus_seg_vstart[0:BUS_COUNT-1];   // carried resolved start value
    reg [BUS_WIDTH-1:0]   bus_seg_target[0:BUS_COUNT-1];   // resolved stop value
    reg [TICK_WIDTH-1:0]  bus_seg_denom [0:BUS_COUNT-1];   // ramp span (its length and its Bresenham denominator)
    reg [BUS_WIDTH:0]     bus_seg_step  [0:BUS_COUNT-1];   // Bresenham base step (gentle: 0)
    reg [BUS_WIDTH:0]     bus_seg_rem   [0:BUS_COUNT-1];   // Bresenham remainder (gentle: d)
    reg                   bus_seg_up    [0:BUS_COUNT-1];   // ramp direction
    reg                   bus_seg_steep [0:BUS_COUNT-1];   // push takes step/rem from the divmod instead
    reg                   bus_seg_isramp[0:BUS_COUNT-1];   // ramp vs edge/hold
    integer bus_seg_i0;
    initial for (bus_seg_i0 = 0; bus_seg_i0 < BUS_COUNT; bus_seg_i0 = bus_seg_i0 + 1) bus_seg_push[bus_seg_i0] = 1'b0;

    // ----- delay runtime (per-channel TTL EVENT SCHEDULER + per-bus DAC action delay) ------------
    // d_ch / d_bus are held CTRL (a delay is constant, never scanned), latched at FIRE.
    reg [TTL_DELAY_WIDTH-1:0]  del_ch_ticks  [0:CHANNEL_COUNT-1];  // per-channel d (0 = passthrough)
    reg [TTL_DELAY_WIDTH-1:0]  del_bus_ticks [0:BUS_COUNT-1];      // per-bus d, shared by the bus's bits
    reg [GTIME_WIDTH-1:0] g_time = {GTIME_WIDTH{1'b0}};         // free-running ticks since FIRE
    reg [CHANNEL_COUNT-1:0] prev_undelayed = {CHANNEL_COUNT{1'b0}};
    wire [CHANNEL_COUNT-1:0] evt_fifo_busy;
    wire [CHANNEL_COUNT-1:0] evt_fifo_overflow;
    wire [BUS_COUNT-1:0] bus_delay_busy;
    wire [BUS_COUNT-1:0] bus_delay_overflow;
    wire delay_runtime_busy = |evt_fifo_busy | |bus_delay_busy;
    localparam integer EVT_ADDR = $clog2(EVT_DEPTH);
    localparam integer BEVT_ADDR = $clog2(BUS_EVT_DEPTH);   // per-bus action FIFO address width
    wire [CHANNEL_COUNT-1:0] evt_out;             // scheduled levels, one owned bit per FIFO
    integer del_i;

    reg reset_meta = 1'b0, reset_sync = 1'b0;
    reg start_meta = 1'b0, start_sync = 1'b0, start_prev = 1'b0;
    reg [TICK_WIDTH+BUS_WIDTH:0] bus_accum_next;
    reg [BUS_WIDTH:0] bus_inc;        // this tick's ramp movement: step or step+1
    reg [BUS_WIDTH:0] bus_v_next;     // widened value+inc for target saturation
    reg [2*BUS_WIDTH+1:0] bus_qr;     // {step, rem} from the deferred ramp divmod

    wire start_event = start_sync && !start_prev;

    // ---- per-channel / per-bus held delay slices ----
    function [TTL_DELAY_WIDTH-1:0] zlc_delay_ch_at;
        input integer ch;
        begin zlc_delay_ch_at = delay_ticks[ch*TTL_DELAY_WIDTH +: TTL_DELAY_WIDTH]; end
    endfunction
    function [TTL_DELAY_WIDTH-1:0] zlc_delay_bus_at;
        input integer b;
        begin zlc_delay_bus_at = bus_delay_ticks[b*TTL_DELAY_WIDTH +: TTL_DELAY_WIDTH]; end
    endfunction

    // Per-channel OUTPUT-delay merge.  delayed_mask[b] marks the bits a delayed channel owns
    // (cleared from the undelayed state_mask); delayed_out[b] is the SCHEDULED level for that
    // channel: evt_out (event scheduler, d >= 2) or prev_undelayed (a register IS a 1-tick
    // delay, d == 1).  Both are 0 until the first scheduled toggle -> out[t] = in[t-d], 0
    // before t = d.  A non-delayed channel passes straight through.
    reg [CHANNEL_COUNT-1:0] delayed_mask;   // bit b set iff channel b is delayed (d_ch != 0)
    reg [CHANNEL_COUNT-1:0] delayed_out;    // delayed value per owned bit
    integer del_m;
    always @(*) begin
        delayed_mask = {CHANNEL_COUNT{1'b0}};
        delayed_out  = {CHANNEL_COUNT{1'b0}};
        for (del_m = 0; del_m < CHANNEL_COUNT; del_m = del_m + 1) begin
            if (del_ch_ticks[del_m] != {TTL_DELAY_WIDTH{1'b0}}) begin
                delayed_mask[del_m] = 1'b1;
                delayed_out[del_m] = (del_ch_ticks[del_m] == {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1})
                                     ? prev_undelayed[del_m] : evt_out[del_m];
            end
        end
    end
    assign out = (state_mask & ~delayed_mask) | delayed_out;

    // ----- per-bus ACTION-DESCRIPTOR OUTPUT delay (g_busseg) -- INSTRUCTION LEVEL ----------------
    // bus_out[t] = bus_value_active[t - d_bus].  The re-player is the live stepper's Bresenham
    // carry + saturate, fed the RESOLVED descriptor (step/remainder already split) on the
    // free-running g_time base, so its output IS the live output shifted by d, by construction.
    // Before its first descriptor emits the bus holds BUS_SAFE_VALUE (mid code = 0 V); g_time keeps
    // advancing while draining so the last d ticks drain (done-tail).  d==0 -> passthrough;
    // d==1 -> one register; d>=2 -> the delayed player.  A ramp always lasts its row's whole
    // duration and the next action on the bus arrives no earlier than the ramp's end, so the
    // descriptor needs no frame-boundary freeze.
    localparam integer O_ISRAMP = 0;
    localparam integer O_UP     = O_ISRAMP + 1;
    localparam integer O_REM    = O_UP     + 1;
    localparam integer O_STEP   = O_REM    + (BUS_WIDTH + 1);
    localparam integer O_SPAN   = O_STEP   + (BUS_WIDTH + 1);
    localparam integer O_TGT    = O_SPAN   + TICK_WIDTH;
    localparam integer O_VST    = O_TGT    + BUS_WIDTH;
    localparam integer O_EMIT   = O_VST    + BUS_WIDTH;
    localparam integer SEG_W    = O_EMIT   + GTIME_WIDTH;
    genvar gbs;
    generate
    for (gbs = 0; gbs < BUS_COUNT; gbs = gbs + 1) begin : g_busseg
        localparam [BUS_WIDTH-1:0] BSAFE = BUS_SAFE_VALUE[BUS_WIDTH-1:0];
        (* ram_style = "distributed" *) reg [SEG_W-1:0] sfifo [0:BUS_EVT_DEPTH-1];
        reg [BEVT_ADDR-1:0] swr = {BEVT_ADDR{1'b0}};
        reg [BEVT_ADDR-1:0] srd = {BEVT_ADDR{1'b0}};
        reg [BEVT_ADDR:0]   scnt = {(BEVT_ADDR+1){1'b0}};
        // delayed re-player registers (the live bus ramp stepper, on the g_time base)
        reg [BUS_WIDTH-1:0] dval  = BUS_SAFE_VALUE[BUS_WIDTH-1:0];   // delayed output (d>=2)
        reg [BUS_WIDTH-1:0] dprev = BUS_SAFE_VALUE[BUS_WIDTH-1:0];   // one-tick register (d==1)
        reg dstarted = 1'b0, dramp = 1'b0, dup = 1'b0;
        reg [BUS_WIDTH-1:0] dtarget = {BUS_WIDTH{1'b0}};
        reg [TICK_WIDTH-1:0] ddenom = {TICK_WIDTH{1'b0}};             // the ramp's span
        reg [TICK_WIDTH-1:0] dleft = {TICK_WIDTH{1'b0}};              // stepping ticks left
        reg [BUS_WIDTH:0] dstep = {(BUS_WIDTH+1){1'b0}}, drem = {(BUS_WIDTH+1){1'b0}};
        reg [TICK_WIDTH+BUS_WIDTH:0] daccum = {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
        reg [TICK_WIDTH+BUS_WIDTH:0] daccum_next;
        reg [BUS_WIDTH:0] dinc, dv_next;
        reg dpushf, dpopf;
        wire [TTL_DELAY_WIDTH-1:0] dbus = del_bus_ticks[gbs];
        wire [SEG_W-1:0] shead = sfifo[srd];                        // async-read FIFO head (LUTRAM)
        wire [GTIME_WIDTH-1:0] h_emit  = shead[O_EMIT  +: GTIME_WIDTH];
        wire [BUS_WIDTH-1:0]   h_vst   = shead[O_VST   +: BUS_WIDTH];
        wire [BUS_WIDTH-1:0]   h_tgt   = shead[O_TGT   +: BUS_WIDTH];
        wire [TICK_WIDTH-1:0]  h_span  = shead[O_SPAN  +: TICK_WIDTH];
        wire [BUS_WIDTH:0]     h_step  = shead[O_STEP  +: (BUS_WIDTH+1)];
        wire [BUS_WIDTH:0]     h_rem   = shead[O_REM   +: (BUS_WIDTH+1)];
        wire                   h_up    = shead[O_UP];
        wire                   h_ramp  = shead[O_ISRAMP];
        // the pushed descriptor: a steep ramp is completed with the divmod of this very tick
        wire [BUS_WIDTH:0] push_step = bus_seg_steep[gbs] ? bus_qr_w[gbs][2*BUS_WIDTH+1:BUS_WIDTH+1] : bus_seg_step[gbs];
        wire [BUS_WIDTH:0] push_rem  = bus_seg_steep[gbs] ? bus_qr_w[gbs][BUS_WIDTH:0] : bus_seg_rem[gbs];
        wire bus_want_pop = (scnt != {(BEVT_ADDR+1){1'b0}}) && (h_emit == g_time);
        assign bus_out[gbs*BUS_WIDTH +: BUS_WIDTH] =
            (dbus == {TTL_DELAY_WIDTH{1'b0}})                    ? bus_value_active[gbs] :   // d==0
            (dbus == {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1})       ? dprev :                    // d==1
            (dstarted ? dval : BSAFE);                                                       // d>=2
        assign bus_delay_busy[gbs] =
            (scnt != {(BEVT_ADDR+1){1'b0}})
            || (bus_out[gbs*BUS_WIDTH +: BUS_WIDTH] != BSAFE);
        assign bus_delay_overflow[gbs] =
            bus_seg_push[gbs] && (scnt == BUS_EVT_DEPTH[BEVT_ADDR:0]) && !bus_want_pop;
        always @(posedge clk) begin
            if (reset_sync) begin
                swr <= {BEVT_ADDR{1'b0}}; srd <= {BEVT_ADDR{1'b0}}; scnt <= {(BEVT_ADDR+1){1'b0}};
                dval <= BUS_SAFE_VALUE[BUS_WIDTH-1:0]; dprev <= BUS_SAFE_VALUE[BUS_WIDTH-1:0];
                dstarted <= 1'b0; dramp <= 1'b0; daccum <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end else begin
                dprev <= bus_value_active[gbs];                    // one-tick register for d==1
                // PUSH a captured descriptor (the apply strobe is from LAST cycle; emit is far
                // enough ahead for d>=2 that it cannot race the pop below).
                dpopf  = bus_want_pop;
                dpushf = bus_seg_push[gbs]
                         && ((scnt != BUS_EVT_DEPTH[BEVT_ADDR:0]) || dpopf);
                if (dpushf) begin
                    sfifo[swr] <= { bus_seg_emit[gbs], bus_seg_vstart[gbs], bus_seg_target[gbs],
                                    bus_seg_denom[gbs], push_step, push_rem,
                                    bus_seg_up[gbs], bus_seg_isramp[gbs] };
                    swr <= swr + 1'b1;
                end
                if (dpopf) begin
                    // emit this descriptor: load the re-player (value shows NEXT tick => out[t]=in[t-d])
                    dval <= h_vst; dstarted <= 1'b1; dramp <= h_ramp;
                    dtarget <= h_tgt; ddenom <= h_span; dleft <= h_span;
                    dstep <= h_step; drem <= h_rem; dup <= h_up;
                    daccum <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                    srd <= srd + 1'b1;
                end else if (dstarted && dramp) begin
                    // step exactly like the live stepper: count the span down, Bresenham carry,
                    // saturate at the target, and land on it on the last tick.
                    if (dleft == {{(TICK_WIDTH-1){1'b0}}, 1'b1}) begin
                        dval <= dtarget; dramp <= 1'b0; daccum <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                    end else begin
                        dleft <= dleft - 1'b1;
                        daccum_next = daccum + drem;
                        if (daccum_next >= ddenom) begin
                            daccum <= daccum_next - ddenom; dinc = dstep + 1'b1;
                        end else begin
                            daccum <= daccum_next; dinc = dstep;
                        end
                        if (dinc != {(BUS_WIDTH+1){1'b0}}) begin
                            if (dup) begin
                                dv_next = {1'b0, dval} + dinc;
                                dval <= (dv_next >= {1'b0, dtarget}) ? dtarget : dv_next[BUS_WIDTH-1:0];
                            end else begin
                                if ({1'b0, dval} <= {1'b0, dtarget} + dinc) dval <= dtarget;
                                else dval <= dval - dinc[BUS_WIDTH-1:0];
                            end
                        end
                    end
                end
                case ({dpushf, dpopf})
                    2'b10: scnt <= scnt + 1'b1;
                    2'b01: scnt <= scnt - 1'b1;
                    default: ;
                endcase
            end
        end
    end
    endgenerate

    // ---- row field access ----
    function [TICK_WIDTH-1:0] row_duration_of;
        input [ROW_BITS-1:0] r;
        begin row_duration_of = r[O_DUR +: TICK_WIDTH]; end
    endfunction
    function [SLOT_SEL_WIDTH-1:0] row_dsel_of;
        input [ROW_BITS-1:0] r;
        begin row_dsel_of = r[O_DSEL +: SLOT_SEL_WIDTH]; end
    endfunction
    function [CHANNEL_COUNT-1:0] row_mask_of;
        input [ROW_BITS-1:0] r;
        begin row_mask_of = r[O_MASK +: CHANNEL_COUNT]; end
    endfunction
    function [BUS_ACTION_BITS-1:0] row_action_of;
        input [ROW_BITS-1:0] r;
        input integer b;
        begin row_action_of = r[O_BUS + b*BUS_ACTION_BITS +: BUS_ACTION_BITS]; end
    endfunction
    // scan slot k-1 (sel >= 1) of a slot vector, the low TICK_WIDTH bits
    function [TICK_WIDTH-1:0] slot_value_of;
        input [SLOT_BITS-1:0] slots;
        input [SLOT_SEL_WIDTH-1:0] sel;
        begin slot_value_of = slots[(sel - 1'b1)*TICK_WIDTH +: TICK_WIDTH]; end
    endfunction
    // the ticks a row lasts for one slot vector; a duration the host never sends (0) plays one tick
    function [TICK_WIDTH-1:0] resolved_duration;
        input [ROW_BITS-1:0] r;
        input [SLOT_BITS-1:0] slots;
        reg [TICK_WIDTH-1:0] d;
        begin
            d = (row_dsel_of(r) == {SLOT_SEL_WIDTH{1'b0}}) ? row_duration_of(r)
                                                          : slot_value_of(slots, row_dsel_of(r));
            resolved_duration = (d == {TICK_WIDTH{1'b0}}) ? {{(TICK_WIDTH-1){1'b0}}, 1'b1} : d;
        end
    endfunction

    // ---- loop table access ----
    function [ROW_ADDR_WIDTH-1:0] loop_first_of;
        input integer i;
        begin loop_first_of = loop_first_flat[i*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH]; end
    endfunction
    function [ROW_ADDR_WIDTH-1:0] loop_last_of;
        input integer i;
        begin loop_last_of = loop_last_flat[i*ROW_ADDR_WIDTH +: ROW_ADDR_WIDTH]; end
    endfunction
    function [31:0] loop_count_of;
        input integer i;
        begin loop_count_of = loop_count_flat[i*32 +: 32]; end
    endfunction

    // ---- scan window: address of point idx for a sweep of bank parity base ----
    // PARAMETERIZATION GUARD: this concatenation assumes SCAN_ADDR_WIDTH == BANK_BITS+1
    // (BANK_SIZE is a power of two and the window is exactly 2 banks); the host
    // (wire.check_rtl_assumptions) rejects other geometries at pack time.
    function [SCAN_ADDR_WIDTH-1:0] scan_addr_of;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        input base;
        begin scan_addr_of = {idx[BANK_BITS] ^ base, idx[BANK_BITS-1:0]}; end
    endfunction
    // bank b is usable for point idx only if it is armed AND actually holds idx's chunk
    // (host writes bank_chunk{0,1} when it loads a chunk): a late refill makes the
    // prefetcher WAIT, never read a stale point.
    function scan_point_resident;
        input [SCAN_COUNT_WIDTH-1:0] idx;
        input base;
        reg b;
        begin
            b = idx[BANK_BITS] ^ base;
            scan_point_resident = bank_ready[b] && ((b ? bank_chunk1 : bank_chunk0) == (idx >> BANK_BITS));
        end
    endfunction
    // n_chunks = ceil(N / BANK_SIZE); STREAMED iff > 2 banks (else resident: never overwritten).
    // The bank parity flips by n_chunks parity ONLY when streamed; resident scans keep base = 0,
    // so chunk 0 always lands in the bank the host fed it one-ahead (cyclic ping-pong).
    wire [SCAN_COUNT_WIDTH-1:0] scan_n_chunks =
        (scan_count + (BANK_SIZE[SCAN_COUNT_WIDTH-1:0] - 1'b1)) >> BANK_BITS;
    wire scan_streamed    = scan_n_chunks > {{(SCAN_COUNT_WIDTH-2){1'b0}}, 2'd2};
    wire scan_wrap_toggle = scan_streamed ? scan_n_chunks[0] : 1'b0;
    wire scan_active_at_start = scan_enable && (scan_count != {SCAN_COUNT_WIDTH{1'b0}});

    // The three outer decisions are independent of the loop table.  Run repeats
    // never move the row cursor; only a completed row can advance or wrap the scan table.
    wire run_repeats_again = runs_remaining != 32'd1;
    wire scan_point_after_current = scan_enable_active
                                    && ((scan_point_index + 1'b1) < active_scan_count);
    wire scan_sweeps_again = scans_remaining != 32'd1;

    task zlc_bus_clear_runtime;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                bus_value_active[i] <= BUS_SAFE_VALUE[BUS_WIDTH-1:0];   // idle DAC = mid-scale = 0 V
                bus_ramp_active[i] <= 1'b0; bus_ramp_dir_up[i] <= 1'b0;
                bus_ramp_left[i] <= {TICK_WIDTH{1'b0}};
                bus_ramp_target[i] <= {BUS_WIDTH{1'b0}}; bus_ramp_step[i] <= {(BUS_WIDTH+1){1'b0}};
                bus_ramp_rem[i] <= {(BUS_WIDTH+1){1'b0}}; bus_ramp_steep[i] <= 1'b0;
                bus_ramp_denom[i] <= {TICK_WIDTH{1'b0}}; bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
            end
        end
    endtask

    // At DONE the undelayed bus snaps to BUS_SAFE_VALUE (zlc_bus_clear_runtime).  For a DELAYED bus
    // that snap must reach the output d ticks LATER (out[t]=in[t-d]); the delayed re-player only sees
    // actions the engine APPLIED during RUN, so capture the terminal SAFE transition as a SAFE-HOLD
    // edge descriptor (emit=g_time+d, isramp=0) -> the re-player drains to SAFE d ticks later,
    // exactly like d==0 (bus_value_active) and d==1 (the dprev register) already do.
    task zlc_bus_capture_safe_hold;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                if (del_bus_ticks[i] > {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1}) begin
                    bus_seg_push[i]   <= 1'b1;
                    bus_seg_emit[i]   <= g_time + {{(GTIME_WIDTH-TTL_DELAY_WIDTH){1'b0}}, del_bus_ticks[i]};
                    bus_seg_vstart[i] <= BUS_SAFE_VALUE[BUS_WIDTH-1:0];   bus_seg_target[i] <= BUS_SAFE_VALUE[BUS_WIDTH-1:0];
                    bus_seg_denom[i]  <= {TICK_WIDTH{1'b0}};    bus_seg_step[i] <= {(BUS_WIDTH+1){1'b0}};
                    bus_seg_rem[i]    <= {(BUS_WIDTH+1){1'b0}}; bus_seg_steep[i] <= 1'b0;
                    bus_seg_up[i]     <= 1'b1;     bus_seg_isramp[i] <= 1'b0;
                end
            end
        end
    endtask

    task zlc_delay_clear_runtime;
        integer i;
        begin
            for (i = 0; i < CHANNEL_COUNT; i = i + 1) del_ch_ticks[i] <= {TTL_DELAY_WIDTH{1'b0}};
            for (i = 0; i < BUS_COUNT; i = i + 1) del_bus_ticks[i] <= {TTL_DELAY_WIDTH{1'b0}};
        end
    endtask


    // Apply one row's action to bus i as the row is entered.  A ramp starts from the
    // level the bus holds when the row is entered -- the target of a ramp still
    // finishing on this very tick, otherwise the live value -- so a ramp carries
    // across periods, loop rewinds and scan points alike (#ramp-carry).
    task zlc_bus_apply_action;
        input integer i;
        input [BUS_ACTION_BITS-1:0] action;
        input [SLOT_BITS-1:0] slot_vec;
        input [TICK_WIDTH-1:0] span;
        reg [1:0] mode;
        reg [SLOT_SEL_WIDTH-1:0] sel;
        reg [BUS_WIDTH-1:0] vstart, vstop;
        reg [BUS_WIDTH:0] d;
        reg [TTL_DELAY_WIDTH-1:0] dly_bus;
        reg [TICK_WIDTH-1:0] resolved;
        begin
            mode = action[A_MODE +: 2];
            sel = action[A_SEL +: SLOT_SEL_WIDTH];
            if (mode != BUS_MODE_HOLD) begin
                // FIRE-safe delay: read the STABLE held CTRL delay slice, NOT the per-run latch
                // del_bus_ticks[i], which is loaded by NBA in the SAME cycle a tick-0 action runs.
                dly_bus = zlc_delay_bus_at(i);
                if (sel == {SLOT_SEL_WIDTH{1'b0}}) vstop = action[A_VALUE +: BUS_WIDTH];
                else begin resolved = slot_value_of(slot_vec, sel); vstop = resolved[BUS_WIDTH-1:0]; end
                vstart = bus_ramp_active[i] ? bus_ramp_target[i] : bus_value_active[i];
                if (mode == BUS_MODE_RAMP && span != {TICK_WIDTH{1'b0}}) begin
                    if (vstop >= vstart) begin bus_ramp_dir_up[i] <= 1'b1; d = vstop - vstart; end
                    else begin bus_ramp_dir_up[i] <= 1'b0; d = vstart - vstop; end
                    // Bresenham split: per-tick base step = d/span, remainder feeds the carry
                    // accumulator.  GENTLE (d <= span) is final as-is; STEEP defers the d/span
                    // divmod to the first stepping tick (see bus_ramp_steep), with rem
                    // temporarily holding d.  Steep => span < d <= 2^BUS_WIDTH-1, so span
                    // fits the divider's BUS_WIDTH+1 bits.
                    bus_ramp_step[i] <= {(BUS_WIDTH+1){1'b0}};
                    bus_ramp_rem[i] <= d;
                    bus_ramp_steep[i] <= (span < d);
                    bus_value_active[i] <= vstart; bus_ramp_active[i] <= 1'b1;
                    bus_ramp_left[i] <= span;
                    bus_ramp_target[i] <= vstop; bus_ramp_denom[i] <= span;
                    bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                    if (dly_bus > {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1}) begin
                        // capture the ramp for the delayed re-player, in the delayed g_time base:
                        // emit = g_time + d (== the shifted row start); it lasts span ticks.
                        bus_seg_push[i]   <= 1'b1;
                        bus_seg_emit[i]   <= g_time + {{(GTIME_WIDTH-TTL_DELAY_WIDTH){1'b0}}, dly_bus};
                        bus_seg_vstart[i] <= vstart;   bus_seg_target[i] <= vstop;
                        bus_seg_denom[i]  <= span;      bus_seg_step[i]   <= {(BUS_WIDTH+1){1'b0}};
                        bus_seg_rem[i]    <= d;         bus_seg_steep[i]  <= (span < d);
                        bus_seg_up[i]     <= (vstop >= vstart);   bus_seg_isramp[i] <= 1'b1;
                    end
                end else begin
                    bus_value_active[i] <= vstop; bus_ramp_active[i] <= 1'b0;
                    bus_ramp_left[i] <= {TICK_WIDTH{1'b0}};
                    bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                    if (dly_bus > {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1}) begin
                        // capture an edge: a constant value, no ramp
                        bus_seg_push[i]   <= 1'b1;
                        bus_seg_emit[i]   <= g_time + {{(GTIME_WIDTH-TTL_DELAY_WIDTH){1'b0}}, dly_bus};
                        bus_seg_vstart[i] <= vstop;   bus_seg_target[i] <= vstop;
                        bus_seg_denom[i]  <= {TICK_WIDTH{1'b0}};    bus_seg_step[i] <= {(BUS_WIDTH+1){1'b0}};
                        bus_seg_rem[i]    <= {(BUS_WIDTH+1){1'b0}}; bus_seg_steep[i] <= 1'b0;
                        bus_seg_up[i]     <= 1'b1;     bus_seg_isramp[i] <= 1'b0;
                    end
                end
            end
        end
    endtask

    // One running tick of every live ramp: move by the Bresenham increment and land
    // on the target on the last stepping tick.  Called BEFORE any row load in the
    // same cycle, so a new action's non-blocking writes win over the final step.
    task zlc_bus_step_ramps;
        integer i;
        begin
            for (i = 0; i < BUS_COUNT; i = i + 1) begin
                if (bus_ramp_active[i] && bus_ramp_left[i] != {TICK_WIDTH{1'b0}}) begin
                    bus_ramp_left[i] <= bus_ramp_left[i] - 1'b1;
                    if (bus_ramp_left[i] == {{(TICK_WIDTH-1){1'b0}}, 1'b1}) begin
                        bus_value_active[i] <= bus_ramp_target[i];
                        bus_ramp_active[i] <= 1'b0;
                        bus_ramp_accum[i] <= {(TICK_WIDTH+BUS_WIDTH+1){1'b0}};
                    end else begin
                        if (bus_ramp_steep[i]) begin
                            // First stepping tick of a STEEP ramp: split d (parked in rem)
                            // into step + remainder from REGISTERED operands.  accum is
                            // still 0 and rem < span by construction, so this tick can
                            // never carry: inc is exactly the new step.
                            bus_qr = bus_qr_w[i];
                            bus_ramp_step[i] <= bus_qr[2*BUS_WIDTH+1:BUS_WIDTH+1];
                            bus_ramp_rem[i] <= bus_qr[BUS_WIDTH:0];
                            bus_ramp_steep[i] <= 1'b0;
                            bus_ramp_accum[i] <= {{(TICK_WIDTH){1'b0}}, bus_qr[BUS_WIDTH:0]};
                            bus_inc = bus_qr[2*BUS_WIDTH+1:BUS_WIDTH+1];
                        end else begin
                            bus_accum_next = bus_ramp_accum[i] + bus_ramp_rem[i];
                            if (bus_accum_next >= bus_ramp_denom[i]) begin
                                bus_ramp_accum[i] <= bus_accum_next - bus_ramp_denom[i];
                                bus_inc = bus_ramp_step[i] + 1'b1;
                            end else begin
                                bus_ramp_accum[i] <= bus_accum_next;
                                bus_inc = bus_ramp_step[i];
                            end
                        end
                        // Move by the full Bresenham increment, saturating AT the target
                        // (widened compares cannot overflow: value+inc <= 2^(W+1)-1).
                        if (bus_inc != {(BUS_WIDTH+1){1'b0}}) begin
                            if (bus_ramp_dir_up[i]) begin
                                bus_v_next = {1'b0, bus_value_active[i]} + bus_inc;
                                bus_value_active[i] <= (bus_v_next >= {1'b0, bus_ramp_target[i]})
                                                       ? bus_ramp_target[i] : bus_v_next[BUS_WIDTH-1:0];
                            end else begin
                                if ({1'b0, bus_value_active[i]} <= {1'b0, bus_ramp_target[i]} + bus_inc)
                                    bus_value_active[i] <= bus_ramp_target[i];
                                else
                                    bus_value_active[i] <= bus_value_active[i] - bus_inc[BUS_WIDTH-1:0];
                            end
                        end
                    end
                end
            end
        end
    endtask

    // Enter the row at the FIFO head with the given slot vector: levels, duration, DAC actions.
    task zlc_load_row;
        input [ROW_BITS-1:0] r;
        input [SLOT_BITS-1:0] slot_vec;
        integer i;
        reg [TICK_WIDTH-1:0] span;
        begin
            span = resolved_duration(r, slot_vec);
            state_mask <= row_mask_of(r);
            left <= span;
            ramp_magic_q <= ramp_reciprocal[span[BUS_WIDTH-1:0]];   // a steep ramp's span is < 2^BUS_WIDTH
            for (i = 0; i < BUS_COUNT; i = i + 1)
                zlc_bus_apply_action(i, row_action_of(r, i), slot_vec, span);
        end
    endtask

    // ---- one FIFO pop: the ring head moves on ----
    task zlc_row_fifo_pop;
        begin rf_rd <= rf_rd + 1'b1; end
    endtask
    task zlc_vec_fifo_pop;
        begin vf_rd <= vf_rd + 1'b1; end
    endtask

    // ----- executor + prefetchers ---------------------------------------------
    reg row_pop, vec_pop, flush;
    reg [FIFO_CNT_W-1:0] rf_after_pop, vf_after_pop;
    reg [FIFO_CNT_W-1:0] rf_inflight, vf_inflight;
    reg row_issue, vec_issue, row_landed, vec_landed;
    reg [SLOT_BITS-1:0] load_slots;
    integer pk, ik;
    // walker temporaries: one issue looks at every stack level in parallel
    reg [ROW_ADDR_WIDTH-1:0] wrow;
    reg [LOOP_DEPTH-1:0] w_hit, w_new, w_fin, w_rew;
    reg [$clog2(LOOP_DEPTH+1)-1:0] w_nstart, w_ntop, w_nfin, w_rewlvl;
    reg w_rewind;
    reg [LOOP_INDEX_WIDTH:0] w_cand;
    reg [LOOP_INDEX_WIDTH-1:0] w_idx [0:LOOP_DEPTH-1];
    reg [31:0] w_rem [0:LOOP_DEPTH-1];
    integer wl;
    reg draining = 1'b0;       // logical program ended; physical delayed tail still owns pins
    reg [1:0] drain_settle = 2'd0; // covers one-tick TTL/DAC registers before DONE
    assign physical_active = running || draining;

    always @(posedge clk) begin
        reset_meta <= reset; reset_sync <= reset_meta;
        start_meta <= start; start_sync <= start_meta; start_prev <= start_sync;

        // DAC action-delay capture strobe: default LOW EVERY cycle so it is a clean 1-cycle pulse
        // regardless of which path runs (zlc_bus_apply_action during RUN, or the done-tail SAFE
        // hold below, both re-assert it LATER in this block -> the last NBA wins).
        for (bus_seg_i0 = 0; bus_seg_i0 < BUS_COUNT; bus_seg_i0 = bus_seg_i0 + 1) bus_seg_push[bus_seg_i0] <= 1'b0;

        row_pop = 1'b0; vec_pop = 1'b0; flush = 1'b0;
        load_slots = slot_active;

        if (reset_sync) begin
            running <= 1'b0; done <= 1'b0; underflow <= 1'b0; overflow <= 1'b0; draining <= 1'b0; drain_settle <= 2'd0;
            state_mask <= {CHANNEL_COUNT{1'b0}};
            left <= {TICK_WIDTH{1'b0}};
            frame_tick <= {TICK_WIDTH{1'b0}};
            scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
            zlc_bus_clear_runtime();
            zlc_delay_clear_runtime();
            // --- periodic flush + refill while reset is held (see the header) ---
            // The host uploads the table WHILE the engine is in reset (CMD_SAFE/CMD_LOAD) and
            // releases reset on CMD_FIRE, so the FIFOs must keep re-reading the most recent
            // table.  One period is 2^ARM_PERIOD_BITS clocks; a refill takes ~FIFO_DEPTH+PIPE.
            arm_timer <= arm_timer + 1'b1;
            if (!arm_kicked || arm_timer == {ARM_PERIOD_BITS{1'b1}}) begin
                arm_kicked <= 1'b1; arm_timer <= {ARM_PERIOD_BITS{1'b0}};
                flush = 1'b1;
            end
        end else begin
            arm_kicked <= 1'b0;
            if (start_event && !running) begin
                // FIRE: latch the run's scalars and the delay amounts from the held CTRL words,
                // then enter row 0 with scan point 0 (both are resident in the FIFOs after arm).
                overflow <= 1'b0; draining <= 1'b0; drain_settle <= 2'd0;
                run_repeat_count_active <= run_repeat_count;
                runs_remaining <= run_repeat_count;
                scan_enable_active <= scan_active_at_start; active_scan_count <= scan_count;
                scans_remaining <= scan_repeat_count;
                scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}};
                scan_cursor <= {SCAN_COUNT_WIDTH{1'b0}};
                frame_tick <= {TICK_WIDTH{1'b0}};
                for (del_i = 0; del_i < CHANNEL_COUNT; del_i = del_i + 1)
                    del_ch_ticks[del_i] <= zlc_delay_ch_at(del_i);
                for (del_i = 0; del_i < BUS_COUNT; del_i = del_i + 1)
                    del_bus_ticks[del_i] <= zlc_delay_bus_at(del_i);
                if (prog_count == {(ROW_ADDR_WIDTH+1){1'b0}}) begin
                    done <= 1'b1;
                end else if (scan_active_at_start && vf_nv == {FIFO_CNT_W{1'b0}}) begin
                    underflow <= 1'b1;           // point 0 is not resident: refuse to start
                end else if (rf_nv == {FIFO_CNT_W{1'b0}}) begin
                    underflow <= 1'b1;           // row 0 is not resident (cannot happen after arm)
                end else begin
                    running <= 1'b1;
                    if (scan_active_at_start) begin
                        load_slots = vf_head; slot_active <= vf_head; vec_pop = 1'b1;
                    end else begin
                        load_slots = {SLOT_BITS{1'b0}}; slot_active <= {SLOT_BITS{1'b0}};
                    end
                    zlc_load_row(rf_head_row, load_slots); row_pop = 1'b1;
                end
            end else if (running) begin
                zlc_bus_step_ramps();
                frame_tick <= frame_tick + 1'b1;
                if (left != {{(TICK_WIDTH-1){1'b0}}, 1'b1}) begin
                    left <= left - 1'b1;
                end else if (rf_nv == {FIFO_CNT_W{1'b0}}) begin
                    underflow <= 1'b1;           // the next row is not resident: hold (cannot happen)
                end else if (!rf_head_fs) begin
                    zlc_load_row(rf_head_row, slot_active); row_pop = 1'b1;
                end else if (run_repeats_again) begin
                    // another Pulse on the same point
                    if (runs_remaining != 32'd0) runs_remaining <= runs_remaining - 1'b1;
                    frame_tick <= {TICK_WIDTH{1'b0}};
                    zlc_load_row(rf_head_row, slot_active); row_pop = 1'b1;
                end else if (scan_point_after_current) begin
                    if (vf_nv == {FIFO_CNT_W{1'b0}}) begin
                        underflow <= 1'b1;       // STALL: the next point's bank is not (yet) resident
                    end else begin
                        scan_point_index <= scan_point_index + 1'b1;
                        scan_cursor <= scan_cursor + 1'b1;
                        runs_remaining <= run_repeat_count_active;
                        frame_tick <= {TICK_WIDTH{1'b0}};
                        load_slots = vf_head; slot_active <= vf_head; vec_pop = 1'b1;
                        zlc_load_row(rf_head_row, load_slots); row_pop = 1'b1;
                    end
                end else if (scan_sweeps_again) begin
                    // CYCLIC re-sweep: point 0 of the next sweep is just the next vector the
                    // prefetcher fetched (its bank parity already flipped at the wrap).
                    if (scan_enable_active && vf_nv == {FIFO_CNT_W{1'b0}}) begin
                        underflow <= 1'b1;
                    end else begin
                        scan_point_index <= {SCAN_COUNT_WIDTH{1'b0}};
                        scan_cursor <= scan_cursor + 1'b1;
                        if (scans_remaining != 32'd0) scans_remaining <= scans_remaining - 1'b1;
                        runs_remaining <= run_repeat_count_active;
                        frame_tick <= {TICK_WIDTH{1'b0}};
                        if (scan_enable_active) begin
                            load_slots = vf_head; slot_active <= vf_head; vec_pop = 1'b1;
                        end
                        zlc_load_row(rf_head_row, load_slots); row_pop = 1'b1;
                    end
                end else begin
                    running <= 1'b0; done <= 1'b0; draining <= 1'b1; drain_settle <= 2'd2;
                    state_mask <= {CHANNEL_COUNT{1'b0}};
                    zlc_bus_clear_runtime();          // undelayed bus -> SAFE now
                    zlc_bus_capture_safe_hold();       // delayed buses drain to SAFE d ticks later
                end
            end else if (draining) begin
                // Public DONE means physical completion: every delayed TTL toggle,
                // delayed DAC action and one-tick register has reached its safe
                // value.  Until then the physical scheduler keeps advancing.
                if (drain_settle != 0) begin
                    drain_settle <= drain_settle - 1'b1;
                end else if (!delay_runtime_busy) begin
                    draining <= 1'b0;
                    done <= 1'b1;
                end
            end
        end

        // ================= ROW PREFETCH: walker + FIFO =================
        if (row_pop) zlc_row_fifo_pop();
        rf_after_pop = row_pop ? (rf_nv - 1'b1) : rf_nv;
        row_landed = pend[PIPE-1];
        rf_inflight = {FIFO_CNT_W{1'b0}};
        for (pk = 0; pk < PIPE; pk = pk + 1) rf_inflight = rf_inflight + {{(FIFO_CNT_W-1){1'b0}}, pend[pk]};
        if (flush) begin
            rf_nv <= {FIFO_CNT_W{1'b0}}; pend <= {PIPE{1'b0}}; pend_fs <= {PIPE{1'b0}};
            rf_rd <= {FIFO_PTR_W{1'b0}}; rf_wr <= {FIFO_PTR_W{1'b0}};
            fetch_row <= {(ROW_ADDR_WIDTH+1){1'b0}}; fetch_fs <= 1'b1;
            next_loop <= {(LOOP_INDEX_WIDTH+1){1'b0}}; stk_n <= 0;
            row_raddr <= {ROW_ADDR_WIDTH{1'b0}};
        end else begin
            if (row_landed) begin
                rf_mem[rf_wr] <= {pend_fs[PIPE-1], row_rdata[ROW_BITS-1:0]};
                rf_wr <= rf_wr + 1'b1;
                rf_nv <= rf_after_pop + 1'b1;
            end else begin
                rf_nv <= rf_after_pop;
            end
            // issue a read iff every read the pipeline OWNS (resident + in flight + this
            // one) has a landing slot
            row_issue = ((rf_after_pop + rf_inflight) < FIFO_DEPTH[FIFO_CNT_W-1:0])
                        && (prog_count != {(ROW_ADDR_WIDTH+1){1'b0}});
            pend <= {pend[PIPE-2:0], row_issue};
            pend_fs <= {pend_fs[PIPE-2:0], fetch_fs};
            if (row_issue) begin
                row_raddr <= fetch_row[ROW_ADDR_WIDTH-1:0];
                // ---- the loop walker: where the row after fetch_row is ----
                wrow = fetch_row[ROW_ADDR_WIDTH-1:0];
                // (1) the table entries starting here are the next ones in table order (the host
                //     stores loops outermost first, start ascending); push each that has a level
                for (wl = 0; wl < LOOP_DEPTH; wl = wl + 1) begin
                    w_cand = next_loop + wl;
                    w_hit[wl] = (w_cand < loop_table_count) && ((stk_n + wl) < LOOP_DEPTH)
                                && (loop_first_of(w_cand) == wrow);
                end
                w_nstart = 0;
                for (wl = 0; wl < LOOP_DEPTH; wl = wl + 1)
                    if (w_nstart == wl && w_hit[wl]) w_nstart = wl + 1;   // leading hits only
                w_ntop = stk_n + w_nstart;
                // (2) the stack after those pushes, level by level: kept entries stay, pushed ones
                //     take their table entry with its full count.  The loops ending here are the
                //     innermost active ones, i.e. the top levels: one on its last iteration is
                //     finished, one with iterations left rewinds.
                for (wl = 0; wl < LOOP_DEPTH; wl = wl + 1) begin
                    w_new[wl] = (wl >= stk_n) && (wl < w_ntop);
                    w_cand = next_loop + (wl - stk_n);
                    w_idx[wl] = w_new[wl] ? w_cand[LOOP_INDEX_WIDTH-1:0]
                                          : stk_idx[wl*LOOP_INDEX_WIDTH +: LOOP_INDEX_WIDTH];
                    w_rem[wl] = w_new[wl] ? loop_count_of(w_idx[wl]) : stk_rem[wl*32 +: 32];
                    w_fin[wl] = (wl < w_ntop) && (loop_last_of(w_idx[wl]) == wrow) && (w_rem[wl] <= 32'd1);
                    w_rew[wl] = (wl < w_ntop) && (loop_last_of(w_idx[wl]) == wrow) && (w_rem[wl] > 32'd1);
                end
                // (3) pop the finished loops from the top down; the first ending loop below them
                //     that still has iterations left rewinds to its first row
                w_nfin = 0;
                for (wl = LOOP_DEPTH - 1; wl >= 0; wl = wl - 1)
                    if ((wl + 1 + w_nfin) == w_ntop && w_fin[wl]) w_nfin = w_nfin + 1;
                w_rewlvl = w_ntop - w_nfin - 1;
                w_rewind = (w_nfin < w_ntop) && w_rew[w_rewlvl];
                if (w_rewind) begin
                    fetch_row <= {1'b0, loop_first_of(w_idx[w_rewlvl])};
                    fetch_fs <= 1'b0;
                    next_loop <= {1'b0, w_idx[w_rewlvl]} + 1'b1;
                    stk_n <= w_ntop - w_nfin;
                end else if ((fetch_row + 1'b1) >= prog_count) begin
                    // wrap: the next read starts a new frame with an empty stack
                    fetch_row <= {(ROW_ADDR_WIDTH+1){1'b0}}; fetch_fs <= 1'b1;
                    stk_n <= 0; next_loop <= {(LOOP_INDEX_WIDTH+1){1'b0}};
                end else begin
                    fetch_row <= fetch_row + 1'b1; fetch_fs <= 1'b0;
                    next_loop <= next_loop + w_nstart;
                    stk_n <= w_ntop - w_nfin;
                end
                // pushed levels take their entry; the rewinding level counts one iteration down
                // (a one-row loop is pushed and rewound in the same issue); popped levels need
                // no clearing, stk_n bounds the stack
                for (wl = 0; wl < LOOP_DEPTH; wl = wl + 1) begin
                    if (w_new[wl]) stk_idx[wl*LOOP_INDEX_WIDTH +: LOOP_INDEX_WIDTH] <= w_idx[wl];
                    if (w_rewind && wl == w_rewlvl) stk_rem[wl*32 +: 32] <= w_rem[wl] - 1'b1;
                    else if (w_new[wl]) stk_rem[wl*32 +: 32] <= w_rem[wl];
                end
            end
        end

        // ================= SCAN PREFETCH: point walker + vector FIFO =================
        if (vec_pop) zlc_vec_fifo_pop();
        vf_after_pop = vec_pop ? (vf_nv - 1'b1) : vf_nv;
        vec_landed = vpend[PIPE-1];
        vf_inflight = {FIFO_CNT_W{1'b0}};
        for (pk = 0; pk < PIPE; pk = pk + 1) vf_inflight = vf_inflight + {{(FIFO_CNT_W-1){1'b0}}, vpend[pk]};
        if (flush) begin
            vf_nv <= {FIFO_CNT_W{1'b0}}; vpend <= {PIPE{1'b0}};
            vf_rd <= {FIFO_PTR_W{1'b0}}; vf_wr <= {FIFO_PTR_W{1'b0}};
            pf_point <= {SCAN_COUNT_WIDTH{1'b0}}; pf_base <= 1'b0;
            scan_raddr <= {SCAN_ADDR_WIDTH{1'b0}};
        end else begin
            if (vec_landed) begin
                vf_mem[vf_wr] <= scan_rdata;
                vf_wr <= vf_wr + 1'b1;
                vf_nv <= vf_after_pop + 1'b1;
            end else begin
                vf_nv <= vf_after_pop;
            end
            vec_issue = scan_active_at_start
                        && ((vf_after_pop + vf_inflight) < VF_DEPTH[FIFO_CNT_W-1:0])
                        && scan_point_resident(pf_point, pf_base);
            vpend <= {vpend[PIPE-2:0], vec_issue};
            if (vec_issue) begin
                scan_raddr <= scan_addr_of(pf_point, pf_base);
                if ((pf_point + 1'b1) >= scan_count) begin
                    pf_point <= {SCAN_COUNT_WIDTH{1'b0}};
                    pf_base <= pf_base ^ scan_wrap_toggle;      // cyclic bank flip (0 if resident)
                end else begin
                    pf_point <= pf_point + 1'b1;
                end
            end
        end

        if (!reset_sync && (|evt_fifo_overflow || |bus_delay_overflow)) overflow <= 1'b1;
    end

    // ----- TTL EVENT SCHEDULER runtime -------------------------------------------------------
    // Timeline (g_time == running tick t, both reset at FIRE):
    //   * cycle t: the undelayed state_mask differs from prev_undelayed on channel ch
    //     (the toggle that happened AT t) -> push {g_time + d_ch - 1, new_level} into ch's
    //     event FIFO (d_ch >= 2 here; d == 1 is the prev_undelayed register, d == 0 bypass).
    //   * cycle u == t + d - 1: the head matches g_time -> evt_out[ch] <= level (NBA),
    //     visible during cycle t + d  ==>  out[t] = in[t-d] exactly, 0 before the first event.
    //   * g_time keeps counting through draining so a long delayed tail completes; CMD_SAFE (reset)
    //     clears the queues and drops the outputs immediately.
    // Per-channel pushes are strictly time-ordered (one toggle per cycle per channel), so a
    // plain FIFO + equality compare is exact.  The host validates that no more than
    // EVT_DEPTH toggles are in flight inside any d-window; the guard below additionally
    // drops (rather than corrupts) on an impossible overflow.
    always @(posedge clk) begin
        if (reset_sync) begin
            g_time <= {GTIME_WIDTH{1'b0}};
            prev_undelayed <= {CHANNEL_COUNT{1'b0}};
        end else begin
            g_time <= g_time + 1'b1;
            prev_undelayed <= state_mask;
        end
    end
    // One independent FIFO per TTL channel. Each is its own 2D distributed-RAM (sync write @wr,
    // async read @rd); a single 3D array would not infer as LUTRAM because every channel has
    // independent pointers.
    genvar gevs;
    generate
    for (gevs = 0; gevs < CHANNEL_COUNT; gevs = gevs + 1) begin : g_evtfifo
        (* ram_style = "distributed" *) reg [GTIME_WIDTH:0] fifo [0:EVT_DEPTH-1];
        reg [EVT_ADDR-1:0] wr  = {EVT_ADDR{1'b0}};
        reg [EVT_ADDR-1:0] rd  = {EVT_ADDR{1'b0}};
        reg [EVT_ADDR:0]   cnt = {(EVT_ADDR+1){1'b0}};
        reg                obit = 1'b0;                        // this channel's scheduled (delayed) level
        reg pushf, popf;
        wire want_push = (state_mask[gevs] != prev_undelayed[gevs])
                         && (del_ch_ticks[gevs] > {{(TTL_DELAY_WIDTH-1){1'b0}}, 1'b1});
        wire [GTIME_WIDTH:0] headw = fifo[rd];                 // async-read FIFO head (LUTRAM read port)
        wire want_pop = (cnt != {(EVT_ADDR+1){1'b0}})
                        && (headw[GTIME_WIDTH:1] == g_time);
        assign evt_fifo_busy[gevs] = (cnt != {(EVT_ADDR+1){1'b0}}) || obit;
        assign evt_fifo_overflow[gevs] = want_push
                                          && (cnt == EVT_DEPTH[EVT_ADDR:0]) && !want_pop;
        assign evt_out[gevs] = obit;
        always @(posedge clk) begin
            if (reset_sync) begin
                wr   <= {EVT_ADDR{1'b0}};
                rd   <= {EVT_ADDR{1'b0}};
                cnt  <= {(EVT_ADDR+1){1'b0}};
                obit <= 1'b0;
            end else begin
                popf  = want_pop;
                pushf = want_push && ((cnt != EVT_DEPTH[EVT_ADDR:0]) || popf);
                if (pushf) begin
                    // zero-EXTEND the 32b delay to the 48b time base
                    fifo[wr] <= {
                        g_time + {{(GTIME_WIDTH-TTL_DELAY_WIDTH){1'b0}}, del_ch_ticks[gevs]} - 1'b1,
                        state_mask[gevs] };
                    wr <= wr + 1'b1;
                end
                if (popf) begin
                    obit <= headw[0];
                    rd <= rd + 1'b1;
                end
                case ({pushf, popf})
                    2'b10: cnt <= cnt + 1'b1;
                    2'b01: cnt <= cnt - 1'b1;
                    default: ;
                endcase
            end
        end
    end
    endgenerate
endmodule
