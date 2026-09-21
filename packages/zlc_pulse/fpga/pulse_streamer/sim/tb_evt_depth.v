`timescale 1ns/1ps
// EVENT-FIFO DEPTH BOUNDARY on the real engine: with deployed EVT_DEPTH=32,
//   * a burst of EXACTLY 32 toggles inside one delay window is delayed
//     tick-exactly (the FIFO full-at-32 boundary must not corrupt anything);
//   * a burst of 34 toggles exceeds capacity and MUST set sticky overflow.
// The exact-depth lane remains tick-exact; the overflowing lane may drop data,
// but that corruption can no longer be silent.
// bit0 carries the 32-toggle burst, bit1 the 34-toggle burst; both delayed by
// d=200 so the whole burst is in flight at once.
module tb_evt_depth;
  localparam integer CH=8, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1, SSW=`ZLC_SLOT_SEL_WIDTH;
  localparam integer BUSC=4, BW=10, ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NE=37;
  localparam integer NT=1200;
  localparam integer D=200;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask;
    begin row_of = {{(BUSC*ABITS){1'b0}}, mask, {SSW{1'b0}}, dur}; end
  endfunction

  // toggle ticks 10,12,...: bit0 toggles at the first 32, bit1 at all 34.
  reg [RBITS-1:0] rowmem [0:63];
  integer i; reg b0; reg b1;
  initial begin
    for (i=0; i<64; i=i+1) rowmem[i]=0;
    rowmem[0]=row_of(32'd10, 8'h00);
    b0=0; b1=0;
    for (i=0; i<34; i=i+1) begin
      b1 = ~b1;
      if (i<32) b0 = ~b0;
      rowmem[1+i] = row_of(32'd2, {6'b0, b1, b0});   // toggle at 10 + 2*i
    end
    rowmem[35]=row_of(32'd422, 8'h00);   // 78 .. 500 all off
    rowmem[36]=row_of(32'd1, 8'h00);     // 500 .. 501: frame end
  end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[5:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  assign delay_ticks_w[0*TDW +: TDW] = D;       // 32-toggle burst, exactly depth
  assign delay_ticks_w[1*TDW +: TDW] = D;       // 34-toggle burst, overflow by 2
  assign delay_ticks_w[CH*TDW-1: 2*TDW] = {(CH-2)*TDW{1'b0}};

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done, overflow, physical_active; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(NE[RAW:0]),.run_repeat_count(32'd1),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done),
    .overflow(overflow),.physical_active(physical_active));

  initial begin
    reset=1; start=0;
    repeat (200) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  // oracle: history of the undelayed stream + per-cycle asserts.
  // ch0 (32 toggles = depth): out0[t] == in0[t-D] EXACTLY.
  // ch1 intentionally overflows; its waveform is invalid once overflow is set.
  reg [CH-1:0] hist [0:NT];
  integer t = -1; integer errs0 = 0; integer started = 0;
  reg exp0;
  always @(posedge clk) begin : oracle
    if (running && !started) begin started = 1; t = -1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) hist[t] = dut.state_mask;
      if (t > 0) begin
        exp0 = (t >= D) ? hist[t-D][0] : 1'b0;
        if (out[0] !== exp0) begin
          errs0 = errs0 + 1;
          if (errs0 <= 5) $display("  MISMATCH ch0 t=%0d out=%b expect=%b", t, out[0], exp0);
        end
      end
    end
  end

  initial begin
    wait(reset==1); wait(reset==0);
    fork
      begin
        wait(done);
        @(posedge clk);
        if (errs0 != 0) $fatal(1, "exact-capacity FIFO corrupted %0d ticks", errs0);
        if (!overflow) $fatal(1, "FIFO overflow was not sticky through DONE");
        if (underflow) $fatal(1, "unexpected scan underflow");
        $display("EVT-DEPTH-STICKY-OVERFLOW-OK cycles=%0d", t);
        $finish;
      end
      begin
        repeat (NT) @(posedge clk);
        $fatal(1, "timeout waiting for overflow run DONE (running=%b draining=%b busy=%b frame_tick=%0d left=%0d cnt0=%0d cnt1=%0d out=%h)",
               running, dut.draining, dut.delay_runtime_busy,
               dut.frame_tick, dut.left,
               dut.g_evtfifo[0].cnt, dut.g_evtfifo[1].cnt, out);
      end
    join
  end
endmodule
