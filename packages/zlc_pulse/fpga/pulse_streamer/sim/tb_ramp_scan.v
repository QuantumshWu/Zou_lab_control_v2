`timescale 1ns/1ps
// REAL-ENGINE proof of the edge+RAMP DAC scan: a ramp whose target reads scan slot 0
// at runtime (value select = 1).  Three scan points (codes 420, 900, 600; the first two
// are the full staircase oracle and the third arrives from a late bank).  STEEP
// Bresenham: delta > span exercises the deferred divmod + multi-code stepping.
// For every tick of both points the engine's bus_out must equal the integer staircase
//   v(t) = vstart + floor((t - t0) * delta / span)   (landing exactly on the code),
// which is byte-identical to the Python preview (analog_levels).
module tb_ramp_scan;
  localparam integer CH=8, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1, SSW=`ZLC_SLOT_SEL_WIDTH;
  localparam integer BUSC=4, BW=10, ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, TDW=32, SAW=2;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer T1=20, SPAN=40, TF=60;         // edge row, ramp row, 60-tick frame
  localparam [BW-1:0] VSTART=10'd320;
  localparam [TW-1:0] P0VAL=32'd420, P1VAL=32'd900, P2VAL=32'd600;

  reg clk=0; always #10 clk=~clk;
  reg reset, start;

  function [ABITS-1:0] act;
    input [1:0] mode; input [SSW-1:0] sel; input [BW-1:0] v;
    begin act = {mode, sel, v}; end
  endfunction
  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask; input [ABITS-1:0] bus0;
    begin row_of = {{((BUSC-1)*ABITS){1'b0}}, bus0, mask, {SSW{1'b0}}, dur}; end
  endfunction

  // rows: r0 = 20 ticks, ch5 on, bus0 edge 320; r1 = 40 ticks, off, bus0 ramp to slot 0
  reg [RBITS-1:0] rowmem [0:3];
  initial begin
    rowmem[0]=row_of(T1[TW-1:0], 8'h20, act(2'd1, 0, VSTART));
    rowmem[1]=row_of(SPAN[TW-1:0], 8'h00, act(2'd2, 1, 0));
    rowmem[2]=0; rowmem[3]=0;
  end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[1:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  // scan BRAM stub (registered, like the real IP): point p -> slot value
  reg [NS*TW-1:0] scanmem [0:3];
  reg [NS*TW-1:0] spipe [0:2];
  wire [SAW-1:0] scan_raddr;
  always @(posedge clk) begin spipe[0]<=scanmem[scan_raddr[1:0]]; spipe[1]<=spipe[0]; spipe[2]<=spipe[1]; end
  wire [NS*TW-1:0] scan_rdata = spipe[2];
  initial begin scanmem[0]=P0VAL; scanmem[1]=P1VAL; scanmem[2]=P2VAL; scanmem[3]=0; end

  reg [1:0] bank_ready;
  reg [31:0] bank_chunk0, bank_chunk1;

  wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .BUS_COUNT(BUSC), .BUS_WIDTH(BW),
                        .SCAN_ADDR_WIDTH(SAW), .BANK_SIZE(2)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(2),.run_repeat_count(32'd1),
    .scan_enable(1'b1),.scan_count(32'd3),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata(scan_rdata),
    .bank_ready(bank_ready),.bank_chunk0(bank_chunk0),.bank_chunk1(bank_chunk1),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*TDW{1'b0}}),.delay_ticks({CH*TDW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  // expected staircase for point value PV at frame tick t (matches the Python preview)
  function [BW-1:0] expect_v;
    input integer t; input [TW-1:0] pv;
    integer delta, k, moves;
    begin
      if (t <= T1) expect_v = VSTART;
      else if (t >= T1 + SPAN) expect_v = pv[BW-1:0];
      else begin
        delta = pv - VSTART;       // both points ramp upward here
        k = t - T1;
        moves = (k * delta) / SPAN;
        if (moves > delta) moves = delta;
        expect_v = VSTART + moves;
      end
    end
  endfunction

  integer t, started, errs, p, ft;
  reg [BW-1:0] hist [0:2*60+10];
  initial begin
    reset=1; start=0; bank_ready=2'b01; bank_chunk0=0; bank_chunk1=1;
    repeat (200) @(posedge clk);                                       // arm with reset held
    t=-1; started=0;
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
    while (t < 2*TF) begin
      @(posedge clk);
      if (running && !started) begin started=1; t=-1; end
      if (started) begin t=t+1; if (t>=0 && t<=2*TF) hist[t]=bus_out[0*BW +: BW]; end
      if (done) t = 2*TF;
    end
    errs = 0;
    for (p = 0; p < 2; p = p + 1) begin
      $write("P%0d bus:", p);
      for (ft = 0; ft < TF; ft = ft + 6) $write(" %0d@%0d", hist[p*TF+ft], ft);
      $write("\n");
      for (ft = 0; ft < TF; ft = ft + 1)
        if (hist[p*TF+ft] !== expect_v(ft, (p==0)?P0VAL:P1VAL)) begin
          if (errs < 8) $display("  MISMATCH p%0d t%0d got %0d expect %0d", p, ft, hist[p*TF+ft], expect_v(ft, (p==0)?P0VAL:P1VAL));
          errs = errs + 1;
        end
    end
    if (errs != 0) $fatal(1, "RAMP-SCAN-BAD mismatches=%0d", errs);

    // Point 2 lives in bank 1, deliberately not ready through the physical seam:
    // the engine holds point 1's last row and flags UNDERFLOW instead of playing
    // a stale point.  Once the bank is resident the prefetcher fetches the point
    // and the seam completes: the next frame restarts from row 0 at VSTART.
    repeat (4) @(posedge clk);
    if (dut.scan_point_index != 1) $fatal(1, "late-bank setup lost point 1 (index=%0d)", dut.scan_point_index);
    if (!underflow) $fatal(1, "a missing bank did not raise UNDERFLOW at the seam");
    if (!running) $fatal(1, "the engine gave up on a late bank instead of holding");
    @(negedge clk);
    bank_ready = 2'b11;
    // the prefetcher reads the point (RD_LAT+2 clocks) and the held seam completes
    for (ft = 0; ft < 12 && dut.scan_point_index != 2; ft = ft + 1) @(posedge clk);
    if (dut.scan_point_index != 2)
      $fatal(1, "late bank did not advance to point 2 (index=%0d)", dut.scan_point_index);
    #1;
    if (bus_out[0*BW +: BW] != VSTART)
      $fatal(1, "point-2 bus did not restart from row 0 (bus=%0d)", bus_out[0*BW +: BW]);
    // and the third point plays its own staircase towards 600, sampled against the
    // engine's own frame tick: the ramp's last step lands on the target at the tick
    // the next row would start, which for the last row of a finite run is the tick
    // the outputs go SAFE, so the row itself is what is checked.
    for (ft = 0; ft < TF - 1; ft = ft + 1) begin
      #1;
      if (bus_out[0*BW +: BW] !== expect_v(dut.frame_tick, P2VAL)) begin
        if (errs < 8) $display("  MISMATCH p2 ft%0d got %0d expect %0d", dut.frame_tick, bus_out[0*BW +: BW], expect_v(dut.frame_tick, P2VAL));
        errs = errs + 1;
      end
      @(posedge clk);
    end
    if (errs != 0) $fatal(1, "RAMP-SCAN-BAD late-point mismatches=%0d", errs);
    repeat (20) @(posedge clk);
    if (!done) $fatal(1, "three-point scan did not finish");
    $display("RAMP-SCAN-OK mismatches=0");
    $finish;
  end
endmodule
