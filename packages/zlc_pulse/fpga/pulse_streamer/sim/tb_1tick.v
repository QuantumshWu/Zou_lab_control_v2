`timescale 1ns/1ps
`include "zlc_geometry.vh"
// 1-tick back-to-back stress on the period table: rows one tick long with ALTERNATING masks,
// so every 20 ns row must be visible on its own cycle.  Verifies FIFO_DEPTH/PIPE sustains the
// design's headline 1-tick capability across a long row (a prefetch bubble) and at the very
// first and last rows of one-row and two-row finite Pulses.
module tb_1tick;
  localparam integer CH=`ZLC_NUM_DELAY_CH, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=`ZLC_NUM_SLOTS;
  localparam integer SSW=`ZLC_SLOT_SEL_WIDTH, BUSC=`ZLC_BUS_COUNT, BW=`ZLC_BUS_WIDTH;
  localparam integer ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NE=20;
  reg clk=0, reset=0, start=0; reg [RAW:0] prog_count=NE; always #10 clk=~clk;

  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask;
    begin row_of = {{(BUSC*ABITS){1'b0}}, mask, {SSW{1'b0}}, dur}; end
  endfunction

  // behavioral row memory: registered address in the engine + 3 pipeline stages = RD_LAT+2
  reg [RBITS-1:0] rowmem [0:(1<<RAW)-1];
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(prog_count),.run_repeat_count(32'd1),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),.bank_ready(2'b11),
    .bank_chunk0(32'd0),.bank_chunk1(32'd0),.scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));
  integer i;
  reg [31:0] starts[0:NE-1]; reg [31:0] masks[0:NE-1];

  task short_case; input [RAW:0] count; begin
    reset=1; start=0; prog_count=count;
    rowmem[0] = row_of(32'd1, 32'h1);
    rowmem[1] = row_of(32'd1, 32'h2);
    repeat (200) @(posedge clk);
    @(negedge clk); reset=0; start=1;
    // reset/start each cross the engine's two-flop synchronizer.
    repeat (3) @(posedge clk); #1;
    if (!running || out !== 32'h1) $fatal(1, "short frame row0 mismatch: count=%0d out=%h run=%b", count, out, running);
    @(negedge clk); start=0;
    @(posedge clk); #1;
    if (count==1) begin
      if (out !== 32'h0) $fatal(1, "1-row terminal mask mismatch: %h", out);
    end else begin
      if (out !== 32'h2) $fatal(1, "2-row second mask mismatch: %h", out);
      @(posedge clk); #1;
      if (out !== 32'h0) $fatal(1, "2-row terminal mask mismatch: %h", out);
    end
    repeat (8) @(posedge clk); #1;
    if (!done || running || underflow || out !== {CH{1'b0}})
      $fatal(1, "short frame did not finish SAFE: count=%0d run=%b done=%b uf=%b out=%h",
             count, running, done, underflow, out);
    $display("SHORT-ONE-SHOT-OK ticks=%0d", count);
    @(negedge clk); reset=1;
    repeat (4) @(posedge clk);
  end endtask

  reg dense_phase=0;
  integer tcount=0; reg [CH-1:0] op=32'hx; integer seen=0; integer bad=0; integer exp_t;
  initial begin
    for (i=0;i<(1<<RAW);i=i+1) rowmem[i]=0;
    short_case(1);
    short_case(2);
    // rows 0..8 one tick each, row 9 lasts to tick 100, rows 10..19 one tick each
    for (i=0;i<NE;i=i+1) begin
      starts[i] = (i<10) ? i : 90+i;
      masks[i]  = (i[0]) ? 32'h001 : 32'h002;
    end
    for (i=0;i<NE;i=i+1) rowmem[i] = row_of((i==9) ? 32'd91 : 32'd1, masks[i]);
    reset=1; start=0; prog_count=NE;
    repeat (200) @(posedge clk);
    tcount=0; seen=0; bad=0; op=32'hx; dense_phase=1;
    @(negedge clk); reset=0; start=1; @(negedge clk); start=0;
  end
  always @(posedge clk) begin
    if (dense_phase && (running||done)) tcount=tcount+1;
    if (dense_phase && (running||done) && out!==op) begin
      // each visible change = the next row starting; expected start tick = starts[seen]+1,
      // and the terminal all-low one tick after the last one-tick row
      exp_t = (seen < NE) ? starts[seen]+1 : starts[NE-1]+2;
      $display("  row#%0d out=0x%h at t=%0d (expect %0d) %s", seen, out[7:0], tcount, exp_t, (tcount==exp_t)?"OK":"**LATE**");
      if (tcount != exp_t) bad=bad+1;
      seen=seen+1; op<=out;
    end
  end
  initial begin
    wait(dense_phase==1); wait(reset==0);
    repeat (400) @(posedge clk);
    $display("==== 1-tick: %0d rows started, %0d off-schedule ====", seen, bad);
    if (seen != NE+1) $fatal(1, "1-tick bench started %0d rows (+terminal low), expected %0d", seen, NE+1);
    if (bad != 0) $fatal(1, "1-tick bench had %0d off-schedule rows", bad);
    if (!done || underflow) $fatal(1, "1-tick bench did not finish clean: done=%b uf=%b", done, underflow);
    $display("ONE-TICK-OK");
    $finish;
  end
endmodule
