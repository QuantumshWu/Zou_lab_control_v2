`timescale 1ns/1ps
`include "zlc_geometry.vh"
// NESTED-LOOP test with a non-zero loop start.  The preamble and tail must each play
// once while the outer bracket (rows 2..4, x3) plays exactly three times and the
// inner bracket (row 3, x2) twice inside every outer replay -- on the TTL outputs
// AND on a DAC bus whose three edges give the preamble, body and tail three
// different codes: the body's code must hold through every rewind, and the tail's
// code must land on the very tick of the tail's TTL edge.
//
// A second program then plays ONE-TICK rows through four nested brackets that share
// start rows and end rows -- a bracket opening where an outer one opens, one-row
// brackets closing where their parents close, a one-row bracket wrapped by another
// of the same span -- and every tick's mask is checked against the hand-expanded
// play order: the walker must open, close and rewind all of them within the one
// clock a one-tick row allows.
module tb_loop;
  localparam integer CH=`ZLC_NUM_DELAY_CH, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=`ZLC_NUM_SLOTS;
  localparam integer SSW=`ZLC_SLOT_SEL_WIDTH, BUSC=`ZLC_BUS_COUNT, BW=`ZLC_BUS_WIDTH;
  localparam integer ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NE=8;
  reg [RAW:0] prog_count_r = NE[RAW:0];
  reg [LIW:0] ltc_r = 2;
  localparam [BW-1:0] DAC_PRE = 10'd100, DAC_BODY = 10'd200, DAC_TAIL = 10'd300;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [ABITS-1:0] act;
    input [1:0] mode; input [SSW-1:0] sel; input [BW-1:0] v;
    begin act = {mode, sel, v}; end
  endfunction
  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask; input [ABITS-1:0] bus0;
    begin row_of = {{((BUSC-1)*ABITS){1'b0}}, bus0, mask, {SSW{1'b0}}, dur}; end
  endfunction

  reg [RBITS-1:0] rowmem [0:(1<<RAW)-1];
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  // loop table: entry 0 = outer rows 2..4 x3, entry 1 = inner row 3 x2 (outermost first)
  reg [ML*RAW-1:0] loop_first = {ML*RAW{1'b0}};
  reg [ML*RAW-1:0] loop_last = {ML*RAW{1'b0}};
  reg [ML*32-1:0] loop_count = {ML*32{1'b0}};

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(prog_count_r),.run_repeat_count(32'd1),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count(ltc_r),.loop_first_flat(loop_first),.loop_last_flat(loop_last),.loop_count_flat(loop_count),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),.bank_ready(2'b11),
    .bank_chunk0(32'd0),.bank_chunk1(32'd0),.scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));
  integer i;
  initial begin
    for (i=0;i<(1<<RAW);i=i+1) rowmem[i]=0;
    // bit0=preamble, bit11=loop body, bit1=post-loop tail; every row 100 ticks.
    rowmem[0] = row_of(32'd100, 32'h1,   act(2'd1, 0, DAC_PRE));
    rowmem[1] = row_of(32'd100, 32'h0,   act(2'd0, 0, 0));
    rowmem[2] = row_of(32'd100, 32'h800, act(2'd1, 0, DAC_BODY));
    rowmem[3] = row_of(32'd100, 32'h0,   act(2'd0, 0, 0));
    rowmem[4] = row_of(32'd100, 32'h0,   act(2'd0, 0, 0));
    rowmem[5] = row_of(32'd100, 32'h2,   act(2'd1, 0, DAC_TAIL));
    rowmem[6] = row_of(32'd100, 32'h0,   act(2'd0, 0, 0));
    rowmem[7] = row_of(32'd100, 32'h0,   act(2'd0, 0, 0));
    loop_first[0*RAW +: RAW] = 2; loop_last[0*RAW +: RAW] = 4; loop_count[0*32 +: 32] = 3;
    loop_first[1*RAW +: RAW] = 3; loop_last[1*RAW +: RAW] = 3; loop_count[1*32 +: 32] = 2;
    reset=1; start=0;
    repeat (200) @(posedge clk); reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end
  integer tcount=0; reg [CH-1:0] previous=0;
  integer lo=-1, np=0, nbad=0, preamble=0, tail=0, orderbad=0, lastrise=-1, nperiodbad=0;
  // The DAC must read the current period's code on EVERY running tick: the
  // preamble's until the body's first TTL edge, the body's through all three
  // plays and both rewinds, the tail's from the tail's TTL edge on.
  integer dacbad=0; reg [BW-1:0] dac_expected = DAC_PRE;
  reg phase1_done = 0, phase2 = 0;
  always @(posedge clk) if (!phase2) begin
    if (running||done) tcount=tcount+1;
    if (running && out[0] && !previous[0]) begin
      if (preamble!=0 || np!=0 || tail!=0) orderbad=orderbad+1;
      preamble=preamble+1;
    end
    if (running && out[11]!==previous[11]) begin
      if (out[11]) begin
        // one outer replay = row2 (100) + row3 x2 (200) + row4 (100) = 400 ticks
        if (lastrise >= 0 && tcount-lastrise != 400) nperiodbad = nperiodbad + 1;
        lastrise = tcount; lo=tcount;
      end else begin np=np+1; if (tcount-lo!=100) nbad=nbad+1;
        $display("[LOOP] emCCD pulse#%0d on=%0d off=%0d width=%0d %s",np,lo,tcount,tcount-lo,(tcount-lo==100)?"OK":"**BAD**"); end
    end
    if (running && out[1] && !previous[1]) begin
      if (preamble!=1 || np!=3 || tail!=0) orderbad=orderbad+1;
      // the tail starts 200 (preamble) + 3 x 400 (outer replays) ticks in
      if (tcount != 1401) orderbad=orderbad+1;
      tail=tail+1;
    end
    if (running) begin
      if (out[11] && !previous[11] && np==0) dac_expected = DAC_BODY;
      if (out[1] && !previous[1]) dac_expected = DAC_TAIL;
      if (bus_out[0 +: BW] !== dac_expected) begin
        dacbad = dacbad + 1;
        if (dacbad <= 8) $display("[LOOP] **BAD** DAC t=%0d out=%0d expected=%0d",
                                 tcount, bus_out[0 +: BW], dac_expected);
      end
    end
    previous<=out;
  end

  // ---- phase 2: one-tick rows, four brackets deep, shared starts and ends ----
  // rows 0..6, one tick each, mask bit = row number.  Loop table (outermost first,
  // start ascending, end descending): A rows 1..5 x2, B rows 1..3 x2, C row 3 x3,
  // E row 3 x2, D row 5 x2.  Hand expansion: 0, (1 2 3x6 1 2 3x6 4 5 5) x2, 6.
  localparam integer NE2 = 7, NT2 = 40;
  reg [3:0] expect2 [0:NT2-1];
  integer k2 = 0, t2 = 0, bad2 = 0, ai, bi, ci;
  initial begin
    expect2[k2] = 0; k2 = k2 + 1;
    for (ai = 0; ai < 2; ai = ai + 1) begin
      for (bi = 0; bi < 2; bi = bi + 1) begin
        expect2[k2] = 1; k2 = k2 + 1;
        expect2[k2] = 2; k2 = k2 + 1;
        for (ci = 0; ci < 6; ci = ci + 1) begin expect2[k2] = 3; k2 = k2 + 1; end
      end
      expect2[k2] = 4; k2 = k2 + 1;
      expect2[k2] = 5; k2 = k2 + 1;
      expect2[k2] = 5; k2 = k2 + 1;
    end
    expect2[k2] = 6; k2 = k2 + 1;
    if (k2 != NT2) $fatal(1, "phase-2 expansion is %0d ticks, not %0d", k2, NT2);
  end
  always @(posedge clk) begin
    if (phase2 && running) begin
      if (t2 >= NT2 || out !== ({{(CH-1){1'b0}}, 1'b1} << expect2[t2])) begin
        bad2 = bad2 + 1;
        if (bad2 <= 8) $display("[LOOP2] **BAD** tick %0d out=%h expected row %0d", t2, out, (t2 < NT2) ? expect2[t2] : -1);
      end
      t2 = t2 + 1;
    end
  end
  initial begin
    wait(phase1_done);
    reset = 1; @(posedge clk);
    for (i = 0; i < (1<<RAW); i = i + 1) rowmem[i] = 0;
    for (i = 0; i < NE2; i = i + 1) rowmem[i] = row_of(32'd1, ({{(CH-1){1'b0}}, 1'b1} << i), act(2'd0, 0, 0));
    loop_first = {ML*RAW{1'b0}}; loop_last = {ML*RAW{1'b0}}; loop_count = {ML*32{1'b0}};
    loop_first[0*RAW +: RAW] = 1; loop_last[0*RAW +: RAW] = 5; loop_count[0*32 +: 32] = 2;   // A
    loop_first[1*RAW +: RAW] = 1; loop_last[1*RAW +: RAW] = 3; loop_count[1*32 +: 32] = 2;   // B
    loop_first[2*RAW +: RAW] = 3; loop_last[2*RAW +: RAW] = 3; loop_count[2*32 +: 32] = 3;   // C
    loop_first[3*RAW +: RAW] = 3; loop_last[3*RAW +: RAW] = 3; loop_count[3*32 +: 32] = 2;   // E
    loop_first[4*RAW +: RAW] = 5; loop_last[4*RAW +: RAW] = 5; loop_count[4*32 +: 32] = 2;   // D
    prog_count_r = NE2[RAW:0]; ltc_r = 5;
    repeat (200) @(posedge clk); reset = 0; @(posedge clk); start = 1; @(posedge clk); start = 0;
    phase2 = 1;
    repeat (400) @(posedge clk);
    $display("==== LOOP2(one-tick rows, 4 deep, shared starts/ends): ticks=%0d expected=%0d bad=%0d ====", t2, NT2, bad2);
    if (t2 != NT2 || bad2 != 0)
      $fatal(1, "nested one-tick loops played %0d ticks with %0d wrong masks", t2, bad2);
    if (!done || running || underflow || out !== {CH{1'b0}})
      $fatal(1, "phase 2 did not finish SAFE: run=%b done=%b uf=%b out=%h", running, done, underflow, out);
    $display("LOOP-OK");
    $finish;
  end

  initial begin
    wait(reset==1); wait(reset==0);
    repeat (3000) @(posedge clk);
    $display("==== LOOP(outer x3, inner x2): pulses=%0d expected=3 width_errors=%0d period_errors=%0d ====", np, nbad, nperiodbad);
    if (np != 3) $fatal(1, "loop bench produced %0d pulses, expected 3", np);
    if (nbad != 0) $fatal(1, "loop bench had %0d bad pulse widths", nbad);
    if (nperiodbad != 0) $fatal(1, "outer replays were not 400 ticks apart (%0d bad)", nperiodbad);
    if (preamble != 1 || tail != 1 || orderbad != 0)
      $fatal(1, "loop order/count mismatch: pre=%0d body=%0d tail=%0d orderbad=%0d",
             preamble, np, tail, orderbad);
    if (dacbad != 0)
      $fatal(1, "DAC left its period's code on %0d running ticks", dacbad);
    if (!done || running || underflow || out !== {CH{1'b0}})
      $fatal(1, "loop did not finish SAFE: run=%b done=%b uf=%b out=%h", running, done, underflow, out);
    $display("LOOP-PHASE1-OK");
    phase1_done = 1;
  end
endmodule
