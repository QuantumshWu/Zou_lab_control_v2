`timescale 1ns/1ps
`include "zlc_geometry.vh"
// NESTED-LOOP test with a non-zero loop start.  The preamble and tail must each play
// once while the outer bracket (rows 2..4, x3) plays exactly three times and the
// inner bracket (row 3, x2) twice inside every outer replay -- on the TTL outputs
// AND on a DAC bus whose three edges give the preamble, body and tail three
// different codes: the body's code must hold through every rewind, and the tail's
// code must land on the very tick of the tail's TTL edge.
module tb_loop;
  localparam integer CH=`ZLC_NUM_DELAY_CH, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=`ZLC_NUM_SLOTS;
  localparam integer SSW=`ZLC_SLOT_SEL_WIDTH, BUSC=`ZLC_BUS_COUNT, BW=`ZLC_BUS_WIDTH;
  localparam integer ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NE=8;
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
    .clk(clk),.reset(reset),.start(start),.prog_count(NE[RAW:0]),.run_repeat_count(32'd1),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count(2),.loop_first_flat(loop_first),.loop_last_flat(loop_last),.loop_count_flat(loop_count),
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
  always @(posedge clk) begin
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
    $display("LOOP-OK");
    $finish;
  end
endmodule
