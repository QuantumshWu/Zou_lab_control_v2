`timescale 1ns/1ps
`include "zlc_geometry.vh"
// Gap sweep: the user's e6/e7/e8 shape with the e4->e5 gap parameterized via `GAP, played
// as an infinite Run-repeat of nine rows.  Proves the emCCD pulse width and the exact
// repeat period across gaps 1,2,3,5,10,50,500 (one-tick rows at a frame seam included).
`ifndef GAP
 `define GAP 500
`endif
module tb_gapsweep;
  localparam integer CH=`ZLC_NUM_DELAY_CH, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=`ZLC_NUM_SLOTS;
  localparam integer SSW=`ZLC_SLOT_SEL_WIDTH, BUSC=`ZLC_BUS_COUNT, BW=`ZLC_BUS_WIDTH;
  localparam integer ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer GAP = `GAP;
  localparam integer NE = 9;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask;
    begin row_of = {{(BUSC*ABITS){1'b0}}, mask, {SSW{1'b0}}, dur}; end
  endfunction

  reg [RBITS-1:0] rowmem [0:15];
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[3:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];
  reg [31:0] LE;

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(NE[RAW:0]),.run_repeat_count(32'd0),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),.bank_ready(2'b11),
    .bank_chunk0(32'd0),.bank_chunk1(32'd0),.scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));
  integer i;
  initial begin
    for (i=0;i<16;i=i+1) rowmem[i]=0;
    // the old edge ticks 0,1000,2500,4500,5000,5000+GAP,+5000,+7000,+9000,+9100 as row durations
    rowmem[0]=row_of(32'd1000, 'h685); rowmem[1]=row_of(32'd1500, 'h200); rowmem[2]=row_of(32'd2000, 'ha08);
    rowmem[3]=row_of(32'd500, 'h200);  rowmem[4]=row_of(GAP, 'h200);      rowmem[5]=row_of(32'd5000, 'h200);
    rowmem[6]=row_of(32'd2000, 'ha00); rowmem[7]=row_of(32'd2000, 'h208); rowmem[8]=row_of(32'd100, 'h0);
    LE = 5000 + GAP + 9100;
    reset=1; start=0;
    repeat (200) @(posedge clk); reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end
  integer tcount=0; reg emp=0; integer lo=-1, nbad=0, np=0, nrise=0, nperiodbad=0;
  integer previous_on[0:1];
  initial begin previous_on[0]=-1; previous_on[1]=-1; end
  always @(posedge clk) begin
    if (running||done) tcount=tcount+1;
    if (running && out[11]!==emp) begin
      if (out[11]) begin
        lo=tcount;
        if (previous_on[nrise & 1] >= 0 && tcount-previous_on[nrise & 1] != LE) begin
          nperiodbad=nperiodbad+1;
          $display("[GAP=%0d] repeat onset delta=%0d expect=%0d **BAD**", GAP,
                   tcount-previous_on[nrise & 1], LE);
        end
        previous_on[nrise & 1]=tcount; nrise=nrise+1;
      end
      else begin np=np+1; if (tcount-lo != 2000) nbad=nbad+1;
        $display("[GAP=%0d] emCCD pulse#%0d on=%0d off=%0d width=%0d %s", GAP, np, lo, tcount, tcount-lo, (tcount-lo==2000)?"OK":"**BAD**"); end
      emp<=out[11];
    end
  end
  initial begin
    wait(reset==1); wait(reset==0);
    repeat (40000) @(posedge clk);
    $display("==== GAP=%0d : pulses=%0d width_errors=%0d ====", GAP, np, nbad);
    if (np < 4 || nrise < 4) $fatal(1, "gap %0d did not cross a repeat seam: pulses=%0d rises=%0d", GAP, np, nrise);
    if (nbad != 0) $fatal(1, "gap %0d had %0d bad pulse widths", GAP, nbad);
    if (nperiodbad != 0) $fatal(1, "gap %0d had %0d bad repeat periods", GAP, nperiodbad);
    if (underflow) $fatal(1, "gap %0d raised underflow", GAP);
    $display("GAPSWEEP-OK GAP=%0d pulses=%0d", GAP, np);
    $finish;
  end
endmodule
