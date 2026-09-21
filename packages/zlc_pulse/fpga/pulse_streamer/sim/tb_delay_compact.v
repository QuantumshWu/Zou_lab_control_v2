`timescale 1ns/1ps
`include "zlc_geometry.vh"
// The compact TTL-only engine owns all 25 channels, each with its own delay FIFO.
// Channel ch gets ch ticks of delay: zero/one-tick bypasses and all six new pins,
// including bit 24, must preserve every edge across repeated frames.
module tb_delay_compact;
  localparam integer CH=`ZLC_NUM_DELAY_CH, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1;
  localparam integer SSW=`ZLC_SLOT_SEL_WIDTH, BUSC=`ZLC_BUS_COUNT, BW=`ZLC_BUS_WIDTH;
  localparam integer ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NE=7;
  localparam integer NT=6000;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask;
    begin row_of = {{(BUSC*ABITS){1'b0}}, mask, {SSW{1'b0}}, dur}; end
  endfunction

  reg [RBITS-1:0] rowmem [0:7];
  initial begin
    rowmem[0]=row_of(32'd5,    {CH{1'b1}});   // all on
    rowmem[1]=row_of(32'd1,    {CH{1'b0}});
    rowmem[2]=row_of(32'd1,    {CH{1'b1}});   // 1-tick pulse (stress)
    rowmem[3]=row_of(32'd93,   {CH{1'b0}});
    rowmem[4]=row_of(32'd60,   {CH{1'b1}});
    rowmem[5]=row_of(32'd1840, {CH{1'b0}});
    rowmem[6]=row_of(32'd400,  {CH{1'b1}});   // frame ends at 2400
    rowmem[7]=0;
  end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[2:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  genvar ch;
  generate for (ch=0; ch<CH; ch=ch+1) begin : delays
    assign delay_ticks_w[ch*TDW +: TDW] = ch;
  end endgenerate

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(NE[RAW:0]),.run_repeat_count(32'd0),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  initial begin
    reset=1; start=0;
    repeat (200) @(posedge clk);         // complete at least two arm flush/refill passes
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  reg [CH-1:0] hist [0:NT];
  integer t = -1; integer errs = 0; integer started = 0;
  integer dch; integer dexp; reg expected;
  always @(posedge clk) begin : oracle
    if (running && !started) begin started = 1; t = -1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) hist[t] = dut.state_mask;
      if (t > 0) begin
        // delayed channels: out[ch] == hist[t-d][ch] (all channels share bit 0's waveform)
        for (dch = 0; dch < CH; dch = dch + 1) begin
          dexp = dch;
          if (dexp == 0) begin
            // Zero delay is the current undelayed state_mask.
            if (out[dch] !== dut.state_mask[dch]) begin
              errs = errs + 1;
              if (errs <= 8) $display("  PASSTHRU MISMATCH t=%0d ch=%0d", t, dch);
            end
          end else begin
            expected = (t >= dexp) ? hist[t-dexp][dch] : 1'b0;
            if (out[dch] !== expected) begin
              errs = errs + 1;
              if (errs <= 8) $display("  DELAY MISMATCH t=%0d ch=%0d d=%0d out=%b exp=%b",
                                       t, dch, dexp, out[dch], expected);
            end
          end
        end
      end
    end
  end

  initial begin
    wait(reset==1); wait(reset==0);
    repeat (NT) @(posedge clk);
    if (errs != 0 || t <= 4000 || underflow)
      $fatal(1,"TTL delay mismatch: channels=%0d cycles=%0d mismatches=%0d",CH,t,errs);
    $display("COMPACT-MAP-OK channels=%0d cycles=%0d",CH,t);
    $finish;
  end
endmodule
