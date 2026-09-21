`timescale 1ns/1ps
// PER-BUS action-scheduler DAC delay on the REAL engine.  One row of 2400 ticks
// carries an EDGE to 1023 on buses 0..2, so each undelayed bus value steps SAFE(512)->1023 at
// FIRE, with per-bus delays d0=5, d1=0 (passthrough), d2=1 (the register path).  ORACLE: record
// the engine's UNDELAYED dut.bus_value_active[b] every cycle and assert the delayed pin output
// bus_out[b] == value[t-d_b], holding the SAFE mid-code (512) before t == d_b.
module tb_bus_delay;
  localparam integer CH=8, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1, SSW=`ZLC_SLOT_SEL_WIDTH;
  localparam integer BUSC=4, BW=10, ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, TDW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NT=3000;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [ABITS-1:0] act;
    input [1:0] mode; input [SSW-1:0] sel; input [BW-1:0] v;
    begin act = {mode, sel, v}; end
  endfunction

  reg [RBITS-1:0] rowmem [0:1];
  initial begin
    rowmem[0] = {act(2'd0, 0, 0), act(2'd1, 0, 10'd1023), act(2'd1, 0, 10'd1023), act(2'd1, 0, 10'd1023),
                 8'h00, {SSW{1'b0}}, 32'd2400};
    rowmem[1] = 0;
  end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  // per-bus delays (32b): bus0=5, bus1=0, bus2=1, bus3=0
  wire [BUSC*TDW-1:0] bus_delay_ticks_w;
  assign bus_delay_ticks_w[0*TDW +: TDW] = 32'd5;
  assign bus_delay_ticks_w[1*TDW +: TDW] = 32'd0;
  assign bus_delay_ticks_w[2*TDW +: TDW] = 32'd1;
  assign bus_delay_ticks_w[3*TDW +: TDW] = 32'd0;

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .BUS_COUNT(BUSC), .BUS_WIDTH(BW)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(1),.run_repeat_count(32'd0),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks(bus_delay_ticks_w),.delay_ticks({CH*TDW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  initial begin
    reset=1; start=0;
    repeat (200) @(posedge clk);         // complete at least two arm flush/refill passes
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  reg [BW-1:0] hist0 [0:NT]; reg [BW-1:0] hist1 [0:NT]; reg [BW-1:0] hist2 [0:NT];
  integer t=-1, errs=0, started=0; reg [BW-1:0] e0,e1,e2;
  always @(posedge clk) begin : oracle
    if (running && !started) begin started=1; t=-1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) begin
        hist0[t]=dut.bus_value_active[0]; hist1[t]=dut.bus_value_active[1]; hist2[t]=dut.bus_value_active[2];
      end
      if (t > 0) begin
        e0 = (t >= 5) ? hist0[t-5] : 10'd512;
        e1 = hist1[t];                       // d=0 passthrough
        e2 = (t >= 1) ? hist2[t-1] : 10'd512;
        if (bus_out[0*BW +: BW] !== e0) begin errs=errs+1;
          if (errs<=6) $display("  BUS0 t=%0d out=%0d exp=%0d", t, bus_out[0*BW +: BW], e0); end
        if (bus_out[1*BW +: BW] !== e1) begin errs=errs+1;
          if (errs<=6) $display("  BUS1 t=%0d out=%0d exp=%0d", t, bus_out[1*BW +: BW], e1); end
        if (bus_out[2*BW +: BW] !== e2) begin errs=errs+1;
          if (errs<=6) $display("  BUS2 t=%0d out=%0d exp=%0d", t, bus_out[2*BW +: BW], e2); end
      end
    end
  end

  initial begin
    wait(reset==1); wait(reset==0);
    repeat (NT) @(posedge clk);
    if (hist0[100] !== 10'd1023) $fatal(1, "the undelayed bus never stepped to 1023");
    $display("==== bus delay: %0d cycles checked, %0d mismatches ====", t, errs);
    $display("%s", (errs==0 && t > 1000) ? "BUS-DELAY-OK" : "**FAIL**");
    $finish;
  end
endmodule
