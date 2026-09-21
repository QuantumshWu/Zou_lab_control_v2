`timescale 1ns/1ps
// TTL EVENT-SCHEDULER delay verification on the REAL engine: channels with delays
// {0, 1, 2, 7, 1000} ticks and a finite dense-toggle program (incl. 1-tick rows),
// with a behavioral aligned-latency row memory.  ORACLE: record the engine's
// UNDELAYED stream (dut.state_mask) every cycle and assert out[t] == in[t-d] for
// every delayed channel on every cycle (0 before t=d), including the physical tail.
module tb_delay_sched;
  localparam integer CH=8, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1, SSW=`ZLC_SLOT_SEL_WIDTH;
  localparam integer BUSC=4, BW=10, ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NE=7;
  localparam integer NT=12000;          // > 3 frames + the 1000-tick delayed tail
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask;
    begin row_of = {{(BUSC*ABITS){1'b0}}, mask, {SSW{1'b0}}, dur}; end
  endfunction

  // rows: dense toggles on bits 0..4 (all channels share the same waveform so each
  // delayed channel can be checked against the same undelayed reference bit).
  reg [RBITS-1:0] rowmem [0:7];
  initial begin
    rowmem[0]=row_of(32'd5,    8'h1F);   // all five test channels ON at t=0
    rowmem[1]=row_of(32'd1,    8'h00);   // off for one tick
    rowmem[2]=row_of(32'd1,    8'h1F);   // 1-tick pulse (stress)
    rowmem[3]=row_of(32'd93,   8'h00);
    rowmem[4]=row_of(32'd60,   8'h1F);
    rowmem[5]=row_of(32'd1840, 8'h00);
    rowmem[6]=row_of(32'd400,  8'h1F);   // frame ends at 2400
    rowmem[7]=0;
  end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[2:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  // per-channel delays: ch0 d=0, ch1 d=1, ch2 d=2, ch3 d=7, ch4 d=1000 (32b fields)
  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  assign delay_ticks_w[0*TDW +: TDW] = 32'd0;
  assign delay_ticks_w[1*TDW +: TDW] = 32'd1;
  assign delay_ticks_w[2*TDW +: TDW] = 32'd2;
  assign delay_ticks_w[3*TDW +: TDW] = 32'd7;
  assign delay_ticks_w[4*TDW +: TDW] = 32'd1000;
  assign delay_ticks_w[CH*TDW-1: 5*TDW] = {(CH-5)*TDW{1'b0}};

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

  // oracle: undelayed history (sampled value DURING each cycle) + per-cycle asserts
  reg [CH-1:0] hist [0:NT];           // hist[t] = undelayed state_mask during cycle t
  integer t = -1; integer errs = 0; integer started = 0;
  integer dch; integer dexp;
  always @(posedge clk) begin : oracle
    integer di; reg expected;
    if (running && !started) begin started = 1; t = -1; end
    if (started) begin
      t = t + 1;
      if (t <= NT) hist[t] = dut.state_mask;
      // check the four delayed channels (the undelayed waveform is bit 0 of hist)
      for (di = 1; di <= 4; di = di + 1) begin
        dexp = (di==1) ? 1 : (di==2) ? 2 : (di==3) ? 7 : 1000;
        expected = (t >= dexp && (t-dexp) <= NT) ? hist[t-dexp][0] : 1'b0;
        if (t > 0 && out[di] !== expected) begin
          errs = errs + 1;
          if (errs <= 8)
            $display("  MISMATCH t=%0d ch=%0d d=%0d out=%b expect=%b", t, di, dexp, out[di], expected);
        end
      end
      // d=0 channel: passthrough equality
      if (t > 0 && out[0] !== dut.state_mask[0]) begin
        errs = errs + 1;
        if (errs <= 8) $display("  MISMATCH d0 t=%0d", t);
      end
    end
  end

  integer saw_drain = 0; integer saw_tail = 0;
  always @(posedge clk) begin
    if (dut.draining) begin
      saw_drain = 1;
      if (|out) saw_tail = 1;
      if (done) $fatal(1, "DONE asserted while delayed outputs were draining");
      if (!physical_active) $fatal(1, "physical_active dropped during delayed tail");
    end
  end
  initial begin
    wait(reset==1); wait(reset==0);
    fork
      begin
        wait(done);
        @(posedge clk);
        if (errs != 0) $fatal(1, "delay scheduler had %0d mismatches", errs);
        if (!saw_drain || !saw_tail) $fatal(1, "finite run did not exercise a non-empty delayed tail");
        if (physical_active) $fatal(1, "physical_active remained high after DONE");
        if (out !== {CH{1'b0}}) $fatal(1, "TTL pins were not safe at DONE: %h", out);
        if (bus_out !== {BUSC{10'd512}}) $fatal(1, "DAC pins were not safe at DONE: %h", bus_out);
        if (underflow || overflow) $fatal(1, "unexpected sticky engine error at DONE");
        $display("DELAY-SCHED-PHYSICAL-DONE-OK cycles=%0d", t);
        $finish;
      end
      begin
        repeat (NT) @(posedge clk);
        $fatal(1, "timeout waiting for physical DONE");
      end
    join
  end
endmodule
