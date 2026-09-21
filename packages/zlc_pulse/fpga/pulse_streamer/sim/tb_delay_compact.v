`timescale 1ns/1ps
`include "zlc_geometry.vh"
// The compact TTL-only engine owns all 25 channels, each with its own delay FIFO.
// Channel ch gets ch ticks of delay: zero/one-tick bypasses and all six new pins,
// including bit 24, must preserve every edge across repeated frames.
module tb_delay_compact;
  localparam integer CH=`ZLC_NUM_DELAY_CH, EAW=12, TW=32, NS=1, CW=16, DTW=32, BUSC=4, BW=10;
  localparam integer NE=8;
  localparam integer NT=6000;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  reg [TW-1:0] etick [0:NE-1]; reg [CH-1:0] emask [0:NE-1];
  initial begin
    etick[0]=0;    emask[0]={CH{1'b1}};   // all on
    etick[1]=5;    emask[1]=0;
    etick[2]=6;    emask[2]={CH{1'b1}};   // 1-tick later back on (stress)
    etick[3]=7;    emask[3]=0;
    etick[4]=100;  emask[4]={CH{1'b1}};
    etick[5]=160;  emask[5]=0;
    etick[6]=2000; emask[6]={CH{1'b1}};
    etick[7]=2400; emask[7]=0;   // frame ends at 2400
  end

  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr[2:0]]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr[2:0]]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[2];
  wire [CH-1:0] edge_mask_rdata = mp[2];

  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  genvar ch;
  generate for (ch=0; ch<CH; ch=ch+1) begin : delays
    assign delay_ticks_w[ch*TDW +: TDW] = ch;
  end endgenerate

  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_edge_streamer #(
    .CHANNEL_COUNT(CH), .NUM_SLOTS(NS)
  ) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd8),.run_repeat_count(32'd0),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd2400),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd1),.scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(1'b0),.bus_prog_bus(2'd0),.bus_prog_addr(6'd0),.bus_prog_start_tick(32'd0),
    .bus_prog_stop_tick(32'd0),.bus_prog_start_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),.bus_prog_start_value(10'd0),
    .bus_prog_stop_value(10'd0),.bus_prog_mode(2'd0),.bus_prog_value_select(3'd0),
    .bus_prog_stop_value_select(3'd0),.bus_counts({BUSC*7{1'b0}}),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  initial begin
    reset=1; start=0;
    repeat (140) @(posedge clk);         // complete at least two 12-step ARM passes
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
