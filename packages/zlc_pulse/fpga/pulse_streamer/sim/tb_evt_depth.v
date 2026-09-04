`timescale 1ns/1ps
// EVENT-FIFO DEPTH BOUNDARY on the real engine: with EVT_DEPTH=16,
//   * a burst of EXACTLY 16 toggles inside one delay window is delayed
//     tick-exactly (the FIFO full-at-16 boundary must not corrupt anything);
//   * a burst of 18 toggles exceeds capacity and MUST set sticky overflow.
// The exact-depth lane remains tick-exact; the overflowing lane may drop data,
// but that corruption can no longer be silent.
// bit0 carries the 16-toggle burst, bit1 the 18-toggle burst; both delayed by
// d=200 so the whole burst is in flight at once.
module tb_evt_depth;
  localparam integer CH=8, EAW=12, TW=32, NS=1, CW=16, DTW=32, BUSC=4, BW=10;
  localparam integer NE=21;
  localparam integer NT=1200;
  localparam integer D=200;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  // toggle ticks 10,12,...: bit0 toggles at the first 16, bit1 at all 18.
  reg [TW-1:0] etick [0:31]; reg [CH-1:0] emask [0:31];
  integer i; reg b0; reg b1;
  initial begin
    etick[0]=0; emask[0]=8'h00;
    b0=0; b1=0;
    for (i=0; i<18; i=i+1) begin
      b1 = ~b1;
      if (i<16) b0 = ~b0;
      etick[1+i] = 10 + 2*i;
      emask[1+i] = {6'b0, b1, b0};
    end
    etick[19]=500; emask[19]=8'h00;   // frame end (all off)
    etick[20]=501; emask[20]=8'h00;   // final off edge
    for (i=21; i<32; i=i+1) begin etick[i]=0; emask[i]=0; end
  end

  wire [EAW-1:0] edge_raddr;
  reg [TW-1:0] tp[0:2]; reg [CH-1:0] mp[0:2];
  always @(posedge clk) begin
    tp[0]<=etick[edge_raddr[4:0]]; tp[1]<=tp[0]; tp[2]<=tp[1];
    mp[0]<=emask[edge_raddr[4:0]]; mp[1]<=mp[0]; mp[2]<=mp[1];
  end
  wire [TW-1:0] edge_tick_rdata = tp[2];
  wire [CH-1:0] edge_mask_rdata = mp[2];

  localparam integer TDW = 32;
  wire [CH*TDW-1:0] delay_ticks_w;
  assign delay_ticks_w[0*TDW +: TDW] = D;       // 16-toggle burst, exactly depth
  assign delay_ticks_w[1*TDW +: TDW] = D;       // 18-toggle burst, overflow by 2
  assign delay_ticks_w[CH*TDW-1: 2*TDW] = {(CH-2)*TDW{1'b0}};

  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done, overflow, physical_active; wire [31:0] scan_cursor; wire underflow;
  zlc_edge_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .EVT_DEPTH(16)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(13'd21),.run_repeat_count(32'd1),
    .loop_start_addr({EAW{1'b0}}),.loop_end_tick(32'd501),.loop_end_coeffs({NS*CW{1'b0}}),
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
    .out(out),.bus_out(bus_out),.running(running),.done(done),
    .overflow(overflow),.physical_active(physical_active));

  initial begin
    reset=1; start=0;
    repeat (200) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  // oracle: history of the undelayed stream + per-cycle asserts.
  // ch0 (16 toggles = depth): out0[t] == in0[t-D] EXACTLY.
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
        $fatal(1, "timeout waiting for overflow run DONE (running=%b draining=%b busy=%b time=%0d final=%0d edge=%0d cnt0=%0d cnt1=%0d out=%h)",
               running, dut.draining, dut.delay_runtime_busy,
               dut.time_count, dut.final_tick, dut.edge_index,
               dut.g_evtfifo[0].cnt, dut.g_evtfifo[1].cnt, out);
      end
    join
  end
endmodule
