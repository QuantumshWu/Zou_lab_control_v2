`timescale 1ns/1ps
`include "zlc_geometry.vh"
// FULL-CHAIN first-frame test: the REAL zlc_pulse_streamer_top + REAL engine + the FIVE REAL
// blk_mem_gen IPs, with a frozen current-layout host word image (9 periods,
// da_bias_y = edge -192(code320)@P0, edge 388(code900)@P1, HOLD after; one
// frame = 116 ticks).  Replays the host
// flow twice: one upload/LOAD then resident FIRE after SAFE.  Also interrupts a
// pending LOAD and retries a FIRE with the same execution ID (lost ACK).
// This covers everything an engine-only TB bypasses: ctrl regfile, mini-loader, command
// sequencing, clk mux, pin map.

// ---- fake JTAG master: tied off ----
module jtag_axi_0(
  input aclk, input aresetn,
  output [0:0] m_axi_awid, output [31:0] m_axi_awaddr, output [7:0] m_axi_awlen,
  output [2:0] m_axi_awsize, output [1:0] m_axi_awburst, output [0:0] m_axi_awlock,
  output [3:0] m_axi_awcache, output [2:0] m_axi_awprot, output [3:0] m_axi_awqos,
  output m_axi_awvalid, input m_axi_awready,
  output [31:0] m_axi_wdata, output [3:0] m_axi_wstrb, output m_axi_wlast,
  output m_axi_wvalid, input m_axi_wready,
  input [0:0] m_axi_bid, input [1:0] m_axi_bresp, input m_axi_bvalid, output m_axi_bready,
  output [0:0] m_axi_arid, output [31:0] m_axi_araddr, output [7:0] m_axi_arlen,
  output [2:0] m_axi_arsize, output [1:0] m_axi_arburst, output [0:0] m_axi_arlock,
  output [3:0] m_axi_arcache, output [2:0] m_axi_arprot, output [3:0] m_axi_arqos,
  output m_axi_arvalid, input m_axi_arready,
  input [0:0] m_axi_rid, input [31:0] m_axi_rdata, input [1:0] m_axi_rresp,
  input m_axi_rlast, input m_axi_rvalid, output m_axi_rready
);
  assign m_axi_awid=0; assign m_axi_awaddr=0; assign m_axi_awlen=0; assign m_axi_awsize=0;
  assign m_axi_awburst=0; assign m_axi_awlock=0; assign m_axi_awcache=0; assign m_axi_awprot=0;
  assign m_axi_awqos=0; assign m_axi_awvalid=0; assign m_axi_wdata=0; assign m_axi_wstrb=0;
  assign m_axi_wlast=0; assign m_axi_wvalid=0; assign m_axi_bready=0;
  assign m_axi_arid=0; assign m_axi_araddr=0; assign m_axi_arlen=0; assign m_axi_arsize=0;
  assign m_axi_arburst=0; assign m_axi_arlock=0; assign m_axi_arcache=0; assign m_axi_arprot=0;
  assign m_axi_arqos=0; assign m_axi_arvalid=0; assign m_axi_rready=0;
endmodule

// ---- scripted bram writer in axi_bram_ctrl_0's place ----
module axi_bram_ctrl_0(
  input s_axi_aclk, input s_axi_aresetn,
  input [0:0] s_axi_awid, input [31:0] s_axi_awaddr, input [7:0] s_axi_awlen,
  input [2:0] s_axi_awsize, input [1:0] s_axi_awburst, input [0:0] s_axi_awlock,
  input [3:0] s_axi_awcache, input [2:0] s_axi_awprot, input s_axi_awvalid, output s_axi_awready,
  input [31:0] s_axi_wdata, input [3:0] s_axi_wstrb, input s_axi_wlast,
  input s_axi_wvalid, output s_axi_wready,
  output [0:0] s_axi_bid, output [1:0] s_axi_bresp, output s_axi_bvalid, input s_axi_bready,
  input [0:0] s_axi_arid, input [31:0] s_axi_araddr, input [7:0] s_axi_arlen,
  input [2:0] s_axi_arsize, input [1:0] s_axi_arburst, input [0:0] s_axi_arlock,
  input [3:0] s_axi_arcache, input [2:0] s_axi_arprot, input s_axi_arvalid, output s_axi_arready,
  output [0:0] s_axi_rid, output [31:0] s_axi_rdata, output [1:0] s_axi_rresp,
  output s_axi_rlast, output s_axi_rvalid, input s_axi_rready,
  output bram_rst_a, output bram_clk_a, output reg bram_en_a,
  output reg [3:0] bram_we_a, output reg [31:0] bram_addr_a,
  output reg [31:0] bram_wrdata_a, input [31:0] bram_rddata_a
);
  assign s_axi_awready=0; assign s_axi_wready=0; assign s_axi_bid=0; assign s_axi_bresp=0;
  assign s_axi_bvalid=0; assign s_axi_arready=0; assign s_axi_rid=0; assign s_axi_rdata=0;
  assign s_axi_rresp=0; assign s_axi_rlast=0; assign s_axi_rvalid=0;
  assign bram_rst_a = 1'b0;
  assign bram_clk_a = s_axi_aclk;

  task wr;
    input [29:0] word; input [31:0] data;
    begin
      @(negedge s_axi_aclk);
      bram_en_a <= 1'b1; bram_we_a <= 4'hF;
      bram_addr_a <= {word, 2'b00}; bram_wrdata_a <= data;
      @(negedge s_axi_aclk);
      bram_en_a <= 1'b0; bram_we_a <= 4'h0;
    end
  endtask
  integer command_identity = 0;
  task issue_cmd;
    input [31:0] x;
    begin
      command_identity = command_identity + 1;
      wr(30'd22, command_identity); wr(30'd1, 32'd0); wr(30'd1, x);
    end
  endtask
  task await_ack;
    input [31:0] expected;
    integer remaining;
    begin
      @(negedge s_axi_aclk); bram_en_a=1; bram_we_a=0; bram_addr_a=23*4;
      remaining=10000;
      repeat(3) @(negedge s_axi_aclk);
      while (bram_rddata_a !== command_identity && remaining > 0) begin
        @(negedge s_axi_aclk); remaining=remaining-1;
      end
      if (remaining==0) $fatal(1,"command %0d did not complete", command_identity);
      bram_addr_a=24*4; repeat(3) @(negedge s_axi_aclk);
      if (bram_rddata_a !== expected) $fatal(1,"command %0d result %h != %h",command_identity,bram_rddata_a,expected);
      bram_en_a=0;
    end
  endtask
  task cmd;
    input [31:0] x;
    begin issue_cmd(x); await_ack(x==8 || x==4 ? 0 : x); end
  endtask
  task upload;
    begin
`include "replay_t.vh"
    end
  endtask
  task prepare_and_fire;
    begin
      cmd(32'd8);                       // CMD_SAFE
      upload;
      // SAFE must preempt the active mini-loader, not wait behind it.
      issue_cmd(32'd1);
      cmd(32'd8);
      cmd(32'd1);                       // CMD_LOAD
      wr(30'd6, 32'd4);                 // four whole-Pulse shots
      wr(30'd16, 32'd3);                // BANK_READY
      cmd(32'd2);                       // CMD_FIRE
      // Same command ID, as when its completion reply was lost: never refire.
      wr(30'd1, 32'd0); wr(30'd1, 32'd2); await_ack(32'd2);
      $display("[TB] FIRE issued at %0t", $time);
    end
  endtask

  initial begin
    bram_en_a=0; bram_we_a=0; bram_addr_a=0; bram_wrdata_a=0;
    repeat (50) @(negedge s_axi_aclk);
    prepare_and_fire;                   // on_pulse #1
    repeat (5 * 116 * 1 + 2000) @(negedge s_axi_aclk);   // ~4+ frames
    cmd(32'd8);
    cmd(32'd2);                         // resident replay: no upload and no LOAD
  end
endmodule

`ifdef ZLC_IVERILOG
// Icarus-only black boxes for the pin-boundary test below.  The test forces
// the engine-side signals and checks the real top-level pin equations; BRAM
// behaviour is deliberately outside this small SAFE-gate proof.
module blk_mem_gen_edge_tick(
  input clka, input ena, input [3:0] wea, input [11:0] addra, input [31:0] dina, output [31:0] douta,
  input clkb, input enb, input [3:0] web, input [11:0] addrb, input [31:0] dinb, output [31:0] doutb);
  assign douta=0; assign doutb=0;
endmodule
module blk_mem_gen_edge_coeff(
  input clka, input ena, input [3:0] wea, input [12:0] addra, input [31:0] dina, output [31:0] douta,
  input clkb, input enb, input [7:0] web, input [11:0] addrb, input [63:0] dinb, output [63:0] doutb);
  assign douta=0; assign doutb=0;
endmodule
module blk_mem_gen_edge_mask(
  input clka, input ena, input [3:0] wea, input [12:0] addra, input [31:0] dina, output [31:0] douta,
  input clkb, input enb, input [7:0] web, input [11:0] addrb, input [63:0] dinb, output [63:0] doutb);
  assign douta=0; assign doutb=0;
endmodule
module blk_mem_gen_scan(
  input clka, input ena, input [3:0] wea, input [13:0] addra, input [31:0] dina, output [31:0] douta,
  input clkb, input enb, input [15:0] web, input [11:0] addrb, input [127:0] dinb, output [127:0] doutb);
  assign douta=0; assign doutb=0;
endmodule
module blk_mem_gen_busimg(
  input clka, input ena, input [3:0] wea, input [10:0] addra, input [31:0] dina, output [31:0] douta,
  input clkb, input enb, input [3:0] web, input [10:0] addrb, input [31:0] dinb, output [31:0] doutb);
  assign douta=0; assign doutb=0;
endmodule

module tb_safe_gate;
  reg clk=0; always #10 clk=~clk;
  wire [1:0] led;
  wire cooling,cooling_pgc,repump,probe,pushout,state_pre,trig,coil,grey_cooling,trap,UV,emCCD;
  wire microwave,address_w,cooling_shutter,repump_shutter,probe_shutter,bias;
  wire GND1,pgc_1D,GND5,GND6,GND7,GND8,GND9,GND10,GND11,GND12,GND13,GND14,GND15;
  wire [9:0] da_dipole,da_bias_y,da_bias_x,da_bias_z;
  wire da_clk0,da_clk1,da_clk2,da_clk3,uart_tx;
  zlc_pulse_streamer_top dut(
    .clk(clk),.uart_rx(1'b1),.uart_tx(uart_tx),.led(led),
    .cooling(cooling),.cooling_pgc(cooling_pgc),.repump(repump),.probe(probe),
    .pushout(pushout),.state_pre(state_pre),.trig(trig),.coil(coil),.grey_cooling(grey_cooling),
    .trap(trap),.UV(UV),.emCCD(emCCD),.microwave(microwave),.address(address_w),
    .GND1(GND1),.pgc_1D(pgc_1D),.GND5(GND5),.GND6(GND6),.GND7(GND7),.GND8(GND8),
    .GND9(GND9),.GND10(GND10),.GND11(GND11),.cooling_shutter(cooling_shutter),
    .GND12(GND12),.repump_shutter(repump_shutter),.GND13(GND13),.probe_shutter(probe_shutter),
    .GND14(GND14),.bias(bias),.GND15(GND15),.da_dipole(da_dipole),.da_clk0(da_clk0),
    .da_bias_y(da_bias_y),.da_clk1(da_clk1),.da_bias_x(da_bias_x),.da_clk2(da_clk2),
    .da_bias_z(da_bias_z),.da_clk3(da_clk3));

  task expect_safe; begin
    #1;
    if ({da_clk3,da_clk2,da_clk1,da_clk0} !== 4'b0000)
      $fatal(1,"DAC clocks were not gated safe: %b",{da_clk3,da_clk2,da_clk1,da_clk0});
    if ({da_bias_z,da_bias_x,da_bias_y,da_dipole} !== {4{10'd512}})
      $fatal(1,"DAC data was not midpoint-safe");
  end endtask

  initial begin
    force dut.clk_en = {`ZLC_CHANNEL_COUNT{1'b1}};
    force dut.out = {`ZLC_CHANNEL_COUNT{1'b0}};
    force dut.zlc_bus_out = {4{10'd37}};
    force dut.zlc_physical_active = 1'b1;
    force dut.eng_reset = 1'b1;
    repeat(3) begin @(posedge clk); expect_safe; @(negedge clk); expect_safe; end

    force dut.eng_reset = 1'b0;
    @(posedge clk); #1;
    if ({da_clk3,da_clk2,da_clk1,da_clk0} !== 4'b0000)
      $fatal(1,"inverted DAC clocks had the wrong active phase");
    @(negedge clk); #1;
    if ({da_clk3,da_clk2,da_clk1,da_clk0} !== 4'b1111)
      $fatal(1,"enabled DAC clocks did not toggle while physically active");

    // A single SAFE assertion dominates an active engine and arbitrary bus data.
    force dut.eng_reset = 1'b1;
    repeat(3) begin @(posedge clk); expect_safe; @(negedge clk); expect_safe; end

    // Decoder faults remain visible without poisoning the pulse engine: a
    // recovered UART retry latches LINK_ERROR, never ENGINE_ERROR.
    force dut.u_protocol_error = 1'b1;
    @(posedge clk); #1; release dut.u_protocol_error;
    repeat(3) @(posedge clk);
    if (!dut.ctrl_reg[2][5] || dut.ctrl_reg[2][3])
      $fatal(1,"UART protocol fault was not isolated in STATUS.LINK_ERROR");
    $display("TOP-SAFE-PIN-GATE-OK");
    $finish;
  end
endmodule
`endif

// ---- the testbench ----
module tb_t_ff;
`include "replay_t_frame.vh"
  localparam integer NFR = 4;                 // frames captured per fire
  reg clk = 0; always #10 clk = ~clk;
  wire [1:0] led;
  wire cooling, cooling_pgc, repump, probe, pushout, state_pre, trig, coil;
  wire grey_cooling, trap, UV, emCCD, microwave, address_w;
  wire GND1,pgc_1D,GND5,GND6,GND7,GND8,GND9,GND10,GND11,GND12,GND13,GND14,GND15;
  wire cooling_shutter, repump_shutter, probe_shutter, bias;
  wire [9:0] da_dipole, da_bias_y, da_bias_x, da_bias_z;
  wire da_clk0, da_clk1, da_clk2, da_clk3;

  // The top's BANK_SIZE default (2048) == streamer_config.json == the real bitstream's
  // geometry (geom.tcl) == the committed replay_t.vh layout.  A
  // mismatched override here would land the bus image in the scan region -- the loader
  // would copy zeros and ALL DA output would be silently wrong (we demonstrated exactly
  // that with a deliberate 512-vs-2048 skew).
  zlc_pulse_streamer_top dut (
    .clk(clk), .led(led), .uart_rx(1'b1),
    .cooling(cooling), .cooling_pgc(cooling_pgc), .repump(repump), .probe(probe),
    .pushout(pushout), .state_pre(state_pre), .trig(trig), .coil(coil),
    .grey_cooling(grey_cooling), .trap(trap), .UV(UV), .emCCD(emCCD),
    .microwave(microwave), .address(address_w),
    .GND1(GND1),.pgc_1D(pgc_1D),.GND5(GND5),.GND6(GND6),.GND7(GND7),.GND8(GND8),
    .GND9(GND9),.GND10(GND10),.GND11(GND11),
    .cooling_shutter(cooling_shutter), .GND12(GND12), .repump_shutter(repump_shutter),
    .GND13(GND13), .probe_shutter(probe_shutter), .GND14(GND14), .bias(bias), .GND15(GND15),
    .da_dipole(da_dipole), .da_clk0(da_clk0),
    .da_bias_y(da_bias_y), .da_clk1(da_clk1),
    .da_bias_x(da_bias_x), .da_clk2(da_clk2),
    .da_bias_z(da_bias_z), .da_clk3(da_clk3)
  );

  // capture da_bias_y + cooling per running tick, per fire
  integer ti = -1, fire_n = 0; reg run_prev = 0;
  reg [9:0] bh [0:2*NFR*200];      // [fire*NFR*T_FRAME + t]
  reg       chh [0:2*NFR*200];
  reg [39:0] all_bus [0:2*NFR*200];
  reg [18:0] all_ttl [0:2*NFR*200];
  always @(posedge clk) begin
    if (led[0] && !run_prev) begin
      $display("[TB] running (fire #%0d) at %0t", fire_n, $time);
      ti = 0;
    end else if (led[0] && ti >= 0) ti = ti + 1;
    if (!led[0] && run_prev) begin fire_n = fire_n + 1; ti = -1; end
    if (led[0] && ti >= 0 && ti < NFR*T_FRAME)
      begin
        bh[fire_n*NFR*T_FRAME + ti] <= da_bias_y; chh[fire_n*NFR*T_FRAME + ti] <= cooling;
        all_bus[fire_n*NFR*T_FRAME + ti] <= {da_bias_z,da_bias_x,da_bias_y,da_dipole};
        all_ttl[fire_n*NFR*T_FRAME + ti] <= {pgc_1D,bias,probe_shutter,repump_shutter,cooling_shutter,address_w,microwave,emCCD,UV,trap,grey_cooling,coil,trig,state_pre,pushout,probe,repump,cooling_pgc,cooling};
      end
    run_prev <= led[0];
  end
  always @(negedge clk) begin
    #1;
    if (led[0] && {da_clk3,da_clk2,da_clk1,da_clk0} !== 4'b1111)
      $fatal(1,"a running replay lost its DAC latch clocks");
  end

  integer f, k, base, prev, errs;
  task report_fire;
    input integer fn;
    begin
      $display("---- fire #%0d ----", fn);
      for (f = 0; f < NFR; f = f + 1) begin
        base = fn*NFR*T_FRAME + f*T_FRAME;
        $write("F%0d bias_y:", f);
        prev = -1;
        for (k = 0; k < T_FRAME; k = k + 1)
          if (bh[base+k] !== prev[9:0]) begin $write(" %0d@%0d", bh[base+k], k); prev = bh[base+k]; end
        $write("   cooling:");
        prev = -1;
        for (k = 0; k < T_FRAME; k = k + 1)
          if ({31'b0,chh[base+k]} !== prev) begin $write(" %0d@%0d", chh[base+k], k); prev = {31'b0,chh[base+k]}; end
        $write("\n");
      end
      errs = 0;
      for (k = 0; k < T_FRAME; k = k + 1) begin
        if (bh[fn*NFR*T_FRAME + 0*T_FRAME + k] !== bh[fn*NFR*T_FRAME + 1*T_FRAME + k]) errs = errs + 1;
        if (bh[fn*NFR*T_FRAME + 1*T_FRAME + k] !== bh[fn*NFR*T_FRAME + 2*T_FRAME + k]) errs = errs + 1;
        if (chh[fn*NFR*T_FRAME + 0*T_FRAME + k] !== chh[fn*NFR*T_FRAME + 1*T_FRAME + k]) errs = errs + 1;
      end
      $display("fire #%0d: F0-vs-F1 + F1-vs-F2 mismatching ticks = %0d  %s",
               fn, errs, (errs == 0) ? "T-FF-OK" : "**T-FF-BAD**");
    end
  endtask

  initial begin
    // fire #1: ~50+300+upload+600+fire, frames 4*116; fire #2 same again
    repeat (50 + 1000 + 600 + 5*T_FRAME + 2000 + 1000 + 600 + 5*T_FRAME + 2000) @(posedge clk);
    report_fire(0);
    report_fire(1);
    if (fire_n != 2) $fatal(1,"expected two physical executions, got %0d",fire_n);
    for (k=0;k<NFR*T_FRAME;k=k+1) begin
      if ($isunknown(all_bus[k]) || all_bus[k] !== all_bus[NFR*T_FRAME+k])
        $fatal(1,"resident DAC output differs at tick %0d",k);
      if ($isunknown(all_ttl[k]) || all_ttl[k] !== all_ttl[NFR*T_FRAME+k])
        $fatal(1,"resident TTL output differs at tick %0d",k);
    end
    $display("RESIDENT-REPLAY-SAFE-INTERRUPT-DEDUP-OK");
    $display("==== DONE ====");
    $finish;
  end
endmodule
