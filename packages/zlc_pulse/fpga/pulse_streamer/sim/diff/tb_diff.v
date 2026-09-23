`timescale 1ns/1ps
`include "zlc_geometry.vh"
// DIFFERENTIAL pin dump: the REAL zlc_pulse_streamer_top + REAL engine + the REAL blk_mem_gen IPs
// of ONE side (edge-table or period-table build), fed the host image that side's own packer made
// for the same authored pulse.  Uploads the image, LOADs, sets the run count, FIREs, and writes
// every running clock's pins (25 TTL, 4x10 DAC data, 4 DAC clocks) to a file until the run ends.
// Two dumps -- one per engine -- are then compared tick for tick outside the simulator.
//   plusargs:  +image=<file of "addr hexdata" lines>  +runs=<RUN_REPEAT_COUNT>  +dump=<output file>

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

  reg [8*256:1] image_path, unused_dump;
  integer runs, bank_ready;
  task upload;
    integer fd, code, a, v, n;
    begin
      fd = $fopen(image_path, "r");
      if (fd == 0) $fatal(1, "cannot open image file");
      n = 0;
      code = $fscanf(fd, "%d %h\n", a, v);
      while (code == 2) begin
        wr(a[29:0], v); n = n + 1;
        code = $fscanf(fd, "%d %h\n", a, v);
      end
      $fclose(fd);
      $display("[TB] uploaded %0d words", n);
    end
  endtask

  integer fargs;
  initial begin
    bram_en_a=0; bram_we_a=0; bram_addr_a=0; bram_wrdata_a=0;
    // xsim.bat splits an argument at '=', so the run's parameters come from
    // a file in the working directory: image path, run count, dump path.
    fargs = $fopen("diff_args.txt", "r");
    if (fargs == 0) $fatal(1, "diff_args.txt missing");
    if ($fscanf(fargs, "%s\n", image_path) != 1) $fatal(1, "diff_args.txt: no image path");
    if ($fscanf(fargs, "%d\n", runs) != 1) $fatal(1, "diff_args.txt: no run count");
    if ($fscanf(fargs, "%s\n", unused_dump) != 1) $fatal(1, "diff_args.txt: no dump path");
    if ($fscanf(fargs, "%d\n", bank_ready) != 1) bank_ready = 3;
    $fclose(fargs);
    repeat (50) @(negedge s_axi_aclk);
    cmd(32'd8);                       // CMD_SAFE
    upload;
    cmd(32'd1);                       // CMD_LOAD
    wr(30'd6, runs);                  // RUN_REPEAT_COUNT: whole-Pulse shots
    wr(30'd16, bank_ready);           // BANK_READY, as the device arms it
    cmd(32'd2);                       // CMD_FIRE
    $display("[TB] FIRE acknowledged at %0t", $time);
  end
endmodule

// ---- the dump bench ----
module tb_diff;
  reg clk = 0; always #10 clk = ~clk;
  wire [1:0] led;
  wire cooling, shutter_420, repump, probe, pushout, state_pre, trig, coil;
  wire grey_cooling, trap, UV, emCCD, microwave, address_w;
  wire GND1,pgc_1D,push_shutter,single_cooling_shutter,cooling_pgc,sweep_trig,push_freq_switch,pgc_1D_freq_switch;
  wire GND11,GND12,GND13,GND14,GND15;
  wire cooling_shutter, repump_shutter, probe_shutter, bias;
  wire [9:0] da_dipole, da_bias_y, da_bias_x, da_bias_z;
  wire da_clk0, da_clk1, da_clk2, da_clk3;
  wire uart_tx;

  zlc_pulse_streamer_top dut (
    .clk(clk), .led(led), .uart_rx(1'b1), .uart_tx(uart_tx),
    .cooling(cooling), .shutter_420(shutter_420), .repump(repump), .probe(probe),
    .pushout(pushout), .state_pre(state_pre), .trig(trig), .coil(coil),
    .grey_cooling(grey_cooling), .trap(trap), .UV(UV), .emCCD(emCCD),
    .microwave(microwave), .address(address_w),
    .GND1(GND1),.pgc_1D(pgc_1D),.push_shutter(push_shutter),.single_cooling_shutter(single_cooling_shutter),
    .cooling_pgc(cooling_pgc),.sweep_trig(sweep_trig),.push_freq_switch(push_freq_switch),
    .pgc_1D_freq_switch(pgc_1D_freq_switch),.GND11(GND11),
    .cooling_shutter(cooling_shutter), .GND12(GND12), .repump_shutter(repump_shutter),
    .GND13(GND13), .probe_shutter(probe_shutter), .GND14(GND14), .bias(bias), .GND15(GND15),
    .da_dipole(da_dipole), .da_clk0(da_clk0),
    .da_bias_y(da_bias_y), .da_clk1(da_clk1),
    .da_bias_x(da_bias_x), .da_clk2(da_clk2),
    .da_bias_z(da_bias_z), .da_clk3(da_clk3)
  );

  wire [24:0] ttl_pins = {pgc_1D_freq_switch,push_freq_switch,sweep_trig,cooling_pgc,single_cooling_shutter,push_shutter,
    pgc_1D,bias,probe_shutter,repump_shutter,cooling_shutter,address_w,microwave,emCCD,UV,trap,grey_cooling,
    coil,trig,state_pre,pushout,probe,repump,shutter_420,cooling};

  reg [8*256:1] dump_path, skip_path;
  integer fdump = 0, ticks = 0, fargs2, skip_runs;
  reg running_seen = 0;
  initial begin
    fargs2 = $fopen("diff_args.txt", "r");
    if (fargs2 == 0) $fatal(1, "diff_args.txt missing");
    if ($fscanf(fargs2, "%s\n", skip_path) != 1) $fatal(1, "diff_args.txt: no image path");
    if ($fscanf(fargs2, "%d\n", skip_runs) != 1) $fatal(1, "diff_args.txt: no run count");
    if ($fscanf(fargs2, "%s\n", dump_path) != 1) $fatal(1, "diff_args.txt: no dump path");
    $fclose(fargs2);
    fdump = $fopen(dump_path, "w");
    if (fdump == 0) $fatal(1, "cannot open dump file");
  end
  // Sample on the rising edge: the values the clock finds on the pins.
  always @(posedge clk) begin
    if (led[0]) begin
      if (!running_seen) $display("[TB] running from %0t", $time);
      running_seen = 1;
      $fwrite(fdump, "%07x %010x %x\n", ttl_pins,
              {da_bias_z, da_bias_x, da_bias_y, da_dipole},
              {da_clk3, da_clk2, da_clk1, da_clk0});
      ticks = ticks + 1;
    end else if (running_seen) begin
      // a few cycles after the run: what the pins settle to
      $fwrite(fdump, "%07x %010x %x\n", ttl_pins,
              {da_bias_z, da_bias_x, da_bias_y, da_dipole},
              {da_clk3, da_clk2, da_clk1, da_clk0});
      ticks = ticks + 1;
      if (ticks > 0 && !led[0]) begin
        $fclose(fdump);
        $display("DIFF-DUMP-DONE ticks=%0d end=%0t", ticks, $time);
        $finish;
      end
    end
  end
  initial begin
    #(20 * 4000000);
    $fatal(1, "timeout: the run never finished");
  end
endmodule
