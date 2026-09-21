`timescale 1ns/1ps
`include "zlc_geometry.vh"
// DEFINITIVE: real zlc_period_streamer + the REAL row IP BRAM (true synthesized latency),
// preloaded through port A with the user's EXACT uploaded program (scaled /500), then FIRE
// and watch emCCD.  Needs the generated blk_mem_gen_rows simulation model from a build.
module tb_real_engine;
  localparam integer CH=`ZLC_NUM_DELAY_CH, RAW=`ZLC_ROW_ADDR_WIDTH, RW=`ZLC_ROW_WORDS, TW=32, NS=`ZLC_NUM_SLOTS;
  localparam integer SSW=`ZLC_SLOT_SEL_WIDTH, BUSC=`ZLC_BUS_COUNT, BW=`ZLC_BUS_WIDTH;
  localparam integer ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS, PBITS=RW*32;
  localparam integer NE=9;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;   // 50MHz

  reg [$clog2((1<<RAW)*RW)-1:0] wa=0; reg [31:0] wd=0; reg [3:0] we=0; reg wen=0;
  wire [RAW-1:0] row_raddr;
  wire [PBITS-1:0] row_rdata_w;
  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;

  blk_mem_gen_rows u_rows(
    .clka(clk),.ena(wen),.wea(we),.addra(wa),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web({(PBITS/8){1'b0}}),.addrb(row_raddr),.dinb({PBITS{1'b0}}),.doutb(row_rdata_w));

  zlc_period_streamer dut (
    .clk(clk),.reset(reset),.start(start),
    .prog_count(NE[RAW:0]),.run_repeat_count(32'd0),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata_w[RBITS-1:0]),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  integer i;
  reg [31:0] durs [0:NE-1]; reg [31:0] masks [0:NE-1];

  task pa_write;
    input [$clog2((1<<RAW)*RW)-1:0] a; input [31:0] d;
    begin
      @(posedge clk); wen<=1; we<=4'hF; wa<=a; wd<=d;
      @(posedge clk); @(posedge clk); wen<=0; we<=4'h0; @(posedge clk);
    end
  endtask

  initial begin
    // the old edge ticks 0,1000,2500,4500,5000,5001,10001,12001,14001,14101 as row durations
    durs[0]=1000; durs[1]=1500; durs[2]=2000; durs[3]=500; durs[4]=1;
    durs[5]=5000; durs[6]=2000; durs[7]=2000; durs[8]=100;
    masks[0]='h685; masks[1]='h200; masks[2]='ha08; masks[3]='h200; masks[4]='h200;
    masks[5]='h200; masks[6]='ha00; masks[7]='h208; masks[8]=0;
    reset=1; start=0; wen=0; we=0;
    // row word 0 = duration; word 1 = {actions..., mask, dsel=0} above bit 32 (mask at bit SSW)
    for (i=0;i<NE;i=i+1) begin
      pa_write(i*RW+0, durs[i]);
      pa_write(i*RW+1, masks[i] << SSW);
      pa_write(i*RW+2, 32'd0);
      pa_write(i*RW+3, 32'd0);
    end
    repeat (200) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  reg rprev=0; reg [CH-1:0] o_prev=0; integer tcount=0;
  always @(posedge clk) begin
    if (running && !rprev) $display("[RUNNING] start");
    rprev<=running;
    if (running && out!==o_prev) begin
      $display("  t=%0d raddr=%0d out=0x%h", tcount, row_raddr, out[15:0]);
      o_prev<=out;
    end
  end
  reg prev=0; reg em; integer lastOn=-1;
  always @(posedge clk) begin
    if (running||done) tcount=tcount+1;
    em=out[11];
    if (running && em!==prev) begin
      if (em) lastOn=tcount;
      else $display("[PULSE] on=%0d off=%0d width=%0d", lastOn, tcount, tcount-lastOn);
      prev=em;
    end
  end
  initial begin
    wait (reset==1); wait (reset==0);
    repeat (52000) @(posedge clk);
    $display("==== DONE (correct = seven emCCD pulses, every width 2000) ====");
    $finish;
  end
endmodule
