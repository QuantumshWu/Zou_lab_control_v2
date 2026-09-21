`timescale 1ns/1ps
// Does the DA-bus delay behave IDENTICALLY to a TTL-channel delay?  (User: "DA delay 和
// TTL delay 是一个机制".)  Drive the REAL engine with a TTL level on ch5 and a DA-bus edge
// on bus0, BOTH in the row that starts at tick T0, and the SAME per-output delay D.  Record
// out[5] and bus_out[bus0] every tick and assert: (a) at D=0 they change on the SAME tick
// (= T0), (b) at D=N both change on the SAME tick (= T0+N) -- i.e. the DA delay shifts
// exactly like the TTL delay, no extra offset, no "delay does nothing".
module tb_da_ttl_align;
  localparam integer CH=8, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1, SSW=`ZLC_SLOT_SEL_WIDTH;
  localparam integer BUSC=4, BW=10, ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, TDW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer NT=400, T0=40, TTLBIT=5;
  localparam [BW-1:0] VA=10'd200, VB=10'd800;   // bus0 steps VA->VB at T0
  reg clk=0; always #10 clk=~clk;
  reg reset, start;
  integer DUT_D;                                 // delay under test (set per run)

  function [ABITS-1:0] act;
    input [1:0] mode; input [SSW-1:0] sel; input [BW-1:0] v;
    begin act = {mode, sel, v}; end
  endfunction
  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask; input [ABITS-1:0] bus0;
    begin row_of = {{((BUSC-1)*ABITS){1'b0}}, bus0, mask, {SSW{1'b0}}, dur}; end
  endfunction

  // rows: r0 [0,T0) mask0 bus0=VA; r1 [T0,T0+40) ch5 high, bus0=VB; r2 to 300 mask0
  reg [RBITS-1:0] rowmem [0:3];
  initial begin
    rowmem[0]=row_of(T0[TW-1:0], 8'h00, act(2'd1, 0, VA));
    rowmem[1]=row_of(32'd40, (8'h1<<TTLBIT), act(2'd1, 0, VB));
    rowmem[2]=row_of(32'd220, 8'h00, act(2'd0, 0, 0));
    rowmem[3]=0;
  end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[1:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];

  // ---- per-output delays: ch5 (TTL) and bus0 (DA) BOTH = DUT_D, everything else 0 ----
  reg [CH*TDW-1:0]   delay_ticks_w   = {CH*TDW{1'b0}};
  reg [BUSC*TDW-1:0] bus_delay_ticks_w = {BUSC*TDW{1'b0}};

  wire [`ZLC_SCAN_ADDR_WIDTH-1:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  zlc_period_streamer #(.CHANNEL_COUNT(CH), .NUM_SLOTS(NS), .BUS_COUNT(BUSC), .BUS_WIDTH(BW)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(3),.run_repeat_count(32'd0),
    .scan_enable(1'b0),.scan_count(32'd0),.scan_repeat_count(32'd1),
    .loop_table_count({(LIW+1){1'b0}}),.loop_first_flat({ML*RAW{1'b0}}),
    .loop_last_flat({ML*RAW{1'b0}}),.loop_count_flat({ML*32{1'b0}}),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),
    .bank_ready(2'b11),.bank_chunk0(32'd0),.bank_chunk1(32'd0),
    .scan_cursor(scan_cursor),.underflow(underflow),
    .bus_delay_ticks(bus_delay_ticks_w),.delay_ticks(delay_ticks_w),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  integer tt0_out, tt0_bus, errs;
  reg [BW-1:0] bhist [0:NT]; reg ohist [0:NT];
  integer t, started, i;

  task run_with_delay(input integer D);
    begin
      DUT_D = D;
      reset=1; start=0;
      delay_ticks_w = {CH*TDW{1'b0}};      delay_ticks_w[TTLBIT*TDW +: TDW] = D[TDW-1:0];
      bus_delay_ticks_w = {BUSC*TDW{1'b0}}; bus_delay_ticks_w[0*TDW +: TDW] = D[TDW-1:0];
      repeat (200) @(posedge clk);         // hold reset: arm refills the prefetch
      t=-1; started=0;
      reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
      // record
      while (t < NT) begin
        @(posedge clk);
        if (running && !started) begin started=1; t=-1; end
        if (started) begin t=t+1; if (t<=NT) begin ohist[t]=out[TTLBIT]; bhist[t]=bus_out[0*BW +: BW]; end end
      end
      // find first rising edge of out[5] and first VA->VB change of bus_out
      tt0_out=-1; tt0_bus=-1;
      for (i=1;i<=NT;i=i+1) begin
        if (tt0_out<0 && ohist[i]==1'b1 && ohist[i-1]==1'b0) tt0_out=i;
        if (tt0_bus<0 && bhist[i]==VB && bhist[i-1]==VA)     tt0_bus=i;
      end
      $display("  D=%0d : TTL out[5] rises @t=%0d , DA bus_out VA->VB @t=%0d  %s",
               D, tt0_out, tt0_bus, (tt0_out==tt0_bus && tt0_out==T0+D) ? "ALIGNED" : "*** MISALIGNED ***");
      if (!(tt0_out==tt0_bus && tt0_out==T0+D)) errs=errs+1;
    end
  endtask

  initial begin
    errs=0;
    run_with_delay(0);     // d=0: TTL and DA must change on the SAME tick
    run_with_delay(5);     // d=5: both shifted by 5, still SAME tick
    run_with_delay(25);    // d=25: both shifted by 25, still SAME tick
    $display("==== da/ttl align: %0d misalignments ====", errs);
    $display("%s", (errs==0) ? "DA-TTL-ALIGN-OK" : "**FAIL**");
    $finish;
  end
endmodule
