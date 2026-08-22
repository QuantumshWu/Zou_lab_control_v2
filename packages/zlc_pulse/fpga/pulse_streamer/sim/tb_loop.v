`timescale 1ns/1ps
// Finite-bracket LOOP test with a non-zero loop start.  The preamble and tail
// must each play once while the bracketed body plays exactly loop_count times.
module tb_loop;
  localparam integer CH=62, EAW=12, TW=32, NS=4, CW=16, DTW=32, BUSC=4, BW=10, NE=8;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;
  reg [12:0] wa=0; reg [31:0] wd=0; reg [3:0] we=0; reg wt=0, wm=0;
  wire [EAW-1:0] edge_raddr; wire [TW-1:0] edge_tick_rdata; wire [63:0] mrd;
  wire [CH-1:0] edge_mask_rdata = mrd[CH-1:0];
  wire [11:0] scan_raddr; wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out;
  wire running, done; wire [31:0] scan_cursor; wire underflow;
  blk_mem_gen_edge_tick u_t(.clka(clk),.ena(wt),.wea(we),.addra(wa[11:0]),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(edge_raddr),.dinb(32'b0),.doutb(edge_tick_rdata));
  blk_mem_gen_edge_mask u_m(.clka(clk),.ena(wm),.wea(we),.addra(wa),.dina(wd),.douta(),
    .clkb(clk),.enb(1'b1),.web(4'b0),.addrb(edge_raddr),.dinb(32'b0),.doutb(mrd));
  reg [31:0] ticks[0:NE-1]; reg [31:0] masks[0:NE-1];
  zlc_edge_streamer #(.CHANNEL_COUNT(CH)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(NE[12:0]),.repeat_forever(1'b0),
    .loop_start_addr(12'd2),.loop_end_tick(32'd500),.loop_end_coeffs({NS*CW{1'b0}}),
    .loop_count(32'd3),.scan_enable(1'b0),.scan_count(32'd0),
    .edge_raddr(edge_raddr),.edge_tick_rdata(edge_tick_rdata),
    .edge_coeff_rdata({NS*CW{1'b0}}),.edge_mask_rdata(edge_mask_rdata),
    .scan_raddr(scan_raddr),.scan_rdata({NS*TW{1'b0}}),.bank_ready(2'b11),
    .bank_chunk0(32'd0),.bank_chunk1(32'd0),.scan_cursor(scan_cursor),.underflow(underflow),
    .bus_prog_we(1'b0),.bus_prog_bus(2'd0),.bus_prog_addr(6'd0),.bus_prog_start_tick(32'd0),
    .bus_prog_stop_tick(32'd0),.bus_prog_start_tick_coeffs({NS*CW{1'b0}}),
    .bus_prog_stop_tick_coeffs({NS*CW{1'b0}}),.bus_prog_start_value(10'd0),
    .bus_prog_stop_value(10'd0),.bus_prog_mode(2'd0),.bus_prog_value_select(3'd0),
    .bus_prog_stop_value_select(3'd0),.bus_counts({BUSC*7{1'b0}}),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));
  integer i;
  task pa; input t; input [12:0] a; input [31:0] d; begin
    @(posedge clk); wt<=t; wm<=~t; we<=4'hF; wa<=a; wd<=d;
    @(posedge clk); @(posedge clk); wt<=0; wm<=0; we<=0; @(posedge clk); end
  endtask
  initial begin
    // bit0=preamble, bit11=loop body, bit1=post-loop tail.
    for (i=0;i<NE;i=i+1) begin ticks[i]=i*100; masks[i]=0; end
    masks[0]=62'h1; masks[1]=0;
    masks[2]=62'h800; masks[3]=0; masks[4]=0;
    masks[5]=62'h2; masks[6]=0; masks[7]=0;
    reset=1; start=0;
    for (i=0;i<NE;i=i+1) pa(1'b1, i, ticks[i]);
    for (i=0;i<NE;i=i+1) begin pa(1'b0, 2*i, masks[i]); pa(1'b0, 2*i+1, 32'd0); end
    repeat (200) @(posedge clk); reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end
  integer tcount=0; reg [CH-1:0] previous=0;
  integer lo=-1, np=0, nbad=0, preamble=0, tail=0, orderbad=0;
  always @(posedge clk) begin
    if (running||done) tcount=tcount+1;
    if (running && out[0] && !previous[0]) begin
      if (preamble!=0 || np!=0 || tail!=0) orderbad=orderbad+1;
      preamble=preamble+1;
    end
    if (running && out[11]!==previous[11]) begin
      if (out[11]) lo=tcount;
      else begin np=np+1; if (tcount-lo!=100) nbad=nbad+1;
        $display("[LOOP] emCCD pulse#%0d on=%0d off=%0d width=%0d %s",np,lo,tcount,tcount-lo,(tcount-lo==100)?"OK":"**BAD**"); end
    end
    if (running && out[1] && !previous[1]) begin
      if (preamble!=1 || np!=3 || tail!=0) orderbad=orderbad+1;
      tail=tail+1;
    end
    previous<=out;
  end
  initial begin
    wait(reset==1); wait(reset==0);
    repeat (3000) @(posedge clk);
    $display("==== LOOP(count=3): pulses=%0d expected=3 width_errors=%0d ====", np, nbad);
    if (np != 3) $fatal(1, "loop bench produced %0d pulses, expected 3", np);
    if (nbad != 0) $fatal(1, "loop bench had %0d bad pulse widths", nbad);
    if (preamble != 1 || tail != 1 || orderbad != 0)
      $fatal(1, "loop order/count mismatch: pre=%0d body=%0d tail=%0d orderbad=%0d",
             preamble, np, tail, orderbad);
    if (!done || running || underflow || out !== {CH{1'b0}})
      $fatal(1, "loop did not finish SAFE: run=%b done=%b uf=%b out=%h", running, done, underflow, out);
    $display("LOOP-OK");
    $finish;
  end
endmodule
