`timescale 1ns/1ps
// Real zlc_period_streamer driven through all three repeat layers and a STREAMED scan with a small
// BANK_SIZE so K = ceil(N/BANK_SIZE) > 2 (here BANK_SIZE=4, N=10 -> K=3, ODD: the case that used to gap).
// A behavioral CYCLIC host-refill model feeds chunks 0,1,..,K-1,0,1,.. one-ahead into the
// alternating ping-pong bank (bank = monotonic_chunk % 2), exactly matching the engine's
// bank parity.  A 3x whole-timeline PulseBracket sits inside 2 Run repeats per row,
// and the ten-row table runs for 3 Scan repeats.  Asserts the exact N*M*S nesting,
// row-only CURSOR motion, finite DONE, and no underflow at either seam.
module tb_scan_wrap;
  localparam integer CH=8, RAW=`ZLC_ROW_ADDR_WIDTH, TW=32, NS=1, SSW=`ZLC_SLOT_SEL_WIDTH;
  localparam integer BUSC=4, BW=10, ML=`ZLC_MAX_LOOPS, LIW=`ZLC_LOOP_INDEX_WIDTH, DTW=32;
  localparam integer ABITS=2+SSW+BW, RBITS=TW+SSW+CH+BUSC*ABITS;
  localparam integer BANK_SIZE=4, SAW=3;          // SAW = clog2(BANK_SIZE)+1 = 3 (2 banks x 4)
  localparam integer NPTS=10;                      // K = ceil(10/4) = 3 (odd)
  localparam integer KCH=(NPTS+BANK_SIZE-1)/BANK_SIZE;
  localparam integer BRACKET_REPEATS=3, RUN_REPEATS=2, SCAN_REPEATS=3;
  reg clk=0, reset=0, start=0; always #10 clk=~clk;

  function [RBITS-1:0] row_of;
    input [TW-1:0] dur; input [CH-1:0] mask;
    begin row_of = {{(BUSC*ABITS){1'b0}}, mask, {SSW{1'b0}}, dur}; end
  endfunction

  // --- period table: row 0 = bit0 high for 5 ticks, row 1 = low for 5 ticks; bracket rows 0..1 x3 ---
  reg [RBITS-1:0] rowmem [0:3];
  initial begin rowmem[0]=row_of(32'd5, 8'h1); rowmem[1]=row_of(32'd5, 8'h0); rowmem[2]=0; rowmem[3]=0; end
  wire [RAW-1:0] row_raddr;
  reg [RBITS-1:0] rp[0:2];
  always @(posedge clk) begin rp[0]<=rowmem[row_raddr[1:0]]; rp[1]<=rp[0]; rp[2]<=rp[1]; end
  wire [RBITS-1:0] row_rdata = rp[2];
  reg [ML*RAW-1:0] loop_first = {ML*RAW{1'b0}};
  reg [ML*RAW-1:0] loop_last = {ML*RAW{1'b0}};
  reg [ML*32-1:0] loop_count = {ML*32{1'b0}};

  // --- behavioral 2-bank scan memory (NS*TW per entry); host writes it; engine reads w/ lat RD_LAT+2 ---
  reg [NS*TW-1:0] scanmem [0:2*BANK_SIZE-1];
  wire [SAW-1:0] scan_raddr;
  reg [NS*TW-1:0] spipe[0:2];
  always @(posedge clk) begin spipe[0]<=scanmem[scan_raddr]; spipe[1]<=spipe[0]; spipe[2]<=spipe[1]; end
  wire [NS*TW-1:0] scan_rdata = spipe[2];

  reg [1:0] bank_ready; reg [TW-1:0] bank_chunk0, bank_chunk1;
  wire [CH-1:0] out; wire [BUSC*BW-1:0] bus_out; wire running, done;
  wire [TW-1:0] scan_cursor_w; wire underflow;

  zlc_period_streamer #(.CHANNEL_COUNT(CH),.SCAN_ADDR_WIDTH(SAW),.BANK_SIZE(BANK_SIZE),
                        .NUM_SLOTS(NS)) dut (
    .clk(clk),.reset(reset),.start(start),.prog_count(2),.run_repeat_count(RUN_REPEATS[31:0]),
    .scan_enable(1'b1),.scan_count(NPTS[31:0]),.scan_repeat_count(SCAN_REPEATS[31:0]),
    .loop_table_count(1),.loop_first_flat(loop_first),.loop_last_flat(loop_last),.loop_count_flat(loop_count),
    .row_raddr(row_raddr),.row_rdata(row_rdata),
    .scan_raddr(scan_raddr),.scan_rdata(scan_rdata),
    .bank_ready(bank_ready),.bank_chunk0(bank_chunk0),.bank_chunk1(bank_chunk1),
    .scan_cursor(scan_cursor_w),.underflow(underflow),
    .bus_delay_ticks({BUSC*DTW{1'b0}}),.delay_ticks({CH*DTW{1'b0}}),
    .out(out),.bus_out(bus_out),.running(running),.done(done));

  // --- behavioral CYCLIC host refill model ---------------------------------------------------
  // scanmem entry for (bank,offset) holds the slot vector (= the point index it represents).
  // chunk c (data) = points [c*BANK_SIZE .. ); host loads chunk (mono mod K) into bank (mono%2)
  // one-ahead, with REFILL_LAT cycles of write latency.
  localparam integer REFILL_LAT=6;
  integer load_at;         // cycle when the in-flight load completes
  integer load_bank, load_chunk;
  reg     load_busy;
  integer cyc;
  task load_chunk_into; input integer datachunk; input integer bank; integer j; integer gpt; begin
    for (j=0;j<BANK_SIZE;j=j+1) begin
      gpt = datachunk*BANK_SIZE + j;                 // global point index (may be >= NPTS in last chunk)
      scanmem[bank*BANK_SIZE + j] = (gpt<NPTS) ? gpt[NS*TW-1:0] : {NS*TW{1'b0}};
    end
  end endtask

  // engine monotonic chunk comes directly from the hardware's cumulative row
  // visit cursor: sweep=cursor/N, row=cursor%N.  Polling need not witness a wrap.
  integer eng_mono;
  reg [NS*TW-1:0] slot_seq [0:255]; integer nseen; integer stalls_after_warmup;
  reg [NS*TW-1:0] body_slots [0:511]; integer nbodies;

  initial begin
    loop_first[0 +: RAW] = 0; loop_last[0 +: RAW] = 1; loop_count[0 +: 32] = BRACKET_REPEATS;
    reset=1; start=0; bank_ready=2'b00; bank_chunk0=0; bank_chunk1=0;
    // preload monotonic chunk 0 -> bank0, chunk1 -> bank1 (base=0)
    load_chunk_into(0,0); bank_chunk0=0;
    load_chunk_into(1%KCH,1); bank_chunk1=(1%KCH);
    bank_ready=2'b11;
    load_busy=0; eng_mono=0; nseen=0; nbodies=0; stalls_after_warmup=0;
    cyc=0; load_at=0; load_bank=0; load_chunk=0;
    repeat (300) @(posedge clk);
    reset=0; @(posedge clk); start=1; @(posedge clk); start=0;
  end

  // host cyclic refill: derive the current monotonic chunk directly and keep
  // the bank for the NEXT monotonic chunk loaded with (chunk mod K).
  always @(posedge clk) begin
    cyc<=cyc+1;
    if (running) begin
      eng_mono = (scan_cursor_w/NPTS)*KCH
                 + ((scan_cursor_w%NPTS)/BANK_SIZE);
      // complete an in-flight load
      if (load_busy && cyc>=load_at) begin
        if (load_bank==0) bank_chunk0<=load_chunk[TW-1:0]; else bank_chunk1<=load_chunk[TW-1:0];
        bank_ready[load_bank]<=1'b1; load_busy<=0;
      end
      // start a one-ahead load if the next bank doesn't already hold the next chunk
      if (!load_busy) begin : refill
        integer nb; integer nc;
        nb = (eng_mono+1) % 2;
        nc = (eng_mono+1) % KCH;
        if (((nb==0)?bank_chunk0:bank_chunk1) != nc[TW-1:0]) begin
          bank_ready[nb]<=1'b0;          // de-arm during rewrite
          load_chunk_into(nc, nb);       // (write data immediately; arm after latency)
          load_bank<=nb; load_chunk<=nc; load_at<=cyc+REFILL_LAT; load_busy<=1;
        end
      end
    end
  end

  // record the slot the engine plays at each body (a rise of bit 0) + watch for wrap stalls
  reg [NS*TW-1:0] slot_prev; integer started=0; reg out_prev=0;
  always @(posedge clk) begin
    if (running) begin
      if (out[0] && !out_prev) begin
        body_slots[nbodies]<=dut.slot_active;
        nbodies<=nbodies+1;
      end
      if (!started) begin started<=1; slot_prev<=dut.slot_active; slot_seq[0]<=dut.slot_active; nseen<=1; end
      else if (dut.slot_active !== slot_prev) begin
        slot_seq[nseen]<=dut.slot_active; nseen<=nseen+1; slot_prev<=dut.slot_active;
      end
      // a stall at the wrap shows as underflow asserting; count it (after warmup)
      if (underflow && cyc>100) stalls_after_warmup<=stalls_after_warmup+1;
    end
    out_prev <= out[0] & running;
  end

  integer i; integer bad; integer timeout_cycles;
  initial begin
    wait(reset==1); wait(reset==0);
    for (timeout_cycles=0; timeout_cycles<8000 && done!=1'b1; timeout_cycles=timeout_cycles+1)
      @(posedge clk);
    if (done!=1'b1) begin
      $display("**FAIL** three-layer scan did not reach finite DONE");
      $finish;
    end
    repeat (10) @(posedge clk);
    // Every body has its row's slot.  Bracket repeats are inner, then Run
    // repeats, then row advance, then Scan repeats.
    bad=0;
    for (i=0;i<nbodies;i=i+1) if (body_slots[i] !== ((i/(BRACKET_REPEATS*RUN_REPEATS)) % NPTS)) begin
      bad=bad+1;
      if (bad<=6) $display("  MISMATCH body[%0d]=%0d expected %0d", i, body_slots[i], (i/(BRACKET_REPEATS*RUN_REPEATS))%NPTS);
    end
    $display("==== three-layer scan: K=%0d N=%0d bracket=%0d run=%0d sweeps=%0d bodies=%0d mismatches=%0d stalls=%0d cursor=%0d ====",
             KCH, NPTS, BRACKET_REPEATS, RUN_REPEATS, SCAN_REPEATS, nbodies, bad, stalls_after_warmup, scan_cursor_w);
    $display("%s", (bad==0 && stalls_after_warmup==0
             && nbodies==NPTS*BRACKET_REPEATS*RUN_REPEATS*SCAN_REPEATS
             && nseen==NPTS*SCAN_REPEATS
             && scan_cursor_w==NPTS*SCAN_REPEATS-1) ? "SEAMLESS-OK" : "**FAIL**");
    $finish;
  end
endmodule
