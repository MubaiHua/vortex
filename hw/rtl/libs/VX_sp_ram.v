`include "VX_platform.vh"

`TRACING_OFF
module VX_sp_ram #(
    parameter DATAW       = 1,
    parameter SIZE        = 1,
    parameter BYTEENW     = 1,
    parameter OUT_REG     = 0,
    parameter NO_RWCHECK  = 0,
    parameter LUTRAM      = 0,
    parameter ADDRW       = $clog2(SIZE),
    parameter INIT_ENABLE = 0,
    parameter INIT_FILE   = "",
    parameter [DATAW-1:0] INIT_VALUE = 0
) (  
    input wire               clk,
    input wire [ADDRW-1:0]   addr,
    input wire [BYTEENW-1:0] wren,
    input wire [DATAW-1:0]   wdata,
    output wire [DATAW-1:0]  rdata
);

    `STATIC_ASSERT((1 == BYTEENW) || ((BYTEENW > 1) && 0 == (BYTEENW % 4)), ("invalid parameter"))

`define RAM_INITIALIZATION                        \
    if (INIT_ENABLE) begin                        \
        if (INIT_FILE != "") begin                \
            initial $readmemh(INIT_FILE, ram);    \
        end else begin                            \
            initial                               \
                for (integer i = 0; i < SIZE; ++i)\
                    ram[i] = INIT_VALUE;          \
        end                                       \
    end

`ifdef __SYNTHESIS__
    if (LUTRAM) begin
        if (OUT_REG) begin        
            reg [DATAW-1:0] rdata_r;

            if (BYTEENW > 1) begin
                `USE_FAST_BRAM reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];

                `RAM_INITIALIZATION

                always @(posedge clk) begin
                    for (integer i = 0; i < BYTEENW; i++) begin
                        if (wren[i])
                            ram[addr][i] <= wdata[i * 8 +: 8];
                    end
                    rdata_r <= ram[addr];
                end
            end else begin
                `USE_FAST_BRAM reg [DATAW-1:0] ram [SIZE-1:0];

                `RAM_INITIALIZATION

                always @(posedge clk) begin
                    if (wren)
                        ram[addr] <= wdata;
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end else begin
            if (BYTEENW > 1) begin
                `USE_FAST_BRAM reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];

               `RAM_INITIALIZATION

                always @(posedge clk) begin
                    for (integer i = 0; i < BYTEENW; i++) begin
                        if (wren[i])
                            ram[addr][i] <= wdata[i * 8 +: 8];
                    end
                end
                assign rdata = ram[addr];
            end else begin
                `USE_FAST_BRAM reg [DATAW-1:0] ram [SIZE-1:0];

                `RAM_INITIALIZATION

                always @(posedge clk) begin
                    if (wren)
                        ram[addr] <= wdata;
                end
                assign rdata = ram[addr];
            end         
        end
    end else begin
        if (OUT_REG) begin
            reg [DATAW-1:0] rdata_r;

            if (BYTEENW > 1) begin
                reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];

                `RAM_INITIALIZATION

                always @(posedge clk) begin
                    for (integer i = 0; i < BYTEENW; i++) begin
                        if (wren[i])
                            ram[addr][i] <= wdata[i * 8 +: 8];
                    end
                    rdata_r <= ram[addr];
                end
            end else begin
                reg [DATAW-1:0] ram [SIZE-1:0];

               `RAM_INITIALIZATION

                always @(posedge clk) begin
                    if (wren)
                        ram[addr] <= wdata;
                    rdata_r <= ram[addr];
                end
            end
            assign rdata = rdata_r;
        end else begin
            if (NO_RWCHECK) begin
                if (BYTEENW > 1) begin
                    `NO_RW_RAM_CHECK reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];

                    `RAM_INITIALIZATION

                    always @(posedge clk) begin
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (wren[i])
                                ram[addr][i] <= wdata[i * 8 +: 8];
                        end
                    end
                    assign rdata = ram[addr];
                end else begin
                    `NO_RW_RAM_CHECK reg [DATAW-1:0] ram [SIZE-1:0];

                    `RAM_INITIALIZATION

                    always @(posedge clk) begin
                        if (wren)
                            ram[addr] <= wdata;
                    end
                    assign rdata = ram[addr];
                end
            end else begin
                if (BYTEENW > 1) begin
                    reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];

                    `RAM_INITIALIZATION

                    always @(posedge clk) begin                        
                        for (integer i = 0; i < BYTEENW; i++) begin
                            if (wren[i])
                                ram[addr][i] <= wdata[i * 8 +: 8];
                        end
                    end
                    assign rdata = ram[addr];
                end else begin
                    reg [DATAW-1:0] ram [SIZE-1:0];

                    `RAM_INITIALIZATION  

                    always @(posedge clk) begin
                        if (wren)
                            ram[addr] <= wdata;
                    end
                    assign rdata = ram[addr];
                end                
            end
        end
    end
`else
    if (OUT_REG) begin
        reg [DATAW-1:0] rdata_r;
        if (BYTEENW > 1) begin
            reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];

            `RAM_INITIALIZATION

            always @(posedge clk) begin
                for (integer i = 0; i < BYTEENW; i++) begin
                    if (wren[i])
                        ram[addr][i] <= wdata[i * 8 +: 8];
                end
                rdata_r <= ram[addr];
            end
        end else begin
            reg [DATAW-1:0] ram [SIZE-1:0];

            `RAM_INITIALIZATION

            always @(posedge clk) begin
                if (wren)
                    ram[addr] <= wdata;
                rdata_r <= ram[addr];
            end
        end
        assign rdata = rdata_r;
    end else begin
        if (BYTEENW > 1) begin
            reg [BYTEENW-1:0][7:0] ram [SIZE-1:0];
            reg [DATAW-1:0] prev_data;
            reg [ADDRW-1:0] prev_addr;
            reg prev_write;

            `RAM_INITIALIZATION

            always @(posedge clk) begin
                for (integer i = 0; i < BYTEENW; i++) begin
                    if (wren[i])
                        ram[addr][i] <= wdata[i * 8 +: 8];
                end
                prev_write <= (| wren);
                prev_data  <= ram[addr];
                prev_addr  <= addr;
            end
            
            if (LUTRAM || !NO_RWCHECK) begin
                `UNUSED_VAR (prev_write)
                `UNUSED_VAR (prev_data)
                `UNUSED_VAR (prev_addr)
                assign rdata = ram[addr];
            end else begin
                assign rdata = (prev_write && (prev_addr == addr)) ? prev_data : ram[addr];
            end
        end else begin
            reg [DATAW-1:0] ram [SIZE-1:0];
            reg [DATAW-1:0] prev_data;
            reg [ADDRW-1:0] prev_addr;
            reg prev_write;

            `RAM_INITIALIZATION  

            always @(posedge clk) begin
                if (wren)
                    ram[addr] <= wdata;
                prev_write <= wren;
                prev_data  <= ram[addr];
                prev_addr  <= addr;
            end
            if (LUTRAM || !NO_RWCHECK) begin
                `UNUSED_VAR (prev_write)
                `UNUSED_VAR (prev_data)
                `UNUSED_VAR (prev_addr)
                assign rdata = ram[addr];
            end else begin
                assign rdata = (prev_write && (prev_addr == addr)) ? prev_data : ram[addr];
            end
        end
    end
`endif    

endmodule
`TRACING_ON