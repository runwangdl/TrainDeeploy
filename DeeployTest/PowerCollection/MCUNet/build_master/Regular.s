
/app/TrainDeeploy/DeeployTest/TEST_GAP9/build_master/Regular:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .data_tiny_fc 0000054c  00000004  1c000004  00001004  2**2
                  CONTENTS, ALLOC, LOAD, DATA
  1 .boot         00000000  1c000550  1c000550  0000f5d0  2**0
                  CONTENTS
  2 .ctors        00000014  1c000550  1c000550  00001550  2**2
                  CONTENTS, ALLOC, LOAD, DATA
  3 .dtors        0000001c  1c000564  1c000564  00001564  2**2
                  CONTENTS, ALLOC, LOAD, DATA
  4 .data_fc      00000040  1c000580  1c000580  00001580  2**2
                  CONTENTS, ALLOC, LOAD, DATA
  5 .stack_irq    00000800  1c0005c0  1c0005c0  000015c0  2**0
                  ALLOC
  6 .vectors_irq  00000094  1c010000  1c010000  00002000  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  7 .cluster_vectors_irq 0000006c  1c010094  1c010094  00002094  2**0
                  ALLOC
  8 .data_tiny_l1 0000004c  00000004  1c010100  00003004  2**2
                  CONTENTS, ALLOC, LOAD, DATA
  9 .l1cluster_g  00000010  10000050  1c01014c  00003050  2**2
                  CONTENTS, ALLOC, LOAD, DATA
 10 .text_l2      00000dd4  1c01015c  1c01015c  0000315c  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 11 .text         0000a780  1c010f30  1c010f30  00003f30  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 12 .cluster.text 00000316  1c01b6b0  1c01b6b0  0000e6b0  2**0
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
 13 .rodata       00000788  1c01b9c8  1c01b9c8  0000e9c8  2**3
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 14 .shbss        00000000  1c01c150  1c01c150  0000f5d0  2**0
                  CONTENTS
 15 .data         00000480  1c01c150  1c01c150  0000f150  2**2
                  CONTENTS, ALLOC, LOAD, DATA
 16 .bss          00001650  1c01c5d0  1c01c5d0  0000f5d0  2**3
                  ALLOC
 17 .ram          00000000  1c01dc20  1c01dc20  0000f5d0  2**0
                  CONTENTS
 18 .comment      0000001a  00000000  00000000  0000f5d0  2**0
                  CONTENTS, READONLY
 19 .Pulp_Chip.Info 00000076  00000000  00000000  0000f5ea  2**0
                  CONTENTS, READONLY
 20 .debug_frame  00000270  00000000  00000000  0000f660  2**2
                  CONTENTS, READONLY, DEBUGGING
 21 .debug_info   00006b26  00000000  00000000  0000f8d0  2**0
                  CONTENTS, READONLY, DEBUGGING
 22 .debug_abbrev 00001aa0  00000000  00000000  000163f6  2**0
                  CONTENTS, READONLY, DEBUGGING
 23 .debug_loc    00007faf  00000000  00000000  00017e96  2**0
                  CONTENTS, READONLY, DEBUGGING
 24 .debug_aranges 000001f8  00000000  00000000  0001fe45  2**0
                  CONTENTS, READONLY, DEBUGGING
 25 .debug_ranges 00000cc0  00000000  00000000  0002003d  2**0
                  CONTENTS, READONLY, DEBUGGING
 26 .debug_line   00003f3c  00000000  00000000  00020cfd  2**0
                  CONTENTS, READONLY, DEBUGGING
 27 .debug_str    00001229  00000000  00000000  00024c39  2**0
                  CONTENTS, READONLY, DEBUGGING

Disassembly of section .vectors_irq:

1c010000 <__irq_vector_base_m__>:
1c010000:	21c0006f          	j	1c01021c <mcause_check>
1c010004:	4340006f          	j	1c010438 <pendSV_handler>
1c010008:	5160006f          	j	1c01051e <default_handler>
1c01000c:	5120006f          	j	1c01051e <default_handler>
1c010010:	50e0006f          	j	1c01051e <default_handler>
1c010014:	50a0006f          	j	1c01051e <default_handler>
1c010018:	5060006f          	j	1c01051e <default_handler>
1c01001c:	5020006f          	j	1c01051e <default_handler>
1c010020:	4fe0006f          	j	1c01051e <default_handler>
1c010024:	4fa0006f          	j	1c01051e <default_handler>
1c010028:	4780006f          	j	1c0104a0 <systick_handler>
1c01002c:	4f20006f          	j	1c01051e <default_handler>
1c010030:	4ee0006f          	j	1c01051e <default_handler>
1c010034:	4ea0006f          	j	1c01051e <default_handler>
1c010038:	4e60006f          	j	1c01051e <default_handler>
1c01003c:	4e20006f          	j	1c01051e <default_handler>
1c010040:	4de0006f          	j	1c01051e <default_handler>
1c010044:	4da0006f          	j	1c01051e <default_handler>
1c010048:	4d60006f          	j	1c01051e <default_handler>
1c01004c:	4d20006f          	j	1c01051e <default_handler>
1c010050:	4ce0006f          	j	1c01051e <default_handler>
1c010054:	4ca0006f          	j	1c01051e <default_handler>
1c010058:	4c60006f          	j	1c01051e <default_handler>
1c01005c:	4c20006f          	j	1c01051e <default_handler>
1c010060:	4be0006f          	j	1c01051e <default_handler>
1c010064:	4680006f          	j	1c0104cc <pmu_event_handler>
1c010068:	43c0006f          	j	1c0104a4 <fc_event_handler>
1c01006c:	4b20006f          	j	1c01051e <default_handler>
1c010070:	4ae0006f          	j	1c01051e <default_handler>
1c010074:	4aa0006f          	j	1c01051e <default_handler>
1c010078:	4a60006f          	j	1c01051e <default_handler>
1c01007c:	4a20006f          	j	1c01051e <default_handler>
1c010080:	0e00006f          	j	1c010160 <_start>
1c010084:	49e0006f          	j	1c010522 <ill_ins_handler>
1c010088:	46c0006f          	j	1c0104f4 <ecall_handler>
1c01008c:	0000                	unimp
	...

1c010090 <__rt_debug_struct_ptr>:
1c010090:	deadbeef          	jal	t4,1bfeb67a <__l1_heapsram_end+0xbfcb67a>

Disassembly of section .text_l2:

1c01015c <_start-0x4>:
1c01015c:	0000                	unimp
	...

1c010160 <_start>:
1c010160:	f1402573          	csrr	a0,mhartid
1c010164:	01f57593          	andi	a1,a0,31
1c010168:	8115                	srli	a0,a0,0x5
1c01016a:	4625                	li	a2,9
1c01016c:	08b61863          	bne	a2,a1,1c0101fc <pi_cl_entry_stub>
1c010170:	8e31                	xor	a2,a2,a2
1c010172:	00002537          	lui	a0,0x2
1c010176:	80050513          	addi	a0,a0,-2048 # 1800 <__STACK_IRQ_SIZE+0x1000>
1c01017a:	30051073          	csrw	mstatus,a0
1c01017e:	00000517          	auipc	a0,0x0
1c010182:	e8250513          	addi	a0,a0,-382 # 1c010000 <__irq_vector_base_m__>
1c010186:	30551073          	csrw	mtvec,a0
1c01018a:	00000517          	auipc	a0,0x0
1c01018e:	e7650513          	addi	a0,a0,-394 # 1c010000 <__irq_vector_base_m__>
1c010192:	00551073          	csrw	0x5,a0
1c010196:	0000c197          	auipc	gp,0xc
1c01019a:	fba18193          	addi	gp,gp,-70 # 1c01c150 <__data_start__>

1c01019e <_fill_zero>:
1c01019e:	0000c517          	auipc	a0,0xc
1c0101a2:	43250513          	addi	a0,a0,1074 # 1c01c5d0 <__data_end__>
1c0101a6:	0000e597          	auipc	a1,0xe
1c0101aa:	a7a58593          	addi	a1,a1,-1414 # 1c01dc20 <__bss_end__>
1c0101ae:	00052023          	sw	zero,0(a0)
1c0101b2:	0511                	addi	a0,a0,4
1c0101b4:	feb56de3          	bltu	a0,a1,1c0101ae <_fill_zero+0x10>
1c0101b8:	ffff1117          	auipc	sp,0xffff1
1c0101bc:	c0810113          	addi	sp,sp,-1016 # 1c000dc0 <__heapfcram_start>
1c0101c0:	00005517          	auipc	a0,0x5
1c0101c4:	e4850513          	addi	a0,a0,-440 # 1c015008 <system_init>
1c0101c8:	000500e7          	jalr	a0
1c0101cc:	0000d117          	auipc	sp,0xd
1c0101d0:	5b810113          	addi	sp,sp,1464 # 1c01d784 <__stack_fc_start__>
1c0101d4:	00012103          	lw	sp,0(sp)
1c0101d8:	ffff1517          	auipc	a0,0xffff1
1c0101dc:	be850513          	addi	a0,a0,-1048 # 1c000dc0 <__heapfcram_start>
1c0101e0:	34051073          	csrw	0x340,a0
1c0101e4:	00005517          	auipc	a0,0x5
1c0101e8:	f5450513          	addi	a0,a0,-172 # 1c015138 <system_init_finalize>
1c0101ec:	000500e7          	jalr	a0
1c0101f0:	0000a617          	auipc	a2,0xa
1c0101f4:	d6260613          	addi	a2,a2,-670 # 1c019f52 <__os_init>
1c0101f8:	000600e7          	jalr	a2

1c0101fc <pi_cl_entry_stub>:
1c0101fc:	0000b297          	auipc	t0,0xb
1c010200:	4b428293          	addi	t0,t0,1204 # 1c01b6b0 <__cluster_text_end>
1c010204:	8282                	jr	t0

1c010206 <__printf_uart_close>:
1c010206:	7175                	addi	sp,sp,-144
1c010208:	c006                	sw	ra,0(sp)
1c01020a:	00005617          	auipc	a2,0x5
1c01020e:	f5260613          	addi	a2,a2,-174 # 1c01515c <system_printf_uart_close>
1c010212:	000600e7          	jalr	a2
1c010216:	4082                	lw	ra,0(sp)
1c010218:	6149                	addi	sp,sp,144
1c01021a:	8082                	ret

1c01021c <mcause_check>:
1c01021c:	34011173          	csrrw	sp,0x340,sp
1c010220:	1121                	addi	sp,sp,-24
1c010222:	c006                	sw	ra,0(sp)
1c010224:	c216                	sw	t0,4(sp)
1c010226:	c41a                	sw	t1,8(sp)
1c010228:	c61e                	sw	t2,12(sp)
1c01022a:	341022f3          	csrr	t0,mepc
1c01022e:	c872                	sw	t3,16(sp)
1c010230:	ca16                	sw	t0,20(sp)
1c010232:	80000337          	lui	t1,0x80000
1c010236:	fff30313          	addi	t1,t1,-1 # 7fffffff <pulp__FC+0x80000000>
1c01023a:	342022f3          	csrr	t0,mcause
1c01023e:	0062f3b3          	and	t2,t0,t1
1c010242:	00239293          	slli	t0,t2,0x2
1c010246:	0000e317          	auipc	t1,0xe
1c01024a:	91630313          	addi	t1,t1,-1770 # 1c01db5c <pi_exception_vector>
1c01024e:	9316                	add	t1,t1,t0
1c010250:	00032303          	lw	t1,0(t1)
1c010254:	00030c63          	beqz	t1,1c01026c <__mcause_check_no_handler>
1c010258:	7175                	addi	sp,sp,-144
1c01025a:	8e1a                	mv	t3,t1
1c01025c:	2d4000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c010260:	000e00e7          	jalr	t3
1c010264:	310000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010268:	6149                	addi	sp,sp,144
1c01026a:	a055                	j	1c01030e <__end_mcause_check>

1c01026c <__mcause_check_no_handler>:
1c01026c:	4e09                	li	t3,2
1c01026e:	42ad                	li	t0,11
1c010270:	09c38363          	beq	t2,t3,1c0102f6 <__ill_ins>
1c010274:	4e05                	li	t3,1
1c010276:	03c38963          	beq	t2,t3,1c0102a8 <__pmp_error>
1c01027a:	4e15                	li	t3,5
1c01027c:	03c38663          	beq	t2,t3,1c0102a8 <__pmp_error>
1c010280:	4e1d                	li	t3,7
1c010282:	03c38363          	beq	t2,t3,1c0102a8 <__pmp_error>
1c010286:	4e31                	li	t3,12
1c010288:	03c38d63          	beq	t2,t3,1c0102c2 <__xip_error_insn>
1c01028c:	4e35                	li	t3,13
1c01028e:	05c38763          	beq	t2,t3,1c0102dc <__xip_error_data>
1c010292:	4e3d                	li	t3,15
1c010294:	05c38463          	beq	t2,t3,1c0102dc <__xip_error_data>
1c010298:	4325                	li	t1,9
1c01029a:	06538563          	beq	t2,t0,1c010304 <__ecall>
1c01029e:	42a1                	li	t0,8
1c0102a0:	06638263          	beq	t2,t1,1c010304 <__ecall>
1c0102a4:	06538063          	beq	t2,t0,1c010304 <__ecall>

1c0102a8 <__pmp_error>:
1c0102a8:	7175                	addi	sp,sp,-144
1c0102aa:	286000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0102ae:	0000a297          	auipc	t0,0xa
1c0102b2:	afa28293          	addi	t0,t0,-1286 # 1c019da8 <pmp_exception_handler>
1c0102b6:	000280e7          	jalr	t0
1c0102ba:	2ba000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0102be:	6149                	addi	sp,sp,144
1c0102c0:	a0b9                	j	1c01030e <__end_mcause_check>

1c0102c2 <__xip_error_insn>:
1c0102c2:	7175                	addi	sp,sp,-144
1c0102c4:	26c000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0102c8:	0000a297          	auipc	t0,0xa
1c0102cc:	b0228293          	addi	t0,t0,-1278 # 1c019dca <xip_insn_exception_handler>
1c0102d0:	000280e7          	jalr	t0
1c0102d4:	2a0000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0102d8:	6149                	addi	sp,sp,144
1c0102da:	a815                	j	1c01030e <__end_mcause_check>

1c0102dc <__xip_error_data>:
1c0102dc:	7175                	addi	sp,sp,-144
1c0102de:	252000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0102e2:	0000a297          	auipc	t0,0xa
1c0102e6:	b4228293          	addi	t0,t0,-1214 # 1c019e24 <xip_data_exception_handler>
1c0102ea:	000280e7          	jalr	t0
1c0102ee:	286000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0102f2:	6149                	addi	sp,sp,144
1c0102f4:	a829                	j	1c01030e <__end_mcause_check>

1c0102f6 <__ill_ins>:
1c0102f6:	34102573          	csrr	a0,mepc
1c0102fa:	0000a297          	auipc	t0,0xa
1c0102fe:	b7828293          	addi	t0,t0,-1160 # 1c019e72 <_ill_insn_print>
1c010302:	9282                	jalr	t0

1c010304 <__ecall>:
1c010304:	1f0000ef          	jal	ra,1c0104f4 <ecall_handler>
1c010308:	42d2                	lw	t0,20(sp)
1c01030a:	0291                	addi	t0,t0,4
1c01030c:	ca16                	sw	t0,20(sp)

1c01030e <__end_mcause_check>:
1c01030e:	42d2                	lw	t0,20(sp)
1c010310:	4e42                	lw	t3,16(sp)
1c010312:	34129073          	csrw	mepc,t0
1c010316:	43b2                	lw	t2,12(sp)
1c010318:	4322                	lw	t1,8(sp)
1c01031a:	4292                	lw	t0,4(sp)
1c01031c:	4082                	lw	ra,0(sp)
1c01031e:	0161                	addi	sp,sp,24
1c010320:	34011173          	csrrw	sp,0x340,sp
1c010324:	30200073          	mret

1c010328 <pi_irq_handler_wrapper>:
1c010328:	34011173          	csrrw	sp,0x340,sp
1c01032c:	7175                	addi	sp,sp,-144
1c01032e:	c006                	sw	ra,0(sp)
1c010330:	200000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c010334:	342022f3          	csrr	t0,mcause
1c010338:	0000e317          	auipc	t1,0xe
1c01033c:	86430313          	addi	t1,t1,-1948 # 1c01db9c <pi_irq_handler_wrapper_vector>
1c010340:	028a                	slli	t0,t0,0x2
1c010342:	9316                	add	t1,t1,t0
1c010344:	00032303          	lw	t1,0(t1)
1c010348:	00030463          	beqz	t1,1c010350 <__pi_irq_handler_wrapper_end>
1c01034c:	000300e7          	jalr	t1

1c010350 <__pi_irq_handler_wrapper_end>:
1c010350:	224000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010354:	4082                	lw	ra,0(sp)
1c010356:	6149                	addi	sp,sp,144
1c010358:	34011173          	csrrw	sp,0x340,sp
1c01035c:	30200073          	mret

1c010360 <asm_do_yield>:
1c010360:	7175                	addi	sp,sp,-144
1c010362:	c006                	sw	ra,0(sp)
1c010364:	34109073          	csrw	mepc,ra
1c010368:	1c8000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c01036c:	e3ff0217          	auipc	tp,0xe3ff0
1c010370:	17422203          	lw	tp,372(tp) # 4e0 <pxCurrentTCB>
1c010374:	00222023          	sw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c010378:	000022b7          	lui	t0,0x2
1c01037c:	80028293          	addi	t0,t0,-2048 # 1800 <__STACK_IRQ_SIZE+0x1000>
1c010380:	30029073          	csrw	mstatus,t0
1c010384:	00008297          	auipc	t0,0x8
1c010388:	52428293          	addi	t0,t0,1316 # 1c0188a8 <vTaskSwitchContext>
1c01038c:	000280e7          	jalr	t0
1c010390:	e3ff0517          	auipc	a0,0xe3ff0
1c010394:	15052503          	lw	a0,336(a0) # 4e0 <pxCurrentTCB>
1c010398:	02a20463          	beq	tp,a0,1c0103c0 <__yield_no_switch>
1c01039c:	21c000ef          	jal	ra,1c0105b8 <ASM_FUNC_SAVE_SAVE_CONTEXT>
1c0103a0:	24c000ef          	jal	ra,1c0105ec <ASM_FUNC_SAVE_CSR_CONTEXT>
1c0103a4:	00056233          	or	tp,a0,zero
1c0103a8:	00022103          	lw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c0103ac:	24e000ef          	jal	ra,1c0105fa <ASM_FUNC_RESTORE_CSR_CONTEXT>
1c0103b0:	222000ef          	jal	ra,1c0105d2 <ASM_FUNC_RESTORE_SAVE_CONTEXT>
1c0103b4:	1c0000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0103b8:	4082                	lw	ra,0(sp)
1c0103ba:	6149                	addi	sp,sp,144
1c0103bc:	30200073          	mret

1c0103c0 <__yield_no_switch>:
1c0103c0:	1b4000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0103c4:	4082                	lw	ra,0(sp)
1c0103c6:	6149                	addi	sp,sp,144
1c0103c8:	30200073          	mret

1c0103cc <asm_do_yield_force>:
1c0103cc:	7175                	addi	sp,sp,-144
1c0103ce:	c006                	sw	ra,0(sp)
1c0103d0:	34109073          	csrw	mepc,ra
1c0103d4:	15c000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0103d8:	e3ff0217          	auipc	tp,0xe3ff0
1c0103dc:	10822203          	lw	tp,264(tp) # 4e0 <pxCurrentTCB>
1c0103e0:	00222023          	sw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c0103e4:	000022b7          	lui	t0,0x2
1c0103e8:	80028293          	addi	t0,t0,-2048 # 1800 <__STACK_IRQ_SIZE+0x1000>
1c0103ec:	30029073          	csrw	mstatus,t0
1c0103f0:	00008297          	auipc	t0,0x8
1c0103f4:	49a28293          	addi	t0,t0,1178 # 1c01888a <vTaskSwitchContextForce>
1c0103f8:	000280e7          	jalr	t0
1c0103fc:	e3ff0517          	auipc	a0,0xe3ff0
1c010400:	0e452503          	lw	a0,228(a0) # 4e0 <pxCurrentTCB>
1c010404:	02a20463          	beq	tp,a0,1c01042c <__yield_no_switch_force>
1c010408:	1b0000ef          	jal	ra,1c0105b8 <ASM_FUNC_SAVE_SAVE_CONTEXT>
1c01040c:	1e0000ef          	jal	ra,1c0105ec <ASM_FUNC_SAVE_CSR_CONTEXT>
1c010410:	00056233          	or	tp,a0,zero
1c010414:	00022103          	lw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c010418:	1e2000ef          	jal	ra,1c0105fa <ASM_FUNC_RESTORE_CSR_CONTEXT>
1c01041c:	1b6000ef          	jal	ra,1c0105d2 <ASM_FUNC_RESTORE_SAVE_CONTEXT>
1c010420:	154000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010424:	4082                	lw	ra,0(sp)
1c010426:	6149                	addi	sp,sp,144
1c010428:	30200073          	mret

1c01042c <__yield_no_switch_force>:
1c01042c:	148000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010430:	4082                	lw	ra,0(sp)
1c010432:	6149                	addi	sp,sp,144
1c010434:	30200073          	mret

1c010438 <pendSV_handler>:
1c010438:	7175                	addi	sp,sp,-144
1c01043a:	c006                	sw	ra,0(sp)
1c01043c:	0f4000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c010440:	e3ff0217          	auipc	tp,0xe3ff0
1c010444:	0a022203          	lw	tp,160(tp) # 4e0 <pxCurrentTCB>
1c010448:	00222023          	sw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c01044c:	34011173          	csrrw	sp,0x340,sp
1c010450:	00008297          	auipc	t0,0x8
1c010454:	45828293          	addi	t0,t0,1112 # 1c0188a8 <vTaskSwitchContext>
1c010458:	000280e7          	jalr	t0
1c01045c:	e3ff0517          	auipc	a0,0xe3ff0
1c010460:	08452503          	lw	a0,132(a0) # 4e0 <pxCurrentTCB>
1c010464:	02a20663          	beq	tp,a0,1c010490 <__pendSV_handler_no_switch>
1c010468:	34011173          	csrrw	sp,0x340,sp
1c01046c:	14c000ef          	jal	ra,1c0105b8 <ASM_FUNC_SAVE_SAVE_CONTEXT>
1c010470:	17c000ef          	jal	ra,1c0105ec <ASM_FUNC_SAVE_CSR_CONTEXT>
1c010474:	00056233          	or	tp,a0,zero
1c010478:	00022103          	lw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c01047c:	17e000ef          	jal	ra,1c0105fa <ASM_FUNC_RESTORE_CSR_CONTEXT>
1c010480:	152000ef          	jal	ra,1c0105d2 <ASM_FUNC_RESTORE_SAVE_CONTEXT>
1c010484:	0f0000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010488:	4082                	lw	ra,0(sp)
1c01048a:	6149                	addi	sp,sp,144
1c01048c:	30200073          	mret

1c010490 <__pendSV_handler_no_switch>:
1c010490:	34011173          	csrrw	sp,0x340,sp
1c010494:	0e0000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010498:	4082                	lw	ra,0(sp)
1c01049a:	6149                	addi	sp,sp,144
1c01049c:	30200073          	mret

1c0104a0 <systick_handler>:
1c0104a0:	30200073          	mret

1c0104a4 <fc_event_handler>:
1c0104a4:	34011173          	csrrw	sp,0x340,sp
1c0104a8:	7175                	addi	sp,sp,-144
1c0104aa:	c006                	sw	ra,0(sp)
1c0104ac:	084000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0104b0:	00001297          	auipc	t0,0x1
1c0104b4:	c7428293          	addi	t0,t0,-908 # 1c011124 <fc_soc_event_handler>
1c0104b8:	000280e7          	jalr	t0
1c0104bc:	0b8000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0104c0:	4082                	lw	ra,0(sp)
1c0104c2:	6149                	addi	sp,sp,144
1c0104c4:	34011173          	csrrw	sp,0x340,sp
1c0104c8:	30200073          	mret

1c0104cc <pmu_event_handler>:
1c0104cc:	34011173          	csrrw	sp,0x340,sp
1c0104d0:	7175                	addi	sp,sp,-144
1c0104d2:	c006                	sw	ra,0(sp)
1c0104d4:	05c000ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0104d8:	00000297          	auipc	t0,0x0
1c0104dc:	64228293          	addi	t0,t0,1602 # 1c010b1a <__pi_pmu_scu_handler>
1c0104e0:	000280e7          	jalr	t0
1c0104e4:	090000ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0104e8:	4082                	lw	ra,0(sp)
1c0104ea:	6149                	addi	sp,sp,144
1c0104ec:	34011173          	csrrw	sp,0x340,sp
1c0104f0:	30200073          	mret

1c0104f4 <ecall_handler>:
1c0104f4:	1101                	addi	sp,sp,-32
1c0104f6:	c006                	sw	ra,0(sp)
1c0104f8:	c216                	sw	t0,4(sp)
1c0104fa:	c41a                	sw	t1,8(sp)
1c0104fc:	00f8f293          	andi	t0,a7,15
1c010500:	0000c317          	auipc	t1,0xc
1c010504:	0bc30313          	addi	t1,t1,188 # 1c01c5bc <_ecall_vector>
1c010508:	028a                	slli	t0,t0,0x2
1c01050a:	9316                	add	t1,t1,t0
1c01050c:	00032283          	lw	t0,0(t1)
1c010510:	000280e7          	jalr	t0
1c010514:	4322                	lw	t1,8(sp)
1c010516:	4292                	lw	t0,4(sp)
1c010518:	4082                	lw	ra,0(sp)
1c01051a:	6105                	addi	sp,sp,32
1c01051c:	8082                	ret

1c01051e <default_handler>:
1c01051e:	30200073          	mret

1c010522 <ill_ins_handler>:
1c010522:	34102573          	csrr	a0,mepc
1c010526:	0000a297          	auipc	t0,0xa
1c01052a:	94c28293          	addi	t0,t0,-1716 # 1c019e72 <_ill_insn_print>
1c01052e:	9282                	jalr	t0

1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>:
1c010530:	c22a                	sw	a0,4(sp)
1c010532:	c42e                	sw	a1,8(sp)
1c010534:	c632                	sw	a2,12(sp)
1c010536:	c836                	sw	a3,16(sp)
1c010538:	ca3a                	sw	a4,20(sp)
1c01053a:	cc3e                	sw	a5,24(sp)
1c01053c:	ce42                	sw	a6,28(sp)
1c01053e:	d046                	sw	a7,32(sp)
1c010540:	d216                	sw	t0,36(sp)
1c010542:	d41a                	sw	t1,40(sp)
1c010544:	d61e                	sw	t2,44(sp)
1c010546:	d872                	sw	t3,48(sp)
1c010548:	da76                	sw	t4,52(sp)
1c01054a:	dc7a                	sw	t5,56(sp)
1c01054c:	de7e                	sw	t6,60(sp)
1c01054e:	7c0022f3          	csrr	t0,0x7c0
1c010552:	dc96                	sw	t0,120(sp)
1c010554:	7c102373          	csrr	t1,priv_emstatus
1c010558:	de9a                	sw	t1,124(sp)
1c01055a:	7c2023f3          	csrr	t2,priv_mepc
1c01055e:	c11e                	sw	t2,128(sp)
1c010560:	7c4022f3          	csrr	t0,0x7c4
1c010564:	c316                	sw	t0,132(sp)
1c010566:	7c502373          	csrr	t1,0x7c5
1c01056a:	c51a                	sw	t1,136(sp)
1c01056c:	7c6023f3          	csrr	t2,0x7c6
1c010570:	c71e                	sw	t2,140(sp)
1c010572:	8082                	ret

1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>:
1c010574:	43ba                	lw	t2,140(sp)
1c010576:	7c639073          	csrw	0x7c6,t2
1c01057a:	432a                	lw	t1,136(sp)
1c01057c:	7c531073          	csrw	0x7c5,t1
1c010580:	429a                	lw	t0,132(sp)
1c010582:	7c429073          	csrw	0x7c4,t0
1c010586:	438a                	lw	t2,128(sp)
1c010588:	7c239073          	csrw	priv_mepc,t2
1c01058c:	5376                	lw	t1,124(sp)
1c01058e:	7c131073          	csrw	priv_emstatus,t1
1c010592:	52e6                	lw	t0,120(sp)
1c010594:	7c029073          	csrw	0x7c0,t0
1c010598:	5ff2                	lw	t6,60(sp)
1c01059a:	5f62                	lw	t5,56(sp)
1c01059c:	5ed2                	lw	t4,52(sp)
1c01059e:	5e42                	lw	t3,48(sp)
1c0105a0:	53b2                	lw	t2,44(sp)
1c0105a2:	5322                	lw	t1,40(sp)
1c0105a4:	5292                	lw	t0,36(sp)
1c0105a6:	5882                	lw	a7,32(sp)
1c0105a8:	4872                	lw	a6,28(sp)
1c0105aa:	47e2                	lw	a5,24(sp)
1c0105ac:	4752                	lw	a4,20(sp)
1c0105ae:	46c2                	lw	a3,16(sp)
1c0105b0:	4632                	lw	a2,12(sp)
1c0105b2:	45a2                	lw	a1,8(sp)
1c0105b4:	4512                	lw	a0,4(sp)
1c0105b6:	8082                	ret

1c0105b8 <ASM_FUNC_SAVE_SAVE_CONTEXT>:
1c0105b8:	c0a2                	sw	s0,64(sp)
1c0105ba:	c2a6                	sw	s1,68(sp)
1c0105bc:	c4ca                	sw	s2,72(sp)
1c0105be:	c6ce                	sw	s3,76(sp)
1c0105c0:	c8d2                	sw	s4,80(sp)
1c0105c2:	cad6                	sw	s5,84(sp)
1c0105c4:	ccda                	sw	s6,88(sp)
1c0105c6:	cede                	sw	s7,92(sp)
1c0105c8:	d0e2                	sw	s8,96(sp)
1c0105ca:	d2e6                	sw	s9,100(sp)
1c0105cc:	d4ea                	sw	s10,104(sp)
1c0105ce:	d6ee                	sw	s11,108(sp)
1c0105d0:	8082                	ret

1c0105d2 <ASM_FUNC_RESTORE_SAVE_CONTEXT>:
1c0105d2:	5db6                	lw	s11,108(sp)
1c0105d4:	5d26                	lw	s10,104(sp)
1c0105d6:	5c96                	lw	s9,100(sp)
1c0105d8:	5c06                	lw	s8,96(sp)
1c0105da:	4bf6                	lw	s7,92(sp)
1c0105dc:	4b66                	lw	s6,88(sp)
1c0105de:	4ad6                	lw	s5,84(sp)
1c0105e0:	4a46                	lw	s4,80(sp)
1c0105e2:	49b6                	lw	s3,76(sp)
1c0105e4:	4926                	lw	s2,72(sp)
1c0105e6:	4496                	lw	s1,68(sp)
1c0105e8:	4406                	lw	s0,64(sp)
1c0105ea:	8082                	ret

1c0105ec <ASM_FUNC_SAVE_CSR_CONTEXT>:
1c0105ec:	300022f3          	csrr	t0,mstatus
1c0105f0:	d896                	sw	t0,112(sp)
1c0105f2:	341022f3          	csrr	t0,mepc
1c0105f6:	da96                	sw	t0,116(sp)
1c0105f8:	8082                	ret

1c0105fa <ASM_FUNC_RESTORE_CSR_CONTEXT>:
1c0105fa:	52d6                	lw	t0,116(sp)
1c0105fc:	34129073          	csrw	mepc,t0
1c010600:	52c6                	lw	t0,112(sp)
1c010602:	30029073          	csrw	mstatus,t0
1c010606:	8082                	ret

1c010608 <ASM_FUNC_SAVE_ADDITIONAL_CONTEXT>:
1c010608:	7c0022f3          	csrr	t0,0x7c0
1c01060c:	dc96                	sw	t0,120(sp)
1c01060e:	7c102373          	csrr	t1,priv_emstatus
1c010612:	de9a                	sw	t1,124(sp)
1c010614:	7c2023f3          	csrr	t2,priv_mepc
1c010618:	c11e                	sw	t2,128(sp)
1c01061a:	7c4022f3          	csrr	t0,0x7c4
1c01061e:	c316                	sw	t0,132(sp)
1c010620:	7c502373          	csrr	t1,0x7c5
1c010624:	c51a                	sw	t1,136(sp)
1c010626:	7c6023f3          	csrr	t2,0x7c6
1c01062a:	c71e                	sw	t2,140(sp)
1c01062c:	8082                	ret

1c01062e <ASM_FUNC_RESTORE_ADDITIONAL_CONTEXT>:
1c01062e:	43ba                	lw	t2,140(sp)
1c010630:	7c639073          	csrw	0x7c6,t2
1c010634:	432a                	lw	t1,136(sp)
1c010636:	7c531073          	csrw	0x7c5,t1
1c01063a:	429a                	lw	t0,132(sp)
1c01063c:	7c429073          	csrw	0x7c4,t0
1c010640:	438a                	lw	t2,128(sp)
1c010642:	7c239073          	csrw	priv_mepc,t2
1c010646:	5376                	lw	t1,124(sp)
1c010648:	7c131073          	csrw	priv_emstatus,t1
1c01064c:	52e6                	lw	t0,120(sp)
1c01064e:	7c029073          	csrw	0x7c0,t0
1c010652:	8082                	ret

1c010654 <__pi_gpio_irq_handler>:
1c010654:	34011173          	csrrw	sp,0x340,sp
1c010658:	7175                	addi	sp,sp,-144
1c01065a:	c006                	sw	ra,0(sp)
1c01065c:	ed5ff0ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c010660:	00006533          	or	a0,zero,zero
1c010664:	00007297          	auipc	t0,0x7
1c010668:	a9228293          	addi	t0,t0,-1390 # 1c0170f6 <__pi_gpio_handler>
1c01066c:	000280e7          	jalr	t0
1c010670:	f05ff0ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010674:	4082                	lw	ra,0(sp)
1c010676:	6149                	addi	sp,sp,144
1c010678:	34011173          	csrrw	sp,0x340,sp
1c01067c:	30200073          	mret

1c010680 <pi_octospi0_handler_asm>:
1c010680:	34011173          	csrrw	sp,0x340,sp
1c010684:	7175                	addi	sp,sp,-144
1c010686:	c006                	sw	ra,0(sp)
1c010688:	ea9ff0ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c01068c:	00007297          	auipc	t0,0x7
1c010690:	59228293          	addi	t0,t0,1426 # 1c017c1e <pi_octospi0_handler>
1c010694:	a819                	j	1c0106aa <__octospi_handler_body>

1c010696 <pi_octospi1_handler_asm>:
1c010696:	34011173          	csrrw	sp,0x340,sp
1c01069a:	7175                	addi	sp,sp,-144
1c01069c:	c006                	sw	ra,0(sp)
1c01069e:	e93ff0ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0106a2:	00007297          	auipc	t0,0x7
1c0106a6:	5a428293          	addi	t0,t0,1444 # 1c017c46 <pi_octospi1_handler>

1c0106aa <__octospi_handler_body>:
1c0106aa:	000280e7          	jalr	t0
1c0106ae:	ec7ff0ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0106b2:	4082                	lw	ra,0(sp)
1c0106b4:	6149                	addi	sp,sp,144
1c0106b6:	34011173          	csrrw	sp,0x340,sp
1c0106ba:	30200073          	mret

1c0106be <__pi_timer_fc_0_1_irq_handler>:
1c0106be:	34011173          	csrrw	sp,0x340,sp
1c0106c2:	7175                	addi	sp,sp,-144
1c0106c4:	c006                	sw	ra,0(sp)
1c0106c6:	e6bff0ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0106ca:	00009297          	auipc	t0,0x9
1c0106ce:	fde28293          	addi	t0,t0,-34 # 1c0196a8 <__pi_timer_fc_0_1_handler>
1c0106d2:	000280e7          	jalr	t0
1c0106d6:	e9fff0ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c0106da:	4082                	lw	ra,0(sp)
1c0106dc:	6149                	addi	sp,sp,144
1c0106de:	34011173          	csrrw	sp,0x340,sp
1c0106e2:	30200073          	mret

1c0106e6 <__pi_timer_fc_1_1_irq_handler>:
1c0106e6:	34011173          	csrrw	sp,0x340,sp
1c0106ea:	7139                	addi	sp,sp,-64
1c0106ec:	c006                	sw	ra,0(sp)
1c0106ee:	e43ff0ef          	jal	ra,1c010530 <ASM_FUNC_SAVE_MINIMAL_CONTEXT>
1c0106f2:	00009297          	auipc	t0,0x9
1c0106f6:	04a28293          	addi	t0,t0,74 # 1c01973c <__pi_timer_fc_1_1_handler>
1c0106fa:	000280e7          	jalr	t0
1c0106fe:	e77ff0ef          	jal	ra,1c010574 <ASM_FUNC_RESTORE_MINIMAL_CONTEXT>
1c010702:	4082                	lw	ra,0(sp)
1c010704:	6121                	addi	sp,sp,64
1c010706:	34011173          	csrrw	sp,0x340,sp
1c01070a:	30200073          	mret

1c01070e <mram_program_resume>:
1c01070e:	4d14                	lw	a3,24(a0)
1c010710:	6785                	lui	a5,0x1
1c010712:	4d50                	lw	a2,28(a0)
1c010714:	e8b6b733          	p.bclr	a4,a3,20,11
1c010718:	80078793          	addi	a5,a5,-2048 # 800 <__STACK_IRQ_SIZE>
1c01071c:	8f99                	sub	a5,a5,a4
1c01071e:	494c                	lw	a1,20(a0)
1c010720:	4118                	lw	a4,0(a0)
1c010722:	04c7d7b3          	p.minu	a5,a5,a2
1c010726:	02072803          	lw	a6,32(a4)
1c01072a:	4889                	li	a7,2
1c01072c:	ce88a833          	p.insert	a6,a7,7,8
1c010730:	03072023          	sw	a6,32(a4)
1c010734:	00b72623          	sw	a1,12(a4)
1c010738:	00f72823          	sw	a5,16(a4)
1c01073c:	00d72c23          	sw	a3,24(a4)
1c010740:	04072223          	sw	zero,68(a4)
1c010744:	4809                	li	a6,2
1c010746:	01072a23          	sw	a6,20(a4)
1c01074a:	96be                	add	a3,a3,a5
1c01074c:	95be                	add	a1,a1,a5
1c01074e:	40f607b3          	sub	a5,a2,a5
1c010752:	cd14                	sw	a3,24(a0)
1c010754:	c94c                	sw	a1,20(a0)
1c010756:	cd5c                	sw	a5,28(a0)
1c010758:	8082                	ret

1c01075a <mram_erase_resume>:
1c01075a:	4d5c                	lw	a5,28(a0)
1c01075c:	f647b733          	p.bclr	a4,a5,27,4
1c010760:	c709                	beqz	a4,1c01076a <mram_erase_resume+0x10>
1c010762:	c607b7b3          	p.bclr	a5,a5,3,0
1c010766:	07c1                	addi	a5,a5,16
1c010768:	cd5c                	sw	a5,28(a0)
1c01076a:	4d14                	lw	a3,24(a0)
1c01076c:	6785                	lui	a5,0x1
1c01076e:	80078793          	addi	a5,a5,-2048 # 800 <__STACK_IRQ_SIZE>
1c010772:	e8b6b733          	p.bclr	a4,a3,20,11
1c010776:	8f99                	sub	a5,a5,a4
1c010778:	4d58                	lw	a4,28(a0)
1c01077a:	04e7d7b3          	p.minu	a5,a5,a4
1c01077e:	473d                	li	a4,15
1c010780:	04f76263          	bltu	a4,a5,1c0107c4 <mram_erase_resume+0x6a>
1c010784:	47c1                	li	a5,16
1c010786:	cd5c                	sw	a5,28(a0)
1c010788:	4118                	lw	a4,0(a0)
1c01078a:	02072603          	lw	a2,32(a4)
1c01078e:	45c1                	li	a1,16
1c010790:	ce85a633          	p.insert	a2,a1,7,8
1c010794:	02c72023          	sw	a2,32(a4)
1c010798:	04072223          	sw	zero,68(a4)
1c01079c:	1d000637          	lui	a2,0x1d000
1c0107a0:	9636                	add	a2,a2,a3
1c0107a2:	02c72223          	sw	a2,36(a4)
1c0107a6:	0047d613          	srli	a2,a5,0x4
1c0107aa:	167d                	addi	a2,a2,-1
1c0107ac:	02c72423          	sw	a2,40(a4)
1c0107b0:	4609                	li	a2,2
1c0107b2:	00c72a23          	sw	a2,20(a4)
1c0107b6:	4d58                	lw	a4,28(a0)
1c0107b8:	96be                	add	a3,a3,a5
1c0107ba:	cd14                	sw	a3,24(a0)
1c0107bc:	40f707b3          	sub	a5,a4,a5
1c0107c0:	cd5c                	sw	a5,28(a0)
1c0107c2:	8082                	ret
1c0107c4:	f647b733          	p.bclr	a4,a5,27,4
1c0107c8:	d361                	beqz	a4,1c010788 <mram_erase_resume+0x2e>
1c0107ca:	c607b7b3          	p.bclr	a5,a5,3,0
1c0107ce:	07c1                	addi	a5,a5,16
1c0107d0:	bf65                	j	1c010788 <mram_erase_resume+0x2e>

1c0107d2 <mram_erase_async>:
1c0107d2:	1141                	addi	sp,sp,-16
1c0107d4:	c606                	sw	ra,12(sp)
1c0107d6:	c422                	sw	s0,8(sp)
1c0107d8:	4508                	lw	a0,8(a0)
1c0107da:	30047473          	csrrci	s0,mstatus,8
1c0107de:	415c                	lw	a5,4(a0)
1c0107e0:	ef91                	bnez	a5,1c0107fc <mram_erase_async+0x2a>
1c0107e2:	4785                	li	a5,1
1c0107e4:	c154                	sw	a3,4(a0)
1c0107e6:	cd0c                	sw	a1,24(a0)
1c0107e8:	cd50                	sw	a2,28(a0)
1c0107ea:	06f50c23          	sb	a5,120(a0)
1c0107ee:	37b5                	jal	1c01075a <mram_erase_resume>
1c0107f0:	30041073          	csrw	mstatus,s0
1c0107f4:	40b2                	lw	ra,12(sp)
1c0107f6:	4422                	lw	s0,8(sp)
1c0107f8:	0141                	addi	sp,sp,16
1c0107fa:	8082                	ret
1c0107fc:	4789                	li	a5,2
1c0107fe:	ce9c                	sw	a5,24(a3)
1c010800:	451c                	lw	a5,8(a0)
1c010802:	cecc                	sw	a1,28(a3)
1c010804:	d290                	sw	a2,32(a3)
1c010806:	c799                	beqz	a5,1c010814 <mram_erase_async+0x42>
1c010808:	455c                	lw	a5,12(a0)
1c01080a:	c394                	sw	a3,0(a5)
1c01080c:	c554                	sw	a3,12(a0)
1c01080e:	0006a023          	sw	zero,0(a3)
1c010812:	bff9                	j	1c0107f0 <mram_erase_async+0x1e>
1c010814:	c514                	sw	a3,8(a0)
1c010816:	bfdd                	j	1c01080c <mram_erase_async+0x3a>

1c010818 <mram_erase_sector_async>:
1c010818:	451c                	lw	a5,8(a0)
1c01081a:	30047773          	csrrci	a4,mstatus,8
1c01081e:	43d4                	lw	a3,4(a5)
1c010820:	ea8d                	bnez	a3,1c010852 <mram_erase_sector_async+0x3a>
1c010822:	c3d0                	sw	a2,4(a5)
1c010824:	439c                	lw	a5,0(a5)
1c010826:	0207a683          	lw	a3,32(a5)
1c01082a:	4621                	li	a2,8
1c01082c:	ce8626b3          	p.insert	a3,a2,7,8
1c010830:	02d7a023          	sw	a3,32(a5)
1c010834:	0407a223          	sw	zero,68(a5)
1c010838:	1d0006b7          	lui	a3,0x1d000
1c01083c:	95b6                	add	a1,a1,a3
1c01083e:	02b7a223          	sw	a1,36(a5)
1c010842:	0207a423          	sw	zero,40(a5)
1c010846:	4689                	li	a3,2
1c010848:	00d7aa23          	sw	a3,20(a5)
1c01084c:	30071073          	csrw	mstatus,a4
1c010850:	8082                	ret
1c010852:	4685                	li	a3,1
1c010854:	ce14                	sw	a3,24(a2)
1c010856:	4794                	lw	a3,8(a5)
1c010858:	ce4c                	sw	a1,28(a2)
1c01085a:	02062023          	sw	zero,32(a2) # 1d000020 <__heapl2ram_size+0xe70020>
1c01085e:	c699                	beqz	a3,1c01086c <mram_erase_sector_async+0x54>
1c010860:	47d4                	lw	a3,12(a5)
1c010862:	c290                	sw	a2,0(a3)
1c010864:	c7d0                	sw	a2,12(a5)
1c010866:	00062023          	sw	zero,0(a2)
1c01086a:	b7cd                	j	1c01084c <mram_erase_sector_async+0x34>
1c01086c:	c790                	sw	a2,8(a5)
1c01086e:	bfdd                	j	1c010864 <mram_erase_sector_async+0x4c>

1c010870 <mram_erase_chip_async>:
1c010870:	451c                	lw	a5,8(a0)
1c010872:	30047773          	csrrci	a4,mstatus,8
1c010876:	43d4                	lw	a3,4(a5)
1c010878:	e295                	bnez	a3,1c01089c <mram_erase_chip_async+0x2c>
1c01087a:	c3cc                	sw	a1,4(a5)
1c01087c:	439c                	lw	a5,0(a5)
1c01087e:	0207a683          	lw	a3,32(a5)
1c010882:	4611                	li	a2,4
1c010884:	ce8626b3          	p.insert	a3,a2,7,8
1c010888:	02d7a023          	sw	a3,32(a5)
1c01088c:	0407a223          	sw	zero,68(a5)
1c010890:	4689                	li	a3,2
1c010892:	00d7aa23          	sw	a3,20(a5)
1c010896:	30071073          	csrw	mstatus,a4
1c01089a:	8082                	ret
1c01089c:	4794                	lw	a3,8(a5)
1c01089e:	0005ac23          	sw	zero,24(a1)
1c0108a2:	0005ae23          	sw	zero,28(a1)
1c0108a6:	0205a023          	sw	zero,32(a1)
1c0108aa:	c699                	beqz	a3,1c0108b8 <mram_erase_chip_async+0x48>
1c0108ac:	47d4                	lw	a3,12(a5)
1c0108ae:	c28c                	sw	a1,0(a3)
1c0108b0:	c7cc                	sw	a1,12(a5)
1c0108b2:	0005a023          	sw	zero,0(a1)
1c0108b6:	b7c5                	j	1c010896 <mram_erase_chip_async+0x26>
1c0108b8:	c78c                	sw	a1,8(a5)
1c0108ba:	bfdd                	j	1c0108b0 <mram_erase_chip_async+0x40>

1c0108bc <pos_mram_handle_event>:
1c0108bc:	4ddc                	lw	a5,28(a1)
1c0108be:	852e                	mv	a0,a1
1c0108c0:	c791                	beqz	a5,1c0108cc <pos_mram_handle_event+0x10>
1c0108c2:	0785c783          	lbu	a5,120(a1)
1c0108c6:	c391                	beqz	a5,1c0108ca <pos_mram_handle_event+0xe>
1c0108c8:	bd49                	j	1c01075a <mram_erase_resume>
1c0108ca:	b591                	j	1c01070e <mram_program_resume>
1c0108cc:	1141                	addi	sp,sp,-16
1c0108ce:	4598                	lw	a4,8(a1)
1c0108d0:	c422                	sw	s0,8(sp)
1c0108d2:	c606                	sw	ra,12(sp)
1c0108d4:	41c0                	lw	s0,4(a1)
1c0108d6:	0005a223          	sw	zero,4(a1)
1c0108da:	cf11                	beqz	a4,1c0108f6 <pos_mram_handle_event+0x3a>
1c0108dc:	431c                	lw	a5,0(a4)
1c0108de:	c59c                	sw	a5,8(a1)
1c0108e0:	4f1c                	lw	a5,24(a4)
1c0108e2:	00b02423          	sw	a1,8(zero) # 8 <__pi_cluster_pool+0x4>
1c0108e6:	0427ad63          	p.beqimm	a5,2,1c010940 <pos_mram_handle_event+0x84>
1c0108ea:	4689                	li	a3,2
1c0108ec:	00f6eb63          	bltu	a3,a5,1c010902 <pos_mram_handle_event+0x46>
1c0108f0:	cf9d                	beqz	a5,1c01092e <pos_mram_handle_event+0x72>
1c0108f2:	0417a263          	p.beqimm	a5,1,1c010936 <pos_mram_handle_event+0x7a>
1c0108f6:	8522                	mv	a0,s0
1c0108f8:	4422                	lw	s0,8(sp)
1c0108fa:	40b2                	lw	ra,12(sp)
1c0108fc:	0141                	addi	sp,sp,16
1c0108fe:	6220906f          	j	1c019f20 <__pi_os_evt_push>
1c010902:	0447ac63          	p.beqimm	a5,4,1c01095a <pos_mram_handle_event+0x9e>
1c010906:	4691                	li	a3,4
1c010908:	04d7e263          	bltu	a5,a3,1c01094c <pos_mram_handle_event+0x90>
1c01090c:	fe57b5e3          	p.bneimm	a5,5,1c0108f6 <pos_mram_handle_event+0x3a>
1c010910:	88ba                	mv	a7,a4
1c010912:	03072803          	lw	a6,48(a4)
1c010916:	575c                	lw	a5,44(a4)
1c010918:	0248a683          	lw	a3,36(a7)
1c01091c:	5718                	lw	a4,40(a4)
1c01091e:	0208a603          	lw	a2,32(a7)
1c010922:	01c8a583          	lw	a1,28(a7)
1c010926:	4501                	li	a0,0
1c010928:	2be050ef          	jal	ra,1c015be6 <mram_copy_2d_async>
1c01092c:	b7e9                	j	1c0108f6 <pos_mram_handle_event+0x3a>
1c01092e:	85ba                	mv	a1,a4
1c010930:	4501                	li	a0,0
1c010932:	3f3d                	jal	1c010870 <mram_erase_chip_async>
1c010934:	b7c9                	j	1c0108f6 <pos_mram_handle_event+0x3a>
1c010936:	4f4c                	lw	a1,28(a4)
1c010938:	863a                	mv	a2,a4
1c01093a:	4501                	li	a0,0
1c01093c:	3df1                	jal	1c010818 <mram_erase_sector_async>
1c01093e:	bf65                	j	1c0108f6 <pos_mram_handle_event+0x3a>
1c010940:	5310                	lw	a2,32(a4)
1c010942:	4f4c                	lw	a1,28(a4)
1c010944:	86ba                	mv	a3,a4
1c010946:	4501                	li	a0,0
1c010948:	3569                	jal	1c0107d2 <mram_erase_async>
1c01094a:	b775                	j	1c0108f6 <pos_mram_handle_event+0x3a>
1c01094c:	5354                	lw	a3,36(a4)
1c01094e:	5310                	lw	a2,32(a4)
1c010950:	4f4c                	lw	a1,28(a4)
1c010952:	4501                	li	a0,0
1c010954:	3d8050ef          	jal	ra,1c015d2c <mram_program_async>
1c010958:	bf79                	j	1c0108f6 <pos_mram_handle_event+0x3a>
1c01095a:	5354                	lw	a3,36(a4)
1c01095c:	5310                	lw	a2,32(a4)
1c01095e:	4f4c                	lw	a1,28(a4)
1c010960:	4501                	li	a0,0
1c010962:	444050ef          	jal	ra,1c015da6 <mram_read_async>
1c010966:	bf41                	j	1c0108f6 <pos_mram_handle_event+0x3a>

1c010968 <pi_mram_baudrate_set>:
1c010968:	1141                	addi	sp,sp,-16
1c01096a:	c422                	sw	s0,8(sp)
1c01096c:	842a                	mv	s0,a0
1c01096e:	4501                	li	a0,0
1c010970:	c606                	sw	ra,12(sp)
1c010972:	2675                	jal	1c010d1e <pi_freq_get>
1c010974:	9522                	add	a0,a0,s0
1c010976:	157d                	addi	a0,a0,-1
1c010978:	02854533          	div	a0,a0,s0
1c01097c:	1a1037b7          	lui	a5,0x1a103
1c010980:	b0078793          	addi	a5,a5,-1280 # 1a102b00 <__l1_heapsram_end+0xa0e2b00>
1c010984:	c2854533          	p.bset	a0,a0,1,8
1c010988:	02a7a623          	sw	a0,44(a5)
1c01098c:	40b2                	lw	ra,12(sp)
1c01098e:	4422                	lw	s0,8(sp)
1c010990:	0141                	addi	sp,sp,16
1c010992:	8082                	ret

1c010994 <mx25u_erase_resume>:
1c010994:	1101                	addi	sp,sp,-32
1c010996:	cc22                	sw	s0,24(sp)
1c010998:	4500                	lw	s0,8(a0)
1c01099a:	ca26                	sw	s1,20(sp)
1c01099c:	c84a                	sw	s2,16(sp)
1c01099e:	14042483          	lw	s1,320(s0)
1c0109a2:	ce06                	sw	ra,28(sp)
1c0109a4:	c64e                	sw	s3,12(sp)
1c0109a6:	892a                	mv	s2,a0
1c0109a8:	e895                	bnez	s1,1c0109dc <mx25u_erase_resume+0x48>
1c0109aa:	300474f3          	csrrci	s1,mstatus,8
1c0109ae:	5048                	lw	a0,36(s0)
1c0109b0:	570090ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c0109b4:	4c54                	lw	a3,28(s0)
1c0109b6:	02042223          	sw	zero,36(s0)
1c0109ba:	c299                	beqz	a3,1c0109c0 <mx25u_erase_resume+0x2c>
1c0109bc:	429c                	lw	a5,0(a3)
1c0109be:	cc5c                	sw	a5,28(s0)
1c0109c0:	30049073          	csrw	mstatus,s1
1c0109c4:	cea9                	beqz	a3,1c010a1e <mx25u_erase_resume+0x8a>
1c0109c6:	4462                	lw	s0,24(sp)
1c0109c8:	40f2                	lw	ra,28(sp)
1c0109ca:	44d2                	lw	s1,20(sp)
1c0109cc:	49b2                	lw	s3,12(sp)
1c0109ce:	5290                	lw	a2,32(a3)
1c0109d0:	4ecc                	lw	a1,28(a3)
1c0109d2:	854a                	mv	a0,s2
1c0109d4:	4942                	lw	s2,16(sp)
1c0109d6:	6105                	addi	sp,sp,32
1c0109d8:	6cb0506f          	j	1c0168a2 <mx25u_erase_async>
1c0109dc:	13c42983          	lw	s3,316(s0)
1c0109e0:	6785                	lui	a5,0x1
1c0109e2:	1c0115b7          	lui	a1,0x1c011
1c0109e6:	e6c9b733          	p.bclr	a4,s3,19,12
1c0109ea:	8f99                	sub	a5,a5,a4
1c0109ec:	862a                	mv	a2,a0
1c0109ee:	99458593          	addi	a1,a1,-1644 # 1c010994 <mx25u_erase_resume>
1c0109f2:	08040513          	addi	a0,s0,128
1c0109f6:	04f4d4b3          	p.minu	s1,s1,a5
1c0109fa:	4f6090ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c0109fe:	862a                	mv	a2,a0
1c010a00:	85ce                	mv	a1,s3
1c010a02:	854a                	mv	a0,s2
1c010a04:	611050ef          	jal	ra,1c016814 <mx25u_erase_sector_async>
1c010a08:	13c42783          	lw	a5,316(s0)
1c010a0c:	97a6                	add	a5,a5,s1
1c010a0e:	12f42e23          	sw	a5,316(s0)
1c010a12:	14042783          	lw	a5,320(s0)
1c010a16:	409784b3          	sub	s1,a5,s1
1c010a1a:	14942023          	sw	s1,320(s0)
1c010a1e:	40f2                	lw	ra,28(sp)
1c010a20:	4462                	lw	s0,24(sp)
1c010a22:	44d2                	lw	s1,20(sp)
1c010a24:	4942                	lw	s2,16(sp)
1c010a26:	49b2                	lw	s3,12(sp)
1c010a28:	6105                	addi	sp,sp,32
1c010a2a:	8082                	ret

1c010a2c <mx25u_program_resume>:
1c010a2c:	7179                	addi	sp,sp,-48
1c010a2e:	d422                	sw	s0,40(sp)
1c010a30:	4500                	lw	s0,8(a0)
1c010a32:	d606                	sw	ra,44(sp)
1c010a34:	d226                	sw	s1,36(sp)
1c010a36:	13842703          	lw	a4,312(s0)
1c010a3a:	d04a                	sw	s2,32(sp)
1c010a3c:	eb01                	bnez	a4,1c010a4c <mx25u_program_resume+0x20>
1c010a3e:	5422                	lw	s0,40(sp)
1c010a40:	50b2                	lw	ra,44(sp)
1c010a42:	5492                	lw	s1,36(sp)
1c010a44:	5902                	lw	s2,32(sp)
1c010a46:	6145                	addi	sp,sp,48
1c010a48:	6cb0506f          	j	1c016912 <mx25u_handle_pending_task>
1c010a4c:	13042483          	lw	s1,304(s0)
1c010a50:	10000793          	li	a5,256
1c010a54:	13442903          	lw	s2,308(s0)
1c010a58:	ee84b6b3          	p.bclr	a3,s1,23,8
1c010a5c:	8f95                	sub	a5,a5,a3
1c010a5e:	04f757b3          	p.minu	a5,a4,a5
1c010a62:	00f486b3          	add	a3,s1,a5
1c010a66:	12d42823          	sw	a3,304(s0)
1c010a6a:	8f1d                	sub	a4,a4,a5
1c010a6c:	00f906b3          	add	a3,s2,a5
1c010a70:	12d42a23          	sw	a3,308(s0)
1c010a74:	12e42c23          	sw	a4,312(s0)
1c010a78:	862a                	mv	a2,a0
1c010a7a:	8522                	mv	a0,s0
1c010a7c:	cc3e                	sw	a5,24(sp)
1c010a7e:	ce32                	sw	a2,28(sp)
1c010a80:	53b050ef          	jal	ra,1c0167ba <mx25u_write_enable>
1c010a84:	4672                	lw	a2,28(sp)
1c010a86:	1c0175b7          	lui	a1,0x1c017
1c010a8a:	c1a58593          	addi	a1,a1,-998 # 1c016c1a <mx25u_check_program>
1c010a8e:	02840513          	addi	a0,s0,40
1c010a92:	45e090ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c010a96:	47e2                	lw	a5,24(sp)
1c010a98:	1c01c837          	lui	a6,0x1c01c
1c010a9c:	c002                	sw	zero,0(sp)
1c010a9e:	88aa                	mv	a7,a0
1c010aa0:	864a                	mv	a2,s2
1c010aa2:	85a6                	mv	a1,s1
1c010aa4:	8522                	mv	a0,s0
1c010aa6:	52480813          	addi	a6,a6,1316 # 1c01c524 <mx25u_program_op>
1c010aaa:	873e                	mv	a4,a5
1c010aac:	86be                	mv	a3,a5
1c010aae:	0b0070ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c010ab2:	50b2                	lw	ra,44(sp)
1c010ab4:	5422                	lw	s0,40(sp)
1c010ab6:	5492                	lw	s1,36(sp)
1c010ab8:	5902                	lw	s2,32(sp)
1c010aba:	6145                	addi	sp,sp,48
1c010abc:	8082                	ret

1c010abe <mx25u_program_async>:
1c010abe:	1101                	addi	sp,sp,-32
1c010ac0:	cc22                	sw	s0,24(sp)
1c010ac2:	4500                	lw	s0,8(a0)
1c010ac4:	c452                	sw	s4,8(sp)
1c010ac6:	8a2e                	mv	s4,a1
1c010ac8:	ca26                	sw	s1,20(sp)
1c010aca:	c84a                	sw	s2,16(sp)
1c010acc:	c64e                	sw	s3,12(sp)
1c010ace:	84aa                	mv	s1,a0
1c010ad0:	89b2                	mv	s3,a2
1c010ad2:	8936                	mv	s2,a3
1c010ad4:	85ba                	mv	a1,a4
1c010ad6:	87b6                	mv	a5,a3
1c010ad8:	8732                	mv	a4,a2
1c010ada:	4881                	li	a7,0
1c010adc:	4801                	li	a6,0
1c010ade:	86d2                	mv	a3,s4
1c010ae0:	4601                	li	a2,0
1c010ae2:	8522                	mv	a0,s0
1c010ae4:	ce06                	sw	ra,28(sp)
1c010ae6:	08d050ef          	jal	ra,1c016372 <mx25u_stall_task>
1c010aea:	e105                	bnez	a0,1c010b0a <mx25u_program_async+0x4c>
1c010aec:	13442823          	sw	s4,304(s0)
1c010af0:	13342a23          	sw	s3,308(s0)
1c010af4:	13242c23          	sw	s2,312(s0)
1c010af8:	4462                	lw	s0,24(sp)
1c010afa:	40f2                	lw	ra,28(sp)
1c010afc:	4942                	lw	s2,16(sp)
1c010afe:	49b2                	lw	s3,12(sp)
1c010b00:	4a22                	lw	s4,8(sp)
1c010b02:	8526                	mv	a0,s1
1c010b04:	44d2                	lw	s1,20(sp)
1c010b06:	6105                	addi	sp,sp,32
1c010b08:	b715                	j	1c010a2c <mx25u_program_resume>
1c010b0a:	40f2                	lw	ra,28(sp)
1c010b0c:	4462                	lw	s0,24(sp)
1c010b0e:	44d2                	lw	s1,20(sp)
1c010b10:	4942                	lw	s2,16(sp)
1c010b12:	49b2                	lw	s3,12(sp)
1c010b14:	4a22                	lw	s4,8(sp)
1c010b16:	6105                	addi	sp,sp,32
1c010b18:	8082                	ret

1c010b1a <__pi_pmu_scu_handler>:
1c010b1a:	1141                	addi	sp,sp,-16
1c010b1c:	08000793          	li	a5,128
1c010b20:	1a107737          	lui	a4,0x1a107
1c010b24:	c606                	sw	ra,12(sp)
1c010b26:	c422                	sw	s0,8(sp)
1c010b28:	c226                	sw	s1,4(sp)
1c010b2a:	00f72823          	sw	a5,16(a4) # 1a107010 <__l1_heapsram_end+0xa0e7010>
1c010b2e:	01402503          	lw	a0,20(zero) # 14 <__pi_pmu_scu_event>
1c010b32:	00002823          	sw	zero,16(zero) # 10 <__pi_pmu_pending_sequence>
1c010b36:	84be                	mv	s1,a5
1c010b38:	c509                	beqz	a0,1c010b42 <__pi_pmu_scu_handler+0x28>
1c010b3a:	3e6090ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c010b3e:	00002a23          	sw	zero,20(zero) # 14 <__pi_pmu_scu_event>
1c010b42:	01802683          	lw	a3,24(zero) # 18 <__pi_pmu_pending_requests>
1c010b46:	ceb9                	beqz	a3,1c010ba4 <__pi_pmu_scu_handler+0x8a>
1c010b48:	4298                	lw	a4,0(a3)
1c010b4a:	1c01e537          	lui	a0,0x1c01e
1c010b4e:	c1c52803          	lw	a6,-996(a0) # 1c01dc1c <__pi_pmu_domains_on>
1c010b52:	00e02c23          	sw	a4,24(zero) # 18 <__pi_pmu_pending_requests>
1c010b56:	4e9c                	lw	a5,24(a3)
1c010b58:	4585                	li	a1,1
1c010b5a:	4ed0                	lw	a2,28(a3)
1c010b5c:	00f59733          	sll	a4,a1,a5
1c010b60:	fff74713          	not	a4,a4
1c010b64:	01077733          	and	a4,a4,a6
1c010b68:	00f61833          	sll	a6,a2,a5
1c010b6c:	01076733          	or	a4,a4,a6
1c010b70:	c0e52e23          	sw	a4,-996(a0)
1c010b74:	00b02823          	sw	a1,16(zero) # 10 <__pi_pmu_pending_sequence>
1c010b78:	0047a563          	p.beqimm	a5,4,1c010b82 <__pi_pmu_scu_handler+0x68>
1c010b7c:	078d                	addi	a5,a5,3
1c010b7e:	0786                	slli	a5,a5,0x1
1c010b80:	97b2                	add	a5,a5,a2
1c010b82:	4705                	li	a4,1
1c010b84:	f647b7b3          	p.bclr	a5,a5,27,4
1c010b88:	00f717b3          	sll	a5,a4,a5
1c010b8c:	1a107737          	lui	a4,0x1a107
1c010b90:	00f72423          	sw	a5,8(a4) # 1a107008 <__l1_heapsram_end+0xa0e7008>
1c010b94:	100007b7          	lui	a5,0x10000
1c010b98:	10278793          	addi	a5,a5,258 # 10000102 <__l1_heapsram_start+0x92>
1c010b9c:	00f72223          	sw	a5,4(a4)
1c010ba0:	00d02a23          	sw	a3,20(zero) # 14 <__pi_pmu_scu_event>
1c010ba4:	40b2                	lw	ra,12(sp)
1c010ba6:	4422                	lw	s0,8(sp)
1c010ba8:	4492                	lw	s1,4(sp)
1c010baa:	0141                	addi	sp,sp,16
1c010bac:	8082                	ret

1c010bae <__pi_fll_freq_set>:
1c010bae:	1101                	addi	sp,sp,-32
1c010bb0:	cc22                	sw	s0,24(sp)
1c010bb2:	842a                	mv	s0,a0
1c010bb4:	4501                	li	a0,0
1c010bb6:	c62e                	sw	a1,12(sp)
1c010bb8:	c432                	sw	a2,8(sp)
1c010bba:	ce06                	sw	ra,28(sp)
1c010bbc:	502090ef          	jal	ra,1c01a0be <pi_pmu_voltage_get>
1c010bc0:	28a00793          	li	a5,650
1c010bc4:	45b2                	lw	a1,12(sp)
1c010bc6:	4622                	lw	a2,8(sp)
1c010bc8:	06f50e63          	beq	a0,a5,1c010c44 <__pi_fll_freq_set+0x96>
1c010bcc:	35a4f7b7          	lui	a5,0x35a4f
1c010bd0:	90078793          	addi	a5,a5,-1792 # 35a4e900 <__heapl2ram_size+0x198be900>
1c010bd4:	8785                	srai	a5,a5,0x1
1c010bd6:	02b7d7b3          	divu	a5,a5,a1
1c010bda:	0ff00713          	li	a4,255
1c010bde:	d015f5d3          	fcvt.s.wu	fa1,a1
1c010be2:	1c01c6b7          	lui	a3,0x1c01c
1c010be6:	0d46a503          	lw	a0,212(a3) # 1c01c0d4 <pow10.1847+0x54>
1c010bea:	04e7d7b3          	p.minu	a5,a5,a4
1c010bee:	d007f7d3          	fcvt.s.w	fa5,a5
1c010bf2:	1c01c737          	lui	a4,0x1c01c
1c010bf6:	0d072703          	lw	a4,208(a4) # 1c01c0d0 <pow10.1847+0x50>
1c010bfa:	10f5f5d3          	fmul.s	fa1,fa1,fa5
1c010bfe:	10e5f5d3          	fmul.s	fa1,fa1,fa4
1c010c02:	10e7f753          	fmul.s	fa4,fa5,fa4
1c010c06:	18a5f5d3          	fdiv.s	fa1,fa1,fa0
1c010c0a:	10a5f553          	fmul.s	fa0,fa1,fa0
1c010c0e:	c01595d3          	fcvt.wu.s	a1,fa1,rtz
1c010c12:	18e57553          	fdiv.s	fa0,fa0,fa4
1c010c16:	4218                	lw	a4,0(a2)
1c010c18:	c0151553          	fcvt.wu.s	a0,fa0,rtz
1c010c1c:	c248                	sw	a0,4(a2)
1c010c1e:	00b72823          	sw	a1,16(a4)
1c010c22:	1a1006b7          	lui	a3,0x1a100
1c010c26:	02c6a703          	lw	a4,44(a3) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010c2a:	c01797d3          	fcvt.wu.s	a5,fa5,rtz
1c010c2e:	040e                	slli	s0,s0,0x3
1c010c30:	c4544433          	p.bset	s0,s0,2,5
1c010c34:	8087a733          	p.insertr	a4,a5,s0
1c010c38:	02e6a623          	sw	a4,44(a3)
1c010c3c:	40f2                	lw	ra,28(sp)
1c010c3e:	4462                	lw	s0,24(sp)
1c010c40:	6105                	addi	sp,sp,32
1c010c42:	8082                	ret
1c010c44:	23c347b7          	lui	a5,0x23c34
1c010c48:	60078793          	addi	a5,a5,1536 # 23c34600 <__heapl2ram_size+0x7aa4600>
1c010c4c:	b761                	j	1c010bd4 <__pi_fll_freq_set+0x26>

1c010c4e <__pi_fll_init>:
1c010c4e:	1141                	addi	sp,sp,-16
1c010c50:	00351793          	slli	a5,a0,0x3
1c010c54:	52800613          	li	a2,1320
1c010c58:	c422                	sw	s0,8(sp)
1c010c5a:	963e                	add	a2,a2,a5
1c010c5c:	c606                	sw	ra,12(sp)
1c010c5e:	4200                	lw	s0,0(a2)
1c010c60:	e191                	bnez	a1,1c010c64 <__pi_fll_init+0x16>
1c010c62:	424c                	lw	a1,4(a2)
1c010c64:	37a9                	jal	1c010bae <__pi_fll_freq_set>
1c010c66:	281717b7          	lui	a5,0x28171
1c010c6a:	a7378793          	addi	a5,a5,-1421 # 28170a73 <__heapl2ram_size+0xbfe0a73>
1c010c6e:	00f42623          	sw	a5,12(s0)
1c010c72:	40b2                	lw	ra,12(sp)
1c010c74:	4422                	lw	s0,8(sp)
1c010c76:	4501                	li	a0,0
1c010c78:	0141                	addi	sp,sp,16
1c010c7a:	8082                	ret

1c010c7c <__pi_fll_deinit>:
1c010c7c:	00351713          	slli	a4,a0,0x3
1c010c80:	52800793          	li	a5,1320
1c010c84:	20e7f783          	p.lw	a5,a4(a5)
1c010c88:	0007a623          	sw	zero,12(a5)
1c010c8c:	1a1007b7          	lui	a5,0x1a100
1c010c90:	0307a783          	lw	a5,48(a5) # 1a100030 <__l1_heapsram_end+0xa0e0030>
1c010c94:	0541                	addi	a0,a0,16
1c010c96:	80a7b533          	p.bclrr	a0,a5,a0
1c010c9a:	1a1007b7          	lui	a5,0x1a100
1c010c9e:	02a7a823          	sw	a0,48(a5) # 1a100030 <__l1_heapsram_end+0xa0e0030>
1c010ca2:	8082                	ret

1c010ca4 <__pi_fll_init_all>:
1c010ca4:	1141                	addi	sp,sp,-16
1c010ca6:	c422                	sw	s0,8(sp)
1c010ca8:	1a1007b7          	lui	a5,0x1a100
1c010cac:	c606                	sw	ra,12(sp)
1c010cae:	c226                	sw	s1,4(sp)
1c010cb0:	52800413          	li	s0,1320
1c010cb4:	00878713          	addi	a4,a5,8 # 1a100008 <__l1_heapsram_end+0xa0e0008>
1c010cb8:	c01c                	sw	a5,0(s0)
1c010cba:	c418                	sw	a4,8(s0)
1c010cbc:	01078713          	addi	a4,a5,16
1c010cc0:	07e1                	addi	a5,a5,24
1c010cc2:	c818                	sw	a4,16(s0)
1c010cc4:	cc1c                	sw	a5,24(s0)
1c010cc6:	1a100737          	lui	a4,0x1a100
1c010cca:	01ff07b7          	lui	a5,0x1ff0
1c010cce:	00f72223          	sw	a5,4(a4) # 1a100004 <__l1_heapsram_end+0xa0e0004>
1c010cd2:	02faf4b7          	lui	s1,0x2faf
1c010cd6:	08048593          	addi	a1,s1,128 # 2faf080 <__l1_heapsram_size+0x2f8f0f0>
1c010cda:	4505                	li	a0,1
1c010cdc:	3f8d                	jal	1c010c4e <__pi_fll_init>
1c010cde:	098975b7          	lui	a1,0x9897
1c010ce2:	80058593          	addi	a1,a1,-2048 # 9896800 <__l1_heapsram_size+0x9876870>
1c010ce6:	4501                	li	a0,0
1c010ce8:	379d                	jal	1c010c4e <__pi_fll_init>
1c010cea:	01040613          	addi	a2,s0,16
1c010cee:	08048593          	addi	a1,s1,128
1c010cf2:	4509                	li	a0,2
1c010cf4:	3d6d                	jal	1c010bae <__pi_fll_freq_set>
1c010cf6:	01840613          	addi	a2,s0,24
1c010cfa:	08048593          	addi	a1,s1,128
1c010cfe:	450d                	li	a0,3
1c010d00:	357d                	jal	1c010bae <__pi_fll_freq_set>
1c010d02:	67d1                	lui	a5,0x14
1c010d04:	32178793          	addi	a5,a5,801 # 14321 <__heapfcram_size+0x50e1>
1c010d08:	1a100737          	lui	a4,0x1a100
1c010d0c:	02f72823          	sw	a5,48(a4) # 1a100030 <__l1_heapsram_end+0xa0e0030>
1c010d10:	4785                	li	a5,1
1c010d12:	d01c                	sw	a5,32(s0)
1c010d14:	40b2                	lw	ra,12(sp)
1c010d16:	4422                	lw	s0,8(sp)
1c010d18:	4492                	lw	s1,4(sp)
1c010d1a:	0141                	addi	sp,sp,16
1c010d1c:	8082                	ret

1c010d1e <pi_freq_get>:
1c010d1e:	00351793          	slli	a5,a0,0x3
1c010d22:	52800513          	li	a0,1320
1c010d26:	953e                	add	a0,a0,a5
1c010d28:	4148                	lw	a0,4(a0)
1c010d2a:	8082                	ret

1c010d2c <pi_freq_set>:
1c010d2c:	1141                	addi	sp,sp,-16
1c010d2e:	c422                	sw	s0,8(sp)
1c010d30:	842a                	mv	s0,a0
1c010d32:	4501                	li	a0,0
1c010d34:	c226                	sw	s1,4(sp)
1c010d36:	c606                	sw	ra,12(sp)
1c010d38:	84ae                	mv	s1,a1
1c010d3a:	384090ef          	jal	ra,1c01a0be <pi_pmu_voltage_get>
1c010d3e:	100555b3          	p.exthz	a1,a0
1c010d42:	8522                	mv	a0,s0
1c010d44:	2e0090ef          	jal	ra,1c01a024 <pi_max_freq_get>
1c010d48:	02956063          	bltu	a0,s1,1c010d68 <pi_freq_set+0x3c>
1c010d4c:	52800793          	li	a5,1320
1c010d50:	00341613          	slli	a2,s0,0x3
1c010d54:	963e                	add	a2,a2,a5
1c010d56:	85a6                	mv	a1,s1
1c010d58:	8522                	mv	a0,s0
1c010d5a:	3d91                	jal	1c010bae <__pi_fll_freq_set>
1c010d5c:	4501                	li	a0,0
1c010d5e:	40b2                	lw	ra,12(sp)
1c010d60:	4422                	lw	s0,8(sp)
1c010d62:	4492                	lw	s1,4(sp)
1c010d64:	0141                	addi	sp,sp,16
1c010d66:	8082                	ret
1c010d68:	557d                	li	a0,-1
1c010d6a:	bfd5                	j	1c010d5e <pi_freq_set+0x32>

1c010d6c <pi_freq_div_by>:
1c010d6c:	1a100737          	lui	a4,0x1a100
1c010d70:	02c72783          	lw	a5,44(a4) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010d74:	00351693          	slli	a3,a0,0x3
1c010d78:	00d7d7b3          	srl	a5,a5,a3
1c010d7c:	ee87b7b3          	p.bclr	a5,a5,23,8
1c010d80:	02b787b3          	mul	a5,a5,a1
1c010d84:	0ff00613          	li	a2,255
1c010d88:	02f64463          	blt	a2,a5,1c010db0 <pi_freq_div_by+0x44>
1c010d8c:	02c72703          	lw	a4,44(a4)
1c010d90:	c456c633          	p.bset	a2,a3,2,5
1c010d94:	80c7a733          	p.insertr	a4,a5,a2
1c010d98:	1a1007b7          	lui	a5,0x1a100
1c010d9c:	02e7a623          	sw	a4,44(a5) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010da0:	52800793          	li	a5,1320
1c010da4:	97b6                	add	a5,a5,a3
1c010da6:	43c8                	lw	a0,4(a5)
1c010da8:	02b55533          	divu	a0,a0,a1
1c010dac:	c3c8                	sw	a0,4(a5)
1c010dae:	8082                	ret
1c010db0:	557d                	li	a0,-1
1c010db2:	8082                	ret

1c010db4 <pi_freq_mult_by>:
1c010db4:	1a100737          	lui	a4,0x1a100
1c010db8:	02c72783          	lw	a5,44(a4) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010dbc:	00351693          	slli	a3,a0,0x3
1c010dc0:	00d7d7b3          	srl	a5,a5,a3
1c010dc4:	ee87b7b3          	p.bclr	a5,a5,23,8
1c010dc8:	02b7d633          	divu	a2,a5,a1
1c010dcc:	557d                	li	a0,-1
1c010dce:	02b7f7b3          	remu	a5,a5,a1
1c010dd2:	e785                	bnez	a5,1c010dfa <pi_freq_mult_by+0x46>
1c010dd4:	c21d                	beqz	a2,1c010dfa <pi_freq_mult_by+0x46>
1c010dd6:	02c72703          	lw	a4,44(a4)
1c010dda:	c456c7b3          	p.bset	a5,a3,2,5
1c010dde:	80f62733          	p.insertr	a4,a2,a5
1c010de2:	1a1007b7          	lui	a5,0x1a100
1c010de6:	02e7a623          	sw	a4,44(a5) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010dea:	52800793          	li	a5,1320
1c010dee:	97b6                	add	a5,a5,a3
1c010df0:	43c8                	lw	a0,4(a5)
1c010df2:	02a58533          	mul	a0,a1,a0
1c010df6:	c3c8                	sw	a0,4(a5)
1c010df8:	8082                	ret
1c010dfa:	8082                	ret

1c010dfc <pi_fll_ioctl>:
1c010dfc:	1101                	addi	sp,sp,-32
1c010dfe:	ce06                	sw	ra,28(sp)
1c010e00:	cc22                	sw	s0,24(sp)
1c010e02:	ca26                	sw	s1,20(sp)
1c010e04:	c84a                	sw	s2,16(sp)
1c010e06:	c64e                	sw	s3,12(sp)
1c010e08:	c452                	sw	s4,8(sp)
1c010e0a:	30047973          	csrrci	s2,mstatus,8
1c010e0e:	0c25ac63          	p.beqimm	a1,2,1c010ee6 <pi_fll_ioctl+0xea>
1c010e12:	4789                	li	a5,2
1c010e14:	00b7ef63          	bltu	a5,a1,1c010e32 <pi_fll_ioctl+0x36>
1c010e18:	0615a063          	p.beqimm	a1,1,1c010e78 <pi_fll_ioctl+0x7c>
1c010e1c:	30091073          	csrw	mstatus,s2
1c010e20:	557d                	li	a0,-1
1c010e22:	40f2                	lw	ra,28(sp)
1c010e24:	4462                	lw	s0,24(sp)
1c010e26:	44d2                	lw	s1,20(sp)
1c010e28:	4942                	lw	s2,16(sp)
1c010e2a:	49b2                	lw	s3,12(sp)
1c010e2c:	4a22                	lw	s4,8(sp)
1c010e2e:	6105                	addi	sp,sp,32
1c010e30:	8082                	ret
1c010e32:	872a                	mv	a4,a0
1c010e34:	0235a663          	p.beqimm	a1,3,1c010e60 <pi_fll_ioctl+0x64>
1c010e38:	fe45b2e3          	p.bneimm	a1,4,1c010e1c <pi_fll_ioctl+0x20>
1c010e3c:	0ff00793          	li	a5,255
1c010e40:	557d                	li	a0,-1
1c010e42:	02c7c863          	blt	a5,a2,1c010e72 <pi_fll_ioctl+0x76>
1c010e46:	1a1006b7          	lui	a3,0x1a100
1c010e4a:	02c6a783          	lw	a5,44(a3) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010e4e:	070e                	slli	a4,a4,0x3
1c010e50:	c4574733          	p.bset	a4,a4,2,5
1c010e54:	80e627b3          	p.insertr	a5,a2,a4
1c010e58:	02f6a623          	sw	a5,44(a3)
1c010e5c:	4501                	li	a0,0
1c010e5e:	a811                	j	1c010e72 <pi_fll_ioctl+0x76>
1c010e60:	1a1007b7          	lui	a5,0x1a100
1c010e64:	02c7a783          	lw	a5,44(a5) # 1a10002c <__l1_heapsram_end+0xa0e002c>
1c010e68:	050e                	slli	a0,a0,0x3
1c010e6a:	00a7d533          	srl	a0,a5,a0
1c010e6e:	ee853533          	p.bclr	a0,a0,23,8
1c010e72:	30091073          	csrw	mstatus,s2
1c010e76:	b775                	j	1c010e22 <pi_fll_ioctl+0x26>
1c010e78:	52800493          	li	s1,1320
1c010e7c:	5098                	lw	a4,32(s1)
1c010e7e:	cf11                	beqz	a4,1c010e9a <pi_fll_ioctl+0x9e>
1c010e80:	52800793          	li	a5,1320
1c010e84:	004450fb          	lp.setupi	x1,4,1c010e94 <pi_fll_ioctl+0x98>
1c010e88:	0087a60b          	p.lw	a2,8(a5!)
1c010e8c:	00c62683          	lw	a3,12(a2)
1c010e90:	c016c6b3          	p.bset	a3,a3,0,1
1c010e94:	00d62623          	sw	a3,12(a2)
1c010e98:	b7d1                	j	1c010e5c <pi_fll_ioctl+0x60>
1c010e9a:	842e                	mv	s0,a1
1c010e9c:	1a100a37          	lui	s4,0x1a100
1c010ea0:	01ff07b7          	lui	a5,0x1ff0
1c010ea4:	00fa2223          	sw	a5,4(s4) # 1a100004 <__l1_heapsram_end+0xa0e0004>
1c010ea8:	02faf9b7          	lui	s3,0x2faf
1c010eac:	08098593          	addi	a1,s3,128 # 2faf080 <__l1_heapsram_size+0x2f8f0f0>
1c010eb0:	4505                	li	a0,1
1c010eb2:	3b71                	jal	1c010c4e <__pi_fll_init>
1c010eb4:	098975b7          	lui	a1,0x9897
1c010eb8:	80058593          	addi	a1,a1,-2048 # 9896800 <__l1_heapsram_size+0x9876870>
1c010ebc:	4501                	li	a0,0
1c010ebe:	3b41                	jal	1c010c4e <__pi_fll_init>
1c010ec0:	01048613          	addi	a2,s1,16
1c010ec4:	08098593          	addi	a1,s3,128
1c010ec8:	4509                	li	a0,2
1c010eca:	31d5                	jal	1c010bae <__pi_fll_freq_set>
1c010ecc:	01848613          	addi	a2,s1,24
1c010ed0:	08098593          	addi	a1,s3,128
1c010ed4:	450d                	li	a0,3
1c010ed6:	39e1                	jal	1c010bae <__pi_fll_freq_set>
1c010ed8:	67d1                	lui	a5,0x14
1c010eda:	32178793          	addi	a5,a5,801 # 14321 <__heapfcram_size+0x50e1>
1c010ede:	02fa2823          	sw	a5,48(s4)
1c010ee2:	d080                	sw	s0,32(s1)
1c010ee4:	bfa5                	j	1c010e5c <pi_fll_ioctl+0x60>
1c010ee6:	68c1                	lui	a7,0x10
1c010ee8:	52800693          	li	a3,1320
1c010eec:	4581                	li	a1,0
1c010eee:	1a100337          	lui	t1,0x1a100
1c010ef2:	18fd                	addi	a7,a7,-1
1c010ef4:	004e50fb          	lp.setupi	x1,4,1c010f2c <pi_fll_ioctl+0x130>
1c010ef8:	0086a70b          	p.lw	a4,8(a3!)
1c010efc:	00c72503          	lw	a0,12(a4)
1c010f00:	00432783          	lw	a5,4(t1) # 1a100004 <__l1_heapsram_end+0xa0e0004>
1c010f04:	c3c5a7b3          	p.insert	a5,a1,1,28
1c010f08:	00f32223          	sw	a5,4(t1)
1c010f0c:	00032783          	lw	a5,0(t1)
1c010f10:	d10797b3          	p.extractu	a5,a5,8,16
1c010f14:	01072803          	lw	a6,16(a4)
1c010f18:	0117f7b3          	and	a5,a5,a7
1c010f1c:	d107a833          	p.insert	a6,a5,8,16
1c010f20:	01072823          	sw	a6,16(a4)
1c010f24:	c01537b3          	p.bclr	a5,a0,0,1
1c010f28:	00f72623          	sw	a5,12(a4)
1c010f2c:	0585                	addi	a1,a1,1
1c010f2e:	b73d                	j	1c010e5c <pi_fll_ioctl+0x60>

Disassembly of section .text:

1c010f30 <__gedf2>:
1c010f30:	002027f3          	frrm	a5
1c010f34:	7ff00e93          	li	t4,2047
1c010f38:	d5459733          	p.extractu	a4,a1,10,20
1c010f3c:	e60597b3          	p.extractu	a5,a1,19,0
1c010f40:	e60698b3          	p.extractu	a7,a3,19,0
1c010f44:	d5469833          	p.extractu	a6,a3,10,20
1c010f48:	832a                	mv	t1,a0
1c010f4a:	81fd                	srli	a1,a1,0x1f
1c010f4c:	8e32                	mv	t3,a2
1c010f4e:	82fd                	srli	a3,a3,0x1f
1c010f50:	01d71a63          	bne	a4,t4,1c010f64 <__gedf2+0x34>
1c010f54:	00a7eeb3          	or	t4,a5,a0
1c010f58:	060e8a63          	beqz	t4,1c010fcc <__gedf2+0x9c>
1c010f5c:	00186073          	csrsi	0x1,16
1c010f60:	5579                	li	a0,-2
1c010f62:	8082                	ret
1c010f64:	01d81a63          	bne	a6,t4,1c010f78 <__gedf2+0x48>
1c010f68:	00c8e633          	or	a2,a7,a2
1c010f6c:	fa65                	bnez	a2,1c010f5c <__gedf2+0x2c>
1c010f6e:	ef19                	bnez	a4,1c010f8c <__gedf2+0x5c>
1c010f70:	8d5d                	or	a0,a0,a5
1c010f72:	00153513          	seqz	a0,a0
1c010f76:	a811                	j	1c010f8a <__gedf2+0x5a>
1c010f78:	ef21                	bnez	a4,1c010fd0 <__gedf2+0xa0>
1c010f7a:	8d5d                	or	a0,a0,a5
1c010f7c:	00153513          	seqz	a0,a0
1c010f80:	00081563          	bnez	a6,1c010f8a <__gedf2+0x5a>
1c010f84:	00c8e633          	or	a2,a7,a2
1c010f88:	c221                	beqz	a2,1c010fc8 <__gedf2+0x98>
1c010f8a:	e901                	bnez	a0,1c010f9a <__gedf2+0x6a>
1c010f8c:	00d59463          	bne	a1,a3,1c010f94 <__gedf2+0x64>
1c010f90:	00e85863          	ble	a4,a6,1c010fa0 <__gedf2+0x70>
1c010f94:	4505                	li	a0,1
1c010f96:	e199                	bnez	a1,1c010f9c <__gedf2+0x6c>
1c010f98:	8082                	ret
1c010f9a:	fefd                	bnez	a3,1c010f98 <__gedf2+0x68>
1c010f9c:	557d                	li	a0,-1
1c010f9e:	8082                	ret
1c010fa0:	01075663          	ble	a6,a4,1c010fac <__gedf2+0x7c>
1c010fa4:	557d                	li	a0,-1
1c010fa6:	d9ed                	beqz	a1,1c010f98 <__gedf2+0x68>
1c010fa8:	4505                	li	a0,1
1c010faa:	8082                	ret
1c010fac:	fef8e4e3          	bltu	a7,a5,1c010f94 <__gedf2+0x64>
1c010fb0:	01179863          	bne	a5,a7,1c010fc0 <__gedf2+0x90>
1c010fb4:	fe6e60e3          	bltu	t3,t1,1c010f94 <__gedf2+0x64>
1c010fb8:	4501                	li	a0,0
1c010fba:	fdc37fe3          	bleu	t3,t1,1c010f98 <__gedf2+0x68>
1c010fbe:	b7dd                	j	1c010fa4 <__gedf2+0x74>
1c010fc0:	ff17e2e3          	bltu	a5,a7,1c010fa4 <__gedf2+0x74>
1c010fc4:	4501                	li	a0,0
1c010fc6:	8082                	ret
1c010fc8:	fd75                	bnez	a0,1c010fc4 <__gedf2+0x94>
1c010fca:	b7e9                	j	1c010f94 <__gedf2+0x64>
1c010fcc:	f8e80ee3          	beq	a6,a4,1c010f68 <__gedf2+0x38>
1c010fd0:	4501                	li	a0,0
1c010fd2:	fa0809e3          	beqz	a6,1c010f84 <__gedf2+0x54>
1c010fd6:	bf5d                	j	1c010f8c <__gedf2+0x5c>

1c010fd8 <__ledf2>:
1c010fd8:	002027f3          	frrm	a5
1c010fdc:	7ff00e93          	li	t4,2047
1c010fe0:	d5459733          	p.extractu	a4,a1,10,20
1c010fe4:	e60597b3          	p.extractu	a5,a1,19,0
1c010fe8:	e60698b3          	p.extractu	a7,a3,19,0
1c010fec:	d5469833          	p.extractu	a6,a3,10,20
1c010ff0:	832a                	mv	t1,a0
1c010ff2:	81fd                	srli	a1,a1,0x1f
1c010ff4:	8e32                	mv	t3,a2
1c010ff6:	82fd                	srli	a3,a3,0x1f
1c010ff8:	01d71a63          	bne	a4,t4,1c01100c <__ledf2+0x34>
1c010ffc:	00a7eeb3          	or	t4,a5,a0
1c011000:	060e8a63          	beqz	t4,1c011074 <__ledf2+0x9c>
1c011004:	00186073          	csrsi	0x1,16
1c011008:	4509                	li	a0,2
1c01100a:	8082                	ret
1c01100c:	01d81a63          	bne	a6,t4,1c011020 <__ledf2+0x48>
1c011010:	00c8e633          	or	a2,a7,a2
1c011014:	fa65                	bnez	a2,1c011004 <__ledf2+0x2c>
1c011016:	ef19                	bnez	a4,1c011034 <__ledf2+0x5c>
1c011018:	8d5d                	or	a0,a0,a5
1c01101a:	00153513          	seqz	a0,a0
1c01101e:	a811                	j	1c011032 <__ledf2+0x5a>
1c011020:	ef21                	bnez	a4,1c011078 <__ledf2+0xa0>
1c011022:	8d5d                	or	a0,a0,a5
1c011024:	00153513          	seqz	a0,a0
1c011028:	00081563          	bnez	a6,1c011032 <__ledf2+0x5a>
1c01102c:	00c8e633          	or	a2,a7,a2
1c011030:	c221                	beqz	a2,1c011070 <__ledf2+0x98>
1c011032:	e901                	bnez	a0,1c011042 <__ledf2+0x6a>
1c011034:	00d59463          	bne	a1,a3,1c01103c <__ledf2+0x64>
1c011038:	00e85863          	ble	a4,a6,1c011048 <__ledf2+0x70>
1c01103c:	4505                	li	a0,1
1c01103e:	e199                	bnez	a1,1c011044 <__ledf2+0x6c>
1c011040:	8082                	ret
1c011042:	fefd                	bnez	a3,1c011040 <__ledf2+0x68>
1c011044:	557d                	li	a0,-1
1c011046:	8082                	ret
1c011048:	01075663          	ble	a6,a4,1c011054 <__ledf2+0x7c>
1c01104c:	557d                	li	a0,-1
1c01104e:	d9ed                	beqz	a1,1c011040 <__ledf2+0x68>
1c011050:	4505                	li	a0,1
1c011052:	8082                	ret
1c011054:	fef8e4e3          	bltu	a7,a5,1c01103c <__ledf2+0x64>
1c011058:	01179863          	bne	a5,a7,1c011068 <__ledf2+0x90>
1c01105c:	fe6e60e3          	bltu	t3,t1,1c01103c <__ledf2+0x64>
1c011060:	4501                	li	a0,0
1c011062:	fdc37fe3          	bleu	t3,t1,1c011040 <__ledf2+0x68>
1c011066:	b7dd                	j	1c01104c <__ledf2+0x74>
1c011068:	ff17e2e3          	bltu	a5,a7,1c01104c <__ledf2+0x74>
1c01106c:	4501                	li	a0,0
1c01106e:	8082                	ret
1c011070:	fd75                	bnez	a0,1c01106c <__ledf2+0x94>
1c011072:	b7e9                	j	1c01103c <__ledf2+0x64>
1c011074:	f8e80ee3          	beq	a6,a4,1c011010 <__ledf2+0x38>
1c011078:	4501                	li	a0,0
1c01107a:	fa0809e3          	beqz	a6,1c01102c <__ledf2+0x54>
1c01107e:	bf5d                	j	1c011034 <__ledf2+0x5c>

1c011080 <__extendsfdf2>:
1c011080:	002027f3          	frrm	a5
1c011084:	cf751633          	p.extractu	a2,a0,7,23
1c011088:	00160713          	addi	a4,a2,1
1c01108c:	ee873733          	p.bclr	a4,a4,23,8
1c011090:	4685                	li	a3,1
1c011092:	ec0517b3          	p.extractu	a5,a0,22,0
1c011096:	817d                	srli	a0,a0,0x1f
1c011098:	00e6d963          	ble	a4,a3,1c0110aa <__extendsfdf2+0x2a>
1c01109c:	0037d713          	srli	a4,a5,0x3
1c0110a0:	38060613          	addi	a2,a2,896
1c0110a4:	07f6                	slli	a5,a5,0x1d
1c0110a6:	4681                	li	a3,0
1c0110a8:	a899                	j	1c0110fe <__extendsfdf2+0x7e>
1c0110aa:	ee05                	bnez	a2,1c0110e2 <__extendsfdf2+0x62>
1c0110ac:	c7ad                	beqz	a5,1c011116 <__extendsfdf2+0x96>
1c0110ae:	477d                	li	a4,31
1c0110b0:	100796b3          	p.fl1	a3,a5
1c0110b4:	40d706b3          	sub	a3,a4,a3
1c0110b8:	4729                	li	a4,10
1c0110ba:	00d74e63          	blt	a4,a3,1c0110d6 <__extendsfdf2+0x56>
1c0110be:	472d                	li	a4,11
1c0110c0:	8f15                	sub	a4,a4,a3
1c0110c2:	01568613          	addi	a2,a3,21
1c0110c6:	00e7d733          	srl	a4,a5,a4
1c0110ca:	00c797b3          	sll	a5,a5,a2
1c0110ce:	38900613          	li	a2,905
1c0110d2:	8e15                	sub	a2,a2,a3
1c0110d4:	bfc9                	j	1c0110a6 <__extendsfdf2+0x26>
1c0110d6:	ff568713          	addi	a4,a3,-11
1c0110da:	00e79733          	sll	a4,a5,a4
1c0110de:	4781                	li	a5,0
1c0110e0:	b7fd                	j	1c0110ce <__extendsfdf2+0x4e>
1c0110e2:	cf8d                	beqz	a5,1c01111c <__extendsfdf2+0x9c>
1c0110e4:	004006b7          	lui	a3,0x400
1c0110e8:	8efd                	and	a3,a3,a5
1c0110ea:	0037d713          	srli	a4,a5,0x3
1c0110ee:	0016b693          	seqz	a3,a3
1c0110f2:	0692                	slli	a3,a3,0x4
1c0110f4:	c1374733          	p.bset	a4,a4,0,19
1c0110f8:	07f6                	slli	a5,a5,0x1d
1c0110fa:	7ff00613          	li	a2,2047
1c0110fe:	4581                	li	a1,0
1c011100:	e60725b3          	p.insert	a1,a4,19,0
1c011104:	d54625b3          	p.insert	a1,a2,10,20
1c011108:	c1f525b3          	p.insert	a1,a0,0,31
1c01110c:	853e                	mv	a0,a5
1c01110e:	c299                	beqz	a3,1c011114 <__extendsfdf2+0x94>
1c011110:	0016a073          	csrs	0x1,a3
1c011114:	8082                	ret
1c011116:	4701                	li	a4,0
1c011118:	4601                	li	a2,0
1c01111a:	b771                	j	1c0110a6 <__extendsfdf2+0x26>
1c01111c:	4701                	li	a4,0
1c01111e:	7ff00613          	li	a2,2047
1c011122:	b751                	j	1c0110a6 <__extendsfdf2+0x26>

1c011124 <fc_soc_event_handler>:
1c011124:	1a1097b7          	lui	a5,0x1a109
1c011128:	53c8                	lw	a0,36(a5)
1c01112a:	03800793          	li	a5,56
1c01112e:	00251713          	slli	a4,a0,0x2
1c011132:	97ba                	add	a5,a5,a4
1c011134:	4398                	lw	a4,0(a5)
1c011136:	c711                	beqz	a4,1c011142 <fc_soc_event_handler+0x1e>
1c011138:	0007a303          	lw	t1,0(a5) # 1a109000 <__l1_heapsram_end+0xa0e9000>
1c01113c:	24c7a583          	lw	a1,588(a5)
1c011140:	8302                	jr	t1
1c011142:	8082                	ret

1c011144 <xPortStartScheduler>:
1c011144:	58f080ef          	jal	ra,1c019ed2 <prvSetupTimerInterrupt>
1c011148:	0000c297          	auipc	t0,0xc
1c01114c:	66028293          	addi	t0,t0,1632 # 1c01d7a8 <ulCriticalNesting>
1c011150:	0002a023          	sw	zero,0(t0)
1c011154:	e3fef217          	auipc	tp,0xe3fef
1c011158:	38c22203          	lw	tp,908(tp) # 4e0 <pxCurrentTCB>
1c01115c:	00022103          	lw	sp,0(tp) # 0 <__pi_cluster_pool-0x4>
1c011160:	52d6                	lw	t0,116(sp)
1c011162:	34129073          	csrw	mepc,t0
1c011166:	52c6                	lw	t0,112(sp)
1c011168:	30029073          	csrw	mstatus,t0
1c01116c:	5db6                	lw	s11,108(sp)
1c01116e:	5d26                	lw	s10,104(sp)
1c011170:	5c96                	lw	s9,100(sp)
1c011172:	5c06                	lw	s8,96(sp)
1c011174:	4bf6                	lw	s7,92(sp)
1c011176:	4b66                	lw	s6,88(sp)
1c011178:	4ad6                	lw	s5,84(sp)
1c01117a:	4a46                	lw	s4,80(sp)
1c01117c:	49b6                	lw	s3,76(sp)
1c01117e:	4926                	lw	s2,72(sp)
1c011180:	4496                	lw	s1,68(sp)
1c011182:	4406                	lw	s0,64(sp)
1c011184:	5ff2                	lw	t6,60(sp)
1c011186:	5f62                	lw	t5,56(sp)
1c011188:	5ed2                	lw	t4,52(sp)
1c01118a:	5e42                	lw	t3,48(sp)
1c01118c:	53b2                	lw	t2,44(sp)
1c01118e:	5322                	lw	t1,40(sp)
1c011190:	5292                	lw	t0,36(sp)
1c011192:	5882                	lw	a7,32(sp)
1c011194:	4872                	lw	a6,28(sp)
1c011196:	47e2                	lw	a5,24(sp)
1c011198:	4752                	lw	a4,20(sp)
1c01119a:	46c2                	lw	a3,16(sp)
1c01119c:	4632                	lw	a2,12(sp)
1c01119e:	45a2                	lw	a1,8(sp)
1c0111a0:	4512                	lw	a0,4(sp)
1c0111a2:	4082                	lw	ra,0(sp)
1c0111a4:	07810113          	addi	sp,sp,120
1c0111a8:	43d2                	lw	t2,20(sp)
1c0111aa:	7c639073          	csrw	0x7c6,t2
1c0111ae:	4342                	lw	t1,16(sp)
1c0111b0:	7c531073          	csrw	0x7c5,t1
1c0111b4:	42b2                	lw	t0,12(sp)
1c0111b6:	7c429073          	csrw	0x7c4,t0
1c0111ba:	43a2                	lw	t2,8(sp)
1c0111bc:	7c239073          	csrw	priv_mepc,t2
1c0111c0:	4312                	lw	t1,4(sp)
1c0111c2:	7c131073          	csrw	priv_emstatus,t1
1c0111c6:	4282                	lw	t0,0(sp)
1c0111c8:	7c029073          	csrw	0x7c0,t0
1c0111cc:	0161                	addi	sp,sp,24
1c0111ce:	30200073          	mret

1c0111d2 <__main>:
1c0111d2:	00004297          	auipc	t0,0x4
1c0111d6:	f8c28293          	addi	t0,t0,-116 # 1c01515e <deallocate_init_stack>
1c0111da:	000280e7          	jalr	t0
1c0111de:	4501                	li	a0,0
1c0111e0:	4581                	li	a1,0
1c0111e2:	00003617          	auipc	a2,0x3
1c0111e6:	afa60613          	addi	a2,a2,-1286 # 1c013cdc <main>
1c0111ea:	000600e7          	jalr	a2
1c0111ee:	00009617          	auipc	a2,0x9
1c0111f2:	dc660613          	addi	a2,a2,-570 # 1c019fb4 <exit>
1c0111f6:	000600e7          	jalr	a2

1c0111fa <__divdi3>:
1c0111fa:	4881                	li	a7,0
1c0111fc:	0005db63          	bgez	a1,1c011212 <__divdi3+0x18>
1c011200:	40a00533          	neg	a0,a0
1c011204:	00a038b3          	snez	a7,a0
1c011208:	40b00733          	neg	a4,a1
1c01120c:	411705b3          	sub	a1,a4,a7
1c011210:	58fd                	li	a7,-1
1c011212:	0006db63          	bgez	a3,1c011228 <__divdi3+0x2e>
1c011216:	40c00633          	neg	a2,a2
1c01121a:	00c037b3          	snez	a5,a2
1c01121e:	40d006b3          	neg	a3,a3
1c011222:	fff8c893          	not	a7,a7
1c011226:	8e9d                	sub	a3,a3,a5
1c011228:	8332                	mv	t1,a2
1c01122a:	8736                	mv	a4,a3
1c01122c:	8eaa                	mv	t4,a0
1c01122e:	8e2e                	mv	t3,a1
1c011230:	20069163          	bnez	a3,1c011432 <__divdi3+0x238>
1c011234:	1c01c6b7          	lui	a3,0x1c01c
1c011238:	a4868693          	addi	a3,a3,-1464 # 1c01ba48 <__clz_tab>
1c01123c:	0ac5fa63          	bleu	a2,a1,1c0112f0 <__divdi3+0xf6>
1c011240:	67c1                	lui	a5,0x10
1c011242:	0af67063          	bleu	a5,a2,1c0112e2 <__divdi3+0xe8>
1c011246:	0ff00f13          	li	t5,255
1c01124a:	00cf3f33          	sltu	t5,t5,a2
1c01124e:	0f0e                	slli	t5,t5,0x3
1c011250:	01e657b3          	srl	a5,a2,t5
1c011254:	96be                	add	a3,a3,a5
1c011256:	0006c703          	lbu	a4,0(a3)
1c01125a:	02000793          	li	a5,32
1c01125e:	9f3a                	add	t5,t5,a4
1c011260:	41e787b3          	sub	a5,a5,t5
1c011264:	cb99                	beqz	a5,1c01127a <__divdi3+0x80>
1c011266:	00f59733          	sll	a4,a1,a5
1c01126a:	01e55f33          	srl	t5,a0,t5
1c01126e:	00f61333          	sll	t1,a2,a5
1c011272:	00ef6e33          	or	t3,t5,a4
1c011276:	00f51eb3          	sll	t4,a0,a5
1c01127a:	01035593          	srli	a1,t1,0x10
1c01127e:	02be5733          	divu	a4,t3,a1
1c011282:	10035633          	p.exthz	a2,t1
1c011286:	010ed693          	srli	a3,t4,0x10
1c01128a:	02be77b3          	remu	a5,t3,a1
1c01128e:	02e60533          	mul	a0,a2,a4
1c011292:	07c2                	slli	a5,a5,0x10
1c011294:	8edd                	or	a3,a3,a5
1c011296:	87ba                	mv	a5,a4
1c011298:	00a6fc63          	bleu	a0,a3,1c0112b0 <__divdi3+0xb6>
1c01129c:	969a                	add	a3,a3,t1
1c01129e:	fff70793          	addi	a5,a4,-1
1c0112a2:	0066e763          	bltu	a3,t1,1c0112b0 <__divdi3+0xb6>
1c0112a6:	00a6f563          	bleu	a0,a3,1c0112b0 <__divdi3+0xb6>
1c0112aa:	ffe70793          	addi	a5,a4,-2
1c0112ae:	969a                	add	a3,a3,t1
1c0112b0:	8e89                	sub	a3,a3,a0
1c0112b2:	02b6f533          	remu	a0,a3,a1
1c0112b6:	02b6d6b3          	divu	a3,a3,a1
1c0112ba:	df052eb3          	p.insert	t4,a0,15,16
1c0112be:	02d60733          	mul	a4,a2,a3
1c0112c2:	8536                	mv	a0,a3
1c0112c4:	00eefb63          	bleu	a4,t4,1c0112da <__divdi3+0xe0>
1c0112c8:	9e9a                	add	t4,t4,t1
1c0112ca:	fff68513          	addi	a0,a3,-1
1c0112ce:	006ee663          	bltu	t4,t1,1c0112da <__divdi3+0xe0>
1c0112d2:	00eef463          	bleu	a4,t4,1c0112da <__divdi3+0xe0>
1c0112d6:	ffe68513          	addi	a0,a3,-2
1c0112da:	07c2                	slli	a5,a5,0x10
1c0112dc:	8fc9                	or	a5,a5,a0
1c0112de:	4e01                	li	t3,0
1c0112e0:	a05d                	j	1c011386 <__divdi3+0x18c>
1c0112e2:	010007b7          	lui	a5,0x1000
1c0112e6:	4f41                	li	t5,16
1c0112e8:	f6f664e3          	bltu	a2,a5,1c011250 <__divdi3+0x56>
1c0112ec:	4f61                	li	t5,24
1c0112ee:	b78d                	j	1c011250 <__divdi3+0x56>
1c0112f0:	e601                	bnez	a2,1c0112f8 <__divdi3+0xfe>
1c0112f2:	4305                	li	t1,1
1c0112f4:	02e35333          	divu	t1,t1,a4
1c0112f8:	67c1                	lui	a5,0x10
1c0112fa:	0af37263          	bleu	a5,t1,1c01139e <__divdi3+0x1a4>
1c0112fe:	0ff00793          	li	a5,255
1c011302:	0067f363          	bleu	t1,a5,1c011308 <__divdi3+0x10e>
1c011306:	4721                	li	a4,8
1c011308:	00e357b3          	srl	a5,t1,a4
1c01130c:	96be                	add	a3,a3,a5
1c01130e:	0006c783          	lbu	a5,0(a3)
1c011312:	02000693          	li	a3,32
1c011316:	97ba                	add	a5,a5,a4
1c011318:	8e9d                	sub	a3,a3,a5
1c01131a:	eac9                	bnez	a3,1c0113ac <__divdi3+0x1b2>
1c01131c:	40658733          	sub	a4,a1,t1
1c011320:	4e05                	li	t3,1
1c011322:	01035593          	srli	a1,t1,0x10
1c011326:	10035633          	p.exthz	a2,t1
1c01132a:	010ed693          	srli	a3,t4,0x10
1c01132e:	02b777b3          	remu	a5,a4,a1
1c011332:	02b75733          	divu	a4,a4,a1
1c011336:	07c2                	slli	a5,a5,0x10
1c011338:	8edd                	or	a3,a3,a5
1c01133a:	02e60533          	mul	a0,a2,a4
1c01133e:	87ba                	mv	a5,a4
1c011340:	00a6fc63          	bleu	a0,a3,1c011358 <__divdi3+0x15e>
1c011344:	969a                	add	a3,a3,t1
1c011346:	fff70793          	addi	a5,a4,-1
1c01134a:	0066e763          	bltu	a3,t1,1c011358 <__divdi3+0x15e>
1c01134e:	00a6f563          	bleu	a0,a3,1c011358 <__divdi3+0x15e>
1c011352:	ffe70793          	addi	a5,a4,-2
1c011356:	969a                	add	a3,a3,t1
1c011358:	8e89                	sub	a3,a3,a0
1c01135a:	02b6f533          	remu	a0,a3,a1
1c01135e:	02b6d6b3          	divu	a3,a3,a1
1c011362:	df052eb3          	p.insert	t4,a0,15,16
1c011366:	02d60733          	mul	a4,a2,a3
1c01136a:	8536                	mv	a0,a3
1c01136c:	00eefb63          	bleu	a4,t4,1c011382 <__divdi3+0x188>
1c011370:	9e9a                	add	t4,t4,t1
1c011372:	fff68513          	addi	a0,a3,-1
1c011376:	006ee663          	bltu	t4,t1,1c011382 <__divdi3+0x188>
1c01137a:	00eef463          	bleu	a4,t4,1c011382 <__divdi3+0x188>
1c01137e:	ffe68513          	addi	a0,a3,-2
1c011382:	07c2                	slli	a5,a5,0x10
1c011384:	8fc9                	or	a5,a5,a0
1c011386:	853e                	mv	a0,a5
1c011388:	85f2                	mv	a1,t3
1c01138a:	00088963          	beqz	a7,1c01139c <__divdi3+0x1a2>
1c01138e:	40f00533          	neg	a0,a5
1c011392:	00a037b3          	snez	a5,a0
1c011396:	41c005b3          	neg	a1,t3
1c01139a:	8d9d                	sub	a1,a1,a5
1c01139c:	8082                	ret
1c01139e:	010007b7          	lui	a5,0x1000
1c0113a2:	4741                	li	a4,16
1c0113a4:	f6f362e3          	bltu	t1,a5,1c011308 <__divdi3+0x10e>
1c0113a8:	4761                	li	a4,24
1c0113aa:	bfb9                	j	1c011308 <__divdi3+0x10e>
1c0113ac:	00d31333          	sll	t1,t1,a3
1c0113b0:	00f5d633          	srl	a2,a1,a5
1c0113b4:	00d59733          	sll	a4,a1,a3
1c0113b8:	00d51eb3          	sll	t4,a0,a3
1c0113bc:	00f555b3          	srl	a1,a0,a5
1c0113c0:	01035513          	srli	a0,t1,0x10
1c0113c4:	02a676b3          	remu	a3,a2,a0
1c0113c8:	8dd9                	or	a1,a1,a4
1c0113ca:	10035733          	p.exthz	a4,t1
1c0113ce:	0105d793          	srli	a5,a1,0x10
1c0113d2:	02a65633          	divu	a2,a2,a0
1c0113d6:	06c2                	slli	a3,a3,0x10
1c0113d8:	8fd5                	or	a5,a5,a3
1c0113da:	02c70833          	mul	a6,a4,a2
1c0113de:	8e32                	mv	t3,a2
1c0113e0:	0107fc63          	bleu	a6,a5,1c0113f8 <__divdi3+0x1fe>
1c0113e4:	979a                	add	a5,a5,t1
1c0113e6:	fff60e13          	addi	t3,a2,-1
1c0113ea:	0067e763          	bltu	a5,t1,1c0113f8 <__divdi3+0x1fe>
1c0113ee:	0107f563          	bleu	a6,a5,1c0113f8 <__divdi3+0x1fe>
1c0113f2:	ffe60e13          	addi	t3,a2,-2
1c0113f6:	979a                	add	a5,a5,t1
1c0113f8:	410787b3          	sub	a5,a5,a6
1c0113fc:	02a7f6b3          	remu	a3,a5,a0
1c011400:	02a7d7b3          	divu	a5,a5,a0
1c011404:	df06a5b3          	p.insert	a1,a3,15,16
1c011408:	02f70733          	mul	a4,a4,a5
1c01140c:	86be                	mv	a3,a5
1c01140e:	00e5fc63          	bleu	a4,a1,1c011426 <__divdi3+0x22c>
1c011412:	959a                	add	a1,a1,t1
1c011414:	fff78693          	addi	a3,a5,-1 # ffffff <__l1_heapsram_size+0xfe006f>
1c011418:	0065e763          	bltu	a1,t1,1c011426 <__divdi3+0x22c>
1c01141c:	00e5f563          	bleu	a4,a1,1c011426 <__divdi3+0x22c>
1c011420:	ffe78693          	addi	a3,a5,-2
1c011424:	959a                	add	a1,a1,t1
1c011426:	0e42                	slli	t3,t3,0x10
1c011428:	40e58733          	sub	a4,a1,a4
1c01142c:	00de6e33          	or	t3,t3,a3
1c011430:	bdcd                	j	1c011322 <__divdi3+0x128>
1c011432:	12d5ea63          	bltu	a1,a3,1c011566 <__divdi3+0x36c>
1c011436:	67c1                	lui	a5,0x10
1c011438:	02f6fd63          	bleu	a5,a3,1c011472 <__divdi3+0x278>
1c01143c:	0ff00813          	li	a6,255
1c011440:	00d83333          	sltu	t1,a6,a3
1c011444:	030e                	slli	t1,t1,0x3
1c011446:	1c01c737          	lui	a4,0x1c01c
1c01144a:	0066d7b3          	srl	a5,a3,t1
1c01144e:	a4870713          	addi	a4,a4,-1464 # 1c01ba48 <__clz_tab>
1c011452:	97ba                	add	a5,a5,a4
1c011454:	0007c803          	lbu	a6,0(a5) # 10000 <__heapfcram_size+0xdc0>
1c011458:	02000e13          	li	t3,32
1c01145c:	981a                	add	a6,a6,t1
1c01145e:	410e0e33          	sub	t3,t3,a6
1c011462:	000e1f63          	bnez	t3,1c011480 <__divdi3+0x286>
1c011466:	4785                	li	a5,1
1c011468:	f0b6efe3          	bltu	a3,a1,1c011386 <__divdi3+0x18c>
1c01146c:	04a637b3          	p.sletu	a5,a2,a0
1c011470:	bf19                	j	1c011386 <__divdi3+0x18c>
1c011472:	010007b7          	lui	a5,0x1000
1c011476:	4341                	li	t1,16
1c011478:	fcf6e7e3          	bltu	a3,a5,1c011446 <__divdi3+0x24c>
1c01147c:	4361                	li	t1,24
1c01147e:	b7e1                	j	1c011446 <__divdi3+0x24c>
1c011480:	01065333          	srl	t1,a2,a6
1c011484:	01c696b3          	sll	a3,a3,t3
1c011488:	00d366b3          	or	a3,t1,a3
1c01148c:	0106de93          	srli	t4,a3,0x10
1c011490:	0105d333          	srl	t1,a1,a6
1c011494:	03d377b3          	remu	a5,t1,t4
1c011498:	01c595b3          	sll	a1,a1,t3
1c01149c:	01055733          	srl	a4,a0,a6
1c0114a0:	8f4d                	or	a4,a4,a1
1c0114a2:	1006d5b3          	p.exthz	a1,a3
1c0114a6:	01075813          	srli	a6,a4,0x10
1c0114aa:	01c61633          	sll	a2,a2,t3
1c0114ae:	03d35333          	divu	t1,t1,t4
1c0114b2:	07c2                	slli	a5,a5,0x10
1c0114b4:	0107e833          	or	a6,a5,a6
1c0114b8:	02658f33          	mul	t5,a1,t1
1c0114bc:	879a                	mv	a5,t1
1c0114be:	01e87c63          	bleu	t5,a6,1c0114d6 <__divdi3+0x2dc>
1c0114c2:	9836                	add	a6,a6,a3
1c0114c4:	fff30793          	addi	a5,t1,-1
1c0114c8:	00d86763          	bltu	a6,a3,1c0114d6 <__divdi3+0x2dc>
1c0114cc:	01e87563          	bleu	t5,a6,1c0114d6 <__divdi3+0x2dc>
1c0114d0:	ffe30793          	addi	a5,t1,-2
1c0114d4:	9836                	add	a6,a6,a3
1c0114d6:	41e80833          	sub	a6,a6,t5
1c0114da:	03d87333          	remu	t1,a6,t4
1c0114de:	03d85833          	divu	a6,a6,t4
1c0114e2:	df032733          	p.insert	a4,t1,15,16
1c0114e6:	030585b3          	mul	a1,a1,a6
1c0114ea:	8342                	mv	t1,a6
1c0114ec:	00b77c63          	bleu	a1,a4,1c011504 <__divdi3+0x30a>
1c0114f0:	9736                	add	a4,a4,a3
1c0114f2:	fff80313          	addi	t1,a6,-1
1c0114f6:	00d76763          	bltu	a4,a3,1c011504 <__divdi3+0x30a>
1c0114fa:	00b77563          	bleu	a1,a4,1c011504 <__divdi3+0x30a>
1c0114fe:	ffe80313          	addi	t1,a6,-2
1c011502:	9736                	add	a4,a4,a3
1c011504:	07c2                	slli	a5,a5,0x10
1c011506:	6ec1                	lui	t4,0x10
1c011508:	0067e7b3          	or	a5,a5,t1
1c01150c:	8f0d                	sub	a4,a4,a1
1c01150e:	fffe8593          	addi	a1,t4,-1 # ffff <__heapfcram_size+0xdbf>
1c011512:	00b7f6b3          	and	a3,a5,a1
1c011516:	0107d813          	srli	a6,a5,0x10
1c01151a:	8df1                	and	a1,a1,a2
1c01151c:	02b68333          	mul	t1,a3,a1
1c011520:	8241                	srli	a2,a2,0x10
1c011522:	02b805b3          	mul	a1,a6,a1
1c011526:	8f2e                	mv	t5,a1
1c011528:	42c68f33          	p.mac	t5,a3,a2
1c01152c:	01035693          	srli	a3,t1,0x10
1c011530:	96fa                	add	a3,a3,t5
1c011532:	02c80833          	mul	a6,a6,a2
1c011536:	00b6f363          	bleu	a1,a3,1c01153c <__divdi3+0x342>
1c01153a:	9876                	add	a6,a6,t4
1c01153c:	0106d613          	srli	a2,a3,0x10
1c011540:	9832                	add	a6,a6,a2
1c011542:	03076063          	bltu	a4,a6,1c011562 <__divdi3+0x368>
1c011546:	d9071ce3          	bne	a4,a6,1c0112de <__divdi3+0xe4>
1c01154a:	6741                	lui	a4,0x10
1c01154c:	177d                	addi	a4,a4,-1
1c01154e:	8ef9                	and	a3,a3,a4
1c011550:	06c2                	slli	a3,a3,0x10
1c011552:	00e37333          	and	t1,t1,a4
1c011556:	01c51533          	sll	a0,a0,t3
1c01155a:	969a                	add	a3,a3,t1
1c01155c:	4e01                	li	t3,0
1c01155e:	e2d574e3          	bleu	a3,a0,1c011386 <__divdi3+0x18c>
1c011562:	17fd                	addi	a5,a5,-1
1c011564:	bbad                	j	1c0112de <__divdi3+0xe4>
1c011566:	4e01                	li	t3,0
1c011568:	4781                	li	a5,0
1c01156a:	bd31                	j	1c011386 <__divdi3+0x18c>

1c01156c <__udivdi3>:
1c01156c:	87b2                	mv	a5,a2
1c01156e:	8736                	mv	a4,a3
1c011570:	88aa                	mv	a7,a0
1c011572:	882e                	mv	a6,a1
1c011574:	1e069d63          	bnez	a3,1c01176e <__udivdi3+0x202>
1c011578:	1c01c337          	lui	t1,0x1c01c
1c01157c:	a4830313          	addi	t1,t1,-1464 # 1c01ba48 <__clz_tab>
1c011580:	0ac5fd63          	bleu	a2,a1,1c01163a <__udivdi3+0xce>
1c011584:	6741                	lui	a4,0x10
1c011586:	0ae67363          	bleu	a4,a2,1c01162c <__udivdi3+0xc0>
1c01158a:	0ff00693          	li	a3,255
1c01158e:	00c6b6b3          	sltu	a3,a3,a2
1c011592:	068e                	slli	a3,a3,0x3
1c011594:	00d65733          	srl	a4,a2,a3
1c011598:	933a                	add	t1,t1,a4
1c01159a:	00034703          	lbu	a4,0(t1)
1c01159e:	02000313          	li	t1,32
1c0115a2:	96ba                	add	a3,a3,a4
1c0115a4:	40d30333          	sub	t1,t1,a3
1c0115a8:	00030c63          	beqz	t1,1c0115c0 <__udivdi3+0x54>
1c0115ac:	00659733          	sll	a4,a1,t1
1c0115b0:	00d556b3          	srl	a3,a0,a3
1c0115b4:	006617b3          	sll	a5,a2,t1
1c0115b8:	00e6e833          	or	a6,a3,a4
1c0115bc:	006518b3          	sll	a7,a0,t1
1c0115c0:	0107d513          	srli	a0,a5,0x10
1c0115c4:	02a85733          	divu	a4,a6,a0
1c0115c8:	1007d5b3          	p.exthz	a1,a5
1c0115cc:	0108d693          	srli	a3,a7,0x10
1c0115d0:	02a87633          	remu	a2,a6,a0
1c0115d4:	02e58833          	mul	a6,a1,a4
1c0115d8:	0642                	slli	a2,a2,0x10
1c0115da:	8ed1                	or	a3,a3,a2
1c0115dc:	863a                	mv	a2,a4
1c0115de:	0106fc63          	bleu	a6,a3,1c0115f6 <__udivdi3+0x8a>
1c0115e2:	96be                	add	a3,a3,a5
1c0115e4:	fff70613          	addi	a2,a4,-1 # ffff <__heapfcram_size+0xdbf>
1c0115e8:	00f6e763          	bltu	a3,a5,1c0115f6 <__udivdi3+0x8a>
1c0115ec:	0106f563          	bleu	a6,a3,1c0115f6 <__udivdi3+0x8a>
1c0115f0:	ffe70613          	addi	a2,a4,-2
1c0115f4:	96be                	add	a3,a3,a5
1c0115f6:	410686b3          	sub	a3,a3,a6
1c0115fa:	02a6f833          	remu	a6,a3,a0
1c0115fe:	02a6d6b3          	divu	a3,a3,a0
1c011602:	df0828b3          	p.insert	a7,a6,15,16
1c011606:	02d58733          	mul	a4,a1,a3
1c01160a:	8536                	mv	a0,a3
1c01160c:	00e8fb63          	bleu	a4,a7,1c011622 <__udivdi3+0xb6>
1c011610:	98be                	add	a7,a7,a5
1c011612:	fff68513          	addi	a0,a3,-1
1c011616:	00f8e663          	bltu	a7,a5,1c011622 <__udivdi3+0xb6>
1c01161a:	00e8f463          	bleu	a4,a7,1c011622 <__udivdi3+0xb6>
1c01161e:	ffe68513          	addi	a0,a3,-2
1c011622:	01061793          	slli	a5,a2,0x10
1c011626:	8fc9                	or	a5,a5,a0
1c011628:	4801                	li	a6,0
1c01162a:	a06d                	j	1c0116d4 <__udivdi3+0x168>
1c01162c:	01000737          	lui	a4,0x1000
1c011630:	46c1                	li	a3,16
1c011632:	f6e661e3          	bltu	a2,a4,1c011594 <__udivdi3+0x28>
1c011636:	46e1                	li	a3,24
1c011638:	bfb1                	j	1c011594 <__udivdi3+0x28>
1c01163a:	e601                	bnez	a2,1c011642 <__udivdi3+0xd6>
1c01163c:	4685                	li	a3,1
1c01163e:	02c6d7b3          	divu	a5,a3,a2
1c011642:	66c1                	lui	a3,0x10
1c011644:	08d7fb63          	bleu	a3,a5,1c0116da <__udivdi3+0x16e>
1c011648:	0ff00693          	li	a3,255
1c01164c:	00f6f363          	bleu	a5,a3,1c011652 <__udivdi3+0xe6>
1c011650:	4721                	li	a4,8
1c011652:	00e7d6b3          	srl	a3,a5,a4
1c011656:	9336                	add	t1,t1,a3
1c011658:	00034683          	lbu	a3,0(t1)
1c01165c:	02000613          	li	a2,32
1c011660:	96ba                	add	a3,a3,a4
1c011662:	8e15                	sub	a2,a2,a3
1c011664:	e251                	bnez	a2,1c0116e8 <__udivdi3+0x17c>
1c011666:	40f58733          	sub	a4,a1,a5
1c01166a:	4805                	li	a6,1
1c01166c:	0107d513          	srli	a0,a5,0x10
1c011670:	1007d5b3          	p.exthz	a1,a5
1c011674:	0108d693          	srli	a3,a7,0x10
1c011678:	02a77633          	remu	a2,a4,a0
1c01167c:	02a75733          	divu	a4,a4,a0
1c011680:	0642                	slli	a2,a2,0x10
1c011682:	8ed1                	or	a3,a3,a2
1c011684:	02e58333          	mul	t1,a1,a4
1c011688:	863a                	mv	a2,a4
1c01168a:	0066fc63          	bleu	t1,a3,1c0116a2 <__udivdi3+0x136>
1c01168e:	96be                	add	a3,a3,a5
1c011690:	fff70613          	addi	a2,a4,-1 # ffffff <__l1_heapsram_size+0xfe006f>
1c011694:	00f6e763          	bltu	a3,a5,1c0116a2 <__udivdi3+0x136>
1c011698:	0066f563          	bleu	t1,a3,1c0116a2 <__udivdi3+0x136>
1c01169c:	ffe70613          	addi	a2,a4,-2
1c0116a0:	96be                	add	a3,a3,a5
1c0116a2:	406686b3          	sub	a3,a3,t1
1c0116a6:	02a6f333          	remu	t1,a3,a0
1c0116aa:	02a6d6b3          	divu	a3,a3,a0
1c0116ae:	df0328b3          	p.insert	a7,t1,15,16
1c0116b2:	02d58733          	mul	a4,a1,a3
1c0116b6:	8536                	mv	a0,a3
1c0116b8:	00e8fb63          	bleu	a4,a7,1c0116ce <__udivdi3+0x162>
1c0116bc:	98be                	add	a7,a7,a5
1c0116be:	fff68513          	addi	a0,a3,-1 # ffff <__heapfcram_size+0xdbf>
1c0116c2:	00f8e663          	bltu	a7,a5,1c0116ce <__udivdi3+0x162>
1c0116c6:	00e8f463          	bleu	a4,a7,1c0116ce <__udivdi3+0x162>
1c0116ca:	ffe68513          	addi	a0,a3,-2
1c0116ce:	01061793          	slli	a5,a2,0x10
1c0116d2:	8fc9                	or	a5,a5,a0
1c0116d4:	853e                	mv	a0,a5
1c0116d6:	85c2                	mv	a1,a6
1c0116d8:	8082                	ret
1c0116da:	010006b7          	lui	a3,0x1000
1c0116de:	4741                	li	a4,16
1c0116e0:	f6d7e9e3          	bltu	a5,a3,1c011652 <__udivdi3+0xe6>
1c0116e4:	4761                	li	a4,24
1c0116e6:	b7b5                	j	1c011652 <__udivdi3+0xe6>
1c0116e8:	00c797b3          	sll	a5,a5,a2
1c0116ec:	00d5d333          	srl	t1,a1,a3
1c0116f0:	0107de13          	srli	t3,a5,0x10
1c0116f4:	00c59733          	sll	a4,a1,a2
1c0116f8:	00c518b3          	sll	a7,a0,a2
1c0116fc:	00d555b3          	srl	a1,a0,a3
1c011700:	03c35533          	divu	a0,t1,t3
1c011704:	8dd9                	or	a1,a1,a4
1c011706:	1007d733          	p.exthz	a4,a5
1c01170a:	0105d693          	srli	a3,a1,0x10
1c01170e:	03c37633          	remu	a2,t1,t3
1c011712:	882a                	mv	a6,a0
1c011714:	02a70333          	mul	t1,a4,a0
1c011718:	0642                	slli	a2,a2,0x10
1c01171a:	8ed1                	or	a3,a3,a2
1c01171c:	0066fc63          	bleu	t1,a3,1c011734 <__udivdi3+0x1c8>
1c011720:	96be                	add	a3,a3,a5
1c011722:	fff50813          	addi	a6,a0,-1
1c011726:	00f6e763          	bltu	a3,a5,1c011734 <__udivdi3+0x1c8>
1c01172a:	0066f563          	bleu	t1,a3,1c011734 <__udivdi3+0x1c8>
1c01172e:	ffe50813          	addi	a6,a0,-2
1c011732:	96be                	add	a3,a3,a5
1c011734:	406686b3          	sub	a3,a3,t1
1c011738:	03c6f633          	remu	a2,a3,t3
1c01173c:	03c6d6b3          	divu	a3,a3,t3
1c011740:	df0625b3          	p.insert	a1,a2,15,16
1c011744:	02d70733          	mul	a4,a4,a3
1c011748:	8636                	mv	a2,a3
1c01174a:	00e5fc63          	bleu	a4,a1,1c011762 <__udivdi3+0x1f6>
1c01174e:	95be                	add	a1,a1,a5
1c011750:	fff68613          	addi	a2,a3,-1 # ffffff <__l1_heapsram_size+0xfe006f>
1c011754:	00f5e763          	bltu	a1,a5,1c011762 <__udivdi3+0x1f6>
1c011758:	00e5f563          	bleu	a4,a1,1c011762 <__udivdi3+0x1f6>
1c01175c:	ffe68613          	addi	a2,a3,-2
1c011760:	95be                	add	a1,a1,a5
1c011762:	0842                	slli	a6,a6,0x10
1c011764:	40e58733          	sub	a4,a1,a4
1c011768:	00c86833          	or	a6,a6,a2
1c01176c:	b701                	j	1c01166c <__udivdi3+0x100>
1c01176e:	12d5ea63          	bltu	a1,a3,1c0118a2 <__udivdi3+0x336>
1c011772:	67c1                	lui	a5,0x10
1c011774:	02f6fd63          	bleu	a5,a3,1c0117ae <__udivdi3+0x242>
1c011778:	0ff00793          	li	a5,255
1c01177c:	00d7b8b3          	sltu	a7,a5,a3
1c011780:	088e                	slli	a7,a7,0x3
1c011782:	1c01c737          	lui	a4,0x1c01c
1c011786:	0116d7b3          	srl	a5,a3,a7
1c01178a:	a4870713          	addi	a4,a4,-1464 # 1c01ba48 <__clz_tab>
1c01178e:	97ba                	add	a5,a5,a4
1c011790:	0007c783          	lbu	a5,0(a5) # 10000 <__heapfcram_size+0xdc0>
1c011794:	02000813          	li	a6,32
1c011798:	97c6                	add	a5,a5,a7
1c01179a:	40f80833          	sub	a6,a6,a5
1c01179e:	00081f63          	bnez	a6,1c0117bc <__udivdi3+0x250>
1c0117a2:	4785                	li	a5,1
1c0117a4:	f2b6e8e3          	bltu	a3,a1,1c0116d4 <__udivdi3+0x168>
1c0117a8:	04a637b3          	p.sletu	a5,a2,a0
1c0117ac:	b725                	j	1c0116d4 <__udivdi3+0x168>
1c0117ae:	010007b7          	lui	a5,0x1000
1c0117b2:	48c1                	li	a7,16
1c0117b4:	fcf6e7e3          	bltu	a3,a5,1c011782 <__udivdi3+0x216>
1c0117b8:	48e1                	li	a7,24
1c0117ba:	b7e1                	j	1c011782 <__udivdi3+0x216>
1c0117bc:	00f658b3          	srl	a7,a2,a5
1c0117c0:	010696b3          	sll	a3,a3,a6
1c0117c4:	00d8e6b3          	or	a3,a7,a3
1c0117c8:	00f5d333          	srl	t1,a1,a5
1c0117cc:	0106de13          	srli	t3,a3,0x10
1c0117d0:	00f55733          	srl	a4,a0,a5
1c0117d4:	03c377b3          	remu	a5,t1,t3
1c0117d8:	1006d8b3          	p.exthz	a7,a3
1c0117dc:	010595b3          	sll	a1,a1,a6
1c0117e0:	8f4d                	or	a4,a4,a1
1c0117e2:	01075593          	srli	a1,a4,0x10
1c0117e6:	01061633          	sll	a2,a2,a6
1c0117ea:	03c35333          	divu	t1,t1,t3
1c0117ee:	07c2                	slli	a5,a5,0x10
1c0117f0:	8ddd                	or	a1,a1,a5
1c0117f2:	02688eb3          	mul	t4,a7,t1
1c0117f6:	879a                	mv	a5,t1
1c0117f8:	01d5fc63          	bleu	t4,a1,1c011810 <__udivdi3+0x2a4>
1c0117fc:	95b6                	add	a1,a1,a3
1c0117fe:	fff30793          	addi	a5,t1,-1
1c011802:	00d5e763          	bltu	a1,a3,1c011810 <__udivdi3+0x2a4>
1c011806:	01d5f563          	bleu	t4,a1,1c011810 <__udivdi3+0x2a4>
1c01180a:	ffe30793          	addi	a5,t1,-2
1c01180e:	95b6                	add	a1,a1,a3
1c011810:	41d585b3          	sub	a1,a1,t4
1c011814:	03c5f333          	remu	t1,a1,t3
1c011818:	03c5d5b3          	divu	a1,a1,t3
1c01181c:	df032733          	p.insert	a4,t1,15,16
1c011820:	02b888b3          	mul	a7,a7,a1
1c011824:	832e                	mv	t1,a1
1c011826:	01177c63          	bleu	a7,a4,1c01183e <__udivdi3+0x2d2>
1c01182a:	9736                	add	a4,a4,a3
1c01182c:	fff58313          	addi	t1,a1,-1
1c011830:	00d76763          	bltu	a4,a3,1c01183e <__udivdi3+0x2d2>
1c011834:	01177563          	bleu	a7,a4,1c01183e <__udivdi3+0x2d2>
1c011838:	ffe58313          	addi	t1,a1,-2
1c01183c:	9736                	add	a4,a4,a3
1c01183e:	07c2                	slli	a5,a5,0x10
1c011840:	6e41                	lui	t3,0x10
1c011842:	0067e7b3          	or	a5,a5,t1
1c011846:	fffe0593          	addi	a1,t3,-1 # ffff <__heapfcram_size+0xdbf>
1c01184a:	00b7f6b3          	and	a3,a5,a1
1c01184e:	41170733          	sub	a4,a4,a7
1c011852:	8df1                	and	a1,a1,a2
1c011854:	0107d893          	srli	a7,a5,0x10
1c011858:	02b68333          	mul	t1,a3,a1
1c01185c:	8241                	srli	a2,a2,0x10
1c01185e:	02b885b3          	mul	a1,a7,a1
1c011862:	8eae                	mv	t4,a1
1c011864:	42c68eb3          	p.mac	t4,a3,a2
1c011868:	01035693          	srli	a3,t1,0x10
1c01186c:	96f6                	add	a3,a3,t4
1c01186e:	02c888b3          	mul	a7,a7,a2
1c011872:	00b6f363          	bleu	a1,a3,1c011878 <__udivdi3+0x30c>
1c011876:	98f2                	add	a7,a7,t3
1c011878:	0106d613          	srli	a2,a3,0x10
1c01187c:	98b2                	add	a7,a7,a2
1c01187e:	03176063          	bltu	a4,a7,1c01189e <__udivdi3+0x332>
1c011882:	db1713e3          	bne	a4,a7,1c011628 <__udivdi3+0xbc>
1c011886:	6741                	lui	a4,0x10
1c011888:	177d                	addi	a4,a4,-1
1c01188a:	8ef9                	and	a3,a3,a4
1c01188c:	06c2                	slli	a3,a3,0x10
1c01188e:	00e37333          	and	t1,t1,a4
1c011892:	01051533          	sll	a0,a0,a6
1c011896:	969a                	add	a3,a3,t1
1c011898:	4801                	li	a6,0
1c01189a:	e2d57de3          	bleu	a3,a0,1c0116d4 <__udivdi3+0x168>
1c01189e:	17fd                	addi	a5,a5,-1
1c0118a0:	b361                	j	1c011628 <__udivdi3+0xbc>
1c0118a2:	4801                	li	a6,0
1c0118a4:	4781                	li	a5,0
1c0118a6:	b53d                	j	1c0116d4 <__udivdi3+0x168>

1c0118a8 <__umoddi3>:
1c0118a8:	88b2                	mv	a7,a2
1c0118aa:	8736                	mv	a4,a3
1c0118ac:	87aa                	mv	a5,a0
1c0118ae:	882e                	mv	a6,a1
1c0118b0:	1a069963          	bnez	a3,1c011a62 <__umoddi3+0x1ba>
1c0118b4:	1c01c6b7          	lui	a3,0x1c01c
1c0118b8:	a4868693          	addi	a3,a3,-1464 # 1c01ba48 <__clz_tab>
1c0118bc:	0ac5f463          	bleu	a2,a1,1c011964 <__umoddi3+0xbc>
1c0118c0:	6341                	lui	t1,0x10
1c0118c2:	08667a63          	bleu	t1,a2,1c011956 <__umoddi3+0xae>
1c0118c6:	0ff00313          	li	t1,255
1c0118ca:	00c37363          	bleu	a2,t1,1c0118d0 <__umoddi3+0x28>
1c0118ce:	4721                	li	a4,8
1c0118d0:	00e65333          	srl	t1,a2,a4
1c0118d4:	969a                	add	a3,a3,t1
1c0118d6:	0006c683          	lbu	a3,0(a3)
1c0118da:	02000313          	li	t1,32
1c0118de:	9736                	add	a4,a4,a3
1c0118e0:	40e30333          	sub	t1,t1,a4
1c0118e4:	00030c63          	beqz	t1,1c0118fc <__umoddi3+0x54>
1c0118e8:	006595b3          	sll	a1,a1,t1
1c0118ec:	00e55733          	srl	a4,a0,a4
1c0118f0:	006618b3          	sll	a7,a2,t1
1c0118f4:	00b76833          	or	a6,a4,a1
1c0118f8:	006517b3          	sll	a5,a0,t1
1c0118fc:	0108d613          	srli	a2,a7,0x10
1c011900:	02c87733          	remu	a4,a6,a2
1c011904:	1008d533          	p.exthz	a0,a7
1c011908:	0107d693          	srli	a3,a5,0x10
1c01190c:	02c85833          	divu	a6,a6,a2
1c011910:	0742                	slli	a4,a4,0x10
1c011912:	8ed9                	or	a3,a3,a4
1c011914:	03050833          	mul	a6,a0,a6
1c011918:	0106f863          	bleu	a6,a3,1c011928 <__umoddi3+0x80>
1c01191c:	96c6                	add	a3,a3,a7
1c01191e:	0116e563          	bltu	a3,a7,1c011928 <__umoddi3+0x80>
1c011922:	0106f363          	bleu	a6,a3,1c011928 <__umoddi3+0x80>
1c011926:	96c6                	add	a3,a3,a7
1c011928:	410686b3          	sub	a3,a3,a6
1c01192c:	02c6f733          	remu	a4,a3,a2
1c011930:	02c6d6b3          	divu	a3,a3,a2
1c011934:	df0727b3          	p.insert	a5,a4,15,16
1c011938:	02d506b3          	mul	a3,a0,a3
1c01193c:	00d7f863          	bleu	a3,a5,1c01194c <__umoddi3+0xa4>
1c011940:	97c6                	add	a5,a5,a7
1c011942:	0117e563          	bltu	a5,a7,1c01194c <__umoddi3+0xa4>
1c011946:	00d7f363          	bleu	a3,a5,1c01194c <__umoddi3+0xa4>
1c01194a:	97c6                	add	a5,a5,a7
1c01194c:	8f95                	sub	a5,a5,a3
1c01194e:	0067d533          	srl	a0,a5,t1
1c011952:	4581                	li	a1,0
1c011954:	8082                	ret
1c011956:	01000337          	lui	t1,0x1000
1c01195a:	4741                	li	a4,16
1c01195c:	f6666ae3          	bltu	a2,t1,1c0118d0 <__umoddi3+0x28>
1c011960:	4761                	li	a4,24
1c011962:	b7bd                	j	1c0118d0 <__umoddi3+0x28>
1c011964:	e601                	bnez	a2,1c01196c <__umoddi3+0xc4>
1c011966:	4605                	li	a2,1
1c011968:	031658b3          	divu	a7,a2,a7
1c01196c:	6641                	lui	a2,0x10
1c01196e:	06c8ff63          	bleu	a2,a7,1c0119ec <__umoddi3+0x144>
1c011972:	0ff00613          	li	a2,255
1c011976:	01167363          	bleu	a7,a2,1c01197c <__umoddi3+0xd4>
1c01197a:	4721                	li	a4,8
1c01197c:	00e8d633          	srl	a2,a7,a4
1c011980:	96b2                	add	a3,a3,a2
1c011982:	0006c603          	lbu	a2,0(a3)
1c011986:	02000313          	li	t1,32
1c01198a:	963a                	add	a2,a2,a4
1c01198c:	40c30333          	sub	t1,t1,a2
1c011990:	06031563          	bnez	t1,1c0119fa <__umoddi3+0x152>
1c011994:	411585b3          	sub	a1,a1,a7
1c011998:	0108d713          	srli	a4,a7,0x10
1c01199c:	1008d533          	p.exthz	a0,a7
1c0119a0:	0107d613          	srli	a2,a5,0x10
1c0119a4:	02e5f6b3          	remu	a3,a1,a4
1c0119a8:	02e5d5b3          	divu	a1,a1,a4
1c0119ac:	06c2                	slli	a3,a3,0x10
1c0119ae:	8ed1                	or	a3,a3,a2
1c0119b0:	02b505b3          	mul	a1,a0,a1
1c0119b4:	00b6f863          	bleu	a1,a3,1c0119c4 <__umoddi3+0x11c>
1c0119b8:	96c6                	add	a3,a3,a7
1c0119ba:	0116e563          	bltu	a3,a7,1c0119c4 <__umoddi3+0x11c>
1c0119be:	00b6f363          	bleu	a1,a3,1c0119c4 <__umoddi3+0x11c>
1c0119c2:	96c6                	add	a3,a3,a7
1c0119c4:	40b685b3          	sub	a1,a3,a1
1c0119c8:	02e5f6b3          	remu	a3,a1,a4
1c0119cc:	02e5d5b3          	divu	a1,a1,a4
1c0119d0:	df06a7b3          	p.insert	a5,a3,15,16
1c0119d4:	02b505b3          	mul	a1,a0,a1
1c0119d8:	00b7f863          	bleu	a1,a5,1c0119e8 <__umoddi3+0x140>
1c0119dc:	97c6                	add	a5,a5,a7
1c0119de:	0117e563          	bltu	a5,a7,1c0119e8 <__umoddi3+0x140>
1c0119e2:	00b7f363          	bleu	a1,a5,1c0119e8 <__umoddi3+0x140>
1c0119e6:	97c6                	add	a5,a5,a7
1c0119e8:	8f8d                	sub	a5,a5,a1
1c0119ea:	b795                	j	1c01194e <__umoddi3+0xa6>
1c0119ec:	01000637          	lui	a2,0x1000
1c0119f0:	4741                	li	a4,16
1c0119f2:	f8c8e5e3          	bltu	a7,a2,1c01197c <__umoddi3+0xd4>
1c0119f6:	4761                	li	a4,24
1c0119f8:	b751                	j	1c01197c <__umoddi3+0xd4>
1c0119fa:	006898b3          	sll	a7,a7,t1
1c0119fe:	00c5d733          	srl	a4,a1,a2
1c011a02:	006517b3          	sll	a5,a0,t1
1c011a06:	00c55633          	srl	a2,a0,a2
1c011a0a:	006595b3          	sll	a1,a1,t1
1c011a0e:	0108d513          	srli	a0,a7,0x10
1c011a12:	8dd1                	or	a1,a1,a2
1c011a14:	02a77633          	remu	a2,a4,a0
1c011a18:	1008d833          	p.exthz	a6,a7
1c011a1c:	0105d693          	srli	a3,a1,0x10
1c011a20:	02a75733          	divu	a4,a4,a0
1c011a24:	0642                	slli	a2,a2,0x10
1c011a26:	8ed1                	or	a3,a3,a2
1c011a28:	02e80733          	mul	a4,a6,a4
1c011a2c:	00e6f863          	bleu	a4,a3,1c011a3c <__umoddi3+0x194>
1c011a30:	96c6                	add	a3,a3,a7
1c011a32:	0116e563          	bltu	a3,a7,1c011a3c <__umoddi3+0x194>
1c011a36:	00e6f363          	bleu	a4,a3,1c011a3c <__umoddi3+0x194>
1c011a3a:	96c6                	add	a3,a3,a7
1c011a3c:	8e99                	sub	a3,a3,a4
1c011a3e:	02a6f733          	remu	a4,a3,a0
1c011a42:	02a6d6b3          	divu	a3,a3,a0
1c011a46:	df0725b3          	p.insert	a1,a4,15,16
1c011a4a:	02d806b3          	mul	a3,a6,a3
1c011a4e:	00d5f863          	bleu	a3,a1,1c011a5e <__umoddi3+0x1b6>
1c011a52:	95c6                	add	a1,a1,a7
1c011a54:	0115e563          	bltu	a1,a7,1c011a5e <__umoddi3+0x1b6>
1c011a58:	00d5f363          	bleu	a3,a1,1c011a5e <__umoddi3+0x1b6>
1c011a5c:	95c6                	add	a1,a1,a7
1c011a5e:	8d95                	sub	a1,a1,a3
1c011a60:	bf25                	j	1c011998 <__umoddi3+0xf0>
1c011a62:	eed5e9e3          	bltu	a1,a3,1c011954 <__umoddi3+0xac>
1c011a66:	6741                	lui	a4,0x10
1c011a68:	04e6f563          	bleu	a4,a3,1c011ab2 <__umoddi3+0x20a>
1c011a6c:	0ff00e93          	li	t4,255
1c011a70:	00deb733          	sltu	a4,t4,a3
1c011a74:	070e                	slli	a4,a4,0x3
1c011a76:	1c01c337          	lui	t1,0x1c01c
1c011a7a:	00e6d8b3          	srl	a7,a3,a4
1c011a7e:	a4830313          	addi	t1,t1,-1464 # 1c01ba48 <__clz_tab>
1c011a82:	989a                	add	a7,a7,t1
1c011a84:	0008ce83          	lbu	t4,0(a7) # 10000 <__heapfcram_size+0xdc0>
1c011a88:	02000e13          	li	t3,32
1c011a8c:	9eba                	add	t4,t4,a4
1c011a8e:	41de0e33          	sub	t3,t3,t4
1c011a92:	020e1763          	bnez	t3,1c011ac0 <__umoddi3+0x218>
1c011a96:	00b6e463          	bltu	a3,a1,1c011a9e <__umoddi3+0x1f6>
1c011a9a:	00c56963          	bltu	a0,a2,1c011aac <__umoddi3+0x204>
1c011a9e:	40c507b3          	sub	a5,a0,a2
1c011aa2:	8d95                	sub	a1,a1,a3
1c011aa4:	00f53533          	sltu	a0,a0,a5
1c011aa8:	40a58833          	sub	a6,a1,a0
1c011aac:	853e                	mv	a0,a5
1c011aae:	85c2                	mv	a1,a6
1c011ab0:	b555                	j	1c011954 <__umoddi3+0xac>
1c011ab2:	010008b7          	lui	a7,0x1000
1c011ab6:	4741                	li	a4,16
1c011ab8:	fb16efe3          	bltu	a3,a7,1c011a76 <__umoddi3+0x1ce>
1c011abc:	4761                	li	a4,24
1c011abe:	bf65                	j	1c011a76 <__umoddi3+0x1ce>
1c011ac0:	01d65733          	srl	a4,a2,t4
1c011ac4:	01c696b3          	sll	a3,a3,t3
1c011ac8:	8ed9                	or	a3,a3,a4
1c011aca:	01d5d7b3          	srl	a5,a1,t4
1c011ace:	0106d813          	srli	a6,a3,0x10
1c011ad2:	0307f333          	remu	t1,a5,a6
1c011ad6:	1006d733          	p.exthz	a4,a3
1c011ada:	01d558b3          	srl	a7,a0,t4
1c011ade:	01c595b3          	sll	a1,a1,t3
1c011ae2:	00b8e5b3          	or	a1,a7,a1
1c011ae6:	0105d893          	srli	a7,a1,0x10
1c011aea:	01c61633          	sll	a2,a2,t3
1c011aee:	01c51533          	sll	a0,a0,t3
1c011af2:	0307d7b3          	divu	a5,a5,a6
1c011af6:	0342                	slli	t1,t1,0x10
1c011af8:	011368b3          	or	a7,t1,a7
1c011afc:	02f70f33          	mul	t5,a4,a5
1c011b00:	833e                	mv	t1,a5
1c011b02:	01e8fc63          	bleu	t5,a7,1c011b1a <__umoddi3+0x272>
1c011b06:	98b6                	add	a7,a7,a3
1c011b08:	fff78313          	addi	t1,a5,-1 # ffffff <__l1_heapsram_size+0xfe006f>
1c011b0c:	00d8e763          	bltu	a7,a3,1c011b1a <__umoddi3+0x272>
1c011b10:	01e8f563          	bleu	t5,a7,1c011b1a <__umoddi3+0x272>
1c011b14:	ffe78313          	addi	t1,a5,-2
1c011b18:	98b6                	add	a7,a7,a3
1c011b1a:	41e888b3          	sub	a7,a7,t5
1c011b1e:	0308f7b3          	remu	a5,a7,a6
1c011b22:	0308d8b3          	divu	a7,a7,a6
1c011b26:	df07a5b3          	p.insert	a1,a5,15,16
1c011b2a:	03170733          	mul	a4,a4,a7
1c011b2e:	87c6                	mv	a5,a7
1c011b30:	00e5fc63          	bleu	a4,a1,1c011b48 <__umoddi3+0x2a0>
1c011b34:	95b6                	add	a1,a1,a3
1c011b36:	fff88793          	addi	a5,a7,-1 # ffffff <__l1_heapsram_size+0xfe006f>
1c011b3a:	00d5e763          	bltu	a1,a3,1c011b48 <__umoddi3+0x2a0>
1c011b3e:	00e5f563          	bleu	a4,a1,1c011b48 <__umoddi3+0x2a0>
1c011b42:	ffe88793          	addi	a5,a7,-2
1c011b46:	95b6                	add	a1,a1,a3
1c011b48:	0342                	slli	t1,t1,0x10
1c011b4a:	6f41                	lui	t5,0x10
1c011b4c:	00f36333          	or	t1,t1,a5
1c011b50:	ffff0793          	addi	a5,t5,-1 # ffff <__heapfcram_size+0xdbf>
1c011b54:	00f37833          	and	a6,t1,a5
1c011b58:	01035313          	srli	t1,t1,0x10
1c011b5c:	8ff1                	and	a5,a5,a2
1c011b5e:	02f808b3          	mul	a7,a6,a5
1c011b62:	8d99                	sub	a1,a1,a4
1c011b64:	01065713          	srli	a4,a2,0x10
1c011b68:	02f307b3          	mul	a5,t1,a5
1c011b6c:	8fbe                	mv	t6,a5
1c011b6e:	42e80fb3          	p.mac	t6,a6,a4
1c011b72:	0108d813          	srli	a6,a7,0x10
1c011b76:	987e                	add	a6,a6,t6
1c011b78:	02e30333          	mul	t1,t1,a4
1c011b7c:	00f87363          	bleu	a5,a6,1c011b82 <__umoddi3+0x2da>
1c011b80:	937a                	add	t1,t1,t5
1c011b82:	01085713          	srli	a4,a6,0x10
1c011b86:	933a                	add	t1,t1,a4
1c011b88:	6741                	lui	a4,0x10
1c011b8a:	177d                	addi	a4,a4,-1
1c011b8c:	00e87833          	and	a6,a6,a4
1c011b90:	0842                	slli	a6,a6,0x10
1c011b92:	00e8f733          	and	a4,a7,a4
1c011b96:	9742                	add	a4,a4,a6
1c011b98:	0065e663          	bltu	a1,t1,1c011ba4 <__umoddi3+0x2fc>
1c011b9c:	00659d63          	bne	a1,t1,1c011bb6 <__umoddi3+0x30e>
1c011ba0:	00e57b63          	bleu	a4,a0,1c011bb6 <__umoddi3+0x30e>
1c011ba4:	40c70633          	sub	a2,a4,a2
1c011ba8:	00c73733          	sltu	a4,a4,a2
1c011bac:	40d30333          	sub	t1,t1,a3
1c011bb0:	40e30333          	sub	t1,t1,a4
1c011bb4:	8732                	mv	a4,a2
1c011bb6:	40e50733          	sub	a4,a0,a4
1c011bba:	00e53533          	sltu	a0,a0,a4
1c011bbe:	406585b3          	sub	a1,a1,t1
1c011bc2:	8d89                	sub	a1,a1,a0
1c011bc4:	01d597b3          	sll	a5,a1,t4
1c011bc8:	01c75733          	srl	a4,a4,t3
1c011bcc:	00e7e533          	or	a0,a5,a4
1c011bd0:	01c5d5b3          	srl	a1,a1,t3
1c011bd4:	b341                	j	1c011954 <__umoddi3+0xac>

1c011bd6 <__adddf3>:
1c011bd6:	00202ef3          	frrm	t4
1c011bda:	e60597b3          	p.extractu	a5,a1,19,0
1c011bde:	00379893          	slli	a7,a5,0x3
1c011be2:	01d55793          	srli	a5,a0,0x1d
1c011be6:	d5459f33          	p.extractu	t5,a1,10,20
1c011bea:	d5469333          	p.extractu	t1,a3,10,20
1c011bee:	0117e7b3          	or	a5,a5,a7
1c011bf2:	e60698b3          	p.extractu	a7,a3,19,0
1c011bf6:	01d65713          	srli	a4,a2,0x1d
1c011bfa:	088e                	slli	a7,a7,0x3
1c011bfc:	00361813          	slli	a6,a2,0x3
1c011c00:	81fd                	srli	a1,a1,0x1f
1c011c02:	406f0633          	sub	a2,t5,t1
1c011c06:	82fd                	srli	a3,a3,0x1f
1c011c08:	050e                	slli	a0,a0,0x3
1c011c0a:	01176733          	or	a4,a4,a7
1c011c0e:	8e32                	mv	t3,a2
1c011c10:	38d59b63          	bne	a1,a3,1c011fa6 <__adddf3+0x3d0>
1c011c14:	12c05263          	blez	a2,1c011d38 <__adddf3+0x162>
1c011c18:	08031663          	bnez	t1,1c011ca4 <__adddf3+0xce>
1c011c1c:	010766b3          	or	a3,a4,a6
1c011c20:	e295                	bnez	a3,1c011c44 <__adddf3+0x6e>
1c011c22:	7ff00713          	li	a4,2047
1c011c26:	38e61b63          	bne	a2,a4,1c011fbc <__adddf3+0x3e6>
1c011c2a:	00a7e733          	or	a4,a5,a0
1c011c2e:	76070763          	beqz	a4,1c01239c <__adddf3+0x7c6>
1c011c32:	0167d713          	srli	a4,a5,0x16
1c011c36:	7ff00313          	li	t1,2047
1c011c3a:	4e01                	li	t3,0
1c011c3c:	40070b63          	beqz	a4,1c012052 <__adddf3+0x47c>
1c011c40:	4881                	li	a7,0
1c011c42:	ac39                	j	1c011e60 <__adddf3+0x28a>
1c011c44:	167d                	addi	a2,a2,-1
1c011c46:	e221                	bnez	a2,1c011c86 <__adddf3+0xb0>
1c011c48:	982a                	add	a6,a6,a0
1c011c4a:	97ba                	add	a5,a5,a4
1c011c4c:	00a83533          	sltu	a0,a6,a0
1c011c50:	00a78733          	add	a4,a5,a0
1c011c54:	4305                	li	t1,1
1c011c56:	00871793          	slli	a5,a4,0x8
1c011c5a:	0e07d963          	bgez	a5,1c011d4c <__adddf3+0x176>
1c011c5e:	0305                	addi	t1,t1,1
1c011c60:	7ff00793          	li	a5,2047
1c011c64:	30f30f63          	beq	t1,a5,1c011f82 <__adddf3+0x3ac>
1c011c68:	c17737b3          	p.bclr	a5,a4,0,23
1c011c6c:	00185693          	srli	a3,a6,0x1
1c011c70:	fc183833          	p.bclr	a6,a6,30,1
1c011c74:	01f79513          	slli	a0,a5,0x1f
1c011c78:	0106e833          	or	a6,a3,a6
1c011c7c:	01056533          	or	a0,a0,a6
1c011c80:	8385                	srli	a5,a5,0x1
1c011c82:	4e01                	li	t3,0
1c011c84:	bf75                	j	1c011c40 <__adddf3+0x6a>
1c011c86:	7ff00693          	li	a3,2047
1c011c8a:	02de1c63          	bne	t3,a3,1c011cc2 <__adddf3+0xec>
1c011c8e:	00a7e733          	or	a4,a5,a0
1c011c92:	70070563          	beqz	a4,1c01239c <__adddf3+0x7c6>
1c011c96:	00979313          	slli	t1,a5,0x9
1c011c9a:	6e035d63          	bgez	t1,1c012394 <__adddf3+0x7be>
1c011c9e:	7ff00313          	li	t1,2047
1c011ca2:	b7c5                	j	1c011c82 <__adddf3+0xac>
1c011ca4:	7ff00693          	li	a3,2047
1c011ca8:	00df1b63          	bne	t5,a3,1c011cbe <__adddf3+0xe8>
1c011cac:	00a7e733          	or	a4,a5,a0
1c011cb0:	6e070663          	beqz	a4,1c01239c <__adddf3+0x7c6>
1c011cb4:	00979613          	slli	a2,a5,0x9
1c011cb8:	7ff00313          	li	t1,2047
1c011cbc:	a0e1                	j	1c011d84 <__adddf3+0x1ae>
1c011cbe:	c1774733          	p.bset	a4,a4,0,23
1c011cc2:	03800693          	li	a3,56
1c011cc6:	06c6c463          	blt	a3,a2,1c011d2e <__adddf3+0x158>
1c011cca:	46fd                	li	a3,31
1c011ccc:	02c6cc63          	blt	a3,a2,1c011d04 <__adddf3+0x12e>
1c011cd0:	02000893          	li	a7,32
1c011cd4:	40c888b3          	sub	a7,a7,a2
1c011cd8:	00c85333          	srl	t1,a6,a2
1c011cdc:	011716b3          	sll	a3,a4,a7
1c011ce0:	01181833          	sll	a6,a6,a7
1c011ce4:	0066e6b3          	or	a3,a3,t1
1c011ce8:	01003833          	snez	a6,a6
1c011cec:	0106e833          	or	a6,a3,a6
1c011cf0:	00c75633          	srl	a2,a4,a2
1c011cf4:	982a                	add	a6,a6,a0
1c011cf6:	963e                	add	a2,a2,a5
1c011cf8:	00a837b3          	sltu	a5,a6,a0
1c011cfc:	00f60733          	add	a4,a2,a5
1c011d00:	837a                	mv	t1,t5
1c011d02:	bf91                	j	1c011c56 <__adddf3+0x80>
1c011d04:	02000313          	li	t1,32
1c011d08:	00c756b3          	srl	a3,a4,a2
1c011d0c:	4881                	li	a7,0
1c011d0e:	00660863          	beq	a2,t1,1c011d1e <__adddf3+0x148>
1c011d12:	04000893          	li	a7,64
1c011d16:	40c88633          	sub	a2,a7,a2
1c011d1a:	00c718b3          	sll	a7,a4,a2
1c011d1e:	0108e833          	or	a6,a7,a6
1c011d22:	01003833          	snez	a6,a6
1c011d26:	0106e833          	or	a6,a3,a6
1c011d2a:	4601                	li	a2,0
1c011d2c:	b7e1                	j	1c011cf4 <__adddf3+0x11e>
1c011d2e:	01076833          	or	a6,a4,a6
1c011d32:	01003833          	snez	a6,a6
1c011d36:	bfd5                	j	1c011d2a <__adddf3+0x154>
1c011d38:	c66d                	beqz	a2,1c011e22 <__adddf3+0x24c>
1c011d3a:	040f1963          	bnez	t5,1c011d8c <__adddf3+0x1b6>
1c011d3e:	00a7e6b3          	or	a3,a5,a0
1c011d42:	ea89                	bnez	a3,1c011d54 <__adddf3+0x17e>
1c011d44:	7ff00793          	li	a5,2047
1c011d48:	04f30663          	beq	t1,a5,1c011d94 <__adddf3+0x1be>
1c011d4c:	04031c63          	bnez	t1,1c011da4 <__adddf3+0x1ce>
1c011d50:	8542                	mv	a0,a6
1c011d52:	a999                	j	1c0121a8 <__adddf3+0x5d2>
1c011d54:	01f63a63          	p.bneimm	a2,-1,1c011d68 <__adddf3+0x192>
1c011d58:	010506b3          	add	a3,a0,a6
1c011d5c:	97ba                	add	a5,a5,a4
1c011d5e:	0106b733          	sltu	a4,a3,a6
1c011d62:	973e                	add	a4,a4,a5
1c011d64:	8836                	mv	a6,a3
1c011d66:	bdc5                	j	1c011c56 <__adddf3+0x80>
1c011d68:	7ff00693          	li	a3,2047
1c011d6c:	fff64613          	not	a2,a2
1c011d70:	04d31163          	bne	t1,a3,1c011db2 <__adddf3+0x1dc>
1c011d74:	01076533          	or	a0,a4,a6
1c011d78:	62050863          	beqz	a0,1c0123a8 <__adddf3+0x7d2>
1c011d7c:	00971613          	slli	a2,a4,0x9
1c011d80:	87ba                	mv	a5,a4
1c011d82:	8542                	mv	a0,a6
1c011d84:	4e01                	li	t3,0
1c011d86:	2c065663          	bgez	a2,1c012052 <__adddf3+0x47c>
1c011d8a:	bd5d                	j	1c011c40 <__adddf3+0x6a>
1c011d8c:	7ff00693          	li	a3,2047
1c011d90:	00d31d63          	bne	t1,a3,1c011daa <__adddf3+0x1d4>
1c011d94:	01076533          	or	a0,a4,a6
1c011d98:	60050863          	beqz	a0,1c0123a8 <__adddf3+0x7d2>
1c011d9c:	00971793          	slli	a5,a4,0x9
1c011da0:	6007d663          	bgez	a5,1c0123ac <__adddf3+0x7d6>
1c011da4:	87ba                	mv	a5,a4
1c011da6:	8542                	mv	a0,a6
1c011da8:	bde9                	j	1c011c82 <__adddf3+0xac>
1c011daa:	40c00633          	neg	a2,a2
1c011dae:	c177c7b3          	p.bset	a5,a5,0,23
1c011db2:	03800693          	li	a3,56
1c011db6:	06c6c263          	blt	a3,a2,1c011e1a <__adddf3+0x244>
1c011dba:	46fd                	li	a3,31
1c011dbc:	02c6cb63          	blt	a3,a2,1c011df2 <__adddf3+0x21c>
1c011dc0:	02000893          	li	a7,32
1c011dc4:	40c888b3          	sub	a7,a7,a2
1c011dc8:	00c55e33          	srl	t3,a0,a2
1c011dcc:	011796b3          	sll	a3,a5,a7
1c011dd0:	01151533          	sll	a0,a0,a7
1c011dd4:	01c6e6b3          	or	a3,a3,t3
1c011dd8:	00a03533          	snez	a0,a0
1c011ddc:	8d55                	or	a0,a0,a3
1c011dde:	00c7d633          	srl	a2,a5,a2
1c011de2:	9542                	add	a0,a0,a6
1c011de4:	00e607b3          	add	a5,a2,a4
1c011de8:	01053733          	sltu	a4,a0,a6
1c011dec:	973e                	add	a4,a4,a5
1c011dee:	882a                	mv	a6,a0
1c011df0:	b59d                	j	1c011c56 <__adddf3+0x80>
1c011df2:	02000e13          	li	t3,32
1c011df6:	00c7d6b3          	srl	a3,a5,a2
1c011dfa:	4881                	li	a7,0
1c011dfc:	01c60863          	beq	a2,t3,1c011e0c <__adddf3+0x236>
1c011e00:	04000893          	li	a7,64
1c011e04:	40c88633          	sub	a2,a7,a2
1c011e08:	00c798b3          	sll	a7,a5,a2
1c011e0c:	00a8e533          	or	a0,a7,a0
1c011e10:	00a03533          	snez	a0,a0
1c011e14:	8d55                	or	a0,a0,a3
1c011e16:	4601                	li	a2,0
1c011e18:	b7e9                	j	1c011de2 <__adddf3+0x20c>
1c011e1a:	8d5d                	or	a0,a0,a5
1c011e1c:	00a03533          	snez	a0,a0
1c011e20:	bfdd                	j	1c011e16 <__adddf3+0x240>
1c011e22:	001f0693          	addi	a3,t5,1
1c011e26:	e8b6b8b3          	p.bclr	a7,a3,20,11
1c011e2a:	4605                	li	a2,1
1c011e2c:	0f164663          	blt	a2,a7,1c011f18 <__adddf3+0x342>
1c011e30:	00a7e6b3          	or	a3,a5,a0
1c011e34:	040f1163          	bnez	t5,1c011e76 <__adddf3+0x2a0>
1c011e38:	de81                	beqz	a3,1c011d50 <__adddf3+0x17a>
1c011e3a:	010766b3          	or	a3,a4,a6
1c011e3e:	52068663          	beqz	a3,1c01236a <__adddf3+0x794>
1c011e42:	982a                	add	a6,a6,a0
1c011e44:	97ba                	add	a5,a5,a4
1c011e46:	00a83533          	sltu	a0,a6,a0
1c011e4a:	00a78733          	add	a4,a5,a0
1c011e4e:	00871313          	slli	t1,a4,0x8
1c011e52:	ee035fe3          	bgez	t1,1c011d50 <__adddf3+0x17a>
1c011e56:	c17737b3          	p.bclr	a5,a4,0,23
1c011e5a:	8542                	mv	a0,a6
1c011e5c:	4881                	li	a7,0
1c011e5e:	4305                	li	t1,1
1c011e60:	f8353733          	p.bclr	a4,a0,28,3
1c011e64:	e775                	bnez	a4,1c011f50 <__adddf3+0x37a>
1c011e66:	040e0b63          	beqz	t3,1c011ebc <__adddf3+0x2e6>
1c011e6a:	fc18b733          	p.bclr	a4,a7,30,1
1c011e6e:	c739                	beqz	a4,1c011ebc <__adddf3+0x2e6>
1c011e70:	c018c8b3          	p.bset	a7,a7,0,1
1c011e74:	a0a1                	j	1c011ebc <__adddf3+0x2e6>
1c011e76:	7ff00613          	li	a2,2047
1c011e7a:	08cf1a63          	bne	t5,a2,1c011f0e <__adddf3+0x338>
1c011e7e:	58068963          	beqz	a3,1c012410 <__adddf3+0x83a>
1c011e82:	004008b7          	lui	a7,0x400
1c011e86:	0117f8b3          	and	a7,a5,a7
1c011e8a:	0018b893          	seqz	a7,a7
1c011e8e:	0892                	slli	a7,a7,0x4
1c011e90:	01e31c63          	bne	t1,t5,1c011ea8 <__adddf3+0x2d2>
1c011e94:	01076633          	or	a2,a4,a6
1c011e98:	c611                	beqz	a2,1c011ea4 <__adddf3+0x2ce>
1c011e9a:	00971613          	slli	a2,a4,0x9
1c011e9e:	00064363          	bltz	a2,1c011ea4 <__adddf3+0x2ce>
1c011ea2:	48c1                	li	a7,16
1c011ea4:	4e068563          	beqz	a3,1c01238e <__adddf3+0x7b8>
1c011ea8:	01076733          	or	a4,a4,a6
1c011eac:	3c070263          	beqz	a4,1c012270 <__adddf3+0x69a>
1c011eb0:	004007b7          	lui	a5,0x400
1c011eb4:	4501                	li	a0,0
1c011eb6:	7ff00313          	li	t1,2047
1c011eba:	4581                	li	a1,0
1c011ebc:	00879713          	slli	a4,a5,0x8
1c011ec0:	00075963          	bgez	a4,1c011ed2 <__adddf3+0x2fc>
1c011ec4:	0305                	addi	t1,t1,1
1c011ec6:	7ff00713          	li	a4,2047
1c011eca:	50e30e63          	beq	t1,a4,1c0123e6 <__adddf3+0x810>
1c011ece:	c177b7b3          	p.bclr	a5,a5,0,23
1c011ed2:	01d79713          	slli	a4,a5,0x1d
1c011ed6:	810d                	srli	a0,a0,0x3
1c011ed8:	7ff00693          	li	a3,2047
1c011edc:	8f49                	or	a4,a4,a0
1c011ede:	838d                	srli	a5,a5,0x3
1c011ee0:	00d31963          	bne	t1,a3,1c011ef2 <__adddf3+0x31c>
1c011ee4:	8f5d                	or	a4,a4,a5
1c011ee6:	4781                	li	a5,0
1c011ee8:	c709                	beqz	a4,1c011ef2 <__adddf3+0x31c>
1c011eea:	000807b7          	lui	a5,0x80
1c011eee:	4701                	li	a4,0
1c011ef0:	4581                	li	a1,0
1c011ef2:	4681                	li	a3,0
1c011ef4:	e607a6b3          	p.insert	a3,a5,19,0
1c011ef8:	d54326b3          	p.insert	a3,t1,10,20
1c011efc:	c1f5a6b3          	p.insert	a3,a1,0,31
1c011f00:	853a                	mv	a0,a4
1c011f02:	85b6                	mv	a1,a3
1c011f04:	50088d63          	beqz	a7,1c01241e <__adddf3+0x848>
1c011f08:	0018a073          	csrs	0x1,a7
1c011f0c:	8082                	ret
1c011f0e:	4881                	li	a7,0
1c011f10:	f8c31ae3          	bne	t1,a2,1c011ea4 <__adddf3+0x2ce>
1c011f14:	4881                	li	a7,0
1c011f16:	bfbd                	j	1c011e94 <__adddf3+0x2be>
1c011f18:	7ff00613          	li	a2,2047
1c011f1c:	02c68063          	beq	a3,a2,1c011f3c <__adddf3+0x366>
1c011f20:	982a                	add	a6,a6,a0
1c011f22:	00a83533          	sltu	a0,a6,a0
1c011f26:	97ba                	add	a5,a5,a4
1c011f28:	97aa                	add	a5,a5,a0
1c011f2a:	01f79513          	slli	a0,a5,0x1f
1c011f2e:	00185813          	srli	a6,a6,0x1
1c011f32:	01056533          	or	a0,a0,a6
1c011f36:	8385                	srli	a5,a5,0x1
1c011f38:	8336                	mv	t1,a3
1c011f3a:	b319                	j	1c011c40 <__adddf3+0x6a>
1c011f3c:	040e8063          	beqz	t4,1c011f7c <__adddf3+0x3a6>
1c011f40:	023ebb63          	p.bneimm	t4,3,1c011f76 <__adddf3+0x3a0>
1c011f44:	cd85                	beqz	a1,1c011f7c <__adddf3+0x3a6>
1c011f46:	57fd                	li	a5,-1
1c011f48:	557d                	li	a0,-1
1c011f4a:	7fe00313          	li	t1,2046
1c011f4e:	4895                	li	a7,5
1c011f50:	c008c8b3          	p.bset	a7,a7,0,0
1c011f54:	482ea663          	p.beqimm	t4,2,1c0123e0 <__adddf3+0x80a>
1c011f58:	463eaf63          	p.beqimm	t4,3,1c0123d6 <__adddf3+0x800>
1c011f5c:	f00e95e3          	bnez	t4,1c011e66 <__adddf3+0x290>
1c011f60:	f6453733          	p.bclr	a4,a0,27,4
1c011f64:	f04721e3          	p.beqimm	a4,4,1c011e66 <__adddf3+0x290>
1c011f68:	00450713          	addi	a4,a0,4
1c011f6c:	00a73533          	sltu	a0,a4,a0
1c011f70:	97aa                	add	a5,a5,a0
1c011f72:	853a                	mv	a0,a4
1c011f74:	bdcd                	j	1c011e66 <__adddf3+0x290>
1c011f76:	fc2eb8e3          	p.bneimm	t4,2,1c011f46 <__adddf3+0x370>
1c011f7a:	d5f1                	beqz	a1,1c011f46 <__adddf3+0x370>
1c011f7c:	7ff00313          	li	t1,2047
1c011f80:	a839                	j	1c011f9e <__adddf3+0x3c8>
1c011f82:	000e8e63          	beqz	t4,1c011f9e <__adddf3+0x3c8>
1c011f86:	003eb963          	p.bneimm	t4,3,1c011f98 <__adddf3+0x3c2>
1c011f8a:	c991                	beqz	a1,1c011f9e <__adddf3+0x3c8>
1c011f8c:	57fd                	li	a5,-1
1c011f8e:	557d                	li	a0,-1
1c011f90:	7fe00313          	li	t1,2046
1c011f94:	4e01                	li	t3,0
1c011f96:	bf65                	j	1c011f4e <__adddf3+0x378>
1c011f98:	fe2ebae3          	p.bneimm	t4,2,1c011f8c <__adddf3+0x3b6>
1c011f9c:	d9e5                	beqz	a1,1c011f8c <__adddf3+0x3b6>
1c011f9e:	4781                	li	a5,0
1c011fa0:	4501                	li	a0,0
1c011fa2:	4895                	li	a7,5
1c011fa4:	bf21                	j	1c011ebc <__adddf3+0x2e6>
1c011fa6:	0ec05563          	blez	a2,1c012090 <__adddf3+0x4ba>
1c011faa:	08031563          	bnez	t1,1c012034 <__adddf3+0x45e>
1c011fae:	010766b3          	or	a3,a4,a6
1c011fb2:	e699                	bnez	a3,1c011fc0 <__adddf3+0x3ea>
1c011fb4:	7ff00713          	li	a4,2047
1c011fb8:	cce60be3          	beq	a2,a4,1c011c8e <__adddf3+0xb8>
1c011fbc:	8332                	mv	t1,a2
1c011fbe:	b1d1                	j	1c011c82 <__adddf3+0xac>
1c011fc0:	167d                	addi	a2,a2,-1
1c011fc2:	e215                	bnez	a2,1c011fe6 <__adddf3+0x410>
1c011fc4:	41050833          	sub	a6,a0,a6
1c011fc8:	8f99                	sub	a5,a5,a4
1c011fca:	01053533          	sltu	a0,a0,a6
1c011fce:	40a78733          	sub	a4,a5,a0
1c011fd2:	4305                	li	t1,1
1c011fd4:	00871613          	slli	a2,a4,0x8
1c011fd8:	d6065ae3          	bgez	a2,1c011d4c <__adddf3+0x176>
1c011fdc:	d1773633          	p.bclr	a2,a4,8,23
1c011fe0:	88c2                	mv	a7,a6
1c011fe2:	8f1a                	mv	t5,t1
1c011fe4:	a4e9                	j	1c0122ae <__adddf3+0x6d8>
1c011fe6:	7ff00693          	li	a3,2047
1c011fea:	ccde01e3          	beq	t3,a3,1c011cac <__adddf3+0xd6>
1c011fee:	03800693          	li	a3,56
1c011ff2:	08c6ca63          	blt	a3,a2,1c012086 <__adddf3+0x4b0>
1c011ff6:	46fd                	li	a3,31
1c011ff8:	06c6c263          	blt	a3,a2,1c01205c <__adddf3+0x486>
1c011ffc:	02000893          	li	a7,32
1c012000:	40c888b3          	sub	a7,a7,a2
1c012004:	00c85333          	srl	t1,a6,a2
1c012008:	011716b3          	sll	a3,a4,a7
1c01200c:	01181833          	sll	a6,a6,a7
1c012010:	0066e6b3          	or	a3,a3,t1
1c012014:	01003833          	snez	a6,a6
1c012018:	0106e833          	or	a6,a3,a6
1c01201c:	00c75633          	srl	a2,a4,a2
1c012020:	41050833          	sub	a6,a0,a6
1c012024:	40c78633          	sub	a2,a5,a2
1c012028:	010537b3          	sltu	a5,a0,a6
1c01202c:	40f60733          	sub	a4,a2,a5
1c012030:	837a                	mv	t1,t5
1c012032:	b74d                	j	1c011fd4 <__adddf3+0x3fe>
1c012034:	7ff00693          	li	a3,2047
1c012038:	00df1f63          	bne	t5,a3,1c012056 <__adddf3+0x480>
1c01203c:	00a7e733          	or	a4,a5,a0
1c012040:	34070e63          	beqz	a4,1c01239c <__adddf3+0x7c6>
1c012044:	00979713          	slli	a4,a5,0x9
1c012048:	7ff00313          	li	t1,2047
1c01204c:	4e01                	li	t3,0
1c01204e:	be0749e3          	bltz	a4,1c011c40 <__adddf3+0x6a>
1c012052:	48c1                	li	a7,16
1c012054:	b531                	j	1c011e60 <__adddf3+0x28a>
1c012056:	c1774733          	p.bset	a4,a4,0,23
1c01205a:	bf51                	j	1c011fee <__adddf3+0x418>
1c01205c:	02000313          	li	t1,32
1c012060:	00c756b3          	srl	a3,a4,a2
1c012064:	4881                	li	a7,0
1c012066:	00660863          	beq	a2,t1,1c012076 <__adddf3+0x4a0>
1c01206a:	04000893          	li	a7,64
1c01206e:	40c88633          	sub	a2,a7,a2
1c012072:	00c718b3          	sll	a7,a4,a2
1c012076:	0108e833          	or	a6,a7,a6
1c01207a:	01003833          	snez	a6,a6
1c01207e:	0106e833          	or	a6,a3,a6
1c012082:	4601                	li	a2,0
1c012084:	bf71                	j	1c012020 <__adddf3+0x44a>
1c012086:	01076833          	or	a6,a4,a6
1c01208a:	01003833          	snez	a6,a6
1c01208e:	bfd5                	j	1c012082 <__adddf3+0x4ac>
1c012090:	c66d                	beqz	a2,1c01217a <__adddf3+0x5a4>
1c012092:	040f1663          	bnez	t5,1c0120de <__adddf3+0x508>
1c012096:	00a7e5b3          	or	a1,a5,a0
1c01209a:	e589                	bnez	a1,1c0120a4 <__adddf3+0x4ce>
1c01209c:	7ff00793          	li	a5,2047
1c0120a0:	85b6                	mv	a1,a3
1c0120a2:	b15d                	j	1c011d48 <__adddf3+0x172>
1c0120a4:	01f63d63          	p.bneimm	a2,-1,1c0120be <__adddf3+0x4e8>
1c0120a8:	40a805b3          	sub	a1,a6,a0
1c0120ac:	40f707b3          	sub	a5,a4,a5
1c0120b0:	00b83733          	sltu	a4,a6,a1
1c0120b4:	40e78733          	sub	a4,a5,a4
1c0120b8:	882e                	mv	a6,a1
1c0120ba:	85b6                	mv	a1,a3
1c0120bc:	bf21                	j	1c011fd4 <__adddf3+0x3fe>
1c0120be:	7ff00593          	li	a1,2047
1c0120c2:	fff64613          	not	a2,a2
1c0120c6:	04b31063          	bne	t1,a1,1c012106 <__adddf3+0x530>
1c0120ca:	01076533          	or	a0,a4,a6
1c0120ce:	2e050263          	beqz	a0,1c0123b2 <__adddf3+0x7dc>
1c0120d2:	00971613          	slli	a2,a4,0x9
1c0120d6:	87ba                	mv	a5,a4
1c0120d8:	8542                	mv	a0,a6
1c0120da:	85b6                	mv	a1,a3
1c0120dc:	b165                	j	1c011d84 <__adddf3+0x1ae>
1c0120de:	7ff00593          	li	a1,2047
1c0120e2:	00b31e63          	bne	t1,a1,1c0120fe <__adddf3+0x528>
1c0120e6:	01076533          	or	a0,a4,a6
1c0120ea:	2c050463          	beqz	a0,1c0123b2 <__adddf3+0x7dc>
1c0120ee:	00971793          	slli	a5,a4,0x9
1c0120f2:	2c07d363          	bgez	a5,1c0123b8 <__adddf3+0x7e2>
1c0120f6:	87ba                	mv	a5,a4
1c0120f8:	8542                	mv	a0,a6
1c0120fa:	85b6                	mv	a1,a3
1c0120fc:	b659                	j	1c011c82 <__adddf3+0xac>
1c0120fe:	40c00633          	neg	a2,a2
1c012102:	c177c7b3          	p.bset	a5,a5,0,23
1c012106:	03800593          	li	a1,56
1c01210a:	06c5c463          	blt	a1,a2,1c012172 <__adddf3+0x59c>
1c01210e:	45fd                	li	a1,31
1c012110:	02c5cd63          	blt	a1,a2,1c01214a <__adddf3+0x574>
1c012114:	02000893          	li	a7,32
1c012118:	40c888b3          	sub	a7,a7,a2
1c01211c:	00c55e33          	srl	t3,a0,a2
1c012120:	011795b3          	sll	a1,a5,a7
1c012124:	01151533          	sll	a0,a0,a7
1c012128:	01c5e5b3          	or	a1,a1,t3
1c01212c:	00a03533          	snez	a0,a0
1c012130:	8d4d                	or	a0,a0,a1
1c012132:	00c7d633          	srl	a2,a5,a2
1c012136:	40a80533          	sub	a0,a6,a0
1c01213a:	40c707b3          	sub	a5,a4,a2
1c01213e:	00a83733          	sltu	a4,a6,a0
1c012142:	40e78733          	sub	a4,a5,a4
1c012146:	882a                	mv	a6,a0
1c012148:	bf8d                	j	1c0120ba <__adddf3+0x4e4>
1c01214a:	02000e13          	li	t3,32
1c01214e:	00c7d5b3          	srl	a1,a5,a2
1c012152:	4881                	li	a7,0
1c012154:	01c60863          	beq	a2,t3,1c012164 <__adddf3+0x58e>
1c012158:	04000893          	li	a7,64
1c01215c:	40c88633          	sub	a2,a7,a2
1c012160:	00c798b3          	sll	a7,a5,a2
1c012164:	00a8e533          	or	a0,a7,a0
1c012168:	00a03533          	snez	a0,a0
1c01216c:	8d4d                	or	a0,a0,a1
1c01216e:	4601                	li	a2,0
1c012170:	b7d9                	j	1c012136 <__adddf3+0x560>
1c012172:	8d5d                	or	a0,a0,a5
1c012174:	00a03533          	snez	a0,a0
1c012178:	bfdd                	j	1c01216e <__adddf3+0x598>
1c01217a:	001f0613          	addi	a2,t5,1
1c01217e:	e8b63633          	p.bclr	a2,a2,20,11
1c012182:	4885                	li	a7,1
1c012184:	10c8c063          	blt	a7,a2,1c012284 <__adddf3+0x6ae>
1c012188:	00a7efb3          	or	t6,a5,a0
1c01218c:	01076633          	or	a2,a4,a6
1c012190:	0a0f1463          	bnez	t5,1c012238 <__adddf3+0x662>
1c012194:	060f9063          	bnez	t6,1c0121f4 <__adddf3+0x61e>
1c012198:	1c061763          	bnez	a2,1c012366 <__adddf3+0x790>
1c01219c:	ffee8593          	addi	a1,t4,-2
1c0121a0:	0015b593          	seqz	a1,a1
1c0121a4:	4701                	li	a4,0
1c0121a6:	4501                	li	a0,0
1c0121a8:	00e567b3          	or	a5,a0,a4
1c0121ac:	22078263          	beqz	a5,1c0123d0 <__adddf3+0x7fa>
1c0121b0:	01f55793          	srli	a5,a0,0x1f
1c0121b4:	00171e13          	slli	t3,a4,0x1
1c0121b8:	9e3e                	add	t3,t3,a5
1c0121ba:	00151793          	slli	a5,a0,0x1
1c0121be:	f837b6b3          	p.bclr	a3,a5,28,3
1c0121c2:	4881                	li	a7,0
1c0121c4:	ce99                	beqz	a3,1c0121e2 <__adddf3+0x60c>
1c0121c6:	1c2ea063          	p.beqimm	t4,2,1c012386 <__adddf3+0x7b0>
1c0121ca:	1a3ea563          	p.beqimm	t4,3,1c012374 <__adddf3+0x79e>
1c0121ce:	4885                	li	a7,1
1c0121d0:	000e9963          	bnez	t4,1c0121e2 <__adddf3+0x60c>
1c0121d4:	f647b6b3          	p.bclr	a3,a5,27,4
1c0121d8:	4885                	li	a7,1
1c0121da:	ffc7b793          	sltiu	a5,a5,-4
1c0121de:	1a46b063          	p.bneimm	a3,4,1c01237e <__adddf3+0x7a8>
1c0121e2:	018e5e13          	srli	t3,t3,0x18
1c0121e6:	001e4e13          	xori	t3,t3,1
1c0121ea:	fc1e3e33          	p.bclr	t3,t3,30,1
1c0121ee:	87ba                	mv	a5,a4
1c0121f0:	4301                	li	t1,0
1c0121f2:	b1bd                	j	1c011e60 <__adddf3+0x28a>
1c0121f4:	16060b63          	beqz	a2,1c01236a <__adddf3+0x794>
1c0121f8:	410508b3          	sub	a7,a0,a6
1c0121fc:	01153333          	sltu	t1,a0,a7
1c012200:	40e78633          	sub	a2,a5,a4
1c012204:	40660633          	sub	a2,a2,t1
1c012208:	00861313          	slli	t1,a2,0x8
1c01220c:	00035c63          	bgez	t1,1c012224 <__adddf3+0x64e>
1c012210:	40a80533          	sub	a0,a6,a0
1c012214:	40f707b3          	sub	a5,a4,a5
1c012218:	00a83733          	sltu	a4,a6,a0
1c01221c:	40e78733          	sub	a4,a5,a4
1c012220:	85b6                	mv	a1,a3
1c012222:	b759                	j	1c0121a8 <__adddf3+0x5d2>
1c012224:	00c8e533          	or	a0,a7,a2
1c012228:	14051363          	bnez	a0,1c01236e <__adddf3+0x798>
1c01222c:	ffee8593          	addi	a1,t4,-2
1c012230:	0015b593          	seqz	a1,a1
1c012234:	4701                	li	a4,0
1c012236:	bf8d                	j	1c0121a8 <__adddf3+0x5d2>
1c012238:	7ff00293          	li	t0,2047
1c01223c:	025f1d63          	bne	t5,t0,1c012276 <__adddf3+0x6a0>
1c012240:	1c0f8463          	beqz	t6,1c012408 <__adddf3+0x832>
1c012244:	004008b7          	lui	a7,0x400
1c012248:	0117f8b3          	and	a7,a5,a7
1c01224c:	0018b893          	seqz	a7,a7
1c012250:	0892                	slli	a7,a7,0x4
1c012252:	03e31763          	bne	t1,t5,1c012280 <__adddf3+0x6aa>
1c012256:	c611                	beqz	a2,1c012262 <__adddf3+0x68c>
1c012258:	00971313          	slli	t1,a4,0x9
1c01225c:	00034363          	bltz	t1,1c012262 <__adddf3+0x68c>
1c012260:	48c1                	li	a7,16
1c012262:	000f9f63          	bnez	t6,1c012280 <__adddf3+0x6aa>
1c012266:	14060d63          	beqz	a2,1c0123c0 <__adddf3+0x7ea>
1c01226a:	87ba                	mv	a5,a4
1c01226c:	8542                	mv	a0,a6
1c01226e:	85b6                	mv	a1,a3
1c012270:	7ff00313          	li	t1,2047
1c012274:	b6f5                	j	1c011e60 <__adddf3+0x28a>
1c012276:	4881                	li	a7,0
1c012278:	fe5315e3          	bne	t1,t0,1c012262 <__adddf3+0x68c>
1c01227c:	4881                	li	a7,0
1c01227e:	bfe1                	j	1c012256 <__adddf3+0x680>
1c012280:	da65                	beqz	a2,1c012270 <__adddf3+0x69a>
1c012282:	b13d                	j	1c011eb0 <__adddf3+0x2da>
1c012284:	410508b3          	sub	a7,a0,a6
1c012288:	01153333          	sltu	t1,a0,a7
1c01228c:	40e78633          	sub	a2,a5,a4
1c012290:	40660633          	sub	a2,a2,t1
1c012294:	00861313          	slli	t1,a2,0x8
1c012298:	06035c63          	bgez	t1,1c012310 <__adddf3+0x73a>
1c01229c:	40a808b3          	sub	a7,a6,a0
1c0122a0:	40f707b3          	sub	a5,a4,a5
1c0122a4:	01183733          	sltu	a4,a6,a7
1c0122a8:	40e78633          	sub	a2,a5,a4
1c0122ac:	85b6                	mv	a1,a3
1c0122ae:	c62d                	beqz	a2,1c012318 <__adddf3+0x742>
1c0122b0:	100616b3          	p.fl1	a3,a2
1c0122b4:	47fd                	li	a5,31
1c0122b6:	40d786b3          	sub	a3,a5,a3
1c0122ba:	ff868313          	addi	t1,a3,-8
1c0122be:	47fd                	li	a5,31
1c0122c0:	0667c463          	blt	a5,t1,1c012328 <__adddf3+0x752>
1c0122c4:	02000793          	li	a5,32
1c0122c8:	406787b3          	sub	a5,a5,t1
1c0122cc:	00661633          	sll	a2,a2,t1
1c0122d0:	00f8d733          	srl	a4,a7,a5
1c0122d4:	00c767b3          	or	a5,a4,a2
1c0122d8:	00689833          	sll	a6,a7,t1
1c0122dc:	09e34063          	blt	t1,t5,1c01235c <__adddf3+0x786>
1c0122e0:	41e30333          	sub	t1,t1,t5
1c0122e4:	00130713          	addi	a4,t1,1
1c0122e8:	46fd                	li	a3,31
1c0122ea:	04e6c563          	blt	a3,a4,1c012334 <__adddf3+0x75e>
1c0122ee:	02000693          	li	a3,32
1c0122f2:	8e99                	sub	a3,a3,a4
1c0122f4:	00e85633          	srl	a2,a6,a4
1c0122f8:	00d79533          	sll	a0,a5,a3
1c0122fc:	00d81833          	sll	a6,a6,a3
1c012300:	8d51                	or	a0,a0,a2
1c012302:	01003833          	snez	a6,a6
1c012306:	01056533          	or	a0,a0,a6
1c01230a:	00e7d733          	srl	a4,a5,a4
1c01230e:	bd69                	j	1c0121a8 <__adddf3+0x5d2>
1c012310:	00c8e533          	or	a0,a7,a2
1c012314:	fd49                	bnez	a0,1c0122ae <__adddf3+0x6d8>
1c012316:	bf19                	j	1c01222c <__adddf3+0x656>
1c012318:	100896b3          	p.fl1	a3,a7
1c01231c:	47fd                	li	a5,31
1c01231e:	40d786b3          	sub	a3,a5,a3
1c012322:	02068693          	addi	a3,a3,32
1c012326:	bf51                	j	1c0122ba <__adddf3+0x6e4>
1c012328:	fd868793          	addi	a5,a3,-40
1c01232c:	00f897b3          	sll	a5,a7,a5
1c012330:	4801                	li	a6,0
1c012332:	b76d                	j	1c0122dc <__adddf3+0x706>
1c012334:	1305                	addi	t1,t1,-31
1c012336:	02000693          	li	a3,32
1c01233a:	0067d333          	srl	t1,a5,t1
1c01233e:	4601                	li	a2,0
1c012340:	00d70763          	beq	a4,a3,1c01234e <__adddf3+0x778>
1c012344:	04000613          	li	a2,64
1c012348:	8e19                	sub	a2,a2,a4
1c01234a:	00c79633          	sll	a2,a5,a2
1c01234e:	00c86533          	or	a0,a6,a2
1c012352:	00a03533          	snez	a0,a0
1c012356:	00a36533          	or	a0,t1,a0
1c01235a:	bde9                	j	1c012234 <__adddf3+0x65e>
1c01235c:	406f0333          	sub	t1,t5,t1
1c012360:	c177b733          	p.bclr	a4,a5,0,23
1c012364:	b2e5                	j	1c011d4c <__adddf3+0x176>
1c012366:	8542                	mv	a0,a6
1c012368:	bd65                	j	1c012220 <__adddf3+0x64a>
1c01236a:	873e                	mv	a4,a5
1c01236c:	bd35                	j	1c0121a8 <__adddf3+0x5d2>
1c01236e:	8732                	mv	a4,a2
1c012370:	8546                	mv	a0,a7
1c012372:	bd1d                	j	1c0121a8 <__adddf3+0x5d2>
1c012374:	4885                	li	a7,1
1c012376:	e60596e3          	bnez	a1,1c0121e2 <__adddf3+0x60c>
1c01237a:	ff87b793          	sltiu	a5,a5,-8
1c01237e:	0017c793          	xori	a5,a5,1
1c012382:	9e3e                	add	t3,t3,a5
1c012384:	bdb9                	j	1c0121e2 <__adddf3+0x60c>
1c012386:	4885                	li	a7,1
1c012388:	e4058de3          	beqz	a1,1c0121e2 <__adddf3+0x60c>
1c01238c:	b7fd                	j	1c01237a <__adddf3+0x7a4>
1c01238e:	87ba                	mv	a5,a4
1c012390:	8542                	mv	a0,a6
1c012392:	bdf9                	j	1c012270 <__adddf3+0x69a>
1c012394:	7ff00313          	li	t1,2047
1c012398:	4e01                	li	t3,0
1c01239a:	b965                	j	1c012052 <__adddf3+0x47c>
1c01239c:	4781                	li	a5,0
1c01239e:	4501                	li	a0,0
1c0123a0:	7ff00313          	li	t1,2047
1c0123a4:	4881                	li	a7,0
1c0123a6:	be19                	j	1c011ebc <__adddf3+0x2e6>
1c0123a8:	4781                	li	a5,0
1c0123aa:	bfed                	j	1c0123a4 <__adddf3+0x7ce>
1c0123ac:	87ba                	mv	a5,a4
1c0123ae:	8542                	mv	a0,a6
1c0123b0:	b7e5                	j	1c012398 <__adddf3+0x7c2>
1c0123b2:	4781                	li	a5,0
1c0123b4:	85b6                	mv	a1,a3
1c0123b6:	b7fd                	j	1c0123a4 <__adddf3+0x7ce>
1c0123b8:	87ba                	mv	a5,a4
1c0123ba:	8542                	mv	a0,a6
1c0123bc:	85b6                	mv	a1,a3
1c0123be:	bfe9                	j	1c012398 <__adddf3+0x7c2>
1c0123c0:	4501                	li	a0,0
1c0123c2:	004007b7          	lui	a5,0x400
1c0123c6:	7ff00313          	li	t1,2047
1c0123ca:	4581                	li	a1,0
1c0123cc:	48c1                	li	a7,16
1c0123ce:	b4fd                	j	1c011ebc <__adddf3+0x2e6>
1c0123d0:	4501                	li	a0,0
1c0123d2:	4301                	li	t1,0
1c0123d4:	bfc1                	j	1c0123a4 <__adddf3+0x7ce>
1c0123d6:	a80598e3          	bnez	a1,1c011e66 <__adddf3+0x290>
1c0123da:	00850713          	addi	a4,a0,8
1c0123de:	b679                	j	1c011f6c <__adddf3+0x396>
1c0123e0:	a80583e3          	beqz	a1,1c011e66 <__adddf3+0x290>
1c0123e4:	bfdd                	j	1c0123da <__adddf3+0x804>
1c0123e6:	4501                	li	a0,0
1c0123e8:	000e8c63          	beqz	t4,1c012400 <__adddf3+0x82a>
1c0123ec:	003eb763          	p.bneimm	t4,3,1c0123fa <__adddf3+0x824>
1c0123f0:	c981                	beqz	a1,1c012400 <__adddf3+0x82a>
1c0123f2:	557d                	li	a0,-1
1c0123f4:	7fe00313          	li	t1,2046
1c0123f8:	a021                	j	1c012400 <__adddf3+0x82a>
1c0123fa:	fe2ebce3          	p.bneimm	t4,2,1c0123f2 <__adddf3+0x81c>
1c0123fe:	d9f5                	beqz	a1,1c0123f2 <__adddf3+0x81c>
1c012400:	0058e893          	ori	a7,a7,5
1c012404:	87aa                	mv	a5,a0
1c012406:	b4f1                	j	1c011ed2 <__adddf3+0x2fc>
1c012408:	4881                	li	a7,0
1c01240a:	e5e31ee3          	bne	t1,t5,1c012266 <__adddf3+0x690>
1c01240e:	b5bd                	j	1c01227c <__adddf3+0x6a6>
1c012410:	b1e302e3          	beq	t1,t5,1c011f14 <__adddf3+0x33e>
1c012414:	87ba                	mv	a5,a4
1c012416:	8542                	mv	a0,a6
1c012418:	7ff00313          	li	t1,2047
1c01241c:	b015                	j	1c011c40 <__adddf3+0x6a>
1c01241e:	8082                	ret

1c012420 <__divdf3>:
1c012420:	1141                	addi	sp,sp,-16
1c012422:	c622                	sw	s0,12(sp)
1c012424:	c426                	sw	s1,8(sp)
1c012426:	c24a                	sw	s2,4(sp)
1c012428:	c04e                	sw	s3,0(sp)
1c01242a:	00202373          	frrm	t1
1c01242e:	d5459e33          	p.extractu	t3,a1,10,20
1c012432:	e6059fb3          	p.extractu	t6,a1,19,0
1c012436:	81fd                	srli	a1,a1,0x1f
1c012438:	020e0463          	beqz	t3,1c012460 <__divdf3+0x40>
1c01243c:	7ff00793          	li	a5,2047
1c012440:	06fe0f63          	beq	t3,a5,1c0124be <__divdf3+0x9e>
1c012444:	0f8e                	slli	t6,t6,0x3
1c012446:	c17fcfb3          	p.bset	t6,t6,0,23
1c01244a:	01d55793          	srli	a5,a0,0x1d
1c01244e:	01f7efb3          	or	t6,a5,t6
1c012452:	00351813          	slli	a6,a0,0x3
1c012456:	c01e0e93          	addi	t4,t3,-1023
1c01245a:	4281                	li	t0,0
1c01245c:	4881                	li	a7,0
1c01245e:	a8a5                	j	1c0124d6 <__divdf3+0xb6>
1c012460:	00afe7b3          	or	a5,t6,a0
1c012464:	c7e1                	beqz	a5,1c01252c <__divdf3+0x10c>
1c012466:	020f8e63          	beqz	t6,1c0124a2 <__divdf3+0x82>
1c01246a:	100f9733          	p.fl1	a4,t6
1c01246e:	47fd                	li	a5,31
1c012470:	40e78733          	sub	a4,a5,a4
1c012474:	ff570e13          	addi	t3,a4,-11 # fff5 <__heapfcram_size+0xdb5>
1c012478:	47f1                	li	a5,28
1c01247a:	03c7cc63          	blt	a5,t3,1c0124b2 <__divdf3+0x92>
1c01247e:	48f5                	li	a7,29
1c012480:	ff870813          	addi	a6,a4,-8
1c012484:	41c888b3          	sub	a7,a7,t3
1c012488:	010f9fb3          	sll	t6,t6,a6
1c01248c:	011558b3          	srl	a7,a0,a7
1c012490:	01f8efb3          	or	t6,a7,t6
1c012494:	01051833          	sll	a6,a0,a6
1c012498:	c0d00e13          	li	t3,-1011
1c01249c:	40ee0eb3          	sub	t4,t3,a4
1c0124a0:	bf6d                	j	1c01245a <__divdf3+0x3a>
1c0124a2:	10051eb3          	p.fl1	t4,a0
1c0124a6:	47fd                	li	a5,31
1c0124a8:	41d78eb3          	sub	t4,a5,t4
1c0124ac:	020e8713          	addi	a4,t4,32
1c0124b0:	b7d1                	j	1c012474 <__divdf3+0x54>
1c0124b2:	fd870f93          	addi	t6,a4,-40
1c0124b6:	01f51fb3          	sll	t6,a0,t6
1c0124ba:	4801                	li	a6,0
1c0124bc:	bff1                	j	1c012498 <__divdf3+0x78>
1c0124be:	882a                	mv	a6,a0
1c0124c0:	00afe533          	or	a0,t6,a0
1c0124c4:	c92d                	beqz	a0,1c012536 <__divdf3+0x116>
1c0124c6:	00cf9793          	slli	a5,t6,0xc
1c0124ca:	7ff00e93          	li	t4,2047
1c0124ce:	428d                	li	t0,3
1c0124d0:	48c1                	li	a7,16
1c0124d2:	f807c5e3          	bltz	a5,1c01245c <__divdf3+0x3c>
1c0124d6:	d5469e33          	p.extractu	t3,a3,10,20
1c0124da:	e6069733          	p.extractu	a4,a3,19,0
1c0124de:	87b2                	mv	a5,a2
1c0124e0:	82fd                	srli	a3,a3,0x1f
1c0124e2:	060e0063          	beqz	t3,1c012542 <__divdf3+0x122>
1c0124e6:	7ff00513          	li	a0,2047
1c0124ea:	0aae0963          	beq	t3,a0,1c01259c <__divdf3+0x17c>
1c0124ee:	070e                	slli	a4,a4,0x3
1c0124f0:	01d65793          	srli	a5,a2,0x1d
1c0124f4:	c1774733          	p.bset	a4,a4,0,23
1c0124f8:	8f5d                	or	a4,a4,a5
1c0124fa:	00361793          	slli	a5,a2,0x3
1c0124fe:	c01e0613          	addi	a2,t3,-1023
1c012502:	4f01                	li	t5,0
1c012504:	40ce8e33          	sub	t3,t4,a2
1c012508:	00229613          	slli	a2,t0,0x2
1c01250c:	01e66633          	or	a2,a2,t5
1c012510:	167d                	addi	a2,a2,-1
1c012512:	4eb9                	li	t4,14
1c012514:	00d5c533          	xor	a0,a1,a3
1c012518:	0aceec63          	bltu	t4,a2,1c0125d0 <__divdf3+0x1b0>
1c01251c:	1c01ceb7          	lui	t4,0x1c01c
1c012520:	060a                	slli	a2,a2,0x2
1c012522:	9d0e8e93          	addi	t4,t4,-1584 # 1c01b9d0 <pi_cl_pe_other_entry+0x10>
1c012526:	20cef603          	p.lw	a2,a2(t4)
1c01252a:	8602                	jr	a2
1c01252c:	4f81                	li	t6,0
1c01252e:	4801                	li	a6,0
1c012530:	4e81                	li	t4,0
1c012532:	4285                	li	t0,1
1c012534:	b725                	j	1c01245c <__divdf3+0x3c>
1c012536:	4f81                	li	t6,0
1c012538:	4801                	li	a6,0
1c01253a:	7ff00e93          	li	t4,2047
1c01253e:	4289                	li	t0,2
1c012540:	bf31                	j	1c01245c <__divdf3+0x3c>
1c012542:	00c767b3          	or	a5,a4,a2
1c012546:	c7b5                	beqz	a5,1c0125b2 <__divdf3+0x192>
1c012548:	cf05                	beqz	a4,1c012580 <__divdf3+0x160>
1c01254a:	10071e33          	p.fl1	t3,a4
1c01254e:	47fd                	li	a5,31
1c012550:	41c78e33          	sub	t3,a5,t3
1c012554:	ff5e0f13          	addi	t5,t3,-11
1c012558:	47f1                	li	a5,28
1c01255a:	03e7cb63          	blt	a5,t5,1c012590 <__divdf3+0x170>
1c01255e:	4575                	li	a0,29
1c012560:	ff8e0793          	addi	a5,t3,-8
1c012564:	41e50533          	sub	a0,a0,t5
1c012568:	00f71733          	sll	a4,a4,a5
1c01256c:	00a65533          	srl	a0,a2,a0
1c012570:	8f49                	or	a4,a4,a0
1c012572:	00f617b3          	sll	a5,a2,a5
1c012576:	c0d00613          	li	a2,-1011
1c01257a:	41c60633          	sub	a2,a2,t3
1c01257e:	b751                	j	1c012502 <__divdf3+0xe2>
1c012580:	10061e33          	p.fl1	t3,a2
1c012584:	47fd                	li	a5,31
1c012586:	41c78e33          	sub	t3,a5,t3
1c01258a:	020e0e13          	addi	t3,t3,32
1c01258e:	b7d9                	j	1c012554 <__divdf3+0x134>
1c012590:	fd8e0713          	addi	a4,t3,-40
1c012594:	00e61733          	sll	a4,a2,a4
1c012598:	4781                	li	a5,0
1c01259a:	bff1                	j	1c012576 <__divdf3+0x156>
1c01259c:	8e59                	or	a2,a2,a4
1c01259e:	ce19                	beqz	a2,1c0125bc <__divdf3+0x19c>
1c0125a0:	00c71613          	slli	a2,a4,0xc
1c0125a4:	02064263          	bltz	a2,1c0125c8 <__divdf3+0x1a8>
1c0125a8:	7ff00613          	li	a2,2047
1c0125ac:	4f0d                	li	t5,3
1c0125ae:	48c1                	li	a7,16
1c0125b0:	bf91                	j	1c012504 <__divdf3+0xe4>
1c0125b2:	4701                	li	a4,0
1c0125b4:	4781                	li	a5,0
1c0125b6:	4601                	li	a2,0
1c0125b8:	4f05                	li	t5,1
1c0125ba:	b7a9                	j	1c012504 <__divdf3+0xe4>
1c0125bc:	4701                	li	a4,0
1c0125be:	4781                	li	a5,0
1c0125c0:	7ff00613          	li	a2,2047
1c0125c4:	4f09                	li	t5,2
1c0125c6:	bf3d                	j	1c012504 <__divdf3+0xe4>
1c0125c8:	7ff00613          	li	a2,2047
1c0125cc:	4f0d                	li	t5,3
1c0125ce:	bf1d                	j	1c012504 <__divdf3+0xe4>
1c0125d0:	01f76663          	bltu	a4,t6,1c0125dc <__divdf3+0x1bc>
1c0125d4:	24ef9963          	bne	t6,a4,1c012826 <__divdf3+0x406>
1c0125d8:	24f86763          	bltu	a6,a5,1c012826 <__divdf3+0x406>
1c0125dc:	01ff9613          	slli	a2,t6,0x1f
1c0125e0:	00185693          	srli	a3,a6,0x1
1c0125e4:	01f81393          	slli	t2,a6,0x1f
1c0125e8:	001fdf93          	srli	t6,t6,0x1
1c0125ec:	00d66833          	or	a6,a2,a3
1c0125f0:	0722                	slli	a4,a4,0x8
1c0125f2:	0187d693          	srli	a3,a5,0x18
1c0125f6:	8ed9                	or	a3,a3,a4
1c0125f8:	0106df13          	srli	t5,a3,0x10
1c0125fc:	03efd2b3          	divu	t0,t6,t5
1c012600:	1006deb3          	p.exthz	t4,a3
1c012604:	00879613          	slli	a2,a5,0x8
1c012608:	01085793          	srli	a5,a6,0x10
1c01260c:	03efffb3          	remu	t6,t6,t5
1c012610:	8716                	mv	a4,t0
1c012612:	025e85b3          	mul	a1,t4,t0
1c012616:	0fc2                	slli	t6,t6,0x10
1c012618:	01f7e7b3          	or	a5,a5,t6
1c01261c:	00b7fc63          	bleu	a1,a5,1c012634 <__divdf3+0x214>
1c012620:	97b6                	add	a5,a5,a3
1c012622:	fff28713          	addi	a4,t0,-1
1c012626:	00d7e763          	bltu	a5,a3,1c012634 <__divdf3+0x214>
1c01262a:	00b7f563          	bleu	a1,a5,1c012634 <__divdf3+0x214>
1c01262e:	ffe28713          	addi	a4,t0,-2
1c012632:	97b6                	add	a5,a5,a3
1c012634:	8f8d                	sub	a5,a5,a1
1c012636:	03e7d2b3          	divu	t0,a5,t5
1c01263a:	03e7f7b3          	remu	a5,a5,t5
1c01263e:	8596                	mv	a1,t0
1c012640:	025e8fb3          	mul	t6,t4,t0
1c012644:	df07a833          	p.insert	a6,a5,15,16
1c012648:	01f87c63          	bleu	t6,a6,1c012660 <__divdf3+0x240>
1c01264c:	9836                	add	a6,a6,a3
1c01264e:	fff28593          	addi	a1,t0,-1
1c012652:	00d86763          	bltu	a6,a3,1c012660 <__divdf3+0x240>
1c012656:	01f87563          	bleu	t6,a6,1c012660 <__divdf3+0x240>
1c01265a:	ffe28593          	addi	a1,t0,-2
1c01265e:	9836                	add	a6,a6,a3
1c012660:	01071793          	slli	a5,a4,0x10
1c012664:	64c1                	lui	s1,0x10
1c012666:	8fcd                	or	a5,a5,a1
1c012668:	41f80833          	sub	a6,a6,t6
1c01266c:	fff48f93          	addi	t6,s1,-1 # ffff <__heapfcram_size+0xdbf>
1c012670:	01f7f733          	and	a4,a5,t6
1c012674:	0107d413          	srli	s0,a5,0x10
1c012678:	01f67fb3          	and	t6,a2,t6
1c01267c:	03f405b3          	mul	a1,s0,t6
1c012680:	01065293          	srli	t0,a2,0x10
1c012684:	03f70933          	mul	s2,a4,t6
1c012688:	89ae                	mv	s3,a1
1c01268a:	42e289b3          	p.mac	s3,t0,a4
1c01268e:	01095713          	srli	a4,s2,0x10
1c012692:	974e                	add	a4,a4,s3
1c012694:	02540433          	mul	s0,s0,t0
1c012698:	00b77363          	bleu	a1,a4,1c01269e <__divdf3+0x27e>
1c01269c:	9426                	add	s0,s0,s1
1c01269e:	64c1                	lui	s1,0x10
1c0126a0:	01075593          	srli	a1,a4,0x10
1c0126a4:	14fd                	addi	s1,s1,-1
1c0126a6:	942e                	add	s0,s0,a1
1c0126a8:	009775b3          	and	a1,a4,s1
1c0126ac:	05c2                	slli	a1,a1,0x10
1c0126ae:	00997733          	and	a4,s2,s1
1c0126b2:	95ba                	add	a1,a1,a4
1c0126b4:	00886763          	bltu	a6,s0,1c0126c2 <__divdf3+0x2a2>
1c0126b8:	873e                	mv	a4,a5
1c0126ba:	02881e63          	bne	a6,s0,1c0126f6 <__divdf3+0x2d6>
1c0126be:	02b3fc63          	bleu	a1,t2,1c0126f6 <__divdf3+0x2d6>
1c0126c2:	93b2                	add	t2,t2,a2
1c0126c4:	00c3b4b3          	sltu	s1,t2,a2
1c0126c8:	94b6                	add	s1,s1,a3
1c0126ca:	9826                	add	a6,a6,s1
1c0126cc:	fff78713          	addi	a4,a5,-1 # 3fffff <__l1_heapsram_size+0x3e006f>
1c0126d0:	0106e663          	bltu	a3,a6,1c0126dc <__divdf3+0x2bc>
1c0126d4:	03069163          	bne	a3,a6,1c0126f6 <__divdf3+0x2d6>
1c0126d8:	00c3ef63          	bltu	t2,a2,1c0126f6 <__divdf3+0x2d6>
1c0126dc:	00886663          	bltu	a6,s0,1c0126e8 <__divdf3+0x2c8>
1c0126e0:	01041b63          	bne	s0,a6,1c0126f6 <__divdf3+0x2d6>
1c0126e4:	00b3f963          	bleu	a1,t2,1c0126f6 <__divdf3+0x2d6>
1c0126e8:	93b2                	add	t2,t2,a2
1c0126ea:	ffe78713          	addi	a4,a5,-2
1c0126ee:	00c3b7b3          	sltu	a5,t2,a2
1c0126f2:	97b6                	add	a5,a5,a3
1c0126f4:	983e                	add	a6,a6,a5
1c0126f6:	40b385b3          	sub	a1,t2,a1
1c0126fa:	00b3b7b3          	sltu	a5,t2,a1
1c0126fe:	40880833          	sub	a6,a6,s0
1c012702:	40f80833          	sub	a6,a6,a5
1c012706:	57fd                	li	a5,-1
1c012708:	0f068563          	beq	a3,a6,1c0127f2 <__divdf3+0x3d2>
1c01270c:	03e85433          	divu	s0,a6,t5
1c012710:	0105d793          	srli	a5,a1,0x10
1c012714:	03e87833          	remu	a6,a6,t5
1c012718:	028e83b3          	mul	t2,t4,s0
1c01271c:	0842                	slli	a6,a6,0x10
1c01271e:	0107e833          	or	a6,a5,a6
1c012722:	87a2                	mv	a5,s0
1c012724:	00787c63          	bleu	t2,a6,1c01273c <__divdf3+0x31c>
1c012728:	9836                	add	a6,a6,a3
1c01272a:	fff40793          	addi	a5,s0,-1
1c01272e:	00d86763          	bltu	a6,a3,1c01273c <__divdf3+0x31c>
1c012732:	00787563          	bleu	t2,a6,1c01273c <__divdf3+0x31c>
1c012736:	ffe40793          	addi	a5,s0,-2
1c01273a:	9836                	add	a6,a6,a3
1c01273c:	40780833          	sub	a6,a6,t2
1c012740:	03e853b3          	divu	t2,a6,t5
1c012744:	03e87833          	remu	a6,a6,t5
1c012748:	027e8eb3          	mul	t4,t4,t2
1c01274c:	df0825b3          	p.insert	a1,a6,15,16
1c012750:	881e                	mv	a6,t2
1c012752:	01d5fc63          	bleu	t4,a1,1c01276a <__divdf3+0x34a>
1c012756:	95b6                	add	a1,a1,a3
1c012758:	fff38813          	addi	a6,t2,-1
1c01275c:	00d5e763          	bltu	a1,a3,1c01276a <__divdf3+0x34a>
1c012760:	01d5f563          	bleu	t4,a1,1c01276a <__divdf3+0x34a>
1c012764:	ffe38813          	addi	a6,t2,-2
1c012768:	95b6                	add	a1,a1,a3
1c01276a:	07c2                	slli	a5,a5,0x10
1c01276c:	0107e833          	or	a6,a5,a6
1c012770:	100857b3          	p.exthz	a5,a6
1c012774:	01085f13          	srli	t5,a6,0x10
1c012778:	02ff8433          	mul	s0,t6,a5
1c01277c:	41d58eb3          	sub	t4,a1,t4
1c012780:	03ff0fb3          	mul	t6,t5,t6
1c012784:	85fe                	mv	a1,t6
1c012786:	42f285b3          	p.mac	a1,t0,a5
1c01278a:	01045793          	srli	a5,s0,0x10
1c01278e:	97ae                	add	a5,a5,a1
1c012790:	03e283b3          	mul	t2,t0,t5
1c012794:	01f7f463          	bleu	t6,a5,1c01279c <__divdf3+0x37c>
1c012798:	65c1                	lui	a1,0x10
1c01279a:	93ae                	add	t2,t2,a1
1c01279c:	6fc1                	lui	t6,0x10
1c01279e:	1ffd                	addi	t6,t6,-1
1c0127a0:	0107df13          	srli	t5,a5,0x10
1c0127a4:	01f7f5b3          	and	a1,a5,t6
1c0127a8:	05c2                	slli	a1,a1,0x10
1c0127aa:	01f47433          	and	s0,s0,t6
1c0127ae:	9f1e                	add	t5,t5,t2
1c0127b0:	95a2                	add	a1,a1,s0
1c0127b2:	01eee663          	bltu	t4,t5,1c0127be <__divdf3+0x39e>
1c0127b6:	87c2                	mv	a5,a6
1c0127b8:	03ee9b63          	bne	t4,t5,1c0127ee <__divdf3+0x3ce>
1c0127bc:	c99d                	beqz	a1,1c0127f2 <__divdf3+0x3d2>
1c0127be:	9eb6                	add	t4,t4,a3
1c0127c0:	fff80793          	addi	a5,a6,-1
1c0127c4:	02dee163          	bltu	t4,a3,1c0127e6 <__divdf3+0x3c6>
1c0127c8:	01eee663          	bltu	t4,t5,1c0127d4 <__divdf3+0x3b4>
1c0127cc:	03ee9163          	bne	t4,t5,1c0127ee <__divdf3+0x3ce>
1c0127d0:	00b67d63          	bleu	a1,a2,1c0127ea <__divdf3+0x3ca>
1c0127d4:	ffe80793          	addi	a5,a6,-2
1c0127d8:	00161813          	slli	a6,a2,0x1
1c0127dc:	00c83633          	sltu	a2,a6,a2
1c0127e0:	96b2                	add	a3,a3,a2
1c0127e2:	9eb6                	add	t4,t4,a3
1c0127e4:	8642                	mv	a2,a6
1c0127e6:	01ee9463          	bne	t4,t5,1c0127ee <__divdf3+0x3ce>
1c0127ea:	00c58463          	beq	a1,a2,1c0127f2 <__divdf3+0x3d2>
1c0127ee:	c007c7b3          	p.bset	a5,a5,0,0
1c0127f2:	3ffe0593          	addi	a1,t3,1023
1c0127f6:	0cb05f63          	blez	a1,1c0128d4 <__divdf3+0x4b4>
1c0127fa:	f837b6b3          	p.bclr	a3,a5,28,3
1c0127fe:	c6c1                	beqz	a3,1c012886 <__divdf3+0x466>
1c012800:	c008c8b3          	p.bset	a7,a7,0,0
1c012804:	08232063          	p.beqimm	t1,2,1c012884 <__divdf3+0x464>
1c012808:	06332a63          	p.beqimm	t1,3,1c01287c <__divdf3+0x45c>
1c01280c:	06031d63          	bnez	t1,1c012886 <__divdf3+0x466>
1c012810:	f647b6b3          	p.bclr	a3,a5,27,4
1c012814:	0646a963          	p.beqimm	a3,4,1c012886 <__divdf3+0x466>
1c012818:	00478693          	addi	a3,a5,4
1c01281c:	00f6b7b3          	sltu	a5,a3,a5
1c012820:	973e                	add	a4,a4,a5
1c012822:	87b6                	mv	a5,a3
1c012824:	a08d                	j	1c012886 <__divdf3+0x466>
1c012826:	1e7d                	addi	t3,t3,-1
1c012828:	4381                	li	t2,0
1c01282a:	b3d9                	j	1c0125f0 <__divdf3+0x1d0>
1c01282c:	852e                	mv	a0,a1
1c01282e:	877e                	mv	a4,t6
1c012830:	87c2                	mv	a5,a6
1c012832:	8f16                	mv	t5,t0
1c012834:	002f2d63          	p.beqimm	t5,2,1c01284e <__divdf3+0x42e>
1c012838:	1c3f2763          	p.beqimm	t5,3,1c012a06 <__divdf3+0x5e6>
1c01283c:	fa1f3be3          	p.bneimm	t5,1,1c0127f2 <__divdf3+0x3d2>
1c012840:	4701                	li	a4,0
1c012842:	4781                	li	a5,0
1c012844:	aa45                	j	1c0129f4 <__divdf3+0x5d4>
1c012846:	8536                	mv	a0,a3
1c012848:	b7f5                	j	1c012834 <__divdf3+0x414>
1c01284a:	c038c8b3          	p.bset	a7,a7,0,3
1c01284e:	4701                	li	a4,0
1c012850:	4781                	li	a5,0
1c012852:	7ff00593          	li	a1,2047
1c012856:	4681                	li	a3,0
1c012858:	e60726b3          	p.insert	a3,a4,19,0
1c01285c:	d545a6b3          	p.insert	a3,a1,10,20
1c012860:	c1f526b3          	p.insert	a3,a0,0,31
1c012864:	85b6                	mv	a1,a3
1c012866:	853e                	mv	a0,a5
1c012868:	00088463          	beqz	a7,1c012870 <__divdf3+0x450>
1c01286c:	0018a073          	csrs	0x1,a7
1c012870:	4432                	lw	s0,12(sp)
1c012872:	44a2                	lw	s1,8(sp)
1c012874:	4912                	lw	s2,4(sp)
1c012876:	4982                	lw	s3,0(sp)
1c012878:	0141                	addi	sp,sp,16
1c01287a:	8082                	ret
1c01287c:	e509                	bnez	a0,1c012886 <__divdf3+0x466>
1c01287e:	00878693          	addi	a3,a5,8
1c012882:	bf69                	j	1c01281c <__divdf3+0x3fc>
1c012884:	fd6d                	bnez	a0,1c01287e <__divdf3+0x45e>
1c012886:	00771693          	slli	a3,a4,0x7
1c01288a:	0006d663          	bgez	a3,1c012896 <__divdf3+0x476>
1c01288e:	c1873733          	p.bclr	a4,a4,0,24
1c012892:	400e0593          	addi	a1,t3,1024
1c012896:	7fe00693          	li	a3,2046
1c01289a:	00b6c963          	blt	a3,a1,1c0128ac <__divdf3+0x48c>
1c01289e:	0037d693          	srli	a3,a5,0x3
1c0128a2:	01d71793          	slli	a5,a4,0x1d
1c0128a6:	8fd5                	or	a5,a5,a3
1c0128a8:	830d                	srli	a4,a4,0x3
1c0128aa:	b775                	j	1c012856 <__divdf3+0x436>
1c0128ac:	02232263          	p.beqimm	t1,2,1c0128d0 <__divdf3+0x4b0>
1c0128b0:	00332b63          	p.beqimm	t1,3,1c0128c6 <__divdf3+0x4a6>
1c0128b4:	00030a63          	beqz	t1,1c0128c8 <__divdf3+0x4a8>
1c0128b8:	57fd                	li	a5,-1
1c0128ba:	7fe00593          	li	a1,2046
1c0128be:	0058e893          	ori	a7,a7,5
1c0128c2:	873e                	mv	a4,a5
1c0128c4:	bf49                	j	1c012856 <__divdf3+0x436>
1c0128c6:	f96d                	bnez	a0,1c0128b8 <__divdf3+0x498>
1c0128c8:	4781                	li	a5,0
1c0128ca:	7ff00593          	li	a1,2047
1c0128ce:	bfc5                	j	1c0128be <__divdf3+0x49e>
1c0128d0:	fd65                	bnez	a0,1c0128c8 <__divdf3+0x4a8>
1c0128d2:	b7dd                	j	1c0128b8 <__divdf3+0x498>
1c0128d4:	4805                	li	a6,1
1c0128d6:	e1a9                	bnez	a1,1c012918 <__divdf3+0x4f8>
1c0128d8:	f837b633          	p.bclr	a2,a5,28,3
1c0128dc:	86ba                	mv	a3,a4
1c0128de:	ca05                	beqz	a2,1c01290e <__divdf3+0x4ee>
1c0128e0:	c008c8b3          	p.bset	a7,a7,0,0
1c0128e4:	02232463          	p.beqimm	t1,2,1c01290c <__divdf3+0x4ec>
1c0128e8:	00332e63          	p.beqimm	t1,3,1c012904 <__divdf3+0x4e4>
1c0128ec:	02031163          	bnez	t1,1c01290e <__divdf3+0x4ee>
1c0128f0:	f647b633          	p.bclr	a2,a5,27,4
1c0128f4:	00462d63          	p.beqimm	a2,4,1c01290e <__divdf3+0x4ee>
1c0128f8:	ffc7b693          	sltiu	a3,a5,-4
1c0128fc:	0016c693          	xori	a3,a3,1
1c012900:	96ba                	add	a3,a3,a4
1c012902:	a031                	j	1c01290e <__divdf3+0x4ee>
1c012904:	e509                	bnez	a0,1c01290e <__divdf3+0x4ee>
1c012906:	ff87b693          	sltiu	a3,a5,-8
1c01290a:	bfcd                	j	1c0128fc <__divdf3+0x4dc>
1c01290c:	fd6d                	bnez	a0,1c012906 <__divdf3+0x4e6>
1c01290e:	82e1                	srli	a3,a3,0x18
1c012910:	0016c693          	xori	a3,a3,1
1c012914:	fc16b833          	p.bclr	a6,a3,30,1
1c012918:	4685                	li	a3,1
1c01291a:	8e8d                	sub	a3,a3,a1
1c01291c:	03800613          	li	a2,56
1c012920:	0ad64b63          	blt	a2,a3,1c0129d6 <__divdf3+0x5b6>
1c012924:	467d                	li	a2,31
1c012926:	04d64663          	blt	a2,a3,1c012972 <__divdf3+0x552>
1c01292a:	41ee0e13          	addi	t3,t3,1054
1c01292e:	01c71633          	sll	a2,a4,t3
1c012932:	00d7d5b3          	srl	a1,a5,a3
1c012936:	01c797b3          	sll	a5,a5,t3
1c01293a:	8e4d                	or	a2,a2,a1
1c01293c:	00f037b3          	snez	a5,a5
1c012940:	8fd1                	or	a5,a5,a2
1c012942:	00d75733          	srl	a4,a4,a3
1c012946:	f837b6b3          	p.bclr	a3,a5,28,3
1c01294a:	cea1                	beqz	a3,1c0129a2 <__divdf3+0x582>
1c01294c:	c008c8b3          	p.bset	a7,a7,0,0
1c012950:	04232863          	p.beqimm	t1,2,1c0129a0 <__divdf3+0x580>
1c012954:	04332263          	p.beqimm	t1,3,1c012998 <__divdf3+0x578>
1c012958:	04031563          	bnez	t1,1c0129a2 <__divdf3+0x582>
1c01295c:	f647b6b3          	p.bclr	a3,a5,27,4
1c012960:	0446a163          	p.beqimm	a3,4,1c0129a2 <__divdf3+0x582>
1c012964:	00478693          	addi	a3,a5,4
1c012968:	00f6b7b3          	sltu	a5,a3,a5
1c01296c:	973e                	add	a4,a4,a5
1c01296e:	87b6                	mv	a5,a3
1c012970:	a80d                	j	1c0129a2 <__divdf3+0x582>
1c012972:	5605                	li	a2,-31
1c012974:	8e0d                	sub	a2,a2,a1
1c012976:	02000e93          	li	t4,32
1c01297a:	00c75633          	srl	a2,a4,a2
1c01297e:	4581                	li	a1,0
1c012980:	01d68663          	beq	a3,t4,1c01298c <__divdf3+0x56c>
1c012984:	43ee0e13          	addi	t3,t3,1086
1c012988:	01c715b3          	sll	a1,a4,t3
1c01298c:	8fcd                	or	a5,a5,a1
1c01298e:	00f037b3          	snez	a5,a5
1c012992:	8fd1                	or	a5,a5,a2
1c012994:	4701                	li	a4,0
1c012996:	bf45                	j	1c012946 <__divdf3+0x526>
1c012998:	e509                	bnez	a0,1c0129a2 <__divdf3+0x582>
1c01299a:	00878693          	addi	a3,a5,8
1c01299e:	b7e9                	j	1c012968 <__divdf3+0x548>
1c0129a0:	fd6d                	bnez	a0,1c01299a <__divdf3+0x57a>
1c0129a2:	00871613          	slli	a2,a4,0x8
1c0129a6:	02065063          	bgez	a2,1c0129c6 <__divdf3+0x5a6>
1c0129aa:	c008c8b3          	p.bset	a7,a7,0,0
1c0129ae:	4701                	li	a4,0
1c0129b0:	4781                	li	a5,0
1c0129b2:	4585                	li	a1,1
1c0129b4:	ea0801e3          	beqz	a6,1c012856 <__divdf3+0x436>
1c0129b8:	fc18b6b3          	p.bclr	a3,a7,30,1
1c0129bc:	e8068de3          	beqz	a3,1c012856 <__divdf3+0x436>
1c0129c0:	c018c8b3          	p.bset	a7,a7,0,1
1c0129c4:	bd49                	j	1c012856 <__divdf3+0x436>
1c0129c6:	0037d693          	srli	a3,a5,0x3
1c0129ca:	01d71793          	slli	a5,a4,0x1d
1c0129ce:	8fd5                	or	a5,a5,a3
1c0129d0:	830d                	srli	a4,a4,0x3
1c0129d2:	4581                	li	a1,0
1c0129d4:	b7c5                	j	1c0129b4 <__divdf3+0x594>
1c0129d6:	8fd9                	or	a5,a5,a4
1c0129d8:	cb99                	beqz	a5,1c0129ee <__divdf3+0x5ce>
1c0129da:	c008c8b3          	p.bset	a7,a7,0,0
1c0129de:	02232163          	p.beqimm	t1,2,1c012a00 <__divdf3+0x5e0>
1c0129e2:	00332b63          	p.beqimm	t1,3,1c0129f8 <__divdf3+0x5d8>
1c0129e6:	4795                	li	a5,5
1c0129e8:	00031a63          	bnez	t1,1c0129fc <__divdf3+0x5dc>
1c0129ec:	838d                	srli	a5,a5,0x3
1c0129ee:	c018c8b3          	p.bset	a7,a7,0,1
1c0129f2:	4701                	li	a4,0
1c0129f4:	4581                	li	a1,0
1c0129f6:	b585                	j	1c012856 <__divdf3+0x436>
1c0129f8:	47a5                	li	a5,9
1c0129fa:	d96d                	beqz	a0,1c0129ec <__divdf3+0x5cc>
1c0129fc:	4785                	li	a5,1
1c0129fe:	b7fd                	j	1c0129ec <__divdf3+0x5cc>
1c012a00:	47a5                	li	a5,9
1c012a02:	f56d                	bnez	a0,1c0129ec <__divdf3+0x5cc>
1c012a04:	bfe5                	j	1c0129fc <__divdf3+0x5dc>
1c012a06:	00080737          	lui	a4,0x80
1c012a0a:	4781                	li	a5,0
1c012a0c:	7ff00593          	li	a1,2047
1c012a10:	4501                	li	a0,0
1c012a12:	b591                	j	1c012856 <__divdf3+0x436>
1c012a14:	00080737          	lui	a4,0x80
1c012a18:	4781                	li	a5,0
1c012a1a:	7ff00593          	li	a1,2047
1c012a1e:	4501                	li	a0,0
1c012a20:	48c1                	li	a7,16
1c012a22:	bd15                	j	1c012856 <__divdf3+0x436>

1c012a24 <__eqdf2>:
1c012a24:	002027f3          	frrm	a5
1c012a28:	7ff00313          	li	t1,2047
1c012a2c:	d5459833          	p.extractu	a6,a1,10,20
1c012a30:	e60597b3          	p.extractu	a5,a1,19,0
1c012a34:	e6069733          	p.extractu	a4,a3,19,0
1c012a38:	d54698b3          	p.extractu	a7,a3,10,20
1c012a3c:	8eaa                	mv	t4,a0
1c012a3e:	81fd                	srli	a1,a1,0x1f
1c012a40:	8e32                	mv	t3,a2
1c012a42:	82fd                	srli	a3,a3,0x1f
1c012a44:	00681963          	bne	a6,t1,1c012a56 <__eqdf2+0x32>
1c012a48:	00a7e333          	or	t1,a5,a0
1c012a4c:	00031f63          	bnez	t1,1c012a6a <__eqdf2+0x46>
1c012a50:	05089e63          	bne	a7,a6,1c012aac <__eqdf2+0x88>
1c012a54:	a019                	j	1c012a5a <__eqdf2+0x36>
1c012a56:	02689a63          	bne	a7,t1,1c012a8a <__eqdf2+0x66>
1c012a5a:	00c76333          	or	t1,a4,a2
1c012a5e:	02030663          	beqz	t1,1c012a8a <__eqdf2+0x66>
1c012a62:	7ff00693          	li	a3,2047
1c012a66:	00d81b63          	bne	a6,a3,1c012a7c <__eqdf2+0x58>
1c012a6a:	8d5d                	or	a0,a0,a5
1c012a6c:	c119                	beqz	a0,1c012a72 <__eqdf2+0x4e>
1c012a6e:	83cd                	srli	a5,a5,0x13
1c012a70:	cf85                	beqz	a5,1c012aa8 <__eqdf2+0x84>
1c012a72:	7ff00793          	li	a5,2047
1c012a76:	4505                	li	a0,1
1c012a78:	02f89b63          	bne	a7,a5,1c012aae <__eqdf2+0x8a>
1c012a7c:	00c767b3          	or	a5,a4,a2
1c012a80:	4505                	li	a0,1
1c012a82:	c795                	beqz	a5,1c012aae <__eqdf2+0x8a>
1c012a84:	834d                	srli	a4,a4,0x13
1c012a86:	c30d                	beqz	a4,1c012aa8 <__eqdf2+0x84>
1c012a88:	8082                	ret
1c012a8a:	4505                	li	a0,1
1c012a8c:	03181163          	bne	a6,a7,1c012aae <__eqdf2+0x8a>
1c012a90:	00e79f63          	bne	a5,a4,1c012aae <__eqdf2+0x8a>
1c012a94:	01de1d63          	bne	t3,t4,1c012aae <__eqdf2+0x8a>
1c012a98:	00d58c63          	beq	a1,a3,1c012ab0 <__eqdf2+0x8c>
1c012a9c:	00081963          	bnez	a6,1c012aae <__eqdf2+0x8a>
1c012aa0:	8fd1                	or	a5,a5,a2
1c012aa2:	00f03533          	snez	a0,a5
1c012aa6:	8082                	ret
1c012aa8:	00186073          	csrsi	0x1,16
1c012aac:	4505                	li	a0,1
1c012aae:	8082                	ret
1c012ab0:	4501                	li	a0,0
1c012ab2:	8082                	ret

1c012ab4 <__muldf3>:
1c012ab4:	1101                	addi	sp,sp,-32
1c012ab6:	ce22                	sw	s0,28(sp)
1c012ab8:	cc26                	sw	s1,24(sp)
1c012aba:	ca4a                	sw	s2,20(sp)
1c012abc:	c84e                	sw	s3,16(sp)
1c012abe:	c652                	sw	s4,12(sp)
1c012ac0:	c456                	sw	s5,8(sp)
1c012ac2:	c25a                	sw	s6,4(sp)
1c012ac4:	00202373          	frrm	t1
1c012ac8:	d5459e33          	p.extractu	t3,a1,10,20
1c012acc:	e60598b3          	p.extractu	a7,a1,19,0
1c012ad0:	81fd                	srli	a1,a1,0x1f
1c012ad2:	020e0463          	beqz	t3,1c012afa <__muldf3+0x46>
1c012ad6:	7ff00793          	li	a5,2047
1c012ada:	06fe0f63          	beq	t3,a5,1c012b58 <__muldf3+0xa4>
1c012ade:	088e                	slli	a7,a7,0x3
1c012ae0:	c178c8b3          	p.bset	a7,a7,0,23
1c012ae4:	01d55793          	srli	a5,a0,0x1d
1c012ae8:	0117e8b3          	or	a7,a5,a7
1c012aec:	00351f93          	slli	t6,a0,0x3
1c012af0:	c01e0e13          	addi	t3,t3,-1023
1c012af4:	4281                	li	t0,0
1c012af6:	4801                	li	a6,0
1c012af8:	a8a5                	j	1c012b70 <__muldf3+0xbc>
1c012afa:	00a8e7b3          	or	a5,a7,a0
1c012afe:	c7e9                	beqz	a5,1c012bc8 <__muldf3+0x114>
1c012b00:	02088e63          	beqz	a7,1c012b3c <__muldf3+0x88>
1c012b04:	100897b3          	p.fl1	a5,a7
1c012b08:	477d                	li	a4,31
1c012b0a:	40f707b3          	sub	a5,a4,a5
1c012b0e:	ff578813          	addi	a6,a5,-11
1c012b12:	4771                	li	a4,28
1c012b14:	03074c63          	blt	a4,a6,1c012b4c <__muldf3+0x98>
1c012b18:	4775                	li	a4,29
1c012b1a:	ff878f93          	addi	t6,a5,-8
1c012b1e:	41070733          	sub	a4,a4,a6
1c012b22:	01f898b3          	sll	a7,a7,t6
1c012b26:	00e55733          	srl	a4,a0,a4
1c012b2a:	011768b3          	or	a7,a4,a7
1c012b2e:	01f51fb3          	sll	t6,a0,t6
1c012b32:	c0d00e13          	li	t3,-1011
1c012b36:	40fe0e33          	sub	t3,t3,a5
1c012b3a:	bf6d                	j	1c012af4 <__muldf3+0x40>
1c012b3c:	47fd                	li	a5,31
1c012b3e:	10051e33          	p.fl1	t3,a0
1c012b42:	41c78e33          	sub	t3,a5,t3
1c012b46:	020e0793          	addi	a5,t3,32
1c012b4a:	b7d1                	j	1c012b0e <__muldf3+0x5a>
1c012b4c:	fd878893          	addi	a7,a5,-40
1c012b50:	011518b3          	sll	a7,a0,a7
1c012b54:	4f81                	li	t6,0
1c012b56:	bff1                	j	1c012b32 <__muldf3+0x7e>
1c012b58:	8faa                	mv	t6,a0
1c012b5a:	00a8e533          	or	a0,a7,a0
1c012b5e:	c935                	beqz	a0,1c012bd2 <__muldf3+0x11e>
1c012b60:	00c89793          	slli	a5,a7,0xc
1c012b64:	7ff00e13          	li	t3,2047
1c012b68:	428d                	li	t0,3
1c012b6a:	4841                	li	a6,16
1c012b6c:	f807c5e3          	bltz	a5,1c012af6 <__muldf3+0x42>
1c012b70:	d5469533          	p.extractu	a0,a3,10,20
1c012b74:	e6069733          	p.extractu	a4,a3,19,0
1c012b78:	87b2                	mv	a5,a2
1c012b7a:	82fd                	srli	a3,a3,0x1f
1c012b7c:	c12d                	beqz	a0,1c012bde <__muldf3+0x12a>
1c012b7e:	7ff00e93          	li	t4,2047
1c012b82:	0bd50b63          	beq	a0,t4,1c012c38 <__muldf3+0x184>
1c012b86:	070e                	slli	a4,a4,0x3
1c012b88:	01d65793          	srli	a5,a2,0x1d
1c012b8c:	c1774733          	p.bset	a4,a4,0,23
1c012b90:	8f5d                	or	a4,a4,a5
1c012b92:	00361793          	slli	a5,a2,0x3
1c012b96:	c0150613          	addi	a2,a0,-1023
1c012b9a:	4e81                	li	t4,0
1c012b9c:	00ce0f33          	add	t5,t3,a2
1c012ba0:	00229613          	slli	a2,t0,0x2
1c012ba4:	01d66633          	or	a2,a2,t4
1c012ba8:	167d                	addi	a2,a2,-1
1c012baa:	43b9                	li	t2,14
1c012bac:	00d5c533          	xor	a0,a1,a3
1c012bb0:	001f0e13          	addi	t3,t5,1
1c012bb4:	0ec3e963          	bltu	t2,a2,1c012ca6 <__muldf3+0x1f2>
1c012bb8:	1c01cf37          	lui	t5,0x1c01c
1c012bbc:	060a                	slli	a2,a2,0x2
1c012bbe:	a0cf0f13          	addi	t5,t5,-1524 # 1c01ba0c <pi_cl_pe_other_entry+0x4c>
1c012bc2:	20cf7603          	p.lw	a2,a2(t5)
1c012bc6:	8602                	jr	a2
1c012bc8:	4881                	li	a7,0
1c012bca:	4f81                	li	t6,0
1c012bcc:	4e01                	li	t3,0
1c012bce:	4285                	li	t0,1
1c012bd0:	b71d                	j	1c012af6 <__muldf3+0x42>
1c012bd2:	4881                	li	a7,0
1c012bd4:	4f81                	li	t6,0
1c012bd6:	7ff00e13          	li	t3,2047
1c012bda:	4289                	li	t0,2
1c012bdc:	bf29                	j	1c012af6 <__muldf3+0x42>
1c012bde:	00c767b3          	or	a5,a4,a2
1c012be2:	c7b5                	beqz	a5,1c012c4e <__muldf3+0x19a>
1c012be4:	cf05                	beqz	a4,1c012c1c <__muldf3+0x168>
1c012be6:	10071533          	p.fl1	a0,a4
1c012bea:	47fd                	li	a5,31
1c012bec:	40a78533          	sub	a0,a5,a0
1c012bf0:	ff550f13          	addi	t5,a0,-11
1c012bf4:	47f1                	li	a5,28
1c012bf6:	03e7cb63          	blt	a5,t5,1c012c2c <__muldf3+0x178>
1c012bfa:	4ef5                	li	t4,29
1c012bfc:	ff850793          	addi	a5,a0,-8
1c012c00:	41ee8eb3          	sub	t4,t4,t5
1c012c04:	00f71733          	sll	a4,a4,a5
1c012c08:	01d65eb3          	srl	t4,a2,t4
1c012c0c:	00eee733          	or	a4,t4,a4
1c012c10:	00f617b3          	sll	a5,a2,a5
1c012c14:	c0d00613          	li	a2,-1011
1c012c18:	8e09                	sub	a2,a2,a0
1c012c1a:	b741                	j	1c012b9a <__muldf3+0xe6>
1c012c1c:	10061533          	p.fl1	a0,a2
1c012c20:	47fd                	li	a5,31
1c012c22:	40a78533          	sub	a0,a5,a0
1c012c26:	02050513          	addi	a0,a0,32
1c012c2a:	b7d9                	j	1c012bf0 <__muldf3+0x13c>
1c012c2c:	fd850713          	addi	a4,a0,-40
1c012c30:	00e61733          	sll	a4,a2,a4
1c012c34:	4781                	li	a5,0
1c012c36:	bff9                	j	1c012c14 <__muldf3+0x160>
1c012c38:	8e59                	or	a2,a2,a4
1c012c3a:	ce19                	beqz	a2,1c012c58 <__muldf3+0x1a4>
1c012c3c:	00c71613          	slli	a2,a4,0xc
1c012c40:	02064263          	bltz	a2,1c012c64 <__muldf3+0x1b0>
1c012c44:	7ff00613          	li	a2,2047
1c012c48:	4e8d                	li	t4,3
1c012c4a:	4841                	li	a6,16
1c012c4c:	bf81                	j	1c012b9c <__muldf3+0xe8>
1c012c4e:	4701                	li	a4,0
1c012c50:	4781                	li	a5,0
1c012c52:	4601                	li	a2,0
1c012c54:	4e85                	li	t4,1
1c012c56:	b799                	j	1c012b9c <__muldf3+0xe8>
1c012c58:	4701                	li	a4,0
1c012c5a:	4781                	li	a5,0
1c012c5c:	7ff00613          	li	a2,2047
1c012c60:	4e89                	li	t4,2
1c012c62:	bf2d                	j	1c012b9c <__muldf3+0xe8>
1c012c64:	7ff00613          	li	a2,2047
1c012c68:	4e8d                	li	t4,3
1c012c6a:	bf0d                	j	1c012b9c <__muldf3+0xe8>
1c012c6c:	00080737          	lui	a4,0x80
1c012c70:	4781                	li	a5,0
1c012c72:	7ff00893          	li	a7,2047
1c012c76:	4501                	li	a0,0
1c012c78:	4841                	li	a6,16
1c012c7a:	4681                	li	a3,0
1c012c7c:	e60726b3          	p.insert	a3,a4,19,0
1c012c80:	d548a6b3          	p.insert	a3,a7,10,20
1c012c84:	c1f526b3          	p.insert	a3,a0,0,31
1c012c88:	85b6                	mv	a1,a3
1c012c8a:	853e                	mv	a0,a5
1c012c8c:	00080463          	beqz	a6,1c012c94 <__muldf3+0x1e0>
1c012c90:	00182073          	csrs	0x1,a6
1c012c94:	4472                	lw	s0,28(sp)
1c012c96:	44e2                	lw	s1,24(sp)
1c012c98:	4952                	lw	s2,20(sp)
1c012c9a:	49c2                	lw	s3,16(sp)
1c012c9c:	4a32                	lw	s4,12(sp)
1c012c9e:	4aa2                	lw	s5,8(sp)
1c012ca0:	4b12                	lw	s6,4(sp)
1c012ca2:	6105                	addi	sp,sp,32
1c012ca4:	8082                	ret
1c012ca6:	6641                	lui	a2,0x10
1c012ca8:	fff60293          	addi	t0,a2,-1 # ffff <__heapfcram_size+0xdbf>
1c012cac:	010fd693          	srli	a3,t6,0x10
1c012cb0:	0057f4b3          	and	s1,a5,t0
1c012cb4:	02968433          	mul	s0,a3,s1
1c012cb8:	005fffb3          	and	t6,t6,t0
1c012cbc:	0107d993          	srli	s3,a5,0x10
1c012cc0:	8ea2                	mv	t4,s0
1c012cc2:	029f87b3          	mul	a5,t6,s1
1c012cc6:	43f98eb3          	p.mac	t4,s3,t6
1c012cca:	0107d593          	srli	a1,a5,0x10
1c012cce:	95f6                	add	a1,a1,t4
1c012cd0:	033683b3          	mul	t2,a3,s3
1c012cd4:	0085f363          	bleu	s0,a1,1c012cda <__muldf3+0x226>
1c012cd8:	93b2                	add	t2,t2,a2
1c012cda:	0055feb3          	and	t4,a1,t0
1c012cde:	0057f7b3          	and	a5,a5,t0
1c012ce2:	0ec2                	slli	t4,t4,0x10
1c012ce4:	9ebe                	add	t4,t4,a5
1c012ce6:	005777b3          	and	a5,a4,t0
1c012cea:	01075913          	srli	s2,a4,0x10
1c012cee:	0105da13          	srli	s4,a1,0x10
1c012cf2:	02f68633          	mul	a2,a3,a5
1c012cf6:	85b2                	mv	a1,a2
1c012cf8:	02ff8733          	mul	a4,t6,a5
1c012cfc:	43f905b3          	p.mac	a1,s2,t6
1c012d00:	8fae                	mv	t6,a1
1c012d02:	01075593          	srli	a1,a4,0x10
1c012d06:	95fe                	add	a1,a1,t6
1c012d08:	032686b3          	mul	a3,a3,s2
1c012d0c:	00c5f463          	bleu	a2,a1,1c012d14 <__muldf3+0x260>
1c012d10:	6641                	lui	a2,0x10
1c012d12:	96b2                	add	a3,a3,a2
1c012d14:	6ac1                	lui	s5,0x10
1c012d16:	fffa8413          	addi	s0,s5,-1 # ffff <__heapfcram_size+0xdbf>
1c012d1a:	0085f2b3          	and	t0,a1,s0
1c012d1e:	0105d613          	srli	a2,a1,0x10
1c012d22:	008775b3          	and	a1,a4,s0
1c012d26:	0108d713          	srli	a4,a7,0x10
1c012d2a:	0088f433          	and	s0,a7,s0
1c012d2e:	02940b33          	mul	s6,s0,s1
1c012d32:	02c2                	slli	t0,t0,0x10
1c012d34:	92ae                	add	t0,t0,a1
1c012d36:	9636                	add	a2,a2,a3
1c012d38:	9a16                	add	s4,s4,t0
1c012d3a:	029704b3          	mul	s1,a4,s1
1c012d3e:	010b5f93          	srli	t6,s6,0x10
1c012d42:	88a6                	mv	a7,s1
1c012d44:	428988b3          	p.mac	a7,s3,s0
1c012d48:	9fc6                	add	t6,t6,a7
1c012d4a:	02e986b3          	mul	a3,s3,a4
1c012d4e:	009ff363          	bleu	s1,t6,1c012d54 <__muldf3+0x2a0>
1c012d52:	96d6                	add	a3,a3,s5
1c012d54:	010fd493          	srli	s1,t6,0x10
1c012d58:	65c1                	lui	a1,0x10
1c012d5a:	94b6                	add	s1,s1,a3
1c012d5c:	fff58693          	addi	a3,a1,-1 # ffff <__heapfcram_size+0xdbf>
1c012d60:	00dff8b3          	and	a7,t6,a3
1c012d64:	00db7fb3          	and	t6,s6,a3
1c012d68:	08c2                	slli	a7,a7,0x10
1c012d6a:	028786b3          	mul	a3,a5,s0
1c012d6e:	98fe                	add	a7,a7,t6
1c012d70:	02f707b3          	mul	a5,a4,a5
1c012d74:	02e90fb3          	mul	t6,s2,a4
1c012d78:	873e                	mv	a4,a5
1c012d7a:	42890733          	p.mac	a4,s2,s0
1c012d7e:	0106d413          	srli	s0,a3,0x10
1c012d82:	9722                	add	a4,a4,s0
1c012d84:	00f77363          	bleu	a5,a4,1c012d8a <__muldf3+0x2d6>
1c012d88:	9fae                	add	t6,t6,a1
1c012d8a:	67c1                	lui	a5,0x10
1c012d8c:	17fd                	addi	a5,a5,-1
1c012d8e:	00f775b3          	and	a1,a4,a5
1c012d92:	8efd                	and	a3,a3,a5
1c012d94:	05c2                	slli	a1,a1,0x10
1c012d96:	93d2                	add	t2,t2,s4
1c012d98:	95b6                	add	a1,a1,a3
1c012d9a:	00c586b3          	add	a3,a1,a2
1c012d9e:	0053b2b3          	sltu	t0,t2,t0
1c012da2:	00568933          	add	s2,a3,t0
1c012da6:	00c6b6b3          	sltu	a3,a3,a2
1c012daa:	8636                	mv	a2,a3
1c012dac:	005936b3          	sltu	a3,s2,t0
1c012db0:	93c6                	add	t2,t2,a7
1c012db2:	8ed1                	or	a3,a3,a2
1c012db4:	0113b8b3          	sltu	a7,t2,a7
1c012db8:	009905b3          	add	a1,s2,s1
1c012dbc:	8341                	srli	a4,a4,0x10
1c012dbe:	0ff6f693          	andi	a3,a3,255
1c012dc2:	01158433          	add	s0,a1,a7
1c012dc6:	96ba                	add	a3,a3,a4
1c012dc8:	0095b733          	sltu	a4,a1,s1
1c012dcc:	85ba                	mv	a1,a4
1c012dce:	01143733          	sltu	a4,s0,a7
1c012dd2:	8f4d                	or	a4,a4,a1
1c012dd4:	0ff77713          	andi	a4,a4,255
1c012dd8:	9736                	add	a4,a4,a3
1c012dda:	977e                	add	a4,a4,t6
1c012ddc:	01745793          	srli	a5,s0,0x17
1c012de0:	0726                	slli	a4,a4,0x9
1c012de2:	8f5d                	or	a4,a4,a5
1c012de4:	00939793          	slli	a5,t2,0x9
1c012de8:	01d7e7b3          	or	a5,a5,t4
1c012dec:	00f037b3          	snez	a5,a5
1c012df0:	0173d393          	srli	t2,t2,0x17
1c012df4:	0077e7b3          	or	a5,a5,t2
1c012df8:	0426                	slli	s0,s0,0x9
1c012dfa:	00771613          	slli	a2,a4,0x7
1c012dfe:	8fc1                	or	a5,a5,s0
1c012e00:	06065963          	bgez	a2,1c012e72 <__muldf3+0x3be>
1c012e04:	0017d693          	srli	a3,a5,0x1
1c012e08:	fc17b7b3          	p.bclr	a5,a5,30,1
1c012e0c:	8fd5                	or	a5,a5,a3
1c012e0e:	01f71693          	slli	a3,a4,0x1f
1c012e12:	8fd5                	or	a5,a5,a3
1c012e14:	8305                	srli	a4,a4,0x1
1c012e16:	3ffe0893          	addi	a7,t3,1023
1c012e1a:	0b105a63          	blez	a7,1c012ece <__muldf3+0x41a>
1c012e1e:	f837b6b3          	p.bclr	a3,a5,28,3
1c012e22:	ceb9                	beqz	a3,1c012e80 <__muldf3+0x3cc>
1c012e24:	c0084833          	p.bset	a6,a6,0,0
1c012e28:	04232b63          	p.beqimm	t1,2,1c012e7e <__muldf3+0x3ca>
1c012e2c:	04332563          	p.beqimm	t1,3,1c012e76 <__muldf3+0x3c2>
1c012e30:	04031863          	bnez	t1,1c012e80 <__muldf3+0x3cc>
1c012e34:	f647b6b3          	p.bclr	a3,a5,27,4
1c012e38:	0446a463          	p.beqimm	a3,4,1c012e80 <__muldf3+0x3cc>
1c012e3c:	00478693          	addi	a3,a5,4 # 10004 <__heapfcram_size+0xdc4>
1c012e40:	00f6b7b3          	sltu	a5,a3,a5
1c012e44:	973e                	add	a4,a4,a5
1c012e46:	87b6                	mv	a5,a3
1c012e48:	a825                	j	1c012e80 <__muldf3+0x3cc>
1c012e4a:	852e                	mv	a0,a1
1c012e4c:	8746                	mv	a4,a7
1c012e4e:	87fe                	mv	a5,t6
1c012e50:	8e96                	mv	t4,t0
1c012e52:	002eab63          	p.beqimm	t4,2,1c012e68 <__muldf3+0x3b4>
1c012e56:	1a3eab63          	p.beqimm	t4,3,1c01300c <__muldf3+0x558>
1c012e5a:	fa1ebee3          	p.bneimm	t4,1,1c012e16 <__muldf3+0x362>
1c012e5e:	4701                	li	a4,0
1c012e60:	4781                	li	a5,0
1c012e62:	aa61                	j	1c012ffa <__muldf3+0x546>
1c012e64:	8536                	mv	a0,a3
1c012e66:	b7f5                	j	1c012e52 <__muldf3+0x39e>
1c012e68:	4701                	li	a4,0
1c012e6a:	4781                	li	a5,0
1c012e6c:	7ff00893          	li	a7,2047
1c012e70:	b529                	j	1c012c7a <__muldf3+0x1c6>
1c012e72:	8e7a                	mv	t3,t5
1c012e74:	b74d                	j	1c012e16 <__muldf3+0x362>
1c012e76:	e509                	bnez	a0,1c012e80 <__muldf3+0x3cc>
1c012e78:	00878693          	addi	a3,a5,8
1c012e7c:	b7d1                	j	1c012e40 <__muldf3+0x38c>
1c012e7e:	fd6d                	bnez	a0,1c012e78 <__muldf3+0x3c4>
1c012e80:	00771693          	slli	a3,a4,0x7
1c012e84:	0006d663          	bgez	a3,1c012e90 <__muldf3+0x3dc>
1c012e88:	c1873733          	p.bclr	a4,a4,0,24
1c012e8c:	400e0893          	addi	a7,t3,1024
1c012e90:	7fe00693          	li	a3,2046
1c012e94:	0116c963          	blt	a3,a7,1c012ea6 <__muldf3+0x3f2>
1c012e98:	0037d693          	srli	a3,a5,0x3
1c012e9c:	01d71793          	slli	a5,a4,0x1d
1c012ea0:	8fd5                	or	a5,a5,a3
1c012ea2:	830d                	srli	a4,a4,0x3
1c012ea4:	bbd9                	j	1c012c7a <__muldf3+0x1c6>
1c012ea6:	02232263          	p.beqimm	t1,2,1c012eca <__muldf3+0x416>
1c012eaa:	00332b63          	p.beqimm	t1,3,1c012ec0 <__muldf3+0x40c>
1c012eae:	00030a63          	beqz	t1,1c012ec2 <__muldf3+0x40e>
1c012eb2:	57fd                	li	a5,-1
1c012eb4:	7fe00893          	li	a7,2046
1c012eb8:	00586813          	ori	a6,a6,5
1c012ebc:	873e                	mv	a4,a5
1c012ebe:	bb75                	j	1c012c7a <__muldf3+0x1c6>
1c012ec0:	f96d                	bnez	a0,1c012eb2 <__muldf3+0x3fe>
1c012ec2:	4781                	li	a5,0
1c012ec4:	7ff00893          	li	a7,2047
1c012ec8:	bfc5                	j	1c012eb8 <__muldf3+0x404>
1c012eca:	fd65                	bnez	a0,1c012ec2 <__muldf3+0x40e>
1c012ecc:	b7dd                	j	1c012eb2 <__muldf3+0x3fe>
1c012ece:	4585                	li	a1,1
1c012ed0:	04089363          	bnez	a7,1c012f16 <__muldf3+0x462>
1c012ed4:	f837b6b3          	p.bclr	a3,a5,28,3
1c012ed8:	863a                	mv	a2,a4
1c012eda:	ca85                	beqz	a3,1c012f0a <__muldf3+0x456>
1c012edc:	c0084833          	p.bset	a6,a6,0,0
1c012ee0:	02232463          	p.beqimm	t1,2,1c012f08 <__muldf3+0x454>
1c012ee4:	00332e63          	p.beqimm	t1,3,1c012f00 <__muldf3+0x44c>
1c012ee8:	02031163          	bnez	t1,1c012f0a <__muldf3+0x456>
1c012eec:	f647b6b3          	p.bclr	a3,a5,27,4
1c012ef0:	0046ad63          	p.beqimm	a3,4,1c012f0a <__muldf3+0x456>
1c012ef4:	ffc7b613          	sltiu	a2,a5,-4
1c012ef8:	00164613          	xori	a2,a2,1
1c012efc:	963a                	add	a2,a2,a4
1c012efe:	a031                	j	1c012f0a <__muldf3+0x456>
1c012f00:	e509                	bnez	a0,1c012f0a <__muldf3+0x456>
1c012f02:	ff87b613          	sltiu	a2,a5,-8
1c012f06:	bfcd                	j	1c012ef8 <__muldf3+0x444>
1c012f08:	fd6d                	bnez	a0,1c012f02 <__muldf3+0x44e>
1c012f0a:	01865593          	srli	a1,a2,0x18
1c012f0e:	0015c593          	xori	a1,a1,1
1c012f12:	fc15b5b3          	p.bclr	a1,a1,30,1
1c012f16:	4605                	li	a2,1
1c012f18:	41160633          	sub	a2,a2,a7
1c012f1c:	03800693          	li	a3,56
1c012f20:	0ac6ce63          	blt	a3,a2,1c012fdc <__muldf3+0x528>
1c012f24:	46fd                	li	a3,31
1c012f26:	04c6c763          	blt	a3,a2,1c012f74 <__muldf3+0x4c0>
1c012f2a:	41ee0e13          	addi	t3,t3,1054
1c012f2e:	01c716b3          	sll	a3,a4,t3
1c012f32:	00c7d8b3          	srl	a7,a5,a2
1c012f36:	01c797b3          	sll	a5,a5,t3
1c012f3a:	0116e6b3          	or	a3,a3,a7
1c012f3e:	00f037b3          	snez	a5,a5
1c012f42:	8fd5                	or	a5,a5,a3
1c012f44:	00c75733          	srl	a4,a4,a2
1c012f48:	f837b6b3          	p.bclr	a3,a5,28,3
1c012f4c:	ceb1                	beqz	a3,1c012fa8 <__muldf3+0x4f4>
1c012f4e:	c0084833          	p.bset	a6,a6,0,0
1c012f52:	04232a63          	p.beqimm	t1,2,1c012fa6 <__muldf3+0x4f2>
1c012f56:	04332463          	p.beqimm	t1,3,1c012f9e <__muldf3+0x4ea>
1c012f5a:	04031763          	bnez	t1,1c012fa8 <__muldf3+0x4f4>
1c012f5e:	f647b6b3          	p.bclr	a3,a5,27,4
1c012f62:	0446a363          	p.beqimm	a3,4,1c012fa8 <__muldf3+0x4f4>
1c012f66:	00478693          	addi	a3,a5,4
1c012f6a:	00f6b7b3          	sltu	a5,a3,a5
1c012f6e:	973e                	add	a4,a4,a5
1c012f70:	87b6                	mv	a5,a3
1c012f72:	a81d                	j	1c012fa8 <__muldf3+0x4f4>
1c012f74:	5685                	li	a3,-31
1c012f76:	411686b3          	sub	a3,a3,a7
1c012f7a:	02000e93          	li	t4,32
1c012f7e:	00d756b3          	srl	a3,a4,a3
1c012f82:	4881                	li	a7,0
1c012f84:	01d60663          	beq	a2,t4,1c012f90 <__muldf3+0x4dc>
1c012f88:	43ee0e13          	addi	t3,t3,1086
1c012f8c:	01c718b3          	sll	a7,a4,t3
1c012f90:	00f8e7b3          	or	a5,a7,a5
1c012f94:	00f037b3          	snez	a5,a5
1c012f98:	8fd5                	or	a5,a5,a3
1c012f9a:	4701                	li	a4,0
1c012f9c:	b775                	j	1c012f48 <__muldf3+0x494>
1c012f9e:	e509                	bnez	a0,1c012fa8 <__muldf3+0x4f4>
1c012fa0:	00878693          	addi	a3,a5,8
1c012fa4:	b7d9                	j	1c012f6a <__muldf3+0x4b6>
1c012fa6:	fd6d                	bnez	a0,1c012fa0 <__muldf3+0x4ec>
1c012fa8:	00871613          	slli	a2,a4,0x8
1c012fac:	02065063          	bgez	a2,1c012fcc <__muldf3+0x518>
1c012fb0:	c0084833          	p.bset	a6,a6,0,0
1c012fb4:	4701                	li	a4,0
1c012fb6:	4781                	li	a5,0
1c012fb8:	4885                	li	a7,1
1c012fba:	cc0580e3          	beqz	a1,1c012c7a <__muldf3+0x1c6>
1c012fbe:	fc1836b3          	p.bclr	a3,a6,30,1
1c012fc2:	ca068ce3          	beqz	a3,1c012c7a <__muldf3+0x1c6>
1c012fc6:	c0184833          	p.bset	a6,a6,0,1
1c012fca:	b945                	j	1c012c7a <__muldf3+0x1c6>
1c012fcc:	0037d693          	srli	a3,a5,0x3
1c012fd0:	01d71793          	slli	a5,a4,0x1d
1c012fd4:	8fd5                	or	a5,a5,a3
1c012fd6:	830d                	srli	a4,a4,0x3
1c012fd8:	4881                	li	a7,0
1c012fda:	b7c5                	j	1c012fba <__muldf3+0x506>
1c012fdc:	8fd9                	or	a5,a5,a4
1c012fde:	cb99                	beqz	a5,1c012ff4 <__muldf3+0x540>
1c012fe0:	c0084833          	p.bset	a6,a6,0,0
1c012fe4:	02232163          	p.beqimm	t1,2,1c013006 <__muldf3+0x552>
1c012fe8:	00332b63          	p.beqimm	t1,3,1c012ffe <__muldf3+0x54a>
1c012fec:	4795                	li	a5,5
1c012fee:	00031a63          	bnez	t1,1c013002 <__muldf3+0x54e>
1c012ff2:	838d                	srli	a5,a5,0x3
1c012ff4:	c0184833          	p.bset	a6,a6,0,1
1c012ff8:	4701                	li	a4,0
1c012ffa:	4881                	li	a7,0
1c012ffc:	b9bd                	j	1c012c7a <__muldf3+0x1c6>
1c012ffe:	47a5                	li	a5,9
1c013000:	d96d                	beqz	a0,1c012ff2 <__muldf3+0x53e>
1c013002:	4785                	li	a5,1
1c013004:	b7fd                	j	1c012ff2 <__muldf3+0x53e>
1c013006:	47a5                	li	a5,9
1c013008:	f56d                	bnez	a0,1c012ff2 <__muldf3+0x53e>
1c01300a:	bfe5                	j	1c013002 <__muldf3+0x54e>
1c01300c:	00080737          	lui	a4,0x80
1c013010:	4781                	li	a5,0
1c013012:	7ff00893          	li	a7,2047
1c013016:	4501                	li	a0,0
1c013018:	b18d                	j	1c012c7a <__muldf3+0x1c6>

1c01301a <__subdf3>:
1c01301a:	00202373          	frrm	t1
1c01301e:	e60597b3          	p.extractu	a5,a1,19,0
1c013022:	00379893          	slli	a7,a5,0x3
1c013026:	01d55793          	srli	a5,a0,0x1d
1c01302a:	0117e7b3          	or	a5,a5,a7
1c01302e:	e60698b3          	p.extractu	a7,a3,19,0
1c013032:	01d65713          	srli	a4,a2,0x1d
1c013036:	d5469f33          	p.extractu	t5,a3,10,20
1c01303a:	088e                	slli	a7,a7,0x3
1c01303c:	00361813          	slli	a6,a2,0x3
1c013040:	7ff00613          	li	a2,2047
1c013044:	d5459eb3          	p.extractu	t4,a1,10,20
1c013048:	050e                	slli	a0,a0,0x3
1c01304a:	81fd                	srli	a1,a1,0x1f
1c01304c:	82fd                	srli	a3,a3,0x1f
1c01304e:	01176733          	or	a4,a4,a7
1c013052:	00cf1563          	bne	t5,a2,1c01305c <__subdf3+0x42>
1c013056:	01076633          	or	a2,a4,a6
1c01305a:	e219                	bnez	a2,1c013060 <__subdf3+0x46>
1c01305c:	0016c693          	xori	a3,a3,1
1c013060:	41ee88b3          	sub	a7,t4,t5
1c013064:	3ab69863          	bne	a3,a1,1c013414 <__subdf3+0x3fa>
1c013068:	13105863          	blez	a7,1c013198 <__subdf3+0x17e>
1c01306c:	080f1863          	bnez	t5,1c0130fc <__subdf3+0xe2>
1c013070:	010766b3          	or	a3,a4,a6
1c013074:	e28d                	bnez	a3,1c013096 <__subdf3+0x7c>
1c013076:	7ff00713          	li	a4,2047
1c01307a:	3ae89863          	bne	a7,a4,1c01342a <__subdf3+0x410>
1c01307e:	00a7e733          	or	a4,a5,a0
1c013082:	78070763          	beqz	a4,1c013810 <__subdf3+0x7f6>
1c013086:	00979613          	slli	a2,a5,0x9
1c01308a:	7ff00f13          	li	t5,2047
1c01308e:	4881                	li	a7,0
1c013090:	06065463          	bgez	a2,1c0130f8 <__subdf3+0xde>
1c013094:	ae09                	j	1c0133a6 <__subdf3+0x38c>
1c013096:	fff88693          	addi	a3,a7,-1 # 3fffff <__l1_heapsram_size+0x3e006f>
1c01309a:	e2a1                	bnez	a3,1c0130da <__subdf3+0xc0>
1c01309c:	982a                	add	a6,a6,a0
1c01309e:	97ba                	add	a5,a5,a4
1c0130a0:	00a83533          	sltu	a0,a6,a0
1c0130a4:	00a78733          	add	a4,a5,a0
1c0130a8:	4f05                	li	t5,1
1c0130aa:	00871613          	slli	a2,a4,0x8
1c0130ae:	46065263          	bgez	a2,1c013512 <__subdf3+0x4f8>
1c0130b2:	0f05                	addi	t5,t5,1
1c0130b4:	7ff00793          	li	a5,2047
1c0130b8:	32ff0c63          	beq	t5,a5,1c0133f0 <__subdf3+0x3d6>
1c0130bc:	c17737b3          	p.bclr	a5,a4,0,23
1c0130c0:	00185693          	srli	a3,a6,0x1
1c0130c4:	fc183833          	p.bclr	a6,a6,30,1
1c0130c8:	01f79513          	slli	a0,a5,0x1f
1c0130cc:	0106e833          	or	a6,a3,a6
1c0130d0:	01056533          	or	a0,a0,a6
1c0130d4:	8385                	srli	a5,a5,0x1
1c0130d6:	4881                	li	a7,0
1c0130d8:	a4f9                	j	1c0133a6 <__subdf3+0x38c>
1c0130da:	7ff00613          	li	a2,2047
1c0130de:	04c89263          	bne	a7,a2,1c013122 <__subdf3+0x108>
1c0130e2:	00a7e733          	or	a4,a5,a0
1c0130e6:	72070563          	beqz	a4,1c013810 <__subdf3+0x7f6>
1c0130ea:	00979713          	slli	a4,a5,0x9
1c0130ee:	7ff00f13          	li	t5,2047
1c0130f2:	4881                	li	a7,0
1c0130f4:	2a074963          	bltz	a4,1c0133a6 <__subdf3+0x38c>
1c0130f8:	4e41                	li	t3,16
1c0130fa:	aac9                	j	1c0132cc <__subdf3+0x2b2>
1c0130fc:	7ff00693          	li	a3,2047
1c013100:	00de9e63          	bne	t4,a3,1c01311c <__subdf3+0x102>
1c013104:	00a7e733          	or	a4,a5,a0
1c013108:	70070463          	beqz	a4,1c013810 <__subdf3+0x7f6>
1c01310c:	00979e13          	slli	t3,a5,0x9
1c013110:	7ff00f13          	li	t5,2047
1c013114:	4881                	li	a7,0
1c013116:	fe0e51e3          	bgez	t3,1c0130f8 <__subdf3+0xde>
1c01311a:	a471                	j	1c0133a6 <__subdf3+0x38c>
1c01311c:	c1774733          	p.bset	a4,a4,0,23
1c013120:	86c6                	mv	a3,a7
1c013122:	03800613          	li	a2,56
1c013126:	06d64463          	blt	a2,a3,1c01318e <__subdf3+0x174>
1c01312a:	467d                	li	a2,31
1c01312c:	02d64c63          	blt	a2,a3,1c013164 <__subdf3+0x14a>
1c013130:	02000893          	li	a7,32
1c013134:	40d888b3          	sub	a7,a7,a3
1c013138:	00d85e33          	srl	t3,a6,a3
1c01313c:	01171633          	sll	a2,a4,a7
1c013140:	01181833          	sll	a6,a6,a7
1c013144:	01c66633          	or	a2,a2,t3
1c013148:	01003833          	snez	a6,a6
1c01314c:	01066833          	or	a6,a2,a6
1c013150:	00d756b3          	srl	a3,a4,a3
1c013154:	982a                	add	a6,a6,a0
1c013156:	96be                	add	a3,a3,a5
1c013158:	00a837b3          	sltu	a5,a6,a0
1c01315c:	00f68733          	add	a4,a3,a5
1c013160:	8f76                	mv	t5,t4
1c013162:	b7a1                	j	1c0130aa <__subdf3+0x90>
1c013164:	02000e13          	li	t3,32
1c013168:	00d75633          	srl	a2,a4,a3
1c01316c:	4881                	li	a7,0
1c01316e:	01c68863          	beq	a3,t3,1c01317e <__subdf3+0x164>
1c013172:	04000893          	li	a7,64
1c013176:	40d886b3          	sub	a3,a7,a3
1c01317a:	00d718b3          	sll	a7,a4,a3
1c01317e:	0108e833          	or	a6,a7,a6
1c013182:	01003833          	snez	a6,a6
1c013186:	01066833          	or	a6,a2,a6
1c01318a:	4681                	li	a3,0
1c01318c:	b7e1                	j	1c013154 <__subdf3+0x13a>
1c01318e:	01076833          	or	a6,a4,a6
1c013192:	01003833          	snez	a6,a6
1c013196:	bfd5                	j	1c01318a <__subdf3+0x170>
1c013198:	0e088a63          	beqz	a7,1c01328c <__subdf3+0x272>
1c01319c:	040e9e63          	bnez	t4,1c0131f8 <__subdf3+0x1de>
1c0131a0:	00a7e6b3          	or	a3,a5,a0
1c0131a4:	e28d                	bnez	a3,1c0131c6 <__subdf3+0x1ac>
1c0131a6:	7ff00793          	li	a5,2047
1c0131aa:	36ff1463          	bne	t5,a5,1c013512 <__subdf3+0x4f8>
1c0131ae:	01076533          	or	a0,a4,a6
1c0131b2:	66050563          	beqz	a0,1c01381c <__subdf3+0x802>
1c0131b6:	00971e93          	slli	t4,a4,0x9
1c0131ba:	87ba                	mv	a5,a4
1c0131bc:	8542                	mv	a0,a6
1c0131be:	4881                	li	a7,0
1c0131c0:	f20edce3          	bgez	t4,1c0130f8 <__subdf3+0xde>
1c0131c4:	a2cd                	j	1c0133a6 <__subdf3+0x38c>
1c0131c6:	01f8ba63          	p.bneimm	a7,-1,1c0131da <__subdf3+0x1c0>
1c0131ca:	010506b3          	add	a3,a0,a6
1c0131ce:	97ba                	add	a5,a5,a4
1c0131d0:	0106b733          	sltu	a4,a3,a6
1c0131d4:	973e                	add	a4,a4,a5
1c0131d6:	8836                	mv	a6,a3
1c0131d8:	bdc9                	j	1c0130aa <__subdf3+0x90>
1c0131da:	7ff00693          	li	a3,2047
1c0131de:	fff8c893          	not	a7,a7
1c0131e2:	02df1e63          	bne	t5,a3,1c01321e <__subdf3+0x204>
1c0131e6:	01076533          	or	a0,a4,a6
1c0131ea:	62050963          	beqz	a0,1c01381c <__subdf3+0x802>
1c0131ee:	00971613          	slli	a2,a4,0x9
1c0131f2:	87ba                	mv	a5,a4
1c0131f4:	8542                	mv	a0,a6
1c0131f6:	bd61                	j	1c01308e <__subdf3+0x74>
1c0131f8:	7ff00693          	li	a3,2047
1c0131fc:	00df1d63          	bne	t5,a3,1c013216 <__subdf3+0x1fc>
1c013200:	01076533          	or	a0,a4,a6
1c013204:	60050c63          	beqz	a0,1c01381c <__subdf3+0x802>
1c013208:	00971793          	slli	a5,a4,0x9
1c01320c:	5e07db63          	bgez	a5,1c013802 <__subdf3+0x7e8>
1c013210:	87ba                	mv	a5,a4
1c013212:	8542                	mv	a0,a6
1c013214:	b5c9                	j	1c0130d6 <__subdf3+0xbc>
1c013216:	411008b3          	neg	a7,a7
1c01321a:	c177c7b3          	p.bset	a5,a5,0,23
1c01321e:	03800693          	li	a3,56
1c013222:	0716c163          	blt	a3,a7,1c013284 <__subdf3+0x26a>
1c013226:	46fd                	li	a3,31
1c013228:	0316cb63          	blt	a3,a7,1c01325e <__subdf3+0x244>
1c01322c:	02000613          	li	a2,32
1c013230:	41160633          	sub	a2,a2,a7
1c013234:	01155e33          	srl	t3,a0,a7
1c013238:	00c796b3          	sll	a3,a5,a2
1c01323c:	00c51533          	sll	a0,a0,a2
1c013240:	01c6e6b3          	or	a3,a3,t3
1c013244:	00a03533          	snez	a0,a0
1c013248:	8d55                	or	a0,a0,a3
1c01324a:	0117d8b3          	srl	a7,a5,a7
1c01324e:	9542                	add	a0,a0,a6
1c013250:	00e887b3          	add	a5,a7,a4
1c013254:	01053733          	sltu	a4,a0,a6
1c013258:	973e                	add	a4,a4,a5
1c01325a:	882a                	mv	a6,a0
1c01325c:	b5b9                	j	1c0130aa <__subdf3+0x90>
1c01325e:	02000e13          	li	t3,32
1c013262:	0117d6b3          	srl	a3,a5,a7
1c013266:	4601                	li	a2,0
1c013268:	01c88863          	beq	a7,t3,1c013278 <__subdf3+0x25e>
1c01326c:	04000613          	li	a2,64
1c013270:	411608b3          	sub	a7,a2,a7
1c013274:	01179633          	sll	a2,a5,a7
1c013278:	8d51                	or	a0,a0,a2
1c01327a:	00a03533          	snez	a0,a0
1c01327e:	8d55                	or	a0,a0,a3
1c013280:	4881                	li	a7,0
1c013282:	b7f1                	j	1c01324e <__subdf3+0x234>
1c013284:	8d5d                	or	a0,a0,a5
1c013286:	00a03533          	snez	a0,a0
1c01328a:	bfdd                	j	1c013280 <__subdf3+0x266>
1c01328c:	001e8693          	addi	a3,t4,1
1c013290:	e8b6be33          	p.bclr	t3,a3,20,11
1c013294:	4605                	li	a2,1
1c013296:	0fc64763          	blt	a2,t3,1c013384 <__subdf3+0x36a>
1c01329a:	00a7e6b3          	or	a3,a5,a0
1c01329e:	040e9263          	bnez	t4,1c0132e2 <__subdf3+0x2c8>
1c0132a2:	26068a63          	beqz	a3,1c013516 <__subdf3+0x4fc>
1c0132a6:	010766b3          	or	a3,a4,a6
1c0132aa:	52068a63          	beqz	a3,1c0137de <__subdf3+0x7c4>
1c0132ae:	982a                	add	a6,a6,a0
1c0132b0:	97ba                	add	a5,a5,a4
1c0132b2:	00a83533          	sltu	a0,a6,a0
1c0132b6:	00a78733          	add	a4,a5,a0
1c0132ba:	00871e13          	slli	t3,a4,0x8
1c0132be:	240e5c63          	bgez	t3,1c013516 <__subdf3+0x4fc>
1c0132c2:	c17737b3          	p.bclr	a5,a4,0,23
1c0132c6:	8542                	mv	a0,a6
1c0132c8:	4e01                	li	t3,0
1c0132ca:	4f05                	li	t5,1
1c0132cc:	f8353733          	p.bclr	a4,a0,28,3
1c0132d0:	e77d                	bnez	a4,1c0133be <__subdf3+0x3a4>
1c0132d2:	04088b63          	beqz	a7,1c013328 <__subdf3+0x30e>
1c0132d6:	fc1e3733          	p.bclr	a4,t3,30,1
1c0132da:	c739                	beqz	a4,1c013328 <__subdf3+0x30e>
1c0132dc:	c01e4e33          	p.bset	t3,t3,0,1
1c0132e0:	a0a1                	j	1c013328 <__subdf3+0x30e>
1c0132e2:	7ff00613          	li	a2,2047
1c0132e6:	08ce9a63          	bne	t4,a2,1c01337a <__subdf3+0x360>
1c0132ea:	58068a63          	beqz	a3,1c01387e <__subdf3+0x864>
1c0132ee:	00400e37          	lui	t3,0x400
1c0132f2:	01c7fe33          	and	t3,a5,t3
1c0132f6:	001e3e13          	seqz	t3,t3
1c0132fa:	0e12                	slli	t3,t3,0x4
1c0132fc:	01df1c63          	bne	t5,t4,1c013314 <__subdf3+0x2fa>
1c013300:	01076633          	or	a2,a4,a6
1c013304:	c611                	beqz	a2,1c013310 <__subdf3+0x2f6>
1c013306:	00971e93          	slli	t4,a4,0x9
1c01330a:	000ec363          	bltz	t4,1c013310 <__subdf3+0x2f6>
1c01330e:	4e41                	li	t3,16
1c013310:	4e068d63          	beqz	a3,1c01380a <__subdf3+0x7f0>
1c013314:	01076733          	or	a4,a4,a6
1c013318:	3c070663          	beqz	a4,1c0136e4 <__subdf3+0x6ca>
1c01331c:	004007b7          	lui	a5,0x400
1c013320:	4501                	li	a0,0
1c013322:	7ff00f13          	li	t5,2047
1c013326:	4581                	li	a1,0
1c013328:	00879613          	slli	a2,a5,0x8
1c01332c:	00065963          	bgez	a2,1c01333e <__subdf3+0x324>
1c013330:	0f05                	addi	t5,t5,1
1c013332:	7ff00713          	li	a4,2047
1c013336:	50ef0f63          	beq	t5,a4,1c013854 <__subdf3+0x83a>
1c01333a:	c177b7b3          	p.bclr	a5,a5,0,23
1c01333e:	01d79713          	slli	a4,a5,0x1d
1c013342:	810d                	srli	a0,a0,0x3
1c013344:	7ff00693          	li	a3,2047
1c013348:	8f49                	or	a4,a4,a0
1c01334a:	838d                	srli	a5,a5,0x3
1c01334c:	00df1963          	bne	t5,a3,1c01335e <__subdf3+0x344>
1c013350:	8f5d                	or	a4,a4,a5
1c013352:	4781                	li	a5,0
1c013354:	c709                	beqz	a4,1c01335e <__subdf3+0x344>
1c013356:	000807b7          	lui	a5,0x80
1c01335a:	4701                	li	a4,0
1c01335c:	4581                	li	a1,0
1c01335e:	4681                	li	a3,0
1c013360:	e607a6b3          	p.insert	a3,a5,19,0
1c013364:	d54f26b3          	p.insert	a3,t5,10,20
1c013368:	c1f5a6b3          	p.insert	a3,a1,0,31
1c01336c:	853a                	mv	a0,a4
1c01336e:	85b6                	mv	a1,a3
1c013370:	500e0e63          	beqz	t3,1c01388c <__subdf3+0x872>
1c013374:	001e2073          	csrs	0x1,t3
1c013378:	8082                	ret
1c01337a:	4e01                	li	t3,0
1c01337c:	f8cf1ae3          	bne	t5,a2,1c013310 <__subdf3+0x2f6>
1c013380:	4e01                	li	t3,0
1c013382:	bfbd                	j	1c013300 <__subdf3+0x2e6>
1c013384:	7ff00613          	li	a2,2047
1c013388:	02c68163          	beq	a3,a2,1c0133aa <__subdf3+0x390>
1c01338c:	982a                	add	a6,a6,a0
1c01338e:	00a83533          	sltu	a0,a6,a0
1c013392:	97ba                	add	a5,a5,a4
1c013394:	97aa                	add	a5,a5,a0
1c013396:	01f79513          	slli	a0,a5,0x1f
1c01339a:	00185813          	srli	a6,a6,0x1
1c01339e:	01056533          	or	a0,a0,a6
1c0133a2:	8385                	srli	a5,a5,0x1
1c0133a4:	8f36                	mv	t5,a3
1c0133a6:	4e01                	li	t3,0
1c0133a8:	b715                	j	1c0132cc <__subdf3+0x2b2>
1c0133aa:	04030063          	beqz	t1,1c0133ea <__subdf3+0x3d0>
1c0133ae:	02333b63          	p.bneimm	t1,3,1c0133e4 <__subdf3+0x3ca>
1c0133b2:	cd85                	beqz	a1,1c0133ea <__subdf3+0x3d0>
1c0133b4:	57fd                	li	a5,-1
1c0133b6:	557d                	li	a0,-1
1c0133b8:	7fe00f13          	li	t5,2046
1c0133bc:	4e15                	li	t3,5
1c0133be:	c00e4e33          	p.bset	t3,t3,0,0
1c0133c2:	48232663          	p.beqimm	t1,2,1c01384e <__subdf3+0x834>
1c0133c6:	46332f63          	p.beqimm	t1,3,1c013844 <__subdf3+0x82a>
1c0133ca:	f00314e3          	bnez	t1,1c0132d2 <__subdf3+0x2b8>
1c0133ce:	f6453733          	p.bclr	a4,a0,27,4
1c0133d2:	f04720e3          	p.beqimm	a4,4,1c0132d2 <__subdf3+0x2b8>
1c0133d6:	00450713          	addi	a4,a0,4
1c0133da:	00a73533          	sltu	a0,a4,a0
1c0133de:	97aa                	add	a5,a5,a0
1c0133e0:	853a                	mv	a0,a4
1c0133e2:	bdc5                	j	1c0132d2 <__subdf3+0x2b8>
1c0133e4:	fc2338e3          	p.bneimm	t1,2,1c0133b4 <__subdf3+0x39a>
1c0133e8:	d5f1                	beqz	a1,1c0133b4 <__subdf3+0x39a>
1c0133ea:	7ff00f13          	li	t5,2047
1c0133ee:	a839                	j	1c01340c <__subdf3+0x3f2>
1c0133f0:	00030e63          	beqz	t1,1c01340c <__subdf3+0x3f2>
1c0133f4:	00333963          	p.bneimm	t1,3,1c013406 <__subdf3+0x3ec>
1c0133f8:	c991                	beqz	a1,1c01340c <__subdf3+0x3f2>
1c0133fa:	57fd                	li	a5,-1
1c0133fc:	557d                	li	a0,-1
1c0133fe:	7fe00f13          	li	t5,2046
1c013402:	4881                	li	a7,0
1c013404:	bf65                	j	1c0133bc <__subdf3+0x3a2>
1c013406:	fe233ae3          	p.bneimm	t1,2,1c0133fa <__subdf3+0x3e0>
1c01340a:	d9e5                	beqz	a1,1c0133fa <__subdf3+0x3e0>
1c01340c:	4781                	li	a5,0
1c01340e:	4501                	li	a0,0
1c013410:	4e15                	li	t3,5
1c013412:	bf19                	j	1c013328 <__subdf3+0x30e>
1c013414:	0f105363          	blez	a7,1c0134fa <__subdf3+0x4e0>
1c013418:	080f1663          	bnez	t5,1c0134a4 <__subdf3+0x48a>
1c01341c:	010766b3          	or	a3,a4,a6
1c013420:	e699                	bnez	a3,1c01342e <__subdf3+0x414>
1c013422:	7ff00713          	li	a4,2047
1c013426:	cae88ee3          	beq	a7,a4,1c0130e2 <__subdf3+0xc8>
1c01342a:	8f46                	mv	t5,a7
1c01342c:	b16d                	j	1c0130d6 <__subdf3+0xbc>
1c01342e:	fff88693          	addi	a3,a7,-1
1c013432:	e295                	bnez	a3,1c013456 <__subdf3+0x43c>
1c013434:	41050833          	sub	a6,a0,a6
1c013438:	8f99                	sub	a5,a5,a4
1c01343a:	01053533          	sltu	a0,a0,a6
1c01343e:	40a78733          	sub	a4,a5,a0
1c013442:	4f05                	li	t5,1
1c013444:	00871e93          	slli	t4,a4,0x8
1c013448:	0c0ed563          	bgez	t4,1c013512 <__subdf3+0x4f8>
1c01344c:	d1773633          	p.bclr	a2,a4,8,23
1c013450:	88c2                	mv	a7,a6
1c013452:	8efa                	mv	t4,t5
1c013454:	a4f9                	j	1c013722 <__subdf3+0x708>
1c013456:	7ff00613          	li	a2,2047
1c01345a:	cac885e3          	beq	a7,a2,1c013104 <__subdf3+0xea>
1c01345e:	03800613          	li	a2,56
1c013462:	08d64763          	blt	a2,a3,1c0134f0 <__subdf3+0x4d6>
1c013466:	467d                	li	a2,31
1c013468:	04d64f63          	blt	a2,a3,1c0134c6 <__subdf3+0x4ac>
1c01346c:	02000893          	li	a7,32
1c013470:	40d888b3          	sub	a7,a7,a3
1c013474:	00d85e33          	srl	t3,a6,a3
1c013478:	01171633          	sll	a2,a4,a7
1c01347c:	01181833          	sll	a6,a6,a7
1c013480:	01c66633          	or	a2,a2,t3
1c013484:	01003833          	snez	a6,a6
1c013488:	01066833          	or	a6,a2,a6
1c01348c:	00d756b3          	srl	a3,a4,a3
1c013490:	41050833          	sub	a6,a0,a6
1c013494:	40d786b3          	sub	a3,a5,a3
1c013498:	010537b3          	sltu	a5,a0,a6
1c01349c:	40f68733          	sub	a4,a3,a5
1c0134a0:	8f76                	mv	t5,t4
1c0134a2:	b74d                	j	1c013444 <__subdf3+0x42a>
1c0134a4:	7ff00693          	li	a3,2047
1c0134a8:	00de9b63          	bne	t4,a3,1c0134be <__subdf3+0x4a4>
1c0134ac:	00a7e733          	or	a4,a5,a0
1c0134b0:	36070063          	beqz	a4,1c013810 <__subdf3+0x7f6>
1c0134b4:	00979e93          	slli	t4,a5,0x9
1c0134b8:	7ff00f13          	li	t5,2047
1c0134bc:	b309                	j	1c0131be <__subdf3+0x1a4>
1c0134be:	c1774733          	p.bset	a4,a4,0,23
1c0134c2:	86c6                	mv	a3,a7
1c0134c4:	bf69                	j	1c01345e <__subdf3+0x444>
1c0134c6:	02000e13          	li	t3,32
1c0134ca:	00d75633          	srl	a2,a4,a3
1c0134ce:	4881                	li	a7,0
1c0134d0:	01c68863          	beq	a3,t3,1c0134e0 <__subdf3+0x4c6>
1c0134d4:	04000893          	li	a7,64
1c0134d8:	40d886b3          	sub	a3,a7,a3
1c0134dc:	00d718b3          	sll	a7,a4,a3
1c0134e0:	0108e833          	or	a6,a7,a6
1c0134e4:	01003833          	snez	a6,a6
1c0134e8:	01066833          	or	a6,a2,a6
1c0134ec:	4681                	li	a3,0
1c0134ee:	b74d                	j	1c013490 <__subdf3+0x476>
1c0134f0:	01076833          	or	a6,a4,a6
1c0134f4:	01003833          	snez	a6,a6
1c0134f8:	bfd5                	j	1c0134ec <__subdf3+0x4d2>
1c0134fa:	0e088a63          	beqz	a7,1c0135ee <__subdf3+0x5d4>
1c0134fe:	040e9d63          	bnez	t4,1c013558 <__subdf3+0x53e>
1c013502:	00a7e633          	or	a2,a5,a0
1c013506:	ea11                	bnez	a2,1c01351a <__subdf3+0x500>
1c013508:	7ff00793          	li	a5,2047
1c01350c:	85b6                	mv	a1,a3
1c01350e:	ccff0ce3          	beq	t5,a5,1c0131e6 <__subdf3+0x1cc>
1c013512:	ce0f1fe3          	bnez	t5,1c013210 <__subdf3+0x1f6>
1c013516:	8542                	mv	a0,a6
1c013518:	a211                	j	1c01361c <__subdf3+0x602>
1c01351a:	01f8bd63          	p.bneimm	a7,-1,1c013534 <__subdf3+0x51a>
1c01351e:	40a805b3          	sub	a1,a6,a0
1c013522:	40f707b3          	sub	a5,a4,a5
1c013526:	00b83733          	sltu	a4,a6,a1
1c01352a:	40e78733          	sub	a4,a5,a4
1c01352e:	882e                	mv	a6,a1
1c013530:	85b6                	mv	a1,a3
1c013532:	bf09                	j	1c013444 <__subdf3+0x42a>
1c013534:	7ff00613          	li	a2,2047
1c013538:	fff8c893          	not	a7,a7
1c01353c:	04cf1063          	bne	t5,a2,1c01357c <__subdf3+0x562>
1c013540:	01076533          	or	a0,a4,a6
1c013544:	2e050263          	beqz	a0,1c013828 <__subdf3+0x80e>
1c013548:	00971793          	slli	a5,a4,0x9
1c01354c:	2c07da63          	bgez	a5,1c013820 <__subdf3+0x806>
1c013550:	87ba                	mv	a5,a4
1c013552:	8542                	mv	a0,a6
1c013554:	85b6                	mv	a1,a3
1c013556:	b641                	j	1c0130d6 <__subdf3+0xbc>
1c013558:	7ff00613          	li	a2,2047
1c01355c:	00cf1c63          	bne	t5,a2,1c013574 <__subdf3+0x55a>
1c013560:	01076533          	or	a0,a4,a6
1c013564:	2c050263          	beqz	a0,1c013828 <__subdf3+0x80e>
1c013568:	00971e13          	slli	t3,a4,0x9
1c01356c:	87ba                	mv	a5,a4
1c01356e:	8542                	mv	a0,a6
1c013570:	85b6                	mv	a1,a3
1c013572:	b64d                	j	1c013114 <__subdf3+0xfa>
1c013574:	411008b3          	neg	a7,a7
1c013578:	c177c7b3          	p.bset	a5,a5,0,23
1c01357c:	03800613          	li	a2,56
1c013580:	07164363          	blt	a2,a7,1c0135e6 <__subdf3+0x5cc>
1c013584:	467d                	li	a2,31
1c013586:	03164d63          	blt	a2,a7,1c0135c0 <__subdf3+0x5a6>
1c01358a:	02000593          	li	a1,32
1c01358e:	411585b3          	sub	a1,a1,a7
1c013592:	01155e33          	srl	t3,a0,a7
1c013596:	00b79633          	sll	a2,a5,a1
1c01359a:	00b51533          	sll	a0,a0,a1
1c01359e:	01c66633          	or	a2,a2,t3
1c0135a2:	00a03533          	snez	a0,a0
1c0135a6:	8d51                	or	a0,a0,a2
1c0135a8:	0117d8b3          	srl	a7,a5,a7
1c0135ac:	40a80533          	sub	a0,a6,a0
1c0135b0:	411707b3          	sub	a5,a4,a7
1c0135b4:	00a83733          	sltu	a4,a6,a0
1c0135b8:	40e78733          	sub	a4,a5,a4
1c0135bc:	882a                	mv	a6,a0
1c0135be:	bf8d                	j	1c013530 <__subdf3+0x516>
1c0135c0:	02000e13          	li	t3,32
1c0135c4:	0117d633          	srl	a2,a5,a7
1c0135c8:	4581                	li	a1,0
1c0135ca:	01c88863          	beq	a7,t3,1c0135da <__subdf3+0x5c0>
1c0135ce:	04000593          	li	a1,64
1c0135d2:	411588b3          	sub	a7,a1,a7
1c0135d6:	011795b3          	sll	a1,a5,a7
1c0135da:	8d4d                	or	a0,a0,a1
1c0135dc:	00a03533          	snez	a0,a0
1c0135e0:	8d51                	or	a0,a0,a2
1c0135e2:	4881                	li	a7,0
1c0135e4:	b7e1                	j	1c0135ac <__subdf3+0x592>
1c0135e6:	8d5d                	or	a0,a0,a5
1c0135e8:	00a03533          	snez	a0,a0
1c0135ec:	bfdd                	j	1c0135e2 <__subdf3+0x5c8>
1c0135ee:	001e8613          	addi	a2,t4,1
1c0135f2:	e8b63633          	p.bclr	a2,a2,20,11
1c0135f6:	4e05                	li	t3,1
1c0135f8:	10ce4063          	blt	t3,a2,1c0136f8 <__subdf3+0x6de>
1c0135fc:	00a7efb3          	or	t6,a5,a0
1c013600:	01076633          	or	a2,a4,a6
1c013604:	0a0e9463          	bnez	t4,1c0136ac <__subdf3+0x692>
1c013608:	060f9063          	bnez	t6,1c013668 <__subdf3+0x64e>
1c01360c:	1c061763          	bnez	a2,1c0137da <__subdf3+0x7c0>
1c013610:	ffe30593          	addi	a1,t1,-2
1c013614:	0015b593          	seqz	a1,a1
1c013618:	4701                	li	a4,0
1c01361a:	4501                	li	a0,0
1c01361c:	00e567b3          	or	a5,a0,a4
1c013620:	20078f63          	beqz	a5,1c01383e <__subdf3+0x824>
1c013624:	01f55793          	srli	a5,a0,0x1f
1c013628:	00171893          	slli	a7,a4,0x1
1c01362c:	98be                	add	a7,a7,a5
1c01362e:	00151793          	slli	a5,a0,0x1
1c013632:	f837b6b3          	p.bclr	a3,a5,28,3
1c013636:	4e01                	li	t3,0
1c013638:	ce99                	beqz	a3,1c013656 <__subdf3+0x63c>
1c01363a:	1c232063          	p.beqimm	t1,2,1c0137fa <__subdf3+0x7e0>
1c01363e:	1a332563          	p.beqimm	t1,3,1c0137e8 <__subdf3+0x7ce>
1c013642:	4e05                	li	t3,1
1c013644:	00031963          	bnez	t1,1c013656 <__subdf3+0x63c>
1c013648:	f647b6b3          	p.bclr	a3,a5,27,4
1c01364c:	4e05                	li	t3,1
1c01364e:	ffc7b793          	sltiu	a5,a5,-4
1c013652:	1a46b063          	p.bneimm	a3,4,1c0137f2 <__subdf3+0x7d8>
1c013656:	0188d893          	srli	a7,a7,0x18
1c01365a:	0018c893          	xori	a7,a7,1
1c01365e:	fc18b8b3          	p.bclr	a7,a7,30,1
1c013662:	87ba                	mv	a5,a4
1c013664:	4f01                	li	t5,0
1c013666:	b19d                	j	1c0132cc <__subdf3+0x2b2>
1c013668:	16060b63          	beqz	a2,1c0137de <__subdf3+0x7c4>
1c01366c:	410508b3          	sub	a7,a0,a6
1c013670:	40e78633          	sub	a2,a5,a4
1c013674:	01153e33          	sltu	t3,a0,a7
1c013678:	41c60633          	sub	a2,a2,t3
1c01367c:	00861e93          	slli	t4,a2,0x8
1c013680:	000edc63          	bgez	t4,1c013698 <__subdf3+0x67e>
1c013684:	40a80533          	sub	a0,a6,a0
1c013688:	40f707b3          	sub	a5,a4,a5
1c01368c:	00a83733          	sltu	a4,a6,a0
1c013690:	40e78733          	sub	a4,a5,a4
1c013694:	85b6                	mv	a1,a3
1c013696:	b759                	j	1c01361c <__subdf3+0x602>
1c013698:	00c8e533          	or	a0,a7,a2
1c01369c:	14051363          	bnez	a0,1c0137e2 <__subdf3+0x7c8>
1c0136a0:	ffe30593          	addi	a1,t1,-2
1c0136a4:	0015b593          	seqz	a1,a1
1c0136a8:	4701                	li	a4,0
1c0136aa:	bf8d                	j	1c01361c <__subdf3+0x602>
1c0136ac:	7ff00293          	li	t0,2047
1c0136b0:	025e9d63          	bne	t4,t0,1c0136ea <__subdf3+0x6d0>
1c0136b4:	1c0f8163          	beqz	t6,1c013876 <__subdf3+0x85c>
1c0136b8:	00400e37          	lui	t3,0x400
1c0136bc:	01c7fe33          	and	t3,a5,t3
1c0136c0:	001e3e13          	seqz	t3,t3
1c0136c4:	0e12                	slli	t3,t3,0x4
1c0136c6:	03df1763          	bne	t5,t4,1c0136f4 <__subdf3+0x6da>
1c0136ca:	c611                	beqz	a2,1c0136d6 <__subdf3+0x6bc>
1c0136cc:	00971e93          	slli	t4,a4,0x9
1c0136d0:	000ec363          	bltz	t4,1c0136d6 <__subdf3+0x6bc>
1c0136d4:	4e41                	li	t3,16
1c0136d6:	000f9f63          	bnez	t6,1c0136f4 <__subdf3+0x6da>
1c0136da:	14060a63          	beqz	a2,1c01382e <__subdf3+0x814>
1c0136de:	87ba                	mv	a5,a4
1c0136e0:	8542                	mv	a0,a6
1c0136e2:	85b6                	mv	a1,a3
1c0136e4:	7ff00f13          	li	t5,2047
1c0136e8:	b6d5                	j	1c0132cc <__subdf3+0x2b2>
1c0136ea:	4e01                	li	t3,0
1c0136ec:	fe5f15e3          	bne	t5,t0,1c0136d6 <__subdf3+0x6bc>
1c0136f0:	4e01                	li	t3,0
1c0136f2:	bfe1                	j	1c0136ca <__subdf3+0x6b0>
1c0136f4:	da65                	beqz	a2,1c0136e4 <__subdf3+0x6ca>
1c0136f6:	b11d                	j	1c01331c <__subdf3+0x302>
1c0136f8:	410508b3          	sub	a7,a0,a6
1c0136fc:	01153e33          	sltu	t3,a0,a7
1c013700:	40e78633          	sub	a2,a5,a4
1c013704:	41c60633          	sub	a2,a2,t3
1c013708:	00861e13          	slli	t3,a2,0x8
1c01370c:	060e5c63          	bgez	t3,1c013784 <__subdf3+0x76a>
1c013710:	40a808b3          	sub	a7,a6,a0
1c013714:	40f707b3          	sub	a5,a4,a5
1c013718:	01183733          	sltu	a4,a6,a7
1c01371c:	40e78633          	sub	a2,a5,a4
1c013720:	85b6                	mv	a1,a3
1c013722:	c62d                	beqz	a2,1c01378c <__subdf3+0x772>
1c013724:	100616b3          	p.fl1	a3,a2
1c013728:	47fd                	li	a5,31
1c01372a:	40d786b3          	sub	a3,a5,a3
1c01372e:	ff868f13          	addi	t5,a3,-8
1c013732:	47fd                	li	a5,31
1c013734:	07e7c463          	blt	a5,t5,1c01379c <__subdf3+0x782>
1c013738:	02000793          	li	a5,32
1c01373c:	41e787b3          	sub	a5,a5,t5
1c013740:	01e61633          	sll	a2,a2,t5
1c013744:	00f8d733          	srl	a4,a7,a5
1c013748:	00c767b3          	or	a5,a4,a2
1c01374c:	01e89833          	sll	a6,a7,t5
1c013750:	09df4063          	blt	t5,t4,1c0137d0 <__subdf3+0x7b6>
1c013754:	41df0f33          	sub	t5,t5,t4
1c013758:	001f0713          	addi	a4,t5,1
1c01375c:	46fd                	li	a3,31
1c01375e:	04e6c563          	blt	a3,a4,1c0137a8 <__subdf3+0x78e>
1c013762:	02000693          	li	a3,32
1c013766:	8e99                	sub	a3,a3,a4
1c013768:	00e85633          	srl	a2,a6,a4
1c01376c:	00d79533          	sll	a0,a5,a3
1c013770:	00d81833          	sll	a6,a6,a3
1c013774:	8d51                	or	a0,a0,a2
1c013776:	01003833          	snez	a6,a6
1c01377a:	01056533          	or	a0,a0,a6
1c01377e:	00e7d733          	srl	a4,a5,a4
1c013782:	bd69                	j	1c01361c <__subdf3+0x602>
1c013784:	00c8e533          	or	a0,a7,a2
1c013788:	fd49                	bnez	a0,1c013722 <__subdf3+0x708>
1c01378a:	bf19                	j	1c0136a0 <__subdf3+0x686>
1c01378c:	100896b3          	p.fl1	a3,a7
1c013790:	47fd                	li	a5,31
1c013792:	40d786b3          	sub	a3,a5,a3
1c013796:	02068693          	addi	a3,a3,32
1c01379a:	bf51                	j	1c01372e <__subdf3+0x714>
1c01379c:	fd868793          	addi	a5,a3,-40
1c0137a0:	00f897b3          	sll	a5,a7,a5
1c0137a4:	4801                	li	a6,0
1c0137a6:	b76d                	j	1c013750 <__subdf3+0x736>
1c0137a8:	1f05                	addi	t5,t5,-31
1c0137aa:	02000693          	li	a3,32
1c0137ae:	01e7df33          	srl	t5,a5,t5
1c0137b2:	4601                	li	a2,0
1c0137b4:	00d70763          	beq	a4,a3,1c0137c2 <__subdf3+0x7a8>
1c0137b8:	04000613          	li	a2,64
1c0137bc:	8e19                	sub	a2,a2,a4
1c0137be:	00c79633          	sll	a2,a5,a2
1c0137c2:	00c86533          	or	a0,a6,a2
1c0137c6:	00a03533          	snez	a0,a0
1c0137ca:	00af6533          	or	a0,t5,a0
1c0137ce:	bde9                	j	1c0136a8 <__subdf3+0x68e>
1c0137d0:	41ee8f33          	sub	t5,t4,t5
1c0137d4:	c177b733          	p.bclr	a4,a5,0,23
1c0137d8:	bb2d                	j	1c013512 <__subdf3+0x4f8>
1c0137da:	8542                	mv	a0,a6
1c0137dc:	bd65                	j	1c013694 <__subdf3+0x67a>
1c0137de:	873e                	mv	a4,a5
1c0137e0:	bd35                	j	1c01361c <__subdf3+0x602>
1c0137e2:	8732                	mv	a4,a2
1c0137e4:	8546                	mv	a0,a7
1c0137e6:	bd1d                	j	1c01361c <__subdf3+0x602>
1c0137e8:	4e05                	li	t3,1
1c0137ea:	e60596e3          	bnez	a1,1c013656 <__subdf3+0x63c>
1c0137ee:	ff87b793          	sltiu	a5,a5,-8
1c0137f2:	0017c793          	xori	a5,a5,1
1c0137f6:	98be                	add	a7,a7,a5
1c0137f8:	bdb9                	j	1c013656 <__subdf3+0x63c>
1c0137fa:	4e05                	li	t3,1
1c0137fc:	e4058de3          	beqz	a1,1c013656 <__subdf3+0x63c>
1c013800:	b7fd                	j	1c0137ee <__subdf3+0x7d4>
1c013802:	87ba                	mv	a5,a4
1c013804:	8542                	mv	a0,a6
1c013806:	4881                	li	a7,0
1c013808:	b8c5                	j	1c0130f8 <__subdf3+0xde>
1c01380a:	87ba                	mv	a5,a4
1c01380c:	8542                	mv	a0,a6
1c01380e:	bdd9                	j	1c0136e4 <__subdf3+0x6ca>
1c013810:	4781                	li	a5,0
1c013812:	4501                	li	a0,0
1c013814:	7ff00f13          	li	t5,2047
1c013818:	4e01                	li	t3,0
1c01381a:	b639                	j	1c013328 <__subdf3+0x30e>
1c01381c:	4781                	li	a5,0
1c01381e:	bfed                	j	1c013818 <__subdf3+0x7fe>
1c013820:	87ba                	mv	a5,a4
1c013822:	8542                	mv	a0,a6
1c013824:	85b6                	mv	a1,a3
1c013826:	b7c5                	j	1c013806 <__subdf3+0x7ec>
1c013828:	4781                	li	a5,0
1c01382a:	85b6                	mv	a1,a3
1c01382c:	b7f5                	j	1c013818 <__subdf3+0x7fe>
1c01382e:	4501                	li	a0,0
1c013830:	004007b7          	lui	a5,0x400
1c013834:	7ff00f13          	li	t5,2047
1c013838:	4581                	li	a1,0
1c01383a:	4e41                	li	t3,16
1c01383c:	b4f5                	j	1c013328 <__subdf3+0x30e>
1c01383e:	4501                	li	a0,0
1c013840:	4f01                	li	t5,0
1c013842:	bfd9                	j	1c013818 <__subdf3+0x7fe>
1c013844:	a80597e3          	bnez	a1,1c0132d2 <__subdf3+0x2b8>
1c013848:	00850713          	addi	a4,a0,8
1c01384c:	b679                	j	1c0133da <__subdf3+0x3c0>
1c01384e:	a80582e3          	beqz	a1,1c0132d2 <__subdf3+0x2b8>
1c013852:	bfdd                	j	1c013848 <__subdf3+0x82e>
1c013854:	4501                	li	a0,0
1c013856:	00030c63          	beqz	t1,1c01386e <__subdf3+0x854>
1c01385a:	00333763          	p.bneimm	t1,3,1c013868 <__subdf3+0x84e>
1c01385e:	c981                	beqz	a1,1c01386e <__subdf3+0x854>
1c013860:	557d                	li	a0,-1
1c013862:	7fe00f13          	li	t5,2046
1c013866:	a021                	j	1c01386e <__subdf3+0x854>
1c013868:	fe233ce3          	p.bneimm	t1,2,1c013860 <__subdf3+0x846>
1c01386c:	d9f5                	beqz	a1,1c013860 <__subdf3+0x846>
1c01386e:	005e6e13          	ori	t3,t3,5
1c013872:	87aa                	mv	a5,a0
1c013874:	b4e9                	j	1c01333e <__subdf3+0x324>
1c013876:	4e01                	li	t3,0
1c013878:	e7df11e3          	bne	t5,t4,1c0136da <__subdf3+0x6c0>
1c01387c:	bd95                	j	1c0136f0 <__subdf3+0x6d6>
1c01387e:	b1df01e3          	beq	t5,t4,1c013380 <__subdf3+0x366>
1c013882:	87ba                	mv	a5,a4
1c013884:	8542                	mv	a0,a6
1c013886:	7ff00f13          	li	t5,2047
1c01388a:	be31                	j	1c0133a6 <__subdf3+0x38c>
1c01388c:	8082                	ret

1c01388e <__fixdfsi>:
1c01388e:	002027f3          	frrm	a5
1c013892:	3fe00793          	li	a5,1022
1c013896:	d54596b3          	p.extractu	a3,a1,10,20
1c01389a:	e6059633          	p.extractu	a2,a1,19,0
1c01389e:	81fd                	srli	a1,a1,0x1f
1c0138a0:	00d7c963          	blt	a5,a3,1c0138b2 <__fixdfsi+0x24>
1c0138a4:	eec1                	bnez	a3,1c01393c <__fixdfsi+0xae>
1c0138a6:	00a66733          	or	a4,a2,a0
1c0138aa:	4781                	li	a5,0
1c0138ac:	eb49                	bnez	a4,1c01393e <__fixdfsi+0xb0>
1c0138ae:	853e                	mv	a0,a5
1c0138b0:	8082                	ret
1c0138b2:	41d00793          	li	a5,1053
1c0138b6:	02d7d663          	ble	a3,a5,1c0138e2 <__fixdfsi+0x54>
1c0138ba:	80000737          	lui	a4,0x80000
1c0138be:	fff74713          	not	a4,a4
1c0138c2:	00e587b3          	add	a5,a1,a4
1c0138c6:	cdb5                	beqz	a1,1c013942 <__fixdfsi+0xb4>
1c0138c8:	41e00593          	li	a1,1054
1c0138cc:	4741                	li	a4,16
1c0138ce:	04b69163          	bne	a3,a1,1c013910 <__fixdfsi+0x82>
1c0138d2:	062e                	slli	a2,a2,0xb
1c0138d4:	01555693          	srli	a3,a0,0x15
1c0138d8:	8e55                	or	a2,a2,a3
1c0138da:	ea1d                	bnez	a2,1c013910 <__fixdfsi+0x82>
1c0138dc:	00b51713          	slli	a4,a0,0xb
1c0138e0:	b7f1                	j	1c0138ac <__fixdfsi+0x1e>
1c0138e2:	c14647b3          	p.bset	a5,a2,0,20
1c0138e6:	43300613          	li	a2,1075
1c0138ea:	8e15                	sub	a2,a2,a3
1c0138ec:	477d                	li	a4,31
1c0138ee:	02c74463          	blt	a4,a2,1c013916 <__fixdfsi+0x88>
1c0138f2:	bed68693          	addi	a3,a3,-1043
1c0138f6:	00d51733          	sll	a4,a0,a3
1c0138fa:	00d797b3          	sll	a5,a5,a3
1c0138fe:	00c55533          	srl	a0,a0,a2
1c013902:	00e03733          	snez	a4,a4
1c013906:	8fc9                	or	a5,a5,a0
1c013908:	c199                	beqz	a1,1c01390e <__fixdfsi+0x80>
1c01390a:	40f007b3          	neg	a5,a5
1c01390e:	d345                	beqz	a4,1c0138ae <__fixdfsi+0x20>
1c013910:	00172073          	csrs	0x1,a4
1c013914:	bf69                	j	1c0138ae <__fixdfsi+0x20>
1c013916:	02000813          	li	a6,32
1c01391a:	4701                	li	a4,0
1c01391c:	01060663          	beq	a2,a6,1c013928 <__fixdfsi+0x9a>
1c013920:	c0d68713          	addi	a4,a3,-1011
1c013924:	00e79733          	sll	a4,a5,a4
1c013928:	8f49                	or	a4,a4,a0
1c01392a:	41300513          	li	a0,1043
1c01392e:	40d506b3          	sub	a3,a0,a3
1c013932:	00e03733          	snez	a4,a4
1c013936:	00d7d7b3          	srl	a5,a5,a3
1c01393a:	b7f9                	j	1c013908 <__fixdfsi+0x7a>
1c01393c:	4781                	li	a5,0
1c01393e:	4705                	li	a4,1
1c013940:	bfc1                	j	1c013910 <__fixdfsi+0x82>
1c013942:	4741                	li	a4,16
1c013944:	b7f1                	j	1c013910 <__fixdfsi+0x82>

1c013946 <__fixunsdfsi>:
1c013946:	002027f3          	frrm	a5
1c01394a:	3fe00713          	li	a4,1022
1c01394e:	d54596b3          	p.extractu	a3,a1,10,20
1c013952:	e60597b3          	p.extractu	a5,a1,19,0
1c013956:	81fd                	srli	a1,a1,0x1f
1c013958:	00d74763          	blt	a4,a3,1c013966 <__fixunsdfsi+0x20>
1c01395c:	e6ad                	bnez	a3,1c0139c6 <__fixunsdfsi+0x80>
1c01395e:	8fc9                	or	a5,a5,a0
1c013960:	4501                	li	a0,0
1c013962:	e3bd                	bnez	a5,1c0139c8 <__fixunsdfsi+0x82>
1c013964:	8082                	ret
1c013966:	e1bd                	bnez	a1,1c0139cc <__fixunsdfsi+0x86>
1c013968:	41e00713          	li	a4,1054
1c01396c:	00d75463          	ble	a3,a4,1c013974 <__fixunsdfsi+0x2e>
1c013970:	557d                	li	a0,-1
1c013972:	a8b1                	j	1c0139ce <__fixunsdfsi+0x88>
1c013974:	43300613          	li	a2,1075
1c013978:	8e15                	sub	a2,a2,a3
1c01397a:	477d                	li	a4,31
1c01397c:	c147c7b3          	p.bset	a5,a5,0,20
1c013980:	02c74163          	blt	a4,a2,1c0139a2 <__fixunsdfsi+0x5c>
1c013984:	bed68693          	addi	a3,a3,-1043
1c013988:	00d51733          	sll	a4,a0,a3
1c01398c:	00d797b3          	sll	a5,a5,a3
1c013990:	00c55533          	srl	a0,a0,a2
1c013994:	00e03733          	snez	a4,a4
1c013998:	8d5d                	or	a0,a0,a5
1c01399a:	cf05                	beqz	a4,1c0139d2 <__fixunsdfsi+0x8c>
1c01399c:	00172073          	csrs	0x1,a4
1c0139a0:	8082                	ret
1c0139a2:	02000593          	li	a1,32
1c0139a6:	4701                	li	a4,0
1c0139a8:	00b60663          	beq	a2,a1,1c0139b4 <__fixunsdfsi+0x6e>
1c0139ac:	c0d68713          	addi	a4,a3,-1011
1c0139b0:	00e79733          	sll	a4,a5,a4
1c0139b4:	8f49                	or	a4,a4,a0
1c0139b6:	41300513          	li	a0,1043
1c0139ba:	8d15                	sub	a0,a0,a3
1c0139bc:	00e03733          	snez	a4,a4
1c0139c0:	00a7d533          	srl	a0,a5,a0
1c0139c4:	bfd9                	j	1c01399a <__fixunsdfsi+0x54>
1c0139c6:	4501                	li	a0,0
1c0139c8:	4705                	li	a4,1
1c0139ca:	bfc9                	j	1c01399c <__fixunsdfsi+0x56>
1c0139cc:	4501                	li	a0,0
1c0139ce:	4741                	li	a4,16
1c0139d0:	b7f1                	j	1c01399c <__fixunsdfsi+0x56>
1c0139d2:	8082                	ret

1c0139d4 <__floatsidf>:
1c0139d4:	c539                	beqz	a0,1c013a22 <__floatsidf+0x4e>
1c0139d6:	01f55613          	srli	a2,a0,0x1f
1c0139da:	00055463          	bgez	a0,1c0139e2 <__floatsidf+0xe>
1c0139de:	40a00533          	neg	a0,a0
1c0139e2:	477d                	li	a4,31
1c0139e4:	100517b3          	p.fl1	a5,a0
1c0139e8:	40f707b3          	sub	a5,a4,a5
1c0139ec:	41e00693          	li	a3,1054
1c0139f0:	4729                	li	a4,10
1c0139f2:	8e9d                	sub	a3,a3,a5
1c0139f4:	02f74263          	blt	a4,a5,1c013a18 <__floatsidf+0x44>
1c0139f8:	472d                	li	a4,11
1c0139fa:	8f1d                	sub	a4,a4,a5
1c0139fc:	07d5                	addi	a5,a5,21
1c0139fe:	00e55733          	srl	a4,a0,a4
1c013a02:	00f51533          	sll	a0,a0,a5
1c013a06:	4781                	li	a5,0
1c013a08:	e60727b3          	p.insert	a5,a4,19,0
1c013a0c:	d546a7b3          	p.insert	a5,a3,10,20
1c013a10:	c1f627b3          	p.insert	a5,a2,0,31
1c013a14:	85be                	mv	a1,a5
1c013a16:	8082                	ret
1c013a18:	17d5                	addi	a5,a5,-11
1c013a1a:	00f51733          	sll	a4,a0,a5
1c013a1e:	4501                	li	a0,0
1c013a20:	b7dd                	j	1c013a06 <__floatsidf+0x32>
1c013a22:	4701                	li	a4,0
1c013a24:	4501                	li	a0,0
1c013a26:	4681                	li	a3,0
1c013a28:	4601                	li	a2,0
1c013a2a:	bff1                	j	1c013a06 <__floatsidf+0x32>

1c013a2c <__floatunsidf>:
1c013a2c:	c131                	beqz	a0,1c013a70 <__floatunsidf+0x44>
1c013a2e:	477d                	li	a4,31
1c013a30:	100517b3          	p.fl1	a5,a0
1c013a34:	40f707b3          	sub	a5,a4,a5
1c013a38:	41e00693          	li	a3,1054
1c013a3c:	4729                	li	a4,10
1c013a3e:	8e9d                	sub	a3,a3,a5
1c013a40:	02f74363          	blt	a4,a5,1c013a66 <__floatunsidf+0x3a>
1c013a44:	472d                	li	a4,11
1c013a46:	8f1d                	sub	a4,a4,a5
1c013a48:	07d5                	addi	a5,a5,21
1c013a4a:	00e55733          	srl	a4,a0,a4
1c013a4e:	00f517b3          	sll	a5,a0,a5
1c013a52:	4601                	li	a2,0
1c013a54:	e6072633          	p.insert	a2,a4,19,0
1c013a58:	d546a633          	p.insert	a2,a3,10,20
1c013a5c:	c1f63733          	p.bclr	a4,a2,0,31
1c013a60:	853e                	mv	a0,a5
1c013a62:	85ba                	mv	a1,a4
1c013a64:	8082                	ret
1c013a66:	17d5                	addi	a5,a5,-11
1c013a68:	00f51733          	sll	a4,a0,a5
1c013a6c:	4781                	li	a5,0
1c013a6e:	b7d5                	j	1c013a52 <__floatunsidf+0x26>
1c013a70:	4701                	li	a4,0
1c013a72:	4781                	li	a5,0
1c013a74:	4681                	li	a3,0
1c013a76:	bff1                	j	1c013a52 <__floatunsidf+0x26>

1c013a78 <ResetTimer>:
1c013a78:	4701                	li	a4,0
1c013a7a:	cc071073          	csrw	0xcc0,a4
1c013a7e:	014027f3          	csrr	a5,uhartid
1c013a82:	f457b7b3          	p.bclr	a5,a5,26,5
1c013a86:	0297a363          	p.beqimm	a5,9,1c013aac <ResetTimer+0x34>
1c013a8a:	014027f3          	csrr	a5,uhartid
1c013a8e:	f457b7b3          	p.bclr	a5,a5,26,5
1c013a92:	0087b963          	p.bneimm	a5,8,1c013aa4 <ResetTimer+0x2c>
1c013a96:	002007b7          	lui	a5,0x200
1c013a9a:	4705                	li	a4,1
1c013a9c:	40078793          	addi	a5,a5,1024 # 200400 <__l1_heapsram_size+0x1e0470>
1c013aa0:	02e7a023          	sw	a4,32(a5)
1c013aa4:	4781                	li	a5,0
1c013aa6:	79f79073          	csrw	pccr31,a5
1c013aaa:	8082                	ret
1c013aac:	1a10c7b7          	lui	a5,0x1a10c
1c013ab0:	4685                	li	a3,1
1c013ab2:	80078793          	addi	a5,a5,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c013ab6:	02d7a023          	sw	a3,32(a5)
1c013aba:	79f71073          	csrw	pccr31,a4
1c013abe:	8082                	ret

1c013ac0 <StartTimer>:
1c013ac0:	014027f3          	csrr	a5,uhartid
1c013ac4:	f457b7b3          	p.bclr	a5,a5,26,5
1c013ac8:	0297a363          	p.beqimm	a5,9,1c013aee <StartTimer+0x2e>
1c013acc:	014027f3          	csrr	a5,uhartid
1c013ad0:	f457b7b3          	p.bclr	a5,a5,26,5
1c013ad4:	0087b963          	p.bneimm	a5,8,1c013ae6 <StartTimer+0x26>
1c013ad8:	002007b7          	lui	a5,0x200
1c013adc:	4705                	li	a4,1
1c013ade:	40078793          	addi	a5,a5,1024 # 200400 <__l1_heapsram_size+0x1e0470>
1c013ae2:	00e7ac23          	sw	a4,24(a5)
1c013ae6:	478d                	li	a5,3
1c013ae8:	cc179073          	csrw	0xcc1,a5
1c013aec:	8082                	ret
1c013aee:	1a10c7b7          	lui	a5,0x1a10c
1c013af2:	4705                	li	a4,1
1c013af4:	80078793          	addi	a5,a5,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c013af8:	b7ed                	j	1c013ae2 <StartTimer+0x22>

1c013afa <StopTimer>:
1c013afa:	014027f3          	csrr	a5,uhartid
1c013afe:	f457b7b3          	p.bclr	a5,a5,26,5
1c013b02:	0297a663          	p.beqimm	a5,9,1c013b2e <StopTimer+0x34>
1c013b06:	014027f3          	csrr	a5,uhartid
1c013b0a:	4701                	li	a4,0
1c013b0c:	cc171073          	csrw	0xcc1,a4
1c013b10:	f457b7b3          	p.bclr	a5,a5,26,5
1c013b14:	0087bc63          	p.bneimm	a5,8,1c013b2c <StopTimer+0x32>
1c013b18:	002007b7          	lui	a5,0x200
1c013b1c:	40078793          	addi	a5,a5,1024 # 200400 <__l1_heapsram_size+0x1e0470>
1c013b20:	0007a703          	lw	a4,0(a5)
1c013b24:	c0002733          	p.insert	a4,zero,0,0
1c013b28:	00e7a023          	sw	a4,0(a5)
1c013b2c:	8082                	ret
1c013b2e:	4781                	li	a5,0
1c013b30:	cc179073          	csrw	0xcc1,a5
1c013b34:	1a10c7b7          	lui	a5,0x1a10c
1c013b38:	80078793          	addi	a5,a5,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c013b3c:	b7d5                	j	1c013b20 <StopTimer+0x26>

1c013b3e <getCycles>:
1c013b3e:	014027f3          	csrr	a5,uhartid
1c013b42:	f457b7b3          	p.bclr	a5,a5,26,5
1c013b46:	0097a963          	p.beqimm	a5,9,1c013b58 <getCycles+0x1a>
1c013b4a:	00200537          	lui	a0,0x200
1c013b4e:	40050513          	addi	a0,a0,1024 # 200400 <__l1_heapsram_size+0x1e0470>
1c013b52:	00852503          	lw	a0,8(a0)
1c013b56:	8082                	ret
1c013b58:	1a10c537          	lui	a0,0x1a10c
1c013b5c:	80050513          	addi	a0,a0,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c013b60:	bfcd                	j	1c013b52 <getCycles+0x14>

1c013b62 <InitNetworkWrapper>:
1c013b62:	01402573          	csrr	a0,uhartid
1c013b66:	45a1                	li	a1,8
1c013b68:	f4553533          	p.bclr	a0,a0,26,5
1c013b6c:	0150006f          	j	1c014380 <InitNetwork>

1c013b70 <RunNetworkWrapper>:
1c013b70:	1141                	addi	sp,sp,-16
1c013b72:	c606                	sw	ra,12(sp)
1c013b74:	3711                	jal	1c013a78 <ResetTimer>
1c013b76:	37a9                	jal	1c013ac0 <StartTimer>
1c013b78:	01402573          	csrr	a0,uhartid
1c013b7c:	45a1                	li	a1,8
1c013b7e:	f4553533          	p.bclr	a0,a0,26,5
1c013b82:	788000ef          	jal	ra,1c01430a <RunNetwork>
1c013b86:	3f65                	jal	1c013b3e <getCycles>
1c013b88:	40b2                	lw	ra,12(sp)
1c013b8a:	1c01d7b7          	lui	a5,0x1c01d
1c013b8e:	76a7ac23          	sw	a0,1912(a5) # 1c01d778 <total_cycles>
1c013b92:	0141                	addi	sp,sp,16
1c013b94:	b79d                	j	1c013afa <StopTimer>

1c013b96 <CompareFloatOnCluster>:
1c013b96:	014027f3          	csrr	a5,uhartid
1c013b9a:	f457b7b3          	p.bclr	a5,a5,26,5
1c013b9e:	0e079963          	bnez	a5,1c013c90 <CompareFloatOnCluster+0xfa>
1c013ba2:	451c                	lw	a5,8(a0)
1c013ba4:	715d                	addi	sp,sp,-80
1c013ba6:	de4e                	sw	s3,60(sp)
1c013ba8:	c23e                	sw	a5,4(sp)
1c013baa:	455c                	lw	a5,12(a0)
1c013bac:	dc52                	sw	s4,56(sp)
1c013bae:	00052983          	lw	s3,0(a0)
1c013bb2:	c43e                	sw	a5,8(sp)
1c013bb4:	491c                	lw	a5,16(a0)
1c013bb6:	00452a03          	lw	s4,4(a0)
1c013bba:	c0ca                	sw	s2,64(sp)
1c013bbc:	1c01c937          	lui	s2,0x1c01c
1c013bc0:	c4a2                	sw	s0,72(sp)
1c013bc2:	c2a6                	sw	s1,68(sp)
1c013bc4:	d06a                	sw	s10,32(sp)
1c013bc6:	ce6e                	sw	s11,28(sp)
1c013bc8:	c686                	sw	ra,76(sp)
1c013bca:	da56                	sw	s5,52(sp)
1c013bcc:	d85a                	sw	s6,48(sp)
1c013bce:	d65e                	sw	s7,44(sp)
1c013bd0:	d462                	sw	s8,40(sp)
1c013bd2:	d266                	sw	s9,36(sp)
1c013bd4:	c63e                	sw	a5,12(sp)
1c013bd6:	4401                	li	s0,0
1c013bd8:	4481                	li	s1,0
1c013bda:	1c01cd37          	lui	s10,0x1c01c
1c013bde:	1c01cdb7          	lui	s11,0x1c01c
1c013be2:	b5090913          	addi	s2,s2,-1200 # 1c01bb50 <__clz_tab+0x108>
1c013be6:	4792                	lw	a5,4(sp)
1c013be8:	02f44363          	blt	s0,a5,1c013c0e <CompareFloatOnCluster+0x78>
1c013bec:	47b2                	lw	a5,12(sp)
1c013bee:	40b6                	lw	ra,76(sp)
1c013bf0:	4426                	lw	s0,72(sp)
1c013bf2:	c384                	sw	s1,0(a5)
1c013bf4:	4906                	lw	s2,64(sp)
1c013bf6:	4496                	lw	s1,68(sp)
1c013bf8:	59f2                	lw	s3,60(sp)
1c013bfa:	5a62                	lw	s4,56(sp)
1c013bfc:	5ad2                	lw	s5,52(sp)
1c013bfe:	5b42                	lw	s6,48(sp)
1c013c00:	5bb2                	lw	s7,44(sp)
1c013c02:	5c22                	lw	s8,40(sp)
1c013c04:	5c92                	lw	s9,36(sp)
1c013c06:	5d02                	lw	s10,32(sp)
1c013c08:	4df2                	lw	s11,28(sp)
1c013c0a:	6161                	addi	sp,sp,80
1c013c0c:	8082                	ret
1c013c0e:	0049ac0b          	p.lw	s8,4(s3!)
1c013c12:	004a2a8b          	p.lw	s5,4(s4!)
1c013c16:	095c7cd3          	fsub.s	fs9,fs8,fs5
1c013c1a:	8566                	mv	a0,s9
1c013c1c:	c64fd0ef          	jal	ra,1c011080 <__extendsfdf2>
1c013c20:	1c01c7b7          	lui	a5,0x1c01c
1c013c24:	b4878793          	addi	a5,a5,-1208 # 1c01bb48 <__clz_tab+0x100>
1c013c28:	4390                	lw	a2,0(a5)
1c013c2a:	43d4                	lw	a3,4(a5)
1c013c2c:	22051b13          	addi.d	s6,a0,0
1c013c30:	ba8fd0ef          	jal	ra,1c010fd8 <__ledf2>
1c013c34:	00054f63          	bltz	a0,1c013c52 <CompareFloatOnCluster+0xbc>
1c013c38:	00092603          	lw	a2,0(s2)
1c013c3c:	00492683          	lw	a3,4(s2)
1c013c40:	220b1513          	addi.d	a0,s6,0
1c013c44:	aecfd0ef          	jal	ra,1c010f30 <__gedf2>
1c013c48:	00a04563          	bgtz	a0,1c013c52 <CompareFloatOnCluster+0xbc>
1c013c4c:	a19ca7d3          	feq.s	a5,fs9,fs9
1c013c50:	ef95                	bnez	a5,1c013c8c <CompareFloatOnCluster+0xf6>
1c013c52:	8562                	mv	a0,s8
1c013c54:	0485                	addi	s1,s1,1
1c013c56:	c2afd0ef          	jal	ra,1c011080 <__extendsfdf2>
1c013c5a:	1c01c7b7          	lui	a5,0x1c01c
1c013c5e:	22051613          	addi.d	a2,a0,0
1c013c62:	b5878513          	addi	a0,a5,-1192 # 1c01bb58 <__clz_tab+0x110>
1c013c66:	217070ef          	jal	ra,1c01b67c <printf_>
1c013c6a:	8556                	mv	a0,s5
1c013c6c:	c14fd0ef          	jal	ra,1c011080 <__extendsfdf2>
1c013c70:	22051613          	addi.d	a2,a0,0
1c013c74:	b6cd0513          	addi	a0,s10,-1172 # 1c01bb6c <__clz_tab+0x124>
1c013c78:	205070ef          	jal	ra,1c01b67c <printf_>
1c013c7c:	47a2                	lw	a5,8(sp)
1c013c7e:	8722                	mv	a4,s0
1c013c80:	220b1613          	addi.d	a2,s6,0
1c013c84:	b80d8513          	addi	a0,s11,-1152 # 1c01bb80 <__clz_tab+0x138>
1c013c88:	1f5070ef          	jal	ra,1c01b67c <printf_>
1c013c8c:	0405                	addi	s0,s0,1
1c013c8e:	bfa1                	j	1c013be6 <CompareFloatOnCluster+0x50>
1c013c90:	8082                	ret

1c013c92 <CL_CompareFloat>:
1c013c92:	002047b7          	lui	a5,0x204
1c013c96:	0ff00713          	li	a4,255
1c013c9a:	08e7a223          	sw	a4,132(a5) # 204084 <__l1_heapsram_size+0x1e40f4>
1c013c9e:	20078713          	addi	a4,a5,512
1c013ca2:	0ff00693          	li	a3,255
1c013ca6:	00d72023          	sw	a3,0(a4) # 80000000 <pulp__FC+0x80000001>
1c013caa:	00d72623          	sw	a3,12(a4)
1c013cae:	22078793          	addi	a5,a5,544
1c013cb2:	1ff00713          	li	a4,511
1c013cb6:	00e7a023          	sw	a4,0(a5)
1c013cba:	00e7a623          	sw	a4,12(a5)
1c013cbe:	1c014737          	lui	a4,0x1c014
1c013cc2:	b9670713          	addi	a4,a4,-1130 # 1c013b96 <CompareFloatOnCluster>
1c013cc6:	002046b7          	lui	a3,0x204
1c013cca:	08e6a023          	sw	a4,128(a3) # 204080 <__l1_heapsram_size+0x1e40f0>
1c013cce:	00204737          	lui	a4,0x204
1c013cd2:	08a72023          	sw	a0,128(a4) # 204080 <__l1_heapsram_size+0x1e40f0>
1c013cd6:	01c7e703          	p.elw	a4,28(a5)
1c013cda:	8082                	ret

1c013cdc <main>:
1c013cdc:	7119                	addi	sp,sp,-128
1c013cde:	de86                	sw	ra,124(sp)
1c013ce0:	dca2                	sw	s0,120(sp)
1c013ce2:	daa6                	sw	s1,116(sp)
1c013ce4:	d8ca                	sw	s2,112(sp)
1c013ce6:	01402673          	csrr	a2,uhartid
1c013cea:	014025f3          	csrr	a1,uhartid
1c013cee:	1c01c537          	lui	a0,0x1c01c
1c013cf2:	8595                	srai	a1,a1,0x5
1c013cf4:	f4563633          	p.bclr	a2,a2,26,5
1c013cf8:	f265b5b3          	p.bclr	a1,a1,25,6
1c013cfc:	bac50513          	addi	a0,a0,-1108 # 1c01bbac <__clz_tab+0x164>
1c013d00:	17d070ef          	jal	ra,1c01b67c <printf_>
1c013d04:	1008                	addi	a0,sp,32
1c013d06:	2af000ef          	jal	ra,1c0147b4 <pi_cluster_conf_init>
1c013d0a:	1c01c437          	lui	s0,0x1c01c
1c013d0e:	100c                	addi	a1,sp,32
1c013d10:	3c440513          	addi	a0,s0,964 # 1c01c3c4 <cluster_dev>
1c013d14:	d002                	sw	zero,32(sp)
1c013d16:	445000ef          	jal	ra,1c01495a <pi_open_from_conf>
1c013d1a:	3c440513          	addi	a0,s0,964
1c013d1e:	7e8000ef          	jal	ra,1c014506 <pi_cluster_open>
1c013d22:	57fd                	li	a5,-1
1c013d24:	1a051a63          	bnez	a0,1c013ed8 <main+0x1fc>
1c013d28:	13c040ef          	jal	ra,1c017e64 <mem_init>
1c013d2c:	0e2040ef          	jal	ra,1c017e0e <open_fs>
1c013d30:	1c01c537          	lui	a0,0x1c01c
1c013d34:	bc450513          	addi	a0,a0,-1084 # 1c01bbc4 <__clz_tab+0x17c>
1c013d38:	145070ef          	jal	ra,1c01b67c <printf_>
1c013d3c:	1c0147b7          	lui	a5,0x1c014
1c013d40:	b6278793          	addi	a5,a5,-1182 # 1c013b62 <InitNetworkWrapper>
1c013d44:	c0be                	sw	a5,64(sp)
1c013d46:	47a1                	li	a5,8
1c013d48:	c8be                	sw	a5,80(sp)
1c013d4a:	6785                	lui	a5,0x1
1c013d4c:	ed878793          	addi	a5,a5,-296 # ed8 <__STACK_IRQ_SIZE+0x6d8>
1c013d50:	008c                	addi	a1,sp,64
1c013d52:	3c440513          	addi	a0,s0,964
1c013d56:	c6be                	sw	a5,76(sp)
1c013d58:	c282                	sw	zero,68(sp)
1c013d5a:	c482                	sw	zero,72(sp)
1c013d5c:	06010423          	sb	zero,104(sp)
1c013d60:	d282                	sw	zero,100(sp)
1c013d62:	2ef000ef          	jal	ra,1c014850 <pi_cluster_send_task_to_cl>
1c013d66:	1c01c537          	lui	a0,0x1c01c
1c013d6a:	bd450513          	addi	a0,a0,-1068 # 1c01bbd4 <__clz_tab+0x18c>
1c013d6e:	10f070ef          	jal	ra,1c01b67c <printf_>
1c013d72:	1c01d7b7          	lui	a5,0x1c01d
1c013d76:	7707a583          	lw	a1,1904(a5) # 1c01d770 <testInputVector>
1c013d7a:	77078493          	addi	s1,a5,1904
1c013d7e:	c185                	beqz	a1,1c013d9e <main+0xc2>
1c013d80:	1c01d7b7          	lui	a5,0x1c01d
1c013d84:	7c07a503          	lw	a0,1984(a5) # 1c01d7c0 <DeeployNetwork_inputs>
1c013d88:	e40007b7          	lui	a5,0xe4000
1c013d8c:	00200737          	lui	a4,0x200
1c013d90:	97aa                	add	a5,a5,a0
1c013d92:	14e7fa63          	bleu	a4,a5,1c013ee6 <main+0x20a>
1c013d96:	10000613          	li	a2,256
1c013d9a:	224060ef          	jal	ra,1c019fbe <memcpy>
1c013d9e:	40cc                	lw	a1,4(s1)
1c013da0:	c185                	beqz	a1,1c013dc0 <main+0xe4>
1c013da2:	1c01d7b7          	lui	a5,0x1c01d
1c013da6:	7c47a503          	lw	a0,1988(a5) # 1c01d7c4 <DeeployNetwork_inputs+0x4>
1c013daa:	e40007b7          	lui	a5,0xe4000
1c013dae:	00200737          	lui	a4,0x200
1c013db2:	97aa                	add	a5,a5,a0
1c013db4:	14e7f163          	bleu	a4,a5,1c013ef6 <main+0x21a>
1c013db8:	10000613          	li	a2,256
1c013dbc:	202060ef          	jal	ra,1c019fbe <memcpy>
1c013dc0:	1c01c537          	lui	a0,0x1c01c
1c013dc4:	be450513          	addi	a0,a0,-1052 # 1c01bbe4 <__clz_tab+0x19c>
1c013dc8:	0b5070ef          	jal	ra,1c01b67c <printf_>
1c013dcc:	1c0147b7          	lui	a5,0x1c014
1c013dd0:	b7078793          	addi	a5,a5,-1168 # 1c013b70 <RunNetworkWrapper>
1c013dd4:	c0be                	sw	a5,64(sp)
1c013dd6:	47a1                	li	a5,8
1c013dd8:	c8be                	sw	a5,80(sp)
1c013dda:	6785                	lui	a5,0x1
1c013ddc:	ed878793          	addi	a5,a5,-296 # ed8 <__STACK_IRQ_SIZE+0x6d8>
1c013de0:	008c                	addi	a1,sp,64
1c013de2:	3c440513          	addi	a0,s0,964
1c013de6:	c6be                	sw	a5,76(sp)
1c013de8:	c282                	sw	zero,68(sp)
1c013dea:	c482                	sw	zero,72(sp)
1c013dec:	06010423          	sb	zero,104(sp)
1c013df0:	d282                	sw	zero,100(sp)
1c013df2:	25f000ef          	jal	ra,1c014850 <pi_cluster_send_task_to_cl>
1c013df6:	1c01c537          	lui	a0,0x1c01c
1c013dfa:	bf450513          	addi	a0,a0,-1036 # 1c01bbf4 <__clz_tab+0x1ac>
1c013dfe:	1c01d937          	lui	s2,0x1c01d
1c013e02:	07b070ef          	jal	ra,1c01b67c <printf_>
1c013e06:	7d092483          	lw	s1,2000(s2) # 1c01d7d0 <DeeployNetwork_outputs>
1c013e0a:	e40007b7          	lui	a5,0xe4000
1c013e0e:	c402                	sw	zero,8(sp)
1c013e10:	97a6                	add	a5,a5,s1
1c013e12:	00200737          	lui	a4,0x200
1c013e16:	02e7e463          	bltu	a5,a4,1c013e3e <main+0x162>
1c013e1a:	f00007b7          	lui	a5,0xf0000
1c013e1e:	97a6                	add	a5,a5,s1
1c013e20:	00040737          	lui	a4,0x40
1c013e24:	00e7ed63          	bltu	a5,a4,1c013e3e <main+0x162>
1c013e28:	10000513          	li	a0,256
1c013e2c:	712060ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c013e30:	7d092583          	lw	a1,2000(s2)
1c013e34:	10000613          	li	a2,256
1c013e38:	84aa                	mv	s1,a0
1c013e3a:	0b4040ef          	jal	ra,1c017eee <ram_read>
1c013e3e:	1c0007b7          	lui	a5,0x1c000
1c013e42:	5907a783          	lw	a5,1424(a5) # 1c000590 <testOutputVector>
1c013e46:	3c440513          	addi	a0,s0,964
1c013e4a:	008c                	addi	a1,sp,64
1c013e4c:	c63e                	sw	a5,12(sp)
1c013e4e:	04000793          	li	a5,64
1c013e52:	ca3e                	sw	a5,20(sp)
1c013e54:	003c                	addi	a5,sp,8
1c013e56:	ce3e                	sw	a5,28(sp)
1c013e58:	1c0147b7          	lui	a5,0x1c014
1c013e5c:	c9278793          	addi	a5,a5,-878 # 1c013c92 <CL_CompareFloat>
1c013e60:	c0be                	sw	a5,64(sp)
1c013e62:	007c                	addi	a5,sp,12
1c013e64:	c2be                	sw	a5,68(sp)
1c013e66:	47a1                	li	a5,8
1c013e68:	c8be                	sw	a5,80(sp)
1c013e6a:	6785                	lui	a5,0x1
1c013e6c:	ed878793          	addi	a5,a5,-296 # ed8 <__STACK_IRQ_SIZE+0x6d8>
1c013e70:	c6be                	sw	a5,76(sp)
1c013e72:	c402                	sw	zero,8(sp)
1c013e74:	c826                	sw	s1,16(sp)
1c013e76:	cc02                	sw	zero,24(sp)
1c013e78:	c482                	sw	zero,72(sp)
1c013e7a:	06010423          	sb	zero,104(sp)
1c013e7e:	d282                	sw	zero,100(sp)
1c013e80:	1d1000ef          	jal	ra,1c014850 <pi_cluster_send_task_to_cl>
1c013e84:	7d092783          	lw	a5,2000(s2)
1c013e88:	e4000737          	lui	a4,0xe4000
1c013e8c:	002006b7          	lui	a3,0x200
1c013e90:	973e                	add	a4,a4,a5
1c013e92:	4422                	lw	s0,8(sp)
1c013e94:	00d76e63          	bltu	a4,a3,1c013eb0 <main+0x1d4>
1c013e98:	f0000737          	lui	a4,0xf0000
1c013e9c:	97ba                	add	a5,a5,a4
1c013e9e:	00040737          	lui	a4,0x40
1c013ea2:	00e7e763          	bltu	a5,a4,1c013eb0 <main+0x1d4>
1c013ea6:	10000593          	li	a1,256
1c013eaa:	8526                	mv	a0,s1
1c013eac:	598060ef          	jal	ra,1c01a444 <pi_l2_free>
1c013eb0:	1c01d7b7          	lui	a5,0x1c01d
1c013eb4:	7787a583          	lw	a1,1912(a5) # 1c01d778 <total_cycles>
1c013eb8:	1c01c537          	lui	a0,0x1c01c
1c013ebc:	c0050513          	addi	a0,a0,-1024 # 1c01bc00 <__clz_tab+0x1b8>
1c013ec0:	7bc070ef          	jal	ra,1c01b67c <printf_>
1c013ec4:	1c01c537          	lui	a0,0x1c01c
1c013ec8:	04000613          	li	a2,64
1c013ecc:	85a2                	mv	a1,s0
1c013ece:	c1850513          	addi	a0,a0,-1000 # 1c01bc18 <__clz_tab+0x1d0>
1c013ed2:	7aa070ef          	jal	ra,1c01b67c <printf_>
1c013ed6:	4781                	li	a5,0
1c013ed8:	50f6                	lw	ra,124(sp)
1c013eda:	5466                	lw	s0,120(sp)
1c013edc:	54d6                	lw	s1,116(sp)
1c013ede:	5946                	lw	s2,112(sp)
1c013ee0:	853e                	mv	a0,a5
1c013ee2:	6109                	addi	sp,sp,128
1c013ee4:	8082                	ret
1c013ee6:	f00007b7          	lui	a5,0xf0000
1c013eea:	97aa                	add	a5,a5,a0
1c013eec:	00040737          	lui	a4,0x40
1c013ef0:	eae7e3e3          	bltu	a5,a4,1c013d96 <main+0xba>
1c013ef4:	b56d                	j	1c013d9e <main+0xc2>
1c013ef6:	f00007b7          	lui	a5,0xf0000
1c013efa:	97aa                	add	a5,a5,a0
1c013efc:	00040737          	lui	a4,0x40
1c013f00:	eae7ece3          	bltu	a5,a4,1c013db8 <main+0xdc>
1c013f04:	bd75                	j	1c013dc0 <main+0xe4>

1c013f06 <mchan_transfer_wait>:
1c013f06:	4785                	li	a5,1
1c013f08:	00202737          	lui	a4,0x202
1c013f0c:	00a79533          	sll	a0,a5,a0
1c013f10:	80470713          	addi	a4,a4,-2044 # 201804 <__l1_heapsram_size+0x1e1874>
1c013f14:	10000613          	li	a2,256
1c013f18:	002046b7          	lui	a3,0x204
1c013f1c:	431c                	lw	a5,0(a4)
1c013f1e:	8fe9                	and	a5,a5,a0
1c013f20:	e391                	bnez	a5,1c013f24 <mchan_transfer_wait+0x1e>
1c013f22:	8082                	ret
1c013f24:	00c6a423          	sw	a2,8(a3) # 204008 <__l1_heapsram_size+0x1e4078>
1c013f28:	03c6e783          	p.elw	a5,60(a3)
1c013f2c:	00c6a223          	sw	a2,4(a3)
1c013f30:	b7f5                	j	1c013f1c <mchan_transfer_wait+0x16>

1c013f32 <_Add_0_cluster_fork>:
1c013f32:	00052803          	lw	a6,0(a0)
1c013f36:	00452883          	lw	a7,4(a0)
1c013f3a:	4508                	lw	a0,8(a0)
1c013f3c:	1c01c7b7          	lui	a5,0x1c01c
1c013f40:	60c78793          	addi	a5,a5,1548 # 1c01c60c <DeeployNetwork__Add_0_tiling_closure_args.73989>
1c013f44:	0107a023          	sw	a6,0(a5)
1c013f48:	0117a223          	sw	a7,4(a5)
1c013f4c:	c788                	sw	a0,8(a5)
1c013f4e:	014027f3          	csrr	a5,uhartid
1c013f52:	f457b7b3          	p.bclr	a5,a5,26,5
1c013f56:	04000713          	li	a4,64
1c013f5a:	078e                	slli	a5,a5,0x3
1c013f5c:	04e7d7b3          	p.minu	a5,a5,a4
1c013f60:	00878313          	addi	t1,a5,8
1c013f64:	04e35333          	p.minu	t1,t1,a4
1c013f68:	00279713          	slli	a4,a5,0x2
1c013f6c:	00e806b3          	add	a3,a6,a4
1c013f70:	00e88633          	add	a2,a7,a4
1c013f74:	972a                	add	a4,a4,a0
1c013f76:	00578593          	addi	a1,a5,5
1c013f7a:	0265e263          	bltu	a1,t1,1c013f9e <_Add_0_cluster_fork+0x6c>
1c013f7e:	00279613          	slli	a2,a5,0x2
1c013f82:	00c80733          	add	a4,a6,a2
1c013f86:	00c886b3          	add	a3,a7,a2
1c013f8a:	962a                	add	a2,a2,a0
1c013f8c:	0667ed63          	bltu	a5,t1,1c014006 <_Add_0_cluster_fork+0xd4>
1c013f90:	002047b7          	lui	a5,0x204
1c013f94:	20078793          	addi	a5,a5,512 # 204200 <__l1_heapsram_size+0x1e4270>
1c013f98:	01c7e703          	p.elw	a4,28(a5)
1c013f9c:	8082                	ret
1c013f9e:	00062e03          	lw	t3,0(a2) # 10000 <__heapfcram_size+0xdc0>
1c013fa2:	428c                	lw	a1,0(a3)
1c013fa4:	0799                	addi	a5,a5,6
1c013fa6:	06e1                	addi	a3,a3,24
1c013fa8:	01c5f5d3          	fadd.s	fa1,fa1,ft8
1c013fac:	0661                	addi	a2,a2,24
1c013fae:	0761                	addi	a4,a4,24
1c013fb0:	feb72423          	sw	a1,-24(a4)
1c013fb4:	fec62e03          	lw	t3,-20(a2)
1c013fb8:	fec6a583          	lw	a1,-20(a3)
1c013fbc:	01c5f5d3          	fadd.s	fa1,fa1,ft8
1c013fc0:	feb72623          	sw	a1,-20(a4)
1c013fc4:	ff062e03          	lw	t3,-16(a2)
1c013fc8:	ff06a583          	lw	a1,-16(a3)
1c013fcc:	01c5f5d3          	fadd.s	fa1,fa1,ft8
1c013fd0:	feb72823          	sw	a1,-16(a4)
1c013fd4:	ff462e03          	lw	t3,-12(a2)
1c013fd8:	ff46a583          	lw	a1,-12(a3)
1c013fdc:	01c5f5d3          	fadd.s	fa1,fa1,ft8
1c013fe0:	feb72a23          	sw	a1,-12(a4)
1c013fe4:	ff862e03          	lw	t3,-8(a2)
1c013fe8:	ff86a583          	lw	a1,-8(a3)
1c013fec:	01c5f5d3          	fadd.s	fa1,fa1,ft8
1c013ff0:	feb72c23          	sw	a1,-8(a4)
1c013ff4:	ffc6a583          	lw	a1,-4(a3)
1c013ff8:	ffc62e03          	lw	t3,-4(a2)
1c013ffc:	01c5f5d3          	fadd.s	fa1,fa1,ft8
1c014000:	feb72e23          	sw	a1,-4(a4)
1c014004:	bf8d                	j	1c013f76 <_Add_0_cluster_fork+0x44>
1c014006:	0047258b          	p.lw	a1,4(a4!)
1c01400a:	0046a50b          	p.lw	a0,4(a3!)
1c01400e:	0785                	addi	a5,a5,1
1c014010:	00a5f5d3          	fadd.s	fa1,fa1,fa0
1c014014:	00b6222b          	p.sw	a1,4(a2!)
1c014018:	bf95                	j	1c013f8c <_Add_0_cluster_fork+0x5a>

1c01401a <cl_wait_task>:
1c01401a:	4691                	li	a3,4
1c01401c:	00204737          	lui	a4,0x204
1c014020:	00054783          	lbu	a5,0(a0)
1c014024:	0ff7f793          	andi	a5,a5,255
1c014028:	c391                	beqz	a5,1c01402c <cl_wait_task+0x12>
1c01402a:	8082                	ret
1c01402c:	00d72423          	sw	a3,8(a4) # 204008 <__l1_heapsram_size+0x1e4078>
1c014030:	03c76783          	p.elw	a5,60(a4)
1c014034:	00d72223          	sw	a3,4(a4)
1c014038:	b7e5                	j	1c014020 <cl_wait_task+0x6>

1c01403a <_Add_0_closure_L3.constprop.1>:
1c01403a:	1c01c7b7          	lui	a5,0x1c01c
1c01403e:	5d078793          	addi	a5,a5,1488 # 1c01c5d0 <__data_end__>
1c014042:	4b94                	lw	a3,16(a5)
1c014044:	7145                	addi	sp,sp,-464
1c014046:	4bd8                	lw	a4,20(a5)
1c014048:	1b312e23          	sw	s3,444(sp)
1c01404c:	1b812423          	sw	s8,424(sp)
1c014050:	1b912223          	sw	s9,420(sp)
1c014054:	1ba12023          	sw	s10,416(sp)
1c014058:	0007a983          	lw	s3,0(a5)
1c01405c:	0047ac03          	lw	s8,4(a5)
1c014060:	0087ac83          	lw	s9,8(a5)
1c014064:	00c7ad03          	lw	s10,12(a5)
1c014068:	c436                	sw	a3,8(sp)
1c01406a:	4f94                	lw	a3,24(a5)
1c01406c:	4fdc                	lw	a5,28(a5)
1c01406e:	07800613          	li	a2,120
1c014072:	4581                	li	a1,0
1c014074:	1028                	addi	a0,sp,40
1c014076:	ca3a                	sw	a4,20(sp)
1c014078:	cc36                	sw	a3,24(sp)
1c01407a:	ce3e                	sw	a5,28(sp)
1c01407c:	1c112623          	sw	ra,460(sp)
1c014080:	1c812423          	sw	s0,456(sp)
1c014084:	1c912223          	sw	s1,452(sp)
1c014088:	1d212023          	sw	s2,448(sp)
1c01408c:	1b712623          	sw	s7,428(sp)
1c014090:	1b412c23          	sw	s4,440(sp)
1c014094:	1b512a23          	sw	s5,436(sp)
1c014098:	1b612823          	sw	s6,432(sp)
1c01409c:	19b12e23          	sw	s11,412(sp)
1c0140a0:	733050ef          	jal	ra,1c019fd2 <memset>
1c0140a4:	07800613          	li	a2,120
1c0140a8:	4581                	li	a1,0
1c0140aa:	1108                	addi	a0,sp,160
1c0140ac:	727050ef          	jal	ra,1c019fd2 <memset>
1c0140b0:	07800613          	li	a2,120
1c0140b4:	4581                	li	a1,0
1c0140b6:	0a28                	addi	a0,sp,280
1c0140b8:	71b050ef          	jal	ra,1c019fd2 <memset>
1c0140bc:	46a2                	lw	a3,8(sp)
1c0140be:	1c01c437          	lui	s0,0x1c01c
1c0140c2:	15040793          	addi	a5,s0,336 # 1c01c150 <__data_start__>
1c0140c6:	0006c703          	lbu	a4,0(a3)
1c0140ca:	1c01c4b7          	lui	s1,0x1c01c
1c0140ce:	1c01c937          	lui	s2,0x1c01c
1c0140d2:	97ba                	add	a5,a5,a4
1c0140d4:	0007cb83          	lbu	s7,0(a5)
1c0140d8:	15040413          	addi	s0,s0,336
1c0140dc:	5f048493          	addi	s1,s1,1520 # 1c01c5f0 <DeeployNetwork__Add_0_closure_args.74049>
1c0140e0:	60090913          	addi	s2,s2,1536 # 1c01c600 <DeeployNetwork__Add_0_cluster_fork_args.74017>
1c0140e4:	46a2                	lw	a3,8(sp)
1c0140e6:	0006c783          	lbu	a5,0(a3)
1c0140ea:	00f40733          	add	a4,s0,a5
1c0140ee:	00174703          	lbu	a4,1(a4)
1c0140f2:	04ebc163          	blt	s7,a4,1c014134 <_Add_0_closure_L3.constprop.1+0xfa>
1c0140f6:	0785                	addi	a5,a5,1
1c0140f8:	00f68023          	sb	a5,0(a3)
1c0140fc:	1cc12083          	lw	ra,460(sp)
1c014100:	1c812403          	lw	s0,456(sp)
1c014104:	1c412483          	lw	s1,452(sp)
1c014108:	1c012903          	lw	s2,448(sp)
1c01410c:	1bc12983          	lw	s3,444(sp)
1c014110:	1b812a03          	lw	s4,440(sp)
1c014114:	1b412a83          	lw	s5,436(sp)
1c014118:	1b012b03          	lw	s6,432(sp)
1c01411c:	1ac12b83          	lw	s7,428(sp)
1c014120:	1a812c03          	lw	s8,424(sp)
1c014124:	1a412c83          	lw	s9,420(sp)
1c014128:	1a012d03          	lw	s10,416(sp)
1c01412c:	19c12d83          	lw	s11,412(sp)
1c014130:	6179                	addi	sp,sp,464
1c014132:	8082                	ret
1c014134:	5b1030ef          	jal	ra,1c017ee4 <get_ram_ptr>
1c014138:	45d2                	lw	a1,20(sp)
1c01413a:	10000793          	li	a5,256
1c01413e:	02810893          	addi	a7,sp,40
1c014142:	4805                	li	a6,1
1c014144:	10000713          	li	a4,256
1c014148:	10000693          	li	a3,256
1c01414c:	8662                	mv	a2,s8
1c01414e:	143000ef          	jal	ra,1c014a90 <pi_cl_ram_copy_2d>
1c014152:	57d2                	lw	a5,52(sp)
1c014154:	c781                	beqz	a5,1c01415c <_Add_0_closure_L3.constprop.1+0x122>
1c014156:	0968                	addi	a0,sp,156
1c014158:	35c9                	jal	1c01401a <cl_wait_task>
1c01415a:	da02                	sw	zero,52(sp)
1c01415c:	589030ef          	jal	ra,1c017ee4 <get_ram_ptr>
1c014160:	45e2                	lw	a1,24(sp)
1c014162:	10000793          	li	a5,256
1c014166:	11810893          	addi	a7,sp,280
1c01416a:	4805                	li	a6,1
1c01416c:	10000713          	li	a4,256
1c014170:	10000693          	li	a3,256
1c014174:	8666                	mv	a2,s9
1c014176:	11b000ef          	jal	ra,1c014a90 <pi_cl_ram_copy_2d>
1c01417a:	12412783          	lw	a5,292(sp)
1c01417e:	c789                	beqz	a5,1c014188 <_Add_0_closure_L3.constprop.1+0x14e>
1c014180:	0368                	addi	a0,sp,396
1c014182:	3d61                	jal	1c01401a <cl_wait_task>
1c014184:	12012223          	sw	zero,292(sp)
1c014188:	57d2                	lw	a5,52(sp)
1c01418a:	c781                	beqz	a5,1c014192 <_Add_0_closure_L3.constprop.1+0x158>
1c01418c:	0968                	addi	a0,sp,156
1c01418e:	3571                	jal	1c01401a <cl_wait_task>
1c014190:	da02                	sw	zero,52(sp)
1c014192:	12412783          	lw	a5,292(sp)
1c014196:	c789                	beqz	a5,1c0141a0 <_Add_0_closure_L3.constprop.1+0x166>
1c014198:	0368                	addi	a0,sp,396
1c01419a:	3541                	jal	1c01401a <cl_wait_task>
1c01419c:	12012223          	sw	zero,292(sp)
1c0141a0:	1c01d7b7          	lui	a5,0x1c01d
1c0141a4:	7cc7aa03          	lw	s4,1996(a5) # 1c01d7cc <DeeployNetwork_MEMORYARENA_L1>
1c0141a8:	0134a023          	sw	s3,0(s1)
1c0141ac:	0184a223          	sw	s8,4(s1)
1c0141b0:	100a0793          	addi	a5,s4,256
1c0141b4:	0194a423          	sw	s9,8(s1)
1c0141b8:	01a4a623          	sw	s10,12(s1)
1c0141bc:	c83e                	sw	a5,16(sp)
1c0141be:	0009c783          	lbu	a5,0(s3)
1c0141c2:	200a0713          	addi	a4,s4,512
1c0141c6:	00202ab7          	lui	s5,0x202
1c0141ca:	97a2                	add	a5,a5,s0
1c0141cc:	0047cd83          	lbu	s11,4(a5)
1c0141d0:	c63a                	sw	a4,12(sp)
1c0141d2:	800a8b13          	addi	s6,s5,-2048 # 201800 <__l1_heapsram_size+0x1e1870>
1c0141d6:	0009c703          	lbu	a4,0(s3)
1c0141da:	00e406b3          	add	a3,s0,a4
1c0141de:	0056c683          	lbu	a3,5(a3)
1c0141e2:	04ddc063          	blt	s11,a3,1c014222 <_Add_0_closure_L3.constprop.1+0x1e8>
1c0141e6:	0705                	addi	a4,a4,1
1c0141e8:	00e98023          	sb	a4,0(s3)
1c0141ec:	4f9030ef          	jal	ra,1c017ee4 <get_ram_ptr>
1c0141f0:	45f2                	lw	a1,28(sp)
1c0141f2:	10000793          	li	a5,256
1c0141f6:	0a010893          	addi	a7,sp,160
1c0141fa:	4801                	li	a6,0
1c0141fc:	10000713          	li	a4,256
1c014200:	10000693          	li	a3,256
1c014204:	866a                	mv	a2,s10
1c014206:	08b000ef          	jal	ra,1c014a90 <pi_cl_ram_copy_2d>
1c01420a:	57ba                	lw	a5,172(sp)
1c01420c:	c781                	beqz	a5,1c014214 <_Add_0_closure_L3.constprop.1+0x1da>
1c01420e:	0a48                	addi	a0,sp,276
1c014210:	3529                	jal	1c01401a <cl_wait_task>
1c014212:	d702                	sw	zero,172(sp)
1c014214:	57ba                	lw	a5,172(sp)
1c014216:	c781                	beqz	a5,1c01421e <_Add_0_closure_L3.constprop.1+0x1e4>
1c014218:	0a48                	addi	a0,sp,276
1c01421a:	3501                	jal	1c01401a <cl_wait_task>
1c01421c:	d702                	sw	zero,172(sp)
1c01421e:	0b85                	addi	s7,s7,1
1c014220:	b5d1                	j	1c0140e4 <_Add_0_closure_L3.constprop.1+0xaa>
1c014222:	47b2                	lw	a5,12(sp)
1c014224:	00160737          	lui	a4,0x160
1c014228:	10070713          	addi	a4,a4,256 # 160100 <__l1_heapsram_size+0x140170>
1c01422c:	000b2683          	lw	a3,0(s6)
1c014230:	00eb2023          	sw	a4,0(s6)
1c014234:	00fb2023          	sw	a5,0(s6)
1c014238:	001607b7          	lui	a5,0x160
1c01423c:	018b2023          	sw	s8,0(s6)
1c014240:	10078793          	addi	a5,a5,256 # 160100 <__l1_heapsram_size+0x140170>
1c014244:	000b2703          	lw	a4,0(s6)
1c014248:	00fb2023          	sw	a5,0(s6)
1c01424c:	47c2                	lw	a5,16(sp)
1c01424e:	00fb2023          	sw	a5,0(s6)
1c014252:	019b2023          	sw	s9,0(s6)
1c014256:	0006ce63          	bltz	a3,1c014272 <_Add_0_closure_L3.constprop.1+0x238>
1c01425a:	8536                	mv	a0,a3
1c01425c:	c236                	sw	a3,4(sp)
1c01425e:	c03a                	sw	a4,0(sp)
1c014260:	315d                	jal	1c013f06 <mchan_transfer_wait>
1c014262:	4692                	lw	a3,4(sp)
1c014264:	4505                	li	a0,1
1c014266:	4702                	lw	a4,0(sp)
1c014268:	00d51533          	sll	a0,a0,a3
1c01426c:	804a8693          	addi	a3,s5,-2044
1c014270:	c288                	sw	a0,0(a3)
1c014272:	00074c63          	bltz	a4,1c01428a <_Add_0_closure_L3.constprop.1+0x250>
1c014276:	853a                	mv	a0,a4
1c014278:	c03a                	sw	a4,0(sp)
1c01427a:	3171                	jal	1c013f06 <mchan_transfer_wait>
1c01427c:	4702                	lw	a4,0(sp)
1c01427e:	4505                	li	a0,1
1c014280:	00e51533          	sll	a0,a0,a4
1c014284:	804a8713          	addi	a4,s5,-2044
1c014288:	c308                	sw	a0,0(a4)
1c01428a:	46b2                	lw	a3,12(sp)
1c01428c:	4742                	lw	a4,16(sp)
1c01428e:	01492423          	sw	s4,8(s2)
1c014292:	00d92023          	sw	a3,0(s2)
1c014296:	00e92223          	sw	a4,4(s2)
1c01429a:	0ff00693          	li	a3,255
1c01429e:	00204737          	lui	a4,0x204
1c0142a2:	08d72223          	sw	a3,132(a4) # 204084 <__l1_heapsram_size+0x1e40f4>
1c0142a6:	20070613          	addi	a2,a4,512
1c0142aa:	00d62023          	sw	a3,0(a2)
1c0142ae:	00d62623          	sw	a3,12(a2)
1c0142b2:	1ff00613          	li	a2,511
1c0142b6:	22070693          	addi	a3,a4,544
1c0142ba:	00c6a023          	sw	a2,0(a3)
1c0142be:	00c6a623          	sw	a2,12(a3)
1c0142c2:	1c0147b7          	lui	a5,0x1c014
1c0142c6:	f3278793          	addi	a5,a5,-206 # 1c013f32 <_Add_0_cluster_fork>
1c0142ca:	08f72023          	sw	a5,128(a4)
1c0142ce:	09272023          	sw	s2,128(a4)
1c0142d2:	01c6e703          	p.elw	a4,28(a3)
1c0142d6:	001406b7          	lui	a3,0x140
1c0142da:	10068693          	addi	a3,a3,256 # 140100 <__l1_heapsram_size+0x120170>
1c0142de:	000b2703          	lw	a4,0(s6)
1c0142e2:	00db2023          	sw	a3,0(s6)
1c0142e6:	014b2023          	sw	s4,0(s6)
1c0142ea:	01ab2023          	sw	s10,0(s6)
1c0142ee:	00074c63          	bltz	a4,1c014306 <_Add_0_closure_L3.constprop.1+0x2cc>
1c0142f2:	853a                	mv	a0,a4
1c0142f4:	c03a                	sw	a4,0(sp)
1c0142f6:	3901                	jal	1c013f06 <mchan_transfer_wait>
1c0142f8:	4702                	lw	a4,0(sp)
1c0142fa:	4505                	li	a0,1
1c0142fc:	00e51533          	sll	a0,a0,a4
1c014300:	804a8713          	addi	a4,s5,-2044
1c014304:	c308                	sw	a0,0(a4)
1c014306:	0d85                	addi	s11,s11,1
1c014308:	b5f9                	j	1c0141d6 <_Add_0_closure_L3.constprop.1+0x19c>

1c01430a <RunNetwork>:
1c01430a:	1c01d7b7          	lui	a5,0x1c01d
1c01430e:	1101                	addi	sp,sp,-32
1c014310:	7b87a583          	lw	a1,1976(a5) # 1c01d7b8 <DeeployNetwork_MEMORYARENA_L2>
1c014314:	1c01c7b7          	lui	a5,0x1c01c
1c014318:	ce06                	sw	ra,28(sp)
1c01431a:	15078793          	addi	a5,a5,336 # 1c01c150 <__data_start__>
1c01431e:	1c01d6b7          	lui	a3,0x1c01d
1c014322:	0087c703          	lbu	a4,8(a5)
1c014326:	7b46a603          	lw	a2,1972(a3) # 1c01d7b4 <DeeployNetwork_input_0>
1c01432a:	1c01d6b7          	lui	a3,0x1c01d
1c01432e:	7c86a683          	lw	a3,1992(a3) # 1c01d7c8 <DeeployNetwork_input_1>
1c014332:	963a                	add	a2,a2,a4
1c014334:	00c7c703          	lbu	a4,12(a5)
1c014338:	0107c783          	lbu	a5,16(a5)
1c01433c:	00e10513          	addi	a0,sp,14
1c014340:	96ba                	add	a3,a3,a4
1c014342:	1c01d737          	lui	a4,0x1c01d
1c014346:	7bc72703          	lw	a4,1980(a4) # 1c01d7bc <DeeployNetwork_output_0>
1c01434a:	00010723          	sb	zero,14(sp)
1c01434e:	000107a3          	sb	zero,15(sp)
1c014352:	973e                	add	a4,a4,a5
1c014354:	1c01c7b7          	lui	a5,0x1c01c
1c014358:	5d078793          	addi	a5,a5,1488 # 1c01c5d0 <__data_end__>
1c01435c:	c388                	sw	a0,0(a5)
1c01435e:	20058513          	addi	a0,a1,512
1c014362:	c3c8                	sw	a0,4(a5)
1c014364:	c7cc                	sw	a1,12(a5)
1c014366:	10058513          	addi	a0,a1,256
1c01436a:	00f10593          	addi	a1,sp,15
1c01436e:	c788                	sw	a0,8(a5)
1c014370:	cb8c                	sw	a1,16(a5)
1c014372:	cbd0                	sw	a2,20(a5)
1c014374:	cf94                	sw	a3,24(a5)
1c014376:	cfd8                	sw	a4,28(a5)
1c014378:	31c9                	jal	1c01403a <_Add_0_closure_L3.constprop.1>
1c01437a:	40f2                	lw	ra,28(sp)
1c01437c:	6105                	addi	sp,sp,32
1c01437e:	8082                	ret

1c014380 <InitNetwork>:
1c014380:	1141                	addi	sp,sp,-16
1c014382:	30000593          	li	a1,768
1c014386:	4501                	li	a0,0
1c014388:	c606                	sw	ra,12(sp)
1c01438a:	c422                	sw	s0,8(sp)
1c01438c:	23a9                	jal	1c0148d6 <pi_cl_l1_malloc>
1c01438e:	1c01d7b7          	lui	a5,0x1c01d
1c014392:	7ca7a623          	sw	a0,1996(a5) # 1c01d7cc <DeeployNetwork_MEMORYARENA_L1>
1c014396:	30000513          	li	a0,768
1c01439a:	1a4060ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c01439e:	1c01d7b7          	lui	a5,0x1c01d
1c0143a2:	7aa7ac23          	sw	a0,1976(a5) # 1c01d7b8 <DeeployNetwork_MEMORYARENA_L2>
1c0143a6:	30000513          	li	a0,768
1c0143aa:	381030ef          	jal	ra,1c017f2a <cl_ram_malloc>
1c0143ae:	1c01d737          	lui	a4,0x1c01d
1c0143b2:	87aa                	mv	a5,a0
1c0143b4:	7aa72823          	sw	a0,1968(a4) # 1c01d7b0 <DeeployNetwork_MEMORYARENA_L3>
1c0143b8:	20050513          	addi	a0,a0,512
1c0143bc:	1c01d737          	lui	a4,0x1c01d
1c0143c0:	7aa72a23          	sw	a0,1972(a4) # 1c01d7b4 <DeeployNetwork_input_0>
1c0143c4:	1c01d737          	lui	a4,0x1c01d
1c0143c8:	7af72e23          	sw	a5,1980(a4) # 1c01d7bc <DeeployNetwork_output_0>
1c0143cc:	1c01d737          	lui	a4,0x1c01d
1c0143d0:	10078693          	addi	a3,a5,256
1c0143d4:	7c070713          	addi	a4,a4,1984 # 1c01d7c0 <DeeployNetwork_inputs>
1c0143d8:	1c01c5b7          	lui	a1,0x1c01c
1c0143dc:	1c01d437          	lui	s0,0x1c01d
1c0143e0:	c308                	sw	a0,0(a4)
1c0143e2:	c354                	sw	a3,4(a4)
1c0143e4:	c3058593          	addi	a1,a1,-976 # 1c01bc30 <__clz_tab+0x1e8>
1c0143e8:	1c01d737          	lui	a4,0x1c01d
1c0143ec:	7cd42423          	sw	a3,1992(s0) # 1c01d7c8 <DeeployNetwork_input_1>
1c0143f0:	7cf72823          	sw	a5,2000(a4) # 1c01d7d0 <DeeployNetwork_outputs>
1c0143f4:	37f030ef          	jal	ra,1c017f72 <load_file_to_ram>
1c0143f8:	7c842503          	lw	a0,1992(s0)
1c0143fc:	4422                	lw	s0,8(sp)
1c0143fe:	40b2                	lw	ra,12(sp)
1c014400:	1c01c5b7          	lui	a1,0x1c01c
1c014404:	c3858593          	addi	a1,a1,-968 # 1c01bc38 <__clz_tab+0x1f0>
1c014408:	0141                	addi	sp,sp,16
1c01440a:	3690306f          	j	1c017f72 <load_file_to_ram>

1c01440e <pi_task_remote_enqueue>:
1c01440e:	1c01d737          	lui	a4,0x1c01d
1c014412:	7fc70793          	addi	a5,a4,2044 # 1c01d7fc <pi_cluster>
1c014416:	4f88                	lw	a0,24(a5)
1c014418:	c915                	beqz	a0,1c01444c <pi_task_remote_enqueue+0x3e>
1c01441a:	4794                	lw	a3,8(a5)
1c01441c:	4290                	lw	a2,0(a3)
1c01441e:	e219                	bnez	a2,1c014424 <pi_task_remote_enqueue+0x16>
1c014420:	7e072e23          	sw	zero,2044(a4)
1c014424:	42d8                	lw	a4,4(a3)
1c014426:	e319                	bnez	a4,1c01442c <pi_task_remote_enqueue+0x1e>
1c014428:	0007a223          	sw	zero,4(a5)
1c01442c:	0007ac23          	sw	zero,24(a5)
1c014430:	4bdc                	lw	a5,20(a5)
1c014432:	0007a023          	sw	zero,0(a5)
1c014436:	fc1537b3          	p.bclr	a5,a0,30,1
1c01443a:	c799                	beqz	a5,1c014448 <pi_task_remote_enqueue+0x3a>
1c01443c:	c0053533          	p.bclr	a0,a0,0,0
1c014440:	00452303          	lw	t1,4(a0)
1c014444:	4508                	lw	a0,8(a0)
1c014446:	8302                	jr	t1
1c014448:	2d90506f          	j	1c019f20 <__pi_os_evt_push>
1c01444c:	8082                	ret

1c01444e <pi_cluster_close>:
1c01444e:	451c                	lw	a5,8(a0)
1c014450:	1141                	addi	sp,sp,-16
1c014452:	c226                	sw	s1,4(sp)
1c014454:	c606                	sw	ra,12(sp)
1c014456:	c422                	sw	s0,8(sp)
1c014458:	1c01e4b7          	lui	s1,0x1c01e
1c01445c:	0287c403          	lbu	s0,40(a5)
1c014460:	55fd                	li	a1,-1
1c014462:	82848513          	addi	a0,s1,-2008 # 1c01d828 <g_cluster_open_close_mutex>
1c014466:	423040ef          	jal	ra,1c019088 <os_sync_obj_take>
1c01446a:	1c01d737          	lui	a4,0x1c01d
1c01446e:	0406                	slli	s0,s0,0x1
1c014470:	7d470713          	addi	a4,a4,2004 # 1c01d7d4 <pi_cluster_nb_open>
1c014474:	9722                	add	a4,a4,s0
1c014476:	00075783          	lhu	a5,0(a4)
1c01447a:	17fd                	addi	a5,a5,-1
1c01447c:	1007d7b3          	p.exthz	a5,a5
1c014480:	00f71023          	sh	a5,0(a4)
1c014484:	cb99                	beqz	a5,1c01449a <pi_cluster_close+0x4c>
1c014486:	82848513          	addi	a0,s1,-2008
1c01448a:	549040ef          	jal	ra,1c0191d2 <os_sync_obj_release>
1c01448e:	40b2                	lw	ra,12(sp)
1c014490:	4422                	lw	s0,8(sp)
1c014492:	4492                	lw	s1,4(sp)
1c014494:	4501                	li	a0,0
1c014496:	0141                	addi	sp,sp,16
1c014498:	8082                	ret
1c01449a:	4601                	li	a2,0
1c01449c:	4581                	li	a1,0
1c01449e:	450d                	li	a0,3
1c0144a0:	7bc000ef          	jal	ra,1c014c5c <pi_pmu_domain_state_change>
1c0144a4:	4509                	li	a0,2
1c0144a6:	fd6fc0ef          	jal	ra,1c010c7c <__pi_fll_deinit>
1c0144aa:	bff1                	j	1c014486 <pi_cluster_close+0x38>

1c0144ac <__pi_cluster_task_set_stack.isra.4>:
1c0144ac:	1101                	addi	sp,sp,-32
1c0144ae:	cc22                	sw	s0,24(sp)
1c0144b0:	4500                	lw	s0,8(a0)
1c0144b2:	c64e                	sw	s3,12(sp)
1c0144b4:	89ae                	mv	s3,a1
1c0144b6:	480c                	lw	a1,16(s0)
1c0144b8:	ca26                	sw	s1,20(sp)
1c0144ba:	c84a                	sw	s2,16(sp)
1c0144bc:	ce06                	sw	ra,28(sp)
1c0144be:	892a                	mv	s2,a0
1c0144c0:	84b2                	mv	s1,a2
1c0144c2:	cd99                	beqz	a1,1c0144e0 <__pi_cluster_task_set_stack.isra.4+0x34>
1c0144c4:	505c                	lw	a5,36(s0)
1c0144c6:	00c79d63          	bne	a5,a2,1c0144e0 <__pi_cluster_task_set_stack.isra.4+0x34>
1c0144ca:	481c                	lw	a5,16(s0)
1c0144cc:	4501                	li	a0,0
1c0144ce:	00f9a023          	sw	a5,0(s3)
1c0144d2:	40f2                	lw	ra,28(sp)
1c0144d4:	4462                	lw	s0,24(sp)
1c0144d6:	44d2                	lw	s1,20(sp)
1c0144d8:	4942                	lw	s2,16(sp)
1c0144da:	49b2                	lw	s3,12(sp)
1c0144dc:	6105                	addi	sp,sp,32
1c0144de:	8082                	ret
1c0144e0:	401c                	lw	a5,0(s0)
1c0144e2:	c799                	beqz	a5,1c0144f0 <__pi_cluster_task_set_stack.isra.4+0x44>
1c0144e4:	c991                	beqz	a1,1c0144f8 <__pi_cluster_task_set_stack.isra.4+0x4c>
1c0144e6:	505c                	lw	a5,36(s0)
1c0144e8:	00f48563          	beq	s1,a5,1c0144f2 <__pi_cluster_task_set_stack.isra.4+0x46>
1c0144ec:	557d                	li	a0,-1
1c0144ee:	b7d5                	j	1c0144d2 <__pi_cluster_task_set_stack.isra.4+0x26>
1c0144f0:	c581                	beqz	a1,1c0144f8 <__pi_cluster_task_set_stack.isra.4+0x4c>
1c0144f2:	5050                	lw	a2,36(s0)
1c0144f4:	854a                	mv	a0,s2
1c0144f6:	2905                	jal	1c014926 <pi_cl_l1_free>
1c0144f8:	d044                	sw	s1,36(s0)
1c0144fa:	85a6                	mv	a1,s1
1c0144fc:	854a                	mv	a0,s2
1c0144fe:	2ee1                	jal	1c0148d6 <pi_cl_l1_malloc>
1c014500:	c808                	sw	a0,16(s0)
1c014502:	f561                	bnez	a0,1c0144ca <__pi_cluster_task_set_stack.isra.4+0x1e>
1c014504:	b7e5                	j	1c0144ec <__pi_cluster_task_set_stack.isra.4+0x40>

1c014506 <pi_cluster_open>:
1c014506:	7179                	addi	sp,sp,-48
1c014508:	c266                	sw	s9,4(sp)
1c01450a:	00452c83          	lw	s9,4(a0)
1c01450e:	d422                	sw	s0,40(sp)
1c014510:	c462                	sw	s8,8(sp)
1c014512:	1c01e437          	lui	s0,0x1c01e
1c014516:	000cac03          	lw	s8,0(s9)
1c01451a:	d226                	sw	s1,36(sp)
1c01451c:	55fd                	li	a1,-1
1c01451e:	84aa                	mv	s1,a0
1c014520:	82840513          	addi	a0,s0,-2008 # 1c01d828 <g_cluster_open_close_mutex>
1c014524:	ce4e                	sw	s3,28(sp)
1c014526:	cc52                	sw	s4,24(sp)
1c014528:	d606                	sw	ra,44(sp)
1c01452a:	d04a                	sw	s2,32(sp)
1c01452c:	ca56                	sw	s5,20(sp)
1c01452e:	c85a                	sw	s6,16(sp)
1c014530:	c65e                	sw	s7,12(sp)
1c014532:	c06a                	sw	s10,0(sp)
1c014534:	1c01d9b7          	lui	s3,0x1c01d
1c014538:	351040ef          	jal	ra,1c019088 <os_sync_obj_take>
1c01453c:	001c1a13          	slli	s4,s8,0x1
1c014540:	7d498793          	addi	a5,s3,2004 # 1c01d7d4 <pi_cluster_nb_open>
1c014544:	97d2                	add	a5,a5,s4
1c014546:	0007d703          	lhu	a4,0(a5)
1c01454a:	c71d                	beqz	a4,1c014578 <pi_cluster_open+0x72>
1c01454c:	0705                	addi	a4,a4,1
1c01454e:	00e79023          	sh	a4,0(a5)
1c014552:	82840513          	addi	a0,s0,-2008
1c014556:	47d040ef          	jal	ra,1c0191d2 <os_sync_obj_release>
1c01455a:	4501                	li	a0,0
1c01455c:	50b2                	lw	ra,44(sp)
1c01455e:	5422                	lw	s0,40(sp)
1c014560:	5492                	lw	s1,36(sp)
1c014562:	5902                	lw	s2,32(sp)
1c014564:	49f2                	lw	s3,28(sp)
1c014566:	4a62                	lw	s4,24(sp)
1c014568:	4ad2                	lw	s5,20(sp)
1c01456a:	4b42                	lw	s6,16(sp)
1c01456c:	4bb2                	lw	s7,12(sp)
1c01456e:	4c22                	lw	s8,8(sp)
1c014570:	4c92                	lw	s9,4(sp)
1c014572:	4d02                	lw	s10,0(sp)
1c014574:	6145                	addi	sp,sp,48
1c014576:	8082                	ret
1c014578:	004ca783          	lw	a5,4(s9)
1c01457c:	008ca703          	lw	a4,8(s9)
1c014580:	00020bb7          	lui	s7,0x20
1c014584:	f90b8b93          	addi	s7,s7,-112 # 1ff90 <__l1_heapsram_size>
1c014588:	00e786b3          	add	a3,a5,a4
1c01458c:	557d                	li	a0,-1
1c01458e:	fcdbe7e3          	bltu	s7,a3,1c01455c <pi_cluster_open+0x56>
1c014592:	10000b37          	lui	s6,0x10000
1c014596:	016c1913          	slli	s2,s8,0x16
1c01459a:	070b0b13          	addi	s6,s6,112 # 10000070 <__l1_heapsram_start>
1c01459e:	8aa2                	mv	s5,s0
1c0145a0:	9b4a                	add	s6,s6,s2
1c0145a2:	1c01d437          	lui	s0,0x1c01d
1c0145a6:	02c00693          	li	a3,44
1c0145aa:	9b3e                	add	s6,s6,a5
1c0145ac:	40fb8bb3          	sub	s7,s7,a5
1c0145b0:	7fc40793          	addi	a5,s0,2044 # 1c01d7fc <pi_cluster>
1c0145b4:	42dc07b3          	p.mac	a5,s8,a3
1c0145b8:	0ffc7d13          	andi	s10,s8,255
1c0145bc:	4581                	li	a1,0
1c0145be:	4509                	li	a0,2
1c0145c0:	40eb8bb3          	sub	s7,s7,a4
1c0145c4:	7d498993          	addi	s3,s3,2004
1c0145c8:	0167a623          	sw	s6,12(a5)
1c0145cc:	c49c                	sw	a5,8(s1)
1c0145ce:	03a78423          	sb	s10,40(a5)
1c0145d2:	d398                	sw	a4,32(a5)
1c0145d4:	e7afc0ef          	jal	ra,1c010c4e <__pi_fll_init>
1c0145d8:	4601                	li	a2,0
1c0145da:	4585                	li	a1,1
1c0145dc:	450d                	li	a0,3
1c0145de:	2dbd                	jal	1c014c5c <pi_pmu_domain_state_change>
1c0145e0:	1c0147b7          	lui	a5,0x1c014
1c0145e4:	1c01e737          	lui	a4,0x1c01e
1c0145e8:	40e78793          	addi	a5,a5,1038 # 1c01440e <pi_task_remote_enqueue>
1c0145ec:	baf72623          	sw	a5,-1108(a4) # 1c01dbac <pi_irq_handler_wrapper_vector+0x10>
1c0145f0:	305026f3          	csrr	a3,mtvec
1c0145f4:	1c0107b7          	lui	a5,0x1c010
1c0145f8:	c006b6b3          	p.bclr	a3,a3,0,0
1c0145fc:	31878793          	addi	a5,a5,792 # 1c010318 <__end_mcause_check+0xa>
1c014600:	8f95                	sub	a5,a5,a3
1c014602:	c1478633          	p.extract	a2,a5,0,20
1c014606:	06f00713          	li	a4,111
1c01460a:	c1f62733          	p.insert	a4,a2,0,31
1c01460e:	d2178633          	p.extract	a2,a5,9,1
1c014612:	d3562733          	p.insert	a4,a2,9,21
1c014616:	c0b78633          	p.extract	a2,a5,0,11
1c01461a:	c1462733          	p.insert	a4,a2,0,20
1c01461e:	cec787b3          	p.extract	a5,a5,7,12
1c014622:	cec7a733          	p.insert	a4,a5,7,12
1c014626:	ca98                	sw	a4,16(a3)
1c014628:	1a1097b7          	lui	a5,0x1a109
1c01462c:	4741                	li	a4,16
1c01462e:	6605                	lui	a2,0x1
1c014630:	c3d8                	sw	a4,4(a5)
1c014632:	100005b7          	lui	a1,0x10000
1c014636:	fff60713          	addi	a4,a2,-1 # fff <__STACK_IRQ_SIZE+0x7ff>
1c01463a:	00400793          	li	a5,4
1c01463e:	95ca                	add	a1,a1,s2
1c014640:	8ff9                	and	a5,a5,a4
1c014642:	102004b7          	lui	s1,0x10200
1c014646:	100008b7          	lui	a7,0x10000
1c01464a:	97ae                	add	a5,a5,a1
1c01464c:	94ca                	add	s1,s1,s2
1c01464e:	4605                	li	a2,1
1c014650:	0408a823          	sw	zero,80(a7) # 10000050 <__pi_cluster_fc_task_lock>
1c014654:	02c4a023          	sw	a2,32(s1) # 10200020 <__l1_heapsram_end+0x1e0020>
1c014658:	00400613          	li	a2,4
1c01465c:	8e79                	and	a2,a2,a4
1c01465e:	1c010537          	lui	a0,0x1c010
1c014662:	962e                	add	a2,a2,a1
1c014664:	10050513          	addi	a0,a0,256 # 1c010100 <__irq_cluster_vector_base_m__>
1c014668:	05c00813          	li	a6,92
1c01466c:	7fc40713          	addi	a4,s0,2044
1c014670:	00400693          	li	a3,4
1c014674:	05088893          	addi	a7,a7,80
1c014678:	8e09                	sub	a2,a2,a0
1c01467a:	00c50333          	add	t1,a0,a2
1c01467e:	13004563          	bgtz	a6,1c0147a8 <pi_cluster_open+0x2a2>
1c014682:	02c00613          	li	a2,44
1c014686:	42cc0733          	p.mac	a4,s8,a2
1c01468a:	10201437          	lui	s0,0x10201
1c01468e:	e0440613          	addi	a2,s0,-508 # 10200e04 <__l1_heapsram_end+0x1e0e04>
1c014692:	964a                	add	a2,a2,s2
1c014694:	8562                	mv	a0,s8
1c014696:	cf50                	sw	a2,28(a4)
1c014698:	e0840613          	addi	a2,s0,-504
1c01469c:	03a70423          	sb	s10,40(a4)
1c0146a0:	964a                	add	a2,a2,s2
1c0146a2:	cb50                	sw	a2,20(a4)
1c0146a4:	018cc603          	lbu	a2,24(s9)
1c0146a8:	c71c                	sw	a5,8(a4)
1c0146aa:	00072823          	sw	zero,16(a4)
1c0146ae:	02c704a3          	sb	a2,41(a4)
1c0146b2:	00072c23          	sw	zero,24(a4)
1c0146b6:	00072023          	sw	zero,0(a4)
1c0146ba:	00072223          	sw	zero,4(a4)
1c0146be:	00204737          	lui	a4,0x204
1c0146c2:	00c88223          	sb	a2,4(a7)
1c0146c6:	10470713          	addi	a4,a4,260 # 204104 <__l1_heapsram_size+0x1e4174>
1c0146ca:	d798                	sw	a4,40(a5)
1c0146cc:	0007a023          	sw	zero,0(a5) # 1a109000 <__l1_heapsram_end+0xa0e9000>
1c0146d0:	0007a223          	sw	zero,4(a5)
1c0146d4:	0007a423          	sw	zero,8(a5)
1c0146d8:	0007ac23          	sw	zero,24(a5)
1c0146dc:	0007a823          	sw	zero,16(a5)
1c0146e0:	0207a023          	sw	zero,32(a5)
1c0146e4:	02078a23          	sb	zero,52(a5)
1c0146e8:	0207ac23          	sw	zero,56(a5)
1c0146ec:	02078b23          	sb	zero,54(a5)
1c0146f0:	6785                	lui	a5,0x1
1c0146f2:	17fd                	addi	a5,a5,-1
1c0146f4:	03c68713          	addi	a4,a3,60
1c0146f8:	8f7d                	and	a4,a4,a5
1c0146fa:	04068693          	addi	a3,a3,64
1c0146fe:	000765a3          	p.sw	zero,a1(a4)
1c014702:	8efd                	and	a3,a3,a5
1c014704:	0006e5a3          	p.sw	zero,a1(a3)
1c014708:	865e                	mv	a2,s7
1c01470a:	85da                	mv	a1,s6
1c01470c:	1b9050ef          	jal	ra,1c01a0c4 <pos_alloc_init_l1>
1c014710:	01cca703          	lw	a4,28(s9)
1c014714:	00277793          	andi	a5,a4,2
1c014718:	c799                	beqz	a5,1c014726 <pi_cluster_open+0x220>
1c01471a:	40040793          	addi	a5,s0,1024
1c01471e:	97ca                	add	a5,a5,s2
1c014720:	4685                	li	a3,1
1c014722:	00d7ac23          	sw	a3,24(a5) # 1018 <__STACK_IRQ_SIZE+0x818>
1c014726:	7fc77793          	andi	a5,a4,2044
1c01472a:	cf81                	beqz	a5,1c014742 <pi_cluster_open+0x23c>
1c01472c:	102017b7          	lui	a5,0x10201
1c014730:	00275693          	srli	a3,a4,0x2
1c014734:	40078793          	addi	a5,a5,1024 # 10201400 <__l1_heapsram_end+0x1e1400>
1c014738:	ec96b6b3          	p.bclr	a3,a3,22,9
1c01473c:	97ca                	add	a5,a5,s2
1c01473e:	00d7ae23          	sw	a3,28(a5)
1c014742:	fc173733          	p.bclr	a4,a4,30,1
1c014746:	cb09                	beqz	a4,1c014758 <pi_cluster_open+0x252>
1c014748:	102017b7          	lui	a5,0x10201
1c01474c:	40078793          	addi	a5,a5,1024 # 10201400 <__l1_heapsram_end+0x1e1400>
1c014750:	993e                	add	s2,s2,a5
1c014752:	57fd                	li	a5,-1
1c014754:	00f92023          	sw	a5,0(s2)
1c014758:	1c0107b7          	lui	a5,0x1c010
1c01475c:	16078793          	addi	a5,a5,352 # 1c010160 <_start>
1c014760:	04f4a023          	sw	a5,64(s1)
1c014764:	04f4a223          	sw	a5,68(s1)
1c014768:	04f4a423          	sw	a5,72(s1)
1c01476c:	04f4a623          	sw	a5,76(s1)
1c014770:	04f4a823          	sw	a5,80(s1)
1c014774:	04f4aa23          	sw	a5,84(s1)
1c014778:	04f4ac23          	sw	a5,88(s1)
1c01477c:	04f4ae23          	sw	a5,92(s1)
1c014780:	06f4a023          	sw	a5,96(s1)
1c014784:	1ff00793          	li	a5,511
1c014788:	00f4a423          	sw	a5,8(s1)
1c01478c:	57fd                	li	a5,-1
1c01478e:	1a104737          	lui	a4,0x1a104
1c014792:	0ef72223          	sw	a5,228(a4) # 1a1040e4 <__l1_heapsram_end+0xa0e40e4>
1c014796:	99d2                	add	s3,s3,s4
1c014798:	0009d783          	lhu	a5,0(s3)
1c01479c:	828a8513          	addi	a0,s5,-2008
1c0147a0:	0785                	addi	a5,a5,1
1c0147a2:	00f99023          	sh	a5,0(s3)
1c0147a6:	bb45                	j	1c014556 <pi_cluster_open+0x50>
1c0147a8:	00452e0b          	p.lw	t3,4(a0!)
1c0147ac:	1871                	addi	a6,a6,-4
1c0147ae:	01c32023          	sw	t3,0(t1)
1c0147b2:	b5e1                	j	1c01467a <pi_cluster_open+0x174>

1c0147b4 <pi_cluster_conf_init>:
1c0147b4:	6785                	lui	a5,0x1
1c0147b6:	80078793          	addi	a5,a5,-2048 # 800 <__STACK_IRQ_SIZE>
1c0147ba:	c15c                	sw	a5,4(a0)
1c0147bc:	4785                	li	a5,1
1c0147be:	00052023          	sw	zero,0(a0)
1c0147c2:	00052c23          	sw	zero,24(a0)
1c0147c6:	00052423          	sw	zero,8(a0)
1c0147ca:	cd5c                	sw	a5,28(a0)
1c0147cc:	1c01e537          	lui	a0,0x1c01e
1c0147d0:	4601                	li	a2,0
1c0147d2:	458d                	li	a1,3
1c0147d4:	82850513          	addi	a0,a0,-2008 # 1c01d828 <g_cluster_open_close_mutex>
1c0147d8:	0630406f          	j	1c01903a <os_sync_obj_init>

1c0147dc <pi_cluster_send_task_to_cl_async>:
1c0147dc:	1101                	addi	sp,sp,-32
1c0147de:	ca26                	sw	s1,20(sp)
1c0147e0:	459c                	lw	a5,8(a1)
1c0147e2:	4504                	lw	s1,8(a0)
1c0147e4:	cc22                	sw	s0,24(sp)
1c0147e6:	c84a                	sw	s2,16(sp)
1c0147e8:	c64e                	sw	s3,12(sp)
1c0147ea:	ce06                	sw	ra,28(sp)
1c0147ec:	842e                	mv	s0,a1
1c0147ee:	89b2                	mv	s3,a2
1c0147f0:	0084a903          	lw	s2,8(s1)
1c0147f4:	cf85                	beqz	a5,1c01482c <pi_cluster_send_task_to_cl_async+0x50>
1c0147f6:	4818                	lw	a4,16(s0)
1c0147f8:	4785                	li	a5,1
1c0147fa:	00042e23          	sw	zero,28(s0)
1c0147fe:	00e797b3          	sll	a5,a5,a4
1c014802:	17fd                	addi	a5,a5,-1
1c014804:	d01c                	sw	a5,32(s0)
1c014806:	01342a23          	sw	s3,20(s0)
1c01480a:	30047773          	csrrci	a4,mstatus,8
1c01480e:	409c                	lw	a5,0(s1)
1c014810:	ef95                	bnez	a5,1c01484c <pi_cluster_send_task_to_cl_async+0x70>
1c014812:	c080                	sw	s0,0(s1)
1c014814:	00092783          	lw	a5,0(s2)
1c014818:	e399                	bnez	a5,1c01481e <pi_cluster_send_task_to_cl_async+0x42>
1c01481a:	00892023          	sw	s0,0(s2)
1c01481e:	4cdc                	lw	a5,28(s1)
1c014820:	0007a023          	sw	zero,0(a5)
1c014824:	30071073          	csrw	mstatus,a4
1c014828:	4781                	li	a5,0
1c01482a:	a809                	j	1c01483c <pi_cluster_send_task_to_cl_async+0x60>
1c01482c:	499c                	lw	a5,16(a1)
1c01482e:	45d0                	lw	a2,12(a1)
1c014830:	05a1                	addi	a1,a1,8
1c014832:	02f60633          	mul	a2,a2,a5
1c014836:	399d                	jal	1c0144ac <__pi_cluster_task_set_stack.isra.4>
1c014838:	57fd                	li	a5,-1
1c01483a:	dd55                	beqz	a0,1c0147f6 <pi_cluster_send_task_to_cl_async+0x1a>
1c01483c:	40f2                	lw	ra,28(sp)
1c01483e:	4462                	lw	s0,24(sp)
1c014840:	44d2                	lw	s1,20(sp)
1c014842:	4942                	lw	s2,16(sp)
1c014844:	49b2                	lw	s3,12(sp)
1c014846:	853e                	mv	a0,a5
1c014848:	6105                	addi	sp,sp,32
1c01484a:	8082                	ret
1c01484c:	cfc0                	sw	s0,28(a5)
1c01484e:	b7d1                	j	1c014812 <pi_cluster_send_task_to_cl_async+0x36>

1c014850 <pi_cluster_send_task_to_cl>:
1c014850:	7119                	addi	sp,sp,-128
1c014852:	dca2                	sw	s0,120(sp)
1c014854:	842a                	mv	s0,a0
1c014856:	0828                	addi	a0,sp,24
1c014858:	de86                	sw	ra,124(sp)
1c01485a:	c62e                	sw	a1,12(sp)
1c01485c:	67e050ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c014860:	45b2                	lw	a1,12(sp)
1c014862:	8522                	mv	a0,s0
1c014864:	0830                	addi	a2,sp,24
1c014866:	3f9d                	jal	1c0147dc <pi_cluster_send_task_to_cl_async>
1c014868:	547d                	li	s0,-1
1c01486a:	e509                	bnez	a0,1c014874 <pi_cluster_send_task_to_cl+0x24>
1c01486c:	842a                	mv	s0,a0
1c01486e:	0828                	addi	a0,sp,24
1c014870:	6ac050ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c014874:	8522                	mv	a0,s0
1c014876:	50f6                	lw	ra,124(sp)
1c014878:	5466                	lw	s0,120(sp)
1c01487a:	6109                	addi	sp,sp,128
1c01487c:	8082                	ret

1c01487e <pi_cl_send_task_to_fc>:
1c01487e:	002047b7          	lui	a5,0x204
1c014882:	0c07e703          	p.elw	a4,192(a5) # 2040c0 <__l1_heapsram_size+0x1e4130>
1c014886:	01402773          	csrr	a4,uhartid
1c01488a:	1c01d7b7          	lui	a5,0x1c01d
1c01488e:	8715                	srai	a4,a4,0x5
1c014890:	f2673733          	p.bclr	a4,a4,25,6
1c014894:	02c00693          	li	a3,44
1c014898:	7fc78793          	addi	a5,a5,2044 # 1c01d7fc <pi_cluster>
1c01489c:	42d707b3          	p.mac	a5,a4,a3
1c0148a0:	00204737          	lui	a4,0x204
1c0148a4:	4691                	li	a3,4
1c0148a6:	4f90                	lw	a2,24(a5)
1c0148a8:	ee01                	bnez	a2,1c0148c0 <pi_cl_send_task_to_fc+0x42>
1c0148aa:	cf88                	sw	a0,24(a5)
1c0148ac:	1a109737          	lui	a4,0x1a109
1c0148b0:	47c1                	li	a5,16
1c0148b2:	00f72823          	sw	a5,16(a4) # 1a109010 <__l1_heapsram_end+0xa0e9010>
1c0148b6:	002047b7          	lui	a5,0x204
1c0148ba:	0c07a023          	sw	zero,192(a5) # 2040c0 <__l1_heapsram_size+0x1e4130>
1c0148be:	8082                	ret
1c0148c0:	0c072023          	sw	zero,192(a4)
1c0148c4:	00d72423          	sw	a3,8(a4)
1c0148c8:	03c76603          	p.elw	a2,60(a4)
1c0148cc:	00d72223          	sw	a3,4(a4)
1c0148d0:	0c076603          	p.elw	a2,192(a4)
1c0148d4:	bfc9                	j	1c0148a6 <pi_cl_send_task_to_fc+0x28>

1c0148d6 <pi_cl_l1_malloc>:
1c0148d6:	1141                	addi	sp,sp,-16
1c0148d8:	c606                	sw	ra,12(sp)
1c0148da:	c422                	sw	s0,8(sp)
1c0148dc:	c226                	sw	s1,4(sp)
1c0148de:	c04a                	sw	s2,0(sp)
1c0148e0:	30047973          	csrrci	s2,mstatus,8
1c0148e4:	cd11                	beqz	a0,1c014900 <pi_cl_l1_malloc+0x2a>
1c0148e6:	451c                	lw	a5,8(a0)
1c0148e8:	0287c783          	lbu	a5,40(a5)
1c0148ec:	cb91                	beqz	a5,1c014900 <pi_cl_l1_malloc+0x2a>
1c0148ee:	4501                	li	a0,0
1c0148f0:	30091073          	csrw	mstatus,s2
1c0148f4:	40b2                	lw	ra,12(sp)
1c0148f6:	4422                	lw	s0,8(sp)
1c0148f8:	4492                	lw	s1,4(sp)
1c0148fa:	4902                	lw	s2,0(sp)
1c0148fc:	0141                	addi	sp,sp,16
1c0148fe:	8082                	ret
1c014900:	1c01d4b7          	lui	s1,0x1c01d
1c014904:	7d848513          	addi	a0,s1,2008 # 1c01d7d8 <pos_alloc_l1>
1c014908:	842e                	mv	s0,a1
1c01490a:	1ff050ef          	jal	ra,1c01a308 <pos_alloc>
1c01490e:	f16d                	bnez	a0,1c0148f0 <pi_cl_l1_malloc+0x1a>
1c014910:	85a2                	mv	a1,s0
1c014912:	4509                	li	a0,2
1c014914:	3c3050ef          	jal	ra,1c01a4d6 <pi_alloc_fail>
1c014918:	f979                	bnez	a0,1c0148ee <pi_cl_l1_malloc+0x18>
1c01491a:	85a2                	mv	a1,s0
1c01491c:	7d848513          	addi	a0,s1,2008
1c014920:	1e9050ef          	jal	ra,1c01a308 <pos_alloc>
1c014924:	b7f1                	j	1c0148f0 <pi_cl_l1_malloc+0x1a>

1c014926 <pi_cl_l1_free>:
1c014926:	1141                	addi	sp,sp,-16
1c014928:	c606                	sw	ra,12(sp)
1c01492a:	c422                	sw	s0,8(sp)
1c01492c:	30047473          	csrrci	s0,mstatus,8
1c014930:	4781                	li	a5,0
1c014932:	c501                	beqz	a0,1c01493a <pi_cl_l1_free+0x14>
1c014934:	451c                	lw	a5,8(a0)
1c014936:	0287c783          	lbu	a5,40(a5)
1c01493a:	1c01d537          	lui	a0,0x1c01d
1c01493e:	02400713          	li	a4,36
1c014942:	7d850513          	addi	a0,a0,2008 # 1c01d7d8 <pos_alloc_l1>
1c014946:	42e78533          	p.mac	a0,a5,a4
1c01494a:	25f050ef          	jal	ra,1c01a3a8 <pos_free>
1c01494e:	30041073          	csrw	mstatus,s0
1c014952:	40b2                	lw	ra,12(sp)
1c014954:	4422                	lw	s0,8(sp)
1c014956:	0141                	addi	sp,sp,16
1c014958:	8082                	ret

1c01495a <pi_open_from_conf>:
1c01495a:	c14c                	sw	a1,4(a0)
1c01495c:	8082                	ret

1c01495e <cl_notify_task_done>:
1c01495e:	300477f3          	csrrci	a5,mstatus,8
1c014962:	4705                	li	a4,1
1c014964:	00e50023          	sb	a4,0(a0)
1c014968:	04058593          	addi	a1,a1,64 # 10000040 <__l1_heapsram_size+0xffe00b0>
1c01496c:	00201737          	lui	a4,0x201
1c014970:	e0870713          	addi	a4,a4,-504 # 200e08 <__l1_heapsram_size+0x1e0e78>
1c014974:	05da                	slli	a1,a1,0x16
1c014976:	95ba                	add	a1,a1,a4
1c014978:	0005a023          	sw	zero,0(a1)
1c01497c:	30079073          	csrw	mstatus,a5
1c014980:	8082                	ret

1c014982 <__pi_ram_cluster_req_done>:
1c014982:	07554583          	lbu	a1,117(a0)
1c014986:	07450513          	addi	a0,a0,116
1c01498a:	bfd1                	j	1c01495e <cl_notify_task_done>

1c01498c <__pi_ram_alloc_cluster_req>:
1c01498c:	1141                	addi	sp,sp,-16
1c01498e:	c422                	sw	s0,8(sp)
1c014990:	842a                	mv	s0,a0
1c014992:	4108                	lw	a0,0(a0)
1c014994:	4410                	lw	a2,8(s0)
1c014996:	c606                	sw	ra,12(sp)
1c014998:	411c                	lw	a5,0(a0)
1c01499a:	00440593          	addi	a1,s0,4
1c01499e:	4b9c                	lw	a5,16(a5)
1c0149a0:	9782                	jalr	a5
1c0149a2:	06a40523          	sb	a0,106(s0)
1c0149a6:	06944583          	lbu	a1,105(s0)
1c0149aa:	06840513          	addi	a0,s0,104
1c0149ae:	4422                	lw	s0,8(sp)
1c0149b0:	40b2                	lw	ra,12(sp)
1c0149b2:	0141                	addi	sp,sp,16
1c0149b4:	b76d                	j	1c01495e <cl_notify_task_done>

1c0149b6 <__pi_ram_cluster_req>:
1c0149b6:	7179                	addi	sp,sp,-48
1c0149b8:	d422                	sw	s0,40(sp)
1c0149ba:	d226                	sw	s1,36(sp)
1c0149bc:	d04a                	sw	s2,32(sp)
1c0149be:	d606                	sw	ra,44(sp)
1c0149c0:	ce4e                	sw	s3,28(sp)
1c0149c2:	862a                	mv	a2,a0
1c0149c4:	07764783          	lbu	a5,119(a2)
1c0149c8:	4100                	lw	s0,0(a0)
1c0149ca:	4504                	lw	s1,8(a0)
1c0149cc:	00452903          	lw	s2,4(a0)
1c0149d0:	4554                	lw	a3,12(a0)
1c0149d2:	07654983          	lbu	s3,118(a0)
1c0149d6:	1c0155b7          	lui	a1,0x1c015
1c0149da:	0561                	addi	a0,a0,24
1c0149dc:	cf85                	beqz	a5,1c014a14 <__pi_ram_cluster_req+0x5e>
1c0149de:	4a18                	lw	a4,16(a2)
1c0149e0:	4a5c                	lw	a5,20(a2)
1c0149e2:	98258593          	addi	a1,a1,-1662 # 1c014982 <__pi_ram_cluster_req_done>
1c0149e6:	c236                	sw	a3,4(sp)
1c0149e8:	c63a                	sw	a4,12(sp)
1c0149ea:	c43e                	sw	a5,8(sp)
1c0149ec:	504050ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c0149f0:	4010                	lw	a2,0(s0)
1c0149f2:	88aa                	mv	a7,a0
1c0149f4:	8522                	mv	a0,s0
1c0149f6:	5422                	lw	s0,40(sp)
1c0149f8:	00c62303          	lw	t1,12(a2)
1c0149fc:	47a2                	lw	a5,8(sp)
1c0149fe:	4732                	lw	a4,12(sp)
1c014a00:	4692                	lw	a3,4(sp)
1c014a02:	50b2                	lw	ra,44(sp)
1c014a04:	884e                	mv	a6,s3
1c014a06:	864a                	mv	a2,s2
1c014a08:	49f2                	lw	s3,28(sp)
1c014a0a:	5902                	lw	s2,32(sp)
1c014a0c:	85a6                	mv	a1,s1
1c014a0e:	5492                	lw	s1,36(sp)
1c014a10:	6145                	addi	sp,sp,48
1c014a12:	8302                	jr	t1
1c014a14:	98258593          	addi	a1,a1,-1662
1c014a18:	c236                	sw	a3,4(sp)
1c014a1a:	4d6050ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c014a1e:	401c                	lw	a5,0(s0)
1c014a20:	4692                	lw	a3,4(sp)
1c014a22:	50b2                	lw	ra,44(sp)
1c014a24:	0087a303          	lw	t1,8(a5)
1c014a28:	87aa                	mv	a5,a0
1c014a2a:	8522                	mv	a0,s0
1c014a2c:	5422                	lw	s0,40(sp)
1c014a2e:	874e                	mv	a4,s3
1c014a30:	864a                	mv	a2,s2
1c014a32:	49f2                	lw	s3,28(sp)
1c014a34:	5902                	lw	s2,32(sp)
1c014a36:	85a6                	mv	a1,s1
1c014a38:	5492                	lw	s1,36(sp)
1c014a3a:	6145                	addi	sp,sp,48
1c014a3c:	8302                	jr	t1

1c014a3e <pi_ram_open>:
1c014a3e:	415c                	lw	a5,4(a0)
1c014a40:	439c                	lw	a5,0(a5)
1c014a42:	0007a303          	lw	t1,0(a5)
1c014a46:	c11c                	sw	a5,0(a0)
1c014a48:	8302                	jr	t1

1c014a4a <pi_cl_ram_copy>:
1c014a4a:	1141                	addi	sp,sp,-16
1c014a4c:	c606                	sw	ra,12(sp)
1c014a4e:	c422                	sw	s0,8(sp)
1c014a50:	c388                	sw	a0,0(a5)
1c014a52:	c3d0                	sw	a2,4(a5)
1c014a54:	c78c                	sw	a1,8(a5)
1c014a56:	c7d4                	sw	a3,12(a5)
1c014a58:	014026f3          	csrr	a3,uhartid
1c014a5c:	8695                	srai	a3,a3,0x5
1c014a5e:	f266b6b3          	p.bclr	a3,a3,25,6
1c014a62:	01878413          	addi	s0,a5,24
1c014a66:	1c0155b7          	lui	a1,0x1c015
1c014a6a:	8522                	mv	a0,s0
1c014a6c:	06d78aa3          	sb	a3,117(a5)
1c014a70:	06078a23          	sb	zero,116(a5)
1c014a74:	06e78b23          	sb	a4,118(a5)
1c014a78:	06078ba3          	sb	zero,119(a5)
1c014a7c:	863e                	mv	a2,a5
1c014a7e:	9b658593          	addi	a1,a1,-1610 # 1c0149b6 <__pi_ram_cluster_req>
1c014a82:	46e050ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c014a86:	8522                	mv	a0,s0
1c014a88:	4422                	lw	s0,8(sp)
1c014a8a:	40b2                	lw	ra,12(sp)
1c014a8c:	0141                	addi	sp,sp,16
1c014a8e:	bbc5                	j	1c01487e <pi_cl_send_task_to_fc>

1c014a90 <pi_cl_ram_copy_2d>:
1c014a90:	1141                	addi	sp,sp,-16
1c014a92:	c606                	sw	ra,12(sp)
1c014a94:	c422                	sw	s0,8(sp)
1c014a96:	00a8a023          	sw	a0,0(a7)
1c014a9a:	00c8a223          	sw	a2,4(a7)
1c014a9e:	00b8a423          	sw	a1,8(a7)
1c014aa2:	00d8a623          	sw	a3,12(a7)
1c014aa6:	00e8a823          	sw	a4,16(a7)
1c014aaa:	00f8aa23          	sw	a5,20(a7)
1c014aae:	014027f3          	csrr	a5,uhartid
1c014ab2:	8795                	srai	a5,a5,0x5
1c014ab4:	f267b7b3          	p.bclr	a5,a5,25,6
1c014ab8:	01888413          	addi	s0,a7,24
1c014abc:	06f88aa3          	sb	a5,117(a7)
1c014ac0:	4785                	li	a5,1
1c014ac2:	1c0155b7          	lui	a1,0x1c015
1c014ac6:	8522                	mv	a0,s0
1c014ac8:	06088a23          	sb	zero,116(a7)
1c014acc:	07088b23          	sb	a6,118(a7)
1c014ad0:	06f88ba3          	sb	a5,119(a7)
1c014ad4:	8646                	mv	a2,a7
1c014ad6:	9b658593          	addi	a1,a1,-1610 # 1c0149b6 <__pi_ram_cluster_req>
1c014ada:	416050ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c014ade:	8522                	mv	a0,s0
1c014ae0:	4422                	lw	s0,8(sp)
1c014ae2:	40b2                	lw	ra,12(sp)
1c014ae4:	0141                	addi	sp,sp,16
1c014ae6:	bb61                	j	1c01487e <pi_cl_send_task_to_fc>

1c014ae8 <pi_cl_ram_alloc>:
1c014ae8:	1141                	addi	sp,sp,-16
1c014aea:	c606                	sw	ra,12(sp)
1c014aec:	c422                	sw	s0,8(sp)
1c014aee:	c208                	sw	a0,0(a2)
1c014af0:	c60c                	sw	a1,8(a2)
1c014af2:	01402773          	csrr	a4,uhartid
1c014af6:	8715                	srai	a4,a4,0x5
1c014af8:	f2673733          	p.bclr	a4,a4,25,6
1c014afc:	01060413          	addi	s0,a2,16
1c014b00:	1c0155b7          	lui	a1,0x1c015
1c014b04:	8522                	mv	a0,s0
1c014b06:	06e604a3          	sb	a4,105(a2)
1c014b0a:	06060423          	sb	zero,104(a2)
1c014b0e:	98c58593          	addi	a1,a1,-1652 # 1c01498c <__pi_ram_alloc_cluster_req>
1c014b12:	3de050ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c014b16:	8522                	mv	a0,s0
1c014b18:	4422                	lw	s0,8(sp)
1c014b1a:	40b2                	lw	ra,12(sp)
1c014b1c:	0141                	addi	sp,sp,16
1c014b1e:	b385                	j	1c01487e <pi_cl_send_task_to_fc>

1c014b20 <__pi_pmu_change_domain_power_async>:
1c014b20:	1141                	addi	sp,sp,-16
1c014b22:	c606                	sw	ra,12(sp)
1c014b24:	c422                	sw	s0,8(sp)
1c014b26:	87aa                	mv	a5,a0
1c014b28:	30047473          	csrrci	s0,mstatus,8
1c014b2c:	1c01e537          	lui	a0,0x1c01e
1c014b30:	0215b263          	p.bneimm	a1,1,1c014b54 <__pi_pmu_change_domain_power_async+0x34>
1c014b34:	c1c52703          	lw	a4,-996(a0) # 1c01dc1c <__pi_pmu_domains_on>
1c014b38:	00f75733          	srl	a4,a4,a5
1c014b3c:	fc173733          	p.bclr	a4,a4,30,1
1c014b40:	cb11                	beqz	a4,1c014b54 <__pi_pmu_change_domain_power_async+0x34>
1c014b42:	8536                	mv	a0,a3
1c014b44:	3dc050ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c014b48:	30041073          	csrw	mstatus,s0
1c014b4c:	40b2                	lw	ra,12(sp)
1c014b4e:	4422                	lw	s0,8(sp)
1c014b50:	0141                	addi	sp,sp,16
1c014b52:	8082                	ret
1c014b54:	c1c52703          	lw	a4,-996(a0)
1c014b58:	fff58813          	addi	a6,a1,-1
1c014b5c:	00183813          	seqz	a6,a6
1c014b60:	80f82733          	p.insertr	a4,a6,a5
1c014b64:	883a                	mv	a6,a4
1c014b66:	c0e52e23          	sw	a4,-996(a0)
1c014b6a:	01000893          	li	a7,16
1c014b6e:	0008a303          	lw	t1,0(a7)
1c014b72:	01000713          	li	a4,16
1c014b76:	04031763          	bnez	t1,1c014bc4 <__pi_pmu_change_domain_power_async+0xa4>
1c014b7a:	00d8a223          	sw	a3,4(a7)
1c014b7e:	4685                	li	a3,1
1c014b80:	00f69733          	sll	a4,a3,a5
1c014b84:	fff74713          	not	a4,a4
1c014b88:	01077733          	and	a4,a4,a6
1c014b8c:	00f59633          	sll	a2,a1,a5
1c014b90:	8f51                	or	a4,a4,a2
1c014b92:	c0e52e23          	sw	a4,-996(a0)
1c014b96:	00d8a023          	sw	a3,0(a7)
1c014b9a:	0047a563          	p.beqimm	a5,4,1c014ba4 <__pi_pmu_change_domain_power_async+0x84>
1c014b9e:	078d                	addi	a5,a5,3
1c014ba0:	0786                	slli	a5,a5,0x1
1c014ba2:	97ae                	add	a5,a5,a1
1c014ba4:	4505                	li	a0,1
1c014ba6:	f647b7b3          	p.bclr	a5,a5,27,4
1c014baa:	00f517b3          	sll	a5,a0,a5
1c014bae:	1a107737          	lui	a4,0x1a107
1c014bb2:	00f72423          	sw	a5,8(a4) # 1a107008 <__l1_heapsram_end+0xa0e7008>
1c014bb6:	100007b7          	lui	a5,0x10000
1c014bba:	10278793          	addi	a5,a5,258 # 10000102 <__l1_heapsram_start+0x92>
1c014bbe:	00f72223          	sw	a5,4(a4)
1c014bc2:	b759                	j	1c014b48 <__pi_pmu_change_domain_power_async+0x28>
1c014bc4:	ce9c                	sw	a5,24(a3)
1c014bc6:	471c                	lw	a5,8(a4)
1c014bc8:	cecc                	sw	a1,28(a3)
1c014bca:	d290                	sw	a2,32(a3)
1c014bcc:	e791                	bnez	a5,1c014bd8 <__pi_pmu_change_domain_power_async+0xb8>
1c014bce:	c714                	sw	a3,8(a4)
1c014bd0:	c754                	sw	a3,12(a4)
1c014bd2:	0006a023          	sw	zero,0(a3)
1c014bd6:	bf8d                	j	1c014b48 <__pi_pmu_change_domain_power_async+0x28>
1c014bd8:	475c                	lw	a5,12(a4)
1c014bda:	c394                	sw	a3,0(a5)
1c014bdc:	bfd5                	j	1c014bd0 <__pi_pmu_change_domain_power_async+0xb0>

1c014bde <__pi_pmu_shutdown>:
1c014bde:	7119                	addi	sp,sp,-128
1c014be0:	d8ca                	sw	s2,112(sp)
1c014be2:	d6ce                	sw	s3,108(sp)
1c014be4:	de86                	sw	ra,124(sp)
1c014be6:	dca2                	sw	s0,120(sp)
1c014be8:	daa6                	sw	s1,116(sp)
1c014bea:	892a                	mv	s2,a0
1c014bec:	89ae                	mv	s3,a1
1c014bee:	300477f3          	csrrci	a5,mstatus,8
1c014bf2:	01000413          	li	s0,16
1c014bf6:	4804                	lw	s1,16(s0)
1c014bf8:	2e5d                	jal	1c014fae <system_exit_printf_flush>
1c014bfa:	485c                	lw	a5,20(s0)
1c014bfc:	c014c4b3          	p.bset	s1,s1,0,1
1c014c00:	ce04a7b3          	p.insert	a5,s1,7,0
1c014c04:	1a1044b7          	lui	s1,0x1a104
1c014c08:	c85c                	sw	a5,20(s0)
1c014c0a:	10f4a223          	sw	a5,260(s1) # 1a104104 <__l1_heapsram_end+0xa0e4104>
1c014c0e:	4601                	li	a2,0
1c014c10:	4585                	li	a1,1
1c014c12:	557d                	li	a0,-1
1c014c14:	56e050ef          	jal	ra,1c01a182 <pos_alloc_power_ctrl>
1c014c18:	6785                	lui	a5,0x1
1c014c1a:	17fd                	addi	a5,a5,-1
1c014c1c:	16f4aa23          	sw	a5,372(s1)
1c014c20:	6791                	lui	a5,0x4
1c014c22:	17fd                	addi	a5,a5,-1
1c014c24:	16f4ac23          	sw	a5,376(s1)
1c014c28:	01844783          	lbu	a5,24(s0)
1c014c2c:	c791                	beqz	a5,1c014c38 <__pi_pmu_shutdown+0x5a>
1c014c2e:	4789                	li	a5,2
1c014c30:	1a104737          	lui	a4,0x1a104
1c014c34:	10f72823          	sw	a5,272(a4) # 1a104110 <__l1_heapsram_end+0xa0e4110>
1c014c38:	0028                	addi	a0,sp,8
1c014c3a:	2a0050ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c014c3e:	86aa                	mv	a3,a0
1c014c40:	864e                	mv	a2,s3
1c014c42:	85ca                	mv	a1,s2
1c014c44:	4511                	li	a0,4
1c014c46:	3de9                	jal	1c014b20 <__pi_pmu_change_domain_power_async>
1c014c48:	a001                	j	1c014c48 <__pi_pmu_shutdown+0x6a>

1c014c4a <pi_pmu_domain_state_change_async>:
1c014c4a:	00453863          	p.bneimm	a0,4,1c014c5a <pi_pmu_domain_state_change_async+0x10>
1c014c4e:	87ae                	mv	a5,a1
1c014c50:	1141                	addi	sp,sp,-16
1c014c52:	85b2                	mv	a1,a2
1c014c54:	853e                	mv	a0,a5
1c014c56:	c606                	sw	ra,12(sp)
1c014c58:	3759                	jal	1c014bde <__pi_pmu_shutdown>
1c014c5a:	b5d9                	j	1c014b20 <__pi_pmu_change_domain_power_async>

1c014c5c <pi_pmu_domain_state_change>:
1c014c5c:	7119                	addi	sp,sp,-128
1c014c5e:	dca2                	sw	s0,120(sp)
1c014c60:	842a                	mv	s0,a0
1c014c62:	0828                	addi	a0,sp,24
1c014c64:	de86                	sw	ra,124(sp)
1c014c66:	c62e                	sw	a1,12(sp)
1c014c68:	c432                	sw	a2,8(sp)
1c014c6a:	270050ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c014c6e:	4622                	lw	a2,8(sp)
1c014c70:	45b2                	lw	a1,12(sp)
1c014c72:	0834                	addi	a3,sp,24
1c014c74:	8522                	mv	a0,s0
1c014c76:	3fd1                	jal	1c014c4a <pi_pmu_domain_state_change_async>
1c014c78:	0828                	addi	a0,sp,24
1c014c7a:	2a2050ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c014c7e:	50f6                	lw	ra,124(sp)
1c014c80:	5466                	lw	s0,120(sp)
1c014c82:	6109                	addi	sp,sp,128
1c014c84:	8082                	ret

1c014c86 <__pi_pmu_init>:
1c014c86:	1a104737          	lui	a4,0x1a104
1c014c8a:	0f872703          	lw	a4,248(a4) # 1a1040f8 <__l1_heapsram_end+0xa0e40f8>
1c014c8e:	01000693          	li	a3,16
1c014c92:	d298                	sw	a4,32(a3)
1c014c94:	c0171733          	p.extractu	a4,a4,0,1
1c014c98:	01000793          	li	a5,16
1c014c9c:	468d                	li	a3,3
1c014c9e:	e311                	bnez	a4,1c014ca2 <__pi_pmu_init+0x1c>
1c014ca0:	4681                	li	a3,0
1c014ca2:	1c01e737          	lui	a4,0x1c01e
1c014ca6:	4641                	li	a2,16
1c014ca8:	c0c72e23          	sw	a2,-996(a4) # 1c01dc1c <__pi_pmu_domains_on>
1c014cac:	4605                	li	a2,1
1c014cae:	00c78c23          	sb	a2,24(a5) # 4018 <__STACK_IRQ_SIZE+0x3818>
1c014cb2:	d3d4                	sw	a3,36(a5)
1c014cb4:	0007a023          	sw	zero,0(a5)
1c014cb8:	0007a423          	sw	zero,8(a5)
1c014cbc:	1a109637          	lui	a2,0x1a109
1c014cc0:	020005b7          	lui	a1,0x2000
1c014cc4:	c24c                	sw	a1,4(a2)
1c014cc6:	08000613          	li	a2,128
1c014cca:	1a1075b7          	lui	a1,0x1a107
1c014cce:	00c5a823          	sw	a2,16(a1) # 1a107010 <__l1_heapsram_end+0xa0e7010>
1c014cd2:	0007a223          	sw	zero,4(a5)
1c014cd6:	e681                	bnez	a3,1c014cde <__pi_pmu_init+0x58>
1c014cd8:	0007aa23          	sw	zero,20(a5)
1c014cdc:	8082                	ret
1c014cde:	5394                	lw	a3,32(a5)
1c014ce0:	cbd4                	sw	a3,20(a5)
1c014ce2:	ce0696b3          	p.extractu	a3,a3,7,0
1c014ce6:	0026d613          	srli	a2,a3,0x2
1c014cea:	f83635b3          	p.bclr	a1,a2,28,3
1c014cee:	cd85                	beqz	a1,1c014d26 <__pi_pmu_init+0xa0>
1c014cf0:	fc163633          	p.bclr	a2,a2,30,1
1c014cf4:	c619                	beqz	a2,1c014d02 <__pi_pmu_init+0x7c>
1c014cf6:	c1c72603          	lw	a2,-996(a4)
1c014cfa:	c0164633          	p.bset	a2,a2,0,1
1c014cfe:	c0c72e23          	sw	a2,-996(a4)
1c014d02:	c0369633          	p.extractu	a2,a3,0,3
1c014d06:	c619                	beqz	a2,1c014d14 <__pi_pmu_init+0x8e>
1c014d08:	c1c72603          	lw	a2,-996(a4)
1c014d0c:	c0364633          	p.bset	a2,a2,0,3
1c014d10:	c0c72e23          	sw	a2,-996(a4)
1c014d14:	c04696b3          	p.extractu	a3,a3,0,4
1c014d18:	c699                	beqz	a3,1c014d26 <__pi_pmu_init+0xa0>
1c014d1a:	c1c72683          	lw	a3,-996(a4)
1c014d1e:	c006c6b3          	p.bset	a3,a3,0,0
1c014d22:	c0d72e23          	sw	a3,-996(a4)
1c014d26:	4fd8                	lw	a4,28(a5)
1c014d28:	1a1046b7          	lui	a3,0x1a104
1c014d2c:	c2374733          	p.bset	a4,a4,1,3
1c014d30:	cfd8                	sw	a4,28(a5)
1c014d32:	10e6a023          	sw	a4,256(a3) # 1a104100 <__l1_heapsram_end+0xa0e4100>
1c014d36:	4bd8                	lw	a4,20(a5)
1c014d38:	aff77713          	andi	a4,a4,-1281
1c014d3c:	cbd8                	sw	a4,20(a5)
1c014d3e:	10e6a223          	sw	a4,260(a3)
1c014d42:	8082                	ret

1c014d44 <__is_irq_mode>:
1c014d44:	c10027f3          	csrr	a5,privlv
1c014d48:	cf91                	beqz	a5,1c014d64 <__is_irq_mode+0x20>
1c014d4a:	342027f3          	csrr	a5,mcause
1c014d4e:	4505                	li	a0,1
1c014d50:	0007cd63          	bltz	a5,1c014d6a <__is_irq_mode+0x26>
1c014d54:	30002573          	csrr	a0,mstatus
1c014d58:	810d                	srli	a0,a0,0x3
1c014d5a:	00154513          	xori	a0,a0,1
1c014d5e:	fc153533          	p.bclr	a0,a0,30,1
1c014d62:	8082                	ret
1c014d64:	04202573          	csrr	a0,0x42
1c014d68:	817d                	srli	a0,a0,0x1f
1c014d6a:	8082                	ret

1c014d6c <__internal_semihost.constprop.4>:
1c014d6c:	85aa                	mv	a1,a0
1c014d6e:	4511                	li	a0,4
1c014d70:	01f01013          	slli	zero,zero,0x1f
1c014d74:	00100073          	ebreak
1c014d78:	40705013          	srai	zero,zero,0x7
1c014d7c:	8082                	ret

1c014d7e <__semihost_putc_cluster_req>:
1c014d7e:	1141                	addi	sp,sp,-16
1c014d80:	c422                	sw	s0,8(sp)
1c014d82:	842a                	mv	s0,a0
1c014d84:	4108                	lw	a0,0(a0)
1c014d86:	c606                	sw	ra,12(sp)
1c014d88:	37d5                	jal	1c014d6c <__internal_semihost.constprop.4>
1c014d8a:	00944783          	lbu	a5,9(s0)
1c014d8e:	30047773          	csrrci	a4,mstatus,8
1c014d92:	4685                	li	a3,1
1c014d94:	00d40423          	sb	a3,8(s0)
1c014d98:	04078793          	addi	a5,a5,64
1c014d9c:	002016b7          	lui	a3,0x201
1c014da0:	e0868693          	addi	a3,a3,-504 # 200e08 <__l1_heapsram_size+0x1e0e78>
1c014da4:	07da                	slli	a5,a5,0x16
1c014da6:	97b6                	add	a5,a5,a3
1c014da8:	0007a023          	sw	zero,0(a5)
1c014dac:	30071073          	csrw	mstatus,a4
1c014db0:	40b2                	lw	ra,12(sp)
1c014db2:	4422                	lw	s0,8(sp)
1c014db4:	0141                	addi	sp,sp,16
1c014db6:	8082                	ret

1c014db8 <__io_lock>:
1c014db8:	52404783          	lbu	a5,1316(zero) # 524 <g_sched_started>
1c014dbc:	7179                	addi	sp,sp,-48
1c014dbe:	d606                	sw	ra,44(sp)
1c014dc0:	d422                	sw	s0,40(sp)
1c014dc2:	d226                	sw	s1,36(sp)
1c014dc4:	d04a                	sw	s2,32(sp)
1c014dc6:	ce4e                	sw	s3,28(sp)
1c014dc8:	cc52                	sw	s4,24(sp)
1c014dca:	0ff7f793          	andi	a5,a5,255
1c014dce:	cb91                	beqz	a5,1c014de2 <__io_lock+0x2a>
1c014dd0:	3f95                	jal	1c014d44 <__is_irq_mode>
1c014dd2:	e901                	bnez	a0,1c014de2 <__io_lock+0x2a>
1c014dd4:	1c01e537          	lui	a0,0x1c01e
1c014dd8:	55fd                	li	a1,-1
1c014dda:	aa850513          	addi	a0,a0,-1368 # 1c01daa8 <g_printf_mutex>
1c014dde:	2aa040ef          	jal	ra,1c019088 <os_sync_obj_take>
1c014de2:	1e000437          	lui	s0,0x1e000
1c014de6:	58040413          	addi	s0,s0,1408 # 1e000580 <__heapl2ram_size+0x1e70580>
1c014dea:	401c                	lw	a5,0(s0)
1c014dec:	01f7aa63          	p.beqimm	a5,-1,1c014e00 <__io_lock+0x48>
1c014df0:	50b2                	lw	ra,44(sp)
1c014df2:	5422                	lw	s0,40(sp)
1c014df4:	5492                	lw	s1,36(sp)
1c014df6:	5902                	lw	s2,32(sp)
1c014df8:	49f2                	lw	s3,28(sp)
1c014dfa:	4a62                	lw	s4,24(sp)
1c014dfc:	6145                	addi	sp,sp,48
1c014dfe:	8082                	ret
1c014e00:	300474f3          	csrrci	s1,mstatus,8
1c014e04:	0068                	addi	a0,sp,12
1c014e06:	13a050ef          	jal	ra,1c019f40 <__pi_os_thread_has_event>
1c014e0a:	c519                	beqz	a0,1c014e18 <__io_lock+0x60>
1c014e0c:	30049073          	csrw	mstatus,s1
1c014e10:	4532                	lw	a0,12(sp)
1c014e12:	132050ef          	jal	ra,1c019f44 <__pi_os_thread_exec_event>
1c014e16:	bfd1                	j	1c014dea <__io_lock+0x32>
1c014e18:	4e002903          	lw	s2,1248(zero) # 4e0 <pxCurrentTCB>
1c014e1c:	854a                	mv	a0,s2
1c014e1e:	6bf030ef          	jal	ra,1c018cdc <sched_ready_list_remove>
1c014e22:	0c6040ef          	jal	ra,1c018ee8 <sched_ready_list_is_empty>
1c014e26:	8a2a                	mv	s4,a0
1c014e28:	854a                	mv	a0,s2
1c014e2a:	020040ef          	jal	ra,1c018e4a <sched_ready_list_add>
1c014e2e:	02092023          	sw	zero,32(s2)
1c014e32:	000a0763          	beqz	s4,1c014e40 <__io_lock+0x88>
1c014e36:	10500073          	wfi
1c014e3a:	30049073          	csrw	mstatus,s1
1c014e3e:	b775                	j	1c014dea <__io_lock+0x32>
1c014e40:	30049073          	csrw	mstatus,s1
1c014e44:	0e4050ef          	jal	ra,1c019f28 <__os_native_yield_force>
1c014e48:	b74d                	j	1c014dea <__io_lock+0x32>

1c014e4a <__io_unlock>:
1c014e4a:	1c0007b7          	lui	a5,0x1c000
1c014e4e:	5807a023          	sw	zero,1408(a5) # 1c000580 <__DTOR_END__>
1c014e52:	52404783          	lbu	a5,1316(zero) # 524 <g_sched_started>
1c014e56:	0ff7f793          	andi	a5,a5,255
1c014e5a:	c385                	beqz	a5,1c014e7a <__io_unlock+0x30>
1c014e5c:	1141                	addi	sp,sp,-16
1c014e5e:	c606                	sw	ra,12(sp)
1c014e60:	35d5                	jal	1c014d44 <__is_irq_mode>
1c014e62:	e909                	bnez	a0,1c014e74 <__io_unlock+0x2a>
1c014e64:	40b2                	lw	ra,12(sp)
1c014e66:	1c01e537          	lui	a0,0x1c01e
1c014e6a:	aa850513          	addi	a0,a0,-1368 # 1c01daa8 <g_printf_mutex>
1c014e6e:	0141                	addi	sp,sp,16
1c014e70:	3620406f          	j	1c0191d2 <os_sync_obj_release>
1c014e74:	40b2                	lw	ra,12(sp)
1c014e76:	0141                	addi	sp,sp,16
1c014e78:	8082                	ret
1c014e7a:	8082                	ret

1c014e7c <__semihost_putc>:
1c014e7c:	7135                	addi	sp,sp,-160
1c014e7e:	cf06                	sw	ra,156(sp)
1c014e80:	cd22                	sw	s0,152(sp)
1c014e82:	cb26                	sw	s1,148(sp)
1c014e84:	c94a                	sw	s2,144(sp)
1c014e86:	c74e                	sw	s3,140(sp)
1c014e88:	014027f3          	csrr	a5,uhartid
1c014e8c:	f457b7b3          	p.bclr	a5,a5,26,5
1c014e90:	0a97a563          	p.beqimm	a5,9,1c014f3a <__semihost_putc+0xbe>
1c014e94:	01402473          	csrr	s0,uhartid
1c014e98:	1c01c937          	lui	s2,0x1c01c
1c014e9c:	f4543433          	p.bclr	s0,s0,26,5
1c014ea0:	08100793          	li	a5,129
1c014ea4:	69c90913          	addi	s2,s2,1692 # 1c01c69c <printf_buffer>
1c014ea8:	42f40933          	p.mac	s2,s0,a5
1c014eac:	1c01d7b7          	lui	a5,0x1c01d
1c014eb0:	b2878713          	addi	a4,a5,-1240 # 1c01cb28 <printf_buffer_index>
1c014eb4:	040a                	slli	s0,s0,0x2
1c014eb6:	9722                	add	a4,a4,s0
1c014eb8:	4304                	lw	s1,0(a4)
1c014eba:	b2878993          	addi	s3,a5,-1240
1c014ebe:	00a944a3          	p.sb	a0,s1(s2)
1c014ec2:	0485                	addi	s1,s1,1
1c014ec4:	c304                	sw	s1,0(a4)
1c014ec6:	07f00713          	li	a4,127
1c014eca:	00e48463          	beq	s1,a4,1c014ed2 <__semihost_putc+0x56>
1c014ece:	04a53863          	p.bneimm	a0,10,1c014f1e <__semihost_putc+0xa2>
1c014ed2:	000944a3          	p.sb	zero,s1(s2)
1c014ed6:	06000613          	li	a2,96
1c014eda:	4581                	li	a1,0
1c014edc:	1008                	addi	a0,sp,32
1c014ede:	0f4050ef          	jal	ra,1c019fd2 <memset>
1c014ee2:	cc4a                	sw	s2,24(sp)
1c014ee4:	ce26                	sw	s1,28(sp)
1c014ee6:	014027f3          	csrr	a5,uhartid
1c014eea:	8795                	srai	a5,a5,0x5
1c014eec:	f267b7b3          	p.bclr	a5,a5,25,6
1c014ef0:	1c0155b7          	lui	a1,0x1c015
1c014ef4:	0830                	addi	a2,sp,24
1c014ef6:	d7e58593          	addi	a1,a1,-642 # 1c014d7e <__semihost_putc_cluster_req>
1c014efa:	1028                	addi	a0,sp,40
1c014efc:	02f100a3          	sb	a5,33(sp)
1c014f00:	006050ef          	jal	ra,1c019f06 <__pi_os_evt_callback_irq_init>
1c014f04:	1028                	addi	a0,sp,40
1c014f06:	979ff0ef          	jal	ra,1c01487e <pi_cl_send_task_to_fc>
1c014f0a:	4691                	li	a3,4
1c014f0c:	00204737          	lui	a4,0x204
1c014f10:	02014783          	lbu	a5,32(sp)
1c014f14:	0ff7f793          	andi	a5,a5,255
1c014f18:	cb91                	beqz	a5,1c014f2c <__semihost_putc+0xb0>
1c014f1a:	0009e423          	p.sw	zero,s0(s3)
1c014f1e:	40fa                	lw	ra,156(sp)
1c014f20:	446a                	lw	s0,152(sp)
1c014f22:	44da                	lw	s1,148(sp)
1c014f24:	494a                	lw	s2,144(sp)
1c014f26:	49ba                	lw	s3,140(sp)
1c014f28:	610d                	addi	sp,sp,160
1c014f2a:	8082                	ret
1c014f2c:	00d72423          	sw	a3,8(a4) # 204008 <__l1_heapsram_size+0x1e4078>
1c014f30:	03c76783          	p.elw	a5,60(a4)
1c014f34:	00d72223          	sw	a3,4(a4)
1c014f38:	bfe1                	j	1c014f10 <__semihost_putc+0x94>
1c014f3a:	c62a                	sw	a0,12(sp)
1c014f3c:	3db5                	jal	1c014db8 <__io_lock>
1c014f3e:	1c01d6b7          	lui	a3,0x1c01d
1c014f42:	77d68613          	addi	a2,a3,1917 # 1c01d77d <g_printf_semihost_index>
1c014f46:	00064783          	lbu	a5,0(a2) # 1a109000 <__l1_heapsram_end+0xa0e9000>
1c014f4a:	4532                	lw	a0,12(sp)
1c014f4c:	1c01c737          	lui	a4,0x1c01c
1c014f50:	61870593          	addi	a1,a4,1560 # 1c01c618 <g_printf_semihost_buffer>
1c014f54:	00a5c7a3          	p.sb	a0,a5(a1)
1c014f58:	0785                	addi	a5,a5,1
1c014f5a:	0ff7f793          	andi	a5,a5,255
1c014f5e:	00f60023          	sb	a5,0(a2)
1c014f62:	07f00613          	li	a2,127
1c014f66:	77d68413          	addi	s0,a3,1917
1c014f6a:	61870493          	addi	s1,a4,1560
1c014f6e:	00c78463          	beq	a5,a2,1c014f76 <__semihost_putc+0xfa>
1c014f72:	00a53b63          	p.bneimm	a0,10,1c014f88 <__semihost_putc+0x10c>
1c014f76:	61870513          	addi	a0,a4,1560
1c014f7a:	0004c7a3          	p.sb	zero,a5(s1)
1c014f7e:	33fd                	jal	1c014d6c <__internal_semihost.constprop.4>
1c014f80:	00040023          	sb	zero,0(s0)
1c014f84:	00048023          	sb	zero,0(s1)
1c014f88:	35c9                	jal	1c014e4a <__io_unlock>
1c014f8a:	bf51                	j	1c014f1e <__semihost_putc+0xa2>

1c014f8c <_putchar>:
1c014f8c:	bdc5                	j	1c014e7c <__semihost_putc>

1c014f8e <puts>:
1c014f8e:	1141                	addi	sp,sp,-16
1c014f90:	c422                	sw	s0,8(sp)
1c014f92:	c606                	sw	ra,12(sp)
1c014f94:	842a                	mv	s0,a0
1c014f96:	0014450b          	p.lbu	a0,1(s0!)
1c014f9a:	c119                	beqz	a0,1c014fa0 <puts+0x12>
1c014f9c:	35c5                	jal	1c014e7c <__semihost_putc>
1c014f9e:	bfe5                	j	1c014f96 <puts+0x8>
1c014fa0:	4529                	li	a0,10
1c014fa2:	3de9                	jal	1c014e7c <__semihost_putc>
1c014fa4:	40b2                	lw	ra,12(sp)
1c014fa6:	4422                	lw	s0,8(sp)
1c014fa8:	4501                	li	a0,0
1c014faa:	0141                	addi	sp,sp,16
1c014fac:	8082                	ret

1c014fae <system_exit_printf_flush>:
1c014fae:	1c01d7b7          	lui	a5,0x1c01d
1c014fb2:	77d78793          	addi	a5,a5,1917 # 1c01d77d <g_printf_semihost_index>
1c014fb6:	0007c703          	lbu	a4,0(a5)
1c014fba:	100766b3          	p.extbs	a3,a4
1c014fbe:	02d05063          	blez	a3,1c014fde <system_exit_printf_flush+0x30>
1c014fc2:	00170693          	addi	a3,a4,1
1c014fc6:	1c01c537          	lui	a0,0x1c01c
1c014fca:	00d78023          	sb	a3,0(a5)
1c014fce:	61850793          	addi	a5,a0,1560 # 1c01c618 <g_printf_semihost_buffer>
1c014fd2:	46a9                	li	a3,10
1c014fd4:	61850513          	addi	a0,a0,1560
1c014fd8:	00d7c723          	p.sb	a3,a4(a5)
1c014fdc:	bb41                	j	1c014d6c <__internal_semihost.constprop.4>
1c014fde:	8082                	ret

1c014fe0 <printf_init>:
1c014fe0:	1c01e537          	lui	a0,0x1c01e
1c014fe4:	1141                	addi	sp,sp,-16
1c014fe6:	aa850513          	addi	a0,a0,-1368 # 1c01daa8 <g_printf_mutex>
1c014fea:	4601                	li	a2,0
1c014fec:	458d                	li	a1,3
1c014fee:	c606                	sw	ra,12(sp)
1c014ff0:	04a040ef          	jal	ra,1c01903a <os_sync_obj_init>
1c014ff4:	40b2                	lw	ra,12(sp)
1c014ff6:	4501                	li	a0,0
1c014ff8:	0141                	addi	sp,sp,16
1c014ffa:	8082                	ret

1c014ffc <printf_deinit>:
1c014ffc:	1c01e537          	lui	a0,0x1c01e
1c015000:	aa850513          	addi	a0,a0,-1368 # 1c01daa8 <g_printf_mutex>
1c015004:	2c40406f          	j	1c0192c8 <os_sync_obj_delete>

1c015008 <system_init>:
1c015008:	1141                	addi	sp,sp,-16
1c01500a:	57fd                	li	a5,-1
1c01500c:	1a106737          	lui	a4,0x1a106
1c015010:	c606                	sw	ra,12(sp)
1c015012:	c422                	sw	s0,8(sp)
1c015014:	00f72223          	sw	a5,4(a4) # 1a106004 <__l1_heapsram_end+0xa0e6004>
1c015018:	00f72423          	sw	a5,8(a4)
1c01501c:	00f72623          	sw	a5,12(a4)
1c015020:	00f72823          	sw	a5,16(a4)
1c015024:	00f72a23          	sw	a5,20(a4)
1c015028:	00f72c23          	sw	a5,24(a4)
1c01502c:	00f72e23          	sw	a5,28(a4)
1c015030:	02f72023          	sw	a5,32(a4)
1c015034:	02f72223          	sw	a5,36(a4)
1c015038:	02f72423          	sw	a5,40(a4)
1c01503c:	02f72623          	sw	a5,44(a4)
1c015040:	02f72823          	sw	a5,48(a4)
1c015044:	02f72a23          	sw	a5,52(a4)
1c015048:	02f72c23          	sw	a5,56(a4)
1c01504c:	02f72e23          	sw	a5,60(a4)
1c015050:	04f72023          	sw	a5,64(a4)
1c015054:	04f72223          	sw	a5,68(a4)
1c015058:	04f72423          	sw	a5,72(a4)
1c01505c:	04f72623          	sw	a5,76(a4)
1c015060:	04f72823          	sw	a5,80(a4)
1c015064:	04f72a23          	sw	a5,84(a4)
1c015068:	04f72c23          	sw	a5,88(a4)
1c01506c:	04f72e23          	sw	a5,92(a4)
1c015070:	06f72023          	sw	a5,96(a4)
1c015074:	1c000737          	lui	a4,0x1c000
1c015078:	58072023          	sw	zero,1408(a4) # 1c000580 <__DTOR_END__>
1c01507c:	1a109737          	lui	a4,0x1a109
1c015080:	00870693          	addi	a3,a4,8 # 1a109008 <__l1_heapsram_end+0xa0e9008>
1c015084:	c29c                	sw	a5,0(a3)
1c015086:	01470693          	addi	a3,a4,20
1c01508a:	c29c                	sw	a5,0(a3)
1c01508c:	80070693          	addi	a3,a4,-2048
1c015090:	c29c                	sw	a5,0(a3)
1c015092:	1a104437          	lui	s0,0x1a104
1c015096:	468d                	li	a3,3
1c015098:	18d42423          	sw	a3,392(s0) # 1a104188 <__l1_heapsram_end+0xa0e4188>
1c01509c:	4d800613          	li	a2,1240
1c0150a0:	c25c                	sw	a5,4(a2)
1c0150a2:	57f9                	li	a5,-2
1c0150a4:	4cf02c23          	sw	a5,1240(zero) # 4d8 <__pi_udma_chan_lin>
1c0150a8:	0fe00693          	li	a3,254
1c0150ac:	4cd02a23          	sw	a3,1236(zero) # 4d4 <__pi_udma_chan_2d>
1c0150b0:	0ff00693          	li	a3,255
1c0150b4:	4cd02823          	sw	a3,1232(zero) # 4d0 <__pi_udma_chan_fifo>
1c0150b8:	040007b7          	lui	a5,0x4000
1c0150bc:	c35c                	sw	a5,4(a4)
1c0150be:	4789                	li	a5,2
1c0150c0:	c35c                	sw	a5,4(a4)
1c0150c2:	37e050ef          	jal	ra,1c01a440 <pi_malloc_init>
1c0150c6:	6505                	lui	a0,0x1
1c0150c8:	4ec050ef          	jal	ra,1c01a5b4 <pi_malloc>
1c0150cc:	1c01d7b7          	lui	a5,0x1c01d
1c0150d0:	78a7a023          	sw	a0,1920(a5) # 1c01d780 <__stack_fc_end__>
1c0150d4:	6785                	lui	a5,0x1
1c0150d6:	953e                	add	a0,a0,a5
1c0150d8:	1c01d7b7          	lui	a5,0x1c01d
1c0150dc:	78a7a223          	sw	a0,1924(a5) # 1c01d784 <__stack_fc_start__>
1c0150e0:	4785                	li	a5,1
1c0150e2:	10f42823          	sw	a5,272(s0)
1c0150e6:	005dc537          	lui	a0,0x5dc
1c0150ea:	053d                	addi	a0,a0,15
1c0150ec:	773040ef          	jal	ra,1c01a05e <pi_ref_fast_clock_frequency_set>
1c0150f0:	0dc42783          	lw	a5,220(s0)
1c0150f4:	c007c7b3          	p.bset	a5,a5,0,0
1c0150f8:	0cf42e23          	sw	a5,220(s0)
1c0150fc:	7ca030ef          	jal	ra,1c0188c6 <os_evt_free_list_nodes_init>
1c015100:	c511                	beqz	a0,1c01510c <system_init+0x104>
1c015102:	7579                	lui	a0,0xffffe
1c015104:	eb850513          	addi	a0,a0,-328 # ffffdeb8 <pulp__FC+0xffffdeb9>
1c015108:	6ad040ef          	jal	ra,1c019fb4 <exit>
1c01510c:	69f030ef          	jal	ra,1c018faa <os_sync_obj_free_list_nodes_init>
1c015110:	f96d                	bnez	a0,1c015102 <system_init+0xfa>
1c015112:	3ae020ef          	jal	ra,1c0174c0 <pi_efuse_reg_val_update>
1c015116:	c509                	beqz	a0,1c015120 <system_init+0x118>
1c015118:	7579                	lui	a0,0xffffe
1c01511a:	eb750513          	addi	a0,a0,-329 # ffffdeb7 <pulp__FC+0xffffdeb8>
1c01511e:	b7ed                	j	1c015108 <system_init+0x100>
1c015120:	0cc42403          	lw	s0,204(s0)
1c015124:	c1f43433          	p.bclr	s0,s0,0,31
1c015128:	1a1047b7          	lui	a5,0x1a104
1c01512c:	0c87a623          	sw	s0,204(a5) # 1a1040cc <__l1_heapsram_end+0xa0e40cc>
1c015130:	40b2                	lw	ra,12(sp)
1c015132:	4422                	lw	s0,8(sp)
1c015134:	0141                	addi	sp,sp,16
1c015136:	8082                	ret

1c015138 <system_init_finalize>:
1c015138:	1141                	addi	sp,sp,-16
1c01513a:	c606                	sw	ra,12(sp)
1c01513c:	300467f3          	csrrsi	a5,mstatus,8
1c015140:	3699                	jal	1c014c86 <__pi_pmu_init>
1c015142:	b63fb0ef          	jal	ra,1c010ca4 <__pi_fll_init_all>
1c015146:	2a35                	jal	1c015282 <pi_bsp_init>
1c015148:	3d61                	jal	1c014fe0 <printf_init>
1c01514a:	c511                	beqz	a0,1c015156 <system_init_finalize+0x1e>
1c01514c:	7579                	lui	a0,0xffffe
1c01514e:	e5450513          	addi	a0,a0,-428 # ffffde54 <pulp__FC+0xffffde55>
1c015152:	663040ef          	jal	ra,1c019fb4 <exit>
1c015156:	40b2                	lw	ra,12(sp)
1c015158:	0141                	addi	sp,sp,16
1c01515a:	8082                	ret

1c01515c <system_printf_uart_close>:
1c01515c:	b545                	j	1c014ffc <printf_deinit>

1c01515e <deallocate_init_stack>:
1c01515e:	1c01d7b7          	lui	a5,0x1c01d
1c015162:	7807a503          	lw	a0,1920(a5) # 1c01d780 <__stack_fc_end__>
1c015166:	3260506f          	j	1c01a48c <pi_free>

1c01516a <system_setup_systick>:
1c01516a:	1101                	addi	sp,sp,-32
1c01516c:	1c01d7b7          	lui	a5,0x1c01d
1c015170:	78a7a423          	sw	a0,1928(a5) # 1c01d788 <tick_rate>
1c015174:	0048                	addi	a0,sp,4
1c015176:	ce06                	sw	ra,28(sp)
1c015178:	cc22                	sw	s0,24(sp)
1c01517a:	6c8040ef          	jal	ra,1c019842 <pi_timer_conf_init>
1c01517e:	1c01e437          	lui	s0,0x1c01e
1c015182:	06400793          	li	a5,100
1c015186:	b5040513          	addi	a0,s0,-1200 # 1c01db50 <sys_timer_hi_prec>
1c01518a:	004c                	addi	a1,sp,4
1c01518c:	c23e                	sw	a5,4(sp)
1c01518e:	00010423          	sb	zero,8(sp)
1c015192:	fc8ff0ef          	jal	ra,1c01495a <pi_open_from_conf>
1c015196:	b5040513          	addi	a0,s0,-1200
1c01519a:	6be040ef          	jal	ra,1c019858 <pi_timer_open>
1c01519e:	c919                	beqz	a0,1c0151b4 <system_setup_systick+0x4a>
1c0151a0:	85aa                	mv	a1,a0
1c0151a2:	1c01c537          	lui	a0,0x1c01c
1c0151a6:	c4050513          	addi	a0,a0,-960 # 1c01bc40 <__clz_tab+0x1f8>
1c0151aa:	4d2060ef          	jal	ra,1c01b67c <printf_>
1c0151ae:	5555                	li	a0,-11
1c0151b0:	605040ef          	jal	ra,1c019fb4 <exit>
1c0151b4:	b5040513          	addi	a0,s0,-1200
1c0151b8:	087040ef          	jal	ra,1c019a3e <pi_timer_start>
1c0151bc:	40f2                	lw	ra,28(sp)
1c0151be:	4462                	lw	s0,24(sp)
1c0151c0:	6105                	addi	sp,sp,32
1c0151c2:	8082                	ret

1c0151c4 <system_core_clock_update>:
1c0151c4:	1101                	addi	sp,sp,-32
1c0151c6:	cc22                	sw	s0,24(sp)
1c0151c8:	1c000437          	lui	s0,0x1c000
1c0151cc:	59840413          	addi	s0,s0,1432 # 1c000598 <SystemCoreClock>
1c0151d0:	401c                	lw	a5,0(s0)
1c0151d2:	ce06                	sw	ra,28(sp)
1c0151d4:	04a78163          	beq	a5,a0,1c015216 <system_core_clock_update+0x52>
1c0151d8:	c008                	sw	a0,0(s0)
1c0151da:	4501                	li	a0,0
1c0151dc:	640040ef          	jal	ra,1c01981c <__pi_timer_stop>
1c0151e0:	0068                	addi	a0,sp,12
1c0151e2:	c602                	sw	zero,12(sp)
1c0151e4:	5fc040ef          	jal	ra,1c0197e0 <__pi_timer_conf_init>
1c0151e8:	00c14783          	lbu	a5,12(sp)
1c0151ec:	4705                	li	a4,1
1c0151ee:	4010                	lw	a2,0(s0)
1c0151f0:	c04727b3          	p.insert	a5,a4,0,4
1c0151f4:	c02727b3          	p.insert	a5,a4,0,2
1c0151f8:	c07727b3          	p.insert	a5,a4,0,7
1c0151fc:	00f10623          	sb	a5,12(sp)
1c015200:	1c01d7b7          	lui	a5,0x1c01d
1c015204:	7887a783          	lw	a5,1928(a5) # 1c01d788 <tick_rate>
1c015208:	45b2                	lw	a1,12(sp)
1c01520a:	4501                	li	a0,0
1c01520c:	02f65633          	divu	a2,a2,a5
1c015210:	167d                	addi	a2,a2,-1
1c015212:	5d4040ef          	jal	ra,1c0197e6 <__pi_timer_init>
1c015216:	40f2                	lw	ra,28(sp)
1c015218:	4462                	lw	s0,24(sp)
1c01521a:	6105                	addi	sp,sp,32
1c01521c:	8082                	ret

1c01521e <system_exit>:
1c01521e:	014027f3          	csrr	a5,uhartid
1c015222:	f457b7b3          	p.bclr	a5,a5,26,5
1c015226:	0497b263          	p.bneimm	a5,9,1c01526a <system_exit+0x4c>
1c01522a:	1141                	addi	sp,sp,-16
1c01522c:	c422                	sw	s0,8(sp)
1c01522e:	c606                	sw	ra,12(sp)
1c015230:	842a                	mv	s0,a0
1c015232:	3bb5                	jal	1c014fae <system_exit_printf_flush>
1c015234:	000205b7          	lui	a1,0x20
1c015238:	c415                	beqz	s0,1c015264 <system_exit+0x46>
1c01523a:	02358593          	addi	a1,a1,35 # 20023 <__l1_heapsram_size+0x93>
1c01523e:	4561                	li	a0,24
1c015240:	01f01013          	slli	zero,zero,0x1f
1c015244:	00100073          	ebreak
1c015248:	40705013          	srai	zero,zero,0x7
1c01524c:	1a1047b7          	lui	a5,0x1a104
1c015250:	c1f44433          	p.bset	s0,s0,0,31
1c015254:	0c87a023          	sw	s0,192(a5) # 1a1040c0 <__l1_heapsram_end+0xa0e40c0>
1c015258:	10500073          	wfi
1c01525c:	40b2                	lw	ra,12(sp)
1c01525e:	4422                	lw	s0,8(sp)
1c015260:	0141                	addi	sp,sp,16
1c015262:	8082                	ret
1c015264:	02658593          	addi	a1,a1,38
1c015268:	bfd9                	j	1c01523e <system_exit+0x20>
1c01526a:	10500073          	wfi
1c01526e:	8082                	ret

1c015270 <__pi_timer_init__>:
1c015270:	06400513          	li	a0,100
1c015274:	bddd                	j	1c01516a <system_setup_systick>

1c015276 <__pi_timer_deinit__>:
1c015276:	1c01e537          	lui	a0,0x1c01e
1c01527a:	b5050513          	addi	a0,a0,-1200 # 1c01db50 <sys_timer_hi_prec>
1c01527e:	0850406f          	j	1c019b02 <pi_timer_close>

1c015282 <pi_bsp_init>:
1c015282:	540007b7          	lui	a5,0x54000
1c015286:	7139                	addi	sp,sp,-64
1c015288:	0785                	addi	a5,a5,1
1c01528a:	cc3e                	sw	a5,24(sp)
1c01528c:	555557b7          	lui	a5,0x55555
1c015290:	55578713          	addi	a4,a5,1365 # 55555555 <__heapl2ram_size+0x393c5555>
1c015294:	0785                	addi	a5,a5,1
1c015296:	d63e                	sw	a5,44(sp)
1c015298:	040c27b7          	lui	a5,0x40c2
1c01529c:	04078793          	addi	a5,a5,64 # 40c2040 <__l1_heapsram_size+0x40a20b0>
1c0152a0:	c43e                	sw	a5,8(sp)
1c0152a2:	0c2897b7          	lui	a5,0xc289
1c0152a6:	20678793          	addi	a5,a5,518 # c289206 <__l1_heapsram_size+0xc269276>
1c0152aa:	c63e                	sw	a5,12(sp)
1c0152ac:	144d27b7          	lui	a5,0x144d2
1c0152b0:	40e78793          	addi	a5,a5,1038 # 144d240e <__l1_heapsram_end+0x44b240e>
1c0152b4:	c83e                	sw	a5,16(sp)
1c0152b6:	2179b7b7          	lui	a5,0x2179b
1c0152ba:	61678793          	addi	a5,a5,1558 # 2179b616 <__heapl2ram_size+0x560b616>
1c0152be:	0828                	addi	a0,sp,24
1c0152c0:	de06                	sw	ra,60(sp)
1c0152c2:	ce3a                	sw	a4,28(sp)
1c0152c4:	d03a                	sw	a4,32(sp)
1c0152c6:	d23a                	sw	a4,36(sp)
1c0152c8:	d43a                	sw	a4,40(sp)
1c0152ca:	ca3e                	sw	a5,20(sp)
1c0152cc:	2a4020ef          	jal	ra,1c017570 <pi_pad_init>
1c0152d0:	0028                	addi	a0,sp,8
1c0152d2:	372020ef          	jal	ra,1c017644 <pi_pad_mux_group_init>
1c0152d6:	2ca9                	jal	1c015530 <pi_register_fpv2_flash_from_devicetree>
1c0152d8:	50f2                	lw	ra,60(sp)
1c0152da:	6121                	addi	sp,sp,64
1c0152dc:	8082                	ret

1c0152de <pi_flash_read>:
1c0152de:	411c                	lw	a5,0(a0)
1c0152e0:	0307a303          	lw	t1,48(a5)
1c0152e4:	8302                	jr	t1

1c0152e6 <__pi_fpv2_ptable_get_partition_by_uuid>:
1c0152e6:	872a                	mv	a4,a0
1c0152e8:	00374803          	lbu	a6,3(a4)
1c0152ec:	0541                	addi	a0,a0,16
1c0152ee:	4781                	li	a5,0
1c0152f0:	0107c463          	blt	a5,a6,1c0152f8 <__pi_fpv2_ptable_get_partition_by_uuid+0x12>
1c0152f4:	4515                	li	a0,5
1c0152f6:	8082                	ret
1c0152f8:	00055883          	lhu	a7,0(a0)
1c0152fc:	00b89d63          	bne	a7,a1,1c015316 <__pi_fpv2_ptable_get_partition_by_uuid+0x30>
1c015300:	c211                	beqz	a2,1c015304 <__pi_fpv2_ptable_get_partition_by_uuid+0x1e>
1c015302:	c21c                	sw	a5,0(a2)
1c015304:	4501                	li	a0,0
1c015306:	dae5                	beqz	a3,1c0152f6 <__pi_fpv2_ptable_get_partition_by_uuid+0x10>
1c015308:	08878793          	addi	a5,a5,136
1c01530c:	078a                	slli	a5,a5,0x2
1c01530e:	20f77783          	p.lw	a5,a5(a4)
1c015312:	c29c                	sw	a5,0(a3)
1c015314:	8082                	ret
1c015316:	02050513          	addi	a0,a0,32
1c01531a:	0785                	addi	a5,a5,1
1c01531c:	bfd1                	j	1c0152f0 <__pi_fpv2_ptable_get_partition_by_uuid+0xa>

1c01531e <__pi_fpv2_vtable_get_next_volume>:
1c01531e:	01554703          	lbu	a4,21(a0)
1c015322:	00271793          	slli	a5,a4,0x2
1c015326:	97ba                	add	a5,a5,a4
1c015328:	078a                	slli	a5,a5,0x2
1c01532a:	07e1                	addi	a5,a5,24
1c01532c:	953e                	add	a0,a0,a5
1c01532e:	8082                	ret

1c015330 <__pi_fpv2_l2_malloc>:
1c015330:	1101                	addi	sp,sp,-32
1c015332:	ce06                	sw	ra,28(sp)
1c015334:	c62a                	sw	a0,12(sp)
1c015336:	208050ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c01533a:	4632                	lw	a2,12(sp)
1c01533c:	4581                	li	a1,0
1c01533e:	495040ef          	jal	ra,1c019fd2 <memset>
1c015342:	86aa                	mv	a3,a0
1c015344:	c901                	beqz	a0,1c015354 <__pi_fpv2_l2_malloc+0x24>
1c015346:	1c01d7b7          	lui	a5,0x1c01d
1c01534a:	78c78793          	addi	a5,a5,1932 # 1c01d78c <debug_info>
1c01534e:	4398                	lw	a4,0(a5)
1c015350:	0705                	addi	a4,a4,1
1c015352:	c398                	sw	a4,0(a5)
1c015354:	40f2                	lw	ra,28(sp)
1c015356:	8536                	mv	a0,a3
1c015358:	6105                	addi	sp,sp,32
1c01535a:	8082                	ret

1c01535c <__pi_fpv2_l2_free>:
1c01535c:	1141                	addi	sp,sp,-16
1c01535e:	c606                	sw	ra,12(sp)
1c015360:	0e4050ef          	jal	ra,1c01a444 <pi_l2_free>
1c015364:	1c01d7b7          	lui	a5,0x1c01d
1c015368:	78c78793          	addi	a5,a5,1932 # 1c01d78c <debug_info>
1c01536c:	4398                	lw	a4,0(a5)
1c01536e:	40b2                	lw	ra,12(sp)
1c015370:	177d                	addi	a4,a4,-1
1c015372:	c398                	sw	a4,0(a5)
1c015374:	0141                	addi	sp,sp,16
1c015376:	8082                	ret

1c015378 <__pi_fpv2_reg_entry_get_by_ref.isra.2>:
1c015378:	1c01d7b7          	lui	a5,0x1c01d
1c01537c:	b507a783          	lw	a5,-1200(a5) # 1c01cb50 <device_reg+0x4>
1c015380:	e399                	bnez	a5,1c015386 <__pi_fpv2_reg_entry_get_by_ref.isra.2+0xe>
1c015382:	853e                	mv	a0,a5
1c015384:	8082                	ret
1c015386:	0007c683          	lbu	a3,0(a5)
1c01538a:	00054703          	lbu	a4,0(a0)
1c01538e:	00e69e63          	bne	a3,a4,1c0153aa <__pi_fpv2_reg_entry_get_by_ref.isra.2+0x32>
1c015392:	0017c683          	lbu	a3,1(a5)
1c015396:	00154703          	lbu	a4,1(a0)
1c01539a:	00e69863          	bne	a3,a4,1c0153aa <__pi_fpv2_reg_entry_get_by_ref.isra.2+0x32>
1c01539e:	0027c683          	lbu	a3,2(a5)
1c0153a2:	00254703          	lbu	a4,2(a0)
1c0153a6:	fce68ee3          	beq	a3,a4,1c015382 <__pi_fpv2_reg_entry_get_by_ref.isra.2+0xa>
1c0153aa:	479c                	lw	a5,8(a5)
1c0153ac:	bfd1                	j	1c015380 <__pi_fpv2_reg_entry_get_by_ref.isra.2+0x8>

1c0153ae <__pi_fpv2_reg_device_get_ref.constprop.9>:
1c0153ae:	cd8d                	beqz	a1,1c0153e8 <__pi_fpv2_reg_device_get_ref.constprop.9+0x3a>
1c0153b0:	1c01d7b7          	lui	a5,0x1c01d
1c0153b4:	b507a783          	lw	a5,-1200(a5) # 1c01cb50 <device_reg+0x4>
1c0153b8:	4701                	li	a4,0
1c0153ba:	4681                	li	a3,0
1c0153bc:	e781                	bnez	a5,1c0153c4 <__pi_fpv2_reg_device_get_ref.constprop.9+0x16>
1c0153be:	eb19                	bnez	a4,1c0153d4 <__pi_fpv2_reg_device_get_ref.constprop.9+0x26>
1c0153c0:	4515                	li	a0,5
1c0153c2:	8082                	ret
1c0153c4:	43d0                	lw	a2,4(a5)
1c0153c6:	00a61563          	bne	a2,a0,1c0153d0 <__pi_fpv2_reg_device_get_ref.constprop.9+0x22>
1c0153ca:	e311                	bnez	a4,1c0153ce <__pi_fpv2_reg_device_get_ref.constprop.9+0x20>
1c0153cc:	86be                	mv	a3,a5
1c0153ce:	0705                	addi	a4,a4,1
1c0153d0:	479c                	lw	a5,8(a5)
1c0153d2:	b7ed                	j	1c0153bc <__pi_fpv2_reg_device_get_ref.constprop.9+0xe>
1c0153d4:	0006d783          	lhu	a5,0(a3)
1c0153d8:	4501                	li	a0,0
1c0153da:	00f59023          	sh	a5,0(a1)
1c0153de:	0026c783          	lbu	a5,2(a3)
1c0153e2:	00f58123          	sb	a5,2(a1)
1c0153e6:	8082                	ret
1c0153e8:	4509                	li	a0,2
1c0153ea:	8082                	ret

1c0153ec <pi_fpv2_ptable_offset_get>:
1c0153ec:	7119                	addi	sp,sp,-128
1c0153ee:	daa6                	sw	s1,116(sp)
1c0153f0:	84ae                	mv	s1,a1
1c0153f2:	006c                	addi	a1,sp,12
1c0153f4:	dca2                	sw	s0,120(sp)
1c0153f6:	d8ca                	sw	s2,112(sp)
1c0153f8:	de86                	sw	ra,124(sp)
1c0153fa:	892a                	mv	s2,a0
1c0153fc:	8432                	mv	s0,a2
1c0153fe:	c402                	sw	zero,8(sp)
1c015400:	395010ef          	jal	ra,1c016f94 <pi_open>
1c015404:	e539                	bnez	a0,1c015452 <pi_fpv2_ptable_offset_get+0x66>
1c015406:	4532                	lw	a0,12(sp)
1c015408:	4691                	li	a3,4
1c01540a:	0030                	addi	a2,sp,8
1c01540c:	4581                	li	a1,0
1c01540e:	3dc1                	jal	1c0152de <pi_flash_read>
1c015410:	45a2                	lw	a1,8(sp)
1c015412:	4532                	lw	a0,12(sp)
1c015414:	06000693          	li	a3,96
1c015418:	0810                	addi	a2,sp,16
1c01541a:	35d1                	jal	1c0152de <pi_flash_read>
1c01541c:	deadc7b7          	lui	a5,0xdeadc
1c015420:	4745                	li	a4,17
1c015422:	eef78793          	addi	a5,a5,-273 # deadbeef <pulp__FC+0xdeadbef0>
1c015426:	02e41463          	bne	s0,a4,1c01544e <pi_fpv2_ptable_offset_get+0x62>
1c01542a:	5716                	lw	a4,100(sp)
1c01542c:	440d                	li	s0,3
1c01542e:	00f70463          	beq	a4,a5,1c015436 <pi_fpv2_ptable_offset_get+0x4a>
1c015432:	c098                	sw	a4,0(s1)
1c015434:	4401                	li	s0,0
1c015436:	854a                	mv	a0,s2
1c015438:	409010ef          	jal	ra,1c017040 <pi_close>
1c01543c:	c111                	beqz	a0,1c015440 <pi_fpv2_ptable_offset_get+0x54>
1c01543e:	842a                	mv	s0,a0
1c015440:	8522                	mv	a0,s0
1c015442:	50f6                	lw	ra,124(sp)
1c015444:	5466                	lw	s0,120(sp)
1c015446:	54d6                	lw	s1,116(sp)
1c015448:	5946                	lw	s2,112(sp)
1c01544a:	6109                	addi	sp,sp,128
1c01544c:	8082                	ret
1c01544e:	4756                	lw	a4,84(sp)
1c015450:	bff1                	j	1c01542c <pi_fpv2_ptable_offset_get+0x40>
1c015452:	4405                	li	s0,1
1c015454:	b7f5                	j	1c015440 <pi_fpv2_ptable_offset_get+0x54>

1c015456 <pi_fpv2_device_register>:
1c015456:	c55d                	beqz	a0,1c015504 <pi_fpv2_device_register+0xae>
1c015458:	00054703          	lbu	a4,0(a0)
1c01545c:	4789                	li	a5,2
1c01545e:	0ae7e463          	bltu	a5,a4,1c015506 <pi_fpv2_device_register+0xb0>
1c015462:	1101                	addi	sp,sp,-32
1c015464:	ca26                	sw	s1,20(sp)
1c015466:	84ae                	mv	s1,a1
1c015468:	cc22                	sw	s0,24(sp)
1c01546a:	006c                	addi	a1,sp,12
1c01546c:	842a                	mv	s0,a0
1c01546e:	8526                	mv	a0,s1
1c015470:	ce06                	sw	ra,28(sp)
1c015472:	3f35                	jal	1c0153ae <__pi_fpv2_reg_device_get_ref.constprop.9>
1c015474:	ed2d                	bnez	a0,1c0154ee <pi_fpv2_device_register+0x98>
1c015476:	00044683          	lbu	a3,0(s0)
1c01547a:	00c14703          	lbu	a4,12(sp)
1c01547e:	47b1                	li	a5,12
1c015480:	06e69a63          	bne	a3,a4,1c0154f4 <pi_fpv2_device_register+0x9e>
1c015484:	00144683          	lbu	a3,1(s0)
1c015488:	00d14703          	lbu	a4,13(sp)
1c01548c:	06e69463          	bne	a3,a4,1c0154f4 <pi_fpv2_device_register+0x9e>
1c015490:	00244683          	lbu	a3,2(s0)
1c015494:	00e14703          	lbu	a4,14(sp)
1c015498:	04e69e63          	bne	a3,a4,1c0154f4 <pi_fpv2_device_register+0x9e>
1c01549c:	8522                	mv	a0,s0
1c01549e:	3de9                	jal	1c015378 <__pi_fpv2_reg_entry_get_by_ref.isra.2>
1c0154a0:	e521                	bnez	a0,1c0154e8 <pi_fpv2_device_register+0x92>
1c0154a2:	4531                	li	a0,12
1c0154a4:	3571                	jal	1c015330 <__pi_fpv2_l2_malloc>
1c0154a6:	20100793          	li	a5,513
1c0154aa:	c529                	beqz	a0,1c0154f4 <pi_fpv2_device_register+0x9e>
1c0154ac:	0ff00793          	li	a5,255
1c0154b0:	00f51023          	sh	a5,0(a0)
1c0154b4:	00050123          	sb	zero,2(a0)
1c0154b8:	57fd                	li	a5,-1
1c0154ba:	c15c                	sw	a5,4(a0)
1c0154bc:	00045783          	lhu	a5,0(s0)
1c0154c0:	00052423          	sw	zero,8(a0)
1c0154c4:	00f51023          	sh	a5,0(a0)
1c0154c8:	00244783          	lbu	a5,2(s0)
1c0154cc:	00f50123          	sb	a5,2(a0)
1c0154d0:	1c01d7b7          	lui	a5,0x1c01d
1c0154d4:	b4c7a703          	lw	a4,-1204(a5) # 1c01cb4c <device_reg>
1c0154d8:	b4c78793          	addi	a5,a5,-1204
1c0154dc:	c315                	beqz	a4,1c015500 <pi_fpv2_device_register+0xaa>
1c0154de:	4794                	lw	a3,8(a5)
1c0154e0:	c688                	sw	a0,8(a3)
1c0154e2:	0705                	addi	a4,a4,1
1c0154e4:	c788                	sw	a0,8(a5)
1c0154e6:	c398                	sw	a4,0(a5)
1c0154e8:	c144                	sw	s1,4(a0)
1c0154ea:	4781                	li	a5,0
1c0154ec:	a021                	j	1c0154f4 <pi_fpv2_device_register+0x9e>
1c0154ee:	4785                	li	a5,1
1c0154f0:	fa5526e3          	p.beqimm	a0,5,1c01549c <pi_fpv2_device_register+0x46>
1c0154f4:	40f2                	lw	ra,28(sp)
1c0154f6:	4462                	lw	s0,24(sp)
1c0154f8:	44d2                	lw	s1,20(sp)
1c0154fa:	853e                	mv	a0,a5
1c0154fc:	6105                	addi	sp,sp,32
1c0154fe:	8082                	ret
1c015500:	c3c8                	sw	a0,4(a5)
1c015502:	b7c5                	j	1c0154e2 <pi_fpv2_device_register+0x8c>
1c015504:	4789                	li	a5,2
1c015506:	853e                	mv	a0,a5
1c015508:	8082                	ret

1c01550a <pi_fpv2_device_get_by_ref>:
1c01550a:	c105                	beqz	a0,1c01552a <pi_fpv2_device_get_by_ref+0x20>
1c01550c:	1141                	addi	sp,sp,-16
1c01550e:	c422                	sw	s0,8(sp)
1c015510:	c606                	sw	ra,12(sp)
1c015512:	842e                	mv	s0,a1
1c015514:	3595                	jal	1c015378 <__pi_fpv2_reg_entry_get_by_ref.isra.2>
1c015516:	4795                	li	a5,5
1c015518:	c501                	beqz	a0,1c015520 <pi_fpv2_device_get_by_ref+0x16>
1c01551a:	415c                	lw	a5,4(a0)
1c01551c:	c01c                	sw	a5,0(s0)
1c01551e:	4781                	li	a5,0
1c015520:	40b2                	lw	ra,12(sp)
1c015522:	4422                	lw	s0,8(sp)
1c015524:	853e                	mv	a0,a5
1c015526:	0141                	addi	sp,sp,16
1c015528:	8082                	ret
1c01552a:	4789                	li	a5,2
1c01552c:	853e                	mv	a0,a5
1c01552e:	8082                	ret

1c015530 <pi_register_fpv2_flash_from_devicetree>:
1c015530:	1141                	addi	sp,sp,-16
1c015532:	c606                	sw	ra,12(sp)
1c015534:	c422                	sw	s0,8(sp)
1c015536:	c226                	sw	s1,4(sp)
1c015538:	c04a                	sw	s2,0(sp)
1c01553a:	3af010ef          	jal	ra,1c0170e8 <dt_fpv2_memory_ref_get>
1c01553e:	842a                	mv	s0,a0
1c015540:	3b3010ef          	jal	ra,1c0170f2 <dt_fpv2_memory_ref_size_get>
1c015544:	892a                	mv	s2,a0
1c015546:	0411                	addi	s0,s0,4
1c015548:	4481                	li	s1,0
1c01554a:	01249963          	bne	s1,s2,1c01555c <pi_register_fpv2_flash_from_devicetree+0x2c>
1c01554e:	4501                	li	a0,0
1c015550:	40b2                	lw	ra,12(sp)
1c015552:	4422                	lw	s0,8(sp)
1c015554:	4492                	lw	s1,4(sp)
1c015556:	4902                	lw	s2,0(sp)
1c015558:	0141                	addi	sp,sp,16
1c01555a:	8082                	ret
1c01555c:	8522                	mv	a0,s0
1c01555e:	ffc5258b          	p.lw	a1,-4(a0!)
1c015562:	0421                	addi	s0,s0,8
1c015564:	3dcd                	jal	1c015456 <pi_fpv2_device_register>
1c015566:	f56d                	bnez	a0,1c015550 <pi_register_fpv2_flash_from_devicetree+0x20>
1c015568:	0485                	addi	s1,s1,1
1c01556a:	b7c5                	j	1c01554a <pi_register_fpv2_flash_from_devicetree+0x1a>

1c01556c <pi_fpv2_ptable_conf_init>:
1c01556c:	c919                	beqz	a0,1c015582 <pi_fpv2_ptable_conf_init+0x16>
1c01556e:	47c1                	li	a5,16
1c015570:	00051023          	sh	zero,0(a0)
1c015574:	00050123          	sb	zero,2(a0)
1c015578:	00052223          	sw	zero,4(a0)
1c01557c:	c51c                	sw	a5,8(a0)
1c01557e:	00051623          	sh	zero,12(a0)
1c015582:	8082                	ret

1c015584 <pi_fpv2_ptable_close>:
1c015584:	cd1d                	beqz	a0,1c0155c2 <pi_fpv2_ptable_close+0x3e>
1c015586:	1141                	addi	sp,sp,-16
1c015588:	c422                	sw	s0,8(sp)
1c01558a:	c226                	sw	s1,4(sp)
1c01558c:	c04a                	sw	s2,0(sp)
1c01558e:	c606                	sw	ra,12(sp)
1c015590:	842a                	mv	s0,a0
1c015592:	22050913          	addi	s2,a0,544
1c015596:	4481                	li	s1,0
1c015598:	00344783          	lbu	a5,3(s0)
1c01559c:	00f4cd63          	blt	s1,a5,1c0155b6 <pi_fpv2_ptable_close+0x32>
1c0155a0:	8522                	mv	a0,s0
1c0155a2:	26000593          	li	a1,608
1c0155a6:	3b5d                	jal	1c01535c <__pi_fpv2_l2_free>
1c0155a8:	40b2                	lw	ra,12(sp)
1c0155aa:	4422                	lw	s0,8(sp)
1c0155ac:	4492                	lw	s1,4(sp)
1c0155ae:	4902                	lw	s2,0(sp)
1c0155b0:	4501                	li	a0,0
1c0155b2:	0141                	addi	sp,sp,16
1c0155b4:	8082                	ret
1c0155b6:	0049250b          	p.lw	a0,4(s2!)
1c0155ba:	45a1                	li	a1,8
1c0155bc:	0485                	addi	s1,s1,1
1c0155be:	3b79                	jal	1c01535c <__pi_fpv2_l2_free>
1c0155c0:	bfe1                	j	1c015598 <pi_fpv2_ptable_close+0x14>
1c0155c2:	4505                	li	a0,1
1c0155c4:	8082                	ret

1c0155c6 <pi_fpv2_ptable_load>:
1c0155c6:	7179                	addi	sp,sp,-48
1c0155c8:	d226                	sw	s1,36(sp)
1c0155ca:	d606                	sw	ra,44(sp)
1c0155cc:	d422                	sw	s0,40(sp)
1c0155ce:	d04a                	sw	s2,32(sp)
1c0155d0:	ce4e                	sw	s3,28(sp)
1c0155d2:	cc52                	sw	s4,24(sp)
1c0155d4:	ca56                	sw	s5,20(sp)
1c0155d6:	c85a                	sw	s6,16(sp)
1c0155d8:	c402                	sw	zero,8(sp)
1c0155da:	4489                	li	s1,2
1c0155dc:	cd29                	beqz	a0,1c015636 <pi_fpv2_ptable_load+0x70>
1c0155de:	892e                	mv	s2,a1
1c0155e0:	c9b9                	beqz	a1,1c015636 <pi_fpv2_ptable_load+0x70>
1c0155e2:	41dc                	lw	a5,4(a1)
1c0155e4:	f647b7b3          	p.bclr	a5,a5,27,4
1c0155e8:	e7b9                	bnez	a5,1c015636 <pi_fpv2_ptable_load+0x70>
1c0155ea:	8aaa                	mv	s5,a0
1c0155ec:	004c                	addi	a1,sp,4
1c0155ee:	854a                	mv	a0,s2
1c0155f0:	3f29                	jal	1c01550a <pi_fpv2_device_get_by_ref>
1c0155f2:	c119                	beqz	a0,1c0155f8 <pi_fpv2_ptable_load+0x32>
1c0155f4:	4485                	li	s1,1
1c0155f6:	a081                	j	1c015636 <pi_fpv2_ptable_load+0x70>
1c0155f8:	4512                	lw	a0,4(sp)
1c0155fa:	002c                	addi	a1,sp,8
1c0155fc:	199010ef          	jal	ra,1c016f94 <pi_open>
1c015600:	f975                	bnez	a0,1c0155f4 <pi_fpv2_ptable_load+0x2e>
1c015602:	26000513          	li	a0,608
1c015606:	332d                	jal	1c015330 <__pi_fpv2_l2_malloc>
1c015608:	842a                	mv	s0,a0
1c01560a:	20100493          	li	s1,513
1c01560e:	c505                	beqz	a0,1c015636 <pi_fpv2_ptable_load+0x70>
1c015610:	4792                	lw	a5,4(sp)
1c015612:	00492583          	lw	a1,4(s2)
1c015616:	862a                	mv	a2,a0
1c015618:	20f52823          	sw	a5,528(a0)
1c01561c:	20b52a23          	sw	a1,532(a0)
1c015620:	4522                	lw	a0,8(sp)
1c015622:	46c1                	li	a3,16
1c015624:	396d                	jal	1c0152de <pi_flash_read>
1c015626:	45ad                	li	a1,11
1c015628:	8522                	mv	a0,s0
1c01562a:	2b4d                	jal	1c015bdc <pi_crc32_compute>
1c01562c:	00b42783          	lw	a5,11(s0)
1c015630:	02f50363          	beq	a0,a5,1c015656 <pi_fpv2_ptable_load+0x90>
1c015634:	44a1                	li	s1,8
1c015636:	47a2                	lw	a5,8(sp)
1c015638:	c781                	beqz	a5,1c015640 <pi_fpv2_ptable_load+0x7a>
1c01563a:	4512                	lw	a0,4(sp)
1c01563c:	205010ef          	jal	ra,1c017040 <pi_close>
1c015640:	50b2                	lw	ra,44(sp)
1c015642:	5422                	lw	s0,40(sp)
1c015644:	8526                	mv	a0,s1
1c015646:	5902                	lw	s2,32(sp)
1c015648:	5492                	lw	s1,36(sp)
1c01564a:	49f2                	lw	s3,28(sp)
1c01564c:	4a62                	lw	s4,24(sp)
1c01564e:	4ad2                	lw	s5,20(sp)
1c015650:	4b42                	lw	s6,16(sp)
1c015652:	6145                	addi	sp,sp,48
1c015654:	8082                	ret
1c015656:	00344683          	lbu	a3,3(s0)
1c01565a:	21442583          	lw	a1,532(s0)
1c01565e:	4522                	lw	a0,8(sp)
1c015660:	01040913          	addi	s2,s0,16
1c015664:	0696                	slli	a3,a3,0x5
1c015666:	864a                	mv	a2,s2
1c015668:	05c1                	addi	a1,a1,16
1c01566a:	3995                	jal	1c0152de <pi_flash_read>
1c01566c:	00344583          	lbu	a1,3(s0)
1c015670:	854a                	mv	a0,s2
1c015672:	0596                	slli	a1,a1,0x5
1c015674:	23a5                	jal	1c015bdc <pi_crc32_compute>
1c015676:	00742783          	lw	a5,7(s0)
1c01567a:	faf51de3          	bne	a0,a5,1c015634 <pi_fpv2_ptable_load+0x6e>
1c01567e:	22040b13          	addi	s6,s0,544
1c015682:	84ca                	mv	s1,s2
1c015684:	4a01                	li	s4,0
1c015686:	00344603          	lbu	a2,3(s0)
1c01568a:	00ca4d63          	blt	s4,a2,1c0156a4 <pi_fpv2_ptable_load+0xde>
1c01568e:	4701                	li	a4,0
1c015690:	4781                	li	a5,0
1c015692:	4485                	li	s1,1
1c015694:	02c76e63          	bltu	a4,a2,1c0156d0 <pi_fpv2_ptable_load+0x10a>
1c015698:	20f41e23          	sh	a5,540(s0)
1c01569c:	fca1                	bnez	s1,1c0155f4 <pi_fpv2_ptable_load+0x2e>
1c01569e:	008aa023          	sw	s0,0(s5)
1c0156a2:	bf51                	j	1c015636 <pi_fpv2_ptable_load+0x70>
1c0156a4:	4521                	li	a0,8
1c0156a6:	3169                	jal	1c015330 <__pi_fpv2_l2_malloc>
1c0156a8:	89aa                	mv	s3,a0
1c0156aa:	c105                	beqz	a0,1c0156ca <pi_fpv2_ptable_load+0x104>
1c0156ac:	c144                	sw	s1,4(a0)
1c0156ae:	006c                	addi	a1,sp,12
1c0156b0:	00648513          	addi	a0,s1,6
1c0156b4:	3d99                	jal	1c01550a <pi_fpv2_device_get_by_ref>
1c0156b6:	02048493          	addi	s1,s1,32
1c0156ba:	fd0d                	bnez	a0,1c0155f4 <pi_fpv2_ptable_load+0x2e>
1c0156bc:	47b2                	lw	a5,12(sp)
1c0156be:	0a05                	addi	s4,s4,1
1c0156c0:	00f9a023          	sw	a5,0(s3)
1c0156c4:	013b222b          	p.sw	s3,4(s6!)
1c0156c8:	bf7d                	j	1c015686 <pi_fpv2_ptable_load+0xc0>
1c0156ca:	20100493          	li	s1,513
1c0156ce:	b7a5                	j	1c015636 <pi_fpv2_ptable_load+0x70>
1c0156d0:	00095683          	lhu	a3,0(s2)
1c0156d4:	00d7e863          	bltu	a5,a3,1c0156e4 <pi_fpv2_ptable_load+0x11e>
1c0156d8:	86be                	mv	a3,a5
1c0156da:	0705                	addi	a4,a4,1
1c0156dc:	02090913          	addi	s2,s2,32
1c0156e0:	87b6                	mv	a5,a3
1c0156e2:	bf4d                	j	1c015694 <pi_fpv2_ptable_load+0xce>
1c0156e4:	4481                	li	s1,0
1c0156e6:	bfd5                	j	1c0156da <pi_fpv2_ptable_load+0x114>

1c0156e8 <pi_fpv2_ptable_partition_info_get>:
1c0156e8:	415c                	lw	a5,4(a0)
1c0156ea:	4501                	li	a0,0
1c0156ec:	0007d703          	lhu	a4,0(a5)
1c0156f0:	00e59023          	sh	a4,0(a1)
1c0156f4:	0027c703          	lbu	a4,2(a5)
1c0156f8:	00e58123          	sb	a4,2(a1)
1c0156fc:	0037c703          	lbu	a4,3(a5)
1c015700:	00e581a3          	sb	a4,3(a1)
1c015704:	0047d703          	lhu	a4,4(a5)
1c015708:	00e59223          	sh	a4,4(a1)
1c01570c:	0067a703          	lw	a4,6(a5)
1c015710:	00e5a323          	sw	a4,6(a1)
1c015714:	00a7a703          	lw	a4,10(a5)
1c015718:	00e5a523          	sw	a4,10(a1)
1c01571c:	00e7a703          	lw	a4,14(a5)
1c015720:	00e5a723          	sw	a4,14(a1)
1c015724:	0127d703          	lhu	a4,18(a5)
1c015728:	0147c783          	lbu	a5,20(a5)
1c01572c:	00e59923          	sh	a4,18(a1)
1c015730:	00f58a23          	sb	a5,20(a1)
1c015734:	8082                	ret

1c015736 <pi_fpv2_ptable_partition_device_get>:
1c015736:	411c                	lw	a5,0(a0)
1c015738:	4501                	li	a0,0
1c01573a:	c19c                	sw	a5,0(a1)
1c01573c:	8082                	ret

1c01573e <pi_fpv2_vtable_close>:
1c01573e:	451c                	lw	a5,8(a0)
1c015740:	7179                	addi	sp,sp,-48
1c015742:	d422                	sw	s0,40(sp)
1c015744:	43dc                	lw	a5,4(a5)
1c015746:	842a                	mv	s0,a0
1c015748:	4148                	lw	a0,4(a0)
1c01574a:	0007d583          	lhu	a1,0(a5)
1c01574e:	0074                	addi	a3,sp,12
1c015750:	4601                	li	a2,0
1c015752:	d226                	sw	s1,36(sp)
1c015754:	d606                	sw	ra,44(sp)
1c015756:	d04a                	sw	s2,32(sp)
1c015758:	ce4e                	sw	s3,28(sp)
1c01575a:	cc52                	sw	s4,24(sp)
1c01575c:	3669                	jal	1c0152e6 <__pi_fpv2_ptable_get_partition_by_uuid>
1c01575e:	4495                	li	s1,5
1c015760:	e901                	bnez	a0,1c015770 <pi_fpv2_vtable_close+0x32>
1c015762:	4489                	li	s1,2
1c015764:	c411                	beqz	s0,1c015770 <pi_fpv2_vtable_close+0x32>
1c015766:	00042903          	lw	s2,0(s0)
1c01576a:	00091c63          	bnez	s2,1c015782 <pi_fpv2_vtable_close+0x44>
1c01576e:	4485                	li	s1,1
1c015770:	50b2                	lw	ra,44(sp)
1c015772:	5422                	lw	s0,40(sp)
1c015774:	8526                	mv	a0,s1
1c015776:	5902                	lw	s2,32(sp)
1c015778:	5492                	lw	s1,36(sp)
1c01577a:	49f2                	lw	s3,28(sp)
1c01577c:	4a62                	lw	s4,24(sp)
1c01577e:	6145                	addi	sp,sp,48
1c015780:	8082                	ret
1c015782:	84aa                	mv	s1,a0
1c015784:	00c40a13          	addi	s4,s0,12
1c015788:	4981                	li	s3,0
1c01578a:	00294783          	lbu	a5,2(s2)
1c01578e:	00f9ec63          	bltu	s3,a5,1c0157a6 <pi_fpv2_vtable_close+0x68>
1c015792:	00392583          	lw	a1,3(s2)
1c015796:	4008                	lw	a0,0(s0)
1c015798:	05bd                	addi	a1,a1,15
1c01579a:	36c9                	jal	1c01535c <__pi_fpv2_l2_free>
1c01579c:	04c00593          	li	a1,76
1c0157a0:	8522                	mv	a0,s0
1c0157a2:	3e6d                	jal	1c01535c <__pi_fpv2_l2_free>
1c0157a4:	b7f1                	j	1c015770 <pi_fpv2_vtable_close+0x32>
1c0157a6:	004a250b          	p.lw	a0,4(s4!)
1c0157aa:	d171                	beqz	a0,1c01576e <pi_fpv2_vtable_close+0x30>
1c0157ac:	45a1                	li	a1,8
1c0157ae:	367d                	jal	1c01535c <__pi_fpv2_l2_free>
1c0157b0:	0985                	addi	s3,s3,1
1c0157b2:	bfe1                	j	1c01578a <pi_fpv2_vtable_close+0x4c>

1c0157b4 <pi_fpv2_vtable_load>:
1c0157b4:	7179                	addi	sp,sp,-48
1c0157b6:	d04a                	sw	s2,32(sp)
1c0157b8:	cc52                	sw	s4,24(sp)
1c0157ba:	ca56                	sw	s5,20(sp)
1c0157bc:	d606                	sw	ra,44(sp)
1c0157be:	d422                	sw	s0,40(sp)
1c0157c0:	d226                	sw	s1,36(sp)
1c0157c2:	ce4e                	sw	s3,28(sp)
1c0157c4:	c85a                	sw	s6,16(sp)
1c0157c6:	892a                	mv	s2,a0
1c0157c8:	8aae                	mv	s5,a1
1c0157ca:	01250693          	addi	a3,a0,18
1c0157ce:	00354583          	lbu	a1,3(a0)
1c0157d2:	c602                	sw	zero,12(sp)
1c0157d4:	4701                	li	a4,0
1c0157d6:	4781                	li	a5,0
1c0157d8:	4a01                	li	s4,0
1c0157da:	0e100513          	li	a0,225
1c0157de:	04b76363          	bltu	a4,a1,1c015824 <pi_fpv2_vtable_load+0x70>
1c0157e2:	14078463          	beqz	a5,1c01592a <pi_fpv2_vtable_load+0x176>
1c0157e6:	4705                	li	a4,1
1c0157e8:	4485                	li	s1,1
1c0157ea:	00f76a63          	bltu	a4,a5,1c0157fe <pi_fpv2_vtable_load+0x4a>
1c0157ee:	04c00513          	li	a0,76
1c0157f2:	b3fff0ef          	jal	ra,1c015330 <__pi_fpv2_l2_malloc>
1c0157f6:	842a                	mv	s0,a0
1c0157f8:	e929                	bnez	a0,1c01584a <pi_fpv2_vtable_load+0x96>
1c0157fa:	20100493          	li	s1,513
1c0157fe:	47b2                	lw	a5,12(sp)
1c015800:	c799                	beqz	a5,1c01580e <pi_fpv2_vtable_load+0x5a>
1c015802:	21092503          	lw	a0,528(s2)
1c015806:	03b010ef          	jal	ra,1c017040 <pi_close>
1c01580a:	10051163          	bnez	a0,1c01590c <pi_fpv2_vtable_load+0x158>
1c01580e:	50b2                	lw	ra,44(sp)
1c015810:	5422                	lw	s0,40(sp)
1c015812:	8526                	mv	a0,s1
1c015814:	5902                	lw	s2,32(sp)
1c015816:	5492                	lw	s1,36(sp)
1c015818:	49f2                	lw	s3,28(sp)
1c01581a:	4a62                	lw	s4,24(sp)
1c01581c:	4ad2                	lw	s5,20(sp)
1c01581e:	4b42                	lw	s6,16(sp)
1c015820:	6145                	addi	sp,sp,48
1c015822:	8082                	ret
1c015824:	0006c603          	lbu	a2,0(a3)
1c015828:	00263d63          	p.bneimm	a2,2,1c015842 <pi_fpv2_vtable_load+0x8e>
1c01582c:	0016c603          	lbu	a2,1(a3)
1c015830:	00a61963          	bne	a2,a0,1c015842 <pi_fpv2_vtable_load+0x8e>
1c015834:	e791                	bnez	a5,1c015840 <pi_fpv2_vtable_load+0x8c>
1c015836:	00271613          	slli	a2,a4,0x2
1c01583a:	964a                	add	a2,a2,s2
1c01583c:	22062a03          	lw	s4,544(a2)
1c015840:	0785                	addi	a5,a5,1
1c015842:	0705                	addi	a4,a4,1
1c015844:	02068693          	addi	a3,a3,32
1c015848:	bf59                	j	1c0157de <pi_fpv2_vtable_load+0x2a>
1c01584a:	21092503          	lw	a0,528(s2)
1c01584e:	006c                	addi	a1,sp,12
1c015850:	448d                	li	s1,3
1c015852:	742010ef          	jal	ra,1c016f94 <pi_open>
1c015856:	f545                	bnez	a0,1c0157fe <pi_fpv2_vtable_load+0x4a>
1c015858:	453d                	li	a0,15
1c01585a:	ad7ff0ef          	jal	ra,1c015330 <__pi_fpv2_l2_malloc>
1c01585e:	004a2783          	lw	a5,4(s4)
1c015862:	862a                	mv	a2,a0
1c015864:	c008                	sw	a0,0(s0)
1c015866:	0097a583          	lw	a1,9(a5)
1c01586a:	84aa                	mv	s1,a0
1c01586c:	4532                	lw	a0,12(sp)
1c01586e:	46bd                	li	a3,15
1c015870:	a6fff0ef          	jal	ra,1c0152de <pi_flash_read>
1c015874:	00042983          	lw	s3,0(s0)
1c015878:	45ad                	li	a1,11
1c01587a:	854e                	mv	a0,s3
1c01587c:	2685                	jal	1c015bdc <pi_crc32_compute>
1c01587e:	00b9a783          	lw	a5,11(s3)
1c015882:	00f50463          	beq	a0,a5,1c01588a <pi_fpv2_vtable_load+0xd6>
1c015886:	44a1                	li	s1,8
1c015888:	bf9d                	j	1c0157fe <pi_fpv2_vtable_load+0x4a>
1c01588a:	0034a483          	lw	s1,3(s1)
1c01588e:	4008                	lw	a0,0(s0)
1c015890:	45bd                	li	a1,15
1c015892:	04bd                	addi	s1,s1,15
1c015894:	ac9ff0ef          	jal	ra,1c01535c <__pi_fpv2_l2_free>
1c015898:	8526                	mv	a0,s1
1c01589a:	a97ff0ef          	jal	ra,1c015330 <__pi_fpv2_l2_malloc>
1c01589e:	c008                	sw	a0,0(s0)
1c0158a0:	89aa                	mv	s3,a0
1c0158a2:	dd21                	beqz	a0,1c0157fa <pi_fpv2_vtable_load+0x46>
1c0158a4:	004a2783          	lw	a5,4(s4)
1c0158a8:	862a                	mv	a2,a0
1c0158aa:	4532                	lw	a0,12(sp)
1c0158ac:	0097a583          	lw	a1,9(a5)
1c0158b0:	86a6                	mv	a3,s1
1c0158b2:	a2dff0ef          	jal	ra,1c0152de <pi_flash_read>
1c0158b6:	4004                	lw	s1,0(s0)
1c0158b8:	01242223          	sw	s2,4(s0)
1c0158bc:	01442423          	sw	s4,8(s0)
1c0158c0:	45ad                	li	a1,11
1c0158c2:	8526                	mv	a0,s1
1c0158c4:	2e21                	jal	1c015bdc <pi_crc32_compute>
1c0158c6:	00b4a783          	lw	a5,11(s1)
1c0158ca:	faf51ee3          	bne	a0,a5,1c015886 <pi_fpv2_vtable_load+0xd2>
1c0158ce:	4004                	lw	s1,0(s0)
1c0158d0:	0034a583          	lw	a1,3(s1)
1c0158d4:	00f48513          	addi	a0,s1,15
1c0158d8:	2611                	jal	1c015bdc <pi_crc32_compute>
1c0158da:	0074a783          	lw	a5,7(s1)
1c0158de:	faf514e3          	bne	a0,a5,1c015886 <pi_fpv2_vtable_load+0xd2>
1c0158e2:	8a22                	mv	s4,s0
1c0158e4:	00ca248b          	p.lw	s1,12(s4!)
1c0158e8:	4b01                	li	s6,0
1c0158ea:	04bd                	addi	s1,s1,15
1c0158ec:	0029c783          	lbu	a5,2(s3)
1c0158f0:	02fb6063          	bltu	s6,a5,1c015910 <pi_fpv2_vtable_load+0x15c>
1c0158f4:	000a8463          	beqz	s5,1c0158fc <pi_fpv2_vtable_load+0x148>
1c0158f8:	008aa023          	sw	s0,0(s5)
1c0158fc:	47b2                	lw	a5,12(sp)
1c0158fe:	4481                	li	s1,0
1c015900:	f00787e3          	beqz	a5,1c01580e <pi_fpv2_vtable_load+0x5a>
1c015904:	21092503          	lw	a0,528(s2)
1c015908:	738010ef          	jal	ra,1c017040 <pi_close>
1c01590c:	84aa                	mv	s1,a0
1c01590e:	b701                	j	1c01580e <pi_fpv2_vtable_load+0x5a>
1c015910:	4521                	li	a0,8
1c015912:	a1fff0ef          	jal	ra,1c015330 <__pi_fpv2_l2_malloc>
1c015916:	c104                	sw	s1,0(a0)
1c015918:	c140                	sw	s0,4(a0)
1c01591a:	00aa222b          	p.sw	a0,4(s4!)
1c01591e:	8526                	mv	a0,s1
1c015920:	9ffff0ef          	jal	ra,1c01531e <__pi_fpv2_vtable_get_next_volume>
1c015924:	84aa                	mv	s1,a0
1c015926:	0b05                	addi	s6,s6,1
1c015928:	b7d1                	j	1c0158ec <pi_fpv2_vtable_load+0x138>
1c01592a:	4495                	li	s1,5
1c01592c:	bdc9                	j	1c0157fe <pi_fpv2_vtable_load+0x4a>

1c01592e <pi_fpv2_vtable_volume_get_by_label>:
1c01592e:	1101                	addi	sp,sp,-32
1c015930:	ca26                	sw	s1,20(sp)
1c015932:	ce06                	sw	ra,28(sp)
1c015934:	cc22                	sw	s0,24(sp)
1c015936:	c84a                	sw	s2,16(sp)
1c015938:	c64e                	sw	s3,12(sp)
1c01593a:	c452                	sw	s4,8(sp)
1c01593c:	c256                	sw	s5,4(sp)
1c01593e:	c05a                	sw	s6,0(sp)
1c015940:	4489                	li	s1,2
1c015942:	c115                	beqz	a0,1c015966 <pi_fpv2_vtable_volume_get_by_label+0x38>
1c015944:	c18d                	beqz	a1,1c015966 <pi_fpv2_vtable_volume_get_by_label+0x38>
1c015946:	c205                	beqz	a2,1c015966 <pi_fpv2_vtable_volume_get_by_label+0x38>
1c015948:	00052903          	lw	s2,0(a0)
1c01594c:	4495                	li	s1,5
1c01594e:	00294b03          	lbu	s6,2(s2)
1c015952:	000b0a63          	beqz	s6,1c015966 <pi_fpv2_vtable_volume_get_by_label+0x38>
1c015956:	8a32                	mv	s4,a2
1c015958:	89ae                	mv	s3,a1
1c01595a:	093d                	addi	s2,s2,15
1c01595c:	00c50413          	addi	s0,a0,12
1c015960:	4a81                	li	s5,0
1c015962:	016aed63          	bltu	s5,s6,1c01597c <pi_fpv2_vtable_volume_get_by_label+0x4e>
1c015966:	40f2                	lw	ra,28(sp)
1c015968:	4462                	lw	s0,24(sp)
1c01596a:	8526                	mv	a0,s1
1c01596c:	4942                	lw	s2,16(sp)
1c01596e:	44d2                	lw	s1,20(sp)
1c015970:	49b2                	lw	s3,12(sp)
1c015972:	4a22                	lw	s4,8(sp)
1c015974:	4a92                	lw	s5,4(sp)
1c015976:	4b02                	lw	s6,0(sp)
1c015978:	6105                	addi	sp,sp,32
1c01597a:	8082                	ret
1c01597c:	4641                	li	a2,16
1c01597e:	00490593          	addi	a1,s2,4
1c015982:	854e                	mv	a0,s3
1c015984:	672040ef          	jal	ra,1c019ff6 <strncmp>
1c015988:	e511                	bnez	a0,1c015994 <pi_fpv2_vtable_volume_get_by_label+0x66>
1c01598a:	cc81                	beqz	s1,1c0159a2 <pi_fpv2_vtable_volume_get_by_label+0x74>
1c01598c:	401c                	lw	a5,0(s0)
1c01598e:	4481                	li	s1,0
1c015990:	00fa2023          	sw	a5,0(s4)
1c015994:	854a                	mv	a0,s2
1c015996:	989ff0ef          	jal	ra,1c01531e <__pi_fpv2_vtable_get_next_volume>
1c01599a:	892a                	mv	s2,a0
1c01599c:	0a85                	addi	s5,s5,1
1c01599e:	0411                	addi	s0,s0,4
1c0159a0:	b7c9                	j	1c015962 <pi_fpv2_vtable_volume_get_by_label+0x34>
1c0159a2:	4485                	li	s1,1
1c0159a4:	b7c9                	j	1c015966 <pi_fpv2_vtable_volume_get_by_label+0x38>

1c0159a6 <pi_fpv2_vtable_active_application_volume_get>:
1c0159a6:	411c                	lw	a5,0(a0)
1c0159a8:	1101                	addi	sp,sp,-32
1c0159aa:	ce06                	sw	ra,28(sp)
1c0159ac:	cc22                	sw	s0,24(sp)
1c0159ae:	ca26                	sw	s1,20(sp)
1c0159b0:	c84a                	sw	s2,16(sp)
1c0159b2:	c64e                	sw	s3,12(sp)
1c0159b4:	c452                	sw	s4,8(sp)
1c0159b6:	c256                	sw	s5,4(sp)
1c0159b8:	c05a                	sw	s6,0(sp)
1c0159ba:	0027ca83          	lbu	s5,2(a5)
1c0159be:	060a8663          	beqz	s5,1c015a2a <pi_fpv2_vtable_active_application_volume_get+0x84>
1c0159c2:	69c1                	lui	s3,0x10
1c0159c4:	842a                	mv	s0,a0
1c0159c6:	8a2e                	mv	s4,a1
1c0159c8:	00f78513          	addi	a0,a5,15
1c0159cc:	4901                	li	s2,0
1c0159ce:	19fd                	addi	s3,s3,-1
1c0159d0:	54fd                	li	s1,-1
1c0159d2:	0ff00b13          	li	s6,255
1c0159d6:	03596663          	bltu	s2,s5,1c015a02 <pi_fpv2_vtable_active_application_volume_get+0x5c>
1c0159da:	4515                	li	a0,5
1c0159dc:	01f4a963          	p.beqimm	s1,-1,1c0159ee <pi_fpv2_vtable_active_application_volume_get+0x48>
1c0159e0:	00249513          	slli	a0,s1,0x2
1c0159e4:	9522                	add	a0,a0,s0
1c0159e6:	455c                	lw	a5,12(a0)
1c0159e8:	4501                	li	a0,0
1c0159ea:	00fa2023          	sw	a5,0(s4)
1c0159ee:	40f2                	lw	ra,28(sp)
1c0159f0:	4462                	lw	s0,24(sp)
1c0159f2:	44d2                	lw	s1,20(sp)
1c0159f4:	4942                	lw	s2,16(sp)
1c0159f6:	49b2                	lw	s3,12(sp)
1c0159f8:	4a22                	lw	s4,8(sp)
1c0159fa:	4a92                	lw	s5,4(sp)
1c0159fc:	4b02                	lw	s6,0(sp)
1c0159fe:	6105                	addi	sp,sp,32
1c015a00:	8082                	ret
1c015a02:	00055783          	lhu	a5,0(a0)
1c015a06:	fc17b7b3          	p.bclr	a5,a5,30,1
1c015a0a:	cf81                	beqz	a5,1c015a22 <pi_fpv2_vtable_active_application_volume_get+0x7c>
1c015a0c:	01754783          	lbu	a5,23(a0)
1c015a10:	01678963          	beq	a5,s6,1c015a22 <pi_fpv2_vtable_active_application_volume_get+0x7c>
1c015a14:	01654783          	lbu	a5,22(a0)
1c015a18:	0137f563          	bleu	s3,a5,1c015a22 <pi_fpv2_vtable_active_application_volume_get+0x7c>
1c015a1c:	100944b3          	p.exths	s1,s2
1c015a20:	89be                	mv	s3,a5
1c015a22:	8fdff0ef          	jal	ra,1c01531e <__pi_fpv2_vtable_get_next_volume>
1c015a26:	0905                	addi	s2,s2,1
1c015a28:	b77d                	j	1c0159d6 <pi_fpv2_vtable_active_application_volume_get+0x30>
1c015a2a:	4515                	li	a0,5
1c015a2c:	b7c9                	j	1c0159ee <pi_fpv2_vtable_active_application_volume_get+0x48>

1c015a2e <pi_fpv2_volume_partition_get_by_label>:
1c015a2e:	c525                	beqz	a0,1c015a96 <pi_fpv2_volume_partition_get_by_label+0x68>
1c015a30:	7179                	addi	sp,sp,-48
1c015a32:	d226                	sw	s1,36(sp)
1c015a34:	d606                	sw	ra,44(sp)
1c015a36:	d422                	sw	s0,40(sp)
1c015a38:	d04a                	sw	s2,32(sp)
1c015a3a:	ce4e                	sw	s3,28(sp)
1c015a3c:	cc52                	sw	s4,24(sp)
1c015a3e:	84aa                	mv	s1,a0
1c015a40:	4509                	li	a0,2
1c015a42:	cd9d                	beqz	a1,1c015a80 <pi_fpv2_volume_partition_get_by_label+0x52>
1c015a44:	ce15                	beqz	a2,1c015a80 <pi_fpv2_volume_partition_get_by_label+0x52>
1c015a46:	4080                	lw	s0,0(s1)
1c015a48:	01444a03          	lbu	s4,20(s0)
1c015a4c:	020a0963          	beqz	s4,1c015a7e <pi_fpv2_volume_partition_get_by_label+0x50>
1c015a50:	86b2                	mv	a3,a2
1c015a52:	892e                	mv	s2,a1
1c015a54:	0461                	addi	s0,s0,24
1c015a56:	4981                	li	s3,0
1c015a58:	0349f363          	bleu	s4,s3,1c015a7e <pi_fpv2_volume_partition_get_by_label+0x50>
1c015a5c:	4641                	li	a2,16
1c015a5e:	00440593          	addi	a1,s0,4
1c015a62:	854a                	mv	a0,s2
1c015a64:	c636                	sw	a3,12(sp)
1c015a66:	590040ef          	jal	ra,1c019ff6 <strncmp>
1c015a6a:	46b2                	lw	a3,12(sp)
1c015a6c:	e115                	bnez	a0,1c015a90 <pi_fpv2_volume_partition_get_by_label+0x62>
1c015a6e:	40dc                	lw	a5,4(s1)
1c015a70:	00245583          	lhu	a1,2(s0)
1c015a74:	4601                	li	a2,0
1c015a76:	43c8                	lw	a0,4(a5)
1c015a78:	86fff0ef          	jal	ra,1c0152e6 <__pi_fpv2_ptable_get_partition_by_uuid>
1c015a7c:	c111                	beqz	a0,1c015a80 <pi_fpv2_volume_partition_get_by_label+0x52>
1c015a7e:	4515                	li	a0,5
1c015a80:	50b2                	lw	ra,44(sp)
1c015a82:	5422                	lw	s0,40(sp)
1c015a84:	5492                	lw	s1,36(sp)
1c015a86:	5902                	lw	s2,32(sp)
1c015a88:	49f2                	lw	s3,28(sp)
1c015a8a:	4a62                	lw	s4,24(sp)
1c015a8c:	6145                	addi	sp,sp,48
1c015a8e:	8082                	ret
1c015a90:	0451                	addi	s0,s0,20
1c015a92:	0985                	addi	s3,s3,1
1c015a94:	b7d1                	j	1c015a58 <pi_fpv2_volume_partition_get_by_label+0x2a>
1c015a96:	4509                	li	a0,2
1c015a98:	8082                	ret

1c015a9a <pi_fpv2_volume_partition_first_get_by_type>:
1c015a9a:	c159                	beqz	a0,1c015b20 <pi_fpv2_volume_partition_first_get_by_type+0x86>
1c015a9c:	7179                	addi	sp,sp,-48
1c015a9e:	d422                	sw	s0,40(sp)
1c015aa0:	d606                	sw	ra,44(sp)
1c015aa2:	d226                	sw	s1,36(sp)
1c015aa4:	d04a                	sw	s2,32(sp)
1c015aa6:	ce4e                	sw	s3,28(sp)
1c015aa8:	cc52                	sw	s4,24(sp)
1c015aaa:	ca56                	sw	s5,20(sp)
1c015aac:	c85a                	sw	s6,16(sp)
1c015aae:	c65e                	sw	s7,12(sp)
1c015ab0:	842a                	mv	s0,a0
1c015ab2:	4509                	li	a0,2
1c015ab4:	c699                	beqz	a3,1c015ac2 <pi_fpv2_volume_partition_first_get_by_type+0x28>
1c015ab6:	00042a03          	lw	s4,0(s0)
1c015aba:	014a4703          	lbu	a4,20(s4)
1c015abe:	ef09                	bnez	a4,1c015ad8 <pi_fpv2_volume_partition_first_get_by_type+0x3e>
1c015ac0:	4515                	li	a0,5
1c015ac2:	50b2                	lw	ra,44(sp)
1c015ac4:	5422                	lw	s0,40(sp)
1c015ac6:	5492                	lw	s1,36(sp)
1c015ac8:	5902                	lw	s2,32(sp)
1c015aca:	49f2                	lw	s3,28(sp)
1c015acc:	4a62                	lw	s4,24(sp)
1c015ace:	4ad2                	lw	s5,20(sp)
1c015ad0:	4b42                	lw	s6,16(sp)
1c015ad2:	4bb2                	lw	s7,12(sp)
1c015ad4:	6145                	addi	sp,sp,48
1c015ad6:	8082                	ret
1c015ad8:	84b6                	mv	s1,a3
1c015ada:	8932                	mv	s2,a2
1c015adc:	8b2e                	mv	s6,a1
1c015ade:	01aa0a93          	addi	s5,s4,26
1c015ae2:	4981                	li	s3,0
1c015ae4:	0fe00b93          	li	s7,254
1c015ae8:	014a4783          	lbu	a5,20(s4)
1c015aec:	fcf9fae3          	bleu	a5,s3,1c015ac0 <pi_fpv2_volume_partition_first_get_by_type+0x26>
1c015af0:	405c                	lw	a5,4(s0)
1c015af2:	000ad583          	lhu	a1,0(s5)
1c015af6:	86a6                	mv	a3,s1
1c015af8:	43c8                	lw	a0,4(a5)
1c015afa:	4601                	li	a2,0
1c015afc:	feaff0ef          	jal	ra,1c0152e6 <__pi_fpv2_ptable_get_partition_by_uuid>
1c015b00:	f161                	bnez	a0,1c015ac0 <pi_fpv2_volume_partition_first_get_by_type+0x26>
1c015b02:	409c                	lw	a5,0(s1)
1c015b04:	43dc                	lw	a5,4(a5)
1c015b06:	0027c703          	lbu	a4,2(a5)
1c015b0a:	01671863          	bne	a4,s6,1c015b1a <pi_fpv2_volume_partition_first_get_by_type+0x80>
1c015b0e:	0037c783          	lbu	a5,3(a5)
1c015b12:	fb2788e3          	beq	a5,s2,1c015ac2 <pi_fpv2_volume_partition_first_get_by_type+0x28>
1c015b16:	fb7906e3          	beq	s2,s7,1c015ac2 <pi_fpv2_volume_partition_first_get_by_type+0x28>
1c015b1a:	0985                	addi	s3,s3,1
1c015b1c:	0ad1                	addi	s5,s5,20
1c015b1e:	b7e9                	j	1c015ae8 <pi_fpv2_volume_partition_first_get_by_type+0x4e>
1c015b20:	4509                	li	a0,2
1c015b22:	8082                	ret

1c015b24 <__pi_crc32_compute_with_init>:
1c015b24:	c955                	beqz	a0,1c015bd8 <__pi_crc32_compute_with_init+0xb4>
1c015b26:	770736b7          	lui	a3,0x77073
1c015b2a:	ee0e6837          	lui	a6,0xee0e6
1c015b2e:	076dc8b7          	lui	a7,0x76dc
1c015b32:	0edb9337          	lui	t1,0xedb9
1c015b36:	1db71e37          	lui	t3,0x1db71
1c015b3a:	3b6e2eb7          	lui	t4,0x3b6e2
1c015b3e:	76dc4f37          	lui	t5,0x76dc4
1c015b42:	edb88fb7          	lui	t6,0xedb88
1c015b46:	95aa                	add	a1,a1,a0
1c015b48:	09668693          	addi	a3,a3,150 # 77073096 <__heapl2ram_size+0x5aee3096>
1c015b4c:	12c80813          	addi	a6,a6,300 # ee0e612c <pulp__FC+0xee0e612d>
1c015b50:	41988893          	addi	a7,a7,1049 # 76dc419 <__l1_heapsram_size+0x76bc489>
1c015b54:	83230313          	addi	t1,t1,-1998 # edb8832 <__l1_heapsram_size+0xed988a2>
1c015b58:	064e0e13          	addi	t3,t3,100 # 1db71064 <__heapl2ram_size+0x19e1064>
1c015b5c:	0c8e8e93          	addi	t4,t4,200 # 3b6e20c8 <__heapl2ram_size+0x1f5520c8>
1c015b60:	190f0f13          	addi	t5,t5,400 # 76dc4190 <__heapl2ram_size+0x5ac34190>
1c015b64:	320f8f93          	addi	t6,t6,800 # edb88320 <pulp__FC+0xedb88321>
1c015b68:	00b51563          	bne	a0,a1,1c015b72 <__pi_crc32_compute_with_init+0x4e>
1c015b6c:	fff64513          	not	a0,a2
1c015b70:	8082                	ret
1c015b72:	0015478b          	p.lbu	a5,1(a0!)
1c015b76:	8e3d                	xor	a2,a2,a5
1c015b78:	01e61713          	slli	a4,a2,0x1e
1c015b7c:	c00607b3          	p.extract	a5,a2,0,0
1c015b80:	877d                	srai	a4,a4,0x1f
1c015b82:	01077733          	and	a4,a4,a6
1c015b86:	8ff5                	and	a5,a5,a3
1c015b88:	8fb9                	xor	a5,a5,a4
1c015b8a:	00865713          	srli	a4,a2,0x8
1c015b8e:	8fb9                	xor	a5,a5,a4
1c015b90:	01d61713          	slli	a4,a2,0x1d
1c015b94:	877d                	srai	a4,a4,0x1f
1c015b96:	01177733          	and	a4,a4,a7
1c015b9a:	8fb9                	xor	a5,a5,a4
1c015b9c:	01c61713          	slli	a4,a2,0x1c
1c015ba0:	877d                	srai	a4,a4,0x1f
1c015ba2:	00677733          	and	a4,a4,t1
1c015ba6:	8fb9                	xor	a5,a5,a4
1c015ba8:	01b61713          	slli	a4,a2,0x1b
1c015bac:	877d                	srai	a4,a4,0x1f
1c015bae:	01c77733          	and	a4,a4,t3
1c015bb2:	8fb9                	xor	a5,a5,a4
1c015bb4:	01a61713          	slli	a4,a2,0x1a
1c015bb8:	877d                	srai	a4,a4,0x1f
1c015bba:	01d77733          	and	a4,a4,t4
1c015bbe:	8fb9                	xor	a5,a5,a4
1c015bc0:	01961713          	slli	a4,a2,0x19
1c015bc4:	877d                	srai	a4,a4,0x1f
1c015bc6:	0662                	slli	a2,a2,0x18
1c015bc8:	01e77733          	and	a4,a4,t5
1c015bcc:	867d                	srai	a2,a2,0x1f
1c015bce:	8fb9                	xor	a5,a5,a4
1c015bd0:	01f67633          	and	a2,a2,t6
1c015bd4:	8e3d                	xor	a2,a2,a5
1c015bd6:	bf49                	j	1c015b68 <__pi_crc32_compute_with_init+0x44>
1c015bd8:	4501                	li	a0,0
1c015bda:	8082                	ret

1c015bdc <pi_crc32_compute>:
1c015bdc:	567d                	li	a2,-1
1c015bde:	b799                	j	1c015b24 <__pi_crc32_compute_with_init>

1c015be0 <mram_reg_set>:
1c015be0:	4501                	li	a0,0
1c015be2:	8082                	ret

1c015be4 <mram_reg_set_async>:
1c015be4:	8082                	ret

1c015be6 <mram_copy_2d_async>:
1c015be6:	08080463          	beqz	a6,1c015c6e <mram_copy_2d_async+0x88>
1c015bea:	4508                	lw	a0,8(a0)
1c015bec:	30047373          	csrrci	t1,mstatus,8
1c015bf0:	00452e03          	lw	t3,4(a0)
1c015bf4:	040e1163          	bnez	t3,1c015c36 <mram_copy_2d_async+0x50>
1c015bf8:	01152223          	sw	a7,4(a0)
1c015bfc:	4108                	lw	a0,0(a0)
1c015bfe:	02052803          	lw	a6,32(a0)
1c015c02:	04000893          	li	a7,64
1c015c06:	ce88a833          	p.insert	a6,a7,7,8
1c015c0a:	03052023          	sw	a6,32(a0)
1c015c0e:	00c52623          	sw	a2,12(a0)
1c015c12:	00d52823          	sw	a3,16(a0)
1c015c16:	00b52c23          	sw	a1,24(a0)
1c015c1a:	02f52e23          	sw	a5,60(a0)
1c015c1e:	04e52023          	sw	a4,64(a0)
1c015c22:	4785                	li	a5,1
1c015c24:	04f52223          	sw	a5,68(a0)
1c015c28:	478d                	li	a5,3
1c015c2a:	00f52a23          	sw	a5,20(a0)
1c015c2e:	30031073          	csrw	mstatus,t1
1c015c32:	4501                	li	a0,0
1c015c34:	8082                	ret
1c015c36:	02f8a623          	sw	a5,44(a7)
1c015c3a:	451c                	lw	a5,8(a0)
1c015c3c:	4e15                	li	t3,5
1c015c3e:	01c8ac23          	sw	t3,24(a7)
1c015c42:	00b8ae23          	sw	a1,28(a7)
1c015c46:	02c8a023          	sw	a2,32(a7)
1c015c4a:	02d8a223          	sw	a3,36(a7)
1c015c4e:	02e8a423          	sw	a4,40(a7)
1c015c52:	0308a823          	sw	a6,48(a7)
1c015c56:	cb89                	beqz	a5,1c015c68 <mram_copy_2d_async+0x82>
1c015c58:	455c                	lw	a5,12(a0)
1c015c5a:	0117a023          	sw	a7,0(a5)
1c015c5e:	01152623          	sw	a7,12(a0)
1c015c62:	0008a023          	sw	zero,0(a7)
1c015c66:	b7e1                	j	1c015c2e <mram_copy_2d_async+0x48>
1c015c68:	01152423          	sw	a7,8(a0)
1c015c6c:	bfcd                	j	1c015c5e <mram_copy_2d_async+0x78>
1c015c6e:	557d                	li	a0,-1
1c015c70:	8082                	ret

1c015c72 <mram_copy_2d>:
1c015c72:	7175                	addi	sp,sp,-144
1c015c74:	c522                	sw	s0,136(sp)
1c015c76:	842a                	mv	s0,a0
1c015c78:	1028                	addi	a0,sp,40
1c015c7a:	c706                	sw	ra,140(sp)
1c015c7c:	ce2e                	sw	a1,28(sp)
1c015c7e:	cc32                	sw	a2,24(sp)
1c015c80:	ca36                	sw	a3,20(sp)
1c015c82:	c83a                	sw	a4,16(sp)
1c015c84:	c63e                	sw	a5,12(sp)
1c015c86:	c442                	sw	a6,8(sp)
1c015c88:	252040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015c8c:	4822                	lw	a6,8(sp)
1c015c8e:	47b2                	lw	a5,12(sp)
1c015c90:	4742                	lw	a4,16(sp)
1c015c92:	46d2                	lw	a3,20(sp)
1c015c94:	4662                	lw	a2,24(sp)
1c015c96:	45f2                	lw	a1,28(sp)
1c015c98:	88aa                	mv	a7,a0
1c015c9a:	8522                	mv	a0,s0
1c015c9c:	37a9                	jal	1c015be6 <mram_copy_2d_async>
1c015c9e:	1028                	addi	a0,sp,40
1c015ca0:	27c040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015ca4:	40ba                	lw	ra,140(sp)
1c015ca6:	442a                	lw	s0,136(sp)
1c015ca8:	4501                	li	a0,0
1c015caa:	6149                	addi	sp,sp,144
1c015cac:	8082                	ret

1c015cae <mram_erase>:
1c015cae:	7119                	addi	sp,sp,-128
1c015cb0:	dca2                	sw	s0,120(sp)
1c015cb2:	842a                	mv	s0,a0
1c015cb4:	0828                	addi	a0,sp,24
1c015cb6:	de86                	sw	ra,124(sp)
1c015cb8:	c62e                	sw	a1,12(sp)
1c015cba:	c432                	sw	a2,8(sp)
1c015cbc:	21e040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015cc0:	4622                	lw	a2,8(sp)
1c015cc2:	45b2                	lw	a1,12(sp)
1c015cc4:	86aa                	mv	a3,a0
1c015cc6:	8522                	mv	a0,s0
1c015cc8:	b0bfa0ef          	jal	ra,1c0107d2 <mram_erase_async>
1c015ccc:	0828                	addi	a0,sp,24
1c015cce:	24e040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015cd2:	50f6                	lw	ra,124(sp)
1c015cd4:	5466                	lw	s0,120(sp)
1c015cd6:	4501                	li	a0,0
1c015cd8:	6109                	addi	sp,sp,128
1c015cda:	8082                	ret

1c015cdc <mram_erase_sector>:
1c015cdc:	7119                	addi	sp,sp,-128
1c015cde:	dca2                	sw	s0,120(sp)
1c015ce0:	842a                	mv	s0,a0
1c015ce2:	0828                	addi	a0,sp,24
1c015ce4:	de86                	sw	ra,124(sp)
1c015ce6:	c62e                	sw	a1,12(sp)
1c015ce8:	1f2040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015cec:	45b2                	lw	a1,12(sp)
1c015cee:	862a                	mv	a2,a0
1c015cf0:	8522                	mv	a0,s0
1c015cf2:	b27fa0ef          	jal	ra,1c010818 <mram_erase_sector_async>
1c015cf6:	0828                	addi	a0,sp,24
1c015cf8:	224040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015cfc:	50f6                	lw	ra,124(sp)
1c015cfe:	5466                	lw	s0,120(sp)
1c015d00:	4501                	li	a0,0
1c015d02:	6109                	addi	sp,sp,128
1c015d04:	8082                	ret

1c015d06 <mram_erase_chip>:
1c015d06:	7159                	addi	sp,sp,-112
1c015d08:	d4a2                	sw	s0,104(sp)
1c015d0a:	842a                	mv	s0,a0
1c015d0c:	0028                	addi	a0,sp,8
1c015d0e:	d686                	sw	ra,108(sp)
1c015d10:	1ca040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015d14:	85aa                	mv	a1,a0
1c015d16:	8522                	mv	a0,s0
1c015d18:	b59fa0ef          	jal	ra,1c010870 <mram_erase_chip_async>
1c015d1c:	0028                	addi	a0,sp,8
1c015d1e:	1fe040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015d22:	50b6                	lw	ra,108(sp)
1c015d24:	5426                	lw	s0,104(sp)
1c015d26:	4501                	li	a0,0
1c015d28:	6165                	addi	sp,sp,112
1c015d2a:	8082                	ret

1c015d2c <mram_program_async>:
1c015d2c:	1141                	addi	sp,sp,-16
1c015d2e:	c606                	sw	ra,12(sp)
1c015d30:	c422                	sw	s0,8(sp)
1c015d32:	4508                	lw	a0,8(a0)
1c015d34:	30047473          	csrrci	s0,mstatus,8
1c015d38:	415c                	lw	a5,4(a0)
1c015d3a:	ef99                	bnez	a5,1c015d58 <mram_program_async+0x2c>
1c015d3c:	c158                	sw	a4,4(a0)
1c015d3e:	cd0c                	sw	a1,24(a0)
1c015d40:	c950                	sw	a2,20(a0)
1c015d42:	cd54                	sw	a3,28(a0)
1c015d44:	06050c23          	sb	zero,120(a0)
1c015d48:	9c7fa0ef          	jal	ra,1c01070e <mram_program_resume>
1c015d4c:	30041073          	csrw	mstatus,s0
1c015d50:	40b2                	lw	ra,12(sp)
1c015d52:	4422                	lw	s0,8(sp)
1c015d54:	0141                	addi	sp,sp,16
1c015d56:	8082                	ret
1c015d58:	478d                	li	a5,3
1c015d5a:	cf1c                	sw	a5,24(a4)
1c015d5c:	451c                	lw	a5,8(a0)
1c015d5e:	cf4c                	sw	a1,28(a4)
1c015d60:	d310                	sw	a2,32(a4)
1c015d62:	d354                	sw	a3,36(a4)
1c015d64:	c799                	beqz	a5,1c015d72 <mram_program_async+0x46>
1c015d66:	455c                	lw	a5,12(a0)
1c015d68:	c398                	sw	a4,0(a5)
1c015d6a:	c558                	sw	a4,12(a0)
1c015d6c:	00072023          	sw	zero,0(a4)
1c015d70:	bff1                	j	1c015d4c <mram_program_async+0x20>
1c015d72:	c518                	sw	a4,8(a0)
1c015d74:	bfdd                	j	1c015d6a <mram_program_async+0x3e>

1c015d76 <mram_program>:
1c015d76:	7119                	addi	sp,sp,-128
1c015d78:	dca2                	sw	s0,120(sp)
1c015d7a:	842a                	mv	s0,a0
1c015d7c:	0828                	addi	a0,sp,24
1c015d7e:	de86                	sw	ra,124(sp)
1c015d80:	c62e                	sw	a1,12(sp)
1c015d82:	c432                	sw	a2,8(sp)
1c015d84:	c236                	sw	a3,4(sp)
1c015d86:	154040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015d8a:	4692                	lw	a3,4(sp)
1c015d8c:	4622                	lw	a2,8(sp)
1c015d8e:	45b2                	lw	a1,12(sp)
1c015d90:	872a                	mv	a4,a0
1c015d92:	8522                	mv	a0,s0
1c015d94:	3f61                	jal	1c015d2c <mram_program_async>
1c015d96:	0828                	addi	a0,sp,24
1c015d98:	184040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015d9c:	50f6                	lw	ra,124(sp)
1c015d9e:	5466                	lw	s0,120(sp)
1c015da0:	4501                	li	a0,0
1c015da2:	6109                	addi	sp,sp,128
1c015da4:	8082                	ret

1c015da6 <mram_read_async>:
1c015da6:	451c                	lw	a5,8(a0)
1c015da8:	30047573          	csrrci	a0,mstatus,8
1c015dac:	0047a803          	lw	a6,4(a5)
1c015db0:	02081a63          	bnez	a6,1c015de4 <mram_read_async+0x3e>
1c015db4:	c3d8                	sw	a4,4(a5)
1c015db6:	439c                	lw	a5,0(a5)
1c015db8:	0207a703          	lw	a4,32(a5)
1c015dbc:	04000813          	li	a6,64
1c015dc0:	ce882733          	p.insert	a4,a6,7,8
1c015dc4:	02e7a023          	sw	a4,32(a5)
1c015dc8:	00c7a623          	sw	a2,12(a5)
1c015dcc:	00d7a823          	sw	a3,16(a5)
1c015dd0:	00b7ac23          	sw	a1,24(a5)
1c015dd4:	0407a223          	sw	zero,68(a5)
1c015dd8:	470d                	li	a4,3
1c015dda:	00e7aa23          	sw	a4,20(a5)
1c015dde:	30051073          	csrw	mstatus,a0
1c015de2:	8082                	ret
1c015de4:	d354                	sw	a3,36(a4)
1c015de6:	4794                	lw	a3,8(a5)
1c015de8:	4811                	li	a6,4
1c015dea:	01072c23          	sw	a6,24(a4)
1c015dee:	cf4c                	sw	a1,28(a4)
1c015df0:	d310                	sw	a2,32(a4)
1c015df2:	c699                	beqz	a3,1c015e00 <mram_read_async+0x5a>
1c015df4:	47d4                	lw	a3,12(a5)
1c015df6:	c298                	sw	a4,0(a3)
1c015df8:	c7d8                	sw	a4,12(a5)
1c015dfa:	00072023          	sw	zero,0(a4)
1c015dfe:	b7c5                	j	1c015dde <mram_read_async+0x38>
1c015e00:	c798                	sw	a4,8(a5)
1c015e02:	bfdd                	j	1c015df8 <mram_read_async+0x52>

1c015e04 <mram_read>:
1c015e04:	7119                	addi	sp,sp,-128
1c015e06:	dca2                	sw	s0,120(sp)
1c015e08:	842a                	mv	s0,a0
1c015e0a:	0828                	addi	a0,sp,24
1c015e0c:	de86                	sw	ra,124(sp)
1c015e0e:	c62e                	sw	a1,12(sp)
1c015e10:	c432                	sw	a2,8(sp)
1c015e12:	c236                	sw	a3,4(sp)
1c015e14:	0c6040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015e18:	4692                	lw	a3,4(sp)
1c015e1a:	4622                	lw	a2,8(sp)
1c015e1c:	45b2                	lw	a1,12(sp)
1c015e1e:	872a                	mv	a4,a0
1c015e20:	8522                	mv	a0,s0
1c015e22:	3751                	jal	1c015da6 <mram_read_async>
1c015e24:	0828                	addi	a0,sp,24
1c015e26:	0f6040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015e2a:	50f6                	lw	ra,124(sp)
1c015e2c:	5466                	lw	s0,120(sp)
1c015e2e:	4501                	li	a0,0
1c015e30:	6109                	addi	sp,sp,128
1c015e32:	8082                	ret

1c015e34 <mram_copy_async>:
1c015e34:	1141                	addi	sp,sp,-16
1c015e36:	c606                	sw	ra,12(sp)
1c015e38:	e719                	bnez	a4,1c015e46 <mram_copy_async+0x12>
1c015e3a:	873e                	mv	a4,a5
1c015e3c:	3dc5                	jal	1c015d2c <mram_program_async>
1c015e3e:	40b2                	lw	ra,12(sp)
1c015e40:	4501                	li	a0,0
1c015e42:	0141                	addi	sp,sp,16
1c015e44:	8082                	ret
1c015e46:	873e                	mv	a4,a5
1c015e48:	3fb9                	jal	1c015da6 <mram_read_async>
1c015e4a:	bfd5                	j	1c015e3e <mram_copy_async+0xa>

1c015e4c <mram_copy>:
1c015e4c:	7119                	addi	sp,sp,-128
1c015e4e:	dca2                	sw	s0,120(sp)
1c015e50:	842a                	mv	s0,a0
1c015e52:	0828                	addi	a0,sp,24
1c015e54:	de86                	sw	ra,124(sp)
1c015e56:	c62e                	sw	a1,12(sp)
1c015e58:	c432                	sw	a2,8(sp)
1c015e5a:	c236                	sw	a3,4(sp)
1c015e5c:	c03a                	sw	a4,0(sp)
1c015e5e:	07c040ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c015e62:	4702                	lw	a4,0(sp)
1c015e64:	4692                	lw	a3,4(sp)
1c015e66:	4622                	lw	a2,8(sp)
1c015e68:	45b2                	lw	a1,12(sp)
1c015e6a:	87aa                	mv	a5,a0
1c015e6c:	8522                	mv	a0,s0
1c015e6e:	37d9                	jal	1c015e34 <mram_copy_async>
1c015e70:	0828                	addi	a0,sp,24
1c015e72:	0aa040ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c015e76:	50f6                	lw	ra,124(sp)
1c015e78:	5466                	lw	s0,120(sp)
1c015e7a:	4501                	li	a0,0
1c015e7c:	6109                	addi	sp,sp,128
1c015e7e:	8082                	ret

1c015e80 <pi_udma_core_lin_alloc>:
1c015e80:	30047673          	csrrci	a2,mstatus,8
1c015e84:	4d802703          	lw	a4,1240(zero) # 4d8 <__pi_udma_chan_lin>
1c015e88:	4d800693          	li	a3,1240
1c015e8c:	e719                	bnez	a4,1c015e9a <pi_udma_core_lin_alloc+0x1a>
1c015e8e:	42d8                	lw	a4,4(a3)
1c015e90:	e31d                	bnez	a4,1c015eb6 <pi_udma_core_lin_alloc+0x36>
1c015e92:	30061073          	csrw	mstatus,a2
1c015e96:	557d                	li	a0,-1
1c015e98:	8082                	ret
1c015e9a:	4781                	li	a5,0
1c015e9c:	10070533          	p.ff1	a0,a4
1c015ea0:	80a73733          	p.bclrr	a4,a4,a0
1c015ea4:	00279593          	slli	a1,a5,0x2
1c015ea8:	00e6e5a3          	p.sw	a4,a1(a3)
1c015eac:	30061073          	csrw	mstatus,a2
1c015eb0:	0796                	slli	a5,a5,0x5
1c015eb2:	953e                	add	a0,a0,a5
1c015eb4:	8082                	ret
1c015eb6:	4785                	li	a5,1
1c015eb8:	b7d5                	j	1c015e9c <pi_udma_core_lin_alloc+0x1c>

1c015eba <mram_reg_get_async>:
1c015eba:	8082                	ret

1c015ebc <mram_reg_get>:
1c015ebc:	4501                	li	a0,0
1c015ebe:	8082                	ret

1c015ec0 <__rt_mram_trim_cfg_exec.isra.0.constprop.8>:
1c015ec0:	411c                	lw	a5,0(a0)
1c015ec2:	0207a703          	lw	a4,32(a5)
1c015ec6:	4685                	li	a3,1
1c015ec8:	ce86a733          	p.insert	a4,a3,7,8
1c015ecc:	02e7a023          	sw	a4,32(a5)
1c015ed0:	00b7a623          	sw	a1,12(a5)
1c015ed4:	4771                	li	a4,28
1c015ed6:	00e7a823          	sw	a4,16(a5)
1c015eda:	0407a223          	sw	zero,68(a5)
1c015ede:	4709                	li	a4,2
1c015ee0:	00e7aa23          	sw	a4,20(a5)
1c015ee4:	8082                	ret

1c015ee6 <__pi_mram_close.part.3>:
1c015ee6:	07954683          	lbu	a3,121(a0)
1c015eea:	300475f3          	csrrci	a1,mstatus,8
1c015eee:	4056d713          	srai	a4,a3,0x5
1c015ef2:	4d800613          	li	a2,1240
1c015ef6:	070a                	slli	a4,a4,0x2
1c015ef8:	9732                	add	a4,a4,a2
1c015efa:	431c                	lw	a5,0(a4)
1c015efc:	f456b6b3          	p.bclr	a3,a3,26,5
1c015f00:	80d7c6b3          	p.bsetr	a3,a5,a3
1c015f04:	c314                	sw	a3,0(a4)
1c015f06:	30059073          	csrw	mstatus,a1
1c015f0a:	07954703          	lbu	a4,121(a0)
1c015f0e:	1a1035b7          	lui	a1,0x1a103
1c015f12:	47c1                	li	a5,16
1c015f14:	0716                	slli	a4,a4,0x5
1c015f16:	972e                	add	a4,a4,a1
1c015f18:	00f72e23          	sw	a5,28(a4)
1c015f1c:	07a54703          	lbu	a4,122(a0)
1c015f20:	300476f3          	csrrci	a3,mstatus,8
1c015f24:	40575793          	srai	a5,a4,0x5
1c015f28:	078a                	slli	a5,a5,0x2
1c015f2a:	97b2                	add	a5,a5,a2
1c015f2c:	4390                	lw	a2,0(a5)
1c015f2e:	f4573733          	p.bclr	a4,a4,26,5
1c015f32:	80e64733          	p.bsetr	a4,a2,a4
1c015f36:	c398                	sw	a4,0(a5)
1c015f38:	30069073          	csrw	mstatus,a3
1c015f3c:	07a54783          	lbu	a5,122(a0)
1c015f40:	4741                	li	a4,16
1c015f42:	0796                	slli	a5,a5,0x5
1c015f44:	97ae                	add	a5,a5,a1
1c015f46:	00e7ae23          	sw	a4,28(a5)
1c015f4a:	07c54703          	lbu	a4,124(a0)
1c015f4e:	4785                	li	a5,1
1c015f50:	00e797b3          	sll	a5,a5,a4
1c015f54:	1a102737          	lui	a4,0x1a102
1c015f58:	00f72423          	sw	a5,8(a4) # 1a102008 <__l1_heapsram_end+0xa0e2008>
1c015f5c:	00f72a23          	sw	a5,20(a4)
1c015f60:	4601                	li	a2,0
1c015f62:	4581                	li	a1,0
1c015f64:	4505                	li	a0,1
1c015f66:	cf7fe06f          	j	1c014c5c <pi_pmu_domain_state_change>

1c015f6a <mram_close>:
1c015f6a:	4508                	lw	a0,8(a0)
1c015f6c:	1c01d737          	lui	a4,0x1c01d
1c015f70:	79870713          	addi	a4,a4,1944 # 1c01d798 <pos_mram_open_count>
1c015f74:	07b54683          	lbu	a3,123(a0)
1c015f78:	40d77783          	p.lbu	a5,a3(a4)
1c015f7c:	17fd                	addi	a5,a5,-1
1c015f7e:	0ff7f793          	andi	a5,a5,255
1c015f82:	00f746a3          	p.sb	a5,a3(a4)
1c015f86:	eb81                	bnez	a5,1c015f96 <mram_close+0x2c>
1c015f88:	1141                	addi	sp,sp,-16
1c015f8a:	c606                	sw	ra,12(sp)
1c015f8c:	3fa9                	jal	1c015ee6 <__pi_mram_close.part.3>
1c015f8e:	40b2                	lw	ra,12(sp)
1c015f90:	4501                	li	a0,0
1c015f92:	0141                	addi	sp,sp,16
1c015f94:	8082                	ret
1c015f96:	4501                	li	a0,0
1c015f98:	8082                	ret

1c015f9a <mram_ioctl>:
1c015f9a:	1101                	addi	sp,sp,-32
1c015f9c:	ce06                	sw	ra,28(sp)
1c015f9e:	cc22                	sw	s0,24(sp)
1c015fa0:	ca26                	sw	s1,20(sp)
1c015fa2:	872a                	mv	a4,a0
1c015fa4:	0215a363          	p.beqimm	a1,1,1c015fca <mram_ioctl+0x30>
1c015fa8:	c989                	beqz	a1,1c015fba <mram_ioctl+0x20>
1c015faa:	0445ac63          	p.beqimm	a1,4,1c016002 <mram_ioctl+0x68>
1c015fae:	40f2                	lw	ra,28(sp)
1c015fb0:	4462                	lw	s0,24(sp)
1c015fb2:	44d2                	lw	s1,20(sp)
1c015fb4:	4501                	li	a0,0
1c015fb6:	6105                	addi	sp,sp,32
1c015fb8:	8082                	ret
1c015fba:	6789                	lui	a5,0x2
1c015fbc:	c21c                	sw	a5,0(a2)
1c015fbe:	67c1                	lui	a5,0x10
1c015fc0:	c25c                	sw	a5,4(a2)
1c015fc2:	002007b7          	lui	a5,0x200
1c015fc6:	c61c                	sw	a5,8(a2)
1c015fc8:	b7dd                	j	1c015fae <mram_ioctl+0x14>
1c015fca:	4500                	lw	s0,8(a0)
1c015fcc:	4501                	li	a0,0
1c015fce:	4004                	lw	s1,0(s0)
1c015fd0:	c632                	sw	a2,12(sp)
1c015fd2:	d4dfa0ef          	jal	ra,1c010d1e <pi_freq_get>
1c015fd6:	4632                	lw	a2,12(sp)
1c015fd8:	c219                	beqz	a2,1c015fde <mram_ioctl+0x44>
1c015fda:	00a64b63          	blt	a2,a0,1c015ff0 <mram_ioctl+0x56>
1c015fde:	c808                	sw	a0,16(s0)
1c015fe0:	4781                	li	a5,0
1c015fe2:	c287c7b3          	p.bset	a5,a5,1,8
1c015fe6:	02f4a623          	sw	a5,44(s1)
1c015fea:	08c42023          	sw	a2,128(s0)
1c015fee:	b7c1                	j	1c015fae <mram_ioctl+0x14>
1c015ff0:	00a607b3          	add	a5,a2,a0
1c015ff4:	17fd                	addi	a5,a5,-1
1c015ff6:	02c7c7b3          	div	a5,a5,a2
1c015ffa:	02f54533          	div	a0,a0,a5
1c015ffe:	c808                	sw	a0,16(s0)
1c016000:	b7cd                	j	1c015fe2 <mram_ioctl+0x48>
1c016002:	1c01d437          	lui	s0,0x1c01d
1c016006:	00261793          	slli	a5,a2,0x2
1c01600a:	79440413          	addi	s0,s0,1940 # 1c01d794 <g_conf>
1c01600e:	943e                	add	s0,s0,a5
1c016010:	1c01d7b7          	lui	a5,0x1c01d
1c016014:	08800693          	li	a3,136
1c016018:	b5878793          	addi	a5,a5,-1192 # 1c01cb58 <g_mram_data>
1c01601c:	42d607b3          	p.mac	a5,a2,a3
1c016020:	4008                	lw	a0,0(s0)
1c016022:	03400593          	li	a1,52
1c016026:	c348                	sw	a0,4(a4)
1c016028:	c71c                	sw	a5,8(a4)
1c01602a:	41a040ef          	jal	ra,1c01a444 <pi_l2_free>
1c01602e:	00042023          	sw	zero,0(s0)
1c016032:	bfb5                	j	1c015fae <mram_ioctl+0x14>

1c016034 <pi_mram_trim_from_efuse>:
1c016034:	415c                	lw	a5,4(a0)
1c016036:	7159                	addi	sp,sp,-112
1c016038:	0028                	addi	a0,sp,8
1c01603a:	d686                	sw	ra,108(sp)
1c01603c:	d4a2                	sw	s0,104(sp)
1c01603e:	5780                	lw	s0,40(a5)
1c016040:	69b030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016044:	1c01d7b7          	lui	a5,0x1c01d
1c016048:	08800713          	li	a4,136
1c01604c:	b5878793          	addi	a5,a5,-1192 # 1c01cb58 <g_mram_data>
1c016050:	42e407b3          	p.mac	a5,s0,a4
1c016054:	1c01e5b7          	lui	a1,0x1c01e
1c016058:	b1058593          	addi	a1,a1,-1264 # 1c01db10 <pi_efuse_regs+0x24>
1c01605c:	c3c8                	sw	a0,4(a5)
1c01605e:	853e                	mv	a0,a5
1c016060:	3585                	jal	1c015ec0 <__rt_mram_trim_cfg_exec.isra.0.constprop.8>
1c016062:	0028                	addi	a0,sp,8
1c016064:	6b9030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016068:	4551                	li	a0,20
1c01606a:	5f8010ef          	jal	ra,1c017662 <pi_time_wait_us>
1c01606e:	50b6                	lw	ra,108(sp)
1c016070:	5426                	lw	s0,104(sp)
1c016072:	6165                	addi	sp,sp,112
1c016074:	8082                	ret

1c016076 <mram_open>:
1c016076:	7175                	addi	sp,sp,-144
1c016078:	d8da                	sw	s6,112(sp)
1c01607a:	00452b03          	lw	s6,4(a0)
1c01607e:	c14a                	sw	s2,128(sp)
1c016080:	dcd2                	sw	s4,120(sp)
1c016082:	028b2903          	lw	s2,40(s6)
1c016086:	1c01da37          	lui	s4,0x1c01d
1c01608a:	c522                	sw	s0,136(sp)
1c01608c:	08800793          	li	a5,136
1c016090:	b58a0413          	addi	s0,s4,-1192 # 1c01cb58 <g_mram_data>
1c016094:	42f90433          	p.mac	s0,s2,a5
1c016098:	1c01d7b7          	lui	a5,0x1c01d
1c01609c:	c706                	sw	ra,140(sp)
1c01609e:	c326                	sw	s1,132(sp)
1c0160a0:	dece                	sw	s3,124(sp)
1c0160a2:	dad6                	sw	s5,116(sp)
1c0160a4:	d6de                	sw	s7,108(sp)
1c0160a6:	d4e2                	sw	s8,104(sp)
1c0160a8:	d2e6                	sw	s9,100(sp)
1c0160aa:	d0ea                	sw	s10,96(sp)
1c0160ac:	79878793          	addi	a5,a5,1944 # 1c01d798 <pos_mram_open_count>
1c0160b0:	4127f983          	p.lbu	s3,s2(a5)
1c0160b4:	c500                	sw	s0,8(a0)
1c0160b6:	0985                	addi	s3,s3,1
1c0160b8:	0ff9f993          	andi	s3,s3,255
1c0160bc:	0137c923          	p.sb	s3,s2(a5)
1c0160c0:	1819b363          	p.bneimm	s3,1,1c016246 <mram_open+0x1d0>
1c0160c4:	0ff97793          	andi	a5,s2,255
1c0160c8:	06f40da3          	sb	a5,123(s0)
1c0160cc:	01590493          	addi	s1,s2,21
1c0160d0:	07d5                	addi	a5,a5,21
1c0160d2:	1a102c37          	lui	s8,0x1a102
1c0160d6:	06f40e23          	sb	a5,124(s0)
1c0160da:	049e                	slli	s1,s1,0x7
1c0160dc:	080c0793          	addi	a5,s8,128 # 1a102080 <__l1_heapsram_end+0xa0e2080>
1c0160e0:	94be                	add	s1,s1,a5
1c0160e2:	02cb2783          	lw	a5,44(s6)
1c0160e6:	00042e23          	sw	zero,28(s0)
1c0160ea:	00042223          	sw	zero,4(s0)
1c0160ee:	00042423          	sw	zero,8(s0)
1c0160f2:	c004                	sw	s1,0(s0)
1c0160f4:	08f42023          	sw	a5,128(s0)
1c0160f8:	030b4783          	lbu	a5,48(s6)
1c0160fc:	8aaa                	mv	s5,a0
1c0160fe:	1a103d37          	lui	s10,0x1a103
1c016102:	08f40223          	sb	a5,132(s0)
1c016106:	3bad                	jal	1c015e80 <pi_udma_core_lin_alloc>
1c016108:	87aa                	mv	a5,a0
1c01610a:	06f40d23          	sb	a5,122(s0)
1c01610e:	6789                	lui	a5,0x2
1c016110:	fe078b93          	addi	s7,a5,-32 # 1fe0 <__STACK_IRQ_SIZE+0x17e0>
1c016114:	0516                	slli	a0,a0,0x5
1c016116:	01757533          	and	a0,a0,s7
1c01611a:	b58a0a13          	addi	s4,s4,-1192
1c01611e:	4cc1                	li	s9,16
1c016120:	956a                	add	a0,a0,s10
1c016122:	01952e23          	sw	s9,28(a0)
1c016126:	4601                	li	a2,0
1c016128:	4585                	li	a1,1
1c01612a:	4505                	li	a0,1
1c01612c:	b31fe0ef          	jal	ra,1c014c5c <pi_pmu_domain_state_change>
1c016130:	3b81                	jal	1c015e80 <pi_udma_core_lin_alloc>
1c016132:	872a                	mv	a4,a0
1c016134:	0516                	slli	a0,a0,0x5
1c016136:	017577b3          	and	a5,a0,s7
1c01613a:	97ea                	add	a5,a5,s10
1c01613c:	06e40ca3          	sb	a4,121(s0)
1c016140:	0197ae23          	sw	s9,28(a5)
1c016144:	07c44783          	lbu	a5,124(s0)
1c016148:	00f999b3          	sll	s3,s3,a5
1c01614c:	013c2823          	sw	s3,16(s8)
1c016150:	013c2223          	sw	s3,4(s8)
1c016154:	07190893          	addi	a7,s2,113
1c016158:	1c011637          	lui	a2,0x1c011
1c01615c:	03800693          	li	a3,56
1c016160:	8bc60613          	addi	a2,a2,-1860 # 1c0108bc <pos_mram_handle_event>
1c016164:	28400793          	li	a5,644
1c016168:	00289593          	slli	a1,a7,0x2
1c01616c:	07290813          	addi	a6,s2,114
1c016170:	0087e5a3          	p.sw	s0,a1(a5)
1c016174:	00c6e5a3          	p.sw	a2,a1(a3)
1c016178:	07390513          	addi	a0,s2,115
1c01617c:	00281593          	slli	a1,a6,0x2
1c016180:	0087e5a3          	p.sw	s0,a1(a5)
1c016184:	00c6e5a3          	p.sw	a2,a1(a3)
1c016188:	00251593          	slli	a1,a0,0x2
1c01618c:	0087e5a3          	p.sw	s0,a1(a5)
1c016190:	00c6e5a3          	p.sw	a2,a1(a3)
1c016194:	07490593          	addi	a1,s2,116
1c016198:	00259313          	slli	t1,a1,0x2
1c01619c:	0087e323          	p.sw	s0,t1(a5)
1c0161a0:	1a1067b7          	lui	a5,0x1a106
1c0161a4:	00c6e323          	p.sw	a2,t1(a3)
1c0161a8:	0917ac23          	sw	a7,152(a5) # 1a106098 <__l1_heapsram_end+0xa0e6098>
1c0161ac:	0907ac23          	sw	a6,152(a5)
1c0161b0:	08a7ac23          	sw	a0,152(a5)
1c0161b4:	08b7ac23          	sw	a1,152(a5)
1c0161b8:	07a44783          	lbu	a5,122(s0)
1c0161bc:	00f4a023          	sw	a5,0(s1)
1c0161c0:	0ff77713          	andi	a4,a4,255
1c0161c4:	00e4a223          	sw	a4,4(s1)
1c0161c8:	030b4703          	lbu	a4,48(s6)
1c0161cc:	00171793          	slli	a5,a4,0x1
1c0161d0:	0716                	slli	a4,a4,0x5
1c0161d2:	8fd9                	or	a5,a5,a4
1c0161d4:	c007c7b3          	p.bset	a5,a5,0,0
1c0161d8:	00f4a423          	sw	a5,8(s1)
1c0161dc:	47bd                	li	a5,15
1c0161de:	02f4aa23          	sw	a5,52(s1)
1c0161e2:	02cb2983          	lw	s3,44(s6)
1c0161e6:	4501                	li	a0,0
1c0161e8:	b37fa0ef          	jal	ra,1c010d1e <pi_freq_get>
1c0161ec:	00098463          	beqz	s3,1c0161f4 <mram_open+0x17e>
1c0161f0:	06a9ca63          	blt	s3,a0,1c016264 <mram_open+0x1ee>
1c0161f4:	08800793          	li	a5,136
1c0161f8:	8752                	mv	a4,s4
1c0161fa:	42f90733          	p.mac	a4,s2,a5
1c0161fe:	4781                	li	a5,0
1c016200:	cb08                	sw	a0,16(a4)
1c016202:	c287c7b3          	p.bset	a5,a5,1,8
1c016206:	02f4a623          	sw	a5,44(s1)
1c01620a:	0204a023          	sw	zero,32(s1)
1c01620e:	08000793          	li	a5,128
1c016212:	02f4a023          	sw	a5,32(s1)
1c016216:	0c000793          	li	a5,192
1c01621a:	02f4a023          	sw	a5,32(s1)
1c01621e:	4515                	li	a0,5
1c016220:	442010ef          	jal	ra,1c017662 <pi_time_wait_us>
1c016224:	0e000793          	li	a5,224
1c016228:	02f4a023          	sw	a5,32(s1)
1c01622c:	06400513          	li	a0,100
1c016230:	432010ef          	jal	ra,1c017662 <pi_time_wait_us>
1c016234:	1c01e7b7          	lui	a5,0x1c01e
1c016238:	aec7a703          	lw	a4,-1300(a5) # 1c01daec <pi_efuse_regs>
1c01623c:	47c5                	li	a5,17
1c01623e:	02e7fc63          	bleu	a4,a5,1c016276 <mram_open+0x200>
1c016242:	8556                	mv	a0,s5
1c016244:	3bc5                	jal	1c016034 <pi_mram_trim_from_efuse>
1c016246:	40ba                	lw	ra,140(sp)
1c016248:	442a                	lw	s0,136(sp)
1c01624a:	449a                	lw	s1,132(sp)
1c01624c:	490a                	lw	s2,128(sp)
1c01624e:	59f6                	lw	s3,124(sp)
1c016250:	5a66                	lw	s4,120(sp)
1c016252:	5ad6                	lw	s5,116(sp)
1c016254:	5b46                	lw	s6,112(sp)
1c016256:	5bb6                	lw	s7,108(sp)
1c016258:	5c26                	lw	s8,104(sp)
1c01625a:	5c96                	lw	s9,100(sp)
1c01625c:	5d06                	lw	s10,96(sp)
1c01625e:	4501                	li	a0,0
1c016260:	6149                	addi	sp,sp,144
1c016262:	8082                	ret
1c016264:	00a987b3          	add	a5,s3,a0
1c016268:	17fd                	addi	a5,a5,-1
1c01626a:	0337c7b3          	div	a5,a5,s3
1c01626e:	02f54533          	div	a0,a0,a5
1c016272:	c808                	sw	a0,16(s0)
1c016274:	b779                	j	1c016202 <mram_open+0x18c>
1c016276:	1c01c4b7          	lui	s1,0x1c01c
1c01627a:	fd799737          	lui	a4,0xfd799
1c01627e:	1e448793          	addi	a5,s1,484 # 1c01c1e4 <trim_cfg_buffer>
1c016282:	d0070713          	addi	a4,a4,-768 # fd798d00 <pulp__FC+0xfd798d01>
1c016286:	c3d8                	sw	a4,4(a5)
1c016288:	62049737          	lui	a4,0x62049
1c01628c:	0d070713          	addi	a4,a4,208 # 620490d0 <__heapl2ram_size+0x45eb90d0>
1c016290:	c798                	sw	a4,8(a5)
1c016292:	04061737          	lui	a4,0x4061
1c016296:	82e70713          	addi	a4,a4,-2002 # 406082e <__l1_heapsram_size+0x404089e>
1c01629a:	c7d8                	sw	a4,12(a5)
1c01629c:	05800737          	lui	a4,0x5800
1c0162a0:	076d                	addi	a4,a4,27
1c0162a2:	cb98                	sw	a4,16(a5)
1c0162a4:	0007a023          	sw	zero,0(a5)
1c0162a8:	0007aa23          	sw	zero,20(a5)
1c0162ac:	0007ac23          	sw	zero,24(a5)
1c0162b0:	0028                	addi	a0,sp,8
1c0162b2:	429030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c0162b6:	08800793          	li	a5,136
1c0162ba:	42f90a33          	p.mac	s4,s2,a5
1c0162be:	1e448593          	addi	a1,s1,484
1c0162c2:	00aa2223          	sw	a0,4(s4)
1c0162c6:	8522                	mv	a0,s0
1c0162c8:	bf9ff0ef          	jal	ra,1c015ec0 <__rt_mram_trim_cfg_exec.isra.0.constprop.8>
1c0162cc:	0028                	addi	a0,sp,8
1c0162ce:	44f030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c0162d2:	4551                	li	a0,20
1c0162d4:	38e010ef          	jal	ra,1c017662 <pi_time_wait_us>
1c0162d8:	b7bd                	j	1c016246 <mram_open+0x1d0>

1c0162da <__pi_mram_init__>:
1c0162da:	1c01d7b7          	lui	a5,0x1c01d
1c0162de:	7947a783          	lw	a5,1940(a5) # 1c01d794 <g_conf>
1c0162e2:	cb89                	beqz	a5,1c0162f4 <__pi_mram_init__+0x1a>
1c0162e4:	1101                	addi	sp,sp,-32
1c0162e6:	0048                	addi	a0,sp,4
1c0162e8:	ce06                	sw	ra,28(sp)
1c0162ea:	c43e                	sw	a5,8(sp)
1c0162ec:	3369                	jal	1c016076 <mram_open>
1c0162ee:	40f2                	lw	ra,28(sp)
1c0162f0:	6105                	addi	sp,sp,32
1c0162f2:	8082                	ret
1c0162f4:	8082                	ret

1c0162f6 <__pi_mram_deinit__>:
1c0162f6:	1141                	addi	sp,sp,-16
1c0162f8:	c422                	sw	s0,8(sp)
1c0162fa:	1c01d437          	lui	s0,0x1c01d
1c0162fe:	79844783          	lbu	a5,1944(s0) # 1c01d798 <pos_mram_open_count>
1c016302:	c606                	sw	ra,12(sp)
1c016304:	c226                	sw	s1,4(sp)
1c016306:	c04a                	sw	s2,0(sp)
1c016308:	c7a1                	beqz	a5,1c016350 <__pi_mram_deinit__+0x5a>
1c01630a:	03400513          	li	a0,52
1c01630e:	1c01d937          	lui	s2,0x1c01d
1c016312:	22c040ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c016316:	79840413          	addi	s0,s0,1944
1c01631a:	b5890493          	addi	s1,s2,-1192 # 1c01cb58 <g_mram_data>
1c01631e:	c505                	beqz	a0,1c016346 <__pi_mram_deinit__+0x50>
1c016320:	1c01c7b7          	lui	a5,0x1c01c
1c016324:	47078793          	addi	a5,a5,1136 # 1c01c470 <mram_api>
1c016328:	c11c                	sw	a5,0(a0)
1c01632a:	07b4c783          	lbu	a5,123(s1)
1c01632e:	d51c                	sw	a5,40(a0)
1c016330:	0804a783          	lw	a5,128(s1)
1c016334:	d55c                	sw	a5,44(a0)
1c016336:	0844c783          	lbu	a5,132(s1)
1c01633a:	02f50823          	sb	a5,48(a0)
1c01633e:	1c01d7b7          	lui	a5,0x1c01d
1c016342:	78a7aa23          	sw	a0,1940(a5) # 1c01d794 <g_conf>
1c016346:	07b4c703          	lbu	a4,123(s1)
1c01634a:	00044783          	lbu	a5,0(s0)
1c01634e:	e799                	bnez	a5,1c01635c <__pi_mram_deinit__+0x66>
1c016350:	40b2                	lw	ra,12(sp)
1c016352:	4422                	lw	s0,8(sp)
1c016354:	4492                	lw	s1,4(sp)
1c016356:	4902                	lw	s2,0(sp)
1c016358:	0141                	addi	sp,sp,16
1c01635a:	8082                	ret
1c01635c:	17fd                	addi	a5,a5,-1
1c01635e:	0ff7f793          	andi	a5,a5,255
1c016362:	00f44723          	p.sb	a5,a4(s0)
1c016366:	f3f5                	bnez	a5,1c01634a <__pi_mram_deinit__+0x54>
1c016368:	b5890513          	addi	a0,s2,-1192
1c01636c:	b7bff0ef          	jal	ra,1c015ee6 <__pi_mram_close.part.3>
1c016370:	bfd9                	j	1c016346 <__pi_mram_deinit__+0x50>

1c016372 <mx25u_stall_task>:
1c016372:	30047373          	csrrci	t1,mstatus,8
1c016376:	01852e03          	lw	t3,24(a0)
1c01637a:	020e0763          	beqz	t3,1c0163a8 <mx25u_stall_task+0x36>
1c01637e:	d1dc                	sw	a5,36(a1)
1c016380:	491c                	lw	a5,16(a0)
1c016382:	cd90                	sw	a2,24(a1)
1c016384:	cdd4                	sw	a3,28(a1)
1c016386:	d198                	sw	a4,32(a1)
1c016388:	0305a423          	sw	a6,40(a1)
1c01638c:	0315a623          	sw	a7,44(a1)
1c016390:	0005a023          	sw	zero,0(a1)
1c016394:	cb81                	beqz	a5,1c0163a4 <mx25u_stall_task+0x32>
1c016396:	495c                	lw	a5,20(a0)
1c016398:	c38c                	sw	a1,0(a5)
1c01639a:	c94c                	sw	a1,20(a0)
1c01639c:	30031073          	csrw	mstatus,t1
1c0163a0:	4505                	li	a0,1
1c0163a2:	8082                	ret
1c0163a4:	c90c                	sw	a1,16(a0)
1c0163a6:	bfd5                	j	1c01639a <mx25u_stall_task+0x28>
1c0163a8:	cd0c                	sw	a1,24(a0)
1c0163aa:	30031073          	csrw	mstatus,t1
1c0163ae:	4501                	li	a0,0
1c0163b0:	8082                	ret

1c0163b2 <mx25u_erase_chip_async>:
1c0163b2:	1141                	addi	sp,sp,-16
1c0163b4:	c422                	sw	s0,8(sp)
1c0163b6:	4500                	lw	s0,8(a0)
1c0163b8:	c226                	sw	s1,4(sp)
1c0163ba:	4881                	li	a7,0
1c0163bc:	84aa                	mv	s1,a0
1c0163be:	4801                	li	a6,0
1c0163c0:	4781                	li	a5,0
1c0163c2:	4701                	li	a4,0
1c0163c4:	4681                	li	a3,0
1c0163c6:	4605                	li	a2,1
1c0163c8:	8522                	mv	a0,s0
1c0163ca:	c606                	sw	ra,12(sp)
1c0163cc:	375d                	jal	1c016372 <mx25u_stall_task>
1c0163ce:	e11d                	bnez	a0,1c0163f4 <mx25u_erase_chip_async+0x42>
1c0163d0:	1c0175b7          	lui	a1,0x1c017
1c0163d4:	8626                	mv	a2,s1
1c0163d6:	bd658593          	addi	a1,a1,-1066 # 1c016bd6 <mx25u_check_erase>
1c0163da:	02840513          	addi	a0,s0,40
1c0163de:	313030ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c0163e2:	4422                	lw	s0,8(sp)
1c0163e4:	40b2                	lw	ra,12(sp)
1c0163e6:	4492                	lw	s1,4(sp)
1c0163e8:	65e1                	lui	a1,0x18
1c0163ea:	6a058593          	addi	a1,a1,1696 # 186a0 <__heapfcram_size+0x9460>
1c0163ee:	0141                	addi	sp,sp,16
1c0163f0:	3350306f          	j	1c019f24 <__pi_os_evt_push_delayed_us>
1c0163f4:	40b2                	lw	ra,12(sp)
1c0163f6:	4422                	lw	s0,8(sp)
1c0163f8:	4492                	lw	s1,4(sp)
1c0163fa:	0141                	addi	sp,sp,16
1c0163fc:	8082                	ret

1c0163fe <mx25u_erase_chip>:
1c0163fe:	7159                	addi	sp,sp,-112
1c016400:	d4a2                	sw	s0,104(sp)
1c016402:	842a                	mv	s0,a0
1c016404:	0028                	addi	a0,sp,8
1c016406:	d686                	sw	ra,108(sp)
1c016408:	2d3030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c01640c:	85aa                	mv	a1,a0
1c01640e:	8522                	mv	a0,s0
1c016410:	374d                	jal	1c0163b2 <mx25u_erase_chip_async>
1c016412:	0028                	addi	a0,sp,8
1c016414:	309030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016418:	50b6                	lw	ra,108(sp)
1c01641a:	5426                	lw	s0,104(sp)
1c01641c:	4501                	li	a0,0
1c01641e:	6165                	addi	sp,sp,112
1c016420:	8082                	ret

1c016422 <mx25u_ioctl>:
1c016422:	1141                	addi	sp,sp,-16
1c016424:	c606                	sw	ra,12(sp)
1c016426:	0225a063          	p.beqimm	a1,2,1c016446 <mx25u_ioctl+0x24>
1c01642a:	0035ae63          	p.beqimm	a1,3,1c016446 <mx25u_ioctl+0x24>
1c01642e:	e981                	bnez	a1,1c01643e <mx25u_ioctl+0x1c>
1c016430:	4154                	lw	a3,4(a0)
1c016432:	0446d703          	lhu	a4,68(a3)
1c016436:	5e94                	lw	a3,56(a3)
1c016438:	c218                	sw	a4,0(a2)
1c01643a:	c614                	sw	a3,8(a2)
1c01643c:	c258                	sw	a4,4(a2)
1c01643e:	40b2                	lw	ra,12(sp)
1c016440:	4501                	li	a0,0
1c016442:	0141                	addi	sp,sp,16
1c016444:	8082                	ret
1c016446:	4508                	lw	a0,8(a0)
1c016448:	027010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c01644c:	bfcd                	j	1c01643e <mx25u_ioctl+0x1c>

1c01644e <mx25u_read_2d_async>:
1c01644e:	7179                	addi	sp,sp,-48
1c016450:	d422                	sw	s0,40(sp)
1c016452:	4500                	lw	s0,8(a0)
1c016454:	d226                	sw	s1,36(sp)
1c016456:	84ae                	mv	s1,a1
1c016458:	d04a                	sw	s2,32(sp)
1c01645a:	ce4e                	sw	s3,28(sp)
1c01645c:	cc52                	sw	s4,24(sp)
1c01645e:	ca56                	sw	s5,20(sp)
1c016460:	c85a                	sw	s6,16(sp)
1c016462:	8932                	mv	s2,a2
1c016464:	8b2a                	mv	s6,a0
1c016466:	89b6                	mv	s3,a3
1c016468:	8a3a                	mv	s4,a4
1c01646a:	8abe                	mv	s5,a5
1c01646c:	85c2                	mv	a1,a6
1c01646e:	88be                	mv	a7,a5
1c016470:	883a                	mv	a6,a4
1c016472:	87b6                	mv	a5,a3
1c016474:	8732                	mv	a4,a2
1c016476:	86a6                	mv	a3,s1
1c016478:	4619                	li	a2,6
1c01647a:	8522                	mv	a0,s0
1c01647c:	d606                	sw	ra,44(sp)
1c01647e:	3dd5                	jal	1c016372 <mx25u_stall_task>
1c016480:	e90d                	bnez	a0,1c0164b2 <mx25u_read_2d_async+0x64>
1c016482:	1c0175b7          	lui	a1,0x1c017
1c016486:	865a                	mv	a2,s6
1c016488:	91258593          	addi	a1,a1,-1774 # 1c016912 <mx25u_handle_pending_task>
1c01648c:	02840513          	addi	a0,s0,40
1c016490:	277030ef          	jal	ra,1c019f06 <__pi_os_evt_callback_irq_init>
1c016494:	4785                	li	a5,1
1c016496:	1c01c837          	lui	a6,0x1c01c
1c01649a:	c03e                	sw	a5,0(sp)
1c01649c:	88aa                	mv	a7,a0
1c01649e:	53080813          	addi	a6,a6,1328 # 1c01c530 <mx25u_read_op>
1c0164a2:	87d6                	mv	a5,s5
1c0164a4:	8752                	mv	a4,s4
1c0164a6:	86ce                	mv	a3,s3
1c0164a8:	864a                	mv	a2,s2
1c0164aa:	85a6                	mv	a1,s1
1c0164ac:	8522                	mv	a0,s0
1c0164ae:	6b0010ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c0164b2:	50b2                	lw	ra,44(sp)
1c0164b4:	5422                	lw	s0,40(sp)
1c0164b6:	5492                	lw	s1,36(sp)
1c0164b8:	5902                	lw	s2,32(sp)
1c0164ba:	49f2                	lw	s3,28(sp)
1c0164bc:	4a62                	lw	s4,24(sp)
1c0164be:	4ad2                	lw	s5,20(sp)
1c0164c0:	4b42                	lw	s6,16(sp)
1c0164c2:	6145                	addi	sp,sp,48
1c0164c4:	8082                	ret

1c0164c6 <mx25u_copy_2d_async>:
1c0164c6:	00080a63          	beqz	a6,1c0164da <mx25u_copy_2d_async+0x14>
1c0164ca:	1141                	addi	sp,sp,-16
1c0164cc:	8846                	mv	a6,a7
1c0164ce:	c606                	sw	ra,12(sp)
1c0164d0:	3fbd                	jal	1c01644e <mx25u_read_2d_async>
1c0164d2:	40b2                	lw	ra,12(sp)
1c0164d4:	4501                	li	a0,0
1c0164d6:	0141                	addi	sp,sp,16
1c0164d8:	8082                	ret
1c0164da:	557d                	li	a0,-1
1c0164dc:	8082                	ret

1c0164de <mx25u_copy_2d>:
1c0164de:	7175                	addi	sp,sp,-144
1c0164e0:	c522                	sw	s0,136(sp)
1c0164e2:	842a                	mv	s0,a0
1c0164e4:	1028                	addi	a0,sp,40
1c0164e6:	ce2e                	sw	a1,28(sp)
1c0164e8:	cc32                	sw	a2,24(sp)
1c0164ea:	ca36                	sw	a3,20(sp)
1c0164ec:	c83a                	sw	a4,16(sp)
1c0164ee:	c63e                	sw	a5,12(sp)
1c0164f0:	c442                	sw	a6,8(sp)
1c0164f2:	c706                	sw	ra,140(sp)
1c0164f4:	1e7030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c0164f8:	4822                	lw	a6,8(sp)
1c0164fa:	45f2                	lw	a1,28(sp)
1c0164fc:	4662                	lw	a2,24(sp)
1c0164fe:	46d2                	lw	a3,20(sp)
1c016500:	4742                	lw	a4,16(sp)
1c016502:	47b2                	lw	a5,12(sp)
1c016504:	557d                	li	a0,-1
1c016506:	00080a63          	beqz	a6,1c01651a <mx25u_copy_2d+0x3c>
1c01650a:	02810813          	addi	a6,sp,40
1c01650e:	8522                	mv	a0,s0
1c016510:	3f3d                	jal	1c01644e <mx25u_read_2d_async>
1c016512:	1028                	addi	a0,sp,40
1c016514:	209030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016518:	4501                	li	a0,0
1c01651a:	40ba                	lw	ra,140(sp)
1c01651c:	442a                	lw	s0,136(sp)
1c01651e:	6149                	addi	sp,sp,144
1c016520:	8082                	ret

1c016522 <pi_octospi_write.constprop.5>:
1c016522:	7175                	addi	sp,sp,-144
1c016524:	c522                	sw	s0,136(sp)
1c016526:	842a                	mv	s0,a0
1c016528:	1028                	addi	a0,sp,40
1c01652a:	c706                	sw	ra,140(sp)
1c01652c:	ce2e                	sw	a1,28(sp)
1c01652e:	ca36                	sw	a3,20(sp)
1c016530:	cc32                	sw	a2,24(sp)
1c016532:	1a9030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016536:	47e2                	lw	a5,24(sp)
1c016538:	46d2                	lw	a3,20(sp)
1c01653a:	45f2                	lw	a1,28(sp)
1c01653c:	88aa                	mv	a7,a0
1c01653e:	8836                	mv	a6,a3
1c016540:	862e                	mv	a2,a1
1c016542:	8522                	mv	a0,s0
1c016544:	c002                	sw	zero,0(sp)
1c016546:	873e                	mv	a4,a5
1c016548:	86be                	mv	a3,a5
1c01654a:	4581                	li	a1,0
1c01654c:	612010ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016550:	1028                	addi	a0,sp,40
1c016552:	1cb030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016556:	40ba                	lw	ra,140(sp)
1c016558:	442a                	lw	s0,136(sp)
1c01655a:	6149                	addi	sp,sp,144
1c01655c:	8082                	ret

1c01655e <mx25u_close>:
1c01655e:	7179                	addi	sp,sp,-48
1c016560:	d422                	sw	s0,40(sp)
1c016562:	4500                	lw	s0,8(a0)
1c016564:	d606                	sw	ra,44(sp)
1c016566:	d226                	sw	s1,36(sp)
1c016568:	14444783          	lbu	a5,324(s0)
1c01656c:	c7b9                	beqz	a5,1c0165ba <mx25u_close+0x5c>
1c01656e:	17fd                	addi	a5,a5,-1
1c016570:	0ff7f793          	andi	a5,a5,255
1c016574:	14f40223          	sb	a5,324(s0)
1c016578:	e3a9                	bnez	a5,1c0165ba <mx25u_close+0x5c>
1c01657a:	000467b7          	lui	a5,0x46
1c01657e:	69978793          	addi	a5,a5,1689 # 46699 <__l1_heapsram_size+0x26709>
1c016582:	4489                	li	s1,2
1c016584:	0034                	addi	a3,sp,8
1c016586:	4601                	li	a2,0
1c016588:	4581                	li	a1,0
1c01658a:	8522                	mv	a0,s0
1c01658c:	c43e                	sw	a5,8(sp)
1c01658e:	c602                	sw	zero,12(sp)
1c016590:	c826                	sw	s1,16(sp)
1c016592:	3f41                	jal	1c016522 <pi_octospi_write.constprop.5>
1c016594:	0004a7b7          	lui	a5,0x4a
1c016598:	96678793          	addi	a5,a5,-1690 # 49966 <__l1_heapsram_size+0x299d6>
1c01659c:	0854                	addi	a3,sp,20
1c01659e:	4601                	li	a2,0
1c0165a0:	4581                	li	a1,0
1c0165a2:	8522                	mv	a0,s0
1c0165a4:	ca3e                	sw	a5,20(sp)
1c0165a6:	cc02                	sw	zero,24(sp)
1c0165a8:	ce26                	sw	s1,28(sp)
1c0165aa:	3fa5                	jal	1c016522 <pi_octospi_write.constprop.5>
1c0165ac:	03c00513          	li	a0,60
1c0165b0:	0b2010ef          	jal	ra,1c017662 <pi_time_wait_us>
1c0165b4:	8522                	mv	a0,s0
1c0165b6:	404010ef          	jal	ra,1c0179ba <pi_octospi_close>
1c0165ba:	50b2                	lw	ra,44(sp)
1c0165bc:	5422                	lw	s0,40(sp)
1c0165be:	5492                	lw	s1,36(sp)
1c0165c0:	4501                	li	a0,0
1c0165c2:	6145                	addi	sp,sp,48
1c0165c4:	8082                	ret

1c0165c6 <mx25u_open>:
1c0165c6:	7131                	addi	sp,sp,-192
1c0165c8:	db26                	sw	s1,180(sp)
1c0165ca:	4144                	lw	s1,4(a0)
1c0165cc:	dd22                	sw	s0,184(sp)
1c0165ce:	df06                	sw	ra,188(sp)
1c0165d0:	5480                	lw	s0,40(s1)
1c0165d2:	d94a                	sw	s2,176(sp)
1c0165d4:	d74e                	sw	s3,172(sp)
1c0165d6:	e409                	bnez	s0,1c0165e0 <mx25u_open+0x1a>
1c0165d8:	1c01d437          	lui	s0,0x1c01d
1c0165dc:	be040413          	addi	s0,s0,-1056 # 1c01cbe0 <mx25u_data>
1c0165e0:	c500                	sw	s0,8(a0)
1c0165e2:	14444783          	lbu	a5,324(s0)
1c0165e6:	cf89                	beqz	a5,1c016600 <mx25u_open+0x3a>
1c0165e8:	0785                	addi	a5,a5,1
1c0165ea:	14f40223          	sb	a5,324(s0)
1c0165ee:	4901                	li	s2,0
1c0165f0:	50fa                	lw	ra,188(sp)
1c0165f2:	546a                	lw	s0,184(sp)
1c0165f4:	854a                	mv	a0,s2
1c0165f6:	54da                	lw	s1,180(sp)
1c0165f8:	594a                	lw	s2,176(sp)
1c0165fa:	59ba                	lw	s3,172(sp)
1c0165fc:	6129                	addi	sp,sp,192
1c0165fe:	8082                	ret
1c016600:	1068                	addi	a0,sp,44
1c016602:	10e010ef          	jal	ra,1c017710 <pi_octospi_conf_init>
1c016606:	8526                	mv	a0,s1
1c016608:	69c010ef          	jal	ra,1c017ca4 <__flash_conf_init>
1c01660c:	54dc                	lw	a5,44(s1)
1c01660e:	106c                	addi	a1,sp,44
1c016610:	8522                	mv	a0,s0
1c016612:	02f10623          	sb	a5,44(sp)
1c016616:	589c                	lw	a5,48(s1)
1c016618:	da02                	sw	zero,52(sp)
1c01661a:	d83e                	sw	a5,48(sp)
1c01661c:	58dc                	lw	a5,52(s1)
1c01661e:	02f106a3          	sb	a5,45(sp)
1c016622:	5cdc                	lw	a5,60(s1)
1c016624:	dc3e                	sw	a5,56(sp)
1c016626:	00448793          	addi	a5,s1,4
1c01662a:	c0be                	sw	a5,64(sp)
1c01662c:	40bc                	lw	a5,64(s1)
1c01662e:	c2be                	sw	a5,68(sp)
1c016630:	b2afe0ef          	jal	ra,1c01495a <pi_open_from_conf>
1c016634:	8522                	mv	a0,s0
1c016636:	0f2010ef          	jal	ra,1c017728 <pi_octospi_open>
1c01663a:	892a                	mv	s2,a0
1c01663c:	0c051e63          	bnez	a0,1c016718 <mx25u_open+0x152>
1c016640:	4601                	li	a2,0
1c016642:	4589                	li	a1,2
1c016644:	00042c23          	sw	zero,24(s0)
1c016648:	00042823          	sw	zero,16(s0)
1c01664c:	02042223          	sw	zero,36(s0)
1c016650:	00042e23          	sw	zero,28(s0)
1c016654:	8522                	mv	a0,s0
1c016656:	618010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c01665a:	000409b7          	lui	s3,0x40
1c01665e:	00698793          	addi	a5,s3,6 # 40006 <__l1_heapsram_size+0x20076>
1c016662:	d03e                	sw	a5,32(sp)
1c016664:	6795                	lui	a5,0x5
1c016666:	20178793          	addi	a5,a5,513 # 5201 <__STACK_IRQ_SIZE+0x4a01>
1c01666a:	1014                	addi	a3,sp,32
1c01666c:	4601                	li	a2,0
1c01666e:	4581                	li	a1,0
1c016670:	8522                	mv	a0,s0
1c016672:	d43e                	sw	a5,40(sp)
1c016674:	d202                	sw	zero,36(sp)
1c016676:	3575                	jal	1c016522 <pi_octospi_write.constprop.5>
1c016678:	679d                	lui	a5,0x7
1c01667a:	24178793          	addi	a5,a5,577 # 7241 <__STACK_IRQ_SIZE+0x6a41>
1c01667e:	00b4                	addi	a3,sp,72
1c016680:	4605                	li	a2,1
1c016682:	086c                	addi	a1,sp,28
1c016684:	c8be                	sw	a5,80(sp)
1c016686:	8522                	mv	a0,s0
1c016688:	4789                	li	a5,2
1c01668a:	07298993          	addi	s3,s3,114
1c01668e:	ce3e                	sw	a5,28(sp)
1c016690:	c4ce                	sw	s3,72(sp)
1c016692:	c682                	sw	zero,76(sp)
1c016694:	3579                	jal	1c016522 <pi_octospi_write.constprop.5>
1c016696:	03c00513          	li	a0,60
1c01669a:	7c9000ef          	jal	ra,1c017662 <pi_time_wait_us>
1c01669e:	0004a7b7          	lui	a5,0x4a
1c0166a2:	f6078793          	addi	a5,a5,-160 # 49f60 <__l1_heapsram_size+0x29fd0>
1c0166a6:	d03e                	sw	a5,32(sp)
1c0166a8:	47a5                	li	a5,9
1c0166aa:	d23e                	sw	a5,36(sp)
1c0166ac:	6791                	lui	a5,0x4
1c0166ae:	04278793          	addi	a5,a5,66 # 4042 <__STACK_IRQ_SIZE+0x3842>
1c0166b2:	00a8                	addi	a0,sp,72
1c0166b4:	d43e                	sw	a5,40(sp)
1c0166b6:	ce02                	sw	zero,28(sp)
1c0166b8:	023030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c0166bc:	4785                	li	a5,1
1c0166be:	88aa                	mv	a7,a0
1c0166c0:	02010813          	addi	a6,sp,32
1c0166c4:	470d                	li	a4,3
1c0166c6:	468d                	li	a3,3
1c0166c8:	0870                	addi	a2,sp,28
1c0166ca:	4581                	li	a1,0
1c0166cc:	c03e                	sw	a5,0(sp)
1c0166ce:	8522                	mv	a0,s0
1c0166d0:	478d                	li	a5,3
1c0166d2:	48c010ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c0166d6:	00a8                	addi	a0,sp,72
1c0166d8:	045030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c0166dc:	01e15783          	lhu	a5,30(sp)
1c0166e0:	4605                	li	a2,1
1c0166e2:	4589                	li	a1,2
1c0166e4:	fc978713          	addi	a4,a5,-55
1c0166e8:	008007b7          	lui	a5,0x800
1c0166ec:	00e797b3          	sll	a5,a5,a4
1c0166f0:	dc9c                	sw	a5,56(s1)
1c0166f2:	8522                	mv	a0,s0
1c0166f4:	57a010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c0166f8:	58dc                	lw	a5,52(s1)
1c0166fa:	cb89                	beqz	a5,1c01670c <mx25u_open+0x146>
1c0166fc:	1c01c637          	lui	a2,0x1c01c
1c016700:	53060613          	addi	a2,a2,1328 # 1c01c530 <mx25u_read_op>
1c016704:	4585                	li	a1,1
1c016706:	8522                	mv	a0,s0
1c016708:	566010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c01670c:	14444783          	lbu	a5,324(s0)
1c016710:	0785                	addi	a5,a5,1
1c016712:	14f40223          	sb	a5,324(s0)
1c016716:	bde9                	j	1c0165f0 <mx25u_open+0x2a>
1c016718:	5979                	li	s2,-2
1c01671a:	bdd9                	j	1c0165f0 <mx25u_open+0x2a>

1c01671c <mx25u_read_async>:
1c01671c:	7179                	addi	sp,sp,-48
1c01671e:	d422                	sw	s0,40(sp)
1c016720:	4500                	lw	s0,8(a0)
1c016722:	d226                	sw	s1,36(sp)
1c016724:	84ae                	mv	s1,a1
1c016726:	d04a                	sw	s2,32(sp)
1c016728:	ce4e                	sw	s3,28(sp)
1c01672a:	cc52                	sw	s4,24(sp)
1c01672c:	8932                	mv	s2,a2
1c01672e:	8a2a                	mv	s4,a0
1c016730:	89b6                	mv	s3,a3
1c016732:	85ba                	mv	a1,a4
1c016734:	87b6                	mv	a5,a3
1c016736:	8732                	mv	a4,a2
1c016738:	4881                	li	a7,0
1c01673a:	4801                	li	a6,0
1c01673c:	86a6                	mv	a3,s1
1c01673e:	4615                	li	a2,5
1c016740:	8522                	mv	a0,s0
1c016742:	d606                	sw	ra,44(sp)
1c016744:	c2fff0ef          	jal	ra,1c016372 <mx25u_stall_task>
1c016748:	e90d                	bnez	a0,1c01677a <mx25u_read_async+0x5e>
1c01674a:	1c0175b7          	lui	a1,0x1c017
1c01674e:	8652                	mv	a2,s4
1c016750:	91258593          	addi	a1,a1,-1774 # 1c016912 <mx25u_handle_pending_task>
1c016754:	02840513          	addi	a0,s0,40
1c016758:	7ae030ef          	jal	ra,1c019f06 <__pi_os_evt_callback_irq_init>
1c01675c:	4785                	li	a5,1
1c01675e:	1c01c837          	lui	a6,0x1c01c
1c016762:	c03e                	sw	a5,0(sp)
1c016764:	88aa                	mv	a7,a0
1c016766:	53080813          	addi	a6,a6,1328 # 1c01c530 <mx25u_read_op>
1c01676a:	87ce                	mv	a5,s3
1c01676c:	874e                	mv	a4,s3
1c01676e:	86ce                	mv	a3,s3
1c016770:	864a                	mv	a2,s2
1c016772:	85a6                	mv	a1,s1
1c016774:	8522                	mv	a0,s0
1c016776:	3e8010ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c01677a:	50b2                	lw	ra,44(sp)
1c01677c:	5422                	lw	s0,40(sp)
1c01677e:	5492                	lw	s1,36(sp)
1c016780:	5902                	lw	s2,32(sp)
1c016782:	49f2                	lw	s3,28(sp)
1c016784:	4a62                	lw	s4,24(sp)
1c016786:	6145                	addi	sp,sp,48
1c016788:	8082                	ret

1c01678a <mx25u_read>:
1c01678a:	7119                	addi	sp,sp,-128
1c01678c:	dca2                	sw	s0,120(sp)
1c01678e:	842a                	mv	s0,a0
1c016790:	0828                	addi	a0,sp,24
1c016792:	de86                	sw	ra,124(sp)
1c016794:	c62e                	sw	a1,12(sp)
1c016796:	c432                	sw	a2,8(sp)
1c016798:	c236                	sw	a3,4(sp)
1c01679a:	740030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c01679e:	4692                	lw	a3,4(sp)
1c0167a0:	4622                	lw	a2,8(sp)
1c0167a2:	45b2                	lw	a1,12(sp)
1c0167a4:	872a                	mv	a4,a0
1c0167a6:	8522                	mv	a0,s0
1c0167a8:	3f95                	jal	1c01671c <mx25u_read_async>
1c0167aa:	0828                	addi	a0,sp,24
1c0167ac:	770030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c0167b0:	50f6                	lw	ra,124(sp)
1c0167b2:	5466                	lw	s0,120(sp)
1c0167b4:	4501                	li	a0,0
1c0167b6:	6109                	addi	sp,sp,128
1c0167b8:	8082                	ret

1c0167ba <mx25u_write_enable>:
1c0167ba:	7179                	addi	sp,sp,-48
1c0167bc:	4601                	li	a2,0
1c0167be:	4589                	li	a1,2
1c0167c0:	d606                	sw	ra,44(sp)
1c0167c2:	d422                	sw	s0,40(sp)
1c0167c4:	d226                	sw	s1,36(sp)
1c0167c6:	842a                	mv	s0,a0
1c0167c8:	4a6010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c0167cc:	000407b7          	lui	a5,0x40
1c0167d0:	6f978793          	addi	a5,a5,1785 # 406f9 <__l1_heapsram_size+0x20769>
1c0167d4:	84aa                	mv	s1,a0
1c0167d6:	ca3e                	sw	a5,20(sp)
1c0167d8:	0d840513          	addi	a0,s0,216
1c0167dc:	4789                	li	a5,2
1c0167de:	ce3e                	sw	a5,28(sp)
1c0167e0:	cc02                	sw	zero,24(sp)
1c0167e2:	c802                	sw	zero,16(sp)
1c0167e4:	6f6030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c0167e8:	88aa                	mv	a7,a0
1c0167ea:	01410813          	addi	a6,sp,20
1c0167ee:	0810                	addi	a2,sp,16
1c0167f0:	8522                	mv	a0,s0
1c0167f2:	c002                	sw	zero,0(sp)
1c0167f4:	4781                	li	a5,0
1c0167f6:	4701                	li	a4,0
1c0167f8:	4681                	li	a3,0
1c0167fa:	4581                	li	a1,0
1c0167fc:	362010ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016800:	8626                	mv	a2,s1
1c016802:	8522                	mv	a0,s0
1c016804:	4589                	li	a1,2
1c016806:	468010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c01680a:	50b2                	lw	ra,44(sp)
1c01680c:	5422                	lw	s0,40(sp)
1c01680e:	5492                	lw	s1,36(sp)
1c016810:	6145                	addi	sp,sp,48
1c016812:	8082                	ret

1c016814 <mx25u_erase_sector_async>:
1c016814:	1101                	addi	sp,sp,-32
1c016816:	cc22                	sw	s0,24(sp)
1c016818:	4500                	lw	s0,8(a0)
1c01681a:	ca26                	sw	s1,20(sp)
1c01681c:	84ae                	mv	s1,a1
1c01681e:	c84a                	sw	s2,16(sp)
1c016820:	85b2                	mv	a1,a2
1c016822:	892a                	mv	s2,a0
1c016824:	4881                	li	a7,0
1c016826:	4801                	li	a6,0
1c016828:	4781                	li	a5,0
1c01682a:	4701                	li	a4,0
1c01682c:	86a6                	mv	a3,s1
1c01682e:	4609                	li	a2,2
1c016830:	8522                	mv	a0,s0
1c016832:	ce06                	sw	ra,28(sp)
1c016834:	b3fff0ef          	jal	ra,1c016372 <mx25u_stall_task>
1c016838:	e91d                	bnez	a0,1c01686e <mx25u_erase_sector_async+0x5a>
1c01683a:	8522                	mv	a0,s0
1c01683c:	3fbd                	jal	1c0167ba <mx25u_write_enable>
1c01683e:	1c0175b7          	lui	a1,0x1c017
1c016842:	864a                	mv	a2,s2
1c016844:	bd658593          	addi	a1,a1,-1066 # 1c016bd6 <mx25u_check_erase>
1c016848:	02840513          	addi	a0,s0,40
1c01684c:	6a4030ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c016850:	1c01c837          	lui	a6,0x1c01c
1c016854:	88aa                	mv	a7,a0
1c016856:	c002                	sw	zero,0(sp)
1c016858:	51880813          	addi	a6,a6,1304 # 1c01c518 <mx25u_erase_op>
1c01685c:	4781                	li	a5,0
1c01685e:	4701                	li	a4,0
1c016860:	4681                	li	a3,0
1c016862:	00c40613          	addi	a2,s0,12
1c016866:	85a6                	mv	a1,s1
1c016868:	8522                	mv	a0,s0
1c01686a:	2f4010ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c01686e:	40f2                	lw	ra,28(sp)
1c016870:	4462                	lw	s0,24(sp)
1c016872:	44d2                	lw	s1,20(sp)
1c016874:	4942                	lw	s2,16(sp)
1c016876:	6105                	addi	sp,sp,32
1c016878:	8082                	ret

1c01687a <mx25u_erase_sector>:
1c01687a:	7119                	addi	sp,sp,-128
1c01687c:	dca2                	sw	s0,120(sp)
1c01687e:	842a                	mv	s0,a0
1c016880:	0828                	addi	a0,sp,24
1c016882:	de86                	sw	ra,124(sp)
1c016884:	c62e                	sw	a1,12(sp)
1c016886:	654030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c01688a:	45b2                	lw	a1,12(sp)
1c01688c:	862a                	mv	a2,a0
1c01688e:	8522                	mv	a0,s0
1c016890:	3751                	jal	1c016814 <mx25u_erase_sector_async>
1c016892:	0828                	addi	a0,sp,24
1c016894:	688030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016898:	50f6                	lw	ra,124(sp)
1c01689a:	5466                	lw	s0,120(sp)
1c01689c:	4501                	li	a0,0
1c01689e:	6109                	addi	sp,sp,128
1c0168a0:	8082                	ret

1c0168a2 <mx25u_erase_async>:
1c0168a2:	451c                	lw	a5,8(a0)
1c0168a4:	30047773          	csrrci	a4,mstatus,8
1c0168a8:	0247a803          	lw	a6,36(a5)
1c0168ac:	02080463          	beqz	a6,1c0168d4 <mx25u_erase_async+0x32>
1c0168b0:	d290                	sw	a2,32(a3)
1c0168b2:	4fd0                	lw	a2,28(a5)
1c0168b4:	450d                	li	a0,3
1c0168b6:	ce88                	sw	a0,24(a3)
1c0168b8:	cecc                	sw	a1,28(a3)
1c0168ba:	0206a223          	sw	zero,36(a3)
1c0168be:	0006a023          	sw	zero,0(a3)
1c0168c2:	c619                	beqz	a2,1c0168d0 <mx25u_erase_async+0x2e>
1c0168c4:	5390                	lw	a2,32(a5)
1c0168c6:	c214                	sw	a3,0(a2)
1c0168c8:	d394                	sw	a3,32(a5)
1c0168ca:	30071073          	csrw	mstatus,a4
1c0168ce:	8082                	ret
1c0168d0:	cfd4                	sw	a3,28(a5)
1c0168d2:	bfdd                	j	1c0168c8 <mx25u_erase_async+0x26>
1c0168d4:	d3d4                	sw	a3,36(a5)
1c0168d6:	30071073          	csrw	mstatus,a4
1c0168da:	12b7ae23          	sw	a1,316(a5)
1c0168de:	14c7a023          	sw	a2,320(a5)
1c0168e2:	8b2fa06f          	j	1c010994 <mx25u_erase_resume>

1c0168e6 <mx25u_erase>:
1c0168e6:	7119                	addi	sp,sp,-128
1c0168e8:	dca2                	sw	s0,120(sp)
1c0168ea:	842a                	mv	s0,a0
1c0168ec:	0828                	addi	a0,sp,24
1c0168ee:	de86                	sw	ra,124(sp)
1c0168f0:	c62e                	sw	a1,12(sp)
1c0168f2:	c432                	sw	a2,8(sp)
1c0168f4:	5e6030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c0168f8:	4622                	lw	a2,8(sp)
1c0168fa:	45b2                	lw	a1,12(sp)
1c0168fc:	0834                	addi	a3,sp,24
1c0168fe:	8522                	mv	a0,s0
1c016900:	374d                	jal	1c0168a2 <mx25u_erase_async>
1c016902:	0828                	addi	a0,sp,24
1c016904:	618030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016908:	50f6                	lw	ra,124(sp)
1c01690a:	5466                	lw	s0,120(sp)
1c01690c:	4501                	li	a0,0
1c01690e:	6109                	addi	sp,sp,128
1c016910:	8082                	ret

1c016912 <mx25u_handle_pending_task>:
1c016912:	1141                	addi	sp,sp,-16
1c016914:	c422                	sw	s0,8(sp)
1c016916:	c226                	sw	s1,4(sp)
1c016918:	c606                	sw	ra,12(sp)
1c01691a:	c04a                	sw	s2,0(sp)
1c01691c:	842a                	mv	s0,a0
1c01691e:	4504                	lw	s1,8(a0)
1c016920:	30047973          	csrrci	s2,mstatus,8
1c016924:	4c88                	lw	a0,24(s1)
1c016926:	5fa030ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c01692a:	4898                	lw	a4,16(s1)
1c01692c:	0004ac23          	sw	zero,24(s1)
1c016930:	c319                	beqz	a4,1c016936 <mx25u_handle_pending_task+0x24>
1c016932:	431c                	lw	a5,0(a4)
1c016934:	c89c                	sw	a5,16(s1)
1c016936:	30091073          	csrw	mstatus,s2
1c01693a:	cf45                	beqz	a4,1c0169f2 <mx25u_handle_pending_task+0xe0>
1c01693c:	4f1c                	lw	a5,24(a4)
1c01693e:	ef81                	bnez	a5,1c016956 <mx25u_handle_pending_task+0x44>
1c016940:	8522                	mv	a0,s0
1c016942:	4422                	lw	s0,8(sp)
1c016944:	40b2                	lw	ra,12(sp)
1c016946:	4492                	lw	s1,4(sp)
1c016948:	4902                	lw	s2,0(sp)
1c01694a:	5354                	lw	a3,36(a4)
1c01694c:	5310                	lw	a2,32(a4)
1c01694e:	4f4c                	lw	a1,28(a4)
1c016950:	0141                	addi	sp,sp,16
1c016952:	96cfa06f          	j	1c010abe <mx25u_program_async>
1c016956:	0017bb63          	p.bneimm	a5,1,1c01696c <mx25u_handle_pending_task+0x5a>
1c01695a:	8522                	mv	a0,s0
1c01695c:	4422                	lw	s0,8(sp)
1c01695e:	40b2                	lw	ra,12(sp)
1c016960:	4492                	lw	s1,4(sp)
1c016962:	4902                	lw	s2,0(sp)
1c016964:	85ba                	mv	a1,a4
1c016966:	0141                	addi	sp,sp,16
1c016968:	a4bff06f          	j	1c0163b2 <mx25u_erase_chip_async>
1c01696c:	0027bb63          	p.bneimm	a5,2,1c016982 <mx25u_handle_pending_task+0x70>
1c016970:	8522                	mv	a0,s0
1c016972:	4422                	lw	s0,8(sp)
1c016974:	40b2                	lw	ra,12(sp)
1c016976:	4492                	lw	s1,4(sp)
1c016978:	4902                	lw	s2,0(sp)
1c01697a:	4f4c                	lw	a1,28(a4)
1c01697c:	863a                	mv	a2,a4
1c01697e:	0141                	addi	sp,sp,16
1c016980:	bd51                	j	1c016814 <mx25u_erase_sector_async>
1c016982:	0037bc63          	p.bneimm	a5,3,1c01699a <mx25u_handle_pending_task+0x88>
1c016986:	8522                	mv	a0,s0
1c016988:	4422                	lw	s0,8(sp)
1c01698a:	40b2                	lw	ra,12(sp)
1c01698c:	4492                	lw	s1,4(sp)
1c01698e:	4902                	lw	s2,0(sp)
1c016990:	5310                	lw	a2,32(a4)
1c016992:	4f4c                	lw	a1,28(a4)
1c016994:	86ba                	mv	a3,a4
1c016996:	0141                	addi	sp,sp,16
1c016998:	a8f9                	j	1c016a76 <mx25u_reg_set_async>
1c01699a:	0047bc63          	p.bneimm	a5,4,1c0169b2 <mx25u_handle_pending_task+0xa0>
1c01699e:	8522                	mv	a0,s0
1c0169a0:	4422                	lw	s0,8(sp)
1c0169a2:	40b2                	lw	ra,12(sp)
1c0169a4:	4492                	lw	s1,4(sp)
1c0169a6:	4902                	lw	s2,0(sp)
1c0169a8:	5310                	lw	a2,32(a4)
1c0169aa:	4f4c                	lw	a1,28(a4)
1c0169ac:	86ba                	mv	a3,a4
1c0169ae:	0141                	addi	sp,sp,16
1c0169b0:	a0b9                	j	1c0169fe <mx25u_reg_get_async>
1c0169b2:	0057bd63          	p.bneimm	a5,5,1c0169cc <mx25u_handle_pending_task+0xba>
1c0169b6:	8522                	mv	a0,s0
1c0169b8:	4422                	lw	s0,8(sp)
1c0169ba:	40b2                	lw	ra,12(sp)
1c0169bc:	4492                	lw	s1,4(sp)
1c0169be:	4902                	lw	s2,0(sp)
1c0169c0:	5354                	lw	a3,36(a4)
1c0169c2:	5310                	lw	a2,32(a4)
1c0169c4:	4f4c                	lw	a1,28(a4)
1c0169c6:	0141                	addi	sp,sp,16
1c0169c8:	d55ff06f          	j	1c01671c <mx25u_read_async>
1c0169cc:	0267b363          	p.bneimm	a5,6,1c0169f2 <mx25u_handle_pending_task+0xe0>
1c0169d0:	883a                	mv	a6,a4
1c0169d2:	8522                	mv	a0,s0
1c0169d4:	4422                	lw	s0,8(sp)
1c0169d6:	575c                	lw	a5,44(a4)
1c0169d8:	40b2                	lw	ra,12(sp)
1c0169da:	4492                	lw	s1,4(sp)
1c0169dc:	4902                	lw	s2,0(sp)
1c0169de:	5718                	lw	a4,40(a4)
1c0169e0:	02482683          	lw	a3,36(a6)
1c0169e4:	02082603          	lw	a2,32(a6)
1c0169e8:	01c82583          	lw	a1,28(a6)
1c0169ec:	0141                	addi	sp,sp,16
1c0169ee:	a61ff06f          	j	1c01644e <mx25u_read_2d_async>
1c0169f2:	40b2                	lw	ra,12(sp)
1c0169f4:	4422                	lw	s0,8(sp)
1c0169f6:	4492                	lw	s1,4(sp)
1c0169f8:	4902                	lw	s2,0(sp)
1c0169fa:	0141                	addi	sp,sp,16
1c0169fc:	8082                	ret

1c0169fe <mx25u_reg_get_async>:
1c0169fe:	1141                	addi	sp,sp,-16
1c016a00:	c422                	sw	s0,8(sp)
1c016a02:	842a                	mv	s0,a0
1c016a04:	c04a                	sw	s2,0(sp)
1c016a06:	00842903          	lw	s2,8(s0)
1c016a0a:	8536                	mv	a0,a3
1c016a0c:	c226                	sw	s1,4(sp)
1c016a0e:	8732                	mv	a4,a2
1c016a10:	84b2                	mv	s1,a2
1c016a12:	86ae                	mv	a3,a1
1c016a14:	4881                	li	a7,0
1c016a16:	85aa                	mv	a1,a0
1c016a18:	4801                	li	a6,0
1c016a1a:	4781                	li	a5,0
1c016a1c:	4611                	li	a2,4
1c016a1e:	854a                	mv	a0,s2
1c016a20:	c606                	sw	ra,12(sp)
1c016a22:	951ff0ef          	jal	ra,1c016372 <mx25u_stall_task>
1c016a26:	ed01                	bnez	a0,1c016a3e <mx25u_reg_get_async+0x40>
1c016a28:	00c95783          	lhu	a5,12(s2)
1c016a2c:	8522                	mv	a0,s0
1c016a2e:	4422                	lw	s0,8(sp)
1c016a30:	00f49023          	sh	a5,0(s1)
1c016a34:	40b2                	lw	ra,12(sp)
1c016a36:	4492                	lw	s1,4(sp)
1c016a38:	4902                	lw	s2,0(sp)
1c016a3a:	0141                	addi	sp,sp,16
1c016a3c:	bdd9                	j	1c016912 <mx25u_handle_pending_task>
1c016a3e:	40b2                	lw	ra,12(sp)
1c016a40:	4422                	lw	s0,8(sp)
1c016a42:	4492                	lw	s1,4(sp)
1c016a44:	4902                	lw	s2,0(sp)
1c016a46:	0141                	addi	sp,sp,16
1c016a48:	8082                	ret

1c016a4a <mx25u_reg_get>:
1c016a4a:	7119                	addi	sp,sp,-128
1c016a4c:	dca2                	sw	s0,120(sp)
1c016a4e:	842a                	mv	s0,a0
1c016a50:	0828                	addi	a0,sp,24
1c016a52:	de86                	sw	ra,124(sp)
1c016a54:	c62e                	sw	a1,12(sp)
1c016a56:	c432                	sw	a2,8(sp)
1c016a58:	482030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016a5c:	4622                	lw	a2,8(sp)
1c016a5e:	45b2                	lw	a1,12(sp)
1c016a60:	86aa                	mv	a3,a0
1c016a62:	8522                	mv	a0,s0
1c016a64:	3f69                	jal	1c0169fe <mx25u_reg_get_async>
1c016a66:	0828                	addi	a0,sp,24
1c016a68:	4b4030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016a6c:	50f6                	lw	ra,124(sp)
1c016a6e:	5466                	lw	s0,120(sp)
1c016a70:	4501                	li	a0,0
1c016a72:	6109                	addi	sp,sp,128
1c016a74:	8082                	ret

1c016a76 <mx25u_reg_set_async>:
1c016a76:	1141                	addi	sp,sp,-16
1c016a78:	c422                	sw	s0,8(sp)
1c016a7a:	842a                	mv	s0,a0
1c016a7c:	c226                	sw	s1,4(sp)
1c016a7e:	4404                	lw	s1,8(s0)
1c016a80:	8536                	mv	a0,a3
1c016a82:	c04a                	sw	s2,0(sp)
1c016a84:	8732                	mv	a4,a2
1c016a86:	8932                	mv	s2,a2
1c016a88:	86ae                	mv	a3,a1
1c016a8a:	4881                	li	a7,0
1c016a8c:	85aa                	mv	a1,a0
1c016a8e:	4801                	li	a6,0
1c016a90:	4781                	li	a5,0
1c016a92:	460d                	li	a2,3
1c016a94:	8526                	mv	a0,s1
1c016a96:	c606                	sw	ra,12(sp)
1c016a98:	8dbff0ef          	jal	ra,1c016372 <mx25u_stall_task>
1c016a9c:	ed01                	bnez	a0,1c016ab4 <mx25u_reg_set_async+0x3e>
1c016a9e:	00095783          	lhu	a5,0(s2)
1c016aa2:	8522                	mv	a0,s0
1c016aa4:	4422                	lw	s0,8(sp)
1c016aa6:	00f49623          	sh	a5,12(s1)
1c016aaa:	40b2                	lw	ra,12(sp)
1c016aac:	4492                	lw	s1,4(sp)
1c016aae:	4902                	lw	s2,0(sp)
1c016ab0:	0141                	addi	sp,sp,16
1c016ab2:	b585                	j	1c016912 <mx25u_handle_pending_task>
1c016ab4:	40b2                	lw	ra,12(sp)
1c016ab6:	4422                	lw	s0,8(sp)
1c016ab8:	4492                	lw	s1,4(sp)
1c016aba:	4902                	lw	s2,0(sp)
1c016abc:	0141                	addi	sp,sp,16
1c016abe:	8082                	ret

1c016ac0 <mx25u_reg_set>:
1c016ac0:	7119                	addi	sp,sp,-128
1c016ac2:	dca2                	sw	s0,120(sp)
1c016ac4:	842a                	mv	s0,a0
1c016ac6:	0828                	addi	a0,sp,24
1c016ac8:	de86                	sw	ra,124(sp)
1c016aca:	c62e                	sw	a1,12(sp)
1c016acc:	c432                	sw	a2,8(sp)
1c016ace:	40c030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016ad2:	4622                	lw	a2,8(sp)
1c016ad4:	45b2                	lw	a1,12(sp)
1c016ad6:	86aa                	mv	a3,a0
1c016ad8:	8522                	mv	a0,s0
1c016ada:	3f71                	jal	1c016a76 <mx25u_reg_set_async>
1c016adc:	0828                	addi	a0,sp,24
1c016ade:	43e030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016ae2:	50f6                	lw	ra,124(sp)
1c016ae4:	5466                	lw	s0,120(sp)
1c016ae6:	4501                	li	a0,0
1c016ae8:	6109                	addi	sp,sp,128
1c016aea:	8082                	ret

1c016aec <mx25u_program>:
1c016aec:	7119                	addi	sp,sp,-128
1c016aee:	dca2                	sw	s0,120(sp)
1c016af0:	842a                	mv	s0,a0
1c016af2:	0828                	addi	a0,sp,24
1c016af4:	de86                	sw	ra,124(sp)
1c016af6:	c62e                	sw	a1,12(sp)
1c016af8:	c432                	sw	a2,8(sp)
1c016afa:	c236                	sw	a3,4(sp)
1c016afc:	3de030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016b00:	4692                	lw	a3,4(sp)
1c016b02:	4622                	lw	a2,8(sp)
1c016b04:	45b2                	lw	a1,12(sp)
1c016b06:	872a                	mv	a4,a0
1c016b08:	8522                	mv	a0,s0
1c016b0a:	fb5f90ef          	jal	ra,1c010abe <mx25u_program_async>
1c016b0e:	0828                	addi	a0,sp,24
1c016b10:	40c030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016b14:	50f6                	lw	ra,124(sp)
1c016b16:	5466                	lw	s0,120(sp)
1c016b18:	4501                	li	a0,0
1c016b1a:	6109                	addi	sp,sp,128
1c016b1c:	8082                	ret

1c016b1e <mx25u_copy_async>:
1c016b1e:	1141                	addi	sp,sp,-16
1c016b20:	c606                	sw	ra,12(sp)
1c016b22:	eb01                	bnez	a4,1c016b32 <mx25u_copy_async+0x14>
1c016b24:	873e                	mv	a4,a5
1c016b26:	f99f90ef          	jal	ra,1c010abe <mx25u_program_async>
1c016b2a:	40b2                	lw	ra,12(sp)
1c016b2c:	4501                	li	a0,0
1c016b2e:	0141                	addi	sp,sp,16
1c016b30:	8082                	ret
1c016b32:	873e                	mv	a4,a5
1c016b34:	be9ff0ef          	jal	ra,1c01671c <mx25u_read_async>
1c016b38:	bfcd                	j	1c016b2a <mx25u_copy_async+0xc>

1c016b3a <mx25u_copy>:
1c016b3a:	7119                	addi	sp,sp,-128
1c016b3c:	dca2                	sw	s0,120(sp)
1c016b3e:	842a                	mv	s0,a0
1c016b40:	0828                	addi	a0,sp,24
1c016b42:	de86                	sw	ra,124(sp)
1c016b44:	c62e                	sw	a1,12(sp)
1c016b46:	c432                	sw	a2,8(sp)
1c016b48:	c236                	sw	a3,4(sp)
1c016b4a:	c03a                	sw	a4,0(sp)
1c016b4c:	38e030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016b50:	4702                	lw	a4,0(sp)
1c016b52:	4692                	lw	a3,4(sp)
1c016b54:	4622                	lw	a2,8(sp)
1c016b56:	45b2                	lw	a1,12(sp)
1c016b58:	8522                	mv	a0,s0
1c016b5a:	083c                	addi	a5,sp,24
1c016b5c:	37c9                	jal	1c016b1e <mx25u_copy_async>
1c016b5e:	547d                	li	s0,-1
1c016b60:	e509                	bnez	a0,1c016b6a <mx25u_copy+0x30>
1c016b62:	842a                	mv	s0,a0
1c016b64:	0828                	addi	a0,sp,24
1c016b66:	3b6030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016b6a:	8522                	mv	a0,s0
1c016b6c:	50f6                	lw	ra,124(sp)
1c016b6e:	5466                	lw	s0,120(sp)
1c016b70:	6109                	addi	sp,sp,128
1c016b72:	8082                	ret

1c016b74 <mx25u_get_status>:
1c016b74:	7175                	addi	sp,sp,-144
1c016b76:	4601                	li	a2,0
1c016b78:	4589                	li	a1,2
1c016b7a:	c706                	sw	ra,140(sp)
1c016b7c:	c522                	sw	s0,136(sp)
1c016b7e:	c326                	sw	s1,132(sp)
1c016b80:	842a                	mv	s0,a0
1c016b82:	0ec010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c016b86:	000407b7          	lui	a5,0x40
1c016b8a:	5fa78793          	addi	a5,a5,1530 # 405fa <__l1_heapsram_size+0x2066a>
1c016b8e:	ce3e                	sw	a5,28(sp)
1c016b90:	4791                	li	a5,4
1c016b92:	84aa                	mv	s1,a0
1c016b94:	d03e                	sw	a5,32(sp)
1c016b96:	1028                	addi	a0,sp,40
1c016b98:	04200793          	li	a5,66
1c016b9c:	d23e                	sw	a5,36(sp)
1c016b9e:	33c030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016ba2:	4785                	li	a5,1
1c016ba4:	c03e                	sw	a5,0(sp)
1c016ba6:	88aa                	mv	a7,a0
1c016ba8:	01c10813          	addi	a6,sp,28
1c016bac:	0830                	addi	a2,sp,24
1c016bae:	4705                	li	a4,1
1c016bb0:	4685                	li	a3,1
1c016bb2:	4581                	li	a1,0
1c016bb4:	8522                	mv	a0,s0
1c016bb6:	7a9000ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016bba:	1028                	addi	a0,sp,40
1c016bbc:	360030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016bc0:	8626                	mv	a2,s1
1c016bc2:	8522                	mv	a0,s0
1c016bc4:	4589                	li	a1,2
1c016bc6:	0a8010ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c016bca:	40ba                	lw	ra,140(sp)
1c016bcc:	442a                	lw	s0,136(sp)
1c016bce:	4562                	lw	a0,24(sp)
1c016bd0:	449a                	lw	s1,132(sp)
1c016bd2:	6149                	addi	sp,sp,144
1c016bd4:	8082                	ret

1c016bd6 <mx25u_check_erase>:
1c016bd6:	1101                	addi	sp,sp,-32
1c016bd8:	cc22                	sw	s0,24(sp)
1c016bda:	4500                	lw	s0,8(a0)
1c016bdc:	862a                	mv	a2,a0
1c016bde:	c632                	sw	a2,12(sp)
1c016be0:	8522                	mv	a0,s0
1c016be2:	ce06                	sw	ra,28(sp)
1c016be4:	3f41                	jal	1c016b74 <mx25u_get_status>
1c016be6:	fc153533          	p.bclr	a0,a0,30,1
1c016bea:	4632                	lw	a2,12(sp)
1c016bec:	c10d                	beqz	a0,1c016c0e <mx25u_check_erase+0x38>
1c016bee:	1c0175b7          	lui	a1,0x1c017
1c016bf2:	bd658593          	addi	a1,a1,-1066 # 1c016bd6 <mx25u_check_erase>
1c016bf6:	02840513          	addi	a0,s0,40
1c016bfa:	2f6030ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c016bfe:	4462                	lw	s0,24(sp)
1c016c00:	40f2                	lw	ra,28(sp)
1c016c02:	6585                	lui	a1,0x1
1c016c04:	38858593          	addi	a1,a1,904 # 1388 <__STACK_IRQ_SIZE+0xb88>
1c016c08:	6105                	addi	sp,sp,32
1c016c0a:	31a0306f          	j	1c019f24 <__pi_os_evt_push_delayed_us>
1c016c0e:	4462                	lw	s0,24(sp)
1c016c10:	40f2                	lw	ra,28(sp)
1c016c12:	8532                	mv	a0,a2
1c016c14:	6105                	addi	sp,sp,32
1c016c16:	cfdff06f          	j	1c016912 <mx25u_handle_pending_task>

1c016c1a <mx25u_check_program>:
1c016c1a:	1101                	addi	sp,sp,-32
1c016c1c:	cc22                	sw	s0,24(sp)
1c016c1e:	4500                	lw	s0,8(a0)
1c016c20:	862a                	mv	a2,a0
1c016c22:	c632                	sw	a2,12(sp)
1c016c24:	8522                	mv	a0,s0
1c016c26:	ce06                	sw	ra,28(sp)
1c016c28:	37b1                	jal	1c016b74 <mx25u_get_status>
1c016c2a:	fc153533          	p.bclr	a0,a0,30,1
1c016c2e:	4632                	lw	a2,12(sp)
1c016c30:	c105                	beqz	a0,1c016c50 <mx25u_check_program+0x36>
1c016c32:	1c0175b7          	lui	a1,0x1c017
1c016c36:	c1a58593          	addi	a1,a1,-998 # 1c016c1a <mx25u_check_program>
1c016c3a:	02840513          	addi	a0,s0,40
1c016c3e:	2b2030ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c016c42:	4462                	lw	s0,24(sp)
1c016c44:	40f2                	lw	ra,28(sp)
1c016c46:	09600593          	li	a1,150
1c016c4a:	6105                	addi	sp,sp,32
1c016c4c:	2d80306f          	j	1c019f24 <__pi_os_evt_push_delayed_us>
1c016c50:	4462                	lw	s0,24(sp)
1c016c52:	40f2                	lw	ra,28(sp)
1c016c54:	8532                	mv	a0,a2
1c016c56:	6105                	addi	sp,sp,32
1c016c58:	dd5f906f          	j	1c010a2c <mx25u_program_resume>

1c016c5c <pi_mx25u51245g_conf_init>:
1c016c5c:	1c01c7b7          	lui	a5,0x1c01c
1c016c60:	4c478793          	addi	a5,a5,1220 # 1c01c4c4 <mx25u51245g_api>
1c016c64:	c11c                	sw	a5,0(a0)
1c016c66:	1c01d7b7          	lui	a5,0x1c01d
1c016c6a:	be078793          	addi	a5,a5,-1056 # 1c01cbe0 <mx25u_data>
1c016c6e:	d51c                	sw	a5,40(a0)
1c016c70:	0bebc7b7          	lui	a5,0xbebc
1c016c74:	20078793          	addi	a5,a5,512 # bebc200 <__l1_heapsram_size+0xbe9c270>
1c016c78:	dd5c                	sw	a5,60(a0)
1c016c7a:	6785                	lui	a5,0x1
1c016c7c:	1141                	addi	sp,sp,-16
1c016c7e:	02052623          	sw	zero,44(a0)
1c016c82:	02052823          	sw	zero,48(a0)
1c016c86:	04f51223          	sh	a5,68(a0)
1c016c8a:	04052023          	sw	zero,64(a0)
1c016c8e:	c422                	sw	s0,8(sp)
1c016c90:	c606                	sw	ra,12(sp)
1c016c92:	842a                	mv	s0,a0
1c016c94:	010010ef          	jal	ra,1c017ca4 <__flash_conf_init>
1c016c98:	40b2                	lw	ra,12(sp)
1c016c9a:	02042a23          	sw	zero,52(s0)
1c016c9e:	4422                	lw	s0,8(sp)
1c016ca0:	0141                	addi	sp,sp,16
1c016ca2:	8082                	ret

1c016ca4 <__powersupplygpio_close>:
1c016ca4:	4518                	lw	a4,8(a0)
1c016ca6:	00074783          	lbu	a5,0(a4) # 5800000 <__l1_heapsram_size+0x57e0070>
1c016caa:	cb85                	beqz	a5,1c016cda <__powersupplygpio_close+0x36>
1c016cac:	1141                	addi	sp,sp,-16
1c016cae:	17fd                	addi	a5,a5,-1
1c016cb0:	c422                	sw	s0,8(sp)
1c016cb2:	c606                	sw	ra,12(sp)
1c016cb4:	0ff7f793          	andi	a5,a5,255
1c016cb8:	4140                	lw	s0,4(a0)
1c016cba:	00f70023          	sb	a5,0(a4)
1c016cbe:	eb89                	bnez	a5,1c016cd0 <__powersupplygpio_close+0x2c>
1c016cc0:	00044503          	lbu	a0,0(s0)
1c016cc4:	4581                	li	a1,0
1c016cc6:	278d                	jal	1c017428 <pi_gpio_pin_write>
1c016cc8:	00044503          	lbu	a0,0(s0)
1c016ccc:	4581                	li	a1,0
1c016cce:	2385                	jal	1c01722e <pi_gpio_pin_configure>
1c016cd0:	40b2                	lw	ra,12(sp)
1c016cd2:	4422                	lw	s0,8(sp)
1c016cd4:	4501                	li	a0,0
1c016cd6:	0141                	addi	sp,sp,16
1c016cd8:	8082                	ret
1c016cda:	4501                	li	a0,0
1c016cdc:	8082                	ret

1c016cde <__powersupplygpio_open>:
1c016cde:	1141                	addi	sp,sp,-16
1c016ce0:	c422                	sw	s0,8(sp)
1c016ce2:	4500                	lw	s0,8(a0)
1c016ce4:	c606                	sw	ra,12(sp)
1c016ce6:	c226                	sw	s1,4(sp)
1c016ce8:	00044783          	lbu	a5,0(s0)
1c016cec:	cb91                	beqz	a5,1c016d00 <__powersupplygpio_open+0x22>
1c016cee:	0785                	addi	a5,a5,1
1c016cf0:	00f40023          	sb	a5,0(s0)
1c016cf4:	40b2                	lw	ra,12(sp)
1c016cf6:	4422                	lw	s0,8(sp)
1c016cf8:	4492                	lw	s1,4(sp)
1c016cfa:	4501                	li	a0,0
1c016cfc:	0141                	addi	sp,sp,16
1c016cfe:	8082                	ret
1c016d00:	4144                	lw	s1,4(a0)
1c016d02:	4585                	li	a1,1
1c016d04:	0004c503          	lbu	a0,0(s1)
1c016d08:	02b000ef          	jal	ra,1c017532 <pi_pad_function_set>
1c016d0c:	0004c503          	lbu	a0,0(s1)
1c016d10:	04000593          	li	a1,64
1c016d14:	2b29                	jal	1c01722e <pi_gpio_pin_configure>
1c016d16:	0004c503          	lbu	a0,0(s1)
1c016d1a:	4585                	li	a1,1
1c016d1c:	2731                	jal	1c017428 <pi_gpio_pin_write>
1c016d1e:	00044783          	lbu	a5,0(s0)
1c016d22:	b7f1                	j	1c016cee <__powersupplygpio_open+0x10>

1c016d24 <aps256xxn_alloc>:
1c016d24:	4508                	lw	a0,8(a0)
1c016d26:	1101                	addi	sp,sp,-32
1c016d28:	cc22                	sw	s0,24(sp)
1c016d2a:	0531                	addi	a0,a0,12
1c016d2c:	842e                	mv	s0,a1
1c016d2e:	85b2                	mv	a1,a2
1c016d30:	0070                	addi	a2,sp,12
1c016d32:	ce06                	sw	ra,28(sp)
1c016d34:	7d1000ef          	jal	ra,1c017d04 <extern_alloc>
1c016d38:	47b2                	lw	a5,12(sp)
1c016d3a:	40f2                	lw	ra,28(sp)
1c016d3c:	c01c                	sw	a5,0(s0)
1c016d3e:	4462                	lw	s0,24(sp)
1c016d40:	6105                	addi	sp,sp,32
1c016d42:	8082                	ret

1c016d44 <aps256xxn_free>:
1c016d44:	4508                	lw	a0,8(a0)
1c016d46:	87b2                	mv	a5,a2
1c016d48:	862e                	mv	a2,a1
1c016d4a:	0531                	addi	a0,a0,12
1c016d4c:	85be                	mv	a1,a5
1c016d4e:	0140106f          	j	1c017d62 <extern_free>

1c016d52 <aps256xxn_close>:
1c016d52:	1141                	addi	sp,sp,-16
1c016d54:	c422                	sw	s0,8(sp)
1c016d56:	4500                	lw	s0,8(a0)
1c016d58:	c606                	sw	ra,12(sp)
1c016d5a:	01044783          	lbu	a5,16(s0)
1c016d5e:	c38d                	beqz	a5,1c016d80 <aps256xxn_close+0x2e>
1c016d60:	17fd                	addi	a5,a5,-1
1c016d62:	0ff7f793          	andi	a5,a5,255
1c016d66:	00f40823          	sb	a5,16(s0)
1c016d6a:	eb99                	bnez	a5,1c016d80 <aps256xxn_close+0x2e>
1c016d6c:	8522                	mv	a0,s0
1c016d6e:	44d000ef          	jal	ra,1c0179ba <pi_octospi_close>
1c016d72:	00c40513          	addi	a0,s0,12
1c016d76:	4422                	lw	s0,8(sp)
1c016d78:	40b2                	lw	ra,12(sp)
1c016d7a:	0141                	addi	sp,sp,16
1c016d7c:	7810006f          	j	1c017cfc <extern_alloc_deinit>
1c016d80:	40b2                	lw	ra,12(sp)
1c016d82:	4422                	lw	s0,8(sp)
1c016d84:	0141                	addi	sp,sp,16
1c016d86:	8082                	ret

1c016d88 <aps256xxn_ioctl>:
1c016d88:	0015ac63          	p.beqimm	a1,1,1c016da0 <aps256xxn_ioctl+0x18>
1c016d8c:	c589                	beqz	a1,1c016d96 <aps256xxn_ioctl+0xe>
1c016d8e:	0025ad63          	p.beqimm	a1,2,1c016da8 <aps256xxn_ioctl+0x20>
1c016d92:	557d                	li	a0,-1
1c016d94:	8082                	ret
1c016d96:	4158                	lw	a4,4(a0)
1c016d98:	8532                	mv	a0,a2
1c016d9a:	5f58                	lw	a4,60(a4)
1c016d9c:	c218                	sw	a4,0(a2)
1c016d9e:	8082                	ret
1c016da0:	4589                	li	a1,2
1c016da2:	4508                	lw	a0,8(a0)
1c016da4:	6cb0006f          	j	1c017c6e <pi_octospi_ioctl>
1c016da8:	458d                	li	a1,3
1c016daa:	bfe5                	j	1c016da2 <aps256xxn_ioctl+0x1a>

1c016dac <aps256xxn_copy_2d_async>:
1c016dac:	1101                	addi	sp,sp,-32
1c016dae:	ce06                	sw	ra,28(sp)
1c016db0:	4508                	lw	a0,8(a0)
1c016db2:	00080a63          	beqz	a6,1c016dc6 <aps256xxn_copy_2d_async+0x1a>
1c016db6:	4805                	li	a6,1
1c016db8:	c042                	sw	a6,0(sp)
1c016dba:	4801                	li	a6,0
1c016dbc:	5a3000ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016dc0:	40f2                	lw	ra,28(sp)
1c016dc2:	6105                	addi	sp,sp,32
1c016dc4:	8082                	ret
1c016dc6:	c002                	sw	zero,0(sp)
1c016dc8:	bfd5                	j	1c016dbc <aps256xxn_copy_2d_async+0x10>

1c016dca <aps256xxn_open>:
1c016dca:	7131                	addi	sp,sp,-192
1c016dcc:	db26                	sw	s1,180(sp)
1c016dce:	4144                	lw	s1,4(a0)
1c016dd0:	df06                	sw	ra,188(sp)
1c016dd2:	dd22                	sw	s0,184(sp)
1c016dd4:	549c                	lw	a5,40(s1)
1c016dd6:	d94a                	sw	s2,176(sp)
1c016dd8:	d74e                	sw	s3,172(sp)
1c016dda:	d552                	sw	s4,168(sp)
1c016ddc:	c391                	beqz	a5,1c016de0 <aps256xxn_open+0x16>
1c016dde:	c51c                	sw	a5,8(a0)
1c016de0:	409c                	lw	a5,0(s1)
1c016de2:	c391                	beqz	a5,1c016de6 <aps256xxn_open+0x1c>
1c016de4:	c11c                	sw	a5,0(a0)
1c016de6:	411c                	lw	a5,0(a0)
1c016de8:	4500                	lw	s0,8(a0)
1c016dea:	c09c                	sw	a5,0(s1)
1c016dec:	d480                	sw	s0,40(s1)
1c016dee:	01044783          	lbu	a5,16(s0)
1c016df2:	cf91                	beqz	a5,1c016e0e <aps256xxn_open+0x44>
1c016df4:	0785                	addi	a5,a5,1
1c016df6:	00f40823          	sb	a5,16(s0)
1c016dfa:	4901                	li	s2,0
1c016dfc:	50fa                	lw	ra,188(sp)
1c016dfe:	546a                	lw	s0,184(sp)
1c016e00:	854a                	mv	a0,s2
1c016e02:	54da                	lw	s1,180(sp)
1c016e04:	594a                	lw	s2,176(sp)
1c016e06:	59ba                	lw	s3,172(sp)
1c016e08:	5a2a                	lw	s4,168(sp)
1c016e0a:	6129                	addi	sp,sp,192
1c016e0c:	8082                	ret
1c016e0e:	40fc                	lw	a5,68(s1)
1c016e10:	5cd0                	lw	a2,60(s1)
1c016e12:	4581                	li	a1,0
1c016e14:	c399                	beqz	a5,1c016e1a <aps256xxn_open+0x50>
1c016e16:	1671                	addi	a2,a2,-4
1c016e18:	4591                	li	a1,4
1c016e1a:	00c40993          	addi	s3,s0,12
1c016e1e:	854e                	mv	a0,s3
1c016e20:	68b000ef          	jal	ra,1c017caa <extern_alloc_init>
1c016e24:	892a                	mv	s2,a0
1c016e26:	10051463          	bnez	a0,1c016f2e <aps256xxn_open+0x164>
1c016e2a:	1068                	addi	a0,sp,44
1c016e2c:	0e5000ef          	jal	ra,1c017710 <pi_octospi_conf_init>
1c016e30:	7d000793          	li	a5,2000
1c016e34:	de3e                	sw	a5,60(sp)
1c016e36:	54dc                	lw	a5,44(s1)
1c016e38:	02f10623          	sb	a5,44(sp)
1c016e3c:	589c                	lw	a5,48(s1)
1c016e3e:	d83e                	sw	a5,48(sp)
1c016e40:	4785                	li	a5,1
1c016e42:	da3e                	sw	a5,52(sp)
1c016e44:	58dc                	lw	a5,52(s1)
1c016e46:	02f106a3          	sb	a5,45(sp)
1c016e4a:	00448793          	addi	a5,s1,4
1c016e4e:	c0be                	sw	a5,64(sp)
1c016e50:	44bc                	lw	a5,72(s1)
1c016e52:	c2be                	sw	a5,68(sp)
1c016e54:	40bc                	lw	a5,64(s1)
1c016e56:	c391                	beqz	a5,1c016e5a <aps256xxn_open+0x90>
1c016e58:	dc3e                	sw	a5,56(sp)
1c016e5a:	8522                	mv	a0,s0
1c016e5c:	106c                	addi	a1,sp,44
1c016e5e:	afdfd0ef          	jal	ra,1c01495a <pi_open_from_conf>
1c016e62:	8522                	mv	a0,s0
1c016e64:	0c5000ef          	jal	ra,1c017728 <pi_octospi_open>
1c016e68:	ed55                	bnez	a0,1c016f24 <aps256xxn_open+0x15a>
1c016e6a:	4601                	li	a2,0
1c016e6c:	4589                	li	a1,2
1c016e6e:	8522                	mv	a0,s0
1c016e70:	5ff000ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c016e74:	000447b7          	lui	a5,0x44
1c016e78:	6985                	lui	s3,0x1
1c016e7a:	04198993          	addi	s3,s3,65 # 1041 <__STACK_IRQ_SIZE+0x841>
1c016e7e:	ca3e                	sw	a5,20(sp)
1c016e80:	00a8                	addi	a0,sp,72
1c016e82:	4795                	li	a5,5
1c016e84:	cc3e                	sw	a5,24(sp)
1c016e86:	ce4e                	sw	s3,28(sp)
1c016e88:	4a05                	li	s4,1
1c016e8a:	00011923          	sh	zero,18(sp)
1c016e8e:	04c030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016e92:	88aa                	mv	a7,a0
1c016e94:	01410813          	addi	a6,sp,20
1c016e98:	4705                	li	a4,1
1c016e9a:	4685                	li	a3,1
1c016e9c:	01210613          	addi	a2,sp,18
1c016ea0:	45a1                	li	a1,8
1c016ea2:	4785                	li	a5,1
1c016ea4:	8522                	mv	a0,s0
1c016ea6:	c052                	sw	s4,0(sp)
1c016ea8:	4b7000ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016eac:	00a8                	addi	a0,sp,72
1c016eae:	06e030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016eb2:	01215783          	lhu	a5,18(sp)
1c016eb6:	00a8                	addi	a0,sp,72
1c016eb8:	d44e                	sw	s3,40(sp)
1c016eba:	c037c7b3          	p.bset	a5,a5,0,3
1c016ebe:	00f11923          	sh	a5,18(sp)
1c016ec2:	0004c7b7          	lui	a5,0x4c
1c016ec6:	d03e                	sw	a5,32(sp)
1c016ec8:	d252                	sw	s4,36(sp)
1c016eca:	010030ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c016ece:	88aa                	mv	a7,a0
1c016ed0:	4785                	li	a5,1
1c016ed2:	02010813          	addi	a6,sp,32
1c016ed6:	4705                	li	a4,1
1c016ed8:	4685                	li	a3,1
1c016eda:	01210613          	addi	a2,sp,18
1c016ede:	45a1                	li	a1,8
1c016ee0:	8522                	mv	a0,s0
1c016ee2:	c002                	sw	zero,0(sp)
1c016ee4:	47b000ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016ee8:	00a8                	addi	a0,sp,72
1c016eea:	032030ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c016eee:	1c01c9b7          	lui	s3,0x1c01c
1c016ef2:	55898613          	addi	a2,s3,1368 # 1c01c558 <aps256xxn_default_op>
1c016ef6:	4581                	li	a1,0
1c016ef8:	8522                	mv	a0,s0
1c016efa:	575000ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c016efe:	58dc                	lw	a5,52(s1)
1c016f00:	c799                	beqz	a5,1c016f0e <aps256xxn_open+0x144>
1c016f02:	55898613          	addi	a2,s3,1368
1c016f06:	4585                	li	a1,1
1c016f08:	8522                	mv	a0,s0
1c016f0a:	565000ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c016f0e:	4605                	li	a2,1
1c016f10:	4589                	li	a1,2
1c016f12:	8522                	mv	a0,s0
1c016f14:	55b000ef          	jal	ra,1c017c6e <pi_octospi_ioctl>
1c016f18:	01044783          	lbu	a5,16(s0)
1c016f1c:	0785                	addi	a5,a5,1
1c016f1e:	00f40823          	sb	a5,16(s0)
1c016f22:	bde9                	j	1c016dfc <aps256xxn_open+0x32>
1c016f24:	854e                	mv	a0,s3
1c016f26:	5d7000ef          	jal	ra,1c017cfc <extern_alloc_deinit>
1c016f2a:	597d                	li	s2,-1
1c016f2c:	bdc1                	j	1c016dfc <aps256xxn_open+0x32>
1c016f2e:	5979                	li	s2,-2
1c016f30:	b5f1                	j	1c016dfc <aps256xxn_open+0x32>

1c016f32 <aps256xxn_copy_async>:
1c016f32:	1101                	addi	sp,sp,-32
1c016f34:	ce06                	sw	ra,28(sp)
1c016f36:	8336                	mv	t1,a3
1c016f38:	88be                	mv	a7,a5
1c016f3a:	4508                	lw	a0,8(a0)
1c016f3c:	cb19                	beqz	a4,1c016f52 <aps256xxn_copy_async+0x20>
1c016f3e:	4785                	li	a5,1
1c016f40:	c03e                	sw	a5,0(sp)
1c016f42:	4801                	li	a6,0
1c016f44:	879a                	mv	a5,t1
1c016f46:	871a                	mv	a4,t1
1c016f48:	417000ef          	jal	ra,1c017b5e <pi_octospi_copy_2d>
1c016f4c:	40f2                	lw	ra,28(sp)
1c016f4e:	6105                	addi	sp,sp,32
1c016f50:	8082                	ret
1c016f52:	c002                	sw	zero,0(sp)
1c016f54:	b7fd                	j	1c016f42 <aps256xxn_copy_async+0x10>

1c016f56 <pi_aps256xxn_conf_init>:
1c016f56:	1c01c7b7          	lui	a5,0x1c01c
1c016f5a:	53c78793          	addi	a5,a5,1340 # 1c01c53c <aps256xxn_api>
1c016f5e:	c11c                	sw	a5,0(a0)
1c016f60:	1c01e7b7          	lui	a5,0x1c01e
1c016f64:	8b878793          	addi	a5,a5,-1864 # 1c01d8b8 <aps256xxn_data>
1c016f68:	d51c                	sw	a5,40(a0)
1c016f6a:	4785                	li	a5,1
1c016f6c:	c17c                	sw	a5,68(a0)
1c016f6e:	d91c                	sw	a5,48(a0)
1c016f70:	0bebc7b7          	lui	a5,0xbebc
1c016f74:	02000737          	lui	a4,0x2000
1c016f78:	20078793          	addi	a5,a5,512 # bebc200 <__l1_heapsram_size+0xbe9c270>
1c016f7c:	02052a23          	sw	zero,52(a0)
1c016f80:	00051223          	sh	zero,4(a0)
1c016f84:	02052c23          	sw	zero,56(a0)
1c016f88:	dd58                	sw	a4,60(a0)
1c016f8a:	02052623          	sw	zero,44(a0)
1c016f8e:	c13c                	sw	a5,64(a0)
1c016f90:	c538                	sw	a4,72(a0)
1c016f92:	8082                	ret

1c016f94 <pi_open>:
1c016f94:	7abf67b7          	lui	a5,0x7abf6
1c016f98:	84978793          	addi	a5,a5,-1975 # 7abf5849 <__heapl2ram_size+0x5ea65849>
1c016f9c:	08f50363          	beq	a0,a5,1c017022 <pi_open+0x8e>
1c016fa0:	02a7e463          	bltu	a5,a0,1c016fc8 <pi_open+0x34>
1c016fa4:	085407b7          	lui	a5,0x8540
1c016fa8:	9fe78713          	addi	a4,a5,-1538 # 853f9fe <__l1_heapsram_size+0x851fa6e>
1c016fac:	00e56c63          	bltu	a0,a4,1c016fc4 <pi_open+0x30>
1c016fb0:	9ff78793          	addi	a5,a5,-1537
1c016fb4:	06a7f263          	bleu	a0,a5,1c017018 <pi_open+0x84>
1c016fb8:	499f27b7          	lui	a5,0x499f2
1c016fbc:	48e78793          	addi	a5,a5,1166 # 499f248e <__heapl2ram_size+0x2d86248e>
1c016fc0:	04f50463          	beq	a0,a5,1c017008 <pi_open+0x74>
1c016fc4:	557d                	li	a0,-1
1c016fc6:	8082                	ret
1c016fc8:	b16a57b7          	lui	a5,0xb16a5
1c016fcc:	f4578793          	addi	a5,a5,-187 # b16a4f45 <pulp__FC+0xb16a4f46>
1c016fd0:	04f50e63          	beq	a0,a5,1c01702c <pi_open+0x98>
1c016fd4:	00a7ed63          	bltu	a5,a0,1c016fee <pi_open+0x5a>
1c016fd8:	8cf2f7b7          	lui	a5,0x8cf2f
1c016fdc:	20c78793          	addi	a5,a5,524 # 8cf2f20c <pulp__FC+0x8cf2f20d>
1c016fe0:	fef512e3          	bne	a0,a5,1c016fc4 <pi_open+0x30>
1c016fe4:	1c01c7b7          	lui	a5,0x1c01c
1c016fe8:	39878793          	addi	a5,a5,920 # 1c01c398 <cam3v3_dev>
1c016fec:	a015                	j	1c017010 <pi_open+0x7c>
1c016fee:	0b7ea7b7          	lui	a5,0xb7ea
1c016ff2:	33278793          	addi	a5,a5,818 # b7ea332 <__l1_heapsram_size+0xb7ca3a2>
1c016ff6:	953e                	add	a0,a0,a5
1c016ff8:	4785                	li	a5,1
1c016ffa:	fca7e5e3          	bltu	a5,a0,1c016fc4 <pi_open+0x30>
1c016ffe:	1c01c7b7          	lui	a5,0x1c01c
1c017002:	38c78793          	addi	a5,a5,908 # 1c01c38c <aps256xxn_dev>
1c017006:	a029                	j	1c017010 <pi_open+0x7c>
1c017008:	1c01c7b7          	lui	a5,0x1c01c
1c01700c:	3c478793          	addi	a5,a5,964 # 1c01c3c4 <cluster_dev>
1c017010:	c19c                	sw	a5,0(a1)
1c017012:	e195                	bnez	a1,1c017036 <pi_open+0xa2>
1c017014:	4501                	li	a0,0
1c017016:	8082                	ret
1c017018:	1c01c7b7          	lui	a5,0x1c01c
1c01701c:	45878793          	addi	a5,a5,1112 # 1c01c458 <mx25u51245g_dev>
1c017020:	bfc5                	j	1c017010 <pi_open+0x7c>
1c017022:	1c01c7b7          	lui	a5,0x1c01c
1c017026:	40478793          	addi	a5,a5,1028 # 1c01c404 <mram_dev>
1c01702a:	b7dd                	j	1c017010 <pi_open+0x7c>
1c01702c:	1c01c7b7          	lui	a5,0x1c01c
1c017030:	46478793          	addi	a5,a5,1124 # 1c01c464 <periph3v3_dev>
1c017034:	bff1                	j	1c017010 <pi_open+0x7c>
1c017036:	4188                	lw	a0,0(a1)
1c017038:	411c                	lw	a5,0(a0)
1c01703a:	0007a303          	lw	t1,0(a5)
1c01703e:	8302                	jr	t1

1c017040 <pi_close>:
1c017040:	7abf67b7          	lui	a5,0x7abf6
1c017044:	84978793          	addi	a5,a5,-1975 # 7abf5849 <__heapl2ram_size+0x5ea65849>
1c017048:	06f50f63          	beq	a0,a5,1c0170c6 <pi_close+0x86>
1c01704c:	02a7e463          	bltu	a5,a0,1c017074 <pi_close+0x34>
1c017050:	085407b7          	lui	a5,0x8540
1c017054:	9fe78713          	addi	a4,a5,-1538 # 853f9fe <__l1_heapsram_size+0x851fa6e>
1c017058:	00e56c63          	bltu	a0,a4,1c017070 <pi_close+0x30>
1c01705c:	9ff78793          	addi	a5,a5,-1537
1c017060:	04a7fa63          	bleu	a0,a5,1c0170b4 <pi_close+0x74>
1c017064:	499f27b7          	lui	a5,0x499f2
1c017068:	48e78793          	addi	a5,a5,1166 # 499f248e <__heapl2ram_size+0x2d86248e>
1c01706c:	06f50763          	beq	a0,a5,1c0170da <pi_close+0x9a>
1c017070:	557d                	li	a0,-1
1c017072:	8082                	ret
1c017074:	b16a57b7          	lui	a5,0xb16a5
1c017078:	f4578793          	addi	a5,a5,-187 # b16a4f45 <pulp__FC+0xb16a4f46>
1c01707c:	04f50a63          	beq	a0,a5,1c0170d0 <pi_close+0x90>
1c017080:	00a7ed63          	bltu	a5,a0,1c01709a <pi_close+0x5a>
1c017084:	8cf2f7b7          	lui	a5,0x8cf2f
1c017088:	20c78793          	addi	a5,a5,524 # 8cf2f20c <pulp__FC+0x8cf2f20d>
1c01708c:	fef512e3          	bne	a0,a5,1c017070 <pi_close+0x30>
1c017090:	1c01c537          	lui	a0,0x1c01c
1c017094:	39850513          	addi	a0,a0,920 # 1c01c398 <cam3v3_dev>
1c017098:	a015                	j	1c0170bc <pi_close+0x7c>
1c01709a:	0b7ea7b7          	lui	a5,0xb7ea
1c01709e:	33278793          	addi	a5,a5,818 # b7ea332 <__l1_heapsram_size+0xb7ca3a2>
1c0170a2:	953e                	add	a0,a0,a5
1c0170a4:	4785                	li	a5,1
1c0170a6:	fca7e5e3          	bltu	a5,a0,1c017070 <pi_close+0x30>
1c0170aa:	1c01c537          	lui	a0,0x1c01c
1c0170ae:	38c50513          	addi	a0,a0,908 # 1c01c38c <aps256xxn_dev>
1c0170b2:	a029                	j	1c0170bc <pi_close+0x7c>
1c0170b4:	1c01c537          	lui	a0,0x1c01c
1c0170b8:	45850513          	addi	a0,a0,1112 # 1c01c458 <mx25u51245g_dev>
1c0170bc:	c505                	beqz	a0,1c0170e4 <pi_close+0xa4>
1c0170be:	411c                	lw	a5,0(a0)
1c0170c0:	0047a303          	lw	t1,4(a5)
1c0170c4:	8302                	jr	t1
1c0170c6:	1c01c537          	lui	a0,0x1c01c
1c0170ca:	40450513          	addi	a0,a0,1028 # 1c01c404 <mram_dev>
1c0170ce:	b7fd                	j	1c0170bc <pi_close+0x7c>
1c0170d0:	1c01c537          	lui	a0,0x1c01c
1c0170d4:	46450513          	addi	a0,a0,1124 # 1c01c464 <periph3v3_dev>
1c0170d8:	b7d5                	j	1c0170bc <pi_close+0x7c>
1c0170da:	1c01c537          	lui	a0,0x1c01c
1c0170de:	3c450513          	addi	a0,a0,964 # 1c01c3c4 <cluster_dev>
1c0170e2:	bfe9                	j	1c0170bc <pi_close+0x7c>
1c0170e4:	4501                	li	a0,0
1c0170e6:	8082                	ret

1c0170e8 <dt_fpv2_memory_ref_get>:
1c0170e8:	1c01c537          	lui	a0,0x1c01c
1c0170ec:	56450513          	addi	a0,a0,1380 # 1c01c564 <fpv2_memory_refs>
1c0170f0:	8082                	ret

1c0170f2 <dt_fpv2_memory_ref_size_get>:
1c0170f2:	4509                	li	a0,2
1c0170f4:	8082                	ret

1c0170f6 <__pi_gpio_handler>:
1c0170f6:	7179                	addi	sp,sp,-48
1c0170f8:	1c01d7b7          	lui	a5,0x1c01d
1c0170fc:	d226                	sw	s1,36(sp)
1c0170fe:	0044                	addi	s1,sp,4
1c017100:	d04a                	sw	s2,32(sp)
1c017102:	d606                	sw	ra,44(sp)
1c017104:	d2878913          	addi	s2,a5,-728 # 1c01cd28 <g_gpio_data>
1c017108:	d422                	sw	s0,40(sp)
1c01710a:	ce4e                	sw	s3,28(sp)
1c01710c:	cc52                	sw	s4,24(sp)
1c01710e:	ca56                	sw	s5,20(sp)
1c017110:	c85a                	sw	s6,16(sp)
1c017112:	c202                	sw	zero,4(sp)
1c017114:	c402                	sw	zero,8(sp)
1c017116:	c602                	sw	zero,12(sp)
1c017118:	86a6                	mv	a3,s1
1c01711a:	d2878793          	addi	a5,a5,-728
1c01711e:	02400713          	li	a4,36
1c017122:	1a1018b7          	lui	a7,0x1a101
1c017126:	0fc00813          	li	a6,252
1c01712a:	0047a58b          	p.lw	a1,4(a5!)
1c01712e:	c599                	beqz	a1,1c01713c <__pi_gpio_handler+0x46>
1c017130:	20e8f503          	p.lw	a0,a4(a7)
1c017134:	4590                	lw	a2,8(a1)
1c017136:	c1c8                	sw	a0,4(a1)
1c017138:	8e69                	and	a2,a2,a0
1c01713a:	c290                	sw	a2,0(a3)
1c01713c:	04870713          	addi	a4,a4,72 # 2000048 <__l1_heapsram_size+0x1fe00b8>
1c017140:	0691                	addi	a3,a3,4
1c017142:	ff0714e3          	bne	a4,a6,1c01712a <__pi_gpio_handler+0x34>
1c017146:	00c90993          	addi	s3,s2,12
1c01714a:	02000a13          	li	s4,32
1c01714e:	4a85                	li	s5,1
1c017150:	00492b0b          	p.lw	s6,4(s2!)
1c017154:	0044a40b          	p.lw	s0,4(s1!)
1c017158:	10041433          	p.fl1	s0,s0
1c01715c:	0ff47413          	andi	s0,s0,255
1c017160:	01441e63          	bne	s0,s4,1c01717c <__pi_gpio_handler+0x86>
1c017164:	ff3916e3          	bne	s2,s3,1c017150 <__pi_gpio_handler+0x5a>
1c017168:	50b2                	lw	ra,44(sp)
1c01716a:	5422                	lw	s0,40(sp)
1c01716c:	5492                	lw	s1,36(sp)
1c01716e:	5902                	lw	s2,32(sp)
1c017170:	49f2                	lw	s3,28(sp)
1c017172:	4a62                	lw	s4,24(sp)
1c017174:	4ad2                	lw	s5,20(sp)
1c017176:	4b42                	lw	s6,16(sp)
1c017178:	6145                	addi	sp,sp,48
1c01717a:	8082                	ret
1c01717c:	00440513          	addi	a0,s0,4
1c017180:	050a                	slli	a0,a0,0x2
1c017182:	20ab7503          	p.lw	a0,a0(s6)
1c017186:	c119                	beqz	a0,1c01718c <__pi_gpio_handler+0x96>
1c017188:	599020ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c01718c:	ffc4a783          	lw	a5,-4(s1)
1c017190:	008a9433          	sll	s0,s5,s0
1c017194:	fff44413          	not	s0,s0
1c017198:	8c7d                	and	s0,s0,a5
1c01719a:	fe84ae23          	sw	s0,-4(s1)
1c01719e:	bf6d                	j	1c017158 <__pi_gpio_handler+0x62>

1c0171a0 <__pi_gpio_deinit>:
1c0171a0:	1141                	addi	sp,sp,-16
1c0171a2:	c606                	sw	ra,12(sp)
1c0171a4:	c422                	sw	s0,8(sp)
1c0171a6:	c226                	sw	s1,4(sp)
1c0171a8:	300474f3          	csrrci	s1,mstatus,8
1c0171ac:	1c01d7b7          	lui	a5,0x1c01d
1c0171b0:	00555693          	srli	a3,a0,0x5
1c0171b4:	d2878613          	addi	a2,a5,-728 # 1c01cd28 <g_gpio_data>
1c0171b8:	00269413          	slli	s0,a3,0x2
1c0171bc:	20867603          	p.lw	a2,s0(a2)
1c0171c0:	c225                	beqz	a2,1c017220 <__pi_gpio_deinit+0x80>
1c0171c2:	4705                	li	a4,1
1c0171c4:	00a71533          	sll	a0,a4,a0
1c0171c8:	4218                	lw	a4,0(a2)
1c0171ca:	fff54513          	not	a0,a0
1c0171ce:	8d79                	and	a0,a0,a4
1c0171d0:	c208                	sw	a0,0(a2)
1c0171d2:	e539                	bnez	a0,1c017220 <__pi_gpio_deinit+0x80>
1c0171d4:	d2878793          	addi	a5,a5,-728
1c0171d8:	4398                	lw	a4,0(a5)
1c0171da:	4318                	lw	a4,0(a4)
1c0171dc:	eb19                	bnez	a4,1c0171f2 <__pi_gpio_deinit+0x52>
1c0171de:	43d8                	lw	a4,4(a5)
1c0171e0:	4318                	lw	a4,0(a4)
1c0171e2:	eb01                	bnez	a4,1c0171f2 <__pi_gpio_deinit+0x52>
1c0171e4:	4798                	lw	a4,8(a5)
1c0171e6:	4318                	lw	a4,0(a4)
1c0171e8:	e709                	bnez	a4,1c0171f2 <__pi_gpio_deinit+0x52>
1c0171ea:	1a109737          	lui	a4,0x1a109
1c0171ee:	65a1                	lui	a1,0x8
1c0171f0:	c70c                	sw	a1,8(a4)
1c0171f2:	4618                	lw	a4,8(a2)
1c0171f4:	04800613          	li	a2,72
1c0171f8:	02c686b3          	mul	a3,a3,a2
1c0171fc:	1a101637          	lui	a2,0x1a101
1c017200:	0691                	addi	a3,a3,4
1c017202:	20d67583          	p.lw	a1,a3(a2)
1c017206:	fff74713          	not	a4,a4
1c01720a:	8f6d                	and	a4,a4,a1
1c01720c:	00e666a3          	p.sw	a4,a3(a2)
1c017210:	943e                	add	s0,s0,a5
1c017212:	4008                	lw	a0,0(s0)
1c017214:	09000593          	li	a1,144
1c017218:	280030ef          	jal	ra,1c01a498 <pi_l2_priv_free>
1c01721c:	00042023          	sw	zero,0(s0)
1c017220:	30049073          	csrw	mstatus,s1
1c017224:	40b2                	lw	ra,12(sp)
1c017226:	4422                	lw	s0,8(sp)
1c017228:	4492                	lw	s1,4(sp)
1c01722a:	0141                	addi	sp,sp,16
1c01722c:	8082                	ret

1c01722e <pi_gpio_pin_configure>:
1c01722e:	1101                	addi	sp,sp,-32
1c017230:	00555793          	srli	a5,a0,0x5
1c017234:	ca26                	sw	s1,20(sp)
1c017236:	c84a                	sw	s2,16(sp)
1c017238:	c452                	sw	s4,8(sp)
1c01723a:	ce06                	sw	ra,28(sp)
1c01723c:	cc22                	sw	s0,24(sp)
1c01723e:	c64e                	sw	s3,12(sp)
1c017240:	c256                	sw	s5,4(sp)
1c017242:	c05a                	sw	s6,0(sp)
1c017244:	fa27b933          	p.bclr	s2,a5,29,2
1c017248:	84aa                	mv	s1,a0
1c01724a:	5a55                	li	s4,-11
1c01724c:	0cf91863          	bne	s2,a5,1c01731c <pi_gpio_pin_configure+0xee>
1c017250:	fa25b7b3          	p.bclr	a5,a1,29,2
1c017254:	852e                	mv	a0,a1
1c017256:	4a01                	li	s4,0
1c017258:	0037b563          	p.bneimm	a5,3,1c017262 <pi_gpio_pin_configure+0x34>
1c01725c:	c015b533          	p.bclr	a0,a1,0,1
1c017260:	4a05                	li	s4,1
1c017262:	1c01d6b7          	lui	a3,0x1c01d
1c017266:	d2868693          	addi	a3,a3,-728 # 1c01cd28 <g_gpio_data>
1c01726a:	00291713          	slli	a4,s2,0x2
1c01726e:	20e6f703          	p.lw	a4,a4(a3)
1c017272:	00455593          	srli	a1,a0,0x4
1c017276:	00255a93          	srli	s5,a0,0x2
1c01727a:	fa253b33          	p.bclr	s6,a0,29,2
1c01727e:	8119                	srli	a0,a0,0x6
1c017280:	0ff57413          	andi	s0,a0,255
1c017284:	4685                	li	a3,1
1c017286:	009696b3          	sll	a3,a3,s1
1c01728a:	fc1439b3          	p.bclr	s3,s0,30,1
1c01728e:	fc15b5b3          	p.bclr	a1,a1,30,1
1c017292:	fa2abab3          	p.bclr	s5,s5,29,2
1c017296:	4710                	lw	a2,8(a4)
1c017298:	475c                	lw	a5,12(a4)
1c01729a:	fff6c513          	not	a0,a3
1c01729e:	08098a63          	beqz	s3,1c017332 <pi_gpio_pin_configure+0x104>
1c0172a2:	8e69                	and	a2,a2,a0
1c0172a4:	c710                	sw	a2,8(a4)
1c0172a6:	8edd                	or	a3,a3,a5
1c0172a8:	c754                	sw	a3,12(a4)
1c0172aa:	8526                	mv	a0,s1
1c0172ac:	2eb9                	jal	1c01760a <pi_pad_schmitt_trigger_set>
1c0172ae:	85d6                	mv	a1,s5
1c0172b0:	8526                	mv	a0,s1
1c0172b2:	2cd1                	jal	1c017586 <pi_pad_drive_strength_set>
1c0172b4:	85da                	mv	a1,s6
1c0172b6:	8526                	mv	a0,s1
1c0172b8:	2e21                	jal	1c0175d0 <pi_pad_pull_set>
1c0172ba:	0ff4f613          	andi	a2,s1,255
1c0172be:	00565713          	srli	a4,a2,0x5
1c0172c2:	00371793          	slli	a5,a4,0x3
1c0172c6:	97ba                	add	a5,a5,a4
1c0172c8:	078e                	slli	a5,a5,0x3
1c0172ca:	1a1015b7          	lui	a1,0x1a101
1c0172ce:	20f5f503          	p.lw	a0,a5(a1)
1c0172d2:	f4563633          	p.bclr	a2,a2,26,5
1c0172d6:	4705                	li	a4,1
1c0172d8:	00c716b3          	sll	a3,a4,a2
1c0172dc:	fff6c693          	not	a3,a3
1c0172e0:	8ee9                	and	a3,a3,a0
1c0172e2:	00c999b3          	sll	s3,s3,a2
1c0172e6:	00d9e9b3          	or	s3,s3,a3
1c0172ea:	0135e7a3          	p.sw	s3,a5(a1)
1c0172ee:	04800793          	li	a5,72
1c0172f2:	02f90933          	mul	s2,s2,a5
1c0172f6:	0911                	addi	s2,s2,4
1c0172f8:	2125f503          	p.lw	a0,s2(a1)
1c0172fc:	f454b7b3          	p.bclr	a5,s1,26,5
1c017300:	00f71733          	sll	a4,a4,a5
1c017304:	fff74713          	not	a4,a4
1c017308:	8f69                	and	a4,a4,a0
1c01730a:	fff44513          	not	a0,s0
1c01730e:	fc153533          	p.bclr	a0,a0,30,1
1c017312:	00f51533          	sll	a0,a0,a5
1c017316:	8d59                	or	a0,a0,a4
1c017318:	00a5e923          	p.sw	a0,s2(a1)
1c01731c:	40f2                	lw	ra,28(sp)
1c01731e:	4462                	lw	s0,24(sp)
1c017320:	8552                	mv	a0,s4
1c017322:	44d2                	lw	s1,20(sp)
1c017324:	4942                	lw	s2,16(sp)
1c017326:	49b2                	lw	s3,12(sp)
1c017328:	4a22                	lw	s4,8(sp)
1c01732a:	4a92                	lw	s5,4(sp)
1c01732c:	4b02                	lw	s6,0(sp)
1c01732e:	6105                	addi	sp,sp,32
1c017330:	8082                	ret
1c017332:	8ed1                	or	a3,a3,a2
1c017334:	c714                	sw	a3,8(a4)
1c017336:	00a7f6b3          	and	a3,a5,a0
1c01733a:	b7bd                	j	1c0172a8 <pi_gpio_pin_configure+0x7a>

1c01733c <__pi_gpio_init>:
1c01733c:	1101                	addi	sp,sp,-32
1c01733e:	c84a                	sw	s2,16(sp)
1c017340:	c64e                	sw	s3,12(sp)
1c017342:	ce06                	sw	ra,28(sp)
1c017344:	cc22                	sw	s0,24(sp)
1c017346:	ca26                	sw	s1,20(sp)
1c017348:	c452                	sw	s4,8(sp)
1c01734a:	00555993          	srli	s3,a0,0x5
1c01734e:	f4553933          	p.bclr	s2,a0,26,5
1c017352:	300474f3          	csrrci	s1,mstatus,8
1c017356:	1c01d437          	lui	s0,0x1c01d
1c01735a:	d2840793          	addi	a5,s0,-728 # 1c01cd28 <g_gpio_data>
1c01735e:	098a                	slli	s3,s3,0x2
1c017360:	2137f783          	p.lw	a5,s3(a5)
1c017364:	e7cd                	bnez	a5,1c01740e <__pi_gpio_init+0xd2>
1c017366:	8a2a                	mv	s4,a0
1c017368:	09000513          	li	a0,144
1c01736c:	266030ef          	jal	ra,1c01a5d2 <pi_l2_priv_malloc>
1c017370:	ed01                	bnez	a0,1c017388 <__pi_gpio_init+0x4c>
1c017372:	30049073          	csrw	mstatus,s1
1c017376:	5555                	li	a0,-11
1c017378:	40f2                	lw	ra,28(sp)
1c01737a:	4462                	lw	s0,24(sp)
1c01737c:	44d2                	lw	s1,20(sp)
1c01737e:	4942                	lw	s2,16(sp)
1c017380:	49b2                	lw	s3,12(sp)
1c017382:	4a22                	lw	s4,8(sp)
1c017384:	6105                	addi	sp,sp,32
1c017386:	8082                	ret
1c017388:	4705                	li	a4,1
1c01738a:	01271733          	sll	a4,a4,s2
1c01738e:	c118                	sw	a4,0(a0)
1c017390:	d2840413          	addi	s0,s0,-728
1c017394:	00052223          	sw	zero,4(a0)
1c017398:	00052423          	sw	zero,8(a0)
1c01739c:	00052623          	sw	zero,12(a0)
1c0173a0:	01050713          	addi	a4,a0,16
1c0173a4:	020250fb          	lp.setupi	x1,32,1c0173ac <__pi_gpio_init+0x70>
1c0173a8:	0007222b          	p.sw	zero,4(a4!) # 1a109004 <__l1_heapsram_end+0xa0e9004>
1c0173ac:	0001                	nop
1c0173ae:	401c                	lw	a5,0(s0)
1c0173b0:	e7a9                	bnez	a5,1c0173fa <__pi_gpio_init+0xbe>
1c0173b2:	405c                	lw	a5,4(s0)
1c0173b4:	e3b9                	bnez	a5,1c0173fa <__pi_gpio_init+0xbe>
1c0173b6:	441c                	lw	a5,8(s0)
1c0173b8:	e3a9                	bnez	a5,1c0173fa <__pi_gpio_init+0xbe>
1c0173ba:	1a1097b7          	lui	a5,0x1a109
1c0173be:	6721                	lui	a4,0x8
1c0173c0:	c3d8                	sw	a4,4(a5)
1c0173c2:	305026f3          	csrr	a3,mtvec
1c0173c6:	1c0107b7          	lui	a5,0x1c010
1c0173ca:	c006b6b3          	p.bclr	a3,a3,0,0
1c0173ce:	61878793          	addi	a5,a5,1560 # 1c010618 <ASM_FUNC_SAVE_ADDITIONAL_CONTEXT+0x10>
1c0173d2:	8f95                	sub	a5,a5,a3
1c0173d4:	c1478633          	p.extract	a2,a5,0,20
1c0173d8:	06f00713          	li	a4,111
1c0173dc:	c1f62733          	p.insert	a4,a2,0,31
1c0173e0:	d2178633          	p.extract	a2,a5,9,1
1c0173e4:	d3562733          	p.insert	a4,a2,9,21
1c0173e8:	c0b78633          	p.extract	a2,a5,0,11
1c0173ec:	c1462733          	p.insert	a4,a2,0,20
1c0173f0:	cec787b3          	p.extract	a5,a5,7,12
1c0173f4:	cec7a733          	p.insert	a4,a5,7,12
1c0173f8:	ded8                	sw	a4,60(a3)
1c0173fa:	00a469a3          	p.sw	a0,s3(s0)
1c0173fe:	4581                	li	a1,0
1c017400:	8552                	mv	a0,s4
1c017402:	e2dff0ef          	jal	ra,1c01722e <pi_gpio_pin_configure>
1c017406:	30049073          	csrw	mstatus,s1
1c01740a:	4501                	li	a0,0
1c01740c:	b7b5                	j	1c017378 <__pi_gpio_init+0x3c>
1c01740e:	4390                	lw	a2,0(a5)
1c017410:	012656b3          	srl	a3,a2,s2
1c017414:	fc16b6b3          	p.bclr	a3,a3,30,1
1c017418:	f6fd                	bnez	a3,1c017406 <__pi_gpio_init+0xca>
1c01741a:	4705                	li	a4,1
1c01741c:	01271733          	sll	a4,a4,s2
1c017420:	8f51                	or	a4,a4,a2
1c017422:	c398                	sw	a4,0(a5)
1c017424:	4581                	li	a1,0
1c017426:	bff1                	j	1c017402 <__pi_gpio_init+0xc6>

1c017428 <pi_gpio_pin_write>:
1c017428:	00555613          	srli	a2,a0,0x5
1c01742c:	1c01d6b7          	lui	a3,0x1c01d
1c017430:	00261793          	slli	a5,a2,0x2
1c017434:	d2868693          	addi	a3,a3,-728 # 1c01cd28 <g_gpio_data>
1c017438:	20f6f683          	p.lw	a3,a5(a3)
1c01743c:	fa2637b3          	p.bclr	a5,a2,29,2
1c017440:	02c79a63          	bne	a5,a2,1c017474 <pi_gpio_pin_write+0x4c>
1c017444:	46d4                	lw	a3,12(a3)
1c017446:	4705                	li	a4,1
1c017448:	00a71733          	sll	a4,a4,a0
1c01744c:	8ef9                	and	a3,a3,a4
1c01744e:	5551                	li	a0,-12
1c017450:	02e69363          	bne	a3,a4,1c017476 <pi_gpio_pin_write+0x4e>
1c017454:	04800693          	li	a3,72
1c017458:	0ff5f593          	andi	a1,a1,255
1c01745c:	02d787b3          	mul	a5,a5,a3
1c017460:	c981                	beqz	a1,1c017470 <pi_gpio_pin_write+0x48>
1c017462:	07c1                	addi	a5,a5,16
1c017464:	1a1016b7          	lui	a3,0x1a101
1c017468:	00e6e7a3          	p.sw	a4,a5(a3)
1c01746c:	4501                	li	a0,0
1c01746e:	8082                	ret
1c017470:	07d1                	addi	a5,a5,20
1c017472:	bfcd                	j	1c017464 <pi_gpio_pin_write+0x3c>
1c017474:	5555                	li	a0,-11
1c017476:	8082                	ret

1c017478 <pi_efuse_ioctl>:
1c017478:	1141                	addi	sp,sp,-16
1c01747a:	c422                	sw	s0,8(sp)
1c01747c:	842a                	mv	s0,a0
1c01747e:	4501                	li	a0,0
1c017480:	c606                	sw	ra,12(sp)
1c017482:	43d020ef          	jal	ra,1c01a0be <pi_pmu_voltage_get>
1c017486:	32000793          	li	a5,800
1c01748a:	02f51963          	bne	a0,a5,1c0174bc <pi_efuse_ioctl+0x44>
1c01748e:	00242f63          	p.beqimm	s0,2,1c0174ac <pi_efuse_ioctl+0x34>
1c017492:	02342163          	p.beqimm	s0,3,1c0174b4 <pi_efuse_ioctl+0x3c>
1c017496:	02143363          	p.bneimm	s0,1,1c0174bc <pi_efuse_ioctl+0x44>
1c01749a:	1a10f7b7          	lui	a5,0x1a10f
1c01749e:	4709                	li	a4,2
1c0174a0:	c398                	sw	a4,0(a5)
1c0174a2:	4501                	li	a0,0
1c0174a4:	40b2                	lw	ra,12(sp)
1c0174a6:	4422                	lw	s0,8(sp)
1c0174a8:	0141                	addi	sp,sp,16
1c0174aa:	8082                	ret
1c0174ac:	1a10f7b7          	lui	a5,0x1a10f
1c0174b0:	4705                	li	a4,1
1c0174b2:	b7fd                	j	1c0174a0 <pi_efuse_ioctl+0x28>
1c0174b4:	1a10f7b7          	lui	a5,0x1a10f
1c0174b8:	4711                	li	a4,4
1c0174ba:	b7dd                	j	1c0174a0 <pi_efuse_ioctl+0x28>
1c0174bc:	557d                	li	a0,-1
1c0174be:	b7dd                	j	1c0174a4 <pi_efuse_ioctl+0x2c>

1c0174c0 <pi_efuse_reg_val_update>:
1c0174c0:	1141                	addi	sp,sp,-16
1c0174c2:	4581                	li	a1,0
1c0174c4:	4509                	li	a0,2
1c0174c6:	c606                	sw	ra,12(sp)
1c0174c8:	3f45                	jal	1c017478 <pi_efuse_ioctl>
1c0174ca:	57fd                	li	a5,-1
1c0174cc:	ed39                	bnez	a0,1c01752a <pi_efuse_reg_val_update+0x6a>
1c0174ce:	1a10f7b7          	lui	a5,0x1a10f
1c0174d2:	20078793          	addi	a5,a5,512 # 1a10f200 <__l1_heapsram_end+0xa0ef200>
1c0174d6:	57b4                	lw	a3,104(a5)
1c0174d8:	1c01e737          	lui	a4,0x1c01e
1c0174dc:	4645                	li	a2,17
1c0174de:	aed72623          	sw	a3,-1300(a4) # 1c01daec <pi_efuse_regs>
1c0174e2:	1c01e737          	lui	a4,0x1c01e
1c0174e6:	af070713          	addi	a4,a4,-1296 # 1c01daf0 <pi_efuse_regs+0x4>
1c0174ea:	0083d0fb          	lp.setupi	x1,8,1c0174f8 <pi_efuse_reg_val_update+0x38>
1c0174ee:	00261593          	slli	a1,a2,0x2
1c0174f2:	20b7f583          	p.lw	a1,a1(a5)
1c0174f6:	0605                	addi	a2,a2,1
1c0174f8:	00b7222b          	p.sw	a1,4(a4!)
1c0174fc:	1c01e7b7          	lui	a5,0x1c01e
1c017500:	1a10f5b7          	lui	a1,0x1a10f
1c017504:	b1078793          	addi	a5,a5,-1264 # 1c01db10 <pi_efuse_regs+0x24>
1c017508:	04200693          	li	a3,66
1c01750c:	20058593          	addi	a1,a1,512 # 1a10f200 <__l1_heapsram_end+0xa0ef200>
1c017510:	0073d0fb          	lp.setupi	x1,7,1c01751e <pi_efuse_reg_val_update+0x5e>
1c017514:	00269613          	slli	a2,a3,0x2
1c017518:	20c5f603          	p.lw	a2,a2(a1)
1c01751c:	0685                	addi	a3,a3,1
1c01751e:	00c7a22b          	p.sw	a2,4(a5!)
1c017522:	4581                	li	a1,0
1c017524:	450d                	li	a0,3
1c017526:	3f89                	jal	1c017478 <pi_efuse_ioctl>
1c017528:	4781                	li	a5,0
1c01752a:	40b2                	lw	ra,12(sp)
1c01752c:	853e                	mv	a0,a5
1c01752e:	0141                	addi	sp,sp,16
1c017530:	8082                	ret

1c017532 <pi_pad_function_set>:
1c017532:	00455793          	srli	a5,a0,0x4
1c017536:	00151693          	slli	a3,a0,0x1
1c01753a:	078a                	slli	a5,a5,0x2
1c01753c:	01e6f613          	andi	a2,a3,30
1c017540:	07c1                	addi	a5,a5,16
1c017542:	1a104837          	lui	a6,0x1a104
1c017546:	20f87683          	p.lw	a3,a5(a6)
1c01754a:	470d                	li	a4,3
1c01754c:	00c71733          	sll	a4,a4,a2
1c017550:	fff74713          	not	a4,a4
1c017554:	8f75                	and	a4,a4,a3
1c017556:	ee85b6b3          	p.bclr	a3,a1,23,8
1c01755a:	00c696b3          	sll	a3,a3,a2
1c01755e:	8f55                	or	a4,a4,a3
1c017560:	00e867a3          	p.sw	a4,a5(a6)
1c017564:	0015b463          	p.bneimm	a1,1,1c01756c <pi_pad_function_set+0x3a>
1c017568:	dd5ff06f          	j	1c01733c <__pi_gpio_init>
1c01756c:	c35ff06f          	j	1c0171a0 <__pi_gpio_deinit>

1c017570 <pi_pad_init>:
1c017570:	4741                	li	a4,16
1c017572:	1a104637          	lui	a2,0x1a104
1c017576:	006350fb          	lp.setupi	x1,6,1c017582 <pi_pad_init+0x12>
1c01757a:	0045268b          	p.lw	a3,4(a0!)
1c01757e:	00d66723          	p.sw	a3,a4(a2)
1c017582:	0711                	addi	a4,a4,4
1c017584:	8082                	ret

1c017586 <pi_pad_drive_strength_set>:
1c017586:	fff58793          	addi	a5,a1,-1
1c01758a:	4689                	li	a3,2
1c01758c:	4701                	li	a4,0
1c01758e:	00f6e963          	bltu	a3,a5,1c0175a0 <pi_pad_drive_strength_set+0x1a>
1c017592:	1c0005b7          	lui	a1,0x1c000
1c017596:	5a458593          	addi	a1,a1,1444 # 1c0005a4 <CSWTCH.46>
1c01759a:	95be                	add	a1,a1,a5
1c01759c:	0005c703          	lbu	a4,0(a1)
1c0175a0:	00351693          	slli	a3,a0,0x3
1c0175a4:	8ae1                	andi	a3,a3,24
1c0175a6:	c2053533          	p.bclr	a0,a0,1,0
1c0175aa:	0689                	addi	a3,a3,2
1c0175ac:	03050513          	addi	a0,a0,48
1c0175b0:	1a104637          	lui	a2,0x1a104
1c0175b4:	20a67583          	p.lw	a1,a0(a2)
1c0175b8:	478d                	li	a5,3
1c0175ba:	00d797b3          	sll	a5,a5,a3
1c0175be:	fff7c793          	not	a5,a5
1c0175c2:	8fed                	and	a5,a5,a1
1c0175c4:	00d71733          	sll	a4,a4,a3
1c0175c8:	8fd9                	or	a5,a5,a4
1c0175ca:	00f66523          	p.sw	a5,a0(a2)
1c0175ce:	8082                	ret

1c0175d0 <pi_pad_pull_set>:
1c0175d0:	4789                	li	a5,2
1c0175d2:	02b7ea63          	bltu	a5,a1,1c017606 <pi_pad_pull_set+0x36>
1c0175d6:	00351713          	slli	a4,a0,0x3
1c0175da:	c2053533          	p.bclr	a0,a0,1,0
1c0175de:	8b61                	andi	a4,a4,24
1c0175e0:	03050513          	addi	a0,a0,48
1c0175e4:	1a1046b7          	lui	a3,0x1a104
1c0175e8:	20a6f603          	p.lw	a2,a0(a3)
1c0175ec:	4785                	li	a5,1
1c0175ee:	00e797b3          	sll	a5,a5,a4
1c0175f2:	fff7c793          	not	a5,a5
1c0175f6:	8ff1                	and	a5,a5,a2
1c0175f8:	00e595b3          	sll	a1,a1,a4
1c0175fc:	8fcd                	or	a5,a5,a1
1c0175fe:	00f6e523          	p.sw	a5,a0(a3)
1c017602:	4501                	li	a0,0
1c017604:	8082                	ret
1c017606:	4505                	li	a0,1
1c017608:	8082                	ret

1c01760a <pi_pad_schmitt_trigger_set>:
1c01760a:	4785                	li	a5,1
1c01760c:	02b7ea63          	bltu	a5,a1,1c017640 <pi_pad_schmitt_trigger_set+0x36>
1c017610:	00351713          	slli	a4,a0,0x3
1c017614:	8b61                	andi	a4,a4,24
1c017616:	c2053533          	p.bclr	a0,a0,1,0
1c01761a:	0711                	addi	a4,a4,4
1c01761c:	03050513          	addi	a0,a0,48
1c017620:	1a1046b7          	lui	a3,0x1a104
1c017624:	20a6f603          	p.lw	a2,a0(a3)
1c017628:	00e797b3          	sll	a5,a5,a4
1c01762c:	fff7c793          	not	a5,a5
1c017630:	8ff1                	and	a5,a5,a2
1c017632:	00e595b3          	sll	a1,a1,a4
1c017636:	8fcd                	or	a5,a5,a1
1c017638:	00f6e523          	p.sw	a5,a0(a3)
1c01763c:	4501                	li	a0,0
1c01763e:	8082                	ret
1c017640:	4505                	li	a0,1
1c017642:	8082                	ret

1c017644 <pi_pad_mux_group_init>:
1c017644:	411c                	lw	a5,0(a0)
1c017646:	1a104737          	lui	a4,0x1a104
1c01764a:	08f72823          	sw	a5,144(a4) # 1a104090 <__l1_heapsram_end+0xa0e4090>
1c01764e:	415c                	lw	a5,4(a0)
1c017650:	08f72a23          	sw	a5,148(a4)
1c017654:	451c                	lw	a5,8(a0)
1c017656:	08f72c23          	sw	a5,152(a4)
1c01765a:	455c                	lw	a5,12(a0)
1c01765c:	08f72e23          	sw	a5,156(a4)
1c017660:	8082                	ret

1c017662 <pi_time_wait_us>:
1c017662:	c90d                	beqz	a0,1c017694 <pi_time_wait_us+0x32>
1c017664:	7159                	addi	sp,sp,-112
1c017666:	d4a2                	sw	s0,104(sp)
1c017668:	842a                	mv	s0,a0
1c01766a:	0028                	addi	a0,sp,8
1c01766c:	d686                	sw	ra,108(sp)
1c01766e:	06d020ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c017672:	4589                	li	a1,2
1c017674:	1c01e537          	lui	a0,0x1c01e
1c017678:	0030                	addi	a2,sp,8
1c01767a:	04b475b3          	p.maxu	a1,s0,a1
1c01767e:	b5050513          	addi	a0,a0,-1200 # 1c01db50 <sys_timer_hi_prec>
1c017682:	528020ef          	jal	ra,1c019baa <pi_timer_task_add>
1c017686:	0028                	addi	a0,sp,8
1c017688:	095020ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c01768c:	50b6                	lw	ra,108(sp)
1c01768e:	5426                	lw	s0,104(sp)
1c017690:	6165                	addi	sp,sp,112
1c017692:	8082                	ret
1c017694:	8082                	ret

1c017696 <pi_udma_core_2d_alloc>:
1c017696:	30047773          	csrrci	a4,mstatus,8
1c01769a:	4d402783          	lw	a5,1236(zero) # 4d4 <__pi_udma_chan_2d>
1c01769e:	cf81                	beqz	a5,1c0176b6 <pi_udma_core_2d_alloc+0x20>
1c0176a0:	10078533          	p.ff1	a0,a5
1c0176a4:	80a7b7b3          	p.bclrr	a5,a5,a0
1c0176a8:	4cf02a23          	sw	a5,1236(zero) # 4d4 <__pi_udma_chan_2d>
1c0176ac:	30071073          	csrw	mstatus,a4
1c0176b0:	04050513          	addi	a0,a0,64
1c0176b4:	8082                	ret
1c0176b6:	30071073          	csrw	mstatus,a4
1c0176ba:	557d                	li	a0,-1
1c0176bc:	8082                	ret

1c0176be <__os_native_irq_handler_set>:
1c0176be:	30502773          	csrr	a4,mtvec
1c0176c2:	050a                	slli	a0,a0,0x2
1c0176c4:	c0073733          	p.bclr	a4,a4,0,0
1c0176c8:	8d89                	sub	a1,a1,a0
1c0176ca:	8d99                	sub	a1,a1,a4
1c0176cc:	c14586b3          	p.extract	a3,a1,0,20
1c0176d0:	06f00793          	li	a5,111
1c0176d4:	c1f6a7b3          	p.insert	a5,a3,0,31
1c0176d8:	d21586b3          	p.extract	a3,a1,9,1
1c0176dc:	d356a7b3          	p.insert	a5,a3,9,21
1c0176e0:	c0b586b3          	p.extract	a3,a1,0,11
1c0176e4:	c146a7b3          	p.insert	a5,a3,0,20
1c0176e8:	cec585b3          	p.extract	a1,a1,7,12
1c0176ec:	cec5a7b3          	p.insert	a5,a1,7,12
1c0176f0:	00f76523          	p.sw	a5,a0(a4)
1c0176f4:	8082                	ret

1c0176f6 <pi_udma_core_2d_free>:
1c0176f6:	300477f3          	csrrci	a5,mstatus,8
1c0176fa:	4d402703          	lw	a4,1236(zero) # 4d4 <__pi_udma_chan_2d>
1c0176fe:	fc050513          	addi	a0,a0,-64
1c017702:	80a74533          	p.bsetr	a0,a4,a0
1c017706:	4ca02a23          	sw	a0,1236(zero) # 4d4 <__pi_udma_chan_2d>
1c01770a:	30079073          	csrw	mstatus,a5
1c01770e:	8082                	ret

1c017710 <pi_octospi_conf_init>:
1c017710:	57fd                	li	a5,-1
1c017712:	00f50023          	sb	a5,0(a0)
1c017716:	4785                	li	a5,1
1c017718:	c51c                	sw	a5,8(a0)
1c01771a:	57fd                	li	a5,-1
1c01771c:	00052623          	sw	zero,12(a0)
1c017720:	00052223          	sw	zero,4(a0)
1c017724:	c91c                	sw	a5,16(a0)
1c017726:	8082                	ret

1c017728 <pi_octospi_open>:
1c017728:	7139                	addi	sp,sp,-64
1c01772a:	d84a                	sw	s2,48(sp)
1c01772c:	00452903          	lw	s2,4(a0)
1c017730:	dc22                	sw	s0,56(sp)
1c017732:	da26                	sw	s1,52(sp)
1c017734:	d64e                	sw	s3,44(sp)
1c017736:	d452                	sw	s4,40(sp)
1c017738:	d256                	sw	s5,36(sp)
1c01773a:	d05a                	sw	s6,32(sp)
1c01773c:	c86a                	sw	s10,16(sp)
1c01773e:	c66e                	sw	s11,12(sp)
1c017740:	de06                	sw	ra,60(sp)
1c017742:	ce5e                	sw	s7,28(sp)
1c017744:	cc62                	sw	s8,24(sp)
1c017746:	ca66                	sw	s9,20(sp)
1c017748:	00090b83          	lb	s7,0(s2)
1c01774c:	00492a03          	lw	s4,4(s2)
1c017750:	4771                	li	a4,28
1c017752:	100bcdb3          	p.exths	s11,s7
1c017756:	003d9a93          	slli	s5,s11,0x3
1c01775a:	41ba8ab3          	sub	s5,s5,s11
1c01775e:	0a8e                	slli	s5,s5,0x3
1c017760:	42ea0ab3          	p.mac	s5,s4,a4
1c017764:	1c01d4b7          	lui	s1,0x1c01d
1c017768:	d7448793          	addi	a5,s1,-652 # 1c01cd74 <pi_octospi_cs>
1c01776c:	00db8413          	addi	s0,s7,13
1c017770:	041e                	slli	s0,s0,0x7
1c017772:	1c01dd37          	lui	s10,0x1c01d
1c017776:	8b2a                	mv	s6,a0
1c017778:	d7448493          	addi	s1,s1,-652
1c01777c:	d34d0d13          	addi	s10,s10,-716 # 1c01cd34 <pi_octospi>
1c017780:	005b9993          	slli	s3,s7,0x5
1c017784:	9abe                	add	s5,s5,a5
1c017786:	1a1027b7          	lui	a5,0x1a102
1c01778a:	08078793          	addi	a5,a5,128 # 1a102080 <__l1_heapsram_end+0xa0e2080>
1c01778e:	943e                	add	s0,s0,a5
1c017790:	1c01d7b7          	lui	a5,0x1c01d
1c017794:	79c78793          	addi	a5,a5,1948 # 1c01d79c <pi_octospi_open_count>
1c017798:	4177fc03          	p.lbu	s8,s7(a5)
1c01779c:	0c05                	addi	s8,s8,1
1c01779e:	0ffc7c13          	andi	s8,s8,255
1c0177a2:	0187cba3          	p.sb	s8,s7(a5)
1c0177a6:	0e1c3763          	p.bneimm	s8,1,1c017894 <pi_octospi_open+0x16c>
1c0177aa:	013d0cb3          	add	s9,s10,s3
1c0177ae:	0ffbf793          	andi	a5,s7,255
1c0177b2:	00fc8d23          	sb	a5,26(s9)
1c0177b6:	07b5                	addi	a5,a5,13
1c0177b8:	00fc8da3          	sb	a5,27(s9)
1c0177bc:	01eb8793          	addi	a5,s7,30
1c0177c0:	000ca223          	sw	zero,4(s9)
1c0177c4:	000ca423          	sw	zero,8(s9)
1c0177c8:	008ca023          	sw	s0,0(s9)
1c0177cc:	000c8e23          	sb	zero,28(s9)
1c0177d0:	00fc17b3          	sll	a5,s8,a5
1c0177d4:	1a109737          	lui	a4,0x1a109
1c0177d8:	c35c                	sw	a5,4(a4)
1c0177da:	03800793          	li	a5,56
1c0177de:	8726                	mv	a4,s1
1c0177e0:	42fb8733          	p.mac	a4,s7,a5
1c0177e4:	0d96                	slli	s11,s11,0x5
1c0177e6:	9dea                	add	s11,s11,s10
1c0177e8:	87ba                	mv	a5,a4
1c0177ea:	01b72023          	sw	s11,0(a4) # 1a109000 <__l1_heapsram_end+0xa0e9000>
1c0177ee:	00072223          	sw	zero,4(a4)
1c0177f2:	00070c23          	sb	zero,24(a4)
1c0177f6:	01b72e23          	sw	s11,28(a4)
1c0177fa:	40000737          	lui	a4,0x40000
1c0177fe:	d398                	sw	a4,32(a5)
1c017800:	03878a23          	sb	s8,52(a5)
1c017804:	e93ff0ef          	jal	ra,1c017696 <pi_udma_core_2d_alloc>
1c017808:	00ac8ca3          	sb	a0,25(s9)
1c01780c:	e8bff0ef          	jal	ra,1c017696 <pi_udma_core_2d_alloc>
1c017810:	01bcc783          	lbu	a5,27(s9)
1c017814:	00ac8c23          	sb	a0,24(s9)
1c017818:	00fc1c33          	sll	s8,s8,a5
1c01781c:	1a1027b7          	lui	a5,0x1a102
1c017820:	0187a823          	sw	s8,16(a5) # 1a102010 <__l1_heapsram_end+0xa0e2010>
1c017824:	0187a223          	sw	s8,4(a5)
1c017828:	019cc703          	lbu	a4,25(s9)
1c01782c:	67c1                	lui	a5,0x10
1c01782e:	f0078793          	addi	a5,a5,-256 # ff00 <__heapfcram_size+0xcc0>
1c017832:	8f5d                	or	a4,a4,a5
1c017834:	00e42023          	sw	a4,0(s0)
1c017838:	0ff57513          	andi	a0,a0,255
1c01783c:	8d5d                	or	a0,a0,a5
1c01783e:	00a42223          	sw	a0,4(s0)
1c017842:	00c92c03          	lw	s8,12(s2)
1c017846:	4501                	li	a0,0
1c017848:	cd6f90ef          	jal	ra,1c010d1e <pi_freq_get>
1c01784c:	40155793          	srai	a5,a0,0x1
1c017850:	000c0463          	beqz	s8,1c017858 <pi_octospi_open+0x130>
1c017854:	12fc4e63          	blt	s8,a5,1c017990 <pi_octospi_open+0x268>
1c017858:	00fdaa23          	sw	a5,20(s11)
1c01785c:	4781                	li	a5,0
1c01785e:	06f42223          	sw	a5,100(s0)
1c017862:	4785                	li	a5,1
1c017864:	00f42423          	sw	a5,8(s0)
1c017868:	06f42023          	sw	a5,96(s0)
1c01786c:	02042423          	sw	zero,40(s0)
1c017870:	400007b7          	lui	a5,0x40000
1c017874:	02f42623          	sw	a5,44(s0)
1c017878:	6791                	lui	a5,0x4
1c01787a:	0c678793          	addi	a5,a5,198 # 40c6 <__STACK_IRQ_SIZE+0x38c6>
1c01787e:	02f42223          	sw	a5,36(s0)
1c017882:	120b9363          	bnez	s7,1c0179a8 <pi_octospi_open+0x280>
1c017886:	1c0105b7          	lui	a1,0x1c010
1c01788a:	68058593          	addi	a1,a1,1664 # 1c010680 <pi_octospi0_handler_asm>
1c01788e:	4579                	li	a0,30
1c017890:	e2fff0ef          	jal	ra,1c0176be <__os_native_irq_handler_set>
1c017894:	04042783          	lw	a5,64(s0)
1c017898:	4705                	li	a4,1
1c01789a:	c01727b3          	p.insert	a5,a4,0,1
1c01789e:	04f42023          	sw	a5,64(s0)
1c0178a2:	04842683          	lw	a3,72(s0)
1c0178a6:	67b5                	lui	a5,0xd
1c0178a8:	001a1613          	slli	a2,s4,0x1
1c0178ac:	c0078793          	addi	a5,a5,-1024 # cc00 <__STACK_IRQ_SIZE+0xc400>
1c0178b0:	4711                	li	a4,4
1c0178b2:	01471733          	sll	a4,a4,s4
1c0178b6:	00c797b3          	sll	a5,a5,a2
1c0178ba:	8fd9                	or	a5,a5,a4
1c0178bc:	00892703          	lw	a4,8(s2)
1c0178c0:	fff7c793          	not	a5,a5
1c0178c4:	8ff5                	and	a5,a5,a3
1c0178c6:	00173d63          	p.bneimm	a4,1,1c0178e0 <pi_octospi_open+0x1b8>
1c0178ca:	00a60593          	addi	a1,a2,10 # 1a10400a <__l1_heapsram_end+0xa0e400a>
1c0178ce:	4689                	li	a3,2
1c0178d0:	00b696b3          	sll	a3,a3,a1
1c0178d4:	0639                	addi	a2,a2,14
1c0178d6:	458d                	li	a1,3
1c0178d8:	00c59633          	sll	a2,a1,a2
1c0178dc:	8ed1                	or	a3,a3,a2
1c0178de:	8fd5                	or	a5,a5,a3
1c0178e0:	03800693          	li	a3,56
1c0178e4:	02db8bb3          	mul	s7,s7,a3
1c0178e8:	46f1                	li	a3,28
1c0178ea:	42da0bb3          	p.mac	s7,s4,a3
1c0178ee:	213d7683          	p.lw	a3,s3(s10)
1c0178f2:	9ba6                	add	s7,s7,s1
1c0178f4:	00fbaa23          	sw	a5,20(s7)
1c0178f8:	04f6a423          	sw	a5,72(a3) # 1a104048 <__l1_heapsram_end+0xa0e4048>
1c0178fc:	05c6a683          	lw	a3,92(a3)
1c017900:	fff70793          	addi	a5,a4,-1 # 3fffffff <__heapl2ram_size+0x23e6ffff>
1c017904:	00f037b3          	snez	a5,a5
1c017908:	0ff7f793          	andi	a5,a5,255
1c01790c:	03042703          	lw	a4,48(s0)
1c017910:	00492583          	lw	a1,4(s2)
1c017914:	e1c5                	bnez	a1,1c0179b4 <pi_octospi_open+0x28c>
1c017916:	c017a733          	p.insert	a4,a5,0,1
1c01791a:	c0002733          	p.insert	a4,zero,0,0
1c01791e:	02e42823          	sw	a4,48(s0)
1c017922:	02442603          	lw	a2,36(s0)
1c017926:	01092703          	lw	a4,16(s2)
1c01792a:	c446c7b3          	p.bset	a5,a3,2,4
1c01792e:	03f72963          	p.beqimm	a4,-1,1c017960 <pi_octospi_open+0x238>
1c017932:	013d07b3          	add	a5,s10,s3
1c017936:	4bdc                	lw	a5,20(a5)
1c017938:	3e800513          	li	a0,1000
1c01793c:	02a7c7b3          	div	a5,a5,a0
1c017940:	02e787b3          	mul	a5,a5,a4
1c017944:	000f4737          	lui	a4,0xf4
1c017948:	24070713          	addi	a4,a4,576 # f4240 <__l1_heapsram_size+0xd42b0>
1c01794c:	02e7d7b3          	divu	a5,a5,a4
1c017950:	17d9                	addi	a5,a5,-10
1c017952:	db27a633          	p.insert	a2,a5,13,18
1c017956:	07a6e793          	ori	a5,a3,122
1c01795a:	e199                	bnez	a1,1c017960 <pi_octospi_open+0x238>
1c01795c:	0756e793          	ori	a5,a3,117
1c017960:	213d7983          	p.lw	s3,s3(s10)
1c017964:	04f9ae23          	sw	a5,92(s3)
1c017968:	02c42223          	sw	a2,36(s0)
1c01796c:	50f2                	lw	ra,60(sp)
1c01796e:	5462                	lw	s0,56(sp)
1c017970:	015b2423          	sw	s5,8(s6)
1c017974:	54d2                	lw	s1,52(sp)
1c017976:	5942                	lw	s2,48(sp)
1c017978:	59b2                	lw	s3,44(sp)
1c01797a:	5a22                	lw	s4,40(sp)
1c01797c:	5a92                	lw	s5,36(sp)
1c01797e:	5b02                	lw	s6,32(sp)
1c017980:	4bf2                	lw	s7,28(sp)
1c017982:	4c62                	lw	s8,24(sp)
1c017984:	4cd2                	lw	s9,20(sp)
1c017986:	4d42                	lw	s10,16(sp)
1c017988:	4db2                	lw	s11,12(sp)
1c01798a:	4501                	li	a0,0
1c01798c:	6121                	addi	sp,sp,64
1c01798e:	8082                	ret
1c017990:	038547b3          	div	a5,a0,s8
1c017994:	03856c33          	rem	s8,a0,s8
1c017998:	000c0363          	beqz	s8,1c01799e <pi_octospi_open+0x276>
1c01799c:	0785                	addi	a5,a5,1
1c01799e:	02f54533          	div	a0,a0,a5
1c0179a2:	00adaa23          	sw	a0,20(s11)
1c0179a6:	bd65                	j	1c01785e <pi_octospi_open+0x136>
1c0179a8:	1c0105b7          	lui	a1,0x1c010
1c0179ac:	69658593          	addi	a1,a1,1686 # 1c010696 <pi_octospi1_handler_asm>
1c0179b0:	457d                	li	a0,31
1c0179b2:	bdf9                	j	1c017890 <pi_octospi_open+0x168>
1c0179b4:	c027a733          	p.insert	a4,a5,0,2
1c0179b8:	b78d                	j	1c01791a <pi_octospi_open+0x1f2>

1c0179ba <pi_octospi_close>:
1c0179ba:	451c                	lw	a5,8(a0)
1c0179bc:	1141                	addi	sp,sp,-16
1c0179be:	c422                	sw	s0,8(sp)
1c0179c0:	4380                	lw	s0,0(a5)
1c0179c2:	c606                	sw	ra,12(sp)
1c0179c4:	1c01d737          	lui	a4,0x1c01d
1c0179c8:	01a44683          	lbu	a3,26(s0)
1c0179cc:	79c70713          	addi	a4,a4,1948 # 1c01d79c <pi_octospi_open_count>
1c0179d0:	40d77783          	p.lbu	a5,a3(a4)
1c0179d4:	17fd                	addi	a5,a5,-1
1c0179d6:	0ff7f793          	andi	a5,a5,255
1c0179da:	00f746a3          	p.sb	a5,a3(a4)
1c0179de:	eb95                	bnez	a5,1c017a12 <pi_octospi_close+0x58>
1c0179e0:	4018                	lw	a4,0(s0)
1c0179e2:	57fd                	li	a5,-1
1c0179e4:	00f72023          	sw	a5,0(a4)
1c0179e8:	00f72223          	sw	a5,4(a4)
1c0179ec:	01944503          	lbu	a0,25(s0)
1c0179f0:	d07ff0ef          	jal	ra,1c0176f6 <pi_udma_core_2d_free>
1c0179f4:	01844503          	lbu	a0,24(s0)
1c0179f8:	cffff0ef          	jal	ra,1c0176f6 <pi_udma_core_2d_free>
1c0179fc:	01b44703          	lbu	a4,27(s0)
1c017a00:	4785                	li	a5,1
1c017a02:	00e797b3          	sll	a5,a5,a4
1c017a06:	1a102737          	lui	a4,0x1a102
1c017a0a:	00f72423          	sw	a5,8(a4) # 1a102008 <__l1_heapsram_end+0xa0e2008>
1c017a0e:	00f72a23          	sw	a5,20(a4)
1c017a12:	40b2                	lw	ra,12(sp)
1c017a14:	4422                	lw	s0,8(sp)
1c017a16:	0141                	addi	sp,sp,16
1c017a18:	8082                	ret

1c017a1a <pi_octospi_set_op>:
1c017a1a:	300478f3          	csrrci	a7,mstatus,8
1c017a1e:	00052e03          	lw	t3,0(a0)
1c017a22:	000e2683          	lw	a3,0(t3)
1c017a26:	0246a783          	lw	a5,36(a3)
1c017a2a:	01854703          	lbu	a4,24(a0)
1c017a2e:	41d0                	lw	a2,4(a1)
1c017a30:	eb55                	bnez	a4,1c017ae4 <pi_octospi_set_op+0xca>
1c017a32:	c80627b3          	p.insert	a5,a2,4,0
1c017a36:	02f6a223          	sw	a5,36(a3)
1c017a3a:	6641                	lui	a2,0x10
1c017a3c:	010e2503          	lw	a0,16(t3)
1c017a40:	00471813          	slli	a6,a4,0x4
1c017a44:	fff60313          	addi	t1,a2,-1 # ffff <__heapfcram_size+0xdbf>
1c017a48:	010317b3          	sll	a5,t1,a6
1c017a4c:	fff7c793          	not	a5,a5
1c017a50:	8fe9                	and	a5,a5,a0
1c017a52:	4588                	lw	a0,8(a1)
1c017a54:	01051533          	sll	a0,a0,a6
1c017a58:	8fc9                	or	a5,a5,a0
1c017a5a:	00fe2823          	sw	a5,16(t3)
1c017a5e:	02f6ae23          	sw	a5,60(a3)
1c017a62:	4188                	lw	a0,0(a1)
1c017a64:	458c                	lw	a1,8(a1)
1c017a66:	006577b3          	and	a5,a0,t1
1c017a6a:	fc15b5b3          	p.bclr	a1,a1,30,1
1c017a6e:	c981                	beqz	a1,1c017a7e <pi_octospi_set_op+0x64>
1c017a70:	07a2                	slli	a5,a5,0x8
1c017a72:	f0060613          	addi	a2,a2,-256
1c017a76:	0067f7b3          	and	a5,a5,t1
1c017a7a:	8e69                	and	a2,a2,a0
1c017a7c:	8fd1                	or	a5,a5,a2
1c017a7e:	0346a603          	lw	a2,52(a3)
1c017a82:	c6584833          	p.bset	a6,a6,3,5
1c017a86:	8107a633          	p.insertr	a2,a5,a6
1c017a8a:	02c6aa23          	sw	a2,52(a3)
1c017a8e:	0486a603          	lw	a2,72(a3)
1c017a92:	15000793          	li	a5,336
1c017a96:	00e797b3          	sll	a5,a5,a4
1c017a9a:	fff7c793          	not	a5,a5
1c017a9e:	8ff1                	and	a5,a5,a2
1c017aa0:	00e51613          	slli	a2,a0,0xe
1c017aa4:	00065863          	bgez	a2,1c017ab4 <pi_octospi_set_op+0x9a>
1c017aa8:	00670593          	addi	a1,a4,6
1c017aac:	4605                	li	a2,1
1c017aae:	00b61633          	sll	a2,a2,a1
1c017ab2:	8fd1                	or	a5,a5,a2
1c017ab4:	00f51613          	slli	a2,a0,0xf
1c017ab8:	00065863          	bgez	a2,1c017ac8 <pi_octospi_set_op+0xae>
1c017abc:	00470593          	addi	a1,a4,4
1c017ac0:	4605                	li	a2,1
1c017ac2:	00b61633          	sll	a2,a2,a1
1c017ac6:	8fd1                	or	a5,a5,a2
1c017ac8:	00d51613          	slli	a2,a0,0xd
1c017acc:	00065763          	bgez	a2,1c017ada <pi_octospi_set_op+0xc0>
1c017ad0:	0721                	addi	a4,a4,8
1c017ad2:	4605                	li	a2,1
1c017ad4:	00e61733          	sll	a4,a2,a4
1c017ad8:	8fd9                	or	a5,a5,a4
1c017ada:	04f6a423          	sw	a5,72(a3)
1c017ade:	30089073          	csrw	mstatus,a7
1c017ae2:	8082                	ret
1c017ae4:	c85627b3          	p.insert	a5,a2,4,5
1c017ae8:	b7b9                	j	1c017a36 <pi_octospi_set_op+0x1c>

1c017aea <pi_octospi_handle_pending.constprop.1>:
1c017aea:	451c                	lw	a5,8(a0)
1c017aec:	cba5                	beqz	a5,1c017b5c <pi_octospi_handle_pending.constprop.1+0x72>
1c017aee:	4398                	lw	a4,0(a5)
1c017af0:	1101                	addi	sp,sp,-32
1c017af2:	cc22                	sw	s0,24(sp)
1c017af4:	ca26                	sw	s1,20(sp)
1c017af6:	c84a                	sw	s2,16(sp)
1c017af8:	c64e                	sw	s3,12(sp)
1c017afa:	c452                	sw	s4,8(sp)
1c017afc:	c256                	sw	s5,4(sp)
1c017afe:	c05a                	sw	s6,0(sp)
1c017b00:	ce06                	sw	ra,28(sp)
1c017b02:	c518                	sw	a4,8(a0)
1c017b04:	c15c                	sw	a5,4(a0)
1c017b06:	5b8c                	lw	a1,48(a5)
1c017b08:	5bc8                	lw	a0,52(a5)
1c017b0a:	0187aa83          	lw	s5,24(a5)
1c017b0e:	01c7aa03          	lw	s4,28(a5)
1c017b12:	0207a983          	lw	s3,32(a5)
1c017b16:	53c4                	lw	s1,36(a5)
1c017b18:	0287a903          	lw	s2,40(a5)
1c017b1c:	57c0                	lw	s0,44(a5)
1c017b1e:	00052b03          	lw	s6,0(a0)
1c017b22:	c199                	beqz	a1,1c017b28 <pi_octospi_handle_pending.constprop.1+0x3e>
1c017b24:	ef7ff0ef          	jal	ra,1c017a1a <pi_octospi_set_op>
1c017b28:	000b2783          	lw	a5,0(s6)
1c017b2c:	0157a623          	sw	s5,12(a5)
1c017b30:	0147a823          	sw	s4,16(a5)
1c017b34:	0337a023          	sw	s3,32(a5)
1c017b38:	0527aa23          	sw	s2,84(a5)
1c017b3c:	0497ac23          	sw	s1,88(a5)
1c017b40:	c0144433          	p.bset	s0,s0,0,1
1c017b44:	0087aa23          	sw	s0,20(a5)
1c017b48:	40f2                	lw	ra,28(sp)
1c017b4a:	4462                	lw	s0,24(sp)
1c017b4c:	44d2                	lw	s1,20(sp)
1c017b4e:	4942                	lw	s2,16(sp)
1c017b50:	49b2                	lw	s3,12(sp)
1c017b52:	4a22                	lw	s4,8(sp)
1c017b54:	4a92                	lw	s5,4(sp)
1c017b56:	4b02                	lw	s6,0(sp)
1c017b58:	6105                	addi	sp,sp,32
1c017b5a:	8082                	ret
1c017b5c:	8082                	ret

1c017b5e <pi_octospi_copy_2d>:
1c017b5e:	4508                	lw	a0,8(a0)
1c017b60:	7179                	addi	sp,sp,-48
1c017b62:	cc52                	sw	s4,24(sp)
1c017b64:	00452a03          	lw	s4,4(a0)
1c017b68:	d226                	sw	s1,36(sp)
1c017b6a:	d606                	sw	ra,44(sp)
1c017b6c:	d422                	sw	s0,40(sp)
1c017b6e:	d04a                	sw	s2,32(sp)
1c017b70:	ce4e                	sw	s3,28(sp)
1c017b72:	ca56                	sw	s5,20(sp)
1c017b74:	c85a                	sw	s6,16(sp)
1c017b76:	c65e                	sw	s7,12(sp)
1c017b78:	54c2                	lw	s1,48(sp)
1c017b7a:	00052303          	lw	t1,0(a0)
1c017b7e:	9a2e                	add	s4,s4,a1
1c017b80:	30047bf3          	csrrci	s7,mstatus,8
1c017b84:	00432583          	lw	a1,4(t1)
1c017b88:	4100                	lw	s0,0(a0)
1c017b8a:	e9a1                	bnez	a1,1c017bda <pi_octospi_copy_2d+0x7c>
1c017b8c:	01132223          	sw	a7,4(t1)
1c017b90:	85c2                	mv	a1,a6
1c017b92:	893e                	mv	s2,a5
1c017b94:	89ba                	mv	s3,a4
1c017b96:	8ab6                	mv	s5,a3
1c017b98:	8b32                	mv	s6,a2
1c017b9a:	00080463          	beqz	a6,1c017ba2 <pi_octospi_copy_2d+0x44>
1c017b9e:	e7dff0ef          	jal	ra,1c017a1a <pi_octospi_set_op>
1c017ba2:	400c                	lw	a1,0(s0)
1c017ba4:	0165a623          	sw	s6,12(a1)
1c017ba8:	0155a823          	sw	s5,16(a1)
1c017bac:	0345a023          	sw	s4,32(a1)
1c017bb0:	0525aa23          	sw	s2,84(a1)
1c017bb4:	0535ac23          	sw	s3,88(a1)
1c017bb8:	c014c4b3          	p.bset	s1,s1,0,1
1c017bbc:	0095aa23          	sw	s1,20(a1)
1c017bc0:	300b9073          	csrw	mstatus,s7
1c017bc4:	50b2                	lw	ra,44(sp)
1c017bc6:	5422                	lw	s0,40(sp)
1c017bc8:	5492                	lw	s1,36(sp)
1c017bca:	5902                	lw	s2,32(sp)
1c017bcc:	49f2                	lw	s3,28(sp)
1c017bce:	4a62                	lw	s4,24(sp)
1c017bd0:	4ad2                	lw	s5,20(sp)
1c017bd2:	4b42                	lw	s6,16(sp)
1c017bd4:	4bb2                	lw	s7,12(sp)
1c017bd6:	6145                	addi	sp,sp,48
1c017bd8:	8082                	ret
1c017bda:	02f8a423          	sw	a5,40(a7) # 1a101028 <__l1_heapsram_end+0xa0e1028>
1c017bde:	441c                	lw	a5,8(s0)
1c017be0:	00c8ac23          	sw	a2,24(a7)
1c017be4:	00d8ae23          	sw	a3,28(a7)
1c017be8:	0348a023          	sw	s4,32(a7)
1c017bec:	02e8a223          	sw	a4,36(a7)
1c017bf0:	0298a623          	sw	s1,44(a7)
1c017bf4:	0308a823          	sw	a6,48(a7)
1c017bf8:	02a8aa23          	sw	a0,52(a7)
1c017bfc:	cb89                	beqz	a5,1c017c0e <pi_octospi_copy_2d+0xb0>
1c017bfe:	445c                	lw	a5,12(s0)
1c017c00:	0117a023          	sw	a7,0(a5)
1c017c04:	01142623          	sw	a7,12(s0)
1c017c08:	0008a023          	sw	zero,0(a7)
1c017c0c:	bf55                	j	1c017bc0 <pi_octospi_copy_2d+0x62>
1c017c0e:	01142423          	sw	a7,8(s0)
1c017c12:	bfcd                	j	1c017c04 <pi_octospi_copy_2d+0xa6>

1c017c14 <pi_octospi_init>:
1c017c14:	1c01d7b7          	lui	a5,0x1c01d
1c017c18:	78079e23          	sh	zero,1948(a5) # 1c01d79c <pi_octospi_open_count>
1c017c1c:	8082                	ret

1c017c1e <pi_octospi0_handler>:
1c017c1e:	1c01d537          	lui	a0,0x1c01d
1c017c22:	1141                	addi	sp,sp,-16
1c017c24:	d3450793          	addi	a5,a0,-716 # 1c01cd34 <pi_octospi>
1c017c28:	c422                	sw	s0,8(sp)
1c017c2a:	c606                	sw	ra,12(sp)
1c017c2c:	43c0                	lw	s0,4(a5)
1c017c2e:	d3450513          	addi	a0,a0,-716
1c017c32:	0007a223          	sw	zero,4(a5)
1c017c36:	eb5ff0ef          	jal	ra,1c017aea <pi_octospi_handle_pending.constprop.1>
1c017c3a:	8522                	mv	a0,s0
1c017c3c:	4422                	lw	s0,8(sp)
1c017c3e:	40b2                	lw	ra,12(sp)
1c017c40:	0141                	addi	sp,sp,16
1c017c42:	2de0206f          	j	1c019f20 <__pi_os_evt_push>

1c017c46 <pi_octospi1_handler>:
1c017c46:	1c01d537          	lui	a0,0x1c01d
1c017c4a:	1141                	addi	sp,sp,-16
1c017c4c:	d3450513          	addi	a0,a0,-716 # 1c01cd34 <pi_octospi>
1c017c50:	c422                	sw	s0,8(sp)
1c017c52:	c606                	sw	ra,12(sp)
1c017c54:	5140                	lw	s0,36(a0)
1c017c56:	02052223          	sw	zero,36(a0)
1c017c5a:	02050513          	addi	a0,a0,32
1c017c5e:	e8dff0ef          	jal	ra,1c017aea <pi_octospi_handle_pending.constprop.1>
1c017c62:	8522                	mv	a0,s0
1c017c64:	4422                	lw	s0,8(sp)
1c017c66:	40b2                	lw	ra,12(sp)
1c017c68:	0141                	addi	sp,sp,16
1c017c6a:	2b60206f          	j	1c019f20 <__pi_os_evt_push>

1c017c6e <pi_octospi_ioctl>:
1c017c6e:	1141                	addi	sp,sp,-16
1c017c70:	c606                	sw	ra,12(sp)
1c017c72:	c422                	sw	s0,8(sp)
1c017c74:	4508                	lw	a0,8(a0)
1c017c76:	30047473          	csrrci	s0,mstatus,8
1c017c7a:	e999                	bnez	a1,1c017c90 <pi_octospi_ioctl+0x22>
1c017c7c:	85b2                	mv	a1,a2
1c017c7e:	d9dff0ef          	jal	ra,1c017a1a <pi_octospi_set_op>
1c017c82:	30041073          	csrw	mstatus,s0
1c017c86:	4501                	li	a0,0
1c017c88:	40b2                	lw	ra,12(sp)
1c017c8a:	4422                	lw	s0,8(sp)
1c017c8c:	0141                	addi	sp,sp,16
1c017c8e:	8082                	ret
1c017c90:	30041073          	csrw	mstatus,s0
1c017c94:	557d                	li	a0,-1
1c017c96:	bfcd                	j	1c017c88 <pi_octospi_ioctl+0x1a>

1c017c98 <pi_flash_open>:
1c017c98:	415c                	lw	a5,4(a0)
1c017c9a:	439c                	lw	a5,0(a5)
1c017c9c:	0007a303          	lw	t1,0(a5)
1c017ca0:	c11c                	sw	a5,0(a0)
1c017ca2:	8302                	jr	t1

1c017ca4 <__flash_conf_init>:
1c017ca4:	00050223          	sb	zero,4(a0)
1c017ca8:	8082                	ret

1c017caa <extern_alloc_init>:
1c017caa:	1101                	addi	sp,sp,-32
1c017cac:	ca26                	sw	s1,20(sp)
1c017cae:	ce06                	sw	ra,28(sp)
1c017cb0:	cc22                	sw	s0,24(sp)
1c017cb2:	84aa                	mv	s1,a0
1c017cb4:	ce1d                	beqz	a2,1c017cf2 <extern_alloc_init+0x48>
1c017cb6:	4531                	li	a0,12
1c017cb8:	00758413          	addi	s0,a1,7
1c017cbc:	c62e                	sw	a1,12(sp)
1c017cbe:	c432                	sw	a2,8(sp)
1c017cc0:	07f020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c017cc4:	c4043433          	p.bclr	s0,s0,2,0
1c017cc8:	45b2                	lw	a1,12(sp)
1c017cca:	4622                	lw	a2,8(sp)
1c017ccc:	c515                	beqz	a0,1c017cf8 <extern_alloc_init+0x4e>
1c017cce:	95b2                	add	a1,a1,a2
1c017cd0:	8d81                	sub	a1,a1,s0
1c017cd2:	4601                	li	a2,0
1c017cd4:	00b05963          	blez	a1,1c017ce6 <extern_alloc_init+0x3c>
1c017cd8:	c088                	sw	a0,0(s1)
1c017cda:	c405b5b3          	p.bclr	a1,a1,2,0
1c017cde:	c10c                	sw	a1,0(a0)
1c017ce0:	00052223          	sw	zero,4(a0)
1c017ce4:	c500                	sw	s0,8(a0)
1c017ce6:	40f2                	lw	ra,28(sp)
1c017ce8:	4462                	lw	s0,24(sp)
1c017cea:	44d2                	lw	s1,20(sp)
1c017cec:	8532                	mv	a0,a2
1c017cee:	6105                	addi	sp,sp,32
1c017cf0:	8082                	ret
1c017cf2:	00052023          	sw	zero,0(a0)
1c017cf6:	bfc5                	j	1c017ce6 <extern_alloc_init+0x3c>
1c017cf8:	567d                	li	a2,-1
1c017cfa:	b7f5                	j	1c017ce6 <extern_alloc_init+0x3c>

1c017cfc <extern_alloc_deinit>:
1c017cfc:	4108                	lw	a0,0(a0)
1c017cfe:	45b1                	li	a1,12
1c017d00:	7440206f          	j	1c01a444 <pi_l2_free>

1c017d04 <extern_alloc>:
1c017d04:	411c                	lw	a5,0(a0)
1c017d06:	059d                	addi	a1,a1,7
1c017d08:	c405b5b3          	p.bclr	a1,a1,2,0
1c017d0c:	4681                	li	a3,0
1c017d0e:	c7b1                	beqz	a5,1c017d5a <extern_alloc+0x56>
1c017d10:	4398                	lw	a4,0(a5)
1c017d12:	02b74863          	blt	a4,a1,1c017d42 <extern_alloc+0x3e>
1c017d16:	1141                	addi	sp,sp,-16
1c017d18:	c226                	sw	s1,4(sp)
1c017d1a:	c606                	sw	ra,12(sp)
1c017d1c:	c422                	sw	s0,8(sp)
1c017d1e:	4784                	lw	s1,8(a5)
1c017d20:	02b71663          	bne	a4,a1,1c017d4c <extern_alloc+0x48>
1c017d24:	43d8                	lw	a4,4(a5)
1c017d26:	c28d                	beqz	a3,1c017d48 <extern_alloc+0x44>
1c017d28:	c2d8                	sw	a4,4(a3)
1c017d2a:	8432                	mv	s0,a2
1c017d2c:	45b1                	li	a1,12
1c017d2e:	853e                	mv	a0,a5
1c017d30:	714020ef          	jal	ra,1c01a444 <pi_l2_free>
1c017d34:	c004                	sw	s1,0(s0)
1c017d36:	40b2                	lw	ra,12(sp)
1c017d38:	4422                	lw	s0,8(sp)
1c017d3a:	4492                	lw	s1,4(sp)
1c017d3c:	4501                	li	a0,0
1c017d3e:	0141                	addi	sp,sp,16
1c017d40:	8082                	ret
1c017d42:	86be                	mv	a3,a5
1c017d44:	43dc                	lw	a5,4(a5)
1c017d46:	b7e1                	j	1c017d0e <extern_alloc+0xa>
1c017d48:	c118                	sw	a4,0(a0)
1c017d4a:	b7c5                	j	1c017d2a <extern_alloc+0x26>
1c017d4c:	40b706b3          	sub	a3,a4,a1
1c017d50:	9726                	add	a4,a4,s1
1c017d52:	c394                	sw	a3,0(a5)
1c017d54:	8f0d                	sub	a4,a4,a1
1c017d56:	c218                	sw	a4,0(a2)
1c017d58:	bff9                	j	1c017d36 <extern_alloc+0x32>
1c017d5a:	57fd                	li	a5,-1
1c017d5c:	c21c                	sw	a5,0(a2)
1c017d5e:	557d                	li	a0,-1
1c017d60:	8082                	ret

1c017d62 <extern_free>:
1c017d62:	7179                	addi	sp,sp,-48
1c017d64:	d422                	sw	s0,40(sp)
1c017d66:	4100                	lw	s0,0(a0)
1c017d68:	059d                	addi	a1,a1,7
1c017d6a:	d226                	sw	s1,36(sp)
1c017d6c:	d04a                	sw	s2,32(sp)
1c017d6e:	ce4e                	sw	s3,28(sp)
1c017d70:	d606                	sw	ra,44(sp)
1c017d72:	89aa                	mv	s3,a0
1c017d74:	c405b933          	p.bclr	s2,a1,2,0
1c017d78:	4481                	li	s1,0
1c017d7a:	c439                	beqz	s0,1c017dc8 <extern_free+0x66>
1c017d7c:	441c                	lw	a5,8(s0)
1c017d7e:	04c7e263          	bltu	a5,a2,1c017dc2 <extern_free+0x60>
1c017d82:	01260733          	add	a4,a2,s2
1c017d86:	04f71163          	bne	a4,a5,1c017dc8 <extern_free+0x66>
1c017d8a:	400c                	lw	a1,0(s0)
1c017d8c:	c410                	sw	a2,8(s0)
1c017d8e:	992e                	add	s2,s2,a1
1c017d90:	01242023          	sw	s2,0(s0)
1c017d94:	c8b1                	beqz	s1,1c017de8 <extern_free+0x86>
1c017d96:	4094                	lw	a3,0(s1)
1c017d98:	4498                	lw	a4,8(s1)
1c017d9a:	9736                	add	a4,a4,a3
1c017d9c:	04e61463          	bne	a2,a4,1c017de4 <extern_free+0x82>
1c017da0:	401c                	lw	a5,0(s0)
1c017da2:	45b1                	li	a1,12
1c017da4:	8522                	mv	a0,s0
1c017da6:	97b6                	add	a5,a5,a3
1c017da8:	c09c                	sw	a5,0(s1)
1c017daa:	405c                	lw	a5,4(s0)
1c017dac:	c0dc                	sw	a5,4(s1)
1c017dae:	696020ef          	jal	ra,1c01a444 <pi_l2_free>
1c017db2:	4501                	li	a0,0
1c017db4:	50b2                	lw	ra,44(sp)
1c017db6:	5422                	lw	s0,40(sp)
1c017db8:	5492                	lw	s1,36(sp)
1c017dba:	5902                	lw	s2,32(sp)
1c017dbc:	49f2                	lw	s3,28(sp)
1c017dbe:	6145                	addi	sp,sp,48
1c017dc0:	8082                	ret
1c017dc2:	84a2                	mv	s1,s0
1c017dc4:	4040                	lw	s0,4(s0)
1c017dc6:	bf55                	j	1c017d7a <extern_free+0x18>
1c017dc8:	4531                	li	a0,12
1c017dca:	c632                	sw	a2,12(sp)
1c017dcc:	772020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c017dd0:	87aa                	mv	a5,a0
1c017dd2:	4632                	lw	a2,12(sp)
1c017dd4:	557d                	li	a0,-1
1c017dd6:	dff9                	beqz	a5,1c017db4 <extern_free+0x52>
1c017dd8:	c3c0                	sw	s0,4(a5)
1c017dda:	0127a023          	sw	s2,0(a5)
1c017dde:	c790                	sw	a2,8(a5)
1c017de0:	843e                	mv	s0,a5
1c017de2:	bf4d                	j	1c017d94 <extern_free+0x32>
1c017de4:	c0c0                	sw	s0,4(s1)
1c017de6:	b7f1                	j	1c017db2 <extern_free+0x50>
1c017de8:	0089a023          	sw	s0,0(s3)
1c017dec:	b7d9                	j	1c017db2 <extern_free+0x50>

1c017dee <cl_wait_task>:
1c017dee:	4691                	li	a3,4
1c017df0:	00204737          	lui	a4,0x204
1c017df4:	00054783          	lbu	a5,0(a0)
1c017df8:	0ff7f793          	andi	a5,a5,255
1c017dfc:	c391                	beqz	a5,1c017e00 <cl_wait_task+0x12>
1c017dfe:	8082                	ret
1c017e00:	00d72423          	sw	a3,8(a4) # 204008 <__l1_heapsram_size+0x1e4078>
1c017e04:	03c76783          	p.elw	a5,60(a4)
1c017e08:	00d72223          	sw	a3,4(a4)
1c017e0c:	b7e5                	j	1c017df4 <cl_wait_task+0x6>

1c017e0e <open_fs>:
1c017e0e:	1141                	addi	sp,sp,-16
1c017e10:	c422                	sw	s0,8(sp)
1c017e12:	1c01d437          	lui	s0,0x1c01d
1c017e16:	64440513          	addi	a0,s0,1604 # 1c01d644 <fs_conf>
1c017e1a:	c606                	sw	ra,12(sp)
1c017e1c:	c226                	sw	s1,4(sp)
1c017e1e:	207000ef          	jal	ra,1c018824 <pi_readfs_conf_init>
1c017e22:	1c01d7b7          	lui	a5,0x1c01d
1c017e26:	64440493          	addi	s1,s0,1604
1c017e2a:	5e478793          	addi	a5,a5,1508 # 1c01d5e4 <flash>
1c017e2e:	c0dc                	sw	a5,4(s1)
1c017e30:	1c01d4b7          	lui	s1,0x1c01d
1c017e34:	63848513          	addi	a0,s1,1592 # 1c01d638 <fs>
1c017e38:	64440593          	addi	a1,s0,1604
1c017e3c:	b1ffc0ef          	jal	ra,1c01495a <pi_open_from_conf>
1c017e40:	63848513          	addi	a0,s1,1592
1c017e44:	24b9                	jal	1c018092 <pi_fs_mount>
1c017e46:	c911                	beqz	a0,1c017e5a <open_fs+0x4c>
1c017e48:	1c01c537          	lui	a0,0x1c01c
1c017e4c:	cd050513          	addi	a0,a0,-816 # 1c01bcd0 <__clz_tab+0x288>
1c017e50:	02d030ef          	jal	ra,1c01b67c <printf_>
1c017e54:	5579                	li	a0,-2
1c017e56:	15e020ef          	jal	ra,1c019fb4 <exit>
1c017e5a:	40b2                	lw	ra,12(sp)
1c017e5c:	4422                	lw	s0,8(sp)
1c017e5e:	4492                	lw	s1,4(sp)
1c017e60:	0141                	addi	sp,sp,16
1c017e62:	8082                	ret

1c017e64 <mem_init>:
1c017e64:	1141                	addi	sp,sp,-16
1c017e66:	c226                	sw	s1,4(sp)
1c017e68:	1c01d4b7          	lui	s1,0x1c01d
1c017e6c:	5f048513          	addi	a0,s1,1520 # 1c01d5f0 <flash_conf>
1c017e70:	c606                	sw	ra,12(sp)
1c017e72:	c422                	sw	s0,8(sp)
1c017e74:	de9fe0ef          	jal	ra,1c016c5c <pi_mx25u51245g_conf_init>
1c017e78:	1c01d437          	lui	s0,0x1c01d
1c017e7c:	5e440513          	addi	a0,s0,1508 # 1c01d5e4 <flash>
1c017e80:	5f048593          	addi	a1,s1,1520
1c017e84:	ad7fc0ef          	jal	ra,1c01495a <pi_open_from_conf>
1c017e88:	5e440513          	addi	a0,s0,1508
1c017e8c:	e0dff0ef          	jal	ra,1c017c98 <pi_flash_open>
1c017e90:	c911                	beqz	a0,1c017ea4 <mem_init+0x40>
1c017e92:	1c01c537          	lui	a0,0x1c01c
1c017e96:	c8450513          	addi	a0,a0,-892 # 1c01bc84 <__clz_tab+0x23c>
1c017e9a:	7e2030ef          	jal	ra,1c01b67c <printf_>
1c017e9e:	557d                	li	a0,-1
1c017ea0:	114020ef          	jal	ra,1c019fb4 <exit>
1c017ea4:	1c01d437          	lui	s0,0x1c01d
1c017ea8:	66440513          	addi	a0,s0,1636 # 1c01d664 <ram_conf>
1c017eac:	8aaff0ef          	jal	ra,1c016f56 <pi_aps256xxn_conf_init>
1c017eb0:	66440593          	addi	a1,s0,1636
1c017eb4:	1c01e437          	lui	s0,0x1c01e
1c017eb8:	b2c40513          	addi	a0,s0,-1236 # 1c01db2c <ram>
1c017ebc:	a9ffc0ef          	jal	ra,1c01495a <pi_open_from_conf>
1c017ec0:	b2c40513          	addi	a0,s0,-1236
1c017ec4:	b7bfc0ef          	jal	ra,1c014a3e <pi_ram_open>
1c017ec8:	c909                	beqz	a0,1c017eda <mem_init+0x76>
1c017eca:	1c01c537          	lui	a0,0x1c01c
1c017ece:	cac50513          	addi	a0,a0,-852 # 1c01bcac <__clz_tab+0x264>
1c017ed2:	7aa030ef          	jal	ra,1c01b67c <printf_>
1c017ed6:	5575                	li	a0,-3
1c017ed8:	b7e1                	j	1c017ea0 <mem_init+0x3c>
1c017eda:	40b2                	lw	ra,12(sp)
1c017edc:	4422                	lw	s0,8(sp)
1c017ede:	4492                	lw	s1,4(sp)
1c017ee0:	0141                	addi	sp,sp,16
1c017ee2:	8082                	ret

1c017ee4 <get_ram_ptr>:
1c017ee4:	1c01e537          	lui	a0,0x1c01e
1c017ee8:	b2c50513          	addi	a0,a0,-1236 # 1c01db2c <ram>
1c017eec:	8082                	ret

1c017eee <ram_read>:
1c017eee:	7119                	addi	sp,sp,-128
1c017ef0:	dca2                	sw	s0,120(sp)
1c017ef2:	842a                	mv	s0,a0
1c017ef4:	0828                	addi	a0,sp,24
1c017ef6:	de86                	sw	ra,124(sp)
1c017ef8:	c62e                	sw	a1,12(sp)
1c017efa:	c432                	sw	a2,8(sp)
1c017efc:	7df010ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c017f00:	1c01e837          	lui	a6,0x1c01e
1c017f04:	b2c82783          	lw	a5,-1236(a6) # 1c01db2c <ram>
1c017f08:	46a2                	lw	a3,8(sp)
1c017f0a:	45b2                	lw	a1,12(sp)
1c017f0c:	0087a883          	lw	a7,8(a5)
1c017f10:	8622                	mv	a2,s0
1c017f12:	87aa                	mv	a5,a0
1c017f14:	4705                	li	a4,1
1c017f16:	b2c80513          	addi	a0,a6,-1236
1c017f1a:	9882                	jalr	a7
1c017f1c:	0828                	addi	a0,sp,24
1c017f1e:	7ff010ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c017f22:	50f6                	lw	ra,124(sp)
1c017f24:	5466                	lw	s0,120(sp)
1c017f26:	6109                	addi	sp,sp,128
1c017f28:	8082                	ret

1c017f2a <cl_ram_malloc>:
1c017f2a:	7119                	addi	sp,sp,-128
1c017f2c:	85aa                	mv	a1,a0
1c017f2e:	1c01e537          	lui	a0,0x1c01e
1c017f32:	860a                	mv	a2,sp
1c017f34:	b2c50513          	addi	a0,a0,-1236 # 1c01db2c <ram>
1c017f38:	de86                	sw	ra,124(sp)
1c017f3a:	baffc0ef          	jal	ra,1c014ae8 <pi_cl_ram_alloc>
1c017f3e:	10a8                	addi	a0,sp,104
1c017f40:	eafff0ef          	jal	ra,1c017dee <cl_wait_task>
1c017f44:	50f6                	lw	ra,124(sp)
1c017f46:	4512                	lw	a0,4(sp)
1c017f48:	6109                	addi	sp,sp,128
1c017f4a:	8082                	ret

1c017f4c <cl_ram_write>:
1c017f4c:	7175                	addi	sp,sp,-144
1c017f4e:	86b2                	mv	a3,a2
1c017f50:	862e                	mv	a2,a1
1c017f52:	85aa                	mv	a1,a0
1c017f54:	1c01e537          	lui	a0,0x1c01e
1c017f58:	003c                	addi	a5,sp,8
1c017f5a:	4701                	li	a4,0
1c017f5c:	b2c50513          	addi	a0,a0,-1236 # 1c01db2c <ram>
1c017f60:	c706                	sw	ra,140(sp)
1c017f62:	ae9fc0ef          	jal	ra,1c014a4a <pi_cl_ram_copy>
1c017f66:	18e8                	addi	a0,sp,124
1c017f68:	e87ff0ef          	jal	ra,1c017dee <cl_wait_task>
1c017f6c:	40ba                	lw	ra,140(sp)
1c017f6e:	6149                	addi	sp,sp,144
1c017f70:	8082                	ret

1c017f72 <load_file_to_ram>:
1c017f72:	715d                	addi	sp,sp,-80
1c017f74:	d85a                	sw	s6,48(sp)
1c017f76:	8b2a                	mv	s6,a0
1c017f78:	1c01d537          	lui	a0,0x1c01d
1c017f7c:	4601                	li	a2,0
1c017f7e:	63850513          	addi	a0,a0,1592 # 1c01d638 <fs>
1c017f82:	c4a2                	sw	s0,72(sp)
1c017f84:	c686                	sw	ra,76(sp)
1c017f86:	c2a6                	sw	s1,68(sp)
1c017f88:	c0ca                	sw	s2,64(sp)
1c017f8a:	de4e                	sw	s3,60(sp)
1c017f8c:	dc52                	sw	s4,56(sp)
1c017f8e:	da56                	sw	s5,52(sp)
1c017f90:	842e                	mv	s0,a1
1c017f92:	2a39                	jal	1c0180b0 <pi_fs_open>
1c017f94:	e919                	bnez	a0,1c017faa <load_file_to_ram+0x38>
1c017f96:	1c01c537          	lui	a0,0x1c01c
1c017f9a:	85a2                	mv	a1,s0
1c017f9c:	c5c50513          	addi	a0,a0,-932 # 1c01bc5c <__clz_tab+0x214>
1c017fa0:	6dc030ef          	jal	ra,1c01b67c <printf_>
1c017fa4:	5571                	li	a0,-4
1c017fa6:	00e020ef          	jal	ra,1c019fb4 <exit>
1c017faa:	00c52a03          	lw	s4,12(a0)
1c017fae:	6985                	lui	s3,0x1
1c017fb0:	892a                	mv	s2,a0
1c017fb2:	4401                	li	s0,0
1c017fb4:	80098993          	addi	s3,s3,-2048 # 800 <__STACK_IRQ_SIZE>
1c017fb8:	1c01dab7          	lui	s5,0x1c01d
1c017fbc:	408a04b3          	sub	s1,s4,s0
1c017fc0:	0534d4b3          	p.minu	s1,s1,s3
1c017fc4:	0034                	addi	a3,sp,8
1c017fc6:	8626                	mv	a2,s1
1c017fc8:	de4a8593          	addi	a1,s5,-540 # 1c01cde4 <buffer>
1c017fcc:	854a                	mv	a0,s2
1c017fce:	2a69                	jal	1c018168 <pi_cl_fs_read>
1c017fd0:	0828                	addi	a0,sp,24
1c017fd2:	e1dff0ef          	jal	ra,1c017dee <cl_wait_task>
1c017fd6:	008b0533          	add	a0,s6,s0
1c017fda:	8626                	mv	a2,s1
1c017fdc:	de4a8593          	addi	a1,s5,-540
1c017fe0:	9426                	add	s0,s0,s1
1c017fe2:	f6bff0ef          	jal	ra,1c017f4c <cl_ram_write>
1c017fe6:	fd446be3          	bltu	s0,s4,1c017fbc <load_file_to_ram+0x4a>
1c017fea:	8522                	mv	a0,s0
1c017fec:	40b6                	lw	ra,76(sp)
1c017fee:	4426                	lw	s0,72(sp)
1c017ff0:	4496                	lw	s1,68(sp)
1c017ff2:	4906                	lw	s2,64(sp)
1c017ff4:	59f2                	lw	s3,60(sp)
1c017ff6:	5a62                	lw	s4,56(sp)
1c017ff8:	5ad2                	lw	s5,52(sp)
1c017ffa:	5b42                	lw	s6,48(sp)
1c017ffc:	6161                	addi	sp,sp,80
1c017ffe:	8082                	ret

1c018000 <__native_cluster_id>:
1c018000:	01402573          	csrr	a0,uhartid
1c018004:	8515                	srai	a0,a0,0x5
1c018006:	f2653533          	p.bclr	a0,a0,25,6
1c01800a:	8082                	ret

1c01800c <cl_notify_task_done>:
1c01800c:	300477f3          	csrrci	a5,mstatus,8
1c018010:	4705                	li	a4,1
1c018012:	00e50023          	sb	a4,0(a0)
1c018016:	04058593          	addi	a1,a1,64
1c01801a:	00201737          	lui	a4,0x201
1c01801e:	e0870713          	addi	a4,a4,-504 # 200e08 <__l1_heapsram_size+0x1e0e78>
1c018022:	05da                	slli	a1,a1,0x16
1c018024:	95ba                	add	a1,a1,a4
1c018026:	0005a023          	sw	zero,0(a1)
1c01802a:	30079073          	csrw	mstatus,a5
1c01802e:	8082                	ret

1c018030 <pi_cl_send_callback_to_fc>:
1c018030:	c0054533          	p.bset	a0,a0,0,0
1c018034:	84bfc06f          	j	1c01487e <pi_cl_send_task_to_fc>

1c018038 <__pi_cl_fs_req>:
1c018038:	411c                	lw	a5,0(a0)
1c01803a:	4b9c                	lw	a5,16(a5)
1c01803c:	4398                	lw	a4,0(a5)
1c01803e:	eb0d                	bnez	a4,1c018070 <__pi_cl_fs_req+0x38>
1c018040:	c388                	sw	a0,0(a5)
1c018042:	c3c8                	sw	a0,4(a5)
1c018044:	00052223          	sw	zero,4(a0)
1c018048:	e71d                	bnez	a4,1c018076 <__pi_cl_fs_req+0x3e>
1c01804a:	1141                	addi	sp,sp,-16
1c01804c:	c422                	sw	s0,8(sp)
1c01804e:	1c0185b7          	lui	a1,0x1c018
1c018052:	00878413          	addi	s0,a5,8
1c018056:	862a                	mv	a2,a0
1c018058:	0d058593          	addi	a1,a1,208 # 1c0180d0 <__pi_cl_fs_req_exec>
1c01805c:	8522                	mv	a0,s0
1c01805e:	c606                	sw	ra,12(sp)
1c018060:	6a7010ef          	jal	ra,1c019f06 <__pi_os_evt_callback_irq_init>
1c018064:	8522                	mv	a0,s0
1c018066:	4422                	lw	s0,8(sp)
1c018068:	40b2                	lw	ra,12(sp)
1c01806a:	0141                	addi	sp,sp,16
1c01806c:	6b50106f          	j	1c019f20 <__pi_os_evt_push>
1c018070:	43d4                	lw	a3,4(a5)
1c018072:	c2c8                	sw	a0,4(a3)
1c018074:	b7f9                	j	1c018042 <__pi_cl_fs_req+0xa>
1c018076:	8082                	ret

1c018078 <pi_fs_conf_init>:
1c018078:	57fd                	li	a5,-1
1c01807a:	00052023          	sw	zero,0(a0)
1c01807e:	00052423          	sw	zero,8(a0)
1c018082:	00052623          	sw	zero,12(a0)
1c018086:	c91c                	sw	a5,16(a0)
1c018088:	00050c23          	sb	zero,24(a0)
1c01808c:	00052e23          	sw	zero,28(a0)
1c018090:	8082                	ret

1c018092 <pi_fs_mount>:
1c018092:	415c                	lw	a5,4(a0)
1c018094:	4398                	lw	a4,0(a5)
1c018096:	c709                	beqz	a4,1c0180a0 <pi_fs_mount+0xe>
1c018098:	4fdc                	lw	a5,28(a5)
1c01809a:	e799                	bnez	a5,1c0180a8 <pi_fs_mount+0x16>
1c01809c:	557d                	li	a0,-1
1c01809e:	8082                	ret
1c0180a0:	1c01c7b7          	lui	a5,0x1c01c
1c0180a4:	57478793          	addi	a5,a5,1396 # 1c01c574 <__pi_readfs_api>
1c0180a8:	0007a303          	lw	t1,0(a5)
1c0180ac:	c11c                	sw	a5,0(a0)
1c0180ae:	8302                	jr	t1

1c0180b0 <pi_fs_open>:
1c0180b0:	411c                	lw	a5,0(a0)
1c0180b2:	0207a303          	lw	t1,32(a5)
1c0180b6:	8302                	jr	t1

1c0180b8 <pi_fs_read_async>:
1c0180b8:	415c                	lw	a5,4(a0)
1c0180ba:	0287a303          	lw	t1,40(a5)
1c0180be:	8302                	jr	t1

1c0180c0 <pi_fs_write_async>:
1c0180c0:	415c                	lw	a5,4(a0)
1c0180c2:	0347a303          	lw	t1,52(a5)
1c0180c6:	8302                	jr	t1

1c0180c8 <pi_fs_direct_read_async>:
1c0180c8:	415c                	lw	a5,4(a0)
1c0180ca:	02c7a303          	lw	t1,44(a5)
1c0180ce:	8302                	jr	t1

1c0180d0 <__pi_cl_fs_req_exec>:
1c0180d0:	1101                	addi	sp,sp,-32
1c0180d2:	ca26                	sw	s1,20(sp)
1c0180d4:	4104                	lw	s1,0(a0)
1c0180d6:	1c0185b7          	lui	a1,0x1c018
1c0180da:	862a                	mv	a2,a0
1c0180dc:	4894                	lw	a3,16(s1)
1c0180de:	cc22                	sw	s0,24(sp)
1c0180e0:	12658593          	addi	a1,a1,294 # 1c018126 <__pi_cl_fs_req_done>
1c0180e4:	06a1                	addi	a3,a3,8
1c0180e6:	842a                	mv	s0,a0
1c0180e8:	8536                	mv	a0,a3
1c0180ea:	c636                	sw	a3,12(sp)
1c0180ec:	ce06                	sw	ra,28(sp)
1c0180ee:	619010ef          	jal	ra,1c019f06 <__pi_os_evt_callback_irq_init>
1c0180f2:	01444783          	lbu	a5,20(s0)
1c0180f6:	4c50                	lw	a2,28(s0)
1c0180f8:	4c0c                	lw	a1,24(s0)
1c0180fa:	46b2                	lw	a3,12(sp)
1c0180fc:	cb99                	beqz	a5,1c018112 <__pi_cl_fs_req_exec+0x42>
1c0180fe:	8526                	mv	a0,s1
1c018100:	fc1ff0ef          	jal	ra,1c0180c0 <pi_fs_write_async>
1c018104:	00a408a3          	sb	a0,17(s0)
1c018108:	40f2                	lw	ra,28(sp)
1c01810a:	4462                	lw	s0,24(sp)
1c01810c:	44d2                	lw	s1,20(sp)
1c01810e:	6105                	addi	sp,sp,32
1c018110:	8082                	ret
1c018112:	01344783          	lbu	a5,19(s0)
1c018116:	8526                	mv	a0,s1
1c018118:	c781                	beqz	a5,1c018120 <__pi_cl_fs_req_exec+0x50>
1c01811a:	fafff0ef          	jal	ra,1c0180c8 <pi_fs_direct_read_async>
1c01811e:	b7dd                	j	1c018104 <__pi_cl_fs_req_exec+0x34>
1c018120:	f99ff0ef          	jal	ra,1c0180b8 <pi_fs_read_async>
1c018124:	b7c5                	j	1c018104 <__pi_cl_fs_req_exec+0x34>

1c018126 <__pi_cl_fs_req_done>:
1c018126:	411c                	lw	a5,0(a0)
1c018128:	1141                	addi	sp,sp,-16
1c01812a:	c226                	sw	s1,4(sp)
1c01812c:	4b84                	lw	s1,16(a5)
1c01812e:	01254583          	lbu	a1,18(a0)
1c018132:	c422                	sw	s0,8(sp)
1c018134:	509c                	lw	a5,32(s1)
1c018136:	c606                	sw	ra,12(sp)
1c018138:	842a                	mv	s0,a0
1c01813a:	00f508a3          	sb	a5,17(a0)
1c01813e:	0541                	addi	a0,a0,16
1c018140:	ecdff0ef          	jal	ra,1c01800c <cl_notify_task_done>
1c018144:	300477f3          	csrrci	a5,mstatus,8
1c018148:	4048                	lw	a0,4(s0)
1c01814a:	c088                	sw	a0,0(s1)
1c01814c:	30079073          	csrw	mstatus,a5
1c018150:	c519                	beqz	a0,1c01815e <__pi_cl_fs_req_done+0x38>
1c018152:	4422                	lw	s0,8(sp)
1c018154:	40b2                	lw	ra,12(sp)
1c018156:	4492                	lw	s1,4(sp)
1c018158:	0141                	addi	sp,sp,16
1c01815a:	f77ff06f          	j	1c0180d0 <__pi_cl_fs_req_exec>
1c01815e:	40b2                	lw	ra,12(sp)
1c018160:	4422                	lw	s0,8(sp)
1c018162:	4492                	lw	s1,4(sp)
1c018164:	0141                	addi	sp,sp,16
1c018166:	8082                	ret

1c018168 <pi_cl_fs_read>:
1c018168:	1101                	addi	sp,sp,-32
1c01816a:	ce06                	sw	ra,28(sp)
1c01816c:	c288                	sw	a0,0(a3)
1c01816e:	ce8c                	sw	a1,24(a3)
1c018170:	ced0                	sw	a2,28(a3)
1c018172:	c636                	sw	a3,12(sp)
1c018174:	e8dff0ef          	jal	ra,1c018000 <__native_cluster_id>
1c018178:	46b2                	lw	a3,12(sp)
1c01817a:	f0000793          	li	a5,-256
1c01817e:	00f69823          	sh	a5,16(a3)
1c018182:	1c0187b7          	lui	a5,0x1c018
1c018186:	40f2                	lw	ra,28(sp)
1c018188:	03878793          	addi	a5,a5,56 # 1c018038 <__pi_cl_fs_req>
1c01818c:	00a68923          	sb	a0,18(a3)
1c018190:	000699a3          	sh	zero,19(a3)
1c018194:	c69c                	sw	a5,8(a3)
1c018196:	c6d4                	sw	a3,12(a3)
1c018198:	0006a223          	sw	zero,4(a3)
1c01819c:	00468513          	addi	a0,a3,4
1c0181a0:	6105                	addi	sp,sp,32
1c0181a2:	e8fff06f          	j	1c018030 <pi_cl_send_callback_to_fc>

1c0181a6 <pi_flash_read_async>:
1c0181a6:	411c                	lw	a5,0(a0)
1c0181a8:	00c7a303          	lw	t1,12(a5)
1c0181ac:	8302                	jr	t1

1c0181ae <__pi_read_fs_copy_async>:
1c0181ae:	882a                	mv	a6,a0
1c0181b0:	4108                	lw	a0,0(a0)
1c0181b2:	00882803          	lw	a6,8(a6)
1c0181b6:	4508                	lw	a0,8(a0)
1c0181b8:	00482803          	lw	a6,4(a6)
1c0181bc:	4148                	lw	a0,4(a0)
1c0181be:	95c2                	add	a1,a1,a6
1c0181c0:	00052883          	lw	a7,0(a0)
1c0181c4:	0288a303          	lw	t1,40(a7)
1c0181c8:	8302                	jr	t1

1c0181ca <__pi_read_fs_copy_2d_async>:
1c0181ca:	832a                	mv	t1,a0
1c0181cc:	4108                	lw	a0,0(a0)
1c0181ce:	00832303          	lw	t1,8(t1)
1c0181d2:	4508                	lw	a0,8(a0)
1c0181d4:	00432e03          	lw	t3,4(t1)
1c0181d8:	4148                	lw	a0,4(a0)
1c0181da:	95f2                	add	a1,a1,t3
1c0181dc:	00052303          	lw	t1,0(a0)
1c0181e0:	02c32303          	lw	t1,44(t1)
1c0181e4:	8302                	jr	t1

1c0181e6 <__pi_read_fs_mkdir>:
1c0181e6:	fdb00513          	li	a0,-37
1c0181ea:	8082                	ret

1c0181ec <__pi_read_fs_dir_open>:
1c0181ec:	4501                	li	a0,0
1c0181ee:	8082                	ret

1c0181f0 <__pi_read_fs_dir_close>:
1c0181f0:	8082                	ret

1c0181f2 <__pi_read_fs_write>:
1c0181f2:	411c                	lw	a5,0(a0)
1c0181f4:	1141                	addi	sp,sp,-16
1c0181f6:	8736                	mv	a4,a3
1c0181f8:	c606                	sw	ra,12(sp)
1c0181fa:	832e                	mv	t1,a1
1c0181fc:	4794                	lw	a3,8(a5)
1c0181fe:	00852883          	lw	a7,8(a0)
1c018202:	30047e73          	csrrci	t3,mstatus,8
1c018206:	0008a783          	lw	a5,0(a7)
1c01820a:	0048a803          	lw	a6,4(a7)
1c01820e:	4548                	lw	a0,12(a0)
1c018210:	010785b3          	add	a1,a5,a6
1c018214:	00c78833          	add	a6,a5,a2
1c018218:	01057463          	bleu	a6,a0,1c018220 <__pi_read_fs_write+0x2e>
1c01821c:	40f50633          	sub	a2,a0,a5
1c018220:	97b2                	add	a5,a5,a2
1c018222:	00f8a023          	sw	a5,0(a7)
1c018226:	300e1073          	csrw	mstatus,t3
1c01822a:	42c8                	lw	a0,4(a3)
1c01822c:	86b2                	mv	a3,a2
1c01822e:	861a                	mv	a2,t1
1c018230:	411c                	lw	a5,0(a0)
1c018232:	4b9c                	lw	a5,16(a5)
1c018234:	9782                	jalr	a5
1c018236:	40b2                	lw	ra,12(sp)
1c018238:	4501                	li	a0,0
1c01823a:	0141                	addi	sp,sp,16
1c01823c:	8082                	ret

1c01823e <__pi_read_fs_seek>:
1c01823e:	4518                	lw	a4,8(a0)
1c018240:	300477f3          	csrrci	a5,mstatus,8
1c018244:	4554                	lw	a3,12(a0)
1c018246:	00d5f763          	bleu	a3,a1,1c018254 <__pi_read_fs_seek+0x16>
1c01824a:	c30c                	sw	a1,0(a4)
1c01824c:	30079073          	csrw	mstatus,a5
1c018250:	4501                	li	a0,0
1c018252:	8082                	ret
1c018254:	30079073          	csrw	mstatus,a5
1c018258:	557d                	li	a0,-1
1c01825a:	8082                	ret

1c01825c <__pi_read_fs_seek_and_read>:
1c01825c:	411c                	lw	a5,0(a0)
1c01825e:	7179                	addi	sp,sp,-48
1c018260:	d422                	sw	s0,40(sp)
1c018262:	d226                	sw	s1,36(sp)
1c018264:	d04a                	sw	s2,32(sp)
1c018266:	cc52                	sw	s4,24(sp)
1c018268:	d606                	sw	ra,44(sp)
1c01826a:	ce4e                	sw	s3,28(sp)
1c01826c:	892a                	mv	s2,a0
1c01826e:	8436                	mv	s0,a3
1c018270:	4504                	lw	s1,8(a0)
1c018272:	0087aa03          	lw	s4,8(a5)
1c018276:	300479f3          	csrrci	s3,mstatus,8
1c01827a:	c632                	sw	a2,12(sp)
1c01827c:	c43a                	sw	a4,8(sp)
1c01827e:	fc1ff0ef          	jal	ra,1c01823e <__pi_read_fs_seek>
1c018282:	4632                	lw	a2,12(sp)
1c018284:	4722                	lw	a4,8(sp)
1c018286:	cd09                	beqz	a0,1c0182a0 <__pi_read_fs_seek_and_read+0x44>
1c018288:	30099073          	csrw	mstatus,s3
1c01828c:	547d                	li	s0,-1
1c01828e:	8522                	mv	a0,s0
1c018290:	50b2                	lw	ra,44(sp)
1c018292:	5422                	lw	s0,40(sp)
1c018294:	5492                	lw	s1,36(sp)
1c018296:	5902                	lw	s2,32(sp)
1c018298:	49f2                	lw	s3,28(sp)
1c01829a:	4a62                	lw	s4,24(sp)
1c01829c:	6145                	addi	sp,sp,48
1c01829e:	8082                	ret
1c0182a0:	409c                	lw	a5,0(s1)
1c0182a2:	00c92683          	lw	a3,12(s2)
1c0182a6:	40cc                	lw	a1,4(s1)
1c0182a8:	00878533          	add	a0,a5,s0
1c0182ac:	00d56463          	bltu	a0,a3,1c0182b4 <__pi_read_fs_seek_and_read+0x58>
1c0182b0:	40f68433          	sub	s0,a3,a5
1c0182b4:	008786b3          	add	a3,a5,s0
1c0182b8:	c094                	sw	a3,0(s1)
1c0182ba:	30099073          	csrw	mstatus,s3
1c0182be:	c801                	beqz	s0,1c0182ce <__pi_read_fs_seek_and_read+0x72>
1c0182c0:	004a2503          	lw	a0,4(s4)
1c0182c4:	86a2                	mv	a3,s0
1c0182c6:	95be                	add	a1,a1,a5
1c0182c8:	edfff0ef          	jal	ra,1c0181a6 <pi_flash_read_async>
1c0182cc:	b7c9                	j	1c01828e <__pi_read_fs_seek_and_read+0x32>
1c0182ce:	853a                	mv	a0,a4
1c0182d0:	451010ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c0182d4:	bf6d                	j	1c01828e <__pi_read_fs_seek_and_read+0x32>

1c0182d6 <__pi_read_fs_direct_read_async>:
1c0182d6:	411c                	lw	a5,0(a0)
1c0182d8:	1141                	addi	sp,sp,-16
1c0182da:	c422                	sw	s0,8(sp)
1c0182dc:	c606                	sw	ra,12(sp)
1c0182de:	8432                	mv	s0,a2
1c0182e0:	8736                	mv	a4,a3
1c0182e2:	4510                	lw	a2,8(a0)
1c0182e4:	0087a803          	lw	a6,8(a5)
1c0182e8:	30047373          	csrrci	t1,mstatus,8
1c0182ec:	421c                	lw	a5,0(a2)
1c0182ee:	4554                	lw	a3,12(a0)
1c0182f0:	00462883          	lw	a7,4(a2)
1c0182f4:	00878533          	add	a0,a5,s0
1c0182f8:	00d56463          	bltu	a0,a3,1c018300 <__pi_read_fs_direct_read_async+0x2a>
1c0182fc:	40f68433          	sub	s0,a3,a5
1c018300:	008786b3          	add	a3,a5,s0
1c018304:	c214                	sw	a3,0(a2)
1c018306:	30031073          	csrw	mstatus,t1
1c01830a:	cc11                	beqz	s0,1c018326 <__pi_read_fs_direct_read_async+0x50>
1c01830c:	00482503          	lw	a0,4(a6)
1c018310:	862e                	mv	a2,a1
1c018312:	86a2                	mv	a3,s0
1c018314:	00f885b3          	add	a1,a7,a5
1c018318:	e8fff0ef          	jal	ra,1c0181a6 <pi_flash_read_async>
1c01831c:	8522                	mv	a0,s0
1c01831e:	40b2                	lw	ra,12(sp)
1c018320:	4422                	lw	s0,8(sp)
1c018322:	0141                	addi	sp,sp,16
1c018324:	8082                	ret
1c018326:	853a                	mv	a0,a4
1c018328:	3f9010ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c01832c:	bfc5                	j	1c01831c <__pi_read_fs_direct_read_async+0x46>

1c01832e <__pi_read_fs_close>:
1c01832e:	1141                	addi	sp,sp,-16
1c018330:	c422                	sw	s0,8(sp)
1c018332:	4500                	lw	s0,8(a0)
1c018334:	c226                	sw	s1,4(sp)
1c018336:	c606                	sw	ra,12(sp)
1c018338:	5c30                	lw	a2,120(s0)
1c01833a:	84aa                	mv	s1,a0
1c01833c:	e21d                	bnez	a2,1c018362 <__pi_read_fs_close+0x34>
1c01833e:	5828                	lw	a0,112(s0)
1c018340:	08800593          	li	a1,136
1c018344:	100020ef          	jal	ra,1c01a444 <pi_l2_free>
1c018348:	8522                	mv	a0,s0
1c01834a:	08800593          	li	a1,136
1c01834e:	0f6020ef          	jal	ra,1c01a444 <pi_l2_free>
1c018352:	4422                	lw	s0,8(sp)
1c018354:	40b2                	lw	ra,12(sp)
1c018356:	8526                	mv	a0,s1
1c018358:	4492                	lw	s1,4(sp)
1c01835a:	45e1                	li	a1,24
1c01835c:	0141                	addi	sp,sp,16
1c01835e:	0e60206f          	j	1c01a444 <pi_l2_free>
1c018362:	411c                	lw	a5,0(a0)
1c018364:	4558                	lw	a4,12(a0)
1c018366:	404c                	lw	a1,4(s0)
1c018368:	479c                	lw	a5,8(a5)
1c01836a:	5c74                	lw	a3,124(s0)
1c01836c:	c20c                	sw	a1,0(a2)
1c01836e:	43c8                	lw	a0,4(a5)
1c018370:	8d95                	sub	a1,a1,a3
1c018372:	c258                	sw	a4,4(a2)
1c018374:	411c                	lw	a5,0(a0)
1c018376:	5bdc                	lw	a5,52(a5)
1c018378:	9782                	jalr	a5
1c01837a:	5c6c                	lw	a1,124(s0)
1c01837c:	5c28                	lw	a0,120(s0)
1c01837e:	b7d9                	j	1c018344 <__pi_read_fs_close+0x16>

1c018380 <__pi_fs_free>:
1c018380:	c91d                	beqz	a0,1c0183b6 <__pi_fs_free+0x36>
1c018382:	1141                	addi	sp,sp,-16
1c018384:	c422                	sw	s0,8(sp)
1c018386:	842a                	mv	s0,a0
1c018388:	5d68                	lw	a0,124(a0)
1c01838a:	c606                	sw	ra,12(sp)
1c01838c:	c901                	beqz	a0,1c01839c <__pi_fs_free+0x1c>
1c01838e:	5c3c                	lw	a5,120(s0)
1c018390:	438c                	lw	a1,0(a5)
1c018392:	059d                	addi	a1,a1,7
1c018394:	c405b5b3          	p.bclr	a1,a1,2,0
1c018398:	0ac020ef          	jal	ra,1c01a444 <pi_l2_free>
1c01839c:	5c28                	lw	a0,120(s0)
1c01839e:	c501                	beqz	a0,1c0183a6 <__pi_fs_free+0x26>
1c0183a0:	45a1                	li	a1,8
1c0183a2:	0a2020ef          	jal	ra,1c01a444 <pi_l2_free>
1c0183a6:	8522                	mv	a0,s0
1c0183a8:	4422                	lw	s0,8(sp)
1c0183aa:	40b2                	lw	ra,12(sp)
1c0183ac:	0f000593          	li	a1,240
1c0183b0:	0141                	addi	sp,sp,16
1c0183b2:	0920206f          	j	1c01a444 <pi_l2_free>
1c0183b6:	8082                	ret

1c0183b8 <__pi_read_fs_open>:
1c0183b8:	1101                	addi	sp,sp,-32
1c0183ba:	c84a                	sw	s2,16(sp)
1c0183bc:	c452                	sw	s4,8(sp)
1c0183be:	c256                	sw	s5,4(sp)
1c0183c0:	ce06                	sw	ra,28(sp)
1c0183c2:	cc22                	sw	s0,24(sp)
1c0183c4:	ca26                	sw	s1,20(sp)
1c0183c6:	c64e                	sw	s3,12(sp)
1c0183c8:	c05a                	sw	s6,0(sp)
1c0183ca:	8a2a                	mv	s4,a0
1c0183cc:	8aae                	mv	s5,a1
1c0183ce:	00852903          	lw	s2,8(a0)
1c0183d2:	0a163063          	p.bneimm	a2,1,1c018472 <__pi_read_fs_open+0xba>
1c0183d6:	08892783          	lw	a5,136(s2)
1c0183da:	c399                	beqz	a5,1c0183e0 <__pi_read_fs_open+0x28>
1c0183dc:	4481                	li	s1,0
1c0183de:	a8bd                	j	1c01845c <__pi_read_fs_open+0xa4>
1c0183e0:	4561                	li	a0,24
1c0183e2:	15c020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c0183e6:	84aa                	mv	s1,a0
1c0183e8:	08800513          	li	a0,136
1c0183ec:	152020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c0183f0:	842a                	mv	s0,a0
1c0183f2:	0e048c63          	beqz	s1,1c0184ea <__pi_read_fs_open+0x132>
1c0183f6:	10050463          	beqz	a0,1c0184fe <__pi_read_fs_open+0x146>
1c0183fa:	8556                	mv	a0,s5
1c0183fc:	419010ef          	jal	ra,1c01a014 <strlen>
1c018400:	00750993          	addi	s3,a0,7
1c018404:	c409b9b3          	p.bclr	s3,s3,2,0
1c018408:	09b1                	addi	s3,s3,12
1c01840a:	8b2a                	mv	s6,a0
1c01840c:	854e                	mv	a0,s3
1c01840e:	130020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c018412:	cd79                	beqz	a0,1c0184f0 <__pi_read_fs_open+0x138>
1c018414:	dc28                	sw	a0,120(s0)
1c018416:	07342e23          	sw	s3,124(s0)
1c01841a:	865a                	mv	a2,s6
1c01841c:	85d6                	mv	a1,s5
1c01841e:	0531                	addi	a0,a0,12
1c018420:	39f010ef          	jal	ra,1c019fbe <memcpy>
1c018424:	5c3c                	lw	a5,120(s0)
1c018426:	0167a423          	sw	s6,8(a5)
1c01842a:	08492783          	lw	a5,132(s2)
1c01842e:	99be                	add	s3,s3,a5
1c018430:	01342223          	sw	s3,4(s0)
1c018434:	0004a623          	sw	zero,12(s1)
1c018438:	57fd                	li	a5,-1
1c01843a:	00042023          	sw	zero,0(s0)
1c01843e:	d87c                	sw	a5,116(s0)
1c018440:	08892423          	sw	s0,136(s2)
1c018444:	1c01c7b7          	lui	a5,0x1c01c
1c018448:	57478793          	addi	a5,a5,1396 # 1c01c574 <__pi_readfs_api>
1c01844c:	09090913          	addi	s2,s2,144
1c018450:	c0dc                	sw	a5,4(s1)
1c018452:	c480                	sw	s0,8(s1)
1c018454:	0144a023          	sw	s4,0(s1)
1c018458:	0124a823          	sw	s2,16(s1)
1c01845c:	40f2                	lw	ra,28(sp)
1c01845e:	4462                	lw	s0,24(sp)
1c018460:	8526                	mv	a0,s1
1c018462:	4942                	lw	s2,16(sp)
1c018464:	44d2                	lw	s1,20(sp)
1c018466:	49b2                	lw	s3,12(sp)
1c018468:	4a22                	lw	s4,8(sp)
1c01846a:	4a92                	lw	s5,4(sp)
1c01846c:	4b02                	lw	s6,0(sp)
1c01846e:	6105                	addi	sp,sp,32
1c018470:	8082                	ret
1c018472:	07c92783          	lw	a5,124(s2)
1c018476:	4481                	li	s1,0
1c018478:	4981                	li	s3,0
1c01847a:	0007ab03          	lw	s6,0(a5)
1c01847e:	00478413          	addi	s0,a5,4
1c018482:	0564c563          	blt	s1,s6,1c0184cc <__pi_read_fs_open+0x114>
1c018486:	f5648be3          	beq	s1,s6,1c0183dc <__pi_read_fs_open+0x24>
1c01848a:	4561                	li	a0,24
1c01848c:	0b2020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c018490:	84aa                	mv	s1,a0
1c018492:	08800513          	li	a0,136
1c018496:	0a8020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c01849a:	842a                	mv	s0,a0
1c01849c:	c4b9                	beqz	s1,1c0184ea <__pi_read_fs_open+0x132>
1c01849e:	c125                	beqz	a0,1c0184fe <__pi_read_fs_open+0x146>
1c0184a0:	08800513          	li	a0,136
1c0184a4:	09a020ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c0184a8:	d828                	sw	a0,112(s0)
1c0184aa:	c139                	beqz	a0,1c0184f0 <__pi_read_fs_open+0x138>
1c0184ac:	0049a783          	lw	a5,4(s3)
1c0184b0:	01092703          	lw	a4,16(s2)
1c0184b4:	c4dc                	sw	a5,12(s1)
1c0184b6:	0009a783          	lw	a5,0(s3)
1c0184ba:	06042c23          	sw	zero,120(s0)
1c0184be:	00042023          	sw	zero,0(s0)
1c0184c2:	97ba                	add	a5,a5,a4
1c0184c4:	c05c                	sw	a5,4(s0)
1c0184c6:	57fd                	li	a5,-1
1c0184c8:	d87c                	sw	a5,116(s0)
1c0184ca:	bfad                	j	1c018444 <__pi_read_fs_open+0x8c>
1c0184cc:	85d6                	mv	a1,s5
1c0184ce:	00c40513          	addi	a0,s0,12
1c0184d2:	311010ef          	jal	ra,1c019fe2 <strcmp>
1c0184d6:	c901                	beqz	a0,1c0184e6 <__pi_read_fs_open+0x12e>
1c0184d8:	441c                	lw	a5,8(s0)
1c0184da:	89a2                	mv	s3,s0
1c0184dc:	0485                	addi	s1,s1,1
1c0184de:	97a2                	add	a5,a5,s0
1c0184e0:	00c78413          	addi	s0,a5,12
1c0184e4:	bf79                	j	1c018482 <__pi_read_fs_open+0xca>
1c0184e6:	89a2                	mv	s3,s0
1c0184e8:	b74d                	j	1c01848a <__pi_read_fs_open+0xd2>
1c0184ea:	4481                	li	s1,0
1c0184ec:	ee0408e3          	beqz	s0,1c0183dc <__pi_read_fs_open+0x24>
1c0184f0:	08800593          	li	a1,136
1c0184f4:	8522                	mv	a0,s0
1c0184f6:	74f010ef          	jal	ra,1c01a444 <pi_l2_free>
1c0184fa:	ee0481e3          	beqz	s1,1c0183dc <__pi_read_fs_open+0x24>
1c0184fe:	45e1                	li	a1,24
1c018500:	8526                	mv	a0,s1
1c018502:	743010ef          	jal	ra,1c01a444 <pi_l2_free>
1c018506:	bdd9                	j	1c0183dc <__pi_read_fs_open+0x24>

1c018508 <__pi_read_fs_ls>:
1c018508:	451c                	lw	a5,8(a0)
1c01850a:	7179                	addi	sp,sp,-48
1c01850c:	cc52                	sw	s4,24(sp)
1c01850e:	5ffc                	lw	a5,124(a5)
1c018510:	1c01c537          	lui	a0,0x1c01c
1c018514:	d1850513          	addi	a0,a0,-744 # 1c01bd18 <__clz_tab+0x2d0>
1c018518:	0007aa03          	lw	s4,0(a5)
1c01851c:	d422                	sw	s0,40(sp)
1c01851e:	d226                	sw	s1,36(sp)
1c018520:	8652                	mv	a2,s4
1c018522:	d606                	sw	ra,44(sp)
1c018524:	d04a                	sw	s2,32(sp)
1c018526:	ce4e                	sw	s3,28(sp)
1c018528:	ca56                	sw	s5,20(sp)
1c01852a:	c85a                	sw	s6,16(sp)
1c01852c:	c65e                	sw	s7,12(sp)
1c01852e:	842e                	mv	s0,a1
1c018530:	00478493          	addi	s1,a5,4
1c018534:	148030ef          	jal	ra,1c01b67c <printf_>
1c018538:	8522                	mv	a0,s0
1c01853a:	2db010ef          	jal	ra,1c01a014 <strlen>
1c01853e:	00153963          	p.bneimm	a0,1,1c018550 <__pi_read_fs_ls+0x48>
1c018542:	00044703          	lbu	a4,0(s0)
1c018546:	02e00793          	li	a5,46
1c01854a:	00f71363          	bne	a4,a5,1c018550 <__pi_read_fs_ls+0x48>
1c01854e:	0405                	addi	s0,s0,1
1c018550:	1c01c937          	lui	s2,0x1c01c
1c018554:	d3490513          	addi	a0,s2,-716 # 1c01bd34 <__clz_tab+0x2ec>
1c018558:	124030ef          	jal	ra,1c01b67c <printf_>
1c01855c:	4981                	li	s3,0
1c01855e:	1c01cbb7          	lui	s7,0x1c01c
1c018562:	0349c263          	blt	s3,s4,1c018586 <__pi_read_fs_ls+0x7e>
1c018566:	d3490513          	addi	a0,s2,-716
1c01856a:	112030ef          	jal	ra,1c01b67c <printf_>
1c01856e:	50b2                	lw	ra,44(sp)
1c018570:	5422                	lw	s0,40(sp)
1c018572:	5492                	lw	s1,36(sp)
1c018574:	5902                	lw	s2,32(sp)
1c018576:	49f2                	lw	s3,28(sp)
1c018578:	4a62                	lw	s4,24(sp)
1c01857a:	4ad2                	lw	s5,20(sp)
1c01857c:	4b42                	lw	s6,16(sp)
1c01857e:	4bb2                	lw	s7,12(sp)
1c018580:	4501                	li	a0,0
1c018582:	6145                	addi	sp,sp,48
1c018584:	8082                	ret
1c018586:	8522                	mv	a0,s0
1c018588:	28d010ef          	jal	ra,1c01a014 <strlen>
1c01858c:	00c48b13          	addi	s6,s1,12
1c018590:	8aaa                	mv	s5,a0
1c018592:	862a                	mv	a2,a0
1c018594:	85da                	mv	a1,s6
1c018596:	8522                	mv	a0,s0
1c018598:	25f010ef          	jal	ra,1c019ff6 <strncmp>
1c01859c:	e901                	bnez	a0,1c0185ac <__pi_read_fs_ls+0xa4>
1c01859e:	40cc                	lw	a1,4(s1)
1c0185a0:	015b0633          	add	a2,s6,s5
1c0185a4:	d3cb8513          	addi	a0,s7,-708 # 1c01bd3c <__clz_tab+0x2f4>
1c0185a8:	0d4030ef          	jal	ra,1c01b67c <printf_>
1c0185ac:	449c                	lw	a5,8(s1)
1c0185ae:	0985                	addi	s3,s3,1
1c0185b0:	94be                	add	s1,s1,a5
1c0185b2:	04b1                	addi	s1,s1,12
1c0185b4:	b77d                	j	1c018562 <__pi_read_fs_ls+0x5a>

1c0185b6 <__pi_read_fs_unmount>:
1c0185b6:	1141                	addi	sp,sp,-16
1c0185b8:	c422                	sw	s0,8(sp)
1c0185ba:	4500                	lw	s0,8(a0)
1c0185bc:	c606                	sw	ra,12(sp)
1c0185be:	4008                	lw	a0,0(s0)
1c0185c0:	a81fe0ef          	jal	ra,1c017040 <pi_close>
1c0185c4:	8522                	mv	a0,s0
1c0185c6:	4422                	lw	s0,8(sp)
1c0185c8:	40b2                	lw	ra,12(sp)
1c0185ca:	0141                	addi	sp,sp,16
1c0185cc:	db5ff06f          	j	1c018380 <__pi_fs_free>

1c0185d0 <__pi_read_fs_remove>:
1c0185d0:	fdb00513          	li	a0,-37
1c0185d4:	8082                	ret

1c0185d6 <__pi_read_fs_dir_read>:
1c0185d6:	fdb00513          	li	a0,-37
1c0185da:	8082                	ret

1c0185dc <__pi_fs_mount_step>:
1c0185dc:	597c                	lw	a5,116(a0)
1c0185de:	711d                	addi	sp,sp,-96
1c0185e0:	cca2                	sw	s0,88(sp)
1c0185e2:	ce86                	sw	ra,92(sp)
1c0185e4:	caa6                	sw	s1,84(sp)
1c0185e6:	c8ca                	sw	s2,80(sp)
1c0185e8:	842a                	mv	s0,a0
1c0185ea:	1227ac63          	p.beqimm	a5,2,1c018722 <__pi_fs_mount_step+0x146>
1c0185ee:	1637aa63          	p.beqimm	a5,3,1c018762 <__pi_fs_mount_step+0x186>
1c0185f2:	0617bb63          	p.bneimm	a5,1,1c018668 <__pi_fs_mount_step+0x8c>
1c0185f6:	01050793          	addi	a5,a0,16
1c0185fa:	1a050263          	beqz	a0,1c01879e <__pi_fs_mount_step+0x1c2>
1c0185fe:	1a078063          	beqz	a5,1c01879e <__pi_fs_mount_step+0x1c2>
1c018602:	1028                	addi	a0,sp,40
1c018604:	f69fc0ef          	jal	ra,1c01556c <pi_fpv2_ptable_conf_init>
1c018608:	7abf6537          	lui	a0,0x7abf6
1c01860c:	4601                	li	a2,0
1c01860e:	106c                	addi	a1,sp,44
1c018610:	84950513          	addi	a0,a0,-1975 # 7abf5849 <__heapl2ram_size+0x5ea65849>
1c018614:	dd9fc0ef          	jal	ra,1c0153ec <pi_fpv2_ptable_offset_get>
1c018618:	18051363          	bnez	a0,1c01879e <__pi_fs_mount_step+0x1c2>
1c01861c:	102c                	addi	a1,sp,40
1c01861e:	0868                	addi	a0,sp,28
1c018620:	ce02                	sw	zero,28(sp)
1c018622:	fa5fc0ef          	jal	ra,1c0155c6 <pi_fpv2_ptable_load>
1c018626:	c539                	beqz	a0,1c018674 <__pi_fs_mount_step+0x98>
1c018628:	448d                	li	s1,3
1c01862a:	5502                	lw	a0,32(sp)
1c01862c:	c119                	beqz	a0,1c018632 <__pi_fs_mount_step+0x56>
1c01862e:	910fd0ef          	jal	ra,1c01573e <pi_fpv2_vtable_close>
1c018632:	4572                	lw	a0,28(sp)
1c018634:	c119                	beqz	a0,1c01863a <__pi_fs_mount_step+0x5e>
1c018636:	f4ffc0ef          	jal	ra,1c015584 <pi_fpv2_ptable_close>
1c01863a:	16049263          	bnez	s1,1c01879e <__pi_fs_mount_step+0x1c2>
1c01863e:	587c                	lw	a5,116(s0)
1c018640:	01840713          	addi	a4,s0,24
1c018644:	1c0185b7          	lui	a1,0x1c018
1c018648:	0785                	addi	a5,a5,1
1c01864a:	d87c                	sw	a5,116(s0)
1c01864c:	8622                	mv	a2,s0
1c01864e:	5dc58593          	addi	a1,a1,1500 # 1c0185dc <__pi_fs_mount_step>
1c018652:	853a                	mv	a0,a4
1c018654:	c63a                	sw	a4,12(sp)
1c018656:	09b010ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c01865a:	4732                	lw	a4,12(sp)
1c01865c:	5c30                	lw	a2,120(s0)
1c01865e:	480c                	lw	a1,16(s0)
1c018660:	46a1                	li	a3,8
1c018662:	4048                	lw	a0,4(s0)
1c018664:	b43ff0ef          	jal	ra,1c0181a6 <pi_flash_read_async>
1c018668:	40f6                	lw	ra,92(sp)
1c01866a:	4466                	lw	s0,88(sp)
1c01866c:	44d6                	lw	s1,84(sp)
1c01866e:	4946                	lw	s2,80(sp)
1c018670:	6125                	addi	sp,sp,96
1c018672:	8082                	ret
1c018674:	4572                	lw	a0,28(sp)
1c018676:	100c                	addi	a1,sp,32
1c018678:	d002                	sw	zero,32(sp)
1c01867a:	93afd0ef          	jal	ra,1c0157b4 <pi_fpv2_vtable_load>
1c01867e:	f54d                	bnez	a0,1c018628 <__pi_fs_mount_step+0x4c>
1c018680:	440c                	lw	a1,8(s0)
1c018682:	d202                	sw	zero,36(sp)
1c018684:	5502                	lw	a0,32(sp)
1c018686:	e1c9                	bnez	a1,1c018708 <__pi_fs_mount_step+0x12c>
1c018688:	104c                	addi	a1,sp,36
1c01868a:	b1cfd0ef          	jal	ra,1c0159a6 <pi_fpv2_vtable_active_application_volume_get>
1c01868e:	e93d                	bnez	a0,1c018704 <__pi_fs_mount_step+0x128>
1c018690:	445c                	lw	a5,12(s0)
1c018692:	ef99                	bnez	a5,1c0186b0 <__pi_fs_mount_step+0xd4>
1c018694:	405c                	lw	a5,4(s0)
1c018696:	cf89                	beqz	a5,1c0186b0 <__pi_fs_mount_step+0xd4>
1c018698:	4398                	lw	a4,0(a5)
1c01869a:	1c01c7b7          	lui	a5,0x1c01c
1c01869e:	47078793          	addi	a5,a5,1136 # 1c01c470 <mram_api>
1c0186a2:	06f71763          	bne	a4,a5,1c018710 <__pi_fs_mount_step+0x134>
1c0186a6:	1c01c7b7          	lui	a5,0x1c01c
1c0186aa:	cfc78793          	addi	a5,a5,-772 # 1c01bcfc <__clz_tab+0x2b4>
1c0186ae:	c45c                	sw	a5,12(s0)
1c0186b0:	444c                	lw	a1,12(s0)
1c0186b2:	5512                	lw	a0,36(sp)
1c0186b4:	e1bd                	bnez	a1,1c01871a <__pi_fs_mount_step+0x13e>
1c0186b6:	0834                	addi	a3,sp,24
1c0186b8:	08100613          	li	a2,129
1c0186bc:	4585                	li	a1,1
1c0186be:	bdcfd0ef          	jal	ra,1c015a9a <pi_fpv2_volume_partition_first_get_by_type>
1c0186c2:	e129                	bnez	a0,1c018704 <__pi_fs_mount_step+0x128>
1c0186c4:	4562                	lw	a0,24(sp)
1c0186c6:	182c                	addi	a1,sp,56
1c0186c8:	820fd0ef          	jal	ra,1c0156e8 <pi_fpv2_ptable_partition_info_get>
1c0186cc:	fd31                	bnez	a0,1c018628 <__pi_fs_mount_step+0x4c>
1c0186ce:	481c                	lw	a5,16(s0)
1c0186d0:	01f7b563          	p.bneimm	a5,-1,1c0186da <__pi_fs_mount_step+0xfe>
1c0186d4:	04112783          	lw	a5,65(sp)
1c0186d8:	c81c                	sw	a5,16(s0)
1c0186da:	03a14783          	lbu	a5,58(sp)
1c0186de:	f417b5e3          	p.bneimm	a5,1,1c018628 <__pi_fs_mount_step+0x4c>
1c0186e2:	03b14703          	lbu	a4,59(sp)
1c0186e6:	08100793          	li	a5,129
1c0186ea:	f2f71fe3          	bne	a4,a5,1c018628 <__pi_fs_mount_step+0x4c>
1c0186ee:	4562                	lw	a0,24(sp)
1c0186f0:	85a2                	mv	a1,s0
1c0186f2:	4481                	li	s1,0
1c0186f4:	842fd0ef          	jal	ra,1c015736 <pi_fpv2_ptable_partition_device_get>
1c0186f8:	4008                	lw	a0,0(s0)
1c0186fa:	00440593          	addi	a1,s0,4
1c0186fe:	897fe0ef          	jal	ra,1c016f94 <pi_open>
1c018702:	b725                	j	1c01862a <__pi_fs_mount_step+0x4e>
1c018704:	4495                	li	s1,5
1c018706:	b715                	j	1c01862a <__pi_fs_mount_step+0x4e>
1c018708:	1050                	addi	a2,sp,36
1c01870a:	a24fd0ef          	jal	ra,1c01592e <pi_fpv2_vtable_volume_get_by_label>
1c01870e:	b741                	j	1c01868e <__pi_fs_mount_step+0xb2>
1c018710:	1c01c7b7          	lui	a5,0x1c01c
1c018714:	d0878793          	addi	a5,a5,-760 # 1c01bd08 <__clz_tab+0x2c0>
1c018718:	bf59                	j	1c0186ae <__pi_fs_mount_step+0xd2>
1c01871a:	0830                	addi	a2,sp,24
1c01871c:	b12fd0ef          	jal	ra,1c015a2e <pi_fpv2_volume_partition_get_by_label>
1c018720:	b74d                	j	1c0186c2 <__pi_fs_mount_step+0xe6>
1c018722:	5d3c                	lw	a5,120(a0)
1c018724:	01052903          	lw	s2,16(a0)
1c018728:	4384                	lw	s1,0(a5)
1c01872a:	049d                	addi	s1,s1,7
1c01872c:	c404b4b3          	p.bclr	s1,s1,2,0
1c018730:	8526                	mv	a0,s1
1c018732:	60d010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c018736:	dc68                	sw	a0,124(s0)
1c018738:	c13d                	beqz	a0,1c01879e <__pi_fs_mount_step+0x1c2>
1c01873a:	587c                	lw	a5,116(s0)
1c01873c:	01840713          	addi	a4,s0,24
1c018740:	1c0185b7          	lui	a1,0x1c018
1c018744:	0785                	addi	a5,a5,1
1c018746:	d87c                	sw	a5,116(s0)
1c018748:	8622                	mv	a2,s0
1c01874a:	5dc58593          	addi	a1,a1,1500 # 1c0185dc <__pi_fs_mount_step>
1c01874e:	853a                	mv	a0,a4
1c018750:	c63a                	sw	a4,12(sp)
1c018752:	79e010ef          	jal	ra,1c019ef0 <__pi_os_evt_callback_no_irq_init>
1c018756:	4732                	lw	a4,12(sp)
1c018758:	86a6                	mv	a3,s1
1c01875a:	5c70                	lw	a2,124(s0)
1c01875c:	00890593          	addi	a1,s2,8
1c018760:	b709                	j	1c018662 <__pi_fs_mount_step+0x86>
1c018762:	5d78                	lw	a4,124(a0)
1c018764:	4681                	li	a3,0
1c018766:	00470793          	addi	a5,a4,4
1c01876a:	430c                	lw	a1,0(a4)
1c01876c:	4701                	li	a4,0
1c01876e:	00b6cd63          	blt	a3,a1,1c018788 <__pi_fs_mount_step+0x1ac>
1c018772:	e315                	bnez	a4,1c018796 <__pi_fs_mount_step+0x1ba>
1c018774:	08f42223          	sw	a5,132(s0)
1c018778:	08042423          	sw	zero,136(s0)
1c01877c:	08042023          	sw	zero,128(s0)
1c018780:	5828                	lw	a0,112(s0)
1c018782:	79e010ef          	jal	ra,1c019f20 <__pi_os_evt_push>
1c018786:	b5cd                	j	1c018668 <__pi_fs_mount_step+0x8c>
1c018788:	4790                	lw	a2,8(a5)
1c01878a:	873e                	mv	a4,a5
1c01878c:	0685                	addi	a3,a3,1
1c01878e:	963e                	add	a2,a2,a5
1c018790:	00c60793          	addi	a5,a2,12
1c018794:	bfe9                	j	1c01876e <__pi_fs_mount_step+0x192>
1c018796:	431c                	lw	a5,0(a4)
1c018798:	4358                	lw	a4,4(a4)
1c01879a:	97ba                	add	a5,a5,a4
1c01879c:	bfe1                	j	1c018774 <__pi_fs_mount_step+0x198>
1c01879e:	57fd                	li	a5,-1
1c0187a0:	08f42023          	sw	a5,128(s0)
1c0187a4:	bff1                	j	1c018780 <__pi_fs_mount_step+0x1a4>

1c0187a6 <__pi_read_fs_mount>:
1c0187a6:	7159                	addi	sp,sp,-112
1c0187a8:	d2a6                	sw	s1,100(sp)
1c0187aa:	d0ca                	sw	s2,96(sp)
1c0187ac:	4144                	lw	s1,4(a0)
1c0187ae:	892a                	mv	s2,a0
1c0187b0:	0f000513          	li	a0,240
1c0187b4:	d4a2                	sw	s0,104(sp)
1c0187b6:	d686                	sw	ra,108(sp)
1c0187b8:	587010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c0187bc:	842a                	mv	s0,a0
1c0187be:	e919                	bnez	a0,1c0187d4 <__pi_read_fs_mount+0x2e>
1c0187c0:	8522                	mv	a0,s0
1c0187c2:	bbfff0ef          	jal	ra,1c018380 <__pi_fs_free>
1c0187c6:	557d                	li	a0,-1
1c0187c8:	50b6                	lw	ra,108(sp)
1c0187ca:	5426                	lw	s0,104(sp)
1c0187cc:	5496                	lw	s1,100(sp)
1c0187ce:	5906                	lw	s2,96(sp)
1c0187d0:	6165                	addi	sp,sp,112
1c0187d2:	8082                	ret
1c0187d4:	40dc                	lw	a5,4(s1)
1c0187d6:	06052c23          	sw	zero,120(a0)
1c0187da:	06052e23          	sw	zero,124(a0)
1c0187de:	c15c                	sw	a5,4(a0)
1c0187e0:	08052823          	sw	zero,144(a0)
1c0187e4:	4521                	li	a0,8
1c0187e6:	559010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c0187ea:	dc28                	sw	a0,120(s0)
1c0187ec:	d971                	beqz	a0,1c0187c0 <__pi_read_fs_mount+0x1a>
1c0187ee:	4785                	li	a5,1
1c0187f0:	d87c                	sw	a5,116(s0)
1c0187f2:	06042e23          	sw	zero,124(s0)
1c0187f6:	0028                	addi	a0,sp,8
1c0187f8:	6e2010ef          	jal	ra,1c019eda <__pi_os_evt_sig_init>
1c0187fc:	449c                	lw	a5,8(s1)
1c0187fe:	d828                	sw	a0,112(s0)
1c018800:	8522                	mv	a0,s0
1c018802:	c41c                	sw	a5,8(s0)
1c018804:	44dc                	lw	a5,12(s1)
1c018806:	c45c                	sw	a5,12(s0)
1c018808:	489c                	lw	a5,16(s1)
1c01880a:	c81c                	sw	a5,16(s0)
1c01880c:	00892423          	sw	s0,8(s2)
1c018810:	dcdff0ef          	jal	ra,1c0185dc <__pi_fs_mount_step>
1c018814:	0028                	addi	a0,sp,8
1c018816:	706010ef          	jal	ra,1c019f1c <__pi_os_evt_wait>
1c01881a:	08042783          	lw	a5,128(s0)
1c01881e:	f3cd                	bnez	a5,1c0187c0 <__pi_read_fs_mount+0x1a>
1c018820:	4501                	li	a0,0
1c018822:	b75d                	j	1c0187c8 <__pi_read_fs_mount+0x22>

1c018824 <pi_readfs_conf_init>:
1c018824:	1141                	addi	sp,sp,-16
1c018826:	c422                	sw	s0,8(sp)
1c018828:	c606                	sw	ra,12(sp)
1c01882a:	842a                	mv	s0,a0
1c01882c:	84dff0ef          	jal	ra,1c018078 <pi_fs_conf_init>
1c018830:	40b2                	lw	ra,12(sp)
1c018832:	00042023          	sw	zero,0(s0)
1c018836:	4422                	lw	s0,8(sp)
1c018838:	0141                	addi	sp,sp,16
1c01883a:	8082                	ret

1c01883c <vTaskStartScheduler>:
1c01883c:	1141                	addi	sp,sp,-16
1c01883e:	c606                	sw	ra,12(sp)
1c018840:	30047073          	csrci	mstatus,8
1c018844:	4e000793          	li	a5,1248
1c018848:	577d                	li	a4,-1
1c01884a:	c3d8                	sw	a4,4(a5)
1c01884c:	4685                	li	a3,1
1c01884e:	1c01d737          	lui	a4,0x1c01d
1c018852:	7ad72023          	sw	a3,1952(a4) # 1c01d7a0 <xSchedulerRunning>
1c018856:	0007a423          	sw	zero,8(a5)
1c01885a:	21bd                	jal	1c018cc8 <sched_start>
1c01885c:	300477f3          	csrrci	a5,mstatus,8
1c018860:	8e5f80ef          	jal	ra,1c011144 <xPortStartScheduler>
1c018864:	cd01                	beqz	a0,1c01887c <vTaskStartScheduler+0x40>
1c018866:	1c01c537          	lui	a0,0x1c01c
1c01886a:	d4450513          	addi	a0,a0,-700 # 1c01bd44 <__clz_tab+0x2fc>
1c01886e:	60f020ef          	jal	ra,1c01b67c <printf_>
1c018872:	757d                	lui	a0,0xfffff
1c018874:	66f50513          	addi	a0,a0,1647 # fffff66f <pulp__FC+0xfffff670>
1c018878:	73c010ef          	jal	ra,1c019fb4 <exit>
1c01887c:	40b2                	lw	ra,12(sp)
1c01887e:	1c0007b7          	lui	a5,0x1c000
1c018882:	5a87a783          	lw	a5,1448(a5) # 1c0005a8 <uxTopUsedPriority>
1c018886:	0141                	addi	sp,sp,16
1c018888:	8082                	ret

1c01888a <vTaskSwitchContextForce>:
1c01888a:	4e002503          	lw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c01888e:	4118                	lw	a4,0(a0)
1c018890:	415c                	lw	a5,4(a0)
1c018892:	00e7ea63          	bltu	a5,a4,1c0188a6 <vTaskSwitchContextForce+0x1c>
1c018896:	1141                	addi	sp,sp,-16
1c018898:	00c50593          	addi	a1,a0,12
1c01889c:	c606                	sw	ra,12(sp)
1c01889e:	446010ef          	jal	ra,1c019ce4 <vApplicationStackOverflowHook>
1c0188a2:	40b2                	lw	ra,12(sp)
1c0188a4:	0141                	addi	sp,sp,16
1c0188a6:	a1f9                	j	1c018d74 <sched_task_switch_tcb_force>

1c0188a8 <vTaskSwitchContext>:
1c0188a8:	4e002503          	lw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c0188ac:	4118                	lw	a4,0(a0)
1c0188ae:	415c                	lw	a5,4(a0)
1c0188b0:	00e7ea63          	bltu	a5,a4,1c0188c4 <vTaskSwitchContext+0x1c>
1c0188b4:	1141                	addi	sp,sp,-16
1c0188b6:	00c50593          	addi	a1,a0,12
1c0188ba:	c606                	sw	ra,12(sp)
1c0188bc:	428010ef          	jal	ra,1c019ce4 <vApplicationStackOverflowHook>
1c0188c0:	40b2                	lw	ra,12(sp)
1c0188c2:	0141                	addi	sp,sp,16
1c0188c4:	a99d                	j	1c018d3a <sched_task_switch_tcb>

1c0188c6 <os_evt_free_list_nodes_init>:
1c0188c6:	1141                	addi	sp,sp,-16
1c0188c8:	c422                	sw	s0,8(sp)
1c0188ca:	1c01c437          	lui	s0,0x1c01c
1c0188ce:	20042783          	lw	a5,512(s0) # 1c01c200 <g_free_list_nodes_nb>
1c0188d2:	4531                	li	a0,12
1c0188d4:	c606                	sw	ra,12(sp)
1c0188d6:	02f50533          	mul	a0,a0,a5
1c0188da:	465010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c0188de:	862a                	mv	a2,a0
1c0188e0:	557d                	li	a0,-1
1c0188e2:	c61d                	beqz	a2,1c018910 <os_evt_free_list_nodes_init+0x4a>
1c0188e4:	20040793          	addi	a5,s0,512
1c0188e8:	00478713          	addi	a4,a5,4
1c0188ec:	c3d8                	sw	a4,4(a5)
1c0188ee:	c798                	sw	a4,8(a5)
1c0188f0:	0007a623          	sw	zero,12(a5)
1c0188f4:	01078713          	addi	a4,a5,16
1c0188f8:	cb98                	sw	a4,16(a5)
1c0188fa:	cbd8                	sw	a4,20(a5)
1c0188fc:	cf90                	sw	a2,24(a5)
1c0188fe:	1c01c5b7          	lui	a1,0x1c01c
1c018902:	4685                	li	a3,1
1c018904:	20458593          	addi	a1,a1,516 # 1c01c204 <g_free_list_nodes>
1c018908:	4398                	lw	a4,0(a5)
1c01890a:	00e6e763          	bltu	a3,a4,1c018918 <os_evt_free_list_nodes_init+0x52>
1c01890e:	4501                	li	a0,0
1c018910:	40b2                	lw	ra,12(sp)
1c018912:	4422                	lw	s0,8(sp)
1c018914:	0141                	addi	sp,sp,16
1c018916:	8082                	ret
1c018918:	1006d533          	p.exthz	a0,a3
1c01891c:	00151713          	slli	a4,a0,0x1
1c018920:	972a                	add	a4,a4,a0
1c018922:	070a                	slli	a4,a4,0x2
1c018924:	9732                	add	a4,a4,a2
1c018926:	c318                	sw	a4,0(a4)
1c018928:	c358                	sw	a4,4(a4)
1c01892a:	00072423          	sw	zero,8(a4)
1c01892e:	43c8                	lw	a0,4(a5)
1c018930:	c34c                	sw	a1,4(a4)
1c018932:	c308                	sw	a0,0(a4)
1c018934:	43c8                	lw	a0,4(a5)
1c018936:	c158                	sw	a4,4(a0)
1c018938:	c3d8                	sw	a4,4(a5)
1c01893a:	0685                	addi	a3,a3,1
1c01893c:	0ff6f693          	andi	a3,a3,255
1c018940:	b7e1                	j	1c018908 <os_evt_free_list_nodes_init+0x42>

1c018942 <os_evt_sig_init>:
1c018942:	4785                	li	a5,1
1c018944:	00f50a23          	sb	a5,20(a0)
1c018948:	04050623          	sb	zero,76(a0)
1c01894c:	04050793          	addi	a5,a0,64
1c018950:	c13c                	sw	a5,64(a0)
1c018952:	c17c                	sw	a5,68(a0)
1c018954:	04052423          	sw	zero,72(a0)
1c018958:	014027f3          	csrr	a5,uhartid
1c01895c:	f457b7b3          	p.bclr	a5,a5,26,5
1c018960:	17dd                	addi	a5,a5,-9
1c018962:	0017b793          	seqz	a5,a5
1c018966:	40f007b3          	neg	a5,a5
1c01896a:	04f506a3          	sb	a5,77(a0)
1c01896e:	02052e23          	sw	zero,60(a0)
1c018972:	00052023          	sw	zero,0(a0)
1c018976:	8082                	ret

1c018978 <os_evt_callback_init>:
1c018978:	00050a23          	sb	zero,20(a0)
1c01897c:	04050623          	sb	zero,76(a0)
1c018980:	c14c                	sw	a1,4(a0)
1c018982:	c510                	sw	a2,8(a0)
1c018984:	014027f3          	csrr	a5,uhartid
1c018988:	f457b7b3          	p.bclr	a5,a5,26,5
1c01898c:	04050713          	addi	a4,a0,64
1c018990:	0297a663          	p.beqimm	a5,9,1c0189bc <os_evt_callback_init+0x44>
1c018994:	c138                	sw	a4,64(a0)
1c018996:	c178                	sw	a4,68(a0)
1c018998:	04052423          	sw	zero,72(a0)
1c01899c:	014027f3          	csrr	a5,uhartid
1c0189a0:	f457b7b3          	p.bclr	a5,a5,26,5
1c0189a4:	17dd                	addi	a5,a5,-9
1c0189a6:	0017b793          	seqz	a5,a5
1c0189aa:	40f007b3          	neg	a5,a5
1c0189ae:	04f506a3          	sb	a5,77(a0)
1c0189b2:	02052e23          	sw	zero,60(a0)
1c0189b6:	00052023          	sw	zero,0(a0)
1c0189ba:	8082                	ret
1c0189bc:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c0189c0:	c138                	sw	a4,64(a0)
1c0189c2:	c178                	sw	a4,68(a0)
1c0189c4:	c53c                	sw	a5,72(a0)
1c0189c6:	bfd9                	j	1c01899c <os_evt_callback_init+0x24>

1c0189c8 <os_evt_callback_irq_init>:
1c0189c8:	4789                	li	a5,2
1c0189ca:	00f50a23          	sb	a5,20(a0)
1c0189ce:	c14c                	sw	a1,4(a0)
1c0189d0:	c510                	sw	a2,8(a0)
1c0189d2:	8082                	ret

1c0189d4 <os_evt_wait>:
1c0189d4:	7139                	addi	sp,sp,-64
1c0189d6:	da26                	sw	s1,52(sp)
1c0189d8:	de06                	sw	ra,60(sp)
1c0189da:	dc22                	sw	s0,56(sp)
1c0189dc:	d84a                	sw	s2,48(sp)
1c0189de:	d64e                	sw	s3,44(sp)
1c0189e0:	d452                	sw	s4,40(sp)
1c0189e2:	d256                	sw	s5,36(sp)
1c0189e4:	d05a                	sw	s6,32(sp)
1c0189e6:	ce5e                	sw	s7,28(sp)
1c0189e8:	84aa                	mv	s1,a0
1c0189ea:	1c01c437          	lui	s0,0x1c01c
1c0189ee:	20040413          	addi	s0,s0,512 # 1c01c200 <g_free_list_nodes_nb>
1c0189f2:	1c01cab7          	lui	s5,0x1c01c
1c0189f6:	00440b13          	addi	s6,s0,4
1c0189fa:	204a8a93          	addi	s5,s5,516 # 1c01c204 <g_free_list_nodes>
1c0189fe:	30047973          	csrrci	s2,mstatus,8
1c018a02:	4985                	li	s3,1
1c018a04:	04048b93          	addi	s7,s1,64
1c018a08:	04c4c783          	lbu	a5,76(s1)
1c018a0c:	1007e7b3          	p.extbs	a5,a5
1c018a10:	cf91                	beqz	a5,1c018a2c <os_evt_wait+0x58>
1c018a12:	30091073          	csrw	mstatus,s2
1c018a16:	50f2                	lw	ra,60(sp)
1c018a18:	5462                	lw	s0,56(sp)
1c018a1a:	54d2                	lw	s1,52(sp)
1c018a1c:	5942                	lw	s2,48(sp)
1c018a1e:	59b2                	lw	s3,44(sp)
1c018a20:	5a22                	lw	s4,40(sp)
1c018a22:	5a92                	lw	s5,36(sp)
1c018a24:	5b02                	lw	s6,32(sp)
1c018a26:	4bf2                	lw	s7,28(sp)
1c018a28:	6121                	addi	sp,sp,64
1c018a2a:	8082                	ret
1c018a2c:	0068                	addi	a0,sp,12
1c018a2e:	21b000ef          	jal	ra,1c019448 <task_native_has_event>
1c018a32:	c909                	beqz	a0,1c018a44 <os_evt_wait+0x70>
1c018a34:	30091073          	csrw	mstatus,s2
1c018a38:	4532                	lw	a0,12(sp)
1c018a3a:	28f000ef          	jal	ra,1c0194c8 <task_native_exec_event>
1c018a3e:	30047973          	csrrci	s2,mstatus,8
1c018a42:	b7d9                	j	1c018a08 <os_evt_wait+0x34>
1c018a44:	0a098563          	beqz	s3,1c018aee <os_evt_wait+0x11a>
1c018a48:	441c                	lw	a5,8(s0)
1c018a4a:	43d8                	lw	a4,4(a5)
1c018a4c:	05679c63          	bne	a5,s6,1c018aa4 <os_evt_wait+0xd0>
1c018a50:	30091073          	csrw	mstatus,s2
1c018a54:	3c000513          	li	a0,960
1c018a58:	2e7010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c018a5c:	d14d                	beqz	a0,1c0189fe <os_evt_wait+0x2a>
1c018a5e:	30047773          	csrrci	a4,mstatus,8
1c018a62:	c108                	sw	a0,0(a0)
1c018a64:	c148                	sw	a0,4(a0)
1c018a66:	c508                	sw	a0,8(a0)
1c018a68:	481c                	lw	a5,16(s0)
1c018a6a:	c11c                	sw	a5,0(a0)
1c018a6c:	01040793          	addi	a5,s0,16
1c018a70:	c15c                	sw	a5,4(a0)
1c018a72:	481c                	lw	a5,16(s0)
1c018a74:	c3c8                	sw	a0,4(a5)
1c018a76:	c808                	sw	a0,16(s0)
1c018a78:	401c                	lw	a5,0(s0)
1c018a7a:	0531                	addi	a0,a0,12
1c018a7c:	05078793          	addi	a5,a5,80
1c018a80:	c01c                	sw	a5,0(s0)
1c018a82:	04f6d0fb          	lp.setupi	x1,79,1c018a9c <os_evt_wait+0xc8>
1c018a86:	c108                	sw	a0,0(a0)
1c018a88:	c148                	sw	a0,4(a0)
1c018a8a:	00052423          	sw	zero,8(a0)
1c018a8e:	4054                	lw	a3,4(s0)
1c018a90:	01552223          	sw	s5,4(a0)
1c018a94:	c114                	sw	a3,0(a0)
1c018a96:	4054                	lw	a3,4(s0)
1c018a98:	c2c8                	sw	a0,4(a3)
1c018a9a:	c048                	sw	a0,4(s0)
1c018a9c:	0531                	addi	a0,a0,12
1c018a9e:	30071073          	csrw	mstatus,a4
1c018aa2:	bfb1                	j	1c0189fe <os_evt_wait+0x2a>
1c018aa4:	fb6706e3          	beq	a4,s6,1c018a50 <os_evt_wait+0x7c>
1c018aa8:	4e002603          	lw	a2,1248(zero) # 4e0 <pxCurrentTCB>
1c018aac:	43cc                	lw	a1,4(a5)
1c018aae:	4394                	lw	a3,0(a5)
1c018ab0:	c2cc                	sw	a1,4(a3)
1c018ab2:	43cc                	lw	a1,4(a5)
1c018ab4:	c194                	sw	a3,0(a1)
1c018ab6:	c39c                	sw	a5,0(a5)
1c018ab8:	c3dc                	sw	a5,4(a5)
1c018aba:	c790                	sw	a2,8(a5)
1c018abc:	40b4                	lw	a3,64(s1)
1c018abe:	0177a223          	sw	s7,4(a5)
1c018ac2:	c394                	sw	a3,0(a5)
1c018ac4:	40b4                	lw	a3,64(s1)
1c018ac6:	c2dc                	sw	a5,4(a3)
1c018ac8:	c0bc                	sw	a5,64(s1)
1c018aca:	4354                	lw	a3,4(a4)
1c018acc:	431c                	lw	a5,0(a4)
1c018ace:	c3d4                	sw	a3,4(a5)
1c018ad0:	4354                	lw	a3,4(a4)
1c018ad2:	c29c                	sw	a5,0(a3)
1c018ad4:	c318                	sw	a4,0(a4)
1c018ad6:	c358                	sw	a4,4(a4)
1c018ad8:	c704                	sw	s1,8(a4)
1c018ada:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c018ade:	5f94                	lw	a3,56(a5)
1c018ae0:	c314                	sw	a3,0(a4)
1c018ae2:	03878693          	addi	a3,a5,56
1c018ae6:	c354                	sw	a3,4(a4)
1c018ae8:	5f94                	lw	a3,56(a5)
1c018aea:	c2d8                	sw	a4,4(a3)
1c018aec:	df98                	sw	a4,56(a5)
1c018aee:	4e002503          	lw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c018af2:	511c                	lw	a5,32(a0)
1c018af4:	c781                	beqz	a5,1c018afc <os_evt_wait+0x128>
1c018af6:	511c                	lw	a5,32(a0)
1c018af8:	0017b563          	p.bneimm	a5,1,1c018b02 <os_evt_wait+0x12e>
1c018afc:	4789                	li	a5,2
1c018afe:	d11c                	sw	a5,32(a0)
1c018b00:	2af1                	jal	1c018cdc <sched_ready_list_remove>
1c018b02:	26dd                	jal	1c018ee8 <sched_ready_list_is_empty>
1c018b04:	89aa                	mv	s3,a0
1c018b06:	c519                	beqz	a0,1c018b14 <os_evt_wait+0x140>
1c018b08:	10500073          	wfi
1c018b0c:	300467f3          	csrrsi	a5,mstatus,8
1c018b10:	4981                	li	s3,0
1c018b12:	b735                	j	1c018a3e <os_evt_wait+0x6a>
1c018b14:	30091073          	csrw	mstatus,s2
1c018b18:	2a75                	jal	1c018cd4 <sched_yield>
1c018b1a:	b715                	j	1c018a3e <os_evt_wait+0x6a>

1c018b1c <os_evt_release>:
1c018b1c:	7179                	addi	sp,sp,-48
1c018b1e:	d606                	sw	ra,44(sp)
1c018b20:	d422                	sw	s0,40(sp)
1c018b22:	d226                	sw	s1,36(sp)
1c018b24:	d04a                	sw	s2,32(sp)
1c018b26:	ce4e                	sw	s3,28(sp)
1c018b28:	cc52                	sw	s4,24(sp)
1c018b2a:	ca56                	sw	s5,20(sp)
1c018b2c:	c85a                	sw	s6,16(sp)
1c018b2e:	c65e                	sw	s7,12(sp)
1c018b30:	c462                	sw	s8,8(sp)
1c018b32:	c266                	sw	s9,4(sp)
1c018b34:	c06a                	sw	s10,0(sp)
1c018b36:	30047bf3          	csrrci	s7,mstatus,8
1c018b3a:	04c54783          	lbu	a5,76(a0)
1c018b3e:	1007e7b3          	p.extbs	a5,a5
1c018b42:	c38d                	beqz	a5,1c018b64 <os_evt_release+0x48>
1c018b44:	300b9073          	csrw	mstatus,s7
1c018b48:	50b2                	lw	ra,44(sp)
1c018b4a:	5422                	lw	s0,40(sp)
1c018b4c:	5492                	lw	s1,36(sp)
1c018b4e:	5902                	lw	s2,32(sp)
1c018b50:	49f2                	lw	s3,28(sp)
1c018b52:	4a62                	lw	s4,24(sp)
1c018b54:	4ad2                	lw	s5,20(sp)
1c018b56:	4b42                	lw	s6,16(sp)
1c018b58:	4bb2                	lw	s7,12(sp)
1c018b5a:	4c22                	lw	s8,8(sp)
1c018b5c:	4c92                	lw	s9,4(sp)
1c018b5e:	4d02                	lw	s10,0(sp)
1c018b60:	6145                	addi	sp,sp,48
1c018b62:	8082                	ret
1c018b64:	892a                	mv	s2,a0
1c018b66:	4785                	li	a5,1
1c018b68:	04f50623          	sb	a5,76(a0)
1c018b6c:	1c01c4b7          	lui	s1,0x1c01c
1c018b70:	4160                	lw	s0,68(a0)
1c018b72:	20048493          	addi	s1,s1,512 # 1c01c200 <g_free_list_nodes_nb>
1c018b76:	04050c13          	addi	s8,a0,64
1c018b7a:	4b01                	li	s6,0
1c018b7c:	00448a13          	addi	s4,s1,4
1c018b80:	03841e63          	bne	s0,s8,1c018bbc <os_evt_release+0xa0>
1c018b84:	04d90783          	lb	a5,77(s2)
1c018b88:	01f7a863          	p.beqimm	a5,-1,1c018b98 <os_evt_release+0x7c>
1c018b8c:	102017b7          	lui	a5,0x10201
1c018b90:	e0078793          	addi	a5,a5,-512 # 10200e00 <__l1_heapsram_end+0x1e0e00>
1c018b94:	0007a223          	sw	zero,4(a5)
1c018b98:	300b9073          	csrw	mstatus,s7
1c018b9c:	fa0b06e3          	beqz	s6,1c018b48 <os_evt_release+0x2c>
1c018ba0:	5422                	lw	s0,40(sp)
1c018ba2:	50b2                	lw	ra,44(sp)
1c018ba4:	5492                	lw	s1,36(sp)
1c018ba6:	5902                	lw	s2,32(sp)
1c018ba8:	49f2                	lw	s3,28(sp)
1c018baa:	4a62                	lw	s4,24(sp)
1c018bac:	4ad2                	lw	s5,20(sp)
1c018bae:	4b42                	lw	s6,16(sp)
1c018bb0:	4bb2                	lw	s7,12(sp)
1c018bb2:	4c22                	lw	s8,8(sp)
1c018bb4:	4c92                	lw	s9,4(sp)
1c018bb6:	4d02                	lw	s10,0(sp)
1c018bb8:	6145                	addi	sp,sp,48
1c018bba:	a8ed                	j	1c018cb4 <sched_run>
1c018bbc:	00842983          	lw	s3,8(s0)
1c018bc0:	00442c83          	lw	s9,4(s0)
1c018bc4:	03c9a783          	lw	a5,60(s3)
1c018bc8:	03898a93          	addi	s5,s3,56
1c018bcc:	03579463          	bne	a5,s5,1c018bf4 <os_evt_release+0xd8>
1c018bd0:	4058                	lw	a4,4(s0)
1c018bd2:	401c                	lw	a5,0(s0)
1c018bd4:	c3d8                	sw	a4,4(a5)
1c018bd6:	4058                	lw	a4,4(s0)
1c018bd8:	c31c                	sw	a5,0(a4)
1c018bda:	c000                	sw	s0,0(s0)
1c018bdc:	c040                	sw	s0,4(s0)
1c018bde:	00042423          	sw	zero,8(s0)
1c018be2:	40dc                	lw	a5,4(s1)
1c018be4:	01442223          	sw	s4,4(s0)
1c018be8:	c01c                	sw	a5,0(s0)
1c018bea:	40dc                	lw	a5,4(s1)
1c018bec:	c3c0                	sw	s0,4(a5)
1c018bee:	c0c0                	sw	s0,4(s1)
1c018bf0:	8466                	mv	s0,s9
1c018bf2:	b779                	j	1c018b80 <os_evt_release+0x64>
1c018bf4:	4798                	lw	a4,8(a5)
1c018bf6:	0047ad03          	lw	s10,4(a5)
1c018bfa:	03271e63          	bne	a4,s2,1c018c36 <os_evt_release+0x11a>
1c018bfe:	43d4                	lw	a3,4(a5)
1c018c00:	4398                	lw	a4,0(a5)
1c018c02:	c354                	sw	a3,4(a4)
1c018c04:	43d4                	lw	a3,4(a5)
1c018c06:	c298                	sw	a4,0(a3)
1c018c08:	c39c                	sw	a5,0(a5)
1c018c0a:	c3dc                	sw	a5,4(a5)
1c018c0c:	0007a423          	sw	zero,8(a5)
1c018c10:	40d8                	lw	a4,4(s1)
1c018c12:	0147a223          	sw	s4,4(a5)
1c018c16:	c398                	sw	a4,0(a5)
1c018c18:	40d8                	lw	a4,4(s1)
1c018c1a:	c35c                	sw	a5,4(a4)
1c018c1c:	c0dc                	sw	a5,4(s1)
1c018c1e:	0209a783          	lw	a5,32(s3)
1c018c22:	0027ba63          	p.bneimm	a5,2,1c018c36 <os_evt_release+0x11a>
1c018c26:	03c9a783          	lw	a5,60(s3)
1c018c2a:	00fa9663          	bne	s5,a5,1c018c36 <os_evt_release+0x11a>
1c018c2e:	854e                	mv	a0,s3
1c018c30:	2c29                	jal	1c018e4a <sched_ready_list_add>
1c018c32:	c111                	beqz	a0,1c018c36 <os_evt_release+0x11a>
1c018c34:	4b05                	li	s6,1
1c018c36:	87ea                	mv	a5,s10
1c018c38:	bf51                	j	1c018bcc <os_evt_release+0xb0>

1c018c3a <os_evt_push>:
1c018c3a:	01450703          	lb	a4,20(a0)
1c018c3e:	00172763          	p.beqimm	a4,1,1c018c4c <os_evt_push+0x12>
1c018c42:	00272763          	p.beqimm	a4,2,1c018c50 <os_evt_push+0x16>
1c018c46:	e31d                	bnez	a4,1c018c6c <os_evt_push+0x32>
1c018c48:	02f0006f          	j	1c019476 <task_native_push_event>
1c018c4c:	ed1ff06f          	j	1c018b1c <os_evt_release>
1c018c50:	1141                	addi	sp,sp,-16
1c018c52:	c606                	sw	ra,12(sp)
1c018c54:	c422                	sw	s0,8(sp)
1c018c56:	30047473          	csrrci	s0,mstatus,8
1c018c5a:	4158                	lw	a4,4(a0)
1c018c5c:	4508                	lw	a0,8(a0)
1c018c5e:	9702                	jalr	a4
1c018c60:	30041073          	csrw	mstatus,s0
1c018c64:	40b2                	lw	ra,12(sp)
1c018c66:	4422                	lw	s0,8(sp)
1c018c68:	0141                	addi	sp,sp,16
1c018c6a:	8082                	ret
1c018c6c:	8082                	ret

1c018c6e <os_evt_push_delayed_us>:
1c018c6e:	1141                	addi	sp,sp,-16
1c018c70:	c606                	sw	ra,12(sp)
1c018c72:	c422                	sw	s0,8(sp)
1c018c74:	30047473          	csrrci	s0,mstatus,8
1c018c78:	e989                	bnez	a1,1c018c8a <os_evt_push_delayed_us+0x1c>
1c018c7a:	fc1ff0ef          	jal	ra,1c018c3a <os_evt_push>
1c018c7e:	30041073          	csrw	mstatus,s0
1c018c82:	40b2                	lw	ra,12(sp)
1c018c84:	4422                	lw	s0,8(sp)
1c018c86:	0141                	addi	sp,sp,16
1c018c88:	8082                	ret
1c018c8a:	862a                	mv	a2,a0
1c018c8c:	1c01e537          	lui	a0,0x1c01e
1c018c90:	b5050513          	addi	a0,a0,-1200 # 1c01db50 <sys_timer_hi_prec>
1c018c94:	717000ef          	jal	ra,1c019baa <pi_timer_task_add>
1c018c98:	b7dd                	j	1c018c7e <os_evt_push_delayed_us+0x10>

1c018c9a <sched_init>:
1c018c9a:	50c00793          	li	a5,1292
1c018c9e:	c3dc                	sw	a5,4(a5)
1c018ca0:	c39c                	sw	a5,0(a5)
1c018ca2:	00878713          	addi	a4,a5,8
1c018ca6:	c7d8                	sw	a4,12(a5)
1c018ca8:	c798                	sw	a4,8(a5)
1c018caa:	01078713          	addi	a4,a5,16
1c018cae:	cbd8                	sw	a4,20(a5)
1c018cb0:	cb98                	sw	a4,16(a5)
1c018cb2:	8082                	ret

1c018cb4 <sched_run>:
1c018cb4:	50c00793          	li	a5,1292
1c018cb8:	0187c783          	lbu	a5,24(a5)
1c018cbc:	0ff7f793          	andi	a5,a5,255
1c018cc0:	c399                	beqz	a5,1c018cc6 <sched_run+0x12>
1c018cc2:	0520106f          	j	1c019d14 <vSetPendSV>
1c018cc6:	8082                	ret

1c018cc8 <sched_start>:
1c018cc8:	50c00793          	li	a5,1292
1c018ccc:	4705                	li	a4,1
1c018cce:	00e78c23          	sb	a4,24(a5)
1c018cd2:	8082                	ret

1c018cd4 <sched_yield>:
1c018cd4:	0400106f          	j	1c019d14 <vSetPendSV>

1c018cd8 <sched_yield_force>:
1c018cd8:	0680106f          	j	1c019d40 <vSetPendSVForce>

1c018cdc <sched_ready_list_remove>:
1c018cdc:	30047673          	csrrci	a2,mstatus,8
1c018ce0:	02450793          	addi	a5,a0,36
1c018ce4:	5154                	lw	a3,36(a0)
1c018ce6:	5518                	lw	a4,40(a0)
1c018ce8:	c314                	sw	a3,0(a4)
1c018cea:	5154                	lw	a3,36(a0)
1c018cec:	c2d8                	sw	a4,4(a3)
1c018cee:	d51c                	sw	a5,40(a0)
1c018cf0:	d15c                	sw	a5,36(a0)
1c018cf2:	01c54783          	lbu	a5,28(a0)
1c018cf6:	50c00713          	li	a4,1292
1c018cfa:	00379693          	slli	a3,a5,0x3
1c018cfe:	078e                	slli	a5,a5,0x3
1c018d00:	20f77783          	p.lw	a5,a5(a4)
1c018d04:	96ba                	add	a3,a3,a4
1c018d06:	00d79e63          	bne	a5,a3,1c018d22 <sched_ready_list_remove+0x46>
1c018d0a:	01c54683          	lbu	a3,28(a0)
1c018d0e:	4785                	li	a5,1
1c018d10:	00d797b3          	sll	a5,a5,a3
1c018d14:	01974683          	lbu	a3,25(a4)
1c018d18:	fff7c793          	not	a5,a5
1c018d1c:	8ff5                	and	a5,a5,a3
1c018d1e:	00f70ca3          	sb	a5,25(a4)
1c018d22:	30061073          	csrw	mstatus,a2
1c018d26:	8082                	ret

1c018d28 <sched_bitmap_highest_prio_get>:
1c018d28:	52504783          	lbu	a5,1317(zero) # 525 <g_sched_highest_prio>
1c018d2c:	10078533          	p.ff1	a0,a5
1c018d30:	e391                	bnez	a5,1c018d34 <sched_bitmap_highest_prio_get+0xc>
1c018d32:	557d                	li	a0,-1
1c018d34:	0ff57513          	andi	a0,a0,255
1c018d38:	8082                	ret

1c018d3a <sched_task_switch_tcb>:
1c018d3a:	1141                	addi	sp,sp,-16
1c018d3c:	c606                	sw	ra,12(sp)
1c018d3e:	febff0ef          	jal	ra,1c018d28 <sched_bitmap_highest_prio_get>
1c018d42:	0ff00793          	li	a5,255
1c018d46:	02f50463          	beq	a0,a5,1c018d6e <sched_task_switch_tcb+0x34>
1c018d4a:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c018d4e:	5314                	lw	a3,32(a4)
1c018d50:	e299                	bnez	a3,1c018d56 <sched_task_switch_tcb+0x1c>
1c018d52:	4685                	li	a3,1
1c018d54:	d314                	sw	a3,32(a4)
1c018d56:	00351713          	slli	a4,a0,0x3
1c018d5a:	50c00513          	li	a0,1292
1c018d5e:	20e57503          	p.lw	a0,a4(a0)
1c018d62:	fdc50713          	addi	a4,a0,-36
1c018d66:	4ee02023          	sw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c018d6a:	fe052e23          	sw	zero,-4(a0)
1c018d6e:	40b2                	lw	ra,12(sp)
1c018d70:	0141                	addi	sp,sp,16
1c018d72:	8082                	ret

1c018d74 <sched_task_switch_tcb_force>:
1c018d74:	1101                	addi	sp,sp,-32
1c018d76:	ce06                	sw	ra,28(sp)
1c018d78:	cc22                	sw	s0,24(sp)
1c018d7a:	ca26                	sw	s1,20(sp)
1c018d7c:	c84a                	sw	s2,16(sp)
1c018d7e:	c64e                	sw	s3,12(sp)
1c018d80:	fa9ff0ef          	jal	ra,1c018d28 <sched_bitmap_highest_prio_get>
1c018d84:	0ff00793          	li	a5,255
1c018d88:	0af50a63          	beq	a0,a5,1c018e3c <sched_task_switch_tcb_force+0xc8>
1c018d8c:	4e002683          	lw	a3,1248(zero) # 4e0 <pxCurrentTCB>
1c018d90:	529c                	lw	a5,32(a3)
1c018d92:	e399                	bnez	a5,1c018d98 <sched_task_switch_tcb_force+0x24>
1c018d94:	4785                	li	a5,1
1c018d96:	d29c                	sw	a5,32(a3)
1c018d98:	50c00793          	li	a5,1292
1c018d9c:	00351713          	slli	a4,a0,0x3
1c018da0:	20e7f783          	p.lw	a5,a4(a5)
1c018da4:	50c00413          	li	s0,1292
1c018da8:	fdc78713          	addi	a4,a5,-36
1c018dac:	4ee02023          	sw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c018db0:	08d71263          	bne	a4,a3,1c018e34 <sched_task_switch_tcb_force+0xc0>
1c018db4:	10055533          	p.exthz	a0,a0
1c018db8:	4398                	lw	a4,0(a5)
1c018dba:	050e                	slli	a0,a0,0x3
1c018dbc:	9522                	add	a0,a0,s0
1c018dbe:	02a70763          	beq	a4,a0,1c018dec <sched_task_switch_tcb_force+0x78>
1c018dc2:	fdc70713          	addi	a4,a4,-36
1c018dc6:	4ee02023          	sw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c018dca:	4390                	lw	a2,0(a5)
1c018dcc:	43d8                	lw	a4,4(a5)
1c018dce:	c310                	sw	a2,0(a4)
1c018dd0:	4390                	lw	a2,0(a5)
1c018dd2:	c258                	sw	a4,4(a2)
1c018dd4:	c3dc                	sw	a5,4(a5)
1c018dd6:	c39c                	sw	a5,0(a5)
1c018dd8:	ff87c703          	lbu	a4,-8(a5)
1c018ddc:	00371613          	slli	a2,a4,0x3
1c018de0:	9622                	add	a2,a2,s0
1c018de2:	424c                	lw	a1,4(a2)
1c018de4:	c390                	sw	a2,0(a5)
1c018de6:	c3cc                	sw	a1,4(a5)
1c018de8:	c19c                	sw	a5,0(a1)
1c018dea:	c25c                	sw	a5,4(a2)
1c018dec:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c018df0:	04d71263          	bne	a4,a3,1c018e34 <sched_task_switch_tcb_force+0xc0>
1c018df4:	ff87c783          	lbu	a5,-8(a5)
1c018df8:	4905                	li	s2,1
1c018dfa:	01940983          	lb	s3,25(s0)
1c018dfe:	00f91933          	sll	s2,s2,a5
1c018e02:	10096933          	p.extbs	s2,s2
1c018e06:	fff94793          	not	a5,s2
1c018e0a:	00f9f7b3          	and	a5,s3,a5
1c018e0e:	00f40ca3          	sb	a5,25(s0)
1c018e12:	f17ff0ef          	jal	ra,1c018d28 <sched_bitmap_highest_prio_get>
1c018e16:	0ff00793          	li	a5,255
1c018e1a:	00f50963          	beq	a0,a5,1c018e2c <sched_task_switch_tcb_force+0xb8>
1c018e1e:	050e                	slli	a0,a0,0x3
1c018e20:	20a47503          	p.lw	a0,a0(s0)
1c018e24:	fdc50513          	addi	a0,a0,-36
1c018e28:	4ea02023          	sw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c018e2c:	01396933          	or	s2,s2,s3
1c018e30:	01240ca3          	sb	s2,25(s0)
1c018e34:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c018e38:	0207a023          	sw	zero,32(a5)
1c018e3c:	40f2                	lw	ra,28(sp)
1c018e3e:	4462                	lw	s0,24(sp)
1c018e40:	44d2                	lw	s1,20(sp)
1c018e42:	4942                	lw	s2,16(sp)
1c018e44:	49b2                	lw	s3,12(sp)
1c018e46:	6105                	addi	sp,sp,32
1c018e48:	8082                	ret

1c018e4a <sched_ready_list_add>:
1c018e4a:	1101                	addi	sp,sp,-32
1c018e4c:	ce06                	sw	ra,28(sp)
1c018e4e:	cc22                	sw	s0,24(sp)
1c018e50:	ca26                	sw	s1,20(sp)
1c018e52:	c84a                	sw	s2,16(sp)
1c018e54:	c64e                	sw	s3,12(sp)
1c018e56:	c452                	sw	s4,8(sp)
1c018e58:	30047a73          	csrrci	s4,mstatus,8
1c018e5c:	511c                	lw	a5,32(a0)
1c018e5e:	4705                	li	a4,1
1c018e60:	17f5                	addi	a5,a5,-3
1c018e62:	00f76d63          	bltu	a4,a5,1c018e7c <sched_ready_list_add+0x32>
1c018e66:	300a1073          	csrw	mstatus,s4
1c018e6a:	4501                	li	a0,0
1c018e6c:	40f2                	lw	ra,28(sp)
1c018e6e:	4462                	lw	s0,24(sp)
1c018e70:	44d2                	lw	s1,20(sp)
1c018e72:	4942                	lw	s2,16(sp)
1c018e74:	49b2                	lw	s3,12(sp)
1c018e76:	4a22                	lw	s4,8(sp)
1c018e78:	6105                	addi	sp,sp,32
1c018e7a:	8082                	ret
1c018e7c:	d118                	sw	a4,32(a0)
1c018e7e:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c018e82:	842a                	mv	s0,a0
1c018e84:	01c54903          	lbu	s2,28(a0)
1c018e88:	89be                	mv	s3,a5
1c018e8a:	50c00493          	li	s1,1292
1c018e8e:	cb11                	beqz	a4,1c018ea2 <sched_ready_list_add+0x58>
1c018e90:	0184c783          	lbu	a5,24(s1)
1c018e94:	0ff7f793          	andi	a5,a5,255
1c018e98:	e799                	bnez	a5,1c018ea6 <sched_ready_list_add+0x5c>
1c018e9a:	e8fff0ef          	jal	ra,1c018d28 <sched_bitmap_highest_prio_get>
1c018e9e:	01257463          	bleu	s2,a0,1c018ea6 <sched_ready_list_add+0x5c>
1c018ea2:	4e802023          	sw	s0,1248(zero) # 4e0 <pxCurrentTCB>
1c018ea6:	02440713          	addi	a4,s0,36
1c018eaa:	00391793          	slli	a5,s2,0x3
1c018eae:	97a6                	add	a5,a5,s1
1c018eb0:	43d4                	lw	a3,4(a5)
1c018eb2:	d05c                	sw	a5,36(s0)
1c018eb4:	d414                	sw	a3,40(s0)
1c018eb6:	c298                	sw	a4,0(a3)
1c018eb8:	c3d8                	sw	a4,4(a5)
1c018eba:	0194c503          	lbu	a0,25(s1)
1c018ebe:	01c44783          	lbu	a5,28(s0)
1c018ec2:	40f55533          	sra	a0,a0,a5
1c018ec6:	00153513          	seqz	a0,a0
1c018eca:	0ff57513          	andi	a0,a0,255
1c018ece:	01c44703          	lbu	a4,28(s0)
1c018ed2:	4785                	li	a5,1
1c018ed4:	00e797b3          	sll	a5,a5,a4
1c018ed8:	0194c703          	lbu	a4,25(s1)
1c018edc:	8fd9                	or	a5,a5,a4
1c018ede:	00f48ca3          	sb	a5,25(s1)
1c018ee2:	300a1073          	csrw	mstatus,s4
1c018ee6:	b759                	j	1c018e6c <sched_ready_list_add+0x22>

1c018ee8 <sched_ready_list_is_empty>:
1c018ee8:	50c00713          	li	a4,1292
1c018eec:	431c                	lw	a5,0(a4)
1c018eee:	00e78463          	beq	a5,a4,1c018ef6 <sched_ready_list_is_empty+0xe>
1c018ef2:	4501                	li	a0,0
1c018ef4:	8082                	ret
1c018ef6:	4794                	lw	a3,8(a5)
1c018ef8:	00878713          	addi	a4,a5,8
1c018efc:	fee69be3          	bne	a3,a4,1c018ef2 <sched_ready_list_is_empty+0xa>
1c018f00:	4b88                	lw	a0,16(a5)
1c018f02:	01078713          	addi	a4,a5,16
1c018f06:	8d19                	sub	a0,a0,a4
1c018f08:	00153513          	seqz	a0,a0
1c018f0c:	0ff57513          	andi	a0,a0,255
1c018f10:	8082                	ret

1c018f12 <os_list_node_insert>:
1c018f12:	411c                	lw	a5,0(a0)
1c018f14:	c1c8                	sw	a0,4(a1)
1c018f16:	c19c                	sw	a5,0(a1)
1c018f18:	411c                	lw	a5,0(a0)
1c018f1a:	c3cc                	sw	a1,4(a5)
1c018f1c:	c10c                	sw	a1,0(a0)
1c018f1e:	8082                	ret

1c018f20 <os_list_node_remove_and_init>:
1c018f20:	4158                	lw	a4,4(a0)
1c018f22:	411c                	lw	a5,0(a0)
1c018f24:	c3d8                	sw	a4,4(a5)
1c018f26:	4158                	lw	a4,4(a0)
1c018f28:	c31c                	sw	a5,0(a4)
1c018f2a:	c108                	sw	a0,0(a0)
1c018f2c:	c148                	sw	a0,4(a0)
1c018f2e:	c50c                	sw	a1,8(a0)
1c018f30:	8082                	ret

1c018f32 <os_sync_obj_free_list_nodes_alloc>:
1c018f32:	1101                	addi	sp,sp,-32
1c018f34:	07800513          	li	a0,120
1c018f38:	cc22                	sw	s0,24(sp)
1c018f3a:	ce06                	sw	ra,28(sp)
1c018f3c:	ca26                	sw	s1,20(sp)
1c018f3e:	c84a                	sw	s2,16(sp)
1c018f40:	c64e                	sw	s3,12(sp)
1c018f42:	5fc010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c018f46:	842a                	mv	s0,a0
1c018f48:	557d                	li	a0,-1
1c018f4a:	c829                	beqz	s0,1c018f9c <os_sync_obj_free_list_nodes_alloc+0x6a>
1c018f4c:	30047973          	csrrci	s2,mstatus,8
1c018f50:	c000                	sw	s0,0(s0)
1c018f52:	c040                	sw	s0,4(s0)
1c018f54:	c400                	sw	s0,8(s0)
1c018f56:	1c01e537          	lui	a0,0x1c01e
1c018f5a:	85a2                	mv	a1,s0
1c018f5c:	b3850513          	addi	a0,a0,-1224 # 1c01db38 <g_sync_obj_mem_slab>
1c018f60:	fb3ff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c018f64:	1c0007b7          	lui	a5,0x1c000
1c018f68:	5ac78793          	addi	a5,a5,1452 # 1c0005ac <g_sync_obj_free_list_nodes_nb>
1c018f6c:	4398                	lw	a4,0(a5)
1c018f6e:	00c40493          	addi	s1,s0,12
1c018f72:	1c01e9b7          	lui	s3,0x1c01e
1c018f76:	0729                	addi	a4,a4,10
1c018f78:	c398                	sw	a4,0(a5)
1c018f7a:	07840413          	addi	s0,s0,120
1c018f7e:	c084                	sw	s1,0(s1)
1c018f80:	c0c4                	sw	s1,4(s1)
1c018f82:	0004a423          	sw	zero,8(s1)
1c018f86:	85a6                	mv	a1,s1
1c018f88:	b4498513          	addi	a0,s3,-1212 # 1c01db44 <g_sync_obj_free_list_nodes>
1c018f8c:	04b1                	addi	s1,s1,12
1c018f8e:	f85ff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c018f92:	fe8496e3          	bne	s1,s0,1c018f7e <os_sync_obj_free_list_nodes_alloc+0x4c>
1c018f96:	30091073          	csrw	mstatus,s2
1c018f9a:	4501                	li	a0,0
1c018f9c:	40f2                	lw	ra,28(sp)
1c018f9e:	4462                	lw	s0,24(sp)
1c018fa0:	44d2                	lw	s1,20(sp)
1c018fa2:	4942                	lw	s2,16(sp)
1c018fa4:	49b2                	lw	s3,12(sp)
1c018fa6:	6105                	addi	sp,sp,32
1c018fa8:	8082                	ret

1c018faa <os_sync_obj_free_list_nodes_init>:
1c018faa:	1101                	addi	sp,sp,-32
1c018fac:	c84a                	sw	s2,16(sp)
1c018fae:	1c000937          	lui	s2,0x1c000
1c018fb2:	5ac92783          	lw	a5,1452(s2) # 1c0005ac <g_sync_obj_free_list_nodes_nb>
1c018fb6:	4531                	li	a0,12
1c018fb8:	c64e                	sw	s3,12(sp)
1c018fba:	02f50533          	mul	a0,a0,a5
1c018fbe:	ce06                	sw	ra,28(sp)
1c018fc0:	cc22                	sw	s0,24(sp)
1c018fc2:	ca26                	sw	s1,20(sp)
1c018fc4:	57a010ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c018fc8:	89aa                	mv	s3,a0
1c018fca:	557d                	li	a0,-1
1c018fcc:	02098d63          	beqz	s3,1c019006 <os_sync_obj_free_list_nodes_init+0x5c>
1c018fd0:	5ac90913          	addi	s2,s2,1452
1c018fd4:	1c01e7b7          	lui	a5,0x1c01e
1c018fd8:	b4478713          	addi	a4,a5,-1212 # 1c01db44 <g_sync_obj_free_list_nodes>
1c018fdc:	b4e7a223          	sw	a4,-1212(a5)
1c018fe0:	c358                	sw	a4,4(a4)
1c018fe2:	00072423          	sw	zero,8(a4)
1c018fe6:	1c01e6b7          	lui	a3,0x1c01e
1c018fea:	b3868713          	addi	a4,a3,-1224 # 1c01db38 <g_sync_obj_mem_slab>
1c018fee:	b2e6ac23          	sw	a4,-1224(a3)
1c018ff2:	c358                	sw	a4,4(a4)
1c018ff4:	01372423          	sw	s3,8(a4)
1c018ff8:	4405                	li	s0,1
1c018ffa:	84be                	mv	s1,a5
1c018ffc:	00092783          	lw	a5,0(s2)
1c019000:	00f46a63          	bltu	s0,a5,1c019014 <os_sync_obj_free_list_nodes_init+0x6a>
1c019004:	4501                	li	a0,0
1c019006:	40f2                	lw	ra,28(sp)
1c019008:	4462                	lw	s0,24(sp)
1c01900a:	44d2                	lw	s1,20(sp)
1c01900c:	4942                	lw	s2,16(sp)
1c01900e:	49b2                	lw	s3,12(sp)
1c019010:	6105                	addi	sp,sp,32
1c019012:	8082                	ret
1c019014:	100457b3          	p.exthz	a5,s0
1c019018:	00179593          	slli	a1,a5,0x1
1c01901c:	95be                	add	a1,a1,a5
1c01901e:	058a                	slli	a1,a1,0x2
1c019020:	95ce                	add	a1,a1,s3
1c019022:	c18c                	sw	a1,0(a1)
1c019024:	c1cc                	sw	a1,4(a1)
1c019026:	0005a423          	sw	zero,8(a1)
1c01902a:	b4448513          	addi	a0,s1,-1212
1c01902e:	0405                	addi	s0,s0,1
1c019030:	ee3ff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c019034:	0ff47413          	andi	s0,s0,255
1c019038:	b7d1                	j	1c018ffc <os_sync_obj_free_list_nodes_init+0x52>

1c01903a <os_sync_obj_init>:
1c01903a:	87aa                	mv	a5,a0
1c01903c:	00b7a22b          	p.sw	a1,4(a5!)
1c019040:	c15c                	sw	a5,4(a0)
1c019042:	c51c                	sw	a5,8(a0)
1c019044:	00052623          	sw	zero,12(a0)
1c019048:	411c                	lw	a5,0(a0)
1c01904a:	0227ab63          	p.beqimm	a5,2,1c019080 <os_sync_obj_init+0x46>
1c01904e:	4709                	li	a4,2
1c019050:	00f76563          	bltu	a4,a5,1c01905a <os_sync_obj_init+0x20>
1c019054:	0017a863          	p.beqimm	a5,1,1c019064 <os_sync_obj_init+0x2a>
1c019058:	8082                	ret
1c01905a:	0237a563          	p.beqimm	a5,3,1c019084 <os_sync_obj_init+0x4a>
1c01905e:	0277a363          	p.beqimm	a5,7,1c019084 <os_sync_obj_init+0x4a>
1c019062:	8082                	ret
1c019064:	ca19                	beqz	a2,1c01907a <os_sync_obj_init+0x40>
1c019066:	421c                	lw	a5,0(a2)
1c019068:	55fd                	li	a1,-1
1c01906a:	c391                	beqz	a5,1c01906e <os_sync_obj_init+0x34>
1c01906c:	438c                	lw	a1,0(a5)
1c01906e:	425c                	lw	a5,4(a2)
1c019070:	4601                	li	a2,0
1c019072:	c391                	beqz	a5,1c019076 <os_sync_obj_init+0x3c>
1c019074:	4390                	lw	a2,0(a5)
1c019076:	0541                	addi	a0,a0,16
1c019078:	acbd                	j	1c0192f6 <os_sem_init>
1c01907a:	4601                	li	a2,0
1c01907c:	55fd                	li	a1,-1
1c01907e:	bfe5                	j	1c019076 <os_sync_obj_init+0x3c>
1c019080:	0541                	addi	a0,a0,16
1c019082:	a455                	j	1c019326 <os_sem_bin_init>
1c019084:	0541                	addi	a0,a0,16
1c019086:	a475                	j	1c019332 <os_mutex_init>

1c019088 <os_sync_obj_take>:
1c019088:	4118                	lw	a4,0(a0)
1c01908a:	479d                	li	a5,7
1c01908c:	14e7e163          	bltu	a5,a4,1c0191ce <os_sync_obj_take+0x146>
1c019090:	4785                	li	a5,1
1c019092:	7139                	addi	sp,sp,-64
1c019094:	00e797b3          	sll	a5,a5,a4
1c019098:	da26                	sw	s1,52(sp)
1c01909a:	de06                	sw	ra,60(sp)
1c01909c:	dc22                	sw	s0,56(sp)
1c01909e:	d84a                	sw	s2,48(sp)
1c0190a0:	d64e                	sw	s3,44(sp)
1c0190a2:	d452                	sw	s4,40(sp)
1c0190a4:	d256                	sw	s5,36(sp)
1c0190a6:	d05a                	sw	s6,32(sp)
1c0190a8:	ce5e                	sw	s7,28(sp)
1c0190aa:	cc62                	sw	s8,24(sp)
1c0190ac:	ca66                	sw	s9,20(sp)
1c0190ae:	08e7f793          	andi	a5,a5,142
1c0190b2:	84aa                	mv	s1,a0
1c0190b4:	4501                	li	a0,0
1c0190b6:	c3bd                	beqz	a5,1c01911c <os_sync_obj_take+0x94>
1c0190b8:	1c01e937          	lui	s2,0x1c01e
1c0190bc:	8b2e                	mv	s6,a1
1c0190be:	01048a13          	addi	s4,s1,16
1c0190c2:	b4490913          	addi	s2,s2,-1212 # 1c01db44 <g_sync_obj_free_list_nodes>
1c0190c6:	00448b93          	addi	s7,s1,4
1c0190ca:	4c09                	li	s8,2
1c0190cc:	30047473          	csrrci	s0,mstatus,8
1c0190d0:	409c                	lw	a5,0(s1)
1c0190d2:	85da                	mv	a1,s6
1c0190d4:	8552                	mv	a0,s4
1c0190d6:	0277b563          	p.bneimm	a5,7,1c019100 <os_sync_obj_take+0x78>
1c0190da:	249d                	jal	1c019340 <os_mutex_lock_recursive>
1c0190dc:	02152e63          	p.beqimm	a0,1,1c019118 <os_sync_obj_take+0x90>
1c0190e0:	00492983          	lw	s3,4(s2)
1c0190e4:	0049ac83          	lw	s9,4(s3)
1c0190e8:	01298463          	beq	s3,s2,1c0190f0 <os_sync_obj_take+0x68>
1c0190ec:	052c9b63          	bne	s9,s2,1c019142 <os_sync_obj_take+0xba>
1c0190f0:	30041073          	csrw	mstatus,s0
1c0190f4:	e3fff0ef          	jal	ra,1c018f32 <os_sync_obj_free_list_nodes_alloc>
1c0190f8:	c511                	beqz	a0,1c019104 <os_sync_obj_take+0x7c>
1c0190fa:	bdfff0ef          	jal	ra,1c018cd8 <sched_yield_force>
1c0190fe:	b7f9                	j	1c0190cc <os_sync_obj_take+0x44>
1c019100:	2401                	jal	1c019300 <os_sem_take>
1c019102:	bfe9                	j	1c0190dc <os_sync_obj_take+0x54>
1c019104:	30047473          	csrrci	s0,mstatus,8
1c019108:	409c                	lw	a5,0(s1)
1c01910a:	4581                	li	a1,0
1c01910c:	8552                	mv	a0,s4
1c01910e:	0277b463          	p.bneimm	a5,7,1c019136 <os_sync_obj_take+0xae>
1c019112:	243d                	jal	1c019340 <os_mutex_lock_recursive>
1c019114:	02153363          	p.bneimm	a0,1,1c01913a <os_sync_obj_take+0xb2>
1c019118:	30041073          	csrw	mstatus,s0
1c01911c:	50f2                	lw	ra,60(sp)
1c01911e:	5462                	lw	s0,56(sp)
1c019120:	54d2                	lw	s1,52(sp)
1c019122:	5942                	lw	s2,48(sp)
1c019124:	59b2                	lw	s3,44(sp)
1c019126:	5a22                	lw	s4,40(sp)
1c019128:	5a92                	lw	s5,36(sp)
1c01912a:	5b02                	lw	s6,32(sp)
1c01912c:	4bf2                	lw	s7,28(sp)
1c01912e:	4c62                	lw	s8,24(sp)
1c019130:	4cd2                	lw	s9,20(sp)
1c019132:	6121                	addi	sp,sp,64
1c019134:	8082                	ret
1c019136:	22e9                	jal	1c019300 <os_sem_take>
1c019138:	bff1                	j	1c019114 <os_sync_obj_take+0x8c>
1c01913a:	00492983          	lw	s3,4(s2)
1c01913e:	0049ac83          	lw	s9,4(s3)
1c019142:	4e002583          	lw	a1,1248(zero) # 4e0 <pxCurrentTCB>
1c019146:	854e                	mv	a0,s3
1c019148:	dd9ff0ef          	jal	ra,1c018f20 <os_list_node_remove_and_init>
1c01914c:	85ce                	mv	a1,s3
1c01914e:	855e                	mv	a0,s7
1c019150:	dc3ff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c019154:	85a6                	mv	a1,s1
1c019156:	8566                	mv	a0,s9
1c019158:	dc9ff0ef          	jal	ra,1c018f20 <os_list_node_remove_and_init>
1c01915c:	4e002503          	lw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c019160:	85e6                	mv	a1,s9
1c019162:	02c50513          	addi	a0,a0,44
1c019166:	dadff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c01916a:	0068                	addi	a0,sp,12
1c01916c:	2cf1                	jal	1c019448 <task_native_has_event>
1c01916e:	c915                	beqz	a0,1c0191a2 <os_sync_obj_take+0x11a>
1c019170:	30041073          	csrw	mstatus,s0
1c019174:	4532                	lw	a0,12(sp)
1c019176:	2e89                	jal	1c0194c8 <task_native_exec_event>
1c019178:	30047473          	csrrci	s0,mstatus,8
1c01917c:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c019180:	0587c703          	lbu	a4,88(a5)
1c019184:	0ff77713          	andi	a4,a4,255
1c019188:	00372563          	p.beqimm	a4,3,1c019192 <os_sync_obj_take+0x10a>
1c01918c:	4ff8                	lw	a4,92(a5)
1c01918e:	fc971ee3          	bne	a4,s1,1c01916a <os_sync_obj_take+0xe2>
1c019192:	04078c23          	sb	zero,88(a5)
1c019196:	0407ae23          	sw	zero,92(a5)
1c01919a:	30041073          	csrw	mstatus,s0
1c01919e:	4b01                	li	s6,0
1c0191a0:	b735                	j	1c0190cc <os_sync_obj_take+0x44>
1c0191a2:	4e002503          	lw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c0191a6:	511c                	lw	a5,32(a0)
1c0191a8:	c781                	beqz	a5,1c0191b0 <os_sync_obj_take+0x128>
1c0191aa:	511c                	lw	a5,32(a0)
1c0191ac:	0017b663          	p.bneimm	a5,1,1c0191b8 <os_sync_obj_take+0x130>
1c0191b0:	03852023          	sw	s8,32(a0)
1c0191b4:	b29ff0ef          	jal	ra,1c018cdc <sched_ready_list_remove>
1c0191b8:	d31ff0ef          	jal	ra,1c018ee8 <sched_ready_list_is_empty>
1c0191bc:	c501                	beqz	a0,1c0191c4 <os_sync_obj_take+0x13c>
1c0191be:	300467f3          	csrrsi	a5,mstatus,8
1c0191c2:	bf5d                	j	1c019178 <os_sync_obj_take+0xf0>
1c0191c4:	30041073          	csrw	mstatus,s0
1c0191c8:	b0dff0ef          	jal	ra,1c018cd4 <sched_yield>
1c0191cc:	b775                	j	1c019178 <os_sync_obj_take+0xf0>
1c0191ce:	4501                	li	a0,0
1c0191d0:	8082                	ret

1c0191d2 <os_sync_obj_release>:
1c0191d2:	4118                	lw	a4,0(a0)
1c0191d4:	479d                	li	a5,7
1c0191d6:	0ee7e763          	bltu	a5,a4,1c0192c4 <os_sync_obj_release+0xf2>
1c0191da:	4785                	li	a5,1
1c0191dc:	715d                	addi	sp,sp,-80
1c0191de:	00e797b3          	sll	a5,a5,a4
1c0191e2:	c4a2                	sw	s0,72(sp)
1c0191e4:	c686                	sw	ra,76(sp)
1c0191e6:	c2a6                	sw	s1,68(sp)
1c0191e8:	c0ca                	sw	s2,64(sp)
1c0191ea:	de4e                	sw	s3,60(sp)
1c0191ec:	dc52                	sw	s4,56(sp)
1c0191ee:	da56                	sw	s5,52(sp)
1c0191f0:	d85a                	sw	s6,48(sp)
1c0191f2:	d65e                	sw	s7,44(sp)
1c0191f4:	d462                	sw	s8,40(sp)
1c0191f6:	d266                	sw	s9,36(sp)
1c0191f8:	d06a                	sw	s10,32(sp)
1c0191fa:	ce6e                	sw	s11,28(sp)
1c0191fc:	08e7f793          	andi	a5,a5,142
1c019200:	842a                	mv	s0,a0
1c019202:	4501                	li	a0,0
1c019204:	cb9d                	beqz	a5,1c01923a <os_sync_obj_release+0x68>
1c019206:	30047bf3          	csrrci	s7,mstatus,8
1c01920a:	00842903          	lw	s2,8(s0)
1c01920e:	00440c13          	addi	s8,s0,4
1c019212:	4a01                	li	s4,0
1c019214:	1c01eab7          	lui	s5,0x1c01e
1c019218:	4c8d                	li	s9,3
1c01921a:	03891f63          	bne	s2,s8,1c019258 <os_sync_obj_release+0x86>
1c01921e:	401c                	lw	a5,0(s0)
1c019220:	01040513          	addi	a0,s0,16
1c019224:	0877be63          	p.bneimm	a5,7,1c0192c0 <os_sync_obj_release+0xee>
1c019228:	2295                	jal	1c01938c <os_mutex_unlock_recursive>
1c01922a:	300b9073          	csrw	mstatus,s7
1c01922e:	000a0663          	beqz	s4,1c01923a <os_sync_obj_release+0x68>
1c019232:	c62a                	sw	a0,12(sp)
1c019234:	a81ff0ef          	jal	ra,1c018cb4 <sched_run>
1c019238:	4532                	lw	a0,12(sp)
1c01923a:	40b6                	lw	ra,76(sp)
1c01923c:	4426                	lw	s0,72(sp)
1c01923e:	4496                	lw	s1,68(sp)
1c019240:	4906                	lw	s2,64(sp)
1c019242:	59f2                	lw	s3,60(sp)
1c019244:	5a62                	lw	s4,56(sp)
1c019246:	5ad2                	lw	s5,52(sp)
1c019248:	5b42                	lw	s6,48(sp)
1c01924a:	5bb2                	lw	s7,44(sp)
1c01924c:	5c22                	lw	s8,40(sp)
1c01924e:	5c92                	lw	s9,36(sp)
1c019250:	5d02                	lw	s10,32(sp)
1c019252:	4df2                	lw	s11,28(sp)
1c019254:	6161                	addi	sp,sp,80
1c019256:	8082                	ret
1c019258:	00892483          	lw	s1,8(s2)
1c01925c:	00492d03          	lw	s10,4(s2)
1c019260:	0304a983          	lw	s3,48(s1)
1c019264:	02c48b13          	addi	s6,s1,44
1c019268:	01699d63          	bne	s3,s6,1c019282 <os_sync_obj_release+0xb0>
1c01926c:	854a                	mv	a0,s2
1c01926e:	4581                	li	a1,0
1c019270:	cb1ff0ef          	jal	ra,1c018f20 <os_list_node_remove_and_init>
1c019274:	85ca                	mv	a1,s2
1c019276:	b44a8513          	addi	a0,s5,-1212 # 1c01db44 <g_sync_obj_free_list_nodes>
1c01927a:	c99ff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c01927e:	896a                	mv	s2,s10
1c019280:	bf69                	j	1c01921a <os_sync_obj_release+0x48>
1c019282:	0089a783          	lw	a5,8(s3)
1c019286:	0049ad83          	lw	s11,4(s3)
1c01928a:	02879963          	bne	a5,s0,1c0192bc <os_sync_obj_release+0xea>
1c01928e:	4581                	li	a1,0
1c019290:	854e                	mv	a0,s3
1c019292:	c8fff0ef          	jal	ra,1c018f20 <os_list_node_remove_and_init>
1c019296:	85ce                	mv	a1,s3
1c019298:	b44a8513          	addi	a0,s5,-1212
1c01929c:	c77ff0ef          	jal	ra,1c018f12 <os_list_node_insert>
1c0192a0:	509c                	lw	a5,32(s1)
1c0192a2:	05948c23          	sb	s9,88(s1)
1c0192a6:	cce0                	sw	s0,92(s1)
1c0192a8:	0027ba63          	p.bneimm	a5,2,1c0192bc <os_sync_obj_release+0xea>
1c0192ac:	589c                	lw	a5,48(s1)
1c0192ae:	00fb1763          	bne	s6,a5,1c0192bc <os_sync_obj_release+0xea>
1c0192b2:	8526                	mv	a0,s1
1c0192b4:	b97ff0ef          	jal	ra,1c018e4a <sched_ready_list_add>
1c0192b8:	c111                	beqz	a0,1c0192bc <os_sync_obj_release+0xea>
1c0192ba:	4a05                	li	s4,1
1c0192bc:	89ee                	mv	s3,s11
1c0192be:	b76d                	j	1c019268 <os_sync_obj_release+0x96>
1c0192c0:	208d                	jal	1c019322 <os_sem_give>
1c0192c2:	b7a5                	j	1c01922a <os_sync_obj_release+0x58>
1c0192c4:	4501                	li	a0,0
1c0192c6:	8082                	ret

1c0192c8 <os_sync_obj_delete>:
1c0192c8:	04054783          	lbu	a5,64(a0)
1c0192cc:	00052023          	sw	zero,0(a0)
1c0192d0:	c789                	beqz	a5,1c0192da <os_sync_obj_delete+0x12>
1c0192d2:	04400593          	li	a1,68
1c0192d6:	16e0106f          	j	1c01a444 <pi_l2_free>
1c0192da:	8082                	ret

1c0192dc <_sem_give>:
1c0192dc:	30047773          	csrrci	a4,mstatus,8
1c0192e0:	415c                	lw	a5,4(a0)
1c0192e2:	c791                	beqz	a5,1c0192ee <_sem_give+0x12>
1c0192e4:	17fd                	addi	a5,a5,-1
1c0192e6:	c15c                	sw	a5,4(a0)
1c0192e8:	e399                	bnez	a5,1c0192ee <_sem_give+0x12>
1c0192ea:	00052423          	sw	zero,8(a0)
1c0192ee:	30071073          	csrw	mstatus,a4
1c0192f2:	4505                	li	a0,1
1c0192f4:	8082                	ret

1c0192f6 <os_sem_init>:
1c0192f6:	c10c                	sw	a1,0(a0)
1c0192f8:	c150                	sw	a2,4(a0)
1c0192fa:	00052423          	sw	zero,8(a0)
1c0192fe:	8082                	ret

1c019300 <os_sem_take>:
1c019300:	4114                	lw	a3,0(a0)
1c019302:	30047773          	csrrci	a4,mstatus,8
1c019306:	415c                	lw	a5,4(a0)
1c019308:	00d7fb63          	bleu	a3,a5,1c01931e <os_sem_take+0x1e>
1c01930c:	0785                	addi	a5,a5,1
1c01930e:	c15c                	sw	a5,4(a0)
1c019310:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c019314:	c51c                	sw	a5,8(a0)
1c019316:	4505                	li	a0,1
1c019318:	30071073          	csrw	mstatus,a4
1c01931c:	8082                	ret
1c01931e:	4501                	li	a0,0
1c019320:	bfe5                	j	1c019318 <os_sem_take+0x18>

1c019322 <os_sem_give>:
1c019322:	fbbff06f          	j	1c0192dc <_sem_give>

1c019326 <os_sem_bin_init>:
1c019326:	4705                	li	a4,1
1c019328:	c118                	sw	a4,0(a0)
1c01932a:	c158                	sw	a4,4(a0)
1c01932c:	00052423          	sw	zero,8(a0)
1c019330:	8082                	ret

1c019332 <os_mutex_init>:
1c019332:	4785                	li	a5,1
1c019334:	c11c                	sw	a5,0(a0)
1c019336:	00052223          	sw	zero,4(a0)
1c01933a:	00052423          	sw	zero,8(a0)
1c01933e:	8082                	ret

1c019340 <os_mutex_lock_recursive>:
1c019340:	87aa                	mv	a5,a0
1c019342:	300476f3          	csrrci	a3,mstatus,8
1c019346:	4518                	lw	a4,8(a0)
1c019348:	c305                	beqz	a4,1c019368 <os_mutex_lock_recursive+0x28>
1c01934a:	4e002603          	lw	a2,1248(zero) # 4e0 <pxCurrentTCB>
1c01934e:	00c71963          	bne	a4,a2,1c019360 <os_mutex_lock_recursive+0x20>
1c019352:	4158                	lw	a4,4(a0)
1c019354:	0705                	addi	a4,a4,1
1c019356:	c158                	sw	a4,4(a0)
1c019358:	30069073          	csrw	mstatus,a3
1c01935c:	4505                	li	a0,1
1c01935e:	8082                	ret
1c019360:	30069073          	csrw	mstatus,a3
1c019364:	4501                	li	a0,0
1c019366:	8082                	ret
1c019368:	410c                	lw	a1,0(a0)
1c01936a:	30047673          	csrrci	a2,mstatus,8
1c01936e:	4158                	lw	a4,4(a0)
1c019370:	4501                	li	a0,0
1c019372:	00b77863          	bleu	a1,a4,1c019382 <os_mutex_lock_recursive+0x42>
1c019376:	0705                	addi	a4,a4,1
1c019378:	c3d8                	sw	a4,4(a5)
1c01937a:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c01937e:	4505                	li	a0,1
1c019380:	c798                	sw	a4,8(a5)
1c019382:	30061073          	csrw	mstatus,a2
1c019386:	30069073          	csrw	mstatus,a3
1c01938a:	8082                	ret

1c01938c <os_mutex_unlock_recursive>:
1c01938c:	1141                	addi	sp,sp,-16
1c01938e:	c606                	sw	ra,12(sp)
1c019390:	c422                	sw	s0,8(sp)
1c019392:	30047473          	csrrci	s0,mstatus,8
1c019396:	451c                	lw	a5,8(a0)
1c019398:	eb81                	bnez	a5,1c0193a8 <os_mutex_unlock_recursive+0x1c>
1c01939a:	30041073          	csrw	mstatus,s0
1c01939e:	4505                	li	a0,1
1c0193a0:	40b2                	lw	ra,12(sp)
1c0193a2:	4422                	lw	s0,8(sp)
1c0193a4:	0141                	addi	sp,sp,16
1c0193a6:	8082                	ret
1c0193a8:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c0193ac:	00e79963          	bne	a5,a4,1c0193be <os_mutex_unlock_recursive+0x32>
1c0193b0:	415c                	lw	a5,4(a0)
1c0193b2:	4705                	li	a4,1
1c0193b4:	00f77563          	bleu	a5,a4,1c0193be <os_mutex_unlock_recursive+0x32>
1c0193b8:	17fd                	addi	a5,a5,-1
1c0193ba:	c15c                	sw	a5,4(a0)
1c0193bc:	bff9                	j	1c01939a <os_mutex_unlock_recursive+0xe>
1c0193be:	f1fff0ef          	jal	ra,1c0192dc <_sem_give>
1c0193c2:	30041073          	csrw	mstatus,s0
1c0193c6:	bfe9                	j	1c0193a0 <os_mutex_unlock_recursive+0x14>

1c0193c8 <task_native_pop_event>:
1c0193c8:	1101                	addi	sp,sp,-32
1c0193ca:	ca26                	sw	s1,20(sp)
1c0193cc:	ce06                	sw	ra,28(sp)
1c0193ce:	cc22                	sw	s0,24(sp)
1c0193d0:	c84a                	sw	s2,16(sp)
1c0193d2:	c64e                	sw	s3,12(sp)
1c0193d4:	c452                	sw	s4,8(sp)
1c0193d6:	84aa                	mv	s1,a0
1c0193d8:	00052023          	sw	zero,0(a0)
1c0193dc:	300479f3          	csrrci	s3,mstatus,8
1c0193e0:	4e002403          	lw	s0,1248(zero) # 4e0 <pxCurrentTCB>
1c0193e4:	893e                	mv	s2,a5
1c0193e6:	4a09                	li	s4,2
1c0193e8:	4838                	lw	a4,80(s0)
1c0193ea:	c118                	sw	a4,0(a0)
1c0193ec:	409c                	lw	a5,0(s1)
1c0193ee:	c38d                	beqz	a5,1c019410 <task_native_pop_event+0x48>
1c0193f0:	483c                	lw	a5,80(s0)
1c0193f2:	439c                	lw	a5,0(a5)
1c0193f4:	c83c                	sw	a5,80(s0)
1c0193f6:	e399                	bnez	a5,1c0193fc <task_native_pop_event+0x34>
1c0193f8:	04042a23          	sw	zero,84(s0)
1c0193fc:	30099073          	csrw	mstatus,s3
1c019400:	40f2                	lw	ra,28(sp)
1c019402:	4462                	lw	s0,24(sp)
1c019404:	44d2                	lw	s1,20(sp)
1c019406:	4942                	lw	s2,16(sp)
1c019408:	49b2                	lw	s3,12(sp)
1c01940a:	4a22                	lw	s4,8(sp)
1c01940c:	6105                	addi	sp,sp,32
1c01940e:	8082                	ret
1c019410:	4e002503          	lw	a0,1248(zero) # 4e0 <pxCurrentTCB>
1c019414:	511c                	lw	a5,32(a0)
1c019416:	c781                	beqz	a5,1c01941e <task_native_pop_event+0x56>
1c019418:	511c                	lw	a5,32(a0)
1c01941a:	0017b663          	p.bneimm	a5,1,1c019426 <task_native_pop_event+0x5e>
1c01941e:	03452023          	sw	s4,32(a0)
1c019422:	8bbff0ef          	jal	ra,1c018cdc <sched_ready_list_remove>
1c019426:	ac3ff0ef          	jal	ra,1c018ee8 <sched_ready_list_is_empty>
1c01942a:	c911                	beqz	a0,1c01943e <task_native_pop_event+0x76>
1c01942c:	10500073          	wfi
1c019430:	300467f3          	csrrsi	a5,mstatus,8
1c019434:	300479f3          	csrrci	s3,mstatus,8
1c019438:	483c                	lw	a5,80(s0)
1c01943a:	c09c                	sw	a5,0(s1)
1c01943c:	bf45                	j	1c0193ec <task_native_pop_event+0x24>
1c01943e:	30099073          	csrw	mstatus,s3
1c019442:	893ff0ef          	jal	ra,1c018cd4 <sched_yield>
1c019446:	b7fd                	j	1c019434 <task_native_pop_event+0x6c>

1c019448 <task_native_has_event>:
1c019448:	00052023          	sw	zero,0(a0)
1c01944c:	300476f3          	csrrci	a3,mstatus,8
1c019450:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c019454:	4bb8                	lw	a4,80(a5)
1c019456:	c118                	sw	a4,0(a0)
1c019458:	c719                	beqz	a4,1c019466 <task_native_has_event+0x1e>
1c01945a:	4bb8                	lw	a4,80(a5)
1c01945c:	4318                	lw	a4,0(a4)
1c01945e:	cbb8                	sw	a4,80(a5)
1c019460:	e319                	bnez	a4,1c019466 <task_native_has_event+0x1e>
1c019462:	0407aa23          	sw	zero,84(a5)
1c019466:	30069073          	csrw	mstatus,a3
1c01946a:	4108                	lw	a0,0(a0)
1c01946c:	00a03533          	snez	a0,a0
1c019470:	0ff57513          	andi	a0,a0,255
1c019474:	8082                	ret

1c019476 <task_native_push_event>:
1c019476:	1141                	addi	sp,sp,-16
1c019478:	c606                	sw	ra,12(sp)
1c01947a:	c422                	sw	s0,8(sp)
1c01947c:	30047473          	csrrci	s0,mstatus,8
1c019480:	453c                	lw	a5,72(a0)
1c019482:	e789                	bnez	a5,1c01948c <task_native_push_event+0x16>
1c019484:	1c01d7b7          	lui	a5,0x1c01d
1c019488:	7ac7a783          	lw	a5,1964(a5) # 1c01d7ac <xHandler0>
1c01948c:	4bf8                	lw	a4,84(a5)
1c01948e:	00052023          	sw	zero,0(a0)
1c019492:	cf09                	beqz	a4,1c0194ac <task_native_push_event+0x36>
1c019494:	c308                	sw	a0,0(a4)
1c019496:	cbe8                	sw	a0,84(a5)
1c019498:	5394                	lw	a3,32(a5)
1c01949a:	4705                	li	a4,1
1c01949c:	00d76b63          	bltu	a4,a3,1c0194b2 <task_native_push_event+0x3c>
1c0194a0:	30041073          	csrw	mstatus,s0
1c0194a4:	40b2                	lw	ra,12(sp)
1c0194a6:	4422                	lw	s0,8(sp)
1c0194a8:	0141                	addi	sp,sp,16
1c0194aa:	8082                	ret
1c0194ac:	cbe8                	sw	a0,84(a5)
1c0194ae:	cba8                	sw	a0,80(a5)
1c0194b0:	b7e5                	j	1c019498 <task_native_push_event+0x22>
1c0194b2:	853e                	mv	a0,a5
1c0194b4:	997ff0ef          	jal	ra,1c018e4a <sched_ready_list_add>
1c0194b8:	30041073          	csrw	mstatus,s0
1c0194bc:	d565                	beqz	a0,1c0194a4 <task_native_push_event+0x2e>
1c0194be:	4422                	lw	s0,8(sp)
1c0194c0:	40b2                	lw	ra,12(sp)
1c0194c2:	0141                	addi	sp,sp,16
1c0194c4:	ff0ff06f          	j	1c018cb4 <sched_run>

1c0194c8 <task_native_exec_event>:
1c0194c8:	01450783          	lb	a5,20(a0)
1c0194cc:	ef81                	bnez	a5,1c0194e4 <task_native_exec_event+0x1c>
1c0194ce:	1141                	addi	sp,sp,-16
1c0194d0:	c422                	sw	s0,8(sp)
1c0194d2:	415c                	lw	a5,4(a0)
1c0194d4:	842a                	mv	s0,a0
1c0194d6:	4508                	lw	a0,8(a0)
1c0194d8:	c606                	sw	ra,12(sp)
1c0194da:	9782                	jalr	a5
1c0194dc:	8522                	mv	a0,s0
1c0194de:	4422                	lw	s0,8(sp)
1c0194e0:	40b2                	lw	ra,12(sp)
1c0194e2:	0141                	addi	sp,sp,16
1c0194e4:	e38ff06f          	j	1c018b1c <os_evt_release>

1c0194e8 <trampoline>:
1c0194e8:	1101                	addi	sp,sp,-32
1c0194ea:	cc22                	sw	s0,24(sp)
1c0194ec:	ce06                	sw	ra,28(sp)
1c0194ee:	842a                	mv	s0,a0
1c0194f0:	300467f3          	csrrsi	a5,mstatus,8
1c0194f4:	c401                	beqz	s0,1c0194fc <trampoline+0x14>
1c0194f6:	501c                	lw	a5,32(s0)
1c0194f8:	0047b663          	p.bneimm	a5,4,1c019504 <trampoline+0x1c>
1c0194fc:	40f2                	lw	ra,28(sp)
1c0194fe:	4462                	lw	s0,24(sp)
1c019500:	6105                	addi	sp,sp,32
1c019502:	8082                	ret
1c019504:	0068                	addi	a0,sp,12
1c019506:	ec3ff0ef          	jal	ra,1c0193c8 <task_native_pop_event>
1c01950a:	4532                	lw	a0,12(sp)
1c01950c:	fbdff0ef          	jal	ra,1c0194c8 <task_native_exec_event>
1c019510:	b7d5                	j	1c0194f4 <trampoline+0xc>

1c019512 <task_native_create>:
1c019512:	7179                	addi	sp,sp,-48
1c019514:	02468693          	addi	a3,a3,36
1c019518:	d04a                	sw	s2,32(sp)
1c01951a:	00269913          	slli	s2,a3,0x2
1c01951e:	ce4e                	sw	s3,28(sp)
1c019520:	89aa                	mv	s3,a0
1c019522:	854a                	mv	a0,s2
1c019524:	ca56                	sw	s5,20(sp)
1c019526:	c85a                	sw	s6,16(sp)
1c019528:	c65e                	sw	s7,12(sp)
1c01952a:	d606                	sw	ra,44(sp)
1c01952c:	d422                	sw	s0,40(sp)
1c01952e:	d226                	sw	s1,36(sp)
1c019530:	cc52                	sw	s4,24(sp)
1c019532:	8aae                	mv	s5,a1
1c019534:	8b32                	mv	s6,a2
1c019536:	8bba                	mv	s7,a4
1c019538:	07c010ef          	jal	ra,1c01a5b4 <pi_malloc>
1c01953c:	c139                	beqz	a0,1c019582 <task_native_create+0x70>
1c01953e:	8a2a                	mv	s4,a0
1c019540:	16800513          	li	a0,360
1c019544:	070010ef          	jal	ra,1c01a5b4 <pi_malloc>
1c019548:	842a                	mv	s0,a0
1c01954a:	e105                	bnez	a0,1c01956a <task_native_create+0x58>
1c01954c:	8552                	mv	a0,s4
1c01954e:	73f000ef          	jal	ra,1c01a48c <pi_free>
1c019552:	8522                	mv	a0,s0
1c019554:	50b2                	lw	ra,44(sp)
1c019556:	5422                	lw	s0,40(sp)
1c019558:	5492                	lw	s1,36(sp)
1c01955a:	5902                	lw	s2,32(sp)
1c01955c:	49f2                	lw	s3,28(sp)
1c01955e:	4a62                	lw	s4,24(sp)
1c019560:	4ad2                	lw	s5,20(sp)
1c019562:	4b42                	lw	s6,16(sp)
1c019564:	4bb2                	lw	s7,12(sp)
1c019566:	6145                	addi	sp,sp,48
1c019568:	8082                	ret
1c01956a:	05800513          	li	a0,88
1c01956e:	046010ef          	jal	ra,1c01a5b4 <pi_malloc>
1c019572:	84aa                	mv	s1,a0
1c019574:	e909                	bnez	a0,1c019586 <task_native_create+0x74>
1c019576:	8522                	mv	a0,s0
1c019578:	715000ef          	jal	ra,1c01a48c <pi_free>
1c01957c:	8552                	mv	a0,s4
1c01957e:	70f000ef          	jal	ra,1c01a48c <pi_free>
1c019582:	4401                	li	s0,0
1c019584:	b7f9                	j	1c019552 <task_native_create+0x40>
1c019586:	1971                	addi	s2,s2,-4
1c019588:	9952                	add	s2,s2,s4
1c01958a:	1c0195b7          	lui	a1,0x1c019
1c01958e:	8622                	mv	a2,s0
1c019590:	4e858593          	addi	a1,a1,1256 # 1c0194e8 <trampoline>
1c019594:	854a                	mv	a0,s2
1c019596:	105000ef          	jal	ra,1c019e9a <pxPortInitialiseStack>
1c01959a:	c008                	sw	a0,0(s0)
1c01959c:	01442223          	sw	s4,4(s0)
1c0195a0:	01242423          	sw	s2,8(s0)
1c0195a4:	0c099363          	bnez	s3,1c01966a <task_native_create+0x158>
1c0195a8:	00040623          	sb	zero,12(s0)
1c0195ac:	4709                	li	a4,2
1c0195ae:	04ebd733          	p.minu	a4,s7,a4
1c0195b2:	00e40e23          	sb	a4,28(s0)
1c0195b6:	00e40ea3          	sb	a4,29(s0)
1c0195ba:	02440793          	addi	a5,s0,36
1c0195be:	d41c                	sw	a5,40(s0)
1c0195c0:	d05c                	sw	a5,36(s0)
1c0195c2:	02c40793          	addi	a5,s0,44
1c0195c6:	d45c                	sw	a5,44(s0)
1c0195c8:	d81c                	sw	a5,48(s0)
1c0195ca:	02042a23          	sw	zero,52(s0)
1c0195ce:	03840793          	addi	a5,s0,56
1c0195d2:	dc1c                	sw	a5,56(s0)
1c0195d4:	dc5c                	sw	a5,60(s0)
1c0195d6:	04042023          	sw	zero,64(s0)
1c0195da:	04042223          	sw	zero,68(s0)
1c0195de:	04042423          	sw	zero,72(s0)
1c0195e2:	865a                	mv	a2,s6
1c0195e4:	85d6                	mv	a1,s5
1c0195e6:	8526                	mv	a0,s1
1c0195e8:	b90ff0ef          	jal	ra,1c018978 <os_evt_callback_init>
1c0195ec:	04048793          	addi	a5,s1,64
1c0195f0:	c0bc                	sw	a5,64(s1)
1c0195f2:	c0fc                	sw	a5,68(s1)
1c0195f4:	c4a0                	sw	s0,72(s1)
1c0195f6:	c824                	sw	s1,80(s0)
1c0195f8:	c864                	sw	s1,84(s0)
1c0195fa:	c464                	sw	s1,76(s0)
1c0195fc:	04040c23          	sb	zero,88(s0)
1c019600:	04042e23          	sw	zero,92(s0)
1c019604:	06040793          	addi	a5,s0,96
1c019608:	d07c                	sw	a5,100(s0)
1c01960a:	d03c                	sw	a5,96(s0)
1c01960c:	52404783          	lbu	a5,1316(zero) # 524 <g_sched_started>
1c019610:	0ff7f793          	andi	a5,a5,255
1c019614:	cf89                	beqz	a5,1c01962e <task_native_create+0x11c>
1c019616:	4e002783          	lw	a5,1248(zero) # 4e0 <pxCurrentTCB>
1c01961a:	06840713          	addi	a4,s0,104
1c01961e:	53f4                	lw	a3,100(a5)
1c019620:	d474                	sw	a3,108(s0)
1c019622:	06078693          	addi	a3,a5,96
1c019626:	d434                	sw	a3,104(s0)
1c019628:	53f4                	lw	a3,100(a5)
1c01962a:	c298                	sw	a4,0(a3)
1c01962c:	d3f8                	sw	a4,100(a5)
1c01962e:	07040513          	addi	a0,s0,112
1c019632:	b10ff0ef          	jal	ra,1c018942 <os_evt_sig_init>
1c019636:	14042823          	sw	zero,336(s0)
1c01963a:	14040c23          	sb	zero,344(s0)
1c01963e:	14042a23          	sw	zero,340(s0)
1c019642:	14040ca3          	sb	zero,345(s0)
1c019646:	15c40793          	addi	a5,s0,348
1c01964a:	16f42023          	sw	a5,352(s0)
1c01964e:	14f42e23          	sw	a5,348(s0)
1c019652:	4785                	li	a5,1
1c019654:	16042223          	sw	zero,356(s0)
1c019658:	d01c                	sw	a5,32(s0)
1c01965a:	8522                	mv	a0,s0
1c01965c:	feeff0ef          	jal	ra,1c018e4a <sched_ready_list_add>
1c019660:	ee0509e3          	beqz	a0,1c019552 <task_native_create+0x40>
1c019664:	e50ff0ef          	jal	ra,1c018cb4 <sched_run>
1c019668:	b5ed                	j	1c019552 <task_native_create+0x40>
1c01966a:	854e                	mv	a0,s3
1c01966c:	00c40713          	addi	a4,s0,12
1c019670:	0104d0fb          	lp.setupi	x1,16,1c019682 <task_native_create+0x170>
1c019674:	00054683          	lbu	a3,0(a0)
1c019678:	00d700ab          	p.sb	a3,1(a4!)
1c01967c:	0015468b          	p.lbu	a3,1(a0!)
1c019680:	c291                	beqz	a3,1c019684 <task_native_create+0x172>
1c019682:	0001                	nop
1c019684:	00040da3          	sb	zero,27(s0)
1c019688:	b715                	j	1c0195ac <task_native_create+0x9a>

1c01968a <task_native_start>:
1c01968a:	8082                	ret

1c01968c <__timer_base_get>:
1c01968c:	478d                	li	a5,3
1c01968e:	00a7e863          	bltu	a5,a0,1c01969e <__timer_base_get+0x12>
1c019692:	8105                	srli	a0,a0,0x1
1c019694:	1a10b7b7          	lui	a5,0x1a10b
1c019698:	052e                	slli	a0,a0,0xb
1c01969a:	953e                	add	a0,a0,a5
1c01969c:	8082                	ret
1c01969e:	00200537          	lui	a0,0x200
1c0196a2:	40050513          	addi	a0,a0,1024 # 200400 <__l1_heapsram_size+0x1e0470>
1c0196a6:	8082                	ret

1c0196a8 <__pi_timer_fc_0_1_handler>:
1c0196a8:	1101                	addi	sp,sp,-32
1c0196aa:	c84a                	sw	s2,16(sp)
1c0196ac:	1c01d937          	lui	s2,0x1c01d
1c0196b0:	6b092503          	lw	a0,1712(s2) # 1c01d6b0 <g_timer_data>
1c0196b4:	1a10b7b7          	lui	a5,0x1a10b
1c0196b8:	ce06                	sw	ra,28(sp)
1c0196ba:	cc22                	sw	s0,24(sp)
1c0196bc:	ca26                	sw	s1,20(sp)
1c0196be:	c64e                	sw	s3,12(sp)
1c0196c0:	0087a603          	lw	a2,8(a5) # 1a10b008 <__l1_heapsram_end+0xa0eb008>
1c0196c4:	00c7a683          	lw	a3,12(a5)
1c0196c8:	8736                	mv	a4,a3
1c0196ca:	4781                	li	a5,0
1c0196cc:	02000813          	li	a6,32
1c0196d0:	4881                	li	a7,0
1c0196d2:	21071733          	sll.d	a4,a4,a6
1c0196d6:	00c76433          	or	s0,a4,a2
1c0196da:	84be                	mv	s1,a5
1c0196dc:	3e800713          	li	a4,1000
1c0196e0:	4781                	li	a5,0
1c0196e2:	20e40433          	add.d	s0,s0,a4
1c0196e6:	6b090913          	addi	s2,s2,1712
1c0196ea:	87aa                	mv	a5,a0
1c0196ec:	e795                	bnez	a5,1c019718 <__pi_timer_fc_0_1_handler+0x70>
1c0196ee:	00f92023          	sw	a5,0(s2)
1c0196f2:	e915                	bnez	a0,1c019726 <__pi_timer_fc_0_1_handler+0x7e>
1c0196f4:	00092783          	lw	a5,0(s2)
1c0196f8:	cb89                	beqz	a5,1c01970a <__pi_timer_fc_0_1_handler+0x62>
1c0196fa:	4bf4                	lw	a3,84(a5)
1c0196fc:	1a10b737          	lui	a4,0x1a10b
1c019700:	00d72a23          	sw	a3,20(a4) # 1a10b014 <__l1_heapsram_end+0xa0eb014>
1c019704:	4bbc                	lw	a5,80(a5)
1c019706:	00f72823          	sw	a5,16(a4)
1c01970a:	40f2                	lw	ra,28(sp)
1c01970c:	4462                	lw	s0,24(sp)
1c01970e:	44d2                	lw	s1,20(sp)
1c019710:	4942                	lw	s2,16(sp)
1c019712:	49b2                	lw	s3,12(sp)
1c019714:	6105                	addi	sp,sp,32
1c019716:	8082                	ret
1c019718:	4bb0                	lw	a2,80(a5)
1c01971a:	4bf4                	lw	a3,84(a5)
1c01971c:	30c43633          	sltu.d	a2,s0,a2
1c019720:	f679                	bnez	a2,1c0196ee <__pi_timer_fc_0_1_handler+0x46>
1c019722:	439c                	lw	a5,0(a5)
1c019724:	b7e1                	j	1c0196ec <__pi_timer_fc_0_1_handler+0x44>
1c019726:	4938                	lw	a4,80(a0)
1c019728:	497c                	lw	a5,84(a0)
1c01972a:	30e43733          	sltu.d	a4,s0,a4
1c01972e:	f379                	bnez	a4,1c0196f4 <__pi_timer_fc_0_1_handler+0x4c>
1c019730:	00052983          	lw	s3,0(a0)
1c019734:	d06ff0ef          	jal	ra,1c018c3a <os_evt_push>
1c019738:	854e                	mv	a0,s3
1c01973a:	bf65                	j	1c0196f2 <__pi_timer_fc_0_1_handler+0x4a>

1c01973c <__pi_timer_fc_1_1_handler>:
1c01973c:	1141                	addi	sp,sp,-16
1c01973e:	c226                	sw	s1,4(sp)
1c019740:	1c01d4b7          	lui	s1,0x1c01d
1c019744:	6b048713          	addi	a4,s1,1712 # 1c01d6b0 <g_timer_data>
1c019748:	4b3c                	lw	a5,80(a4)
1c01974a:	5b08                	lw	a0,48(a4)
1c01974c:	c422                	sw	s0,8(sp)
1c01974e:	1a10c437          	lui	s0,0x1a10c
1c019752:	80040413          	addi	s0,s0,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c019756:	c606                	sw	ra,12(sp)
1c019758:	c04a                	sw	s2,0(sp)
1c01975a:	00c42403          	lw	s0,12(s0)
1c01975e:	0207f793          	andi	a5,a5,32
1c019762:	6b048493          	addi	s1,s1,1712
1c019766:	3e840413          	addi	s0,s0,1000
1c01976a:	ef89                	bnez	a5,1c019784 <__pi_timer_fc_1_1_handler+0x48>
1c01976c:	e129                	bnez	a0,1c0197ae <__pi_timer_fc_1_1_handler+0x72>
1c01976e:	0204a823          	sw	zero,48(s1)
1c019772:	a01d                	j	1c019798 <__pi_timer_fc_1_1_handler+0x5c>
1c019774:	493c                	lw	a5,80(a0)
1c019776:	00f46863          	bltu	s0,a5,1c019786 <__pi_timer_fc_1_1_handler+0x4a>
1c01977a:	00052903          	lw	s2,0(a0)
1c01977e:	cbcff0ef          	jal	ra,1c018c3a <os_evt_push>
1c019782:	854a                	mv	a0,s2
1c019784:	f965                	bnez	a0,1c019774 <__pi_timer_fc_1_1_handler+0x38>
1c019786:	d888                	sw	a0,48(s1)
1c019788:	cd11                	beqz	a0,1c0197a4 <__pi_timer_fc_1_1_handler+0x68>
1c01978a:	4938                	lw	a4,80(a0)
1c01978c:	1a10c7b7          	lui	a5,0x1a10c
1c019790:	80078793          	addi	a5,a5,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c019794:	00e7aa23          	sw	a4,20(a5)
1c019798:	40b2                	lw	ra,12(sp)
1c01979a:	4422                	lw	s0,8(sp)
1c01979c:	4492                	lw	s1,4(sp)
1c01979e:	4902                	lw	s2,0(sp)
1c0197a0:	0141                	addi	sp,sp,16
1c0197a2:	8082                	ret
1c0197a4:	1a1097b7          	lui	a5,0x1a109
1c0197a8:	6709                	lui	a4,0x2
1c0197aa:	c798                	sw	a4,8(a5)
1c0197ac:	b7f5                	j	1c019798 <__pi_timer_fc_1_1_handler+0x5c>
1c0197ae:	4100                	lw	s0,0(a0)
1c0197b0:	c8aff0ef          	jal	ra,1c018c3a <os_evt_push>
1c0197b4:	8522                	mv	a0,s0
1c0197b6:	bf5d                	j	1c01976c <__pi_timer_fc_1_1_handler+0x30>

1c0197b8 <__pi_timer_clock_freq_get>:
1c0197b8:	8105                	srli	a0,a0,0x1
1c0197ba:	1c01d7b7          	lui	a5,0x1c01d
1c0197be:	ee853533          	p.bclr	a0,a0,23,8
1c0197c2:	6b078793          	addi	a5,a5,1712 # 1c01d6b0 <g_timer_data>
1c0197c6:	03000713          	li	a4,48
1c0197ca:	42e507b3          	p.mac	a5,a0,a4
1c0197ce:	02078783          	lb	a5,32(a5)
1c0197d2:	0007c563          	bltz	a5,1c0197dc <__pi_timer_clock_freq_get+0x24>
1c0197d6:	4505                	li	a0,1
1c0197d8:	d46f706f          	j	1c010d1e <pi_freq_get>
1c0197dc:	0c90006f          	j	1c01a0a4 <pi_ref_fast_clock_frequency_get>

1c0197e0 <__pi_timer_conf_init>:
1c0197e0:	478d                	li	a5,3
1c0197e2:	c11c                	sw	a5,0(a0)
1c0197e4:	8082                	ret

1c0197e6 <__pi_timer_init>:
1c0197e6:	1101                	addi	sp,sp,-32
1c0197e8:	cc22                	sw	s0,24(sp)
1c0197ea:	ce06                	sw	ra,28(sp)
1c0197ec:	842a                	mv	s0,a0
1c0197ee:	c62e                	sw	a1,12(sp)
1c0197f0:	c432                	sw	a2,8(sp)
1c0197f2:	e9bff0ef          	jal	ra,1c01968c <__timer_base_get>
1c0197f6:	fc1437b3          	p.bclr	a5,s0,30,1
1c0197fa:	45b2                	lw	a1,12(sp)
1c0197fc:	4622                	lw	a2,8(sp)
1c0197fe:	078a                	slli	a5,a5,0x2
1c019800:	01078713          	addi	a4,a5,16
1c019804:	00c56723          	p.sw	a2,a4(a0)
1c019808:	00878713          	addi	a4,a5,8
1c01980c:	00056723          	p.sw	zero,a4(a0)
1c019810:	00b567a3          	p.sw	a1,a5(a0)
1c019814:	40f2                	lw	ra,28(sp)
1c019816:	4462                	lw	s0,24(sp)
1c019818:	6105                	addi	sp,sp,32
1c01981a:	8082                	ret

1c01981c <__pi_timer_stop>:
1c01981c:	1141                	addi	sp,sp,-16
1c01981e:	c422                	sw	s0,8(sp)
1c019820:	c606                	sw	ra,12(sp)
1c019822:	842a                	mv	s0,a0
1c019824:	e69ff0ef          	jal	ra,1c01968c <__timer_base_get>
1c019828:	fc1437b3          	p.bclr	a5,s0,30,1
1c01982c:	078a                	slli	a5,a5,0x2
1c01982e:	20f57703          	p.lw	a4,a5(a0)
1c019832:	c0002733          	p.insert	a4,zero,0,0
1c019836:	00e567a3          	p.sw	a4,a5(a0)
1c01983a:	40b2                	lw	ra,12(sp)
1c01983c:	4422                	lw	s0,8(sp)
1c01983e:	0141                	addi	sp,sp,16
1c019840:	8082                	ret

1c019842 <pi_timer_conf_init>:
1c019842:	010107b7          	lui	a5,0x1010
1c019846:	10078793          	addi	a5,a5,256 # 1010100 <__l1_heapsram_size+0xff0170>
1c01984a:	c15c                	sw	a5,4(a0)
1c01984c:	4785                	li	a5,1
1c01984e:	00052023          	sw	zero,0(a0)
1c019852:	00f50423          	sb	a5,8(a0)
1c019856:	8082                	ret

1c019858 <pi_timer_open>:
1c019858:	1101                	addi	sp,sp,-32
1c01985a:	ce06                	sw	ra,28(sp)
1c01985c:	cc22                	sw	s0,24(sp)
1c01985e:	ca26                	sw	s1,20(sp)
1c019860:	c84a                	sw	s2,16(sp)
1c019862:	c64e                	sw	s3,12(sp)
1c019864:	300474f3          	csrrci	s1,mstatus,8
1c019868:	00452903          	lw	s2,4(a0)
1c01986c:	00494783          	lbu	a5,4(s2)
1c019870:	c51c                	sw	a5,8(a0)
1c019872:	cf89                	beqz	a5,1c01988c <pi_timer_open+0x34>
1c019874:	0e37a163          	p.beqimm	a5,3,1c019956 <pi_timer_open+0xfe>
1c019878:	30049073          	csrw	mstatus,s1
1c01987c:	557d                	li	a0,-1
1c01987e:	40f2                	lw	ra,28(sp)
1c019880:	4462                	lw	s0,24(sp)
1c019882:	44d2                	lw	s1,20(sp)
1c019884:	4942                	lw	s2,16(sp)
1c019886:	49b2                	lw	s3,12(sp)
1c019888:	6105                	addi	sp,sp,32
1c01988a:	8082                	ret
1c01988c:	1c01d437          	lui	s0,0x1c01d
1c019890:	6b040413          	addi	s0,s0,1712 # 1c01d6b0 <g_timer_data>
1c019894:	505c                	lw	a5,36(s0)
1c019896:	00178713          	addi	a4,a5,1
1c01989a:	d058                	sw	a4,36(s0)
1c01989c:	ebcd                	bnez	a5,1c01994e <pi_timer_open+0xf6>
1c01989e:	00092783          	lw	a5,0(s2)
1c0198a2:	800009b7          	lui	s3,0x80000
1c0198a6:	08498993          	addi	s3,s3,132 # 80000084 <pulp__FC+0x80000085>
1c0198aa:	c81c                	sw	a5,16(s0)
1c0198ac:	00042023          	sw	zero,0(s0)
1c0198b0:	00042a23          	sw	zero,20(s0)
1c0198b4:	03342023          	sw	s3,32(s0)
1c0198b8:	4501                	li	a0,0
1c0198ba:	effff0ef          	jal	ra,1c0197b8 <__pi_timer_clock_freq_get>
1c0198be:	00092583          	lw	a1,0(s2)
1c0198c2:	000f4737          	lui	a4,0xf4
1c0198c6:	23f70713          	addi	a4,a4,575 # f423f <__l1_heapsram_size+0xd42af>
1c0198ca:	4781                	li	a5,0
1c0198cc:	22071813          	addi.d	a6,a4,0
1c0198d0:	72b52833          	p.macu.d	a6,a0,a1
1c0198d4:	000f4637          	lui	a2,0xf4
1c0198d8:	24060613          	addi	a2,a2,576 # f4240 <__l1_heapsram_size+0xd42b0>
1c0198dc:	4681                	li	a3,0
1c0198de:	22081513          	addi.d	a0,a6,0
1c0198e2:	919f70ef          	jal	ra,1c0111fa <__divdi3>
1c0198e6:	cc08                	sw	a0,24(s0)
1c0198e8:	00042e23          	sw	zero,28(s0)
1c0198ec:	305026f3          	csrr	a3,mtvec
1c0198f0:	1c0107b7          	lui	a5,0x1c010
1c0198f4:	c006b6b3          	p.bclr	a3,a3,0,0
1c0198f8:	69678793          	addi	a5,a5,1686 # 1c010696 <pi_octospi1_handler_asm>
1c0198fc:	8f95                	sub	a5,a5,a3
1c0198fe:	c1478633          	p.extract	a2,a5,0,20
1c019902:	06f00713          	li	a4,111
1c019906:	c1f62733          	p.insert	a4,a2,0,31
1c01990a:	d2178633          	p.extract	a2,a5,9,1
1c01990e:	d3562733          	p.insert	a4,a2,9,21
1c019912:	c0b78633          	p.extract	a2,a5,0,11
1c019916:	c1462733          	p.insert	a4,a2,0,20
1c01991a:	cec787b3          	p.extract	a5,a5,7,12
1c01991e:	cec7a733          	p.insert	a4,a5,7,12
1c019922:	d698                	sw	a4,40(a3)
1c019924:	1a1097b7          	lui	a5,0x1a109
1c019928:	40000713          	li	a4,1024
1c01992c:	c3d8                	sw	a4,4(a5)
1c01992e:	1a10b7b7          	lui	a5,0x1a10b
1c019932:	0007a623          	sw	zero,12(a5) # 1a10b00c <__l1_heapsram_end+0xa0eb00c>
1c019936:	577d                	li	a4,-1
1c019938:	00e7aa23          	sw	a4,20(a5)
1c01993c:	4685                	li	a3,1
1c01993e:	c016a9b3          	p.insert	s3,a3,0,1
1c019942:	00e7a823          	sw	a4,16(a5)
1c019946:	0007a423          	sw	zero,8(a5)
1c01994a:	0137a023          	sw	s3,0(a5)
1c01994e:	30049073          	csrw	mstatus,s1
1c019952:	4501                	li	a0,0
1c019954:	b72d                	j	1c01987e <pi_timer_open+0x26>
1c019956:	1c01d437          	lui	s0,0x1c01d
1c01995a:	6b040413          	addi	s0,s0,1712 # 1c01d6b0 <g_timer_data>
1c01995e:	487c                	lw	a5,84(s0)
1c019960:	00178713          	addi	a4,a5,1
1c019964:	c878                	sw	a4,84(s0)
1c019966:	f7e5                	bnez	a5,1c01994e <pi_timer_open+0xf6>
1c019968:	00694603          	lbu	a2,6(s2)
1c01996c:	00894783          	lbu	a5,8(s2)
1c019970:	00594683          	lbu	a3,5(s2)
1c019974:	4981                	li	s3,0
1c019976:	00794703          	lbu	a4,7(s2)
1c01997a:	c02629b3          	p.insert	s3,a2,0,2
1c01997e:	00092583          	lw	a1,0(s2)
1c019982:	c047a9b3          	p.insert	s3,a5,0,4
1c019986:	c056a9b3          	p.insert	s3,a3,0,5
1c01998a:	c07729b3          	p.insert	s3,a4,0,7
1c01998e:	c02c                	sw	a1,64(s0)
1c019990:	04c40ea3          	sb	a2,93(s0)
1c019994:	04d40e23          	sb	a3,92(s0)
1c019998:	04e40f23          	sb	a4,94(s0)
1c01999c:	04f40fa3          	sb	a5,95(s0)
1c0199a0:	02042823          	sw	zero,48(s0)
1c0199a4:	04042223          	sw	zero,68(s0)
1c0199a8:	05342823          	sw	s3,80(s0)
1c0199ac:	450d                	li	a0,3
1c0199ae:	e0bff0ef          	jal	ra,1c0197b8 <__pi_timer_clock_freq_get>
1c0199b2:	00092583          	lw	a1,0(s2)
1c0199b6:	000f4737          	lui	a4,0xf4
1c0199ba:	23f70713          	addi	a4,a4,575 # f423f <__l1_heapsram_size+0xd42af>
1c0199be:	4781                	li	a5,0
1c0199c0:	22071813          	addi.d	a6,a4,0
1c0199c4:	72b52833          	p.macu.d	a6,a0,a1
1c0199c8:	000f4637          	lui	a2,0xf4
1c0199cc:	24060613          	addi	a2,a2,576 # f4240 <__l1_heapsram_size+0xd42b0>
1c0199d0:	4681                	li	a3,0
1c0199d2:	22081513          	addi.d	a0,a6,0
1c0199d6:	825f70ef          	jal	ra,1c0111fa <__divdi3>
1c0199da:	c428                	sw	a0,72(s0)
1c0199dc:	04042623          	sw	zero,76(s0)
1c0199e0:	305026f3          	csrr	a3,mtvec
1c0199e4:	1c0107b7          	lui	a5,0x1c010
1c0199e8:	c006b6b3          	p.bclr	a3,a3,0,0
1c0199ec:	6b278793          	addi	a5,a5,1714 # 1c0106b2 <__octospi_handler_body+0x8>
1c0199f0:	8f95                	sub	a5,a5,a3
1c0199f2:	c1478633          	p.extract	a2,a5,0,20
1c0199f6:	06f00713          	li	a4,111
1c0199fa:	c1f62733          	p.insert	a4,a2,0,31
1c0199fe:	d2178633          	p.extract	a2,a5,9,1
1c019a02:	d3562733          	p.insert	a4,a2,9,21
1c019a06:	c0b78633          	p.extract	a2,a5,0,11
1c019a0a:	c1462733          	p.insert	a4,a2,0,20
1c019a0e:	cec787b3          	p.extract	a5,a5,7,12
1c019a12:	cec7a733          	p.insert	a4,a5,7,12
1c019a16:	dad8                	sw	a4,52(a3)
1c019a18:	1a1097b7          	lui	a5,0x1a109
1c019a1c:	6709                	lui	a4,0x2
1c019a1e:	c3d8                	sw	a4,4(a5)
1c019a20:	4785                	li	a5,1
1c019a22:	c017a9b3          	p.insert	s3,a5,0,1
1c019a26:	1a10c7b7          	lui	a5,0x1a10c
1c019a2a:	80078793          	addi	a5,a5,-2048 # 1a10b800 <__l1_heapsram_end+0xa0eb800>
1c019a2e:	577d                	li	a4,-1
1c019a30:	00e7aa23          	sw	a4,20(a5)
1c019a34:	0007a623          	sw	zero,12(a5)
1c019a38:	0137a223          	sw	s3,4(a5)
1c019a3c:	bf09                	j	1c01994e <pi_timer_open+0xf6>

1c019a3e <pi_timer_start>:
1c019a3e:	1141                	addi	sp,sp,-16
1c019a40:	c606                	sw	ra,12(sp)
1c019a42:	c422                	sw	s0,8(sp)
1c019a44:	c226                	sw	s1,4(sp)
1c019a46:	c04a                	sw	s2,0(sp)
1c019a48:	300474f3          	csrrci	s1,mstatus,8
1c019a4c:	4508                	lw	a0,8(a0)
1c019a4e:	1c01d7b7          	lui	a5,0x1c01d
1c019a52:	03000713          	li	a4,48
1c019a56:	0ff57413          	andi	s0,a0,255
1c019a5a:	00145693          	srli	a3,s0,0x1
1c019a5e:	6b078793          	addi	a5,a5,1712 # 1c01d6b0 <g_timer_data>
1c019a62:	42e687b3          	p.mac	a5,a3,a4
1c019a66:	5798                	lw	a4,40(a5)
1c019a68:	00170693          	addi	a3,a4,1 # 2001 <__STACK_IRQ_SIZE+0x1801>
1c019a6c:	d794                	sw	a3,40(a5)
1c019a6e:	cb11                	beqz	a4,1c019a82 <pi_timer_start+0x44>
1c019a70:	30049073          	csrw	mstatus,s1
1c019a74:	40b2                	lw	ra,12(sp)
1c019a76:	4422                	lw	s0,8(sp)
1c019a78:	4492                	lw	s1,4(sp)
1c019a7a:	4902                	lw	s2,0(sp)
1c019a7c:	4501                	li	a0,0
1c019a7e:	0141                	addi	sp,sp,16
1c019a80:	8082                	ret
1c019a82:	5398                	lw	a4,32(a5)
1c019a84:	4905                	li	s2,1
1c019a86:	fc143433          	p.bclr	s0,s0,30,1
1c019a8a:	c0092733          	p.insert	a4,s2,0,0
1c019a8e:	d398                	sw	a4,32(a5)
1c019a90:	0ff57513          	andi	a0,a0,255
1c019a94:	040a                	slli	s0,s0,0x2
1c019a96:	bf7ff0ef          	jal	ra,1c01968c <__timer_base_get>
1c019a9a:	0461                	addi	s0,s0,24
1c019a9c:	01256423          	p.sw	s2,s0(a0)
1c019aa0:	bfc1                	j	1c019a70 <pi_timer_start+0x32>

1c019aa2 <pi_timer_stop>:
1c019aa2:	1141                	addi	sp,sp,-16
1c019aa4:	c606                	sw	ra,12(sp)
1c019aa6:	c422                	sw	s0,8(sp)
1c019aa8:	c226                	sw	s1,4(sp)
1c019aaa:	300474f3          	csrrci	s1,mstatus,8
1c019aae:	4508                	lw	a0,8(a0)
1c019ab0:	1c01d7b7          	lui	a5,0x1c01d
1c019ab4:	03000713          	li	a4,48
1c019ab8:	0ff57413          	andi	s0,a0,255
1c019abc:	00145693          	srli	a3,s0,0x1
1c019ac0:	6b078793          	addi	a5,a5,1712 # 1c01d6b0 <g_timer_data>
1c019ac4:	42e687b3          	p.mac	a5,a3,a4
1c019ac8:	5798                	lw	a4,40(a5)
1c019aca:	177d                	addi	a4,a4,-1
1c019acc:	d798                	sw	a4,40(a5)
1c019ace:	cb01                	beqz	a4,1c019ade <pi_timer_stop+0x3c>
1c019ad0:	30049073          	csrw	mstatus,s1
1c019ad4:	40b2                	lw	ra,12(sp)
1c019ad6:	4422                	lw	s0,8(sp)
1c019ad8:	4492                	lw	s1,4(sp)
1c019ada:	0141                	addi	sp,sp,16
1c019adc:	8082                	ret
1c019ade:	5398                	lw	a4,32(a5)
1c019ae0:	0ff57513          	andi	a0,a0,255
1c019ae4:	fc143433          	p.bclr	s0,s0,30,1
1c019ae8:	c0002733          	p.insert	a4,zero,0,0
1c019aec:	d398                	sw	a4,32(a5)
1c019aee:	b9fff0ef          	jal	ra,1c01968c <__timer_base_get>
1c019af2:	040a                	slli	s0,s0,0x2
1c019af4:	20857783          	p.lw	a5,s0(a0)
1c019af8:	c00027b3          	p.insert	a5,zero,0,0
1c019afc:	00f56423          	p.sw	a5,s0(a0)
1c019b00:	bfc1                	j	1c019ad0 <pi_timer_stop+0x2e>

1c019b02 <pi_timer_close>:
1c019b02:	1141                	addi	sp,sp,-16
1c019b04:	c606                	sw	ra,12(sp)
1c019b06:	c422                	sw	s0,8(sp)
1c019b08:	c226                	sw	s1,4(sp)
1c019b0a:	c04a                	sw	s2,0(sp)
1c019b0c:	30047973          	csrrci	s2,mstatus,8
1c019b10:	4504                	lw	s1,8(a0)
1c019b12:	1c01d437          	lui	s0,0x1c01d
1c019b16:	6b040413          	addi	s0,s0,1712 # 1c01d6b0 <g_timer_data>
1c019b1a:	0ff4f793          	andi	a5,s1,255
1c019b1e:	8385                	srli	a5,a5,0x1
1c019b20:	03000713          	li	a4,48
1c019b24:	42e78433          	p.mac	s0,a5,a4
1c019b28:	505c                	lw	a5,36(s0)
1c019b2a:	17fd                	addi	a5,a5,-1
1c019b2c:	d05c                	sw	a5,36(s0)
1c019b2e:	cb89                	beqz	a5,1c019b40 <pi_timer_close+0x3e>
1c019b30:	30091073          	csrw	mstatus,s2
1c019b34:	40b2                	lw	ra,12(sp)
1c019b36:	4422                	lw	s0,8(sp)
1c019b38:	4492                	lw	s1,4(sp)
1c019b3a:	4902                	lw	s2,0(sp)
1c019b3c:	0141                	addi	sp,sp,16
1c019b3e:	8082                	ret
1c019b40:	f63ff0ef          	jal	ra,1c019aa2 <pi_timer_stop>
1c019b44:	22001793          	addi.d	a5,zero,0
1c019b48:	00042023          	sw	zero,0(s0)
1c019b4c:	c81c                	sw	a5,16(s0)
1c019b4e:	01042a23          	sw	a6,20(s0)
1c019b52:	02042023          	sw	zero,32(s0)
1c019b56:	30091073          	csrw	mstatus,s2
1c019b5a:	4785                	li	a5,1
1c019b5c:	04a9                	addi	s1,s1,10
1c019b5e:	009794b3          	sll	s1,a5,s1
1c019b62:	1a1097b7          	lui	a5,0x1a109
1c019b66:	c784                	sw	s1,8(a5)
1c019b68:	b7f1                	j	1c019b34 <pi_timer_close+0x32>

1c019b6a <pi_long_timer_current_value_read>:
1c019b6a:	00854503          	lbu	a0,8(a0)
1c019b6e:	1101                	addi	sp,sp,-32
1c019b70:	ce06                	sw	ra,28(sp)
1c019b72:	c62e                	sw	a1,12(sp)
1c019b74:	b19ff0ef          	jal	ra,1c01968c <__timer_base_get>
1c019b78:	00852783          	lw	a5,8(a0)
1c019b7c:	45b2                	lw	a1,12(sp)
1c019b7e:	c19c                	sw	a5,0(a1)
1c019b80:	0005a223          	sw	zero,4(a1)
1c019b84:	00c52503          	lw	a0,12(a0)
1c019b88:	872a                	mv	a4,a0
1c019b8a:	02000613          	li	a2,32
1c019b8e:	4681                	li	a3,0
1c019b90:	4781                	li	a5,0
1c019b92:	20c71733          	sll.d	a4,a4,a2
1c019b96:	4190                	lw	a2,0(a1)
1c019b98:	41d4                	lw	a3,4(a1)
1c019b9a:	40f2                	lw	ra,28(sp)
1c019b9c:	4501                	li	a0,0
1c019b9e:	20e60733          	add.d	a4,a2,a4
1c019ba2:	c198                	sw	a4,0(a1)
1c019ba4:	c1dc                	sw	a5,4(a1)
1c019ba6:	6105                	addi	sp,sp,32
1c019ba8:	8082                	ret

1c019baa <pi_timer_task_add>:
1c019baa:	715d                	addi	sp,sp,-80
1c019bac:	c686                	sw	ra,76(sp)
1c019bae:	c4a2                	sw	s0,72(sp)
1c019bb0:	c2a6                	sw	s1,68(sp)
1c019bb2:	c0ca                	sw	s2,64(sp)
1c019bb4:	de4e                	sw	s3,60(sp)
1c019bb6:	dc52                	sw	s4,56(sp)
1c019bb8:	da56                	sw	s5,52(sp)
1c019bba:	d85a                	sw	s6,48(sp)
1c019bbc:	d65e                	sw	s7,44(sp)
1c019bbe:	d462                	sw	s8,40(sp)
1c019bc0:	d266                	sw	s9,36(sp)
1c019bc2:	d06a                	sw	s10,32(sp)
1c019bc4:	ce6e                	sw	s11,28(sp)
1c019bc6:	30047b73          	csrrci	s6,mstatus,8
1c019bca:	451c                	lw	a5,8(a0)
1c019bcc:	22001693          	addi.d	a3,zero,0
1c019bd0:	c436                	sw	a3,8(sp)
1c019bd2:	0ff7f413          	andi	s0,a5,255
1c019bd6:	c63a                	sw	a4,12(sp)
1c019bd8:	c40d                	beqz	s0,1c019c02 <pi_timer_task_add+0x58>
1c019bda:	02342463          	p.beqimm	s0,3,1c019c02 <pi_timer_task_add+0x58>
1c019bde:	300b1073          	csrw	mstatus,s6
1c019be2:	5555                	li	a0,-11
1c019be4:	40b6                	lw	ra,76(sp)
1c019be6:	4426                	lw	s0,72(sp)
1c019be8:	4496                	lw	s1,68(sp)
1c019bea:	4906                	lw	s2,64(sp)
1c019bec:	59f2                	lw	s3,60(sp)
1c019bee:	5a62                	lw	s4,56(sp)
1c019bf0:	5ad2                	lw	s5,52(sp)
1c019bf2:	5b42                	lw	s6,48(sp)
1c019bf4:	5bb2                	lw	s7,44(sp)
1c019bf6:	5c22                	lw	s8,40(sp)
1c019bf8:	5c92                	lw	s9,36(sp)
1c019bfa:	5d02                	lw	s10,32(sp)
1c019bfc:	4df2                	lw	s11,28(sp)
1c019bfe:	6161                	addi	sp,sp,80
1c019c00:	8082                	ret
1c019c02:	892a                	mv	s2,a0
1c019c04:	0ff7f513          	andi	a0,a5,255
1c019c08:	8a32                	mv	s4,a2
1c019c0a:	8dae                	mv	s11,a1
1c019c0c:	a81ff0ef          	jal	ra,1c01968c <__timer_base_get>
1c019c10:	002c                	addi	a1,sp,8
1c019c12:	8aaa                	mv	s5,a0
1c019c14:	854a                	mv	a0,s2
1c019c16:	f55ff0ef          	jal	ra,1c019b6a <pi_long_timer_current_value_read>
1c019c1a:	1c01dbb7          	lui	s7,0x1c01d
1c019c1e:	03000793          	li	a5,48
1c019c22:	00145493          	srli	s1,s0,0x1
1c019c26:	6b0b8d13          	addi	s10,s7,1712 # 1c01d6b0 <g_timer_data>
1c019c2a:	42f48d33          	p.mac	s10,s1,a5
1c019c2e:	856e                	mv	a0,s11
1c019c30:	4581                	li	a1,0
1c019c32:	4981                	li	s3,0
1c019c34:	010d2c03          	lw	s8,16(s10)
1c019c38:	014d2c83          	lw	s9,20(s10)
1c019c3c:	220c1613          	addi.d	a2,s8,0
1c019c40:	c69f70ef          	jal	ra,1c0118a8 <__umoddi3>
1c019c44:	8dc9                	or	a1,a1,a0
1c019c46:	00b03933          	snez	s2,a1
1c019c4a:	220c1613          	addi.d	a2,s8,0
1c019c4e:	856e                	mv	a0,s11
1c019c50:	4581                	li	a1,0
1c019c52:	91bf70ef          	jal	ra,1c01156c <__udivdi3>
1c019c56:	01cd2783          	lw	a5,28(s10)
1c019c5a:	20a90533          	add.d	a0,s2,a0
1c019c5e:	018d2903          	lw	s2,24(s10)
1c019c62:	02a787b3          	mul	a5,a5,a0
1c019c66:	46b2                	lw	a3,12(sp)
1c019c68:	4622                	lw	a2,8(sp)
1c019c6a:	6b0b8713          	addi	a4,s7,1712
1c019c6e:	42b907b3          	p.mac	a5,s2,a1
1c019c72:	72a95533          	p.mulu.d	a0,s2,a0
1c019c76:	95be                	add	a1,a1,a5
1c019c78:	000d2783          	lw	a5,0(s10)
1c019c7c:	20c50533          	add.d	a0,a0,a2
1c019c80:	04aa2823          	sw	a0,80(s4)
1c019c84:	04ba2a23          	sw	a1,84(s4)
1c019c88:	4681                	li	a3,0
1c019c8a:	cb89                	beqz	a5,1c019c9c <pi_timer_task_add+0xf2>
1c019c8c:	0507a803          	lw	a6,80(a5) # 1a109050 <__l1_heapsram_end+0xa0e9050>
1c019c90:	0547a883          	lw	a7,84(a5)
1c019c94:	34a83833          	p.sletu.d	a6,a6,a0
1c019c98:	00081b63          	bnez	a6,1c019cae <pi_timer_task_add+0x104>
1c019c9c:	ce81                	beqz	a3,1c019cb4 <pi_timer_task_add+0x10a>
1c019c9e:	0146a023          	sw	s4,0(a3)
1c019ca2:	00fa2023          	sw	a5,0(s4)
1c019ca6:	300b1073          	csrw	mstatus,s6
1c019caa:	4501                	li	a0,0
1c019cac:	bf25                	j	1c019be4 <pi_timer_task_add+0x3a>
1c019cae:	86be                	mv	a3,a5
1c019cb0:	439c                	lw	a5,0(a5)
1c019cb2:	bfe1                	j	1c019c8a <pi_timer_task_add+0xe0>
1c019cb4:	03000693          	li	a3,48
1c019cb8:	42d48733          	p.mac	a4,s1,a3
1c019cbc:	01472023          	sw	s4,0(a4)
1c019cc0:	e411                	bnez	s0,1c019ccc <pi_timer_task_add+0x122>
1c019cc2:	00baaa23          	sw	a1,20(s5)
1c019cc6:	00aaa823          	sw	a0,16(s5)
1c019cca:	bfe1                	j	1c019ca2 <pi_timer_task_add+0xf8>
1c019ccc:	fc143433          	p.bclr	s0,s0,30,1
1c019cd0:	040a                	slli	s0,s0,0x2
1c019cd2:	01040693          	addi	a3,s0,16
1c019cd6:	00aae6a3          	p.sw	a0,a3(s5)
1c019cda:	0461                	addi	s0,s0,24
1c019cdc:	4705                	li	a4,1
1c019cde:	00eae423          	p.sw	a4,s0(s5)
1c019ce2:	b7c1                	j	1c019ca2 <pi_timer_task_add+0xf8>

1c019ce4 <vApplicationStackOverflowHook>:
1c019ce4:	1141                	addi	sp,sp,-16
1c019ce6:	c422                	sw	s0,8(sp)
1c019ce8:	842a                	mv	s0,a0
1c019cea:	1c01c537          	lui	a0,0x1c01c
1c019cee:	d7450513          	addi	a0,a0,-652 # 1c01bd74 <__clz_tab+0x32c>
1c019cf2:	c606                	sw	ra,12(sp)
1c019cf4:	189010ef          	jal	ra,1c01b67c <printf_>
1c019cf8:	4058                	lw	a4,4(s0)
1c019cfa:	4014                	lw	a3,0(s0)
1c019cfc:	4410                	lw	a2,8(s0)
1c019cfe:	1c01c537          	lui	a0,0x1c01c
1c019d02:	85a2                	mv	a1,s0
1c019d04:	d9850513          	addi	a0,a0,-616 # 1c01bd98 <__clz_tab+0x350>
1c019d08:	175010ef          	jal	ra,1c01b67c <printf_>
1c019d0c:	757d                	lui	a0,0xfffff
1c019d0e:	f1f50513          	addi	a0,a0,-225 # ffffef1f <pulp__FC+0xffffef20>
1c019d12:	244d                	jal	1c019fb4 <exit>

1c019d14 <vSetPendSV>:
1c019d14:	342027f3          	csrr	a5,mcause
1c019d18:	0007d763          	bgez	a5,1c019d26 <vSetPendSV+0x12>
1c019d1c:	1a1097b7          	lui	a5,0x1a109
1c019d20:	4709                	li	a4,2
1c019d22:	cb98                	sw	a4,16(a5)
1c019d24:	8082                	ret
1c019d26:	1141                	addi	sp,sp,-16
1c019d28:	c606                	sw	ra,12(sp)
1c019d2a:	c422                	sw	s0,8(sp)
1c019d2c:	30047473          	csrrci	s0,mstatus,8
1c019d30:	e30f60ef          	jal	ra,1c010360 <asm_do_yield>
1c019d34:	30041073          	csrw	mstatus,s0
1c019d38:	40b2                	lw	ra,12(sp)
1c019d3a:	4422                	lw	s0,8(sp)
1c019d3c:	0141                	addi	sp,sp,16
1c019d3e:	8082                	ret

1c019d40 <vSetPendSVForce>:
1c019d40:	342027f3          	csrr	a5,mcause
1c019d44:	0007d763          	bgez	a5,1c019d52 <vSetPendSVForce+0x12>
1c019d48:	1a1097b7          	lui	a5,0x1a109
1c019d4c:	4709                	li	a4,2
1c019d4e:	cb98                	sw	a4,16(a5)
1c019d50:	8082                	ret
1c019d52:	1141                	addi	sp,sp,-16
1c019d54:	c606                	sw	ra,12(sp)
1c019d56:	c422                	sw	s0,8(sp)
1c019d58:	30047473          	csrrci	s0,mstatus,8
1c019d5c:	e70f60ef          	jal	ra,1c0103cc <asm_do_yield_force>
1c019d60:	30041073          	csrw	mstatus,s0
1c019d64:	40b2                	lw	ra,12(sp)
1c019d66:	4422                	lw	s0,8(sp)
1c019d68:	0141                	addi	sp,sp,16
1c019d6a:	8082                	ret

1c019d6c <_procid>:
1c019d6c:	f1402573          	csrr	a0,mhartid
1c019d70:	f6453533          	p.bclr	a0,a0,27,4
1c019d74:	8082                	ret

1c019d76 <_clusterid>:
1c019d76:	f1402573          	csrr	a0,mhartid
1c019d7a:	8115                	srli	a0,a0,0x5
1c019d7c:	8082                	ret

1c019d7e <_priv_drop>:
1c019d7e:	300027f3          	csrr	a5,mstatus
1c019d82:	c2b7b7b3          	p.bclr	a5,a5,1,11
1c019d86:	30079073          	csrw	mstatus,a5
1c019d8a:	4501                	li	a0,0
1c019d8c:	8082                	ret

1c019d8e <_isr_default>:
1c019d8e:	1141                	addi	sp,sp,-16
1c019d90:	c606                	sw	ra,12(sp)
1c019d92:	0001                	nop
1c019d94:	4501                	li	a0,0
1c019d96:	2c39                	jal	1c019fb4 <exit>

1c019d98 <_puts>:
1c019d98:	1141                	addi	sp,sp,-16
1c019d9a:	c606                	sw	ra,12(sp)
1c019d9c:	9f2fb0ef          	jal	ra,1c014f8e <puts>
1c019da0:	40b2                	lw	ra,12(sp)
1c019da2:	4501                	li	a0,0
1c019da4:	0141                	addi	sp,sp,16
1c019da6:	8082                	ret

1c019da8 <pmp_exception_handler>:
1c019da8:	1c01c537          	lui	a0,0x1c01c
1c019dac:	1141                	addi	sp,sp,-16
1c019dae:	df850513          	addi	a0,a0,-520 # 1c01bdf8 <__func__.47299+0x38>
1c019db2:	c606                	sw	ra,12(sp)
1c019db4:	0c9010ef          	jal	ra,1c01b67c <printf_>
1c019db8:	343015f3          	csrrw	a1,0x343,zero
1c019dbc:	1c01c537          	lui	a0,0x1c01c
1c019dc0:	e1050513          	addi	a0,a0,-496 # 1c01be10 <__func__.47299+0x50>
1c019dc4:	0b9010ef          	jal	ra,1c01b67c <printf_>
1c019dc8:	a001                	j	1c019dc8 <pmp_exception_handler+0x20>

1c019dca <xip_insn_exception_handler>:
1c019dca:	1141                	addi	sp,sp,-16
1c019dcc:	c606                	sw	ra,12(sp)
1c019dce:	c422                	sw	s0,8(sp)
1c019dd0:	34102673          	csrr	a2,mepc
1c019dd4:	343016f3          	csrrw	a3,0x343,zero
1c019dd8:	1a150437          	lui	s0,0x1a150
1c019ddc:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c019de0:	444c                	lw	a1,12(s0)
1c019de2:	1c01c537          	lui	a0,0x1c01c
1c019de6:	00c70793          	addi	a5,a4,12
1c019dea:	d005b5b3          	p.bclr	a1,a1,8,0
1c019dee:	e4050513          	addi	a0,a0,-448 # 1c01be40 <__func__.47299+0x80>
1c019df2:	08b010ef          	jal	ra,1c01b67c <printf_>
1c019df6:	444c                	lw	a1,12(s0)
1c019df8:	1c01c537          	lui	a0,0x1c01c
1c019dfc:	e6050513          	addi	a0,a0,-416 # 1c01be60 <__func__.47299+0xa0>
1c019e00:	d005b5b3          	p.bclr	a1,a1,8,0
1c019e04:	079010ef          	jal	ra,1c01b67c <printf_>
1c019e08:	20000793          	li	a5,512
1c019e0c:	c45c                	sw	a5,12(s0)
1c019e0e:	444c                	lw	a1,12(s0)
1c019e10:	1c01c537          	lui	a0,0x1c01c
1c019e14:	e8050513          	addi	a0,a0,-384 # 1c01be80 <__func__.47299+0xc0>
1c019e18:	d005b5b3          	p.bclr	a1,a1,8,0
1c019e1c:	061010ef          	jal	ra,1c01b67c <printf_>
1c019e20:	4501                	li	a0,0
1c019e22:	2a49                	jal	1c019fb4 <exit>

1c019e24 <xip_data_exception_handler>:
1c019e24:	1141                	addi	sp,sp,-16
1c019e26:	c606                	sw	ra,12(sp)
1c019e28:	c422                	sw	s0,8(sp)
1c019e2a:	c226                	sw	s1,4(sp)
1c019e2c:	34102673          	csrr	a2,mepc
1c019e30:	34301473          	csrrw	s0,0x343,zero
1c019e34:	1a1504b7          	lui	s1,0x1a150
1c019e38:	44cc                	lw	a1,12(s1)
1c019e3a:	4e002703          	lw	a4,1248(zero) # 4e0 <pxCurrentTCB>
1c019e3e:	1c01c537          	lui	a0,0x1c01c
1c019e42:	86a2                	mv	a3,s0
1c019e44:	00c70793          	addi	a5,a4,12
1c019e48:	0ff5f593          	andi	a1,a1,255
1c019e4c:	e2050513          	addi	a0,a0,-480 # 1c01be20 <__func__.47299+0x60>
1c019e50:	02d010ef          	jal	ra,1c01b67c <printf_>
1c019e54:	44dc                	lw	a5,12(s1)
1c019e56:	1c01d7b7          	lui	a5,0x1c01d
1c019e5a:	7a47a303          	lw	t1,1956(a5) # 1c01d7a4 <g_xip_tlb_data_handler>
1c019e5e:	00030863          	beqz	t1,1c019e6e <xip_data_exception_handler+0x4a>
1c019e62:	8522                	mv	a0,s0
1c019e64:	4422                	lw	s0,8(sp)
1c019e66:	40b2                	lw	ra,12(sp)
1c019e68:	4492                	lw	s1,4(sp)
1c019e6a:	0141                	addi	sp,sp,16
1c019e6c:	8302                	jr	t1
1c019e6e:	4501                	li	a0,0
1c019e70:	2291                	jal	1c019fb4 <exit>

1c019e72 <_ill_insn_print>:
1c019e72:	4e002683          	lw	a3,1248(zero) # 4e0 <pxCurrentTCB>
1c019e76:	862a                	mv	a2,a0
1c019e78:	1c01c5b7          	lui	a1,0x1c01c
1c019e7c:	1c01c537          	lui	a0,0x1c01c
1c019e80:	1141                	addi	sp,sp,-16
1c019e82:	00c68713          	addi	a4,a3,12
1c019e86:	dc058593          	addi	a1,a1,-576 # 1c01bdc0 <__func__.47299>
1c019e8a:	dd050513          	addi	a0,a0,-560 # 1c01bdd0 <__func__.47299+0x10>
1c019e8e:	c606                	sw	ra,12(sp)
1c019e90:	7ec010ef          	jal	ra,1c01b67c <printf_>
1c019e94:	f3900513          	li	a0,-199
1c019e98:	2a31                	jal	1c019fb4 <exit>

1c019e9a <pxPortInitialiseStack>:
1c019e9a:	feed57b7          	lui	a5,0xfeed5
1c019e9e:	1ac78793          	addi	a5,a5,428 # feed51ac <pulp__FC+0xfeed51ad>
1c019ea2:	fef52e2b          	p.sw	a5,-4(a0!)
1c019ea6:	c60537b3          	p.bclr	a5,a0,3,0
1c019eaa:	f7078513          	addi	a0,a5,-144
1c019eae:	86aa                	mv	a3,a0
1c019eb0:	024250fb          	lp.setupi	x1,36,1c019eb8 <pxPortInitialiseStack+0x1e>
1c019eb4:	0006a22b          	p.sw	zero,4(a3!)
1c019eb8:	0001                	nop
1c019eba:	6709                	lui	a4,0x2
1c019ebc:	80070713          	addi	a4,a4,-2048 # 1800 <__STACK_IRQ_SIZE+0x1000>
1c019ec0:	f6b7a823          	sw	a1,-144(a5)
1c019ec4:	f6c7aa23          	sw	a2,-140(a5)
1c019ec8:	fee7a023          	sw	a4,-32(a5)
1c019ecc:	feb7a223          	sw	a1,-28(a5)
1c019ed0:	8082                	ret

1c019ed2 <prvSetupTimerInterrupt>:
1c019ed2:	06400513          	li	a0,100
1c019ed6:	a94fb06f          	j	1c01516a <system_setup_systick>

1c019eda <__pi_os_evt_sig_init>:
1c019eda:	1141                	addi	sp,sp,-16
1c019edc:	c422                	sw	s0,8(sp)
1c019ede:	c606                	sw	ra,12(sp)
1c019ee0:	842a                	mv	s0,a0
1c019ee2:	a61fe0ef          	jal	ra,1c018942 <os_evt_sig_init>
1c019ee6:	8522                	mv	a0,s0
1c019ee8:	40b2                	lw	ra,12(sp)
1c019eea:	4422                	lw	s0,8(sp)
1c019eec:	0141                	addi	sp,sp,16
1c019eee:	8082                	ret

1c019ef0 <__pi_os_evt_callback_no_irq_init>:
1c019ef0:	1141                	addi	sp,sp,-16
1c019ef2:	c422                	sw	s0,8(sp)
1c019ef4:	c606                	sw	ra,12(sp)
1c019ef6:	842a                	mv	s0,a0
1c019ef8:	a81fe0ef          	jal	ra,1c018978 <os_evt_callback_init>
1c019efc:	8522                	mv	a0,s0
1c019efe:	40b2                	lw	ra,12(sp)
1c019f00:	4422                	lw	s0,8(sp)
1c019f02:	0141                	addi	sp,sp,16
1c019f04:	8082                	ret

1c019f06 <__pi_os_evt_callback_irq_init>:
1c019f06:	1141                	addi	sp,sp,-16
1c019f08:	c422                	sw	s0,8(sp)
1c019f0a:	c606                	sw	ra,12(sp)
1c019f0c:	842a                	mv	s0,a0
1c019f0e:	abbfe0ef          	jal	ra,1c0189c8 <os_evt_callback_irq_init>
1c019f12:	8522                	mv	a0,s0
1c019f14:	40b2                	lw	ra,12(sp)
1c019f16:	4422                	lw	s0,8(sp)
1c019f18:	0141                	addi	sp,sp,16
1c019f1a:	8082                	ret

1c019f1c <__pi_os_evt_wait>:
1c019f1c:	ab9fe06f          	j	1c0189d4 <os_evt_wait>

1c019f20 <__pi_os_evt_push>:
1c019f20:	d1bfe06f          	j	1c018c3a <os_evt_push>

1c019f24 <__pi_os_evt_push_delayed_us>:
1c019f24:	d4bfe06f          	j	1c018c6e <os_evt_push_delayed_us>

1c019f28 <__os_native_yield_force>:
1c019f28:	e19ff06f          	j	1c019d40 <vSetPendSVForce>

1c019f2c <__pi_os_thread_create_task>:
1c019f2c:	87b2                	mv	a5,a2
1c019f2e:	0ff77713          	andi	a4,a4,255
1c019f32:	862e                	mv	a2,a1
1c019f34:	85aa                	mv	a1,a0
1c019f36:	853e                	mv	a0,a5
1c019f38:	ddaff06f          	j	1c019512 <task_native_create>

1c019f3c <__pi_os_thread_start>:
1c019f3c:	f4eff06f          	j	1c01968a <task_native_start>

1c019f40 <__pi_os_thread_has_event>:
1c019f40:	d08ff06f          	j	1c019448 <task_native_has_event>

1c019f44 <__pi_os_thread_exec_event>:
1c019f44:	d84ff06f          	j	1c0194c8 <task_native_exec_event>

1c019f48 <__pi_os_thread_set_main_handler>:
1c019f48:	1c01d7b7          	lui	a5,0x1c01d
1c019f4c:	7aa7a623          	sw	a0,1964(a5) # 1c01d7ac <xHandler0>
1c019f50:	8082                	ret

1c019f52 <__os_init>:
1c019f52:	1141                	addi	sp,sp,-16
1c019f54:	c606                	sw	ra,12(sp)
1c019f56:	c422                	sw	s0,8(sp)
1c019f58:	d43fe0ef          	jal	ra,1c018c9a <sched_init>
1c019f5c:	1c01c637          	lui	a2,0x1c01c
1c019f60:	1c011537          	lui	a0,0x1c011
1c019f64:	4705                	li	a4,1
1c019f66:	20000693          	li	a3,512
1c019f6a:	ea860613          	addi	a2,a2,-344 # 1c01bea8 <__func__.47299+0xe8>
1c019f6e:	4581                	li	a1,0
1c019f70:	1d250513          	addi	a0,a0,466 # 1c0111d2 <__main>
1c019f74:	fb9ff0ef          	jal	ra,1c019f2c <__pi_os_thread_create_task>
1c019f78:	e919                	bnez	a0,1c019f8e <__os_init+0x3c>
1c019f7a:	1c01c537          	lui	a0,0x1c01c
1c019f7e:	eb050513          	addi	a0,a0,-336 # 1c01beb0 <__func__.47299+0xf0>
1c019f82:	6fa010ef          	jal	ra,1c01b67c <printf_>
1c019f86:	757d                	lui	a0,0xfffff
1c019f88:	f1f50513          	addi	a0,a0,-225 # ffffef1f <pulp__FC+0xffffef20>
1c019f8c:	2025                	jal	1c019fb4 <exit>
1c019f8e:	842a                	mv	s0,a0
1c019f90:	fb9ff0ef          	jal	ra,1c019f48 <__pi_os_thread_set_main_handler>
1c019f94:	8522                	mv	a0,s0
1c019f96:	fa7ff0ef          	jal	ra,1c019f3c <__pi_os_thread_start>
1c019f9a:	1a1097b7          	lui	a5,0x1a109
1c019f9e:	4709                	li	a4,2
1c019fa0:	c3d8                	sw	a4,4(a5)
1c019fa2:	300467f3          	csrrsi	a5,mstatus,8
1c019fa6:	897fe0ef          	jal	ra,1c01883c <vTaskStartScheduler>
1c019faa:	40b2                	lw	ra,12(sp)
1c019fac:	4422                	lw	s0,8(sp)
1c019fae:	4501                	li	a0,0
1c019fb0:	0141                	addi	sp,sp,16
1c019fb2:	8082                	ret

1c019fb4 <exit>:
1c019fb4:	1141                	addi	sp,sp,-16
1c019fb6:	c606                	sw	ra,12(sp)
1c019fb8:	a66fb0ef          	jal	ra,1c01521e <system_exit>
1c019fbc:	a001                	j	1c019fbc <exit+0x8>

1c019fbe <memcpy>:
1c019fbe:	87aa                	mv	a5,a0
1c019fc0:	962e                	add	a2,a2,a1
1c019fc2:	00c59363          	bne	a1,a2,1c019fc8 <memcpy+0xa>
1c019fc6:	8082                	ret
1c019fc8:	0015c70b          	p.lbu	a4,1(a1!)
1c019fcc:	00e780ab          	p.sb	a4,1(a5!) # 1a109001 <__l1_heapsram_end+0xa0e9001>
1c019fd0:	bfcd                	j	1c019fc2 <memcpy+0x4>

1c019fd2 <memset>:
1c019fd2:	87aa                	mv	a5,a0
1c019fd4:	962a                	add	a2,a2,a0
1c019fd6:	00c79363          	bne	a5,a2,1c019fdc <memset+0xa>
1c019fda:	8082                	ret
1c019fdc:	00b780ab          	p.sb	a1,1(a5!)
1c019fe0:	bfdd                	j	1c019fd6 <memset+0x4>

1c019fe2 <strcmp>:
1c019fe2:	0015478b          	p.lbu	a5,1(a0!)
1c019fe6:	0015c70b          	p.lbu	a4,1(a1!)
1c019fea:	c399                	beqz	a5,1c019ff0 <strcmp+0xe>
1c019fec:	fee78be3          	beq	a5,a4,1c019fe2 <strcmp>
1c019ff0:	40e78533          	sub	a0,a5,a4
1c019ff4:	8082                	ret

1c019ff6 <strncmp>:
1c019ff6:	962e                	add	a2,a2,a1
1c019ff8:	00c58c63          	beq	a1,a2,1c01a010 <strncmp+0x1a>
1c019ffc:	0015478b          	p.lbu	a5,1(a0!)
1c01a000:	0015c70b          	p.lbu	a4,1(a1!)
1c01a004:	c399                	beqz	a5,1c01a00a <strncmp+0x14>
1c01a006:	fee789e3          	beq	a5,a4,1c019ff8 <strncmp+0x2>
1c01a00a:	40e78533          	sub	a0,a5,a4
1c01a00e:	8082                	ret
1c01a010:	4501                	li	a0,0
1c01a012:	8082                	ret

1c01a014 <strlen>:
1c01a014:	872a                	mv	a4,a0
1c01a016:	87aa                	mv	a5,a0
1c01a018:	40e78533          	sub	a0,a5,a4
1c01a01c:	0017c68b          	p.lbu	a3,1(a5!)
1c01a020:	fee5                	bnez	a3,1c01a018 <strlen+0x4>
1c01a022:	8082                	ret

1c01a024 <pi_max_freq_get>:
1c01a024:	1141                	addi	sp,sp,-16
1c01a026:	c606                	sw	ra,12(sp)
1c01a028:	e581                	bnez	a1,1c01a030 <pi_max_freq_get+0xc>
1c01a02a:	4501                	li	a0,0
1c01a02c:	2849                	jal	1c01a0be <pi_pmu_voltage_get>
1c01a02e:	85aa                	mv	a1,a0
1c01a030:	03200793          	li	a5,50
1c01a034:	d7658593          	addi	a1,a1,-650
1c01a038:	02f5d5b3          	divu	a1,a1,a5
1c01a03c:	1c0007b7          	lui	a5,0x1c000
1c01a040:	5b078793          	addi	a5,a5,1456 # 1c0005b0 <pi_max_freq>
1c01a044:	40b2                	lw	ra,12(sp)
1c01a046:	0586                	slli	a1,a1,0x1
1c01a048:	95be                	add	a1,a1,a5
1c01a04a:	0005d503          	lhu	a0,0(a1)
1c01a04e:	000f47b7          	lui	a5,0xf4
1c01a052:	24078793          	addi	a5,a5,576 # f4240 <__l1_heapsram_size+0xd42b0>
1c01a056:	02f50533          	mul	a0,a0,a5
1c01a05a:	0141                	addi	sp,sp,16
1c01a05c:	8082                	ret

1c01a05e <pi_ref_fast_clock_frequency_set>:
1c01a05e:	017707b7          	lui	a5,0x1770
1c01a062:	03f78793          	addi	a5,a5,63 # 177003f <__l1_heapsram_size+0x17500af>
1c01a066:	02a7d7b3          	divu	a5,a5,a0
1c01a06a:	10079733          	p.fl1	a4,a5
1c01a06e:	1007b7b3          	p.cnt	a5,a5
1c01a072:	469d                	li	a3,7
1c01a074:	02e6e563          	bltu	a3,a4,1c01a09e <pi_ref_fast_clock_frequency_set+0x40>
1c01a078:	4605                	li	a2,1
1c01a07a:	56fd                	li	a3,-1
1c01a07c:	02f66263          	bltu	a2,a5,1c01a0a0 <pi_ref_fast_clock_frequency_set+0x42>
1c01a080:	1141                	addi	sp,sp,-16
1c01a082:	c0374733          	p.bset	a4,a4,0,3
1c01a086:	1a1047b7          	lui	a5,0x1a104
1c01a08a:	c606                	sw	ra,12(sp)
1c01a08c:	10e7ac23          	sw	a4,280(a5) # 1a104118 <__l1_heapsram_end+0xa0e4118>
1c01a090:	934fb0ef          	jal	ra,1c0151c4 <system_core_clock_update>
1c01a094:	40b2                	lw	ra,12(sp)
1c01a096:	4681                	li	a3,0
1c01a098:	8536                	mv	a0,a3
1c01a09a:	0141                	addi	sp,sp,16
1c01a09c:	8082                	ret
1c01a09e:	56fd                	li	a3,-1
1c01a0a0:	8536                	mv	a0,a3
1c01a0a2:	8082                	ret

1c01a0a4 <pi_ref_fast_clock_frequency_get>:
1c01a0a4:	1a1047b7          	lui	a5,0x1a104
1c01a0a8:	1187a783          	lw	a5,280(a5) # 1a104118 <__l1_heapsram_end+0xa0e4118>
1c01a0ac:	01770537          	lui	a0,0x1770
1c01a0b0:	f837b7b3          	p.bclr	a5,a5,28,3
1c01a0b4:	03f50513          	addi	a0,a0,63 # 177003f <__l1_heapsram_size+0x17500af>
1c01a0b8:	00f55533          	srl	a0,a0,a5
1c01a0bc:	8082                	ret

1c01a0be <pi_pmu_voltage_get>:
1c01a0be:	54c02503          	lw	a0,1356(zero) # 54c <pi_pmu_soc_voltage>
1c01a0c2:	8082                	ret

1c01a0c4 <pos_alloc_init_l1>:
1c01a0c4:	1c01d7b7          	lui	a5,0x1c01d
1c01a0c8:	02400713          	li	a4,36
1c01a0cc:	7d878793          	addi	a5,a5,2008 # 1c01d7d8 <pos_alloc_l1>
1c01a0d0:	42e507b3          	p.mac	a5,a0,a4
1c01a0d4:	853e                	mv	a0,a5
1c01a0d6:	a429                	j	1c01a2e0 <pos_alloc_init>

1c01a0d8 <pos_allocs_init>:
1c01a0d8:	1141                	addi	sp,sp,-16
1c01a0da:	1c0015b7          	lui	a1,0x1c001
1c01a0de:	c606                	sw	ra,12(sp)
1c01a0e0:	c422                	sw	s0,8(sp)
1c01a0e2:	c226                	sw	s1,4(sp)
1c01a0e4:	c04a                	sw	s2,0(sp)
1c01a0e6:	dc058793          	addi	a5,a1,-576 # 1c000dc0 <__heapfcram_start>
1c01a0ea:	1c010637          	lui	a2,0x1c010
1c01a0ee:	08c7c663          	blt	a5,a2,1c01a17a <pos_allocs_init+0xa2>
1c01a0f2:	4581                	li	a1,0
1c01a0f4:	4601                	li	a2,0
1c01a0f6:	1c01e437          	lui	s0,0x1c01e
1c01a0fa:	86c40513          	addi	a0,s0,-1940 # 1c01d86c <pos_alloc_l2>
1c01a0fe:	22cd                	jal	1c01a2e0 <pos_alloc_init>
1c01a100:	1c01e5b7          	lui	a1,0x1c01e
1c01a104:	c2058793          	addi	a5,a1,-992 # 1c01dc20 <__bss_end__>
1c01a108:	1c190937          	lui	s2,0x1c190
1c01a10c:	40f90933          	sub	s2,s2,a5
1c01a110:	1c01e4b7          	lui	s1,0x1c01e
1c01a114:	864a                	mv	a2,s2
1c01a116:	c2058593          	addi	a1,a1,-992
1c01a11a:	89048513          	addi	a0,s1,-1904 # 1c01d890 <pos_alloc_l2+0x24>
1c01a11e:	22c9                	jal	1c01a2e0 <pos_alloc_init>
1c01a120:	86c40693          	addi	a3,s0,-1940
1c01a124:	4785                	li	a5,1
1c01a126:	1c01d737          	lui	a4,0x1c01d
1c01a12a:	dadc                	sw	a5,52(a3)
1c01a12c:	71070793          	addi	a5,a4,1808 # 1c01d710 <pos_alloc_account_0>
1c01a130:	de9c                	sw	a5,56(a3)
1c01a132:	1c01d7b7          	lui	a5,0x1c01d
1c01a136:	74078613          	addi	a2,a5,1856 # 1c01d740 <pos_alloc_account_1>
1c01a13a:	ded0                	sw	a2,60(a3)
1c01a13c:	71070713          	addi	a4,a4,1808
1c01a140:	74078793          	addi	a5,a5,1856
1c01a144:	86c40413          	addi	s0,s0,-1940
1c01a148:	00c250fb          	lp.setupi	x1,12,1c01a150 <pos_allocs_init+0x78>
1c01a14c:	0007222b          	p.sw	zero,4(a4!)
1c01a150:	0007a22b          	p.sw	zero,4(a5!)
1c01a154:	47c5                	li	a5,17
1c01a156:	c03c                	sw	a5,64(s0)
1c01a158:	1c0107b7          	lui	a5,0x1c010
1c01a15c:	c07c                	sw	a5,68(s0)
1c01a15e:	4422                	lw	s0,8(sp)
1c01a160:	40b2                	lw	ra,12(sp)
1c01a162:	ff890613          	addi	a2,s2,-8 # 1c18fff8 <__bss_end__+0x1723d8>
1c01a166:	89048513          	addi	a0,s1,-1904
1c01a16a:	4902                	lw	s2,0(sp)
1c01a16c:	4492                	lw	s1,4(sp)
1c01a16e:	1c01e5b7          	lui	a1,0x1c01e
1c01a172:	c2858593          	addi	a1,a1,-984 # 1c01dc28 <__bss_end__+0x8>
1c01a176:	0141                	addi	sp,sp,16
1c01a178:	aaa9                	j	1c01a2d2 <pos_alloc_account_free>
1c01a17a:	8e1d                	sub	a2,a2,a5
1c01a17c:	dc058593          	addi	a1,a1,-576
1c01a180:	bf9d                	j	1c01a0f6 <pos_allocs_init+0x1e>

1c01a182 <pos_alloc_power_ctrl>:
1c01a182:	c59d                	beqz	a1,1c01a1b0 <pos_alloc_power_ctrl+0x2e>
1c01a184:	1a104737          	lui	a4,0x1a104
1c01a188:	15c72683          	lw	a3,348(a4) # 1a10415c <__l1_heapsram_end+0xa0e415c>
1c01a18c:	16072603          	lw	a2,352(a4)
1c01a190:	01051793          	slli	a5,a0,0x10
1c01a194:	8d5d                	or	a0,a0,a5
1c01a196:	fff54793          	not	a5,a0
1c01a19a:	8e7d                	and	a2,a2,a5
1c01a19c:	16c72023          	sw	a2,352(a4)
1c01a1a0:	8ff5                	and	a5,a5,a3
1c01a1a2:	14f72e23          	sw	a5,348(a4)
1c01a1a6:	18c72783          	lw	a5,396(a4)
1c01a1aa:	8fe9                	and	a5,a5,a0
1c01a1ac:	ffed                	bnez	a5,1c01a1a6 <pos_alloc_power_ctrl+0x24>
1c01a1ae:	8082                	ret
1c01a1b0:	1a1047b7          	lui	a5,0x1a104
1c01a1b4:	15c7a683          	lw	a3,348(a5) # 1a10415c <__l1_heapsram_end+0xa0e415c>
1c01a1b8:	1607a703          	lw	a4,352(a5)
1c01a1bc:	c609                	beqz	a2,1c01a1c6 <pos_alloc_power_ctrl+0x44>
1c01a1be:	8d55                	or	a0,a0,a3
1c01a1c0:	14a7ae23          	sw	a0,348(a5)
1c01a1c4:	8082                	ret
1c01a1c6:	01051793          	slli	a5,a0,0x10
1c01a1ca:	8edd                	or	a3,a3,a5
1c01a1cc:	1a104637          	lui	a2,0x1a104
1c01a1d0:	14d62e23          	sw	a3,348(a2) # 1a10415c <__l1_heapsram_end+0xa0e415c>
1c01a1d4:	8d5d                	or	a0,a0,a5
1c01a1d6:	8d59                	or	a0,a0,a4
1c01a1d8:	16a62023          	sw	a0,352(a2)
1c01a1dc:	8082                	ret

1c01a1de <pos_alloc_dump>:
1c01a1de:	1141                	addi	sp,sp,-16
1c01a1e0:	c422                	sw	s0,8(sp)
1c01a1e2:	842a                	mv	s0,a0
1c01a1e4:	1c01c537          	lui	a0,0x1c01c
1c01a1e8:	ec050513          	addi	a0,a0,-320 # 1c01bec0 <__func__.47299+0x100>
1c01a1ec:	c226                	sw	s1,4(sp)
1c01a1ee:	c04a                	sw	s2,0(sp)
1c01a1f0:	c606                	sw	ra,12(sp)
1c01a1f2:	48a010ef          	jal	ra,1c01b67c <printf_>
1c01a1f6:	4000                	lw	s0,0(s0)
1c01a1f8:	1c01c4b7          	lui	s1,0x1c01c
1c01a1fc:	1c01c937          	lui	s2,0x1c01c
1c01a200:	c00d                	beqz	s0,1c01a222 <pos_alloc_dump+0x44>
1c01a202:	4054                	lw	a3,4(s0)
1c01a204:	4010                	lw	a2,0(s0)
1c01a206:	85a2                	mv	a1,s0
1c01a208:	ef048513          	addi	a0,s1,-272 # 1c01bef0 <__func__.47299+0x130>
1c01a20c:	470010ef          	jal	ra,1c01b67c <printf_>
1c01a210:	405c                	lw	a5,4(s0)
1c01a212:	02879363          	bne	a5,s0,1c01a238 <pos_alloc_dump+0x5a>
1c01a216:	1c01c537          	lui	a0,0x1c01c
1c01a21a:	f1c50513          	addi	a0,a0,-228 # 1c01bf1c <__func__.47299+0x15c>
1c01a21e:	45e010ef          	jal	ra,1c01b67c <printf_>
1c01a222:	4422                	lw	s0,8(sp)
1c01a224:	40b2                	lw	ra,12(sp)
1c01a226:	4492                	lw	s1,4(sp)
1c01a228:	4902                	lw	s2,0(sp)
1c01a22a:	1c01c537          	lui	a0,0x1c01c
1c01a22e:	f2850513          	addi	a0,a0,-216 # 1c01bf28 <__func__.47299+0x168>
1c01a232:	0141                	addi	sp,sp,16
1c01a234:	4480106f          	j	1c01b67c <printf_>
1c01a238:	bfc90513          	addi	a0,s2,-1028 # 1c01bbfc <__clz_tab+0x1b4>
1c01a23c:	440010ef          	jal	ra,1c01b67c <printf_>
1c01a240:	4040                	lw	s0,4(s0)
1c01a242:	bf7d                	j	1c01a200 <pos_alloc_dump+0x22>

1c01a244 <pos_alloc_account>:
1c01a244:	01c52803          	lw	a6,28(a0)
1c01a248:	511c                	lw	a5,32(a0)
1c01a24a:	4885                	li	a7,1
1c01a24c:	010898b3          	sll	a7,a7,a6
1c01a250:	8d9d                	sub	a1,a1,a5
1c01a252:	411007b3          	neg	a5,a7
1c01a256:	8eaa                	mv	t4,a0
1c01a258:	00b7f533          	and	a0,a5,a1
1c01a25c:	40b887b3          	sub	a5,a7,a1
1c01a260:	01055833          	srl	a6,a0,a6
1c01a264:	97aa                	add	a5,a5,a0
1c01a266:	04c7d7b3          	p.minu	a5,a5,a2
1c01a26a:	00281e13          	slli	t3,a6,0x2
1c01a26e:	4501                	li	a0,0
1c01a270:	4f05                	li	t5,1
1c01a272:	ea01                	bnez	a2,1c01a282 <pos_alloc_account+0x3e>
1c01a274:	c539                	beqz	a0,1c01a2c2 <pos_alloc_account+0x7e>
1c01a276:	0685                	addi	a3,a3,1
1c01a278:	863a                	mv	a2,a4
1c01a27a:	0016b593          	seqz	a1,a3
1c01a27e:	f05ff06f          	j	1c01a182 <pos_alloc_power_ctrl>
1c01a282:	cf0d                	beqz	a4,1c01a2bc <pos_alloc_account+0x78>
1c01a284:	018ea303          	lw	t1,24(t4)
1c01a288:	9372                	add	t1,t1,t3
1c01a28a:	00032583          	lw	a1,0(t1)
1c01a28e:	01f6b863          	p.bneimm	a3,-1,1c01a29e <pos_alloc_account+0x5a>
1c01a292:	00b89663          	bne	a7,a1,1c01a29e <pos_alloc_account+0x5a>
1c01a296:	010f1fb3          	sll	t6,t5,a6
1c01a29a:	01f56533          	or	a0,a0,t6
1c01a29e:	42f685b3          	p.mac	a1,a3,a5
1c01a2a2:	00b32023          	sw	a1,0(t1)
1c01a2a6:	01159563          	bne	a1,a7,1c01a2b0 <pos_alloc_account+0x6c>
1c01a2aa:	010f15b3          	sll	a1,t5,a6
1c01a2ae:	8d4d                	or	a0,a0,a1
1c01a2b0:	8e1d                	sub	a2,a2,a5
1c01a2b2:	0805                	addi	a6,a6,1
1c01a2b4:	0e11                	addi	t3,t3,4
1c01a2b6:	04c8d7b3          	p.minu	a5,a7,a2
1c01a2ba:	bf65                	j	1c01a272 <pos_alloc_account+0x2e>
1c01a2bc:	014ea303          	lw	t1,20(t4)
1c01a2c0:	b7e1                	j	1c01a288 <pos_alloc_account+0x44>
1c01a2c2:	8082                	ret

1c01a2c4 <pos_alloc_account_alloc>:
1c01a2c4:	491c                	lw	a5,16(a0)
1c01a2c6:	c789                	beqz	a5,1c01a2d0 <pos_alloc_account_alloc+0xc>
1c01a2c8:	4701                	li	a4,0
1c01a2ca:	56fd                	li	a3,-1
1c01a2cc:	f79ff06f          	j	1c01a244 <pos_alloc_account>
1c01a2d0:	8082                	ret

1c01a2d2 <pos_alloc_account_free>:
1c01a2d2:	491c                	lw	a5,16(a0)
1c01a2d4:	c789                	beqz	a5,1c01a2de <pos_alloc_account_free+0xc>
1c01a2d6:	4701                	li	a4,0
1c01a2d8:	4685                	li	a3,1
1c01a2da:	f6bff06f          	j	1c01a244 <pos_alloc_account>
1c01a2de:	8082                	ret

1c01a2e0 <pos_alloc_init>:
1c01a2e0:	00758793          	addi	a5,a1,7
1c01a2e4:	c407b7b3          	p.bclr	a5,a5,2,0
1c01a2e8:	40b785b3          	sub	a1,a5,a1
1c01a2ec:	00052823          	sw	zero,16(a0)
1c01a2f0:	c11c                	sw	a5,0(a0)
1c01a2f2:	00052623          	sw	zero,12(a0)
1c01a2f6:	8e0d                	sub	a2,a2,a1
1c01a2f8:	00c05763          	blez	a2,1c01a306 <pos_alloc_init+0x26>
1c01a2fc:	c4063633          	p.bclr	a2,a2,2,0
1c01a300:	c390                	sw	a2,0(a5)
1c01a302:	0007a223          	sw	zero,4(a5)
1c01a306:	8082                	ret

1c01a308 <pos_alloc>:
1c01a308:	1101                	addi	sp,sp,-32
1c01a30a:	cc22                	sw	s0,24(sp)
1c01a30c:	058d                	addi	a1,a1,3
1c01a30e:	4100                	lw	s0,0(a0)
1c01a310:	ca26                	sw	s1,20(sp)
1c01a312:	47a1                	li	a5,8
1c01a314:	c205b4b3          	p.bclr	s1,a1,1,0
1c01a318:	c64e                	sw	s3,12(sp)
1c01a31a:	ce06                	sw	ra,28(sp)
1c01a31c:	c84a                	sw	s2,16(sp)
1c01a31e:	c452                	sw	s4,8(sp)
1c01a320:	04f4f4b3          	p.maxu	s1,s1,a5
1c01a324:	4981                	li	s3,0
1c01a326:	471d                	li	a4,7
1c01a328:	c41d                	beqz	s0,1c01a356 <pos_alloc+0x4e>
1c01a32a:	401c                	lw	a5,0(s0)
1c01a32c:	409786b3          	sub	a3,a5,s1
1c01a330:	00d74463          	blt	a4,a3,1c01a338 <pos_alloc+0x30>
1c01a334:	02979a63          	bne	a5,s1,1c01a368 <pos_alloc+0x60>
1c01a338:	892a                	mv	s2,a0
1c01a33a:	00840593          	addi	a1,s0,8
1c01a33e:	ff848613          	addi	a2,s1,-8
1c01a342:	02979963          	bne	a5,s1,1c01a374 <pos_alloc+0x6c>
1c01a346:	854a                	mv	a0,s2
1c01a348:	f7dff0ef          	jal	ra,1c01a2c4 <pos_alloc_account_alloc>
1c01a34c:	405c                	lw	a5,4(s0)
1c01a34e:	02098063          	beqz	s3,1c01a36e <pos_alloc+0x66>
1c01a352:	00f9a223          	sw	a5,4(s3)
1c01a356:	8522                	mv	a0,s0
1c01a358:	40f2                	lw	ra,28(sp)
1c01a35a:	4462                	lw	s0,24(sp)
1c01a35c:	44d2                	lw	s1,20(sp)
1c01a35e:	4942                	lw	s2,16(sp)
1c01a360:	49b2                	lw	s3,12(sp)
1c01a362:	4a22                	lw	s4,8(sp)
1c01a364:	6105                	addi	sp,sp,32
1c01a366:	8082                	ret
1c01a368:	89a2                	mv	s3,s0
1c01a36a:	4040                	lw	s0,4(s0)
1c01a36c:	bf75                	j	1c01a328 <pos_alloc+0x20>
1c01a36e:	00f92023          	sw	a5,0(s2)
1c01a372:	b7d5                	j	1c01a356 <pos_alloc+0x4e>
1c01a374:	854a                	mv	a0,s2
1c01a376:	f4fff0ef          	jal	ra,1c01a2c4 <pos_alloc_account_alloc>
1c01a37a:	00940a33          	add	s4,s0,s1
1c01a37e:	85d2                	mv	a1,s4
1c01a380:	4621                	li	a2,8
1c01a382:	854a                	mv	a0,s2
1c01a384:	f41ff0ef          	jal	ra,1c01a2c4 <pos_alloc_account_alloc>
1c01a388:	400c                	lw	a1,0(s0)
1c01a38a:	405c                	lw	a5,4(s0)
1c01a38c:	409584b3          	sub	s1,a1,s1
1c01a390:	009a2023          	sw	s1,0(s4)
1c01a394:	00fa2223          	sw	a5,4(s4)
1c01a398:	00098563          	beqz	s3,1c01a3a2 <pos_alloc+0x9a>
1c01a39c:	0149a223          	sw	s4,4(s3)
1c01a3a0:	bf5d                	j	1c01a356 <pos_alloc+0x4e>
1c01a3a2:	01492023          	sw	s4,0(s2)
1c01a3a6:	bf45                	j	1c01a356 <pos_alloc+0x4e>

1c01a3a8 <pos_free>:
1c01a3a8:	c9d9                	beqz	a1,1c01a43e <pos_free+0x96>
1c01a3aa:	1101                	addi	sp,sp,-32
1c01a3ac:	cc22                	sw	s0,24(sp)
1c01a3ae:	ca26                	sw	s1,20(sp)
1c01a3b0:	842e                	mv	s0,a1
1c01a3b2:	00360493          	addi	s1,a2,3
1c01a3b6:	410c                	lw	a1,0(a0)
1c01a3b8:	c204b4b3          	p.bclr	s1,s1,1,0
1c01a3bc:	47a1                	li	a5,8
1c01a3be:	c64e                	sw	s3,12(sp)
1c01a3c0:	ce06                	sw	ra,28(sp)
1c01a3c2:	c84a                	sw	s2,16(sp)
1c01a3c4:	04f4f4b3          	p.maxu	s1,s1,a5
1c01a3c8:	4981                	li	s3,0
1c01a3ca:	c199                	beqz	a1,1c01a3d0 <pos_free+0x28>
1c01a3cc:	0485e963          	bltu	a1,s0,1c01a41e <pos_free+0x76>
1c01a3d0:	009407b3          	add	a5,s0,s1
1c01a3d4:	892a                	mv	s2,a0
1c01a3d6:	04f59763          	bne	a1,a5,1c01a424 <pos_free+0x7c>
1c01a3da:	419c                	lw	a5,0(a1)
1c01a3dc:	4621                	li	a2,8
1c01a3de:	97a6                	add	a5,a5,s1
1c01a3e0:	c01c                	sw	a5,0(s0)
1c01a3e2:	41dc                	lw	a5,4(a1)
1c01a3e4:	c05c                	sw	a5,4(s0)
1c01a3e6:	eedff0ef          	jal	ra,1c01a2d2 <pos_alloc_account_free>
1c01a3ea:	04098763          	beqz	s3,1c01a438 <pos_free+0x90>
1c01a3ee:	0009a703          	lw	a4,0(s3)
1c01a3f2:	00e987b3          	add	a5,s3,a4
1c01a3f6:	02f41a63          	bne	s0,a5,1c01a42a <pos_free+0x82>
1c01a3fa:	401c                	lw	a5,0(s0)
1c01a3fc:	8626                	mv	a2,s1
1c01a3fe:	85a2                	mv	a1,s0
1c01a400:	97ba                	add	a5,a5,a4
1c01a402:	00f9a023          	sw	a5,0(s3)
1c01a406:	405c                	lw	a5,4(s0)
1c01a408:	00f9a223          	sw	a5,4(s3)
1c01a40c:	4462                	lw	s0,24(sp)
1c01a40e:	40f2                	lw	ra,28(sp)
1c01a410:	44d2                	lw	s1,20(sp)
1c01a412:	49b2                	lw	s3,12(sp)
1c01a414:	854a                	mv	a0,s2
1c01a416:	4942                	lw	s2,16(sp)
1c01a418:	6105                	addi	sp,sp,32
1c01a41a:	eb9ff06f          	j	1c01a2d2 <pos_alloc_account_free>
1c01a41e:	89ae                	mv	s3,a1
1c01a420:	41cc                	lw	a1,4(a1)
1c01a422:	b765                	j	1c01a3ca <pos_free+0x22>
1c01a424:	c004                	sw	s1,0(s0)
1c01a426:	c04c                	sw	a1,4(s0)
1c01a428:	b7c9                	j	1c01a3ea <pos_free+0x42>
1c01a42a:	0089a223          	sw	s0,4(s3)
1c01a42e:	ff848613          	addi	a2,s1,-8
1c01a432:	00840593          	addi	a1,s0,8
1c01a436:	bfd9                	j	1c01a40c <pos_free+0x64>
1c01a438:	00892023          	sw	s0,0(s2)
1c01a43c:	bfcd                	j	1c01a42e <pos_free+0x86>
1c01a43e:	8082                	ret

1c01a440 <pi_malloc_init>:
1c01a440:	c99ff06f          	j	1c01a0d8 <pos_allocs_init>

1c01a444 <pi_l2_free>:
1c01a444:	1141                	addi	sp,sp,-16
1c01a446:	c422                	sw	s0,8(sp)
1c01a448:	c606                	sw	ra,12(sp)
1c01a44a:	c226                	sw	s1,4(sp)
1c01a44c:	842e                	mv	s0,a1
1c01a44e:	300474f3          	csrrci	s1,mstatus,8
1c01a452:	862e                	mv	a2,a1
1c01a454:	85aa                	mv	a1,a0
1c01a456:	1c01e537          	lui	a0,0x1c01e
1c01a45a:	89050513          	addi	a0,a0,-1904 # 1c01d890 <pos_alloc_l2+0x24>
1c01a45e:	f4bff0ef          	jal	ra,1c01a3a8 <pos_free>
1c01a462:	1c01e7b7          	lui	a5,0x1c01e
1c01a466:	86c78793          	addi	a5,a5,-1940 # 1c01d86c <pos_alloc_l2>
1c01a46a:	5b98                	lw	a4,48(a5)
1c01a46c:	fc173733          	p.bclr	a4,a4,30,1
1c01a470:	c719                	beqz	a4,1c01a47e <pi_l2_free+0x3a>
1c01a472:	57d8                	lw	a4,44(a5)
1c01a474:	041d                	addi	s0,s0,7
1c01a476:	c4043433          	p.bclr	s0,s0,2,0
1c01a47a:	943a                	add	s0,s0,a4
1c01a47c:	d7c0                	sw	s0,44(a5)
1c01a47e:	30049073          	csrw	mstatus,s1
1c01a482:	40b2                	lw	ra,12(sp)
1c01a484:	4422                	lw	s0,8(sp)
1c01a486:	4492                	lw	s1,4(sp)
1c01a488:	0141                	addi	sp,sp,16
1c01a48a:	8082                	ret

1c01a48c <pi_free>:
1c01a48c:	c509                	beqz	a0,1c01a496 <pi_free+0xa>
1c01a48e:	1571                	addi	a0,a0,-4
1c01a490:	410c                	lw	a1,0(a0)
1c01a492:	fb3ff06f          	j	1c01a444 <pi_l2_free>
1c01a496:	8082                	ret

1c01a498 <pi_l2_priv_free>:
1c01a498:	1141                	addi	sp,sp,-16
1c01a49a:	c606                	sw	ra,12(sp)
1c01a49c:	c422                	sw	s0,8(sp)
1c01a49e:	862e                	mv	a2,a1
1c01a4a0:	30047473          	csrrci	s0,mstatus,8
1c01a4a4:	85aa                	mv	a1,a0
1c01a4a6:	1c01e537          	lui	a0,0x1c01e
1c01a4aa:	86c50513          	addi	a0,a0,-1940 # 1c01d86c <pos_alloc_l2>
1c01a4ae:	efbff0ef          	jal	ra,1c01a3a8 <pos_free>
1c01a4b2:	30041073          	csrw	mstatus,s0
1c01a4b6:	40b2                	lw	ra,12(sp)
1c01a4b8:	4422                	lw	s0,8(sp)
1c01a4ba:	0141                	addi	sp,sp,16
1c01a4bc:	8082                	ret

1c01a4be <pi_l2_malloc_dump>:
1c01a4be:	1c01e537          	lui	a0,0x1c01e
1c01a4c2:	89050513          	addi	a0,a0,-1904 # 1c01d890 <pos_alloc_l2+0x24>
1c01a4c6:	d19ff06f          	j	1c01a1de <pos_alloc_dump>

1c01a4ca <pi_l2_priv_malloc_dump>:
1c01a4ca:	1c01e537          	lui	a0,0x1c01e
1c01a4ce:	86c50513          	addi	a0,a0,-1940 # 1c01d86c <pos_alloc_l2>
1c01a4d2:	d0dff06f          	j	1c01a1de <pos_alloc_dump>

1c01a4d6 <pi_alloc_fail>:
1c01a4d6:	1141                	addi	sp,sp,-16
1c01a4d8:	c422                	sw	s0,8(sp)
1c01a4da:	c226                	sw	s1,4(sp)
1c01a4dc:	842a                	mv	s0,a0
1c01a4de:	84ae                	mv	s1,a1
1c01a4e0:	85aa                	mv	a1,a0
1c01a4e2:	1c01c537          	lui	a0,0x1c01c
1c01a4e6:	f5850513          	addi	a0,a0,-168 # 1c01bf58 <__func__.47299+0x198>
1c01a4ea:	c606                	sw	ra,12(sp)
1c01a4ec:	190010ef          	jal	ra,1c01b67c <printf_>
1c01a4f0:	c0a1                	beqz	s1,1c01a530 <pi_alloc_fail+0x5a>
1c01a4f2:	02142563          	p.beqimm	s0,1,1c01a51c <pi_alloc_fail+0x46>
1c01a4f6:	c819                	beqz	s0,1c01a50c <pi_alloc_fail+0x36>
1c01a4f8:	02242563          	p.beqimm	s0,2,1c01a522 <pi_alloc_fail+0x4c>
1c01a4fc:	1c01c537          	lui	a0,0x1c01c
1c01a500:	85a2                	mv	a1,s0
1c01a502:	f9850513          	addi	a0,a0,-104 # 1c01bf98 <__func__.47299+0x1d8>
1c01a506:	176010ef          	jal	ra,1c01b67c <printf_>
1c01a50a:	a019                	j	1c01a510 <pi_alloc_fail+0x3a>
1c01a50c:	fbfff0ef          	jal	ra,1c01a4ca <pi_l2_priv_malloc_dump>
1c01a510:	40b2                	lw	ra,12(sp)
1c01a512:	4422                	lw	s0,8(sp)
1c01a514:	4492                	lw	s1,4(sp)
1c01a516:	557d                	li	a0,-1
1c01a518:	0141                	addi	sp,sp,16
1c01a51a:	8082                	ret
1c01a51c:	fa3ff0ef          	jal	ra,1c01a4be <pi_l2_malloc_dump>
1c01a520:	bfc5                	j	1c01a510 <pi_alloc_fail+0x3a>
1c01a522:	1c01d537          	lui	a0,0x1c01d
1c01a526:	7d850513          	addi	a0,a0,2008 # 1c01d7d8 <pos_alloc_l1>
1c01a52a:	cb5ff0ef          	jal	ra,1c01a1de <pos_alloc_dump>
1c01a52e:	b7cd                	j	1c01a510 <pi_alloc_fail+0x3a>
1c01a530:	1c01c537          	lui	a0,0x1c01c
1c01a534:	fb450513          	addi	a0,a0,-76 # 1c01bfb4 <__func__.47299+0x1f4>
1c01a538:	144010ef          	jal	ra,1c01b67c <printf_>
1c01a53c:	bfd1                	j	1c01a510 <pi_alloc_fail+0x3a>

1c01a53e <pi_l2_malloc>:
1c01a53e:	1101                	addi	sp,sp,-32
1c01a540:	cc22                	sw	s0,24(sp)
1c01a542:	ce06                	sw	ra,28(sp)
1c01a544:	ca26                	sw	s1,20(sp)
1c01a546:	c84a                	sw	s2,16(sp)
1c01a548:	c64e                	sw	s3,12(sp)
1c01a54a:	842a                	mv	s0,a0
1c01a54c:	300479f3          	csrrci	s3,mstatus,8
1c01a550:	1c01e937          	lui	s2,0x1c01e
1c01a554:	85aa                	mv	a1,a0
1c01a556:	89090513          	addi	a0,s2,-1904 # 1c01d890 <pos_alloc_l2+0x24>
1c01a55a:	dafff0ef          	jal	ra,1c01a308 <pos_alloc>
1c01a55e:	84aa                	mv	s1,a0
1c01a560:	ed01                	bnez	a0,1c01a578 <pi_l2_malloc+0x3a>
1c01a562:	85a2                	mv	a1,s0
1c01a564:	4505                	li	a0,1
1c01a566:	f71ff0ef          	jal	ra,1c01a4d6 <pi_alloc_fail>
1c01a56a:	e519                	bnez	a0,1c01a578 <pi_l2_malloc+0x3a>
1c01a56c:	85a2                	mv	a1,s0
1c01a56e:	89090513          	addi	a0,s2,-1904
1c01a572:	d97ff0ef          	jal	ra,1c01a308 <pos_alloc>
1c01a576:	84aa                	mv	s1,a0
1c01a578:	1c01e7b7          	lui	a5,0x1c01e
1c01a57c:	86c78793          	addi	a5,a5,-1940 # 1c01d86c <pos_alloc_l2>
1c01a580:	5b98                	lw	a4,48(a5)
1c01a582:	fc173733          	p.bclr	a4,a4,30,1
1c01a586:	cf09                	beqz	a4,1c01a5a0 <pi_l2_malloc+0x62>
1c01a588:	cc81                	beqz	s1,1c01a5a0 <pi_l2_malloc+0x62>
1c01a58a:	57d8                	lw	a4,44(a5)
1c01a58c:	041d                	addi	s0,s0,7
1c01a58e:	c4043433          	p.bclr	s0,s0,2,0
1c01a592:	40870433          	sub	s0,a4,s0
1c01a596:	5798                	lw	a4,40(a5)
1c01a598:	d7c0                	sw	s0,44(a5)
1c01a59a:	00e47363          	bleu	a4,s0,1c01a5a0 <pi_l2_malloc+0x62>
1c01a59e:	d780                	sw	s0,40(a5)
1c01a5a0:	30099073          	csrw	mstatus,s3
1c01a5a4:	40f2                	lw	ra,28(sp)
1c01a5a6:	4462                	lw	s0,24(sp)
1c01a5a8:	8526                	mv	a0,s1
1c01a5aa:	4942                	lw	s2,16(sp)
1c01a5ac:	44d2                	lw	s1,20(sp)
1c01a5ae:	49b2                	lw	s3,12(sp)
1c01a5b0:	6105                	addi	sp,sp,32
1c01a5b2:	8082                	ret

1c01a5b4 <pi_malloc>:
1c01a5b4:	1141                	addi	sp,sp,-16
1c01a5b6:	c422                	sw	s0,8(sp)
1c01a5b8:	00450413          	addi	s0,a0,4
1c01a5bc:	8522                	mv	a0,s0
1c01a5be:	c606                	sw	ra,12(sp)
1c01a5c0:	f7fff0ef          	jal	ra,1c01a53e <pi_l2_malloc>
1c01a5c4:	c119                	beqz	a0,1c01a5ca <pi_malloc+0x16>
1c01a5c6:	0085222b          	p.sw	s0,4(a0!)
1c01a5ca:	40b2                	lw	ra,12(sp)
1c01a5cc:	4422                	lw	s0,8(sp)
1c01a5ce:	0141                	addi	sp,sp,16
1c01a5d0:	8082                	ret

1c01a5d2 <pi_l2_priv_malloc>:
1c01a5d2:	1101                	addi	sp,sp,-32
1c01a5d4:	ca26                	sw	s1,20(sp)
1c01a5d6:	ce06                	sw	ra,28(sp)
1c01a5d8:	cc22                	sw	s0,24(sp)
1c01a5da:	c84a                	sw	s2,16(sp)
1c01a5dc:	c64e                	sw	s3,12(sp)
1c01a5de:	84aa                	mv	s1,a0
1c01a5e0:	300479f3          	csrrci	s3,mstatus,8
1c01a5e4:	1c01e937          	lui	s2,0x1c01e
1c01a5e8:	85aa                	mv	a1,a0
1c01a5ea:	86c90513          	addi	a0,s2,-1940 # 1c01d86c <pos_alloc_l2>
1c01a5ee:	d1bff0ef          	jal	ra,1c01a308 <pos_alloc>
1c01a5f2:	842a                	mv	s0,a0
1c01a5f4:	e919                	bnez	a0,1c01a60a <pi_l2_priv_malloc+0x38>
1c01a5f6:	85a6                	mv	a1,s1
1c01a5f8:	edfff0ef          	jal	ra,1c01a4d6 <pi_alloc_fail>
1c01a5fc:	e519                	bnez	a0,1c01a60a <pi_l2_priv_malloc+0x38>
1c01a5fe:	85a6                	mv	a1,s1
1c01a600:	86c90513          	addi	a0,s2,-1940
1c01a604:	d05ff0ef          	jal	ra,1c01a308 <pos_alloc>
1c01a608:	842a                	mv	s0,a0
1c01a60a:	30099073          	csrw	mstatus,s3
1c01a60e:	8522                	mv	a0,s0
1c01a610:	40f2                	lw	ra,28(sp)
1c01a612:	4462                	lw	s0,24(sp)
1c01a614:	44d2                	lw	s1,20(sp)
1c01a616:	4942                	lw	s2,16(sp)
1c01a618:	49b2                	lw	s3,12(sp)
1c01a61a:	6105                	addi	sp,sp,32
1c01a61c:	8082                	ret

1c01a61e <_out_null>:
1c01a61e:	8082                	ret

1c01a620 <_out_rev>:
1c01a620:	7179                	addi	sp,sp,-48
1c01a622:	d422                	sw	s0,40(sp)
1c01a624:	d226                	sw	s1,36(sp)
1c01a626:	d04a                	sw	s2,32(sp)
1c01a628:	ce4e                	sw	s3,28(sp)
1c01a62a:	cc52                	sw	s4,24(sp)
1c01a62c:	ca56                	sw	s5,20(sp)
1c01a62e:	c85a                	sw	s6,16(sp)
1c01a630:	c65e                	sw	s7,12(sp)
1c01a632:	c462                	sw	s8,8(sp)
1c01a634:	843e                	mv	s0,a5
1c01a636:	d606                	sw	ra,44(sp)
1c01a638:	c266                	sw	s9,4(sp)
1c01a63a:	c06a                	sw	s10,0(sp)
1c01a63c:	fa28b7b3          	p.bclr	a5,a7,29,2
1c01a640:	8aaa                	mv	s5,a0
1c01a642:	8b2e                	mv	s6,a1
1c01a644:	89b2                	mv	s3,a2
1c01a646:	8bb6                	mv	s7,a3
1c01a648:	84ba                	mv	s1,a4
1c01a64a:	8c42                	mv	s8,a6
1c01a64c:	8a46                	mv	s4,a7
1c01a64e:	8932                	mv	s2,a2
1c01a650:	cfb1                	beqz	a5,1c01a6ac <_out_rev+0x8c>
1c01a652:	fff48c93          	addi	s9,s1,-1
1c01a656:	94a2                	add	s1,s1,s0
1c01a658:	00990d33          	add	s10,s2,s1
1c01a65c:	409d0633          	sub	a2,s10,s1
1c01a660:	14fd                	addi	s1,s1,-1
1c01a662:	05949863          	bne	s1,s9,1c01a6b2 <_out_rev+0x92>
1c01a666:	002a7a13          	andi	s4,s4,2
1c01a66a:	01240633          	add	a2,s0,s2
1c01a66e:	060a1063          	bnez	s4,1c01a6ce <_out_rev+0xae>
1c01a672:	50b2                	lw	ra,44(sp)
1c01a674:	5422                	lw	s0,40(sp)
1c01a676:	5492                	lw	s1,36(sp)
1c01a678:	5902                	lw	s2,32(sp)
1c01a67a:	49f2                	lw	s3,28(sp)
1c01a67c:	4a62                	lw	s4,24(sp)
1c01a67e:	4ad2                	lw	s5,20(sp)
1c01a680:	4b42                	lw	s6,16(sp)
1c01a682:	4bb2                	lw	s7,12(sp)
1c01a684:	4c22                	lw	s8,8(sp)
1c01a686:	4c92                	lw	s9,4(sp)
1c01a688:	4d02                	lw	s10,0(sp)
1c01a68a:	8532                	mv	a0,a2
1c01a68c:	6145                	addi	sp,sp,48
1c01a68e:	8082                	ret
1c01a690:	864a                	mv	a2,s2
1c01a692:	00190d13          	addi	s10,s2,1
1c01a696:	86de                	mv	a3,s7
1c01a698:	85da                	mv	a1,s6
1c01a69a:	02000513          	li	a0,32
1c01a69e:	9a82                	jalr	s5
1c01a6a0:	896a                	mv	s2,s10
1c01a6a2:	012c87b3          	add	a5,s9,s2
1c01a6a6:	ff87e5e3          	bltu	a5,s8,1c01a690 <_out_rev+0x70>
1c01a6aa:	b765                	j	1c01a652 <_out_rev+0x32>
1c01a6ac:	40c40cb3          	sub	s9,s0,a2
1c01a6b0:	bfcd                	j	1c01a6a2 <_out_rev+0x82>
1c01a6b2:	0004c503          	lbu	a0,0(s1)
1c01a6b6:	86de                	mv	a3,s7
1c01a6b8:	85da                	mv	a1,s6
1c01a6ba:	9a82                	jalr	s5
1c01a6bc:	b745                	j	1c01a65c <_out_rev+0x3c>
1c01a6be:	86de                	mv	a3,s7
1c01a6c0:	85da                	mv	a1,s6
1c01a6c2:	02000513          	li	a0,32
1c01a6c6:	00160413          	addi	s0,a2,1
1c01a6ca:	9a82                	jalr	s5
1c01a6cc:	8622                	mv	a2,s0
1c01a6ce:	413607b3          	sub	a5,a2,s3
1c01a6d2:	ff87e6e3          	bltu	a5,s8,1c01a6be <_out_rev+0x9e>
1c01a6d6:	bf71                	j	1c01a672 <_out_rev+0x52>

1c01a6d8 <_ntoa_format>:
1c01a6d8:	8346                	mv	t1,a7
1c01a6da:	48a2                	lw	a7,8(sp)
1c01a6dc:	8e42                	mv	t3,a6
1c01a6de:	4e82                	lw	t4,0(sp)
1c01a6e0:	0028ff13          	andi	t5,a7,2
1c01a6e4:	4812                	lw	a6,4(sp)
1c01a6e6:	040f1263          	bnez	t5,1c01a72a <_ntoa_format+0x52>
1c01a6ea:	fc18bfb3          	p.bclr	t6,a7,30,1
1c01a6ee:	00080b63          	beqz	a6,1c01a704 <_ntoa_format+0x2c>
1c01a6f2:	000f8963          	beqz	t6,1c01a704 <_ntoa_format+0x2c>
1c01a6f6:	000e1663          	bnez	t3,1c01a702 <_ntoa_format+0x2a>
1c01a6fa:	00c8ff13          	andi	t5,a7,12
1c01a6fe:	000f0363          	beqz	t5,1c01a704 <_ntoa_format+0x2c>
1c01a702:	187d                	addi	a6,a6,-1
1c01a704:	00f70f33          	add	t5,a4,a5
1c01a708:	02000293          	li	t0,32
1c01a70c:	03000393          	li	t2,48
1c01a710:	05d7eb63          	bltu	a5,t4,1c01a766 <_ntoa_format+0x8e>
1c01a714:	00f70f33          	add	t5,a4,a5
1c01a718:	42fd                	li	t0,31
1c01a71a:	03000393          	li	t2,48
1c01a71e:	000f8663          	beqz	t6,1c01a72a <_ntoa_format+0x52>
1c01a722:	0107f463          	bleu	a6,a5,1c01a72a <_ntoa_format+0x52>
1c01a726:	04f2f663          	bleu	a5,t0,1c01a772 <_ntoa_format+0x9a>
1c01a72a:	0108ff13          	andi	t5,a7,16
1c01a72e:	060f0e63          	beqz	t5,1c01a7aa <_ntoa_format+0xd2>
1c01a732:	4008ff13          	andi	t5,a7,1024
1c01a736:	040f1363          	bnez	t5,1c01a77c <_ntoa_format+0xa4>
1c01a73a:	c3a9                	beqz	a5,1c01a77c <_ntoa_format+0xa4>
1c01a73c:	01d78463          	beq	a5,t4,1c01a744 <_ntoa_format+0x6c>
1c01a740:	03079e63          	bne	a5,a6,1c01a77c <_ntoa_format+0xa4>
1c01a744:	fff78e93          	addi	t4,a5,-1
1c01a748:	020e8963          	beqz	t4,1c01a77a <_ntoa_format+0xa2>
1c01a74c:	4f41                	li	t5,16
1c01a74e:	17f9                	addi	a5,a5,-2
1c01a750:	03e30963          	beq	t1,t5,1c01a782 <_ntoa_format+0xaa>
1c01a754:	87f6                	mv	a5,t4
1c01a756:	04233263          	p.bneimm	t1,2,1c01a79a <_ntoa_format+0xc2>
1c01a75a:	437d                	li	t1,31
1c01a75c:	06f36763          	bltu	t1,a5,1c01a7ca <_ntoa_format+0xf2>
1c01a760:	06200313          	li	t1,98
1c01a764:	a805                	j	1c01a794 <_ntoa_format+0xbc>
1c01a766:	fa5787e3          	beq	a5,t0,1c01a714 <_ntoa_format+0x3c>
1c01a76a:	0785                	addi	a5,a5,1
1c01a76c:	007f00ab          	p.sb	t2,1(t5!)
1c01a770:	b745                	j	1c01a710 <_ntoa_format+0x38>
1c01a772:	0785                	addi	a5,a5,1
1c01a774:	007f00ab          	p.sb	t2,1(t5!)
1c01a778:	b75d                	j	1c01a71e <_ntoa_format+0x46>
1c01a77a:	4781                	li	a5,0
1c01a77c:	4ec1                	li	t4,16
1c01a77e:	fdd31ce3          	bne	t1,t4,1c01a756 <_ntoa_format+0x7e>
1c01a782:	0208f313          	andi	t1,a7,32
1c01a786:	04031463          	bnez	t1,1c01a7ce <_ntoa_format+0xf6>
1c01a78a:	437d                	li	t1,31
1c01a78c:	02f36f63          	bltu	t1,a5,1c01a7ca <_ntoa_format+0xf2>
1c01a790:	07800313          	li	t1,120
1c01a794:	006747a3          	p.sb	t1,a5(a4)
1c01a798:	0785                	addi	a5,a5,1
1c01a79a:	437d                	li	t1,31
1c01a79c:	02f36763          	bltu	t1,a5,1c01a7ca <_ntoa_format+0xf2>
1c01a7a0:	03000313          	li	t1,48
1c01a7a4:	006747a3          	p.sb	t1,a5(a4)
1c01a7a8:	0785                	addi	a5,a5,1
1c01a7aa:	437d                	li	t1,31
1c01a7ac:	00f36f63          	bltu	t1,a5,1c01a7ca <_ntoa_format+0xf2>
1c01a7b0:	02d00313          	li	t1,45
1c01a7b4:	000e1863          	bnez	t3,1c01a7c4 <_ntoa_format+0xec>
1c01a7b8:	0048f313          	andi	t1,a7,4
1c01a7bc:	00030f63          	beqz	t1,1c01a7da <_ntoa_format+0x102>
1c01a7c0:	02b00313          	li	t1,43
1c01a7c4:	006747a3          	p.sb	t1,a5(a4)
1c01a7c8:	0785                	addi	a5,a5,1
1c01a7ca:	e57ff06f          	j	1c01a620 <_out_rev>
1c01a7ce:	437d                	li	t1,31
1c01a7d0:	fef36de3          	bltu	t1,a5,1c01a7ca <_ntoa_format+0xf2>
1c01a7d4:	05800313          	li	t1,88
1c01a7d8:	bf75                	j	1c01a794 <_ntoa_format+0xbc>
1c01a7da:	0088f313          	andi	t1,a7,8
1c01a7de:	fe0306e3          	beqz	t1,1c01a7ca <_ntoa_format+0xf2>
1c01a7e2:	02000313          	li	t1,32
1c01a7e6:	bff9                	j	1c01a7c4 <_ntoa_format+0xec>

1c01a7e8 <_ntoa_long>:
1c01a7e8:	7139                	addi	sp,sp,-64
1c01a7ea:	de06                	sw	ra,60(sp)
1c01a7ec:	83be                	mv	t2,a5
1c01a7ee:	4e96                	lw	t4,68(sp)
1c01a7f0:	e319                	bnez	a4,1c01a7f6 <_ntoa_long+0xe>
1c01a7f2:	c04ebeb3          	p.bclr	t4,t4,0,4
1c01a7f6:	400ef313          	andi	t1,t4,1024
1c01a7fa:	00030463          	beqz	t1,1c01a802 <_ntoa_long+0x1a>
1c01a7fe:	4781                	li	a5,0
1c01a800:	c331                	beqz	a4,1c01a844 <_ntoa_long+0x5c>
1c01a802:	020efe13          	andi	t3,t4,32
1c01a806:	06100313          	li	t1,97
1c01a80a:	000e0463          	beqz	t3,1c01a812 <_ntoa_long+0x2a>
1c01a80e:	04100313          	li	t1,65
1c01a812:	01010f93          	addi	t6,sp,16
1c01a816:	4781                	li	a5,0
1c01a818:	40a5                	li	ra,9
1c01a81a:	ff630e13          	addi	t3,t1,-10
1c01a81e:	020950fb          	lp.setupi	x1,32,1c01a842 <_ntoa_long+0x5a>
1c01a822:	030772b3          	remu	t0,a4,a6
1c01a826:	0ff2f313          	andi	t1,t0,255
1c01a82a:	0250e963          	bltu	ra,t0,1c01a85c <_ntoa_long+0x74>
1c01a82e:	03030313          	addi	t1,t1,48
1c01a832:	03075733          	divu	a4,a4,a6
1c01a836:	0ff37313          	andi	t1,t1,255
1c01a83a:	006f80ab          	p.sb	t1,1(t6!)
1c01a83e:	0785                	addi	a5,a5,1
1c01a840:	c311                	beqz	a4,1c01a844 <_ntoa_long+0x5c>
1c01a842:	0001                	nop
1c01a844:	4706                	lw	a4,64(sp)
1c01a846:	c046                	sw	a7,0(sp)
1c01a848:	c476                	sw	t4,8(sp)
1c01a84a:	c23a                	sw	a4,4(sp)
1c01a84c:	88c2                	mv	a7,a6
1c01a84e:	0818                	addi	a4,sp,16
1c01a850:	881e                	mv	a6,t2
1c01a852:	e87ff0ef          	jal	ra,1c01a6d8 <_ntoa_format>
1c01a856:	50f2                	lw	ra,60(sp)
1c01a858:	6121                	addi	sp,sp,64
1c01a85a:	8082                	ret
1c01a85c:	9372                	add	t1,t1,t3
1c01a85e:	bfd1                	j	1c01a832 <_ntoa_long+0x4a>

1c01a860 <_ntoa_long_long>:
1c01a860:	7119                	addi	sp,sp,-128
1c01a862:	d2a6                	sw	s1,100(sp)
1c01a864:	d0ca                	sw	s2,96(sp)
1c01a866:	cece                	sw	s3,92(sp)
1c01a868:	ccd2                	sw	s4,88(sp)
1c01a86a:	cad6                	sw	s5,84(sp)
1c01a86c:	c2e6                	sw	s9,68(sp)
1c01a86e:	8ab6                	mv	s5,a3
1c01a870:	d686                	sw	ra,108(sp)
1c01a872:	d4a2                	sw	s0,104(sp)
1c01a874:	c8da                	sw	s6,80(sp)
1c01a876:	c6de                	sw	s7,76(sp)
1c01a878:	c4e2                	sw	s8,72(sp)
1c01a87a:	c0ea                	sw	s10,64(sp)
1c01a87c:	dec6                	sw	a7,124(sp)
1c01a87e:	30e036b3          	sltu.d	a3,zero,a4
1c01a882:	892a                	mv	s2,a0
1c01a884:	89ae                	mv	s3,a1
1c01a886:	8a32                	mv	s4,a2
1c01a888:	448a                	lw	s1,128(sp)
1c01a88a:	4cba                	lw	s9,140(sp)
1c01a88c:	e299                	bnez	a3,1c01a892 <_ntoa_long_long+0x32>
1c01a88e:	c04cbcb3          	p.bclr	s9,s9,0,4
1c01a892:	400cf693          	andi	a3,s9,1024
1c01a896:	c689                	beqz	a3,1c01a8a0 <_ntoa_long_long+0x40>
1c01a898:	3017369b          	sltiu.d	a3,a4,1
1c01a89c:	4c01                	li	s8,0
1c01a89e:	eaad                	bnez	a3,1c01a910 <_ntoa_long_long+0xb0>
1c01a8a0:	020cf693          	andi	a3,s9,32
1c01a8a4:	06100413          	li	s0,97
1c01a8a8:	c299                	beqz	a3,1c01a8ae <_ntoa_long_long+0x4e>
1c01a8aa:	04100413          	li	s0,65
1c01a8ae:	02010d13          	addi	s10,sp,32
1c01a8b2:	4c01                	li	s8,0
1c01a8b4:	4ba5                	li	s7,9
1c01a8b6:	1459                	addi	s0,s0,-10
1c01a8b8:	02000b13          	li	s6,32
1c01a8bc:	22071513          	addi.d	a0,a4,0
1c01a8c0:	8646                	mv	a2,a7
1c01a8c2:	86a6                	mv	a3,s1
1c01a8c4:	ce3a                	sw	a4,28(sp)
1c01a8c6:	cc3e                	sw	a5,24(sp)
1c01a8c8:	ca42                	sw	a6,20(sp)
1c01a8ca:	c846                	sw	a7,16(sp)
1c01a8cc:	fddf60ef          	jal	ra,1c0118a8 <__umoddi3>
1c01a8d0:	0ff57513          	andi	a0,a0,255
1c01a8d4:	4772                	lw	a4,28(sp)
1c01a8d6:	47e2                	lw	a5,24(sp)
1c01a8d8:	4852                	lw	a6,20(sp)
1c01a8da:	48c2                	lw	a7,16(sp)
1c01a8dc:	06abe563          	bltu	s7,a0,1c01a946 <_ntoa_long_long+0xe6>
1c01a8e0:	03050513          	addi	a0,a0,48
1c01a8e4:	0ff57513          	andi	a0,a0,255
1c01a8e8:	00ad00ab          	p.sb	a0,1(s10!)
1c01a8ec:	8646                	mv	a2,a7
1c01a8ee:	86a6                	mv	a3,s1
1c01a8f0:	22071513          	addi.d	a0,a4,0
1c01a8f4:	ca42                	sw	a6,20(sp)
1c01a8f6:	c846                	sw	a7,16(sp)
1c01a8f8:	c75f60ef          	jal	ra,1c01156c <__udivdi3>
1c01a8fc:	3015369b          	sltiu.d	a3,a0,1
1c01a900:	0c05                	addi	s8,s8,1
1c01a902:	22051713          	addi.d	a4,a0,0
1c01a906:	4852                	lw	a6,20(sp)
1c01a908:	48c2                	lw	a7,16(sp)
1c01a90a:	e299                	bnez	a3,1c01a910 <_ntoa_long_long+0xb0>
1c01a90c:	fb6c18e3          	bne	s8,s6,1c01a8bc <_ntoa_long_long+0x5c>
1c01a910:	47aa                	lw	a5,136(sp)
1c01a912:	c466                	sw	s9,8(sp)
1c01a914:	1018                	addi	a4,sp,32
1c01a916:	c23e                	sw	a5,4(sp)
1c01a918:	479a                	lw	a5,132(sp)
1c01a91a:	86d6                	mv	a3,s5
1c01a91c:	8652                	mv	a2,s4
1c01a91e:	c03e                	sw	a5,0(sp)
1c01a920:	85ce                	mv	a1,s3
1c01a922:	87e2                	mv	a5,s8
1c01a924:	854a                	mv	a0,s2
1c01a926:	db3ff0ef          	jal	ra,1c01a6d8 <_ntoa_format>
1c01a92a:	50b6                	lw	ra,108(sp)
1c01a92c:	5426                	lw	s0,104(sp)
1c01a92e:	5496                	lw	s1,100(sp)
1c01a930:	5906                	lw	s2,96(sp)
1c01a932:	49f6                	lw	s3,92(sp)
1c01a934:	4a66                	lw	s4,88(sp)
1c01a936:	4ad6                	lw	s5,84(sp)
1c01a938:	4b46                	lw	s6,80(sp)
1c01a93a:	4bb6                	lw	s7,76(sp)
1c01a93c:	4c26                	lw	s8,72(sp)
1c01a93e:	4c96                	lw	s9,68(sp)
1c01a940:	4d06                	lw	s10,64(sp)
1c01a942:	6109                	addi	sp,sp,128
1c01a944:	8082                	ret
1c01a946:	9522                	add	a0,a0,s0
1c01a948:	bf71                	j	1c01a8e4 <_ntoa_long_long+0x84>

1c01a94a <_out_char>:
1c01a94a:	c119                	beqz	a0,1c01a950 <_out_char+0x6>
1c01a94c:	e40fa06f          	j	1c014f8c <_putchar>
1c01a950:	8082                	ret

1c01a952 <_ftoa>:
1c01a952:	7175                	addi	sp,sp,-144
1c01a954:	d8da                	sw	s6,112(sp)
1c01a956:	d6de                	sw	s7,108(sp)
1c01a958:	d4e2                	sw	s8,104(sp)
1c01a95a:	d02a                	sw	a0,32(sp)
1c01a95c:	8b2e                	mv	s6,a1
1c01a95e:	8bb2                	mv	s7,a2
1c01a960:	8c36                	mv	s8,a3
1c01a962:	22071513          	addi.d	a0,a4,0
1c01a966:	22071613          	addi.d	a2,a4,0
1c01a96a:	c522                	sw	s0,136(sp)
1c01a96c:	c326                	sw	s1,132(sp)
1c01a96e:	c14a                	sw	s2,128(sp)
1c01a970:	dece                	sw	s3,124(sp)
1c01a972:	dcd2                	sw	s4,120(sp)
1c01a974:	c706                	sw	ra,140(sp)
1c01a976:	dad6                	sw	s5,116(sp)
1c01a978:	d2e6                	sw	s9,100(sp)
1c01a97a:	d0ea                	sw	s10,96(sp)
1c01a97c:	ceee                	sw	s11,92(sp)
1c01a97e:	22071913          	addi.d	s2,a4,0
1c01a982:	8442                	mv	s0,a6
1c01a984:	8a46                	mv	s4,a7
1c01a986:	44ca                	lw	s1,144(sp)
1c01a988:	89cf80ef          	jal	ra,1c012a24 <__eqdf2>
1c01a98c:	cd19                	beqz	a0,1c01a9aa <_ftoa+0x58>
1c01a98e:	1c01c737          	lui	a4,0x1c01c
1c01a992:	88a6                	mv	a7,s1
1c01a994:	8852                	mv	a6,s4
1c01a996:	478d                	li	a5,3
1c01a998:	07470713          	addi	a4,a4,116 # 1c01c074 <__func__.47299+0x2b4>
1c01a99c:	5502                	lw	a0,32(sp)
1c01a99e:	86e2                	mv	a3,s8
1c01a9a0:	865e                	mv	a2,s7
1c01a9a2:	85da                	mv	a1,s6
1c01a9a4:	c7dff0ef          	jal	ra,1c01a620 <_out_rev>
1c01a9a8:	a065                	j	1c01aa50 <_ftoa+0xfe>
1c01a9aa:	1c01c7b7          	lui	a5,0x1c01c
1c01a9ae:	04078793          	addi	a5,a5,64 # 1c01c040 <__func__.47299+0x280>
1c01a9b2:	4390                	lw	a2,0(a5)
1c01a9b4:	43d4                	lw	a3,4(a5)
1c01a9b6:	22091513          	addi.d	a0,s2,0
1c01a9ba:	e1ef60ef          	jal	ra,1c010fd8 <__ledf2>
1c01a9be:	00055a63          	bgez	a0,1c01a9d2 <_ftoa+0x80>
1c01a9c2:	1c01c737          	lui	a4,0x1c01c
1c01a9c6:	88a6                	mv	a7,s1
1c01a9c8:	8852                	mv	a6,s4
1c01a9ca:	4791                	li	a5,4
1c01a9cc:	07870713          	addi	a4,a4,120 # 1c01c078 <__func__.47299+0x2b8>
1c01a9d0:	b7f1                	j	1c01a99c <_ftoa+0x4a>
1c01a9d2:	1c01c7b7          	lui	a5,0x1c01c
1c01a9d6:	04878793          	addi	a5,a5,72 # 1c01c048 <__func__.47299+0x288>
1c01a9da:	4390                	lw	a2,0(a5)
1c01a9dc:	43d4                	lw	a3,4(a5)
1c01a9de:	22091513          	addi.d	a0,s2,0
1c01a9e2:	d4ef60ef          	jal	ra,1c010f30 <__gedf2>
1c01a9e6:	02a05363          	blez	a0,1c01aa0c <_ftoa+0xba>
1c01a9ea:	0044f793          	andi	a5,s1,4
1c01a9ee:	eb89                	bnez	a5,1c01aa00 <_ftoa+0xae>
1c01a9f0:	1c01c737          	lui	a4,0x1c01c
1c01a9f4:	07070713          	addi	a4,a4,112 # 1c01c070 <__func__.47299+0x2b0>
1c01a9f8:	478d                	li	a5,3
1c01a9fa:	88a6                	mv	a7,s1
1c01a9fc:	8852                	mv	a6,s4
1c01a9fe:	bf79                	j	1c01a99c <_ftoa+0x4a>
1c01aa00:	1c01c737          	lui	a4,0x1c01c
1c01aa04:	06870713          	addi	a4,a4,104 # 1c01c068 <__func__.47299+0x2a8>
1c01aa08:	4791                	li	a5,4
1c01aa0a:	bfc5                	j	1c01a9fa <_ftoa+0xa8>
1c01aa0c:	1c01c7b7          	lui	a5,0x1c01c
1c01aa10:	05078793          	addi	a5,a5,80 # 1c01c050 <__func__.47299+0x290>
1c01aa14:	4390                	lw	a2,0(a5)
1c01aa16:	43d4                	lw	a3,4(a5)
1c01aa18:	22091513          	addi.d	a0,s2,0
1c01aa1c:	d14f60ef          	jal	ra,1c010f30 <__gedf2>
1c01aa20:	00a04e63          	bgtz	a0,1c01aa3c <_ftoa+0xea>
1c01aa24:	1c01c7b7          	lui	a5,0x1c01c
1c01aa28:	05878793          	addi	a5,a5,88 # 1c01c058 <__func__.47299+0x298>
1c01aa2c:	4390                	lw	a2,0(a5)
1c01aa2e:	43d4                	lw	a3,4(a5)
1c01aa30:	22091513          	addi.d	a0,s2,0
1c01aa34:	da4f60ef          	jal	ra,1c010fd8 <__ledf2>
1c01aa38:	02055b63          	bgez	a0,1c01aa6e <_ftoa+0x11c>
1c01aa3c:	5502                	lw	a0,32(sp)
1c01aa3e:	c026                	sw	s1,0(sp)
1c01aa40:	88d2                	mv	a7,s4
1c01aa42:	8822                	mv	a6,s0
1c01aa44:	22091713          	addi.d	a4,s2,0
1c01aa48:	86e2                	mv	a3,s8
1c01aa4a:	865e                	mv	a2,s7
1c01aa4c:	85da                	mv	a1,s6
1c01aa4e:	2441                	jal	1c01acce <_etoa>
1c01aa50:	40ba                	lw	ra,140(sp)
1c01aa52:	442a                	lw	s0,136(sp)
1c01aa54:	449a                	lw	s1,132(sp)
1c01aa56:	490a                	lw	s2,128(sp)
1c01aa58:	59f6                	lw	s3,124(sp)
1c01aa5a:	5a66                	lw	s4,120(sp)
1c01aa5c:	5ad6                	lw	s5,116(sp)
1c01aa5e:	5b46                	lw	s6,112(sp)
1c01aa60:	5bb6                	lw	s7,108(sp)
1c01aa62:	5c26                	lw	s8,104(sp)
1c01aa64:	5c96                	lw	s9,100(sp)
1c01aa66:	5d06                	lw	s10,96(sp)
1c01aa68:	4df6                	lw	s11,92(sp)
1c01aa6a:	6149                	addi	sp,sp,144
1c01aa6c:	8082                	ret
1c01aa6e:	4601                	li	a2,0
1c01aa70:	4681                	li	a3,0
1c01aa72:	22091513          	addi.d	a0,s2,0
1c01aa76:	d62f60ef          	jal	ra,1c010fd8 <__ledf2>
1c01aa7a:	12055a63          	bgez	a0,1c01abae <_ftoa+0x25c>
1c01aa7e:	22091613          	addi.d	a2,s2,0
1c01aa82:	4501                	li	a0,0
1c01aa84:	4581                	li	a1,0
1c01aa86:	d94f80ef          	jal	ra,1c01301a <__subdf3>
1c01aa8a:	4705                	li	a4,1
1c01aa8c:	22051913          	addi.d	s2,a0,0
1c01aa90:	d23a                	sw	a4,36(sp)
1c01aa92:	4004f793          	andi	a5,s1,1024
1c01aa96:	10078e63          	beqz	a5,1c01abb2 <_ftoa+0x260>
1c01aa9a:	1818                	addi	a4,sp,48
1c01aa9c:	4c81                	li	s9,0
1c01aa9e:	46a5                	li	a3,9
1c01aaa0:	03000613          	li	a2,48
1c01aaa4:	0203d0fb          	lp.setupi	x1,32,1c01aab2 <_ftoa+0x160>
1c01aaa8:	0086f663          	bleu	s0,a3,1c01aab4 <_ftoa+0x162>
1c01aaac:	00c700ab          	p.sb	a2,1(a4!)
1c01aab0:	0c85                	addi	s9,s9,1
1c01aab2:	147d                	addi	s0,s0,-1
1c01aab4:	22091513          	addi.d	a0,s2,0
1c01aab8:	dd7f80ef          	jal	ra,1c01388e <__fixdfsi>
1c01aabc:	1c01c737          	lui	a4,0x1c01c
1c01aac0:	08070713          	addi	a4,a4,128 # 1c01c080 <pow10.1847>
1c01aac4:	00341793          	slli	a5,s0,0x3
1c01aac8:	97ba                	add	a5,a5,a4
1c01aaca:	4394                	lw	a3,0(a5)
1c01aacc:	43d8                	lw	a4,4(a5)
1c01aace:	8aaa                	mv	s5,a0
1c01aad0:	d436                	sw	a3,40(sp)
1c01aad2:	d63a                	sw	a4,44(sp)
1c01aad4:	f01f80ef          	jal	ra,1c0139d4 <__floatsidf>
1c01aad8:	22051613          	addi.d	a2,a0,0
1c01aadc:	22091513          	addi.d	a0,s2,0
1c01aae0:	d3af80ef          	jal	ra,1c01301a <__subdf3>
1c01aae4:	5622                	lw	a2,40(sp)
1c01aae6:	56b2                	lw	a3,44(sp)
1c01aae8:	1c01cdb7          	lui	s11,0x1c01c
1c01aaec:	fc9f70ef          	jal	ra,1c012ab4 <__muldf3>
1c01aaf0:	22051713          	addi.d	a4,a0,0
1c01aaf4:	c83e                	sw	a5,16(sp)
1c01aaf6:	ca2a                	sw	a0,20(sp)
1c01aaf8:	e4ff80ef          	jal	ra,1c013946 <__fixunsdfsi>
1c01aafc:	8d2a                	mv	s10,a0
1c01aafe:	f2ff80ef          	jal	ra,1c013a2c <__floatunsidf>
1c01ab02:	4752                	lw	a4,20(sp)
1c01ab04:	47c2                	lw	a5,16(sp)
1c01ab06:	22051613          	addi.d	a2,a0,0
1c01ab0a:	22071513          	addi.d	a0,a4,0
1c01ab0e:	d0cf80ef          	jal	ra,1c01301a <__subdf3>
1c01ab12:	060d8693          	addi	a3,s11,96 # 1c01c060 <__func__.47299+0x2a0>
1c01ab16:	4298                	lw	a4,0(a3)
1c01ab18:	42dc                	lw	a5,4(a3)
1c01ab1a:	22051813          	addi.d	a6,a0,0
1c01ab1e:	ca3a                	sw	a4,20(sp)
1c01ab20:	22071613          	addi.d	a2,a4,0
1c01ab24:	c83e                	sw	a5,16(sp)
1c01ab26:	cc46                	sw	a7,24(sp)
1c01ab28:	ce2a                	sw	a0,28(sp)
1c01ab2a:	c06f60ef          	jal	ra,1c010f30 <__gedf2>
1c01ab2e:	4752                	lw	a4,20(sp)
1c01ab30:	47c2                	lw	a5,16(sp)
1c01ab32:	4872                	lw	a6,28(sp)
1c01ab34:	48e2                	lw	a7,24(sp)
1c01ab36:	08a05063          	blez	a0,1c01abb6 <_ftoa+0x264>
1c01ab3a:	0d05                	addi	s10,s10,1
1c01ab3c:	856a                	mv	a0,s10
1c01ab3e:	eeff80ef          	jal	ra,1c013a2c <__floatunsidf>
1c01ab42:	22051613          	addi.d	a2,a0,0
1c01ab46:	5522                	lw	a0,40(sp)
1c01ab48:	55b2                	lw	a1,44(sp)
1c01ab4a:	c8ef60ef          	jal	ra,1c010fd8 <__ledf2>
1c01ab4e:	00a04463          	bgtz	a0,1c01ab56 <_ftoa+0x204>
1c01ab52:	0a85                	addi	s5,s5,1
1c01ab54:	4d01                	li	s10,0
1c01ab56:	e079                	bnez	s0,1c01ac1c <_ftoa+0x2ca>
1c01ab58:	8556                	mv	a0,s5
1c01ab5a:	e7bf80ef          	jal	ra,1c0139d4 <__floatsidf>
1c01ab5e:	22051613          	addi.d	a2,a0,0
1c01ab62:	22091513          	addi.d	a0,s2,0
1c01ab66:	cb4f80ef          	jal	ra,1c01301a <__subdf3>
1c01ab6a:	1c01c737          	lui	a4,0x1c01c
1c01ab6e:	06070713          	addi	a4,a4,96 # 1c01c060 <__func__.47299+0x2a0>
1c01ab72:	00072903          	lw	s2,0(a4)
1c01ab76:	00472983          	lw	s3,4(a4)
1c01ab7a:	22051d13          	addi.d	s10,a0,0
1c01ab7e:	22091613          	addi.d	a2,s2,0
1c01ab82:	c56f60ef          	jal	ra,1c010fd8 <__ledf2>
1c01ab86:	00055a63          	bgez	a0,1c01ab9a <_ftoa+0x248>
1c01ab8a:	22091613          	addi.d	a2,s2,0
1c01ab8e:	220d1513          	addi.d	a0,s10,0
1c01ab92:	b9ef60ef          	jal	ra,1c010f30 <__gedf2>
1c01ab96:	00a05663          	blez	a0,1c01aba2 <_ftoa+0x250>
1c01ab9a:	fc1ab7b3          	p.bclr	a5,s5,30,1
1c01ab9e:	c391                	beqz	a5,1c01aba2 <_ftoa+0x250>
1c01aba0:	0a85                	addi	s5,s5,1
1c01aba2:	1814                	addi	a3,sp,48
1c01aba4:	01968733          	add	a4,a3,s9
1c01aba8:	467d                	li	a2,31
1c01abaa:	46a9                	li	a3,10
1c01abac:	a8d1                	j	1c01ac80 <_ftoa+0x32e>
1c01abae:	d202                	sw	zero,36(sp)
1c01abb0:	b5cd                	j	1c01aa92 <_ftoa+0x140>
1c01abb2:	4419                	li	s0,6
1c01abb4:	b5dd                	j	1c01aa9a <_ftoa+0x148>
1c01abb6:	22071613          	addi.d	a2,a4,0
1c01abba:	22081513          	addi.d	a0,a6,0
1c01abbe:	c1af60ef          	jal	ra,1c010fd8 <__ledf2>
1c01abc2:	f8054ae3          	bltz	a0,1c01ab56 <_ftoa+0x204>
1c01abc6:	000d0563          	beqz	s10,1c01abd0 <_ftoa+0x27e>
1c01abca:	fc1d37b3          	p.bclr	a5,s10,30,1
1c01abce:	d7c1                	beqz	a5,1c01ab56 <_ftoa+0x204>
1c01abd0:	0d05                	addi	s10,s10,1
1c01abd2:	b751                	j	1c01ab56 <_ftoa+0x204>
1c01abd4:	02dd77b3          	remu	a5,s10,a3
1c01abd8:	147d                	addi	s0,s0,-1
1c01abda:	0c85                	addi	s9,s9,1
1c01abdc:	02dd5d33          	divu	s10,s10,a3
1c01abe0:	03078793          	addi	a5,a5,48
1c01abe4:	00f700ab          	p.sb	a5,1(a4!)
1c01abe8:	040d1063          	bnez	s10,1c01ac28 <_ftoa+0x2d6>
1c01abec:	181c                	addi	a5,sp,48
1c01abee:	9466                	add	s0,s0,s9
1c01abf0:	1814                	addi	a3,sp,48
1c01abf2:	01978733          	add	a4,a5,s9
1c01abf6:	9436                	add	s0,s0,a3
1c01abf8:	02000793          	li	a5,32
1c01abfc:	03000613          	li	a2,48
1c01ac00:	02fc8663          	beq	s9,a5,1c01ac2c <_ftoa+0x2da>
1c01ac04:	001c8693          	addi	a3,s9,1
1c01ac08:	04e41d63          	bne	s0,a4,1c01ac62 <_ftoa+0x310>
1c01ac0c:	0898                	addi	a4,sp,80
1c01ac0e:	9cba                	add	s9,s9,a4
1c01ac10:	02e00793          	li	a5,46
1c01ac14:	fefc8023          	sb	a5,-32(s9)
1c01ac18:	8cb6                	mv	s9,a3
1c01ac1a:	b761                	j	1c01aba2 <_ftoa+0x250>
1c01ac1c:	181c                	addi	a5,sp,48
1c01ac1e:	01978733          	add	a4,a5,s9
1c01ac22:	02000613          	li	a2,32
1c01ac26:	46a9                	li	a3,10
1c01ac28:	facc96e3          	bne	s9,a2,1c01abd4 <_ftoa+0x282>
1c01ac2c:	fa24b7b3          	p.bclr	a5,s1,29,2
1c01ac30:	0417bf63          	p.bneimm	a5,1,1c01ac8e <_ftoa+0x33c>
1c01ac34:	040a0d63          	beqz	s4,1c01ac8e <_ftoa+0x33c>
1c01ac38:	5712                	lw	a4,36(sp)
1c01ac3a:	e701                	bnez	a4,1c01ac42 <_ftoa+0x2f0>
1c01ac3c:	00c4f793          	andi	a5,s1,12
1c01ac40:	c391                	beqz	a5,1c01ac44 <_ftoa+0x2f2>
1c01ac42:	1a7d                	addi	s4,s4,-1
1c01ac44:	1814                	addi	a3,sp,48
1c01ac46:	019687b3          	add	a5,a3,s9
1c01ac4a:	477d                	li	a4,31
1c01ac4c:	03000693          	li	a3,48
1c01ac50:	034cff63          	bleu	s4,s9,1c01ac8e <_ftoa+0x33c>
1c01ac54:	03977963          	bleu	s9,a4,1c01ac86 <_ftoa+0x334>
1c01ac58:	88a6                	mv	a7,s1
1c01ac5a:	8852                	mv	a6,s4
1c01ac5c:	87e6                	mv	a5,s9
1c01ac5e:	1818                	addi	a4,sp,48
1c01ac60:	bb35                	j	1c01a99c <_ftoa+0x4a>
1c01ac62:	00c700ab          	p.sb	a2,1(a4!)
1c01ac66:	8cb6                	mv	s9,a3
1c01ac68:	bf61                	j	1c01ac00 <_ftoa+0x2ae>
1c01ac6a:	02dae7b3          	rem	a5,s5,a3
1c01ac6e:	0c85                	addi	s9,s9,1
1c01ac70:	02dacab3          	div	s5,s5,a3
1c01ac74:	03078793          	addi	a5,a5,48
1c01ac78:	00f700ab          	p.sb	a5,1(a4!)
1c01ac7c:	fa0a88e3          	beqz	s5,1c01ac2c <_ftoa+0x2da>
1c01ac80:	ff9675e3          	bleu	s9,a2,1c01ac6a <_ftoa+0x318>
1c01ac84:	b765                	j	1c01ac2c <_ftoa+0x2da>
1c01ac86:	0c85                	addi	s9,s9,1
1c01ac88:	00d780ab          	p.sb	a3,1(a5!)
1c01ac8c:	b7d1                	j	1c01ac50 <_ftoa+0x2fe>
1c01ac8e:	47fd                	li	a5,31
1c01ac90:	fd97e4e3          	bltu	a5,s9,1c01ac58 <_ftoa+0x306>
1c01ac94:	5712                	lw	a4,36(sp)
1c01ac96:	cb11                	beqz	a4,1c01acaa <_ftoa+0x358>
1c01ac98:	0894                	addi	a3,sp,80
1c01ac9a:	019687b3          	add	a5,a3,s9
1c01ac9e:	02d00713          	li	a4,45
1c01aca2:	fee78023          	sb	a4,-32(a5)
1c01aca6:	0c85                	addi	s9,s9,1
1c01aca8:	bf45                	j	1c01ac58 <_ftoa+0x306>
1c01acaa:	0044f793          	andi	a5,s1,4
1c01acae:	c799                	beqz	a5,1c01acbc <_ftoa+0x36a>
1c01acb0:	0898                	addi	a4,sp,80
1c01acb2:	019707b3          	add	a5,a4,s9
1c01acb6:	02b00713          	li	a4,43
1c01acba:	b7e5                	j	1c01aca2 <_ftoa+0x350>
1c01acbc:	0084f793          	andi	a5,s1,8
1c01acc0:	dfc1                	beqz	a5,1c01ac58 <_ftoa+0x306>
1c01acc2:	0894                	addi	a3,sp,80
1c01acc4:	019687b3          	add	a5,a3,s9
1c01acc8:	02000713          	li	a4,32
1c01accc:	bfd9                	j	1c01aca2 <_ftoa+0x350>

1c01acce <_etoa>:
1c01acce:	7119                	addi	sp,sp,-128
1c01acd0:	d2d6                	sw	s5,100(sp)
1c01acd2:	cce2                	sw	s8,88(sp)
1c01acd4:	cae6                	sw	s9,84(sp)
1c01acd6:	8aaa                	mv	s5,a0
1c01acd8:	8c2e                	mv	s8,a1
1c01acda:	dc32                	sw	a2,56(sp)
1c01acdc:	8cb6                	mv	s9,a3
1c01acde:	863a                	mv	a2,a4
1c01ace0:	86be                	mv	a3,a5
1c01ace2:	853a                	mv	a0,a4
1c01ace4:	85be                	mv	a1,a5
1c01ace6:	daa6                	sw	s1,116(sp)
1c01ace8:	d6ce                	sw	s3,108(sp)
1c01acea:	d4d2                	sw	s4,104(sp)
1c01acec:	c8ea                	sw	s10,80(sp)
1c01acee:	de86                	sw	ra,124(sp)
1c01acf0:	dca2                	sw	s0,120(sp)
1c01acf2:	d8ca                	sw	s2,112(sp)
1c01acf4:	d0da                	sw	s6,96(sp)
1c01acf6:	cede                	sw	s7,92(sp)
1c01acf8:	c6ee                	sw	s11,76(sp)
1c01acfa:	84ba                	mv	s1,a4
1c01acfc:	8a3e                	mv	s4,a5
1c01acfe:	89c2                	mv	s3,a6
1c01ad00:	da46                	sw	a7,52(sp)
1c01ad02:	4d0a                	lw	s10,128(sp)
1c01ad04:	d21f70ef          	jal	ra,1c012a24 <__eqdf2>
1c01ad08:	e90d                	bnez	a0,1c01ad3a <_etoa+0x6c>
1c01ad0a:	1c01c7b7          	lui	a5,0x1c01c
1c01ad0e:	04878793          	addi	a5,a5,72 # 1c01c048 <__func__.47299+0x288>
1c01ad12:	4390                	lw	a2,0(a5)
1c01ad14:	43d4                	lw	a3,4(a5)
1c01ad16:	8526                	mv	a0,s1
1c01ad18:	85d2                	mv	a1,s4
1c01ad1a:	a16f60ef          	jal	ra,1c010f30 <__gedf2>
1c01ad1e:	00a04e63          	bgtz	a0,1c01ad3a <_etoa+0x6c>
1c01ad22:	1c01c7b7          	lui	a5,0x1c01c
1c01ad26:	04078793          	addi	a5,a5,64 # 1c01c040 <__func__.47299+0x280>
1c01ad2a:	4390                	lw	a2,0(a5)
1c01ad2c:	43d4                	lw	a3,4(a5)
1c01ad2e:	8526                	mv	a0,s1
1c01ad30:	85d2                	mv	a1,s4
1c01ad32:	aa6f60ef          	jal	ra,1c010fd8 <__ledf2>
1c01ad36:	02055b63          	bgez	a0,1c01ad6c <_etoa+0x9e>
1c01ad3a:	5466                	lw	s0,120(sp)
1c01ad3c:	c16a                	sw	s10,128(sp)
1c01ad3e:	58d2                	lw	a7,52(sp)
1c01ad40:	5662                	lw	a2,56(sp)
1c01ad42:	50f6                	lw	ra,124(sp)
1c01ad44:	5946                	lw	s2,112(sp)
1c01ad46:	5b06                	lw	s6,96(sp)
1c01ad48:	4bf6                	lw	s7,92(sp)
1c01ad4a:	4d46                	lw	s10,80(sp)
1c01ad4c:	4db6                	lw	s11,76(sp)
1c01ad4e:	884e                	mv	a6,s3
1c01ad50:	8726                	mv	a4,s1
1c01ad52:	59b6                	lw	s3,108(sp)
1c01ad54:	54d6                	lw	s1,116(sp)
1c01ad56:	87d2                	mv	a5,s4
1c01ad58:	86e6                	mv	a3,s9
1c01ad5a:	5a26                	lw	s4,104(sp)
1c01ad5c:	4cd6                	lw	s9,84(sp)
1c01ad5e:	85e2                	mv	a1,s8
1c01ad60:	8556                	mv	a0,s5
1c01ad62:	4c66                	lw	s8,88(sp)
1c01ad64:	5a96                	lw	s5,100(sp)
1c01ad66:	6109                	addi	sp,sp,128
1c01ad68:	bebff06f          	j	1c01a952 <_ftoa>
1c01ad6c:	4601                	li	a2,0
1c01ad6e:	4681                	li	a3,0
1c01ad70:	8526                	mv	a0,s1
1c01ad72:	85d2                	mv	a1,s4
1c01ad74:	a64f60ef          	jal	ra,1c010fd8 <__ledf2>
1c01ad78:	de26                	sw	s1,60(sp)
1c01ad7a:	8452                	mv	s0,s4
1c01ad7c:	00055663          	bgez	a0,1c01ad88 <_etoa+0xba>
1c01ad80:	80000437          	lui	s0,0x80000
1c01ad84:	01444433          	xor	s0,s0,s4
1c01ad88:	400d7713          	andi	a4,s10,1024
1c01ad8c:	d83a                	sw	a4,48(sp)
1c01ad8e:	e311                	bnez	a4,1c01ad92 <_etoa+0xc4>
1c01ad90:	4999                	li	s3,6
1c01ad92:	03400313          	li	t1,52
1c01ad96:	4381                	li	t2,0
1c01ad98:	8726                	mv	a4,s1
1c01ad9a:	87a2                	mv	a5,s0
1c01ad9c:	20675533          	srl.d	a0,a4,t1
1c01ada0:	e8b53533          	p.bclr	a0,a0,20,11
1c01ada4:	c0150513          	addi	a0,a0,-1023
1c01ada8:	cc1a                	sw	t1,24(sp)
1c01adaa:	ca1e                	sw	t2,20(sp)
1c01adac:	d226                	sw	s1,36(sp)
1c01adae:	d022                	sw	s0,32(sp)
1c01adb0:	c25f80ef          	jal	ra,1c0139d4 <__floatsidf>
1c01adb4:	1c01c6b7          	lui	a3,0x1c01c
1c01adb8:	fd868693          	addi	a3,a3,-40 # 1c01bfd8 <__func__.47299+0x218>
1c01adbc:	4290                	lw	a2,0(a3)
1c01adbe:	42d4                	lw	a3,4(a3)
1c01adc0:	cf5f70ef          	jal	ra,1c012ab4 <__muldf3>
1c01adc4:	1c01c6b7          	lui	a3,0x1c01c
1c01adc8:	fe068693          	addi	a3,a3,-32 # 1c01bfe0 <__func__.47299+0x220>
1c01adcc:	4290                	lw	a2,0(a3)
1c01adce:	42d4                	lw	a3,4(a3)
1c01add0:	e07f60ef          	jal	ra,1c011bd6 <__adddf3>
1c01add4:	5782                	lw	a5,32(sp)
1c01add6:	22051b13          	addi.d	s6,a0,0
1c01adda:	5712                	lw	a4,36(sp)
1c01addc:	d747b5b3          	p.bclr	a1,a5,11,20
1c01ade0:	1c01c7b7          	lui	a5,0x1c01c
1c01ade4:	fe878793          	addi	a5,a5,-24 # 1c01bfe8 <__func__.47299+0x228>
1c01ade8:	4390                	lw	a2,0(a5)
1c01adea:	43d4                	lw	a3,4(a5)
1c01adec:	d345c7b3          	p.bset	a5,a1,9,20
1c01adf0:	853a                	mv	a0,a4
1c01adf2:	85be                	mv	a1,a5
1c01adf4:	a26f80ef          	jal	ra,1c01301a <__subdf3>
1c01adf8:	1c01c7b7          	lui	a5,0x1c01c
1c01adfc:	ff078793          	addi	a5,a5,-16 # 1c01bff0 <__func__.47299+0x230>
1c01ae00:	4390                	lw	a2,0(a5)
1c01ae02:	43d4                	lw	a3,4(a5)
1c01ae04:	cb1f70ef          	jal	ra,1c012ab4 <__muldf3>
1c01ae08:	22051613          	addi.d	a2,a0,0
1c01ae0c:	220b1513          	addi.d	a0,s6,0
1c01ae10:	dc7f60ef          	jal	ra,1c011bd6 <__adddf3>
1c01ae14:	a7bf80ef          	jal	ra,1c01388e <__fixdfsi>
1c01ae18:	892a                	mv	s2,a0
1c01ae1a:	bbbf80ef          	jal	ra,1c0139d4 <__floatsidf>
1c01ae1e:	1c01c7b7          	lui	a5,0x1c01c
1c01ae22:	ff878793          	addi	a5,a5,-8 # 1c01bff8 <__func__.47299+0x238>
1c01ae26:	4390                	lw	a2,0(a5)
1c01ae28:	43d4                	lw	a3,4(a5)
1c01ae2a:	22051b13          	addi.d	s6,a0,0
1c01ae2e:	c87f70ef          	jal	ra,1c012ab4 <__muldf3>
1c01ae32:	1c01c7b7          	lui	a5,0x1c01c
1c01ae36:	06078793          	addi	a5,a5,96 # 1c01c060 <__func__.47299+0x2a0>
1c01ae3a:	4390                	lw	a2,0(a5)
1c01ae3c:	43d4                	lw	a3,4(a5)
1c01ae3e:	d99f60ef          	jal	ra,1c011bd6 <__adddf3>
1c01ae42:	a4df80ef          	jal	ra,1c01388e <__fixdfsi>
1c01ae46:	1c01c7b7          	lui	a5,0x1c01c
1c01ae4a:	00078793          	mv	a5,a5
1c01ae4e:	4390                	lw	a2,0(a5)
1c01ae50:	43d4                	lw	a3,4(a5)
1c01ae52:	8daa                	mv	s11,a0
1c01ae54:	220b1513          	addi.d	a0,s6,0
1c01ae58:	c5df70ef          	jal	ra,1c012ab4 <__muldf3>
1c01ae5c:	22051b13          	addi.d	s6,a0,0
1c01ae60:	856e                	mv	a0,s11
1c01ae62:	b73f80ef          	jal	ra,1c0139d4 <__floatsidf>
1c01ae66:	1c01c7b7          	lui	a5,0x1c01c
1c01ae6a:	00878793          	addi	a5,a5,8 # 1c01c008 <__func__.47299+0x248>
1c01ae6e:	4390                	lw	a2,0(a5)
1c01ae70:	43d4                	lw	a3,4(a5)
1c01ae72:	3ffd8d93          	addi	s11,s11,1023
1c01ae76:	c3ff70ef          	jal	ra,1c012ab4 <__muldf3>
1c01ae7a:	22051613          	addi.d	a2,a0,0
1c01ae7e:	220b1513          	addi.d	a0,s6,0
1c01ae82:	998f80ef          	jal	ra,1c01301a <__subdf3>
1c01ae86:	22051713          	addi.d	a4,a0,0
1c01ae8a:	22051613          	addi.d	a2,a0,0
1c01ae8e:	d03e                	sw	a5,32(sp)
1c01ae90:	d22a                	sw	a0,36(sp)
1c01ae92:	c23f70ef          	jal	ra,1c012ab4 <__muldf3>
1c01ae96:	5712                	lw	a4,36(sp)
1c01ae98:	5782                	lw	a5,32(sp)
1c01ae9a:	22051b13          	addi.d	s6,a0,0
1c01ae9e:	22071613          	addi.d	a2,a4,0
1c01aea2:	22071513          	addi.d	a0,a4,0
1c01aea6:	d31f60ef          	jal	ra,1c011bd6 <__adddf3>
1c01aeaa:	1c01c6b7          	lui	a3,0x1c01c
1c01aeae:	01068693          	addi	a3,a3,16 # 1c01c010 <__func__.47299+0x250>
1c01aeb2:	4290                	lw	a2,0(a3)
1c01aeb4:	42d4                	lw	a3,4(a3)
1c01aeb6:	22051813          	addi.d	a6,a0,0
1c01aeba:	220b1513          	addi.d	a0,s6,0
1c01aebe:	d642                	sw	a6,44(sp)
1c01aec0:	d446                	sw	a7,40(sp)
1c01aec2:	d5ef70ef          	jal	ra,1c012420 <__divdf3>
1c01aec6:	1c01ce37          	lui	t3,0x1c01c
1c01aeca:	018e0e13          	addi	t3,t3,24 # 1c01c018 <__func__.47299+0x258>
1c01aece:	000e2603          	lw	a2,0(t3)
1c01aed2:	004e2683          	lw	a3,4(t3)
1c01aed6:	ce72                	sw	t3,28(sp)
1c01aed8:	cfff60ef          	jal	ra,1c011bd6 <__adddf3>
1c01aedc:	22051613          	addi.d	a2,a0,0
1c01aee0:	220b1513          	addi.d	a0,s6,0
1c01aee4:	d3cf70ef          	jal	ra,1c012420 <__divdf3>
1c01aee8:	1c01c6b7          	lui	a3,0x1c01c
1c01aeec:	02068693          	addi	a3,a3,32 # 1c01c020 <__func__.47299+0x260>
1c01aef0:	4290                	lw	a2,0(a3)
1c01aef2:	42d4                	lw	a3,4(a3)
1c01aef4:	ce3f60ef          	jal	ra,1c011bd6 <__adddf3>
1c01aef8:	22051613          	addi.d	a2,a0,0
1c01aefc:	220b1513          	addi.d	a0,s6,0
1c01af00:	d20f70ef          	jal	ra,1c012420 <__divdf3>
1c01af04:	5712                	lw	a4,36(sp)
1c01af06:	5782                	lw	a5,32(sp)
1c01af08:	22051b13          	addi.d	s6,a0,0
1c01af0c:	1c01c5b7          	lui	a1,0x1c01c
1c01af10:	22071613          	addi.d	a2,a4,0
1c01af14:	02858793          	addi	a5,a1,40 # 1c01c028 <__func__.47299+0x268>
1c01af18:	4388                	lw	a0,0(a5)
1c01af1a:	43cc                	lw	a1,4(a5)
1c01af1c:	8fef80ef          	jal	ra,1c01301a <__subdf3>
1c01af20:	22051613          	addi.d	a2,a0,0
1c01af24:	220b1513          	addi.d	a0,s6,0
1c01af28:	caff60ef          	jal	ra,1c011bd6 <__adddf3>
1c01af2c:	5832                	lw	a6,44(sp)
1c01af2e:	58a2                	lw	a7,40(sp)
1c01af30:	22051613          	addi.d	a2,a0,0
1c01af34:	22081513          	addi.d	a0,a6,0
1c01af38:	ce8f70ef          	jal	ra,1c012420 <__divdf3>
1c01af3c:	1c01c7b7          	lui	a5,0x1c01c
1c01af40:	03078793          	addi	a5,a5,48 # 1c01c030 <__func__.47299+0x270>
1c01af44:	4390                	lw	a2,0(a5)
1c01af46:	43d4                	lw	a3,4(a5)
1c01af48:	c8ff60ef          	jal	ra,1c011bd6 <__adddf3>
1c01af4c:	4362                	lw	t1,24(sp)
1c01af4e:	43d2                	lw	t2,20(sp)
1c01af50:	866e                	mv	a2,s11
1c01af52:	41fdd693          	srai	a3,s11,0x1f
1c01af56:	20661633          	sll.d	a2,a2,t1
1c01af5a:	b5bf70ef          	jal	ra,1c012ab4 <__muldf3>
1c01af5e:	22051613          	addi.d	a2,a0,0
1c01af62:	22051b13          	addi.d	s6,a0,0
1c01af66:	8526                	mv	a0,s1
1c01af68:	85a2                	mv	a1,s0
1c01af6a:	86ef60ef          	jal	ra,1c010fd8 <__ledf2>
1c01af6e:	4e72                	lw	t3,28(sp)
1c01af70:	00055d63          	bgez	a0,1c01af8a <_etoa+0x2bc>
1c01af74:	000e2603          	lw	a2,0(t3)
1c01af78:	004e2683          	lw	a3,4(t3)
1c01af7c:	220b1513          	addi.d	a0,s6,0
1c01af80:	197d                	addi	s2,s2,-1
1c01af82:	c9ef70ef          	jal	ra,1c012420 <__divdf3>
1c01af86:	22051b13          	addi.d	s6,a0,0
1c01af8a:	0c600793          	li	a5,198
1c01af8e:	06390d93          	addi	s11,s2,99
1c01af92:	01b7bdb3          	sltu	s11,a5,s11
1c01af96:	014d1793          	slli	a5,s10,0x14
1c01af9a:	0d91                	addi	s11,s11,4
1c01af9c:	0407d363          	bgez	a5,1c01afe2 <_etoa+0x314>
1c01afa0:	1c01c7b7          	lui	a5,0x1c01c
1c01afa4:	b5078793          	addi	a5,a5,-1200 # 1c01bb50 <__clz_tab+0x108>
1c01afa8:	4390                	lw	a2,0(a5)
1c01afaa:	43d4                	lw	a3,4(a5)
1c01afac:	8526                	mv	a0,s1
1c01afae:	85a2                	mv	a1,s0
1c01afb0:	f81f50ef          	jal	ra,1c010f30 <__gedf2>
1c01afb4:	0e054f63          	bltz	a0,1c01b0b2 <_etoa+0x3e4>
1c01afb8:	1c01c7b7          	lui	a5,0x1c01c
1c01afbc:	03878793          	addi	a5,a5,56 # 1c01c038 <__func__.47299+0x278>
1c01afc0:	4390                	lw	a2,0(a5)
1c01afc2:	43d4                	lw	a3,4(a5)
1c01afc4:	8526                	mv	a0,s1
1c01afc6:	85a2                	mv	a1,s0
1c01afc8:	810f60ef          	jal	ra,1c010fd8 <__ledf2>
1c01afcc:	0e055363          	bgez	a0,1c01b0b2 <_etoa+0x3e4>
1c01afd0:	0d395f63          	ble	s3,s2,1c01b0ae <_etoa+0x3e0>
1c01afd4:	412989b3          	sub	s3,s3,s2
1c01afd8:	19fd                	addi	s3,s3,-1
1c01afda:	c0ad4d33          	p.bset	s10,s10,0,10
1c01afde:	4d81                	li	s11,0
1c01afe0:	4901                	li	s2,0
1c01afe2:	57d2                	lw	a5,52(sp)
1c01afe4:	4881                	li	a7,0
1c01afe6:	00fdf463          	bleu	a5,s11,1c01afee <_etoa+0x320>
1c01afea:	41b788b3          	sub	a7,a5,s11
1c01afee:	002d7713          	andi	a4,s10,2
1c01aff2:	d83a                	sw	a4,48(sp)
1c01aff4:	c701                	beqz	a4,1c01affc <_etoa+0x32e>
1c01aff6:	000d8363          	beqz	s11,1c01affc <_etoa+0x32e>
1c01affa:	4881                	li	a7,0
1c01affc:	00090c63          	beqz	s2,1c01b014 <_etoa+0x346>
1c01b000:	85a2                	mv	a1,s0
1c01b002:	220b1613          	addi.d	a2,s6,0
1c01b006:	8526                	mv	a0,s1
1c01b008:	d446                	sw	a7,40(sp)
1c01b00a:	c16f70ef          	jal	ra,1c012420 <__divdf3>
1c01b00e:	58a2                	lw	a7,40(sp)
1c01b010:	de2a                	sw	a0,60(sp)
1c01b012:	842e                	mv	s0,a1
1c01b014:	4601                	li	a2,0
1c01b016:	4681                	li	a3,0
1c01b018:	8526                	mv	a0,s1
1c01b01a:	85d2                	mv	a1,s4
1c01b01c:	d446                	sw	a7,40(sp)
1c01b01e:	fbbf50ef          	jal	ra,1c010fd8 <__ledf2>
1c01b022:	58a2                	lw	a7,40(sp)
1c01b024:	00055563          	bgez	a0,1c01b02e <_etoa+0x360>
1c01b028:	800007b7          	lui	a5,0x80000
1c01b02c:	8c3d                	xor	s0,s0,a5
1c01b02e:	5672                	lw	a2,60(sp)
1c01b030:	c0bd37b3          	p.bclr	a5,s10,0,11
1c01b034:	c03e                	sw	a5,0(sp)
1c01b036:	8732                	mv	a4,a2
1c01b038:	5662                	lw	a2,56(sp)
1c01b03a:	884e                	mv	a6,s3
1c01b03c:	87a2                	mv	a5,s0
1c01b03e:	86e6                	mv	a3,s9
1c01b040:	85e2                	mv	a1,s8
1c01b042:	8556                	mv	a0,s5
1c01b044:	90fff0ef          	jal	ra,1c01a952 <_ftoa>
1c01b048:	862a                	mv	a2,a0
1c01b04a:	040d8263          	beqz	s11,1c01b08e <_etoa+0x3c0>
1c01b04e:	020d7d13          	andi	s10,s10,32
1c01b052:	04500513          	li	a0,69
1c01b056:	000d1463          	bnez	s10,1c01b05e <_etoa+0x390>
1c01b05a:	06500513          	li	a0,101
1c01b05e:	86e6                	mv	a3,s9
1c01b060:	85e2                	mv	a1,s8
1c01b062:	00160413          	addi	s0,a2,1
1c01b066:	9a82                	jalr	s5
1c01b068:	4795                	li	a5,5
1c01b06a:	1dfd                	addi	s11,s11,-1
1c01b06c:	c23e                	sw	a5,4(sp)
1c01b06e:	8622                	mv	a2,s0
1c01b070:	01f95793          	srli	a5,s2,0x1f
1c01b074:	c06e                	sw	s11,0(sp)
1c01b076:	4881                	li	a7,0
1c01b078:	4829                	li	a6,10
1c01b07a:	04090733          	p.avg	a4,s2,zero
1c01b07e:	86e6                	mv	a3,s9
1c01b080:	85e2                	mv	a1,s8
1c01b082:	8556                	mv	a0,s5
1c01b084:	f64ff0ef          	jal	ra,1c01a7e8 <_ntoa_long>
1c01b088:	57c2                	lw	a5,48(sp)
1c01b08a:	862a                	mv	a2,a0
1c01b08c:	e3a9                	bnez	a5,1c01b0ce <_etoa+0x400>
1c01b08e:	50f6                	lw	ra,124(sp)
1c01b090:	5466                	lw	s0,120(sp)
1c01b092:	54d6                	lw	s1,116(sp)
1c01b094:	5946                	lw	s2,112(sp)
1c01b096:	59b6                	lw	s3,108(sp)
1c01b098:	5a26                	lw	s4,104(sp)
1c01b09a:	5a96                	lw	s5,100(sp)
1c01b09c:	5b06                	lw	s6,96(sp)
1c01b09e:	4bf6                	lw	s7,92(sp)
1c01b0a0:	4c66                	lw	s8,88(sp)
1c01b0a2:	4cd6                	lw	s9,84(sp)
1c01b0a4:	4d46                	lw	s10,80(sp)
1c01b0a6:	4db6                	lw	s11,76(sp)
1c01b0a8:	8532                	mv	a0,a2
1c01b0aa:	6109                	addi	sp,sp,128
1c01b0ac:	8082                	ret
1c01b0ae:	4981                	li	s3,0
1c01b0b0:	b72d                	j	1c01afda <_etoa+0x30c>
1c01b0b2:	f20988e3          	beqz	s3,1c01afe2 <_etoa+0x314>
1c01b0b6:	5742                	lw	a4,48(sp)
1c01b0b8:	d70d                	beqz	a4,1c01afe2 <_etoa+0x314>
1c01b0ba:	19fd                	addi	s3,s3,-1
1c01b0bc:	b71d                	j	1c01afe2 <_etoa+0x314>
1c01b0be:	86e6                	mv	a3,s9
1c01b0c0:	85e2                	mv	a1,s8
1c01b0c2:	02000513          	li	a0,32
1c01b0c6:	00160413          	addi	s0,a2,1
1c01b0ca:	9a82                	jalr	s5
1c01b0cc:	8622                	mv	a2,s0
1c01b0ce:	5762                	lw	a4,56(sp)
1c01b0d0:	40e607b3          	sub	a5,a2,a4
1c01b0d4:	5752                	lw	a4,52(sp)
1c01b0d6:	fee7e4e3          	bltu	a5,a4,1c01b0be <_etoa+0x3f0>
1c01b0da:	bf55                	j	1c01b08e <_etoa+0x3c0>

1c01b0dc <_vsnprintf>:
1c01b0dc:	711d                	addi	sp,sp,-96
1c01b0de:	caa6                	sw	s1,84(sp)
1c01b0e0:	c8ca                	sw	s2,80(sp)
1c01b0e2:	c6ce                	sw	s3,76(sp)
1c01b0e4:	c4d2                	sw	s4,72(sp)
1c01b0e6:	dc62                	sw	s8,56(sp)
1c01b0e8:	ce86                	sw	ra,92(sp)
1c01b0ea:	cca2                	sw	s0,88(sp)
1c01b0ec:	c2d6                	sw	s5,68(sp)
1c01b0ee:	c0da                	sw	s6,64(sp)
1c01b0f0:	de5e                	sw	s7,60(sp)
1c01b0f2:	da66                	sw	s9,52(sp)
1c01b0f4:	d86a                	sw	s10,48(sp)
1c01b0f6:	d66e                	sw	s11,44(sp)
1c01b0f8:	8a2e                	mv	s4,a1
1c01b0fa:	8932                	mv	s2,a2
1c01b0fc:	89b6                	mv	s3,a3
1c01b0fe:	8c3a                	mv	s8,a4
1c01b100:	84aa                	mv	s1,a0
1c01b102:	e589                	bnez	a1,1c01b10c <_vsnprintf+0x30>
1c01b104:	1c01a4b7          	lui	s1,0x1c01a
1c01b108:	61e48493          	addi	s1,s1,1566 # 1c01a61e <_out_null>
1c01b10c:	6b41                	lui	s6,0x10
1c01b10e:	4d81                	li	s11,0
1c01b110:	1b7d                	addi	s6,s6,-1
1c01b112:	4ba9                	li	s7,10
1c01b114:	0009c503          	lbu	a0,0(s3)
1c01b118:	e915                	bnez	a0,1c01b14c <_vsnprintf+0x70>
1c01b11a:	866e                	mv	a2,s11
1c01b11c:	012de463          	bltu	s11,s2,1c01b124 <_vsnprintf+0x48>
1c01b120:	fff90613          	addi	a2,s2,-1
1c01b124:	86ca                	mv	a3,s2
1c01b126:	85d2                	mv	a1,s4
1c01b128:	4501                	li	a0,0
1c01b12a:	9482                	jalr	s1
1c01b12c:	40f6                	lw	ra,92(sp)
1c01b12e:	4466                	lw	s0,88(sp)
1c01b130:	856e                	mv	a0,s11
1c01b132:	44d6                	lw	s1,84(sp)
1c01b134:	4946                	lw	s2,80(sp)
1c01b136:	49b6                	lw	s3,76(sp)
1c01b138:	4a26                	lw	s4,72(sp)
1c01b13a:	4a96                	lw	s5,68(sp)
1c01b13c:	4b06                	lw	s6,64(sp)
1c01b13e:	5bf2                	lw	s7,60(sp)
1c01b140:	5c62                	lw	s8,56(sp)
1c01b142:	5cd2                	lw	s9,52(sp)
1c01b144:	5d42                	lw	s10,48(sp)
1c01b146:	5db2                	lw	s11,44(sp)
1c01b148:	6125                	addi	sp,sp,96
1c01b14a:	8082                	ret
1c01b14c:	02500793          	li	a5,37
1c01b150:	0985                	addi	s3,s3,1
1c01b152:	00f50863          	beq	a0,a5,1c01b162 <_vsnprintf+0x86>
1c01b156:	001d8413          	addi	s0,s11,1
1c01b15a:	86ca                	mv	a3,s2
1c01b15c:	866e                	mv	a2,s11
1c01b15e:	85d2                	mv	a1,s4
1c01b160:	a275                	j	1c01b30c <_vsnprintf+0x230>
1c01b162:	4781                	li	a5,0
1c01b164:	02b00613          	li	a2,43
1c01b168:	03000593          	li	a1,48
1c01b16c:	02000513          	li	a0,32
1c01b170:	02300813          	li	a6,35
1c01b174:	a811                	j	1c01b188 <_vsnprintf+0xac>
1c01b176:	02d00893          	li	a7,45
1c01b17a:	03170c63          	beq	a4,a7,1c01b1b2 <_vsnprintf+0xd6>
1c01b17e:	02b71063          	bne	a4,a1,1c01b19e <_vsnprintf+0xc2>
1c01b182:	c007c7b3          	p.bset	a5,a5,0,0
1c01b186:	89b6                	mv	s3,a3
1c01b188:	86ce                	mv	a3,s3
1c01b18a:	0016c70b          	p.lbu	a4,1(a3!)
1c01b18e:	02c70563          	beq	a4,a2,1c01b1b8 <_vsnprintf+0xdc>
1c01b192:	fee662e3          	bltu	a2,a4,1c01b176 <_vsnprintf+0x9a>
1c01b196:	02a70463          	beq	a4,a0,1c01b1be <_vsnprintf+0xe2>
1c01b19a:	03070563          	beq	a4,a6,1c01b1c4 <_vsnprintf+0xe8>
1c01b19e:	fd070613          	addi	a2,a4,-48
1c01b1a2:	0ff67613          	andi	a2,a2,255
1c01b1a6:	45a5                	li	a1,9
1c01b1a8:	06c5e563          	bltu	a1,a2,1c01b212 <_vsnprintf+0x136>
1c01b1ac:	4c81                	li	s9,0
1c01b1ae:	45a5                	li	a1,9
1c01b1b0:	a015                	j	1c01b1d4 <_vsnprintf+0xf8>
1c01b1b2:	c017c7b3          	p.bset	a5,a5,0,1
1c01b1b6:	bfc1                	j	1c01b186 <_vsnprintf+0xaa>
1c01b1b8:	c027c7b3          	p.bset	a5,a5,0,2
1c01b1bc:	b7e9                	j	1c01b186 <_vsnprintf+0xaa>
1c01b1be:	c037c7b3          	p.bset	a5,a5,0,3
1c01b1c2:	b7d1                	j	1c01b186 <_vsnprintf+0xaa>
1c01b1c4:	c047c7b3          	p.bset	a5,a5,0,4
1c01b1c8:	bf7d                	j	1c01b186 <_vsnprintf+0xaa>
1c01b1ca:	437c8733          	p.mac	a4,s9,s7
1c01b1ce:	89b2                	mv	s3,a2
1c01b1d0:	fd070c93          	addi	s9,a4,-48
1c01b1d4:	864e                	mv	a2,s3
1c01b1d6:	0016470b          	p.lbu	a4,1(a2!)
1c01b1da:	fd070693          	addi	a3,a4,-48
1c01b1de:	0ff6f693          	andi	a3,a3,255
1c01b1e2:	fed5f4e3          	bleu	a3,a1,1c01b1ca <_vsnprintf+0xee>
1c01b1e6:	0009c683          	lbu	a3,0(s3)
1c01b1ea:	02e00713          	li	a4,46
1c01b1ee:	0ae69363          	bne	a3,a4,1c01b294 <_vsnprintf+0x1b8>
1c01b1f2:	0019c603          	lbu	a2,1(s3)
1c01b1f6:	45a5                	li	a1,9
1c01b1f8:	00198713          	addi	a4,s3,1
1c01b1fc:	fd060693          	addi	a3,a2,-48
1c01b200:	0ff6f693          	andi	a3,a3,255
1c01b204:	c0a7c7b3          	p.bset	a5,a5,0,10
1c01b208:	06d5e963          	bltu	a1,a3,1c01b27a <_vsnprintf+0x19e>
1c01b20c:	4d01                	li	s10,0
1c01b20e:	45a5                	li	a1,9
1c01b210:	a805                	j	1c01b240 <_vsnprintf+0x164>
1c01b212:	02a00613          	li	a2,42
1c01b216:	4c81                	li	s9,0
1c01b218:	fcc717e3          	bne	a4,a2,1c01b1e6 <_vsnprintf+0x10a>
1c01b21c:	000c2c83          	lw	s9,0(s8)
1c01b220:	004c0713          	addi	a4,s8,4
1c01b224:	000cd663          	bgez	s9,1c01b230 <_vsnprintf+0x154>
1c01b228:	c017c7b3          	p.bset	a5,a5,0,1
1c01b22c:	41900cb3          	neg	s9,s9
1c01b230:	8c3a                	mv	s8,a4
1c01b232:	89b6                	mv	s3,a3
1c01b234:	bf4d                	j	1c01b1e6 <_vsnprintf+0x10a>
1c01b236:	437d0833          	p.mac	a6,s10,s7
1c01b23a:	8732                	mv	a4,a2
1c01b23c:	fd080d13          	addi	s10,a6,-48
1c01b240:	863a                	mv	a2,a4
1c01b242:	0016480b          	p.lbu	a6,1(a2!)
1c01b246:	fd080693          	addi	a3,a6,-48
1c01b24a:	0ff6f693          	andi	a3,a3,255
1c01b24e:	fed5f4e3          	bleu	a3,a1,1c01b236 <_vsnprintf+0x15a>
1c01b252:	00074683          	lbu	a3,0(a4)
1c01b256:	06c00613          	li	a2,108
1c01b25a:	00170993          	addi	s3,a4,1
1c01b25e:	04c68963          	beq	a3,a2,1c01b2b0 <_vsnprintf+0x1d4>
1c01b262:	02d66c63          	bltu	a2,a3,1c01b29a <_vsnprintf+0x1be>
1c01b266:	06800613          	li	a2,104
1c01b26a:	04c68c63          	beq	a3,a2,1c01b2c2 <_vsnprintf+0x1e6>
1c01b26e:	06a00613          	li	a2,106
1c01b272:	0ac68363          	beq	a3,a2,1c01b318 <_vsnprintf+0x23c>
1c01b276:	89ba                	mv	s3,a4
1c01b278:	a899                	j	1c01b2ce <_vsnprintf+0x1f2>
1c01b27a:	02a00693          	li	a3,42
1c01b27e:	4d01                	li	s10,0
1c01b280:	fcd619e3          	bne	a2,a3,1c01b252 <_vsnprintf+0x176>
1c01b284:	000c2303          	lw	t1,0(s8)
1c01b288:	00298713          	addi	a4,s3,2
1c01b28c:	0c11                	addi	s8,s8,4
1c01b28e:	04036d33          	p.max	s10,t1,zero
1c01b292:	b7c1                	j	1c01b252 <_vsnprintf+0x176>
1c01b294:	874e                	mv	a4,s3
1c01b296:	4d01                	li	s10,0
1c01b298:	bf6d                	j	1c01b252 <_vsnprintf+0x176>
1c01b29a:	07400613          	li	a2,116
1c01b29e:	00c68663          	beq	a3,a2,1c01b2aa <_vsnprintf+0x1ce>
1c01b2a2:	07a00613          	li	a2,122
1c01b2a6:	fcc698e3          	bne	a3,a2,1c01b276 <_vsnprintf+0x19a>
1c01b2aa:	c087c7b3          	p.bset	a5,a5,0,8
1c01b2ae:	a005                	j	1c01b2ce <_vsnprintf+0x1f2>
1c01b2b0:	00174603          	lbu	a2,1(a4)
1c01b2b4:	fed61be3          	bne	a2,a3,1c01b2aa <_vsnprintf+0x1ce>
1c01b2b8:	c287c7b3          	p.bset	a5,a5,1,8
1c01b2bc:	00270993          	addi	s3,a4,2
1c01b2c0:	a039                	j	1c01b2ce <_vsnprintf+0x1f2>
1c01b2c2:	00174603          	lbu	a2,1(a4)
1c01b2c6:	04d60663          	beq	a2,a3,1c01b312 <_vsnprintf+0x236>
1c01b2ca:	c077c7b3          	p.bset	a5,a5,0,7
1c01b2ce:	0019c50b          	p.lbu	a0,1(s3!)
1c01b2d2:	06500713          	li	a4,101
1c01b2d6:	22e50563          	beq	a0,a4,1c01b500 <_vsnprintf+0x424>
1c01b2da:	0ea76d63          	bltu	a4,a0,1c01b3d4 <_vsnprintf+0x2f8>
1c01b2de:	04700713          	li	a4,71
1c01b2e2:	20e50f63          	beq	a0,a4,1c01b500 <_vsnprintf+0x424>
1c01b2e6:	02a76c63          	bltu	a4,a0,1c01b31e <_vsnprintf+0x242>
1c01b2ea:	04500713          	li	a4,69
1c01b2ee:	20e50963          	beq	a0,a4,1c01b500 <_vsnprintf+0x424>
1c01b2f2:	1ca76d63          	bltu	a4,a0,1c01b4cc <_vsnprintf+0x3f0>
1c01b2f6:	02500793          	li	a5,37
1c01b2fa:	e4f51ee3          	bne	a0,a5,1c01b156 <_vsnprintf+0x7a>
1c01b2fe:	001d8413          	addi	s0,s11,1
1c01b302:	86ca                	mv	a3,s2
1c01b304:	866e                	mv	a2,s11
1c01b306:	85d2                	mv	a1,s4
1c01b308:	02500513          	li	a0,37
1c01b30c:	9482                	jalr	s1
1c01b30e:	8da2                	mv	s11,s0
1c01b310:	b511                	j	1c01b114 <_vsnprintf+0x38>
1c01b312:	c267c7b3          	p.bset	a5,a5,1,6
1c01b316:	b75d                	j	1c01b2bc <_vsnprintf+0x1e0>
1c01b318:	c097c7b3          	p.bset	a5,a5,0,9
1c01b31c:	bf4d                	j	1c01b2ce <_vsnprintf+0x1f2>
1c01b31e:	06200713          	li	a4,98
1c01b322:	00e50863          	beq	a0,a4,1c01b332 <_vsnprintf+0x256>
1c01b326:	0aa76063          	bltu	a4,a0,1c01b3c6 <_vsnprintf+0x2ea>
1c01b32a:	05800713          	li	a4,88
1c01b32e:	e2e514e3          	bne	a0,a4,1c01b156 <_vsnprintf+0x7a>
1c01b332:	07800713          	li	a4,120
1c01b336:	32e50d63          	beq	a0,a4,1c01b670 <_vsnprintf+0x594>
1c01b33a:	05800713          	li	a4,88
1c01b33e:	32e50763          	beq	a0,a4,1c01b66c <_vsnprintf+0x590>
1c01b342:	06f00713          	li	a4,111
1c01b346:	32e50763          	beq	a0,a4,1c01b674 <_vsnprintf+0x598>
1c01b34a:	06200713          	li	a4,98
1c01b34e:	32e50563          	beq	a0,a4,1c01b678 <_vsnprintf+0x59c>
1c01b352:	06900713          	li	a4,105
1c01b356:	c047b7b3          	p.bclr	a5,a5,0,4
1c01b35a:	4829                	li	a6,10
1c01b35c:	00e50863          	beq	a0,a4,1c01b36c <_vsnprintf+0x290>
1c01b360:	06400713          	li	a4,100
1c01b364:	00e50463          	beq	a0,a4,1c01b36c <_vsnprintf+0x290>
1c01b368:	c227b7b3          	p.bclr	a5,a5,1,2
1c01b36c:	4007f713          	andi	a4,a5,1024
1c01b370:	c319                	beqz	a4,1c01b376 <_vsnprintf+0x29a>
1c01b372:	c007b7b3          	p.bclr	a5,a5,0,0
1c01b376:	06900693          	li	a3,105
1c01b37a:	2007f713          	andi	a4,a5,512
1c01b37e:	00d50663          	beq	a0,a3,1c01b38a <_vsnprintf+0x2ae>
1c01b382:	06400693          	li	a3,100
1c01b386:	0ed51663          	bne	a0,a3,1c01b472 <_vsnprintf+0x396>
1c01b38a:	c34d                	beqz	a4,1c01b42c <_vsnprintf+0x350>
1c01b38c:	0c1d                	addi	s8,s8,7
1c01b38e:	c40c3c33          	p.bclr	s8,s8,2,0
1c01b392:	004c2683          	lw	a3,4(s8)
1c01b396:	000c2603          	lw	a2,0(s8)
1c01b39a:	88c2                	mv	a7,a6
1c01b39c:	01f6d713          	srli	a4,a3,0x1f
1c01b3a0:	c63e                	sw	a5,12(sp)
1c01b3a2:	0ff77813          	andi	a6,a4,255
1c01b3a6:	008c0413          	addi	s0,s8,8
1c01b3aa:	c466                	sw	s9,8(sp)
1c01b3ac:	c26a                	sw	s10,4(sp)
1c01b3ae:	c002                	sw	zero,0(sp)
1c01b3b0:	24060733          	p.abs.d	a4,a2
1c01b3b4:	86ca                	mv	a3,s2
1c01b3b6:	866e                	mv	a2,s11
1c01b3b8:	85d2                	mv	a1,s4
1c01b3ba:	8526                	mv	a0,s1
1c01b3bc:	ca4ff0ef          	jal	ra,1c01a860 <_ntoa_long_long>
1c01b3c0:	8daa                	mv	s11,a0
1c01b3c2:	8c22                	mv	s8,s0
1c01b3c4:	bb81                	j	1c01b114 <_vsnprintf+0x38>
1c01b3c6:	06300713          	li	a4,99
1c01b3ca:	18e50063          	beq	a0,a4,1c01b54a <_vsnprintf+0x46e>
1c01b3ce:	06400713          	li	a4,100
1c01b3d2:	bfb1                	j	1c01b32e <_vsnprintf+0x252>
1c01b3d4:	06f00713          	li	a4,111
1c01b3d8:	f4e50de3          	beq	a0,a4,1c01b332 <_vsnprintf+0x256>
1c01b3dc:	00a76b63          	bltu	a4,a0,1c01b3f2 <_vsnprintf+0x316>
1c01b3e0:	06700713          	li	a4,103
1c01b3e4:	10e50e63          	beq	a0,a4,1c01b500 <_vsnprintf+0x424>
1c01b3e8:	0ee56263          	bltu	a0,a4,1c01b4cc <_vsnprintf+0x3f0>
1c01b3ec:	06900713          	li	a4,105
1c01b3f0:	bf3d                	j	1c01b32e <_vsnprintf+0x252>
1c01b3f2:	07300713          	li	a4,115
1c01b3f6:	1ae50f63          	beq	a0,a4,1c01b5b4 <_vsnprintf+0x4d8>
1c01b3fa:	02a76263          	bltu	a4,a0,1c01b41e <_vsnprintf+0x342>
1c01b3fe:	07000713          	li	a4,112
1c01b402:	d4e51ae3          	bne	a0,a4,1c01b156 <_vsnprintf+0x7a>
1c01b406:	0217e793          	ori	a5,a5,33
1c01b40a:	000c2703          	lw	a4,0(s8)
1c01b40e:	c23e                	sw	a5,4(sp)
1c01b410:	47a1                	li	a5,8
1c01b412:	004c0413          	addi	s0,s8,4
1c01b416:	c03e                	sw	a5,0(sp)
1c01b418:	88ea                	mv	a7,s10
1c01b41a:	4841                	li	a6,16
1c01b41c:	a841                	j	1c01b4ac <_vsnprintf+0x3d0>
1c01b41e:	07500713          	li	a4,117
1c01b422:	f0e508e3          	beq	a0,a4,1c01b332 <_vsnprintf+0x256>
1c01b426:	07800713          	li	a4,120
1c01b42a:	b711                	j	1c01b32e <_vsnprintf+0x252>
1c01b42c:	1007f713          	andi	a4,a5,256
1c01b430:	004c0413          	addi	s0,s8,4
1c01b434:	c30d                	beqz	a4,1c01b456 <_vsnprintf+0x37a>
1c01b436:	000c2703          	lw	a4,0(s8)
1c01b43a:	c23e                	sw	a5,4(sp)
1c01b43c:	c066                	sw	s9,0(sp)
1c01b43e:	01f75793          	srli	a5,a4,0x1f
1c01b442:	88ea                	mv	a7,s10
1c01b444:	04070733          	p.avg	a4,a4,zero
1c01b448:	86ca                	mv	a3,s2
1c01b44a:	866e                	mv	a2,s11
1c01b44c:	85d2                	mv	a1,s4
1c01b44e:	8526                	mv	a0,s1
1c01b450:	b98ff0ef          	jal	ra,1c01a7e8 <_ntoa_long>
1c01b454:	b7b5                	j	1c01b3c0 <_vsnprintf+0x2e4>
1c01b456:	0407f693          	andi	a3,a5,64
1c01b45a:	000c2703          	lw	a4,0(s8)
1c01b45e:	c681                	beqz	a3,1c01b466 <_vsnprintf+0x38a>
1c01b460:	ee873733          	p.bclr	a4,a4,23,8
1c01b464:	bfd9                	j	1c01b43a <_vsnprintf+0x35e>
1c01b466:	0807f693          	andi	a3,a5,128
1c01b46a:	dae1                	beqz	a3,1c01b43a <_vsnprintf+0x35e>
1c01b46c:	10074733          	p.exths	a4,a4
1c01b470:	b7e9                	j	1c01b43a <_vsnprintf+0x35e>
1c01b472:	c31d                	beqz	a4,1c01b498 <_vsnprintf+0x3bc>
1c01b474:	0c1d                	addi	s8,s8,7
1c01b476:	c40c3c33          	p.bclr	s8,s8,2,0
1c01b47a:	000c2603          	lw	a2,0(s8)
1c01b47e:	004c2683          	lw	a3,4(s8)
1c01b482:	88c2                	mv	a7,a6
1c01b484:	c63e                	sw	a5,12(sp)
1c01b486:	008c0413          	addi	s0,s8,8
1c01b48a:	c466                	sw	s9,8(sp)
1c01b48c:	c26a                	sw	s10,4(sp)
1c01b48e:	c002                	sw	zero,0(sp)
1c01b490:	4801                	li	a6,0
1c01b492:	22061713          	addi.d	a4,a2,0
1c01b496:	bf39                	j	1c01b3b4 <_vsnprintf+0x2d8>
1c01b498:	1007f713          	andi	a4,a5,256
1c01b49c:	004c0413          	addi	s0,s8,4
1c01b4a0:	cb01                	beqz	a4,1c01b4b0 <_vsnprintf+0x3d4>
1c01b4a2:	000c2703          	lw	a4,0(s8)
1c01b4a6:	c23e                	sw	a5,4(sp)
1c01b4a8:	c066                	sw	s9,0(sp)
1c01b4aa:	88ea                	mv	a7,s10
1c01b4ac:	4781                	li	a5,0
1c01b4ae:	bf69                	j	1c01b448 <_vsnprintf+0x36c>
1c01b4b0:	0407f693          	andi	a3,a5,64
1c01b4b4:	000c2703          	lw	a4,0(s8)
1c01b4b8:	c681                	beqz	a3,1c01b4c0 <_vsnprintf+0x3e4>
1c01b4ba:	ee873733          	p.bclr	a4,a4,23,8
1c01b4be:	b7e5                	j	1c01b4a6 <_vsnprintf+0x3ca>
1c01b4c0:	0807f693          	andi	a3,a5,128
1c01b4c4:	d2ed                	beqz	a3,1c01b4a6 <_vsnprintf+0x3ca>
1c01b4c6:	01677733          	and	a4,a4,s6
1c01b4ca:	bff1                	j	1c01b4a6 <_vsnprintf+0x3ca>
1c01b4cc:	04600713          	li	a4,70
1c01b4d0:	00e51463          	bne	a0,a4,1c01b4d8 <_vsnprintf+0x3fc>
1c01b4d4:	c057c7b3          	p.bset	a5,a5,0,5
1c01b4d8:	0c1d                	addi	s8,s8,7
1c01b4da:	c40c3733          	p.bclr	a4,s8,2,0
1c01b4de:	4310                	lw	a2,0(a4)
1c01b4e0:	4354                	lw	a3,4(a4)
1c01b4e2:	00870c13          	addi	s8,a4,8
1c01b4e6:	c03e                	sw	a5,0(sp)
1c01b4e8:	88e6                	mv	a7,s9
1c01b4ea:	22061713          	addi.d	a4,a2,0
1c01b4ee:	886a                	mv	a6,s10
1c01b4f0:	86ca                	mv	a3,s2
1c01b4f2:	866e                	mv	a2,s11
1c01b4f4:	85d2                	mv	a1,s4
1c01b4f6:	8526                	mv	a0,s1
1c01b4f8:	c5aff0ef          	jal	ra,1c01a952 <_ftoa>
1c01b4fc:	8daa                	mv	s11,a0
1c01b4fe:	b919                	j	1c01b114 <_vsnprintf+0x38>
1c01b500:	0df57713          	andi	a4,a0,223
1c01b504:	04700693          	li	a3,71
1c01b508:	00d71863          	bne	a4,a3,1c01b518 <_vsnprintf+0x43c>
1c01b50c:	6705                	lui	a4,0x1
1c01b50e:	80070713          	addi	a4,a4,-2048 # 800 <__STACK_IRQ_SIZE>
1c01b512:	8fd9                	or	a5,a5,a4
1c01b514:	0fd57513          	andi	a0,a0,253
1c01b518:	04500713          	li	a4,69
1c01b51c:	00e51463          	bne	a0,a4,1c01b524 <_vsnprintf+0x448>
1c01b520:	c057c7b3          	p.bset	a5,a5,0,5
1c01b524:	0c1d                	addi	s8,s8,7
1c01b526:	c40c3733          	p.bclr	a4,s8,2,0
1c01b52a:	4310                	lw	a2,0(a4)
1c01b52c:	4354                	lw	a3,4(a4)
1c01b52e:	00870c13          	addi	s8,a4,8
1c01b532:	c03e                	sw	a5,0(sp)
1c01b534:	88e6                	mv	a7,s9
1c01b536:	22061713          	addi.d	a4,a2,0
1c01b53a:	886a                	mv	a6,s10
1c01b53c:	86ca                	mv	a3,s2
1c01b53e:	866e                	mv	a2,s11
1c01b540:	85d2                	mv	a1,s4
1c01b542:	8526                	mv	a0,s1
1c01b544:	f8aff0ef          	jal	ra,1c01acce <_etoa>
1c01b548:	bf55                	j	1c01b4fc <_vsnprintf+0x420>
1c01b54a:	0027f413          	andi	s0,a5,2
1c01b54e:	c439                	beqz	s0,1c01b59c <_vsnprintf+0x4c0>
1c01b550:	8aee                	mv	s5,s11
1c01b552:	4785                	li	a5,1
1c01b554:	000c4503          	lbu	a0,0(s8)
1c01b558:	86ca                	mv	a3,s2
1c01b55a:	cc3e                	sw	a5,24(sp)
1c01b55c:	8656                	mv	a2,s5
1c01b55e:	85d2                	mv	a1,s4
1c01b560:	9482                	jalr	s1
1c01b562:	004c0d13          	addi	s10,s8,4
1c01b566:	001a8d93          	addi	s11,s5,1
1c01b56a:	47e2                	lw	a5,24(sp)
1c01b56c:	c801                	beqz	s0,1c01b57c <_vsnprintf+0x4a0>
1c01b56e:	17fd                	addi	a5,a5,-1
1c01b570:	41578433          	sub	s0,a5,s5
1c01b574:	008d87b3          	add	a5,s11,s0
1c01b578:	0397e463          	bltu	a5,s9,1c01b5a0 <_vsnprintf+0x4c4>
1c01b57c:	8c6a                	mv	s8,s10
1c01b57e:	be59                	j	1c01b114 <_vsnprintf+0x38>
1c01b580:	86ca                	mv	a3,s2
1c01b582:	8656                	mv	a2,s5
1c01b584:	85d2                	mv	a1,s4
1c01b586:	02000513          	li	a0,32
1c01b58a:	9482                	jalr	s1
1c01b58c:	01bd0ab3          	add	s5,s10,s11
1c01b590:	002d0793          	addi	a5,s10,2
1c01b594:	0d05                	addi	s10,s10,1
1c01b596:	ff9d65e3          	bltu	s10,s9,1c01b580 <_vsnprintf+0x4a4>
1c01b59a:	bf6d                	j	1c01b554 <_vsnprintf+0x478>
1c01b59c:	4d01                	li	s10,0
1c01b59e:	b7fd                	j	1c01b58c <_vsnprintf+0x4b0>
1c01b5a0:	866e                	mv	a2,s11
1c01b5a2:	001d8c13          	addi	s8,s11,1
1c01b5a6:	86ca                	mv	a3,s2
1c01b5a8:	85d2                	mv	a1,s4
1c01b5aa:	02000513          	li	a0,32
1c01b5ae:	9482                	jalr	s1
1c01b5b0:	8de2                	mv	s11,s8
1c01b5b2:	b7c9                	j	1c01b574 <_vsnprintf+0x498>
1c01b5b4:	004c0893          	addi	a7,s8,4
1c01b5b8:	ce46                	sw	a7,28(sp)
1c01b5ba:	000c2803          	lw	a6,0(s8)
1c01b5be:	577d                	li	a4,-1
1c01b5c0:	000d0363          	beqz	s10,1c01b5c6 <_vsnprintf+0x4ea>
1c01b5c4:	876a                	mv	a4,s10
1c01b5c6:	00e806b3          	add	a3,a6,a4
1c01b5ca:	8742                	mv	a4,a6
1c01b5cc:	00074603          	lbu	a2,0(a4)
1c01b5d0:	c219                	beqz	a2,1c01b5d6 <_vsnprintf+0x4fa>
1c01b5d2:	04e69063          	bne	a3,a4,1c01b612 <_vsnprintf+0x536>
1c01b5d6:	4007fc13          	andi	s8,a5,1024
1c01b5da:	41070733          	sub	a4,a4,a6
1c01b5de:	000c0463          	beqz	s8,1c01b5e6 <_vsnprintf+0x50a>
1c01b5e2:	05a75733          	p.minu	a4,a4,s10
1c01b5e6:	0027f413          	andi	s0,a5,2
1c01b5ea:	c429                	beqz	s0,1c01b634 <_vsnprintf+0x558>
1c01b5ec:	8aba                	mv	s5,a4
1c01b5ee:	866e                	mv	a2,s11
1c01b5f0:	41060db3          	sub	s11,a2,a6
1c01b5f4:	010d8633          	add	a2,s11,a6
1c01b5f8:	0018450b          	p.lbu	a0,1(a6!)
1c01b5fc:	c519                	beqz	a0,1c01b60a <_vsnprintf+0x52e>
1c01b5fe:	040c0063          	beqz	s8,1c01b63e <_vsnprintf+0x562>
1c01b602:	fffd0793          	addi	a5,s10,-1
1c01b606:	020d1b63          	bnez	s10,1c01b63c <_vsnprintf+0x560>
1c01b60a:	8db2                	mv	s11,a2
1c01b60c:	ec29                	bnez	s0,1c01b666 <_vsnprintf+0x58a>
1c01b60e:	4c72                	lw	s8,28(sp)
1c01b610:	b611                	j	1c01b114 <_vsnprintf+0x38>
1c01b612:	0705                	addi	a4,a4,1
1c01b614:	bf65                	j	1c01b5cc <_vsnprintf+0x4f0>
1c01b616:	cc42                	sw	a6,24(sp)
1c01b618:	86ca                	mv	a3,s2
1c01b61a:	85d2                	mv	a1,s4
1c01b61c:	02000513          	li	a0,32
1c01b620:	9482                	jalr	s1
1c01b622:	4862                	lw	a6,24(sp)
1c01b624:	87d6                	mv	a5,s5
1c01b626:	00fd8633          	add	a2,s11,a5
1c01b62a:	00178a93          	addi	s5,a5,1 # 80000001 <pulp__FC+0x80000002>
1c01b62e:	ff97e4e3          	bltu	a5,s9,1c01b616 <_vsnprintf+0x53a>
1c01b632:	bf7d                	j	1c01b5f0 <_vsnprintf+0x514>
1c01b634:	87ba                	mv	a5,a4
1c01b636:	40ed8db3          	sub	s11,s11,a4
1c01b63a:	b7f5                	j	1c01b626 <_vsnprintf+0x54a>
1c01b63c:	8d3e                	mv	s10,a5
1c01b63e:	cc42                	sw	a6,24(sp)
1c01b640:	86ca                	mv	a3,s2
1c01b642:	85d2                	mv	a1,s4
1c01b644:	9482                	jalr	s1
1c01b646:	4862                	lw	a6,24(sp)
1c01b648:	b775                	j	1c01b5f4 <_vsnprintf+0x518>
1c01b64a:	866e                	mv	a2,s11
1c01b64c:	001d8c13          	addi	s8,s11,1
1c01b650:	86ca                	mv	a3,s2
1c01b652:	85d2                	mv	a1,s4
1c01b654:	02000513          	li	a0,32
1c01b658:	9482                	jalr	s1
1c01b65a:	8de2                	mv	s11,s8
1c01b65c:	01b407b3          	add	a5,s0,s11
1c01b660:	ff97e5e3          	bltu	a5,s9,1c01b64a <_vsnprintf+0x56e>
1c01b664:	b76d                	j	1c01b60e <_vsnprintf+0x532>
1c01b666:	40ca8433          	sub	s0,s5,a2
1c01b66a:	bfcd                	j	1c01b65c <_vsnprintf+0x580>
1c01b66c:	c057c7b3          	p.bset	a5,a5,0,5
1c01b670:	4841                	li	a6,16
1c01b672:	b9dd                	j	1c01b368 <_vsnprintf+0x28c>
1c01b674:	4821                	li	a6,8
1c01b676:	b1ed                	j	1c01b360 <_vsnprintf+0x284>
1c01b678:	4809                	li	a6,2
1c01b67a:	b1dd                	j	1c01b360 <_vsnprintf+0x284>

1c01b67c <printf_>:
1c01b67c:	7139                	addi	sp,sp,-64
1c01b67e:	d636                	sw	a3,44(sp)
1c01b680:	86aa                	mv	a3,a0
1c01b682:	1c01b537          	lui	a0,0x1c01b
1c01b686:	d22e                	sw	a1,36(sp)
1c01b688:	d432                	sw	a2,40(sp)
1c01b68a:	d83a                	sw	a4,48(sp)
1c01b68c:	002c                	addi	a1,sp,8
1c01b68e:	1058                	addi	a4,sp,36
1c01b690:	567d                	li	a2,-1
1c01b692:	94a50513          	addi	a0,a0,-1718 # 1c01a94a <_out_char>
1c01b696:	ce06                	sw	ra,28(sp)
1c01b698:	da3e                	sw	a5,52(sp)
1c01b69a:	dc42                	sw	a6,56(sp)
1c01b69c:	de46                	sw	a7,60(sp)
1c01b69e:	c63a                	sw	a4,12(sp)
1c01b6a0:	a3dff0ef          	jal	ra,1c01b0dc <_vsnprintf>
1c01b6a4:	40f2                	lw	ra,28(sp)
1c01b6a6:	6121                	addi	sp,sp,64
1c01b6a8:	8082                	ret
1c01b6aa:	0000                	unimp
1c01b6ac:	0000                	unimp
	...

Disassembly of section .cluster.text:

1c01b6b0 <pi_cl_entry>:
1c01b6b0:	f1402573          	csrr	a0,mhartid
1c01b6b4:	4581                	li	a1,0
1c01b6b6:	000f02b7          	lui	t0,0xf0
1c01b6ba:	00204337          	lui	t1,0x204
1c01b6be:	00532023          	sw	t0,0(t1) # 204000 <__l1_heapsram_size+0x1e4070>
1c01b6c2:	43a1                	li	t2,8
1c01b6c4:	00750463          	beq	a0,t2,1c01b6cc <pi_cl_cc_entry>
1c01b6c8:	2a00006f          	j	1c01b968 <pi_cl_pe_entry>

1c01b6cc <pi_cl_cc_entry>:
1c01b6cc:	ffff5397          	auipc	t2,0xffff5
1c01b6d0:	a3438393          	addi	t2,t2,-1484 # 1c010100 <__irq_cluster_vector_base_m__>
1c01b6d4:	30539073          	csrw	mtvec,t2
1c01b6d8:	e3fe5417          	auipc	s0,0xe3fe5
1c01b6dc:	92c40413          	addi	s0,s0,-1748 # 4 <__pi_cluster_pool>
1c01b6e0:	002049b7          	lui	s3,0x204
1c01b6e4:	4a09                	li	s4,2
1c01b6e6:	00002b97          	auipc	s7,0x2
1c01b6ea:	116b8b93          	addi	s7,s7,278 # 1c01d7fc <pi_cluster>
1c01b6ee:	00000097          	auipc	ra,0x0
1c01b6f2:	0c808093          	addi	ra,ra,200 # 1c01b7b6 <pi_cl_cc_wait_task>
1c01b6f6:	02c00393          	li	t2,44
1c01b6fa:	ffff9917          	auipc	s2,0xffff9
1c01b6fe:	18490913          	addi	s2,s2,388 # 1c01487e <pi_cl_send_task_to_fc>
1c01b702:	02b383b3          	mul	t2,t2,a1
1c01b706:	9b9e                	add	s7,s7,t2
1c01b708:	00cba103          	lw	sp,12(s7)
1c01b70c:	0be1                	addi	s7,s7,24
1c01b70e:	1a109cb7          	lui	s9,0x1a109
1c01b712:	010c8c93          	addi	s9,s9,16 # 1a109010 <__l1_heapsram_end+0xa0e9010>
1c01b716:	002044b7          	lui	s1,0x204
1c01b71a:	4f048493          	addi	s1,s1,1264 # 2044f0 <__l1_heapsram_size+0x1e4560>
1c01b71e:	20000293          	li	t0,512
1c01b722:	0059aa23          	sw	t0,20(s3) # 204014 <__l1_heapsram_size+0x1e4084>
1c01b726:	60000293          	li	t0,1536
1c01b72a:	01000337          	lui	t1,0x1000
1c01b72e:	ff030313          	addi	t1,t1,-16 # fffff0 <__l1_heapsram_size+0xfe0060>
1c01b732:	0069e2a3          	p.sw	t1,t0(s3)
1c01b736:	61000293          	li	t0,1552
1c01b73a:	00010337          	lui	t1,0x10
1c01b73e:	fff30313          	addi	t1,t1,-1 # ffff <__heapfcram_size+0xdbf>
1c01b742:	0069e2a3          	p.sw	t1,t0(s3)
1c01b746:	63000293          	li	t0,1584
1c01b74a:	0ff00313          	li	t1,255
1c01b74e:	0069e2a3          	p.sw	t1,t0(s3)
1c01b752:	62000293          	li	t0,1568
1c01b756:	0fc00313          	li	t1,252
1c01b75a:	0069e2a3          	p.sw	t1,t0(s3)
1c01b75e:	00000d17          	auipc	s10,0x0
1c01b762:	120d0d13          	addi	s10,s10,288 # 1c01b87e <pi_cl_pe_set_stack>
1c01b766:	001d6d13          	ori	s10,s10,1
1c01b76a:	30045073          	csrwi	mstatus,8
1c01b76e:	4c41                	li	s8,16
1c01b770:	002043b7          	lui	t2,0x204
1c01b774:	60038393          	addi	t2,t2,1536 # 204600 <__l1_heapsram_size+0x1e4670>
1c01b778:	00008337          	lui	t1,0x8
1c01b77c:	fff30313          	addi	t1,t1,-1 # 7fff <__STACK_IRQ_SIZE+0x77ff>
1c01b780:	0063a023          	sw	t1,0(t2)
1c01b784:	a80d                	j	1c01b7b6 <pi_cl_cc_wait_task>

1c01b786 <pi_cl_cc_handle_tasklet>:
1c01b786:	c1444f33          	p.bset	t5,s0,0,20

1c01b78a <pi_cl_cc_handle_tasklet_lock>:
1c01b78a:	008f2383          	lw	t2,8(t5)
1c01b78e:	fff3aee3          	p.beqimm	t2,-1,1c01b78a <pi_cl_cc_handle_tasklet_lock>
1c01b792:	01c3ae83          	lw	t4,28(t2)
1c01b796:	01d42423          	sw	t4,8(s0)

1c01b79a <pi_cl_cc_exec_tasklet>:
1c01b79a:	0003a283          	lw	t0,0(t2)
1c01b79e:	0043a503          	lw	a0,4(t2)
1c01b7a2:	8282                	jr	t0

1c01b7a4 <pi_cl_cc_handle_callback>:
1c01b7a4:	008eaf03          	lw	t5,8(t4)
1c01b7a8:	000ea283          	lw	t0,0(t4)
1c01b7ac:	004ea503          	lw	a0,4(t4)
1c01b7b0:	03e42023          	sw	t5,32(s0)
1c01b7b4:	8282                	jr	t0

1c01b7b6 <pi_cl_cc_wait_task>:
1c01b7b6:	00842383          	lw	t2,8(s0)
1c01b7ba:	00442303          	lw	t1,4(s0)
1c01b7be:	00042b03          	lw	s6,0(s0)
1c01b7c2:	02c42e03          	lw	t3,44(s0)
1c01b7c6:	fc0390e3          	bnez	t2,1c01b786 <pi_cl_cc_handle_tasklet>
1c01b7ca:	08031663          	bnez	t1,1c01b856 <pi_cl_cc_handle_prio_task>
1c01b7ce:	02042e83          	lw	t4,32(s0)
1c01b7d2:	03842303          	lw	t1,56(s0)
1c01b7d6:	060e1a63          	bnez	t3,1c01b84a <pi_cl_cc_handle_yield_task>
1c01b7da:	fc0e95e3          	bnez	t4,1c01b7a4 <pi_cl_cc_handle_callback>
1c01b7de:	000b1563          	bnez	s6,1c01b7e8 <pi_cl_cc_update_task_loop>
1c01b7e2:	08030763          	beqz	t1,1c01b870 <pi_cl_cc_sleep>
1c01b7e6:	8b1a                	mv	s6,t1

1c01b7e8 <pi_cl_cc_update_task_loop>:
1c01b7e8:	01cb2e83          	lw	t4,28(s6) # 1001c <__heapfcram_size+0xddc>
1c01b7ec:	02040aa3          	sb	zero,53(s0)
1c01b7f0:	01d42023          	sw	t4,0(s0)
1c01b7f4:	01cb2f03          	lw	t5,28(s6)
1c01b7f8:	ffee98e3          	bne	t4,t5,1c01b7e8 <pi_cl_cc_update_task_loop>

1c01b7fc <pi_cl_cc_exec_task>:
1c01b7fc:	00000097          	auipc	ra,0x0
1c01b800:	09408093          	addi	ra,ra,148 # 1c01b890 <pi_cl_cc_wait_task_end>
1c01b804:	03642823          	sw	s6,48(s0)
1c01b808:	000b2283          	lw	t0,0(s6)
1c01b80c:	004b2503          	lw	a0,4(s6)
1c01b810:	010b2f83          	lw	t6,16(s6)
1c01b814:	020b2f03          	lw	t5,32(s6)
1c01b818:	05f02423          	sw	t6,72(zero) # 48 <__pi_cluster_nb_active_pe>
1c01b81c:	21e9a023          	sw	t5,512(s3)
1c01b820:	21e9a623          	sw	t5,524(s3)
1c01b824:	09e9a223          	sw	t5,132(s3)
1c01b828:	100f6f93          	ori	t6,t5,256
1c01b82c:	23f9a023          	sw	t6,544(s3)
1c01b830:	23f9a623          	sw	t6,556(s3)
1c01b834:	008b2e03          	lw	t3,8(s6)
1c01b838:	00cb2383          	lw	t2,12(s6)
1c01b83c:	09a9a023          	sw	s10,128(s3)
1c01b840:	0879a023          	sw	t2,128(s3)
1c01b844:	09c9a023          	sw	t3,128(s3)
1c01b848:	8282                	jr	t0

1c01b84a <pi_cl_cc_handle_yield_task>:
1c01b84a:	02042623          	sw	zero,44(s0)
1c01b84e:	02040aa3          	sb	zero,53(s0)
1c01b852:	8b72                	mv	s6,t3
1c01b854:	b765                	j	1c01b7fc <pi_cl_cc_exec_task>

1c01b856 <pi_cl_cc_handle_prio_task>:
1c01b856:	8b1a                	mv	s6,t1
1c01b858:	4385                	li	t2,1
1c01b85a:	01cb2e83          	lw	t4,28(s6)
1c01b85e:	02740aa3          	sb	t2,53(s0)
1c01b862:	01d42223          	sw	t4,4(s0)
1c01b866:	01cb2f03          	lw	t5,28(s6)
1c01b86a:	ffee96e3          	bne	t4,t5,1c01b856 <pi_cl_cc_handle_prio_task>
1c01b86e:	b779                	j	1c01b7fc <pi_cl_cc_exec_task>

1c01b870 <pi_cl_cc_sleep>:
1c01b870:	0149a423          	sw	s4,8(s3)
1c01b874:	03c9e003          	p.elw	zero,60(s3)
1c01b878:	0149a223          	sw	s4,4(s3)
1c01b87c:	8082                	ret

1c01b87e <pi_cl_pe_set_stack>:
1c01b87e:	08092283          	lw	t0,128(s2)
1c01b882:	00198f13          	addi	t5,s3,1
1c01b886:	02af0eb3          	mul	t4,t5,a0
1c01b88a:	005e8133          	add	sp,t4,t0
1c01b88e:	8082                	ret

1c01b890 <pi_cl_cc_wait_task_end>:
1c01b890:	03440303          	lb	t1,52(s0)
1c01b894:	03640383          	lb	t2,54(s0)
1c01b898:	04031263          	bnez	t1,1c01b8dc <pi_cl_cc_wait_task_end_yield>
1c01b89c:	02039d63          	bnez	t2,1c01b8d6 <pi_cl_cc_wait_task_end_background>
1c01b8a0:	00842383          	lw	t2,8(s0)
1c01b8a4:	4808                	lw	a0,16(s0)
1c01b8a6:	02042e83          	lw	t4,32(s0)
1c01b8aa:	ec039ee3          	bnez	t2,1c01b786 <pi_cl_cc_handle_tasklet>
1c01b8ae:	ee0e9be3          	bnez	t4,1c01b7a4 <pi_cl_cc_handle_callback>
1c01b8b2:	02050a63          	beqz	a0,1c01b8e6 <pi_cl_cc_check_task_end>

1c01b8b6 <pi_cl_cc_handle_workitem_end>:
1c01b8b6:	04002283          	lw	t0,64(zero) # 40 <__pi_cluster_pending_task>
1c01b8ba:	12fd                	addi	t0,t0,-1
1c01b8bc:	04502023          	sw	t0,64(zero) # 40 <__pi_cluster_pending_task>
1c01b8c0:	0c09e283          	p.elw	t0,192(s3)
1c01b8c4:	00452e83          	lw	t4,4(a0)
1c01b8c8:	00852f03          	lw	t5,8(a0)
1c01b8cc:	01d42823          	sw	t4,16(s0)
1c01b8d0:	0c09a023          	sw	zero,192(s3)
1c01b8d4:	8f02                	jr	t5

1c01b8d6 <pi_cl_cc_wait_task_end_background>:
1c01b8d6:	02040b23          	sb	zero,54(s0)
1c01b8da:	bdf1                	j	1c01b7b6 <pi_cl_cc_wait_task>

1c01b8dc <pi_cl_cc_wait_task_end_yield>:
1c01b8dc:	02040a23          	sb	zero,52(s0)
1c01b8e0:	03642623          	sw	s6,44(s0)
1c01b8e4:	bdc9                	j	1c01b7b6 <pi_cl_cc_wait_task>

1c01b8e6 <pi_cl_cc_check_task_end>:
1c01b8e6:	04002283          	lw	t0,64(zero) # 40 <__pi_cluster_pending_task>
1c01b8ea:	f80293e3          	bnez	t0,1c01b870 <pi_cl_cc_sleep>
1c01b8ee:	014b2503          	lw	a0,20(s6)
1c01b8f2:	00000097          	auipc	ra,0x0
1c01b8f6:	ec408093          	addi	ra,ra,-316 # 1c01b7b6 <pi_cl_cc_wait_task>
1c01b8fa:	8902                	jr	s2

1c01b8fc <pi_cl_cc_check_task_end_yield>:
1c01b8fc:	02c42b03          	lw	s6,44(s0)
1c01b900:	02042623          	sw	zero,44(s0)
1c01b904:	bde5                	j	1c01b7fc <pi_cl_cc_exec_task>

1c01b906 <__pi_cl_irq_handler>:
1c01b906:	7175                	addi	sp,sp,-144
1c01b908:	c006                	sw	ra,0(sp)
1c01b90a:	c22a                	sw	a0,4(sp)
1c01b90c:	c42e                	sw	a1,8(sp)
1c01b90e:	c632                	sw	a2,12(sp)
1c01b910:	c836                	sw	a3,16(sp)
1c01b912:	ca3a                	sw	a4,20(sp)
1c01b914:	cc3e                	sw	a5,24(sp)
1c01b916:	ce42                	sw	a6,28(sp)
1c01b918:	d046                	sw	a7,32(sp)
1c01b91a:	d216                	sw	t0,36(sp)
1c01b91c:	d41a                	sw	t1,40(sp)
1c01b91e:	d61e                	sw	t2,44(sp)
1c01b920:	d872                	sw	t3,48(sp)
1c01b922:	da76                	sw	t4,52(sp)
1c01b924:	dc7a                	sw	t5,56(sp)
1c01b926:	de7e                	sw	t6,60(sp)
1c01b928:	342022f3          	csrr	t0,mcause
1c01b92c:	028a                	slli	t0,t0,0x2
1c01b92e:	00001317          	auipc	t1,0x1
1c01b932:	83630313          	addi	t1,t1,-1994 # 1c01c164 <__pi_cl_irq_handlers>
1c01b936:	20537383          	p.lw	t2,t0(t1)
1c01b93a:	00038463          	beqz	t2,1c01b942 <__pi_cl_irq_handler_end>
1c01b93e:	000380e7          	jalr	t2

1c01b942 <__pi_cl_irq_handler_end>:
1c01b942:	5ff2                	lw	t6,60(sp)
1c01b944:	5f62                	lw	t5,56(sp)
1c01b946:	5ed2                	lw	t4,52(sp)
1c01b948:	5e42                	lw	t3,48(sp)
1c01b94a:	53b2                	lw	t2,44(sp)
1c01b94c:	5322                	lw	t1,40(sp)
1c01b94e:	5292                	lw	t0,36(sp)
1c01b950:	5882                	lw	a7,32(sp)
1c01b952:	4872                	lw	a6,28(sp)
1c01b954:	47e2                	lw	a5,24(sp)
1c01b956:	4752                	lw	a4,20(sp)
1c01b958:	46c2                	lw	a3,16(sp)
1c01b95a:	4632                	lw	a2,12(sp)
1c01b95c:	45a2                	lw	a1,8(sp)
1c01b95e:	4512                	lw	a0,4(sp)
1c01b960:	4082                	lw	ra,0(sp)
1c01b962:	6149                	addi	sp,sp,144
1c01b964:	30200073          	mret

1c01b968 <pi_cl_pe_entry>:
1c01b968:	00000a17          	auipc	s4,0x0
1c01b96c:	03ea0a13          	addi	s4,s4,62 # 1c01b9a6 <pi_cl_pe_fork_return>
1c01b970:	00000a97          	auipc	s5,0x0
1c01b974:	03aa8a93          	addi	s5,s5,58 # 1c01b9aa <pi_cl_pe_wait_for_dispatch>
1c01b978:	00204937          	lui	s2,0x204
1c01b97c:	f14029f3          	csrr	s3,mhartid
1c01b980:	e3fe4417          	auipc	s0,0xe3fe4
1c01b984:	68440413          	addi	s0,s0,1668 # 4 <__pi_cluster_pool>
1c01b988:	002044b7          	lui	s1,0x204
1c01b98c:	4f048493          	addi	s1,s1,1264 # 2044f0 <__l1_heapsram_size+0x1e4560>
1c01b990:	00204cb7          	lui	s9,0x204
1c01b994:	104c8c93          	addi	s9,s9,260 # 204104 <__l1_heapsram_size+0x1e4174>
1c01b998:	00000d17          	auipc	s10,0x0
1c01b99c:	ee6d0d13          	addi	s10,s10,-282 # 1c01b87e <pi_cl_pe_set_stack>
1c01b9a0:	001d6d13          	ori	s10,s10,1
1c01b9a4:	a019                	j	1c01b9aa <pi_cl_pe_wait_for_dispatch>

1c01b9a6 <pi_cl_pe_fork_return>:
1c01b9a6:	23c96283          	p.elw	t0,572(s2) # 20423c <__l1_heapsram_size+0x1e42ac>

1c01b9aa <pi_cl_pe_wait_for_dispatch>:
1c01b9aa:	08096283          	p.elw	t0,128(s2)
1c01b9ae:	08092503          	lw	a0,128(s2)
1c01b9b2:	0012f313          	andi	t1,t0,1
1c01b9b6:	00031563          	bnez	t1,1c01b9c0 <pi_cl_pe_other_entry>

1c01b9ba <pi_cl_pe_fork_entry>:
1c01b9ba:	000a00b3          	add	ra,s4,zero
1c01b9be:	8282                	jr	t0

1c01b9c0 <pi_cl_pe_other_entry>:
1c01b9c0:	000a80b3          	add	ra,s5,zero
1c01b9c4:	8282                	jr	t0
