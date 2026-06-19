In archive /app/TrainDeeploy/DeeployTest/TEST_GAP9/build_master/lib/libdeeploybasic.a:

BatchNorm_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.BatchNorm_fp32 000000bc  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .rodata.BatchNorm_fp32.cst4 00000004  00000000  00000000  000000f0  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  5 .comment      0000001b  00000000  00000000  000000f4  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  0000010f  2**0
                  CONTENTS, READONLY

Disassembly of section .text.BatchNorm_fp32:

00000000 <BatchNorm_fp32>:
   0:	715d                	addi	sp,sp,-80
   2:	de4e                	sw	s3,60(sp)
   4:	49c6                	lw	s3,80(sp)
   6:	c0ca                	sw	s2,64(sp)
   8:	dc52                	sw	s4,56(sp)
   a:	00299913          	slli	s2,s3,0x2
   e:	03288a33          	mul	s4,a7,s2
  12:	c2a6                	sw	s1,68(sp)
  14:	84aa                	mv	s1,a0
  16:	00000537          	lui	a0,0x0
  1a:	da56                	sw	s5,52(sp)
  1c:	00052a83          	lw	s5,0(a0) # 0 <BatchNorm_fp32>
  20:	c4a2                	sw	s0,72(sp)
  22:	c686                	sw	ra,76(sp)
  24:	d85a                	sw	s6,48(sp)
  26:	d65e                	sw	s7,44(sp)
  28:	d462                	sw	s8,40(sp)
  2a:	4401                	li	s0,0

0000002c <.L2>:
  2c:	01144e63          	blt	s0,a7,48 <.L7>
  30:	40b6                	lw	ra,76(sp)
  32:	4426                	lw	s0,72(sp)
  34:	4496                	lw	s1,68(sp)
  36:	4906                	lw	s2,64(sp)
  38:	59f2                	lw	s3,60(sp)
  3a:	5a62                	lw	s4,56(sp)
  3c:	5ad2                	lw	s5,52(sp)
  3e:	5b42                	lw	s6,48(sp)
  40:	5bb2                	lw	s7,44(sp)
  42:	5c22                	lw	s8,40(sp)
  44:	6161                	addi	sp,sp,80
  46:	8082                	ret

00000048 <.L7>:
  48:	0047250b          	p.lw	a0,4(a4!)
  4c:	0046ab0b          	p.lw	s6,4(a3!)
  50:	0045ab8b          	p.lw	s7,4(a1!)
  54:	01557553          	fadd.s	fa0,fa0,fs5
  58:	00462c0b          	p.lw	s8,4(a2!)
  5c:	ca2e                	sw	a1,20(sp)
  5e:	ce36                	sw	a3,28(sp)
  60:	c832                	sw	a2,16(sp)
  62:	cc3a                	sw	a4,24(sp)
  64:	c63e                	sw	a5,12(sp)
  66:	c442                	sw	a6,8(sp)
  68:	c246                	sw	a7,4(sp)
  6a:	00000097          	auipc	ra,0x0
  6e:	000080e7          	jalr	ra
  72:	45d2                	lw	a1,20(sp)
  74:	4642                	lw	a2,16(sp)
  76:	46f2                	lw	a3,28(sp)
  78:	4762                	lw	a4,24(sp)
  7a:	47b2                	lw	a5,12(sp)
  7c:	4822                	lw	a6,8(sp)
  7e:	4892                	lw	a7,4(sp)
  80:	4e01                	li	t3,0
  82:	4f01                	li	t5,0

00000084 <.L3>:
  84:	030f5863          	ble	a6,t5,b4 <.L5>
  88:	009e02b3          	add	t0,t3,s1
  8c:	00fe0fb3          	add	t6,t3,a5
  90:	4e81                	li	t4,0
  92:	a821                	j	aa <.L6>

00000094 <.L4>:
  94:	0042a30b          	p.lw	t1,4(t0!)
  98:	0e85                	addi	t4,t4,1
  9a:	09637353          	fsub.s	ft6,ft6,fs6
  9e:	18a37353          	fdiv.s	ft6,ft6,fa0
  a2:	c06bf343          	fmadd.s	ft6,fs7,ft6,fs8
  a6:	006fa22b          	p.sw	t1,4(t6!)

000000aa <.L6>:
  aa:	ff3ec5e3          	blt	t4,s3,94 <.L4>
  ae:	0f05                	addi	t5,t5,1
  b0:	9e52                	add	t3,t3,s4
  b2:	bfc9                	j	84 <.L3>

000000b4 <.L5>:
  b4:	0405                	addi	s0,s0,1
  b6:	94ca                	add	s1,s1,s2
  b8:	97ca                	add	a5,a5,s2
  ba:	bf8d                	j	2c <.L2>

ConvTranspose1d_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.ConvTranspose1d_fp32 000000fc  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  00000130  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  0000014b  2**0
                  CONTENTS, READONLY

Disassembly of section .text.ConvTranspose1d_fp32:

00000000 <ConvTranspose1d_fp32>:
   0:	7179                	addi	sp,sp,-48
   2:	d04e                	sw	s3,32(sp)
   4:	5e62                	lw	t3,56(sp)
   6:	59d2                	lw	s3,52(sp)
   8:	c662                	sw	s8,12(sp)
   a:	03014c03          	lbu	s8,48(sp)
   e:	d622                	sw	s0,44(sp)
  10:	d426                	sw	s1,40(sp)
  12:	d24a                	sw	s2,36(sp)
  14:	ce52                	sw	s4,28(sp)
  16:	cc56                	sw	s5,24(sp)
  18:	ca5a                	sw	s6,20(sp)
  1a:	c85e                	sw	s7,16(sp)
  1c:	c466                	sw	s9,8(sp)
  1e:	c26a                	sw	s10,4(sp)
  20:	c06e                	sw	s11,0(sp)
  22:	002e1293          	slli	t0,t3,0x2
  26:	8ece                	mv	t4,s3
  28:	4301                	li	t1,0

0000002a <.L2>:
  2a:	02e31a63          	bne	t1,a4,5e <.L17>
  2e:	00279a93          	slli	s5,a5,0x2
  32:	03530bb3          	mul	s7,t1,s5
  36:	00261b13          	slli	s6,a2,0x2
  3a:	4f81                	li	t6,0
  3c:	4901                	li	s2,0

0000003e <.L5>:
  3e:	0b230163          	beq	t1,s2,e0 <.L1>
  42:	84aa                	mv	s1,a0
  44:	8436                	mv	s0,a3
  46:	4381                	li	t2,0
  48:	a835                	j	84 <.L16>

0000004a <.L3>:
  4a:	00000393          	li	t2,0
  4e:	0f05                	addi	t5,t5,1
  50:	007fa22b          	p.sw	t2,4(t6!)

00000054 <.L4>:
  54:	ffcf1be3          	bne	t5,t3,4a <.L3>
  58:	0305                	addi	t1,t1,1
  5a:	9e96                	add	t4,t4,t0
  5c:	b7f9                	j	2a <.L2>

0000005e <.L17>:
  5e:	8ff6                	mv	t6,t4
  60:	4f01                	li	t5,0
  62:	bfcd                	j	54 <.L4>

00000064 <.L9>:
  64:	01f28733          	add	a4,t0,t6
  68:	004cad0b          	p.lw	s10,4(s9!)
  6c:	070a                	slli	a4,a4,0x2
  6e:	974e                	add	a4,a4,s3
  70:	4e81                	li	t4,0

00000072 <.L6>:
  72:	02fe9263          	bne	t4,a5,96 <.L8>
  76:	0a05                	addi	s4,s4,1
  78:	92c2                	add	t0,t0,a6

0000007a <.L10>:
  7a:	feca15e3          	bne	s4,a2,64 <.L9>
  7e:	0385                	addi	t2,t2,1
  80:	94da                	add	s1,s1,s6
  82:	945e                	add	s0,s0,s7

00000084 <.L16>:
  84:	02b39a63          	bne	t2,a1,b8 <.L18>
  88:	020c1c63          	bnez	s8,c0 <.L11>

0000008c <.L14>:
  8c:	0905                	addi	s2,s2,1
  8e:	0891                	addi	a7,a7,4
  90:	96d6                	add	a3,a3,s5
  92:	9ff2                	add	t6,t6,t3
  94:	b76d                	j	3e <.L5>

00000096 <.L8>:
  96:	005e8f33          	add	t5,t4,t0
  9a:	01cf7c63          	bleu	t3,t5,b2 <.L7>
  9e:	002e9f13          	slli	t5,t4,0x2
  a2:	21e47f03          	p.lw	t5,t5(s0)
  a6:	00072d83          	lw	s11,0(a4)
  aa:	d9ed7f43          	fmadd.s	ft10,fs10,ft10,fs11
  ae:	01e72023          	sw	t5,0(a4)

000000b2 <.L7>:
  b2:	0e85                	addi	t4,t4,1
  b4:	0711                	addi	a4,a4,4
  b6:	bf75                	j	72 <.L6>

000000b8 <.L18>:
  b8:	8ca6                	mv	s9,s1
  ba:	4281                	li	t0,0
  bc:	4a01                	li	s4,0
  be:	bf75                	j	7a <.L10>

000000c0 <.L11>:
  c0:	002f9713          	slli	a4,t6,0x2
  c4:	974e                	add	a4,a4,s3
  c6:	4e81                	li	t4,0

000000c8 <.L12>:
  c8:	fdce82e3          	beq	t4,t3,8c <.L14>
  cc:	00072283          	lw	t0,0(a4)
  d0:	0008af03          	lw	t5,0(a7)
  d4:	0e85                	addi	t4,t4,1
  d6:	005f7f53          	fadd.s	ft10,ft10,ft5
  da:	01e7222b          	p.sw	t5,4(a4!)
  de:	b7ed                	j	c8 <.L12>

000000e0 <.L1>:
  e0:	5432                	lw	s0,44(sp)
  e2:	54a2                	lw	s1,40(sp)
  e4:	5912                	lw	s2,36(sp)
  e6:	5982                	lw	s3,32(sp)
  e8:	4a72                	lw	s4,28(sp)
  ea:	4ae2                	lw	s5,24(sp)
  ec:	4b52                	lw	s6,20(sp)
  ee:	4bc2                	lw	s7,16(sp)
  f0:	4c32                	lw	s8,12(sp)
  f2:	4ca2                	lw	s9,8(sp)
  f4:	4d12                	lw	s10,4(sp)
  f6:	4d82                	lw	s11,0(sp)
  f8:	6145                	addi	sp,sp,48
  fa:	8082                	ret

Convolution_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Conv2d_fp32_fp32_fp32_NCHW 000001e0  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.Conv1d_fp32_fp32_fp32 000000b6  00000000  00000000  00000214  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .comment      0000001b  00000000  00000000  000002ca  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  000002e5  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Conv2d_fp32_fp32_fp32_NCHW:

00000000 <Conv2d_fp32_fp32_fp32_NCHW>:
   0:	7159                	addi	sp,sp,-112
   2:	5346                	lw	t1,112(sp)
   4:	d82a                	sw	a0,48(sp)
   6:	41060533          	sub	a0,a2,a6
   a:	02655533          	divu	a0,a0,t1
   e:	5e56                	lw	t3,116(sp)
  10:	c2ea                	sw	s10,68(sp)
  12:	07c14e83          	lbu	t4,124(sp)
  16:	c0ee                	sw	s11,64(sp)
  18:	da3e                	sw	a5,52(sp)
  1a:	d6a2                	sw	s0,108(sp)
  1c:	d4a6                	sw	s1,104(sp)
  1e:	d2ca                	sw	s2,100(sp)
  20:	d0ce                	sw	s3,96(sp)
  22:	ced2                	sw	s4,92(sp)
  24:	ccd6                	sw	s5,88(sp)
  26:	cada                	sw	s6,84(sp)
  28:	c8de                	sw	s7,80(sp)
  2a:	c6e2                	sw	s8,76(sp)
  2c:	c4e6                	sw	s9,72(sp)
  2e:	478a                	lw	a5,128(sp)
  30:	00259d93          	slli	s11,a1,0x2
  34:	00150d13          	addi	s10,a0,1
  38:	41168533          	sub	a0,a3,a7
  3c:	03c55533          	divu	a0,a0,t3
  40:	0e0a                	slli	t3,t3,0x2
  42:	dc72                	sw	t3,56(sp)
  44:	068a                	slli	a3,a3,0x2
  46:	0505                	addi	a0,a0,1
  48:	00251293          	slli	t0,a0,0x2
  4c:	cc2a                	sw	a0,24(sp)
  4e:	ca16                	sw	t0,20(sp)
  50:	00289513          	slli	a0,a7,0x2
  54:	020e9a63          	bnez	t4,88 <.L2>
  58:	031d8db3          	mul	s11,s11,a7
  5c:	025d03b3          	mul	t2,s10,t0
  60:	030d8db3          	mul	s11,s11,a6
  64:	d61e                	sw	t2,44(sp)
  66:	02668333          	mul	t1,a3,t1
  6a:	de6e                	sw	s11,60(sp)
  6c:	02c68633          	mul	a2,a3,a2
  70:	d41a                	sw	t1,40(sp)
  72:	030504b3          	mul	s1,a0,a6
  76:	d032                	sw	a2,32(sp)
  78:	d226                	sw	s1,36(sp)

0000007a <.L3>:
  7a:	5cd2                	lw	s9,52(sp)
  7c:	159e8463          	beq	t4,s9,1c4 <.L1>
  80:	5e42                	lw	t3,48(sp)
  82:	833e                	mv	t1,a5
  84:	4601                	li	a2,0
  86:	a229                	j	190 <.L17>

00000088 <.L2>:
  88:	031d8db3          	mul	s11,s11,a7
  8c:	4952                	lw	s2,20(sp)
  8e:	5f66                	lw	t5,120(sp)
  90:	4e81                	li	t4,0
  92:	032d0933          	mul	s2,s10,s2
  96:	030d8db3          	mul	s11,s11,a6
  9a:	de4a                	sw	s2,60(sp)
  9c:	02d30333          	mul	t1,t1,a3
  a0:	c66e                	sw	s11,12(sp)
  a2:	02d60633          	mul	a2,a2,a3
  a6:	d61a                	sw	t1,44(sp)
  a8:	02a80cb3          	mul	s9,a6,a0
  ac:	d232                	sw	a2,36(sp)
  ae:	d466                	sw	s9,40(sp)

000000b0 <.L4>:
  b0:	54d2                	lw	s1,52(sp)
  b2:	109e8963          	beq	t4,s1,1c4 <.L1>
  b6:	5e42                	lw	t3,48(sp)
  b8:	833e                	mv	t1,a5
  ba:	4601                	li	a2,0
  bc:	a8a1                	j	114 <.L11>

000000be <.L5>:
  be:	44c2                	lw	s1,16(sp)
  c0:	004c2d8b          	p.lw	s11,4(s8!)
  c4:	004ca38b          	p.lw	t2,4(s9!)
  c8:	0b05                	addi	s6,s6,1
  ca:	487df4c3          	fmadd.s	fs1,fs11,ft7,fs1
  ce:	c826                	sw	s1,16(sp)

000000d0 <.L6>:
  d0:	ff1b17e3          	bne	s6,a7,be <.L5>
  d4:	0985                	addi	s3,s3,1
  d6:	9a36                	add	s4,s4,a3
  d8:	9aaa                	add	s5,s5,a0

000000da <.L7>:
  da:	05099663          	bne	s3,a6,126 <.L18>
  de:	4ca2                	lw	s9,8(sp)
  e0:	5d92                	lw	s11,36(sp)
  e2:	53a2                	lw	t2,40(sp)
  e4:	0405                	addi	s0,s0,1
  e6:	9cee                	add	s9,s9,s11
  e8:	c466                	sw	s9,8(sp)
  ea:	991e                	add	s2,s2,t2

000000ec <.L8>:
  ec:	04b41163          	bne	s0,a1,12e <.L19>
  f0:	000f2403          	lw	s0,0(t5)
  f4:	44c2                	lw	s1,16(sp)
  f6:	5962                	lw	s2,56(sp)
  f8:	0f85                	addi	t6,t6,1
  fa:	009473d3          	fadd.s	ft7,fs0,fs1
  fe:	92ca                	add	t0,t0,s2
 100:	007ba22b          	p.sw	t2,4(s7!)

00000104 <.L9>:
 104:	4ce2                	lw	s9,24(sp)
 106:	039fe863          	bltu	t6,s9,136 <.L20>
 10a:	4dd2                	lw	s11,20(sp)
 10c:	5fb2                	lw	t6,44(sp)
 10e:	0605                	addi	a2,a2,1
 110:	936e                	add	t1,t1,s11
 112:	9e7e                	add	t3,t3,t6

00000114 <.L11>:
 114:	03a66663          	bltu	a2,s10,140 <.L21>
 118:	52f2                	lw	t0,60(sp)
 11a:	43b2                	lw	t2,12(sp)
 11c:	0e85                	addi	t4,t4,1
 11e:	9796                	add	a5,a5,t0
 120:	0f11                	addi	t5,t5,4
 122:	971e                	add	a4,a4,t2
 124:	b771                	j	b0 <.L4>

00000126 <.L18>:
 126:	8cd6                	mv	s9,s5
 128:	8c52                	mv	s8,s4
 12a:	4b01                	li	s6,0
 12c:	b755                	j	d0 <.L6>

0000012e <.L19>:
 12e:	8aca                	mv	s5,s2
 130:	4a22                	lw	s4,8(sp)
 132:	4981                	li	s3,0
 134:	b75d                	j	da <.L7>

00000136 <.L20>:
 136:	893a                	mv	s2,a4
 138:	c416                	sw	t0,8(sp)
 13a:	c802                	sw	zero,16(sp)
 13c:	4401                	li	s0,0
 13e:	b77d                	j	ec <.L8>

00000140 <.L21>:
 140:	82f2                	mv	t0,t3
 142:	8b9a                	mv	s7,t1
 144:	4f81                	li	t6,0
 146:	bf7d                	j	104 <.L9>

00000148 <.L12>:
 148:	004bac8b          	p.lw	s9,4(s7!)
 14c:	0a85                	addi	s5,s5,1
 14e:	ce66                	sw	s9,28(sp)
 150:	4df2                	lw	s11,28(sp)
 152:	004c2c8b          	p.lw	s9,4(s8!)
 156:	299df2c3          	fmadd.s	ft5,fs11,fs9,ft5

0000015a <.L13>:
 15a:	ff1a97e3          	bne	s5,a7,148 <.L12>
 15e:	0905                	addi	s2,s2,1
 160:	99b6                	add	s3,s3,a3
 162:	9a2a                	add	s4,s4,a0

00000164 <.L14>:
 164:	03091e63          	bne	s2,a6,1a0 <.L22>
 168:	5902                	lw	s2,32(sp)
 16a:	5c92                	lw	s9,36(sp)
 16c:	0385                	addi	t2,t2,1
 16e:	944a                	add	s0,s0,s2
 170:	94e6                	add	s1,s1,s9

00000172 <.L15>:
 172:	02b39b63          	bne	t2,a1,1a8 <.L23>
 176:	5de2                	lw	s11,56(sp)
 178:	005b222b          	p.sw	t0,4(s6!)
 17c:	0f05                	addi	t5,t5,1
 17e:	9fee                	add	t6,t6,s11

00000180 <.L16>:
 180:	42e2                	lw	t0,24(sp)
 182:	025f6763          	bltu	t5,t0,1b0 <.L24>
 186:	43d2                	lw	t2,20(sp)
 188:	54a2                	lw	s1,40(sp)
 18a:	0605                	addi	a2,a2,1
 18c:	931e                	add	t1,t1,t2
 18e:	9e26                	add	t3,t3,s1

00000190 <.L17>:
 190:	03a66663          	bltu	a2,s10,1bc <.L25>
 194:	5632                	lw	a2,44(sp)
 196:	5972                	lw	s2,60(sp)
 198:	0e85                	addi	t4,t4,1
 19a:	97b2                	add	a5,a5,a2
 19c:	974a                	add	a4,a4,s2
 19e:	bdf1                	j	7a <.L3>

000001a0 <.L22>:
 1a0:	8c52                	mv	s8,s4
 1a2:	8bce                	mv	s7,s3
 1a4:	4a81                	li	s5,0
 1a6:	bf55                	j	15a <.L13>

000001a8 <.L23>:
 1a8:	8a26                	mv	s4,s1
 1aa:	89a2                	mv	s3,s0
 1ac:	4901                	li	s2,0
 1ae:	bf5d                	j	164 <.L14>

000001b0 <.L24>:
 1b0:	84ba                	mv	s1,a4
 1b2:	847e                	mv	s0,t6
 1b4:	00000293          	li	t0,0
 1b8:	4381                	li	t2,0
 1ba:	bf65                	j	172 <.L15>

000001bc <.L25>:
 1bc:	8ff2                	mv	t6,t3
 1be:	8b1a                	mv	s6,t1
 1c0:	4f01                	li	t5,0
 1c2:	bf7d                	j	180 <.L16>

000001c4 <.L1>:
 1c4:	5436                	lw	s0,108(sp)
 1c6:	54a6                	lw	s1,104(sp)
 1c8:	5916                	lw	s2,100(sp)
 1ca:	5986                	lw	s3,96(sp)
 1cc:	4a76                	lw	s4,92(sp)
 1ce:	4ae6                	lw	s5,88(sp)
 1d0:	4b56                	lw	s6,84(sp)
 1d2:	4bc6                	lw	s7,80(sp)
 1d4:	4c36                	lw	s8,76(sp)
 1d6:	4ca6                	lw	s9,72(sp)
 1d8:	4d16                	lw	s10,68(sp)
 1da:	4d86                	lw	s11,64(sp)
 1dc:	6165                	addi	sp,sp,112
 1de:	8082                	ret

Disassembly of section .text.Conv1d_fp32_fp32_fp32:

00000000 <Conv1d_fp32_fp32_fp32>:
   0:	7179                	addi	sp,sp,-48
   2:	d04e                	sw	s3,32(sp)
   4:	00279993          	slli	s3,a5,0x2
   8:	c662                	sw	s8,12(sp)
   a:	02b98c33          	mul	s8,s3,a1
   e:	ce52                	sw	s4,28(sp)
  10:	5a62                	lw	s4,56(sp)
  12:	d426                	sw	s1,40(sp)
  14:	c85e                	sw	s7,16(sp)
  16:	54d2                	lw	s1,52(sp)
  18:	03014b83          	lbu	s7,48(sp)
  1c:	d622                	sw	s0,44(sp)
  1e:	ca5a                	sw	s6,20(sp)
  20:	c466                	sw	s9,8(sp)
  22:	d24a                	sw	s2,36(sp)
  24:	cc56                	sw	s5,24(sp)
  26:	c26a                	sw	s10,4(sp)
  28:	002a1b13          	slli	s6,s4,0x2
  2c:	00261c93          	slli	s9,a2,0x2
  30:	4401                	li	s0,0

00000032 <.L28>:
  32:	06e41963          	bne	s0,a4,a4 <.L37>
  36:	5432                	lw	s0,44(sp)
  38:	54a2                	lw	s1,40(sp)
  3a:	5912                	lw	s2,36(sp)
  3c:	5982                	lw	s3,32(sp)
  3e:	4a72                	lw	s4,28(sp)
  40:	4ae2                	lw	s5,24(sp)
  42:	4b52                	lw	s6,20(sp)
  44:	4bc2                	lw	s7,16(sp)
  46:	4c32                	lw	s8,12(sp)
  48:	4ca2                	lw	s9,8(sp)
  4a:	4d12                	lw	s10,4(sp)
  4c:	6145                	addi	sp,sp,48
  4e:	8082                	ret

00000050 <.L30>:
  50:	00728eb3          	add	t4,t0,t2
  54:	00cefa63          	bleu	a2,t4,68 <.L29>
  58:	00229e93          	slli	t4,t0,0x2
  5c:	21de7d03          	p.lw	s10,t4(t3)
  60:	21dffe83          	p.lw	t4,t4(t6)
  64:	31dd7343          	fmadd.s	ft6,fs10,ft9,ft6

00000068 <.L29>:
  68:	0285                	addi	t0,t0,1

0000006a <.L31>:
  6a:	fef293e3          	bne	t0,a5,50 <.L30>
  6e:	0f05                	addi	t5,t5,1
  70:	9e66                	add	t3,t3,s9
  72:	9fce                	add	t6,t6,s3

00000074 <.L34>:
  74:	02bf1663          	bne	t5,a1,a0 <.L36>
  78:	000b8663          	beqz	s7,84 <.L32>
  7c:	0008ae03          	lw	t3,0(a7)
  80:	01c37353          	fadd.s	ft6,ft6,ft8

00000084 <.L32>:
  84:	006aa22b          	p.sw	t1,4(s5!)
  88:	0905                	addi	s2,s2,1
  8a:	93c2                	add	t2,t2,a6

0000008c <.L35>:
  8c:	03490063          	beq	s2,s4,ac <.L33>
  90:	00239e13          	slli	t3,t2,0x2
  94:	9e2a                	add	t3,t3,a0
  96:	8fb6                	mv	t6,a3
  98:	00000313          	li	t1,0
  9c:	4f01                	li	t5,0
  9e:	bfd9                	j	74 <.L34>

000000a0 <.L36>:
  a0:	4281                	li	t0,0
  a2:	b7e1                	j	6a <.L31>

000000a4 <.L37>:
  a4:	8aa6                	mv	s5,s1
  a6:	4381                	li	t2,0
  a8:	4901                	li	s2,0
  aa:	b7cd                	j	8c <.L35>

000000ac <.L33>:
  ac:	0405                	addi	s0,s0,1
  ae:	94da                	add	s1,s1,s6
  b0:	0891                	addi	a7,a7,4
  b2:	96e2                	add	a3,a3,s8
  b4:	bfbd                	j	32 <.L28>

Convolution_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Conv2d_s8_s8_s32_NCHW 0000013a  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000016e  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000189  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Conv2d_s8_s8_s32_NCHW:

00000000 <Conv2d_s8_s8_s32_NCHW>:
   0:	7159                	addi	sp,sp,-112
   2:	5e46                	lw	t3,112(sp)
   4:	41060333          	sub	t1,a2,a6
   8:	53d6                	lw	t2,116(sp)
   a:	03c35333          	divu	t1,t1,t3
   e:	c63e                	sw	a5,12(sp)
  10:	d0ce                	sw	s3,96(sp)
  12:	c6e2                	sw	s8,76(sp)
  14:	5c66                	lw	s8,120(sp)
  16:	d6a2                	sw	s0,108(sp)
  18:	d4a6                	sw	s1,104(sp)
  1a:	d2ca                	sw	s2,100(sp)
  1c:	ced2                	sw	s4,92(sp)
  1e:	ccd6                	sw	s5,88(sp)
  20:	cada                	sw	s6,84(sp)
  22:	c8de                	sw	s7,80(sp)
  24:	c4e6                	sw	s9,72(sp)
  26:	c2ea                	sw	s10,68(sp)
  28:	c0ee                	sw	s11,64(sp)
  2a:	cc62                	sw	s8,24(sp)
  2c:	0305                	addi	t1,t1,1
  2e:	d01a                	sw	t1,32(sp)
  30:	41168333          	sub	t1,a3,a7
  34:	02735333          	divu	t1,t1,t2
  38:	5982                	lw	s3,32(sp)
  3a:	0305                	addi	t1,t1,1
  3c:	00231793          	slli	a5,t1,0x2
  40:	d21a                	sw	t1,36(sp)
  42:	02b88333          	mul	t1,a7,a1
  46:	ce3e                	sw	a5,28(sp)
  48:	03180eb3          	mul	t4,a6,a7
  4c:	033789b3          	mul	s3,a5,s3
  50:	da76                	sw	t4,52(sp)
  52:	4e81                	li	t4,0
  54:	03030333          	mul	t1,t1,a6
  58:	dc4e                	sw	s3,56(sp)
  5a:	02de0e33          	mul	t3,t3,a3
  5e:	de1a                	sw	t1,60(sp)
  60:	02d60633          	mul	a2,a2,a3
  64:	d672                	sw	t3,44(sp)
  66:	d432                	sw	a2,40(sp)

00000068 <.L2>:
  68:	47b2                	lw	a5,12(sp)
  6a:	0afe8a63          	beq	t4,a5,11e <.L1>
  6e:	49e2                	lw	s3,24(sp)
  70:	8e2a                	mv	t3,a0
  72:	40a00333          	neg	t1,a0
  76:	ca4e                	sw	s3,20(sp)
  78:	4d81                	li	s11,0
  7a:	a0a5                	j	e2 <.L9>

0000007c <.L3>:
  7c:	001d0c0b          	p.lb	s8,1(s10!)
  80:	001b878b          	p.lb	a5,1(s7!)
  84:	d862                	sw	s8,48(sp)
  86:	5c76                	lw	s8,124(sp)
  88:	97e2                	add	a5,a5,s8
  8a:	5c42                	lw	s8,48(sp)
  8c:	43878633          	p.mac	a2,a5,s8

00000090 <.L4>:
  90:	016b8c33          	add	s8,s7,s6
  94:	ff1c64e3          	bltu	s8,a7,7c <.L3>
  98:	0985                	addi	s3,s3,1
  9a:	9a36                	add	s4,s4,a3
  9c:	9ac6                	add	s5,s5,a7
  9e:	40db0b33          	sub	s6,s6,a3

000000a2 <.L5>:
  a2:	05099b63          	bne	s3,a6,f8 <.L10>
  a6:	57a2                	lw	a5,40(sp)
  a8:	59d2                	lw	s3,52(sp)
  aa:	0385                	addi	t2,t2,1
  ac:	943e                	add	s0,s0,a5
  ae:	94ce                	add	s1,s1,s3
  b0:	40f90933          	sub	s2,s2,a5

000000b4 <.L6>:
  b4:	04b39563          	bne	t2,a1,fe <.L11>
  b8:	4c0a                	lw	s8,128(sp)
  ba:	53d6                	lw	t2,116(sp)
  bc:	0f05                	addi	t5,t5,1
  be:	9662                	add	a2,a2,s8
  c0:	00cca22b          	p.sw	a2,4(s9!)
  c4:	9f9e                	add	t6,t6,t2
  c6:	407282b3          	sub	t0,t0,t2

000000ca <.L7>:
  ca:	5612                	lw	a2,36(sp)
  cc:	02cf6e63          	bltu	t5,a2,108 <.L12>
  d0:	47d2                	lw	a5,20(sp)
  d2:	49f2                	lw	s3,28(sp)
  d4:	5c32                	lw	s8,44(sp)
  d6:	0d85                	addi	s11,s11,1
  d8:	97ce                	add	a5,a5,s3
  da:	ca3e                	sw	a5,20(sp)
  dc:	9e62                	add	t3,t3,s8
  de:	41830333          	sub	t1,t1,s8

000000e2 <.L9>:
  e2:	5f02                	lw	t5,32(sp)
  e4:	03ede863          	bltu	s11,t5,114 <.L13>
  e8:	4362                	lw	t1,24(sp)
  ea:	53e2                	lw	t2,56(sp)
  ec:	5672                	lw	a2,60(sp)
  ee:	0e85                	addi	t4,t4,1
  f0:	931e                	add	t1,t1,t2
  f2:	cc1a                	sw	t1,24(sp)
  f4:	9732                	add	a4,a4,a2
  f6:	bf8d                	j	68 <.L2>

000000f8 <.L10>:
  f8:	8d56                	mv	s10,s5
  fa:	8bd2                	mv	s7,s4
  fc:	bf51                	j	90 <.L4>

000000fe <.L11>:
  fe:	8b4a                	mv	s6,s2
 100:	8aa6                	mv	s5,s1
 102:	8a22                	mv	s4,s0
 104:	4981                	li	s3,0
 106:	bf71                	j	a2 <.L5>

00000108 <.L12>:
 108:	8916                	mv	s2,t0
 10a:	84ba                	mv	s1,a4
 10c:	847e                	mv	s0,t6
 10e:	4601                	li	a2,0
 110:	4381                	li	t2,0
 112:	b74d                	j	b4 <.L6>

00000114 <.L13>:
 114:	829a                	mv	t0,t1
 116:	8ff2                	mv	t6,t3
 118:	4cd2                	lw	s9,20(sp)
 11a:	4f01                	li	t5,0
 11c:	b77d                	j	ca <.L7>

0000011e <.L1>:
 11e:	5436                	lw	s0,108(sp)
 120:	54a6                	lw	s1,104(sp)
 122:	5916                	lw	s2,100(sp)
 124:	5986                	lw	s3,96(sp)
 126:	4a76                	lw	s4,92(sp)
 128:	4ae6                	lw	s5,88(sp)
 12a:	4b56                	lw	s6,84(sp)
 12c:	4bc6                	lw	s7,80(sp)
 12e:	4c36                	lw	s8,76(sp)
 130:	4ca6                	lw	s9,72(sp)
 132:	4d16                	lw	s10,68(sp)
 134:	4d86                	lw	s11,64(sp)
 136:	6165                	addi	sp,sp,112
 138:	8082                	ret

DWConvolution_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.DWConv2d_fp32_fp32_fp32_NCHW 000001ea  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000021e  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000239  2**0
                  CONTENTS, READONLY

Disassembly of section .text.DWConv2d_fp32_fp32_fp32_NCHW:

00000000 <DWConv2d_fp32_fp32_fp32_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	5e16                	lw	t3,100(sp)
   4:	5306                	lw	t1,96(sp)
   6:	ce3a                	sw	a4,28(sp)
   8:	41168733          	sub	a4,a3,a7
   c:	03c75733          	divu	a4,a4,t3
  10:	d03e                	sw	a5,32(sp)
  12:	410607b3          	sub	a5,a2,a6
  16:	06c14e83          	lbu	t4,108(sp)
  1a:	0e0a                	slli	t3,t3,0x2
  1c:	da6a                	sw	s10,52(sp)
  1e:	d272                	sw	t3,36(sp)
  20:	cea2                	sw	s0,92(sp)
  22:	cca6                	sw	s1,88(sp)
  24:	caca                	sw	s2,84(sp)
  26:	c8ce                	sw	s3,80(sp)
  28:	c6d2                	sw	s4,76(sp)
  2a:	c4d6                	sw	s5,72(sp)
  2c:	c2da                	sw	s6,68(sp)
  2e:	c0de                	sw	s7,64(sp)
  30:	de62                	sw	s8,60(sp)
  32:	dc66                	sw	s9,56(sp)
  34:	d86e                	sw	s11,48(sp)
  36:	068a                	slli	a3,a3,0x2
  38:	00289d13          	slli	s10,a7,0x2
  3c:	0267d7b3          	divu	a5,a5,t1
  40:	0705                	addi	a4,a4,1
  42:	00271e13          	slli	t3,a4,0x2
  46:	0785                	addi	a5,a5,1
  48:	0c0e9763          	bnez	t4,116 <.L2>
  4c:	02d60633          	mul	a2,a2,a3
  50:	4f81                	li	t6,0
  52:	4281                	li	t0,0
  54:	02d30333          	mul	t1,t1,a3
  58:	cc32                	sw	a2,24(sp)
  5a:	02fe0e33          	mul	t3,t3,a5
  5e:	ca1a                	sw	t1,20(sp)
  60:	d472                	sw	t3,40(sp)
  62:	030d0bb3          	mul	s7,s10,a6
  66:	aa11                	j	17a <.L19>

00000068 <.L8>:
  68:	4322                	lw	t1,8(sp)
  6a:	02b2d633          	divu	a2,t0,a1
  6e:	54a6                	lw	s1,104(sp)
  70:	8976                	mv	s2,t4
  72:	4b12                	lw	s6,4(sp)
  74:	59c6                	lw	s3,112(sp)
  76:	02b35333          	divu	t1,t1,a1
  7a:	00261f13          	slli	t5,a2,0x2
  7e:	9f26                	add	t5,t5,s1
  80:	44f2                	lw	s1,28(sp)
  82:	ca1a                	sw	t1,20(sp)
  84:	8372                	mv	t1,t3
  86:	42c78333          	p.mac	t1,a5,a2
  8a:	42e30933          	p.mac	s2,t1,a4
  8e:	436604b3          	p.mac	s1,a2,s6
  92:	00291313          	slli	t1,s2,0x2
  96:	934e                	add	t1,t1,s3

00000098 <.L4>:
  98:	4cd2                	lw	s9,20(sp)
  9a:	07966563          	bltu	a2,s9,104 <.L22>
  9e:	5d92                	lw	s11,36(sp)
  a0:	0e85                	addi	t4,t4,1
  a2:	946e                	add	s0,s0,s11

000000a4 <.L9>:
  a4:	fceee2e3          	bltu	t4,a4,68 <.L8>
  a8:	4362                	lw	t1,24(sp)
  aa:	0e05                	addi	t3,t3,1
  ac:	9f9a                	add	t6,t6,t1

000000ae <.L12>:
  ae:	06fe6163          	bltu	t3,a5,110 <.L23>
  b2:	5422                	lw	s0,40(sp)
  b4:	42a2                	lw	t0,8(sp)
  b6:	9522                	add	a0,a0,s0

000000b8 <.L10>:
  b8:	10b38b63          	beq	t2,a1,1ce <.L1>
  bc:	5602                	lw	a2,32(sp)
  be:	0385                	addi	t2,t2,1
  c0:	8faa                	mv	t6,a0
  c2:	9616                	add	a2,a2,t0
  c4:	c432                	sw	a2,8(sp)
  c6:	4e01                	li	t3,0
  c8:	b7dd                	j	ae <.L12>

000000ca <.L5>:
  ca:	004bac8b          	p.lw	s9,4(s7!)
  ce:	004c2d8b          	p.lw	s11,4(s8!)
  d2:	0b05                	addi	s6,s6,1
  d4:	91bcf943          	fmadd.s	fs2,fs9,fs11,fs2

000000d8 <.L6>:
  d8:	ff1b19e3          	bne	s6,a7,ca <.L5>
  dc:	0985                	addi	s3,s3,1
  de:	9a36                	add	s4,s4,a3
  e0:	9aea                	add	s5,s5,s10

000000e2 <.L7>:
  e2:	01099d63          	bne	s3,a6,fc <.L21>
  e6:	004f298b          	p.lw	s3,4(t5!)
  ea:	4b12                	lw	s6,4(sp)
  ec:	0605                	addi	a2,a2,1
  ee:	0129f953          	fadd.s	fs2,fs3,fs2
  f2:	59b2                	lw	s3,44(sp)
  f4:	94da                	add	s1,s1,s6
  f6:	012369ab          	p.sw	s2,s3(t1!)
  fa:	bf79                	j	98 <.L4>

000000fc <.L21>:
  fc:	8c56                	mv	s8,s5
  fe:	8bd2                	mv	s7,s4
 100:	4b01                	li	s6,0
 102:	bfd9                	j	d8 <.L6>

00000104 <.L22>:
 104:	8aa6                	mv	s5,s1
 106:	8a22                	mv	s4,s0
 108:	00000913          	li	s2,0
 10c:	4981                	li	s3,0
 10e:	bfd1                	j	e2 <.L7>

00000110 <.L23>:
 110:	847e                	mv	s0,t6
 112:	4e81                	li	t4,0
 114:	bf41                	j	a4 <.L9>

00000116 <.L2>:
 116:	02d60633          	mul	a2,a2,a3
 11a:	4281                	li	t0,0
 11c:	4381                	li	t2,0
 11e:	02d30333          	mul	t1,t1,a3
 122:	d432                	sw	a2,40(sp)
 124:	02fe0e33          	mul	t3,t3,a5
 128:	cc1a                	sw	t1,24(sp)
 12a:	03a804b3          	mul	s1,a6,s10
 12e:	d672                	sw	t3,44(sp)
 130:	c226                	sw	s1,4(sp)
 132:	b759                	j	b8 <.L10>

00000134 <.L17>:
 134:	02bfd333          	divu	t1,t6,a1
 138:	4922                	lw	s2,8(sp)
 13a:	8672                	mv	a2,t3
 13c:	89f6                	mv	s3,t4
 13e:	4472                	lw	s0,28(sp)
 140:	5b46                	lw	s6,112(sp)
 142:	02b95933          	divu	s2,s2,a1
 146:	42678633          	p.mac	a2,a5,t1
 14a:	c84a                	sw	s2,16(sp)
 14c:	42e609b3          	p.mac	s3,a2,a4
 150:	43730433          	p.mac	s0,t1,s7
 154:	00299613          	slli	a2,s3,0x2
 158:	965a                	add	a2,a2,s6

0000015a <.L13>:
 15a:	4dc2                	lw	s11,16(sp)
 15c:	07b36063          	bltu	t1,s11,1bc <.L25>
 160:	5312                	lw	t1,36(sp)
 162:	0e85                	addi	t4,t4,1
 164:	939a                	add	t2,t2,t1

00000166 <.L18>:
 166:	fceee7e3          	bltu	t4,a4,134 <.L17>
 16a:	4452                	lw	s0,20(sp)
 16c:	0e05                	addi	t3,t3,1
 16e:	9f22                	add	t5,t5,s0

00000170 <.L20>:
 170:	04fe6c63          	bltu	t3,a5,1c8 <.L26>
 174:	44e2                	lw	s1,24(sp)
 176:	4fa2                	lw	t6,8(sp)
 178:	9526                	add	a0,a0,s1

0000017a <.L19>:
 17a:	04b28a63          	beq	t0,a1,1ce <.L1>
 17e:	5602                	lw	a2,32(sp)
 180:	0285                	addi	t0,t0,1
 182:	8f2a                	mv	t5,a0
 184:	967e                	add	a2,a2,t6
 186:	c432                	sw	a2,8(sp)
 188:	4e01                	li	t3,0
 18a:	b7dd                	j	170 <.L20>

0000018c <.L14>:
 18c:	004c2d8b          	p.lw	s11,4(s8!)
 190:	004cab0b          	p.lw	s6,4(s9!)
 194:	0a85                	addi	s5,s5,1
 196:	496df4c3          	fmadd.s	fs1,fs11,fs6,fs1

0000019a <.L15>:
 19a:	ff1a99e3          	bne	s5,a7,18c <.L14>
 19e:	0905                	addi	s2,s2,1
 1a0:	99b6                	add	s3,s3,a3
 1a2:	9a6a                	add	s4,s4,s10

000001a4 <.L16>:
 1a4:	01091863          	bne	s2,a6,1b4 <.L24>
 1a8:	5ca2                	lw	s9,40(sp)
 1aa:	0305                	addi	t1,t1,1
 1ac:	945e                	add	s0,s0,s7
 1ae:	00966cab          	p.sw	s1,s9(a2!)
 1b2:	b765                	j	15a <.L13>

000001b4 <.L24>:
 1b4:	8cd2                	mv	s9,s4
 1b6:	8c4e                	mv	s8,s3
 1b8:	4a81                	li	s5,0
 1ba:	b7c5                	j	19a <.L15>

000001bc <.L25>:
 1bc:	8a22                	mv	s4,s0
 1be:	899e                	mv	s3,t2
 1c0:	00000493          	li	s1,0
 1c4:	4901                	li	s2,0
 1c6:	bff9                	j	1a4 <.L16>

000001c8 <.L26>:
 1c8:	83fa                	mv	t2,t5
 1ca:	4e81                	li	t4,0
 1cc:	bf69                	j	166 <.L18>

000001ce <.L1>:
 1ce:	4476                	lw	s0,92(sp)
 1d0:	44e6                	lw	s1,88(sp)
 1d2:	4956                	lw	s2,84(sp)
 1d4:	49c6                	lw	s3,80(sp)
 1d6:	4a36                	lw	s4,76(sp)
 1d8:	4aa6                	lw	s5,72(sp)
 1da:	4b16                	lw	s6,68(sp)
 1dc:	4b86                	lw	s7,64(sp)
 1de:	5c72                	lw	s8,60(sp)
 1e0:	5ce2                	lw	s9,56(sp)
 1e2:	5d52                	lw	s10,52(sp)
 1e4:	5dc2                	lw	s11,48(sp)
 1e6:	6125                	addi	sp,sp,96
 1e8:	8082                	ret

DWConvolution_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.DWConv2d_s8_s8_s32_NCHW 00000108  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000013c  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000157  2**0
                  CONTENTS, READONLY

Disassembly of section .text.DWConv2d_s8_s8_s32_NCHW:

00000000 <DWConv2d_s8_s8_s32_NCHW>:
   0:	40f60333          	sub	t1,a2,a5
   4:	03135333          	divu	t1,t1,a7
   8:	715d                	addi	sp,sp,-80
   a:	c0ce                	sw	s3,64(sp)
   c:	49c6                	lw	s3,80(sp)
   e:	d662                	sw	s8,44(sp)
  10:	d06e                	sw	s11,32(sp)
  12:	4e56                	lw	t3,84(sp)
  14:	c6a2                	sw	s0,76(sp)
  16:	c4a6                	sw	s1,72(sp)
  18:	c2ca                	sw	s2,68(sp)
  1a:	de52                	sw	s4,60(sp)
  1c:	dc56                	sw	s5,56(sp)
  1e:	da5a                	sw	s6,52(sp)
  20:	d85e                	sw	s7,48(sp)
  22:	d466                	sw	s9,40(sp)
  24:	d26a                	sw	s10,36(sp)
  26:	40a00eb3          	neg	t4,a0
  2a:	0305                	addi	t1,t1,1
  2c:	c41a                	sw	t1,8(sp)
  2e:	41068333          	sub	t1,a3,a6
  32:	03335333          	divu	t1,t1,s3
  36:	4c22                	lw	s8,8(sp)
  38:	0305                	addi	t1,t1,1
  3a:	00231d93          	slli	s11,t1,0x2
  3e:	c61a                	sw	t1,12(sp)
  40:	038d8c33          	mul	s8,s11,s8
  44:	03078333          	mul	t1,a5,a6
  48:	cc62                	sw	s8,24(sp)
  4a:	02d60633          	mul	a2,a2,a3
  4e:	ce1a                	sw	t1,28(sp)
  50:	4301                	li	t1,0
  52:	02d888b3          	mul	a7,a7,a3
  56:	c032                	sw	a2,0(sp)
  58:	c846                	sw	a7,16(sp)

0000005a <.L2>:
  5a:	0ab31263          	bne	t1,a1,fe <.L11>
  5e:	4436                	lw	s0,76(sp)
  60:	44a6                	lw	s1,72(sp)
  62:	4916                	lw	s2,68(sp)
  64:	4986                	lw	s3,64(sp)
  66:	5a72                	lw	s4,60(sp)
  68:	5ae2                	lw	s5,56(sp)
  6a:	5b52                	lw	s6,52(sp)
  6c:	5bc2                	lw	s7,48(sp)
  6e:	5c32                	lw	s8,44(sp)
  70:	5ca2                	lw	s9,40(sp)
  72:	5d12                	lw	s10,36(sp)
  74:	5d82                	lw	s11,32(sp)
  76:	6161                	addi	sp,sp,80
  78:	8082                	ret

0000007a <.L3>:
  7a:	001d0c0b          	p.lb	s8,1(s10!)
  7e:	001b860b          	p.lb	a2,1(s7!)
  82:	ca62                	sw	s8,20(sp)
  84:	4c66                	lw	s8,88(sp)
  86:	9662                	add	a2,a2,s8
  88:	4c52                	lw	s8,20(sp)
  8a:	438608b3          	p.mac	a7,a2,s8

0000008e <.L4>:
  8e:	016b8c33          	add	s8,s7,s6
  92:	ff0c64e3          	bltu	s8,a6,7a <.L3>
  96:	0985                	addi	s3,s3,1
  98:	9a36                	add	s4,s4,a3
  9a:	9ac2                	add	s5,s5,a6
  9c:	40db0b33          	sub	s6,s6,a3

000000a0 <.L5>:
  a0:	04f99163          	bne	s3,a5,e2 <.L8>
  a4:	4676                	lw	a2,92(sp)
  a6:	0405                	addi	s0,s0,1
  a8:	98b2                	add	a7,a7,a2
  aa:	011ca22b          	p.sw	a7,4(s9!)
  ae:	48c6                	lw	a7,80(sp)
  b0:	94c6                	add	s1,s1,a7
  b2:	41190933          	sub	s2,s2,a7

000000b6 <.L6>:
  b6:	49b2                	lw	s3,12(sp)
  b8:	03346863          	bltu	s0,s3,e8 <.L9>
  bc:	4c42                	lw	s8,16(sp)
  be:	0f05                	addi	t5,t5,1
  c0:	9fee                	add	t6,t6,s11
  c2:	92e2                	add	t0,t0,s8
  c4:	418383b3          	sub	t2,t2,s8

000000c8 <.L7>:
  c8:	4622                	lw	a2,8(sp)
  ca:	02cf6563          	bltu	t5,a2,f4 <.L10>
  ce:	4982                	lw	s3,0(sp)
  d0:	48e2                	lw	a7,24(sp)
  d2:	4c72                	lw	s8,28(sp)
  d4:	0305                	addi	t1,t1,1
  d6:	9e46                	add	t3,t3,a7
  d8:	954e                	add	a0,a0,s3
  da:	413e8eb3          	sub	t4,t4,s3
  de:	9762                	add	a4,a4,s8
  e0:	bfad                	j	5a <.L2>

000000e2 <.L8>:
  e2:	8d56                	mv	s10,s5
  e4:	8bd2                	mv	s7,s4
  e6:	b765                	j	8e <.L4>

000000e8 <.L9>:
  e8:	8b4a                	mv	s6,s2
  ea:	8aba                	mv	s5,a4
  ec:	8a26                	mv	s4,s1
  ee:	4881                	li	a7,0
  f0:	4981                	li	s3,0
  f2:	b77d                	j	a0 <.L5>

000000f4 <.L10>:
  f4:	891e                	mv	s2,t2
  f6:	8496                	mv	s1,t0
  f8:	8cfe                	mv	s9,t6
  fa:	4401                	li	s0,0
  fc:	bf6d                	j	b6 <.L6>

000000fe <.L11>:
  fe:	83f6                	mv	t2,t4
 100:	82aa                	mv	t0,a0
 102:	8ff2                	mv	t6,t3
 104:	4f01                	li	t5,0
 106:	b7c9                	j	c8 <.L7>

Div_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Div_fp32_fp32_fp32 0000001c  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  00000050  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  0000006b  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Div_fp32_fp32_fp32:

00000000 <Div_fp32_fp32_fp32>:
   0:	4781                	li	a5,0

00000002 <.L2>:
   2:	00d7c363          	blt	a5,a3,8 <.L3>
   6:	8082                	ret

00000008 <.L3>:
   8:	0045270b          	p.lw	a4,4(a0!)
   c:	0045a80b          	p.lw	a6,4(a1!)
  10:	0785                	addi	a5,a5,1
  12:	19077753          	fdiv.s	fa4,fa4,fa6
  16:	00e6222b          	p.sw	a4,4(a2!)
  1a:	b7e5                	j	2 <.L2>

Div_s32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Div_s32_s32 0000011a  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000014e  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000169  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Div_s32_s32:

00000000 <Div_s32_s32>:
   0:	7119                	addi	sp,sp,-128
   2:	cede                	sw	s7,92(sp)
   4:	02f74bb3          	div	s7,a4,a5
   8:	468a                	lw	a3,128(sp)
   a:	d4d2                	sw	s4,104(sp)
   c:	d2d6                	sw	s5,100(sp)
   e:	d0da                	sw	s6,96(sp)
  10:	c8ea                	sw	s10,80(sp)
  12:	de86                	sw	ra,124(sp)
  14:	dca2                	sw	s0,120(sp)
  16:	daa6                	sw	s1,116(sp)
  18:	d8ca                	sw	s2,112(sp)
  1a:	d6ce                	sw	s3,108(sp)
  1c:	cce2                	sw	s8,88(sp)
  1e:	cae6                	sw	s9,84(sp)
  20:	c6ee                	sw	s11,76(sp)
  22:	de2a                	sw	a0,60(sp)
  24:	00279d13          	slli	s10,a5,0x2
  28:	8a2e                	mv	s4,a1
  2a:	4a81                	li	s5,0
  2c:	4b01                	li	s6,0
  2e:	41f6d293          	srai	t0,a3,0x1f
  32:	03764733          	div	a4,a2,s7
  36:	461a                	lw	a2,132(sp)
  38:	031608b3          	mul	a7,a2,a7

0000003c <.L2>:
  3c:	0ceb5063          	ble	a4,s6,fc <.L1>
  40:	56f2                	lw	a3,60(sp)
  42:	8cd2                	mv	s9,s4
  44:	4481                	li	s1,0
  46:	4981                	li	s3,0
  48:	03578433          	mul	s0,a5,s5
  4c:	8c1d                	sub	s0,s0,a5
  4e:	040a                	slli	s0,s0,0x2
  50:	00868db3          	add	s11,a3,s0
  54:	9442                	add	s0,s0,a6
  56:	a815                	j	8a <.L7>

00000058 <.L5>:
  58:	460a                	lw	a2,128(sp)
  5a:	d616                	sw	t0,44(sp)
  5c:	004ca68b          	p.lw	a3,4(s9!)
  60:	d432                	sw	a2,40(sp)
  62:	5f22                	lw	t5,40(sp)
  64:	461a                	lw	a2,132(sp)
  66:	5fb2                	lw	t6,44(sp)
  68:	4901                	li	s2,0
  6a:	4c01                	li	s8,0
  6c:	72d60f33          	p.mac.d	t5,a2,a3
  70:	01b486b3          	add	a3,s1,s11
  74:	00848633          	add	a2,s1,s0
  78:	da36                	sw	a3,52(sp)
  7a:	dc32                	sw	a2,56(sp)
  7c:	641f5e13          	srai.d	t3,t5,0x1

00000080 <.L3>:
  80:	996a                	add	s2,s2,s10
  82:	017c4a63          	blt	s8,s7,96 <.L4>
  86:	0985                	addi	s3,s3,1
  88:	0491                	addi	s1,s1,4

0000008a <.L7>:
  8a:	fcf9c7e3          	blt	s3,a5,58 <.L5>
  8e:	0b05                	addi	s6,s6,1
  90:	9a6a                	add	s4,s4,s10
  92:	9ade                	add	s5,s5,s7
  94:	b765                	j	3c <.L2>

00000096 <.L4>:
  96:	56d2                	lw	a3,52(sp)
  98:	0c05                	addi	s8,s8,1
  9a:	2126f503          	p.lw	a0,s2(a3)
  9e:	c216                	sw	t0,4(sp)
  a0:	cc3a                	sw	a4,24(sp)
  a2:	72a8c333          	p.muls.d	t1,a7,a0
  a6:	c63e                	sw	a5,12(sp)
  a8:	c442                	sw	a6,8(sp)
  aa:	d246                	sw	a7,36(sp)
  ac:	d072                	sw	t3,32(sp)
  ae:	ce76                	sw	t4,28(sp)
  b0:	ca7a                	sw	t5,20(sp)
  b2:	c87e                	sw	t6,16(sp)
  b4:	fff3c513          	not	a0,t2
  b8:	01f3d693          	srli	a3,t2,0x1f
  bc:	817d                	srli	a0,a0,0x1f
  be:	8d15                	sub	a0,a0,a3
  c0:	02ae86b3          	mul	a3,t4,a0
  c4:	41f55613          	srai	a2,a0,0x1f
  c8:	43c606b3          	p.mac	a3,a2,t3
  cc:	73c55533          	p.mulu.d	a0,a0,t3
  d0:	95b6                	add	a1,a1,a3
  d2:	20650533          	add.d	a0,a0,t1
  d6:	220f1613          	addi.d	a2,t5,0
  da:	00000097          	auipc	ra,0x0
  de:	000080e7          	jalr	ra
  e2:	5662                	lw	a2,56(sp)
  e4:	4fc2                	lw	t6,16(sp)
  e6:	4f52                	lw	t5,20(sp)
  e8:	00a66923          	p.sw	a0,s2(a2)
  ec:	4ef2                	lw	t4,28(sp)
  ee:	5e02                	lw	t3,32(sp)
  f0:	5892                	lw	a7,36(sp)
  f2:	4822                	lw	a6,8(sp)
  f4:	47b2                	lw	a5,12(sp)
  f6:	4762                	lw	a4,24(sp)
  f8:	4292                	lw	t0,4(sp)
  fa:	b759                	j	80 <.L3>

000000fc <.L1>:
  fc:	50f6                	lw	ra,124(sp)
  fe:	5466                	lw	s0,120(sp)
 100:	54d6                	lw	s1,116(sp)
 102:	5946                	lw	s2,112(sp)
 104:	59b6                	lw	s3,108(sp)
 106:	5a26                	lw	s4,104(sp)
 108:	5a96                	lw	s5,100(sp)
 10a:	5b06                	lw	s6,96(sp)
 10c:	4bf6                	lw	s7,92(sp)
 10e:	4c66                	lw	s8,88(sp)
 110:	4cd6                	lw	s9,84(sp)
 112:	4d46                	lw	s10,80(sp)
 114:	4db6                	lw	s11,76(sp)
 116:	6109                	addi	sp,sp,128
 118:	8082                	ret

GELU_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.GELU_fp32_fp32 00000094  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.GELU_fp32_fp32_sigmoid 00000066  00000000  00000000  000000c8  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .text.GELU_fp32_fp32_sigmoid_grad_chunk 000000b4  00000000  00000000  0000012e  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  6 .rodata.GELU_fp32_fp32.cst4 00000014  00000000  00000000  000001e4  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  7 .rodata.GELU_fp32_fp32_sigmoid.cst4 00000004  00000000  00000000  000001f8  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  8 .rodata.GELU_fp32_fp32_sigmoid_grad_chunk.cst4 0000000c  00000000  00000000  000001fc  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  9 .comment      0000001b  00000000  00000000  00000208  2**0
                  CONTENTS, READONLY
 10 .Pulp_Chip.Info 00000072  00000000  00000000  00000223  2**0
                  CONTENTS, READONLY

Disassembly of section .text.GELU_fp32_fp32:

00000000 <GELU_fp32_fp32>:
   0:	7139                	addi	sp,sp,-64
   2:	000007b7          	lui	a5,0x0
   6:	d452                	sw	s4,40(sp)
   8:	0007aa03          	lw	s4,0(a5) # 0 <GELU_fp32_fp32>
   c:	000007b7          	lui	a5,0x0
  10:	d256                	sw	s5,36(sp)
  12:	0007aa83          	lw	s5,0(a5) # 0 <GELU_fp32_fp32>
  16:	000007b7          	lui	a5,0x0
  1a:	d05a                	sw	s6,32(sp)
  1c:	0007ab03          	lw	s6,0(a5) # 0 <GELU_fp32_fp32>
  20:	000007b7          	lui	a5,0x0
  24:	ce5e                	sw	s7,28(sp)
  26:	0007ab83          	lw	s7,0(a5) # 0 <GELU_fp32_fp32>
  2a:	dc22                	sw	s0,56(sp)
  2c:	d84a                	sw	s2,48(sp)
  2e:	d64e                	sw	s3,44(sp)
  30:	de06                	sw	ra,60(sp)
  32:	da26                	sw	s1,52(sp)
  34:	892a                	mv	s2,a0
  36:	89ae                	mv	s3,a1
  38:	4401                	li	s0,0

0000003a <.L2>:
  3a:	00c44d63          	blt	s0,a2,54 <.L3>
  3e:	50f2                	lw	ra,60(sp)
  40:	5462                	lw	s0,56(sp)
  42:	54d2                	lw	s1,52(sp)
  44:	5942                	lw	s2,48(sp)
  46:	59b2                	lw	s3,44(sp)
  48:	5a22                	lw	s4,40(sp)
  4a:	5a92                	lw	s5,36(sp)
  4c:	5b02                	lw	s6,32(sp)
  4e:	4bf2                	lw	s7,28(sp)
  50:	6121                	addi	sp,sp,64
  52:	8082                	ret

00000054 <.L3>:
  54:	0049248b          	p.lw	s1,4(s2!)
  58:	85d2                	mv	a1,s4
  5a:	c632                	sw	a2,12(sp)
  5c:	8526                	mv	a0,s1
  5e:	0405                	addi	s0,s0,1
  60:	00000097          	auipc	ra,0x0
  64:	000080e7          	jalr	ra
  68:	49557543          	fmadd.s	fa0,fa0,fs5,fs1
  6c:	11657553          	fmul.s	fa0,fa0,fs6
  70:	00000097          	auipc	ra,0x0
  74:	000080e7          	jalr	ra
  78:	01757553          	fadd.s	fa0,fa0,fs7
  7c:	000007b7          	lui	a5,0x0
  80:	0007a783          	lw	a5,0(a5) # 0 <GELU_fp32_fp32>
  84:	4632                	lw	a2,12(sp)
  86:	10f57553          	fmul.s	fa0,fa0,fa5
  8a:	10957553          	fmul.s	fa0,fa0,fs1
  8e:	00a9a22b          	p.sw	a0,4(s3!)
  92:	b765                	j	3a <.L2>

Disassembly of section .text.GELU_fp32_fp32_sigmoid:

00000000 <GELU_fp32_fp32_sigmoid>:
   0:	7179                	addi	sp,sp,-48
   2:	000007b7          	lui	a5,0x0
   6:	cc52                	sw	s4,24(sp)
   8:	0007aa03          	lw	s4,0(a5) # 0 <GELU_fp32_fp32_sigmoid>
   c:	000007b7          	lui	a5,0x0
  10:	d04a                	sw	s2,32(sp)
  12:	0007a903          	lw	s2,0(a5) # 0 <GELU_fp32_fp32_sigmoid>
  16:	d422                	sw	s0,40(sp)
  18:	d226                	sw	s1,36(sp)
  1a:	d606                	sw	ra,44(sp)
  1c:	ce4e                	sw	s3,28(sp)
  1e:	84aa                	mv	s1,a0
  20:	4401                	li	s0,0

00000022 <.L6>:
  22:	00c44a63          	blt	s0,a2,36 <.L7>
  26:	50b2                	lw	ra,44(sp)
  28:	5422                	lw	s0,40(sp)
  2a:	5492                	lw	s1,36(sp)
  2c:	5902                	lw	s2,32(sp)
  2e:	49f2                	lw	s3,28(sp)
  30:	4a62                	lw	s4,24(sp)
  32:	6145                	addi	sp,sp,48
  34:	8082                	ret

00000036 <.L7>:
  36:	0044a98b          	p.lw	s3,4(s1!)
  3a:	c62e                	sw	a1,12(sp)
  3c:	c432                	sw	a2,8(sp)
  3e:	1149f553          	fmul.s	fa0,fs3,fs4
  42:	0405                	addi	s0,s0,1
  44:	20a51553          	fneg.s	fa0,fa0
  48:	00000097          	auipc	ra,0x0
  4c:	000080e7          	jalr	ra
  50:	01257553          	fadd.s	fa0,fa0,fs2
  54:	45b2                	lw	a1,12(sp)
  56:	4622                	lw	a2,8(sp)
  58:	18a97553          	fdiv.s	fa0,fs2,fa0
  5c:	11357553          	fmul.s	fa0,fa0,fs3
  60:	00a5a22b          	p.sw	a0,4(a1!)
  64:	bf7d                	j	22 <.L6>

Disassembly of section .text.GELU_fp32_fp32_sigmoid_grad_chunk:

00000000 <GELU_fp32_fp32_sigmoid_grad_chunk>:
   0:	7139                	addi	sp,sp,-64
   2:	00269793          	slli	a5,a3,0x2
   6:	d84a                	sw	s2,48(sp)
   8:	d64e                	sw	s3,44(sp)
   a:	d452                	sw	s4,40(sp)
   c:	00f589b3          	add	s3,a1,a5
  10:	00f50a33          	add	s4,a0,a5
  14:	00f60933          	add	s2,a2,a5
  18:	000007b7          	lui	a5,0x0
  1c:	d05a                	sw	s6,32(sp)
  1e:	0007ab03          	lw	s6,0(a5) # 0 <GELU_fp32_fp32_sigmoid_grad_chunk>
  22:	000007b7          	lui	a5,0x0
  26:	d256                	sw	s5,36(sp)
  28:	0007aa83          	lw	s5,0(a5) # 0 <GELU_fp32_fp32_sigmoid_grad_chunk>
  2c:	000007b7          	lui	a5,0x0
  30:	ce5e                	sw	s7,28(sp)
  32:	0007ab83          	lw	s7,0(a5) # 0 <GELU_fp32_fp32_sigmoid_grad_chunk>
  36:	000007b7          	lui	a5,0x0
  3a:	cc62                	sw	s8,24(sp)
  3c:	0007ac03          	lw	s8,0(a5) # 0 <GELU_fp32_fp32_sigmoid_grad_chunk>
  40:	de06                	sw	ra,60(sp)
  42:	dc22                	sw	s0,56(sp)
  44:	da26                	sw	s1,52(sp)

00000046 <.L10>:
  46:	00e6ce63          	blt	a3,a4,62 <.L11>
  4a:	50f2                	lw	ra,60(sp)
  4c:	5462                	lw	s0,56(sp)
  4e:	54d2                	lw	s1,52(sp)
  50:	5942                	lw	s2,48(sp)
  52:	59b2                	lw	s3,44(sp)
  54:	5a22                	lw	s4,40(sp)
  56:	5a92                	lw	s5,36(sp)
  58:	5b02                	lw	s6,32(sp)
  5a:	4bf2                	lw	s7,28(sp)
  5c:	4c62                	lw	s8,24(sp)
  5e:	6121                	addi	sp,sp,64
  60:	8082                	ret

00000062 <.L11>:
  62:	0049a40b          	p.lw	s0,4(s3!)
  66:	c636                	sw	a3,12(sp)
  68:	c43a                	sw	a4,8(sp)
  6a:	11647553          	fmul.s	fa0,fs0,fs6
  6e:	00000097          	auipc	ra,0x0
  72:	000080e7          	jalr	ra
  76:	84aa                	mv	s1,a0
  78:	11547553          	fmul.s	fa0,fs0,fs5
  7c:	10857553          	fmul.s	fa0,fa0,fs0
  80:	00000097          	auipc	ra,0x0
  84:	000080e7          	jalr	ra
  88:	10857453          	fmul.s	fs0,fa0,fs0
  8c:	0174f7d3          	fadd.s	fa5,fs1,fs7
  90:	000005b7          	lui	a1,0x0
  94:	0005a503          	lw	a0,0(a1) # 0 <GELU_fp32_fp32_sigmoid_grad_chunk>
  98:	004a260b          	p.lw	a2,4(s4!)
  9c:	11847453          	fmul.s	fs0,fs0,fs8
  a0:	46b2                	lw	a3,12(sp)
  a2:	4722                	lw	a4,8(sp)
  a4:	0685                	addi	a3,a3,1
  a6:	40a7f443          	fmadd.s	fs0,fa5,fa0,fs0
  aa:	10c47453          	fmul.s	fs0,fs0,fa2
  ae:	0089222b          	p.sw	s0,4(s2!)
  b2:	bf51                	j	46 <.L10>

GELU_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.GELU_s8_s32 00000048  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000007c  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000097  2**0
                  CONTENTS, READONLY

Disassembly of section .text.GELU_s8_s32:

00000000 <GELU_s8_s32>:
   0:	8e2a                	mv	t3,a0
   2:	40d00eb3          	neg	t4,a3

00000006 <.L2>:
   6:	40ae0833          	sub	a6,t3,a0
   a:	00c84363          	blt	a6,a2,10 <.L3>
   e:	8082                	ret

00000010 <.L3>:
  10:	001e080b          	p.lb	a6,1(t3!)
  14:	8f3a                	mv	t5,a4
  16:	983e                	add	a6,a6,a5
  18:	01f85893          	srli	a7,a6,0x1f
  1c:	01002333          	sgtz	t1,a6
  20:	41130333          	sub	t1,t1,a7
  24:	026808b3          	mul	a7,a6,t1
  28:	051ec8b3          	p.min	a7,t4,a7
  2c:	98b6                	add	a7,a7,a3
  2e:	43189f33          	p.msu	t5,a7,a7
  32:	88fa                	mv	a7,t5
  34:	8f3a                	mv	t5,a4
  36:	42688f33          	p.mac	t5,a7,t1
  3a:	401f5893          	srai	a7,t5,0x1
  3e:	03088833          	mul	a6,a7,a6
  42:	0105a22b          	p.sw	a6,4(a1!)
  46:	b7c1                	j	6 <.L2>

Gemm_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Gemm_fp32_fp32_fp32_fp32 0000009a  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  000000ce  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  000000e9  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Gemm_fp32_fp32_fp32_fp32:

00000000 <Gemm_fp32_fp32_fp32_fp32>:
   0:	1101                	addi	sp,sp,-32
   2:	ce22                	sw	s0,28(sp)
   4:	c652                	sw	s4,12(sp)
   6:	cc26                	sw	s1,24(sp)
   8:	ca4a                	sw	s2,20(sp)
   a:	c84e                	sw	s3,16(sp)
   c:	c456                	sw	s5,8(sp)
   e:	c25a                	sw	s6,4(sp)
  10:	c05e                	sw	s7,0(sp)
  12:	00281a13          	slli	s4,a6,0x2
  16:	4f81                	li	t6,0
  18:	4401                	li	s0,0
  1a:	4281                	li	t0,0

0000001c <.L2>:
  1c:	06e28563          	beq	t0,a4,86 <.L1>
  20:	01f609b3          	add	s3,a2,t6
  24:	01f68933          	add	s2,a3,t6
  28:	4381                	li	t2,0
  2a:	4f01                	li	t5,0
  2c:	a089                	j	6e <.L8>

0000002e <.L5>:
  2e:	8e56                	mv	t3,s5
  30:	00089463          	bnez	a7,38 <.L3>
  34:	008e8e33          	add	t3,t4,s0

00000038 <.L3>:
  38:	5b82                	lw	s7,32(sp)
  3a:	835a                	mv	t1,s6
  3c:	000b8463          	beqz	s7,44 <.L4>
  40:	007e8333          	add	t1,t4,t2

00000044 <.L4>:
  44:	0e0a                	slli	t3,t3,0x2
  46:	030a                	slli	t1,t1,0x2
  48:	21c57e03          	p.lw	t3,t3(a0)
  4c:	2065f303          	p.lw	t1,t1(a1)
  50:	0e85                	addi	t4,t4,1
  52:	9b42                	add	s6,s6,a6
  54:	486e74c3          	fmadd.s	fs1,ft8,ft6,fs1
  58:	9aba                	add	s5,s5,a4

0000005a <.L6>:
  5a:	fcfe9ae3          	bne	t4,a5,2e <.L5>
  5e:	0049a30b          	p.lw	t1,4(s3!)
  62:	0f05                	addi	t5,t5,1
  64:	93be                	add	t2,t2,a5
  66:	00937353          	fadd.s	ft6,ft6,fs1
  6a:	0069222b          	p.sw	t1,4(s2!)

0000006e <.L8>:
  6e:	010f1663          	bne	t5,a6,7a <.L11>
  72:	0285                	addi	t0,t0,1
  74:	943e                	add	s0,s0,a5
  76:	9fd2                	add	t6,t6,s4
  78:	b755                	j	1c <.L2>

0000007a <.L11>:
  7a:	8a96                	mv	s5,t0
  7c:	8b7a                	mv	s6,t5
  7e:	4e81                	li	t4,0
  80:	00000493          	li	s1,0
  84:	bfd9                	j	5a <.L6>

00000086 <.L1>:
  86:	4472                	lw	s0,28(sp)
  88:	44e2                	lw	s1,24(sp)
  8a:	4952                	lw	s2,20(sp)
  8c:	49c2                	lw	s3,16(sp)
  8e:	4a32                	lw	s4,12(sp)
  90:	4aa2                	lw	s5,8(sp)
  92:	4b12                	lw	s6,4(sp)
  94:	4b82                	lw	s7,0(sp)
  96:	6105                	addi	sp,sp,32
  98:	8082                	ret

Gemm_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Gemm_s8_s8_s32_s32 000001d2  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  00000206  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000221  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Gemm_s8_s8_s32_s32:

00000000 <Gemm_s8_s8_s32_s32>:
   0:	7179                	addi	sp,sp,-48
   2:	d622                	sw	s0,44(sp)
   4:	5452                	lw	s0,52(sp)
   6:	d426                	sw	s1,40(sp)
   8:	d24a                	sw	s2,36(sp)
   a:	d04e                	sw	s3,32(sp)
   c:	ce52                	sw	s4,28(sp)
   e:	cc56                	sw	s5,24(sp)
  10:	ca5a                	sw	s6,20(sp)
  12:	c85e                	sw	s7,16(sp)
  14:	c662                	sw	s8,12(sp)
  16:	c466                	sw	s9,8(sp)
  18:	c26a                	sw	s10,4(sp)
  1a:	5e42                	lw	t3,48(sp)
  1c:	5362                	lw	t1,56(sp)
  1e:	5ef2                	lw	t4,60(sp)
  20:	4f06                	lw	t5,64(sp)
  22:	4f96                	lw	t6,68(sp)
  24:	42a6                	lw	t0,72(sp)
  26:	00281393          	slli	t2,a6,0x2
  2a:	e03d                	bnez	s0,90 <.L2>
  2c:	0e031063          	bnez	t1,10c <.L3>
  30:	40a009b3          	neg	s3,a0
  34:	4481                	li	s1,0
  36:	4901                	li	s2,0

00000038 <.L4>:
  38:	18e90063          	beq	s2,a4,1b8 <.L1>
  3c:	00960b33          	add	s6,a2,s1
  40:	00968ab3          	add	s5,a3,s1
  44:	4401                	li	s0,0
  46:	a805                	j	76 <.L9>

00000048 <.L5>:
  48:	001a0b8b          	p.lb	s7,1(s4!)
  4c:	010cfc0b          	p.lb	s8,a6(s9!)
  50:	9bf6                	add	s7,s7,t4
  52:	9c7a                	add	s8,s8,t5
  54:	437c0333          	p.mac	t1,s8,s7

00000058 <.L7>:
  58:	013a0bb3          	add	s7,s4,s3
  5c:	fefbe6e3          	bltu	s7,a5,48 <.L5>
  60:	004b2a0b          	p.lw	s4,4(s6!)
  64:	0405                	addi	s0,s0,1
  66:	9a7e                	add	s4,s4,t6
  68:	03130333          	mul	t1,t1,a7
  6c:	43ca0333          	p.mac	t1,s4,t3
  70:	9316                	add	t1,t1,t0
  72:	006aa22b          	p.sw	t1,4(s5!)

00000076 <.L9>:
  76:	01040763          	beq	s0,a6,84 <.L6>
  7a:	00858cb3          	add	s9,a1,s0
  7e:	8a2a                	mv	s4,a0
  80:	4301                	li	t1,0
  82:	bfd9                	j	58 <.L7>

00000084 <.L6>:
  84:	0905                	addi	s2,s2,1
  86:	953e                	add	a0,a0,a5
  88:	40f989b3          	sub	s3,s3,a5
  8c:	949e                	add	s1,s1,t2
  8e:	b76d                	j	38 <.L4>

00000090 <.L2>:
  90:	02142063          	p.beqimm	s0,1,b0 <.L10>

00000094 <.L12>:
  94:	00e50bb3          	add	s7,a0,a4
  98:	4401                	li	s0,0

0000009a <.L11>:
  9a:	11750f63          	beq	a0,s7,1b8 <.L1>
  9e:	00860b33          	add	s6,a2,s0
  a2:	00868ab3          	add	s5,a3,s0
  a6:	89ae                	mv	s3,a1
  a8:	40b00933          	neg	s2,a1
  ac:	4481                	li	s1,0
  ae:	a8e5                	j	1a6 <.L24>

000000b0 <.L10>:
  b0:	fe0312e3          	bnez	t1,94 <.L12>
  b4:	00e50b33          	add	s6,a0,a4
  b8:	4481                	li	s1,0

000000ba <.L13>:
  ba:	0f650f63          	beq	a0,s6,1b8 <.L1>
  be:	00960ab3          	add	s5,a2,s1
  c2:	00968a33          	add	s4,a3,s1
  c6:	4401                	li	s0,0
  c8:	a03d                	j	f6 <.L17>

000000ca <.L14>:
  ca:	00ec7b8b          	p.lb	s7,a4(s8!)
  ce:	010cf98b          	p.lb	s3,a6(s9!)
  d2:	0905                	addi	s2,s2,1
  d4:	9bf6                	add	s7,s7,t4
  d6:	99fa                	add	s3,s3,t5
  d8:	433b8333          	p.mac	t1,s7,s3

000000dc <.L16>:
  dc:	fef917e3          	bne	s2,a5,ca <.L14>
  e0:	004aa90b          	p.lw	s2,4(s5!)
  e4:	0405                	addi	s0,s0,1
  e6:	997e                	add	s2,s2,t6
  e8:	03130333          	mul	t1,t1,a7
  ec:	43c90333          	p.mac	t1,s2,t3
  f0:	9316                	add	t1,t1,t0
  f2:	006a222b          	p.sw	t1,4(s4!)

000000f6 <.L17>:
  f6:	01040863          	beq	s0,a6,106 <.L15>
  fa:	00858cb3          	add	s9,a1,s0
  fe:	8c2a                	mv	s8,a0
 100:	4301                	li	t1,0
 102:	4901                	li	s2,0
 104:	bfe1                	j	dc <.L16>

00000106 <.L15>:
 106:	0505                	addi	a0,a0,1
 108:	949e                	add	s1,s1,t2
 10a:	bf45                	j	ba <.L13>

0000010c <.L3>:
 10c:	f81334e3          	p.bneimm	t1,1,94 <.L12>
 110:	40a00a33          	neg	s4,a0
 114:	4401                	li	s0,0
 116:	4981                	li	s3,0

00000118 <.L18>:
 118:	0ae98063          	beq	s3,a4,1b8 <.L1>
 11c:	00860bb3          	add	s7,a2,s0
 120:	00868b33          	add	s6,a3,s0
 124:	892e                	mv	s2,a1
 126:	4481                	li	s1,0
 128:	a80d                	j	15a <.L21>

0000012a <.L19>:
 12a:	001a8c8b          	p.lb	s9,1(s5!)
 12e:	001d0c0b          	p.lb	s8,1(s10!)
 132:	9cf6                	add	s9,s9,t4
 134:	9c7a                	add	s8,s8,t5
 136:	438c8333          	p.mac	t1,s9,s8

0000013a <.L20>:
 13a:	014a8c33          	add	s8,s5,s4
 13e:	fefc66e3          	bltu	s8,a5,12a <.L19>
 142:	004baa8b          	p.lw	s5,4(s7!)
 146:	0485                	addi	s1,s1,1
 148:	993e                	add	s2,s2,a5
 14a:	9afe                	add	s5,s5,t6
 14c:	03130333          	mul	t1,t1,a7
 150:	43ca8333          	p.mac	t1,s5,t3
 154:	9316                	add	t1,t1,t0
 156:	006b222b          	p.sw	t1,4(s6!)

0000015a <.L21>:
 15a:	01049863          	bne	s1,a6,16a <.L25>
 15e:	0985                	addi	s3,s3,1
 160:	953e                	add	a0,a0,a5
 162:	40fa0a33          	sub	s4,s4,a5
 166:	941e                	add	s0,s0,t2
 168:	bf45                	j	118 <.L18>

0000016a <.L25>:
 16a:	8d4a                	mv	s10,s2
 16c:	8aaa                	mv	s5,a0
 16e:	4301                	li	t1,0
 170:	b7e9                	j	13a <.L20>

00000172 <.L22>:
 172:	001a0c0b          	p.lb	s8,1(s4!)
 176:	00ed7c8b          	p.lb	s9,a4(s10!)
 17a:	9c7a                	add	s8,s8,t5
 17c:	9cf6                	add	s9,s9,t4
 17e:	438c8333          	p.mac	t1,s9,s8

00000182 <.L23>:
 182:	012a0c33          	add	s8,s4,s2
 186:	fefc66e3          	bltu	s8,a5,172 <.L22>
 18a:	004b2a0b          	p.lw	s4,4(s6!)
 18e:	0485                	addi	s1,s1,1
 190:	99be                	add	s3,s3,a5
 192:	9a7e                	add	s4,s4,t6
 194:	40f90933          	sub	s2,s2,a5
 198:	03130333          	mul	t1,t1,a7
 19c:	43ca0333          	p.mac	t1,s4,t3
 1a0:	9316                	add	t1,t1,t0
 1a2:	006aa22b          	p.sw	t1,4(s5!)

000001a6 <.L24>:
 1a6:	01049563          	bne	s1,a6,1b0 <.L26>
 1aa:	0505                	addi	a0,a0,1
 1ac:	941e                	add	s0,s0,t2
 1ae:	b5f5                	j	9a <.L11>

000001b0 <.L26>:
 1b0:	8a4e                	mv	s4,s3
 1b2:	8d2a                	mv	s10,a0
 1b4:	4301                	li	t1,0
 1b6:	b7f1                	j	182 <.L23>

000001b8 <.L1>:
 1b8:	5432                	lw	s0,44(sp)
 1ba:	54a2                	lw	s1,40(sp)
 1bc:	5912                	lw	s2,36(sp)
 1be:	5982                	lw	s3,32(sp)
 1c0:	4a72                	lw	s4,28(sp)
 1c2:	4ae2                	lw	s5,24(sp)
 1c4:	4b52                	lw	s6,20(sp)
 1c6:	4bc2                	lw	s7,16(sp)
 1c8:	4c32                	lw	s8,12(sp)
 1ca:	4ca2                	lw	s9,8(sp)
 1cc:	4d12                	lw	s10,4(sp)
 1ce:	6145                	addi	sp,sp,48
 1d0:	8082                	ret

Hardswish_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.iHardswish_s8_s32 0000002c  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  00000060  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  0000007b  2**0
                  CONTENTS, READONLY

Disassembly of section .text.iHardswish_s8_s32:

00000000 <iHardswish_s8_s32>:
   0:	8e2a                	mv	t3,a0

00000002 <.L2>:
   2:	40ae08b3          	sub	a7,t3,a0
   6:	00c8c363          	blt	a7,a2,c <.L3>
   a:	8082                	ret

0000000c <.L3>:
   c:	001e030b          	p.lb	t1,1(t3!)
  10:	010308b3          	add	a7,t1,a6
  14:	98ba                	add	a7,a7,a4
  16:	0408e8b3          	p.max	a7,a7,zero
  1a:	0517c8b3          	p.min	a7,a5,a7
  1e:	031688b3          	mul	a7,a3,a7
  22:	031308b3          	mul	a7,t1,a7
  26:	0115a22b          	p.sw	a7,4(a1!)
  2a:	bfe1                	j	2 <.L2>

Layernorm_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Layernorm_fp32_fp32 000000c4  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.LayernormGrad_fp32_fp32 00000130  00000000  00000000  000000f8  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .rodata.LayernormGrad_fp32_fp32.cst4 00000004  00000000  00000000  00000228  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  6 .comment      0000001b  00000000  00000000  0000022c  2**0
                  CONTENTS, READONLY
  7 .Pulp_Chip.Info 00000072  00000000  00000000  00000247  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Layernorm_fp32_fp32:

00000000 <Layernorm_fp32_fp32>:
   0:	7139                	addi	sp,sp,-64
   2:	d452                	sw	s4,40(sp)
   4:	0307ca33          	div	s4,a5,a6
   8:	da26                	sw	s1,52(sp)
   a:	d84a                	sw	s2,48(sp)
   c:	d64e                	sw	s3,44(sp)
   e:	de06                	sw	ra,60(sp)
  10:	dc22                	sw	s0,56(sp)
  12:	00281993          	slli	s3,a6,0x2
  16:	84aa                	mv	s1,a0
  18:	4901                	li	s2,0

0000001a <.L2>:
  1a:	07494863          	blt	s2,s4,8a <.L9>
  1e:	50f2                	lw	ra,60(sp)
  20:	5462                	lw	s0,56(sp)
  22:	54d2                	lw	s1,52(sp)
  24:	5942                	lw	s2,48(sp)
  26:	59b2                	lw	s3,44(sp)
  28:	5a22                	lw	s4,40(sp)
  2a:	6121                	addi	sp,sp,64
  2c:	8082                	ret

0000002e <.L3>:
  2e:	0047a30b          	p.lw	t1,4(a5!)
  32:	0505                	addi	a0,a0,1
  34:	0068f8d3          	fadd.s	fa7,fa7,ft6

00000038 <.L8>:
  38:	ff054be3          	blt	a0,a6,2e <.L3>
  3c:	d00877d3          	fcvt.s.w	fa5,a6
  40:	00000513          	li	a0,0
  44:	8e26                	mv	t3,s1
  46:	18f8f453          	fdiv.s	fs0,fa7,fa5
  4a:	4301                	li	t1,0

0000004c <.L4>:
  4c:	05034463          	blt	t1,a6,94 <.L5>
  50:	ce2e                	sw	a1,28(sp)
  52:	cc32                	sw	a2,24(sp)
  54:	ca36                	sw	a3,20(sp)
  56:	c83a                	sw	a4,16(sp)
  58:	c642                	sw	a6,12(sp)
  5a:	18f57553          	fdiv.s	fa0,fa0,fa5
  5e:	00e57553          	fadd.s	fa0,fa0,fa4
  62:	00000097          	auipc	ra,0x0
  66:	000080e7          	jalr	ra
  6a:	4662                	lw	a2,24(sp)
  6c:	46d2                	lw	a3,20(sp)
  6e:	45f2                	lw	a1,28(sp)
  70:	4742                	lw	a4,16(sp)
  72:	4832                	lw	a6,12(sp)
  74:	8e32                	mv	t3,a2
  76:	8f36                	mv	t5,a3
  78:	87ae                	mv	a5,a1
  7a:	8ea6                	mv	t4,s1
  7c:	4301                	li	t1,0

0000007e <.L6>:
  7e:	03034363          	blt	t1,a6,a4 <.L7>
  82:	0905                	addi	s2,s2,1
  84:	94ce                	add	s1,s1,s3
  86:	95ce                	add	a1,a1,s3
  88:	bf49                	j	1a <.L2>

0000008a <.L9>:
  8a:	87a6                	mv	a5,s1
  8c:	4501                	li	a0,0
  8e:	00000893          	li	a7,0
  92:	b75d                	j	38 <.L8>

00000094 <.L5>:
  94:	004e288b          	p.lw	a7,4(t3!)
  98:	0305                	addi	t1,t1,1
  9a:	0888f8d3          	fsub.s	fa7,fa7,fs0
  9e:	5118f543          	fmadd.s	fa0,fa7,fa7,fa0
  a2:	b76d                	j	4c <.L4>

000000a4 <.L7>:
  a4:	004ea88b          	p.lw	a7,4(t4!)
  a8:	004e228b          	p.lw	t0,4(t3!)
  ac:	004f2f8b          	p.lw	t6,4(t5!)
  b0:	0888f8d3          	fsub.s	fa7,fa7,fs0
  b4:	0305                	addi	t1,t1,1
  b6:	18a8f8d3          	fdiv.s	fa7,fa7,fa0
  ba:	f858f8c3          	fmadd.s	fa7,fa7,ft5,ft11
  be:	0117a22b          	p.sw	a7,4(a5!)
  c2:	bf75                	j	7e <.L6>

Disassembly of section .text.LayernormGrad_fp32_fp32:

00000000 <LayernormGrad_fp32_fp32>:
   0:	715d                	addi	sp,sp,-80
   2:	da56                	sw	s5,52(sp)
   4:	03184ab3          	div	s5,a6,a7
   8:	00000737          	lui	a4,0x0
   c:	d85a                	sw	s6,48(sp)
   e:	00072b03          	lw	s6,0(a4) # 0 <LayernormGrad_fp32_fp32>
  12:	c2a6                	sw	s1,68(sp)
  14:	de4e                	sw	s3,60(sp)
  16:	dc52                	sw	s4,56(sp)
  18:	c686                	sw	ra,76(sp)
  1a:	c4a2                	sw	s0,72(sp)
  1c:	c0ca                	sw	s2,64(sp)
  1e:	d65e                	sw	s7,44(sp)
  20:	d462                	sw	s8,40(sp)
  22:	00289a13          	slli	s4,a7,0x2
  26:	84aa                	mv	s1,a0
  28:	4981                	li	s3,0

0000002a <.L12>:
  2a:	0959c863          	blt	s3,s5,ba <.L21>
  2e:	40b6                	lw	ra,76(sp)
  30:	4426                	lw	s0,72(sp)
  32:	4496                	lw	s1,68(sp)
  34:	4906                	lw	s2,64(sp)
  36:	59f2                	lw	s3,60(sp)
  38:	5a62                	lw	s4,56(sp)
  3a:	5ad2                	lw	s5,52(sp)
  3c:	5b42                	lw	s6,48(sp)
  3e:	5bb2                	lw	s7,44(sp)
  40:	5c22                	lw	s8,40(sp)
  42:	6161                	addi	sp,sp,80
  44:	8082                	ret

00000046 <.L13>:
  46:	0045280b          	p.lw	a6,4(a0!)
  4a:	0705                	addi	a4,a4,1
  4c:	01047453          	fadd.s	fs0,fs0,fa6

00000050 <.L20>:
  50:	ff174be3          	blt	a4,a7,46 <.L13>
  54:	d008f953          	fcvt.s.w	fs2,a7
  58:	00000513          	li	a0,0
  5c:	832e                	mv	t1,a1
  5e:	19247453          	fdiv.s	fs0,fs0,fs2
  62:	4801                	li	a6,0

00000064 <.L14>:
  64:	07184063          	blt	a6,a7,c4 <.L15>
  68:	ce2e                	sw	a1,28(sp)
  6a:	cc32                	sw	a2,24(sp)
  6c:	ca36                	sw	a3,20(sp)
  6e:	c83e                	sw	a5,16(sp)
  70:	c646                	sw	a7,12(sp)
  72:	19257553          	fdiv.s	fa0,fa0,fs2
  76:	00f57553          	fadd.s	fa0,fa0,fa5
  7a:	00000097          	auipc	ra,0x0
  7e:	000080e7          	jalr	ra
  82:	18ab7553          	fdiv.s	fa0,fs6,fa0
  86:	00000e13          	li	t3,0
  8a:	46d2                	lw	a3,20(sp)
  8c:	45f2                	lw	a1,28(sp)
  8e:	8372                	mv	t1,t3
  90:	4662                	lw	a2,24(sp)
  92:	47c2                	lw	a5,16(sp)
  94:	48b2                	lw	a7,12(sp)
  96:	8eb6                	mv	t4,a3
  98:	83b6                	mv	t2,a3
  9a:	82ae                	mv	t0,a1
  9c:	8fa6                	mv	t6,s1
  9e:	4801                	li	a6,0

000000a0 <.L16>:
  a0:	03184a63          	blt	a6,a7,d4 <.L17>
  a4:	82b2                	mv	t0,a2
  a6:	8f26                	mv	t5,s1
  a8:	8fae                	mv	t6,a1
  aa:	4801                	li	a6,0

000000ac <.L18>:
  ac:	05184663          	blt	a6,a7,f8 <.L19>
  b0:	0985                	addi	s3,s3,1
  b2:	95d2                	add	a1,a1,s4
  b4:	94d2                	add	s1,s1,s4
  b6:	9652                	add	a2,a2,s4
  b8:	bf8d                	j	2a <.L12>

000000ba <.L21>:
  ba:	852e                	mv	a0,a1
  bc:	4701                	li	a4,0
  be:	00000413          	li	s0,0
  c2:	b779                	j	50 <.L20>

000000c4 <.L15>:
  c4:	0043270b          	p.lw	a4,4(t1!)
  c8:	0805                	addi	a6,a6,1
  ca:	08877753          	fsub.s	fa4,fa4,fs0
  ce:	50e77543          	fmadd.s	fa0,fa4,fa4,fa0
  d2:	bf49                	j	64 <.L14>

000000d4 <.L17>:
  d4:	004fa70b          	p.lw	a4,4(t6!)
  d8:	0042af0b          	p.lw	t5,4(t0!)
  dc:	0043ab8b          	p.lw	s7,4(t2!)
  e0:	00e37353          	fadd.s	ft6,ft6,fa4
  e4:	088f7f53          	fsub.s	ft10,ft10,fs0
  e8:	11777753          	fmul.s	fa4,fa4,fs7
  ec:	0805                	addi	a6,a6,1
  ee:	11e77753          	fmul.s	fa4,fa4,ft10
  f2:	e0a77e43          	fmadd.s	ft8,fa4,fa0,ft8
  f6:	b76d                	j	a0 <.L16>

000000f8 <.L19>:
  f8:	004fa70b          	p.lw	a4,4(t6!)
  fc:	19237c53          	fdiv.s	fs8,ft6,fs2
 100:	004f2b8b          	p.lw	s7,4(t5!)
 104:	08877753          	fsub.s	fa4,fa4,fs0
 108:	004ea38b          	p.lw	t2,4(t4!)
 10c:	0805                	addi	a6,a6,1
 10e:	10a3f3d3          	fmul.s	ft7,ft7,fa0
 112:	10a77753          	fmul.s	fa4,fa4,fa0
 116:	10a77753          	fmul.s	fa4,fa4,fa0
 11a:	19277753          	fdiv.s	fa4,fa4,fs2
 11e:	098bfbd3          	fsub.s	fs7,fs7,fs8
 122:	b9c7774b          	fnmsub.s	fa4,fa4,ft8,fs7
 126:	10777753          	fmul.s	fa4,fa4,ft7
 12a:	00e2a22b          	p.sw	a4,4(t0!)
 12e:	bfbd                	j	ac <.L18>

Layernorm_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text._plp_sqrt_q32 00000044  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.Layernorm_s8_s8 00000172  00000000  00000000  00000078  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .comment      0000001b  00000000  00000000  000001ea  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  00000205  2**0
                  CONTENTS, READONLY

Disassembly of section .text._plp_sqrt_q32:

00000000 <_plp_sqrt_q32>:
   0:	00052883          	lw	a7,0(a0)
   4:	03105d63          	blez	a7,3e <.L2>
   8:	672d                	lui	a4,0xb
   a:	50670713          	addi	a4,a4,1286 # b506 <.L18+0xb422>
   e:	4801                	li	a6,0
  10:	4501                	li	a0,0

00000012 <.L6>:
  12:	02e827db          	p.addn	a5,a6,a4,1
  16:	02f786b3          	mul	a3,a5,a5
  1a:	40b6d6b3          	sra	a3,a3,a1
  1e:	01168e63          	beq	a3,a7,3a <.L8>
  22:	0116d963          	ble	a7,a3,34 <.L4>
  26:	00178813          	addi	a6,a5,1
  2a:	853e                	mv	a0,a5

0000002c <.L5>:
  2c:	ff0753e3          	ble	a6,a4,12 <.L6>

00000030 <.L3>:
  30:	c208                	sw	a0,0(a2)
  32:	8082                	ret

00000034 <.L4>:
  34:	fff78713          	addi	a4,a5,-1
  38:	bfd5                	j	2c <.L5>

0000003a <.L8>:
  3a:	853e                	mv	a0,a5
  3c:	bfd5                	j	30 <.L3>

0000003e <.L2>:
  3e:	00062023          	sw	zero,0(a2)
  42:	8082                	ret

Disassembly of section .text.Layernorm_s8_s8:

00000000 <Layernorm_s8_s8>:
   0:	7135                	addi	sp,sp,-160
   2:	cd22                	sw	s0,152(sp)
   4:	c94a                	sw	s2,144(sp)
   6:	c74e                	sw	s3,140(sp)
   8:	c552                	sw	s4,136(sp)
   a:	c356                	sw	s5,132(sp)
   c:	dae6                	sw	s9,116(sp)
   e:	d8ea                	sw	s10,112(sp)
  10:	d6ee                	sw	s11,108(sp)
  12:	cf06                	sw	ra,156(sp)
  14:	cb26                	sw	s1,148(sp)
  16:	c15a                	sw	s6,128(sp)
  18:	dede                	sw	s7,124(sp)
  1a:	dce2                	sw	s8,120(sp)
  1c:	89aa                	mv	s3,a0
  1e:	c4ae                	sw	a1,72(sp)
  20:	c6b2                	sw	a2,76(sp)
  22:	8cb6                	mv	s9,a3
  24:	8442                	mv	s0,a6
  26:	8a46                	mv	s4,a7
  28:	40a00ab3          	neg	s5,a0
  2c:	4901                	li	s2,0
  2e:	41f75d13          	srai	s10,a4,0x1f
  32:	41f8dd93          	srai	s11,a7,0x1f

00000036 <.L11>:
  36:	02f94163          	blt	s2,a5,58 <.L19>
  3a:	40fa                	lw	ra,156(sp)
  3c:	446a                	lw	s0,152(sp)
  3e:	44da                	lw	s1,148(sp)
  40:	494a                	lw	s2,144(sp)
  42:	49ba                	lw	s3,140(sp)
  44:	4a2a                	lw	s4,136(sp)
  46:	4a9a                	lw	s5,132(sp)
  48:	4b0a                	lw	s6,128(sp)
  4a:	5bf6                	lw	s7,124(sp)
  4c:	5c66                	lw	s8,120(sp)
  4e:	5cd6                	lw	s9,116(sp)
  50:	5d46                	lw	s10,112(sp)
  52:	5db6                	lw	s11,108(sp)
  54:	610d                	addi	sp,sp,160
  56:	8082                	ret

00000058 <.L19>:
  58:	01298c33          	add	s8,s3,s2
  5c:	cc82                	sw	zero,88(sp)
  5e:	86e2                	mv	a3,s8
  60:	4481                	li	s1,0

00000062 <.L12>:
  62:	01568633          	add	a2,a3,s5
  66:	06864163          	blt	a2,s0,c8 <.L13>
  6a:	0284c4b3          	div	s1,s1,s0
  6e:	85e2                	mv	a1,s8
  70:	4681                	li	a3,0
  72:	4601                	li	a2,0
  74:	40970533          	sub	a0,a4,s1

00000078 <.L14>:
  78:	01558333          	add	t1,a1,s5
  7c:	04834b63          	blt	t1,s0,d2 <.L15>
  80:	c291                	beqz	a3,84 <.L16>
  82:	ccb2                	sw	a2,88(sp)

00000084 <.L16>:
  84:	46e6                	lw	a3,88(sp)
  86:	08f0                	addi	a2,sp,92
  88:	4581                	li	a1,0
  8a:	0286c6b3          	div	a3,a3,s0
  8e:	08a8                	addi	a0,sp,88
  90:	c43a                	sw	a4,8(sp)
  92:	c23e                	sw	a5,4(sp)
  94:	0685                	addi	a3,a3,1
  96:	ccb6                	sw	a3,88(sp)
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	4f76                	lw	t5,92(sp)
  a2:	4626                	lw	a2,72(sp)
  a4:	4336                	lw	t1,76(sp)
  a6:	4722                	lw	a4,8(sp)
  a8:	4792                	lw	a5,4(sp)
  aa:	41ff5293          	srai	t0,t5,0x1f
  ae:	8e66                	mv	t3,s9
  b0:	01260eb3          	add	t4,a2,s2
  b4:	41f4d393          	srai	t2,s1,0x1f

000000b8 <.L17>:
  b8:	015c06b3          	add	a3,s8,s5
  bc:	0286c463          	blt	a3,s0,e4 <.L18>
  c0:	9922                	add	s2,s2,s0
  c2:	408a8ab3          	sub	s5,s5,s0
  c6:	bf85                	j	36 <.L11>

000000c8 <.L13>:
  c8:	0016860b          	p.lb	a2,1(a3!)
  cc:	963a                	add	a2,a2,a4
  ce:	94b2                	add	s1,s1,a2
  d0:	bf49                	j	62 <.L12>

000000d2 <.L15>:
  d2:	0015868b          	p.lb	a3,1(a1!)
  d6:	96aa                	add	a3,a3,a0
  d8:	1006c6b3          	p.exths	a3,a3
  dc:	42d68633          	p.mac	a2,a3,a3
  e0:	4685                	li	a3,1
  e2:	bf59                	j	78 <.L14>

000000e4 <.L18>:
  e4:	001c068b          	p.lb	a3,1(s8!)
  e8:	d83a                	sw	a4,48(sp)
  ea:	da6a                	sw	s10,52(sp)
  ec:	d436                	sw	a3,40(sp)
  ee:	86fd                	srai	a3,a3,0x1f
  f0:	d636                	sw	a3,44(sp)
  f2:	5b22                	lw	s6,40(sp)
  f4:	5bb2                	lw	s7,44(sp)
  f6:	5842                	lw	a6,48(sp)
  f8:	58d2                	lw	a7,52(sp)
  fa:	de1e                	sw	t2,60(sp)
  fc:	dc26                	sw	s1,56(sp)
  fe:	0043260b          	p.lw	a2,4(t1!)
 102:	210b0533          	add.d	a0,s6,a6
 106:	5b62                	lw	s6,56(sp)
 108:	5bf2                	lw	s7,60(sp)
 10a:	41f65693          	srai	a3,a2,0x1f
 10e:	004e2f8b          	p.lw	t6,4(t3!)
 112:	61650533          	sub.d	a0,a0,s6
 116:	02a686b3          	mul	a3,a3,a0
 11a:	c816                	sw	t0,16(sp)
 11c:	d01a                	sw	t1,32(sp)
 11e:	c61e                	sw	t2,12(sp)
 120:	c43a                	sw	a4,8(sp)
 122:	c23e                	sw	a5,4(sp)
 124:	ce72                	sw	t3,28(sp)
 126:	cc76                	sw	t4,24(sp)
 128:	ca7a                	sw	t5,20(sp)
 12a:	d27e                	sw	t6,36(sp)
 12c:	42b606b3          	p.mac	a3,a2,a1
 130:	72c55533          	p.mulu.d	a0,a0,a2
 134:	867a                	mv	a2,t5
 136:	95b6                	add	a1,a1,a3
 138:	8696                	mv	a3,t0
 13a:	00000097          	auipc	ra,0x0
 13e:	000080e7          	jalr	ra
 142:	5f92                	lw	t6,36(sp)
 144:	4ee2                	lw	t4,24(sp)
 146:	c0fe                	sw	t6,64(sp)
 148:	41ffdf93          	srai	t6,t6,0x1f
 14c:	c2fe                	sw	t6,68(sp)
 14e:	4b06                	lw	s6,64(sp)
 150:	4b96                	lw	s7,68(sp)
 152:	21650533          	add.d	a0,a0,s6
 156:	8b52                	mv	s6,s4
 158:	8bee                	mv	s7,s11
 15a:	61655533          	sra.d	a0,a0,s6
 15e:	00ae80ab          	p.sb	a0,1(t4!)
 162:	4f52                	lw	t5,20(sp)
 164:	4e72                	lw	t3,28(sp)
 166:	4792                	lw	a5,4(sp)
 168:	4722                	lw	a4,8(sp)
 16a:	43b2                	lw	t2,12(sp)
 16c:	5302                	lw	t1,32(sp)
 16e:	42c2                	lw	t0,16(sp)
 170:	b7a1                	j	b8 <.L17>

MatMul_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.MatMul_fp32_fp32_fp32 00000084  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.MatMul_fp32_fp32_fp32_unroll1x7 00000110  00000000  00000000  000000b8  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .comment      0000001b  00000000  00000000  000001c8  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  000001e3  2**0
                  CONTENTS, READONLY

Disassembly of section .text.MatMul_fp32_fp32_fp32:

00000000 <MatMul_fp32_fp32_fp32>:
   0:	00279293          	slli	t0,a5,0x2
   4:	00271393          	slli	t2,a4,0x2
   8:	4301                	li	t1,0

0000000a <.L9>:
   a:	04d31163          	bne	t1,a3,4c <.L15>
   e:	8082                	ret

00000010 <.L7>:
  10:	8fb2                	mv	t6,a2
  12:	4881                	li	a7,0
  14:	a8b1                	j	70 <.L6>

00000016 <.L4>:
  16:	0305                	addi	t1,t1,1
  18:	9616                	add	a2,a2,t0
  1a:	951e                	add	a0,a0,t2
  1c:	fed31ae3          	bne	t1,a3,10 <.L7>
  20:	4432                	lw	s0,12(sp)
  22:	44a2                	lw	s1,8(sp)
  24:	0141                	addi	sp,sp,16
  26:	8082                	ret

00000028 <.L17>:
  28:	00289813          	slli	a6,a7,0x2
  2c:	4e81                	li	t4,0
  2e:	982e                	add	a6,a6,a1
  30:	8f2a                	mv	t5,a0
  32:	00000e13          	li	t3,0
  36:	00ee9e63          	bne	t4,a4,52 <.L16>
  3a:	01cfa22b          	p.sw	t3,4(t6!)
  3e:	0885                	addi	a7,a7,1

00000040 <.L11>:
  40:	fef894e3          	bne	a7,a5,28 <.L17>
  44:	0305                	addi	t1,t1,1
  46:	9616                	add	a2,a2,t0
  48:	951e                	add	a0,a0,t2
  4a:	b7c1                	j	a <.L9>

0000004c <.L15>:
  4c:	8fb2                	mv	t6,a2
  4e:	4881                	li	a7,0
  50:	bfc5                	j	40 <.L11>

00000052 <.L16>:
  52:	1141                	addi	sp,sp,-16
  54:	c622                	sw	s0,12(sp)
  56:	c426                	sw	s1,8(sp)

00000058 <.L3>:
  58:	004f248b          	p.lw	s1,4(t5!)
  5c:	2058740b          	p.lw	s0,t0(a6!)
  60:	0e85                	addi	t4,t4,1
  62:	e084fe43          	fmadd.s	ft8,fs1,fs0,ft8

00000066 <.L5>:
  66:	feee99e3          	bne	t4,a4,58 <.L3>
  6a:	01cfa22b          	p.sw	t3,4(t6!)
  6e:	0885                	addi	a7,a7,1

00000070 <.L6>:
  70:	faf883e3          	beq	a7,a5,16 <.L4>
  74:	00289813          	slli	a6,a7,0x2
  78:	982e                	add	a6,a6,a1
  7a:	8f2a                	mv	t5,a0
  7c:	4e81                	li	t4,0
  7e:	00000e13          	li	t3,0
  82:	b7d5                	j	66 <.L5>

Disassembly of section .text.MatMul_fp32_fp32_fp32_unroll1x7:

00000000 <MatMul_fp32_fp32_fp32_unroll1x7>:
   0:	481d                	li	a6,7
   2:	0307f833          	remu	a6,a5,a6
   6:	7179                	addi	sp,sp,-48
   8:	d622                	sw	s0,44(sp)
   a:	d426                	sw	s1,40(sp)
   c:	d24a                	sw	s2,36(sp)
   e:	d04e                	sw	s3,32(sp)
  10:	ce52                	sw	s4,28(sp)
  12:	cc56                	sw	s5,24(sp)
  14:	ca5a                	sw	s6,20(sp)
  16:	c85e                	sw	s7,16(sp)
  18:	c662                	sw	s8,12(sp)
  1a:	c466                	sw	s9,8(sp)
  1c:	00271493          	slli	s1,a4,0x2
  20:	00279913          	slli	s2,a5,0x2
  24:	4401                	li	s0,0
  26:	410783b3          	sub	t2,a5,a6
  2a:	00239993          	slli	s3,t2,0x2

0000002e <.L19>:
  2e:	0ad41063          	bne	s0,a3,ce <.L28>
  32:	5432                	lw	s0,44(sp)
  34:	54a2                	lw	s1,40(sp)
  36:	5912                	lw	s2,36(sp)
  38:	5982                	lw	s3,32(sp)
  3a:	4a72                	lw	s4,28(sp)
  3c:	4ae2                	lw	s5,24(sp)
  3e:	4b52                	lw	s6,20(sp)
  40:	4bc2                	lw	s7,16(sp)
  42:	4c32                	lw	s8,12(sp)
  44:	4ca2                	lw	s9,8(sp)
  46:	6145                	addi	sp,sp,48
  48:	8082                	ret

0000004a <.L20>:
  4a:	00432e8b          	p.lw	t4,4(t1!)
  4e:	00082c83          	lw	s9,0(a6)
  52:	0885                	addi	a7,a7,1
  54:	f99effc3          	fmadd.s	ft11,ft9,fs9,ft11
  58:	00482c83          	lw	s9,4(a6)
  5c:	299ef2c3          	fmadd.s	ft5,ft9,fs9,ft5
  60:	00882c83          	lw	s9,8(a6)
  64:	a19efa43          	fmadd.s	fs4,ft9,fs9,fs4
  68:	00c82c83          	lw	s9,12(a6)
  6c:	a99efac3          	fmadd.s	fs5,ft9,fs9,fs5
  70:	01082c83          	lw	s9,16(a6)
  74:	b19efb43          	fmadd.s	fs6,ft9,fs9,fs6
  78:	01482c83          	lw	s9,20(a6)
  7c:	b99efbc3          	fmadd.s	fs7,ft9,fs9,fs7
  80:	01882c83          	lw	s9,24(a6)
  84:	984a                	add	a6,a6,s2
  86:	c19efc43          	fmadd.s	fs8,ft9,fs9,fs8

0000008a <.L22>:
  8a:	fce890e3          	bne	a7,a4,4a <.L20>
  8e:	01fe2023          	sw	t6,0(t3)
  92:	005e2223          	sw	t0,4(t3)
  96:	014e2423          	sw	s4,8(t3)
  9a:	015e2623          	sw	s5,12(t3)
  9e:	016e2823          	sw	s6,16(t3)
  a2:	017e2a23          	sw	s7,20(t3)
  a6:	018e2c23          	sw	s8,24(t3)
  aa:	0f1d                	addi	t5,t5,7
  ac:	0e71                	addi	t3,t3,28

000000ae <.L27>:
  ae:	027f7363          	bleu	t2,t5,d4 <.L21>
  b2:	00000c13          	li	s8,0
  b6:	002f1813          	slli	a6,t5,0x2
  ba:	982e                	add	a6,a6,a1
  bc:	832a                	mv	t1,a0
  be:	8be2                	mv	s7,s8
  c0:	8b62                	mv	s6,s8
  c2:	8ae2                	mv	s5,s8
  c4:	8a62                	mv	s4,s8
  c6:	82e2                	mv	t0,s8
  c8:	8fe2                	mv	t6,s8
  ca:	4881                	li	a7,0
  cc:	bf7d                	j	8a <.L22>

000000ce <.L28>:
  ce:	8e32                	mv	t3,a2
  d0:	4f01                	li	t5,0
  d2:	bff1                	j	ae <.L27>

000000d4 <.L21>:
  d4:	01360f33          	add	t5,a2,s3
  d8:	889e                	mv	a7,t2

000000da <.L23>:
  da:	02f8f763          	bleu	a5,a7,108 <.L25>
  de:	00289813          	slli	a6,a7,0x2
  e2:	982e                	add	a6,a6,a1
  e4:	8eaa                	mv	t4,a0
  e6:	00000e13          	li	t3,0
  ea:	4301                	li	t1,0
  ec:	a801                	j	fc <.L26>

000000ee <.L24>:
  ee:	004ea28b          	p.lw	t0,4(t4!)
  f2:	21287f8b          	p.lw	t6,s2(a6!)
  f6:	0305                	addi	t1,t1,1
  f8:	e1f2fe43          	fmadd.s	ft8,ft5,ft11,ft8

000000fc <.L26>:
  fc:	fee319e3          	bne	t1,a4,ee <.L24>
 100:	01cf222b          	p.sw	t3,4(t5!)
 104:	0885                	addi	a7,a7,1
 106:	bfd1                	j	da <.L23>

00000108 <.L25>:
 108:	0405                	addi	s0,s0,1
 10a:	9526                	add	a0,a0,s1
 10c:	964a                	add	a2,a2,s2
 10e:	b705                	j	2e <.L19>

MatMul_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.MatMul_s8_s8_s32 00000266  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000029a  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  000002b5  2**0
                  CONTENTS, READONLY

Disassembly of section .text.MatMul_s8_s8_s32:

00000000 <MatMul_s8_s8_s32>:
   0:	7119                	addi	sp,sp,-128
   2:	0017d313          	srli	t1,a5,0x1
   6:	d41a                	sw	t1,40(sp)
   8:	4e8a                	lw	t4,128(sp)
   a:	0306                	slli	t1,t1,0x1
   c:	932e                	add	t1,t1,a1
   e:	00179f13          	slli	t5,a5,0x1
  12:	d4d6                	sw	s5,104(sp)
  14:	d0de                	sw	s7,96(sp)
  16:	0016d293          	srli	t0,a3,0x1
  1a:	00175393          	srli	t2,a4,0x1
  1e:	00171a93          	slli	s5,a4,0x1
  22:	00e50bb3          	add	s7,a0,a4
  26:	c69a                	sw	t1,76(sp)
  28:	de7a                	sw	t5,60(sp)
  2a:	dea2                	sw	s0,124(sp)
  2c:	dca6                	sw	s1,120(sp)
  2e:	daca                	sw	s2,116(sp)
  30:	d8ce                	sw	s3,112(sp)
  32:	d6d2                	sw	s4,108(sp)
  34:	d2da                	sw	s6,100(sp)
  36:	cee2                	sw	s8,92(sp)
  38:	cce6                	sw	s9,88(sp)
  3a:	caea                	sw	s10,84(sp)
  3c:	c8ee                	sw	s11,80(sp)
  3e:	d616                	sw	t0,44(sp)
  40:	c49e                	sw	t2,72(sp)
  42:	ce32                	sw	a2,28(sp)
  44:	c0d6                	sw	s5,64(sp)
  46:	d25e                	sw	s7,36(sp)
  48:	00279e13          	slli	t3,a5,0x2
  4c:	d02a                	sw	a0,32(sp)
  4e:	8f32                	mv	t5,a2
  50:	4f81                	li	t6,0
  52:	4301                	li	t1,0
  54:	cc02                	sw	zero,24(sp)

00000056 <.L2>:
  56:	53b2                	lw	t2,44(sp)
  58:	4ae2                	lw	s5,24(sp)
  5a:	0d538e63          	beq	t2,s5,136 <.L6>
  5e:	01cf0bb3          	add	s7,t5,t3
  62:	c2de                	sw	s7,68(sp)
  64:	8fae                	mv	t6,a1
  66:	83de                	mv	t2,s7
  68:	a851                	j	fc <.L7>

0000006a <.L3>:
  6a:	00040a03          	lb	s4,0(s0)
  6e:	00048b83          	lb	s7,0(s1)
  72:	00148a83          	lb	s5,1(s1)
  76:	9a42                	add	s4,s4,a6
  78:	d852                	sw	s4,48(sp)
  7a:	00090a03          	lb	s4,0(s2)
  7e:	9bc2                	add	s7,s7,a6
  80:	9ac2                	add	s5,s5,a6
  82:	9a46                	add	s4,s4,a7
  84:	da52                	sw	s4,52(sp)
  86:	c85e                	sw	s7,16(sp)
  88:	c656                	sw	s5,12(sp)
  8a:	5bd2                	lw	s7,52(sp)
  8c:	5ac2                	lw	s5,48(sp)
  8e:	42d2                	lw	t0,20(sp)
  90:	00140d03          	lb	s10,1(s0)
  94:	00098c03          	lb	s8,0(s3)
  98:	437a82b3          	p.mac	t0,s5,s7
  9c:	9d42                	add	s10,s10,a6
  9e:	9c46                	add	s8,s8,a7
  a0:	00190a03          	lb	s4,1(s2)
  a4:	0305                	addi	t1,t1,1
  a6:	0409                	addi	s0,s0,2
  a8:	9a46                	add	s4,s4,a7
  aa:	dc52                	sw	s4,56(sp)
  ac:	00198a03          	lb	s4,1(s3)
  b0:	0489                	addi	s1,s1,2
  b2:	438d02b3          	p.mac	t0,s10,s8
  b6:	9a46                	add	s4,s4,a7
  b8:	ca16                	sw	t0,20(sp)
  ba:	52e2                	lw	t0,56(sp)
  bc:	425a8db3          	p.mac	s11,s5,t0
  c0:	4ac2                	lw	s5,16(sp)
  c2:	437a8cb3          	p.mac	s9,s5,s7
  c6:	4bb2                	lw	s7,12(sp)
  c8:	425a8b33          	p.mac	s6,s5,t0
  cc:	52f2                	lw	t0,60(sp)
  ce:	9916                	add	s2,s2,t0
  d0:	9996                	add	s3,s3,t0
  d2:	434d0db3          	p.mac	s11,s10,s4
  d6:	438b8cb3          	p.mac	s9,s7,s8
  da:	434b8b33          	p.mac	s6,s7,s4

000000de <.L5>:
  de:	4aa6                	lw	s5,72(sp)
  e0:	f86a95e3          	bne	s5,t1,6a <.L3>
  e4:	4bd2                	lw	s7,20(sp)
  e6:	01bf2223          	sw	s11,4(t5)
  ea:	03a1                	addi	t2,t2,8
  ec:	017f2023          	sw	s7,0(t5)
  f0:	ff93ac23          	sw	s9,-8(t2)
  f4:	ff63ae23          	sw	s6,-4(t2)
  f8:	0f21                	addi	t5,t5,8
  fa:	0f89                	addi	t6,t6,2

000000fc <.L7>:
  fc:	42b6                	lw	t0,76(sp)
  fe:	005f8d63          	beq	t6,t0,118 <.L4>
 102:	01f789b3          	add	s3,a5,t6
 106:	897e                	mv	s2,t6
 108:	5492                	lw	s1,36(sp)
 10a:	5402                	lw	s0,32(sp)
 10c:	8b76                	mv	s6,t4
 10e:	8cf6                	mv	s9,t4
 110:	8df6                	mv	s11,t4
 112:	ca76                	sw	t4,20(sp)
 114:	4301                	li	t1,0
 116:	b7e1                	j	de <.L5>

00000118 <.L4>:
 118:	4f86                	lw	t6,64(sp)
 11a:	43e2                	lw	t2,24(sp)
 11c:	5b82                	lw	s7,32(sp)
 11e:	5292                	lw	t0,36(sp)
 120:	4a96                	lw	s5,68(sp)
 122:	9bfe                	add	s7,s7,t6
 124:	92fe                	add	t0,t0,t6
 126:	0385                	addi	t2,t2,1
 128:	cc1e                	sw	t2,24(sp)
 12a:	015e0f33          	add	t5,t3,s5
 12e:	d05e                	sw	s7,32(sp)
 130:	d216                	sw	t0,36(sp)
 132:	5fa2                	lw	t6,40(sp)
 134:	b70d                	j	56 <.L2>

00000136 <.L6>:
 136:	52b2                	lw	t0,44(sp)
 138:	0306                	slli	t1,t1,0x1
 13a:	0f86                	slli	t6,t6,0x1
 13c:	00129f13          	slli	t5,t0,0x1
 140:	03e69f63          	bne	a3,t5,17e <.L8>
 144:	00e30f63          	beq	t1,a4,162 <.L9>

00000148 <.L13>:
 148:	8b2e                	mv	s6,a1
 14a:	42f30b33          	p.mac	s6,t1,a5
 14e:	006503b3          	add	t2,a0,t1
 152:	40a00433          	neg	s0,a0
 156:	4281                	li	t0,0

00000158 <.L10>:
 158:	03e28563          	beq	t0,t5,182 <.L14>
 15c:	49f2                	lw	s3,28(sp)
 15e:	4301                	li	t1,0
 160:	a89d                	j	1d6 <.L18>

00000162 <.L9>:
 162:	02ff8a63          	beq	t6,a5,196 <.L1>

00000166 <.L19>:
 166:	002f9313          	slli	t1,t6,0x2
 16a:	9332                	add	t1,t1,a2
 16c:	40a004b3          	neg	s1,a0
 170:	842a                	mv	s0,a0
 172:	4381                	li	t2,0

00000174 <.L12>:
 174:	01e38963          	beq	t2,t5,186 <.L20>
 178:	8b9a                	mv	s7,t1
 17a:	82fe                	mv	t0,t6
 17c:	a851                	j	210 <.L24>

0000017e <.L8>:
 17e:	fce315e3          	bne	t1,a4,148 <.L13>

00000182 <.L14>:
 182:	feff92e3          	bne	t6,a5,166 <.L19>

00000186 <.L20>:
 186:	43e70533          	p.mac	a0,a4,t5
 18a:	43cf0633          	p.mac	a2,t5,t3
 18e:	40a00fb3          	neg	t6,a0

00000192 <.L25>:
 192:	0cdf6163          	bltu	t5,a3,254 <.L32>

00000196 <.L1>:
 196:	5476                	lw	s0,124(sp)
 198:	54e6                	lw	s1,120(sp)
 19a:	5956                	lw	s2,116(sp)
 19c:	59c6                	lw	s3,112(sp)
 19e:	5a36                	lw	s4,108(sp)
 1a0:	5aa6                	lw	s5,104(sp)
 1a2:	5b16                	lw	s6,100(sp)
 1a4:	5b86                	lw	s7,96(sp)
 1a6:	4c76                	lw	s8,92(sp)
 1a8:	4ce6                	lw	s9,88(sp)
 1aa:	4d56                	lw	s10,84(sp)
 1ac:	4dc6                	lw	s11,80(sp)
 1ae:	6109                	addi	sp,sp,128
 1b0:	8082                	ret

000001b2 <.L15>:
 1b2:	00190a0b          	p.lb	s4,1(s2!)
 1b6:	00fbfa8b          	p.lb	s5,a5(s7!)
 1ba:	9a42                	add	s4,s4,a6
 1bc:	9ac6                	add	s5,s5,a7
 1be:	434a84b3          	p.mac	s1,s5,s4

000001c2 <.L17>:
 1c2:	00890a33          	add	s4,s2,s0
 1c6:	feea66e3          	bltu	s4,a4,1b2 <.L15>
 1ca:	0009a903          	lw	s2,0(s3)
 1ce:	0305                	addi	t1,t1,1
 1d0:	94ca                	add	s1,s1,s2
 1d2:	0099a22b          	p.sw	s1,4(s3!)

000001d6 <.L18>:
 1d6:	01f30763          	beq	t1,t6,1e4 <.L16>
 1da:	01630bb3          	add	s7,t1,s6
 1de:	891e                	mv	s2,t2
 1e0:	4481                	li	s1,0
 1e2:	b7c5                	j	1c2 <.L17>

000001e4 <.L16>:
 1e4:	4372                	lw	t1,28(sp)
 1e6:	0285                	addi	t0,t0,1
 1e8:	93ba                	add	t2,t2,a4
 1ea:	9372                	add	t1,t1,t3
 1ec:	ce1a                	sw	t1,28(sp)
 1ee:	8c19                	sub	s0,s0,a4
 1f0:	b7a5                	j	158 <.L10>

000001f2 <.L21>:
 1f2:	00198a0b          	p.lb	s4,1(s3!)
 1f6:	00fb7a8b          	p.lb	s5,a5(s6!)
 1fa:	9a42                	add	s4,s4,a6
 1fc:	9ac6                	add	s5,s5,a7
 1fe:	434a8933          	p.mac	s2,s5,s4

00000202 <.L23>:
 202:	00998a33          	add	s4,s3,s1
 206:	feea66e3          	bltu	s4,a4,1f2 <.L21>
 20a:	012ba22b          	p.sw	s2,4(s7!)
 20e:	0285                	addi	t0,t0,1

00000210 <.L24>:
 210:	00f2f763          	bleu	a5,t0,21e <.L22>
 214:	00558b33          	add	s6,a1,t0
 218:	89a2                	mv	s3,s0
 21a:	8976                	mv	s2,t4
 21c:	b7dd                	j	202 <.L23>

0000021e <.L22>:
 21e:	0385                	addi	t2,t2,1
 220:	9372                	add	t1,t1,t3
 222:	943a                	add	s0,s0,a4
 224:	8c99                	sub	s1,s1,a4
 226:	b7b9                	j	174 <.L12>

00000228 <.L26>:
 228:	0013840b          	p.lb	s0,1(t2!)
 22c:	00f9748b          	p.lb	s1,a5(s2!)
 230:	9442                	add	s0,s0,a6
 232:	94c6                	add	s1,s1,a7
 234:	428482b3          	p.mac	t0,s1,s0

00000238 <.L28>:
 238:	01f38433          	add	s0,t2,t6
 23c:	fee466e3          	bltu	s0,a4,228 <.L26>
 240:	0059a22b          	p.sw	t0,4(s3!)
 244:	0305                	addi	t1,t1,1

00000246 <.L29>:
 246:	00f30a63          	beq	t1,a5,25a <.L27>
 24a:	00658933          	add	s2,a1,t1
 24e:	83aa                	mv	t2,a0
 250:	82f6                	mv	t0,t4
 252:	b7dd                	j	238 <.L28>

00000254 <.L32>:
 254:	89b2                	mv	s3,a2
 256:	4301                	li	t1,0
 258:	b7fd                	j	246 <.L29>

0000025a <.L27>:
 25a:	0f05                	addi	t5,t5,1
 25c:	9672                	add	a2,a2,t3
 25e:	953a                	add	a0,a0,a4
 260:	40ef8fb3          	sub	t6,t6,a4
 264:	b73d                	j	192 <.L25>

MaxPool_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.MaxPool2d_fp32_fp32_NCHW 00000110  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.MaxPool1d_fp32_fp32 00000096  00000000  00000000  00000144  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .rodata.MaxPool2d_fp32_fp32_NCHW.cst4 00000004  00000000  00000000  000001dc  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  6 .comment      0000001b  00000000  00000000  000001e0  2**0
                  CONTENTS, READONLY
  7 .Pulp_Chip.Info 00000072  00000000  00000000  000001fb  2**0
                  CONTENTS, READONLY

Disassembly of section .text.MaxPool2d_fp32_fp32_NCHW:

00000000 <MaxPool2d_fp32_fp32_NCHW>:
   0:	10e66763          	bltu	a2,a4,10e <.L24>
   4:	10f6e563          	bltu	a3,a5,10e <.L24>
   8:	10080363          	beqz	a6,10e <.L24>
   c:	10088163          	beqz	a7,10e <.L24>
  10:	40f68eb3          	sub	t4,a3,a5
  14:	40e60e33          	sub	t3,a2,a4
  18:	031edeb3          	divu	t4,t4,a7
  1c:	7139                	addi	sp,sp,-64
  1e:	cc66                	sw	s9,24(sp)
  20:	dc26                	sw	s1,56(sp)
  22:	4486                	lw	s1,64(sp)
  24:	de22                	sw	s0,60(sp)
  26:	da4a                	sw	s2,52(sp)
  28:	d84e                	sw	s3,48(sp)
  2a:	d652                	sw	s4,44(sp)
  2c:	d456                	sw	s5,40(sp)
  2e:	d25a                	sw	s6,36(sp)
  30:	d05e                	sw	s7,32(sp)
  32:	ce62                	sw	s8,28(sp)
  34:	ca6a                	sw	s10,20(sp)
  36:	c86e                	sw	s11,16(sp)
  38:	4901                	li	s2,0
  3a:	4401                	li	s0,0
  3c:	030e5e33          	divu	t3,t3,a6
  40:	0e85                	addi	t4,t4,1
  42:	002e9c93          	slli	s9,t4,0x2
  46:	0e05                	addi	t3,t3,1
  48:	03cc8333          	mul	t1,s9,t3
  4c:	02d60f33          	mul	t5,a2,a3
  50:	c41a                	sw	t1,8(sp)
  52:	03068333          	mul	t1,a3,a6
  56:	c67a                	sw	t5,12(sp)
  58:	00269f13          	slli	t5,a3,0x2
  5c:	c07a                	sw	t5,0(sp)
  5e:	c21a                	sw	t1,4(sp)

00000060 <.L3>:
  60:	0ab41263          	bne	s0,a1,104 <.L15>
  64:	5472                	lw	s0,60(sp)
  66:	54e2                	lw	s1,56(sp)
  68:	5952                	lw	s2,52(sp)
  6a:	59c2                	lw	s3,48(sp)
  6c:	5a32                	lw	s4,44(sp)
  6e:	5aa2                	lw	s5,40(sp)
  70:	5b12                	lw	s6,36(sp)
  72:	5b82                	lw	s7,32(sp)
  74:	4c72                	lw	s8,28(sp)
  76:	4ce2                	lw	s9,24(sp)
  78:	4d52                	lw	s10,20(sp)
  7a:	4dc2                	lw	s11,16(sp)
  7c:	6121                	addi	sp,sp,64
  7e:	8082                	ret

00000080 <.L11>:
  80:	00000337          	lui	t1,0x0
  84:	00030313          	mv	t1,t1
  88:	00032303          	lw	t1,0(t1) # 0 <MaxPool2d_fp32_fp32_NCHW>
  8c:	00000f13          	li	t5,0
  90:	4281                	li	t0,0
  92:	19e37c53          	fdiv.s	fs8,ft6,ft10
  96:	015f8333          	add	t1,t6,s5
  9a:	030a                	slli	t1,t1,0x2
  9c:	932a                	add	t1,t1,a0

0000009e <.L4>:
  9e:	02e29563          	bne	t0,a4,c8 <.L10>
  a2:	018d222b          	p.sw	s8,4(s10!)
  a6:	0b85                	addi	s7,s7,1
  a8:	9fc6                	add	t6,t6,a7

000000aa <.L12>:
  aa:	fddbebe3          	bltu	s7,t4,80 <.L11>
  ae:	4312                	lw	t1,4(sp)
  b0:	0985                	addi	s3,s3,1
  b2:	9a66                	add	s4,s4,s9
  b4:	9a9a                	add	s5,s5,t1
  b6:	9b42                	add	s6,s6,a6

000000b8 <.L13>:
  b8:	05c9e263          	bltu	s3,t3,fc <.L14>
  bc:	4f22                	lw	t5,8(sp)
  be:	4332                	lw	t1,12(sp)
  c0:	0405                	addi	s0,s0,1
  c2:	94fa                	add	s1,s1,t5
  c4:	991a                	add	s2,s2,t1
  c6:	bf69                	j	60 <.L3>

000000c8 <.L10>:
  c8:	01628f33          	add	t5,t0,s6
  cc:	00cf7563          	bleu	a2,t5,d6 <.L5>
  d0:	4f01                	li	t5,0

000000d2 <.L6>:
  d2:	00ff1663          	bne	t5,a5,de <.L9>

000000d6 <.L5>:
  d6:	4f02                	lw	t5,0(sp)
  d8:	0285                	addi	t0,t0,1
  da:	937a                	add	t1,t1,t5
  dc:	b7c9                	j	9e <.L4>

000000de <.L9>:
  de:	01ff03b3          	add	t2,t5,t6
  e2:	00d3fb63          	bleu	a3,t2,f8 <.L7>
  e6:	002f1393          	slli	t2,t5,0x2
  ea:	20737383          	p.lw	t2,t2(t1)
  ee:	a07c1dd3          	flt.s	s11,fs8,ft7
  f2:	000d8363          	beqz	s11,f8 <.L7>
  f6:	8c1e                	mv	s8,t2

000000f8 <.L7>:
  f8:	0f05                	addi	t5,t5,1
  fa:	bfe1                	j	d2 <.L6>

000000fc <.L14>:
  fc:	8d52                	mv	s10,s4
  fe:	4f81                	li	t6,0
 100:	4b81                	li	s7,0
 102:	b765                	j	aa <.L12>

00000104 <.L15>:
 104:	8aca                	mv	s5,s2
 106:	8a26                	mv	s4,s1
 108:	4b01                	li	s6,0
 10a:	4981                	li	s3,0
 10c:	b775                	j	b8 <.L13>

0000010e <.L24>:
 10e:	8082                	ret

Disassembly of section .text.MaxPool1d_fp32_fp32:

00000000 <MaxPool1d_fp32_fp32>:
   0:	08d66a63          	bltu	a2,a3,94 <.L42>
   4:	cb41                	beqz	a4,94 <.L42>
   6:	40d60833          	sub	a6,a2,a3
   a:	02e85833          	divu	a6,a6,a4
   e:	1101                	addi	sp,sp,-32
  10:	000008b7          	lui	a7,0x0
  14:	c84e                	sw	s3,16(sp)
  16:	c652                	sw	s4,12(sp)
  18:	00000993          	li	s3,0
  1c:	0008aa03          	lw	s4,0(a7) # 0 <MaxPool1d_fp32_fp32>
  20:	ce22                	sw	s0,28(sp)
  22:	cc26                	sw	s1,24(sp)
  24:	ca4a                	sw	s2,20(sp)
  26:	c456                	sw	s5,8(sp)
  28:	00261913          	slli	s2,a2,0x2
  2c:	00271413          	slli	s0,a4,0x2
  30:	4e01                	li	t3,0
  32:	0805                	addi	a6,a6,1
  34:	00281493          	slli	s1,a6,0x2

00000038 <.L29>:
  38:	04be1963          	bne	t3,a1,8a <.L36>
  3c:	4472                	lw	s0,28(sp)
  3e:	44e2                	lw	s1,24(sp)
  40:	4952                	lw	s2,20(sp)
  42:	49c2                	lw	s3,16(sp)
  44:	4a32                	lw	s4,12(sp)
  46:	4aa2                	lw	s5,8(sp)
  48:	6105                	addi	sp,sp,32
  4a:	8082                	ret

0000004c <.L34>:
  4c:	193a72d3          	fdiv.s	ft5,fs4,fs3
  50:	4881                	li	a7,0

00000052 <.L30>:
  52:	00d89d63          	bne	a7,a3,6c <.L33>
  56:	0053a22b          	p.sw	t0,4(t2!)
  5a:	0e85                	addi	t4,t4,1
  5c:	9f3a                	add	t5,t5,a4
  5e:	9fa2                	add	t6,t6,s0

00000060 <.L35>:
  60:	ff0ee6e3          	bltu	t4,a6,4c <.L34>
  64:	0e05                	addi	t3,t3,1
  66:	97a6                	add	a5,a5,s1
  68:	954a                	add	a0,a0,s2
  6a:	b7f9                	j	38 <.L29>

0000006c <.L33>:
  6c:	01e88333          	add	t1,a7,t5
  70:	00c37b63          	bleu	a2,t1,86 <.L31>
  74:	00289313          	slli	t1,a7,0x2
  78:	206ff303          	p.lw	t1,t1(t6)
  7c:	a0629ad3          	flt.s	s5,ft5,ft6
  80:	000a8363          	beqz	s5,86 <.L31>
  84:	829a                	mv	t0,t1

00000086 <.L31>:
  86:	0885                	addi	a7,a7,1
  88:	b7e9                	j	52 <.L30>

0000008a <.L36>:
  8a:	8faa                	mv	t6,a0
  8c:	83be                	mv	t2,a5
  8e:	4f01                	li	t5,0
  90:	4e81                	li	t4,0
  92:	b7f9                	j	60 <.L35>

00000094 <.L42>:
  94:	8082                	ret

MaxPool_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.MaxPool2d_s8_s8_NCHW 00000108  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.MaxPool1d_s8_s8 00000084  00000000  00000000  0000013c  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .comment      0000001b  00000000  00000000  000001c0  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  000001db  2**0
                  CONTENTS, READONLY

Disassembly of section .text.MaxPool2d_s8_s8_NCHW:

00000000 <MaxPool2d_s8_s8_NCHW>:
   0:	10e66363          	bltu	a2,a4,106 <.L21>
   4:	10f6e163          	bltu	a3,a5,106 <.L21>
   8:	0e080f63          	beqz	a6,106 <.L21>
   c:	0e088d63          	beqz	a7,106 <.L21>
  10:	40e60333          	sub	t1,a2,a4
  14:	03035333          	divu	t1,t1,a6
  18:	715d                	addi	sp,sp,-80
  1a:	da5a                	sw	s6,52(sp)
  1c:	d06e                	sw	s11,32(sp)
  1e:	4e46                	lw	t3,80(sp)
  20:	c6a2                	sw	s0,76(sp)
  22:	c4a6                	sw	s1,72(sp)
  24:	c2ca                	sw	s2,68(sp)
  26:	c0ce                	sw	s3,64(sp)
  28:	de52                	sw	s4,60(sp)
  2a:	dc56                	sw	s5,56(sp)
  2c:	d85e                	sw	s7,48(sp)
  2e:	d662                	sw	s8,44(sp)
  30:	d466                	sw	s9,40(sp)
  32:	d26a                	sw	s10,36(sp)
  34:	40a00f33          	neg	t5,a0
  38:	41c00fb3          	neg	t6,t3
  3c:	4e81                	li	t4,0
  3e:	0305                	addi	t1,t1,1
  40:	c41a                	sw	t1,8(sp)
  42:	40f68333          	sub	t1,a3,a5
  46:	03135333          	divu	t1,t1,a7
  4a:	4b22                	lw	s6,8(sp)
  4c:	0305                	addi	t1,t1,1
  4e:	02d60633          	mul	a2,a2,a3
  52:	03068833          	mul	a6,a3,a6
  56:	c032                	sw	a2,0(sp)
  58:	026b0db3          	mul	s11,s6,t1
  5c:	c642                	sw	a6,12(sp)

0000005e <.L3>:
  5e:	08be9e63          	bne	t4,a1,fa <.L13>
  62:	4436                	lw	s0,76(sp)
  64:	44a6                	lw	s1,72(sp)
  66:	4916                	lw	s2,68(sp)
  68:	4986                	lw	s3,64(sp)
  6a:	5a72                	lw	s4,60(sp)
  6c:	5ae2                	lw	s5,56(sp)
  6e:	5b52                	lw	s6,52(sp)
  70:	5bc2                	lw	s7,48(sp)
  72:	5c32                	lw	s8,44(sp)
  74:	5ca2                	lw	s9,40(sp)
  76:	5d12                	lw	s10,36(sp)
  78:	5d82                	lw	s11,32(sp)
  7a:	6161                	addi	sp,sp,80
  7c:	8082                	ret

0000007e <.L5>:
  7e:	001c8d0b          	p.lb	s10,1(s9!)
  82:	4656                	lw	a2,84(sp)
  84:	966a                	add	a2,a2,s10
  86:	ce32                	sw	a2,28(sp)
  88:	4d72                	lw	s10,28(sp)
  8a:	01a85363          	ble	s10,a6,90 <.L6>
  8e:	4872                	lw	a6,28(sp)

00000090 <.L6>:
  90:	018c8d33          	add	s10,s9,s8
  94:	fefd65e3          	bltu	s10,a5,7e <.L5>
  98:	0b05                	addi	s6,s6,1
  9a:	9bb6                	add	s7,s7,a3
  9c:	40dc0c33          	sub	s8,s8,a3

000000a0 <.L7>:
  a0:	04eb1163          	bne	s6,a4,e2 <.L10>
  a4:	4b66                	lw	s6,88(sp)
  a6:	9a46                	add	s4,s4,a7
  a8:	411a8ab3          	sub	s5,s5,a7
  ac:	985a                	add	a6,a6,s6
  ae:	010980ab          	p.sb	a6,1(s3!)

000000b2 <.L8>:
  b2:	01298833          	add	a6,s3,s2
  b6:	02686863          	bltu	a6,t1,e6 <.L11>
  ba:	4d32                	lw	s10,12(sp)
  bc:	0285                	addi	t0,t0,1
  be:	939a                	add	t2,t2,t1
  c0:	946a                	add	s0,s0,s10
  c2:	41a484b3          	sub	s1,s1,s10
  c6:	40690933          	sub	s2,s2,t1

000000ca <.L9>:
  ca:	4622                	lw	a2,8(sp)
  cc:	02c2e363          	bltu	t0,a2,f2 <.L12>
  d0:	4802                	lw	a6,0(sp)
  d2:	0e85                	addi	t4,t4,1
  d4:	9e6e                	add	t3,t3,s11
  d6:	9542                	add	a0,a0,a6
  d8:	410f0f33          	sub	t5,t5,a6
  dc:	41bf8fb3          	sub	t6,t6,s11
  e0:	bfbd                	j	5e <.L3>

000000e2 <.L10>:
  e2:	8cde                	mv	s9,s7
  e4:	b775                	j	90 <.L6>

000000e6 <.L11>:
  e6:	8c56                	mv	s8,s5
  e8:	8bd2                	mv	s7,s4
  ea:	f8000813          	li	a6,-128
  ee:	4b01                	li	s6,0
  f0:	bf45                	j	a0 <.L7>

000000f2 <.L12>:
  f2:	8aa6                	mv	s5,s1
  f4:	8a22                	mv	s4,s0
  f6:	899e                	mv	s3,t2
  f8:	bf6d                	j	b2 <.L8>

000000fa <.L13>:
  fa:	897e                	mv	s2,t6
  fc:	84fa                	mv	s1,t5
  fe:	842a                	mv	s0,a0
 100:	83f2                	mv	t2,t3
 102:	4281                	li	t0,0
 104:	b7d9                	j	ca <.L9>

00000106 <.L21>:
 106:	8082                	ret

Disassembly of section .text.MaxPool1d_s8_s8:

00000000 <MaxPool1d_s8_s8>:
   0:	06d66a63          	bltu	a2,a3,74 <.L37>
   4:	cb25                	beqz	a4,74 <.L37>
   6:	40d60333          	sub	t1,a2,a3
   a:	02e35333          	divu	t1,t1,a4
   e:	4f81                	li	t6,0
  10:	40f002b3          	neg	t0,a5
  14:	40a003b3          	neg	t2,a0
  18:	0305                	addi	t1,t1,1
  1a:	04bf9e63          	bne	t6,a1,76 <.L39>
  1e:	8082                	ret

00000020 <.L28>:
  20:	012e84b3          	add	s1,t4,s2
  24:	00c4f963          	bleu	a2,s1,36 <.L27>
  28:	01df04b3          	add	s1,t5,t4
  2c:	00048483          	lb	s1,0(s1)
  30:	94c2                	add	s1,s1,a6
  32:	049e6e33          	p.max	t3,t3,s1

00000036 <.L27>:
  36:	0e85                	addi	t4,t4,1

00000038 <.L30>:
  38:	fede94e3          	bne	t4,a3,20 <.L28>
  3c:	9e46                	add	t3,t3,a7
  3e:	01c400ab          	p.sb	t3,1(s0!)
  42:	9f3a                	add	t5,t5,a4

00000044 <.L31>:
  44:	00540e33          	add	t3,s0,t0
  48:	006e7863          	bleu	t1,t3,58 <.L29>
  4c:	007f0933          	add	s2,t5,t2
  50:	4e81                	li	t4,0
  52:	f8000e13          	li	t3,-128
  56:	b7cd                	j	38 <.L30>

00000058 <.L29>:
  58:	0f85                	addi	t6,t6,1
  5a:	979a                	add	a5,a5,t1
  5c:	9532                	add	a0,a0,a2
  5e:	406282b3          	sub	t0,t0,t1
  62:	40c383b3          	sub	t2,t2,a2
  66:	00bf9c63          	bne	t6,a1,7e <.L32>
  6a:	4432                	lw	s0,12(sp)
  6c:	44a2                	lw	s1,8(sp)
  6e:	4912                	lw	s2,4(sp)
  70:	0141                	addi	sp,sp,16
  72:	8082                	ret

00000074 <.L37>:
  74:	8082                	ret

00000076 <.L39>:
  76:	1141                	addi	sp,sp,-16
  78:	c622                	sw	s0,12(sp)
  7a:	c426                	sw	s1,8(sp)
  7c:	c24a                	sw	s2,4(sp)

0000007e <.L32>:
  7e:	8f2a                	mv	t5,a0
  80:	843e                	mv	s0,a5
  82:	b7c9                	j	44 <.L31>

Pow_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Pow_fp32_fp32_fp32 00000040  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.Pow_fp32_scalar_fp32 0000003e  00000000  00000000  00000074  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .comment      0000001b  00000000  00000000  000000b2  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  000000cd  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Pow_fp32_fp32_fp32:

00000000 <Pow_fp32_fp32_fp32>:
   0:	1101                	addi	sp,sp,-32
   2:	cc22                	sw	s0,24(sp)
   4:	ca26                	sw	s1,20(sp)
   6:	c84a                	sw	s2,16(sp)
   8:	ce06                	sw	ra,28(sp)
   a:	84aa                	mv	s1,a0
   c:	892e                	mv	s2,a1
   e:	4401                	li	s0,0

00000010 <.L2>:
  10:	00d44863          	blt	s0,a3,20 <.L3>
  14:	40f2                	lw	ra,28(sp)
  16:	4462                	lw	s0,24(sp)
  18:	44d2                	lw	s1,20(sp)
  1a:	4942                	lw	s2,16(sp)
  1c:	6105                	addi	sp,sp,32
  1e:	8082                	ret

00000020 <.L3>:
  20:	0049258b          	p.lw	a1,4(s2!)
  24:	0044a50b          	p.lw	a0,4(s1!)
  28:	c632                	sw	a2,12(sp)
  2a:	c436                	sw	a3,8(sp)
  2c:	00000097          	auipc	ra,0x0
  30:	000080e7          	jalr	ra
  34:	4632                	lw	a2,12(sp)
  36:	0405                	addi	s0,s0,1
  38:	46a2                	lw	a3,8(sp)
  3a:	00a6222b          	p.sw	a0,4(a2!)
  3e:	bfc9                	j	10 <.L2>

Disassembly of section .text.Pow_fp32_scalar_fp32:

00000000 <Pow_fp32_scalar_fp32>:
   0:	1101                	addi	sp,sp,-32
   2:	c84a                	sw	s2,16(sp)
   4:	892e                	mv	s2,a1
   6:	cc22                	sw	s0,24(sp)
   8:	ca26                	sw	s1,20(sp)
   a:	ce06                	sw	ra,28(sp)
   c:	84aa                	mv	s1,a0
   e:	4401                	li	s0,0

00000010 <.L6>:
  10:	00d44863          	blt	s0,a3,20 <.L7>
  14:	40f2                	lw	ra,28(sp)
  16:	4462                	lw	s0,24(sp)
  18:	44d2                	lw	s1,20(sp)
  1a:	4942                	lw	s2,16(sp)
  1c:	6105                	addi	sp,sp,32
  1e:	8082                	ret

00000020 <.L7>:
  20:	0044a50b          	p.lw	a0,4(s1!)
  24:	85ca                	mv	a1,s2
  26:	c632                	sw	a2,12(sp)
  28:	c436                	sw	a3,8(sp)
  2a:	00000097          	auipc	ra,0x0
  2e:	000080e7          	jalr	ra
  32:	4632                	lw	a2,12(sp)
  34:	0405                	addi	s0,s0,1
  36:	46a2                	lw	a3,8(sp)
  38:	00a6222b          	p.sw	a0,4(a2!)
  3c:	bfd1                	j	10 <.L6>

RQDiv_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.RQDiv_s32_s8 0000014a  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000017e  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000199  2**0
                  CONTENTS, READONLY

Disassembly of section .text.RQDiv_s32_s8:

00000000 <RQDiv_s32_s8>:
   0:	7119                	addi	sp,sp,-128
   2:	d6ce                	sw	s3,108(sp)
   4:	02f749b3          	div	s3,a4,a5
   8:	dc2a                	sw	a0,56(sp)
   a:	451a                	lw	a0,132(sp)
   c:	cede                	sw	s7,92(sp)
   e:	c8ea                	sw	s10,80(sp)
  10:	0407ebb3          	p.max	s7,a5,zero
  14:	4685                	li	a3,1
  16:	d4d2                	sw	s4,104(sp)
  18:	d2d6                	sw	s5,100(sp)
  1a:	d0da                	sw	s6,96(sp)
  1c:	c6ee                	sw	s11,76(sp)
  1e:	de86                	sw	ra,124(sp)
  20:	dca2                	sw	s0,120(sp)
  22:	daa6                	sw	s1,116(sp)
  24:	d8ca                	sw	s2,112(sp)
  26:	cce2                	sw	s8,88(sp)
  28:	cae6                	sw	s9,84(sp)
  2a:	00279d93          	slli	s11,a5,0x2
  2e:	8a2e                	mv	s4,a1
  30:	4a81                	li	s5,0
  32:	4b01                	li	s6,0
  34:	0409ed33          	p.max	s10,s3,zero
  38:	031508b3          	mul	a7,a0,a7
  3c:	03ab8bb3          	mul	s7,s7,s10
  40:	03364733          	div	a4,a2,s3
  44:	464a                	lw	a2,144(sp)
  46:	fff60f93          	addi	t6,a2,-1
  4a:	01f692b3          	sll	t0,a3,t6
  4e:	468a                	lw	a3,128(sp)
  50:	86fd                	srai	a3,a3,0x1f
  52:	de36                	sw	a3,60(sp)

00000054 <.L2>:
  54:	0ceb5c63          	ble	a4,s6,12c <.L1>
  58:	8cd2                	mv	s9,s4
  5a:	8942                	mv	s2,a6
  5c:	4481                	li	s1,0
  5e:	035786b3          	mul	a3,a5,s5
  62:	da36                	sw	a3,52(sp)
  64:	a81d                	j	9a <.L8>

00000066 <.L6>:
  66:	4e0a                	lw	t3,128(sp)
  68:	5ef2                	lw	t4,60(sp)
  6a:	004ca68b          	p.lw	a3,4(s9!)
  6e:	461a                	lw	a2,132(sp)
  70:	220e1513          	addi.d	a0,t3,0
  74:	8c4a                	mv	s8,s2
  76:	72d60533          	p.mac.d	a0,a2,a3
  7a:	56d2                	lw	a3,52(sp)
  7c:	00d48433          	add	s0,s1,a3
  80:	040a                	slli	s0,s0,0x2
  82:	d42a                	sw	a0,40(sp)
  84:	64155f13          	srai.d	t5,a0,0x1
  88:	5562                	lw	a0,56(sp)
  8a:	d62e                	sw	a1,44(sp)
  8c:	942a                	add	s0,s0,a0

0000008e <.L3>:
  8e:	412c06b3          	sub	a3,s8,s2
  92:	0136cb63          	blt	a3,s3,a8 <.L5>
  96:	996a                	add	s2,s2,s10
  98:	0485                	addi	s1,s1,1

0000009a <.L8>:
  9a:	fcf4c6e3          	blt	s1,a5,66 <.L6>
  9e:	985e                	add	a6,a6,s7
  a0:	0b05                	addi	s6,s6,1
  a2:	9a6e                	add	s4,s4,s11
  a4:	9ace                	add	s5,s5,s3
  a6:	b77d                	j	54 <.L2>

000000a8 <.L5>:
  a8:	00042303          	lw	t1,0(s0)
  ac:	cc16                	sw	t0,24(sp)
  ae:	ca3a                	sw	a4,20(sp)
  b0:	7268c333          	p.muls.d	t1,a7,t1
  b4:	c83e                	sw	a5,16(sp)
  b6:	c642                	sw	a6,12(sp)
  b8:	d246                	sw	a7,36(sp)
  ba:	c472                	sw	t3,8(sp)
  bc:	c276                	sw	t4,4(sp)
  be:	d07a                	sw	t5,32(sp)
  c0:	ce7e                	sw	t6,28(sp)
  c2:	fff3c513          	not	a0,t2
  c6:	01f3d693          	srli	a3,t2,0x1f
  ca:	817d                	srli	a0,a0,0x1f
  cc:	8d15                	sub	a0,a0,a3
  ce:	02af86b3          	mul	a3,t6,a0
  d2:	41f55613          	srai	a2,a0,0x1f
  d6:	43e606b3          	p.mac	a3,a2,t5
  da:	5622                	lw	a2,40(sp)
  dc:	73e55533          	p.mulu.d	a0,a0,t5
  e0:	95b6                	add	a1,a1,a3
  e2:	56b2                	lw	a3,44(sp)
  e4:	20650533          	add.d	a0,a0,t1
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	46ba                	lw	a3,140(sp)
  f2:	45aa                	lw	a1,136(sp)
  f4:	464a                	lw	a2,144(sp)
  f6:	42e2                	lw	t0,24(sp)
  f8:	42a586b3          	p.mac	a3,a1,a0
  fc:	4752                	lw	a4,20(sp)
  fe:	47c2                	lw	a5,16(sp)
 100:	4832                	lw	a6,12(sp)
 102:	5892                	lw	a7,36(sp)
 104:	4e22                	lw	t3,8(sp)
 106:	4e92                	lw	t4,4(sp)
 108:	5f02                	lw	t5,32(sp)
 10a:	4ff2                	lw	t6,28(sp)
 10c:	40c2a6db          	p.addnr	a3,t0,a2
 110:	07f00613          	li	a2,127
 114:	00d64863          	blt	a2,a3,124 <.L4>
 118:	f8000613          	li	a2,-128
 11c:	00c6c463          	blt	a3,a2,124 <.L4>
 120:	1006e633          	p.extbs	a2,a3

00000124 <.L4>:
 124:	00cc00ab          	p.sb	a2,1(s8!)
 128:	946e                	add	s0,s0,s11
 12a:	b795                	j	8e <.L3>

0000012c <.L1>:
 12c:	50f6                	lw	ra,124(sp)
 12e:	5466                	lw	s0,120(sp)
 130:	54d6                	lw	s1,116(sp)
 132:	5946                	lw	s2,112(sp)
 134:	59b6                	lw	s3,108(sp)
 136:	5a26                	lw	s4,104(sp)
 138:	5a96                	lw	s5,100(sp)
 13a:	5b06                	lw	s6,96(sp)
 13c:	4bf6                	lw	s7,92(sp)
 13e:	4c66                	lw	s8,88(sp)
 140:	4cd6                	lw	s9,84(sp)
 142:	4d46                	lw	s10,80(sp)
 144:	4db6                	lw	s11,76(sp)
 146:	6109                	addi	sp,sp,128
 148:	8082                	ret

RQGELU_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.RQGELU_s8_s8 000000a0  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  000000d4  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  000000ef  2**0
                  CONTENTS, READONLY

Disassembly of section .text.RQGELU_s8_s8:

00000000 <RQGELU_s8_s8>:
   0:	1101                	addi	sp,sp,-32
   2:	5f02                	lw	t5,32(sp)
   4:	5f92                	lw	t6,36(sp)
   6:	ce22                	sw	s0,28(sp)
   8:	cc26                	sw	s1,24(sp)
   a:	ca4a                	sw	s2,20(sp)
   c:	c84e                	sw	s3,16(sp)
   e:	c652                	sw	s4,12(sp)
  10:	8eae                	mv	t4,a1
  12:	40d002b3          	neg	t0,a3
  16:	4385                	li	t2,1
  18:	07f00413          	li	s0,127
  1c:	f8000493          	li	s1,-128

00000020 <.L2>:
  20:	40be8333          	sub	t1,t4,a1
  24:	00c34963          	blt	t1,a2,36 <.L4>
  28:	4472                	lw	s0,28(sp)
  2a:	44e2                	lw	s1,24(sp)
  2c:	4952                	lw	s2,20(sp)
  2e:	49c2                	lw	s3,16(sp)
  30:	4a32                	lw	s4,12(sp)
  32:	6105                	addi	sp,sp,32
  34:	8082                	ret

00000036 <.L4>:
  36:	0015030b          	p.lb	t1,1(a0!)
  3a:	8a3a                	mv	s4,a4
  3c:	000fa983          	lw	s3,0(t6)
  40:	933e                	add	t1,t1,a5
  42:	01f35e13          	srli	t3,t1,0x1f
  46:	00602933          	sgtz	s2,t1
  4a:	41c90933          	sub	s2,s2,t3
  4e:	03230e33          	mul	t3,t1,s2
  52:	05c2ce33          	p.min	t3,t0,t3
  56:	9e36                	add	t3,t3,a3
  58:	43ce1a33          	p.msu	s4,t3,t3
  5c:	8e52                	mv	t3,s4
  5e:	8a3a                	mv	s4,a4
  60:	432e0a33          	p.mac	s4,t3,s2
  64:	0008a903          	lw	s2,0(a7)
  68:	401a5e13          	srai	t3,s4,0x1
  6c:	026e0e33          	mul	t3,t3,t1
  70:	000f2303          	lw	t1,0(t5)
  74:	432e0333          	p.mac	t1,t3,s2
  78:	fff98e13          	addi	t3,s3,-1
  7c:	01c39e33          	sll	t3,t2,t3
  80:	413e235b          	p.addnr	t1,t3,s3
  84:	9342                	add	t1,t1,a6
  86:	07f00e13          	li	t3,127
  8a:	00644863          	blt	s0,t1,9a <.L3>
  8e:	f8000e13          	li	t3,-128
  92:	00934463          	blt	t1,s1,9a <.L3>
  96:	10036e33          	p.extbs	t3,t1

0000009a <.L3>:
  9a:	01ce80ab          	p.sb	t3,1(t4!)
  9e:	b749                	j	20 <.L2>

RQHardswish.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.RQiHardswish_s8_s8 00000072  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  000000a6  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  000000c1  2**0
                  CONTENTS, READONLY

Disassembly of section .text.RQiHardswish_s8_s8:

00000000 <RQiHardswish_s8_s8>:
   0:	1141                	addi	sp,sp,-16
   2:	4fe2                	lw	t6,24(sp)
   4:	42c2                	lw	t0,16(sp)
   6:	43d2                	lw	t2,20(sp)
   8:	ffff8313          	addi	t1,t6,-1
   c:	4f05                	li	t5,1
   e:	c622                	sw	s0,12(sp)
  10:	c426                	sw	s1,8(sp)
  12:	006f1f33          	sll	t5,t5,t1
  16:	8eae                	mv	t4,a1
  18:	07f00413          	li	s0,127
  1c:	f8000493          	li	s1,-128

00000020 <.L2>:
  20:	40be8333          	sub	t1,t4,a1
  24:	00c34663          	blt	t1,a2,30 <.L4>
  28:	4432                	lw	s0,12(sp)
  2a:	44a2                	lw	s1,8(sp)
  2c:	0141                	addi	sp,sp,16
  2e:	8082                	ret

00000030 <.L4>:
  30:	00150e0b          	p.lb	t3,1(a0!)
  34:	010e0333          	add	t1,t3,a6
  38:	933a                	add	t1,t1,a4
  3a:	04036333          	p.max	t1,t1,zero
  3e:	0467c333          	p.min	t1,a5,t1
  42:	02668333          	mul	t1,a3,t1
  46:	026e0333          	mul	t1,t3,t1
  4a:	8e1e                	mv	t3,t2
  4c:	42530e33          	p.mac	t3,t1,t0
  50:	8372                	mv	t1,t3
  52:	41ff235b          	p.addnr	t1,t5,t6
  56:	9346                	add	t1,t1,a7
  58:	07f00e13          	li	t3,127
  5c:	00644863          	blt	s0,t1,6c <.L3>
  60:	f8000e13          	li	t3,-128
  64:	00934463          	blt	t1,s1,6c <.L3>
  68:	10036e33          	p.extbs	t3,t1

0000006c <.L3>:
  6c:	01ce80ab          	p.sb	t3,1(t4!)
  70:	bf45                	j	20 <.L2>

Relu_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Relu_fp32_fp32 00000024  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  00000058  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000073  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Relu_fp32_fp32:

00000000 <Relu_fp32_fp32>:
   0:	00000693          	li	a3,0
   4:	4781                	li	a5,0

00000006 <.L2>:
   6:	00c7c363          	blt	a5,a2,c <.L4>
   a:	8082                	ret

0000000c <.L4>:
   c:	0045270b          	p.lw	a4,4(a0!)
  10:	a0e69853          	flt.s	a6,fa3,fa4
  14:	00081463          	bnez	a6,1c <.L3>
  18:	00000713          	li	a4,0

0000001c <.L3>:
  1c:	00e5a22b          	p.sw	a4,4(a1!)
  20:	0785                	addi	a5,a5,1
  22:	b7d5                	j	6 <.L2>

RequantShift_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.RequantShift_s8_s8_NHWC 0000006a  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.RequantShift_s16_s8_NHWC 0000006a  00000000  00000000  0000009e  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .text.RequantShift_s32_s8_NHWC 0000006a  00000000  00000000  00000108  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  6 .text.RequantShift_s8_s8_NCHW 0000006a  00000000  00000000  00000172  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  7 .text.RequantShift_s16_s8_NCHW 0000006a  00000000  00000000  000001dc  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  8 .text.RequantShift_s32_s8_NCHW 0000006a  00000000  00000000  00000246  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  9 .comment      0000001b  00000000  00000000  000002b0  2**0
                  CONTENTS, READONLY
 10 .Pulp_Chip.Info 00000072  00000000  00000000  000002cb  2**0
                  CONTENTS, READONLY

Disassembly of section .text.RequantShift_s8_s8_NHWC:

00000000 <RequantShift_s8_s8_NHWC>:
   0:	1141                	addi	sp,sp,-16
   2:	c622                	sw	s0,12(sp)
   4:	43c2                	lw	t2,16(sp)
   6:	01410f83          	lb	t6,20(sp)
   a:	01810283          	lb	t0,24(sp)
   e:	01c14403          	lbu	s0,28(sp)
  12:	fff78313          	addi	t1,a5,-1
  16:	4e85                	li	t4,1
  18:	c426                	sw	s1,8(sp)
  1a:	006e9eb3          	sll	t4,t4,t1
  1e:	4e01                	li	t3,0

00000020 <.L2>:
  20:	00be4663          	blt	t3,a1,2c <.L5>
  24:	4432                	lw	s0,12(sp)
  26:	44a2                	lw	s1,8(sp)
  28:	0141                	addi	sp,sp,16
  2a:	8082                	ret

0000002c <.L5>:
  2c:	030e6333          	rem	t1,t3,a6
  30:	00150f0b          	p.lb	t5,1(a0!)
  34:	9f46                	add	t5,t5,a7
  36:	030a                	slli	t1,t1,0x2
  38:	20667483          	p.lw	s1,t1(a2)
  3c:	2066f303          	p.lw	t1,t1(a3)
  40:	429f0333          	p.mac	t1,t5,s1
  44:	c401                	beqz	s0,4c <.L3>
  46:	00f05363          	blez	a5,4c <.L3>
  4a:	9376                	add	t1,t1,t4

0000004c <.L3>:
  4c:	40f35333          	sra	t1,t1,a5
  50:	931e                	add	t1,t1,t2
  52:	8f16                	mv	t5,t0
  54:	0062c763          	blt	t0,t1,62 <.L4>
  58:	8f7e                	mv	t5,t6
  5a:	01f34463          	blt	t1,t6,62 <.L4>
  5e:	10036f33          	p.extbs	t5,t1

00000062 <.L4>:
  62:	01e700ab          	p.sb	t5,1(a4!)
  66:	0e05                	addi	t3,t3,1
  68:	bf65                	j	20 <.L2>

Disassembly of section .text.RequantShift_s16_s8_NHWC:

00000000 <RequantShift_s16_s8_NHWC>:
   0:	1141                	addi	sp,sp,-16
   2:	c622                	sw	s0,12(sp)
   4:	43c2                	lw	t2,16(sp)
   6:	01410f83          	lb	t6,20(sp)
   a:	01810283          	lb	t0,24(sp)
   e:	01c14403          	lbu	s0,28(sp)
  12:	fff78313          	addi	t1,a5,-1
  16:	4e85                	li	t4,1
  18:	c426                	sw	s1,8(sp)
  1a:	006e9eb3          	sll	t4,t4,t1
  1e:	4e01                	li	t3,0

00000020 <.L13>:
  20:	00be4663          	blt	t3,a1,2c <.L16>
  24:	4432                	lw	s0,12(sp)
  26:	44a2                	lw	s1,8(sp)
  28:	0141                	addi	sp,sp,16
  2a:	8082                	ret

0000002c <.L16>:
  2c:	030e6333          	rem	t1,t3,a6
  30:	00251f0b          	p.lh	t5,2(a0!)
  34:	9f46                	add	t5,t5,a7
  36:	030a                	slli	t1,t1,0x2
  38:	20667483          	p.lw	s1,t1(a2)
  3c:	2066f303          	p.lw	t1,t1(a3)
  40:	429f0333          	p.mac	t1,t5,s1
  44:	c401                	beqz	s0,4c <.L14>
  46:	00f05363          	blez	a5,4c <.L14>
  4a:	9376                	add	t1,t1,t4

0000004c <.L14>:
  4c:	40f35333          	sra	t1,t1,a5
  50:	931e                	add	t1,t1,t2
  52:	8f16                	mv	t5,t0
  54:	0062c763          	blt	t0,t1,62 <.L15>
  58:	8f7e                	mv	t5,t6
  5a:	01f34463          	blt	t1,t6,62 <.L15>
  5e:	10036f33          	p.extbs	t5,t1

00000062 <.L15>:
  62:	01e700ab          	p.sb	t5,1(a4!)
  66:	0e05                	addi	t3,t3,1
  68:	bf65                	j	20 <.L13>

Disassembly of section .text.RequantShift_s32_s8_NHWC:

00000000 <RequantShift_s32_s8_NHWC>:
   0:	1141                	addi	sp,sp,-16
   2:	c622                	sw	s0,12(sp)
   4:	43c2                	lw	t2,16(sp)
   6:	01410f83          	lb	t6,20(sp)
   a:	01810283          	lb	t0,24(sp)
   e:	01c14403          	lbu	s0,28(sp)
  12:	fff78313          	addi	t1,a5,-1
  16:	4e85                	li	t4,1
  18:	c426                	sw	s1,8(sp)
  1a:	006e9eb3          	sll	t4,t4,t1
  1e:	4e01                	li	t3,0

00000020 <.L24>:
  20:	00be4663          	blt	t3,a1,2c <.L27>
  24:	4432                	lw	s0,12(sp)
  26:	44a2                	lw	s1,8(sp)
  28:	0141                	addi	sp,sp,16
  2a:	8082                	ret

0000002c <.L27>:
  2c:	030e6333          	rem	t1,t3,a6
  30:	00452f0b          	p.lw	t5,4(a0!)
  34:	9f46                	add	t5,t5,a7
  36:	030a                	slli	t1,t1,0x2
  38:	20667483          	p.lw	s1,t1(a2)
  3c:	2066f303          	p.lw	t1,t1(a3)
  40:	429f0333          	p.mac	t1,t5,s1
  44:	c401                	beqz	s0,4c <.L25>
  46:	00f05363          	blez	a5,4c <.L25>
  4a:	9376                	add	t1,t1,t4

0000004c <.L25>:
  4c:	40f35333          	sra	t1,t1,a5
  50:	931e                	add	t1,t1,t2
  52:	8f16                	mv	t5,t0
  54:	0062c763          	blt	t0,t1,62 <.L26>
  58:	8f7e                	mv	t5,t6
  5a:	01f34463          	blt	t1,t6,62 <.L26>
  5e:	10036f33          	p.extbs	t5,t1

00000062 <.L26>:
  62:	01e700ab          	p.sb	t5,1(a4!)
  66:	0e05                	addi	t3,t3,1
  68:	bf65                	j	20 <.L24>

Disassembly of section .text.RequantShift_s8_s8_NCHW:

00000000 <RequantShift_s8_s8_NCHW>:
   0:	1141                	addi	sp,sp,-16
   2:	c622                	sw	s0,12(sp)
   4:	43c2                	lw	t2,16(sp)
   6:	01410f83          	lb	t6,20(sp)
   a:	01810283          	lb	t0,24(sp)
   e:	01c14403          	lbu	s0,28(sp)
  12:	fff78313          	addi	t1,a5,-1
  16:	4e85                	li	t4,1
  18:	c426                	sw	s1,8(sp)
  1a:	006e9eb3          	sll	t4,t4,t1
  1e:	4e01                	li	t3,0

00000020 <.L35>:
  20:	00be4663          	blt	t3,a1,2c <.L38>
  24:	4432                	lw	s0,12(sp)
  26:	44a2                	lw	s1,8(sp)
  28:	0141                	addi	sp,sp,16
  2a:	8082                	ret

0000002c <.L38>:
  2c:	030e4333          	div	t1,t3,a6
  30:	00150f0b          	p.lb	t5,1(a0!)
  34:	9f46                	add	t5,t5,a7
  36:	030a                	slli	t1,t1,0x2
  38:	20667483          	p.lw	s1,t1(a2)
  3c:	2066f303          	p.lw	t1,t1(a3)
  40:	429f0333          	p.mac	t1,t5,s1
  44:	c401                	beqz	s0,4c <.L36>
  46:	00f05363          	blez	a5,4c <.L36>
  4a:	9376                	add	t1,t1,t4

0000004c <.L36>:
  4c:	40f35333          	sra	t1,t1,a5
  50:	931e                	add	t1,t1,t2
  52:	8f16                	mv	t5,t0
  54:	0062c763          	blt	t0,t1,62 <.L37>
  58:	8f7e                	mv	t5,t6
  5a:	01f34463          	blt	t1,t6,62 <.L37>
  5e:	10036f33          	p.extbs	t5,t1

00000062 <.L37>:
  62:	01e700ab          	p.sb	t5,1(a4!)
  66:	0e05                	addi	t3,t3,1
  68:	bf65                	j	20 <.L35>

Disassembly of section .text.RequantShift_s16_s8_NCHW:

00000000 <RequantShift_s16_s8_NCHW>:
   0:	1141                	addi	sp,sp,-16
   2:	c622                	sw	s0,12(sp)
   4:	43c2                	lw	t2,16(sp)
   6:	01410f83          	lb	t6,20(sp)
   a:	01810283          	lb	t0,24(sp)
   e:	01c14403          	lbu	s0,28(sp)
  12:	fff78313          	addi	t1,a5,-1
  16:	4e85                	li	t4,1
  18:	c426                	sw	s1,8(sp)
  1a:	006e9eb3          	sll	t4,t4,t1
  1e:	4e01                	li	t3,0

00000020 <.L46>:
  20:	00be4663          	blt	t3,a1,2c <.L49>
  24:	4432                	lw	s0,12(sp)
  26:	44a2                	lw	s1,8(sp)
  28:	0141                	addi	sp,sp,16
  2a:	8082                	ret

0000002c <.L49>:
  2c:	030e4333          	div	t1,t3,a6
  30:	00251f0b          	p.lh	t5,2(a0!)
  34:	9f46                	add	t5,t5,a7
  36:	030a                	slli	t1,t1,0x2
  38:	20667483          	p.lw	s1,t1(a2)
  3c:	2066f303          	p.lw	t1,t1(a3)
  40:	429f0333          	p.mac	t1,t5,s1
  44:	c401                	beqz	s0,4c <.L47>
  46:	00f05363          	blez	a5,4c <.L47>
  4a:	9376                	add	t1,t1,t4

0000004c <.L47>:
  4c:	40f35333          	sra	t1,t1,a5
  50:	931e                	add	t1,t1,t2
  52:	8f16                	mv	t5,t0
  54:	0062c763          	blt	t0,t1,62 <.L48>
  58:	8f7e                	mv	t5,t6
  5a:	01f34463          	blt	t1,t6,62 <.L48>
  5e:	10036f33          	p.extbs	t5,t1

00000062 <.L48>:
  62:	01e700ab          	p.sb	t5,1(a4!)
  66:	0e05                	addi	t3,t3,1
  68:	bf65                	j	20 <.L46>

Disassembly of section .text.RequantShift_s32_s8_NCHW:

00000000 <RequantShift_s32_s8_NCHW>:
   0:	1141                	addi	sp,sp,-16
   2:	c622                	sw	s0,12(sp)
   4:	43c2                	lw	t2,16(sp)
   6:	01410f83          	lb	t6,20(sp)
   a:	01810283          	lb	t0,24(sp)
   e:	01c14403          	lbu	s0,28(sp)
  12:	fff78313          	addi	t1,a5,-1
  16:	4e85                	li	t4,1
  18:	c426                	sw	s1,8(sp)
  1a:	006e9eb3          	sll	t4,t4,t1
  1e:	4e01                	li	t3,0

00000020 <.L57>:
  20:	00be4663          	blt	t3,a1,2c <.L60>
  24:	4432                	lw	s0,12(sp)
  26:	44a2                	lw	s1,8(sp)
  28:	0141                	addi	sp,sp,16
  2a:	8082                	ret

0000002c <.L60>:
  2c:	030e4333          	div	t1,t3,a6
  30:	00452f0b          	p.lw	t5,4(a0!)
  34:	9f46                	add	t5,t5,a7
  36:	030a                	slli	t1,t1,0x2
  38:	20667483          	p.lw	s1,t1(a2)
  3c:	2066f303          	p.lw	t1,t1(a3)
  40:	429f0333          	p.mac	t1,t5,s1
  44:	c401                	beqz	s0,4c <.L58>
  46:	00f05363          	blez	a5,4c <.L58>
  4a:	9376                	add	t1,t1,t4

0000004c <.L58>:
  4c:	40f35333          	sra	t1,t1,a5
  50:	931e                	add	t1,t1,t2
  52:	8f16                	mv	t5,t0
  54:	0062c763          	blt	t0,t1,62 <.L59>
  58:	8f7e                	mv	t5,t6
  5a:	01f34463          	blt	t1,t6,62 <.L59>
  5e:	10036f33          	p.extbs	t5,t1

00000062 <.L59>:
  62:	01e700ab          	p.sb	t5,1(a4!)
  66:	0e05                	addi	t3,t3,1
  68:	bf65                	j	20 <.L57>

Softmax_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Softmax_fp32_fp32 000000d0  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.SoftmaxGrad_fp32_fp32_fp32 000000aa  00000000  00000000  00000104  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .rodata.Softmax_fp32_fp32.cst4 00000008  00000000  00000000  000001b0  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
  6 .comment      0000001b  00000000  00000000  000001b8  2**0
                  CONTENTS, READONLY
  7 .Pulp_Chip.Info 00000072  00000000  00000000  000001d3  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Softmax_fp32_fp32:

00000000 <Softmax_fp32_fp32>:
   0:	715d                	addi	sp,sp,-80
   2:	d462                	sw	s8,40(sp)
   4:	02d64c33          	div	s8,a2,a3
   8:	000007b7          	lui	a5,0x0
   c:	d06a                	sw	s10,32(sp)
   e:	0007ad03          	lw	s10,0(a5) # 0 <Softmax_fp32_fp32>
  12:	000007b7          	lui	a5,0x0
  16:	d266                	sw	s9,36(sp)
  18:	ce6e                	sw	s11,28(sp)
  1a:	00000c93          	li	s9,0
  1e:	0007ad83          	lw	s11,0(a5) # 0 <Softmax_fp32_fp32>
  22:	c4a2                	sw	s0,72(sp)
  24:	c0ca                	sw	s2,64(sp)
  26:	da56                	sw	s5,52(sp)
  28:	c686                	sw	ra,76(sp)
  2a:	c2a6                	sw	s1,68(sp)
  2c:	de4e                	sw	s3,60(sp)
  2e:	dc52                	sw	s4,56(sp)
  30:	d85a                	sw	s6,48(sp)
  32:	d65e                	sw	s7,44(sp)
  34:	00269a93          	slli	s5,a3,0x2
  38:	842a                	mv	s0,a0
  3a:	4901                	li	s2,0

0000003c <.L2>:
  3c:	03894163          	blt	s2,s8,5e <.L10>
  40:	40b6                	lw	ra,76(sp)
  42:	4426                	lw	s0,72(sp)
  44:	4496                	lw	s1,68(sp)
  46:	4906                	lw	s2,64(sp)
  48:	59f2                	lw	s3,60(sp)
  4a:	5a62                	lw	s4,56(sp)
  4c:	5ad2                	lw	s5,52(sp)
  4e:	5b42                	lw	s6,48(sp)
  50:	5bb2                	lw	s7,44(sp)
  52:	5c22                	lw	s8,40(sp)
  54:	5c92                	lw	s9,36(sp)
  56:	5d02                	lw	s10,32(sp)
  58:	4df2                	lw	s11,28(sp)
  5a:	6161                	addi	sp,sp,80
  5c:	8082                	ret

0000005e <.L10>:
  5e:	199d74d3          	fdiv.s	fs1,fs10,fs9
  62:	8522                	mv	a0,s0
  64:	4781                	li	a5,0

00000066 <.L3>:
  66:	02d7c163          	blt	a5,a3,88 <.L5>
  6a:	00000993          	li	s3,0
  6e:	8bae                	mv	s7,a1
  70:	8b22                	mv	s6,s0
  72:	4a01                	li	s4,0

00000074 <.L6>:
  74:	02da4363          	blt	s4,a3,9a <.L7>
  78:	872e                	mv	a4,a1
  7a:	4781                	li	a5,0

0000007c <.L8>:
  7c:	04d7c163          	blt	a5,a3,be <.L9>
  80:	0905                	addi	s2,s2,1
  82:	95d6                	add	a1,a1,s5
  84:	9456                	add	s0,s0,s5
  86:	bf5d                	j	3c <.L2>

00000088 <.L5>:
  88:	0045270b          	p.lw	a4,4(a0!)
  8c:	a0e49653          	flt.s	a2,fs1,fa4
  90:	e211                	bnez	a2,94 <.L4>
  92:	8726                	mv	a4,s1

00000094 <.L4>:
  94:	0785                	addi	a5,a5,1
  96:	84ba                	mv	s1,a4
  98:	b7f9                	j	66 <.L3>

0000009a <.L7>:
  9a:	004b250b          	p.lw	a0,4(s6!)
  9e:	c62e                	sw	a1,12(sp)
  a0:	c436                	sw	a3,8(sp)
  a2:	08957553          	fsub.s	fa0,fa0,fs1
  a6:	0a05                	addi	s4,s4,1
  a8:	00000097          	auipc	ra,0x0
  ac:	000080e7          	jalr	ra
  b0:	00aba22b          	p.sw	a0,4(s7!)
  b4:	00a9f9d3          	fadd.s	fs3,fs3,fa0
  b8:	46a2                	lw	a3,8(sp)
  ba:	45b2                	lw	a1,12(sp)
  bc:	bf65                	j	74 <.L6>

000000be <.L9>:
  be:	193df553          	fdiv.s	fa0,fs11,fs3
  c2:	4310                	lw	a2,0(a4)
  c4:	0785                	addi	a5,a5,1
  c6:	10c57553          	fmul.s	fa0,fa0,fa2
  ca:	00a7222b          	p.sw	a0,4(a4!)
  ce:	b77d                	j	7c <.L8>

Disassembly of section .text.SoftmaxGrad_fp32_fp32_fp32:

00000000 <SoftmaxGrad_fp32_fp32_fp32>:
   0:	02e6c6b3          	div	a3,a3,a4
   4:	00271293          	slli	t0,a4,0x2
   8:	4781                	li	a5,0
   a:	4e81                	li	t4,0

0000000c <.L20>:
   c:	02dec263          	blt	t4,a3,30 <.L29>
  10:	8082                	ret

00000012 <.L16>:
  12:	0043280b          	p.lw	a6,4(t1!)
  16:	004e238b          	p.lw	t2,4(t3!)
  1a:	0f05                	addi	t5,t5,1
  1c:	09187853          	fsub.s	fa6,fa6,fa7
  20:	10787853          	fmul.s	fa6,fa6,ft7
  24:	010fa22b          	p.sw	a6,4(t6!)
  28:	a08d                	j	8a <.L15>

0000002a <.L12>:
  2a:	4432                	lw	s0,12(sp)
  2c:	0141                	addi	sp,sp,16
  2e:	8082                	ret

00000030 <.L29>:
  30:	00f50333          	add	t1,a0,a5
  34:	00f58e33          	add	t3,a1,a5
  38:	4801                	li	a6,0
  3a:	8ff2                	mv	t6,t3
  3c:	8f1a                	mv	t5,t1
  3e:	00000893          	li	a7,0
  42:	02e84663          	blt	a6,a4,6e <.L28>
  46:	00f60fb3          	add	t6,a2,a5
  4a:	4f01                	li	t5,0

0000004c <.L22>:
  4c:	00ef4563          	blt	t5,a4,56 <.L23>
  50:	0e85                	addi	t4,t4,1
  52:	9796                	add	a5,a5,t0
  54:	bf65                	j	c <.L20>

00000056 <.L23>:
  56:	0043280b          	p.lw	a6,4(t1!)
  5a:	004e238b          	p.lw	t2,4(t3!)
  5e:	0f05                	addi	t5,t5,1
  60:	09187853          	fsub.s	fa6,fa6,fa7
  64:	10787853          	fmul.s	fa6,fa6,ft7
  68:	010fa22b          	p.sw	a6,4(t6!)
  6c:	b7c5                	j	4c <.L22>

0000006e <.L28>:
  6e:	1141                	addi	sp,sp,-16
  70:	c622                	sw	s0,12(sp)

00000072 <.L14>:
  72:	004f240b          	p.lw	s0,4(t5!)
  76:	004fa38b          	p.lw	t2,4(t6!)
  7a:	0805                	addi	a6,a6,1
  7c:	887478c3          	fmadd.s	fa7,fs0,ft7,fa7

00000080 <.L18>:
  80:	fee849e3          	blt	a6,a4,72 <.L14>
  84:	00f60fb3          	add	t6,a2,a5
  88:	4f01                	li	t5,0

0000008a <.L15>:
  8a:	f8ef44e3          	blt	t5,a4,12 <.L16>
  8e:	0e85                	addi	t4,t4,1
  90:	9796                	add	a5,a5,t0
  92:	f8dedce3          	ble	a3,t4,2a <.L12>
  96:	00f50333          	add	t1,a0,a5
  9a:	00f58e33          	add	t3,a1,a5
  9e:	8ff2                	mv	t6,t3
  a0:	8f1a                	mv	t5,t1
  a2:	4801                	li	a6,0
  a4:	00000893          	li	a7,0
  a8:	bfe1                	j	80 <.L18>

Softmax_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Softmax_s8_s8 00000132  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.ITAMax_s8 000000a0  00000000  00000000  00000166  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .text.ITAPartialMax_s8 000000da  00000000  00000000  00000206  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  6 .comment      0000001b  00000000  00000000  000002e0  2**0
                  CONTENTS, READONLY
  7 .Pulp_Chip.Info 00000072  00000000  00000000  000002fb  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Softmax_s8_s8:

00000000 <Softmax_s8_s8>:
   0:	711d                	addi	sp,sp,-96
   2:	c4d2                	sw	s4,72(sp)
   4:	8a2a                	mv	s4,a0
   6:	00269513          	slli	a0,a3,0x2
   a:	c8ca                	sw	s2,80(sp)
   c:	c2d6                	sw	s5,68(sp)
   e:	5916                	lw	s2,100(sp)
  10:	dc62                	sw	s8,56(sp)
  12:	ce46                	sw	a7,28(sp)
  14:	ca2e                	sw	a1,20(sp)
  16:	c832                	sw	a2,16(sp)
  18:	c636                	sw	a3,12(sp)
  1a:	c43a                	sw	a4,8(sp)
  1c:	c23e                	sw	a5,4(sp)
  1e:	ce86                	sw	ra,92(sp)
  20:	cca2                	sw	s0,88(sp)
  22:	caa6                	sw	s1,84(sp)
  24:	c6ce                	sw	s3,76(sp)
  26:	c0da                	sw	s6,64(sp)
  28:	de5e                	sw	s7,60(sp)
  2a:	da66                	sw	s9,52(sp)
  2c:	d86a                	sw	s10,48(sp)
  2e:	d66e                	sw	s11,44(sp)
  30:	cc42                	sw	a6,24(sp)
  32:	5a86                	lw	s5,96(sp)
  34:	00000097          	auipc	ra,0x0
  38:	000080e7          	jalr	ra
  3c:	4642                	lw	a2,16(sp)
  3e:	46b2                	lw	a3,12(sp)
  40:	45d2                	lw	a1,20(sp)
  42:	4722                	lw	a4,8(sp)
  44:	02d65633          	divu	a2,a2,a3
  48:	4792                	lw	a5,4(sp)
  4a:	00195f13          	srli	t5,s2,0x1
  4e:	0fff7f13          	andi	t5,t5,255
  52:	40b002b3          	neg	t0,a1
  56:	41400333          	neg	t1,s4
  5a:	4881                	li	a7,0
  5c:	4f81                	li	t6,0
  5e:	4c7d                	li	s8,31
  60:	fff90e13          	addi	t3,s2,-1

00000064 <.L2>:
  64:	0bf60563          	beq	a2,t6,10e <.L8>
  68:	011a0833          	add	a6,s4,a7
  6c:	83c2                	mv	t2,a6
  6e:	f8000e93          	li	t4,-128
  72:	a029                	j	7c <.L9>

00000074 <.L3>:
  74:	0013890b          	p.lb	s2,1(t2!)
  78:	05d96eb3          	p.max	t4,s2,t4

0000007c <.L9>:
  7c:	00638933          	add	s2,t2,t1
  80:	fed96ae3          	bltu	s2,a3,74 <.L3>
  84:	8b2a                	mv	s6,a0
  86:	8baa                	mv	s7,a0
  88:	4381                	li	t2,0

0000008a <.L4>:
  8a:	00680933          	add	s2,a6,t1
  8e:	00d96f63          	bltu	s2,a3,ac <.L5>
  92:	01158eb3          	add	t4,a1,a7

00000096 <.L6>:
  96:	005e8833          	add	a6,t4,t0
  9a:	04d86f63          	bltu	a6,a3,f8 <.L7>
  9e:	0f85                	addi	t6,t6,1
  a0:	40d282b3          	sub	t0,t0,a3
  a4:	40d30333          	sub	t1,t1,a3
  a8:	98b6                	add	a7,a7,a3
  aa:	bf6d                	j	64 <.L2>

000000ac <.L5>:
  ac:	0018090b          	p.lb	s2,1(a6!)
  b0:	4462                	lw	s0,24(sp)
  b2:	44f2                	lw	s1,28(sp)
  b4:	41d90933          	sub	s2,s2,t4
  b8:	03594cb3          	div	s9,s2,s5
  bc:	41900cb3          	neg	s9,s9
  c0:	100cecb3          	p.extbs	s9,s9
  c4:	040cecb3          	p.max	s9,s9,zero
  c8:	058cccb3          	p.min	s9,s9,s8
  cc:	015c995b          	p.macu	s2,s9,s5
  d0:	10094933          	p.exths	s2,s2
  d4:	993e                	add	s2,s2,a5
  d6:	03290933          	mul	s2,s2,s2
  da:	02e90933          	mul	s2,s2,a4
  de:	8d4a                	mv	s10,s2
  e0:	41f95d93          	srai	s11,s2,0x1f
  e4:	208d0933          	add.d	s2,s10,s0
  e8:	8466                	mv	s0,s9
  ea:	4481                	li	s1,0
  ec:	20895933          	srl.d	s2,s2,s0
  f0:	012ba22b          	p.sw	s2,4(s7!)
  f4:	93ca                	add	t2,t2,s2
  f6:	bf51                	j	8a <.L4>

000000f8 <.L7>:
  f8:	004b280b          	p.lw	a6,4(s6!)
  fc:	030e0833          	mul	a6,t3,a6
 100:	02785833          	divu	a6,a6,t2
 104:	41e80833          	sub	a6,a6,t5
 108:	010e80ab          	p.sb	a6,1(t4!)
 10c:	b769                	j	96 <.L6>

0000010e <.L8>:
 10e:	4466                	lw	s0,88(sp)
 110:	40f6                	lw	ra,92(sp)
 112:	44d6                	lw	s1,84(sp)
 114:	4946                	lw	s2,80(sp)
 116:	49b6                	lw	s3,76(sp)
 118:	4a26                	lw	s4,72(sp)
 11a:	4a96                	lw	s5,68(sp)
 11c:	4b06                	lw	s6,64(sp)
 11e:	5bf2                	lw	s7,60(sp)
 120:	5c62                	lw	s8,56(sp)
 122:	5cd2                	lw	s9,52(sp)
 124:	5d42                	lw	s10,48(sp)
 126:	5db2                	lw	s11,44(sp)
 128:	6125                	addi	sp,sp,96
 12a:	00000317          	auipc	t1,0x0
 12e:	00030067          	jr	t1

Disassembly of section .text.ITAMax_s8:

00000000 <ITAMax_s8>:
   0:	02e6d6b3          	divu	a3,a3,a4
   4:	1141                	addi	sp,sp,-16
   6:	00879f13          	slli	t5,a5,0x8
   a:	8385                	srli	a5,a5,0x1
   c:	c426                	sw	s1,8(sp)
   e:	c24a                	sw	s2,4(sp)
  10:	c622                	sw	s0,12(sp)
  12:	c04e                	sw	s3,0(sp)
  14:	f00f0f13          	addi	t5,t5,-256
  18:	0ff7f793          	andi	a5,a5,255
  1c:	40a002b3          	neg	t0,a0
  20:	4301                	li	t1,0
  22:	4f81                	li	t6,0
  24:	00c704b3          	add	s1,a4,a2
  28:	10000913          	li	s2,256

0000002c <.L12>:
  2c:	07f68463          	beq	a3,t6,94 <.L11>
  30:	00650eb3          	add	t4,a0,t1
  34:	8876                	mv	a6,t4
  36:	f8000e13          	li	t3,-128
  3a:	a029                	j	44 <.L19>

0000003c <.L13>:
  3c:	0018088b          	p.lb	a7,1(a6!)
  40:	05c8ee33          	p.max	t3,a7,t3

00000044 <.L19>:
  44:	005808b3          	add	a7,a6,t0
  48:	fee8eae3          	bltu	a7,a4,3c <.L13>
  4c:	83b2                	mv	t2,a2
  4e:	8432                	mv	s0,a2
  50:	4881                	li	a7,0

00000052 <.L14>:
  52:	00941d63          	bne	s0,s1,6c <.L15>
  56:	031f58b3          	divu	a7,t5,a7
  5a:	00658e33          	add	t3,a1,t1

0000005e <.L16>:
  5e:	02939263          	bne	t2,s1,82 <.L17>
  62:	0f85                	addi	t6,t6,1
  64:	40e282b3          	sub	t0,t0,a4
  68:	933a                	add	t1,t1,a4
  6a:	b7c9                	j	2c <.L12>

0000006c <.L15>:
  6c:	001e880b          	p.lb	a6,1(t4!)
  70:	0b0e785b          	p.mac.zh.zh	a6,t3,a6,t0
  74:	89c2                	mv	s3,a6
  76:	01095833          	srl	a6,s2,a6
  7a:	013400ab          	p.sb	s3,1(s0!)
  7e:	98c2                	add	a7,a7,a6
  80:	bfc9                	j	52 <.L14>

00000082 <.L17>:
  82:	0013c80b          	p.lbu	a6,1(t2!)
  86:	0108d833          	srl	a6,a7,a6
  8a:	40f80833          	sub	a6,a6,a5
  8e:	010e00ab          	p.sb	a6,1(t3!)
  92:	b7f1                	j	5e <.L16>

00000094 <.L11>:
  94:	4432                	lw	s0,12(sp)
  96:	44a2                	lw	s1,8(sp)
  98:	4912                	lw	s2,4(sp)
  9a:	4982                	lw	s3,0(sp)
  9c:	0141                	addi	sp,sp,16
  9e:	8082                	ret

Disassembly of section .text.ITAPartialMax_s8:

00000000 <ITAPartialMax_s8>:
   0:	02d65633          	divu	a2,a2,a3
   4:	8385                	srli	a5,a5,0x1
   6:	1101                	addi	sp,sp,-32
   8:	00879293          	slli	t0,a5,0x8
   c:	ca4a                	sw	s2,20(sp)
   e:	ce22                	sw	s0,28(sp)
  10:	cc26                	sw	s1,24(sp)
  12:	c84e                	sw	s3,16(sp)
  14:	c652                	sw	s4,12(sp)
  16:	c456                	sw	s5,8(sp)
  18:	c25a                	sw	s6,4(sp)
  1a:	f0028293          	addi	t0,t0,-256
  1e:	40a00eb3          	neg	t4,a0
  22:	4f01                	li	t5,0
  24:	4381                	li	t2,0
  26:	10000913          	li	s2,256

0000002a <.L22>:
  2a:	08760f63          	beq	a2,t2,c8 <.L21>
  2e:	01e50e33          	add	t3,a0,t5
  32:	8a76                	mv	s4,t4
  34:	89f2                	mv	s3,t3
  36:	f8000893          	li	a7,-128
  3a:	4801                	li	a6,0
  3c:	4a81                	li	s5,0
  3e:	02e6db33          	divu	s6,a3,a4
  42:	a82d                	j	7c <.L31>

00000044 <.L23>:
  44:	001f840b          	p.lb	s0,1(t6!)
  48:	04836333          	p.max	t1,t1,s0

0000004c <.L27>:
  4c:	014f8433          	add	s0,t6,s4
  50:	fee46ae3          	bltu	s0,a4,44 <.L23>
  54:	4f81                	li	t6,0
  56:	0068d463          	ble	t1,a7,5e <.L24>
  5a:	0b137fdb          	p.mac.zh.zh	t6,t1,a7,t0

0000005e <.L24>:
  5e:	0468e8b3          	p.max	a7,a7,t1
  62:	84ce                	mv	s1,s3
  64:	4401                	li	s0,0

00000066 <.L25>:
  66:	01448333          	add	t1,s1,s4
  6a:	02e36c63          	bltu	t1,a4,a2 <.L26>
  6e:	01f85833          	srl	a6,a6,t6
  72:	9822                	add	a6,a6,s0
  74:	0a85                	addi	s5,s5,1
  76:	99ba                	add	s3,s3,a4
  78:	40ea0a33          	sub	s4,s4,a4

0000007c <.L31>:
  7c:	015b1f63          	bne	s6,s5,9a <.L33>
  80:	0302d433          	divu	s0,t0,a6
  84:	01e58333          	add	t1,a1,t5

00000088 <.L28>:
  88:	01de0fb3          	add	t6,t3,t4
  8c:	02dfe363          	bltu	t6,a3,b2 <.L29>
  90:	0385                	addi	t2,t2,1
  92:	40de8eb3          	sub	t4,t4,a3
  96:	9f36                	add	t5,t5,a3
  98:	bf49                	j	2a <.L22>

0000009a <.L33>:
  9a:	8fce                	mv	t6,s3
  9c:	f8000313          	li	t1,-128
  a0:	b775                	j	4c <.L27>

000000a2 <.L26>:
  a2:	0014830b          	p.lb	t1,1(s1!)
  a6:	0a68f35b          	p.mac.zh.zh	t1,a7,t1,t0
  aa:	00695333          	srl	t1,s2,t1
  ae:	941a                	add	s0,s0,t1
  b0:	bf5d                	j	66 <.L25>

000000b2 <.L29>:
  b2:	001e080b          	p.lb	a6,1(t3!)
  b6:	0b08f85b          	p.mac.zh.zh	a6,a7,a6,t0
  ba:	01045833          	srl	a6,s0,a6
  be:	40f80833          	sub	a6,a6,a5
  c2:	010300ab          	p.sb	a6,1(t1!) # 12b <.L8+0x1d>
  c6:	b7c9                	j	88 <.L28>

000000c8 <.L21>:
  c8:	4472                	lw	s0,28(sp)
  ca:	44e2                	lw	s1,24(sp)
  cc:	4952                	lw	s2,20(sp)
  ce:	49c2                	lw	s3,16(sp)
  d0:	4a32                	lw	s4,12(sp)
  d2:	4aa2                	lw	s5,8(sp)
  d4:	4b12                	lw	s6,4(sp)
  d6:	6105                	addi	sp,sp,32
  d8:	8082                	ret

Sqrt_fp32.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.Sqrt_fp32_fp32 00000036  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .comment      0000001b  00000000  00000000  0000006a  2**0
                  CONTENTS, READONLY
  5 .Pulp_Chip.Info 00000072  00000000  00000000  00000085  2**0
                  CONTENTS, READONLY

Disassembly of section .text.Sqrt_fp32_fp32:

00000000 <Sqrt_fp32_fp32>:
   0:	1101                	addi	sp,sp,-32
   2:	cc22                	sw	s0,24(sp)
   4:	ca26                	sw	s1,20(sp)
   6:	ce06                	sw	ra,28(sp)
   8:	84aa                	mv	s1,a0
   a:	4401                	li	s0,0

0000000c <.L2>:
   c:	00c44763          	blt	s0,a2,1a <.L3>
  10:	40f2                	lw	ra,28(sp)
  12:	4462                	lw	s0,24(sp)
  14:	44d2                	lw	s1,20(sp)
  16:	6105                	addi	sp,sp,32
  18:	8082                	ret

0000001a <.L3>:
  1a:	0044a50b          	p.lw	a0,4(s1!)
  1e:	c62e                	sw	a1,12(sp)
  20:	c432                	sw	a2,8(sp)
  22:	00000097          	auipc	ra,0x0
  26:	000080e7          	jalr	ra
  2a:	45b2                	lw	a1,12(sp)
  2c:	0405                	addi	s0,s0,1
  2e:	4622                	lw	a2,8(sp)
  30:	00a5a22b          	p.sw	a0,4(a1!)
  34:	bfe1                	j	c <.L2>

Util.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text.PrintMatrix_s8_NCHW 00000110  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.PrintMatrix_s8_NHWC 00000102  00000000  00000000  00000144  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .text.PrintMatrix_s16_NCHW 00000100  00000000  00000000  00000246  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  6 .text.PrintMatrix_s16_NHWC 00000114  00000000  00000000  00000346  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  7 .text.PrintMatrix_s32_NCHW 000000fc  00000000  00000000  0000045a  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  8 .text.PrintMatrix_s32_NHWC 00000114  00000000  00000000  00000556  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  9 .text.PrintArray_s8 0000004e  00000000  00000000  0000066a  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 10 .text.PrintArray_s16 00000054  00000000  00000000  000006b8  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 11 .text.PrintArray_s32 00000050  00000000  00000000  0000070c  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 12 .text.PrintMatrix_u8_NCHW 00000110  00000000  00000000  0000075c  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 13 .text.PrintMatrix_u8_NHWC 00000102  00000000  00000000  0000086c  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 14 .text.PrintMatrix_u16_NCHW 00000100  00000000  00000000  0000096e  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 15 .text.PrintMatrix_u16_NHWC 00000114  00000000  00000000  00000a6e  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 16 .text.PrintMatrix_u32_NCHW 000000fc  00000000  00000000  00000b82  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 17 .text.PrintMatrix_u32_NHWC 00000114  00000000  00000000  00000c7e  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 18 .text.PrintArray_u8 0000004e  00000000  00000000  00000d92  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 19 .text.PrintArray_u16 00000054  00000000  00000000  00000de0  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 20 .text.PrintArray_u32 00000050  00000000  00000000  00000e34  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
 21 .rodata.PrintArray_s8.str1.4 00000003  00000000  00000000  00000e84  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 22 .rodata.PrintMatrix_s16_NCHW.str1.4 00000006  00000000  00000000  00000e88  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 23 .rodata.PrintMatrix_s32_NCHW.str1.4 00000007  00000000  00000000  00000e90  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 24 .rodata.PrintMatrix_s8_NCHW.str1.4 00000020  00000000  00000000  00000e98  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 25 .rodata.PrintMatrix_u16_NCHW.str1.4 00000006  00000000  00000000  00000eb8  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 26 .rodata.PrintMatrix_u32_NCHW.str1.4 00000007  00000000  00000000  00000ec0  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 27 .rodata.PrintMatrix_u8_NCHW.str1.4 00000005  00000000  00000000  00000ec8  2**2
                  CONTENTS, ALLOC, LOAD, READONLY, DATA
 28 .comment      0000001b  00000000  00000000  00000ecd  2**0
                  CONTENTS, READONLY
 29 .Pulp_Chip.Info 00000072  00000000  00000000  00000ee8  2**0
                  CONTENTS, READONLY

Disassembly of section .text.PrintMatrix_s8_NCHW:

00000000 <PrintMatrix_s8_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	c0da                	sw	s6,64(sp)
   4:	02e68b33          	mul	s6,a3,a4
   8:	de5e                	sw	s7,60(sp)
   a:	cca2                	sw	s0,88(sp)
   c:	c8ca                	sw	s2,80(sp)
   e:	c6ce                	sw	s3,76(sp)
  10:	c4d2                	sw	s4,72(sp)
  12:	dc62                	sw	s8,56(sp)
  14:	da66                	sw	s9,52(sp)
  16:	ce86                	sw	ra,92(sp)
  18:	caa6                	sw	s1,84(sp)
  1a:	03660bb3          	mul	s7,a2,s6
  1e:	c2d6                	sw	s5,68(sp)
  20:	d86a                	sw	s10,48(sp)
  22:	d66e                	sw	s11,44(sp)
  24:	c82e                	sw	a1,16(sp)
  26:	8c32                	mv	s8,a2
  28:	8cb6                	mv	s9,a3
  2a:	8a3a                	mv	s4,a4
  2c:	ca3e                	sw	a5,20(sp)
  2e:	842a                	mv	s0,a0
  30:	00e509b3          	add	s3,a0,a4
  34:	4901                	li	s2,0

00000036 <.L2>:
  36:	47c2                	lw	a5,16(sp)
  38:	02f91163          	bne	s2,a5,5a <.L9>
  3c:	40f6                	lw	ra,92(sp)
  3e:	4466                	lw	s0,88(sp)
  40:	44d6                	lw	s1,84(sp)
  42:	4946                	lw	s2,80(sp)
  44:	49b6                	lw	s3,76(sp)
  46:	4a26                	lw	s4,72(sp)
  48:	4a96                	lw	s5,68(sp)
  4a:	4b06                	lw	s6,64(sp)
  4c:	5bf2                	lw	s7,60(sp)
  4e:	5c62                	lw	s8,56(sp)
  50:	5cd2                	lw	s9,52(sp)
  52:	5d42                	lw	s10,48(sp)
  54:	5db2                	lw	s11,44(sp)
  56:	6125                	addi	sp,sp,96
  58:	8082                	ret

0000005a <.L9>:
  5a:	000007b7          	lui	a5,0x0
  5e:	00078513          	mv	a0,a5
  62:	00000097          	auipc	ra,0x0
  66:	000080e7          	jalr	ra
  6a:	4481                	li	s1,0
  6c:	4a81                	li	s5,0

0000006e <.L3>:
  6e:	018a9e63          	bne	s5,s8,8a <.L8>
  72:	00000537          	lui	a0,0x0
  76:	00050513          	mv	a0,a0
  7a:	00000097          	auipc	ra,0x0
  7e:	000080e7          	jalr	ra
  82:	0905                	addi	s2,s2,1
  84:	945e                	add	s0,s0,s7
  86:	99de                	add	s3,s3,s7
  88:	b77d                	j	36 <.L2>

0000008a <.L8>:
  8a:	000007b7          	lui	a5,0x0
  8e:	00078513          	mv	a0,a5
  92:	00000097          	auipc	ra,0x0
  96:	000080e7          	jalr	ra
  9a:	008487b3          	add	a5,s1,s0
  9e:	cc3e                	sw	a5,24(sp)
  a0:	013487b3          	add	a5,s1,s3
  a4:	4d01                	li	s10,0
  a6:	4d81                	li	s11,0
  a8:	ce3e                	sw	a5,28(sp)

000000aa <.L4>:
  aa:	059d8863          	beq	s11,s9,fa <.L6>
  ae:	47e2                	lw	a5,24(sp)
  b0:	01a78733          	add	a4,a5,s10
  b4:	47f2                	lw	a5,28(sp)
  b6:	01a78633          	add	a2,a5,s10
  ba:	a01d                	j	e0 <.L7>

000000bc <.L5>:
  bc:	0017058b          	p.lb	a1,1(a4!)
  c0:	47d2                	lw	a5,20(sp)
  c2:	c432                	sw	a2,8(sp)
  c4:	c63a                	sw	a4,12(sp)
  c6:	95be                	add	a1,a1,a5
  c8:	000007b7          	lui	a5,0x0
  cc:	1005e5b3          	p.extbs	a1,a1
  d0:	00078513          	mv	a0,a5
  d4:	00000097          	auipc	ra,0x0
  d8:	000080e7          	jalr	ra
  dc:	4732                	lw	a4,12(sp)
  de:	4622                	lw	a2,8(sp)

000000e0 <.L7>:
  e0:	fcc71ee3          	bne	a4,a2,bc <.L5>
  e4:	000007b7          	lui	a5,0x0
  e8:	00078513          	mv	a0,a5
  ec:	00000097          	auipc	ra,0x0
  f0:	000080e7          	jalr	ra
  f4:	0d85                	addi	s11,s11,1
  f6:	9d52                	add	s10,s10,s4
  f8:	bf4d                	j	aa <.L4>

000000fa <.L6>:
  fa:	000007b7          	lui	a5,0x0
  fe:	00078513          	mv	a0,a5
 102:	00000097          	auipc	ra,0x0
 106:	000080e7          	jalr	ra
 10a:	0a85                	addi	s5,s5,1
 10c:	94da                	add	s1,s1,s6
 10e:	b785                	j	6e <.L3>

Disassembly of section .text.PrintMatrix_s8_NHWC:

00000000 <PrintMatrix_s8_NHWC>:
   0:	711d                	addi	sp,sp,-96
   2:	ce3e                	sw	a5,28(sp)
   4:	02e687b3          	mul	a5,a3,a4
   8:	caa6                	sw	s1,84(sp)
   a:	c8ca                	sw	s2,80(sp)
   c:	c6ce                	sw	s3,76(sp)
   e:	c4d2                	sw	s4,72(sp)
  10:	c2d6                	sw	s5,68(sp)
  12:	dc62                	sw	s8,56(sp)
  14:	da66                	sw	s9,52(sp)
  16:	ce86                	sw	ra,92(sp)
  18:	cca2                	sw	s0,88(sp)
  1a:	02c787b3          	mul	a5,a5,a2
  1e:	c0da                	sw	s6,64(sp)
  20:	de5e                	sw	s7,60(sp)
  22:	d86a                	sw	s10,48(sp)
  24:	d66e                	sw	s11,44(sp)
  26:	cc2e                	sw	a1,24(sp)
  28:	89b2                	mv	s3,a2
  2a:	8ab6                	mv	s5,a3
  2c:	8a3a                	mv	s4,a4
  2e:	892a                	mv	s2,a0
  30:	ca3e                	sw	a5,20(sp)
  32:	02e607b3          	mul	a5,a2,a4
  36:	4481                	li	s1,0
  38:	00000c37          	lui	s8,0x0
  3c:	00000cb7          	lui	s9,0x0
  40:	c83e                	sw	a5,16(sp)

00000042 <.L12>:
  42:	47e2                	lw	a5,24(sp)
  44:	02f49163          	bne	s1,a5,66 <.L18>
  48:	40f6                	lw	ra,92(sp)
  4a:	4466                	lw	s0,88(sp)
  4c:	44d6                	lw	s1,84(sp)
  4e:	4946                	lw	s2,80(sp)
  50:	49b6                	lw	s3,76(sp)
  52:	4a26                	lw	s4,72(sp)
  54:	4a96                	lw	s5,68(sp)
  56:	4b06                	lw	s6,64(sp)
  58:	5bf2                	lw	s7,60(sp)
  5a:	5c62                	lw	s8,56(sp)
  5c:	5cd2                	lw	s9,52(sp)
  5e:	5d42                	lw	s10,48(sp)
  60:	5db2                	lw	s11,44(sp)
  62:	6125                	addi	sp,sp,96
  64:	8082                	ret

00000066 <.L18>:
  66:	000007b7          	lui	a5,0x0
  6a:	00078513          	mv	a0,a5
  6e:	00000097          	auipc	ra,0x0
  72:	000080e7          	jalr	ra
  76:	4401                	li	s0,0
  78:	00000d37          	lui	s10,0x0
  7c:	00000db7          	lui	s11,0x0

00000080 <.L13>:
  80:	01341e63          	bne	s0,s3,9c <.L17>
  84:	00000537          	lui	a0,0x0
  88:	00050513          	mv	a0,a0
  8c:	00000097          	auipc	ra,0x0
  90:	000080e7          	jalr	ra
  94:	47d2                	lw	a5,20(sp)
  96:	0485                	addi	s1,s1,1
  98:	993e                	add	s2,s2,a5
  9a:	b765                	j	42 <.L12>

0000009c <.L17>:
  9c:	000d0513          	mv	a0,s10
  a0:	00000097          	auipc	ra,0x0
  a4:	000080e7          	jalr	ra
  a8:	01240bb3          	add	s7,s0,s2
  ac:	4b01                	li	s6,0

000000ae <.L14>:
  ae:	055b1763          	bne	s6,s5,fc <.L19>
  b2:	000d8513          	mv	a0,s11
  b6:	00000097          	auipc	ra,0x0
  ba:	000080e7          	jalr	ra
  be:	0405                	addi	s0,s0,1
  c0:	b7c1                	j	80 <.L13>

000000c2 <.L15>:
  c2:	4136758b          	p.lbu	a1,s3(a2!)
  c6:	47f2                	lw	a5,28(sp)
  c8:	000c0513          	mv	a0,s8
  cc:	c632                	sw	a2,12(sp)
  ce:	95be                	add	a1,a1,a5
  d0:	1005e5b3          	p.extbs	a1,a1
  d4:	c436                	sw	a3,8(sp)
  d6:	00000097          	auipc	ra,0x0
  da:	000080e7          	jalr	ra
  de:	46a2                	lw	a3,8(sp)
  e0:	4632                	lw	a2,12(sp)
  e2:	0685                	addi	a3,a3,1

000000e4 <.L16>:
  e4:	fd469fe3          	bne	a3,s4,c2 <.L15>
  e8:	000c8513          	mv	a0,s9
  ec:	00000097          	auipc	ra,0x0
  f0:	000080e7          	jalr	ra
  f4:	47c2                	lw	a5,16(sp)
  f6:	0b05                	addi	s6,s6,1
  f8:	9bbe                	add	s7,s7,a5
  fa:	bf55                	j	ae <.L14>

000000fc <.L19>:
  fc:	865e                	mv	a2,s7
  fe:	4681                	li	a3,0
 100:	b7d5                	j	e4 <.L16>

Disassembly of section .text.PrintMatrix_s16_NCHW:

00000000 <PrintMatrix_s16_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	c4d2                	sw	s4,72(sp)
   4:	00171a13          	slli	s4,a4,0x1
   8:	c2d6                	sw	s5,68(sp)
   a:	03468ab3          	mul	s5,a3,s4
   e:	ce3e                	sw	a5,28(sp)
  10:	cca2                	sw	s0,88(sp)
  12:	caa6                	sw	s1,84(sp)
  14:	c0da                	sw	s6,64(sp)
  16:	de5e                	sw	s7,60(sp)
  18:	dc62                	sw	s8,56(sp)
  1a:	ce86                	sw	ra,92(sp)
  1c:	c8ca                	sw	s2,80(sp)
  1e:	c6ce                	sw	s3,76(sp)
  20:	035607b3          	mul	a5,a2,s5
  24:	da66                	sw	s9,52(sp)
  26:	d86a                	sw	s10,48(sp)
  28:	d66e                	sw	s11,44(sp)
  2a:	cc2e                	sw	a1,24(sp)
  2c:	8b32                	mv	s6,a2
  2e:	8bb6                	mv	s7,a3
  30:	8c3a                	mv	s8,a4
  32:	84aa                	mv	s1,a0
  34:	4401                	li	s0,0
  36:	ca3e                	sw	a5,20(sp)

00000038 <.L22>:
  38:	47e2                	lw	a5,24(sp)
  3a:	02f41163          	bne	s0,a5,5c <.L28>
  3e:	40f6                	lw	ra,92(sp)
  40:	4466                	lw	s0,88(sp)
  42:	44d6                	lw	s1,84(sp)
  44:	4946                	lw	s2,80(sp)
  46:	49b6                	lw	s3,76(sp)
  48:	4a26                	lw	s4,72(sp)
  4a:	4a96                	lw	s5,68(sp)
  4c:	4b06                	lw	s6,64(sp)
  4e:	5bf2                	lw	s7,60(sp)
  50:	5c62                	lw	s8,56(sp)
  52:	5cd2                	lw	s9,52(sp)
  54:	5d42                	lw	s10,48(sp)
  56:	5db2                	lw	s11,44(sp)
  58:	6125                	addi	sp,sp,96
  5a:	8082                	ret

0000005c <.L28>:
  5c:	000007b7          	lui	a5,0x0
  60:	00078513          	mv	a0,a5
  64:	00000097          	auipc	ra,0x0
  68:	000080e7          	jalr	ra
  6c:	89a6                	mv	s3,s1
  6e:	4901                	li	s2,0
  70:	00000db7          	lui	s11,0x0

00000074 <.L23>:
  74:	01691e63          	bne	s2,s6,90 <.L27>
  78:	00000537          	lui	a0,0x0
  7c:	00050513          	mv	a0,a0
  80:	00000097          	auipc	ra,0x0
  84:	000080e7          	jalr	ra
  88:	47d2                	lw	a5,20(sp)
  8a:	0405                	addi	s0,s0,1
  8c:	94be                	add	s1,s1,a5
  8e:	b76d                	j	38 <.L22>

00000090 <.L27>:
  90:	000007b7          	lui	a5,0x0
  94:	00078513          	mv	a0,a5
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	8d4e                	mv	s10,s3
  a2:	4c81                	li	s9,0

000000a4 <.L24>:
  a4:	057c9b63          	bne	s9,s7,fa <.L29>
  a8:	000d8513          	mv	a0,s11
  ac:	00000097          	auipc	ra,0x0
  b0:	000080e7          	jalr	ra
  b4:	0905                	addi	s2,s2,1
  b6:	99d6                	add	s3,s3,s5
  b8:	bf75                	j	74 <.L23>

000000ba <.L25>:
  ba:	0026158b          	p.lh	a1,2(a2!)
  be:	47f2                	lw	a5,28(sp)
  c0:	c836                	sw	a3,16(sp)
  c2:	c632                	sw	a2,12(sp)
  c4:	95be                	add	a1,a1,a5
  c6:	000007b7          	lui	a5,0x0
  ca:	1005c5b3          	p.exths	a1,a1
  ce:	00078513          	mv	a0,a5
  d2:	00000097          	auipc	ra,0x0
  d6:	000080e7          	jalr	ra
  da:	46c2                	lw	a3,16(sp)
  dc:	4632                	lw	a2,12(sp)
  de:	0685                	addi	a3,a3,1

000000e0 <.L26>:
  e0:	fd869de3          	bne	a3,s8,ba <.L25>
  e4:	000007b7          	lui	a5,0x0
  e8:	00078513          	mv	a0,a5
  ec:	00000097          	auipc	ra,0x0
  f0:	000080e7          	jalr	ra
  f4:	0c85                	addi	s9,s9,1
  f6:	9d52                	add	s10,s10,s4
  f8:	b775                	j	a4 <.L24>

000000fa <.L29>:
  fa:	866a                	mv	a2,s10
  fc:	4681                	li	a3,0
  fe:	b7cd                	j	e0 <.L26>

Disassembly of section .text.PrintMatrix_s16_NHWC:

00000000 <PrintMatrix_s16_NHWC>:
   0:	711d                	addi	sp,sp,-96
   2:	ce3e                	sw	a5,28(sp)
   4:	02e687b3          	mul	a5,a3,a4
   8:	cca2                	sw	s0,88(sp)
   a:	c8ca                	sw	s2,80(sp)
   c:	c6ce                	sw	s3,76(sp)
   e:	c4d2                	sw	s4,72(sp)
  10:	c2d6                	sw	s5,68(sp)
  12:	dc62                	sw	s8,56(sp)
  14:	d66e                	sw	s11,44(sp)
  16:	ce86                	sw	ra,92(sp)
  18:	caa6                	sw	s1,84(sp)
  1a:	02c787b3          	mul	a5,a5,a2
  1e:	c0da                	sw	s6,64(sp)
  20:	de5e                	sw	s7,60(sp)
  22:	da66                	sw	s9,52(sp)
  24:	d86a                	sw	s10,48(sp)
  26:	ca2a                	sw	a0,20(sp)
  28:	cc2e                	sw	a1,24(sp)
  2a:	8432                	mv	s0,a2
  2c:	8ab6                	mv	s5,a3
  2e:	8a3a                	mv	s4,a4
  30:	c83e                	sw	a5,16(sp)
  32:	00171793          	slli	a5,a4,0x1
  36:	02c787b3          	mul	a5,a5,a2
  3a:	00161d93          	slli	s11,a2,0x1
  3e:	4981                	li	s3,0
  40:	4901                	li	s2,0
  42:	00000c37          	lui	s8,0x0
  46:	c63e                	sw	a5,12(sp)

00000048 <.L32>:
  48:	47e2                	lw	a5,24(sp)
  4a:	02f91163          	bne	s2,a5,6c <.L38>
  4e:	40f6                	lw	ra,92(sp)
  50:	4466                	lw	s0,88(sp)
  52:	44d6                	lw	s1,84(sp)
  54:	4946                	lw	s2,80(sp)
  56:	49b6                	lw	s3,76(sp)
  58:	4a26                	lw	s4,72(sp)
  5a:	4a96                	lw	s5,68(sp)
  5c:	4b06                	lw	s6,64(sp)
  5e:	5bf2                	lw	s7,60(sp)
  60:	5c62                	lw	s8,56(sp)
  62:	5cd2                	lw	s9,52(sp)
  64:	5d42                	lw	s10,48(sp)
  66:	5db2                	lw	s11,44(sp)
  68:	6125                	addi	sp,sp,96
  6a:	8082                	ret

0000006c <.L38>:
  6c:	00000737          	lui	a4,0x0
  70:	00070513          	mv	a0,a4
  74:	00000097          	auipc	ra,0x0
  78:	000080e7          	jalr	ra
  7c:	4481                	li	s1,0
  7e:	00000cb7          	lui	s9,0x0
  82:	00000d37          	lui	s10,0x0

00000086 <.L33>:
  86:	00849e63          	bne	s1,s0,a2 <.L37>
  8a:	00000537          	lui	a0,0x0
  8e:	00050513          	mv	a0,a0
  92:	00000097          	auipc	ra,0x0
  96:	000080e7          	jalr	ra
  9a:	4742                	lw	a4,16(sp)
  9c:	0905                	addi	s2,s2,1
  9e:	99ba                	add	s3,s3,a4
  a0:	b765                	j	48 <.L32>

000000a2 <.L37>:
  a2:	000c8513          	mv	a0,s9
  a6:	00000097          	auipc	ra,0x0
  aa:	000080e7          	jalr	ra
  ae:	4752                	lw	a4,20(sp)
  b0:	013487b3          	add	a5,s1,s3
  b4:	0786                	slli	a5,a5,0x1
  b6:	00f70bb3          	add	s7,a4,a5
  ba:	4b01                	li	s6,0

000000bc <.L34>:
  bc:	055b1963          	bne	s6,s5,10e <.L39>
  c0:	000d0513          	mv	a0,s10
  c4:	00000097          	auipc	ra,0x0
  c8:	000080e7          	jalr	ra
  cc:	0485                	addi	s1,s1,1
  ce:	bf65                	j	86 <.L33>

000000d0 <.L35>:
  d0:	51b6758b          	p.lhu	a1,s11(a2!)
  d4:	47f2                	lw	a5,28(sp)
  d6:	00000737          	lui	a4,0x0
  da:	00070513          	mv	a0,a4
  de:	95be                	add	a1,a1,a5
  e0:	1005c5b3          	p.exths	a1,a1
  e4:	c432                	sw	a2,8(sp)
  e6:	c236                	sw	a3,4(sp)
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	4692                	lw	a3,4(sp)
  f2:	4622                	lw	a2,8(sp)
  f4:	0685                	addi	a3,a3,1

000000f6 <.L36>:
  f6:	fd469de3          	bne	a3,s4,d0 <.L35>
  fa:	000c0513          	mv	a0,s8
  fe:	00000097          	auipc	ra,0x0
 102:	000080e7          	jalr	ra
 106:	47b2                	lw	a5,12(sp)
 108:	0b05                	addi	s6,s6,1
 10a:	9bbe                	add	s7,s7,a5
 10c:	bf45                	j	bc <.L34>

0000010e <.L39>:
 10e:	865e                	mv	a2,s7
 110:	4681                	li	a3,0
 112:	b7d5                	j	f6 <.L36>

Disassembly of section .text.PrintMatrix_s32_NCHW:

00000000 <PrintMatrix_s32_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	c4d2                	sw	s4,72(sp)
   4:	00271a13          	slli	s4,a4,0x2
   8:	c2d6                	sw	s5,68(sp)
   a:	03468ab3          	mul	s5,a3,s4
   e:	ce3e                	sw	a5,28(sp)
  10:	cca2                	sw	s0,88(sp)
  12:	caa6                	sw	s1,84(sp)
  14:	c0da                	sw	s6,64(sp)
  16:	de5e                	sw	s7,60(sp)
  18:	dc62                	sw	s8,56(sp)
  1a:	ce86                	sw	ra,92(sp)
  1c:	c8ca                	sw	s2,80(sp)
  1e:	c6ce                	sw	s3,76(sp)
  20:	035607b3          	mul	a5,a2,s5
  24:	da66                	sw	s9,52(sp)
  26:	d86a                	sw	s10,48(sp)
  28:	d66e                	sw	s11,44(sp)
  2a:	cc2e                	sw	a1,24(sp)
  2c:	8b32                	mv	s6,a2
  2e:	8bb6                	mv	s7,a3
  30:	8c3a                	mv	s8,a4
  32:	84aa                	mv	s1,a0
  34:	4401                	li	s0,0
  36:	ca3e                	sw	a5,20(sp)

00000038 <.L42>:
  38:	47e2                	lw	a5,24(sp)
  3a:	02f41163          	bne	s0,a5,5c <.L48>
  3e:	40f6                	lw	ra,92(sp)
  40:	4466                	lw	s0,88(sp)
  42:	44d6                	lw	s1,84(sp)
  44:	4946                	lw	s2,80(sp)
  46:	49b6                	lw	s3,76(sp)
  48:	4a26                	lw	s4,72(sp)
  4a:	4a96                	lw	s5,68(sp)
  4c:	4b06                	lw	s6,64(sp)
  4e:	5bf2                	lw	s7,60(sp)
  50:	5c62                	lw	s8,56(sp)
  52:	5cd2                	lw	s9,52(sp)
  54:	5d42                	lw	s10,48(sp)
  56:	5db2                	lw	s11,44(sp)
  58:	6125                	addi	sp,sp,96
  5a:	8082                	ret

0000005c <.L48>:
  5c:	000007b7          	lui	a5,0x0
  60:	00078513          	mv	a0,a5
  64:	00000097          	auipc	ra,0x0
  68:	000080e7          	jalr	ra
  6c:	89a6                	mv	s3,s1
  6e:	4901                	li	s2,0
  70:	00000db7          	lui	s11,0x0

00000074 <.L43>:
  74:	01691e63          	bne	s2,s6,90 <.L47>
  78:	00000537          	lui	a0,0x0
  7c:	00050513          	mv	a0,a0
  80:	00000097          	auipc	ra,0x0
  84:	000080e7          	jalr	ra
  88:	47d2                	lw	a5,20(sp)
  8a:	0405                	addi	s0,s0,1
  8c:	94be                	add	s1,s1,a5
  8e:	b76d                	j	38 <.L42>

00000090 <.L47>:
  90:	000007b7          	lui	a5,0x0
  94:	00078513          	mv	a0,a5
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	8d4e                	mv	s10,s3
  a2:	4c81                	li	s9,0

000000a4 <.L44>:
  a4:	057c9963          	bne	s9,s7,f6 <.L49>
  a8:	000d8513          	mv	a0,s11
  ac:	00000097          	auipc	ra,0x0
  b0:	000080e7          	jalr	ra
  b4:	0905                	addi	s2,s2,1
  b6:	99d6                	add	s3,s3,s5
  b8:	bf75                	j	74 <.L43>

000000ba <.L45>:
  ba:	0046258b          	p.lw	a1,4(a2!)
  be:	47f2                	lw	a5,28(sp)
  c0:	c836                	sw	a3,16(sp)
  c2:	c632                	sw	a2,12(sp)
  c4:	95be                	add	a1,a1,a5
  c6:	000007b7          	lui	a5,0x0
  ca:	00078513          	mv	a0,a5
  ce:	00000097          	auipc	ra,0x0
  d2:	000080e7          	jalr	ra
  d6:	46c2                	lw	a3,16(sp)
  d8:	4632                	lw	a2,12(sp)
  da:	0685                	addi	a3,a3,1

000000dc <.L46>:
  dc:	fd869fe3          	bne	a3,s8,ba <.L45>
  e0:	000007b7          	lui	a5,0x0
  e4:	00078513          	mv	a0,a5
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	0c85                	addi	s9,s9,1
  f2:	9d52                	add	s10,s10,s4
  f4:	bf45                	j	a4 <.L44>

000000f6 <.L49>:
  f6:	866a                	mv	a2,s10
  f8:	4681                	li	a3,0
  fa:	b7cd                	j	dc <.L46>

Disassembly of section .text.PrintMatrix_s32_NHWC:

00000000 <PrintMatrix_s32_NHWC>:
   0:	711d                	addi	sp,sp,-96
   2:	cc3e                	sw	a5,24(sp)
   4:	02e687b3          	mul	a5,a3,a4
   8:	c4d2                	sw	s4,72(sp)
   a:	8a3a                	mv	s4,a4
   c:	cca2                	sw	s0,88(sp)
   e:	c8ca                	sw	s2,80(sp)
  10:	c6ce                	sw	s3,76(sp)
  12:	c2d6                	sw	s5,68(sp)
  14:	dc62                	sw	s8,56(sp)
  16:	da66                	sw	s9,52(sp)
  18:	ce86                	sw	ra,92(sp)
  1a:	02c787b3          	mul	a5,a5,a2
  1e:	caa6                	sw	s1,84(sp)
  20:	c0da                	sw	s6,64(sp)
  22:	de5e                	sw	s7,60(sp)
  24:	d86a                	sw	s10,48(sp)
  26:	d66e                	sw	s11,44(sp)
  28:	c82a                	sw	a0,16(sp)
  2a:	ca2e                	sw	a1,20(sp)
  2c:	8432                	mv	s0,a2
  2e:	8ab6                	mv	s5,a3
  30:	c63e                	sw	a5,12(sp)
  32:	00271793          	slli	a5,a4,0x2
  36:	02c787b3          	mul	a5,a5,a2
  3a:	00261713          	slli	a4,a2,0x2
  3e:	ce3a                	sw	a4,28(sp)
  40:	4981                	li	s3,0
  42:	4901                	li	s2,0
  44:	00000c37          	lui	s8,0x0
  48:	00000cb7          	lui	s9,0x0
  4c:	c43e                	sw	a5,8(sp)

0000004e <.L52>:
  4e:	47d2                	lw	a5,20(sp)
  50:	02f91163          	bne	s2,a5,72 <.L58>
  54:	40f6                	lw	ra,92(sp)
  56:	4466                	lw	s0,88(sp)
  58:	44d6                	lw	s1,84(sp)
  5a:	4946                	lw	s2,80(sp)
  5c:	49b6                	lw	s3,76(sp)
  5e:	4a26                	lw	s4,72(sp)
  60:	4a96                	lw	s5,68(sp)
  62:	4b06                	lw	s6,64(sp)
  64:	5bf2                	lw	s7,60(sp)
  66:	5c62                	lw	s8,56(sp)
  68:	5cd2                	lw	s9,52(sp)
  6a:	5d42                	lw	s10,48(sp)
  6c:	5db2                	lw	s11,44(sp)
  6e:	6125                	addi	sp,sp,96
  70:	8082                	ret

00000072 <.L58>:
  72:	000007b7          	lui	a5,0x0
  76:	00078513          	mv	a0,a5
  7a:	00000097          	auipc	ra,0x0
  7e:	000080e7          	jalr	ra
  82:	4481                	li	s1,0
  84:	00000d37          	lui	s10,0x0
  88:	00000db7          	lui	s11,0x0

0000008c <.L53>:
  8c:	00849e63          	bne	s1,s0,a8 <.L57>
  90:	00000537          	lui	a0,0x0
  94:	00050513          	mv	a0,a0
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	4732                	lw	a4,12(sp)
  a2:	0905                	addi	s2,s2,1
  a4:	99ba                	add	s3,s3,a4
  a6:	b765                	j	4e <.L52>

000000a8 <.L57>:
  a8:	000d0513          	mv	a0,s10
  ac:	00000097          	auipc	ra,0x0
  b0:	000080e7          	jalr	ra
  b4:	4742                	lw	a4,16(sp)
  b6:	013487b3          	add	a5,s1,s3
  ba:	078a                	slli	a5,a5,0x2
  bc:	00f70bb3          	add	s7,a4,a5
  c0:	4b01                	li	s6,0

000000c2 <.L54>:
  c2:	055b1663          	bne	s6,s5,10e <.L59>
  c6:	000d8513          	mv	a0,s11
  ca:	00000097          	auipc	ra,0x0
  ce:	000080e7          	jalr	ra
  d2:	0485                	addi	s1,s1,1
  d4:	bf65                	j	8c <.L53>

000000d6 <.L55>:
  d6:	47f2                	lw	a5,28(sp)
  d8:	4762                	lw	a4,24(sp)
  da:	000c0513          	mv	a0,s8
  de:	20f6758b          	p.lw	a1,a5(a2!)
  e2:	c036                	sw	a3,0(sp)
  e4:	95ba                	add	a1,a1,a4
  e6:	c232                	sw	a2,4(sp)
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	4682                	lw	a3,0(sp)
  f2:	4612                	lw	a2,4(sp)
  f4:	0685                	addi	a3,a3,1

000000f6 <.L56>:
  f6:	ff4690e3          	bne	a3,s4,d6 <.L55>
  fa:	000c8513          	mv	a0,s9
  fe:	00000097          	auipc	ra,0x0
 102:	000080e7          	jalr	ra
 106:	47a2                	lw	a5,8(sp)
 108:	0b05                	addi	s6,s6,1
 10a:	9bbe                	add	s7,s7,a5
 10c:	bf5d                	j	c2 <.L54>

0000010e <.L59>:
 10e:	865e                	mv	a2,s7
 110:	4681                	li	a3,0
 112:	b7d5                	j	f6 <.L56>

Disassembly of section .text.PrintArray_s8:

00000000 <PrintArray_s8>:
   0:	1101                	addi	sp,sp,-32
   2:	cc22                	sw	s0,24(sp)
   4:	ca26                	sw	s1,20(sp)
   6:	c84a                	sw	s2,16(sp)
   8:	ce06                	sw	ra,28(sp)
   a:	842a                	mv	s0,a0
   c:	00b504b3          	add	s1,a0,a1
  10:	00000937          	lui	s2,0x0

00000014 <.L62>:
  14:	00941f63          	bne	s0,s1,32 <.L63>
  18:	4462                	lw	s0,24(sp)
  1a:	40f2                	lw	ra,28(sp)
  1c:	44d2                	lw	s1,20(sp)
  1e:	4942                	lw	s2,16(sp)
  20:	00000537          	lui	a0,0x0
  24:	00050513          	mv	a0,a0
  28:	6105                	addi	sp,sp,32
  2a:	00000317          	auipc	t1,0x0
  2e:	00030067          	jr	t1

00000032 <.L63>:
  32:	0014058b          	p.lb	a1,1(s0!)
  36:	00090513          	mv	a0,s2
  3a:	c632                	sw	a2,12(sp)
  3c:	95b2                	add	a1,a1,a2
  3e:	1005e5b3          	p.extbs	a1,a1
  42:	00000097          	auipc	ra,0x0
  46:	000080e7          	jalr	ra
  4a:	4632                	lw	a2,12(sp)
  4c:	b7e1                	j	14 <.L62>

Disassembly of section .text.PrintArray_s16:

00000000 <PrintArray_s16>:
   0:	7179                	addi	sp,sp,-48
   2:	d422                	sw	s0,40(sp)
   4:	d226                	sw	s1,36(sp)
   6:	d04a                	sw	s2,32(sp)
   8:	ce4e                	sw	s3,28(sp)
   a:	d606                	sw	ra,44(sp)
   c:	892e                	mv	s2,a1
   e:	84aa                	mv	s1,a0
  10:	4401                	li	s0,0
  12:	000009b7          	lui	s3,0x0

00000016 <.L66>:
  16:	03241063          	bne	s0,s2,36 <.L67>
  1a:	5422                	lw	s0,40(sp)
  1c:	50b2                	lw	ra,44(sp)
  1e:	5492                	lw	s1,36(sp)
  20:	5902                	lw	s2,32(sp)
  22:	49f2                	lw	s3,28(sp)
  24:	00000537          	lui	a0,0x0
  28:	00050513          	mv	a0,a0
  2c:	6145                	addi	sp,sp,48
  2e:	00000317          	auipc	t1,0x0
  32:	00030067          	jr	t1

00000036 <.L67>:
  36:	0024958b          	p.lh	a1,2(s1!)
  3a:	00098513          	mv	a0,s3
  3e:	c632                	sw	a2,12(sp)
  40:	95b2                	add	a1,a1,a2
  42:	1005c5b3          	p.exths	a1,a1
  46:	00000097          	auipc	ra,0x0
  4a:	000080e7          	jalr	ra
  4e:	0405                	addi	s0,s0,1
  50:	4632                	lw	a2,12(sp)
  52:	b7d1                	j	16 <.L66>

Disassembly of section .text.PrintArray_s32:

00000000 <PrintArray_s32>:
   0:	7179                	addi	sp,sp,-48
   2:	d422                	sw	s0,40(sp)
   4:	d226                	sw	s1,36(sp)
   6:	d04a                	sw	s2,32(sp)
   8:	ce4e                	sw	s3,28(sp)
   a:	d606                	sw	ra,44(sp)
   c:	892e                	mv	s2,a1
   e:	84aa                	mv	s1,a0
  10:	4401                	li	s0,0
  12:	000009b7          	lui	s3,0x0

00000016 <.L70>:
  16:	03241063          	bne	s0,s2,36 <.L71>
  1a:	5422                	lw	s0,40(sp)
  1c:	50b2                	lw	ra,44(sp)
  1e:	5492                	lw	s1,36(sp)
  20:	5902                	lw	s2,32(sp)
  22:	49f2                	lw	s3,28(sp)
  24:	00000537          	lui	a0,0x0
  28:	00050513          	mv	a0,a0
  2c:	6145                	addi	sp,sp,48
  2e:	00000317          	auipc	t1,0x0
  32:	00030067          	jr	t1

00000036 <.L71>:
  36:	0044a58b          	p.lw	a1,4(s1!)
  3a:	00098513          	mv	a0,s3
  3e:	c632                	sw	a2,12(sp)
  40:	95b2                	add	a1,a1,a2
  42:	00000097          	auipc	ra,0x0
  46:	000080e7          	jalr	ra
  4a:	0405                	addi	s0,s0,1
  4c:	4632                	lw	a2,12(sp)
  4e:	b7e1                	j	16 <.L70>

Disassembly of section .text.PrintMatrix_u8_NCHW:

00000000 <PrintMatrix_u8_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	c0da                	sw	s6,64(sp)
   4:	02e68b33          	mul	s6,a3,a4
   8:	de5e                	sw	s7,60(sp)
   a:	cca2                	sw	s0,88(sp)
   c:	c8ca                	sw	s2,80(sp)
   e:	c6ce                	sw	s3,76(sp)
  10:	c4d2                	sw	s4,72(sp)
  12:	dc62                	sw	s8,56(sp)
  14:	da66                	sw	s9,52(sp)
  16:	ce86                	sw	ra,92(sp)
  18:	caa6                	sw	s1,84(sp)
  1a:	03660bb3          	mul	s7,a2,s6
  1e:	c2d6                	sw	s5,68(sp)
  20:	d86a                	sw	s10,48(sp)
  22:	d66e                	sw	s11,44(sp)
  24:	c82e                	sw	a1,16(sp)
  26:	8c32                	mv	s8,a2
  28:	8cb6                	mv	s9,a3
  2a:	8a3a                	mv	s4,a4
  2c:	ca3e                	sw	a5,20(sp)
  2e:	842a                	mv	s0,a0
  30:	00e509b3          	add	s3,a0,a4
  34:	4901                	li	s2,0

00000036 <.L74>:
  36:	47c2                	lw	a5,16(sp)
  38:	02f91163          	bne	s2,a5,5a <.L81>
  3c:	40f6                	lw	ra,92(sp)
  3e:	4466                	lw	s0,88(sp)
  40:	44d6                	lw	s1,84(sp)
  42:	4946                	lw	s2,80(sp)
  44:	49b6                	lw	s3,76(sp)
  46:	4a26                	lw	s4,72(sp)
  48:	4a96                	lw	s5,68(sp)
  4a:	4b06                	lw	s6,64(sp)
  4c:	5bf2                	lw	s7,60(sp)
  4e:	5c62                	lw	s8,56(sp)
  50:	5cd2                	lw	s9,52(sp)
  52:	5d42                	lw	s10,48(sp)
  54:	5db2                	lw	s11,44(sp)
  56:	6125                	addi	sp,sp,96
  58:	8082                	ret

0000005a <.L81>:
  5a:	000007b7          	lui	a5,0x0
  5e:	00078513          	mv	a0,a5
  62:	00000097          	auipc	ra,0x0
  66:	000080e7          	jalr	ra
  6a:	4481                	li	s1,0
  6c:	4a81                	li	s5,0

0000006e <.L75>:
  6e:	018a9e63          	bne	s5,s8,8a <.L80>
  72:	00000537          	lui	a0,0x0
  76:	00050513          	mv	a0,a0
  7a:	00000097          	auipc	ra,0x0
  7e:	000080e7          	jalr	ra
  82:	0905                	addi	s2,s2,1
  84:	945e                	add	s0,s0,s7
  86:	99de                	add	s3,s3,s7
  88:	b77d                	j	36 <.L74>

0000008a <.L80>:
  8a:	000007b7          	lui	a5,0x0
  8e:	00078513          	mv	a0,a5
  92:	00000097          	auipc	ra,0x0
  96:	000080e7          	jalr	ra
  9a:	008487b3          	add	a5,s1,s0
  9e:	cc3e                	sw	a5,24(sp)
  a0:	013487b3          	add	a5,s1,s3
  a4:	4d01                	li	s10,0
  a6:	4d81                	li	s11,0
  a8:	ce3e                	sw	a5,28(sp)

000000aa <.L76>:
  aa:	059d8863          	beq	s11,s9,fa <.L78>
  ae:	47e2                	lw	a5,24(sp)
  b0:	01a78733          	add	a4,a5,s10
  b4:	47f2                	lw	a5,28(sp)
  b6:	01a78633          	add	a2,a5,s10
  ba:	a01d                	j	e0 <.L79>

000000bc <.L77>:
  bc:	0017458b          	p.lbu	a1,1(a4!) # 1 <PrintMatrix_u8_NCHW+0x1>
  c0:	47d2                	lw	a5,20(sp)
  c2:	c432                	sw	a2,8(sp)
  c4:	c63a                	sw	a4,12(sp)
  c6:	95be                	add	a1,a1,a5
  c8:	000007b7          	lui	a5,0x0
  cc:	0ff5f593          	andi	a1,a1,255
  d0:	00078513          	mv	a0,a5
  d4:	00000097          	auipc	ra,0x0
  d8:	000080e7          	jalr	ra
  dc:	4732                	lw	a4,12(sp)
  de:	4622                	lw	a2,8(sp)

000000e0 <.L79>:
  e0:	fcc71ee3          	bne	a4,a2,bc <.L77>
  e4:	000007b7          	lui	a5,0x0
  e8:	00078513          	mv	a0,a5
  ec:	00000097          	auipc	ra,0x0
  f0:	000080e7          	jalr	ra
  f4:	0d85                	addi	s11,s11,1
  f6:	9d52                	add	s10,s10,s4
  f8:	bf4d                	j	aa <.L76>

000000fa <.L78>:
  fa:	000007b7          	lui	a5,0x0
  fe:	00078513          	mv	a0,a5
 102:	00000097          	auipc	ra,0x0
 106:	000080e7          	jalr	ra
 10a:	0a85                	addi	s5,s5,1
 10c:	94da                	add	s1,s1,s6
 10e:	b785                	j	6e <.L75>

Disassembly of section .text.PrintMatrix_u8_NHWC:

00000000 <PrintMatrix_u8_NHWC>:
   0:	711d                	addi	sp,sp,-96
   2:	ce3e                	sw	a5,28(sp)
   4:	02e687b3          	mul	a5,a3,a4
   8:	caa6                	sw	s1,84(sp)
   a:	c8ca                	sw	s2,80(sp)
   c:	c6ce                	sw	s3,76(sp)
   e:	c4d2                	sw	s4,72(sp)
  10:	c2d6                	sw	s5,68(sp)
  12:	dc62                	sw	s8,56(sp)
  14:	da66                	sw	s9,52(sp)
  16:	ce86                	sw	ra,92(sp)
  18:	cca2                	sw	s0,88(sp)
  1a:	02c787b3          	mul	a5,a5,a2
  1e:	c0da                	sw	s6,64(sp)
  20:	de5e                	sw	s7,60(sp)
  22:	d86a                	sw	s10,48(sp)
  24:	d66e                	sw	s11,44(sp)
  26:	cc2e                	sw	a1,24(sp)
  28:	89b2                	mv	s3,a2
  2a:	8ab6                	mv	s5,a3
  2c:	8a3a                	mv	s4,a4
  2e:	892a                	mv	s2,a0
  30:	ca3e                	sw	a5,20(sp)
  32:	02e607b3          	mul	a5,a2,a4
  36:	4481                	li	s1,0
  38:	00000c37          	lui	s8,0x0
  3c:	00000cb7          	lui	s9,0x0
  40:	c83e                	sw	a5,16(sp)

00000042 <.L84>:
  42:	47e2                	lw	a5,24(sp)
  44:	02f49163          	bne	s1,a5,66 <.L90>
  48:	40f6                	lw	ra,92(sp)
  4a:	4466                	lw	s0,88(sp)
  4c:	44d6                	lw	s1,84(sp)
  4e:	4946                	lw	s2,80(sp)
  50:	49b6                	lw	s3,76(sp)
  52:	4a26                	lw	s4,72(sp)
  54:	4a96                	lw	s5,68(sp)
  56:	4b06                	lw	s6,64(sp)
  58:	5bf2                	lw	s7,60(sp)
  5a:	5c62                	lw	s8,56(sp)
  5c:	5cd2                	lw	s9,52(sp)
  5e:	5d42                	lw	s10,48(sp)
  60:	5db2                	lw	s11,44(sp)
  62:	6125                	addi	sp,sp,96
  64:	8082                	ret

00000066 <.L90>:
  66:	000007b7          	lui	a5,0x0
  6a:	00078513          	mv	a0,a5
  6e:	00000097          	auipc	ra,0x0
  72:	000080e7          	jalr	ra
  76:	4401                	li	s0,0
  78:	00000d37          	lui	s10,0x0
  7c:	00000db7          	lui	s11,0x0

00000080 <.L85>:
  80:	01341e63          	bne	s0,s3,9c <.L89>
  84:	00000537          	lui	a0,0x0
  88:	00050513          	mv	a0,a0
  8c:	00000097          	auipc	ra,0x0
  90:	000080e7          	jalr	ra
  94:	47d2                	lw	a5,20(sp)
  96:	0485                	addi	s1,s1,1
  98:	993e                	add	s2,s2,a5
  9a:	b765                	j	42 <.L84>

0000009c <.L89>:
  9c:	000d0513          	mv	a0,s10
  a0:	00000097          	auipc	ra,0x0
  a4:	000080e7          	jalr	ra
  a8:	01240bb3          	add	s7,s0,s2
  ac:	4b01                	li	s6,0

000000ae <.L86>:
  ae:	055b1763          	bne	s6,s5,fc <.L91>
  b2:	000d8513          	mv	a0,s11
  b6:	00000097          	auipc	ra,0x0
  ba:	000080e7          	jalr	ra
  be:	0405                	addi	s0,s0,1
  c0:	b7c1                	j	80 <.L85>

000000c2 <.L87>:
  c2:	4136758b          	p.lbu	a1,s3(a2!)
  c6:	47f2                	lw	a5,28(sp)
  c8:	000c0513          	mv	a0,s8
  cc:	c632                	sw	a2,12(sp)
  ce:	95be                	add	a1,a1,a5
  d0:	0ff5f593          	andi	a1,a1,255
  d4:	c436                	sw	a3,8(sp)
  d6:	00000097          	auipc	ra,0x0
  da:	000080e7          	jalr	ra
  de:	46a2                	lw	a3,8(sp)
  e0:	4632                	lw	a2,12(sp)
  e2:	0685                	addi	a3,a3,1

000000e4 <.L88>:
  e4:	fd469fe3          	bne	a3,s4,c2 <.L87>
  e8:	000c8513          	mv	a0,s9
  ec:	00000097          	auipc	ra,0x0
  f0:	000080e7          	jalr	ra
  f4:	47c2                	lw	a5,16(sp)
  f6:	0b05                	addi	s6,s6,1
  f8:	9bbe                	add	s7,s7,a5
  fa:	bf55                	j	ae <.L86>

000000fc <.L91>:
  fc:	865e                	mv	a2,s7
  fe:	4681                	li	a3,0
 100:	b7d5                	j	e4 <.L88>

Disassembly of section .text.PrintMatrix_u16_NCHW:

00000000 <PrintMatrix_u16_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	c4d2                	sw	s4,72(sp)
   4:	00171a13          	slli	s4,a4,0x1
   8:	c2d6                	sw	s5,68(sp)
   a:	03468ab3          	mul	s5,a3,s4
   e:	ce3e                	sw	a5,28(sp)
  10:	cca2                	sw	s0,88(sp)
  12:	caa6                	sw	s1,84(sp)
  14:	c0da                	sw	s6,64(sp)
  16:	de5e                	sw	s7,60(sp)
  18:	dc62                	sw	s8,56(sp)
  1a:	ce86                	sw	ra,92(sp)
  1c:	c8ca                	sw	s2,80(sp)
  1e:	c6ce                	sw	s3,76(sp)
  20:	035607b3          	mul	a5,a2,s5
  24:	da66                	sw	s9,52(sp)
  26:	d86a                	sw	s10,48(sp)
  28:	d66e                	sw	s11,44(sp)
  2a:	cc2e                	sw	a1,24(sp)
  2c:	8b32                	mv	s6,a2
  2e:	8bb6                	mv	s7,a3
  30:	8c3a                	mv	s8,a4
  32:	84aa                	mv	s1,a0
  34:	4401                	li	s0,0
  36:	ca3e                	sw	a5,20(sp)

00000038 <.L94>:
  38:	47e2                	lw	a5,24(sp)
  3a:	02f41163          	bne	s0,a5,5c <.L100>
  3e:	40f6                	lw	ra,92(sp)
  40:	4466                	lw	s0,88(sp)
  42:	44d6                	lw	s1,84(sp)
  44:	4946                	lw	s2,80(sp)
  46:	49b6                	lw	s3,76(sp)
  48:	4a26                	lw	s4,72(sp)
  4a:	4a96                	lw	s5,68(sp)
  4c:	4b06                	lw	s6,64(sp)
  4e:	5bf2                	lw	s7,60(sp)
  50:	5c62                	lw	s8,56(sp)
  52:	5cd2                	lw	s9,52(sp)
  54:	5d42                	lw	s10,48(sp)
  56:	5db2                	lw	s11,44(sp)
  58:	6125                	addi	sp,sp,96
  5a:	8082                	ret

0000005c <.L100>:
  5c:	000007b7          	lui	a5,0x0
  60:	00078513          	mv	a0,a5
  64:	00000097          	auipc	ra,0x0
  68:	000080e7          	jalr	ra
  6c:	89a6                	mv	s3,s1
  6e:	4901                	li	s2,0
  70:	00000db7          	lui	s11,0x0

00000074 <.L95>:
  74:	01691e63          	bne	s2,s6,90 <.L99>
  78:	00000537          	lui	a0,0x0
  7c:	00050513          	mv	a0,a0
  80:	00000097          	auipc	ra,0x0
  84:	000080e7          	jalr	ra
  88:	47d2                	lw	a5,20(sp)
  8a:	0405                	addi	s0,s0,1
  8c:	94be                	add	s1,s1,a5
  8e:	b76d                	j	38 <.L94>

00000090 <.L99>:
  90:	000007b7          	lui	a5,0x0
  94:	00078513          	mv	a0,a5
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	8d4e                	mv	s10,s3
  a2:	4c81                	li	s9,0

000000a4 <.L96>:
  a4:	057c9b63          	bne	s9,s7,fa <.L101>
  a8:	000d8513          	mv	a0,s11
  ac:	00000097          	auipc	ra,0x0
  b0:	000080e7          	jalr	ra
  b4:	0905                	addi	s2,s2,1
  b6:	99d6                	add	s3,s3,s5
  b8:	bf75                	j	74 <.L95>

000000ba <.L97>:
  ba:	0026558b          	p.lhu	a1,2(a2!)
  be:	47f2                	lw	a5,28(sp)
  c0:	c836                	sw	a3,16(sp)
  c2:	c632                	sw	a2,12(sp)
  c4:	95be                	add	a1,a1,a5
  c6:	000007b7          	lui	a5,0x0
  ca:	1005d5b3          	p.exthz	a1,a1
  ce:	00078513          	mv	a0,a5
  d2:	00000097          	auipc	ra,0x0
  d6:	000080e7          	jalr	ra
  da:	46c2                	lw	a3,16(sp)
  dc:	4632                	lw	a2,12(sp)
  de:	0685                	addi	a3,a3,1

000000e0 <.L98>:
  e0:	fd869de3          	bne	a3,s8,ba <.L97>
  e4:	000007b7          	lui	a5,0x0
  e8:	00078513          	mv	a0,a5
  ec:	00000097          	auipc	ra,0x0
  f0:	000080e7          	jalr	ra
  f4:	0c85                	addi	s9,s9,1
  f6:	9d52                	add	s10,s10,s4
  f8:	b775                	j	a4 <.L96>

000000fa <.L101>:
  fa:	866a                	mv	a2,s10
  fc:	4681                	li	a3,0
  fe:	b7cd                	j	e0 <.L98>

Disassembly of section .text.PrintMatrix_u16_NHWC:

00000000 <PrintMatrix_u16_NHWC>:
   0:	711d                	addi	sp,sp,-96
   2:	ce3e                	sw	a5,28(sp)
   4:	02e687b3          	mul	a5,a3,a4
   8:	cca2                	sw	s0,88(sp)
   a:	c8ca                	sw	s2,80(sp)
   c:	c6ce                	sw	s3,76(sp)
   e:	c4d2                	sw	s4,72(sp)
  10:	c2d6                	sw	s5,68(sp)
  12:	dc62                	sw	s8,56(sp)
  14:	d66e                	sw	s11,44(sp)
  16:	ce86                	sw	ra,92(sp)
  18:	caa6                	sw	s1,84(sp)
  1a:	02c787b3          	mul	a5,a5,a2
  1e:	c0da                	sw	s6,64(sp)
  20:	de5e                	sw	s7,60(sp)
  22:	da66                	sw	s9,52(sp)
  24:	d86a                	sw	s10,48(sp)
  26:	ca2a                	sw	a0,20(sp)
  28:	cc2e                	sw	a1,24(sp)
  2a:	8432                	mv	s0,a2
  2c:	8ab6                	mv	s5,a3
  2e:	8a3a                	mv	s4,a4
  30:	c83e                	sw	a5,16(sp)
  32:	00171793          	slli	a5,a4,0x1
  36:	02c787b3          	mul	a5,a5,a2
  3a:	00161d93          	slli	s11,a2,0x1
  3e:	4981                	li	s3,0
  40:	4901                	li	s2,0
  42:	00000c37          	lui	s8,0x0
  46:	c63e                	sw	a5,12(sp)

00000048 <.L104>:
  48:	47e2                	lw	a5,24(sp)
  4a:	02f91163          	bne	s2,a5,6c <.L110>
  4e:	40f6                	lw	ra,92(sp)
  50:	4466                	lw	s0,88(sp)
  52:	44d6                	lw	s1,84(sp)
  54:	4946                	lw	s2,80(sp)
  56:	49b6                	lw	s3,76(sp)
  58:	4a26                	lw	s4,72(sp)
  5a:	4a96                	lw	s5,68(sp)
  5c:	4b06                	lw	s6,64(sp)
  5e:	5bf2                	lw	s7,60(sp)
  60:	5c62                	lw	s8,56(sp)
  62:	5cd2                	lw	s9,52(sp)
  64:	5d42                	lw	s10,48(sp)
  66:	5db2                	lw	s11,44(sp)
  68:	6125                	addi	sp,sp,96
  6a:	8082                	ret

0000006c <.L110>:
  6c:	00000737          	lui	a4,0x0
  70:	00070513          	mv	a0,a4
  74:	00000097          	auipc	ra,0x0
  78:	000080e7          	jalr	ra
  7c:	4481                	li	s1,0
  7e:	00000cb7          	lui	s9,0x0
  82:	00000d37          	lui	s10,0x0

00000086 <.L105>:
  86:	00849e63          	bne	s1,s0,a2 <.L109>
  8a:	00000537          	lui	a0,0x0
  8e:	00050513          	mv	a0,a0
  92:	00000097          	auipc	ra,0x0
  96:	000080e7          	jalr	ra
  9a:	4742                	lw	a4,16(sp)
  9c:	0905                	addi	s2,s2,1
  9e:	99ba                	add	s3,s3,a4
  a0:	b765                	j	48 <.L104>

000000a2 <.L109>:
  a2:	000c8513          	mv	a0,s9
  a6:	00000097          	auipc	ra,0x0
  aa:	000080e7          	jalr	ra
  ae:	4752                	lw	a4,20(sp)
  b0:	013487b3          	add	a5,s1,s3
  b4:	0786                	slli	a5,a5,0x1
  b6:	00f70bb3          	add	s7,a4,a5
  ba:	4b01                	li	s6,0

000000bc <.L106>:
  bc:	055b1963          	bne	s6,s5,10e <.L111>
  c0:	000d0513          	mv	a0,s10
  c4:	00000097          	auipc	ra,0x0
  c8:	000080e7          	jalr	ra
  cc:	0485                	addi	s1,s1,1
  ce:	bf65                	j	86 <.L105>

000000d0 <.L107>:
  d0:	51b6758b          	p.lhu	a1,s11(a2!)
  d4:	47f2                	lw	a5,28(sp)
  d6:	00000737          	lui	a4,0x0
  da:	00070513          	mv	a0,a4
  de:	95be                	add	a1,a1,a5
  e0:	1005d5b3          	p.exthz	a1,a1
  e4:	c432                	sw	a2,8(sp)
  e6:	c236                	sw	a3,4(sp)
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	4692                	lw	a3,4(sp)
  f2:	4622                	lw	a2,8(sp)
  f4:	0685                	addi	a3,a3,1

000000f6 <.L108>:
  f6:	fd469de3          	bne	a3,s4,d0 <.L107>
  fa:	000c0513          	mv	a0,s8
  fe:	00000097          	auipc	ra,0x0
 102:	000080e7          	jalr	ra
 106:	47b2                	lw	a5,12(sp)
 108:	0b05                	addi	s6,s6,1
 10a:	9bbe                	add	s7,s7,a5
 10c:	bf45                	j	bc <.L106>

0000010e <.L111>:
 10e:	865e                	mv	a2,s7
 110:	4681                	li	a3,0
 112:	b7d5                	j	f6 <.L108>

Disassembly of section .text.PrintMatrix_u32_NCHW:

00000000 <PrintMatrix_u32_NCHW>:
   0:	711d                	addi	sp,sp,-96
   2:	c4d2                	sw	s4,72(sp)
   4:	00271a13          	slli	s4,a4,0x2
   8:	c2d6                	sw	s5,68(sp)
   a:	03468ab3          	mul	s5,a3,s4
   e:	ce3e                	sw	a5,28(sp)
  10:	cca2                	sw	s0,88(sp)
  12:	caa6                	sw	s1,84(sp)
  14:	c0da                	sw	s6,64(sp)
  16:	de5e                	sw	s7,60(sp)
  18:	dc62                	sw	s8,56(sp)
  1a:	ce86                	sw	ra,92(sp)
  1c:	c8ca                	sw	s2,80(sp)
  1e:	c6ce                	sw	s3,76(sp)
  20:	035607b3          	mul	a5,a2,s5
  24:	da66                	sw	s9,52(sp)
  26:	d86a                	sw	s10,48(sp)
  28:	d66e                	sw	s11,44(sp)
  2a:	cc2e                	sw	a1,24(sp)
  2c:	8b32                	mv	s6,a2
  2e:	8bb6                	mv	s7,a3
  30:	8c3a                	mv	s8,a4
  32:	84aa                	mv	s1,a0
  34:	4401                	li	s0,0
  36:	ca3e                	sw	a5,20(sp)

00000038 <.L114>:
  38:	47e2                	lw	a5,24(sp)
  3a:	02f41163          	bne	s0,a5,5c <.L120>
  3e:	40f6                	lw	ra,92(sp)
  40:	4466                	lw	s0,88(sp)
  42:	44d6                	lw	s1,84(sp)
  44:	4946                	lw	s2,80(sp)
  46:	49b6                	lw	s3,76(sp)
  48:	4a26                	lw	s4,72(sp)
  4a:	4a96                	lw	s5,68(sp)
  4c:	4b06                	lw	s6,64(sp)
  4e:	5bf2                	lw	s7,60(sp)
  50:	5c62                	lw	s8,56(sp)
  52:	5cd2                	lw	s9,52(sp)
  54:	5d42                	lw	s10,48(sp)
  56:	5db2                	lw	s11,44(sp)
  58:	6125                	addi	sp,sp,96
  5a:	8082                	ret

0000005c <.L120>:
  5c:	000007b7          	lui	a5,0x0
  60:	00078513          	mv	a0,a5
  64:	00000097          	auipc	ra,0x0
  68:	000080e7          	jalr	ra
  6c:	89a6                	mv	s3,s1
  6e:	4901                	li	s2,0
  70:	00000db7          	lui	s11,0x0

00000074 <.L115>:
  74:	01691e63          	bne	s2,s6,90 <.L119>
  78:	00000537          	lui	a0,0x0
  7c:	00050513          	mv	a0,a0
  80:	00000097          	auipc	ra,0x0
  84:	000080e7          	jalr	ra
  88:	47d2                	lw	a5,20(sp)
  8a:	0405                	addi	s0,s0,1
  8c:	94be                	add	s1,s1,a5
  8e:	b76d                	j	38 <.L114>

00000090 <.L119>:
  90:	000007b7          	lui	a5,0x0
  94:	00078513          	mv	a0,a5
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	8d4e                	mv	s10,s3
  a2:	4c81                	li	s9,0

000000a4 <.L116>:
  a4:	057c9963          	bne	s9,s7,f6 <.L121>
  a8:	000d8513          	mv	a0,s11
  ac:	00000097          	auipc	ra,0x0
  b0:	000080e7          	jalr	ra
  b4:	0905                	addi	s2,s2,1
  b6:	99d6                	add	s3,s3,s5
  b8:	bf75                	j	74 <.L115>

000000ba <.L117>:
  ba:	0046258b          	p.lw	a1,4(a2!)
  be:	47f2                	lw	a5,28(sp)
  c0:	c836                	sw	a3,16(sp)
  c2:	c632                	sw	a2,12(sp)
  c4:	95be                	add	a1,a1,a5
  c6:	000007b7          	lui	a5,0x0
  ca:	00078513          	mv	a0,a5
  ce:	00000097          	auipc	ra,0x0
  d2:	000080e7          	jalr	ra
  d6:	46c2                	lw	a3,16(sp)
  d8:	4632                	lw	a2,12(sp)
  da:	0685                	addi	a3,a3,1

000000dc <.L118>:
  dc:	fd869fe3          	bne	a3,s8,ba <.L117>
  e0:	000007b7          	lui	a5,0x0
  e4:	00078513          	mv	a0,a5
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	0c85                	addi	s9,s9,1
  f2:	9d52                	add	s10,s10,s4
  f4:	bf45                	j	a4 <.L116>

000000f6 <.L121>:
  f6:	866a                	mv	a2,s10
  f8:	4681                	li	a3,0
  fa:	b7cd                	j	dc <.L118>

Disassembly of section .text.PrintMatrix_u32_NHWC:

00000000 <PrintMatrix_u32_NHWC>:
   0:	711d                	addi	sp,sp,-96
   2:	cc3e                	sw	a5,24(sp)
   4:	02e687b3          	mul	a5,a3,a4
   8:	c4d2                	sw	s4,72(sp)
   a:	8a3a                	mv	s4,a4
   c:	cca2                	sw	s0,88(sp)
   e:	c8ca                	sw	s2,80(sp)
  10:	c6ce                	sw	s3,76(sp)
  12:	c2d6                	sw	s5,68(sp)
  14:	dc62                	sw	s8,56(sp)
  16:	da66                	sw	s9,52(sp)
  18:	ce86                	sw	ra,92(sp)
  1a:	02c787b3          	mul	a5,a5,a2
  1e:	caa6                	sw	s1,84(sp)
  20:	c0da                	sw	s6,64(sp)
  22:	de5e                	sw	s7,60(sp)
  24:	d86a                	sw	s10,48(sp)
  26:	d66e                	sw	s11,44(sp)
  28:	c82a                	sw	a0,16(sp)
  2a:	ca2e                	sw	a1,20(sp)
  2c:	8432                	mv	s0,a2
  2e:	8ab6                	mv	s5,a3
  30:	c63e                	sw	a5,12(sp)
  32:	00271793          	slli	a5,a4,0x2
  36:	02c787b3          	mul	a5,a5,a2
  3a:	00261713          	slli	a4,a2,0x2
  3e:	ce3a                	sw	a4,28(sp)
  40:	4981                	li	s3,0
  42:	4901                	li	s2,0
  44:	00000c37          	lui	s8,0x0
  48:	00000cb7          	lui	s9,0x0
  4c:	c43e                	sw	a5,8(sp)

0000004e <.L124>:
  4e:	47d2                	lw	a5,20(sp)
  50:	02f91163          	bne	s2,a5,72 <.L130>
  54:	40f6                	lw	ra,92(sp)
  56:	4466                	lw	s0,88(sp)
  58:	44d6                	lw	s1,84(sp)
  5a:	4946                	lw	s2,80(sp)
  5c:	49b6                	lw	s3,76(sp)
  5e:	4a26                	lw	s4,72(sp)
  60:	4a96                	lw	s5,68(sp)
  62:	4b06                	lw	s6,64(sp)
  64:	5bf2                	lw	s7,60(sp)
  66:	5c62                	lw	s8,56(sp)
  68:	5cd2                	lw	s9,52(sp)
  6a:	5d42                	lw	s10,48(sp)
  6c:	5db2                	lw	s11,44(sp)
  6e:	6125                	addi	sp,sp,96
  70:	8082                	ret

00000072 <.L130>:
  72:	000007b7          	lui	a5,0x0
  76:	00078513          	mv	a0,a5
  7a:	00000097          	auipc	ra,0x0
  7e:	000080e7          	jalr	ra
  82:	4481                	li	s1,0
  84:	00000d37          	lui	s10,0x0
  88:	00000db7          	lui	s11,0x0

0000008c <.L125>:
  8c:	00849e63          	bne	s1,s0,a8 <.L129>
  90:	00000537          	lui	a0,0x0
  94:	00050513          	mv	a0,a0
  98:	00000097          	auipc	ra,0x0
  9c:	000080e7          	jalr	ra
  a0:	4732                	lw	a4,12(sp)
  a2:	0905                	addi	s2,s2,1
  a4:	99ba                	add	s3,s3,a4
  a6:	b765                	j	4e <.L124>

000000a8 <.L129>:
  a8:	000d0513          	mv	a0,s10
  ac:	00000097          	auipc	ra,0x0
  b0:	000080e7          	jalr	ra
  b4:	4742                	lw	a4,16(sp)
  b6:	013487b3          	add	a5,s1,s3
  ba:	078a                	slli	a5,a5,0x2
  bc:	00f70bb3          	add	s7,a4,a5
  c0:	4b01                	li	s6,0

000000c2 <.L126>:
  c2:	055b1663          	bne	s6,s5,10e <.L131>
  c6:	000d8513          	mv	a0,s11
  ca:	00000097          	auipc	ra,0x0
  ce:	000080e7          	jalr	ra
  d2:	0485                	addi	s1,s1,1
  d4:	bf65                	j	8c <.L125>

000000d6 <.L127>:
  d6:	47f2                	lw	a5,28(sp)
  d8:	4762                	lw	a4,24(sp)
  da:	000c0513          	mv	a0,s8
  de:	20f6758b          	p.lw	a1,a5(a2!)
  e2:	c036                	sw	a3,0(sp)
  e4:	95ba                	add	a1,a1,a4
  e6:	c232                	sw	a2,4(sp)
  e8:	00000097          	auipc	ra,0x0
  ec:	000080e7          	jalr	ra
  f0:	4682                	lw	a3,0(sp)
  f2:	4612                	lw	a2,4(sp)
  f4:	0685                	addi	a3,a3,1

000000f6 <.L128>:
  f6:	ff4690e3          	bne	a3,s4,d6 <.L127>
  fa:	000c8513          	mv	a0,s9
  fe:	00000097          	auipc	ra,0x0
 102:	000080e7          	jalr	ra
 106:	47a2                	lw	a5,8(sp)
 108:	0b05                	addi	s6,s6,1
 10a:	9bbe                	add	s7,s7,a5
 10c:	bf5d                	j	c2 <.L126>

0000010e <.L131>:
 10e:	865e                	mv	a2,s7
 110:	4681                	li	a3,0
 112:	b7d5                	j	f6 <.L128>

Disassembly of section .text.PrintArray_u8:

00000000 <PrintArray_u8>:
   0:	1101                	addi	sp,sp,-32
   2:	cc22                	sw	s0,24(sp)
   4:	ca26                	sw	s1,20(sp)
   6:	c84a                	sw	s2,16(sp)
   8:	ce06                	sw	ra,28(sp)
   a:	842a                	mv	s0,a0
   c:	00b504b3          	add	s1,a0,a1
  10:	00000937          	lui	s2,0x0

00000014 <.L134>:
  14:	00941f63          	bne	s0,s1,32 <.L135>
  18:	4462                	lw	s0,24(sp)
  1a:	40f2                	lw	ra,28(sp)
  1c:	44d2                	lw	s1,20(sp)
  1e:	4942                	lw	s2,16(sp)
  20:	00000537          	lui	a0,0x0
  24:	00050513          	mv	a0,a0
  28:	6105                	addi	sp,sp,32
  2a:	00000317          	auipc	t1,0x0
  2e:	00030067          	jr	t1

00000032 <.L135>:
  32:	0014458b          	p.lbu	a1,1(s0!)
  36:	00090513          	mv	a0,s2
  3a:	c632                	sw	a2,12(sp)
  3c:	95b2                	add	a1,a1,a2
  3e:	0ff5f593          	andi	a1,a1,255
  42:	00000097          	auipc	ra,0x0
  46:	000080e7          	jalr	ra
  4a:	4632                	lw	a2,12(sp)
  4c:	b7e1                	j	14 <.L134>

Disassembly of section .text.PrintArray_u16:

00000000 <PrintArray_u16>:
   0:	7179                	addi	sp,sp,-48
   2:	d422                	sw	s0,40(sp)
   4:	d226                	sw	s1,36(sp)
   6:	d04a                	sw	s2,32(sp)
   8:	ce4e                	sw	s3,28(sp)
   a:	d606                	sw	ra,44(sp)
   c:	892e                	mv	s2,a1
   e:	84aa                	mv	s1,a0
  10:	4401                	li	s0,0
  12:	000009b7          	lui	s3,0x0

00000016 <.L138>:
  16:	03241063          	bne	s0,s2,36 <.L139>
  1a:	5422                	lw	s0,40(sp)
  1c:	50b2                	lw	ra,44(sp)
  1e:	5492                	lw	s1,36(sp)
  20:	5902                	lw	s2,32(sp)
  22:	49f2                	lw	s3,28(sp)
  24:	00000537          	lui	a0,0x0
  28:	00050513          	mv	a0,a0
  2c:	6145                	addi	sp,sp,48
  2e:	00000317          	auipc	t1,0x0
  32:	00030067          	jr	t1

00000036 <.L139>:
  36:	0024d58b          	p.lhu	a1,2(s1!)
  3a:	00098513          	mv	a0,s3
  3e:	c632                	sw	a2,12(sp)
  40:	95b2                	add	a1,a1,a2
  42:	1005d5b3          	p.exthz	a1,a1
  46:	00000097          	auipc	ra,0x0
  4a:	000080e7          	jalr	ra
  4e:	0405                	addi	s0,s0,1
  50:	4632                	lw	a2,12(sp)
  52:	b7d1                	j	16 <.L138>

Disassembly of section .text.PrintArray_u32:

00000000 <PrintArray_u32>:
   0:	7179                	addi	sp,sp,-48
   2:	d422                	sw	s0,40(sp)
   4:	d226                	sw	s1,36(sp)
   6:	d04a                	sw	s2,32(sp)
   8:	ce4e                	sw	s3,28(sp)
   a:	d606                	sw	ra,44(sp)
   c:	892e                	mv	s2,a1
   e:	84aa                	mv	s1,a0
  10:	4401                	li	s0,0
  12:	000009b7          	lui	s3,0x0

00000016 <.L142>:
  16:	03241063          	bne	s0,s2,36 <.L143>
  1a:	5422                	lw	s0,40(sp)
  1c:	50b2                	lw	ra,44(sp)
  1e:	5492                	lw	s1,36(sp)
  20:	5902                	lw	s2,32(sp)
  22:	49f2                	lw	s3,28(sp)
  24:	00000537          	lui	a0,0x0
  28:	00050513          	mv	a0,a0
  2c:	6145                	addi	sp,sp,48
  2e:	00000317          	auipc	t1,0x0
  32:	00030067          	jr	t1

00000036 <.L143>:
  36:	0044a58b          	p.lw	a1,4(s1!)
  3a:	00098513          	mv	a0,s3
  3e:	c632                	sw	a2,12(sp)
  40:	95b2                	add	a1,a1,a2
  42:	00000097          	auipc	ra,0x0
  46:	000080e7          	jalr	ra
  4a:	0405                	addi	s0,s0,1
  4c:	4632                	lw	a2,12(sp)
  4e:	b7e1                	j	16 <.L142>

iRMSNorm_s8.c.obj:     file format elf32-littleriscv

Sections:
Idx Name          Size      VMA       LMA       File off  Algn
  0 .text         00000000  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 .data         00000000  00000000  00000000  00000034  2**0
                  CONTENTS, ALLOC, LOAD, DATA
  2 .bss          00000000  00000000  00000000  00000034  2**0
                  ALLOC
  3 .text._plp_sqrt_q32 00000044  00000000  00000000  00000034  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  4 .text.iRMSnorm_s8_s8 000000f2  00000000  00000000  00000078  2**1
                  CONTENTS, ALLOC, LOAD, RELOC, READONLY, CODE
  5 .comment      0000001b  00000000  00000000  0000016a  2**0
                  CONTENTS, READONLY
  6 .Pulp_Chip.Info 00000072  00000000  00000000  00000185  2**0
                  CONTENTS, READONLY

Disassembly of section .text._plp_sqrt_q32:

00000000 <_plp_sqrt_q32>:
   0:	00052883          	lw	a7,0(a0)
   4:	03105d63          	blez	a7,3e <.L2>
   8:	672d                	lui	a4,0xb
   a:	50670713          	addi	a4,a4,1286 # b506 <.L16+0xb41a>
   e:	4801                	li	a6,0
  10:	4501                	li	a0,0

00000012 <.L6>:
  12:	02e827db          	p.addn	a5,a6,a4,1
  16:	02f786b3          	mul	a3,a5,a5
  1a:	40b6d6b3          	sra	a3,a3,a1
  1e:	01168e63          	beq	a3,a7,3a <.L8>
  22:	0116d963          	ble	a7,a3,34 <.L4>
  26:	00178813          	addi	a6,a5,1
  2a:	853e                	mv	a0,a5

0000002c <.L5>:
  2c:	ff0753e3          	ble	a6,a4,12 <.L6>

00000030 <.L3>:
  30:	c208                	sw	a0,0(a2)
  32:	8082                	ret

00000034 <.L4>:
  34:	fff78713          	addi	a4,a5,-1
  38:	bfd5                	j	2c <.L5>

0000003a <.L8>:
  3a:	853e                	mv	a0,a5
  3c:	bfd5                	j	30 <.L3>

0000003e <.L2>:
  3e:	00062023          	sw	zero,0(a2)
  42:	8082                	ret

Disassembly of section .text.iRMSnorm_s8_s8:

00000000 <iRMSnorm_s8_s8>:
   0:	715d                	addi	sp,sp,-80
   2:	d85a                	sw	s6,48(sp)
   4:	02f74b33          	div	s6,a4,a5
   8:	c4a2                	sw	s0,72(sp)
   a:	c2a6                	sw	s1,68(sp)
   c:	c0ca                	sw	s2,64(sp)
   e:	de4e                	sw	s3,60(sp)
  10:	da56                	sw	s5,52(sp)
  12:	d65e                	sw	s7,44(sp)
  14:	d462                	sw	s8,40(sp)
  16:	d266                	sw	s9,36(sp)
  18:	d06a                	sw	s10,32(sp)
  1a:	c686                	sw	ra,76(sp)
  1c:	dc52                	sw	s4,56(sp)
  1e:	8aaa                	mv	s5,a0
  20:	89ae                	mv	s3,a1
  22:	8bb2                	mv	s7,a2
  24:	40b00433          	neg	s0,a1
  28:	4481                	li	s1,0
  2a:	4901                	li	s2,0
  2c:	40a58c33          	sub	s8,a1,a0
  30:	07f00c93          	li	s9,127
  34:	f8000d13          	li	s10,-128

00000038 <.L11>:
  38:	03694063          	blt	s2,s6,58 <.L18>
  3c:	40b6                	lw	ra,76(sp)
  3e:	4426                	lw	s0,72(sp)
  40:	4496                	lw	s1,68(sp)
  42:	4906                	lw	s2,64(sp)
  44:	59f2                	lw	s3,60(sp)
  46:	5a62                	lw	s4,56(sp)
  48:	5ad2                	lw	s5,52(sp)
  4a:	5b42                	lw	s6,48(sp)
  4c:	5bb2                	lw	s7,44(sp)
  4e:	5c22                	lw	s8,40(sp)
  50:	5c92                	lw	s9,36(sp)
  52:	5d02                	lw	s10,32(sp)
  54:	6161                	addi	sp,sp,80
  56:	8082                	ret

00000058 <.L18>:
  58:	01548a33          	add	s4,s1,s5
  5c:	cc02                	sw	zero,24(sp)
  5e:	85d2                	mv	a1,s4
  60:	4601                	li	a2,0
  62:	4701                	li	a4,0
  64:	008c08b3          	add	a7,s8,s0

00000068 <.L12>:
  68:	00b88533          	add	a0,a7,a1
  6c:	04f54263          	blt	a0,a5,b0 <.L13>
  70:	c211                	beqz	a2,74 <.L14>
  72:	cc3a                	sw	a4,24(sp)

00000074 <.L14>:
  74:	4662                	lw	a2,24(sp)
  76:	0828                	addi	a0,sp,24
  78:	4581                	li	a1,0
  7a:	02f64633          	div	a2,a2,a5
  7e:	c636                	sw	a3,12(sp)
  80:	c43e                	sw	a5,8(sp)
  82:	c242                	sw	a6,4(sp)
  84:	0605                	addi	a2,a2,1
  86:	cc32                	sw	a2,24(sp)
  88:	0870                	addi	a2,sp,28
  8a:	00000097          	auipc	ra,0x0
  8e:	000080e7          	jalr	ra
  92:	48f2                	lw	a7,28(sp)
  94:	46b2                	lw	a3,12(sp)
  96:	47a2                	lw	a5,8(sp)
  98:	4812                	lw	a6,4(sp)
  9a:	00998733          	add	a4,s3,s1
  9e:	855e                	mv	a0,s7

000000a0 <.L15>:
  a0:	00870633          	add	a2,a4,s0
  a4:	00f64f63          	blt	a2,a5,c2 <.L17>
  a8:	0905                	addi	s2,s2,1
  aa:	8c1d                	sub	s0,s0,a5
  ac:	94be                	add	s1,s1,a5
  ae:	b769                	j	38 <.L11>

000000b0 <.L13>:
  b0:	0015860b          	p.lb	a2,1(a1!)
  b4:	9636                	add	a2,a2,a3
  b6:	10064633          	p.exths	a2,a2
  ba:	42c60733          	p.mac	a4,a2,a2
  be:	4605                	li	a2,1
  c0:	b765                	j	68 <.L12>

000000c2 <.L17>:
  c2:	001a060b          	p.lb	a2,1(s4!)
  c6:	0045258b          	p.lw	a1,4(a0!)
  ca:	9636                	add	a2,a2,a3
  cc:	02b60633          	mul	a2,a2,a1
  d0:	07f00593          	li	a1,127
  d4:	03164633          	div	a2,a2,a7
  d8:	41065633          	sra	a2,a2,a6
  dc:	00ccc863          	blt	s9,a2,ec <.L16>
  e0:	f8000593          	li	a1,-128
  e4:	01a64463          	blt	a2,s10,ec <.L16>
  e8:	100665b3          	p.extbs	a1,a2

000000ec <.L16>:
  ec:	00b700ab          	p.sb	a1,1(a4!)
  f0:	bf45                	j	a0 <.L15>
