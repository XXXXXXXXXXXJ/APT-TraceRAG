/*
 * YARA rules for x64 ELF compiler detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "elf"

rule gcc_470_rhel_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "GCC"
		version = "4.7.0"
		extra = "RHEL"
	strings:
		$1 = { 31 ED 49 89 D1 5E 48 89 E2 48 83 E4 F0 50 54 49 C7 C0 ?0 ?? ?? 0? 48 C7 C1 ?0 ?? ?? 0? 48 C7 C7 ?? ?? ?? 0? E8 ?? ?? ?? ?? F4 66 90 48 83 EC 08 48 8B 05 ?? ?? ?? 0? 48 85 C0 74 02 FF D0 48 83 C4 08 C3 }
	condition:
		$1 at elf.entry_point
}

rule gcc_463_ubuntu_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "GCC"
		version = "4.6.3"
		extra = "Ubuntu"
	strings:
		$1 = { 31 ED 49 89 D1 5E 48 89 E2 48 83 E4 F0 50 54 49 C7 C0 ?0 ?? ?? 00 48 C7 C1 ?0 ?? ?? 00 48 C7 C7 ?? ?? 40 00 E8 ?? ?? ?? ?? F4 90 90 ?? ?? ?? ?? 48 8? ?? ?? ?? ?? ?? ?? ?? ?0 ?? ?? ?? ?? 48 83 C4 08 C3 }
	condition:
		$1 at elf.entry_point
}

rule gcc_472_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "GCC"
		version = "4.7.2"
	strings:
		$1 = { 31 ED 5E 89 E1 83 E4 F0 50 54 52 68 ?0 ?? 0? 08 68 ?0 ?? 0? 08 51 56 68 ?? ?? 0? 08 E8 ?? ?? ?? ?? F4 66 90 }
	condition:
		$1 at elf.entry_point
}

rule tcc_0_9_26_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "Tiny C Compiler"
		version = "0.9.26"
		source = "Made by Retdec Team"
	strings:
		$1 = { 31 ED 49 89 D1 5E 48 89 E2 48 83 E4 F0 50 54 49 C7 C0 ?0 ?? 0? 08 48 C7 C1 ?0 ?? 0? 08 48 C7 C7 ?? ?? 0? 08 E8 ?7 ?? 0? 00 F4 90 90 48 83 EC 08 48 8B 05 ?9 ?? 0? 00 48 85 C0 74 02 FF D0 48 83 C4 08 C3 55 48 89 E5 48 81 EC ?0 0? 00 00 }
	condition:
		$1 at elf.entry_point
}

rule gc_1_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 77 08 48 8B 3F 48 8D 05 02 00 00 00 FF E0 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 48 8B 7C 24 08 8B 74 24 10 8B 54 24 14 4C 8B 54 24 18 4C 8B 44 24 20 B8 C6 01 00 00 0F 05 89 44 24 28 C3 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8B 7C 24 08 8B 74 24 10 B8 C7 01 00 00 0F 05 C3 }
	condition:
		$1 at elf.entry_point
}

rule gc_2_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 74 24 08 48 8B 3C 24 48 8D 05 10 00 00 00 FF E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 48 8B 7C 24 08 48 8B 74 24 10 48 8B 54 24 18 B8 35 01 00 00 0F 05 73 03 48 F7 D8 89 44 24 20 C3 }
	condition:
		$1 at elf.entry_point
}

rule gc_3_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 74 24 08 48 8B 3C 24 48 8D 05 10 00 00 00 FF E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 48 83 EC 2? 4? 8? ?? 24 ?? 4? 8? ?C 24 ?0 4C 8B ?4 24 4? 4? 8B ?C 24 ?8 4? 8B ?4 24 ?0 ?8 }
	condition:
		$1 at elf.entry_point
}

rule gc_4_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 74 24 08 48 8B 3C 24 48 8D 05 10 00 00 00 FF E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 48 83 EC 10 48 89 6C 24 08 48 8D 6C 24 08 48 8B 6C 24 08 48 83 C4 10 C3 }
	condition:
		$1 at elf.entry_point
}

rule gc_5_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 74 24 08 48 8B 3C 24 48 8D 05 10 00 00 00 FF E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 8B 7C 24 08 B8 E7 00 00 00 0F 05 C3 ?? ?? ?? ?? 48 8B 7C 24 08 8B 74 24 10 8B 54 24 14 B8 02 00 00 00 0F 05 48 3D 01 F0 FF FF 76 05 B8 FF FF FF FF 89 44 24 18 C3 }
	condition:
		$1 at elf.entry_point
}

rule gc_6_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 74 24 08 48 8B 3C 24 48 8D 05 10 00 00 00 FF E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 8B 7C 24 08 B8 E7 00 00 00 0F 05 C3 ?? ?? ?? ?? 48 8B 7C 24 }
	condition:
		$1 at elf.entry_point
}

rule gc_7_x64_elf
{
	meta:
        category = "compiler"
        author = "RetDec Team"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 77 08 48 8B 3F 48 8D 05 02 00 00 00 FF E0 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 48 8B 7C 24 08 8B 74 24 10 8B 54 24 14 B8 D5 01 00 00 0F 05 73 03 48 F7 D8 89 44 24 18 C3 ?? ?? 48 8B 7C 24 08 8B 74 24 10 B8 D6 01 00 00 0F 05 73 03 48 F7 D8 89 44 24 18 C3 ?? ?? ?? ?? ?? ?? 48 8B 7C 24 }
	condition:
		$1 at elf.entry_point
}
