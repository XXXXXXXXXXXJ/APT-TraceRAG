/*
 * YARA rules for x64 PE compiler detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "pe"

rule aut2exe_3300_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.0.0"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 40 53 48 83 EC 20 48 8D 05 7B 60 09 00 48 8B D9 48 89 01 E8 F8 00 00 00 48 8B 4B 08 48 83 C4 20 5B E9 CE 75 01 00 CC CC CC CC CC CC CC CC CC CC 48 83 EC 28 48 8D 0D F5 EA 0B 00 E8 20 01 00 00 83 3D 59 EB 0B 00 00 76 37 48 89 5C 24 30 33 DB 48 89 7C 24 20 48 8B FB 48 8B 0D 39 EB 0B 00 48 8B 0C 0F E8 }
	condition:
		$1 at 0x400
}

rule aut2exe_338x_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.8.x"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 48 83 EC 28 83 3D B5 20 0C 00 00 0F 85 DF BB 02 00 48 83 C4 28 C3 CC CC CC CC CC CC CC CC CC CC 40 53 48 83 EC 20 48 8B D9 E8 62 00 00 00 84 C0 0F 84 1A B8 02 00 48 8B 43 10 48 83 C4 20 5B C3 48 83 EC 28 48 83 79 10 00 75 36 48 89 5C 24 20 48 8B D9 48 8D 0D 6E 66 09 00 FF 15 B8 25 09 00 48 89 03 48 }
	condition:
		$1 at 0x400
}

rule aut2exe_33100_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.10.0"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 48 8B C4 48 89 58 08 48 89 68 10 48 89 70 18 48 89 78 20 41 56 48 83 EC 20 48 8B 7C 24 50 4D 8B D9 45 8B F0 8B DA 48 8B F1 4C 8B D1 41 8B D6 49 8B CA E8 29 00 00 00 4C 8B D0 48 85 C0 0F 85 C9 43 09 00 48 8B 5C 24 30 48 8B 6C 24 38 48 8B 74 24 40 48 8B 7C 24 48 48 83 C4 20 41 5E C3 CC CC 44 8B D2 EB }
	condition:
		$1 at 0x400
}

rule aut2exe_33102_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.10.2"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 48 89 5C 24 08 55 56 57 41 54 41 55 41 56 41 57 48 83 EC 60 48 8B 7A 10 48 83 64 24 50 00 83 CD FF 49 8B D8 48 8B F2 4C 8B F1 44 8B FD 44 8B E5 44 8B ED 89 AC 24 B8 00 00 00 89 AC 24 A8 00 00 00 48 83 FF 07 0F 87 31 96 08 00 48 83 FF 06 76 10 48 8B 46 08 48 8B 48 30 E8 F2 D1 00 00 44 8B F8 48 83 FF }
	condition:
		$1 at 0x400
}

rule aut2exe_33140_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.14.0"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 48 83 EC 28 48 8D 0D B5 F2 0D 00 E8 70 04 04 00 48 8D 0D C5 09 04 00 48 83 C4 28 E9 34 5A 02 00 48 83 EC 28 E8 53 8C 01 00 48 8D 0D B8 09 04 00 48 83 C4 28 E9 1B 5A 02 00 CC CC CC 48 83 EC 28 E8 27 8E 01 00 48 8D 0D A8 09 04 00 48 83 C4 28 E9 FF 59 02 00 CC CC CC 48 8D 0D 9D 09 04 00 E9 F0 59 02 00 }
	condition:
		$1 at 0x400
}

rule aut2exe_33142_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.14.2"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 48 83 EC 28 48 8D 0D B5 02 0E 00 E8 B0 06 04 00 48 8D 0D 05 0C 04 00 48 83 C4 28 E9 74 5C 02 00 48 83 EC 28 E8 73 8E 01 00 48 8D 0D F8 0B 04 00 48 83 C4 28 E9 5B 5C 02 00 CC CC CC 48 83 EC 28 E8 47 90 01 00 48 8D 0D E8 0B 04 00 48 83 C4 28 E9 3F 5C 02 00 CC CC CC 48 8D 0D DD 0B 04 00 E9 30 5C 02 00 }
	condition:
		$1 at 0x400
}

rule aut2exe_33143_x64 {
	meta:
        category = "compiler"
		tool = "C"
		name = "Aut2Exe"
		version = "3.3.14.3"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
		absoluteStart = 1024
	strings:
		$1 = { 48 83 EC 28 48 8D 0D B5 22 0E 00 E8 C0 07 04 00 48 8D 0D 15 0D 04 00 48 83 C4 28 E9 84 5D 02 00 48 83 EC 28 E8 73 8E 01 00 48 8D 0D 08 0D 04 00 48 83 C4 28 E9 6B 5D 02 00 CC CC CC 48 83 EC 28 E8 47 90 01 00 48 8D 0D F8 0C 04 00 48 83 C4 28 E9 4F 5D 02 00 CC CC CC 48 8D 0D ED 0C 04 00 E9 40 5D 02 00 }
	condition:
		$1 at 0x400
}

rule msvc_general_x64
{
	meta:
        category = "compiler"
		tool = "C"
		name = "MSVC"
		source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 ?? ?? 00 00 48 83 C4 28 E9 ?? FE FF FF CC CC }
	condition:
		$1 at pe.entry_point
}

rule gc_x64
{
	meta:
        category = "compiler"
		tool = "C"
		name = "gc"
		language = "Go"
	strings:
		$1 = { 48 8D 74 24 08 48 8B 3C 24 48 8D 05 10 00 00 00 FF E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 48 8D 05 ?9 C? FF FF FF E0 ?? ?? ?? ?? ?? ?? ?? 51 48 8B 01 48 8B 71 10 48 8B 49 08 65 48 8B 3C 25 30 00 00 00 C7 47 68 00 00 00 00 48 81 EC 80 00 00 00 83 F9 04 7E 11 83 F9 10 7E 02 CD 03 48 89 E7 FC F3 48 A5 48 89 E6 48 8B 0E 48 8B 56 08 4C 8B 46 10 4C 8B 4E 18 }
	condition:
		$1 at pe.entry_point
}

rule gc_mingw_x64
{
	meta:
        category = "compiler"
		tool = "C"
		name = "gc"
		language = "Go"
		strings:
		$1 = { 48 83 EC 28 48 8B 05 ?5 ?? ?? 00 C7 00 00 00 00 00 E8 ?A ?? ?? 00 E8 95 FC FF FF 90 90 48 83 C4 28 C3 90 90 90 90 90 90 90 90 90 90 90 90 90 90 55 48 89 E5 5D C3 66 2E 0F 1F 84 00 00 00 00 00 55 48 89 E5 48 83 EC 20 48 83 3D ?0 ?? ?? 00 00 74 30 48 8D 0D A7 ?A ?? 00 FF 15 ?? ?? ?? 00 48 85 C0 74 2F }
	condition:
		$1 at pe.entry_point
}
