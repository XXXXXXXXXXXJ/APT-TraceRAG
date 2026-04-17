/*
 * YARA rules for x86 ELF installer detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "elf"

rule p7zip_904_elf
{
	meta:
        category = "installer"
        author = "RetDec Team"
		tool = "I"
		name = "p7zip SFX"
		source = "Made by RetDec Team"
	strings:
		$1 = { 31 ED 5E 89 E1 83 E4 F0 50 54 52 68 ?0 ?? 09 08 68 ?? ?? 04 08 51 56 68 ?? ?? 0? 08 E8 ?? FD FF FF F4 89 F6 55 89 E5 83 EC 14 53 E8 00 00 00 00 5B 81 C3 ?? ?? 0? 00 8B 83 ?? 02 00 00 85 C0 74 02 FF D0 5B C9 C3 89 F6 90 90 90 90 90 90 90 90 55 89 E5 83 EC 08 83 3D ?C ?? 0? 08 00 75 3E EB 12 A1 ?8 }
	condition:
		$1 at elf.entry_point
}
