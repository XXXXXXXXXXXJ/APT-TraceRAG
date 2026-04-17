/*
 * YARA rules for x86 PE packer detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "pe"
import "dotnet"

rule blizzard_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BlizzardProtector"
		version = "1.0"
		description = "BlizzardProtector"
	condition:
		filesize > 5MB and
		(pe.sections[4].name == "_RDATA" or pe.sections[5].name == "_RDATA" or pe.sections[6].name == "_RDATA" or pe.sections[7].name == "_RDATA") and
		(pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_TLS].virtual_address != 0) and
		(
			(
				pe.machine == pe.MACHINE_I386 and
				pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_TLS].size == 0x18
			)
			or
			(
				pe.machine == pe.MACHINE_AMD64 and
				pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_TLS].size == 0x28
			)
		)
		and
		(
			(
				pe.number_of_imports <= 2 and
				(pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_IMPORT].size & 0xFFF00000) != 0 and
				pe.imports("user32.dll", "MessageBoxW")
			)
			or
			(
				uint32(pe.sections[2].raw_data_offset + pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_EXPORT].virtual_address - pe.sections[2].virtual_address) == 0 and
				uint32(pe.sections[2].raw_data_offset + pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_EXPORT].virtual_address - pe.sections[2].virtual_address + 4) == 0xFFFFFFFF
			)
		)
}

rule ep_exepack_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "!EP"
		version = "1.0"
		description = "ExePack"
	strings:
		$1 = { 60 68 ?? ?? ?? ?? B8 ?? ?? ?? ?? FF 10 68 ?? ?? ?? ?? 50 B8 ?? ?? ?? ?? FF 10 68 ?? ?? ?? ?? 6A 40 FF D0 89 05 ?? ?? ?? ?? 89 C7 BE ?? ?? ?? ?? 60 FC B2 80 31 DB A4 B3 02 E8 6D 00 00 00 73 F6 31 C9 E8 64 00 00 00 73 1C 31 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 10 C0 73 F7 75 3F AA EB D4 E8 4D 00 00 00 29 D9 75 10 E8 42 00 00 00 EB 28 AC D1 E8 74 4D 11 C9 EB 1C 91 48 C1 E0 08 AC E8 2C 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 89 E8 B3 01 56 89 FE 29 C6 F3 A4 5E EB 8E 00 D2 75 05 8A 16 46 10 D2 C3 }
	condition:
		$1 at pe.entry_point
}

rule ep_exepack_14lb2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "!EP"
		version = "1.4 lite b2"
		description = "ExePack"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 55 53 45 52 33 32 2E 44 4C 4C 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 4D 65 73 73 61 67 65 42 6F 78 41 00 00 00 00 00 EB 4C 47 6C 6F 62 61 6C 41 6C 6C 6F 63 00 47 6C 6F 62 61 6C 46 72 65 65 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ep_exepack_14lf_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "!EP"
		version = "1.4 lite final"
		description = "ExePack"
	strings:
		$1 = { 33 C0 8B C0 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule ep_exepack_14lf_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "!EP"
		version = "1.4 lite final"
		description = "ExePack"
	strings:
		$1 = { 90 90 90 90 61 B8 ?? ?? ?? ?? FF E0 55 8B EC 60 55 8B 75 08 8B 7D 0C E8 02 00 00 00 EB 04 8B 1C 24 C3 81 C3 00 02 00 00 53 57 8B 07 89 03 83 C7 04 83 C3 04 4E 75 F3 5F 5E FC B2 80 8A 06 46 88 07 47 02 D2 75 05 8A 16 46 12 D2 73 EF 02 D2 75 05 8A 16 46 12 D2 73 4A 33 C0 02 D2 75 05 8A 16 46 12 D2 0F 83 D6 00 00 00 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 13 C0 74 06 57 2B F8 8A 07 5F 88 07 47 EB A0 B8 01 00 00 00 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 72 EA 83 E8 02 75 28 B9 01 00 00 00 02 D2 75 05 8A 16 46 12 D2 13 C9 02 D2 75 05 8A 16 46 12 D2 72 EA 56 8B F7 2B F5 F3 A4 5E E9 58 FF FF FF 48 C1 E0 08 8A 06 46 8B E8 B9 01 00 00 00 02 D2 75 05 8A 16 46 12 D2 13 C9 02 D2 75 05 8A 16 46 12 D2 72 EA 3D 00 7D 00 00 73 1A 3D 00 05 00 00 72 0E 41 56 8B F7 2B F0 F3 A4 5E E9 18 FF FF FF 83 F8 7F 77 03 83 C1 02 56 8B F7 2B F0 F3 A4 5E E9 03 FF FF FF 8A 06 46 33 C9 C0 E8 01 74 12 83 D1 02 8B E8 56 8B F7 2B F0 F3 A4 5E E9 E7 FE FF FF 5D 2B 7D 0C 89 7D FC 61 5D C3 }
	condition:
		$1 at pe.entry_point
}

rule ep_exepack_360_406 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "!EP"
		version = "3.60 - 4.06"
		description = "ExePack"
	strings:
		$1 = { 8C C0 05 ?? ?? 0E 1F A3 ?? ?? 03 ?? ?? ?? 8E C0 8B ?? ?? ?? 8B ?? 4F 8B F7 FD F3 A4 }
	condition:
		$1 at pe.entry_point
}

rule eziriz_dotnet_reactor_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Eziriz .NET Reactor"
		version = "2.0 - 2.1"
	condition:
		pe.number_of_sections > 3 and
		pe.sections[1].raw_data_size == 0 and
		pe.sections[2].raw_data_size == 0 and
		pe.sections[3].raw_data_size == 0 and
		(
			pe.sections[0].name == "reacto" or
			pe.sections[1].name == "reacto" or
			pe.sections[2].name == "reacto" or
			pe.sections[3].name == "reacto"
		)
}

rule eziriz_dotnet_reactor_40_60 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Eziriz .NET Reactor"
		version = "4.0.0.0 - 6.0.0.0"
		description = ".NET protection"
		strength = "high"
	condition:
		pe.number_of_sections == 4 and
		pe.sections[1].name == ".sdata" and
		pe.sections[1].characteristics == 0xC0000040 and
		pe.imports("mscoree.dll")
}

rule eziriz_dotnet_reactor_62_or_newer {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Eziriz .NET Reactor"
		version = "6.2.0.0 or newer"
		description = ".NET protection"
	condition:
		pe.number_of_sections == 3 and
		pe.imports("mscoree.dll") and
		dotnet.number_of_user_strings > 8 and
		dotnet.user_strings[dotnet.number_of_user_strings - 8] == "{\x001\x001\x001\x001\x001\x00-\x002\x002\x002\x002\x002\x00-\x002\x000\x000\x000\x001\x00-\x000\x000\x000\x000\x001\x00}\x00" and
		dotnet.user_strings[dotnet.number_of_user_strings - 6] == "{\x001\x001\x001\x001\x001\x00-\x002\x002\x002\x002\x002\x00-\x003\x000\x000\x000\x001\x00-\x000\x000\x000\x000\x001\x00}\x00" and
		dotnet.user_strings[dotnet.number_of_user_strings - 4] == "{\x001\x001\x001\x001\x001\x00-\x002\x002\x002\x002\x002\x00-\x004\x000\x000\x000\x001\x00-\x000\x000\x000\x000\x001\x00}\x00" and
		dotnet.user_strings[dotnet.number_of_user_strings - 2] == "{\x001\x001\x001\x001\x001\x00-\x002\x002\x002\x002\x002\x00-\x005\x000\x000\x000\x001\x00-\x000\x000\x000\x000\x001\x00}\x00"
}

rule spirit_15_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "$pirit"
		version = "1.5"
	strings:
		$1 = { ?? ?? ?? 5B 24 55 50 44 FB 32 2E 31 5D }
	condition:
		$1 at pe.entry_point
}

rule spirit_15_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "$pirit"
		version = "1.5"
	strings:
		$1 = { B4 4D CD 21 E8 ?? ?? FD E8 ?? ?? B4 51 CD 21 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1"
	strings:
		$1 = { 90 90 90 90 68 ?? ?? ?? ?? 67 64 FF 36 00 00 67 64 89 26 00 00 F1 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_32lite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [32Lite 0.03]"
	strings:
		$1 = { 60 06 FC 1E 07 BE 90 90 90 90 6A 04 68 90 10 90 90 68 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_armadillo {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [Armadillo 3.00]"
	strings:
		$1 = { 60 E8 2A 00 00 00 5D 50 51 EB 0F B9 EB 0F B8 EB 07 B9 EB 0F 90 EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC E9 59 58 50 51 EB 85 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_cdcops {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [CD-Cops II]"
	strings:
		$1 = { 53 60 BD 90 90 90 90 8D 45 90 8D 5D 90 E8 00 00 00 00 8D 01 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_codesafe {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [CodeSafe 2.0]"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 EB 0B 83 EC 10 53 56 57 E8 C4 01 00 85 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_crunch {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [Crunch/PE Heuristic]"
	strings:
		$1 = { 55 E8 0E 00 00 00 5D 83 ED 06 8B C5 55 60 89 AD ?? ?? ?? ?? 2B 85 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_dxpack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [DxPack 1.0]"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 8B FD 81 ED 90 90 90 90 2B B9 00 00 00 00 81 EF 90 90 90 90 83 BD 90 90 90 90 90 0F 84 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_fsg {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [FSG 1.31]"
	strings:
		$1 = { BE 90 90 90 00 BF 90 90 90 00 BB 90 90 90 00 53 BB 90 90 90 00 B2 80 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_gleam {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [Gleam 1.00]"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 EB 0B 83 EC 0C 53 56 57 E8 24 02 00 FF }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_jdpack_jdprotect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [JDPack 1.x / JDProtect 0.9]"
	strings:
		$1 = { 60 E8 22 00 00 00 5D 8B D5 81 ED 90 90 90 90 2B 95 90 90 90 90 81 EA 06 90 90 90 89 95 90 90 90 90 83 BD 45 00 01 00 01 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_lcc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [LCC Win32 1.x]"
	strings:
		$1 = { 64 A1 01 00 00 00 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 9A 10 40 90 50 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_lockless_intro_pack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [Lockless Intro Pack]"
	strings:
		$1 = { 2C E8 EB 1A 90 90 5D 8B C5 81 ED F6 73 90 90 2B 85 90 90 90 90 83 E8 06 89 85 FF 01 EC AD }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_mew {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [MEW 11 SE 1.0]"
	strings:
		$1 = { E9 09 00 00 00 00 00 00 02 00 00 00 0C 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_msvc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [MSVC 7.0 DLL]"
	strings:
		$1 = { 55 8D 6C 01 00 81 EC 00 00 00 00 8B 45 90 83 F8 01 56 0F 84 00 00 00 00 85 C0 0F 84 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_mingw {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [MinGW GCC 2.x]"
	strings:
		$1 = { 55 89 E5 E8 02 00 00 00 C9 C3 90 90 45 58 45 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_pe_pack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [PE Pack 0.99]"
	strings:
		$1 = { 60 E8 11 00 00 00 5D 83 ED 06 80 BD E0 04 90 90 01 0F 84 F2 FF CC 0A }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_peprotect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [PE-Protect 0.9]"
	strings:
		$1 = { 52 51 55 57 64 67 A1 30 00 85 C0 78 0D E8 07 00 00 00 58 83 C0 07 C6 90 C3 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_peshield {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [PE-SHiELD 0.25]"
	strings:
		$1 = { 60 E8 2B 00 00 00 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 CC CC }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_realbasic {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [REALBasic]"
	strings:
		$1 = { 55 89 E5 90 90 90 90 90 90 90 90 90 90 50 90 90 90 90 90 00 01 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_vbox_stealthpe {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [VBOX 4.3 MTE / Ste@lth PE 1.01]"
	strings:
		$1 = { 0B C0 0B C0 0B C0 0B C0 0B C0 0B C0 0B C0 0B C0 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_vob_protectcd {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [VOB ProtectCD 5]"
	strings:
		$1 = { 36 3E 26 8A C0 60 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_asprotect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [ASProtect]"
	strings:
		$1 = { 60 90 90 90 90 90 90 5D 90 90 90 90 90 90 90 90 90 90 90 03 DD }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_upx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [UPX 0.6]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 83 E8 3D 50 8D B8 00 00 00 FF 57 8D B0 E8 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_watcom {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [WATCOM C/C++]"
	strings:
		$1 = { E9 00 00 00 00 90 90 90 90 57 41 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_02_xcr {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 - 0.2 [XCR 0.11]"
	strings:
		$1 = { 60 8B F0 33 DB 83 C3 01 83 C0 01 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_acprotect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [ACProtect 1.09]"
	strings:
		$1 = { 60 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 EB 02 00 00 90 90 90 04 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_borland_delphi_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Borland Delphi 3.0]"
	strings:
		$1 = { 55 8B EC 83 C4 90 90 90 90 68 ?? ?? ?? ?? 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_borland_delphi_50 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Borland Delphi 5.0 KOL/MCK]"
	strings:
		$1 = { 55 8B EC 90 90 90 90 68 ?? ?? ?? ?? 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 00 FF 90 90 90 90 90 90 90 90 00 01 90 90 90 90 90 90 90 90 90 EB 04 00 00 00 01 90 90 90 90 90 90 90 00 01 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_def {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [DEF 1.0]"
	strings:
		$1 = { BE 00 01 40 00 6A 05 59 80 7E 07 00 74 11 8B 46 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 83 C1 01 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_exesmasher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [ExeSmasher]"
	strings:
		$1 = { 9C FE 03 90 60 BE 90 90 41 90 8D BE 90 10 FF FF 57 83 CD FF EB 10 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 FE 0B E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_lcc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [LCC Win32 DLL]"
	strings:
		$1 = { 55 89 E5 53 56 57 83 7D 0C 01 75 05 E8 17 90 90 90 FF 75 10 FF 75 0C FF 75 08 A1 ?? ?? ?? ?? E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_ltc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [LTC 1.3]"
	strings:
		$1 = { 54 E8 00 00 00 00 5D 8B C5 81 ED F6 73 40 00 2B 85 87 75 40 00 83 E8 06 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_msvb {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Microsoft Visual Basic 5.0 - 6.0]"
	strings:
		$1 = { 68 ?? ?? ?? ?? E8 0A 00 00 00 00 00 00 00 00 00 30 00 00 00 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_msvc_50 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [MSVC 5.0+ (MFC)]"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_msvc_60_debug {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [MSVC 6.0 (Debug)]"
	strings:
		$1 = { 55 8B EC 51 90 90 90 01 01 90 90 90 90 68 ?? ?? ?? ?? 90 90 90 90 90 90 90 90 90 90 90 90 00 01 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 00 01 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_morphine {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Morphine 1.2]"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 EB 06 00 90 90 90 90 90 90 90 90 EB 08 E8 90 00 00 00 66 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 51 66 90 90 90 59 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_neolite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Neolite 2.0]"
	strings:
		$1 = { E9 A6 00 00 00 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_pe_shrinker {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [NorthStar PE Shrinker 1.3]"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 B3 85 40 00 2D AC 85 40 00 2B E8 8D B5 00 00 00 00 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_pack_master {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Pack Master 1.0 (PeX Clone)]"
	strings:
		$1 = { 60 E8 01 01 00 00 E8 83 C4 04 E8 01 90 90 90 E9 5D 81 ED D3 22 40 90 E8 04 02 90 90 E8 EB 08 EB 02 CD 20 FF 24 24 9A 66 BE 47 46 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_pe_intro {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [PE Intro 1.0]"
	strings:
		$1 = { 8B 04 24 9C 60 E8 14 00 00 00 5D 81 ED 0A 45 40 90 80 BD 67 44 40 90 90 0F 85 48 FF ED 0A E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_pe_ninja {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [PE Ninja 1.31]"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_penightmare {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [PENightMare 2 Beta]"
	strings:
		$1 = { 60 E9 10 00 00 00 EF 40 03 A7 07 8F 07 1C 37 5D 43 A7 04 B9 2C 3A E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_pex {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [PEX 0.99]"
	strings:
		$1 = { 60 E8 01 00 00 00 55 83 C4 04 E8 01 00 00 00 90 5D 81 FF FF FF 00 01 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_video_lan_client {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [Video-Lan-Client]"
	strings:
		$1 = { 55 89 E5 83 EC 08 90 90 90 90 90 90 90 90 90 90 90 90 90 90 01 FF FF 01 01 01 00 01 90 90 90 90 90 90 90 90 90 90 90 90 90 90 00 01 00 01 00 01 90 90 00 01 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_01_yodas_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.1 [yoda's Protector 1.02]"
	strings:
		$1 = { E8 03 00 00 00 EB 01 90 90 E9 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_bfjnt_11b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [.BJFNT 1.1b]"
	strings:
		$1 = { EB 01 EA 9C EB 01 EA 53 EB 01 EA 51 EB 01 EA 52 EB 01 EA 56 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_bfjnt_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [.BJFNT 1.2]"
	strings:
		$1 = { EB 02 69 B1 83 EC 04 EB 03 CD 20 EB EB 01 EB 9C EB 01 EB EB 00 }
	condition:
		$1 at pe.entry_point
}
rule pseudosigner_02_borlandcpp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Borland C++]"
	strings:
		$1 = { EB 10 66 62 3A 43 2B 2B 48 4F 4F 4B 90 E9 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_borland_delphi {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Borland Delphi]"
	strings:
		$1 = { 55 8B EC 83 C4 B4 B8 90 90 90 90 E8 00 00 00 00 E8 00 00 00 00 8D 40 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_borland_delphi_sm {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Borland Delphi Setup Module]"
	strings:
		$1 = { 55 8B EC 83 C4 90 53 56 57 33 C0 89 45 F0 89 45 D4 89 45 D0 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_codelock {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Code-Lock]"
	strings:
		$1 = { 43 4F 44 45 2D 4C 4F 43 4B 2E 4F 43 58 00 01 28 01 50 4B 47 05 4C 3F B4 04 4D 4C 47 4B }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_def {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [DEF 1.0]"
	strings:
		$1 = { BE 00 01 40 00 6A 05 59 80 7E 07 00 74 11 8B 46 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 83 C1 01 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_exesmasher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [ExeSmasher]"
	strings:
		$1 = { 9C FE 03 90 60 BE 90 90 41 90 8D BE 90 10 FF FF 57 83 CD FF EB 10 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 FE 0B }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_lcc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [LCC Win32 DLL]"
	strings:
		$1 = { 55 89 E5 53 56 57 83 7D 0C 01 75 05 E8 17 90 90 90 FF 75 10 FF 75 0C FF 75 08 A1 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_msvb {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Microsoft Visual Basic 5.0 - 6.0]"
	strings:
		$1 = { 68 ?? ?? ?? ?? E8 0A 00 00 00 00 00 00 00 00 00 30 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_peshrinker {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [NorthStar PE Shrinker 1.3]"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 B3 85 40 00 2D AC 85 40 00 2B E8 8D B5 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_pe_intro {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [PE Intro 1.0]"
	strings:
		$1 = { 8B 04 24 9C 60 E8 14 00 00 00 5D 81 ED 0A 45 40 90 80 BD 67 44 40 90 90 0F 85 48 FF ED 0A }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_penightmare {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [PENightMare 2 Beta]"
	strings:
		$1 = { 60 E9 10 00 00 00 EF 40 03 A7 07 8F 07 1C 37 5D 43 A7 04 B9 2C 3A }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_pex {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [PEX 0.99]"
	strings:
		$1 = { 60 E8 01 00 00 00 55 83 C4 04 E8 01 00 00 00 90 5D 81 FF FF FF 00 01 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_video_lan_client {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Video-Lan-Client]"
	strings:
		$1 = { 55 89 E5 83 EC 08 90 90 90 90 90 90 90 90 90 90 90 90 90 90 01 FF FF 01 01 01 00 01 90 90 90 90 90 90 90 90 90 90 90 90 90 90 00 01 00 01 00 01 90 90 00 01 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_watcom {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [Watcom C/C++]"
	strings:
		$1 = { 53 56 57 55 8B 74 24 14 8B 7C 24 18 8B 6C 24 1C 83 FF 03 0F 87 01 00 00 00 F1 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_yodas_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [yoda's Protector 1.02]"
	strings:
		$1 = { E8 03 00 00 00 EB 01 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_zcode {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "* PseudoSigner"
		version = "0.2 [ZCode 1.01]"
	strings:
		$1 = { E9 12 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E9 FB FF FF FF C3 68 00 00 00 00 64 FF 35 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pseudosigner_02_ddem_pe_engine {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "*** Protector"
		version = "1.1.11 (DDeM->PE Engine v0.9, DDeM->CI v0.9.2)"
	strings:
		$1 = { 53 51 56 E8 00 00 00 00 5B 81 EB 08 10 00 00 8D B3 34 10 00 00 B9 F3 03 00 00 BA 63 17 2A EE 31 16 83 C6 04 }
	condition:
		$1 at pe.entry_point
}

rule bfjnt_11b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = ".BJFnt"
		version = "1.1b"
	strings:
		$1 = { EB 01 EA 9C EB 01 EA 53 EB 01 EA 51 EB 01 EA 52 EB 01 EA 56 }
	condition:
		$1 at pe.entry_point
}

rule bfjnt_12rc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = ".BJFnt"
		version = "1.2rc"
	strings:
		$1 = { EB 02 69 B1 83 EC 04 EB 03 CD 20 EB EB 01 EB 9C EB 01 EB EB }
	condition:
		$1 at pe.entry_point
}

rule bfjnt_13 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = ".BJFnt"
		version = "1.3"
	strings:
		$1 = { EB ?? 3A ?? ?? 1E EB ?? CD 20 9C EB ?? CD 20 EB ?? CD 20 60 EB }
	condition:
		$1 at pe.entry_point
}

rule lite32_003a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "32Lite"
		version = "0.03a"
	strings:
		$1 = { 60 06 FC 1E 07 BE ?? ?? ?? ?? 6A 04 68 ?? 10 ?? ?? 68 }
	condition:
		$1 at pe.entry_point
}

rule anticrack_software_protector_109_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anticrack Software Protector"
		version = "1.09"
	strings:
		$1 = { 60 ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 ?? ?? ?? 04 }
	condition:
		$1 at pe.entry_point
}

rule anticrack_software_protector_109_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anticrack Software Protector"
		version = "1.09"
	strings:
		$1 = { 60 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 83 04 24 06 C3 ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.1"
	strings:
		$1 = { 60 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 E8 0A 00 00 00 E8 EB 0C 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_031a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.31a"
	strings:
		$1 = { 60 D1 CB 0F CA C1 CA E0 D1 CA 0F C8 EB 01 F1 0F C0 C9 D2 D1 0F C1 C0 D3 DA C0 D6 A8 EB 01 DE D0 EC 0F C1 CB D0 CF 0F C1 D1 D2 DB 0F C8 EB 01 BC C0 E9 C6 C1 D0 91 0F CB EB 01 73 0F CA 87 D9 87 D2 D0 CF 87 D9 0F C8 EB 01 C1 EB 01 A2 86 CA D0 E1 0F C0 CB 0F }
	condition:
		$1 at pe.entry_point
}

rule mslrh_031 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.31"
	strings:
		$1 = { 60 D1 CB 0F CA C1 CA E0 D1 CA 0F C8 EB 01 F1 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a"
	strings:
		$1 = { E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 81 83 C4 04 E8 0A 00 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a"
	strings:
		$1 = { EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [.BJFNT 1.3]"
	strings:
		$1 = { EB 03 3A 4D 3A 1E EB 02 CD 20 9C EB 02 CD 20 EB 02 CD 20 60 EB 02 C7 05 EB 02 CD 20 E8 03 00 00 00 E9 EB 04 58 40 50 C3 61 9D 1F EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_aspack_211d {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [ASPack 2.11d]"
	strings:
		$1 = { 60 E8 02 00 00 00 EB 09 5D 55 81 ED 39 39 44 00 C3 61 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_asppack_212 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [ASPack 2.12]"
	strings:
		$1 = { 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 A0 02 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_exe32pack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [EXE32Pack 1.3x]"
	strings:
		$1 = { 3B C0 74 02 81 83 55 3B C0 74 02 81 83 53 3B C9 74 01 BC 56 3B D2 74 02 81 85 57 E8 00 00 00 00 3B DB 74 01 90 83 C4 14 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_msvc_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [MSVC]"
	strings:
		$1 = { 55 8B EC 6A FF 68 CA 37 41 00 68 06 38 41 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 64 8F 05 00 00 00 00 83 C4 0C 5D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_msvc_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [MSVC]"
	strings:
		$1 = { 55 8B EC 56 57 BF 01 00 00 00 8B 75 0C 85 F6 5F 5E 5D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_msvc_60 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [MSVC 6.0]"
	strings:
		$1 = { 55 8B EC 53 8B 5D 08 56 8B 75 0C 57 8B 7D 10 85 F6 5F 5E 5B 5D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_msvc_70 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [MSVC 7.0]"
	strings:
		$1 = { 55 8B EC 53 8B 5D 08 56 8B 75 0C 5E 5B 5D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_neolite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [Neolite 2.0]"
	strings:
		$1 = { E9 A6 00 00 00 B0 7B 40 00 78 60 40 00 7C 60 40 00 00 00 00 00 B0 3F 00 00 12 62 40 00 4E 65 6F 4C 69 74 65 20 45 78 65 63 75 74 61 62 6C 65 20 46 69 6C 65 20 43 6F 6D 70 72 65 73 73 6F 72 0D 0A 43 6F 70 79 72 69 67 68 74 20 28 63 29 20 31 39 39 38 2C 31 39 39 39 20 4E 65 6F 57 6F 72 78 20 49 6E 63 0D 0A 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 20 28 63 29 20 31 39 39 37 2D 31 39 39 39 20 4C 65 65 20 48 61 73 69 75 6B 0D 0A 41 6C 6C 20 52 69 67 68 74 73 20 52 65 73 65 72 76 65 64 2E 00 00 00 00 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_nspack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [NsPacK 1.3]"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 B3 85 40 00 2D AC 85 40 00 2B E8 8D B5 D3 FE FF FF 8B 06 83 F8 00 74 11 8D B5 DF FE FF FF 8B 06 83 F8 01 0F 84 F1 01 00 00 61 9D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pc_guard {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PC Guard 4.xx]"
	strings:
		$1 = { FC 55 50 E8 00 00 00 00 5D EB 01 E3 60 E8 03 00 00 00 D2 EB 0B 58 EB 01 48 40 EB 01 35 FF E0 E7 61 58 5D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pecrypt {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PE Crypt 1.02]"
	strings:
		$1 = { E8 00 00 00 00 5B 83 EB 05 EB 04 52 4E 44 21 85 C0 73 02 F7 05 50 E8 08 00 00 00 EA FF 58 EB 18 EB 01 0F EB 02 CD 20 EB 03 EA CD 20 58 58 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_peshield {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PE-SHiELD 0.25]"
	strings:
		$1 = { 60 E8 2B 00 00 00 0D 0A 0D 0A 0D 0A 52 65 67 69 73 74 41 72 65 64 20 74 6F 3A 20 4E 4F 4E 2D 43 4F 4D 4D 45 52 43 49 41 4C 21 21 0D 0A 0D 0A 0D 00 58 61 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pebundle_03_3x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PEBundle 0.2 - 3.x]"
	strings:
		$1 = { 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 07 30 40 00 87 DD 61 9D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pebundle_20_24 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PEBundle 2.0x - 2.4x]"
	strings:
		$1 = { 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 07 30 40 00 87 DD 83 BD 9C 38 40 00 01 61 9D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pecompact {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PECompact 1.4x]"
	strings:
		$1 = { EB 06 68 2E A8 00 00 C3 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 90 40 00 61 9D EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pelock {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PELock NT 2.04]"
	strings:
		$1 = { EB 03 CD 20 C7 1E EB 03 CD 20 EA 9C EB 02 EB 01 EB 01 EB 60 EB 03 CD 20 EB EB 01 EB E8 03 00 00 00 E9 EB 04 58 40 50 C3 EB 03 CD 20 EB EB 03 CD 20 03 61 9D 83 C4 04 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_petite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [Petite 2.1]"
	strings:
		$1 = { B8 00 50 40 00 6A 00 68 BB 21 40 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 66 9C 60 50 83 C4 04 61 66 9D 64 8F 05 00 00 00 00 83 C4 08 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_pex {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [PeX 0.99]"
	strings:
		$1 = { 60 E8 01 00 00 00 E8 83 C4 04 E8 01 00 00 00 E9 5D 81 ED FF 22 40 00 61 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_svkp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [SVKP 1.11]"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 00 00 00 64 A0 23 00 00 00 83 C5 06 61 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 2B 04 24 74 04 75 02 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_upx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [UPX 0.89.6 - 1.02 / 1.05 - 1.24]"
	strings:
		$1 = { 60 BE 00 90 8B 00 8D BE 00 80 B4 FF 57 83 CD FF EB 3A 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 19 8B 1E 83 EE FC 11 DB 72 10 58 61 90 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_wwpack32 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [WWPack32 1.x]"
	strings:
		$1 = { 53 55 8B E8 33 DB EB 60 0D 0A 0D 0A 57 57 50 61 63 6B 33 32 20 64 65 63 6F 6D 70 72 65 73 73 69 6F 6E 20 72 6F 75 74 69 6E 65 20 76 65 72 73 69 6F 6E 20 31 2E 31 32 0D 0A 28 63 29 20 31 39 39 38 20 50 69 6F 74 72 20 57 61 72 65 7A 61 6B 20 61 6E 64 20 52 61 66 61 6C 20 57 69 65 72 7A 62 69 63 6B 69 0D 0A 0D 0A 5D 5B 90 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mslrh_032a_yodas_cryptor_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "[MSLRH]"
		version = "0.32a [yoda's cryptor 1.2]"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED F3 1D 40 00 B9 7B 09 00 00 8D BD 3B 1E 40 00 8B F7 AC 90 2C 8A C0 C0 78 90 04 62 EB 01 00 61 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF 83 C4 08 74 04 75 02 EB 02 EB 01 81 50 E8 02 00 00 00 29 5A 58 6B C0 03 E8 02 00 00 00 29 5A 83 C4 04 58 74 04 75 02 EB 02 EB 01 81 0F 31 50 0F 31 E8 0A 00 00 00 E8 EB 0C 00 00 E8 F6 FF FF FF E8 F2 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule box {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "_BOX_"
	strings:
		$1 = { 58 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? 00 00 50 68 ?? ?? ?? 00 C3 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule aase_crypter {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Aase Crypter"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 B8 A0 3E 00 10 E8 93 DE FF FF 68 F8 42 00 10 E8 79 DF FF FF 68 00 43 00 10 68 0C 43 00 10 E8 42 DF FF FF 50 E8 44 DF FF FF A3 98 66 00 10 83 3D 98 66 00 10 00 75 13 6A 00 68 18 43 00 10 68 1C 43 00 10 6A 00 E8 4B DF FF FF 68 2C 43 00 10 68 0C 43 ?? ?? ?? ?? DF FF FF 50 E8 0E DF FF FF A3 94 66 00 10 83 3D 94 66 00 10 00 75 13 6A 00 68 18 43 00 10 68 38 43 00 10 6A 00 E8 15 DF FF FF 68 48 43 00 10 68 0C 43 00 10 E8 D6 DE FF FF 50 E8 D8 DE FF FF A3 A0 66 00 10 83 3D A0 66 00 10 00 75 13 6A 00 68 18 43 00 10 68 58 43 00 10 6A 00 E8 DF DE FF FF 68 6C 43 00 10 68 0C 43 00 10 E8 A0 DE FF FF 50 E8 A2 DE FF FF }
	condition:
		$1 at pe.entry_point
}

rule abc_cryptor_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ABC Cryptor"
		version = "1.0"
	strings:
		$1 = { 68 FF 64 24 F0 68 58 58 58 58 90 FF D4 50 8B 40 F2 05 B0 95 F6 95 0F 85 01 81 BB FF 68 ?? ?? ?? ?? BF 00 ?? ?? ?? B9 00 ?? ?? ?? 80 37 ?? 47 39 CF 75 F8 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? BF 00 ?? ?? ?? B9 00 ?? ?? ?? 80 37 ?? 47 39 CF 75 F8 }
	condition:
		$1 at pe.entry_point
}

rule ace_compression_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACE COMPRESSION"
	strings:
		$1 = { 3? 3? 26 8A C0 60 E8 ?? 00 00 00 ?? ?? 48 FA ?? ?? ?? ?? 6A 77 38 39 33 6A 73 39 32 6A 61 39 73 6A 73 39 33 61 5F 3B 28 25 4C 49 2C 3A 00 EF BE AD DE ?? ?? ?? 78 ?? 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ace_compression_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACE COMPRESSION"
	strings:
		$1 = { 3? 3? 26 8A C0 60 E8 ?? 00 00 00 ?? ?? 48 FA 4D 45 54 49 4E 46 ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 EF BE AD DE ?? ?? ?? 78 ?? 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ace_compression_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACE COMPRESSION"
	strings:
		$1 = { E9 ?? ?? ?? ?? 43 6F 70 79 72 69 67 68 74 20 62 79 20 41 43 45 20 43 6F 6D 70 72 65 73 73 69 6F 6E 20 53 6F 66 74 77 61 72 65 20 28 31 39 39 38 2D 32 30 30 30 29 }
	condition:
		$1 at pe.entry_point
}

rule acidcrypt_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AcidCrypt"
	strings:
		$1 = { 60 B9 ?? ?? ?? 00 BA ?? ?? ?? 00 BE ?? ?? ?? 00 02 38 40 4E 75 FA 8B C2 8A 18 32 DF C0 CB }
	condition:
		$1 at pe.entry_point
}

rule acidcrypt_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AcidCrypt"
	strings:
		$1 = { BE ?? ?? ?? ?? 02 38 40 4E 75 FA 8B C2 8A 18 32 DF C0 CB }
	condition:
		$1 at pe.entry_point
}

rule acprotect_109 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.09"
	strings:
		$1 = { 60 F9 50 E8 01 00 00 00 7C 58 58 49 50 E8 01 00 00 00 7E 58 58 79 04 66 B9 B8 72 E8 01 00 00 00 7A 83 C4 04 85 C8 EB 01 EB C1 F8 BE 72 03 73 01 74 0F 81 01 00 00 00 F9 EB 01 75 F9 E8 01 00 00 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_135_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.35"
	strings:
		$1 = { 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 53 45 52 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 4D 65 73 73 61 67 65 42 6F 78 41 00 90 4D 69 6E 65 49 6D 70 6F 72 74 5F }
	condition:
		$1 at pe.entry_point
}

rule acprotect_135_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.35"
	strings:
		$1 = { 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 53 45 52 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 47 65 74 50 72 6F 63 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_13x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.3x"
	strings:
		$1 = { 60 50 E8 01 00 00 00 75 83 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_141_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.41"
	strings:
		$1 = { 60 76 03 77 01 7B 74 03 75 01 78 47 87 EE E8 01 00 00 00 76 83 C4 04 85 EE EB 01 7F 85 F2 EB 01 79 0F 86 01 00 00 00 FC EB 01 78 79 02 87 F2 61 51 8F 05 19 38 01 01 60 EB 01 E9 E9 01 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_141_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.41"
	strings:
		$1 = { E8 01 00 00 00 ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_14x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.4x"
	strings:
		$1 = { 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 4D 65 73 73 61 67 65 42 6F 78 41 00 90 4D 69 6E 65 49 6D 70 6F 72 74 5F 45 6E 64 73 73 00 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_14x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.4x"
	strings:
		$1 = { 60 E8 01 00 00 00 7C 83 04 24 06 C3 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_190 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "1.90"
	strings:
		$1 = { 60 0F 87 02 00 00 00 1B F8 E8 01 00 00 00 73 83 04 24 06 C3 }
	condition:
		$1 at pe.entry_point
}

rule acprotect_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ACProtect"
		version = "2.0"
	strings:
		$1 = { 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule activemark_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ActiveMark"
	strings:
		$1 = { 00 54 4D 53 41 4D 56 4F 48 A4 9B FD FF 26 24 E9 D7 F1 D6 F0 D6 AE BE FC D6 DF B5 C1 D0 1F 07 CE EF EE DD DE 4F F1 D1 AE BE 6B 62 A0 9B A4 9B FD FF 26 21 EC CE F1 D6 F0 D6 AE BE 01 00 14 00 }
	condition:
		$1 at pe.entry_point
}

rule activemark_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ActiveMark"
	strings:
		$1 = { 89 25 ?? ?? ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule activemark_5x{
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ActiveMark"
		version = "5.x"
	strings:
		$1 = { 20 2D 2D 4D 50 52 4D 4D 47 56 41 2D 2D 00 75 73 65 72 33 32 2E 64 6C 6C 00 4D 65 73 73 61 67 65 42 6F 78 41 00 54 68 69 73 20 61 70 70 6C 69 63 61 74 69 6F 6E 20 63 61 6E 6E 6F 74 20 72 75 6E 20 77 69 74 68 20 61 6E 20 61 63 74 69 76 65 20 64 65 62 75 67 67 65 72 20 69 6E 20 6D 65 6D 6F 72 79 2E 0D 0A 50 6C 65 61 73 65 20 75 6E 6C 6F 61 64 20 74 68 65 20 64 65 62 75 67 67 65 72 20 61 6E 64 20 72 65 73 74 61 72 74 20 74 68 65 20 61 70 70 6C 69 63 61 74 69 6F 6E 2E 00 57 61 72 6E 69 6E 67 }
	condition:
		$1 at pe.entry_point
}

rule activemark_531 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ActiveMark"
		version = "5.31"
	strings:
		$1 = { 79 11 7F AB 9A 4A 83 B5 C9 6B 1A 48 F9 27 B4 25 }
	condition:
		$1 at pe.entry_point
}

rule adflt2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AdFlt2"
	strings:
		$1 = { 68 00 01 9C 0F A0 0F A8 60 FD 6A 00 0F A1 BE ?? ?? AD }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_041 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 - 0.41"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_aspack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [ASPack 2.12]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_asprotect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [ASProtect 1.0]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 60 E8 01 00 00 00 90 5D 81 ED 00 00 00 00 BB 00 00 00 00 03 DD 2B 9D }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_borland_delphi {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [Borland Delphi 6.0 - 7.0]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 53 8B D8 33 C0 A3 00 00 00 00 6A 00 E8 00 00 00 FF A3 00 00 00 00 A1 00 00 00 00 A3 00 00 00 00 33 C0 A3 00 00 00 00 33 C0 A3 00 00 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_kkryptor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [k.kryptor 9 / kryptor a]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 60 E8 ?? ?? ?? ?? 5E B9 00 00 00 00 2B C0 02 04 0E D3 C0 49 79 F8 41 8D 7E 2C 33 46 ?? 66 B9 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_msvc_70 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [MSVC 7.0]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 6A 00 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? BF ?? ?? ?? ?? 8B C7 E8 ?? ?? ?? ?? 89 65 00 8B F4 89 3E 56 FF 15 ?? ?? ?? ?? 8B 4E ?? 89 0D ?? ?? ?? 00 8B 46 00 A3 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_pcguard {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [PCGuard 4.03 - 4.15]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 FC 55 50 E8 00 00 00 00 5D EB 01 E3 60 E8 03 00 00 00 D2 EB 0B 58 EB 01 48 40 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_pecrypt {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [PE-Crypt 1.02]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 E8 00 00 00 00 5B 83 EB 05 EB 04 52 4E 44 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_pelock {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [PELock NT 2.04]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 EB 03 CD 20 C7 1E EB 03 CD 20 EA 9C EB 02 EB 01 EB 01 EB 60 EB 03 CD 20 EB EB 01 EB }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_peshield {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [PE-SHiELD 2.x]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 60 E8 00 00 00 00 41 4E 41 4B 49 4E 5D 83 ED 06 EB 02 EA 04 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_petite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [Petite 2.2]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 B8 00 00 00 00 68 00 00 00 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 66 9C 60 50 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_spalsher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [Spalsher 1.x - 3.x]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 9C 60 8B 44 24 24 E8 00 00 00 00 5D 81 ED 00 00 00 00 50 E8 ED 02 00 00 8C C0 0F 84 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_stones_pe_encryptor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [Stone's PE Encryptor 2.0]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 53 51 52 56 57 55 E8 00 00 00 00 5D 81 ED 42 30 40 00 FF 95 32 35 40 00 B8 37 30 40 00 03 C5 2B 85 1B 34 40 00 89 85 27 34 40 00 83 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_svkp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [SVKP 1.3x]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 60 E8 00 00 00 00 5D 81 ED 06 00 00 00 EB 05 B8 00 00 00 00 64 A0 23 00 00 00 EB 03 C7 84 E8 84 C0 EB 03 C7 84 E9 75 67 B9 49 00 00 00 8D B5 C5 02 00 00 56 80 06 44 46 E2 FA 8B 8D C1 02 00 00 5E 55 51 6A 00 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_telock {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [tElock 0.61]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 E9 00 00 00 00 60 E8 00 00 00 00 58 83 C0 08 F3 EB FF E0 83 C0 28 50 E8 00 00 00 00 5E B3 33 8D 46 0E 8D 76 31 28 18 F8 73 00 C3 8B FE B9 3C 02 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_virus {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [VIRUS / I-Worm Hybris]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 EB 16 A8 54 00 00 47 41 42 4C 4B 43 47 43 00 00 00 00 00 00 52 49 53 00 FC 68 4C 70 40 00 FF 15 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_vob_protectcd {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [VOB ProtectCD]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 5F 81 EF 00 00 00 00 BE 00 00 40 00 8B 87 00 00 00 00 03 C6 57 56 8C A7 00 00 00 00 FF 10 89 87 00 00 00 00 5E 5F }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_extreme_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [Xtreme-Protector 1.05]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 E8 00 00 00 00 5D 81 00 00 00 00 00 6A 45 E8 A3 00 00 00 68 00 00 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule ahteam_ep_protector_03_zcode {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AHTeam EP Protector"
		version = "0.3 [ZCode 1.01]"
	strings:
		$1 = { 90 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 90 FF E0 E9 12 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E9 FB FF FF FF C3 68 00 00 00 00 64 FF 35 }
	condition:
		$1 at pe.entry_point
}

rule ai1_creator_1b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AI1 Creator"
		version = "1b2"
	strings:
		$1 = { E8 FE FD FF FF 6A 00 E8 0D 00 00 00 CC FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule alex_protector_04b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Alex Protector"
		version = "0.4b1"
	strings:
		$1 = { 60 E8 01 00 00 00 C7 83 C4 04 33 C9 E8 01 00 00 00 68 83 C4 04 E8 01 00 00 00 68 83 C4 04 B9 ?? 00 00 00 E8 01 00 00 00 68 83 C4 04 E8 00 00 00 00 E8 01 00 00 00 C7 83 C4 04 8B 2C 24 83 C4 04 E8 01 00 00 00 A9 83 C4 04 81 ED 3C 13 40 00 E8 01 00 00 00 68 83 C4 04 E8 00 00 00 00 E8 00 00 00 00 49 E8 01 00 00 00 68 83 C4 04 85 C9 75 DF E8 B9 02 00 00 E8 01 00 00 00 C7 83 C4 04 8D 95 63 14 40 00 E8 01 00 00 00 C7 83 C4 04 90 90 90 E8 CA 01 00 00 01 02 03 04 05 68 90 60 8B 74 24 24 8B 7C 24 28 FC B2 }
	condition:
		$1 at pe.entry_point
}

rule alex_protector_10b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Alex Protector"
		version = "1.0b2"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 10 40 00 E8 24 00 00 00 EB 01 E9 8B 44 24 0C EB 03 EB 03 C7 EB FB E8 01 00 00 00 A8 83 C4 04 83 80 B8 00 00 00 02 33 C0 EB 01 E9 C3 58 83 C4 04 EB 03 EB 03 C7 EB FB E8 01 00 00 00 A8 83 C4 04 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 EB 01 E9 FF FF 60 EB 03 EB 03 C7 EB FB E8 01 00 00 00 A8 83 C4 04 0F 31 8B D8 EB 03 EB 03 C7 EB FB E8 01 00 00 00 A8 83 C4 04 8B CA EB 03 EB 03 C7 EB FB E8 01 00 00 00 A8 83 C4 04 0F 31 2B C3 EB 03 EB 03 C7 EB FB E8 01 00 00 00 A8 83 }
	condition:
		$1 at pe.entry_point
}

rule alex_protector_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Alex Protector"
		version = "1.0"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 10 40 00 E8 24 00 00 00 EB 01 E9 8B }
	condition:
		$1 at pe.entry_point
}

rule alloy_1x2000 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Alloy"
		version = "1.x.2000"
	strings:
		$1 = { 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 07 20 40 ?? 87 DD 6A 04 68 ?? 10 ?? ?? 68 ?? 02 ?? ?? 6A ?? FF 95 46 23 40 ?? 0B }
	condition:
		$1 at pe.entry_point
}

rule alloy_4x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Alloy"
		version = "4.x"
	strings:
		$1 = { 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 07 30 40 00 87 DD 6A 04 68 00 10 00 00 68 00 02 00 00 6A 00 FF 95 A8 33 40 00 0B C0 0F 84 F6 01 00 00 89 85 2E 33 40 00 83 BD E8 32 40 00 01 74 0D 83 BD E4 32 40 00 01 74 2A 8B F8 EB 3E 68 D8 01 00 00 50 FF 95 CC 33 40 00 50 8D 85 28 33 40 00 50 FF B5 2E 33 40 00 FF 95 D0 33 40 00 58 83 C0 05 EB 0C 68 D8 01 00 00 50 FF 95 C0 33 40 00 8B BD 2E 33 40 00 03 F8 C6 07 5C 47 8D B5 00 33 40 00 AC 0A C0 74 03 AA EB F8 83 BD DC 32 40 00 01 }
	condition:
		$1 at pe.entry_point
}

rule andpakk_2006 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ANDpakk"
		version = "2.0.06"
	strings:
		$1 = { 60 FC BE ?? ?? ?? ?? BF ?? ?? ?? ?? 57 83 CD FF 33 C9 F9 EB 05 A4 02 DB 75 05 8A 1E 46 12 DB 72 F4 33 C0 40 02 DB 75 05 8A 1E 46 12 DB 13 C0 02 DB 75 05 8A 1E 46 12 DB 72 0E 48 02 DB 75 05 8A 1E 46 12 DB 13 C0 EB DC 83 E8 03 72 0F C1 E0 08 AC 83 F0 FF 74 4D D1 F8 8B E8 EB 09 02 DB 75 05 8A 1E 46 12 DB 13 C9 02 DB 75 05 8A 1E 46 12 DB 13 C9 75 1A 41 02 DB 75 05 8A 1E 46 12 DB 13 C9 02 DB 75 05 8A 1E 46 12 DB 73 EA 83 C1 02 81 FD ?? ?? ?? ?? 83 D1 01 56 8D 34 2F F3 A4 5E E9 73 FF FF FF C3 }
	condition:
		$1 at pe.entry_point
}

rule andpakk_2018 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ANDpakk"
		version = "2.0.18"
	strings:
		$1 = { FC BE ?? ?? ?? ?? BF ?? ?? ?? ?? 57 83 CD FF 33 C9 F9 EB 05 A4 02 DB 75 05 8A 1E 46 12 DB 72 F4 33 C0 40 02 DB 75 05 8A 1E 46 12 DB 13 C0 02 DB 75 05 8A 1E 46 12 DB 72 0E 48 02 DB 75 05 8A 1E 46 12 DB 13 C0 EB DC 83 E8 03 72 0F C1 E0 08 AC 83 F0 FF 74 4D D1 F8 8B E8 EB 09 02 DB 75 05 8A 1E 46 12 DB 13 C9 02 DB 75 05 8A 1E 46 12 DB 13 C9 75 1A 41 02 DB 75 05 8A 1E 46 12 DB 13 C9 02 DB 75 05 8A 1E 46 12 DB 73 EA 83 C1 02 81 FD ?? ?? ?? ?? 83 D1 01 56 8D 34 2F F3 A4 5E E9 73 FF FF FF C3 }
	condition:
		$1 at pe.entry_point
}

rule anskya_binder_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anskya Binder"
		version = "1.1"
	strings:
		$1 = { BE ?? ?? ?? 00 BB F8 11 40 00 33 ED 83 EE 04 39 2E 74 11 }
	condition:
		$1 at pe.entry_point
}

rule anskya_ntpacker_generator {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anskya NTPacker Generator"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 B8 88 1D 00 10 E8 C7 FA FF FF 6A 0A 68 20 1E 00 10 A1 14 31 00 10 50 E8 71 FB FF FF 8B D8 85 DB 74 2F 53 A1 14 31 00 10 50 E8 97 FB FF FF 85 C0 74 1F 53 A1 14 31 00 10 50 E8 5F FB FF FF 85 C0 74 0F 50 E8 5D FB FF FF 85 C0 74 05 E8 70 FC FF FF 5B E8 F2 F6 FF FF 00 00 48 45 41 52 54 }
	condition:
		$1 at pe.entry_point
}

rule anslym_fud_crypter {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anslym FUD Crypter"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 56 B8 38 17 05 10 E8 5A 45 FB FF 33 C0 55 68 21 1C 05 10 64 FF 30 64 89 20 EB 08 FC FC FC FC FC FC 27 54 E8 85 4C FB FF 6A 00 E8 0E 47 FB FF 6A 0A E8 27 49 FB FF E8 EA 47 FB FF 6A 0A }
	condition:
		$1 at pe.entry_point
}

rule anti007_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anti007"
		description = "NsPacK Private"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 10 00 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 60 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 60 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 60 00 00 E0 }
	condition:
		$1 at pe.entry_point
}

rule anti007_10_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anti007"
		version = "1.0 - 2.x"
		description = "NsPacK Private"
	strings:
		$1 = { 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule anti007_26 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anti007"
		version = "2.6"
	strings:
		$1 = { 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 47 65 74 53 79 73 74 65 6D 44 69 72 65 63 74 6F 72 79 41 00 00 00 43 72 65 61 74 65 46 69 6C 65 41 00 00 00 57 72 69 74 65 46 69 6C 65 00 00 00 43 6C 6F 73 65 48 61 6E 64 6C 65 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 }
	condition:
		$1 at pe.entry_point
}

rule anti007_27_35 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Anti007"
		version = "2.7 - 3.5"
	strings:
		$1 = { 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 47 65 74 54 65 6D 70 50 61 74 68 41 00 00 00 43 72 65 61 74 65 46 69 6C 65 41 00 00 00 57 72 69 74 65 46 69 6C 65 00 00 00 43 6C 6F 73 65 48 61 6E 64 6C 65 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule antidote_10_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiDote"
		version = "1.0 - 1.4"
	strings:
		$1 = { 00 00 00 00 09 01 47 65 74 43 6F 6D 6D 61 6E 64 4C 69 6E 65 41 00 DB 01 47 65 74 56 65 72 73 69 6F 6E 45 78 41 00 73 01 47 65 74 4D 6F 64 75 6C 65 46 69 6C 65 4E 61 6D 65 41 00 00 7A 03 57 61 69 74 46 6F 72 53 69 6E 67 6C 65 4F 62 6A 65 63 74 00 BF 02 52 65 73 75 6D 65 54 68 72 65 61 64 00 00 29 03 53 65 74 54 68 72 65 61 64 43 6F 6E 74 65 78 74 00 00 94 03 57 72 69 74 65 50 72 6F 63 65 73 73 4D 65 6D 6F 72 79 00 00 6B 03 56 69 72 74 75 61 6C 41 6C 6C 6F 63 45 78 00 00 A6 02 52 65 61 64 50 72 6F 63 65 73 73 4D 65 6D 6F 72 79 00 CA 01 47 65 74 54 68 72 65 61 64 43 6F 6E 74 65 78 74 00 00 62 00 43 72 65 61 74 65 50 72 6F 63 65 73 73 41 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C }
	condition:
		$1 at pe.entry_point
}

rule antidote_10b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiDote"
		version = "1.0b"
	strings:
		$1 = { E8 BB FF FF FF 84 C0 74 2F 68 04 01 00 00 68 C0 23 60 00 6A 00 FF 15 08 10 60 00 E8 40 FF FF FF 50 68 78 11 60 00 68 68 11 60 00 68 C0 23 60 00 E8 AB FD FF FF 83 C4 10 33 C0 C2 10 00 90 90 90 8B 4C 24 08 56 8B 74 24 08 33 D2 8B C6 F7 F1 8B C6 85 D2 74 08 33 D2 F7 F1 40 0F AF C1 5E C3 90 8B 44 24 04 53 55 56 8B 48 3C 57 03 C8 33 D2 8B 79 54 8B 71 38 8B C7 F7 F6 85 D2 74 0C 8B C7 33 D2 F7 F6 8B F8 47 0F AF FE 33 C0 33 DB 66 8B 41 14 8D 54 08 18 33 C0 66 8B 41 06 89 54 24 14 8D 68 FF 85 ED 7C 37 33 C0 }
	condition:
		$1 at pe.entry_point
}

rule antidote_12b_demo {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiDote"
		version = "1.2b demo"
	strings:
		$1 = { 68 69 D6 00 00 E8 C6 FD FF FF 68 69 D6 00 00 E8 BC FD FF FF 83 C4 08 E8 A4 FF FF FF 84 C0 74 2F 68 04 01 00 00 68 B0 21 60 00 6A 00 FF 15 08 10 60 00 E8 29 FF FF FF 50 68 88 10 60 00 68 78 10 60 00 68 B0 21 60 00 E8 A4 FD FF FF 83 C4 10 33 C0 C2 10 00 90 90 90 90 90 90 90 90 90 90 90 90 8B 4C 24 08 56 8B 74 24 08 33 D2 8B C6 F7 F1 8B C6 85 D2 74 08 33 D2 F7 F1 40 0F AF C1 5E C3 90 8B 44 24 04 53 55 56 8B 48 3C 57 03 C8 33 D2 8B 79 54 8B 71 38 8B C7 F7 F6 85 D2 74 0C 8B C7 33 D2 F7 F6 8B F8 47 0F AF FE 33 C0 33 DB 66 8B 41 14 8D 54 08 18 33 C0 }
	condition:
		$1 at pe.entry_point
}

rule antidote_12_demo {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiDote"
		version = "1.2 demo"
	strings:
		$1 = { E8 F7 FE FF FF 05 CB 22 00 00 FF E0 E8 EB FE FF FF 05 BB 19 00 00 FF E0 E8 BD 00 00 00 08 B2 62 00 01 52 17 0C 0F 2C 2B 20 7F 52 79 01 30 07 17 29 4F 01 3C 30 2B 5A 3D C7 26 11 26 06 59 0E 78 2E 10 14 0B 13 1A 1A 3F 64 1D 71 33 57 21 09 24 8B 1B 09 37 08 61 0F 1D 1D 2A 01 87 35 4C 07 39 0B }
	condition:
		$1 at pe.entry_point
}

rule antidote_12_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiDote"
		version = "1.2, 1.4"
	strings:
		$1 = { EB 10 66 62 3A 43 2B 2B 48 4F 4F 4B 90 E9 08 32 90 90 90 90 90 90 90 90 90 90 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 0B 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 8B 1E 83 EE FC 11 DB 73 E4 31 C9 83 E8 03 72 0D C1 E0 08 8A 06 46 83 F0 FF 74 74 89 C5 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C9 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C9 75 20 41 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C9 01 DB 73 EF 75 09 8B 1E 83 EE FC 11 DB 73 E4 83 C1 02 81 FD 00 F3 FF FF 83 D1 01 8D 14 2F 83 FD FC 76 0F 8A 02 42 88 07 47 49 75 F7 E9 63 FF FF FF 90 8B 02 83 C2 04 89 07 83 C7 04 83 E9 04 77 F1 01 CF E9 4C FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule antidote_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiDote"
		version = "1.4"
	strings:
		$1 = { 68 90 03 00 00 E8 C6 FD FF FF 68 90 03 00 00 E8 BC FD FF FF 68 90 03 00 00 E8 B2 FD FF FF 50 E8 AC FD FF FF 50 E8 A6 FD FF FF 68 69 D6 00 00 E8 9C FD FF FF 50 E8 96 FD FF FF 50 E8 90 FD FF FF 83 C4 20 E8 78 FF FF FF 84 C0 74 4F 68 04 01 00 00 68 10 22 60 00 6A 00 FF 15 08 10 60 00 68 90 03 00 00 E8 68 FD FF FF 68 69 D6 00 00 E8 5E FD FF FF 50 E8 58 FD FF FF 50 E8 52 FD FF FF E8 DD FE FF FF 50 68 A4 10 60 00 68 94 10 60 00 68 10 22 60 00 E8 58 FD FF FF 83 C4 20 33 C0 C2 10 00 8B 4C 24 08 56 8B 74 24 08 33 D2 8B C6 F7 F1 8B C6 85 D2 74 08 33 D2 F7 F1 40 0F AF C1 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule antivirus_vaccine_103 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AntiVirus Vaccine"
		version = "1.03"
	strings:
		$1 = { FA 33 DB B9 ?? ?? 0E 1F 33 F6 FC AD 35 ?? ?? 03 D8 E2 }
	condition:
		$1 at pe.entry_point
}

rule apatch_gui_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "APatch GUI"
		version = "1.1"
	strings:
		$1 = { 52 31 C0 E8 FF FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule apex_30a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Apex"
		version = "3.0a"
	strings:
		$1 = { 5F B9 14 00 00 00 51 BE 00 10 40 00 B9 00 ?? ?? 00 8A 07 30 06 46 E2 FB 47 59 E2 EA 68 ?? ?? ?? 00 C3 }
	condition:
		$1 at pe.entry_point
}

rule apex_c_blt_apex_40 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "APEX_C"
		version = "BLT Apex 4.0"
	strings:
		$1 = { 68 ?? ?? ?? ?? B9 FF FF FF 00 01 D0 F7 E2 72 01 48 E2 F7 B9 FF 00 00 00 8B 34 24 80 36 FD 46 E2 FA C3 }
	condition:
		$1 at pe.entry_point
}

rule app_encryptor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "App Encryptor"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 1F 1F 40 00 B9 7B 09 00 00 8D BD 67 1F 40 00 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule app_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "App Protector"
	strings:
		$1 = { E9 97 00 00 00 0D 0A 53 69 6C 65 6E 74 20 54 65 61 6D 20 41 70 70 20 50 72 6F 74 65 63 74 6F 72 0D 0A 43 72 65 61 74 65 64 20 62 79 20 53 69 6C 65 6E 74 20 53 6F 66 74 77 61 72 65 0D 0A 54 68 65 6E 6B 7A 20 74 6F 20 44 6F 63 68 74 6F 72 20 58 0D 0A 0D 0A }
	condition:
		$1 at pe.entry_point
}

rule arm_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ARM Protector"
		version = "0.1 - 0.3"
	strings:
		$1 = { E8 04 00 00 00 83 60 EB 0C 5D EB 05 45 55 EB 04 B8 EB F9 00 C3 E8 00 00 00 00 5D EB 01 00 81 ED 5E 1F 40 00 EB 02 83 09 8D B5 EF 1F 40 00 EB 02 83 09 BA A3 11 00 00 EB 01 00 8D 8D 92 31 40 00 8B 09 E8 14 00 00 00 83 EB 01 00 8B FE E8 00 00 00 00 58 83 C0 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
	strings:
		$1 = { E8 ?? ?? ?? ?? E9 ?? ?? ?? ?? 6A 0C 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8B 4D 08 33 FF 3B CF 76 2E 6A E0 58 33 D2 F7 F1 3B 45 0C 1B C0 40 75 1F E8 ?? ?? ?? ?? C7 00 0C 00 00 00 57 57 57 57 57 E8 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_19x_200b1_250b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "1.9x, 2.00b1, 2.50b1"
	strings:
		$1 = { 55 8B EC 6A FF 68 98 ?? ?? ?? 68 10 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_200 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.00"
	strings:
		$1 = { 55 8B EC 6A FF 68 00 02 41 00 68 C4 A0 40 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_250_250b3 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.50, 2.50b3"
	strings:
		$1 = { 55 8B EC 6A FF 68 B8 ?? ?? ?? 68 F8 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 20 ?? ?? ?? 33 D2 8A D4 89 15 D0 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_251 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.51"
	strings:
		$1 = { 55 8B EC 6A FF 68 B8 ?? ?? ?? 68 D0 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 20 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_252b2_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.52b2"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? B0 ?? ?? ?? ?? 68 60 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF ?? ?? ?? 15 24 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_252b2_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.52b2"
	strings:
		$1 = { 55 8B EC 6A FF 68 B0 ?? ?? ?? 68 60 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 24 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_252_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.52"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? E0 ?? ?? ?? ?? 68 D4 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF ?? ?? ?? 15 38 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_252_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.52"
	strings:
		$1 = { 55 8B EC 6A FF 68 E0 ?? ?? ?? 68 D4 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 38 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_253_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.53"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 40 ?? ?? ?? ?? 68 54 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF ?? ?? ?? 15 58 33 D2 8A D4 89 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_253_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.53"
	strings:
		$1 = { 55 8B EC 6A FF 68 40 ?? ?? ?? 68 54 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 58 ?? ?? ?? 33 D2 8A D4 89 15 EC }
	condition:
		$1 at pe.entry_point
}

rule armadillo_253b3 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.53b3"
	strings:
		$1 = { 55 8B EC 6A FF 68 D8 ?? ?? ?? 68 14 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_25x_26x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.5x - 2.6x"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 58 ?? ?? ?? 33 D2 8A D4 89 15 EC }
	condition:
		$1 at pe.entry_point
}

rule armadillo_260a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.60a"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 94 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 6C ?? ?? ?? 33 D2 8A D4 89 15 B4 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_260b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.60b1"
	strings:
		$1 = { 55 8B EC 6A FF 68 50 ?? ?? ?? 68 74 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 58 ?? ?? ?? 33 D2 8A D4 89 15 FC }
	condition:
		$1 at pe.entry_point
}

rule armadillo_260b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.60b2"
	strings:
		$1 = { 55 8B EC 6A FF 68 90 ?? ?? ?? 68 24 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 60 ?? ?? ?? 33 D2 8A D4 89 15 3C }
	condition:
		$1 at pe.entry_point
}

rule armadillo_260c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.60c"
	strings:
		$1 = { 55 8B EC 6A FF 68 40 ?? ?? ?? 68 F4 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 6C ?? ?? ?? 33 D2 8A D4 89 15 F4 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_260 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.60"
	strings:
		$1 = { 55 8B EC 6A FF 68 D0 ?? ?? ?? 68 34 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 68 ?? ?? ?? 33 D2 8A D4 89 15 84 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_261 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.61"
	strings:
		$1 = { 55 8B EC 6A FF 68 28 ?? ?? ?? 68 E4 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 6C ?? ?? ?? 33 D2 8A D4 89 15 0C }
	condition:
		$1 at pe.entry_point
}

rule armadillo_265b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.65b1"
	strings:
		$1 = { 55 8B EC 6A FF 68 38 ?? ?? ?? 68 40 ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 28 ?? ?? ?? 33 D2 8A D4 89 15 F4 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_275_285 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.75 - 2.85"
	strings:
		$1 = { 55 8B EC 6A FF 68 68 ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 58 53 56 57 89 65 E8 FF 15 28 ?? ?? ?? 33 D2 8A D4 89 15 24 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "2.xx (CopyMem II)"
	strings:
		$1 = { 6A ?? 8B B5 ?? ?? ?? ?? C1 E6 04 8B 85 ?? ?? ?? ?? 25 07 ?? ?? 80 79 05 48 83 C8 F8 40 33 C9 8A 88 ?? ?? ?? ?? 8B 95 ?? ?? ?? ?? 81 E2 07 ?? ?? 80 79 05 4A 83 CA F8 42 33 C0 8A 82 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_300_305 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.00 - 3.05"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 50 51 EB 0F ?? EB 0F ?? EB 07 ?? EB 0F ?? EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC ?? 59 58 50 51 EB 0F ?? EB 0F ?? EB 07 ?? EB 0F ?? EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC ?? 59 58 50 51 EB 0F }
	condition:
		$1 at pe.entry_point
}

rule armadillo_300_37x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.00a, 3.01 - 3.50a, 3.01 - 3.50, 3.6x, 3.7x"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 50 51 EB 0F ?? EB 0F ?? EB 07 ?? EB 0F ?? EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC ?? 59 58 50 51 EB 0F ?? EB 0F ?? EB 07 ?? EB 0F ?? EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC ?? 59 58 50 51 EB 0F ?? EB 0F ?? EB 07 ?? EB 0F ?? EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC ?? 59 58 60 33 C9 75 02 EB 15 ?? 33 C9 75 18 7A 0C 70 0E EB 0D ?? 72 0E 79 F1 ?? ?? ?? 79 09 74 F0 ?? 87 DB 7A F0 ?? ?? 61 50 51 EB 0F ?? EB 0F ?? EB 07 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_310_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.10"
	strings:
		$1 = { 55 8B EC 6A FF 68 E0 97 44 00 68 20 C0 42 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 4C 41 44 00 33 D2 8A D4 89 15 90 A1 44 00 8B C8 81 E1 FF 00 00 00 89 0D 8C A1 44 00 C1 E1 08 03 CA 89 0D 88 A1 44 00 C1 E8 10 A3 84 A1 44 00 33 F6 56 E8 72 16 00 00 59 85 C0 75 08 6A 1C E8 B0 00 00 00 59 89 75 FC E8 3D 13 00 00 FF 15 30 40 44 00 A3 84 B7 44 00 E8 FB 11 00 00 A3 E0 A1 44 00 E8 A4 0F 00 00 E8 E6 0E 00 00 E8 4E F6 FF FF 89 75 D0 8D 45 A4 50 FF 15 38 40 44 00 E8 77 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_310_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.10"
	strings:
		$1 = { 55 8B EC 6A FF 68 E0 97 44 00 68 20 C0 42 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 4C 41 44 00 33 D2 8A D4 89 15 90 A1 44 00 8B C8 81 E1 FF 00 00 00 89 0D 8C A1 44 00 C1 E1 08 03 CA 89 0D 88 A1 44 00 C1 E8 10 A3 84 A1 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_378 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.78"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 50 51 0F CA F7 D2 9C F7 D2 0F CA EB 0F B9 EB 0F B8 EB 07 B9 EB 0F 90 EB 08 FD EB 0B }
	condition:
		$1 at pe.entry_point
}

rule armadillo_3xx_6xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.xx - 6.xx"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 50 51 0F CA F7 D2 9C F7 D2 0F CA EB 0F B9 EB 0F B8 EB 07 B9 EB 0F 90 EB 08 FD EB 0B F2 EB F5 EB F6 F2 EB 08 FD EB E9 F3 EB E4 FC E9 9D 0F C9 8B CA F7 D1 59 58 50 51 0F CA F7 D2 9C F7 D2 0F CA EB 0F B9 EB 0F B8 EB 07 B9 EB 0F 90 EB 08 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_3xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "3.xx"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 50 51 EB 0F B9 EB 0F B8 EB 07 B9 EB 0F 90 EB 08 FD EB 0B }
	condition:
		$1 at pe.entry_point
}

rule armadillo_400 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "4.00"
	strings:
		$1 = { 55 8B EC 6A FF 68 20 8B 4B 00 68 80 E4 48 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 88 31 4B 00 33 D2 8A D4 89 15 A4 A1 4B 00 8B C8 81 E1 FF 00 00 00 89 0D A0 A1 4B 00 C1 E1 08 03 CA 89 0D 9C A1 4B 00 C1 E8 10 A3 98 A1 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_410 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "4.10"
		comment = "Silicon Realms Toolworks"
	strings:
		$1 = { 55 8B EC 6A FF 68 F8 8E 4C 00 68 D0 EA 49 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 88 31 4C 00 33 D2 8A D4 89 15 7C A5 4C 00 8B C8 81 E1 FF 00 00 00 89 0D 78 A5 4C 00 C1 E1 08 03 CA 89 0D 74 A5 4C 00 C1 E8 10 A3 70 A5 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_420 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "4.20"
		comment = "Silicon Realms Toolworks"
	strings:
		$1 = { 55 8B EC 6A FF 68 F8 8E 4C 00 68 F0 EA 49 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 88 31 4C 00 33 D2 8A D4 89 15 84 A5 4C 00 8B C8 81 E1 FF 00 00 00 89 0D 80 A5 4C 00 C1 E1 08 03 CA 89 0D 7C A5 4C 00 C1 E8 10 A3 78 A5 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_430a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "4.30a"
	strings:
		$1 = { 44 64 65 44 61 74 61 20 69 6E 69 74 69 61 6C 69 7A 65 64 20 28 41 4E 53 49 29 2C 20 61 70 70 20 73 74 72 69 6E 67 73 20 61 72 65 20 27 25 73 27 20 61 6E 64 20 27 25 73 27 00 00 00 44 64 65 44 61 74 61 20 69 6E 69 74 69 61 6C 69 7A 65 64 20 28 55 4E 49 43 4F 44 45 29 2C 20 61 70 70 20 73 74 72 69 6E 67 73 20 61 72 65 20 27 25 53 27 20 61 6E 64 20 27 25 53 27 00 00 00 00 50 75 74 53 74 72 69 6E 67 28 27 25 73 27 29 00 47 65 74 53 74 72 69 6E 67 28 29 2C 20 66 61 6C 73 65 00 00 47 65 74 53 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_430_440 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "4.30 - 4.40"
	strings:
		$1 = { 55 8B EC 6A FF 68 40 ?? ?? 00 68 80 ?? ?? 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 88 ?? ?? 00 33 D2 8A D4 89 15 30 ?? ?? 00 8B C8 81 E1 FF 00 00 00 89 0D 2C ?? ?? 00 C1 E1 08 03 CA 89 0D 28 ?? ?? 00 C1 E8 10 A3 24 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_440 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "4.40"
	strings:
		$1 = { 31 2E 31 2E 34 00 00 00 C2 E0 94 BE 93 FC DE C6 B6 24 83 F7 D2 A4 92 77 40 27 CF EB D8 6F 50 B4 B5 29 24 FA 45 08 04 52 D5 1B D2 8C 8A 1E 6E FF 8C 5F 42 89 F1 83 B1 27 C5 69 57 FC 55 0A DD 44 BE 2A 02 97 6B 65 15 AA 31 E9 28 7D 49 1B DF B5 5D 08 A8 BA A8 73 DC F6 D1 05 42 55 53 79 73 74 65 6D 00 00 53 00 79 00 73 00 74 00 65 00 6D 00 00 00 00 00 44 44 45 20 50 72 6F 63 65 73 73 69 6E 67 00 00 53 77 50 44 44 45 00 00 44 00 44 00 45 00 20 00 50 00 72 00 6F 00 63 00 65 00 73 00 73 00 69 00 6E 00 67 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_50x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "5.0x"
	strings:
		$1 = { E8 E3 40 00 00 E9 16 FE FF FF 6A 0C 68 ?? ?? ?? ?? E8 44 15 00 00 8B 4D 08 33 FF 3B CF 76 2E 6A E0 58 33 D2 F7 F1 3B 45 0C 1B C0 40 75 1F E8 36 13 00 00 C7 00 0C 00 00 00 57 57 57 57 57 E8 C7 12 00 00 83 C4 14 33 C0 E9 D5 00 00 00 0F AF 4D 0C 8B F1 89 75 08 3B F7 75 03 33 F6 46 33 DB 89 5D E4 83 FE E0 77 69 83 3D ?? ?? ?? ?? 03 75 4B 83 C6 0F 83 E6 F0 89 75 0C 8B 45 08 3B 05 ?? ?? ?? ?? 77 37 6A 04 E8 48 11 00 00 59 89 7D FC ?? 75 08 E8 01 49 00 00 59 89 45 E4 C7 45 FC FE FF FF FF E8 5F 00 00 00 8B 5D E4 3B DF 74 11 FF 75 08 57 53 E8 66 D3 FF FF 83 C4 0C 3B DF 75 61 56 6A 08 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B D8 3B DF 75 4C 39 3D ?? ?? ?? ?? 74 33 56 E8 AF F9 FF FF 59 85 C0 0F 85 72 FF FF FF 8B 45 10 3B C7 0F 84 50 FF FF FF C7 00 0C 00 00 00 E9 45 FF FF FF 33 FF 8B 75 0C 6A 04 E8 EE 0F 00 00 59 C3 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_5xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "5.xx"
	strings:
		$1 = { 83 7C 24 08 01 75 05 E8 ?? ?? ?? ?? FF 74 24 04 8B 4C 24 10 8B 54 24 0C E8 ?? ?? ?? ?? 59 C2 0C 00 6A 0C 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8B 4D 08 33 FF 3B CF 76 2E 6A E0 58 33 D2 F7 F1 3B 45 0C 1B C0 40 75 1F E8 ?? ?? ?? ?? C7 00 0C 00 00 00 57 57 57 57 57 E8 ?? ?? ?? ?? 83 C4 14 33 C0 E9 D5 00 00 00 0F AF 4D 0C 8B F1 89 75 08 3B F7 75 03 33 F6 46 33 DB 89 5D E4 83 FE E0 77 69 83 3D ?? ?? ?? ?? 03 75 4B 83 C6 0F 83 E6 F0 89 75 0C 8B 45 08 3B 05 ?? ?? ?? ?? 77 37 6A 04 E8 ?? ?? ?? ?? 59 89 7D FC FF 75 08 E8 ?? ?? ?? ?? 59 89 45 E4 C7 45 FC FE FF FF FF E8 ?? ?? ?? ?? 8B 5D E4 3B DF 74 11 FF 75 08 57 53 E8 ?? ?? ?? ?? 83 C4 0C 3B DF 75 61 56 6A 08 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B D8 3B DF 75 4C 39 3D ?? ?? ?? ?? 74 33 56 E8 ?? ?? ?? ?? 59 85 C0 0F 85 72 FF FF FF 8B 45 10 3B C7 0F 84 50 FF FF FF C7 00 0C 00 00 00 E9 45 FF FF FF 33 FF 8B 75 0C 6A 04 E8 ?? ?? ?? ?? 59 C3 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_520b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "5.20b1"
	strings:
		$1 = { E8 8E 3F 00 00 E9 16 FE FF FF 6A 0C 68 ?? ?? ?? ?? E8 9E 16 00 00 8B 4D 08 33 FF 3B CF 76 2E 6A E0 58 33 D2 F7 F1 3B 45 0C 1B C0 40 75 1F E8 F5 14 00 00 C7 00 0C 00 00 00 57 57 57 57 57 E8 86 14 00 00 83 C4 14 33 C0 E9 D5 00 00 00 0F AF 4D 0C 8B F1 89 75 08 3B F7 75 03 33 F6 46 33 DB 89 5D E4 83 FE E0 77 69 83 3D ?? ?? ?? ?? ?? 75 4B 83 C6 0F 83 E6 F0 89 75 0C 8B 45 08 3B 05 ?? ?? ?? ?? 77 37 6A 04 E8 07 13 00 00 59 89 7D FC FF 75 08 E8 AC 47 00 00 59 89 45 E4 C7 45 FC FE FF FF FF E8 5F 00 00 00 8B 5D E4 3B DF 74 11 FF 75 08 57 53 E8 7C D3 FF FF 83 C4 0C 3B DF 75 61 56 6A 08 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B D8 3B DF 75 4C 39 3D ?? ?? ?? ?? 74 33 56 E8 C7 F9 FF FF 59 85 C0 0F 85 72 FF FF FF 8B 45 10 3B C7 0F 84 50 FF FF FF C7 00 0C 00 00 00 E9 45 FF FF FF 33 FF 8B 75 0C 6A 04 E8 AD 11 00 00 59 C3 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_520 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "5.20"
	strings:
		$1 = { E8 38 3D 00 00 E9 16 FE FF FF 6A 0C 68 ?? ?? ?? ?? E8 98 1E 00 00 8B 4D 08 33 FF 3B CF 76 2E 6A E0 58 33 D2 F7 F1 3B 45 0C 1B C0 40 75 1F E8 EC 1C 00 00 C7 00 0C 00 00 00 57 57 57 57 57 E8 7D 1C 00 00 83 C4 14 33 C0 E9 D5 00 00 00 0F AF 4D 0C 8B F1 89 75 08 3B F7 75 03 33 F6 46 33 DB 89 5D E4 83 FE E0 77 69 83 3D ?? ?? ?? ?? ?? 75 4B 83 C6 0F 83 E6 F0 89 75 0C 8B 45 08 3B 05 ?? ?? ?? ?? 77 37 6A 04 E8 FE 1A 00 00 59 89 7D FC FF 75 08 E8 56 45 00 00 59 89 45 E4 C7 45 FC FE FF FF FF E8 5F 00 00 00 8B 5D E4 3B DF 74 11 FF 75 08 57 53 E8 96 D3 FF FF 83 C4 0C 3B DF 75 61 56 6A 08 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B D8 3B DF 75 4C 39 3D ?? ?? ?? ?? 74 33 56 E8 C0 FA FF FF 59 85 C0 0F 85 72 FF FF FF 8B 45 10 3B C7 0F 84 50 FF FF FF C7 00 0C 00 00 00 E9 45 FF FF FF 33 FF 8B 75 0C 6A 04 E8 A4 19 00 00 59 C3 3B DF 75 0D 8B 45 10 3B C7 74 06 C7 00 0C 00 00 00 8B C3 E8 CC 1D 00 00 C3 55 8B EC 51 83 65 FC 00 57 8D 45 FC 50 FF 75 0C FF 75 08 E8 CA FE FF FF 8B F8 83 C4 0C 85 FF 75 19 56 8B 75 FC 85 F6 74 10 E8 C9 1B 00 00 85 C0 74 07 E8 C0 1B 00 00 89 30 5E 8B C7 5F C9 C3 6A 0C 68 ?? ?? ?? ?? E8 3B 1D 00 00 8B 75 08 85 F6 74 75 83 3D ?? ?? ?? ?? ?? 75 43 6A 04 E8 FF 19 00 00 59 83 65 FC 00 56 E8 84 3C 00 00 59 89 45 E4 85 C0 74 09 56 50 E8 A0 3C 00 00 59 59 C7 45 FC FE FF FF FF E8 0B 00 00 00 83 7D E4 00 75 37 FF 75 08 EB 0A 6A 04 E8 ED 18 00 00 59 C3 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_540_542 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "5.40 - 5.42"
	strings:
		$1 = { E8 93 3E 00 00 E9 16 FE FF FF 6A 0C 68 ?? ?? ?? ?? E8 B4 1F 00 00 8B 4D 08 33 FF 3B CF 76 2E 6A E0 58 33 D2 F7 F1 3B 45 0C 1B C0 40 75 1F E8 AF 1D 00 00 C7 00 0C 00 00 00 57 57 57 57 57 E8 40 1D 00 00 83 C4 14 33 C0 E9 D5 00 00 00 0F AF 4D 0C 8B F1 89 75 08 3B F7 75 03 33 F6 46 33 DB 89 5D E4 83 FE E0 77 69 83 3D ?? ?? ?? ?? ?? 75 4B 83 C6 0F 83 E6 F0 89 75 0C 8B 45 08 3B 05 ?? ?? ?? ?? 77 37 6A 04 E8 C1 1B 00 00 59 89 7D FC FF 75 08 E8 B1 46 00 00 59 89 45 E4 C7 45 FC FE FF FF FF E8 5F 00 00 00 8B 5D E4 3B DF 74 11 FF 75 08 57 53 E8 86 D3 FF FF 83 C4 0C 3B DF 75 61 56 6A 08 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B D8 3B DF 75 4C 39 3D ?? ?? ?? ?? 74 33 56 E8 C4 FA FF FF 59 85 C0 0F 85 72 FF FF FF 8B 45 10 3B C7 0F 84 50 FF FF FF C7 00 0C 00 00 00 E9 45 FF FF FF 33 FF 8B 75 0C 6A 04 E8 67 1A 00 00 59 C3 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_6xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "6.xx"
		comment = "Silicon Realms Toolworks * Sign.By.fly * 20081227"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 60 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 D0 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 60 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 60 01 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 80 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 42 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 }
	condition:
		$1 at pe.entry_point
}

rule armadillo_6xx_minimu_protection {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Armadillo"
		version = "6.xx Minimum Protection"
	strings:
		$1 = { E8 ?? ?? ?? ?? E9 ?? ?? ?? ?? 6A 0C 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 65 E4 00 8B 75 08 3B 35 ?? ?? ?? ?? 77 22 6A 04 E8 ?? ?? ?? ?? 59 83 65 FC 00 56 E8 ?? ?? ?? ?? 59 89 45 E4 C7 45 FC FE FF }
	condition:
		$1 at pe.entry_point
}

rule ascrypt_01_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AsCrypt"
		version = "0.1"
	strings:
		$1 = { 80 ?? ?? ?? 83 ?? ?? ?? ?? 90 90 90 51 ?? ?? ?? 01 00 00 00 83 ?? ?? E2 }
	condition:
		$1 at pe.entry_point
}

rule ascrypt_01_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AsCrypt"
		version = "0.1"
	strings:
		$1 = { 80 ?? ?? ?? 83 ?? ?? ?? ?? 90 90 90 83 ?? ?? E2 }
	condition:
		$1 at pe.entry_point
}

rule ascrypt_01_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AsCrypt"
		version = "0.1"
	strings:
		$1 = { 80 ?? ?? ?? 83 ?? ?? ?? ?? 90 90 90 E2 }
	condition:
		$1 at pe.entry_point
}

rule ascrypt_01_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AsCrypt"
		version = "0.1"
	strings:
		$1 = { B9 ?? ?? ?? ?? 81 ?? ?? ?? ?? ?? ?? 83 04 24 04 ?? 90 90 90 83 E9 03 E2 EC EB ?? 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule asdpack_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASDPack"
	strings:
		$1 = { 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 8D 49 00 1F 01 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 90 }
	condition:
		$1 at pe.entry_point
}

rule asdpack_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASDPack"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 56 53 E8 5C 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 10 00 00 ?? ?? ?? 00 00 00 00 00 00 00 40 00 00 ?? ?? 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 ?? ?? 00 00 10 00 00 00 ?? 00 00 00 ?? ?? 00 00 ?? ?? 00 00 ?? ?? 00 00 ?? 00 00 00 ?? ?? 00 00 ?? 00 00 00 ?? ?? 00 00 ?? 00 00 00 ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 5B 81 EB E6 1D 40 00 83 7D 0C 01 75 11 55 E8 4F 01 00 00 E8 6A 01 00 00 5D E8 2C 00 00 00 8B B3 1A 1E 40 00 03 B3 FA 1D 40 00 8B 76 0C AD 0B C0 74 0D FF 75 10 FF 75 0C FF 75 08 FF D0 EB EE B8 01 00 00 00 5B 5E C9 C2 0C 00 55 6A 00 FF 93 20 21 40 00 89 83 FA 1D 40 00 6A 40 68 00 10 00 00 FF B3 02 1E 40 00 6A 00 FF 93 2C 21 40 00 89 83 06 1E 40 00 8B 83 F2 1D 40 00 03 83 FA 1D 40 00 50 FF B3 06 1E 40 00 50 E8 6D 01 00 00 5F }
	condition:
		$1 at pe.entry_point
}

rule asdpack_20_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASDPack"
		version = "2.0"
	strings:
		$1 = { 5B 43 83 7B 74 00 0F 84 08 00 00 00 89 43 14 E9 }
	condition:
		$1 at pe.entry_point
}

rule asdpack_20_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASDPack"
		version = "2.0"
	strings:
		$1 = { 8B 44 24 04 56 57 53 E8 CD 01 00 00 C3 00 00 00 00 00 00 00 00 00 00 00 00 00 10 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E8 ?? ?? ?? 00 E9 ?? ?? ?? ?? ?? ?? 00 8B FE B9 97 00 00 00 AD 35 78 56 34 12 AB 49 75 F6 EB }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E8 ?? ?? ?? 00 EB 33 87 DB 90 }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E8 ?? ?? ?? 00 EB 4? 00 00 00 00 00 00 00 00 }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 1)
}

rule aspack_uv_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E8 ?? ?? 00 00 EB 09 5D 55 81 ED 39 39 44 00 C3 E9 ?? ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED ?? ?? ?? 00 B8 ?? ?? ?? 00 03 C5 2B 85 ?? ?? ?? 00 89 85 ?? ?? ?? 00 80 BD ?? ?? ?? 00 00 75 15 FE 85 ?? ?? ?? 00 E8 1D 00 00 00 E8 ?? ?? 00 00 E8 ?? ?? 00 00 8B 85 }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_07 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 76 AA 44 00 BB 70 AA 44 00 03 DD 2B 9D E1 B2 44 00 83 BD DC B2 44 00 00 89 9D ED B0 }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_08 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = { 60 E9 3D 04 00 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_uv_09 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$ep = { 75 00 E9 }
		$1 = {
			60 E8 00 00 00 00 5D 81 ED ?? ?? ?? ?? B8 ?? ?? ?? ?? 03 C5 2B 85 ??
			?? ?? ?? 89 85 ?? ?? ?? ?? 80 BD ?? ?? ?? ?? 00 75 15 FE 85 ?? ?? ??
			?? E8 ?? ?? 00 00 E8 ?? ?? 00 00 E8 ?? ?? 00 00 8B 85 ?? ?? ?? ?? 03
			85 ?? ?? ?? ?? 89 44 24 1C 61 FF E0
		}
	condition:
		$ep in (pe.entry_point .. pe.entry_point + 2) and $1
}

rule aspack_uv_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$1 = {
			60 E8 03 00 00 00 ?? EB 04 5D 45 55 C3 E8 01 00 00 00 ?? 5D BB ?? ??
			FF FF ?? ?? 81 EB ?? ?? ?? ?? 83 BD 22 04 00 00 00 89 9D 22 04 00 00
			0F 85 ?? ?? 00 00 8D 85 ?? ?? 00 00 50 FF 95 ?? ?? 00 00 89 85 ?? ??
			00 00 ?? ?? 8D 5D ?? 53 50 FF 95 ?? ?? 00 00 89 85 ?? ?? 00 00 8D 5D
			?? 53 57 FF 95 ?? ?? 00 00 89 85 ?? ?? 00 00 8D 45 ?? FF E0
		}
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 2)
}

rule aspack_uv_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
	strings:
		$ep = { 75 01 ?? E9 }
		$1 = {
			60 EB 0A 5D EB 02 ?? ?? 45 FF E5 ?? ?? E8 ?? FF FF FF ?? 81 ED ?? ??
			?? ?? BB ?? ?? ?? ?? ?? ?? 2B 9D ?? ?? ?? ?? 8D 8D ?? ?? ?? ?? 51 FF
			95 ?? ?? ?? ?? 8D 8D ?? ?? ?? ?? 51 50 FF 95 ?? ?? ?? ?? 6A 00 54 6A
			04 68 ?? ?? ?? ?? 8D 8D ?? ?? ?? ?? 51 FF D0 58 89 9D ?? ?? ?? ?? 80
			BD ?? ?? ?? ?? 00 75 1A FE 85 ?? ?? ?? ?? E8 ?? 00 00 00 E8 ?? ?? 00
			00 E8 ?? ?? 00 00 E8 ?? ?? 00 00 8B 85 ?? ?? ?? ?? 03 85 ?? ?? ?? ??
			89 44 24 ?? 61
		}
	condition:
		$ep in (pe.entry_point .. pe.entry_point + 2) and $1
}

rule aspack_100b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.00b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED 92 1A 44 ?? B8 8C 1A 44 ?? 03 C5 2B 85 CD 1D 44 ?? 89 85 D9 1D 44 ?? 80 BD C4 1D 44 }
	condition:
		$1 at pe.entry_point
}

rule aspack_101b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.01b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED D2 2A 44 ?? B8 CC 2A 44 ?? 03 C5 2B 85 A5 2E 44 ?? 89 85 B1 2E 44 ?? 80 BD 9C 2E 44 }
	condition:
		$1 at pe.entry_point
}

rule aspack_102b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.02a"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED 3E D9 43 ?? B8 38 ?? ?? ?? 03 C5 2B 85 0B DE 43 ?? 89 85 17 DE 43 ?? 80 BD 01 DE 43 ?? ?? 75 15 FE 85 01 DE 43 ?? E8 1D ?? ?? ?? E8 79 02 ?? ?? E8 12 03 ?? ?? 8B 85 03 DE 43 ?? 03 85 17 DE 43 ?? 89 44 24 1C 61 FF }
	condition:
		$1 at pe.entry_point
}

rule aspack_102b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.02b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED 96 78 43 ?? B8 90 78 43 ?? 03 C5 2B 85 7D 7C 43 ?? 89 85 89 7C 43 ?? 80 BD 74 7C 43 }
	condition:
		$1 at pe.entry_point
}

rule aspack_102b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.02b"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 96 78 43 00 B8 90 78 43 00 03 C5 }
	condition:
		$1 at pe.entry_point
}

rule aspack_103b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.03b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED AE 98 43 ?? B8 A8 98 43 ?? 03 C5 2B 85 18 9D 43 ?? 89 85 24 9D 43 ?? 80 BD 0E 9D 43 }
	condition:
		$1 at pe.entry_point
}

rule aspack_104b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.04b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? B8 ?? ?? ?? ?? 03 C5 2B 85 ?? 12 9D ?? 89 85 1E 9D ?? ?? 80 BD 08 9D }
	condition:
		$1 at pe.entry_point
}

rule aspack_105b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.05b"
	strings:
		$1 = { 75 00 E9 }
	condition:
		$1 at pe.entry_point
}

rule aspack_105b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.05b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED CE 3A 44 ?? B8 C8 3A 44 ?? 03 C5 2B 85 B5 3E 44 ?? 89 85 C1 3E 44 ?? 80 BD AC 3E 44 }
	condition:
		$1 at pe.entry_point
}

rule aspack_106b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.06b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED EA A8 43 ?? B8 E4 A8 43 ?? 03 C5 2B 85 78 AD 43 ?? 89 85 84 AD 43 ?? 80 BD 6E AD 43 }
	condition:
		$1 at pe.entry_point
}

rule aspack_106b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.06b"
	strings:
		$1 = { 90 90 90 75 00 E9 }
	condition:
		$1 at pe.entry_point
}

rule aspack_107b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.07b"
	strings:
		$1 = { 60 E8 00 00 00 00 5D ?? ?? ?? ?? ?? ?? B8 ?? ?? ?? ?? 03 C5 }
	condition:
		$1 at pe.entry_point
}

rule aspack_107b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.07b"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? B8 ?? ?? ?? ?? 03 C5 2B 85 ?? 0B DE ?? 89 85 17 DE ?? ?? 80 BD 01 DE }
	condition:
		$1 at pe.entry_point
}

rule aspack_108_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.08"
	strings:
		$1 = { 90 90 90 75 01 90 E9 }
	condition:
		$1 at pe.entry_point
}

rule aspack_108_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.08"
	strings:
		$1 = { 90 90 90 75 01 FF E9 }
	condition:
		$1 at pe.entry_point
}

rule aspack_10801_10802 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.08.01 - 1.08.02"
	strings:
		$1 = { 60 EB 0A 5D EB 02 FF 25 45 FF E5 E8 E9 E8 F1 FF FF FF E9 81 ?? ?? ?? 44 ?? BB 10 ?? 44 ?? 03 DD 2B 9D }
	condition:
		$1 at pe.entry_point
}

rule aspack_10803 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.08.03"
	strings:
		$1 = { 60 E8 00 00 00 00 5D ?? ?? ?? ?? ?? ?? BB ?? ?? ?? ?? 03 DD 2B 9D B1 50 44 00 83 BD AC 50 44 00 00 89 9D BB 4E }
	condition:
		$1 at pe.entry_point
}

rule aspack_10804 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.08.04"
	strings:
		$1 = { 60 E8 41 06 00 00 EB 41 }
	condition:
		$1 at pe.entry_point
}

rule aspack_1080x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "1.08.x"
	strings:
		$1 = { 60 EB 03 5D FF E5 E8 F8 FF FF FF 81 ED 1B 6A 44 00 BB 10 6A 44 00 03 DD 2B 9D 2A }
	condition:
		$1 at pe.entry_point
}

rule aspack_2000 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.000"
	strings:
		$1 = { 60 E8 70 05 00 00 EB 4C }
	condition:
		$1 at pe.entry_point
}

rule aspack_2001 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.001"
	strings:
		$1 = { 60 E8 72 05 00 00 EB 4C }
	condition:
		$1 at pe.entry_point
}

rule aspack_21 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.1"
	strings:
		$1 = { 60 E8 72 05 00 00 EB 33 87 DB 90 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_211b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.11b"
	strings:
		$1 = { 60 E8 02 00 00 00 EB 09 5D 55 81 ED 39 39 44 00 C3 E9 3D 04 00 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_211c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.11c"
	strings:
		$1 = { 60 E8 02 00 00 00 EB 09 5D 55 81 ED 39 39 44 00 C3 E9 59 04 00 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_211d {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.11d"
	strings:
		$1 = { 60 E8 02 00 00 00 EB 09 5D 55 81 ED 39 39 44 00 C3 6? }
	condition:
		$1 at pe.entry_point
}

rule aspack_asprotect_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack or ASProtect"
		version = "2.xx"
	strings:
		$1 = { 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB }
	condition:
		$1 at pe.entry_point
}

rule aspack_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.xx"
	strings:
		$1 = { A8 03 ?? ?? 61 75 08 B8 01 ?? ?? ?? C2 0C ?? 68 ?? ?? ?? ?? C3 8B 85 26 04 ?? ?? 8D 8D 3B 04 ?? ?? 51 50 FF 95 }
	condition:
		$1 at pe.entry_point
}

rule aspack_212 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.12"
	strings:
		$1 = { 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 ?0 ?? 00 83 BD 22 04 00 00 00 89 9D 22 04 00 00 0F 85 65 03 00 00 8D 85 2E 04 00 00 50 FF 95 4D 0F 00 00 89 85 26 04 00 00 8B F8 8D 5D 5E 53 50 FF 95 49 0F 00 00 89 85 4D 05 00 00 8D 5D 6B 53 57 FF 95 49 0F 00 }
	condition:
		$1 at pe.entry_point
}

rule aspack_220 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.20"
	strings:
		$1 = { 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 ?0 ?? 00 83 BD 7D 04 00 00 00 89 9D 7D 04 00 00 0F 85 C0 03 00 00 8D 85 89 04 00 00 50 FF 95 09 0F 00 00 89 85 81 04 00 00 8B F0 8D 7D 51 57 56 FF 95 05 0F 00 00 AB B0 00 AE 75 FD 38 07 75 EE 8D 45 7A FF E0 56 }
	condition:
		$1 at pe.entry_point
}

rule aspack_224_228 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack"
		version = "2.24, 2.28"
	strings:
		$1 = { 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 ?0 ?? 00 83 BD 88 04 00 00 00 89 9D 88 04 00 00 0F 85 CB 03 00 00 8D 85 94 04 00 00 50 FF 95 A9 0F 00 00 89 85 8C 04 00 00 8B F0 8D 7D 51 57 56 FF 95 A5 0F 00 00 AB B0 00 AE 75 FD 38 07 75 EE 8D 45 7A FF E0 56 }
	condition:
		$1 at pe.entry_point
}

rule aspack_asprotect_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack-ASPROTECT"
	strings:
		$1 = { 60 E9 D5 04 00 00 F7 10 0F 0F 0F 9F 6C 90 FC DB 8C 54 0F CA CF 8C 54 0F 12 EC 3A AC 27 95 54 0F 92 CC 0F 94 54 0F 0F 98 AC 0F 94 54 0F 1E 94 58 12 0F 0F D6 94 D2 8C 54 0F 0F 0F 0F 0F 9C 94 17 }
	condition:
		$1 at pe.entry_point
}

rule aspack_asprotect_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack-ASPROTECT"
	strings:
		$1 = { 60 E9 DB 05 00 00 47 60 5F 5F 5F EF BC E0 4C A7 15 A4 5F 1A 9B 15 A4 5F 62 3C 8A FC E0 1D A4 5F E2 1C D7 1C A4 5F 5F E8 FC D7 1C A4 5F 6E E4 A8 62 5F 5F 26 E4 9E 15 A4 5F 5F 5F 5F 5F EC E4 DF }
	condition:
		$1 at pe.entry_point
}

rule aspack_asprotect_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack-ASPROTECT"
	strings:
		$1 = { 60 E9 DC 05 00 00 C1 9E B7 BB D9 2C 15 3D C6 E5 6D 01 D9 57 F4 71 1E 9D BA 98 04 3A 39 7A 1E 9D 3A 79 51 5A FD BB D9 25 55 34 96 E2 B7 CA 5E E6 BA BB D9 63 3D FB 8F E2 B7 BB D9 9C B7 48 5E 1D }
	condition:
		$1 at pe.entry_point
}

rule aspack_asprotect_uv_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack-ASPROTECT"
	strings:
		$1 = { 60 E9 F3 05 00 00 67 A0 22 E7 8F 31 7F 66 62 E9 94 A2 8F 1A 1E 51 CA A1 21 3A A4 3C A3 59 CA A1 A1 5A F7 1C 67 E7 8F 28 BF 9F 32 E4 22 E8 0A E8 21 E7 8F 66 A7 D8 39 E4 22 E7 8F A1 22 6A 0A 21 }
	condition:
		$1 at pe.entry_point
}
rule aspack_asprotect_uv_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPack-ASPROTECT"
	strings:
		$1 = { E8 01 00 00 00 EB 5D BB ?? FF FF FF 03 DD 81 EB 00 8A 0F 00 EB 02 EB 39 C6 45 10 00 33 C0 8B 73 3C FF 74 33 58 0F B7 54 33 06 4A 4A 8D BC 33 F8 00 00 00 8B 77 0C 8B 4F 10 0B C9 74 07 03 F3 }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 1)
}

rule aspr_stripper_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASPR Stripper"
		version = "2.x"
	strings:
		$1 = { BB ?? ?? ?? ?? E9 ?? ?? ?? ?? 60 9C FC BF ?? ?? ?? ?? B9 ?? ?? ?? ?? F3 AA 9D 61 C3 55 8B EC }
	condition:
		$1 at pe.entry_point
}

rule asprotect_ske_21_22_21x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect SKE"
		version = "2.1, 2.2, 2.1x"
	strings:
		$1 = { 90 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 ?? ?? ?? 80 7D 4D 01 75 0C 8B 74 24 28 83 FE 01 89 5D 4E 75 31 8D 45 53 50 53 FF B5 ED 09 00 00 8D 45 35 50 E9 82 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 2B 95 CD 3C 40 00 81 EA 2C 00 00 00 80 BD 08 3D 40 00 00 74 18 8B 85 ED 3C 40 00 03 85 F7 3C 40 00 3B ?? 74 01 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
	strings:
		$1 = { 60 ?? ?? ?? ?? ?? 90 5D ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 03 DD }
	condition:
		$1 at pe.entry_point
}

rule asprotect_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
	strings:
		$1 = { 68 ?? ?? ?? ?? E8 ?? ?? 00 00 C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_uv_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
	strings:
		$1 = { 90 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 ?? ?? 00 80 7D 4D 01 75 0C 8B 74 24 28 83 FE 01 89 5D 4E 75 31 8D 45 53 50 53 FF B5 DD 09 00 00 8D 45 35 50 E9 82 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_uv_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
	strings:
		$1 = { 5D 81 ED ?? ?? ?? 00 BB ?? ?? ?? 00 03 DD 2B 9D ?? ?? ?? 00 83 BD ?? ?? ?? 00 00 89 9D ?? ?? ?? 00 0F 85 ?? ?? 00 00 8D 85 ?? ?? ?? 00 50 FF 95 ?? ?? ?? 00 89 85 }
	condition:
		$1 in (pe.entry_point + 6 .. pe.entry_point + 7)
}

rule asprotect_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.0"
	strings:
		$1 = { 60 E8 01 ?? ?? ?? 90 5D 81 ED ?? ?? ?? ?? BB ?? ?? ?? ?? 03 DD 2B 9D }
	condition:
		$1 at pe.entry_point
}

rule asprotect_11_brs {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.1 BRS"
	strings:
		$1 = { 60 E9 ?? 05 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_11_mte {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.1 MTE"
	strings:
		$1 = { 60 E9 ?? ?? ?? ?? 91 78 79 79 79 E9 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_11b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.1b"
	strings:
		$1 = { 90 60 E9 ?? 04 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_11c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.1c"
	strings:
		$1 = { 90 60 E8 1B ?? ?? ?? E9 FC }
	condition:
		$1 at pe.entry_point
}

rule asprotect_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.1"
	strings:
		$1 = { 60 E9 ?? 04 ?? ?? E9 ?? ?? ?? ?? ?? ?? ?? EE }
	condition:
		$1 at pe.entry_point
}

rule asprotect_12_21 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.2 - 2.1"
	strings:
		$1 = { 68 01 ?? ?? ?? E8 01 ?? ?? ?? C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_12_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.2"
	strings:
		$1 = { 68 01 ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_12_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.2"
	strings:
		$1 = { 90 60 E8 1B 00 00 00 E9 FC 8D B5 0F 06 00 00 8B FE B9 97 00 00 00 AD 35 78 56 34 12 AB 49 75 F6 EB 04 5D 45 55 C3 E9 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_123_rc4_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.23 RC4"
	strings:
		$1 = { 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB 00 ?? ?? ?? 80 7D 4D 01 75 0C 8B 74 24 28 83 FE 01 89 5D 4E 75 31 8D 45 53 50 53 FF B5 D5 09 00 00 8D 45 35 50 E9 82 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule asprotect_123_rc4_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.23 RC4"
	strings:
		$1 = { 90 60 E8 03 00 00 00 E9 EB 04 5D 45 55 C3 E8 01 00 00 00 EB 5D BB ED FF FF FF 03 DD 81 EB ?? ?? ?? ?? 80 7D 4D 01 75 0C 8B 74 24 28 83 FE 01 89 5D 4E 75 31 8D 45 53 50 53 FF B5 D5 09 00 00 8D 45 35 50 E9 82 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 B8 F8 C0 A5 23 50 50 03 45 4E 5B 85 C0 74 1C EB 01 E8 81 FB F8 C0 A5 23 74 35 33 D2 56 6A 00 56 FF 75 4E FF D0 5E 83 FE 00 75 24 33 D2 8B 45 41 85 C0 74 07 52 52 FF 75 35 FF D0 8B 45 35 85 C0 74 0D }
	condition:
		$1 at pe.entry_point
}

rule asprotect_12x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ASProtect"
		version = "1.2x"
	strings:
		$1 = { 00 00 68 01 ?? ?? ?? C3 AA }
	condition:
		$1 at pe.entry_point
}

rule ass_crypter {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ass - crypter"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 ?? ?? ?? ?? 89 45 EC B8 98 40 00 10 E8 AC EA FF FF 33 C0 55 68 78 51 00 10 64 ?? ?? ?? ?? 20 6A 0A 68 88 51 00 10 A1 E0 97 00 10 50 E8 D8 EA FF FF 8B D8 53 A1 E0 97 00 10 50 E8 12 EB FF FF 8B F8 53 A1 E0 97 00 10 50 E8 DC EA FF FF 8B D8 53 E8 DC EA FF FF 8B F0 85 F6 74 26 8B D7 4A B8 F0 97 00 10 E8 C9 E7 FF FF B8 F0 97 00 10 E8 B7 E7 FF FF 8B CF 8B D6 E8 EE EA FF FF 53 E8 98 EA FF FF 8D 4D EC BA 9C 51 00 10 A1 F0 97 00 10 E8 22 EB FF FF 8B 55 EC B8 F0 97 00 10 E8 89 E6 FF FF B8 F0 97 00 10 E8 7F E7 FF FF E8 6E EC FF FF 33 C0 5A 59 59 64 89 10 68 7F 51 00 10 8D 45 EC E8 11 E6 FF FF C3 E9 FF DF FF FF EB F0 5F 5E 5B E8 0D E5 FF FF 00 53 45 54 54 49 4E 47 53 00 00 00 00 FF FF FF FF 1C 00 00 00 45 4E 54 45 52 20 59 4F 55 52 20 4F 57 4E 20 50 41 53 53 57 4F 52 44 20 48 45 52 45 }
	condition:
		$1 at pe.entry_point
}

rule avercryptor_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AverCryptor"
		version = "1.0"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 75 17 40 00 8B BD 9C 18 40 00 8B 8D A4 18 40 00 B8 BC 18 40 00 03 C5 80 30 05 83 F9 00 74 71 81 7F 1C AB 00 00 00 75 62 8B 57 0C 03 95 A0 18 40 00 33 C0 51 33 C9 66 B9 FA 00 66 83 F9 00 74 49 8B 57 0C 03 95 A0 18 40 00 8B 85 A8 18 40 00 83 F8 02 75 06 81 C2 00 02 00 00 51 8B 4F 10 83 F8 02 75 06 81 E9 00 02 00 00 57 BF C8 00 00 00 8B F1 E8 27 00 00 00 8B C8 5F B8 BC 18 40 00 03 C5 E8 24 00 00 00 59 49 EB B1 59 83 C7 28 49 EB 8A 8B 85 98 18 40 00 89 44 24 1C 61 FF E0 56 57 4F F7 D7 23 F7 8B C6 5F 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule avercryptor_102b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AverCryptor"
		version = "1.02b"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 0C 17 40 00 8B BD 33 18 40 00 8B 8D 3B 18 40 00 B8 51 18 40 00 03 C5 80 30 05 83 F9 00 74 71 81 7F 1C AB 00 00 00 75 62 8B 57 0C 03 95 37 18 40 00 33 C0 51 33 C9 66 B9 F7 00 66 83 F9 00 74 49 8B 57 0C 03 95 37 18 40 00 8B 85 3F 18 40 00 83 F8 02 75 06 81 C2 00 02 00 00 51 8B 4F 10 83 F8 02 75 06 81 E9 00 02 00 00 57 BF C8 00 00 00 8B F1 E8 27 00 00 00 8B C8 5F B8 51 18 40 00 03 C5 E8 24 00 00 00 59 49 EB B1 59 83 C7 28 49 EB 8A 8B 85 2F 18 40 00 89 44 24 1C 61 FF E0 56 57 4F F7 D7 23 F7 8B C6 5F 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule azpprotect_0001_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AZProtect"
		version = "0001"

	strings:
		$1 = { EB 70 FC 60 8C 80 4D 11 00 70 25 81 00 40 0D 91 BB 60 8C 80 4D 11 00 70 21 81 1D 61 0D 81 00 40 CE 60 8C 80 4D 11 00 70 25 81 25 81 25 81 25 81 29 61 41 81 31 61 1D 61 00 40 B7 30 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 60 BE 00 ?? ?? 00 BF 00 00 40 00 EB 17 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 00 00 00 00 FF 25 ?? ?? ?? 00 8B C6 03 C7 8B F8 57 55 8B EC 05 7F 00 00 00 50 E8 E5 FF FF FF BA 8C ?? ?? 00 89 02 E9 1A 01 00 00 ?? 00 00 00 47 65 74 4D 6F 64 75 6C 65 46 69 6C 65 4E 61 6D 65 41 00 47 65 74 56 6F 6C 75 6D 65 49 6E 66 6F 72 6D 61 74 69 6F 6E 41 00 4D 65 73 73 61 67 65 42 6F 78 41 00 45 78 69 74 50 72 6F 63 65 73 73 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 }
	condition:
		$1 at pe.entry_point
}

rule azpprotect_0001_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "AZProtect"
		version = "0001"
	strings:
		$1 = { FC 33 C9 49 8B D1 33 C0 33 DB AC 32 C1 8A CD 8A EA 8A D6 B6 08 66 D1 EB 66 D1 D8 73 09 66 35 20 83 66 81 F3 B8 ED FE CE 75 EB 33 C8 33 D3 4F 75 D5 F7 D2 F7 D1 8B C2 C1 C0 10 66 8B C1 C3 F0 DA 55 8B EC 53 56 33 C9 33 DB 8B 4D 0C 8B 55 10 8B 75 08 4E 4A 83 FB 08 72 05 33 DB 43 EB 01 43 33 C0 8A 04 31 8A 24 13 2A C4 88 04 31 E2 E6 5E 5B C9 C2 0C }
	condition:
		$1 at pe.entry_point
}

rule bambam_001 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BamBam"
		version = "0.01"
	strings:
		$1 = { 6A 14 E8 9A 05 00 00 8B D8 53 68 FB ?? ?? ?? E8 6C FD FF FF B9 05 00 00 00 8B F3 BF FB ?? ?? ?? 53 F3 A5 E8 8D 05 00 00 8B 3D 03 ?? ?? ?? A1 ?? ?? ?? ?? 66 8B 15 ?? ?? ?? ?? B9 ?? ?? ?? ?? 2B CF 89 45 E8 89 0D ?? ?? ?? ?? 66 89 55 EC 8B 41 3C 33 D2 03 C1 }
	condition:
		$1 at pe.entry_point
}

rule bambvam_004 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BamBam"
		version = "0.04"
	strings:
		$1 = { BF ?? ?? ?? ?? 83 C9 FF 33 C0 68 ?? ?? ?? ?? F2 AE F7 D1 49 51 68 ?? ?? ?? ?? E8 11 0A 00 00 83 C4 0C 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B F0 BF ?? ?? ?? ?? 83 C9 FF 33 C0 F2 AE F7 D1 49 BF ?? ?? ?? ?? 8B D1 68 ?? ?? ?? ?? C1 E9 02 F3 AB 8B CA 83 E1 03 F3 AA BF ?? ?? ?? ?? 83 C9 FF 33 C0 F2 AE F7 D1 49 51 68 ?? ?? ?? ?? E8 C0 09 00 00 }
	condition:
		$1 at pe.entry_point
}

rule beria_007 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Beria"
		version = "0.07 public WIP"
	strings:
		$1 = { 83 EC 18 53 8B 1D 00 30 ?? ?? 55 56 57 68 30 07 00 00 33 ED 55 FF D3 8B F0 3B F5 74 0D 89 AE 20 07 00 00 E8 88 0F 00 00 EB 02 33 F6 6A 10 55 89 35 30 40 ?? ?? FF D3 8B F0 3B F5 74 09 89 2E E8 3C FE FF FF EB 02 33 F6 6A 18 55 89 35 D8 43 ?? ?? FF D3 8B F0 3B F5 74 37 8B 46 0C 3B C5 8B 3D 04 30 ?? ?? 89 2E 89 6E 04 89 6E 08 74 06 50 FF D7 89 6E 0C 8B 46 10 3B C5 74 06 50 FF D7 89 6E 10 8B 46 14 3B C5 74 0A 50 FF D7 89 6E 14 EB 02 33 F6 6A 10 55 89 35 A4 40 ?? ?? FF D3 8B F0 3B F5 74 09 E8 08 12 00 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00"
	strings:
		$1 = { BA ?? ?? ?? ?? 8D B2 ?? ?? ?? ?? 8B 46 ?? 85 C0 74 51 03 C2 8B 7E ?? 8B 1E 85 DB 75 02 8B DF 03 DA 03 FA 52 57 50 FF 15 ?? ?? ?? ?? 5F 5A 85 C0 74 2F 8B C8 8B 03 85 C0 74 22 0F BA F0 1F 72 04 8D 44 ?? ?? 51 52 57 50 51 FF 15 ?? ?? ?? ?? 5F 5A 59 85 C0 74 0B AB 83 C3 04 EB D8 83 C6 14 EB AA 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100_lzbrr_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00 [LZBRR]"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? BF ?? ?? ?? ?? FC B2 80 33 DB A4 B3 02 E8 ?? ?? ?? ?? 73 F6 33 C9 E8 ?? ?? ?? ?? 73 1C 33 C0 E8 ?? ?? ?? ?? 73 23 B3 02 41 B0 10 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100_lzbrr_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00 [LZBRR]"
	strings:
		$1 = { 83 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? BF ?? ?? ?? ?? FC B2 80 33 DB A4 B3 02 E8 ?? ?? ?? ?? 73 F6 33 C9 E8 ?? ?? ?? ?? 73 1C 33 C0 E8 ?? ?? ?? ?? 73 23 B3 02 41 B0 10 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100_lzbrs_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00 [LZBRS]"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? BF ?? ?? ?? ?? FC AD 8D 1C 07 B0 80 3B FB 73 3B E8 ?? ?? ?? ?? 72 03 A4 EB F2 E8 ?? ?? ?? ?? 8D 51 FF E8 ?? ?? ?? ?? 56 8B F7 2B F2 F3 A4 5E EB DB 02 C0 75 03 AC 12 C0 C3 33 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100_lzbrs_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00 [LZBRS]"
	strings:
		$1 = { 83 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? BF ?? ?? ?? ?? FC AD 8D 1C 07 B0 80 3B FB 73 3B E8 ?? ?? ?? ?? 72 03 A4 EB F2 E8 ?? ?? ?? ?? 8D 51 FF E8 ?? ?? ?? ?? 56 8B F7 2B F2 F3 A4 5E EB DB 02 C0 75 03 AC 12 C0 C3 33 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100_lzma_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00 [LZMA]"
	strings:
		$1 = { 60 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? BE ?? ?? ?? ?? B9 04 00 00 00 8B F9 81 FE ?? ?? ?? ?? 7F 10 AC 47 04 18 2C 02 73 F0 29 3E 03 F1 03 F9 EB E8 }
	condition:
		$1 at pe.entry_point
}

rule beroexepacker_100_lzma_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
		version = "1.00 [LZMA] DLL"
	strings:
		$1 = { 83 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? BE ?? ?? ?? ?? B9 ?? ?? ?? ?? 8B F9 81 FE ?? ?? ?? ?? 7F 10 AC 47 04 18 2C 02 73 F0 29 3E 03 F1 03 F9 EB E8 }
	condition:
		$1 at pe.entry_point
}

private rule beroexepacker_uv_prologue {
	strings:
		$1 = { 60 FC B9 ?? ?? ?? ?? BE ?? ?? ?? ?? BF ?? ?? ?? ?? F3 A4 }
		$2 = { 60 E8 00 00 00 00 }
		$3 = { 60 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? }
		$4 = { 60 C8 94 0C 00 60 }
	condition:
		for any of them : ( $ in (pe.entry_point .. pe.entry_point + 11))
}

rule beroexepacker_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BeRoEXEPacker"
	strings:
		 $1 = { 5E 81 C6 A7 00 00 00 BF ?? ?? ?? ?? 57 FC B2 80 33 DB A4 B3 02 E8 71 00 00 00 73 F6 33 C9 E8 68 00 00 00 73 1C 33 C0 E8 5F 00 00 00 73 23 B3 02 41 B0 10 E8 53 00 00 00 12 C0 73 }
		 $2 = { B9 04 00 00 00 2B CE 81 FE ?? ?? ?? ?? 77 1E AC 04 18 2C 02 72 0D 3C 25 75 ED 8A 06 24 F0 3C 80 75 E5 46 8D 3C 0E 29 3E 83 C6 04 EB DA }
		 $3 = { AD 89 45 FC 33 C0 F7 D0 89 45 F8 F7 D0 B4 08 B9 23 03 00 00 8D BD 6C F3 FF FF F3 AB BF ?? ?? ?? ?? E9 AC 00 00 00 }
		 $4 = { B9 ?? ?? ?? ?? BB ?? ?? ?? ?? BE ?? ?? ?? ?? 8B FB FC F3 A4 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 53 B8 ?? ?? ?? ?? FF D0 }
	condition:
		beroexepacker_uv_prologue and 1 of them
}

rule blackenergy_ddos_bot_crypter {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BlackEnergy DDoS Bot Crypter"
	strings:
		$1 = { 55 ?? ?? 81 EC 1C 01 00 00 53 56 57 6A 04 BE 00 30 00 00 56 FF 35 00 20 11 13 6A 00 E8 ?? 03 00 00 ?? ?? 83 C4 10 ?? FF 89 7D F4 0F }
	condition:
		$1 at pe.entry_point
}

rule blade_joiner_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Blade Joiner"
		version = "1.5"
	strings:
		$1 = { 55 8B EC 81 C4 E4 FE FF FF 53 56 57 33 C0 89 45 F0 89 85 }
	condition:
		$1 at pe.entry_point
}

rule berio_100b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Berio"
		version = "1.00b"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 E9 01 12 00 }
	condition:
		$1 at pe.entry_point
}

rule berio200b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Berio"
		version = "2.00b"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 E9 01 74 01 }
	condition:
		$1 at pe.entry_point
}

rule blindspot_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BlindSpot"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 81 EC 50 02 00 00 8D 85 B0 FE FF FF 53 56 A3 90 12 40 00 57 8D 85 B0 FD FF FF 68 00 01 00 00 33 F6 50 56 FF 15 24 10 40 00 56 68 80 00 00 00 6A 03 56 56 8D 85 B0 FD FF FF 68 00 00 00 80 50 FF 15 20 10 40 00 56 56 68 00 08 00 00 50 89 45 FC FF 15 1C 10 40 00 8D 45 F8 8B 1D 18 10 40 00 56 50 6A 34 FF 35 90 12 40 00 FF 75 FC FF D3 85 C0 0F 84 7F 01 00 00 39 75 F8 0F 84 76 01 00 00 A1 90 12 40 00 66 8B 40 30 66 3D 01 00 75 14 8D 85 E4 FE FF FF 68 04 01 00 00 50 FF 15 14 10 40 00 EB 2C 66 3D 02 00 75 14 8D 85 E4 FE FF FF 50 68 04 01 00 00 FF 15 10 10 40 00 EB 12 8D 85 E4 FE FF FF 68 04 01 00 00 50 FF 15 0C 10 40 00 8B 3D 08 10 40 00 8D 85 E4 FE FF FF 68 54 10 40 00 50 }
	condition:
		$1 at pe.entry_point
}

rule bobpack_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BobPack"
		version = "1.00"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 0C 24 89 CD 83 E9 06 81 ED ?? ?? ?? ?? E8 3D 00 00 00 89 85 ?? ?? ?? ?? 89 C2 B8 5D 0A 00 00 8D 04 08 E8 E4 00 00 00 8B 70 04 01 D6 E8 76 00 00 00 E8 51 01 00 00 E8 01 01 }
	condition:
		$1 at pe.entry_point
}

rule bobcrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BopCrypt"
		version = "1.0"
	strings:
		$1 = { 60 BD ?? ?? ?? ?? E8 ?? ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_precompiled_header {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Borland precompiled header file"
	strings:
		$1 = { 54 50 53 }
	condition:
		$1 at pe.entry_point
}

rule ci_crypt_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "C.I Crypt"
		version = "0.1"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ci_crypt_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "C.I Crypt"
		version = "0.2"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule cd_cops_ii {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CD-Cops"
		version = "II"
	strings:
		$1 = { 53 60 BD ?? ?? ?? ?? 8D 45 ?? 8D 5D ?? E8 ?? ?? ?? ?? 8D }
	condition:
		$1 at pe.entry_point
}

rule cds_ss_10b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CDS SS"
		version = "1.0b1"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED CA 47 40 00 FF 74 24 20 E8 D3 03 00 00 0B C0 0F 84 13 03 00 00 89 85 B8 4E 40 00 66 8C D8 A8 04 74 0C C7 85 8C 4E 40 00 01 00 00 00 EB 12 64 A1 30 00 00 00 0F B6 40 02 0A C0 0F 85 E8 02 00 00 8D 85 F6 4C 40 00 50 FF B5 B8 4E 40 00 E8 FC 03 00 00 0B C0 0F 84 CE 02 00 00 E8 1E 03 00 00 89 85 90 4E 40 00 8D 85 03 4D 40 00 50 FF B5 B8 }
	condition:
		$1 at pe.entry_point
}

rule celsius_crypt_21_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Celsius Crypt"
		version = "2.1"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 01 00 00 00 FF 15 84 92 44 00 E8 C8 FE FF FF 90 8D B4 26 00 00 00 00 55 89 E5 83 EC 08 C7 04 24 02 00 00 00 FF 15 84 92 44 00 E8 A8 FE FF FF 90 8D B4 26 00 00 00 00 55 8B 0D C4 92 44 00 89 E5 5D FF E1 8D 74 26 00 55 8B 0D AC 92 44 00 89 E5 5D FF E1 90 90 90 90 55 89 E5 5D E9 77 C2 00 00 90 90 90 90 90 90 90 55 89 E5 83 EC 28 8B 45 10 89 04 24 E8 3F 14 01 00 48 89 45 FC 8B 45 0C 48 89 45 F4 8D 45 F4 89 44 24 04 8D 45 FC 89 04 24 E8 12 A3 03 00 8B 00 89 45 F8 8B 45 FC 89 45 F0 C6 45 EF 01 C7 45 E8 00 00 00 00 8B 45 E8 3B 45 F8 73 39 80 7D EF 00 74 33 8B 45 F0 89 44 24 04 8B 45 10 89 04 24 E8 1C 1A 01 00 89 C1 8B 45 08 8B 55 E8 01 C2 0F B6 01 3A 02 0F 94 C0 88 45 EF 8D 45 F0 FF 08 8D 45 E8 FF 00 EB BF 83 7D F0 00 74 34 80 7D EF 00 74 2E 8B 45 F0 89 44 24 04 8B 45 10 89 04 24 E8 DD 19 01 00 89 C1 8B 45 08 8B 55 F8 01 C2 0F B6 01 3A 02 0F 94 C0 88 45 EF 8D 45 F0 FF 08 EB C6 C7 44 24 04 00 00 00 00 8B 45 10 89 04 24 E8 AE 19 01 00 89 C1 8B 45 08 8B 55 F8 01 C2 0F B6 01 3A 02 7F 0C 0F B6 45 EF 83 E0 01 88 45 E7 EB 04 C6 45 E7 00 0F B6 45 E7 88 45 EF 0F B6 45 EF C9 C3 }
	condition:
		$1 at pe.entry_point
}

rule celsius_crypt_21_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Celsius Crypt"
		version = "2.1"
	strings:
		$1 = { 55 89 E5 83 EC 28 8B 45 10 89 04 24 E8 3F 14 01 00 48 89 45 FC 8B 45 0C 48 89 45 F4 8D 45 F4 89 44 24 04 8D 45 FC 89 04 24 E8 12 A3 03 00 8B 00 89 45 F8 8B 45 FC 89 45 F0 C6 45 EF 01 C7 45 E8 00 00 00 00 8B 45 E8 3B 45 F8 73 39 80 7D EF 00 74 33 8B 45 F0 89 44 24 04 8B 45 10 89 04 24 E8 1C 1A 01 00 89 C1 8B 45 08 8B 55 E8 01 C2 0F B6 01 3A 02 0F 94 C0 88 45 EF 8D 45 F0 FF 08 8D 45 E8 FF 00 EB BF 83 7D F0 00 74 34 80 7D EF 00 74 2E 8B 45 F0 89 44 24 04 8B 45 10 89 04 24 E8 DD 19 01 00 89 C1 8B 45 08 8B 55 F8 01 C2 0F B6 01 3A 02 0F 94 C0 88 45 EF 8D 45 F0 FF 08 EB C6 C7 44 24 04 00 00 00 00 8B 45 10 89 04 24 E8 AE 19 01 00 89 C1 8B 45 08 8B 55 F8 01 C2 0F B6 01 3A 02 7F 0C 0F B6 45 EF 83 E0 01 88 45 E7 EB 04 C6 45 E7 00 0F B6 45 E7 88 45 EF 0F B6 45 EF C9 C3 }
	condition:
		$1 at pe.entry_point
}

rule cexe_10a_10b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CExe"
		version = "1.0a, 1.0b"
	strings:
		$1 = { 55 8B EC 81 EC 0C 02 ?? ?? 56 BE 04 01 ?? ?? 8D 85 F8 FE FF FF 56 50 6A ?? FF 15 ?? 10 40 00 8A 8D F8 FE FF FF 33 D2 84 C9 8D 85 F8 FE FF FF 74 }
	condition:
		$1 at pe.entry_point
}

rule checkprg {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CHECKPRG"
	strings:
		$1 = { 33 C0 BE ?? ?? 8B D8 B9 ?? ?? BF ?? ?? BA ?? ?? 47 4A 74 }
	condition:
		$1 at pe.entry_point
}

rule chinaprotect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ChinaProtect"
	strings:
		$1 = { C3 E8 ?? ?? ?? ?? B9 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 30 C3 B9 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 30 C3 B9 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 30 C3 B9 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 30 C3 56 8B ?? ?? ?? 6A 40 68 00 10 00 00 8D ?? ?? 50 6A 00 E8 ?? ?? ?? ?? 89 30 83 C0 04 5E C3 8B 44 ?? ?? 56 8D ?? ?? 68 00 40 00 00 FF 36 56 E8 ?? ?? ?? ?? 68 00 80 00 00 6A 00 56 E8 ?? ?? ?? ?? 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule cicompress {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CICompress"
		version = "1.0"
	strings:
		$1 = { 6A 04 68 00 10 00 00 FF 35 9C 14 40 00 6A 00 FF 15 38 10 40 00 A3 FC 10 40 00 97 BE 00 20 40 00 E8 71 00 00 00 3B 05 9C 14 40 00 75 61 6A 00 6A 20 6A 02 6A 00 6A 03 68 00 00 00 C0 68 94 10 40 00 FF 15 2C 10 40 00 A3 F8 10 40 00 6A 00 68 F4 10 40 00 FF 35 9C 14 40 00 FF 35 FC 10 40 00 FF 35 F8 10 40 00 FF 15 34 10 40 00 FF 35 F8 10 40 00 FF 15 30 10 40 00 68 00 40 00 00 FF 35 9C 14 40 00 FF 35 FC 10 40 00 FF 15 3C 10 40 00 6A 00 FF 15 28 10 40 00 60 33 DB 33 C9 E8 7F 00 00 00 73 0A B1 08 E8 82 00 }
	condition:
		$1 at pe.entry_point
}

rule code_virtualizer_1310 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Code Virtualizer"
		version = "1.3.1.0"
	strings:
		$1 = { 60 9C FC E8 00 00 00 00 5F 81 EF ?? ?? ?? ?? 8B C7 81 C7 ?? ?? ?? ?? 3B 47 2C 75 02 EB 2E 89 47 2C B9 A7 00 00 00 EB 05 01 44 8F ?? 49 0B C9 75 F7 83 7F 40 00 74 15 8B 77 40 03 F0 EB 09 8B 1E 03 D8 01 03 83 C6 04 83 3E 00 75 F2 8B 74 24 24 8B DE 03 F0 B9 01 00 00 00 33 C0 F0 0F B1 4F 30 75 F7 AC }
	condition:
		$1 at pe.entry_point
}

rule codelock {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Code-Lock"
	strings:
		$1 = { 43 4F 44 45 2D 4C 4F 43 4B 2E 4F 43 58 00 }
	condition:
		$1 at pe.entry_point
}

rule codecrypt_014b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CodeCrypt"
		version = "0.14b"
	strings:
		$1 = { E9 C5 02 00 00 EB 02 83 3D 58 EB 02 FF 1D 5B EB 02 0F C7 5F }
	condition:
		$1 at pe.entry_point
}

rule codecrypt_015b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CodeCrypt"
		version = "0.15b"
	strings:
		$1 = { E9 31 03 00 00 EB 02 83 3D 58 EB 02 FF 1D 5B EB 02 0F C7 5F }
	condition:
		$1 at pe.entry_point
}

rule codecrypt_016_0164 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CodeCrypt"
		version = "0.16 - 0.164"
	strings:
		$1 = { E9 2E 03 00 00 EB 02 83 3D 58 EB 02 FF 1D 5B EB 02 0F C7 5F }
	condition:
		$1 at pe.entry_point
}

rule codecrypter_031 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "codeCrypter"
		version = "0.31"
	strings:
		$1 = { 50 58 53 5B 90 BB ?? ?? ?? 00 FF E3 90 CC CC CC 55 8B EC 5D C3 CC CC CC CC CC CC CC CC CC CC CC }
	condition:
		$1 at pe.entry_point
}

rule bitshape_pe_crypt_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "BitShape PE Crypt"
		version = "1.5"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED ?? ?? ?? ?? B9 7B 09 00 00 8D BD ?? ?? ?? ?? 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule codesafe_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CodeSafe"
		version = "2.0"
		start = 23
	strings:
		$1 = { ?8 3E C1 05 35 65 7E 8C 40 10 0? }
	condition:
		$1 at pe.entry_point + 23
}

rule codeveil_12_13 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CodeVeil"
		version = "1.2 - 1.3"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8D 64 24 00 55 8B EC 53 56 57 8B 4D 10 83 81 B8 00 00 00 05 83 A1 C0 00 00 00 DF 33 C0 5F 5E 5B C9 C3 8B FF 60 E8 01 00 00 00 B8 5E E8 01 00 00 00 B8 58 2D 31 01 00 00 8B 00 2B F0 81 E6 00 00 FF FF 03 76 3C 33 C9 66 8B 4E 14 8D 74 31 18 8B 5E 0C 03 DE 81 E3 00 F0 FF FF 8B 56 08 E8 05 00 00 00 E9 ?? 00 00 00 55 8B EC 83 C4 F0 B9 E9 00 00 00 8B F3 03 DA E8 01 00 00 00 B8 58 2D 77 01 00 00 8B 00 03 C6 89 45 F4 E8 01 00 00 00 B8 5A 81 EA 86 }
	condition:
		$1 at pe.entry_point
}

rule copy_prtector_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Copy Protector"
		version = "2.0"
	strings:
		$1 = { 2E A2 ?? ?? 53 51 52 1E 06 B4 ?? 1E 0E 1F BA ?? ?? CD 21 1F }
	condition:
		$1 at pe.entry_point
}

rule copycontrol_303 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CopyControl"
		version = "3.03"
	strings:
		$1 = { CC 90 90 EB 0B 01 50 51 52 53 54 61 33 61 2D 35 CA D1 07 52 D1 A1 3C }
	condition:
		$1 at pe.entry_point
}

rule copyminder {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CopyMinder"
	strings:
		$1 = { 83 25 ?? ?? ?? ?? EF 6A 00 E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? CC FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 ?? ?? ?? ?? FF 25 }
	condition:
		$1 at pe.entry_point
}

rule cpav {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CPAV"
	strings:
		$1 = { E8 ?? ?? 4D 5A B1 01 93 01 00 00 02 }
	condition:
		$1 at pe.entry_point
}

rule crinkler_01_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crinkler"
		version = "0.1 - 0.2"
	strings:
		$1 = { B9 ?? ?? ?? ?? 01 C0 68 ?? ?? ?? ?? 6A 00 58 50 6A 00 5F 48 5D BB 03 00 00 00 BE ?? ?? ?? ?? E9 }
	condition:
		$1 at pe.entry_point
}

rule crinkler_03_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crinkler"
		version = "0.3 - 0.4"
	strings:
		$1 = { B8 00 00 42 00 31 DB 43 EB 58 }
	condition:
		$1 at pe.entry_point
}

rule crunch_5fusion4 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crunch"
		version = "5 Fusion 4"
	strings:
		$1 = { EB 15 03 ?? ?? ?? 06 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 68 ?? ?? ?? ?? 55 E8 }
	condition:
		$1 at pe.entry_point
}

rule crunch_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crunch"
		version = "1.0"
	strings:
		$1 = { 55 E8 ?? ?? ?? ?? 5D 83 ED 06 8B C5 55 60 89 AD ?? ?? ?? ?? 2B 85 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? 80 BD ?? ?? ?? ?? ?? 75 09 C6 85 }
	condition:
		$1 at pe.entry_point
}

rule crunch_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crunch"
		version = "2.0"
	strings:
		$1 = { 55 E8 ?? ?? ?? ?? 5D 83 ED 06 8B C5 55 60 89 AD ?? ?? ?? ?? 2B 85 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? 55 BB ?? ?? ?? ?? 03 DD 53 64 67 FF 36 ?? ?? 64 67 89 26 }
	condition:
		$1 at pe.entry_point
}

rule crunch_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crunch"
		version = "3.0"
	strings:
		$1 = { EB 10 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 E8 ?? ?? ?? ?? 5D 81 ED 18 ?? ?? ?? 8B C5 55 60 9C 2B 85 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? FF 74 }
	condition:
		$1 at pe.entry_point
}

rule crunch_40 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crunch"
		version = "4.0"
	strings:
		$1 = { EB 10 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 E8 ?? ?? ?? ?? 5D 81 ED 18 ?? ?? ?? 8B C5 55 60 9C 2B 85 E9 06 ?? ?? 89 85 E1 06 ?? ?? FF 74 24 2C E8 BB 01 00 00 0F 82 92 05 00 00 E8 F1 03 00 00 49 0F 88 86 05 00 00 68 6C D9 B2 96 33 C0 50 E8 24 }
	condition:
		$1 at pe.entry_point
}

rule crunch_50 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crunch"
		version = "5.0"
	strings:
		$1 = { EB 15 03 00 00 00 06 00 00 00 00 00 00 00 00 00 00 00 68 00 00 00 00 55 E8 00 00 00 00 5D 81 ED 1D 00 00 00 8B C5 55 60 9C 2B 85 FC 07 00 00 89 85 E8 07 00 00 FF 74 24 2C E8 20 02 00 00 0F 82 94 06 00 00 E8 F3 04 00 00 49 0F 88 88 06 00 00 8B B5 E8 07 00 }
	condition:
		$1 at pe.entry_point
}

rule cruncher_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Cruncher"
		version = "1.0"
	strings:
		$1 = { 2E ?? ?? ?? ?? 2E ?? ?? ?? B4 30 CD 21 3C 03 73 ?? BB ?? ?? 8E DB 8D ?? ?? ?? B4 09 CD 21 06 33 C0 50 CB }
	condition:
		$1 at pe.entry_point
}

rule dirty_cryptor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DirTy CrYpt0r"
	strings:
		$1 = { B8 ?? ?? ?? ?? 32 DB FE C3 30 18 40 3D ?? ?? ?? ?? 7E ?? 68 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule crypkey_5x_6x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CrypKey"
		version = "5.x - 6.x"
	strings:
		$1 = { E8 ?? ?? ?? ?? 58 83 E8 05 50 5F 57 8B F7 81 EF ?? ?? ?? ?? 83 C6 39 BA ?? ?? ?? ?? 8B DF B9 0B ?? ?? ?? 8B 06 }
	condition:
		$1 at pe.entry_point
}

rule crypkey_56x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CrypKey"
		version = "5.6.x"
	strings:
		$1 = { 8B 1D ?? ?? ?? ?? 83 FB 00 75 0A E8 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule crypkey_56x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CrypKey"
		version = "5.6.x"
	strings:
		$1 = { E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 F8 00 75 07 6A 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule crypkey_61x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CrypKey"
		version = "6.1.x"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? 00 75 34 68 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule crypter_31 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crypter"
		version = "3.1"
	strings:
		$1 = { 68 FF 64 24 F0 68 58 58 58 58 FF D4 50 8B 40 F2 05 B0 95 F6 95 0F 85 01 81 BB FF 68 }
	condition:
		$1 at pe.entry_point
}

rule cryptic_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Cryptic"
		version = "2.0"
	strings:
		$1 = { B8 00 00 40 00 BB ?? ?? ?? 00 B9 00 10 00 00 BA ?? ?? ?? 00 03 D8 03 C8 03 D1 3B CA 74 06 80 31 ?? 41 EB F6 FF E3 }
	condition:
		$1 at pe.entry_point
}

rule cryptolock_202 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Crypto-Lock"
		version = "2.02"
	strings:
		$1 = { 60 BE ?? 90 40 00 8D BE ?? ?? FF FF 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 }
	condition:
		$1 at pe.entry_point
}

rule cryptocracks_pe_protector_092 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CRYPToCRACk's PE Protector"
		version = "0.9.2"
	strings:
		$1 = { E8 01 00 00 00 E8 58 5B 81 E3 00 FF FF FF 66 81 3B 4D 5A 75 37 84 DB 75 33 8B F3 03 ?? ?? 81 3E 50 45 00 00 75 26 }
	condition:
		$1 at pe.entry_point
}

rule cryptocracks_pe_protector_093 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CRYPToCRACk's PE Protector"
		version = "0.9.3"
	strings:
		$1 = { 5B 81 E3 00 FF FF FF 66 81 3B 4D 5A 75 33 8B F3 03 73 3C 81 3E 50 45 00 00 75 26 0F B7 46 18 8B C8 69 C0 AD 0B 00 00 F7 E0 2D AB 5D 41 4B 69 C9 DE C0 00 00 03 C1 }
	condition:
		$1 at pe.entry_point
}

rule crypwrap {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "CrypWrap"
	strings:
		$1 = { E8 B8 ?? ?? ?? E8 90 02 ?? ?? 83 F8 ?? 75 07 6A ?? E8 ?? ?? ?? ?? FF 15 49 8F 40 ?? A9 ?? ?? ?? 80 74 0E }
	condition:
		$1 at pe.entry_point
}

rule cygwin32 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Cygwin32"
	strings:
		$1 = { 55 89 E5 83 EC 04 83 3D }
	condition:
		$1 at pe.entry_point
}

rule d1s1g_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "D1NS1G"
	strings:
		$1 = { 18 37 00 00 00 00 00 00 01 00 0A 00 00 00 18 00 00 80 00 00 00 00 ?? ?? 18 37 00 00 00 00 02 00 00 00 88 00 00 80 38 00 00 80 96 00 00 80 50 00 00 80 00 00 00 00 ?? ?? 18 37 00 00 00 00 00 00 01 00 00 00 00 00 68 00 00 00 00 00 00 00 ?? ?? 18 37 00 00 00 00 00 00 01 00 00 00 00 00 78 00 00 00 B0 F0 00 00 10 00 00 00 00 00 00 00 00 00 00 00 C0 F0 00 00 60 00 00 00 00 00 00 00 00 00 00 00 06 00 44 00 56 00 43 00 4C 00 41 00 4C 00 0B 00 50 00 41 00 43 00 4B 00 41 00 47 00 45 00 49 00 4E 00 46 00 4F 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule d1s1g_11b_scrambled_exe {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "D1S1G"
		version = "1.1b scrambled EXE"
	strings:
		$1 = { E8 07 00 00 00 E8 1E 00 00 00 C3 90 58 89 C2 89 C2 25 00 F0 FF FF 50 83 C0 55 8D 00 FF 30 8D 40 04 FF 30 52 C3 8D 40 00 55 8B EC 83 C4 E8 53 56 57 8B 4D 10 8B 45 08 89 45 F8 8B 45 0C 89 45 F4 8D 41 61 8B 38 8D 41 65 8B 00 03 C7 89 45 FC 8D 41 69 8B 00 03 C7 8D 51 6D 8B 12 03 D7 83 C1 71 8B 09 03 CF 2B CA 72 0A 41 87 D1 80 31 FF 41 4A 75 F9 89 45 F0 EB 71 8B }
	condition:
		$1 at pe.entry_point
}

rule d1s1g_11b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "D1S1G"
		version = "1.1b"
	strings:
		$1 = { 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 01 00 0A 00 00 00 18 00 00 80 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 02 00 00 00 88 00 00 80 38 00 00 80 96 00 00 80 50 00 00 80 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 01 00 00 00 00 00 68 00 00 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 01 00 00 00 00 00 78 00 00 00 B0 ?? ?? 00 10 00 00 00 00 00 00 00 00 00 00 00 C0 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 06 00 44 00 56 00 43 00 4C 00 41 00 4C 00 0B 00 50 00 41 00 43 00 4B 00 41 00 47 00 45 00 49 00 4E 00 46 00 4F 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule daemon_protect {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DAEMON Protect"
		version = "0.6.7"
	strings:
		$1 = { 60 60 9C 8C C9 32 C9 E3 0C 52 0F 01 4C 24 FE 5A 83 C2 0C 8B 1A 9D 61 }
	condition:
		$1 at pe.entry_point
}

rule dalkrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DalKrypt"
		version = "1.0"
	strings:
		$1 = { 68 00 10 40 00 58 68 ?? ?? ?? 00 5F 33 DB EB 0D 8A 14 03 80 EA 07 80 F2 04 88 14 03 43 81 FB ?? ?? ?? 00 72 EB FF E7 }
	condition:
		$1 at pe.entry_point
}

rule dcrypt_private {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DCrypt Private"
		version = "0.9b"
	strings:
		$1 = { B9 ?? ?? ?? 00 E8 00 00 00 00 58 68 ?? ?? ?? 00 83 E8 0B 0F 18 00 D0 00 48 E2 FB C3 }
	condition:
		$1 at pe.entry_point
}

rule def_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DEF"
		version = "1.0"
	strings:
		$1 = { BE ?? 01 40 00 6A ?? 59 80 7E 07 00 74 11 8B 46 }
	condition:
		$1 at pe.entry_point
}

rule def_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DEF"
		version = "1.0"
	strings:
		$1 = { BE ?? ?? 40 00 6A ?? 59 80 7E 07 00 74 11 8B 46 0C 05 00 00 40 00 8B 56 10 30 10 40 4A 75 FA 83 C6 28 E2 E4 68 ?? ?? 40 00 C3 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule diamondcs {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DIAMONDCS"
	strings:
		$1 = { 60 EB 0A 44 69 61 6D 6F 6E 64 43 53 00 EB 02 EB 05 E8 F9 FF FF FF 58 2D 13 00 00 00 F2 EB 02 85 }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_007 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "0.07"
	strings:
		$1 = { 55 57 56 52 51 53 E8 00 00 00 00 5D 8B D5 81 ED 23 35 40 00 }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_08 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "0.8 Phantasm"
	strings:
		$1 = { 55 57 56 52 51 53 E8 00 00 00 00 5D 8B D5 81 ED 0D 39 40 00 }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_10_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "1.0, 1.1 Phantasm"
	strings:
		$1 = { 55 57 56 52 51 53 66 81 C3 EB 02 EB FC 66 81 C3 EB 02 EB FC }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_15b3 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "1.5b3 Phantasm"
	strings:
		$1 = { 9C 55 57 56 52 51 53 9C FA E8 ?? ?? ?? ?? 5D 81 ED 5B 53 40 ?? B0 }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_210_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "2.10"
	strings:
		$1 = { 9C 6A 10 73 0B EB 02 C1 51 E8 06 ?? ?? ?? C4 11 73 F7 5B CD 83 C4 04 EB 02 99 EB FF 0C 24 71 01 E8 79 E0 7A 01 75 83 C4 04 9D EB 01 75 68 5F 20 40 ?? E8 B0 EF FF FF 72 03 73 01 75 BE }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_210_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "2.10"
	strings:
		$1 = { EB 20 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9C 55 57 56 52 51 53 9C E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? EB 58 75 73 65 72 33 32 2E 64 6C 6C ?? 4D 65 73 73 61 67 65 42 6F 78 41 ?? 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C ?? 53 6C 65 65 70 ?? 47 65 74 54 69 63 6B 43 6F 75 6E 74 }
	condition:
		$1 at pe.entry_point
}

rule ding_boys_pe_lock_233 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ding Boy's PE-lock"
		version = "2.33"
	strings:
		$1 = { EB 20 ?? ?? 40 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9C 55 57 56 52 51 53 9C E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? 9C 6A 10 73 0B EB 02 C1 51 E8 06 ?? ?? ?? C4 11 73 F7 5B CD 83 C4 04 EB 02 99 EB FF 0C 24 71 01 E8 79 E0 7A 01 75 83 }
	condition:
		$1 at pe.entry_point
}

rule dipacker_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "diPacker"
		version = "1.x"
	strings:
		$1 = { 0F 00 2D E9 01 00 A0 E3 68 01 00 EB 8C 00 00 EB 2B 00 00 EB 00 00 20 E0 1C 10 8F E2 8E 20 8F E2 00 30 A0 E3 67 01 00 EB 0F 00 BD E8 00 C0 8F E2 00 F0 9C E5 }
	condition:
		$1 at pe.entry_point
}

rule diprotector_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "diProtector"
		version = "1.x"
	strings:
		$1 = { 01 00 A0 E3 14 00 00 EB 00 00 20 E0 44 10 9F E5 03 2A A0 E3 40 30 A0 E3 AE 00 00 EB 30 00 8F E5 00 20 A0 E1 3A 0E 8F E2 00 00 80 E2 1C 10 9F E5 20 30 8F E2 0E 00 00 EB 14 00 9F E5 14 10 9F E5 7F 20 A0 E3 C5 00 00 EB 04 C0 8F E2 00 F0 9C E5 }
	condition:
		$1 at pe.entry_point
}

rule djoin_07_rc4 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DJoin"
		version = "0.7 [RC4]"
	strings:
		$1 = { C6 05 ?? ?? 40 00 00 C6 05 ?? ?? 40 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule djoin_07_xor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DJoin"
		version = "0.7 [XOR]"
	strings:
		$1 = { C6 05 ?? ?? 40 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule dotfix_nice_protect_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DotFix Nice Protect"
	strings:
		$1 = { 60 E8 55 00 00 00 8D BD 00 10 40 00 68 ?? ?? ?? 00 03 3C 24 8B F7 90 68 31 10 40 00 9B DB E3 55 DB 04 24 8B C7 DB 44 24 04 DE C1 DB 1C 24 8B 1C 24 66 AD 51 DB 04 24 90 90 DA 8D 77 10 40 00 DB 1C 24 D1 E1 29 }
	condition:
		$1 at pe.entry_point
}

rule dotfix_nice_protect_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DotFix Nice Protect"
		version = "2.x"
	strings:
		$1 = { E9 FF 00 00 00 60 8B 74 24 24 8B 7C 24 28 FC B2 80 33 DB A4 B3 02 E8 6D 00 00 00 73 F6 33 C9 E8 64 00 00 00 73 1C 33 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 12 C0 73 F7 75 3F AA EB D4 E8 4D 00 00 00 2B CB 75 10 E8 42 00 00 00 EB 28 AC D1 E8 74 4D 13 C9 EB 1C 91 48 C1 E0 08 AC E8 2C 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 8B C5 B3 01 56 8B F7 2B F0 F3 A4 5E EB 8E 02 D2 75 05 8A 16 46 12 D2 C3 33 C9 41 E8 EE FF FF FF 13 C9 E8 E7 FF FF FF 72 F2 C3 2B 7C 24 28 89 7C 24 1C 61 C3 60 B8 ?? ?? ?? ?? 03 C5 50 B8 ?? ?? ?? ?? 03 C5 FF 10 BB ?? ?? ?? ?? 03 DD 83 C3 0C 53 50 B8 ?? ?? ?? ?? 03 C5 FF 10 6A 40 68 00 10 00 00 FF 74 24 2C 6A 00 FF D0 89 44 24 1C 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule dragonarmor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DragonArmor"
	strings:
		$1 = { BF 4C ?? ?? 00 83 C9 FF 33 C0 68 34 ?? ?? 00 F2 AE F7 D1 49 51 68 4C ?? ?? 00 E8 11 0A 00 00 83 C4 0C 68 4C ?? ?? 00 FF 15 00 ?? ?? 00 8B F0 BF 4C ?? ?? 00 83 C9 FF 33 C0 F2 AE F7 D1 49 BF 4C ?? ?? 00 8B D1 68 34 ?? ?? 00 C1 E9 02 F3 AB 8B CA 83 E1 03 F3 AA BF 5C ?? ?? 00 83 C9 FF 33 C0 F2 AE F7 D1 49 51 68 5C ?? ?? 00 E8 C0 09 00 00 8B 1D 04 ?? ?? 00 83 C4 0C 68 5C ?? ?? 00 56 FF D3 A3 D4 ?? ?? 00 BF 5C ?? ?? 00 83 C9 FF 33 C0 F2 AE F7 D1 49 BF 5C ?? ?? 00 8B D1 68 34 ?? ?? 00 C1 E9 02 F3 AB 8B CA 83 E1 }
	condition:
		$1 at pe.entry_point
}

rule dropper_creator_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Dropper Creator"
		version = "0.1"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 8D 05 ?? ?? ?? ?? 29 C5 8D 85 ?? ?? ?? ?? 31 C0 64 03 40 30 78 0C 8B 40 0C 8B 70 1C AD 8B 40 08 EB 09 }
	condition:
		$1 at pe.entry_point
}

rule dshield {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DSHIELD"
	strings:
		$1 = { 06 E8 ?? ?? 5E 83 EE ?? 16 17 9C 58 B9 ?? ?? 25 ?? ?? 2E }
	condition:
		$1 at pe.entry_point
}

rule duals_crypt {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Dual's Cryptor"
	strings:
		$1 = { 55 8B EC 81 EC 00 05 00 00 E8 00 00 00 00 5D 81 ED 0E }
	condition:
		$1 at pe.entry_point
}

rule dup_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "dUP"
		version = "2.x"
	strings:
		$1 = { E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8B F0 6A 00 68 ?? ?? ?? ?? 56 E8 ?? ?? ?? ?? A2 ?? ?? ?? ?? 6A 00 68 ?? ?? ?? ?? 56 E8 ?? ?? ?? ?? A2 ?? ?? ?? ?? 6A 00 68 ?? ?? ?? ?? 56 E8 ?? ?? ?? ?? A2 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 56 E8 ?? ?? ?? ?? 3C 01 75 19 BE ?? ?? ?? ?? 68 00 02 00 00 56 68 }
	condition:
		$1 at pe.entry_point
}

rule dup_2x_patcher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "dUP"
		version = "2.x patcher"
	strings:
		$1 = { 8B CB 85 C9 74 ?? 80 3A 01 74 08 AC AE 75 0A 42 49 EB EF 47 46 42 49 EB E9 }
	condition:
		$1 at pe.entry_point
}

rule dxpack_086 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DxPack"
		version = "0.86"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 8B FD 81 ED 06 10 40 00 2B BD 94 12 40 00 81 EF 06 00 00 00 83 BD 14 13 40 00 01 0F 84 2F 01 00 00 }
	condition:
		$1 at pe.entry_point
}

rule dxpack_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DxPack"
		version = "1.0"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 8B FD 81 ED ?? ?? ?? ?? 2B B9 ?? ?? ?? ?? 81 EF ?? ?? ?? ?? 83 BD ?? ?? ?? ?? ?? 0F 84 }
	condition:
		$1 at pe.entry_point
}

rule dza_patcher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "DZA Patcher"
		version = "1.3"
	strings:
		$1 = { BF 00 40 40 00 99 68 48 20 40 00 68 00 20 40 00 52 52 52 52 52 52 52 57 E8 15 01 00 00 85 C0 75 1C 99 52 52 57 52 E8 CB 00 00 00 FF 35 4C 20 40 00 E8 D2 00 00 00 6A 00 E8 BF 00 00 00 99 68 58 20 40 00 52 52 68 63 10 40 00 52 52 E8 DB 00 00 00 6A FF FF 35 48 20 40 00 E8 C2 00 00 00 E8 C8 FF FF FF BF 40 40 40 00 FF 35 4C 20 40 00 E8 A1 00 00 00 8B 0F 83 F9 00 74 B1 60 6A 00 6A 04 6A 01 51 FF 35 48 20 40 00 E8 75 00 00 00 61 60 BB 5C 20 40 00 6A 00 6A 01 53 51 FF 35 48 20 40 00 E8 75 00 00 00 61 A0 }
	condition:
		$1 at pe.entry_point
}

rule e_you_di_dai {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "E.You.Di.Dai"
	strings:
		$1 = { 55 8B EC B8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 53 56 57 0F 31 8B D8 0F 31 8B D0 2B D3 C1 EA 10 B8 ?? ?? ?? ?? 0F 6E C0 B8 ?? ?? ?? ?? 0F 6E C8 0F F5 C1 0F 7E C0 0F 77 03 C2 ?? ?? ?? ?? ?? FF E0 }
	condition:
		$1 at pe.entry_point
}

rule elicense_system_4000 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Elicense System"
		version = "4.0.0.0"
	strings:
		$1 = { 00 00 00 00 63 79 62 00 65 6C 69 63 65 6E 34 30 2E 64 6C 6C 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule embedpe_100_124 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EmbedPE"
		version = "1.00 - 1.24"
	strings:
		$1 = { 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule embedpe_113_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EmbedPE"
		version = "1.13"
	strings:
		$1 = { 83 EC 50 60 68 5D B9 52 5A E8 2F 99 00 00 DC 99 F3 57 05 68 }
	condition:
		$1 at pe.entry_point
}

rule embedpe_113_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EmbedPE"
		version = "1.13"
	strings:
		$1 = { 83 EC 50 60 68 5D B9 52 5A E8 2F 99 00 00 DC 99 F3 57 05 68 B8 5E 2D C6 DA FD 48 63 05 3C 71 B8 5E 97 7C 36 7E 32 7C 08 4F 06 51 64 10 A3 F1 4E CF 25 CB 80 D2 99 54 46 ED E1 D3 46 86 2D 10 68 93 83 5C 46 4D 43 9B 8C D6 7C BB 99 69 97 71 2A 2F A3 38 6B 33 A3 F5 0B 85 97 7C BA 1D 96 DD 07 F8 FD D2 3A 98 83 CC 46 99 9D DF 6F 89 92 54 46 9F 94 43 CC 41 43 9B 8C 61 B9 D8 6F 96 3B D1 07 32 24 DD 07 05 8E CB 6F A1 07 5C 62 20 E0 DB BA 9D 83 54 46 E6 83 51 7A 2B 94 54 64 8A 83 05 68 D7 5E 2D C6 B7 57 00 }
	condition:
		$1 at pe.entry_point
}

rule embedpe_124 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EmbedPE"
		version = "1.24"
	strings:
		$1 = { 83 EC 50 60 68 ?? ?? ?? ?? E8 CB FF 00 00 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_12003318_12003518 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "1.2003.3.18 - 1.2003.5.18"
	strings:
		$1 = { 60 9C 64 FF 35 00 00 00 00 E8 79 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_22004616_22006630 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2004.6.16 - 2.2006.6.30"
	strings:
		$1 = { 60 9C 64 FF 35 00 00 00 00 E8 7A }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_22006115 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2006.1.15"
	strings:
		$1 = { 45 50 45 3A 20 45 6E 63 72 79 70 74 50 45 20 56 32 2E 32 30 30 36 2E 31 2E 31 35 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_22006710_220061025 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2006.7.10 - 2.2006.10.25"
	strings:
		$1 = { 60 9C 64 FF 35 00 00 00 00 E8 73 01 00 00 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_220070411 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2007.04.11"
	strings:
		$1 = { 60 9C 64 FF 35 00 00 00 00 E8 1B 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 47 65 74 54 65 6D 70 50 61 74 68 41 00 00 00 43 72 65 61 74 65 46 69 6C 65 41 00 00 00 43 72 65 61 74 65 46 69 6C 65 4D 61 70 70 69 6E 67 41 00 00 00 4D 61 70 56 69 65 77 4F 66 46 69 6C 65 00 00 00 55 6E 6D 61 70 56 69 65 77 4F 66 46 69 6C 65 00 00 00 43 6C 6F 73 65 48 61 6E 64 6C 65 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_22007121 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2007.12.1"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 45 50 45 3A 20 45 6E 63 72 79 70 74 50 45 20 56 32 2E 32 30 30 37 2E 31 32 2E 31 2C 20 43 6F 70 79 72 69 67 68 74 20 28 43 29 20 57 46 53 00 00 48 6F 6D 65 50 61 67 65 3A 20 77 77 77 2E 65 6E 63 72 79 70 74 70 65 2E 63 6F 6D 00 00 00 00 00 45 4D 61 69 6C 3A 20 77 66 73 23 65 6E 63 72 79 70 74 70 65 2E 63 6F 6D 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_22008618_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2008.6.18"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 47 65 74 54 65 6D 70 50 61 74 68 41 00 43 72 65 61 74 65 46 69 6C 65 41 00 43 72 65 61 74 65 46 69 6C 65 4D 61 70 70 69 6E 67 41 00 4D 61 70 56 69 65 77 4F 66 46 69 6C 65 00 55 6E 6D 61 70 56 69 65 77 4F 66 46 69 6C 65 00 43 6C 6F 73 65 48 61 6E 64 6C 65 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 56 32 32 30 30 38 30 36 31 38 2E 45 50 45 00 00 00 45 6E 63 72 79 70 74 50 45 5F 49 6E 69 74 00 }
	condition:
		$1 at pe.entry_point
}

rule encryptpe_22008618_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EncryptPE"
		version = "2.2008.6.18"
	strings:
		$1 = { 68 ?? ?? ?? 00 E8 52 01 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 47 65 74 54 65 6D 70 50 61 74 68 41 00 43 72 65 61 74 65 }
	condition:
		$1 at pe.entry_point
}

rule enigma_0x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "0.x beta"
	strings:
		$1 = { 60 E8 24 00 00 00 ?? ?? ?? EB 02 ?? ?? 8B 44 24 0C 83 80 B8 00 00 00 03 31 C0 C3 83 C0 08 EB 02 ?? ?? 89 C4 61 EB 2E ?? ?? ?? 83 04 24 03 EB 01 ?? 31 C0 EB 01 ?? 64 FF 30 EB 01 ?? 64 89 20 EB 02 ?? ?? 89 00 }
	condition:
		@1 < pe.overlay.offset or $1
}

rule enigma_102 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.02"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 83 ED 06 81 ED ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 9A 83 C4 04 EB 02 FF 35 60 E8 24 00 00 00 00 00 FF EB 02 CD 20 8B 44 24 0C 83 80 B8 00 00 00 03 31 C0 C3 83 C0 08 EB 02 FF 15 89 C4 61 EB 2E EA EB 2B 83 04 24 03 EB 01 00 31 C0 EB 01 85 64 FF 30 EB 01 83 64 89 20 EB 02 CD 20 89 00 9A 64 8F 05 00 00 00 00 EB 02 C1 90 58 61 EB 01 3E BE 01 00 00 00 C1 E6 02 83 EC 04 87 DE 89 1C 24 }
	condition:
		@1 < pe.overlay.offset or $1
}

rule enigma_11x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.1x"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 83 ED 06 8B F5 57 56 50 53 33 D8 8A C3 33 D8 EB 13 2A C3 05 77 02 00 00 81 EB 9A 09 00 00 5B 58 5E 5F EB 05 83 C3 17 EB E8 57 56 50 53 33 D8 8A C3 33 D8 EB 13 2A C3 05 77 }
	condition:
		$1 at pe.entry_point
}

rule enigma_11x_15x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.1x - 1.5x"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 00 10 40 00 E8 ?? ?? ?? ?? 9A 83 C4 10 8B E5 5D E9 }
	condition:
		$1 at pe.entry_point
}

rule enigma_10_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.0 - 1.2"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 83 ?? ?? 81 }
	condition:
		$1 at pe.entry_point
}

rule enigma_110_unregistred {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.10 unregistered"
	strings:
		$1 = { 60 72 80 72 88 72 8C 72 90 72 94 72 98 72 9C 72 A0 72 A4 59 A8 B0 5C E8 39 D5 39 E4 39 F1 31 F9 5C 3D 58 CA 5F 56 B1 2D 20 7A 2E 30 16 32 72 2B 72 36 1C A5 33 A9 9C AD 9C B1 9C B5 9C B9 9C BD 9C C1 9C C5 9C C9 9C CD 9C D1 9C D5 9C D9 9C DD 9C E1 9C E5 89 }
	condition:
		$1 at pe.entry_point
}

rule enigma_1x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.x"
	strings:
		$1 = { 45 6E 69 67 6D 61 20 70 72 6F 74 65 63 74 6F 72 20 76 31 }
	condition:
		$1 at pe.entry_point
}

rule enigma_1x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.x"
	strings:
		$1 = { 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 4D 65 73 73 61 67 65 42 6F 78 41 00 00 00 52 65 67 43 6C 6F 73 65 4B 65 79 00 00 00 53 79 73 46 72 65 65 53 74 72 69 6E 67 00 00 00 43 72 65 61 74 65 46 6F 6E 74 41 00 00 00 53 68 65 6C 6C 45 78 65 63 75 74 65 41 00 00 }
	condition:
		$1 at pe.entry_point
}

rule enigma_1x_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.x+"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 00 00 00 81 ED ?? ?? ?? ?? E9 ?? 00 00 00 45 4E 49 47 4D 41 }
	condition:
		@1 < pe.overlay.offset or $1
}

rule enigma_131 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Enigma"
		version = "1.31"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 00 00 00 81 ED ?? ?? ?? ?? E9 49 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 8A 84 24 28 00 00 00 80 F8 01 0F 84 07 00 00 00 B8 ?? ?? ?? ?? FF E0 E9 04 00 00 00 ?? ?? ?? ?? B8 ?? ?? ?? ?? 03 C5 81 C0 ?? ?? ?? ?? B9 ?? ?? ?? ?? BA ?? ?? ?? ?? 30 10 40 49 0F 85 F6 FF FF FF E9 04 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ep_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EP"
		version = "1.0"
	strings:
		$1 = { 50 83 C0 17 8B F0 97 33 C0 33 C9 B1 24 AC 86 C4 AC AA 86 C4 AA E2 F6 00 B8 40 00 03 00 3C 40 D2 33 8B 66 14 50 70 8B 8D 34 02 44 8B 18 10 48 70 03 BA 0C ?? ?? ?? ?? C0 33 FE 8B 30 AC 30 D0 C1 F0 10 C2 D0 30 F0 30 C2 C1 AA 10 42 42 CA C1 E2 04 5F E9 5E B1 C0 30 ?? 68 ?? ?? F3 00 C3 AA }
	condition:
		$1 at pe.entry_point
}

rule ep_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EP"
		version = "2.0"
	strings:
		$1 = { 6A ?? 60 E9 01 01 }
	condition:
		$1 at pe.entry_point
}

rule escargot_01_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Escargot"
		version = "0.1"
	strings:
		$1 = { EB 04 40 30 2E 31 60 68 61 }
	condition:
		$1 at pe.entry_point
}

rule escargot_01_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Escargot"
		version = "0.1"
	strings:
		$1 = { EB 08 28 65 73 63 30 2E 31 29 60 68 2B ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 B8 5C ?? ?? ?? 8B 00 FF D0 50 BE 00 10 ?? ?? B9 00 ?? ?? 00 EB 05 49 80 34 31 40 0B C9 75 F7 58 0B C0 74 08 33 C0 C7 00 DE C0 AD 0B BE ?? ?? ?? ?? E9 AC 00 00 00 8B 46 0C BB 00 00 ?? ?? 03 C3 50 50 B8 54 ?? ?? ?? 8B 00 FF D0 5F 80 3F 00 74 06 C6 07 00 47 EB F5 33 FF 8B 16 0B D2 75 03 8B 56 10 03 D3 03 D7 8B 0A C7 02 00 00 00 00 0B C9 74 4B F7 C1 00 00 00 80 74 14 81 E1 FF FF 00 00 50 51 50 B8 50 }
	condition:
		$1 at pe.entry_point
}

rule escargot_01f {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Escargot"
		version = "0.1f"
	strings:
		$1 = { EB 04 40 30 2E 31 60 68 61 ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 B8 92 ?? ?? ?? 8B 00 FF D0 50 B8 CD ?? ?? ?? 81 38 DE C0 37 13 75 2D 68 C9 ?? ?? ?? 6A 40 68 00 ?? 00 00 68 00 00 ?? ?? B8 96 ?? ?? ?? 8B 00 FF D0 8B 44 24 F0 8B 4C 24 F4 EB 05 49 C6 04 01 40 0B C9 75 F7 BE 00 10 ?? ?? B9 00 ?? ?? 00 EB 05 49 80 34 31 40 0B C9 75 F7 58 0B C0 74 08 33 C0 C7 00 DE C0 AD 0B BE ?? ?? ?? ?? E9 AC 00 00 00 8B 46 0C BB 00 00 ?? ?? 03 C3 50 50 }
	condition:
		$1 at pe.entry_point
}

rule excalibur_103_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Excalibur"
		version = "1.03"
	strings:
		$1 = { E9 00 00 00 00 60 E8 14 00 00 00 5D 81 ED 00 00 00 00 6A 45 E8 A3 00 00 00 68 00 00 00 00 E8 58 61 EB 39 }
	condition:
		$1 at pe.entry_point
}
rule excalibur_103_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Excalibur"
		version = "1.03"
	strings:
		$1 = { E9 00 00 00 00 60 E8 14 00 00 00 5D 81 ED 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule exe_guarder_18 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Exe Guarder"
		version = "1.8"
	strings:
		$1 = { 55 8B EC 83 C4 D0 53 56 57 8D 75 FC 8B 44 24 30 25 00 00 FF FF 81 38 4D 5A 90 00 74 07 2D 00 10 00 00 EB F1 89 45 FC E8 C8 FF FF FF 2D B2 04 00 00 89 45 F4 8B 06 8B 40 3C 03 06 8B 40 78 03 06 8B C8 8B 51 20 03 16 8B 59 24 03 1E 89 5D F0 8B 59 1C 03 1E 89 5D EC 8B 41 18 8B C8 49 85 C9 72 5A 41 33 C0 8B D8 C1 E3 02 03 DA 8B 3B 03 3E 81 3F 47 65 74 50 75 40 8B DF 83 C3 04 81 3B 72 6F 63 41 75 33 8B DF 83 C3 08 81 3B 64 64 72 65 75 26 83 C7 0C 66 81 3F 73 73 75 1C 8B D0 03 D2 03 55 F0 0F B7 12 C1 E2 }
	condition:
		$1 at pe.entry_point
}

rule exe_locker_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Exe Locker"
		version = "1.0"
	strings:
		$1 = { E8 00 00 00 00 60 8B 6C 24 20 81 ED 05 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule exe_manager_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Manager"
		version = "3.0"
		source = "(c) Solar Designer"
	strings:
		$1 = { B4 30 1E 06 CD 21 2E ?? ?? ?? BF ?? ?? B9 ?? ?? 33 C0 2E ?? ?? 47 E2 }
	condition:
		$1 at pe.entry_point
}

rule exe_packer_70 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Packer"
		version = "7.0"
	strings:
		$1 = { 1E 06 8C C3 83 ?? ?? 2E ?? ?? ?? ?? B9 ?? ?? 8C C8 8E D8 8B F1 4E 8B FE }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "1.1"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED FB 1D 40 00 B9 7B 09 00 00 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_250 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.50"
	strings:
		$1 = { 60 90 EB 22 45 78 65 53 74 65 61 6C 74 68 20 2D 20 77 77 77 2E 77 65 62 74 6F 6F 6C 6D 61 73 74 65 72 2E 63 6F 6D E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_27 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.7"
	strings:
		$1 = { EB 00 60 EB 00 E8 00 00 00 00 5D 81 ED D3 26 40 }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_271 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.71"
	strings:
		$1 = { EB 00 60 EB 00 E8 00 00 00 00 5D 81 ED B0 27 40 }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_273 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.73"
	strings:
		$1 = { EB 00 EB 2F 53 68 61 72 65 77 61 72 65 20 2D 20 45 78 65 53 74 65 61 6C 74 68 00 EB 16 77 77 77 2E 77 65 62 74 6F 6F 6C 6D 61 73 74 65 72 2E 63 6F 6D 00 60 90 E8 00 00 00 00 5D 81 ED F0 27 40 00 B9 15 00 00 00 83 C1 05 EB 05 EB FE 83 C7 56 EB 00 83 E9 02 81 C1 78 43 27 65 EB 00 81 C1 10 25 94 00 81 E9 63 85 00 00 B9 77 0C 00 00 90 8D BD 61 28 40 00 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_274_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.74"
	strings:
		$1 = { EB 00 EB 17 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 60 90 E8 00 00 00 00 5D }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_274_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.74"
	strings:
		$1 = { EB 00 EB 17 53 68 61 72 65 77 61 72 65 20 2D 20 45 78 65 53 74 65 61 6C 74 68 00 60 90 E8 00 00 00 00 5D 81 ED C4 27 40 00 B9 15 00 00 00 83 C1 04 83 C1 01 EB 05 EB FE 83 C7 56 EB 00 83 E9 02 81 C1 78 43 27 65 EB 00 81 C1 10 25 94 00 81 E9 63 85 00 00 B9 91 0C 00 00 90 8D BD 38 28 40 00 8B F7 AC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? AA E2 CC }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_275a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Exe Stealth"
		version = "2.75a"
	strings:
		$1 = { EB 58 53 68 61 72 65 77 61 72 65 2D 56 65 72 73 69 6F 6E 20 45 78 65 53 74 65 61 6C 74 68 2C 20 63 6F 6E 74 61 63 74 20 73 75 70 70 6F 72 74 40 77 65 62 74 6F 6F 6C 6D 61 73 74 65 72 2E 63 6F 6D 20 2D 20 77 77 77 2E 77 65 62 74 6F 6F 6C 6D 61 73 74 65 72 2E 63 6F 6D 00 90 60 90 E8 00 00 00 00 5D 81 ED F7 27 40 00 B9 15 00 00 00 83 C1 04 83 C1 01 EB 05 EB FE 83 C7 56 EB 00 EB 00 83 E9 02 81 C1 78 43 27 65 EB 00 81 C1 10 25 94 00 81 E9 63 85 00 00 B9 96 0C 00 00 90 8D BD 74 28 40 00 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_275 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.75"
	strings:
		$1 = { 90 60 90 E8 00 00 00 00 5D 81 ED D1 27 40 00 B9 15 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_276_unreg {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.76 unregistered"
	strings:
		$1 = { EB ?? 45 78 65 53 74 65 61 6C 74 68 20 56 32 20 53 68 61 72 65 77 61 72 65 20 }
	condition:
		$1 at pe.entry_point
}

rule exe_stealth_276 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE Stealth"
		version = "2.76"
	strings:
		$1 = { EB 65 45 78 65 53 74 65 61 6C 74 68 20 56 32 20 2D 20 77 77 77 2E 77 65 62 74 6F 6F 6C 6D 61 73 74 65 72 2E 63 6F 6D 20 59 4F 55 52 20 41 44 20 48 45 52 45 21 50 69 52 41 43 59 20 69 53 20 41 }
	condition:
		$1 at pe.entry_point
}

rule exe32pack_13x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE32Pack"
		version = "1.3x"
	strings:
		$1 = { 3B ?? 74 02 81 ?? 55 3B ?? 74 02 81 ?? 53 3B ?? 74 01 ?? ?? ?? ?? ?? 02 81 ?? ?? ?? ?? ?? ?? ?? 3B ?? 74 01 ?? 5D 8B D5 81 ED ?? ?? 40 }
	condition:
		$1 at pe.entry_point
}

rule exe32pack_13x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXE32Pack"
		version = "1.3x"
	strings:
		$1 = { 3B ?? 74 02 81 83 55 3B ?? 74 02 81 ?? 53 3B ?? 74 01 ?? ?? ?? ?? ?? 02 81 ?? ?? E8 ?? ?? ?? ?? 3B 74 01 ?? 5D 8B D5 81 ED }
	condition:
		$1 at pe.entry_point
}

rule exebundle_30_small {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeBundle"
		version = "3.0 small loader"
	strings:
		$1 = { 00 00 00 00 60 BE 00 F0 40 00 8D BE 00 20 FF FF 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 }
	condition:
		$1 at pe.entry_point
}

rule exebundle_30_standard {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeBundle"
		version = "3.0 standard loader"
	strings:
		$1 = { 00 00 00 00 60 BE 00 B0 42 00 8D BE 00 60 FD FF C7 87 B0 E4 02 00 31 3C 4B DF 57 83 CD FF EB 0E 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB }
	condition:
		$1 at pe.entry_point
}

rule execrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECrypt"
		version = "1.0"
	strings:
		$1 = { 90 90 60 E8 00 00 00 00 5D 81 ED D1 27 40 00 B9 15 00 00 00 83 C1 04 83 C1 01 EB 05 EB FE 83 C7 56 EB 00 EB 00 83 E9 02 81 C1 78 43 27 65 EB 00 81 C1 10 25 94 00 81 E9 63 85 00 00 B9 96 0C 00 00 90 8D BD 4E 28 40 00 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { 83 EC 04 50 53 E8 01 00 00 00 CC 58 8B D8 40 2D ?? ?? ?? ?? 2D ?? ?? 5F 00 05 ?? ?? 5F 00 80 3B CC 75 19 C6 03 00 BB 00 10 00 00 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 53 50 E8 0A 00 00 00 83 C0 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E8 ?? ?? ?? 00 05 ?? ?? ?? ?? FF E0 E8 ?? ?? ?? 00 05 ?? ?? 00 00 FF E0 E8 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E8 ?? ?? ?? FF 05 ?? ?? 00 00 FF E0 E8 ?? ?? ?? FF 05 ?? ?? 00 00 FF E0 E8 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E9 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 83 EC 0C 53 56 57 E8 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E9 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 83 EC 10 53 56 57 E8 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E9 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8A 06 ?? ?? ?? 47 0? DB 75 07 8B 1E 83 EE FC }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_07 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E9 ?? ?? ?? ?? 66 9C 60 50 8B D8 03 00 68 54 BC 00 00 6A 00 FF 50 14 8B CC }
	condition:
		$1 at pe.entry_point
}

rule execryptor_uv_08 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
	strings:
		$1 = { E8 24 ?? ?? ?? 8B 4C 24 0C C7 01 17 ?? 01 ?? C7 81 B8 ?? ?? ?? ?? ?? ?? ?? 31 C0 89 41 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_13045 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "1.3.0.45"
	strings:
		$1 = { E8 24 ?? ?? ?? 8B 4C 24 0C C7 01 17 ?? 01 ?? C7 81 ?? ?? ?? ?? ?? ?? ?? 31 C0 89 41 14 89 41 18 80 A1 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_151_153 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "1.5.1 - 1.5.3"
	strings:
		$1 = { E8 24 ?? ?? ?? 8B 4C 24 0C C7 01 17 ?? 01 ?? C7 81 B8 ?? ?? ?? ?? ?? ?? ?? 31 C0 89 41 14 89 41 18 80 A1 C1 ?? ?? ?? FE C3 31 C0 64 FF 30 64 89 20 CC C3 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_20_21 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.0, 2.1"
	strings:
		$1 = { 55 8B EC 83 C4 F4 56 57 53 BE ?? ?? ?? ?? B8 00 00 ?? ?? 89 45 FC 89 C2 8B 46 0C 09 C0 0F 84 ?? 00 00 00 01 D0 89 C3 50 FF 15 94 ?? ?? ?? 09 C0 0F 85 0F 00 00 00 53 FF 15 98 ?? ?? ?? 09 C0 0F 84 ?? 00 00 00 89 45 F8 6A 00 8F 45 F4 8B 06 09 C0 8B 55 FC 0F 85 03 00 00 00 8B 46 10 01 D0 03 45 F4 8B 18 8B 7E 10 01 D7 03 7D F4 09 DB 0F 84 ?? 00 00 00 F7 C3 00 00 00 80 0F 85 04 00 00 00 8D 5C 13 02 81 E3 FF FF FF ?? 53 FF 75 F8 FF 15 9C ?? ?? ?? 09 C0 0F 84 ?? 00 00 00 89 07 83 45 F4 04 E9 A6 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule execryptor_20_21_iat {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.0, 2.1 protected IAT"
	strings:
		$1 = { A4 ?? ?? ?? 00 00 00 00 FF FF FF FF 3C ?? ?? ?? 94 ?? ?? ?? D8 ?? ?? ?? 00 00 00 00 FF FF FF FF B8 ?? ?? ?? D4 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 60 ?? ?? ?? 70 ?? ?? ?? 84 ?? ?? ?? 00 00 00 00 75 73 65 72 33 32 2E 64 6C 6C 00 00 00 00 4D 65 73 73 61 67 65 42 6F 78 41 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_2117 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.1.17"
	strings:
		$1 = { BE ?? ?? ?? ?? B8 00 00 ?? ?? 89 45 FC 89 C2 8B 46 0C 09 C0 0F 84 ?? 00 00 00 01 D0 89 C3 50 FF 15 94 ?? ?? ?? 09 C0 0F 85 0F 00 00 00 53 FF 15 98 ?? ?? ?? 09 C0 0F 84 ?? 00 00 00 89 45 F8 6A 00 8F 45 F4 8B 06 09 C0 8B 55 FC 0F 85 03 00 00 00 8B 46 10 01 D0 03 45 F4 8B 18 8B 7E 10 01 D7 03 7D F4 09 DB 0F 84 ?? 00 00 00 F7 C3 00 00 00 80 0F 85 04 00 00 00 8D 5C 13 02 81 E3 FF FF FF 7F 53 FF 75 F8 FF 15 9C ?? ?? ?? 09 C0 0F 84 ?? 00 00 00 89 07 83 45 F4 04 E9 A6 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule execryptor_21x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.1.x"
	strings:
		$1 = { 83 C6 14 8B 55 FC E9 ?? FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule execryptor_21x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.1.x"
	strings:
		$1 = { E9 ?? ?? ?? ?? 66 9C 60 50 8D 88 ?? ?? ?? ?? 8D 90 04 16 ?? ?? 8B DC 8B E1 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_224_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2.4"
	strings:
		$1 = { 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_224_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2.4"
	strings:
		$1 = { E8 F7 FE FF FF 05 ?? ?? 00 00 FF E0 E8 EB FE FF FF 05 ?? ?? 00 00 FF E0 E8 ?? 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_226_min_prot_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2.6 minimum protection"
	strings:
		$1 = { 50 68 ?? ?? ?? ?? 58 81 E0 ?? ?? ?? ?? E9 ?? ?? ?? 00 87 0C 24 59 E8 ?? ?? ?? 00 89 45 F8 E9 ?? ?? ?? ?? 0F 83 ?? ?? ?? 00 E9 ?? ?? ?? ?? 87 14 24 5A 57 68 ?? ?? ?? ?? E9 ?? ?? ?? ?? 58 81 C0 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? 81 C8 ?? ?? ?? ?? 81 E0 ?? ?? ?? ?? E9 ?? ?? ?? 00 C3 E9 ?? ?? ?? ?? C3 BF ?? ?? ?? ?? 81 CB ?? ?? ?? ?? BA ?? ?? ?? ?? 52 E9 ?? ?? ?? 00 E8 ?? ?? ?? 00 E9 ?? ?? ?? 00 E9 ?? ?? ?? ?? 87 34 24 5E 66 8B 00 66 25 ?? ?? E9 ?? ?? ?? ?? 8B CD 87 0C 24 8B EC 51 89 EC 5D 8B 05 ?? ?? ?? ?? 09 C0 E9 ?? ?? ?? ?? 59 81 C1 ?? ?? ?? ?? C1 C1 ?? 23 0D ?? ?? ?? ?? 81 F9 ?? ?? ?? ?? E9 ?? ?? ?? ?? C3 E9 ?? ?? ?? 00 13 D0 0B F9 E9 ?? ?? ?? ?? 51 E8 ?? ?? ?? ?? 8B 64 24 08 31 C0 64 8F 05 00 00 00 00 5A E9 ?? ?? ?? ?? 3C A4 0F 85 ?? ?? ?? 00 8B 45 FC 66 81 38 ?? ?? 0F 84 05 00 00 00 E9 ?? ?? ?? ?? 0F 84 ?? ?? ?? ?? E9 ?? ?? ?? ?? 87 3C 24 5F 31 DB 31 C9 31 D2 68 ?? ?? ?? ?? E9 ?? ?? ?? ?? 89 45 FC 33 C0 89 45 F4 83 7D FC 00 E9 ?? ?? ?? ?? 53 52 8B D1 87 14 24 81 C0 ?? ?? ?? ?? 0F 88 ?? ?? ?? ?? 3B CB }
	condition:
		$1 at pe.entry_point
}

rule execryptor_226_min_prot_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2.6 minimum protection"
	strings:
		$1 = { 50 8B C6 87 04 24 68 ?? ?? ?? ?? 5E E9 ?? ?? ?? ?? 85 C8 E9 ?? ?? ?? ?? 81 C3 ?? ?? ?? ?? 0F 81 ?? ?? ?? 00 81 FA ?? ?? ?? ?? 33 D0 E9 ?? ?? ?? 00 0F 8D ?? ?? ?? 00 81 D5 ?? ?? ?? ?? F7 D1 0B 15 ?? ?? ?? ?? C1 C2 ?? 81 C2 ?? ?? ?? ?? 9D E9 ?? ?? ?? ?? C1 E2 ?? C1 E8 ?? 81 EA ?? ?? ?? ?? 13 DA 81 E9 ?? ?? ?? ?? 87 04 24 8B C8 E9 ?? ?? ?? ?? 55 8B EC 83 C4 F8 89 45 FC 8B 45 FC 89 45 F8 8B 45 08 E9 ?? ?? ?? ?? 8B 45 E0 C6 00 00 FF 45 E4 E9 ?? ?? ?? ?? FF 45 E4 E9 ?? ?? ?? 00 F7 D3 0F 81 ?? ?? ?? ?? E9 ?? ?? ?? ?? 87 34 24 5E 8B 45 F4 E8 ?? ?? ?? 00 8B 45 F4 8B E5 5D C3 E9 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_22_23_compressed_code {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2, 2.3 compressed code"
	strings:
		$1 = { E8 00 00 00 00 58 ?? ?? ?? ?? ?? 8B 1C 24 81 EB ?? ?? ?? ?? B8 ?? ?? ?? ?? 50 6A 04 68 00 10 00 00 50 6A 00 B8 C4 ?? ?? ?? 8B 04 18 FF D0 59 BA ?? ?? ?? ?? 01 DA 52 53 50 89 C7 89 D6 FC F3 A4 B9 ?? ?? ?? ?? 01 D9 FF D1 58 8B 1C 24 68 00 80 00 00 6A 00 50 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_22_23_iat {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2, 2.3 protected IAT"
	strings:
		$1 = { CC ?? ?? ?? 00 00 00 00 FF FF FF FF 3C ?? ?? ?? B4 ?? ?? ?? 08 ?? ?? ?? 00 00 00 00 FF FF FF FF E8 ?? ?? ?? 04 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 00 00 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 4C ?? ?? ?? 60 ?? ?? ?? 70 ?? ?? ?? 84 ?? ?? ?? 94 ?? ?? ?? A4 ?? ?? ?? 00 00 00 00 75 73 65 72 33 32 2E 64 6C 6C 00 00 00 00 4D 65 73 73 61 67 65 42 6F 78 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_22x_24x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2x - 2.4x"
	strings:
		$1 = { E8 ?? ?? ?? ?? 05 ?? ?? ?? ?? FF E0 E8 ?? ?? ?? ?? 05 ?? ?? ?? ?? FF E0 E8 04 00 00 00 FF FF FF FF 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_22x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.2x"
	strings:
		$1 = { FF E0 E8 04 00 00 00 FF FF FF FF 5E C3 00 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_239_compressed_res_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.3.9 compressed resources"
	strings:
		$1 = { 51 68 ?? ?? ?? ?? 59 81 F1 12 3C CB 98 E9 53 2C 00 00 F7 D7 E9 EB 60 00 00 83 45 F8 02 E9 E3 36 00 00 F6 45 F8 20 0F 84 1E 21 00 00 55 E9 80 62 00 00 87 0C 24 8B E9 ?? ?? ?? ?? 00 00 23 C1 81 E9 ?? ?? ?? ?? 57 E9 ED 00 00 00 0F 88 ?? ?? ?? ?? E9 2C 0D 00 00 81 ED BB 43 CB 79 C1 E0 1C E9 9E 14 00 00 0B 15 ?? ?? ?? ?? 81 E2 2A 70 7F 49 81 C2 9D 83 12 3B E8 0C 50 00 00 E9 A0 16 00 00 59 5B C3 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 E8 41 42 00 00 E9 93 33 00 00 31 DB 89 D8 59 5B C3 A1 ?? ?? ?? ?? 8A 00 2C 99 E9 82 30 00 00 0F 8A ?? ?? ?? ?? B8 01 00 00 00 31 D2 0F A2 25 FF 0F 00 00 E9 72 21 00 00 0F 86 57 0B 00 00 E9 ?? ?? ?? ?? C1 C0 03 E8 F0 36 00 00 E9 41 0A 00 00 81 F7 B3 6E 85 EA 81 C7 ?? ?? ?? ?? 87 3C 24 E9 74 52 00 00 0F 8E ?? ?? ?? ?? E8 5E 37 00 00 68 B1 74 96 13 5A E9 A1 04 00 00 81 D1 49 C0 12 27 E9 50 4E 00 00 C1 C8 1B 1B C3 81 E1 96 36 E5 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_239_compressed_res_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.3.9 compressed resources"
	strings:
		$1 = { 50 68 ?? ?? ?? ?? 58 C1 C0 0F E9 ?? ?? ?? 00 87 04 24 58 89 45 FC E9 ?? ?? ?? FF FF 05 ?? ?? ?? ?? E9 ?? ?? ?? 00 C1 C3 18 E9 ?? ?? ?? ?? 8B 55 08 09 42 F8 E9 ?? ?? ?? FF 83 7D F0 01 0F 85 ?? ?? ?? ?? E9 ?? ?? ?? 00 87 34 24 5E 8B 45 FC 33 D2 56 8B F2 E9 ?? ?? ?? 00 BA ?? ?? ?? ?? E8 ?? ?? ?? 00 A3 ?? ?? ?? ?? C3 E9 ?? ?? ?? 00 C3 83 C4 04 C3 E9 ?? ?? ?? FF 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 E8 ?? ?? ?? 00 E9 ?? ?? ?? FF C1 C2 03 81 CA ?? ?? ?? ?? 81 C2 ?? ?? ?? ?? 03 C2 5A E9 ?? ?? ?? FF 81 E7 ?? ?? ?? ?? 81 EF ?? ?? ?? ?? 81 C7 ?? ?? ?? ?? 89 07 E9 ?? ?? ?? ?? 0F 89 ?? ?? ?? ?? 87 14 24 5A 50 C1 C8 10 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_239_min_prot_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.3.9 minimum protection"
	strings:
		$1 = { 68 ?? ?? ?? ?? E9 ?? ?? ?? FF 50 C1 C8 18 89 05 ?? ?? ?? ?? C3 C1 C0 18 51 E9 ?? ?? ?? FF 84 C0 0F 84 6A F9 FF FF E9 ?? ?? ?? FF C3 E9 ?? ?? ?? FF E8 CF E9 FF FF B8 01 00 00 00 E9 ?? ?? ?? FF 2B D0 68 A0 36 80 D4 59 81 C9 64 98 FF 99 E9 ?? ?? ?? FF 84 C0 0F 84 8E EC FF FF E9 ?? ?? ?? FF C3 87 3C 24 5F 8B 00 03 45 FC 83 C0 18 E9 ?? ?? ?? FF 87 0C 24 59 B8 01 00 00 00 D3 E0 23 D0 E9 02 18 00 00 0F 8D DB 00 00 00 C1 E8 14 E9 CA 00 00 00 9D 87 0C 24 59 87 1C 24 68 AE 73 B9 96 E9 C5 10 00 00 0F 8A ?? ?? ?? ?? E9 ?? ?? ?? FF 81 FD F5 FF 8F 07 E9 4F 10 00 00 C3 E9 5E 12 00 00 87 3C 24 E9 ?? ?? ?? FF E8 ?? ?? ?? FF 83 3D ?? ?? ?? ?? 00 0F 85 ?? ?? ?? ?? 8D 55 EC B8 ?? ?? ?? ?? E9 ?? ?? ?? FF E8 A7 1A 00 00 E8 2A CB FF FF E9 ?? ?? ?? FF C3 E9 ?? ?? ?? FF 59 89 45 E0 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_239_min_prot_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.3.9 minimum protection"
	strings:
		$1 = { 51 68 ?? ?? ?? ?? 87 2C 24 8B CD 5D 81 E1 ?? ?? ?? ?? E9 ?? ?? ?? 00 89 45 F8 51 68 ?? ?? ?? ?? 59 81 F1 ?? ?? ?? ?? 0B 0D ?? ?? ?? ?? 81 E9 ?? ?? ?? ?? E9 ?? ?? ?? 00 81 C2 ?? ?? ?? ?? E8 ?? ?? ?? 00 87 0C 24 59 51 64 8B 05 30 00 00 00 8B 40 0C 8B 40 0C E9 ?? ?? ?? 00 F7 D6 2B D5 E9 ?? ?? ?? 00 87 3C 24 8B CF 5F 87 14 24 1B CA E9 ?? ?? ?? 00 83 C4 08 68 ?? ?? ?? ?? E9 ?? ?? ?? 00 C3 E9 ?? ?? ?? 00 E9 ?? ?? ?? 00 50 8B C5 87 04 24 8B EC 51 0F 88 ?? ?? ?? 00 FF 05 ?? ?? ?? ?? E9 ?? ?? ?? 00 87 0C 24 59 99 03 04 24 E9 ?? ?? ?? 00 C3 81 D5 ?? ?? ?? ?? 9C E9 ?? ?? ?? 00 81 FA ?? ?? ?? ?? E9 ?? ?? ?? 00 C1 C3 15 81 CB ?? ?? ?? ?? 81 F3 ?? ?? ?? ?? 81 C3 ?? ?? ?? ?? 87 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.x.x"
	strings:
		$1 = { A4 ?? ?? 00 00 00 00 00 FF FF FF FF 3C ?? ?? 00 94 ?? ?? 00 D8 ?? ?? 00 00 00 00 00 FF FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule execryptor_2xx_compressed_res {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.x.x compressed resources"
	strings:
		$1 = { 56 57 53 31 DB 89 C6 89 D7 0F B6 06 89 C2 83 E0 1F C1 EA 05 74 2D 4A 74 15 8D 5C 13 02 46 C1 E0 08 89 FA 0F B6 0E 46 29 CA 4A 29 C2 EB 32 C1 E3 05 8D 5C 03 04 46 89 FA 0F B7 0E 29 CA 4A 83 C6 02 EB 1D C1 E3 04 46 89 C1 83 E1 0F 01 CB C1 E8 05 73 07 43 89 F2 01 DE EB 06 85 DB 74 0E EB A9 56 89 D6 89 D9 F3 A4 31 DB 5E EB 9D 89 F0 5B 5F 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule execryptor_2xx_max_compressed_res {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXECryptor"
		version = "2.x.x max. compressed resources"
	strings:
		$1 = { 55 8B EC 83 C4 EC FC 53 57 56 89 45 FC 89 55 F8 89 C6 89 D7 66 81 3E 4A 43 0F 85 23 01 00 00 83 C6 0A C7 45 F4 08 00 00 00 31 DB BA 00 00 00 80 43 31 C0 E8 11 01 00 00 73 0E 8B 4D F0 E8 1F 01 00 00 02 45 EF AA EB E9 E8 FC 00 00 00 0F 82 97 00 00 00 E8 F1 00 00 00 73 5B B9 04 00 00 00 E8 FD 00 00 00 48 74 DE 0F 89 C7 00 00 00 E8 D7 00 00 00 73 1B 55 BD 00 01 00 00 E8 D7 00 00 00 88 07 47 4D 75 F5 E8 BF 00 00 00 72 E9 5D EB A2 B9 01 00 00 00 E8 C8 00 00 00 83 C0 07 89 45 F0 C6 45 EF 00 83 F8 08 74 89 E8 A9 00 00 00 88 45 EF E9 7C FF FF FF B9 07 00 00 00 E8 A2 00 00 00 50 }
	condition:
		$1 at pe.entry_point
}

rule exejoiner {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeJoiner"
	strings:
		$1 = { A1 14 A1 40 00 C1 E0 02 A3 18 A1 40 }
	condition:
		$1 at pe.entry_point
}

rule exejoiner_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeJoiner"
		version = "1.0"
	strings:
		$1 = { 68 00 10 40 00 68 04 01 00 00 E8 39 03 00 00 05 00 10 40 00 C6 00 5C 68 04 01 00 00 68 04 11 40 00 6A 00 E8 1A 03 00 00 6A 00 68 80 00 00 00 6A 03 6A 00 6A 01 68 00 00 00 80 68 04 11 40 00 E8 EC 02 00 00 83 F8 FF 0F 84 83 02 00 00 A3 08 12 40 00 6A 00 50 E8 E2 02 00 00 83 F8 FF 0F 84 6D 02 00 00 A3 0C 12 40 00 8B D8 83 EB 04 6A 00 6A 00 53 FF 35 08 12 40 00 E8 E3 02 00 00 6A 00 68 3C 12 40 00 6A 04 68 1E 12 40 00 FF 35 08 12 40 00 E8 C4 02 00 00 83 EB 04 6A 00 6A 00 53 FF 35 08 12 40 00 E8 B7 02 }
	condition:
		$1 at pe.entry_point
}

rule exejoiner_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeJoiner"
		version = "1.0"
	strings:
		$1 = { 68 00 10 40 00 68 04 01 00 00 E8 39 03 00 00 05 00 10 40 C6 00 5C 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule exelock_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeLock"
		version = "1.0"
	strings:
		$1 = { 06 8C C8 8E C0 BE ?? ?? 26 ?? ?? 34 ?? 26 ?? ?? 46 81 ?? ?? ?? 75 ?? 40 B3 ?? B3 ?? F3 }
	condition:
		$1 at pe.entry_point
}

rule exelock_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeLock"
		version = "1.5"
	strings:
		$1 = { BA ?? ?? BF ?? ?? EB ?? EA ?? ?? ?? ?? 79 ?? 7F ?? 7E ?? 1C ?? 48 78 ?? E3 ?? 45 14 ?? 5A E9 }
	condition:
		$1 at pe.entry_point
}

rule exepack_531009 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXEPACK"
		version = "5.31.009"
	strings:
		$1 = { 8B E8 8C C0 }
	condition:
		$1 at pe.entry_point
}

rule epack_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Epack"
		version = "1.4"
	strings:
		$1 = { 33 C0 8B C0 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule exerefractor_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EXERefactor"
		version = "0.1"
	strings:
		$1 = { 55 8B EC 81 EC 90 0B 00 00 53 56 57 E9 58 8C 01 00 55 53 43 41 54 49 4F 4E }
	condition:
		$1 at pe.entry_point
}

rule exesafeguard_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSafeguard"
		version = "1.0"
	strings:
		$1 = { C0 5D EB 4E EB 47 DF 69 4E 58 DF 59 74 F3 EB 01 DF 75 EE 9A 59 9C 81 C1 E2 FF FF FF EB 01 DF 9D FF E1 E8 51 E8 EB FF FF FF DF 22 3F 9A C0 81 ED 19 18 40 00 EB 48 EB 47 DF 69 4E 58 DF 59 79 EE EB 01 DF 78 E9 DF 59 9C 81 C1 E5 FF FF FF 9D FF E1 EB 51 E8 EE FF FF FF DF BA A3 22 3F 9A C0 60 EB 4D EB 47 DF 69 4E 58 DF 59 79 F3 EB 01 DF 78 EE DF 59 9C 81 C1 E5 FF FF FF 9D FF E1 EB 51 E8 EE FF FF FF E8 BA A3 22 3F 9A C0 8D B5 EE 19 40 00 EB 47 EB 47 DF 69 4E 58 DF 59 7A EE EB 01 DF 7B E9 DF 59 9C 81 C1 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_01b_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "0.1b - 0.6"
	strings:
		$1 = { E8 04 00 00 00 83 60 EB 0C 5D EB 05 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_01b_08 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "0.1b - 0.8"
	strings:
		$1 = { E8 04 00 00 00 83 ?? ?? ?? 5D EB 05 45 55 EB 04 ?? EB F9 ?? C3 E8 00 00 00 00 5D EB 01 ?? 81 ?? ?? ?? ?? ?? EB 02 ?? ?? 8D ?? ?? ?? ?? ?? EB 02 ?? ?? BA 9F 11 00 00 EB 01 ?? 8D ?? ?? ?? ?? ?? 8B 09 E8 14 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 58 40 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_17 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "1.7"
	strings:
		$1 = { EB 06 68 90 1F 06 00 C3 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 90 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_27 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "2.7"
	strings:
		$1 = { EB 06 68 F4 86 06 00 C3 9C 60 E8 02 00 00 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_27b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "2.7b"
	strings:
		$1 = { EB 06 68 40 85 06 00 C3 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 90 40 00 87 DD 8B 85 E6 90 40 00 01 85 33 90 40 00 66 C7 85 30 90 40 00 90 90 01 85 DA 90 40 00 01 85 DE 90 40 00 01 85 E2 90 40 00 BB 7B 11 00 00 03 9D EA 90 40 00 03 9D E6 90 40 00 53 8B C3 8B FB 2D AC 90 40 00 89 85 AD 90 40 00 8D B5 AC 90 40 00 B9 40 04 00 00 F3 A5 8B FB C3 BD 00 00 00 00 8B F7 83 C6 54 81 C7 FF 10 00 00 56 57 57 56 FF 95 DA 90 40 00 8B C8 5E 5F 8B C1 C1 F9 02 F3 A5 03 C8 83 E1 03 F3 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_29 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "2.9"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 0B 20 40 00 B9 EB 08 00 00 8D BD 53 20 40 00 8B F7 AC ?? ?? ?? F8 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_36 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "3.6"
	strings:
		$1 = { B8 ?? ?? ?? 00 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 6F 6D 70 61 63 74 32 00 CE 1E 42 AF F8 D6 CC E9 FB C8 4F 1B 22 7C B4 C8 0D BD 71 A9 C8 1F 5F B1 29 8F 11 73 8F 00 D1 88 87 A9 3F 4D 00 6C 3C BF C0 80 F7 AD 35 23 EB 84 82 6F 8C B9 0A FC EC E4 82 97 AE 0F 18 D2 47 1B 65 EA 46 A5 FD 3E 9D 75 2A 62 80 60 F9 B0 0D E1 AC 12 0E 9D 24 D5 43 CE 9A D6 18 BF 22 DA 1F 72 76 B0 98 5B C2 64 BC AE D8 }
	condition:
		$1 at pe.entry_point
}

rule exeshield_36_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
		version = "3.6 Protector"
	strings:
		$1 = { B8 ?? ?? ?? 00 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 6F 6D 70 61 63 74 32 00 CE 1E 42 AF F8 D6 CC }
	condition:
		$1 at pe.entry_point
}

rule exeshield_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeShield"
	strings:
		$1 = { 65 78 65 73 68 6C 2E 64 6C 6C C0 5D 00 }
	condition:
		$1 at pe.entry_point
}

rule exesmasher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSmasher"
	strings:
		$1 = { 9C FE 03 ?? 60 BE ?? ?? 41 ?? 8D BE ?? 10 FF FF 57 83 CD FF EB 10 }
	condition:
		$1 at pe.entry_point
}

rule exesplitter_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSplitter"
		version = "1.2"
	strings:
		$1 = { E9 95 02 00 00 64 A1 00 00 00 00 83 38 FF 74 04 8B 00 EB F7 8B 40 04 C3 55 8B EC B8 00 00 00 00 8B 75 08 81 E6 00 00 FF FF B9 06 00 00 00 56 56 E8 B0 00 00 00 5E 83 F8 01 75 06 8B C6 C9 C2 04 00 81 EE 00 00 01 00 E2 E5 C9 C2 04 00 55 8B EC 8B 75 0C 8B DE 03 76 3C 8D 76 18 8D 76 60 8B 36 03 F3 56 8B 76 20 03 F3 33 D2 8B C6 8B 36 03 F3 8B 7D 08 B9 0E 00 00 00 FC F3 A6 0B C9 75 02 EB 08 }
	condition:
		$1 at pe.entry_point
}

rule exesplitter_13_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSplitter"
		version = "1.3"
		description = "split only"
	strings:
		$1 = { E8 00 00 00 00 5D 81 ED 08 12 40 00 E8 66 FE FF FF 55 50 8D 9D 81 11 40 00 53 8D 9D 21 11 40 00 53 6A 08 E8 76 FF FF FF 6A 40 68 00 30 00 00 68 00 01 00 00 6A 00 FF 95 89 11 40 00 89 85 61 10 40 00 50 68 00 01 00 00 FF 95 85 11 40 00 8D 85 65 10 40 00 50 FF B5 61 10 40 00 FF 95 8D 11 40 00 6A 00 68 80 00 00 00 6A 02 6A 00 ?? ?? ?? ?? 01 1F 00 FF B5 61 10 40 00 FF 95 91 11 40 00 89 85 72 10 40 00 6A 00 8D ?? ?? ?? ?? 00 50 FF B5 09 10 40 00 8D 85 F5 12 40 00 50 FF B5 72 10 40 00 FF 95 95 11 40 00 FF B5 72 10 40 00 FF 95 99 11 40 00 8D 85 0D 10 40 00 50 8D 85 1D 10 40 00 50 B9 07 00 00 00 6A 00 E2 FC }
	condition:
		$1 at pe.entry_point
}

rule exesplitter_13_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSplitter"
		version = "1.3"
		description = "split only"
	strings:
		$1 = { E9 FE 01 00 00 ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 73 76 63 45 72 30 31 31 2E 74 6D 70 00 00 00 00 00 00 00 00 00 64 A1 30 00 00 00 8B 40 0C 8B 40 0C 8B 00 85 C0 0F 84 5F 02 00 00 8B 48 30 80 39 6B 74 07 80 39 4B 74 02 EB E7 80 79 0C 33 74 02 EB DF 8B 40 18 C3 }
	condition:
		$1 at pe.entry_point
}

rule exesplitter_13_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSplitter"
		version = "1.3"
		description = "split + crypt"
	strings:
		$1 = { 15 10 05 23 14 56 57 57 48 12 0B 16 66 66 66 66 66 66 66 66 66 02 C7 56 66 66 66 ED 26 6A ED 26 6A ED 66 E3 A6 69 E2 39 64 66 66 ED 2E 56 E6 5F 0D 12 61 E6 5F 2D 12 64 8D 81 E6 1F 6A 55 12 64 8D B9 ED 26 7E A5 33 ED 8A 8D 69 21 03 12 36 14 09 05 27 02 02 14 03 15 15 27 ED 2B 6A ED 13 6E ED B8 65 10 5A EB 10 7E EB 10 06 ED 50 65 95 30 ED 10 46 65 95 55 B4 ED A0 ED 50 65 95 37 ED 2B 6A EB DF AB 76 26 66 3F DF 68 66 66 66 9A 95 C0 6D AF 13 64 }
	condition:
		$1 at pe.entry_point
}

rule exesplitter_13_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ExeSplitter"
		version = "1.3"
		description = "split + crypt"
	strings:
		$1 = { E8 00 00 00 00 5D 81 ED 05 10 40 00 B9 ?? ?? ?? ?? 8D 85 1D 10 40 00 80 30 66 40 E2 FA 8F 98 67 66 66 ?? ?? ?? ?? ?? ?? ?? 66 }
	condition:
		$1 at pe.entry_point
}

rule expressor_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.0"
	strings:
		$1 = { E9 35 14 00 00 E9 31 13 00 00 E9 98 12 00 00 E9 EF 0C 00 00 E9 42 13 00 00 E9 E9 02 00 00 E9 EF 0B 00 00 E9 1B 0D 00 00 }
	condition:
		$1 at pe.entry_point
}

rule expressor_11_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.1"
	strings:
		$1 = { E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? 12 00 00 E9 ?? 0C 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule expressor_11_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXpressor"
		version = "1.1"
	strings:
		$1 = { E9 15 13 00 00 E9 F0 12 00 00 E9 58 12 00 00 E9 AF 0C 00 00 E9 AE 02 00 00 E9 B4 0B 00 00 E9 E0 0C 00 00 }
	condition:
		$1 at pe.entry_point
}

rule expressor_12_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.2"
	strings:
		$1 = { 45 78 50 72 2D 76 2E 31 2E 32 2E }
	condition:
		$1 at pe.entry_point
}

rule expressor_12_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXpressor"
		version = "1.2"
	strings:
		$1 = { 55 8B EC 81 EC ?? ?? ?? ?? 53 56 57 EB ?? 45 78 50 72 2D 76 }
	condition:
		$1 at pe.entry_point
}

rule expressor_13_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.3"
	strings:
		$1 = { 45 78 50 72 2D 76 2E 31 2E 33 2E }
	condition:
		$1 at pe.entry_point
}

rule expressor_13_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.3"
	strings:
		$1 = { 55 8B EC 83 EC ?? 53 56 57 EB 0C 45 78 50 72 2D 76 2E 31 2E 33 2E 2E B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? 83 3D ?? ?? ?? ?? 00 74 13 A1 ?? ?? ?? ?? 03 05 ?? ?? ?? ?? 89 ?? ?? E9 ?? ?? 00 00 C7 05 }
	condition:
		$1 at pe.entry_point
}

rule expressor_13_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.3"
	strings:
		$1 = { 55 8B EC 83 EC ?? 53 56 57 EB 0C 45 }
	condition:
		$1 at pe.entry_point
}

rule expressor_14_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.4"
	strings:
		$1 = { 55 8B EC 83 EC ?? 53 56 57 EB 0C 45 78 50 72 2D 76 2E 31 2E 34 2E 2E B8 }
	condition:
		$1 at pe.entry_point
}

rule expressor_14_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.4"
	strings:
		$1 = { 65 58 50 72 2D 76 2E 31 2E 34 2E }
	condition:
		$1 at pe.entry_point
}

rule expressor_145 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXpressor"
		version = "1.4.5"
	strings:
		$1 = { 55 8B EC 83 EC 58 53 56 57 83 65 DC 00 F3 EB 0C }
	condition:
		$1 at pe.entry_point
}

rule expressor_1451_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.4.5.1"
	strings:
		$1 = { 55 8B EC 83 EC 58 53 56 57 83 65 DC 00 F3 EB 0C 65 58 50 72 2D 76 2E 31 2E 34 2E 00 A1 00 ?? ?? ?? 05 00 ?? ?? ?? A3 08 ?? ?? ?? A1 08 ?? ?? ?? B9 81 ?? ?? ?? 2B 48 18 89 0D 0C ?? ?? ?? 83 3D 10 ?? ?? ?? 00 74 16 A1 08 ?? ?? ?? 8B 0D 0C ?? ?? ?? 03 48 14 }
	condition:
		$1 at pe.entry_point
}

rule expressor_1451_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.4.5.1"
	strings:
		$1 = { 55 8B EC 83 EC 58 53 56 57 83 65 DC 00 F3 EB 0C 65 58 50 72 2D 76 2E 31 2E 34 2E 00 A1 00 ?? ?? 00 05 00 ?? ?? 00 A3 08 ?? ?? 00 A1 08 ?? ?? 00 B9 81 ?? ?? 00 2B 48 18 89 0D 0C ?? ?? 00 83 3D }
	condition:
		$1 at pe.entry_point
}

rule expressor_150x_pack {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.5.0.x .Pack"
	strings:
		$1 = { 55 8B EC 81 EC ?? ?? ?? ?? 53 56 57 83 A5 ?? ?? ?? ?? ?? F3 EB 0C 65 58 50 72 2D 76 2E 31 2E 35 2E 00 83 7D 0C ?? 75 23 8B 45 08 A3 ?? ?? ?? ?? 6A 04 68 00 10 00 00 68 20 03 00 00 6A 00 FF 15 ?? ?? ?? ?? A3 ?? ?? ?? ?? EB 04 }
	condition:
		$1 at pe.entry_point
}

rule expressor_150x_protection {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.5.0.x .Protection"
	strings:
		$1 = { EB 01 68 EB 01 ?? ?? ?? ?? 83 EC 0C 53 56 57 EB 01 ?? 83 3D ?? ?? ?? ?? 00 74 08 EB 01 E9 E9 56 01 00 00 EB 02 E8 E9 C7 05 ?? ?? ?? ?? 01 00 00 00 EB 01 C2 E8 E2 05 00 00 EB 02 DA 9F 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? B8 ?? ?? ?? ?? FF D0 59 59 EB 01 C8 EB 02 66 F0 68 ?? ?? ?? ?? E8 0E 05 00 00 59 EB 01 DD 83 65 F4 00 EB 07 8B 45 F4 40 89 45 F4 83 7D F4 61 73 1F EB 02 DA 1A 8B 45 F4 0F ?? ?? ?? ?? ?? ?? 33 45 F4 8B 4D F4 88 ?? ?? ?? ?? ?? EB 01 EB EB }
	condition:
		$1 at pe.entry_point
}

rule expressor_1601_full {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.6.0.1 .Full Support"
	strings:
		$1 = { 55 8B EC 81 EC 74 02 00 00 53 56 57 83 A5 C8 FD FF FF 00 F3 EB 0C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? A1 ?? ?? ?? ?? 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? A1 ?? ?? ?? ?? 8B 40 04 25 ?? ?? ?? ?? 85 C0 74 5C 83 7D 0C 01 75 2A 8B 45 08 A3 ?? ?? ?? ?? 83 3D ?? ?? ?? ?? ?? 75 19 6A 04 68 00 10 00 00 68 20 03 00 00 6A 00 FF ?? ?? ?? ?? ?? A3 ?? ?? ?? ?? 83 7D 0C 00 75 0E 83 3D ?? ?? ?? ?? ?? 74 05 E9 F4 0A 00 00 83 3D ?? ?? ?? ?? ?? 74 05 E9 BB 09 00 00 C7 05 ?? ?? ?? ?? ?? ?? ?? ?? A1 ?? ?? ?? ?? 83 78 60 00 75 1C 6A 10 6A 00 E8 E8 19 00 00 59 50 6A 01 E8 DF 19 00 00 59 50 6A 00 FF 15 ?? ?? ?? ?? E8 27 FF FF FF A3 ?? ?? ?? ?? 6A 04 68 00 10 00 00 68 80 00 00 00 6A 00 FF 15 ?? ?? ?? ?? 89 85 E8 FD FF FF 68 04 01 00 00 8D 85 F0 FD FF FF 50 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8D 84 05 EF FD FF FF 89 85 D4 FD FF FF 8B 85 D4 FD FF FF 0F BE 00 83 F8 5C 74 0F 8B 85 D4 FD FF FF 48 89 85 D4 FD FF FF EB E3 8B 85 D4 FD FF FF 40 89 85 D4 FD FF FF 8B 85 D4 FD FF FF 8D 8D F0 FD FF FF 2B C1 89 85 AC FD FF FF 8B 8D AC FD FF FF 8D B5 F0 FD FF FF 8D BD FC FE FF FF 8B C1 C1 E9 02 F3 A5 8B C8 83 E1 03 F3 A4 8B 85 AC FD FF FF 80 A4 05 FD FE FF FF 00 83 A5 D8 FD FF FF 00 A1 ?? ?? ?? ?? 8B 40 04 25 ?? ?? ?? ?? 85 C0 75 11 A1 ?? ?? ?? ?? 8B 40 04 25 ?? ?? ?? ?? 85 C0 74 43 E8 11 0C 00 00 89 85 D8 FD FF FF A1 ?? ?? ?? ?? 8B 40 04 25 ?? ?? ?? ?? 85 C0 74 27 83 BD D8 FD FF FF 00 74 1E 6A 10 FF B5 D4 FD FF FF 6A 18 E8 C3 18 00 00 59 50 6A 00 FF 15 ?? ?? ?? ?? E9 8F 09 00 00 }
	condition:
		$1 at pe.entry_point
}

rule expressor_1601_light {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.6.0.1 .Light"
	strings:
		$1 = { 55 8B EC 81 EC 68 02 00 00 53 56 57 83 A5 D0 FD FF FF 00 F3 EB 0C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? A1 ?? ?? ?? ?? 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? A1 ?? ?? ?? ?? 83 78 60 00 75 14 6A 10 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 FF 15 ?? ?? ?? ?? E8 9C FF FF FF A3 ?? ?? ?? ?? 68 04 01 00 00 8D 85 F0 FD FF FF 50 FF 35 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8D 84 05 EF FD FF FF 89 85 DC FD FF FF 8B 85 DC FD FF FF 0F BE 00 83 F8 5C 74 0F 8B 85 DC FD FF FF 48 89 85 DC FD FF FF EB E3 8B 85 DC FD FF FF 40 89 85 DC FD FF FF 8B 85 DC FD FF FF 8D 8D F0 FD FF FF 2B C1 89 85 B4 FD FF FF 8B 8D B4 FD FF FF 8D B5 F0 FD FF FF 8D BD FC FE FF FF 8B C1 C1 E9 02 F3 A5 8B C8 83 E1 03 F3 A4 8B 85 B4 FD FF FF 80 A4 05 FD FE FF FF 00 83 A5 E0 FD FF FF 00 A1 ?? ?? ?? ?? 8B 40 04 25 ?? ?? ?? ?? 85 C0 75 11 A1 ?? ?? ?? ?? 8B 40 04 25 00 00 00 02 85 C0 74 2A E8 5B 06 00 00 89 85 E0 FD FF FF A1 ?? ?? ?? ?? 8B 40 04 25 ?? ?? ?? ?? 85 C0 74 0E 83 BD E0 FD FF FF 00 74 05 E9 34 06 00 00 }
	condition:
		$1 at pe.entry_point
}

rule expressor_1601_protection {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "eXPressor"
		version = "1.6.0.1 .Protection "
	strings:
		$1 = { EB 01 ?? EB 01 ?? 55 8B EC 83 EC 0C 53 56 57 EB 01 ?? 83 3D ?? ?? ?? ?? ?? 74 08 EB 01 ?? E9 56 01 00 00 EB 02 ?? ?? C7 05 ?? ?? ?? ?? ?? ?? ?? ?? EB 01 ?? E8 E2 05 00 00 EB 02 ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? B8 ?? ?? ?? ?? FF D0 59 59 EB 01 ?? EB 02 ?? ?? 68 ?? ?? ?? ?? E8 0E 05 00 00 59 EB 01 ?? 83 65 F4 00 EB 07 8B 45 F4 40 89 45 F4 83 7D F4 61 73 1F EB 02 ?? ?? 8B 45 F4 0F B6 ?? ?? ?? ?? ?? 33 45 F4 8B 4D F4 88 81 ?? ?? ?? ?? EB 01 ?? EB D4 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? B8 ?? ?? ?? ?? FF D0 83 C4 10 89 45 FC EB 02 ?? ?? 83 7D FC 00 75 0A 6A 00 A1 ?? ?? ?? ?? FF 50 14 EB 01 ?? F3 E8 A0 05 00 00 A1 ?? ?? ?? ?? 05 ?? ?? ?? ?? 89 45 F8 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? FF 75 FC E8 01 00 00 00 ?? 83 04 24 06 C3 }
	condition:
		$1 at pe.entry_point
}

rule ezip_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "EZIP"
		version = "1.0"
	strings:
		$1 = { E9 19 32 00 00 E9 7C 2A 00 00 E9 19 24 00 00 E9 FF 23 00 00 E9 1E 2E 00 00 E9 88 2E 00 00 E9 2C }
	condition:
		$1 at pe.entry_point
}

rule fakeninja_28_ad {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FakeNinja"
		version = "2.8 anti debug"
	strings:
		$1 = { 64 A1 18 00 00 00 EB 02 C3 11 8B 40 30 EB 01 0F 0F B6 40 02 83 F8 01 74 FE EB 01 E8 90 C0 FF FF EB 03 BD F4 B5 64 A1 30 00 00 00 0F B6 40 02 74 01 BA 74 E0 50 00 64 A1 30 00 00 00 83 C0 68 8B 00 EB 00 83 F8 70 74 CF EB 02 EB FE 90 90 90 0F 31 33 C9 03 C8 0F 31 2B C1 3D FF 0F 00 00 73 EA E8 08 00 00 00 C1 3D FF 0F 00 00 74 AA EB 07 E8 8B 40 30 EB 08 EA 64 A1 18 00 00 00 EB F2 90 90 90 BA ?? ?? ?? ?? FF E2 64 11 40 00 FF 35 84 11 40 00 E8 40 11 00 00 6A 00 6A 00 FF 35 70 11 40 00 FF 35 84 11 40 00 E8 25 11 00 00 FF }
	condition:
		$1 at pe.entry_point
}

rule fakeninja_28_private {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FakeNinja"
		version = "2.8 private"
	strings:
		$1 = { 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 17 E5 FF 60 }
	condition:
		$1 at pe.entry_point
}

rule fakeninja_28 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FakeNinja"
		version = "2.8"
	strings:
		$1 = { BA ?? ?? ?? ?? FF E2 64 11 40 00 FF 35 84 11 40 00 E8 40 }
	condition:
		$1 at pe.entry_point
}

rule feokt {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Feokt"
	strings:
		$1 = { 89 25 A8 11 40 00 BF ?? ?? ?? 00 31 C0 B9 ?? ?? ?? 00 29 F9 FC F3 AA ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 ?? ?? 00 00 BE ?? ?? 40 00 BF }
	condition:
		$1 at pe.entry_point
}

rule fileshield {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FileShield"
	strings:
		$1 = { 50 1E EB ?? 90 00 00 8B D8 }
	condition:
		$1 at pe.entry_point
}

rule flash_player {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Flash Player"
	strings:
		$1 = { 83 ?? ?? 56 FF 15 ?? ?? ?? ?? 8B F0 8A 06 3C ?? 75 ?? 8A ?? ?? 46 3C ?? 74 ?? 84 C0 74 ?? 8A ?? ?? 46 3C ?? 75 ?? 80 ?? ?? 75 ?? 46 EB ?? 3C ?? 7E ?? 8A }
	condition:
		$1 at pe.entry_point
}

rule flash_player_80 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Flash Player"
		version = "8.0"
	strings:
		$1 = { 83 ?? ?? 56 FF 15 ?? ?? ?? ?? 8B F0 8A 06 3C ?? 75 ?? 8A ?? ?? 46 3C ?? 74 ?? 84 C0 75 ?? 3C ?? 75 ?? 46 EB ?? 3C ?? 76 ?? 8D A4 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_101_shield_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.01 shield"
	strings:
		$1 = { 55 8B EC 83 C4 D0 53 56 57 8B 45 10 83 C0 0C 8B 00 89 45 DC 83 7D DC 00 75 08 E8 AD FF FF FF 89 45 DC E8 C1 FE FF FF 8B 10 03 55 DC 89 55 E4 83 C0 04 8B 10 89 55 FC 83 C0 04 8B 10 89 55 F4 83 C0 04 8B 10 89 55 F8 83 C0 04 8B 10 89 55 F0 83 C0 04 8B 10 89 55 EC 83 C0 04 8B 00 89 45 E8 8B 45 E4 8B 58 04 03 5D E4 8B FB 8B 45 E4 8B 30 4E 85 F6 72 2B 46 C7 45 E0 00 00 00 00 83 7B 04 00 74 14 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_101_shield_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.01 shield"
	strings:
		$1 = { 60 E8 12 FE FF FF C3 90 09 00 00 00 2C 00 00 00 ?? ?? ?? ?? C4 03 00 00 BC A0 00 00 00 40 01 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 99 00 00 00 00 8A 00 00 00 10 00 00 28 88 00 00 40 ?? 4B 00 00 00 02 00 00 00 A0 00 00 18 01 00 00 40 ?? 4C 00 00 00 0C 00 00 00 B0 00 00 38 0A 00 00 40 ?? 4E 00 00 00 00 00 00 00 C0 00 00 40 39 00 00 40 ?? 4E 00 00 00 08 00 00 00 00 01 00 C8 06 00 00 40 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_102_packer {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.02 packer"
	strings:
		$1 = { 60 E8 07 00 00 00 61 68 ?? ?? ?? ?? C3 5E 56 8B 56 02 ?? ?? ?? AD 01 D0 5B 36 89 43 02 66 3E C7 43 FA EB 05 53 AD 01 D0 89 C3 C7 43 FC 00 10 00 00 C7 43 F8 00 80 00 00 89 53 F4 AD 01 D0 89 43 F0 AD 01 D0 89 43 10 52 6A 04 FF 73 FC AD 50 6A 00 3E FF 53 08 89 C5 6A 04 FF 73 FC AD 50 6A 00 3E FF 53 08 89 C1 5A 83 EE 08 AD 50 55 AD AD 50 AD 01 D0 50 6A 02 6A 00 6A ?? 51 89 CF FF 53 10 83 C4 20 FF 73 F8 6A 00 57 FF 53 0C }
	condition:
		$1 at pe.entry_point
}

rule fishpe_10x_packer_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.0x packer"
	strings:
		$1 = { 60 E8 21 00 00 00 EB 18 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 5E 56 8B 56 1C 89 F3 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_10x_packer_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.0x"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? C3 90 09 00 00 00 2C 00 00 00 ?? ?? ?? ?? C4 03 00 00 BC A0 00 00 00 40 01 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 99 00 00 00 00 8A 00 00 00 10 00 00 ?? ?? 00 00 ?? ?? ?? ?? 00 00 02 00 00 00 A0 00 00 18 01 00 00 ?? ?? ?? ?? 00 00 0C 00 00 00 B0 00 00 38 0A 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 C0 00 00 40 39 00 00 ?? ?? ?? ?? 00 00 08 00 00 00 00 01 00 C8 06 00 00 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_104_10x_packer {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.04 - 1.0x packer"
	strings:
		$1 = { 60 B8 ?? ?? ?? ?? FF D0 5A 00 00 ?? ?? ?? ?? 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 57 56 53 55 89 E5 8B 45 20 01 45 24 50 FC 8B 75 18 01 75 1C 56 8B 75 14 AD 92 52 8A 4E FE 83 C8 FF D3 E0 F7 D0 50 88 F1 83 C8 FF D3 E0 F7 D0 50 00 D1 89 F7 83 EC 0C 29 C0 40 50 50 50 50 50 57 AD 89 C1 AD 29 F6 56 83 CB FF F3 AB 6A 05 59 E8 9C 02 00 00 E2 F9 8D 36 8D 3F 8B 7D FC 8B 45 F0 2B 7D 20 21 F8 89 45 E8 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_112_116_shield_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.12, 1.16 shield"
	strings:
		$1 = { 55 8B EC 83 C4 D0 53 56 57 8B 45 10 83 C0 0C 8B 00 89 45 DC 83 7D DC 00 75 08 E8 BD FE FF FF 89 45 DC E8 E1 FD FF FF 8B 00 03 45 DC 89 45 E4 E8 DC FE FF FF 8B D8 BA 8E 4E 0E EC 8B C3 E8 2E FF FF FF 89 45 F4 BA 04 49 32 D3 8B C3 E8 1F FF FF FF 89 45 F8 BA 54 CA AF 91 8B C3 E8 10 FF FF FF 89 45 F0 BA AC 33 06 03 8B C3 E8 01 FF FF FF 89 45 EC BA 1B C6 46 79 8B C3 E8 F2 FE FF FF 89 45 E8 BA AA FC 0D 7C 8B C3 E8 E3 FE FF FF 89 45 FC 8B 45 E4 8B 58 04 03 5D E4 8B FB 8B 45 E4 8B 30 4E 85 F6 72 2B }
	condition:
		$1 at pe.entry_point
}

rule fishpe_112_116_shield_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.12, 1.16 shield"
	strings:
		$1 = { 60 E8 EA FD FF FF FF D0 C3 8D 40 00 ?? 00 00 00 2C 00 00 00 ?? ?? ?? 00 ?? ?? 00 00 ?? ?? ?? 00 00 ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? 00 00 00 00 ?? ?? 00 ?? ?? 00 00 ?? 00 00 00 00 ?? ?? 00 00 10 00 00 ?? ?? ?? 00 40 ?? ?? ?? 00 00 ?? ?? 00 00 ?? ?? 00 ?? ?? ?? 00 40 ?? ?? ?? 00 00 ?? 00 00 00 ?? ?? 00 ?? ?? 00 00 40 }
	condition:
		$1 at pe.entry_point
}

rule fishpe_11x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FishPE"
		version = "1.1x"
	strings:
		$1 = { 50 45 00 00 4C 01 0A 00 19 5E 42 2A 00 00 00 00 00 00 00 00 E0 00 8E 81 0B 01 02 19 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 10 00 00 00 02 00 00 04 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 ?? ?? ?? ?? 00 04 00 00 00 00 00 00 02 00 00 00 00 00 10 00 00 40 00 00 00 00 10 00 00 10 00 00 00 00 00 00 10 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 18 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C2 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E0 00 00 E0 }
	condition:
		$1 at pe.entry_point
}

rule fixuppak_120_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FixupPak"
		version = "1.20"
	strings:
		$1 = { 55 E8 00 00 00 00 5D 81 ED ?? ?? 00 00 BE 00 ?? 00 00 03 F5 BA 00 00 ?? ?? 2B D5 8B DD 33 C0 AC 3C 00 74 3D 3C 01 74 0E 3C 02 74 0E 3C 03 74 0D 03 D8 29 13 EB E7 66 AD EB F6 AD EB F3 AC 0F B6 C8 3C 00 74 06 3C 01 74 09 EB 0A 66 AD 0F B7 C8 EB 03 AD 8B C8 }
	condition:
		$1 at pe.entry_point
}

rule fixuppak_120_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FixupPak"
		version = "1.20"
	strings:
		$1 = { 55 E8 00 00 00 00 5D 81 ED ?? ?? 00 00 BE 00 ?? 00 00 03 F5 BA 00 00 ?? ?? 2B D5 8B DD 33 C0 AC 3C 00 74 3D 3C 01 74 0E 3C 02 74 0E 3C 03 74 0D 03 D8 29 13 EB E7 66 AD EB F6 AD EB F3 AC 0F B6 C8 3C 00 74 06 3C 01 74 09 EB 0A 66 AD 0F B7 C8 EB 03 AD 8B C8 AC 0F B6 C0 03 D8 29 13 E2 FA EB BC 8D 85 ?? ?? 00 00 5D FF E0 00 00 00 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule flash_projector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Flash Projector"
	strings:
		$1 = { 83 EC 44 56 FF 15 ?? ?? ?? ?? 8B F0 8A 06 3C 22 75 1C 8A 46 01 46 3C 22 74 0C 84 C0 74 08 8A 46 01 46 3C 22 75 F4 80 3E 22 75 0F 46 EB 0C 3C 20 7E 08 8A 46 01 46 3C 20 7F F8 8A 06 84 C0 74 0C 3C 20 7F 08 8A 46 01 46 84 C0 75 F4 8D 44 24 04 C7 44 24 30 00 00 00 00 50 FF 15 }
	condition:
		$1 at pe.entry_point
}

rule flash_projector_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Flash Projector"
		version = "3.0"
	strings:
		$1 = { 55 8B EC 83 EC 44 56 FF 15 94 13 42 00 8B F0 B1 22 8A 06 3A C1 75 13 8A 46 01 46 3A C1 74 04 84 C0 75 F4 38 0E 75 0D 46 EB 0A 3C 20 7E 06 }
	condition:
		$1 at pe.entry_point
}

rule flycrypter_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Fly-Crypter"
		version = "1.0"
	strings:
		$1 = { 53 56 57 55 BB 2C ?? ?? 44 BE 00 30 44 44 BF 20 ?? ?? 44 80 7B 28 00 75 16 83 3F 00 74 11 8B 17 89 D0 33 D2 89 17 8B E8 FF D5 83 3F 00 75 EF 83 3D 04 30 44 44 00 74 06 FF 15 58 30 44 44 80 7B 28 02 75 0A 83 3E 00 75 05 33 C0 89 43 0C FF 15 20 30 44 44 80 7B 28 01 76 05 83 3E 00 74 22 8B 43 10 85 C0 74 1B FF 15 18 30 44 44 8B 53 10 8B 42 10 3B 42 04 74 0A 85 C0 74 06 50 E8 2F FA FF FF FF 15 24 30 44 44 80 7B 28 01 75 03 FF 53 24 80 7B 28 00 74 05 E8 35 FF FF FF 83 3B 00 75 17 83 3D 10 ?? ?? 44 00 74 06 FF 15 10 ?? ?? 44 8B 06 50 E8 51 FA FF FF 8B 03 56 8B F0 8B FB B9 0B 00 00 00 F3 A5 5E E9 73 FF FF FF 5D 5F 5E 5B C3 A3 00 30 44 44 E8 26 FF FF FF C3 }
	condition:
		$1 at pe.entry_point
}

rule flycrypter_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Fly-Crypter"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 B8 18 22 44 44 E8 7F F7 FF FF E8 0A F1 FF FF B8 09 00 00 00 E8 5C F1 FF FF 8B D8 85 DB 75 05 E8 85 FD FF FF 83 FB 01 75 05 E8 7B FD FF FF 83 FB 02 75 05 E8 D1 FD FF FF 83 FB 03 75 05 E8 87 FE FF FF 83 FB 04 75 05 E8 5D FD FF FF 83 FB 05 75 05 E8 B3 FD FF FF 83 FB 06 75 05 E8 69 FE FF FF 83 FB 07 75 05 E8 5F FE FF FF 83 FB 08 75 05 E8 95 FD FF FF 83 FB 09 75 05 E8 4B FE FF FF 5B E8 9D F2 FF FF 90 }
	condition:
		$1 at pe.entry_point
}

rule freecryptor_01001 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeCryptor"
		version = "0.1.001"
	strings:
		$1 = { 8B 04 24 40 90 83 C0 07 80 38 90 90 74 02 EB FF 68 26 ?? ?? 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 FF E4 90 8B 04 24 64 A3 00 00 00 00 8B 64 24 08 90 83 C4 08 }
	condition:
		$1 at pe.entry_point
}

rule freecryptor_01002 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeCryptor"
		version = "0.1.002"
	strings:
		$1 = { 8B 04 24 40 90 83 C0 07 80 38 90 90 74 02 EB FF 90 68 27 ?? ?? 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 FF E4 90 8B 04 24 64 A3 00 00 00 00 8B 64 24 08 90 83 C4 08 }
	condition:
		$1 at pe.entry_point
}

rule freecryptor_02002 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeCryptor"
		version = "0.2.002"
	strings:
		$1 = { 33 D2 90 1E 68 1B ?? ?? ?? 0F A0 1F 8B 02 90 50 54 8F 02 90 90 8E 64 24 08 FF E2 58 50 33 D2 52 83 F8 01 9B 40 8A 10 89 14 24 90 D9 04 24 90 D9 FA D9 5C 24 FC 8B 5C 24 FC 81 F3 C2 FC 1D 1C 75 E3 74 01 62 FF D0 90 5A 33 C0 8B 54 24 08 90 64 8F 00 90 83 C2 08 52 5C 5A }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_151 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "1.5.1"
	strings:
		$1 = { 90 87 FF 90 90 B9 2B 00 00 00 BA 07 10 40 00 83 C2 03 90 87 FF 90 90 B9 04 00 00 00 90 87 FF 90 33 C9 C7 05 09 30 40 00 00 00 00 00 68 00 01 00 00 68 21 30 40 00 6A 00 E8 B7 02 00 00 6A 00 68 80 00 00 00 6A 03 6A 00 6A 00 68 00 00 00 80 68 21 30 40 00 E8 8F 02 00 00 A3 19 30 40 00 90 87 FF 90 8B 15 09 30 40 00 81 C2 04 01 00 00 F7 DA 6A 02 6A 00 52 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_152 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "1.5.2 stub engine 1.6"
	strings:
		$1 = { E8 46 FD FF FF 50 E8 0C 00 00 00 FF 25 08 20 40 00 FF 25 0C 20 40 00 FF 25 10 20 40 00 FF 25 14 20 40 00 FF 25 18 20 40 00 FF 25 1C 20 40 00 FF 25 20 20 40 00 FF 25 24 20 40 00 FF 25 28 20 40 00 FF 25 00 20 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_153_17 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "1.5.3 stub engine 1.7"
	strings:
		$1 = { E8 33 FD FF FF 50 E8 0D 00 00 00 CC FF 25 08 20 40 00 FF 25 0C 20 40 00 FF 25 10 20 40 00 FF 25 14 20 40 00 FF 25 18 20 40 00 FF 25 1C 20 40 00 FF 25 20 20 40 00 FF 25 24 20 40 00 FF 25 28 20 40 00 FF 25 00 20 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_153_171 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "1.5.3 stub engine 1.7.1"
	strings:
		$1 = { E8 02 FD FF FF 6A 00 E8 0D 00 00 00 CC FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A8 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_small_014_021 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "small build 014 - 021"
	strings:
		$1 = { E8 ?? ?? FF FF 6A 00 E8 0D 00 00 00 CC FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_small_023 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "small build 023"
	strings:
		$1 = { E8 E1 FD FF FF 6A 00 E8 0C 00 00 00 FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_small_029 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "small build 029"
	strings:
		$1 = { 50 32 C4 8A C3 58 E8 DE FD FF FF 6A 00 E8 0D 00 00 00 CC FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_small_031_032 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "small build 031, 032"
	strings:
		$1 = { 50 32 ?? 66 8B C3 58 E8 ?? FD FF FF 6A 00 E8 0D 00 00 00 CC FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_small_033 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "small build 033"
	strings:
		$1 = { 50 66 33 C3 66 8B C1 58 E8 AC FD FF FF 6A 00 E8 0D 00 00 00 CC FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freejoiner_small_035 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FreeJoiner"
		version = "small build 035"
	strings:
		$1 = { 51 33 CB 86 C9 59 E8 9E FD FF FF 66 87 DB 6A 00 E8 0C 00 00 00 FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 AC 10 40 00 }
	condition:
		$1 at pe.entry_point
}

rule freshbind_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Freshbind"
		version = "2.0"
	strings:
		$1 = { 64 A1 00 00 00 00 55 89 E5 6A FF 68 1C A0 41 00 }
	condition:
		$1 at pe.entry_point
}

rule frusion {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Frusion"
	strings:
		$1 = { 83 EC 0C 53 55 56 57 68 04 01 00 00 C7 44 24 14 }
	condition:
		$1 at pe.entry_point
}

rule fsg_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
	strings:
		$1 = { ?? ?? ?? ?? ?? 81 C2 F1 4F 53 05 52 52 81 C2 FC 04 00 00 89 D1 5A E8 12 00 00 00 05 44 34 67 55 29 02 C1 02 08 83 C2 04 39 D1 75 EA C3 }
	condition:
		$1 at pe.entry_point
}

rule fsg_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
	strings:
		$1 = { 8D ?? ?? ?? ?? 00 00 BA ?? ?? ?? ?? 81 C2 ?? ?? ?? ?? 52 52 81 C2 1C 05 00 00 89 D1 5A 6A ?? 6A ?? 6A ?? E8 ?? 00 00 00 05 ?? ?? ?? ?? 31 02 C1 02 }
	condition:
		$1 at pe.entry_point
}

rule fsg_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.00"
	strings:
		$1 = { BB D0 01 40 00 BF 00 10 40 00 BE ?? ?? ?? ?? 53 E8 0A 00 00 00 02 D2 75 05 8A 16 46 12 D2 C3 FC B2 80 A4 6A 02 5B FF 14 24 73 F7 33 C9 FF 14 24 73 18 33 C0 FF 14 24 73 21 B3 02 41 B0 10 FF 14 24 12 C0 73 F9 75 3F AA EB DC E8 43 00 00 00 2B CB 75 10 E8 38 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_asm {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MASM32, TASM32"
	strings:
		$1 = { 03 F7 23 FE 33 FB EB 02 CD 20 BB 80 ?? 40 00 EB 01 86 EB 01 90 B8 F4 00 00 00 83 EE 05 2B F2 81 F6 EE 00 00 00 EB 02 CD 20 8A 0B E8 02 00 00 00 A9 54 5E C1 EE 07 F7 D7 EB 01 DE 81 E9 B7 96 A0 C4 EB 01 6B EB 02 CD 20 80 E9 4B C1 CF 08 EB 01 71 80 E9 1C EB }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_bdelhpi_msvc_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland Delphi, MSVC"
	strings:
		$1 = { C1 C8 10 EB 01 0F BF 03 74 66 77 C1 E9 1D 68 83 ?? ?? 77 EB 02 CD 20 5E EB 02 CD 20 2B F7 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_bdelhpi_msvc_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland Delphi, MSVC"
	strings:
		$1 = { 1B DB E8 02 00 00 00 1A 0D 5B 68 80 ?? ?? 00 E8 01 00 00 00 EA 5A 58 EB 02 CD 20 68 F4 00 00 00 EB 02 CD 20 5E 0F B6 D0 80 CA 5C 8B 38 EB 01 35 EB 02 DC 97 81 EF F7 65 17 43 E8 02 00 00 00 97 CB 5B 81 C7 B2 8B A1 0C 8B D1 83 EF 17 EB 02 0C 65 83 EF 43 13 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_borland_cpp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland C++"
	strings:
		$1 = { 23 CA EB 02 5A 0D E8 02 00 00 00 6A 35 58 C1 C9 10 BE 80 ?? ?? 00 0F B6 C9 EB 02 CD 20 BB F4 00 00 00 EB 02 04 FA EB 01 FA EB 01 5F EB 02 CD 20 8A 16 EB 02 11 31 80 E9 31 EB 02 30 11 C1 E9 11 80 EA 04 EB 02 F0 EA 33 CB 81 EA AB AB 19 08 04 D5 03 C2 80 EA }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_borland_cpp_1999 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland C++ 1999"
	strings:
		$1 = { EB 02 CD 20 2B C8 68 80 ?? ?? 00 EB 02 1E BB 5E EB 02 CD 20 68 B1 2B 6E 37 40 5B 0F B6 C9 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_borland_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland Delphi or C++"
	strings:
		$1 = { 2B C2 E8 02 00 00 00 95 4A 59 8D 3D 52 F1 2A E8 C1 C8 1C BE 2E ?? ?? 18 EB 02 AB A0 03 F7 EB 02 CD 20 68 F4 00 00 00 0B C7 5B 03 CB 8A 06 8A 16 E8 02 00 00 00 8D 46 59 EB 01 A4 02 D3 EB 02 CD 20 02 D3 E8 02 00 00 00 57 AB 58 81 C2 AA 87 AC B9 0F BE C9 80 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_borland_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland Delphi or C++"
	strings:
		$1 = { EB 01 2E EB 02 A5 55 BB 80 ?? ?? 00 87 FE 8D 05 AA CE E0 63 EB 01 75 BA 5E CE E0 63 EB 02 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_lcc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 4.x, LCC Win32 1.x"
	strings:
		$1 = { 2C 71 1B CA EB 01 2A EB 01 65 8D 35 80 ?? ?? 00 80 C9 84 80 C9 68 BB F4 00 00 00 EB 01 EB }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_50_60 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 5.0, 6.0"
	strings:
		$1 = { 33 D2 0F BE D2 EB 01 C7 EB 01 D8 8D 05 80 ?? ?? ?? EB 02 CD 20 EB 01 F8 BE F4 00 00 00 EB }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { C1 CE 10 C1 F6 0F 68 00 ?? ?? 00 2B FA 5B 23 F9 8D 15 80 ?? ?? 00 E8 01 00 00 00 B6 5E 0B }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { D1 E9 03 C0 68 80 ?? ?? 00 EB 02 CD 20 5E 40 BB F4 00 00 00 33 CA 2B C7 0F B6 16 EB 01 3E }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { E8 01 00 00 00 0E 59 E8 01 00 00 00 58 58 BE 80 ?? ?? 00 EB 02 61 E9 68 F4 00 00 00 C1 C8 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { 03 DE EB 01 F8 B8 80 ?? 42 00 EB 02 CD 20 68 17 A0 B3 AB EB 01 E8 59 0F B6 DB 68 0B A1 B3 AB EB 02 CD 20 5E 80 CB AA 2B F1 EB 02 CD 20 43 0F BE 38 13 D6 80 C3 47 2B FE EB 01 F4 03 FE EB 02 4F 4E 81 EF 93 53 7C 3C 80 C3 29 81 F7 8A 8F 67 8B 80 C3 C7 2B FE }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { EB 01 4D 83 F6 4C 68 80 ?? ?? 00 EB 02 CD 20 5B EB 01 23 68 48 1C 2B 3A E8 02 00 00 00 38 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { EB 02 AB 35 EB 02 B5 C6 8D 05 80 ?? ?? 00 C1 C2 11 BE F4 00 00 00 F7 DB F7 DB 0F BE 38 E8 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_07 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { EB 02 CD 20 ?? CF ?? ?? 80 ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_08 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { 91 EB 02 CD 20 BF 50 BC 04 6F 91 BE D0 ?? ?? 6F EB 02 CD 20 2B F7 EB 02 F0 46 8D 1D F4 00 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_09 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0"
	strings:
		$1 = { F7 D0 EB 02 CD 20 BE BB 74 1C FB EB 02 CD 20 BF 3B ?? ?? FB C1 C1 03 33 F7 EB 02 CD 20 68 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_70_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0, 7.0"
	strings:
		$1 = { F7 D8 40 49 EB 02 E0 0A 8D 35 80 ?? ?? ?? 0F B6 C2 EB 01 9C 8D 1D F4 00 00 00 EB 01 3C 80 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_70_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0, 7.0"
	strings:
		$1 = { 87 FE E8 02 00 00 00 98 CC 5F BB 80 ?? ?? 00 EB 02 CD 20 68 F4 00 00 00 E8 01 00 00 00 E3 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_70_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0, 7.0"
	strings:
		$1 = { F7 DB 80 EA BF B9 2F 40 67 BA EB 01 01 68 AF ?? ?? BA 80 EA 9D 58 C1 C2 09 2B C1 8B D7 68 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvc_60_70_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0, 7.0"
	strings:
		$1 = { 0B D0 8B DA E8 02 00 00 00 40 A0 5A EB 01 9D B8 80 ?? ?? 00 EB 02 CD 20 03 D3 8D 35 F4 00 00 00 EB 01 35 EB 01 88 80 CA 7C 80 F3 74 8B 38 EB 02 AC BA 03 DB E8 01 00 00 00 A5 5B C1 C2 0B 81 C7 DA 10 0A 4E EB 01 08 2B D1 83 EF 14 EB 02 CD 20 33 D3 83 EF 27 }
	condition:
		$1 at pe.entry_point
}

rule fsg_msvc_60_70_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MSVC 6.0, 7.0"
	strings:
		$1 = { E8 01 00 00 00 5A 5E E8 02 00 00 00 BA DD 5E 03 F2 EB 01 64 BB 80 ?? ?? 00 8B FA EB 01 A8 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_winrar_sfx_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "WinRAR SFX"
	strings:
		$1 = { 80 E9 A1 C1 C1 13 68 E4 16 75 46 C1 C1 05 5E EB 01 9D 68 64 86 37 46 EB 02 8C E0 5F F7 D0 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_winrar_sfx_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "WinRAR SFX"
	strings:
		$1 = { EB 01 02 EB 02 CD 20 B8 80 ?? 42 00 EB 01 55 BE F4 00 00 00 13 DF 13 D8 0F B6 38 D1 F3 F7 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_msvb_50_60 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Microsoft Visual Basic 5.0, 6.0"
	strings:
		$1 = { C1 CB 10 EB 01 0F B9 03 74 F6 EE 0F B6 D3 8D 05 83 ?? ?? EF 80 F3 F6 2B C1 EB 01 DE 68 77 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_borland_delphi_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Borland Delphi 2.0"
	strings:
		$1 = { EB 01 56 E8 02 00 00 00 B2 D9 59 68 80 ?? 41 00 E8 02 00 00 00 65 32 59 5E EB 02 CD 20 BB }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_masm32 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "MASM32"
	strings:
		$1 = { EB 01 DB E8 02 00 00 00 86 43 5E 8D 1D D0 75 CF 83 C1 EE 1D 68 50 ?? 8F 83 EB 02 3D 0F 5A }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_mvb_masm32 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Microsoft Visual Basic, MASM32"
	strings:
		$1 = { EB 02 09 94 0F B7 FF 68 80 ?? ?? 00 81 F6 8E 00 00 00 5B EB 02 11 C2 8D 05 F4 00 00 00 47 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_watcom {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
		description = "Watcom C/C++"
	strings:
		$1 = { EB 02 CD 20 03 ?? 8D ?? 80 ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? EB 02 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
	strings:
		$1 = { BB D0 01 40 ?? BF ?? 10 40 ?? BE ?? ?? ?? ?? FC B2 80 8A 06 46 88 07 47 02 D2 75 05 8A 16 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
	strings:
		$1 = { EB 02 CD 20 EB 02 CD 20 EB 02 CD 20 C1 E6 18 BB 80 ?? ?? 00 EB 02 82 B8 EB 01 10 8D 05 F4 }
	condition:
		$1 at pe.entry_point
}

rule fsg_110_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.10"
	strings:
		$1 = { F7 D8 0F BE C2 BE 80 ?? ?? 00 0F BE C9 BF 08 3B 65 07 EB 02 D8 29 BB EC C5 9A F8 EB 01 94 }
	condition:
		$1 at pe.entry_point
}

rule fsg_120_delphi_msvc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.20"
		description = "Borland Delphi, MSVC"
	strings:
		$1 = { 0F B6 D0 E8 01 00 00 00 0C 5A B8 80 ?? ?? 00 EB 02 00 DE 8D 35 F4 00 00 00 F7 D2 EB 02 0E EA 8B 38 EB 01 A0 C1 F3 11 81 EF 84 88 F4 4C EB 02 CD 20 83 F7 22 87 D3 33 FE C1 C3 19 83 F7 26 E8 02 00 00 00 BC DE 5A 81 EF F7 EF 6F 18 EB 02 CD 20 83 EF 7F EB 01 F7 2B FE EB 01 7F 81 EF DF 30 90 1E EB 02 CD 20 87 FA 88 10 80 EA 03 40 EB 01 20 4E EB 01 3D 83 FE 00 75 A2 EB 02 CD 20 EB 01 C3 78 73 42 F7 35 6C 2D 3F ED 33 97 ?? ?? ?? 5D F0 45 29 55 57 55 71 63 02 72 E9 1F 2D 67 B1 C0 91 FD 10 58 A3 90 71 6C }
	condition:
		$1 at pe.entry_point
}

rule fsg_120_borland {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.20"
		description = "Borland Delphi or C++"
	strings:
		$1 = { 0F BE C1 EB 01 0E 8D 35 C3 BE B6 22 F7 D1 68 43 ?? ?? 22 EB 02 B5 15 5F C1 F1 15 33 F7 80 E9 F9 BB F4 00 00 00 EB 02 8F D0 EB 02 08 AD 8A 16 2B C7 1B C7 80 C2 7A 41 80 EA 10 EB 01 3C 81 EA CF AE F1 AA EB 01 EC 81 EA BB C6 AB EE 2C E3 32 D3 0B CB 81 EA AB EE 90 14 2C 77 2A D3 EB 01 87 2A D3 E8 01 00 00 00 92 59 88 16 EB 02 52 08 46 EB 02 CD 20 4B 80 F1 C2 85 DB 75 AE C1 E0 04 EB 00 DA B2 82 5C 9B C7 89 98 4F 8A F7 ?? ?? ?? B1 4D DF B8 AD AC AB D4 07 27 D4 50 CF 9A D5 1C EC F2 27 77 18 40 4E A4 A8 }
	condition:
		$1 at pe.entry_point
}

rule fsg_120_borland_cpp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.20"
		description = "Borland C++"
	strings:
		$1 = { C1 F0 07 EB 02 CD 20 BE 80 ?? ?? 00 1B C6 8D 1D F4 00 00 00 0F B6 06 EB 02 CD 20 8A 16 0F B6 C3 E8 01 00 00 00 DC 59 80 EA 37 EB 02 CD 20 2A D3 EB 02 CD 20 80 EA 73 1B CF 32 D3 C1 C8 0E 80 EA 23 0F B6 C9 02 D3 EB 01 B5 02 D3 EB 02 DB 5B 81 C2 F6 56 7B F6 EB 02 56 7B 2A D3 E8 01 00 00 00 ED 58 88 16 13 C3 46 EB 02 CD 20 4B EB 02 CD 20 2B C9 3B D9 75 A1 E8 02 00 00 00 D7 6B 58 EB 00 9E 96 6A 28 67 AB 69 54 03 3E 7F ?? ?? ?? 31 0D 63 44 35 38 37 18 87 9F 10 8C 37 C6 41 80 4C 5E 8B DB 60 4C 3A 28 08 }
	condition:
		$1 at pe.entry_point
}

rule fsg_120_asm {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.20"
		description = "MASM32, TASM32"
	strings:
		$1 = { 33 C2 2C FB 8D 3D 7E 45 B4 80 E8 02 00 00 00 8A 45 58 68 02 ?? 8C 7F EB 02 CD 20 5E 80 C9 16 03 F7 EB 02 40 B0 68 F4 00 00 00 80 F1 2C 5B C1 E9 05 0F B6 C9 8A 16 0F B6 C9 0F BF C7 2A D3 E8 02 00 00 00 99 4C 58 80 EA 53 C1 C9 16 2A D3 E8 02 00 00 00 9D CE 58 80 EA 33 C1 E1 12 32 D3 48 80 C2 26 EB 02 CD 20 88 16 F7 D8 46 EB 01 C0 4B 40 8D 0D 00 00 00 00 3B D9 75 B7 EB 01 14 EB 01 0A CF C5 93 53 90 DA 96 67 54 8D CC ?? ?? 51 8E 18 74 53 82 83 80 47 B4 D2 41 FB 64 31 6A AF 7D 89 BC 0A 91 D7 83 37 39 }
	condition:
		$1 at pe.entry_point
}

rule fsg_120_msvc_60 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.20"
		description = "MSVC 6.0"
	strings:
		$1 = { C1 E0 06 EB 02 CD 20 EB 01 27 EB 01 24 BE 80 ?? 42 00 49 EB 01 99 8D 1D F4 00 00 00 EB 01 5C F7 D8 1B CA EB 01 31 8A 16 80 E9 41 EB 01 C2 C1 E0 0A EB 01 A1 81 EA A8 8C 18 A1 34 46 E8 01 00 00 00 62 59 32 D3 C1 C9 02 EB 01 68 80 F2 1A 0F BE C9 F7 D1 2A D3 }
	condition:
		$1 at pe.entry_point
}

rule fsg_120_msvc_60_70 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.20"
		description = "MSVC 6.0, 7.0"
	strings:
		$1 = { EB 02 CD 20 EB 01 91 8D 35 80 ?? ?? 00 33 C2 68 83 93 7E 7D 0C A4 5B 23 C3 68 77 93 7E 7D EB 01 FA 5F E8 02 00 00 00 F7 FB 58 33 DF EB 01 3F E8 02 00 00 00 11 88 58 0F B6 16 EB 02 CD 20 EB 02 86 2F 2A D3 EB 02 CD 20 80 EA 2F EB 01 52 32 D3 80 E9 CD 80 EA 73 8B CF 81 C2 96 44 EB 04 EB 02 CD 20 88 16 E8 02 00 00 00 44 A2 59 46 E8 01 00 00 00 AD 59 4B 80 C1 13 83 FB 00 75 B2 F7 D9 96 8F 80 4D 0C 4C 91 50 1C 0C 50 8A ?? ?? ?? 50 E9 34 16 50 4C 4C 0E 7E 9B 49 C6 32 02 3E 7E 7B 5E 8C C5 6B 50 3F 0E 0F }
	condition:
		$1 at pe.entry_point
}

rule fsg_120 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		description = "1.20"
	strings:
		$1 = { 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 ?? 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule fsg_130 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.30"
	strings:
		$1 = { BB D0 01 40 00 BF 00 10 40 00 BE ?? ?? ?? ?? 53 E8 0A 00 00 00 02 D2 75 05 8A 16 46 12 D2 C3 B2 80 A4 6A 02 5B FF 14 24 73 F7 33 C9 FF 14 24 73 18 33 C0 FF 14 24 73 21 B3 02 41 B0 10 FF 14 24 12 C0 73 F9 75 3F AA EB DC E8 43 00 00 00 2B CB 75 10 E8 38 00 00 00 EB 28 AC D1 E8 74 41 13 C9 EB 1C 91 48 C1 E0 08 AC E8 22 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 8B C5 B3 01 56 8B F7 2B F0 F3 A4 5E EB 96 33 C9 41 FF 54 24 04 13 C9 FF 54 24 04 72 F4 C3 5F 5B 0F B7 3B 4F 74 08 }
	condition:
		$1 at pe.entry_point
}

rule fsg_131_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.31"
	strings:
		$1 = { BB D0 01 40 00 BF 00 10 40 00 BE ?? ?? ?? ?? 53 BB ?? ?? ?? ?? B2 80 A4 B6 80 FF D3 73 F9 33 C9 FF D3 73 16 33 C0 FF D3 73 23 B6 80 41 B0 10 FF D3 12 C0 73 FA 75 42 AA EB E0 E8 46 00 00 00 02 F6 83 D9 01 75 10 E8 38 00 00 00 EB 28 AC D1 E8 74 48 13 C9 EB 1C 91 48 C1 E0 08 AC E8 22 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 8B C5 B6 00 56 8B F7 2B F0 F3 A4 5E EB 97 33 C9 41 FF D3 13 C9 FF D3 72 F8 C3 02 D2 75 05 8A 16 46 12 D2 C3 5B 5B 0F B7 3B 4F 74 08 4F 74 13 C1 E7 0C }
	condition:
		$1 at pe.entry_point
}

rule fsg_131_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.31"
	strings:
		$1 = { BE ?? ?? ?? 00 BF ?? ?? ?? 00 BB ?? ?? ?? 00 53 BB ?? ?? ?? 00 B2 80 }
	condition:
		$1 at pe.entry_point
}

rule fsg_133 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "1.33"
	strings:
		$1 = { BE A4 01 40 00 AD 93 AD 97 AD 56 96 B2 80 A4 B6 80 FF 13 73 F9 33 C9 FF 13 73 16 33 C0 FF 13 73 1F B6 80 41 B0 10 FF 13 12 C0 73 FA 75 3C AA EB E0 FF 53 08 02 F6 83 D9 01 75 0E FF 53 04 EB 26 AC D1 E8 74 2F 13 C9 EB 1A 91 48 C1 E0 08 AC FF 53 04 3D 00 7D }
	condition:
		$1 at pe.entry_point
}

rule fsg_200 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "FSG"
		version = "2.00"
	strings:
		$1 = { 87 25 ?? ?? ?? 00 61 94 55 A4 B6 80 FF 13 73 F9 33 C9 FF 13 73 16 33 C0 FF 13 73 1F B6 80 41 B0 10 FF 13 12 C0 73 FA 75 3A AA EB E0 }
	condition:
		$1 at pe.entry_point
}

rule fucknjoy_10c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Fuck'n'Joy"
		version = "1.0c"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED D8 05 40 00 FF 74 24 20 E8 8C 02 00 00 0B C0 0F 84 2C 01 00 00 89 85 6C 08 40 00 8D 85 2F 08 40 00 50 FF B5 6C 08 40 00 E8 EF 02 00 00 0B C0 0F 84 0C 01 00 00 89 85 3B 08 40 00 8D 85 3F 08 40 00 50 FF B5 6C 08 40 00 E8 CF 02 00 }
	condition:
		$1 at pe.entry_point
}

rule fusion_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Fusion"
		version = "1.0"
	strings:
		$1 = { 68 04 30 40 00 68 04 30 40 00 E8 09 03 00 00 68 04 30 40 00 E8 C7 02 00 00 }
	condition:
		$1 at pe.entry_point
}

rule gameguard_20065xx_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "GameGuard"
		version = "2006.5.x.x"
	strings:
		$1 = { 31 FF 74 06 61 E9 4A 4D 50 30 BA 4C 00 00 00 80 7C 24 08 01 0F 85 ?? 01 00 00 60 BE 00 }
	condition:
		$1 at pe.entry_point
}

rule gameguard_20065xx_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "GameGuard"
		version = "2006.5.x.x"
	strings:
		$1 = { 31 FF 74 06 61 E9 4A 4D 50 30 5A BA 7D 00 00 00 80 7C 24 08 01 E9 00 00 00 00 60 BE 00 }
	condition:
		$1 at pe.entry_point
}

rule gamehouse_media_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Gamehouse Media Protector"
	strings:
		$1 = { 68 ?? ?? ?? ?? 6A 00 FF 15 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ghf_protector_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "GHF Protector"
	strings:
		$1 = { 60 68 ?? ?? ?? ?? B8 ?? ?? ?? ?? FF 10 68 ?? ?? ?? ?? 50 B8 ?? ?? ?? ?? FF 10 68 00 A0 00 00 6A 40 FF D0 89 05 ?? ?? ?? ?? 89 C7 BE ?? ?? ?? ?? 60 FC B2 80 31 DB A4 B3 02 E8 6D 00 00 00 73 F6 }
	condition:
		$1 at pe.entry_point
}

rule ghf_protector_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "GHF Protector"
	strings:
		$1 = { 60 68 ?? ?? ?? ?? B8 ?? ?? ?? ?? FF 10 68 ?? ?? ?? ?? 50 B8 ?? ?? ?? ?? FF 10 68 00 00 00 00 6A 40 FF D0 89 05 ?? ?? ?? ?? 89 C7 BE ?? ?? ?? ?? 60 FC B2 80 31 DB A4 B3 02 E8 6D 00 00 00 73 F6 31 C9 E8 64 00 00 00 73 1C 31 C0 E8 5B 00 00 00 73 23 B3 02 41 }
	condition:
		$1 at pe.entry_point
}

rule goatrs_pe_mutilator_16 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Goat's PE Mutilator"
		version = "1.6"
	strings:
		$1 = { E8 EA 0B 00 00 ?? ?? ?? 8B 1C 79 F6 63 D8 8D 22 B0 BF F6 49 08 C3 02 BD 3B 6C 29 46 13 28 5D 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0F 53 0F DE 0F 55 0F 60 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule gix_protector_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "G!X Protector"
		version = "1.2"
	strings:
		$1 = { 60 EB 05 E8 EB 04 40 00 EB FA E8 0A 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule hardlock_dongle {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Hardlock dongle"
	strings:
		$1 = { 5C 5C 2E 5C 48 41 52 44 4C 4F 43 4B 2E 56 58 44 00 00 00 00 5C 5C 2E 5C 46 45 6E 74 65 44 65 76 }
	condition:
		$1 at pe.entry_point
}

rule hasp_dongle {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "HASP Dongle"
	strings:
		$1 = { 50 53 51 52 57 56 8B 75 1C 8B 3E ?? ?? ?? ?? ?? 8B 5D 08 8A FB ?? ?? 03 5D 10 8B 45 0C 8B 4D 14 8B 55 18 80 FF 32 }
	condition:
		$1 at pe.entry_point
}

rule hasp_protection_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "HASP Protection"
	strings:
		$1 = { 6A ?? 60 2E FF 35 ?? ?? ?? ?? 2E FF 35 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 66 83 C4 ?? 2E FF 35 ?? ?? ?? ?? 2E FF 35 ?? ?? ?? ?? B8 ?? ?? ?? ?? 83 C0 ?? 50 }
	condition:
		$1 at pe.entry_point
}

rule hasp_protection_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "HASP Protection"
	strings:
		$1 = { 55 8B EC 53 56 57 60 8B C4 A3 50 ?? ?? ?? B8 90 ?? ?? ?? 2B 05 B0 ?? ?? ?? A3 B0 ?? ?? ?? 83 3D 4C ?? ?? ?? 00 0F 84 11 00 00 00 A1 50 ?? ?? ?? 50 FF 15 4C ?? ?? ?? E9 69 00 00 00 C7 05 70 }
	condition:
		$1 at pe.entry_point
}

rule hasp_protection_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "HASP Protection"
		version = "1.x"
	strings:
		$1 = { 55 8B EC 53 56 57 60 8B C4 A3 ?? ?? ?? ?? B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? 83 3D ?? ?? ?? ?? 00 74 15 8B 0D ?? ?? ?? ?? 51 FF 15 ?? ?? ?? ?? 83 C4 04 E9 A5 00 00 00 68 }
	condition:
		$1 at pe.entry_point
}

rule hide_pe_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Hide PE"
		version = "1.01"
	strings:
		$1 = { ?? BA ?? ?? ?? 00 B8 ?? ?? ?? ?? 89 02 83 C2 04 B8 ?? ?? ?? ?? 89 02 83 C2 04 B8 ?? ?? ?? ?? 89 02 83 C2 F8 FF E2 0D 0A 2D 3D 5B 20 48 69 64 65 50 45 20 62 79 20 42 47 43 6F 72 70 20 5D 3D 2D }
	condition:
		$1 at pe.entry_point
}

rule hide_protect_1016c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Hide&Protect"
		version = "1.016C"
	strings:
		$1 = { 90 90 90 E9 D8 }
	condition:
		$1 at pe.entry_point
}

rule hmimys_protect_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "hmimys Protect"
		version = "1.0"
	strings:
		$1 = { E8 BA 00 00 00 ?? 00 00 00 00 ?? ?? 00 00 10 40 00 ?? ?? ?? 00 ?? ?? ?? 00 00 ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? 00 00 00 00 00 00 00 ?? ?? ?? 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 56 69 72 }
	condition:
		$1 at pe.entry_point
}

rule hmimys_pepack_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "hmimys's PE-Pack"
		version = "0.1"
	strings:
		$1 = { E8 00 00 00 00 5D 83 ED 05 6A 00 FF 95 E1 0E 00 00 89 85 85 0E 00 00 8B 58 3C 03 D8 81 C3 F8 00 00 00 80 AD 89 0E 00 00 01 89 9D 63 0F 00 00 8B 4B 0C 03 8D 85 0E 00 00 8B 53 08 80 BD 89 0E 00 00 00 75 0C 03 8D 91 0E 00 00 2B 95 91 0E 00 00 89 8D 57 0F 00 00 89 95 5B 0F 00 00 8B 5B 10 89 9D 5F 0F 00 00 8B 9D 5F 0F 00 00 8B 85 57 0F 00 00 53 50 E8 B7 0B 00 00 89 85 73 0F 00 00 6A 04 68 00 10 00 00 50 6A 00 FF 95 E9 0E 00 00 89 85 6B 0F 00 00 6A 04 68 00 10 00 00 68 D8 7C 00 00 6A 00 FF 95 E9 0E 00 00 89 85 6F 0F 00 00 8D 85 67 0F 00 00 8B 9D 73 0F 00 00 8B 8D 6B 0F 00 00 8B 95 5B 0F 00 00 83 EA 0E 8B B5 57 0F 00 00 83 C6 0E 8B BD 6F 0F 00 00 50 53 51 52 56 68 D8 7C 00 00 57 E8 01 01 00 00 8B 9D 57 0F 00 00 8B 03 3C 01 75 }
	condition:
		$1 at pe.entry_point
}

rule hmimys_packer_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "hmimys's Packer"
		version = "1.0"
	strings:
		$1 = { 5E 83 C6 64 AD 50 AD 50 83 EE 6C AD 50 AD 50 AD 50 AD 50 AD 50 E8 E7 07 }
	condition:
		$1 at pe.entry_point
}

rule hmimys_packer_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "hmimys's Packer"
		version = "1.2"
	strings:
		$1 = { E8 95 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 5E AD 50 AD 50 97 AD 50 AD 50 AD 50 E8 C0 01 00 00 AD 50 AD 93 87 DE B9 ?? ?? ?? ?? E3 1D 8A 07 47 04 ?? 3C ?? 73 F7 8B 07 3C ?? 75 F3 B0 00 0F C8 05 ?? ?? ?? ?? 2B C7 AB E2 E3 AD 85 C0 74 2B 97 56 FF 13 8B E8 AC 84 C0 75 FB 66 AD 66 85 C0 74 E9 AC 83 EE 03 84 C0 74 08 56 55 FF 53 04 AB EB E4 AD 50 55 FF 53 04 AB EB E0 C3 8B 0A 3B 4A 04 75 0A C7 42 10 01 00 00 00 0C FF C3 }
	condition:
		$1 at pe.entry_point
}

rule hpa {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "HPA"
	strings:
		$1 = { E8 ?? ?? 5E 8B D6 83 ?? ?? 83 ?? ?? 06 0E 1E 0E 1F 33 FF 8C D3 }
	condition:
		$1 at pe.entry_point
}

rule icebergLock_protector_3101x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "IcebergLock Protector"
		version = "3.10.1.36, 3.10.1.41"
	strings:
		$1 = { E8 D7 FF FF FF ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 8B EC 33 C0 55 ?? ?? ?? ?? ?? 64 FF 30 64 89 20 33 C0 5A 59 59 64 89 10 ?? ?? ?? ?? ?? C3 E9 ?? ?? ?? ?? EB F8 5D C3 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8B EC 83 ?? ?? B8 ?? ?? ?? ?? E8 ?? ?? FD FF E8 ?? ?? FF FF B8 ?? ?? ?? ?? E8 71 FE FF FF E8 ?? ?? FD FF }
	condition:
		$1 at pe.entry_point
}

rule icrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ICrypt"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 56 57 33 C0 89 45 EC B8 70 3B 00 10 E8 3C FA FF FF 33 C0 55 68 6C 3C 00 10 64 FF 30 64 89 20 6A 0A 68 7C 3C 00 10 A1 50 56 00 10 50 E8 D8 FA FF FF 8B D8 53 A1 50 56 00 10 50 E8 0A FB FF FF 8B F8 53 A1 50 56 00 10 50 E8 D4 FA FF FF 8B D8 53 E8 D4 FA FF FF 8B F0 85 F6 74 26 8B D7 4A B8 64 56 00 10 E8 25 F6 FF FF B8 64 56 00 10 E8 13 F6 FF FF 8B CF 8B D6 E8 E6 FA FF FF 53 E8 90 FA FF FF 8D 4D EC BA 8C 3C 00 10 A1 64 56 00 10 E8 16 FB FF FF 8B 55 EC B8 64 56 00 10 E8 C5 F4 FF FF B8 64 56 00 10 E8 DB F5 FF FF E8 56 FC FF FF 33 C0 5A 59 59 64 89 10 68 73 3C 00 10 8D 45 EC E8 4D F4 FF FF C3 E9 E3 EE FF FF EB F0 5F 5E 5B E8 4D F3 FF FF 00 53 45 54 ?? ?? ?? ?? 00 FF FF FF FF 08 00 00 00 76 6F 74 72 65 63 6C 65 }
	condition:
		$1 at pe.entry_point
}

rule id_application_protector_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ID Application Protector"
		version = "1.2"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED F2 0B 47 00 B9 19 22 47 00 81 E9 EA 0E 47 00 89 EA 81 C2 EA 0E 47 00 8D 3A 89 FE 31 C0 E9 D3 02 00 00 CC CC CC CC E9 CA 02 00 00 43 3A 5C 57 69 6E 64 6F 77 73 5C 53 6F 66 74 57 61 72 65 50 72 6F 74 65 63 74 6F 72 5C }
	condition:
		$1 at pe.entry_point
}

rule ilucrypt_4015 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "iLUCRYPT"
		version = "4.015"
	strings:
		$1 = { 8B EC FA C7 46 F7 ?? ?? 42 81 FA ?? ?? 75 F9 FF 66 F7 }
	condition:
		$1 at pe.entry_point
}

rule imp_packer_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "IMP-Packer"
		version = "1.0"
	strings:
		$1 = { 28 ?? ?? ?? 00 00 00 00 00 00 00 00 40 ?? ?? ?? 34 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4C ?? ?? ?? 5C ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 }
	condition:
		$1 at pe.entry_point
}

rule imploder_104 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Imploder"
		version = "1.04"
	strings:
		$1 = { 60 E8 A0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 36 ?? ?? ?? 2E ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 80 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 44 }
	condition:
		$1 at pe.entry_point
}

rule impostor_pack_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "IMPostor Pack"
		version = "1.0"
	strings:
		$1 = { BE ?? ?? ?? 00 83 C6 01 FF E6 00 00 00 00 ?? ?? 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 ?? 02 ?? ?? 00 10 00 00 00 02 00 }
	condition:
		$1 at pe.entry_point
}

rule inbuild_10_hard {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Inbuild"
		version = "1.0 hard"
	strings:
		$1 = { B9 ?? ?? BB ?? ?? 2E ?? ?? 2E ?? ?? 43 E2 }
	condition:
		$1 at pe.entry_point
}

rule incrypter_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "INCrypter"
		version = "0.3 INinY"
	strings:
		$1 = { 60 64 A1 30 00 00 00 8B 40 0C 8B 40 0C 8D 58 20 C7 03 00 00 00 00 E8 00 00 00 00 5D 81 ED 4D 16 40 00 8B 9D 0E 17 40 00 64 A1 18 00 00 00 8B 40 30 0F B6 40 02 83 F8 01 75 05 03 DB C1 CB 10 8B 8D 12 17 40 00 8B B5 06 17 40 00 51 81 3E 2E 72 73 72 74 65 8B 85 16 17 40 00 E8 23 00 00 00 8B 85 1A 17 40 00 E8 18 00 00 00 8B 85 1E 17 40 00 E8 0D 00 00 00 8B 85 22 17 40 00 E8 02 00 00 00 EB 18 8B D6 3B 46 0C 72 0A 83 F9 01 74 0B 3B 46 34 72 06 BA 00 00 00 00 C3 58 83 FA 00 75 1A 8B 4E 10 8B 7E 0C 03 BD 02 17 40 00 83 F9 00 74 09 F6 17 31 0F 31 1F 47 E2 F7 59 83 C6 28 49 83 F9 00 75 88 8B 85 0A 17 40 00 89 44 24 1C 61 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule interlok_551 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "InterLok"
		version = "5.51"
	strings:
		$1 = { EB 03 ?? ?? ?? 55 EB 03 ?? ?? ?? EB 04 ?? EB 06 ?? 8B EC EB F9 ?? EB 02 ?? ?? 81 EC A8 00 00 00 EB 02 ?? ?? EB 01 ?? 53 EB 03 ?? ?? ?? EB 05 ?? ?? EB 15 ?? EB 03 ?? ?? ?? 56 EB 04 ?? EB F2 ?? EB 01 ?? EB F8 ?? ?? ?? EB 0F ?? 33 F6 EB 10 ?? ?? ?? EB F7 ?? ?? EB FA ?? EB 01 ?? EB F8 ?? EB 01 ?? 57 EB 03 ?? ?? ?? EB 11 ?? ?? ?? EB 03 ?? ?? ?? ?? ?? ?? ?? ?? ?? EB 08 ?? EB F0 ?? EB 07 ?? ?? EB FA ?? ?? ?? EB 02 ?? ?? BB ?? ?? ?? ?? EB 03 ?? ?? ?? 0F 85 ?? ?? ?? ?? EB 07 }
	condition:
		$1 at pe.entry_point
}

rule interlok_5xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "InterLok"
		version = "5.xx"
	strings:
		$1 = { 55 8B EC 81 EC A4 00 00 00 53 56 33 F6 57 39 35 ?? ?? ?? ?? 75 53 8D 45 DC 6A 1C 50 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 85 C0 74 05 8B 45 E0 EB 22 8B 7D 08 6A 02 57 FF 15 ?? ?? ?? ?? 85 C0 75 0B 66 81 3F 4D 5A 75 04 8B C7 EB 07 56 FF 15 ?? ?? ?? ?? A3 ?? ?? ?? ?? A3 ?? ?? ?? ?? 8B 48 3C 03 C8 89 ?? ?? ?? ?? ?? EB 06 8B ?? ?? ?? ?? ?? 66 8B 59 16 C1 EB 0D 83 E3 01 74 0A 83 7D 0C 01 0F 85 38 01 00 00 8D 45 F8 50 8D 45 FC 50 E8 47 01 00 00 8B F8 59 3B FE 59 75 52 83 7D FC FF FF 75 F8 75 17 8D 85 5C FF FF FF 68 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? 83 C4 0C EB 18 FF 75 FC 8D 85 5C FF FF FF 68 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? 83 C4 10 6A 30 8D 85 5C FF FF FF 68 ?? ?? ?? ?? 50 56 FF 15 ?? ?? ?? ?? E9 BB 00 00 00 68 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? 57 FF D7 57 6A 01 8B F0 FF 15 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? 85 F6 0F 84 96 00 00 00 83 FE F6 7F 32 74 29 83 FE 97 74 75 83 FE F3 74 18 83 FE F4 74 0C 83 FE F5 75 2B B8 ?? ?? ?? ?? EB 4F B8 ?? ?? ?? ?? EB 48 B8 ?? ?? ?? ?? EB 41 B8 ?? ?? ?? ?? EB 3A 83 FE FA 74 30 83 FE FC 74 24 83 FE FD 74 18 56 8D 45 E0 68 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? 83 C4 0C 8D 45 E0 EB 13 B8 ?? ?? ?? ?? EB 0C B8 ?? ?? ?? ?? EB 05 B8 ?? ?? ?? ?? 6A 30 68 ?? ?? ?? ?? 50 6A 00 FF 15 ?? ?? ?? ?? 85 DB 75 08 6A 01 FF 15 ?? ?? ?? ?? 33 C0 5F 5E 5B C9 C2 0C 00 }
	condition:
		$1 at pe.entry_point
}

rule ionic_wind_sw {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ionic Wind Software"
	strings:
		$1 = { 9B DB E3 9B DB E2 D9 2D 00 ?? ?? 00 55 89 E5 E8 }
	condition:
		$1 at pe.entry_point
}

rule ipbprotect_013_017 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "iPBProtect"
		version = "0.1.3 - 0.1.7"
	strings:
		$1 = { 55 8B EC 6A FF 68 4B 43 55 46 68 54 49 48 53 64 A1 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ipbprotect_013 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "iPBProtect"
		version = "0.1.3"
	strings:
		$1 = { 55 8B EC 6A FF 68 4B 43 55 46 68 54 49 48 53 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 68 53 56 57 89 65 FA 33 DB 89 5D F8 6A 02 EB 01 F8 58 5F 5E 5B 64 8B 25 00 00 00 00 64 8F 05 00 00 00 00 58 58 58 5D 68 9F 6F 56 B6 50 E8 5D 00 00 00 EB FF 71 78 C2 50 00 EB D3 5B F3 68 89 5C 24 48 5C 24 58 FF 8D 5C 24 58 5B 83 C3 4C 75 F4 5A 8D 71 78 75 09 81 F3 EB FF 52 BA 01 00 83 EB FC 4A FF 71 0F 75 19 8B 5C 24 00 00 81 33 50 53 8B 1B 0F FF C6 75 1B 81 F3 EB 87 1C 24 8B 8B 04 24 83 EC FC EB 01 E8 83 }
	condition:
		$1 at pe.entry_point
}

rule iprotect_10_fxlib {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "IProtect"
		version = "1.0 fxlib.dll mode"
	strings:
		$1 = { EB 33 2E 46 55 58 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 46 78 4C 69 62 2E 64 6C 6C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 60 E8 00 00 00 00 5D 81 ED 71 10 40 00 FF 74 24 20 E8 40 00 00 00 0B C0 74 2F 89 85 63 10 40 00 8D 85 3C 10 40 00 50 FF B5 63 10 40 00 E8 92 00 00 00 0B C0 74 13 89 85 5F 10 40 00 8D 85 49 10 40 00 50 FF 95 5F 10 40 00 8B 85 67 10 40 00 89 44 24 1C 61 FF E0 8B 7C 24 04 8D 85 00 10 40 00 50 64 FF 35 00 00 00 00 8D 85 53 10 40 00 89 20 89 68 04 8D 9D 0A 11 40 00 89 58 08 64 89 25 00 00 00 00 81 E7 00 00 FF FF 66 81 3F 4D 5A 75 0F 8B F7 03 76 3C 81 3E 50 45 00 00 75 02 EB 17 81 EF 00 00 01 00 81 FF 00 00 00 70 73 07 BF 00 00 F7 BF EB 02 EB D3 97 64 8F 05 00 00 00 00 83 C4 04 C2 04 00 8D 85 00 10 40 00 50 64 FF 35 00 00 00 00 8D 85 53 10 40 00 89 20 89 68 04 8D 9D 0A 11 40 00 89 58 08 64 89 25 00 00 00 00 8B 74 24 0C 66 81 3E 4D 5A 74 05 E9 8A 00 00 00 03 76 3C 81 3E 50 45 00 00 74 02 EB 7D 8B 7C 24 10 B9 96 00 00 00 32 C0 F2 AE 8B CF 2B 4C 24 10 8B 56 78 03 54 24 0C 8B 5A 20 03 5C 24 0C 33 C0 8B 3B 03 7C 24 0C 8B 74 24 10 51 F3 A6 75 05 83 C4 04 EB 0A 59 83 C3 04 40 3B 42 18 75 E2 3B 42 18 75 02 EB 35 8B 72 24 03 74 24 0C 52 BB 02 00 00 00 33 D2 F7 E3 5A 03 C6 33 C9 66 8B 08 8B 7A 1C 33 D2 BB 04 00 00 00 8B C1 F7 E3 03 44 24 0C 03 C7 8B 00 03 44 24 0C EB 02 33 C0 64 8F 05 00 00 00 00 83 C4 04 C2 08 00 E8 FA FD FF FF }
	condition:
		$1 at pe.entry_point
}

rule iprotect_10_fxsub {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "IProtect"
		version = "1.0 fxsub.dll mode"
	strings:
		$1 = { EB 33 2E 46 55 58 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 46 78 53 75 62 2E 64 6C 6C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? 00 60 E8 00 00 00 00 5D 81 ED B6 13 40 00 FF 74 24 20 E8 40 00 00 00 0B C0 74 2F 89 85 A8 13 40 00 8D 85 81 13 40 00 50 FF B5 A8 13 40 00 E8 92 00 00 00 0B C0 74 13 89 85 A4 13 40 00 8D 85 8E 13 40 00 50 FF 95 A4 13 40 00 8B 85 AC 13 40 00 89 44 24 1C 61 FF E0 8B 7C 24 04 8D 85 00 10 40 00 50 64 FF 35 00 00 00 00 8D 85 98 13 40 00 89 20 89 68 04 8D 9D 4F 14 40 00 89 58 08 64 89 25 00 00 00 00 81 E7 00 00 FF FF 66 81 3F 4D 5A 75 0F 8B F7 03 76 3C 81 3E 50 45 00 00 75 02 EB 17 81 EF 00 00 01 00 81 FF 00 00 00 70 73 07 BF 00 00 F7 BF EB 02 EB D3 97 64 8F 05 00 00 00 00 83 C4 04 C2 04 00 8D 85 00 10 40 00 50 64 FF 35 00 00 00 00 8D 85 98 13 40 00 89 20 89 68 04 8D 9D 4F 14 40 00 89 58 08 64 89 25 00 00 00 00 8B 74 24 0C 66 81 3E 4D 5A 74 05 E9 8A 00 00 00 03 76 3C 81 3E 50 45 00 00 74 02 EB 7D 8B 7C 24 10 B9 96 00 00 00 32 C0 F2 AE 8B CF 2B 4C 24 10 8B 56 78 03 54 24 0C 8B 5A 20 03 5C 24 0C 33 C0 8B 3B 03 7C 24 0C 8B 74 24 10 51 F3 A6 75 05 83 C4 04 EB 0A 59 83 C3 04 40 3B 42 18 75 E2 3B 42 18 75 02 EB 35 8B 72 24 03 74 24 0C 52 BB 02 00 00 00 33 D2 F7 E3 5A 03 C6 33 C9 66 8B 08 8B 7A 1C 33 D2 BB 04 00 00 00 8B C1 F7 E3 03 44 24 0C 03 C7 8B 00 03 44 24 0C EB 02 33 C0 64 8F 05 00 00 00 00 83 C4 04 C2 08 00 E8 B5 FA FF FF }
	condition:
		$1 at pe.entry_point
}

rule java_loader_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JAVA Loader"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 48 24 05 01 64 A1 30 00 00 00 8B 40 0C 8B 70 1C AD 8B 40 08 89 85 76 2E 05 01 8D 9D 7E 2E 05 01 53 FF B5 76 2E 05 01 E8 04 02 00 00 89 85 21 2F 05 01 8D 9D 8B 2E 05 }
	condition:
		$1 at pe.entry_point
}

rule java_loader_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JAVA Loader"
	strings:
		$1 = { E8 ?? ?? ?? ?? 85 C0 75 10 6A 01 E8 ?? ?? ?? ?? 59 6A 01 FF 15 ?? ?? ?? ?? 33 C0 50 50 50 50 E8 D2 F8 FF FF C3 }
	condition:
		$1 at pe.entry_point
}

rule java_loader_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JAVA Loader"
	strings:
		$1 = { E8 ?? ?? ?? ?? 85 C0 75 10 6A 01 E8 ?? ?? ?? ?? 59 6A 01 FF 15 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? E8 D9 F8 FF FF 59 59 C3 }
	condition:
		$1 at pe.entry_point
}

rule jcpack_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JDPack"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 8B D5 81 ED ?? ?? ?? ?? 2B 95 ?? ?? ?? ?? 81 EA 06 ?? ?? ?? 89 95 ?? ?? ?? ?? 83 BD 45 }
	condition:
		$1 at pe.entry_point
}

rule jdpack_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JDPack"
		version = "2.0"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 ?? ?? ?? E8 01 00 00 00 ?? ?? ?? ?? ?? ?? 05 00 00 00 00 83 C4 0C 5D 60 E8 00 00 00 00 5D 8B D5 64 FF 35 00 00 00 00 EB }
	condition:
		$1 at pe.entry_point
}

rule jdpack_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JDPack"
		version = "2.x"
	strings:
		$1 = { 55 8B EC 6A FF 68 68 51 40 00 68 04 25 40 00 64 A1 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule jexecompressor_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "JExeCompressor"
		version = "1.0"
	strings:
		$1 = { 8D 2D D3 4A E5 14 0F BB F7 0F BA E5 73 0F AF D5 8D 0D 0C 9F E6 11 C0 F8 EF F6 DE 80 DC 5B F6 DA 0F A5 C1 0F C1 F1 1C F3 4A 81 E1 8C 1F 66 91 0F BE C6 11 EE 0F C0 E7 33 D9 64 F2 C0 DC 73 0F C0 D5 55 8B EC BA C0 1F 41 00 8B C2 B9 97 00 00 00 80 32 79 50 B8 02 00 00 00 50 03 14 24 58 58 51 2B C9 B9 01 00 00 00 83 EA 01 E2 FB 59 E2 E1 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule joiner {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Joiner"
	strings:
		$1 = { 81 EC 04 01 00 00 8B F4 68 04 01 00 00 56 6A 00 E8 7C 01 00 00 33 C0 6A 00 68 80 00 00 00 6A 03 6A 00 6A 00 68 00 00 00 80 56 E8 50 01 00 00 8B D8 6A 00 6A 00 6A 00 6A 02 6A 00 53 E8 44 01 }
	condition:
		$1 at pe.entry_point
}

rule kbys_022 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KByS"
		version = "0.22"
	strings:
		$1 = { 68 ?? ?? ?? ?? E8 01 00 00 00 C3 C3 11 55 07 8B EC B8 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule kbys_028b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KBys"
		version = "0.28b"
	strings:
		$1 = { 68 85 AE 01 01 E8 01 00 00 00 C3 C3 60 8B 74 24 24 8B 7C 24 28 FC B2 80 33 DB A4 B3 02 E8 6D 00 00 00 73 F6 33 C9 E8 64 00 00 00 73 1C 33 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule kbys_028b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KBys"
		version = "0.28b"
	strings:
		$1 = { 60 E8 00 00 00 00 5E 83 EE 0A 8B 06 03 C2 8B 08 89 4E F3 83 EE 0F 56 52 8B F0 AD AD 03 C2 8B D8 6A 04 BF 00 10 00 00 57 57 6A 00 FF 53 08 5A 59 BD 00 80 00 00 55 6A 00 50 51 52 50 89 06 AD AD 03 C2 50 AD 03 C2 FF D0 6A 04 57 AD 50 6A 00 FF 53 }
	condition:
		$1 at pe.entry_point
}

rule kbys_028 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KByS"
		version = "0.28"
	strings:
		$1 = { B8 ?? ?? ?? ?? BA ?? ?? ?? ?? 03 C2 FF E0 ?? ?? ?? ?? 60 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule kenpack_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KenPack"
		version = "0.3"
	strings:
		$1 = { 6A 18 E8 14 00 00 00 58 8D 4A 18 51 8D 92 ?? ?? ?? ?? 64 8B 08 FF 31 89 21 89 10 5A FF E2 72 8B 44 24 0C 8B ?? A8 00 00 00 8D 8A ?? ?? ?? ?? 60 }
	condition:
		$1 at pe.entry_point
}

rule kgcrypt {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KGCrypt"
	strings:
		$1 = { E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? 64 A1 30 ?? ?? ?? 84 C0 74 ?? 64 A1 20 ?? ?? ?? 0B C0 74 }
	condition:
		$1 at pe.entry_point
}

rule kkrunchy_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kkrunchy"
	strings:
		$1 = { BD 08 ?? ?? 00 C7 45 00 ?? ?? ?? 00 FF 4D 08 C6 45 0C 05 8D 7D 14 31 C0 B4 04 89 C1 F3 AB BF ?? ?? ?? 00 57 BE ?? ?? ?? 00 31 C9 41 FF 4D 0C 8D 9C 8D A0 00 00 00 FF D6 }
	condition:
		$1 at pe.entry_point
}

rule kkrunchy_017 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kkrunchy"
		version = "0.17"
	strings:
		$1 = { FC FF 4D 08 31 D2 8D 7D 30 BE }
	condition:
		$1 at pe.entry_point
}

rule kkrunchy_023a2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kkrunchy"
		version = "0.23a2"
	strings:
		$1 = { BD ?? ?? ?? ?? C7 45 00 ?? ?? ?? 00 B8 ?? ?? ?? 00 89 45 04 89 45 54 50 C7 45 10 ?? ?? ?? 00 FF 4D 0C FF 45 14 FF 45 58 C6 45 1C 08 B8 00 08 00 00 8D 7D 30 AB AB AB AB BB 00 00 D8 00 BF }
	condition:
		$1 at pe.entry_point
}

rule krypton_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Krypton"
		version = "0.2"
	strings:
		$1 = { 8B 0C 24 E9 0A 7C 01 ?? AD 42 40 BD BE 9D 7A 04 }
	condition:
		$1 at pe.entry_point
}

rule krypton_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Krypton"
		version = "0.3"
	strings:
		$1 = { 8B 0C 24 E9 C0 8D 01 ?? C1 3A 6E CA 5D 7E 79 6D B3 64 5A 71 EA }
	condition:
		$1 at pe.entry_point
}

rule krypton_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Krypton"
		version = "0.4"
	strings:
		$1 = { 54 E8 ?? ?? ?? ?? 5D 8B C5 81 ED 61 34 ?? ?? 2B 85 60 37 ?? ?? 83 E8 06 }
	condition:
		$1 at pe.entry_point
}

rule krypton_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Krypton"
		version = "0.5"
	strings:
		$1 = { 54 E8 ?? ?? ?? ?? 5D 8B C5 81 ED 71 44 ?? ?? 2B 85 64 60 ?? ?? EB 43 DF }
	condition:
		$1 at pe.entry_point
}

rule kryptor_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kryptor"
	strings:
		$1 = { EB 66 87 DB }
	condition:
		$1 at pe.entry_point
}
rule kryptor_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kryptor"
	strings:
		$1 = { EB 6A 87 DB }
	condition:
		$1 at pe.entry_point
}

rule kryptor_5 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kryptor"
		version = "5"
	strings:
		$1 = { E8 03 ?? ?? ?? E9 EB 6C 58 40 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule kryptor_6 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kryptor"
		version = "6"
	strings:
		$1 = { E8 03 ?? ?? ?? E9 EB 68 58 33 D2 74 02 E9 E9 40 42 75 02 }
	condition:
		$1 at pe.entry_point
}

rule kryptor_9 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "kryptor"
		version = "9"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5E B9 ?? ?? ?? ?? 2B C0 02 04 0E D3 C0 49 79 F8 41 8D 7E 2C 33 46 ?? 66 B9 }
	condition:
		$1 at pe.entry_point
}

rule lamecrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LameCrypt"
		version = "1.0"
	strings:
		$1 = { 60 66 9C BB ?? ?? ?? ?? 80 B3 00 10 40 00 90 4B 83 FB FF 75 F3 66 9D 61 }
	condition:
		$1 at pe.entry_point
}

rule larp_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LARP"
		version = "2.0"
	strings:
		$1 = { E8 01 00 00 00 81 E8 02 00 00 00 81 84 E8 EF 01 00 00 81 84 E8 01 00 00 00 64 E8 02 00 00 00 E8 81 E8 81 00 00 00 C3 81 84 E8 04 00 00 00 01 31 00 00 50 68 23 31 40 00 E8 A1 01 00 00 81 68 D7 17 40 00 3B D1 0F 87 32 04 00 00 0F 86 52 28 00 00 81 84 68 F1 17 40 00 85 C9 0F 85 84 28 00 00 0F 84 42 04 00 00 81 E8 D4 18 00 00 68 5B 50 E8 76 01 00 00 81 84 68 14 18 40 00 68 B3 2C 40 00 85 C0 0F 84 27 28 00 00 0F 85 FA 03 00 00 81 84 58 83 04 24 01 83 C4 04 0B E4 74 04 FF 64 24 FC 81 E8 4B 01 00 00 81 E8 01 00 00 00 84 E8 06 00 00 00 81 84 74 00 81 84 0B E4 74 ?? ?? ?? ?? ?? ?? 00 0B E4 74 02 FF E0 81 E8 00 00 00 00 68 ?? ?? ?? ?? E8 02 00 00 00 75 BA F8 72 02 73 02 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 E8 FA 00 00 00 81 84 0B E4 74 27 E8 EF 00 00 00 81 84 E8 01 00 00 00 50 E8 02 00 00 00 81 84 0B E4 E8 D9 00 00 00 81 84 74 08 ?? ?? ?? ?? ?? ?? FF E2 }
	condition:
		$1 at pe.entry_point
}

rule launch_anywhere_4001 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LaunchAnywhere"
		version = "4.0.0.1"
	strings:
		$1 = { 55 58 9E 55 38 3E C4 85 5B 8F FF FF FF F5 05 06 8E 03 E4 20 06 4F F3 50 00 00 00 06 48 92 50 00 00 00 06 8C 06 94 40 0E 8E 48 0F FF F5 9E 84 E2 90 00 0E 8C 90 D0 00 08 5C 07 50 86 AF FE 86 E2 B0 00 05 9E 8A 82 C0 00 0E 82 32 E0 00 0F F1 54 CC 24 40 08 9C 3E B1 93 C2 27 51 48 9C 08 D4 00 04 38 A0 38 4C 07 40 43 C2 27 5F 53 C2 27 50 14 38 A0 38 4C 07 40 B3 C2 07 40 73 C0 97 5D 9E B0 14 38 A0 38 4C 07 40 43 C2 07 EF 58 D4 5B 85 0F F1 5E 4C 14 40 08 B4 5E 42 50 10 00 00 07 40 60 FB 74 5E 8E B0 5B 80 A? }
	condition:
		$1 at pe.entry_point
}

rule launcher_generator_103 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Launcher Generator"
		version = "1.03"
	strings:
		$1 = { 68 00 20 40 00 68 10 20 40 00 6A 00 6A 00 6A 20 6A 00 6A 00 6A 00 68 F0 22 40 00 6A 00 E8 93 00 00 00 85 C0 0F 84 7E 00 00 00 B8 00 00 00 00 3B 05 68 20 40 00 74 13 6A ?? 68 60 23 40 00 68 20 23 40 00 6A 00 E8 83 00 00 00 A1 58 20 40 00 3B 05 6C 20 40 00 74 51 C1 E0 02 A3 5C 20 40 00 BB 70 21 40 00 03 C3 8B 18 68 60 20 40 00 53 B8 F0 21 40 00 03 05 5C 20 40 00 8B D8 8B 03 05 70 20 40 00 50 B8 70 22 40 00 03 05 5C 20 40 00 FF 30 FF 35 00 20 40 00 E8 26 00 00 00 A1 58 20 40 00 40 A3 58 20 40 00 EB }
	condition:
		$1 at pe.entry_point
}

rule lock98_10028 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LOCK98"
		version = "1.00.28"
	strings:
		$1 = { 55 E8 00 00 00 00 5D 81 ?? ?? ?? ?? ?? EB 05 E9 ?? ?? ?? ?? EB 08 }
	condition:
		$1 at pe.entry_point
}

rule locked_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LOCKED?"
	strings:
		$1 = { 29 23 BE 84 E1 6C D6 AE 52 90 49 F1 F1 BB E9 EB B3 A6 DB 3C 87 0C 3E 99 24 5E 0D 1C 06 B7 47 DE B3 12 4D C8 43 BB 8B A6 1F 03 5A 7D 09 38 25 1F }
	condition:
		$1 at pe.entry_point
}

rule lockless_intro_pack_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Lockless Intro Pack"
	strings:
		$1 = { 2C E8 ?? ?? ?? ?? 5D 8B C5 81 ED F6 73 ?? ?? 2B 85 ?? ?? ?? ?? 83 E8 06 89 85 }
	condition:
		$1 at pe.entry_point
}

rule ltc_13 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LTC"
		version = "1.3"
	strings:
		$1 = { 54 E8 00 00 00 00 5D 8B C5 81 ED F6 73 40 00 2B 85 87 75 40 00 83 E8 06 }
	condition:
		$1 at pe.entry_point
}

rule ly_wgkx_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LY_WGKX"
	strings:
		$1 = { 4D 79 46 75 6E 00 62 73 }
	condition:
		$1 at pe.entry_point
}

rule ly_wgkx_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "LY_WGKX"
		version = "2.x"
	strings:
		$1 = { 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4C 59 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 01 00 4D 79 46 75 6E 00 62 73 00 00 }
	condition:
		$1 at pe.entry_point
}

rule macromedia_windows {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Macromedia Windows"
		version = "6.0"
	strings:
		$1 = { 83 EC 44 56 FF 15 24 81 49 00 8B F0 8A 06 3C 22 75 1C 8A 46 01 46 3C 22 74 0C 84 C0 74 08 8A 46 01 46 3C 22 75 F4 80 3E 22 75 0F 46 EB 0C }
	condition:
		$1 at pe.entry_point
}

rule macromedia_windows_flash_projector_40 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Macromedia Windows Flash Projector"
		version = "4.0"
	strings:
		$1 = { 83 EC 44 56 FF 15 24 41 43 00 8B F0 8A 06 3C 22 75 1C 8A 46 01 46 3C 22 74 0C 84 C0 74 08 8A 46 01 46 3C 22 75 F4 80 3E 22 75 0F 46 EB 0C }
	condition:
		$1 at pe.entry_point
}

rule macromedia_windows_flash_projector_50 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Macromedia Windows Flash Projector"
		version = "5.0"
	strings:
		$1 = { 83 EC 44 56 FF 15 70 61 44 00 8B F0 8A 06 3C 22 75 1C 8A 46 01 46 3C 22 74 0C 84 C0 74 08 8A 46 01 46 3C 22 75 F4 80 3E 22 75 0F 46 EB 0C 3C 20 7E 08 8A 46 01 46 3C 20 7F F8 8A 06 84 C0 74 0C 3C 20 7F 08 8A 46 01 46 84 C0 75 F4 8D 44 24 04 C7 44 24 30 00 00 00 00 50 FF 15 80 61 44 00 F6 44 24 30 01 74 0B 8B 44 24 34 25 FF FF 00 00 EB 05 B8 0A 00 00 00 50 56 6A 00 6A 00 FF 15 74 61 44 00 50 E8 18 00 00 00 50 FF 15 78 61 44 00 5E 83 C4 44 C3 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule marjinz_exescrambler_se {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MarjinZ EXE-Scrambler SE"
	strings:
		$1 = { E8 A3 02 00 00 E9 35 FD FF FF FF 25 C8 20 00 10 6A 14 68 C0 21 00 10 E8 E4 01 00 00 FF 35 7C 33 00 10 8B 35 8C 20 00 10 FF D6 59 89 45 E4 83 F8 FF 75 0C FF 75 08 FF 15 88 20 00 10 59 EB 61 6A 08 E8 02 03 00 00 59 83 65 FC 00 FF 35 7C 33 00 10 FF D6 89 45 E4 FF 35 78 33 00 10 FF D6 89 45 E0 8D 45 E0 50 8D 45 E4 50 FF 75 08 E8 D1 02 00 00 89 45 DC FF 75 E4 8B 35 74 20 00 10 FF D6 A3 7C 33 00 10 FF 75 E0 FF D6 83 C4 1C A3 78 33 00 10 C7 45 FC FE FF FF FF E8 09 00 00 00 8B 45 DC E8 A0 01 00 00 C3 }
	condition:
		$1 at pe.entry_point
}

rule maskpe_16 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MaskPE"
		version = "1.6"
	strings:
		$1 = { 36 81 2C 24 ?? ?? ?? 00 C3 60 }
	condition:
		$1 at pe.entry_point
}

rule maskpe_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MaskPE"
		version = "2.0"
	strings:
		$1 = { B8 18 00 00 00 64 8B 18 83 C3 30 C3 40 3E 0F B6 00 C1 E0 ?? 83 C0 ?? 36 01 04 24 C3 }
	condition:
		$1 at pe.entry_point
}

rule matrix_dongle_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Matrix Dongle"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 E8 B6 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? E8 00 00 00 00 5B 2B D9 8B F8 8B 4C 24 2C 33 C0 2B CF F2 AA 8B 3C 24 8B 0A 2B CF 89 5C 24 20 80 37 A2 47 49 75 F9 8D 64 24 04 FF 64 24 FC 60 C7 42 08 ?? ?? ?? ?? E8 C5 FF FF FF C3 C2 F7 29 4E 29 5A 29 E6 86 8A 89 63 5C A2 65 E2 A3 A2 }
	condition:
		$1 at pe.entry_point
}

rule matrix_dongle_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Matrix Dongle"
	strings:
		$1 = { E8 00 00 00 00 E8 00 00 00 00 59 5A 2B CA 2B D1 E8 1A FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule mew_10_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MEW"
		version = "10 1.0"
	strings:
		$1 = { 33 C0 E9 ?? ?0 ?? FF }
	condition:
		$1 at pe.entry_point
}

rule mew_11_se_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MEW"
		version = "11 SE 1.0"
	strings:
		$1 = { E9 ?? ?? ?? ?? 00 00 00 02 00 00 00 0C ?0 }
	condition:
		$1 at pe.entry_point
}

rule mew_11_se_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MEW"
		version = "11 SE 1.2"
    start = 48
	strings:
		$1 = { EB 02 FA 04 E8 49 00 00 00 69 E8 49 00 00 00 95 E8 4F 00 00 00 68 E8 1F 00 00 00 49 E8 E9 FF FF FF 67 E8 1F 00 00 00 93 E8 31 00 00 00 78 E8 DD FF FF FF 38 E8 E3 FF FF FF 66 E8 0D 00 00 00 04 E8 E3 FF FF FF 70 E8 CB FF FF FF 69 E8 DD FF FF FF 58 E8 DD FF FF FF 69 E8 E3 FF FF FF 79 E8 BF FF FF FF 69 83 C4 40 E8 00 00 00 00 5D 81 ED 9D 11 40 00 8D 95 B4 11 }
	condition:
		$1 at pe.entry_point + 48
}

rule mew_5xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MEW"
		version = "5.x.x"
	strings:
		$1 = { BE 5B 00 40 00 AD 91 AD 93 53 AD 96 56 5F AC C0 C0 }
	condition:
		$1 at pe.entry_point
}

rule mew_501 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MEW"
		version = "5.0.1"
	strings:
		$1 = { BE 5B 00 40 00 AD 91 AD 93 53 AD 96 56 5F AC C0 C0 ?? 04 ?? C0 C8 ?? AA E2 F4 C3 00 ?? ?? 00 ?? ?? ?? 00 00 10 40 00 4D 45 57 20 30 2E 31 20 62 79 20 4E 6F 72 74 68 66 6F 78 00 4D 45 57 20 30 2E 31 20 62 79 20 4E 6F 72 74 68 66 6F 78 00 4D 45 57 20 30 2E 31 20 62 79 20 4E 6F 72 74 68 66 6F 78 00 4D 45 57 20 30 2E 31 20 62 79 20 4E 6F 72 74 68 66 6F 78 00 4D }
	condition:
		$1 at pe.entry_point
}

rule microdog_win32shell_4093 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MicroDog Win32Shell"
		version = "4.0.9.3"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 19 FF FF FF E9 AF EC FF FF 90 90 90 90 90 55 8B EC 83 EC 18 53 56 57 8B 45 08 8B 00 C1 E8 10 89 45 FC 8B 45 08 8B 00 25 FF FF 00 00 89 45 F8 C7 45 F4 5A 01 00 00 C7 45 EC 35 4E 00 00 8B 45 F4 0F AF 45 F8 25 FF FF 00 00 89 45 F0 83 7D FC 00 74 0F 8B 45 EC 0F AF 45 FC 25 FF FF 00 00 01 45 F0 8B 45 EC 0F AF 45 F8 8B 4D F0 C1 E1 10 81 E1 00 00 FF FF 03 C1 40 89 45 E8 8B 45 E8 8B 4D 08 89 01 C1 6D E8 10 81 65 E8 FF 7F 00 00 66 8B 45 E8 EB 00 5F 5E 5B C9 C3 }
	condition:
		$1 at pe.entry_point
}

rule microdog_win32shell_4x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MicroDog Win32Shell"
		version = "4.x"
	strings:
		$1 = { 60 55 8B EC 81 EC ?? ?? ?? ?? 53 56 57 C6 85 ?? ?? ?? ?? ?? C6 85 ?? ?? ?? ?? ?? C6 85 ?? ?? ?? ?? ?? C6 85 ?? ?? ?? ?? ?? 8D BD ?? ?? ?? ?? 33 C0 B9 3F ?? ?? ?? F3 AB C6 85 ?? ?? ?? ?? ?? C6 85 ?? ?? ?? ?? ?? C6 85 ?? ?? ?? ?? ?? C6 85 ?? ?? ?? ?? ?? 8D BD ?? ?? ?? ?? 33 C0 B9 3F ?? ?? ?? F3 AB C7 85 ?? ?? ?? ?? ?? ?? ?? ?? 66 C7 ?? ?? ?? ?? ?? ?? ?? E9 13 09 00 00 ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 ?? ?? 89 ?? ?? 83 ?? ?? ?? 75 05 E9 C1 11 00 00 68 ?? ?? ?? ?? A1 ?? ?? ?? ?? 50 8B ?? ?? 50 E8 ?? ?? ?? ?? 83 ?? ?? A1 ?? ?? ?? ?? 33 ?? ?? ?? ?? ?? 89 ?? ?? ?? ?? ?? 8B ?? ?? ?? ?? ?? 89 ?? ?? E9 CE 0E 00 00 E9 36 11 00 00 E9 3D 11 00 00 E9 38 11 00 00 66 ?? ?? ?? ?? ?? EB 04 66 ?? ?? ?? 8B ?? ?? 25 FF FF 00 00 83 ?? ?? 0F 8D DF 00 00 00 8B ?? ?? 25 FF FF 00 00 8B ?? ?? 81 E1 FF FF 00 00 0F AF C1 8B ?? ?? 81 E1 FF FF 00 00 0F AF C1 8B ?? ?? 81 E1 FF FF 00 00 0F AF C1 83 ?? ?? 89 ?? ?? ?? ?? ?? EB 7E }
	condition:
		$1 at pe.entry_point
}

rule microjoiner_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MicroJoiner"
		version = "1.1"
	strings:
		$1 = { BE 0C 70 40 00 BB F8 11 40 00 33 ED 83 EE 04 39 2E 74 11 }
	condition:
		$1 at pe.entry_point
}

rule microjoiner_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MicroJoiner"
		version = "1.5"
	strings:
		$1 = { BF 05 10 40 00 83 EC 30 8B EC E8 C8 FF FF FF E8 C3 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule microjoiner_16 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MicroJoiner"
		version = "1.6"
	strings:
		$1 = { 33 C0 64 8B 38 48 8B C8 F2 AF AF 8B 1F 66 33 DB 66 81 3B }
	condition:
		$1 at pe.entry_point
}

rule microjoiner_17 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MicroJoiner"
		version = "1.7"
	strings:
		$1 = { BF 00 10 40 00 8D 5F 21 6A 0A 58 6A 04 59 60 57 E8 8E 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule minke_101_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Minke"
		version = "1.0.1"
	strings:
		$1 = { 26 3D 4F 38 C2 82 37 B8 F3 24 42 03 17 9B 3A 83 01 00 00 CC 00 00 00 00 06 00 00 00 01 64 53 74 75 62 00 10 55 54 79 70 65 73 00 00 C7 53 79 73 74 65 6D 00 00 81 53 79 73 49 6E 69 74 00 0C 4B 57 69 6E 64 6F 77 73 00 00 8A 75 46 75 6E 63 74 69 6F 6E 73 }
	condition:
		$1 at pe.entry_point
}

rule minke_101_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Minke"
		version = "1.0.1"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 ?? ?? ?? ?? ?? 10 E8 7A F6 FF FF BE 68 66 00 10 33 C0 55 68 DB 40 00 10 64 FF 30 64 89 20 E8 FA F8 FF FF BA EC 40 00 10 8B C6 E8 F2 FA FF FF 8B D8 B8 6C 66 00 10 8B 16 E8 88 F2 FF FF B8 6C 66 00 10 E8 76 F2 FF FF 8B D0 8B C3 8B 0E E8 E3 E4 FF FF E8 2A F9 FF FF E8 C1 F8 FF FF B8 6C 66 00 10 8B 16 E8 6D FA FF FF E8 14 F9 FF FF E8 AB F8 FF FF 8B 06 E8 B8 E3 FF FF 8B D8 B8 6C 66 00 10 E8 38 F2 FF FF 8B D3 8B 0E E8 A7 E4 FF ?? ?? ?? ?? C4 FB FF FF E8 E7 F8 FF FF 8B C3 E8 B0 E3 FF FF E8 DB F8 FF FF 33 C0 5A 59 59 64 89 10 68 E2 40 00 10 C3 E9 50 EB FF FF EB F8 5E 5B E8 BB EF FF FF 00 00 00 43 41 31 38 }
	condition:
		$1 at pe.entry_point
}

rule mkfpack_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "mkfpack"
	strings:
		$1 = { E8 00 00 00 00 5B 81 EB 05 00 00 00 8B 93 9F 08 00 00 53 6A ?? 68 ?? ?? ?? ?? 52 6A 00 FF 93 32 08 00 00 5B 8B F0 8B BB 9B 08 00 00 03 FB 56 57 E8 86 08 00 00 83 C4 08 8D 93 BB 08 00 00 52 53 FF E6 }
	condition:
		$1 at pe.entry_point
}

rule molebox_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
	strings:
		$1 = { 60 E8 4F 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule molebox_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
	strings:
		$1 = { 55 8B EC 6A FF 68 00 00 00 00 68 00 00 00 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule molebox_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
	strings:
		$1 = { 58 4F 4A 55 4D 41 4E 4A }
	condition:
		$1 in (pe.overlay.offset .. pe.overlay.offset + pe.overlay.size)
}

rule molebox_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
		version = "2.0"
	strings:
		$1 = { E8 ?? ?? ?? ?? 60 E8 }
	condition:
		$1 at pe.entry_point
}

rule molebox_230 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
		version = "2.3.0"
	strings:
		$1 = { 42 04 E8 ?? ?? 00 00 A3 ?? ?? ?? 00 8B 4D F0 8B 11 89 15 ?? ?? ?? 00 ?? 45 FC A3 ?? ?? ?? 00 5F 5E 8B E5 5D C3 CC CC CC CC CC CC CC CC CC CC CC E8 EB FB FF FF 58 E8 ?? 07 00 00 58 89 44 24 20 61 58 FF D0 E8 ?? ?? 00 00 CC CC CC CC CC CC CC }
	condition:
		$1 at pe.entry_point
}

rule molebox_23x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
		version = "2.3.x"
	strings:
		$1 = { E8 00 00 00 00 60 E8 4F 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule molebox_236 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
		version = "2.3.6"
	strings:
		$1 = { EB 16 8B 15 ?? ?? ?? ?? FF 32 8F 05 ?? ?? ?? ?? EB 06 8F 05 ?? ?? ?? ?? B8 ?? ?? ?? ?? 83 38 00 74 20 50 }
	condition:
		$1 at pe.entry_point
}

rule molebox_254 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
		version = "2.5.4"
	strings:
		$1 = { ?? ?? ?? 00 8B 4D F0 8B 11 89 15 ?? ?? ?? 00 8B 45 FC A3 ?? ?? ?? 00 5F 5E 8B E5 5D C3 CC CC CC E8 EB FB FF FF 58 E8 ?? 07 00 00 58 89 44 24 24 61 58 58 FF D0 E8 ?? ?? 00 00 6A 00 FF 15 ?? ?? ?? 00 CC CC CC CC CC CC CC CC CC CC CC CC CC CC }
	condition:
		$1 at pe.entry_point
}

rule molebox_pro_255 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox Pro"
		version = "2.5.5"
	strings:
		$1 = { E8 00 00 00 00 60 E8 4F 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E9 CA 69 00 00 E9 DE 69 00 00 E9 D9 69 00 00 E8 5E FB FF FF 3E F4 00 00 }
	condition:
		$1 at pe.entry_point
}

rule molebox_42321 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBox"
		version = "4.2321"
	strings:
		$1 = { 6A 28 68 70 20 40 00 E8 74 02 00 00 3? FF 57 FF 15 ?? ?? ?? ?? 66 81 38 4D 5A }
	condition:
		$1 at pe.entry_point
}

rule molebox_pro_43018 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MoleBoxPro"
		version = "4.3018"
	strings:
		$1 = { 55 89 E5 ?? ?C 0? ?? 0? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?0 ?? ?? ?? 00 00 ?? 0? ?? ?? ?? ?? ?? 0? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 0? 0? ?? ?? ?? B? ?? ?0 00 ?? ?? ?? ?? ?? ?? ?4 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?0 ?? ?? ?? 89 }
	condition:
		$1 at pe.entry_point
}

rule kaos_pe_exe_undetecter_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "KaOs PE eXecutable Undetecter"
	strings:
		$1 = { 60 FC 0F B6 05 ?? ?? ?? ?? ?? ?? 75 31 B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? E8 9A 00 00 00 A3 }
	condition:
		$1 at pe.entry_point
}

rule k_kryptor_011 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "K!Cryptor"
		version = "0.11"
	strings:
		$1 = { 55 8B EC 83 EC ?? 53 56 57 33 DB 53 FF 15 ?? ?? ?? ?? 8B 3D ?? ?? ?? ?? 89 45 ?? B8 ?? ?? ?? ?? FF 30 BE ?? ?? ?? ?? 56 E8 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A ?? E8 ?? ?? ?? ?? 83 C4 ?? 6A ?? 68 ?? ?? ?? ?? 57 53 FF D0 }
	condition:
		$1 at pe.entry_point
}

rule morphnah_beta {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Morphnah"
		version = "Beta"
	strings:
		$1 = { 2E 6E 61 68 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 A0 00 00 E0 }
	condition:
		$1 at pe.entry_point
}

rule mpack_002 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "mPack"
		version = "0.0.2"
	strings:
		$1 = { E9 00 00 00 00 60 E8 14 00 00 00 5D 81 ED 00 00 00 00 6A 45 E8 A3 00 00 00 68 00 00 00 00 E8 58 61 E8 AA 00 00 00 4E }
	condition:
		$1 at pe.entry_point
}
rule mpack_003_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "mPack"
		version = "0.0.3"
	strings:
		$1 = { 55 8B EC 83 C4 F0 33 C0 89 45 F0 B8 A8 76 00 10 E8 67 C4 FF FF 33 C0 55 68 C2 78 00 10 64 FF 30 64 89 20 8D 55 F0 33 C0 E8 93 C8 FF FF 8B 45 F0 E8 87 CB FF FF A3 08 A5 00 10 33 C0 55 68 A5 78 00 10 64 FF 30 64 89 20 A1 08 A5 00 10 E8 FA C9 FF FF 83 F8 FF 75 0A E8 88 B2 FF FF E9 1B 01 00 00 C7 05 14 A5 00 10 32 00 00 00 A1 08 A5 00 10 8B 15 14 A5 00 10 E8 C9 C9 FF FF BA 14 A5 00 10 A1 08 A5 00 10 B9 04 00 00 00 E8 C5 C9 FF FF 83 3D 14 A5 00 10 32 77 0A E8 47 B2 FF FF E9 DA 00 00 00 A1 08 A5 00 10 8B 15 14 A5 00 10 E8 92 C9 FF FF BA 18 A5 }
	condition:
		$1 at pe.entry_point
}

rule mpack_003_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "mPack"
		version = "0.0.3"
	strings:
		$1 = { 55 8B EC 83 ?? ?? 33 C0 89 45 F0 B8 ?? ?? ?? ?? E8 67 C4 FF FF 33 C0 55 68 ?? ?? ?? ?? 64 FF 30 64 89 20 8D 55 F0 33 C0 E8 93 C8 FF FF 8B 45 F0 E8 87 CB FF FF A3 ?? ?? ?? ?? 33 C0 55 68 ?? ?? ?? ?? 64 FF 30 64 89 20 A1 ?? ?? ?? ?? E8 FA C9 FF FF 83 F8 FF 75 0A E8 88 B2 FF FF E9 1B 01 00 00 C7 05 ?? ?? ?? ?? 32 00 00 00 A1 ?? ?? ?? ?? 8B 15 ?? ?? ?? ?? E8 C9 C9 FF FF BA ?? ?? ?? ?? A1 ?? ?? ?? ?? B9 04 00 00 00 E8 C5 C9 FF FF 83 3D ?? ?? ?? ?? 32 77 0A E8 47 B2 FF FF E9 DA 00 00 00 A1 ?? ?? ?? ?? 8B 15 ?? ?? ?? ?? E8 92 C9 FF FF BA 18 A5 00 10 A1 ?? ?? ?? ?? B9 04 00 00 00 E8 8E C9 FF FF 83 F8 04 74 0A E8 14 B2 FF FF E9 A7 00 00 00 E8 0A CB FF FF A3 ?? ?? ?? ?? A1 ?? ?? ?? ?? E8 63 C9 FF FF 83 F8 FF 75 0A E8 F1 B1 FF FF E9 84 00 00 00 6A 00 6A 00 B8 ?? ?? ?? ?? 8B 15 ?? ?? ?? ?? E8 D4 CD FF FF 84 C0 75 07 E8 CF B1 FF FF EB 65 8B 0D ?? ?? ?? ?? 8B 15 ?? ?? ?? ?? A1 ?? ?? ?? ?? E8 0F FA FF FF 3B 05 ?? ?? ?? ?? 75 0D A1 ?? ?? ?? ?? 8B 40 3C E8 6E FB FF FF 6A 03 E8 07 C4 FF FF A1 ?? ?? ?? ?? E8 C1 C6 FF FF 33 C0 5A 59 59 64 89 10 68 ?? ?? ?? ?? A1 ?? ?? ?? ?? E8 AA C6 FF FF A1 ?? ?? ?? ?? E8 A0 C6 FF FF C3 E9 AE B0 FF FF EB E4 33 C0 5A 59 59 64 89 10 68 ?? ?? ?? ?? 8D 45 F0 E8 A7 B5 FF FF C3 E9 91 B0 FF FF EB F0 E8 62 B4 FF FF }
	condition:
		$1 at pe.entry_point
}

rule mpress_071a_075b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "0.71a - 0.75b"
	strings:
		$1 = { 57 56 53 51 52 55 E8 10 00 00 00 E8 7A 00 00 00 5D 5A 59 5B 5E 5F E9 84 01 00 00 E8 00 00 00 00 58 05 84 01 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 AD 2B C8 03 F1 8B C8 49 8A 44 39 06 74 05 88 04 31 EB F4 88 04 31 2B C0 AC 0A C0 74 37 8A C8 24 3F 80 E1 C0 C1 E0 10 66 AD 80 F9 C0 74 1E F6 C1 40 75 0A 8B C8 2B C0 F3 AA 75 FC EB D9 8B D6 8B CF 03 F0 E8 8F 00 00 00 03 F8 EB CA 8B C8 F3 A4 75 FC EB C2 C3 E8 00 00 00 00 5F 81 C7 71 FF FF FF B0 E9 AA B8 9A 01 00 00 AB 2B FF E8 00 00 00 00 58 05 FE 00 00 00 8B 78 08 8B D7 8B 78 04 0B FF 74 53 8B 30 03 F0 2B F2 8B EE 8B C2 8B 45 3C 03 C5 8B 48 34 2B CD 74 3D E8 00 00 00 00 58 05 DD 00 00 00 8B 10 03 F2 03 FE 2B C0 AD 3B F7 73 25 8B D8 AD 3B F7 73 1E 8B D0 83 EA 08 03 D6 66 AD 0A E4 74 0B 25 FF 0F 00 00 03 C3 03 C5 29 08 3B F2 73 D8 EB E9 C3 }
	condition:
		$1 at pe.entry_point
}

rule mpress_077b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "0.77b"
	strings:
		$1 = { 60 E8 0B 00 00 00 E8 77 00 00 00 61 E9 75 01 00 00 E8 00 00 00 00 58 05 75 01 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 AD 2B C8 03 F1 8B C8 49 8A 44 39 06 74 05 88 04 31 EB F4 88 04 31 2B C0 3B FE 73 3A AC 0A C0 74 35 8A C8 24 3F 80 E1 C0 C1 E0 10 66 AD 80 F9 C0 74 1C F6 C1 40 75 08 8B C8 2B C0 F3 AA EB D7 8B D6 8B CF 03 F0 E8 7E 00 00 00 03 F8 EB C8 8B C8 F3 A4 75 FC EB C0 C3 E8 00 00 00 00 5F 81 C7 79 FF FF FF B0 E9 AA B8 81 01 00 00 AB 2B FF E8 00 00 00 00 58 05 ED 00 00 00 8B 78 08 8B D7 8B 78 04 0B FF 74 42 8B 30 03 F0 2B F2 8B EE 8B 48 10 2B CD 74 33 8B 50 0C 03 F2 03 FE 2B C0 AD 3B F7 73 25 8B D8 AD 3B F7 73 1E 8B D0 83 EA 08 03 D6 66 AD 0A E4 74 0B 25 FF 0F 00 00 03 C3 03 C5 29 08 3B F2 73 D8 EB E9 C3 }
	condition:
		$1 at pe.entry_point
}

rule mpress_085_092 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "0.85 - 0.92"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 48 01 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 49 8A 44 39 06 74 05 88 04 31 EB F4 88 04 31 2B C0 3B FE 73 28 AC 0A C0 74 23 8A C8 24 3F C1 E0 10 66 AD 80 E1 40 74 0F 8B D6 8B CF 03 F0 E8 5F 00 00 00 03 F8 EB D8 8B C8 F3 A4 EB D2 5E 5A 83 EA 05 2B C9 3B CA 73 25 8B D9 AC 41 24 FE 3C E8 75 F2 83 C1 04 AD 0B C0 78 06 3B C2 73 E6 EB 06 03 C3 78 E0 03 C2 2B C3 89 46 FC EB D7 E8 00 00 00 00 5F 81 C7 6A FF FF FF B0 E9 AA B8 44 01 00 00 AB E8 00 00 00 00 58 05 A3 00 00 00 E9 93 00 00 00 53 56 57 8B F9 8B F2 8B DA 03 D8 51 55 33 C0 8B EB 8B DE 2B D2 2B C9 EB 4F 3B DD 73 6C 2B C9 66 8B 03 8D 5B 02 8A CC 80 E4 0F 0B C0 75 02 B4 10 C0 E9 04 80 C1 03 80 F9 12 72 19 8A 0B 66 83 C1 12 43 66 81 F9 11 01 72 0B 66 8B 0B 81 C1 11 01 00 00 43 43 8B F7 2B F0 F3 A4 12 D2 74 0A 72 B9 8A 03 43 88 07 47 EB F2 3B DD 73 1D 0A 13 F9 74 03 43 EB E6 8B 43 01 89 07 8B 43 05 89 47 04 8D 5B 09 8D 7F 08 33 C0 EB DF 5D 8B C7 59 2B C1 5F 5E 5B C3 E9 }
	condition:
		$1 at pe.entry_point
}

rule mpress_097_099 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "0.97 - 0.99"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 49 01 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 49 8A 44 39 06 74 05 88 04 31 EB F4 88 04 31 2B C0 3B FE 73 28 AC 0A C0 74 23 8A C8 24 3F C1 E0 10 66 AD 80 E1 40 74 0F 8B D6 8B CF 03 F0 E8 60 00 00 00 03 F8 EB D8 8B C8 F3 A4 EB D2 5E 5A 83 EA 05 2B C9 3B CA 73 26 8B D9 AC 41 24 FE 3C E8 75 F2 43 83 C1 04 AD 0B C0 78 06 3B C2 73 E5 EB 06 03 C3 78 DF 03 C2 2B C3 89 46 FC EB D6 E8 00 00 00 00 5F 81 C7 69 FF FF FF B0 E9 AA B8 45 01 00 00 AB E8 00 00 00 00 58 05 A3 00 00 00 E9 93 00 00 00 53 56 57 8B F9 8B F2 8B DA 03 D8 51 55 33 C0 8B EB 8B DE 2B D2 2B C9 EB 4F 3B DD 73 6C 2B C9 66 8B 03 8D 5B 02 8A CC 80 E4 0F 0B C0 75 02 B4 10 C0 E9 04 80 C1 03 80 F9 12 72 19 8A 0B 66 83 C1 12 43 66 81 F9 11 01 72 0B 66 8B 0B 81 C1 11 01 00 00 43 43 8B F7 2B F0 F3 A4 12 D2 74 0A 72 B9 8A 03 43 88 07 47 EB F2 3B DD 73 1D 0A 13 F9 74 03 43 EB E6 8B 43 01 89 07 8B 43 05 89 47 04 8D 5B 09 8D 7F 08 33 C0 EB DF 5D 8B C7 59 2B C1 5F 5E 5B C3 E9 }
	condition:
		$1 at pe.entry_point
}

rule mpress_101_105 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "1.01 - 1.05"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 B6 02 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_107_127 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "1.07 - 1.27"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 9E 02 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 74 05 88 04 31 EB F4 88 04 31 8B D6 8B CF E8 56 00 00 00 5E 5A 83 EA 05 2B C9 3B CA 73 26 8B D9 AC 41 24 FE 3C E8 75 F2 43 83 C1 04 AD 0B C0 78 06 3B C2 73 E5 EB 06 03 C3 78 DF 03 C2 2B C3 89 46 FC EB D6 E8 00 00 00 00 5F 81 C7 8D FF FF FF B0 E9 AA B8 9A 02 00 00 AB E8 00 00 00 00 58 05 1C 02 00 00 E9 0C 02 00 00 }
	condition:
		$1 at pe.entry_point
}

rule mpress_1x_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "1.x - 2.x"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 ?? ?? ?? ?? 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_201_lzma {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "2.01 [LZMA]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 5E 0B 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_201_lzmat {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "2.01 [LZMAT]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 99 02 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_205_lzma {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "2.05 [LZMA]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 57 0B 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_205_lzmat {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "2.05 [LZMAT]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 9C 02 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_212_219_lzma {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "2.12 - 2.19 [LZMA]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 5A 0B 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule mpress_212_219_lzmat {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MPRESS"
		version = "2.12 - 2.19 [LZMAT]"
	strings:
		$1 = { 60 E8 00 00 00 00 58 05 9F 02 00 00 8B 30 03 F0 2B C0 8B FE 66 AD C1 E0 0C 8B C8 50 AD 2B C8 03 F1 8B C8 57 51 49 8A 44 39 06 }
	condition:
		$1 at pe.entry_point
}

rule muckis_protector_i {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "mucki's protector"
		version = "I"
	strings:
		$1 = { BE ?? ?? ?? ?? B9 ?? ?? ?? ?? 8A 06 F6 D0 88 06 46 E2 F7 E9 }
	condition:
		$1 at pe.entry_point
}

rule muckis_protector_ii_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Muckis protector"
		version = "II"
	strings:
		$1 = { E8 24 00 00 00 8B 4C 24 0C C7 01 17 00 01 00 C7 81 B8 00 00 00 00 00 00 00 31 C0 89 41 14 89 41 18 80 6A 00 }
	condition:
		$1 at pe.entry_point
}

rule muckis_protector_ii_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "mucki's protector"
		version = "II"
	strings:
		$1 = { E8 24 00 00 00 8B 4C 24 0C C7 01 17 00 01 00 C7 81 B8 00 00 00 00 00 00 00 31 C0 89 41 14 89 41 18 80 6A 00 E8 85 C0 74 12 64 8B 3D 18 00 00 00 8B 7F 30 0F B6 47 02 85 C0 74 01 C3 C7 04 24 ?? ?? ?? ?? BE ?? ?? ?? ?? B9 ?? ?? ?? ?? 8A 06 F6 D0 88 06 46 E2 F7 C3 }
	condition:
		$1 at pe.entry_point
}

rule mz0ope_106b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MZ0oPE"
		version = "1.0.6b"
	strings:
		$1 = { EB CA 89 03 83 C3 04 87 FE 32 C0 AE 75 FD 87 FE 80 3E FF 75 E2 46 5B 83 C3 04 53 8B 1B 80 3F FF 75 C9 8B E5 61 68 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule mz_crypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "MZ-Crypt"
		version = "1.0"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 25 14 40 00 8B BD 77 14 40 00 8B 8D 7F 14 40 00 EB 28 83 7F 1C 07 75 1E 8B 77 0C 03 B5 7B 14 40 00 33 C0 EB 0C 50 8A A5 83 14 40 00 30 26 58 40 46 3B 47 10 76 EF 83 C7 28 49 0B C9 75 D4 8B 85 73 14 40 00 89 44 24 1C 61 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule njoiner_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "N-Joiner"
		version = "0.1"
		description = "asm version"
	strings:
		$1 = { 6A 00 68 00 14 40 00 68 00 10 40 00 6A 00 E8 14 00 00 00 6A 00 E8 13 00 00 00 CC FF 25 AC 12 40 00 FF 25 B0 12 40 00 FF 25 B4 12 40 00 FF 25 B8 12 40 00 FF 25 BC 12 40 00 FF 25 C0 12 40 00 FF 25 C4 12 40 00 FF 25 C8 12 40 00 FF 25 CC 12 40 00 FF 25 D0 12 40 00 FF 25 D4 12 40 00 FF 25 D8 12 40 00 FF 25 DC 12 40 00 FF 25 E4 12 40 00 FF 25 EC 12 40 00 }
	condition:
		$1 at pe.entry_point
}

rule njoy_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "N-Joy"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 9C 3B 40 00 E8 8C FC FF FF 6A 00 68 E4 39 40 00 6A 0A 6A 00 E8 40 FD FF FF E8 EF F5 FF FF 8D 40 00 }
	condition:
		$1 at pe.entry_point
}

rule njoy_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "N-Joy"
		version = "1.1"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 0C 3C 40 00 E8 24 FC FF FF 6A 00 68 28 3A 40 00 6A 0A 6A 00 E8 D8 FC FF FF E8 7F F5 FF FF 8D 40 00 }
	condition:
		$1 at pe.entry_point
}

rule njoy_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "N-Joy"
		version = "1.2"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 A4 32 40 00 E8 E8 F1 FF FF 6A 00 68 54 2A 40 00 6A 0A 6A 00 E8 A8 F2 FF FF E8 C7 EA FF FF 8D 40 00 }
	condition:
		$1 at pe.entry_point
}

rule njoy_13 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "N-Joy"
		version = "1.3"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 48 36 40 00 E8 54 EE FF FF 6A 00 68 D8 2B 40 00 6A 0A 6A 00 E8 2C EF FF FF E8 23 E7 FF FF 8D 40 00 }
	condition:
		$1 at pe.entry_point
}

rule maked_packer_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Naked Packer"
		version = "1.0"
	strings:
		$1 = { 60 FC 0F B6 05 ?? ?? ?? ?? 85 C0 75 31 B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? A1 ?? ?? ?? ?? 03 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? E8 9A 00 00 00 A3 ?? ?? ?? ?? C6 05 ?? ?? ?? ?? 01 83 3D ?? ?? ?? ?? 00 75 07 61 FF 25 ?? ?? ?? ?? 61 FF 74 24 04 6A 00 FF 15 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? C3 FF 74 24 04 6A 00 FF 15 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule maked_packer_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Naked Packer"
		version = "1.x"
	strings:
		$1 = { 6A ?? E8 9A 05 00 00 8B D8 53 68 ?? ?? ?? ?? E8 6C FD FF FF B9 05 00 00 00 8B F3 BF ?? ?? ?? ?? 53 F3 A5 E8 8D 05 00 00 8B 3D ?? ?? ?? ?? A1 ?? ?? ?? ?? 66 8B 15 ?? ?? ?? ?? B9 ?? ?? ?? ?? 2B CF 89 45 E8 89 0D ?? ?? ?? ?? 66 89 55 EC 8B 41 3C 33 D2 03 C1 83 C4 10 66 8B 48 06 66 8B 50 14 81 E1 FF FF 00 00 8D 5C 02 18 8D 41 FF E8 00 00 00 00 EB 01 ?? 89 45 F0 C6 45 FF 00 8D 7D E8 8B F3 8A 0E 8A 17 8A C1 3A CA 75 1E 84 C0 74 16 8A 56 01 8A 4F 01 8A C2 3A D1 75 0E 83 C6 02 83 C7 02 84 C0 75 DC 33 C0 EB 05 }
	condition:
		$1 at pe.entry_point
}

rule nakedbind_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Nakedbind"
		version = "1.0"
	strings:
		$1 = { 64 8B 38 48 8B C8 F2 AF AF 8B 1F 66 33 DB 66 81 3B 4D 5A 74 08 81 EB 00 00 }
	condition:
		$1 at pe.entry_point
}

rule native_ud_packer_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Native UD Packer"
		version = "1.1"
	strings:
		$1 = { 31 C0 31 DB 31 C9 EB 0E 6A 00 6A 00 6A 00 6A 00 FF 15 28 41 40 00 FF 15 94 40 40 00 89 C7 68 88 13 00 00 FF 15 98 40 40 00 FF 15 94 40 40 00 81 C7 88 13 00 00 39 F8 73 05 E9 84 00 00 00 6A 40 68 00 10 00 00 FF 35 04 30 40 00 6A 00 FF 15 A4 40 40 00 89 C7 FF 35 04 30 40 00 68 CA 10 40 00 50 FF 15 A8 40 40 00 6A 40 68 00 10 00 00 FF 35 08 30 40 00 6A 00 FF 15 A4 40 40 00 89 C6 68 00 30 40 00 FF 35 04 30 40 00 57 FF 35 08 30 40 00 50 6A 02 FF 15 4E 41 40 00 6A 00 6A 00 6A 00 56 6A 00 6A 00 FF 15 9C 40 40 00 50 6A 00 6A 00 6A 11 50 FF 15 4A 41 40 00 58 6A FF 50 FF 15 AC 40 40 00 6A 00 FF 15 A0 40 }
	condition:
		$1 at pe.entry_point
}

rule nbinder_361 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nBinder"
		version = "3.6.1"
	strings:
		$1 = { 6E 35 36 34 35 36 35 33 32 33 34 35 34 33 5F 6E 62 33 5C 00 5C 6E 35 36 34 35 36 35 33 32 33 34 35 34 33 5F 6E 62 33 5C }
	condition:
		$1 at pe.entry_point
}

rule nbinder_40 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nBinder"
		version = "4.0"
	strings:
		$1 = { 5C 6E 62 34 5F 74 6D 70 5F 30 31 33 32 34 35 34 33 35 30 5C 00 00 00 00 00 00 00 00 00 E9 55 43 4C FF 01 1A 00 00 00 00 96 30 07 77 2C 61 0E EE BA 51 09 99 19 C4 6D 07 8F F4 6A 70 35 A5 63 E9 A3 95 64 9E 32 88 DB 0E A4 B8 DC 79 }
	condition:
		$1 at pe.entry_point
}

rule nbuild_10_soft {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nbuild"
		version = "1.0 soft"
	strings:
		$1 = { B9 ?? ?? BB ?? ?? C0 ?? ?? 80 ?? ?? 43 E2 }
	condition:
		$1 at pe.entry_point
}

rule ncode_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "N-Code"
		version = "0.2"
	strings:
		$1 = { 90 66 BE ?? ?? 66 83 FE ?? 74 ?? 66 B8 ?? ?? 66 BE ?? ?? 66 83 FE ?? 74 ?? 66 83 E8 ?? 66 BB ?? ?? 66 83 C3 ?? 66 43 66 81 FB ?? ?? 74 ?? 66 83 F8 }
	condition:
		$1 at pe.entry_point
}

rule neolite_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NeoLite"
		version = "1.0"
	strings:
		$1 = { E9 9B 00 00 00 A0 }
	condition:
		$1 at pe.entry_point
}

rule neolite_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NeoLite"
		version = "1.0"
	strings:
		$1 = { 8B 44 24 04 8D 54 24 FC 23 05 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? 50 FF 25 }
	condition:
		$1 at pe.entry_point
}

rule neolite_20_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NeoLite"
		version = "2.0"
	strings:
		$1 = { 8B 44 24 04 23 05 ?? ?? ?? ?? 50 E8 ?? ?? ?? ?? 83 C4 04 FE 05 ?? ?? ?? ?? 0B C0 74 }
	condition:
		$1 at pe.entry_point
}

rule neolite_20_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NeoLite"
		version = "2.0"
	strings:
		$1 = { E9 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 4E 65 6F 4C 69 74 65 }
	condition:
		$1 at pe.entry_point
}

rule neolite_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NeoLite"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9E 37 00 00 ?? ?? 48 ?? ?? ?? 6F 4C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 61 }
	condition:
		$1 at pe.entry_point
}

rule nfo_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NFO"
		version = "1.0"
	strings:
		$1 = { 8D 50 12 2B C9 B1 1E 8A 02 34 77 88 02 42 E2 F7 C8 8C }
	condition:
		$1 at pe.entry_point
}

rule nfo_1x_modified {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NFO"
		version = "1.x modified"
	strings:
		$1 = { 60 9C 8D 50 }
	condition:
		$1 at pe.entry_point
}

rule noc_packer_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NOS Packer"
	strings:
		$1 = { 50 E8 00 00 00 00 5B 81 EB ?? ?? ?? ?? B9 ?? ?? ?? ?? 2B D9 8B F3 81 EB ?? ?? ?? ?? 8B FB 81 EB ?? ?? ?? ?? 57 51 56 E8 ?? ?? ?? ?? 83 C4 ?? 8B AB ?? ?? ?? ?? 8D 2C 2B 4D 8A 4D ?? 80 F9 ?? 74 ?? 83 ED ?? 8B D3 2B 53 }
	condition:
		$1 at pe.entry_point
}

rule ningishzida_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ningishzida"
		version = "1.0"
	strings:
		$1 = { 9C 60 96 E8 00 00 00 00 5D 81 ED 03 25 40 00 B9 04 1B 00 00 8D BD 4B 25 40 00 8B F7 AC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? AA E2 CC }
	condition:
		$1 at pe.entry_point
}

rule nmacro_recorder_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nMacro recorder"
		version = "1.0"
	strings:
		$1 = { 5C 6E 6D 72 5F 74 65 6D 70 2E 6E 6D 72 00 00 00 72 62 00 00 58 C7 41 00 10 F8 41 00 11 01 00 00 00 00 00 00 46 E1 00 00 46 E1 00 00 35 00 00 00 F6 88 41 00 }
	condition:
		$1 at pe.entry_point
}

rule nme_11_public {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NME"
		version = "1.1 public"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 56 B8 30 35 14 13 E8 9A E6 FF FF 33 C0 55 68 6C 36 14 13 64 FF 30 64 89 20 B8 08 5C 14 13 BA 84 36 14 13 E8 7D E2 FF FF E8 C0 EA FF FF 8B 15 CC 45 14 13 A1 C8 45 14 13 E8 04 F8 FF FF 8B 15 D0 45 14 13 A1 C8 45 14 13 E8 F4 F7 FF FF 8B 15 CC 45 14 13 A1 C8 45 14 13 E8 2C F9 FF FF A3 F8 5A 14 13 8B 15 D0 45 14 13 A1 C8 45 14 13 E8 17 F9 FF FF A3 FC 5A 14 13 B8 04 5C 14 13 E8 20 FB FF FF 8B D8 85 DB 74 48 B8 00 5B 14 13 8B 15 C4 45 14 13 E8 1E E7 FF FF A1 04 5C 14 13 E8 A8 DA FF FF ?? ?? ?? ?? 5C 14 13 50 8B CE 8B D3 B8 00 5B 14 13 ?? ?? ?? ?? FF 8B C6 E8 DF FB FF FF 8B C6 E8 9C DA FF FF B8 00 5B 14 13 E8 72 E7 FF FF 33 C0 5A 59 59 64 89 10 68 73 36 14 13 C3 E9 0F DF FF FF EB F8 5E 5B E8 7E E0 FF FF 00 00 FF FF FF FF 0C 00 00 00 4E 4D 45 20 31 2E 31 20 53 74 75 62 }
	condition:
		$1 at pe.entry_point
}

rule noobyprotect_1004 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NoobyProtect"
		version = "1.0.0.4"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 45 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 9C 81 44 24 04 }
	condition:
		$1 at pe.entry_point
}

rule noobyprotect_1090_1098 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NoobyProtect"
		version = "1.0.9.0 - 1.0.9.8"
	strings:
		$1 = { 53 51 E8 00 00 00 00 8B 1C 24 83 C3 25 33 C9 87 4B FC 83 F9 00 74 06 80 33 ?? 43 E2 FA 83 C4 04 59 5B 9D E9 04 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule noobyprotect_109x_se_public {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NoobyProtect"
		version = "1.0.9.x SE public"
	strings:
		$1 = { 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 75 73 65 72 33 32 2E 64 6C 6C 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? 00 00 4D 65 73 73 61 67 65 42 6F 78 41 00 ?? ?? ?? ?? 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule noobyprotect_1100_se_public_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NoobyProtect"
		version = "1.1.0.0 SE public"
	strings:
		$1 = { 4E 6F 6F 62 79 50 72 6F 74 65 63 74 20 53 45 20 31 2E 31 2E 30 2E 30 }
	condition:
		$1 at pe.entry_point
}

rule noobyprotect_1100_se_public_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NoobyProtect"
		version = "1.1.0.0 SE public"
	strings:
		$1 = { 9C 53 51 E8 00 00 00 00 8B 1C 24 83 C3 25 33 C9 87 4B FC 83 F9 00 74 06 80 33 ?? 43 E2 FA 83 C4 04 59 5B 9D E9 04 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule noodlecrypt_200 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NoodleCrypt"
		version = "2.00"
	strings:
		$1 = { EB 01 9A E8 ?? 00 00 00 EB 01 9A E8 ?? ?? 00 00 EB 01 9A E8 ?? ?? 00 00 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule npack_11150b_11200b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.150b - 1.1.200b"
	strings:
		$1 = { 83 3D 40 ?? ?? ?? 00 75 05 E9 01 00 00 00 C3 E8 41 00 00 00 B8 80 ?? ?? ?? 2B 05 08 ?? ?? ?? A3 3C ?? ?? ?? E8 5E 00 00 00 E8 E0 01 00 00 E8 EC 06 00 00 E8 F7 05 00 00 }
	condition:
		$1 at pe.entry_point
}

rule npack_111502006b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.150.2006b"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? ?? 75 05 E9 01 00 00 00 C3 E8 41 00 00 00 B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? E8 5E 00 00 00 E8 E0 01 00 00 E8 EC 06 00 00 E8 F7 05 00 00 A1 ?? ?? ?? ?? C7 05 ?? ?? ?? ?? ?? ?? ?? ?? 01 05 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? C3 C3 56 57 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B 35 ?? ?? ?? ?? 8B F8 68 ?? ?? ?? ?? 57 FF D6 68 ?? ?? ?? ?? 57 A3 ?? ?? ?? ?? FF D6 5F A3 ?? ?? ?? ?? 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule npack_11150206b_112002006b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.150.2006b, 1.1.200.2006b"
	strings:
		$1 = { 55 89 E5 83 EC 08 83 C4 F4 6A 02 A1 C8 ?? ?? ?? FF D0 E8 ?? ?? ?? ?? C9 C3 }
	condition:
		$1 at pe.entry_point
}

rule npack_112002006b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.200.2006b"
	strings:
		$1 = { 83 3D 40 ?? ?? ?? 00 75 05 E9 01 00 00 00 C3 E8 41 00 00 00 B8 80 ?? ?? ?? 2B 05 08 ?? ?? ?? A3 3C ?? ?? ?? E8 5E 00 00 00 E8 EC 01 00 00 E8 F8 06 00 00 E8 03 06 00 00 A1 3C ?? ?? ?? C7 05 40 ?? ?? ?? 01 00 00 00 01 05 00 ?? ?? ?? FF 35 00 ?? ?? ?? C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule npack_11250 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.250"
	strings:
		$1 = { 83 3D 04 ?? ?? ?? 00 75 05 E9 01 00 00 00 C3 E8 46 00 00 00 E8 73 00 00 00 B8 2E ?? ?? ?? 2B 05 08 ?? ?? ?? A3 00 ?? ?? ?? E8 9C 00 00 00 E8 04 02 00 00 E8 FB 06 00 00 E8 1B 06 00 00 A1 00 ?? ?? ?? C7 05 04 ?? ?? ?? 01 00 00 00 01 05 00 ?? ?? ?? FF 35 00 ?? ?? ?? C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule npack_112752006b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.275.2006b"
	strings:
		$1 = { 55 8B EC 51 51 56 57 BE ?? ?? ?? ?? 8D 7D F8 66 A5 A4 BE ?? ?? ?? ?? 8D 7D FC 8D 45 FC 66 A5 50 8D 45 F8 50 A4 FF 15 ?? ?? ?? ?? 83 3D ?? ?? ?? ?? ?? 5F 5E 75 05 E8 02 00 00 00 C9 C3 E8 46 00 00 00 E8 73 00 00 00 B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? E8 9C 00 00 00 E8 1A 02 00 00 E8 CA 06 00 00 E8 19 06 00 00 A1 ?? ?? ?? ?? C7 05 ?? ?? ?? ?? ?? ?? ?? ?? 01 05 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? C3 C3 56 57 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B ?? ?? ?? ?? ?? 8B F8 68 ?? ?? ?? ?? 57 FF D6 68 ?? ?? ?? ?? 57 A3 ?? ?? ?? ?? FF D6 5F A3 ?? ?? ?? ?? 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule npack_113002006b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.300.2006b"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? ?? 75 05 E9 01 00 00 00 C3 E8 46 00 00 00 E8 73 00 00 00 B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? E8 9C 00 00 00 E8 2D 02 00 00 E8 DD 06 00 00 E8 2C 06 00 00 A1 ?? ?? ?? ?? C7 05 ?? ?? ?? ?? ?? ?? ?? ?? 01 05 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? C3 C3 56 57 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B 35 ?? ?? ?? ?? 8B F8 68 ?? ?? ?? ?? 57 FF D6 68 ?? ?? ?? ?? 57 A3 ?? ?? ?? ?? FF D6 5F A3 ?? ?? ?? ?? 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule npack_115002008b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "nPack"
		version = "1.1.500.2008b"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? ?? 75 05 E9 01 00 00 00 C3 E8 46 00 00 00 E8 73 00 00 00 B8 ?? ?? ?? ?? 2B 05 ?? ?? ?? ?? A3 ?? ?? ?? ?? E8 9C 00 00 00 E8 48 02 00 00 E8 F8 06 00 00 E8 47 06 00 00 A1 ?? ?? ?? ?? C7 05 ?? ?? ?? ?? ?? ?? ?? ?? 01 05 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? C3 C3 56 57 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? 8B 35 ?? ?? ?? ?? 8B F8 68 ?? ?? ?? ?? 57 FF D6 68 ?? ?? ?? ?? 57 A3 ?? ?? ?? ?? FF D6 5F A3 ?? ?? ?? ?? 5E C3 56 57 6A ?? 68 ?? ?? ?? ?? 6A ?? 6A ?? FF 15 ?? ?? ?? ?? 8B F0 BF ?? ?? ?? ?? 56 57 E8 23 FE FF FF 6A ?? 56 57 E8 F4 FC FF FF 83 C4 14 68 ?? ?? ?? ?? 6A ?? 56 FF 15 ?? ?? ?? ?? 5F 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule nspack_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "1.1"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 57 84 40 00 2D 50 84 40 00 }
	condition:
		$1
}

rule nspack_13 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "1.3"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 B3 85 40 00 2D AC 85 40 00 2B E8 8D B5 73 ?? FF FF 8B 06 83 F8 00 74 11 8D B5 7F ?? FF FF 8B 06 83 F8 01 0F 84 F1 01 00 00 C7 06 01 00 00 00 8B D5 8B 85 4F ?? FF FF 2B D0 89 95 4F ?? FF FF 01 95 67 ?? FF FF 8D B5 83 ?? FF FF 01 16 8B 36 8B FD 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 A3 ?? FF FF 85 C0 0F 84 06 03 00 00 89 85 63 ?? FF FF E8 00 00 00 00 5B B9 31 89 40 00 81 E9 2E 86 40 00 03 D9 50 53 E8 3D 02 00 00 61 03 BD 47 ?? FF FF 8B DF 83 3F 00 75 0A 83 C7 }
	condition:
		$1
}

rule nspack_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "1.4"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 B1 85 40 00 2D AA 85 40 00 }
	condition:
		$1
}

rule nspack_23_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "2.3"
	strings:
		$1 = { 9C 60 70 61 63 6B 24 40 }
	condition:
		$1
}

rule nspack_23_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "2.3"
	strings:
		$1 = { 9C 60 E8 ?? ?? 00 00 5D B8 07 00 00 00 2B E8 8D B5 ?? ?? FF FF }
	condition:
		$1
}

rule nspack_23_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "2.3"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 07 00 00 00 2B E8 8D B5 ?? ?? FF FF 8B 06 83 F8 00 74 11 8D B5 ?? ?? FF FF 8B 06 83 F8 01 0F 84 4B 02 00 00 C7 06 01 00 00 00 8B D5 8B 85 ?? ?? FF FF 2B D0 89 95 ?? ?? FF FF 01 95 ?? ?? FF FF 8D B5 ?? ?? FF FF 01 16 8B 36 8B FD 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 ?? ?? FF FF 85 C0 0F 84 56 03 00 00 89 85 ?? ?? FF FF E8 00 00 00 00 5B B9 54 03 00 00 03 D9 50 53 E8 9D 02 00 00 61 }
	condition:
		$1
}

rule nspack_29 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "2.9"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 07 00 00 00 2B E8 8D B5 ?? ?? FF FF 8A 06 3C 00 74 12 8B F5 8D B5 ?? ?? FF FF 8A 06 3C 01 0F 84 42 02 00 00 C6 06 01 8B D5 2B 95 ?? ?? FF FF 89 95 ?? ?? FF FF 01 95 ?? ?? FF FF 8D B5 ?? ?? FF FF 01 16 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 ?? ?? FF FF 85 C0 0F 84 6A 03 00 00 89 85 ?? ?? FF FF E8 00 00 00 00 5B B9 68 03 00 00 03 D9 50 53 E8 B1 02 00 00 61 8B 36 8B FD 03 BD ?? ?? FF FF 8B DF 83 3F 00 75 0A 83 C7 04 B9 00 00 00 00 EB 16 B9 01 00 00 00 03 3B 83 C3 04 83 3B 00 74 36 }
	condition:
		$1
}

rule nspack_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "3.0"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 07 00 00 00 2B E8 8D B5 ?? ?? ?? ?? 66 8B 06 66 83 F8 00 74 15 8B F5 8D B5 ?? ?? FF FF 66 8B 06 66 83 F8 01 0F 84 42 02 00 00 C6 06 01 8B D5 2B 95 ?? ?? FF FF 89 95 ?? ?? FF FF 01 95 ?? ?? FF FF 8D B5 ?? ?? FF FF 01 16 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 ?? ?? FF FF 85 C0 0F 84 6A 03 00 00 89 85 ?? ?? FF FF E8 00 00 00 00 5B B9 68 03 00 00 03 D9 50 53 E8 B1 02 00 00 61 8B 36 8B FD 03 BD ?? ?? FF FF 8B DF 83 3F 00 75 0A 83 C7 04 B9 00 00 00 00 EB 16 B9 01 00 00 00 03 3B 83 C3 04 83 3B 00 74 36 }
	condition:
		$1
}

rule nspack_31 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "3.1"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D 83 ED 07 8D 9D ?? ?? ?? ?? 8A 03 3C 00 74 10 8D 9D ?? ?? FF FF 8A 03 3C 01 0F 84 42 02 00 00 C6 03 01 8B D5 2B 95 ?? ?? FF FF 89 95 ?? ?? FF FF 01 95 ?? ?? FF FF 8D B5 ?? ?? FF FF 01 16 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 ?? ?? FF FF 85 C0 0F 84 6A 03 00 00 89 85 ?? ?? FF FF E8 00 00 00 00 5B B9 68 03 00 00 03 D9 50 53 E8 B1 02 00 00 61 8B 36 8B FD 03 BD ?? ?? FF FF 8B DF 83 3F 00 75 0A 83 C7 04 B9 00 00 00 00 EB 16 B9 01 00 00 00 03 3B 83 C3 04 83 3B 00 74 36 01 13 8B 33 03 7B 04 57 51 52 53 FF B5 ?? ?? FF FF FF B5 ?? ?? FF FF 8B D6 8B CF 8B 85 ?? ?? FF FF 05 AA 05 00 00 FF D0 5B 5A 59 5F 83 F9 00 74 05 83 C3 08 EB C5 68 00 80 00 00 6A 00 }
	condition:
		$1
}

rule nspack_33 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "3.3"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D 83 ED 07 8D 85 ?? ?? ?? ?? 80 38 00 74 }
	condition:
		$1
}

rule nspack_34 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "3.4"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D 83 ED 07 8D 85 ?? ?? FF FF 80 38 01 0F 84 42 02 00 00 C6 00 01 8B D5 2B 95 ?? ?? FF FF 89 95 ?? ?? FF FF 01 95 ?? ?? FF FF 8D B5 ?? ?? FF FF 01 16 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 ?? ?? FF FF 85 C0 0F 84 6A 03 00 00 89 85 ?? ?? FF FF E8 00 00 00 00 5B B9 68 03 00 00 03 D9 50 53 E8 B1 02 00 00 61 8B 36 8B FD 03 BD ?? ?? FF FF 8B DF 83 3F 00 75 0A 83 C7 04 B9 00 00 00 00 EB 16 B9 01 00 00 00 03 3B 83 C3 04 83 3B 00 74 36 01 13 8B 33 03 7B 04 57 51 52 53 FF B5 ?? ?? FF FF FF B5 ?? ?? FF FF 8B D6 8B CF 8B 85 ?? ?? FF FF 05 AA 05 00 00 FF D0 5B 5A 59 5F 83 F9 00 74 05 83 C3 08 EB C5 }
	condition:
		$1
}

rule nspack_36 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "3.6"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D 83 ED 07 8D ?? ?? ?? ?? ?? 83 38 01 0F 84 47 02 00 00 }
	condition:
		$1
}

rule nspack_37 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NsPacK"
		version = "3.7"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D 83 ED 07 8D ?? ?? ?? ?? ?? 80 39 01 0F ?? ?? ?? 00 00 }
	condition:
		$1
}

rule ntkrnl_packer {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NTKrnl"
		description = "Packer"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 34 10 00 00 28 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 }
	condition:
		$1 at pe.entry_point
}

rule ntkrnl_secure_suite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NTkrnl"
		description = "Secure Suite"
	strings:
		$1 = { 34 10 00 00 28 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 41 10 00 00 50 10 00 00 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 }
	condition:
		$1 at pe.entry_point
}

rule ntpacker_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NTPacker"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 E0 53 33 C0 89 45 E0 89 45 E4 89 45 E8 89 45 EC B8 ?? ?? 40 00 E8 ?? ?? FF FF 33 C0 55 68 ?? ?? 40 00 64 FF 30 64 89 20 8D 4D EC BA ?? ?? 40 00 A1 ?? ?? 40 00 E8 ?? FC FF FF 8B 55 EC B8 ?? ?? 40 00 E8 ?? ?? FF FF 8D 4D E8 BA ?? ?? 40 00 A1 ?? ?? 40 00 E8 ?? FE FF FF 8B 55 E8 B8 ?? ?? 40 00 E8 ?? ?? FF FF B8 ?? ?? 40 00 E8 ?? FB FF FF 8B D8 A1 ?? ?? 40 00 BA ?? ?? 40 00 E8 ?? ?? FF FF 75 26 8B D3 A1 ?? ?? 40 00 E8 ?? ?? FF FF 84 C0 75 2A 8D 55 E4 33 C0 E8 ?? ?? FF FF 8B 45 E4 8B D3 E8 ?? ?? FF FF EB 14 8D 55 E0 33 C0 E8 ?? ?? FF FF 8B 45 E0 8B D3 E8 ?? ?? FF FF 6A 00 E8 ?? ?? FF FF 33 C0 5A 59 59 64 89 10 68 ?? ?? 40 00 8D 45 E0 BA 04 00 00 00 E8 ?? ?? FF FF C3 E9 ?? ?? FF FF EB EB 5B E8 ?? ?? FF FF 00 00 00 FF FF FF FF 01 00 00 00 25 00 00 00 FF FF FF FF 01 00 00 00 5C 00 00 00 FF FF FF FF 06 00 00 00 53 45 52 56 45 52 00 00 FF FF FF FF 01 00 00 00 31 }
	condition:
		$1 at pe.entry_point
}

rule ntpacker_2x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NTPacker"
		version = "2.x"
	strings:
		$1 = { 4B 57 69 6E 64 6F 77 73 00 10 55 54 79 70 65 73 00 00 3F 75 6E 74 4D 61 69 6E 46 75 6E 63 74 69 6F 6E 73 00 00 47 75 6E 74 42 79 70 61 73 73 00 00 B7 61 50 4C 69 62 75 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule nx_pe_packer_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "NX PE Packer"
		version = "1.0"
	strings:
		$1 = { FF 60 FF CA FF 00 BA DC 0D E0 40 00 50 00 60 00 70 00 80 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_10059f {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.0.0.59f"
	strings:
		$1 = { E8 AB 1C }
	condition:
		$1 at pe.entry_point
}

rule obsidium_10061 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.0.0.61"
	strings:
		$1 = { E8 AF 1C 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1111 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.1.1.1"
	strings:
		$1 = { EB 02 ?? ?? E8 E7 1C 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1200_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.2.0.0"
	strings:
		$1 = { EB 02 ?? ?? E8 3F 1E 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1200_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.2.0.0"
	strings:
		$1 = { EB 02 ?? ?? E8 77 1E 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1250 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.2.5.0"
	strings:
		$1 = { E8 0E 00 00 00 8B 54 24 0C 83 82 B8 00 00 00 0D 33 C0 C3 64 67 FF 36 00 00 64 67 89 26 00 00 50 33 C0 8B 00 C3 E9 FA 00 00 00 E8 D5 FF FF FF 58 64 67 8F 06 00 00 83 C4 04 E8 2B 13 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1258 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.2.5.8"
	strings:
		$1 = { EB 01 ?? E8 29 00 00 00 EB 02 ?? ?? EB 01 ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 24 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 02 ?? ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 01 ?? 64 67 89 26 00 00 EB 03 ?? ?? ?? EB 01 ?? 50 EB 03 ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 01 ?? E9 FA 00 00 00 EB 02 ?? ?? E8 D5 FF FF FF EB 04 ?? ?? ?? ?? EB 03 ?? ?? ?? EB 01 ?? 58 EB 01 ?? EB 02 ?? ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 01 ?? E8 7B 21 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_12xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.2.x.x"
	strings:
		$1 = { E8 0E 00 00 00 33 C0 8B 54 24 0C 83 82 B8 00 00 00 0D C3 64 67 FF 36 00 00 64 67 89 26 00 00 50 33 C0 8B 00 C3 E9 FA 00 00 00 E8 D5 FF FF FF 58 64 67 8F 06 00 00 83 C4 04 E8 2B 13 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1300 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.0"
	strings:
		$1 = { EB 04 ?? ?? ?? ?? E8 29 00 00 00 EB 02 ?? ?? EB 01 ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 22 EB 02 ?? ?? 33 C0 EB 04 ?? ?? ?? ?? C3 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 04 ?? ?? ?? ?? EB 01 ?? 50 EB 03 ?? ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 01 ?? C3 EB 04 ?? ?? ?? ?? E9 FA 00 00 00 EB 01 ?? E8 D5 FF FF FF EB 02 ?? ?? EB 03 ?? ?? ?? 58 EB 04 ?? ?? ?? ?? EB 01 ?? 64 67 8F 06 00 00 EB 02 ?? ?? 83 C4 04 EB 02 ?? ?? E8 47 26 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1304 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.4"
	strings:
		$1 = { EB 02 ?? ?? E8 25 00 00 00 EB 04 ?? ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 01 ?? 83 82 B8 00 00 00 23 EB 01 ?? 33 C0 EB 02 ?? ?? C3 EB 02 ?? ?? EB 04 ?? ?? ?? ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 01 ?? 50 EB 01 ?? 33 C0 EB 01 ?? 8B 00 EB 01 ?? C3 EB 02 ?? ?? E9 FA 00 00 00 EB 02 ?? ?? E8 D5 FF FF FF EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 58 EB 02 ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 01 ?? E8 3B 26 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_13013 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.13"
	strings:
		$1 = { EB 01 ?? E8 26 00 00 00 EB 02 ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 01 ?? 83 82 B8 00 00 00 21 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 01 ?? EB 04 ?? ?? ?? ?? 64 67 FF 36 00 00 EB 02 ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 03 ?? ?? ?? 50 EB 01 ?? 33 C0 EB 03 ?? ?? ?? 8B 00 EB 02 ?? ?? C3 EB 02 ?? ?? E9 FA 00 00 00 EB 01 ?? E8 D5 FF FF FF EB 03 ?? ?? ?? EB 02 ?? ?? 58 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 03 ?? ?? ?? E8 13 26 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_13017 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.17"
	strings:
		$1 = { EB 02 ?? ?? E8 28 00 00 00 EB 04 ?? ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 01 ?? 83 82 B8 00 00 00 25 EB 02 ?? ?? 33 C0 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 01 ?? 64 67 89 26 00 00 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 01 ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 04 ?? ?? ?? ?? EB 02 ?? ?? 58 EB 03 ?? ?? ?? EB 01 ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 02 ?? ?? E8 4F 26 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_13021 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.21"
	strings:
		$1 = { EB 03 ?? ?? ?? E8 2E 00 00 00 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 23 EB 01 ?? 33 C0 EB 04 ?? ?? ?? ?? C3 EB 03 ?? ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 01 ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 02 ?? ?? 50 EB 01 ?? 33 C0 EB 03 ?? ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 01 ?? EB 01 ?? 58 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 04 ?? ?? ?? ?? E8 2B 26 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_13037 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.37"
	strings:
		$1 = { EB 02 ?? ?? E8 26 00 00 00 EB 03 ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 26 EB 01 ?? 33 C0 EB 02 ?? ?? C3 EB 01 ?? EB 04 ?? ?? ?? ?? 64 67 FF 36 00 00 EB 01 ?? 64 67 89 26 00 00 EB 01 ?? EB 03 ?? ?? ?? 50 EB 03 ?? ?? ?? 33 C0 EB 03 ?? ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 04 ?? ?? ?? ?? EB 01 ?? 58 EB 02 ?? ?? EB 03 ?? ?? ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 03 ?? ?? ?? E8 23 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_130x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.0.x"
	strings:
		$1 = { EB 03 ?? ?? ?? E8 2E 00 00 00 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 8B ?? ?? ?? EB 04 ?? ?? ?? ?? 83 ?? ?? ?? ?? ?? ?? EB 01 ?? 33 C0 EB 04 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1311 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.1.1"
	strings:
		$1 = { EB 02 ?? ?? E8 27 00 00 00 EB 02 ?? ?? EB 03 ?? ?? ?? 8B 54 24 0C EB 01 ?? 83 82 B8 00 00 00 22 EB 04 ?? ?? ?? ?? 33 C0 EB 01 ?? C3 EB 02 ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 03 ?? ?? ?? 50 EB 03 ?? ?? ?? 33 C0 EB 01 ?? 8B 00 EB 03 ?? ?? ?? C3 EB 01 ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 01 ?? EB 03 ?? ?? ?? 58 EB 03 ?? ?? ?? EB 01 ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 03 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1322 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.2.2"
	strings:
		$1 = { EB 04 ?? ?? ?? ?? E8 2A 00 00 00 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 26 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 01 ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 02 ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 01 ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? 8B 00 EB 02 ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 02 ?? ?? EB 04 ?? ?? ?? ?? 58 EB 01 ?? EB 01 ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 04 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1331 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.1"
	strings:
		$1 = { EB 01 ?? E8 29 00 00 00 EB 02 ?? ?? EB 03 ?? ?? ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 24 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 02 ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 02 ?? ?? 50 EB 01 ?? 33 C0 EB 04 ?? ?? ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 02 ?? ?? E8 D5 FF FF FF EB 01 ?? EB 04 ?? ?? ?? ?? 58 EB 02 ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 02 ?? ?? E8 5F 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1332 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.2"
	strings:
		$1 = { EB 01 ?? E8 2B 00 00 00 EB 02 ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 24 EB 04 ?? ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? C3 EB 02 ?? ?? EB 01 ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 02 ?? ?? 50 EB 02 ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 02 ?? ?? C3 EB 04 ?? ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 03 ?? ?? ?? EB 01 ?? 58 EB 01 ?? EB 02 ?? ?? 64 67 8F 06 00 00 EB 02 ?? ?? 83 C4 04 EB 02 ?? ?? E8 3B 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1333 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.3"
	strings:
		$1 = { EB 02 ?? ?? E8 29 00 00 00 EB 03 ?? ?? ?? EB 03 ?? ?? ?? 8B ?? 24 0C EB 01 ?? 83 ?? B8 00 00 00 28 EB 03 ?? ?? ?? 33 C0 EB 01 ?? C3 EB 04 ?? ?? ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 04 ?? ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 01 ?? 8B 00 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 58 EB 01 ?? EB 03 ?? ?? ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 04 ?? ?? ?? ?? E8 2B 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1334 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.4"
	strings:
		$1 = { EB 02 ?? ?? E8 29 00 00 00 EB 03 ?? ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 25 EB 02 ?? ?? 33 C0 EB 02 ?? ?? C3 EB 03 ?? ?? ?? EB 01 ?? 64 67 FF 36 00 00 EB 02 ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 04 ?? ?? ?? ?? 50 EB 02 ?? ?? 33 C0 EB 01 ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 02 ?? ?? E8 D5 FF FF FF EB 02 ?? ?? EB 03 ?? ?? ?? 58 EB 02 ?? ?? EB 03 ?? ?? ?? 64 67 8F 06 00 00 EB 03 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1336 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.6"
	strings:
		$1 = { EB 04 ?? ?? ?? ?? E8 28 00 00 00 EB 01 ?? ?? ?? ?? ?? ?? ?? 8B 54 24 0C EB 01 ?? 83 82 B8 00 00 00 26 EB 04 ?? ?? ?? ?? 33 C0 EB 01 ?? C3 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 50 EB 01 ?? 33 C0 EB 02 ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 04 ?? ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 01 ?? EB 03 ?? ?? ?? 58 EB 02 ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 04 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1337_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.7"
	strings:
		$1 = { EB 02 ?? ?? E8 2C 00 00 00 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 27 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 02 ?? ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 03 ?? ?? ?? EB 01 ?? 50 EB 02 ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 02 ?? ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 02 ?? ?? EB 04 ?? ?? ?? ?? 58 EB 04 ?? ?? ?? ?? EB 03 ?? ?? ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 03 ?? ?? ?? E8 23 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1337_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.7"
	strings:
		$1 = { EB 02 ?? ?? E8 27 00 00 00 EB 03 ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 23 EB 03 ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 01 ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 01 ?? 50 EB 02 ?? ?? 33 C0 EB 01 ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 02 ?? ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 01 ?? EB 01 ?? 58 EB 04 ?? ?? ?? ?? EB 01 ?? 64 67 8F 06 00 00 EB 02 ?? ?? 83 C4 04 EB 01 ?? E8 F7 26 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1338 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.8"
	strings:
		$1 = { EB 04 ?? ?? ?? ?? E8 28 00 00 00 EB 01 ?? EB 01 ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 ?? EB 04 ?? ?? ?? ?? 33 C0 EB 03 ?? ?? ?? C3 EB 01 ?? EB 01 ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 01 ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 02 ?? ?? EB 04 ?? ?? ?? ?? 58 EB 04 ?? ?? ?? ?? EB 02 ?? ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 04 ?? ?? ?? ?? E8 57 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1339 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.3.9"
	strings:
		$1 = { EB 02 ?? ?? E8 29 00 00 00 EB 03 ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 28 EB 02 ?? ?? 33 C0 EB 02 ?? ?? C3 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 01 ?? 50 EB 03 ?? ?? ?? 33 C0 EB 03 ?? ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 04 ?? ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 02 ?? ?? EB 04 ?? ?? ?? ?? 58 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 04 ?? ?? ?? ?? E8 CF 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1341 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.4.1"
	strings:
		$1 = { EB 01 ?? E8 2A 00 00 00 EB 04 ?? ?? ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 21 EB 02 ?? ?? 33 C0 EB 03 ?? ?? ?? C3 EB 02 ?? ?? EB 01 ?? 64 67 FF 36 00 00 EB 01 ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 03 ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 02 ?? ?? E9 FA 00 00 00 EB 02 ?? ?? E8 D5 FF FF FF EB 01 ?? EB 01 ?? 58 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 02 ?? ?? E8 C3 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1342 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.4.2"
	strings:
		$1 = { EB 02 ?? ?? E8 26 00 00 00 EB 03 ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 24 EB 03 ?? ?? ?? 33 C0 EB 01 ?? C3 EB 02 ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 03 ?? ?? ?? EB 03 ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 03 ?? ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 03 ?? ?? ?? E8 D5 FF FF FF EB 01 ?? EB 03 ?? ?? ?? 58 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 01 ?? E8 C3 27 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1350 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.5.0"
	strings:
		$1 = { EB 03 ?? ?? ?? E8 ?? ?? ?? ?? EB 02 ?? ?? EB 04 ?? ?? ?? ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 20 EB 03 ?? ?? ?? 33 C0 EB 01 ?? C3 EB 02 ?? ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 04 ?? ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 02 ?? ?? E9 FA 00 00 00 EB 01 ?? E8 ?? ?? ?? ?? EB 01 ?? EB 02 ?? ?? 58 EB 04 ?? ?? ?? ?? EB 02 ?? ?? 64 67 8F 06 00 00 EB 02 ?? ?? 83 C4 04 EB 01 ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1352 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.5.2"
	strings:
		$1 = { EB 04 ?? ?? ?? ?? E8 28 00 00 00 EB 01 ?? EB 01 ?? 8B 54 24 0C EB 01 ?? 83 82 B8 00 00 00 25 EB 03 ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? C3 EB 04 ?? ?? ?? ?? EB 01 ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 03 ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 01 ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 02 ?? ?? EB 04 ?? ?? ?? ?? 58 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 03 ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1353 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.5.3"
	strings:
		$1 = { EB 02 ?? ?? E8 2B 00 00 00 EB 04 ?? ?? ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 24 EB 02 ?? ?? 33 C0 EB 02 ?? ?? C3 EB 04 ?? ?? ?? ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 01 ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 01 ?? EB 01 ?? 58 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 02 ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1354 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.5.4"
	strings:
		$1 = { EB 03 ?? ?? ?? E8 2D 00 00 00 EB 04 ?? ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 25 EB 03 ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? C3 EB 03 ?? ?? ?? EB 01 ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 03 ?? ?? ?? EB 02 ?? ?? 50 EB 01 ?? 33 C0 EB 02 ?? ?? 8B 00 EB 04 ?? ?? ?? ?? C3 EB 01 ?? E9 FA 00 00 00 EB 04 ?? ?? ?? ?? E8 D5 FF FF FF EB 03 ?? ?? ?? EB 02 ?? ?? 58 EB 04 ?? ?? ?? ?? EB 03 ?? ?? ?? 64 67 8F 06 00 00 EB 03 ?? ?? ?? 83 C4 04 EB 04 ?? ?? ?? ?? E8 5B 28 00 00 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1355 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.5.5"
	strings:
		$1 = { EB 01 ?? E8 2B 00 00 00 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 23 EB 03 ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 03 ?? ?? ?? EB 02 ?? ?? 64 67 FF 36 00 00 EB 01 ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 02 ?? ?? 50 EB 03 ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? E9 ?? ?? ?? ?? EB 01 ?? E8 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? EB 01 ?? 58 EB 03 ?? ?? ?? EB 02 ?? ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 01 ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1357 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.5.7"
	strings:
		$1 = { EB 01 ?? E8 ?? 00 00 00 EB 03 ?? ?? ?? EB 01 ?? 8B 54 24 0C EB 02 ?? ?? 83 82 B8 00 00 00 24 EB 03 ?? ?? ?? 33 C0 EB 02 ?? ?? C3 EB 02 ?? ?? EB 01 ?? 64 67 FF 36 00 00 EB 04 ?? ?? ?? ?? 64 67 89 26 00 00 EB 01 ?? EB 02 ?? ?? 50 EB 03 ?? ?? ?? 33 C0 EB 01 ?? 8B 00 EB 03 ?? ?? ?? C3 EB 01 ?? E9 ?? ?? ?? ?? EB 03 ?? ?? ?? E8 ?? ?? ?? ?? EB 03 ?? ?? ?? EB 03 ?? ?? ?? 58 EB 01 ?? EB 02 ?? ?? 64 67 8F 06 00 00 EB 01 ?? 83 C4 04 EB 01 ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1360 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.6.0"
	strings:
		$1 = { EB 02 ?? ?? 50 EB 01 ?? E8 ?? 00 00 00 EB 03 ?? ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 04 ?? ?? ?? ?? 83 82 B8 00 00 00 1F EB 04 ?? ?? ?? ?? 33 C0 EB 01 ?? C3 EB 03 ?? ?? ?? EB 02 ?? ?? 33 C0 EB 01 ?? 64 FF 30 EB 04 ?? ?? ?? ?? 64 89 20 EB 03 ?? ?? ?? EB 02 ?? ?? 8B 00 EB 01 ?? C3 EB 02 ?? ?? E9 ?? 00 00 00 EB 01 ?? E8 ?? FF FF FF EB 01 ?? EB 03 ?? ?? ?? EB 02 ?? ?? EB 02 ?? ?? 64 8F 00 EB 01 ?? 83 C4 04 EB 03 ?? ?? ?? 58 EB 04 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1361 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.6.1"
	strings:
		$1 = { EB 04 ?? ?? ?? ?? 50 EB 02 ?? ?? E8 ?? 00 00 00 EB 03 ?? ?? ?? EB 02 ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 ?? EB 02 ?? ?? 33 C0 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? EB 01 ?? 33 C0 EB 04 ?? ?? ?? ?? 64 FF 30 EB 04 ?? ?? ?? ?? 64 89 20 EB 01 ?? EB 03 ?? ?? ?? 8B 00 EB 02 ?? ?? C3 EB 03 ?? ?? ?? E9 FA 00 00 00 EB 01 ?? E8 ?? FF FF FF EB 01 ?? EB 03 ?? ?? ?? EB 01 ?? EB 03 ?? ?? ?? 64 8F 00 EB 03 ?? ?? ?? 83 C4 04 EB 01 ?? 58 EB 02 ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1363 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.3.6.3"
	strings:
		$1 = { EB 03 ?? ?? ?? 50 EB 04 ?? ?? ?? ?? E8 ?? 00 00 00 EB 04 ?? ?? ?? ?? EB 03 ?? ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 26 EB 03 ?? ?? ?? 33 C0 EB 03 ?? ?? ?? C3 EB 03 ?? ?? ?? EB 02 ?? ?? 33 C0 EB 02 ?? ?? 64 FF 30 EB 01 ?? 64 89 20 EB 01 ?? EB 02 ?? ?? 8B 00 EB 03 ?? ?? ?? C3 EB 04 ?? ?? ?? ?? E9 ?? 00 00 00 EB 03 ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_1400b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
		version = "1.4.0.0b"
	strings:
		$1 = { EB 01 ?? E8 2F 00 00 00 EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 8B 54 24 0C EB 03 ?? ?? ?? 83 82 B8 00 00 00 21 EB 04 ?? ?? ?? ?? 33 C0 EB 04 ?? ?? ?? ?? C3 EB 03 ?? ?? ?? EB 03 ?? ?? ?? 64 67 FF 36 00 00 EB 03 ?? ?? ?? 64 67 89 26 00 00 EB 02 ?? ?? EB 03 ?? ?? ?? 50 EB 04 ?? ?? ?? ?? 33 C0 EB 02 ?? ?? 8B 00 EB 01 ?? C3 EB 01 ?? E9 ?? ?? ?? ?? EB 01 ?? E8 D5 FF FF FF EB 03 ?? ?? ?? EB 04 ?? ?? ?? ?? 58 EB 04 ?? ?? ?? ?? EB 04 ?? ?? ?? ?? 64 67 8F 06 00 00 EB 04 ?? ?? ?? ?? 83 C4 04 EB 04 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule obsidium_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Obsidium"
	strings:
		$1 = { E8 47 19 }
	condition:
		$1 at pe.entry_point
}

rule open_source_code_crypter {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Open Source Code Crypter"
	strings:
		$1 = { 55 8B EC B9 09 00 00 00 6A 00 6A 00 49 75 F9 53 56 57 B8 34 44 40 00 E8 28 F8 FF FF 33 C0 55 68 9F 47 40 00 64 FF 30 64 89 20 BA B0 47 40 00 B8 1C 67 40 00 E8 07 FD FF FF 8B D8 85 DB 75 07 6A 00 E8 C2 F8 FF FF BA 28 67 40 00 8B C3 8B 0D 1C 67 40 00 E8 F0 E0 FF FF BE 01 00 00 00 B8 2C 68 40 00 E8 E1 F0 FF FF BF 0A 00 00 00 8D 55 EC 8B C6 E8 92 FC FF FF 8B 4D EC B8 2C 68 40 00 BA BC 47 40 00 E8 54 F2 FF FF A1 2C 68 40 00 E8 52 F3 FF FF 8B D0 B8 20 67 40 00 E8 A2 FC FF FF 8B D8 85 DB 0F 84 52 02 00 00 B8 24 67 40 00 8B 15 20 67 40 00 E8 78 F4 FF FF B8 24 67 40 00 E8 7A F3 FF FF 8B D0 8B C3 8B 0D 20 67 40 00 E8 77 E0 FF FF 8D 55 E8 A1 24 67 40 00 E8 42 FD FF FF 8B 55 E8 B8 24 67 40 00 }
	condition:
		$1 at pe.entry_point
}

rule orien_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ORiEN"
	strings:
		$1 = { E9 ?? ?? ?? 00 CE D1 CE ?? 0D 0A 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 2D 0D 0A 2D 20 4F 52 69 45 4E 20 65 78 65 63 75 74 61 62 6C 65 20 66 69 6C 65 73 20 70 72 6F }
	condition:
		$1 at pe.entry_point
}

rule orien_1xx_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ORiEN"
		version = "1.xx - 2.xx"
	strings:
		$1 = { 4F 52 69 45 4E 20 65 78 65 63 75 74 61 62 6C 65 20 66 69 6C 65 73 20 70 72 6F 74 65 63 74 69 6F 6E 20 73 79 73 74 65 6D }
	condition:
		$1 at pe.entry_point
}

rule orien_211_212 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ORiEN"
		version = "2.11 - 2.12"
	strings:
		$1 = { E9 5D 01 00 00 CE D1 CE ?? 0D }
	condition:
		$1 at pe.entry_point
}

rule pack_master_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pack Master"
		version = "1.0"
	strings:
		$1 = { 60 E8 01 ?? ?? ?? E8 83 C4 04 E8 01 ?? ?? ?? E9 5D 81 ED D3 22 40 ?? E8 04 02 ?? ?? E8 EB 08 EB 02 CD 20 FF 24 24 9A 66 BE 47 46 }
	condition:
		$1 at pe.entry_point
}

rule pack_master_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pack Master"
		version = "1.0"
	strings:
		$1 = { 60 E8 01 00 00 00 E8 83 C4 04 E8 01 00 00 00 E9 5D 81 ED D3 22 40 00 E8 04 02 00 00 E8 EB 08 EB 02 CD 20 FF 24 24 9A 66 BE 47 46 }
	condition:
		$1 at pe.entry_point
}

rule packanoid_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Packanoid"
	strings:
		$1 = { BF ?? ?? ?? ?? BE ?? ?? ?? ?? E8 9D 00 00 00 B8 }
	condition:
		$1 at pe.entry_point
}

rule packitbitch_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PackItBitch"
		version = "1.0"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 ?? 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule packitbitch_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PackItBitch"
		version = "1.0"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 28 ?? ?? ?? 35 ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 41 ?? ?? ?? 50 ?? ?? ?? 00 00 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 ?? ?? ?? ?? ?? ?? ?? 79 ?? ?? ?? 7D ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pacman_0001_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Packman"
		version = "0.0.0.1"
	strings:
		$1 = { 60 E8 00 00 00 00 58 8D ?? ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 8D }
	condition:
		$1 at pe.entry_point
}

rule pacman_0001_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Packman"
		version = "0.0.0.1"
	strings:
		$1 = { 0F 85 ?? FF FF FF 8D B3 ?? ?? ?? ?? EB 3D 8B 46 0C 03 C3 50 FF 55 00 56 8B 36 0B F6 75 02 8B F7 03 F3 03 FB EB 1B D1 C1 D1 E9 73 05 0F B7 C9 EB 05 03 CB 8D 49 02 50 51 50 FF 55 04 AB 58 83 C6 04 8B 0E 85 C9 75 DF 5E 83 C6 14 8B 7E 10 85 FF 75 BC 8D 8B 00 00 ?? ?? B8 00 ?? ?? 00 0B C0 74 34 03 C3 EB 2A 8D 70 08 03 40 04 33 ED 33 D2 66 8B 2E 66 0F A4 EA 04 80 FA 03 75 0D 81 E5 FF 0F 00 00 03 EF 03 EB 01 4D 00 46 46 3B F0 75 DC 8B 38 85 FF 75 D0 61 E9 ?? FE FF FF 02 D2 75 05 8A 16 46 12 D2 C3 }
	condition:
		$1 at pe.entry_point
}

rule pacman_1000 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Packman"
		version = "1.0.0.0"
	strings:
		$1 = { 60 E8 00 00 00 00 5B 8D 5B C6 01 1B 8B 13 8D 73 14 6A 08 59 01 16 AD 49 75 FA }
	condition:
		$1 at pe.entry_point
}

rule passlock_2000_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PassLock 2000"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 50 40 00 66 2E F7 05 34 20 40 00 04 00 0F 85 98 00 00 00 E8 1F 01 00 00 C7 43 60 01 00 00 00 8D 83 E4 01 00 00 50 FF 15 F0 61 40 00 83 EC 44 C7 04 24 44 00 00 00 C7 44 24 2C 00 00 00 00 54 FF 15 E8 61 40 00 B8 0A 00 00 00 F7 44 24 2C 01 00 00 00 74 05 0F B7 44 24 30 83 C4 44 89 43 56 FF 15 D0 61 40 00 E8 9E 00 00 00 89 43 4C FF 15 D4 61 40 00 89 43 48 6A 00 FF 15 E4 61 40 00 89 43 5C E8 F9 00 00 00 E8 AA 00 00 00 B8 FF 00 00 00 72 0D 53 E8 96 00 00 00 5B FF 4B 10 FF 4B 18 }
	condition:
		$1 at pe.entry_point
}

rule password_protector_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Password Protector"
	strings:
		$1 = { 06 0E 0E 07 1F E8 00 00 5B 83 EB 08 BA 27 01 03 D3 E8 3C 02 BA EA }
	condition:
		$1 at pe.entry_point
}

rule password_protector_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Password Protector"
	strings:
		$1 = { E8 ?? ?? ?? ?? 5D 8B FD 81 ?? ?? ?? ?? ?? 81 ?? ?? ?? ?? ?? 83 ?? ?? 89 ?? ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 46 80 ?? ?? 74 }
	condition:
		$1 at pe.entry_point
}

rule patch_creation_wizzard_12_byte_patch {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Patch Creation Wizard"
		version = "1.2 Byte Patch"
	strings:
		$1 = { E8 7F 03 00 00 6A 00 E8 24 03 00 00 A3 B8 33 40 00 6A 00 68 29 10 40 00 6A 00 6A 01 50 E8 2C 03 00 00 6A 00 E8 EF 02 00 00 55 8B EC 56 51 57 8B 45 0C 98 3D 10 01 00 00 0F 85 C1 00 00 00 6A 01 FF 35 B8 33 40 00 E8 1B 03 00 00 50 6A 01 68 80 00 00 00 FF 75 08 E8 1D 03 00 00 68 5F 30 40 00 6A 65 FF 75 08 E8 14 03 00 00 68 B0 30 40 00 6A 67 FF 75 08 E8 05 03 00 00 68 01 31 40 00 6A 66 FF 75 08 E8 F6 02 00 00 6A 00 FF 75 08 E8 C8 02 00 00 A3 B4 33 40 00 C7 05 BC 33 40 00 2C 00 00 00 C7 05 C0 33 40 00 }
	condition:
		$1 at pe.entry_point
}

rule patch_creation_wizzard_12_mem_patch {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Patch Creation Wizard"
		version = "1.2 Memory Patch"
	strings:
		$1 = { 6A 00 E8 9B 02 00 00 A3 7A 33 40 00 6A 00 68 8E 10 40 00 6A 00 6A 01 50 E8 B5 02 00 00 68 5A 31 40 00 68 12 31 40 00 6A 00 6A 00 6A 04 6A 01 6A 00 6A 00 68 A2 30 40 00 6A 00 E8 51 02 00 00 85 C0 74 31 FF 35 62 31 40 00 6A 00 6A 30 E8 62 02 00 00 E8 0B 01 00 00 FF 35 5A 31 40 00 E8 22 02 00 00 FF 35 5E 31 40 00 E8 53 02 00 00 6A 00 E8 22 02 00 00 6A 10 68 F7 30 40 00 68 FE 30 40 00 6A 00 E8 63 02 00 00 6A 00 E8 08 02 00 00 55 8B EC 56 51 57 8B 45 0C 98 3D 10 01 00 00 75 6B 6A 01 FF 35 7A 33 40 00 }
	condition:
		$1 at pe.entry_point
}

rule patch_creation_wizzard_12_seekndestroy {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Patch Creation Wizard"
		version = "1.2 Seek and Destroy Patch"
	strings:
		$1 = { E8 C5 05 00 00 6A 00 E8 5E 05 00 00 A3 CE 39 40 00 6A 00 68 29 10 40 00 6A 00 6A 01 50 E8 72 05 00 00 6A 00 E8 2F 05 00 00 55 8B EC 56 51 57 8B 45 0C 98 3D 10 01 00 00 0F 85 C1 00 00 00 6A 01 FF 35 CE 39 40 00 E8 61 05 00 00 50 6A 01 68 80 00 00 00 FF 75 08 E8 63 05 00 00 68 5F 30 40 00 6A 65 FF 75 08 E8 5A 05 00 00 68 B0 30 40 00 6A 67 FF 75 08 E8 4B 05 00 00 68 01 31 40 00 6A 66 FF 75 08 E8 3C 05 00 00 6A 00 FF 75 08 E8 0E 05 00 00 A3 CA 39 40 00 C7 05 D2 39 40 00 2C 00 00 00 C7 05 D6 39 40 00 }
	condition:
		$1 at pe.entry_point
}

rule pawning_antivirus_cryptor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pawning AntiVirus Cryptor"
	strings:
		$1 = { 53 56 57 55 BB 2C ?? ?? 70 BE 00 30 00 70 BF 20 ?? ?? 70 80 7B 28 00 75 16 83 3F 00 74 11 8B 17 89 D0 33 D2 89 17 8B E8 FF D5 83 3F 00 75 EF 83 3D 04 30 00 70 00 74 06 FF 15 54 30 00 70 80 7B 28 02 75 0A 83 3E 00 75 05 33 C0 89 43 0C FF 15 1C 30 00 70 80 7B 28 01 76 05 83 3E 00 74 22 8B 43 10 85 C0 74 1B FF 15 14 30 00 70 8B 53 10 8B 42 10 3B 42 04 74 0A 85 C0 74 06 50 E8 8F FA FF FF FF 15 20 30 00 70 80 7B 28 01 75 03 FF 53 24 80 7B 28 00 74 05 E8 35 FF FF FF 83 3B 00 75 17 83 3D 10 ?? ?? 70 00 74 06 FF 15 10 ?? ?? 70 8B 06 50 E8 A9 FA FF FF 8B 03 56 8B F0 8B FB B9 0B 00 00 00 F3 A5 5E E9 73 FF FF FF 5D 5F 5E 5B C3 A3 00 30 00 70 E8 26 FF FF FF C3 90 8F 05 04 30 00 70 E9 E9 FF FF FF C3 }
	condition:
		$1 at pe.entry_point
}

rule pc_guard_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PC Guard"
	strings:
		$1 = { FC 55 50 E8 00 00 00 00 5D EB 01 E3 60 E8 03 00 00 00 D2 EB 0B 58 EB 01 48 40 EB 01 35 FF E0 E7 61 B8 ?? ?? ?? ?? 60 E8 03 00 00 00 83 EB 0E EB 01 0C 58 EB 01 35 40 EB 01 36 FF E0 0B 61 2B E8 }
	condition:
		$1 at pe.entry_point
}

rule pc_guard_303d_305d {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PC Guard"
		version = "3.03d, 3.05d"
	strings:
		$1 = { 55 50 E8 ?? ?? ?? ?? 5D EB 01 E3 60 E8 03 ?? ?? ?? D2 EB 0B 58 EB 01 48 40 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule pc_guard_405d_410d_415d {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PC Guard"
		version = "4.05d, 4.10d, 4.15d"
	strings:
		$1 = { FC 55 50 E8 00 00 00 00 5D EB 01 }
	condition:
		$1 at pe.entry_point
}

rule pc_guard_500 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PC Guard"
		version = "5.00"
	strings:
		$1 = { FC 55 50 E8 00 00 00 00 5D 60 E8 03 00 00 00 83 EB 0E EB 01 0C 58 EB 01 35 40 EB 01 36 FF E0 0B 61 B8 ?? ?? ?? ?? EB 01 E3 60 E8 03 00 00 00 D2 EB 0B 58 EB 01 48 40 EB 01 35 FF E0 E7 61 2B E8 }
	condition:
		$1 at pe.entry_point
}

rule pcpec_alpha {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCPEC"
		version = "alpha"
	strings:
		$1 = { 53 51 52 56 57 55 E8 ?? ?? ?? ?? 5D 8B CD 81 ?? ?? ?? ?? ?? 2B ?? ?? ?? ?? ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule pcpec_alpha_preview {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCPEC"
		version = "alpha preview"
	strings:
		$1 = { 53 51 52 56 57 55 E8 00 00 00 00 5D 8B CD 81 ED 33 30 40 }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_020 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
		version = "0.20"
	strings:
		$1 = { E8 E8 01 ?? ?? 60 01 AD B3 27 40 ?? 68 }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_029 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
		version = "0.29"
	strings:
		$1 = { ?? BD ?? ?? ?? ?? 01 AD 55 39 40 ?? 8D B5 35 39 40 }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_040b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
		version = "0.40b"
	strings:
		$1 = { 9C 60 BD ?? ?? ?? ?? 01 ?? ?? ?? ?? ?? FF ?? ?? ?? ?? ?? 6A ?? FF ?? ?? ?? ?? ?? 50 50 2D }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_045 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
		version = "0.45"
	strings:
		$1 = { ?? BD ?? ?? ?? ?? 01 AD E3 38 40 ?? FF B5 DF 38 40 }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_071b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
		version = "0.71b"
	strings:
		$1 = { 01 AD 54 3A 40 00 FF B5 50 3A 40 00 6A 40 FF 95 88 3A 40 00 }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_071 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
		version = "0.71"
	strings:
		$1 = { 9C 60 BD ?? ?? ?? ?? 01 AD 54 3A 40 ?? FF B5 50 3A 40 ?? 6A 40 FF 95 88 3A 40 ?? 50 50 2D ?? ?? ?? ?? 89 85 }
	condition:
		$1 at pe.entry_point
}

rule pcshrinker_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PCShrinker"
	strings:
		$1 = { 9C 60 BD ?? ?? ?? ?? 01 ?? ?? ?? ?? ?? FF ?? ?? ?? ?? ?? 6A ?? FF ?? ?? ?? ?? ?? 50 50 2D }
	condition:
		$1 at pe.entry_point
}

rule pecrypt_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Crypt"
	strings:
		$1 = { 55 8B EC 83 C4 E0 53 56 33 C0 89 45 E4 89 45 E0 89 45 EC ?? ?? ?? ?? 64 82 40 00 E8 7C C7 FF FF 33 C0 55 68 BE 84 40 00 64 FF 30 64 89 20 68 CC 84 40 00 ?? ?? ?? ?? 00 A1 10 A7 40 00 50 E8 1D C8 FF FF 8B D8 85 DB 75 39 E8 3A C8 FF FF 6A 00 6A 00 68 A0 A9 40 00 68 00 04 00 00 50 6A 00 68 00 13 00 00 E8 FF C7 FF FF 6A 00 68 E0 84 40 00 A1 A0 A9 40 00 50 6A 00 E8 ?? ?? ?? ?? E9 7D 01 00 00 53 A1 10 A7 40 00 50 E8 42 C8 FF FF 8B F0 85 F6 75 18 6A 00 68 E0 84 40 00 68 E4 84 40 00 6A 00 E8 71 C8 FF FF E9 53 01 00 00 53 6A 00 E8 2C C8 FF FF A3 ?? ?? ?? ?? 83 3D 48 A8 40 00 00 75 18 6A 00 68 E0 84 40 00 68 F8 84 40 00 6A 00 E8 43 C8 FF FF E9 25 01 00 00 56 E8 F8 C7 FF FF A3 4C A8 40 00 A1 48 A8 40 00 E8 91 A1 FF FF 8B D8 8B 15 48 A8 40 00 85 D2 7C 16 42 33 C0 8B 0D 4C A8 40 00 03 C8 8A 09 8D 34 18 88 0E 40 4A 75 ED 8B 15 48 A8 40 00 85 D2 7C 32 42 33 C0 8D 34 18 8A 0E 80 F9 01 75 05 C6 06 FF EB 1C 8D 0C 18 8A 09 84 ?? ?? ?? ?? ?? 00 EB 0E 8B 0D 4C A8 40 00 03 C8 0F B6 09 49 88 0E 40 4A 75 D1 8D ?? ?? ?? ?? E8 A5 A3 FF FF 8B 45 E8 8D 55 EC E8 56 D5 FF FF 8D 45 EC BA 18 85 40 00 E8 79 BA FF FF 8B 45 EC E8 39 BB FF FF 8B D0 B8 54 A8 40 00 E8 31 A6 FF FF BA 01 00 00 00 B8 54 A8 40 00 E8 12 A9 FF FF E8 DD A1 FF FF 68 50 A8 40 00 8B D3 8B 0D 48 A8 40 00 B8 54 A8 40 00 E8 56 A7 FF FF E8 C1 A1 FF FF }
	condition:
		$1 at pe.entry_point
}

rule pecrypt_100_102_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Crypt"
		version = "1.00 - 1.02"
	strings:
		$1 = { E8 00 00 00 00 5B 83 ?? ?? EB ?? 52 4E 44 21 }
	condition:
		$1 at pe.entry_point
}

rule pecrypt_100_102_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Crypt"
		version = "1.00 - 1.02"
	strings:
		$1 = { E8 ?? ?? ?? ?? 5B 83 EB 05 EB 04 52 4E 44 }
	condition:
		$1 at pe.entry_point
}

rule pecrypt_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Crypt"
		version = "1.5"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 55 20 40 00 B9 7B 09 00 00 8D BD 9D 20 40 00 8B F7 AC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? AA E2 CC }
	condition:
		$1 at pe.entry_point
}

rule pe_diminisher_01_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Diminisher"
		version = "0.1"
	strings:
		$1 = { 53 51 52 56 57 55 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_diminisher_01_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Diminisher"
		version = "0.1"
	strings:
		$1 = { 5D 8B D5 81 ED A2 30 40 ?? 2B 95 91 33 40 ?? 81 EA 0B ?? ?? ?? 89 95 9A 33 40 ?? 80 BD 99 }
	condition:
		$1 at pe.entry_point
}

rule pe_intro_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Intro"
		version = "1.0"
	strings:
		$1 = { 8B 04 24 9C 60 E8 ?? ?? ?? ?? 5D 81 ED 0A 45 40 ?? 80 BD 67 44 40 ?? ?? 0F 85 48 }
	condition:
		$1 at pe.entry_point
}

rule pe_ninja_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Ninja"
	strings:
		$1 = { 5D 8B C5 81 ED B2 2C 40 00 2B 85 94 3E 40 00 2D 71 02 00 00 89 85 98 3E 40 00 0F B6 B5 9C 3E 40 00 8B FD }
	condition:
		$1 at pe.entry_point
}

rule pe_ninja_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Ninja"
	strings:
		$1 = { 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pe_ninja_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Ninja"
		version = "1.0"
	strings:
		$1 = { BE 5B 2A 40 00 BF 35 12 00 00 E8 40 12 00 00 3D 22 83 A3 C6 0F 85 67 0F 00 00 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pe_packer_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Packer"
	strings:
		$1 = { FC 8B 35 70 01 40 ?? 83 EE 40 6A 40 68 ?? 30 10 }
	condition:
		$1 at pe.entry_point
}

rule pe_password_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Password"
		version = "0.2 SMT/SMF"
	strings:
		$1 = { E8 04 ?? ?? ?? 8B EC 5D C3 33 C0 5D 8B FD 81 ED 33 26 40 ?? 81 EF ?? ?? ?? ?? 83 EF 05 89 AD 88 27 40 ?? 8D 9D 07 29 40 ?? 8D B5 62 28 40 ?? 46 80 }
	condition:
		$1 at pe.entry_point
}

rule pe_protector_260 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE Protector"
		version = "2.60 hying's PE-Armor V0.460"
	strings:
		$1 = { 55 53 51 52 56 57 E8 E1 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 5D 81 ED 0B 00 00 00 8B 9D 9B 00 00 00 03 9D 9F 00 00 00 0B DB 74 14 8B 83 7F 46 00 00 03 83 87 46 00 00 5F 5E 5A 59 5B 5D FF E0 8D 75 43 56 FF 55 54 8D B5 A3 00 00 00 56 50 FF 55 50 89 85 B0 00 00 00 8D 75 43 56 FF 55 54 8D B5 B4 00 00 00 56 50 FF 55 50 89 85 C0 00 00 00 8D 75 43 56 FF 55 54 8D B5 C4 00 00 00 56 50 FF 55 50 89 85 D0 00 00 00 6A 40 68 00 10 00 00 FF B5 97 00 00 00 6A 00 FF 95 B0 00 00 00 89 85 9B 00 00 00 55 8D 9D F2 01 00 00 53 8D 9D CC 01 00 00 FF D3 8B 74 24 04 8B 7C 24 0C F7 46 04 07 00 00 00 75 08 81 3E 27 00 00 C0 75 06 B8 00 00 00 00 C3 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_0460 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.460"
	strings:
		$1 = { E8 AA 00 00 00 2D ?? ?? 00 00 00 00 00 00 00 00 00 3D ?? ?? 00 2D ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B ?? ?? 00 5C ?? ?? 00 6F ?? ?? 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 A2 01 00 00 ?? ?? 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 00 00 00 ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_0490 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.490"
	strings:
		$1 = { 56 52 51 53 55 E8 15 01 00 00 32 ?? ?? 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_0460_0759 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.460 - 0.759"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_0750 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.750"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? 00 00 00 00 00 00 ?? ?? 01 00 00 00 00 00 00 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 74 ?? ?? ?? 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_076 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.760"
	strings:
		$1 = { E9 00 00 00 00 60 E8 14 00 00 00 5D 81 ED 00 00 00 00 6A ?? E8 A3 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_0760_0765 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.760 - 0.765"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 00 00 08 00 00 00 00 00 00 00 60 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_armor_07xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Armor"
		version = "0.7xx"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED ?? ?? ?? ?? 8D B5 ?? ?? ?? ?? 55 56 81 C5 ?? ?? ?? ?? 55 C3 }
	condition:
		$1 at pe.entry_point
}

rule pe_crypter_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Crypter"
	strings:
		$1 = { 60 E8 00 00 00 00 5D EB 26 }
	condition:
		$1 at pe.entry_point
}

rule re_pack_099 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-PACK"
		version = "0.99"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 83 ED 06 80 BD E0 04 ?? ?? 01 0F 84 F2 }
	condition:
		$1 at pe.entry_point
}

rule re_pack_100_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-PACK"
		version = "1.00"
	strings:
		$1 = { C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 0D 0A 20 2D 3D FE 20 50 45 2D 50 41 43 4B 20 76 31 2E 30 20 2D FE 2D 20 28 43 29 20 43 6F 70 79 72 69 67 68 74 20 31 39 39 38 20 62 79 20 41 4E 41 4B 69 4E 20 FE 3D 2D 20 0D 0A C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 }
	condition:
		$1
}

rule pe_protect_09_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Protect"
		version = "0.9"
	strings:
		$1 = { 50 45 2D 50 52 4F 54 45 43 54 20 30 2E 39 }
	condition:
		$1 at pe.entry_point
}

rule pe_protect_09_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Protect"
		version = "0.9"
	strings:
		$1 = { 52 51 55 57 64 67 A1 30 00 85 C0 78 0D E8 ?? ?? ?? ?? 58 83 C0 07 C6 ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule pe_protect_09_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Protect"
		version = "0.9"
	strings:
		$1 = { E9 ?? 00 00 00 0D 0A 0D 0A C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 0D 0A 50 45 2D 50 52 4F 54 45 43 54 20 30 2E 39 20 28 43 29 6F }
	condition:
		$1 at pe.entry_point
}
rule pe_protect_09_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-Protect"
		version = "0.9"
	strings:
		$1 = { E9 CF 00 00 00 0D 0A 0D 0A C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 }
	condition:
		$1 at pe.entry_point
}

rule pe_shield_01b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-SHiELD"
		version = "0.1b MTE"
	strings:
		$1 = { E8 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? B9 1B 01 ?? ?? D1 }
	condition:
		$1 at pe.entry_point
}

rule pe_shield_02_02b_02b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-SHiELD"
		version = "0.2, 0.2b, 0.2b2"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 41 4E 41 4B 49 4E 5D 83 ED 06 EB 02 EA 04 }
	condition:
		$1 at pe.entry_point
}

rule pe_shield_025 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-SHiELD"
		version = "0.25"
	strings:
		$1 = { 60 E8 2B 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pe_shield_0251 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE-SHiELD"
		version = "0.251"
	strings:
		$1 = { 5D 83 ED 06 EB 02 EA 04 8D }
	condition:
		$1 at pe.entry_point
}

rule pe123_2006412 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pe123"
		version = "2006.4.12"
	strings:
		$1 = { 8B C0 60 9C E8 01 00 00 00 C3 53 E8 72 00 00 00 50 E8 1C 03 00 00 8B D8 FF D3 5B C3 8B C0 E8 00 00 00 00 58 83 C0 05 C3 8B C0 55 8B EC 60 8B 4D 10 8B 7D 0C 8B 75 08 F3 A4 61 5D C2 0C 00 E8 00 00 00 00 58 83 E8 05 C3 8B C0 E8 00 00 00 00 58 83 C0 05 C3 8B C0 E8 00 00 00 00 58 C1 E8 0C C1 E0 0C 66 81 38 4D 5A 74 0C 2D 00 10 00 00 66 81 38 4D 5A 75 F4 C3 E8 00 00 00 00 58 83 E8 05 C3 8B C0 55 8B EC 81 C4 4C FE FF FF 53 6A 40 8D 85 44 FF FF FF 50 E8 BC FF FF FF 50 E8 8A FF FF FF 68 F8 00 00 00 8D 85 }
	condition:
		$1 at pe.entry_point
}

rule pe123_200644 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pe123"
		version = "2006.4.4"
	strings:
		$1 = { 8B C0 EB 01 34 60 EB 01 2A 9C EB 02 EA C8 E8 0F 00 00 00 EB 03 3D 23 23 EB 01 4A EB 01 5B C3 8D 40 00 53 EB 01 6C EB 01 7E EB 01 8F E8 15 01 00 00 50 E8 67 04 00 00 EB 01 9A 8B D8 FF D3 5B C3 8B C0 E8 00 00 00 00 58 83 C0 05 C3 8B C0 55 8B EC 60 8B 4D 10 8B 7D 0C 8B 75 08 F3 A4 61 5D C2 0C 00 E8 00 00 00 00 58 83 E8 05 C3 8B C0 E8 00 00 00 00 58 83 C0 05 C3 8B C0 E8 00 00 00 00 58 C1 E8 0C C1 E0 0C 66 81 38 4D 5A 74 0C 2D 00 10 00 00 66 81 38 4D 5A 75 F4 C3 E8 00 00 00 00 58 83 E8 05 C3 8B C0 55 }
	condition:
		$1 at pe.entry_point
}

rule pe123_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pe123"
	strings:
		$1 = { 8B C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? C0 }
	condition:
		$1 at pe.entry_point
}

rule pezip_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEZip"
		version = "1.0"
	strings:
		$1 = { D9 D0 F8 74 02 23 DB F5 F5 50 51 52 53 8D 44 24 10 50 55 56 57 D9 D0 22 C9 C1 F7 A0 55 66 C1 C8 B0 5D 81 E6 FF FF FF FF F8 77 07 52 76 03 72 01 90 5A C1 E0 60 90 BD 1F 01 00 00 87 E8 E2 07 E3 05 17 5D 47 E4 42 41 7F 06 50 66 83 EE 00 58 25 FF FF FF FF 51 0F B6 C9 66 83 F6 00 3D CB 60 47 92 50 40 58 FC E2 EE 59 F8 7C 08 53 74 04 78 02 84 C9 5B 66 0B ED F8 F5 BA 9F FA FF FF 52 57 77 04 78 02 84 E4 5F 5A 50 80 EF 00 58 50 81 E0 FF FF FF FF 58 3C EF FC 7A 05 3D DF DA AC D1 05 00 00 00 00 73 05 71 03 }
	condition:
		$1 at pe.entry_point
}

rule pe_admin_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PE_Admin"
		version = "1.0 EncryptPE 1.2003.5.18"
	strings:
		$1 = { 60 9C 64 FF 35 00 00 00 00 E8 79 01 00 00 90 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 47 65 74 53 79 73 74 65 6D 44 69 72 65 63 74 6F 72 79 41 00 00 00 43 72 65 61 74 65 46 69 6C 65 41 00 00 00 43 72 65 61 74 65 46 69 6C 65 4D 61 70 70 69 6E 67 41 00 00 00 4D 61 70 56 69 65 77 4F 66 46 69 6C 65 00 00 00 55 6E 6D 61 70 56 69 65 77 4F 66 46 69 6C 65 00 00 00 43 6C 6F 73 65 48 61 6E 64 6C 65 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 }
	condition:
		$1 at pe.entry_point
}

rule pebundle_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEBundle"
	strings:
		$1 = { 9C 60 E8 ?? ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 07 ?? 40 00 87 DD ?? ?? ?? ?? 40 00 01 }
	condition:
		$1 at pe.entry_point
}

rule pebundle_020_20x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEBundle"
		version = "0.20 - 2.0x"
	strings:
		$1 = { 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB ?? ?? 40 ?? 87 DD 6A 04 68 ?? 10 ?? ?? 68 ?? 02 ?? ?? 6A ?? FF 95 }
	condition:
		$1 at pe.entry_point
}

rule pebundle_200b5_230 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEBundle"
		version = "2.00b5 - 2.30"
	strings:
		$1 = { 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB ?? ?? 40 ?? 87 DD 01 AD ?? ?? ?? ?? 01 AD }
	condition:
		$1 at pe.entry_point
}

rule pebundle_244 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEBundle"
		version = "2.44"
	strings:
		$1 = { 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB ?? ?? 40 ?? 87 DD 83 BD }
	condition:
		$1 at pe.entry_point
}

rule pebundle_310 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEBundle"
		version = "3.10"
	strings:
		$1 = { 9C 60 E8 02 00 00 00 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 07 20 40 00 87 DD ?? ?? ?? ?? 40 00 01 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_090 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.90"
	strings:
		$1 = { EB 06 68 ?? ?? 40 00 C3 9C 60 BD ?? ?? 00 00 B9 02 00 00 00 B0 90 8D BD 7A 42 40 00 F3 AA 01 AD D9 43 40 00 FF B5 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_092 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.92"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 BD ?? ?? ?? ?? B9 02 ?? ?? ?? B0 90 8D BD A5 4F 40 ?? F3 AA 01 AD 04 51 40 ?? FF B5 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_094 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.94"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 ?? ?? ?? ?? 5D 55 58 81 ED ?? ?? ?? ?? 2B 85 ?? ?? ?? ?? 01 85 ?? ?? ?? ?? 50 B9 02 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_0971_0976 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.971 - 0.976"
	strings:
		$1 = { EB 06 68 C3 9C 60 E8 5D 55 5B 81 ED 8B 85 01 85 66 C7 85 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_0977 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.977"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB A0 86 40 ?? 87 DD 8B 85 2A 87 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_0978 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.978"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 24 88 40 ?? 87 DD 8B 85 A9 88 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_09781 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.978.1"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 49 87 40 ?? 87 DD 8B 85 CE 87 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_09782 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.978.2"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB D1 84 40 ?? 87 DD 8B 85 56 85 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_098 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.98"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB D7 84 40 ?? 87 DD 8B 85 5C 85 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_099 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "0.99"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 2F 85 40 ?? 87 DD 8B 85 B4 85 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.00"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB C4 84 40 ?? 87 DD 8B 85 49 85 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b1"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 28 63 40 ?? 87 DD 8B 85 AD 63 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b2"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 60 40 ?? 87 DD 8B 85 94 60 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b3 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b3"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 60 40 ?? 87 DD 8B 85 95 60 40 ?? 01 85 03 60 40 ?? 66 C7 85 ?? 60 40 ?? 90 90 BB 95 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b4 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b4"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 60 40 ?? 87 DD 8B 85 95 60 40 ?? 01 85 03 60 40 ?? 66 C7 85 ?? 60 40 ?? 90 90 BB 44 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b5 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b5"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 60 40 ?? 87 DD 8B 85 95 60 40 ?? 01 85 03 60 40 ?? 66 C7 85 ?? 60 40 ?? 90 90 BB 49 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b6 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b6"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 60 ?? 00 87 DD 8B 85 9A 60 40 ?? 01 85 03 60 40 ?? 66 C7 85 ?? 60 40 ?? 90 90 01 85 92 60 40 ?? BB B7 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_110b7 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.10b7"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 60 40 ?? 87 DD 8B 85 9A 60 40 ?? 01 85 03 60 40 ?? 66 C7 85 ?? 60 40 ?? 90 90 01 85 92 60 40 ?? BB 14 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_120_1201 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.20 - 1.20.1"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 70 40 ?? 87 DD 8B 85 9A 70 40 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_122 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.22"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 70 40 ?? 87 DD 8B 85 A6 70 40 ?? 01 85 03 70 40 ?? 66 C7 85 ?? 70 40 ?? 90 90 01 85 9E 70 40 ?? BB F3 08 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_123b3_1241 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.23b3 - 1.24.1"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 70 40 ?? 87 DD 8B 85 A6 70 40 ?? 01 85 03 70 40 ?? 66 C7 85 70 40 90 ?? 90 01 85 9E 70 40 BB ?? D2 08 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_1242_1243 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.24.2 - 1.24.3"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 70 40 ?? 87 DD 8B 85 A6 70 40 ?? 01 85 03 70 40 ?? 66 C7 85 70 40 90 ?? 90 01 85 9E 70 40 BB ?? D2 09 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_125 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.25"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 70 40 ?? 87 DD 8B 85 A6 70 40 ?? 01 85 03 70 40 ?? 66 C7 85 70 40 90 ?? 90 01 85 9E 70 40 BB ?? F3 0D }
	condition:
		$1 at pe.entry_point
}

rule pecompact_126b1_126b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.26b1 - 1.26b2"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 70 40 ?? 87 DD 8B 85 A6 70 40 ?? 01 85 03 70 40 ?? 66 C7 85 70 40 90 ?? 90 01 85 9E 70 40 BB ?? 05 0E }
	condition:
		$1 at pe.entry_point
}

rule pecompact_133 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.33"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 80 40 ?? 87 DD 8B 85 A6 80 40 ?? 01 85 03 80 40 ?? 66 C7 85 00 80 40 ?? 90 90 01 85 9E 80 40 ?? BB E8 0E }
	condition:
		$1 at pe.entry_point
}

rule pecompact_134_140b1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.34 - 1.40b1"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 80 40 ?? 87 DD 8B 85 A6 80 40 ?? 01 85 03 80 40 ?? 66 C7 85 ?? 00 80 ?? 40 90 90 01 85 9E 80 ?? 40 BB F8 10 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_140_145 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.40 - 1.45"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F A0 40 ?? 87 DD 8B 85 A6 A0 40 ?? 01 85 03 A0 40 ?? 66 C7 85 ?? A0 40 ?? 90 90 01 85 9E A0 40 ?? BB C3 11 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_140b2_140b4 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.40b2 - 1.40b4"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F A0 40 ?? 87 DD 8B 85 A6 A0 40 ?? 01 85 03 A0 40 ?? 66 C7 85 ?? A0 40 ?? 90 90 01 85 9E A0 40 ?? BB 86 11 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_140b5_140b6 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.40b5 - 1.40b6"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F A0 40 ?? 87 DD 8B 85 A6 A0 40 ?? 01 85 03 A0 40 ?? 66 C7 85 ?? A0 40 ?? 90 90 01 85 9E A0 40 ?? BB 8A 11 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_146 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.46"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F A0 40 ?? 87 DD 8B 85 A6 A0 40 ?? 01 85 03 A0 40 ?? 66 C7 85 ?? A0 40 ?? 90 90 01 85 9E A0 40 ?? BB 60 12 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_147_150 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.47 - 1.50"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F A0 40 ?? 87 DD 8B 85 A6 A0 40 ?? 01 85 03 A0 40 ?? 66 C7 85 ?? A0 40 ?? 90 90 01 85 9E A0 40 ?? BB 5B 12 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_155 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.55"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 80 40 ?? 87 DD 8B 85 A2 80 40 ?? 01 85 03 80 40 ?? 66 C7 85 ?? 80 40 ?? 90 90 01 85 9E 80 40 ?? BB 2D 12 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_156 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.56"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 0F 90 40 ?? 87 DD 8B 85 A2 90 40 ?? 01 85 03 90 40 ?? 66 C7 85 ?? 90 40 ?? 90 90 01 85 9E 90 40 ?? BB 2D 12 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_160_165 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.60 - 1.65"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 80 40 ?? 87 DD 8B 85 D2 80 40 ?? 01 85 33 80 40 ?? 66 C7 85 ?? 80 40 ?? 90 90 01 85 CE 80 40 ?? BB BB 12 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_166 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.66"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 90 40 ?? 87 DD 8B 85 E6 90 40 ?? 01 85 33 90 40 ?? 66 C7 85 ?? 90 40 ?? 90 90 01 85 DA 90 40 ?? 01 85 DE 90 40 ?? 01 85 E2 90 40 ?? BB 5B 11 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_167 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.67"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 90 40 87 DD 8B 85 E6 90 40 01 85 33 90 40 66 C7 85 90 40 90 90 01 85 DA 90 40 01 85 DE 90 40 01 85 E2 90 40 BB 8B 11 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_168_184 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.68 - 1.84"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 02 ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB 3F 90 40 87 DD 8B 85 E6 90 40 01 85 33 90 40 66 C7 85 90 40 90 90 01 85 DA 90 40 01 85 DE 90 40 01 85 E2 90 40 BB 7B 11 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_1xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "1.xx"
	strings:
		$1 = { EB 06 68 ?? ?? ?? ?? C3 9C 60 E8 ?? ?? ?? ?? 33 C0 8B C4 83 C0 04 93 8B E3 8B 5B FC 81 EB ?? ?? 40 00 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_200a38 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.00a38"
	strings:
		$1 = { B8 ?? ?? ?? ?? 80 B8 BF 10 00 10 01 74 7A C6 80 BF 10 00 10 01 9C 55 53 51 57 52 56 8D 98 0F 10 00 10 8B 53 14 8B E8 6A 40 68 00 10 00 00 FF 73 04 6A 00 8B 4B 10 03 CA 8B 01 FF D0 8B F8 50 8B 33 8B 53 14 03 F2 8B 4B 0C 03 CA 8D 85 B7 10 00 10 FF 73 04 8F 00 50 57 56 FF D1 58 03 43 08 8B F8 8B 53 14 8B F0 8B 46 FC 83 C0 04 2B F0 89 56 08 8B 4B 10 89 4E 18 FF D7 89 85 BB 10 00 10 5E 5A 5F 59 5B 5D 9D FF E0 8B 80 BB 10 00 10 FF E0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_200b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.00b"
	strings:
		$1 = { B8 ?? ?? ?? ?? 05 ?? ?? ?? ?? 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 CC 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_25_retail_slim {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.5 retail slim"
	strings:
		$1 = { B8 ?? ?? ?? 01 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 32 00 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_25_retail {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.5 retail"
	strings:
		$1 = { B8 ?? ?? ?? 01 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 6F 6D 70 61 63 74 32 00 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_253_slim {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.53 slim"
	strings:
		$1 = { B8 ?? ?? ?? ?? 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 32 00 00 08 0C 00 48 E1 01 56 57 53 55 8B 5C 24 1C 85 DB 0F 84 AB 21 E8 BD 0E E6 60 0D 0B 6B 65 72 6E 6C 33 32 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_253 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.53"
	strings:
		$1 = { B8 ?? ?? ?? ?? 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 6F 6D 70 61 63 74 32 00 00 00 00 08 0C 00 48 E1 01 56 57 53 55 8B 5C 24 1C 85 DB 0F 84 AB 21 E8 BD 0E E6 60 0D }
	condition:
		$1 at pe.entry_point
}

rule pecompact_253_276 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.53 - 2.76"
	strings:
		$1 = { B8 ?? ?? ?? ?? 55 53 51 57 56 52 8D 98 C9 11 00 10 8B 53 18 52 8B E8 6A 40 68 00 10 00 00 FF 73 04 6A 00 8B 4B 10 03 CA 8B 01 FF D0 5A 8B F8 50 52 8B 33 8B 43 20 03 C2 8B 08 89 4B 20 8B 43 1C 03 C2 8B 08 89 4B 1C 03 F2 8B 4B 0C 03 CA 8D 43 1C 50 57 56 FF }
	condition:
		$1 at pe.entry_point
}

rule pecompact_2xxb {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.xxb"
	strings:
		$1 = { B8 ?? ?? ?? 00 80 00 28 40 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_2xx_slim {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.xx slim"
	strings:
		$1 = { B8 ?? ?? ?? ?? 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 32 00 }
	condition:
		$1 at pe.entry_point
}

rule pecompact_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECompact"
		version = "2.xx"
	strings:
		$1 = { B8 ?? ?? ?? ?? 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 33 C0 89 08 50 45 43 6F 6D 70 61 63 74 32 00 }
	condition:
		$1 at pe.entry_point
}

rule pecrc32_088 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PECrc32"
		version = "0.88"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED B6 A4 45 00 8D BD B0 A4 45 00 81 EF 82 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule peid_bundle_100_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEiD-Bundle"
		version = "1.00 - 1.01"
	strings:
		$1 = { 60 E8 ?? 02 00 00 8B 44 24 04 52 48 66 31 C0 66 81 38 4D 5A 75 F5 8B 50 3C 81 3C 02 50 45 00 00 75 E9 5A C2 04 00 60 89 DD 89 C3 8B 45 3C 8B 54 28 78 01 EA 52 8B 52 20 01 EA 31 C9 41 8B 34 8A }
	condition:
		$1 at pe.entry_point
}

rule peid_bundle_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEiD-Bundle"
		version = "1.00"
	strings:
		$1 = { 60 E8 21 02 00 00 8B 44 24 04 52 48 66 31 C0 66 81 38 4D 5A 75 F5 8B 50 3C 81 3C 02 50 45 00 00 75 E9 5A C2 04 00 60 89 DD 89 C3 8B 45 3C 8B 54 28 78 01 EA 52 8B 52 20 01 EA 31 C9 41 8B 34 8A }
	condition:
		$1 at pe.entry_point
}

rule peid_bundle_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEiD-Bundle"
		version = "1.01"
	strings:
		$1 = { 60 E8 23 02 00 00 8B 44 24 04 52 48 66 31 C0 66 81 38 4D 5A 75 F5 8B 50 3C 81 3C 02 50 45 00 00 75 E9 5A C2 04 00 60 89 DD 89 C3 8B 45 3C 8B 54 28 78 01 EA 52 8B 52 20 01 EA 31 C9 41 8B 34 8A }
	condition:
		$1 at pe.entry_point
}

rule peid_bundle_102_103_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEiD-Bundle"
		version = "1.02 - 1.03"
	strings:
		$1 = { 83 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 E8 9C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 41 00 08 00 39 00 08 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 80 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule peid_bundle_102_103_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEiD-Bundle"
		version = "1.02 - 1.03"
	strings:
		$1 = { 60 E8 9C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 36 ?? ?? ?? 2E ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 80 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 44 }
	condition:
		$1 at pe.entry_point
}

rule peid_bundle_102_104 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEiD-Bundle"
		version = "1.02 - 1.04"
	strings:
		$1 = { 60 E8 ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 36 ?? ?? ?? 2E ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 01 00 00 80 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 44 }
	condition:
		$1 at pe.entry_point
}

rule pelock_106 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PELock"
		version = "1.06"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 4B 45 }
	condition:
		$1 at pe.entry_point
}

rule pelock_nt_201 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PELock"
		version = "NT 2.01"
	strings:
		$1 = { EB 03 CD 20 EB EB 01 EB 1E EB 01 EB EB 02 CD 20 9C EB 03 CD }
	condition:
		$1 at pe.entry_point
}

rule pelock_nt_202c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PELock"
		version = "NT 2.02c"
	strings:
		$1 = { EB 02 C7 85 1E EB 03 CD 20 EB EB 01 EB 9C EB 01 EB EB 02 CD }
	condition:
		$1 at pe.entry_point
}

rule pelock_nt_203 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PELock"
		version = "NT 2.03"
	strings:
		$1 = { EB 02 C7 85 1E EB 03 CD 20 C7 9C EB 02 69 B1 60 EB 02 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule pelock_nt_204 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PELock"
		version = "NT 2.04"
	strings:
		$1 = { EB ?? CD ?? ?? ?? ?? ?? CD ?? ?? ?? ?? ?? EB ?? EB ?? EB ?? EB ?? CD ?? ?? ?? ?? ?? E8 ?? ?? ?? ?? E9 ?? ?? ?? ?? 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule pemangle_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEMangle"
	strings:
		$1 = { 60 9C BE ?? ?? ?? ?? 8B FE B9 ?? ?? ?? ?? BB 44 52 4F 4C AD 33 C3 }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
	strings:
		$1 = { 55 8B EC 81 EC 78 05 00 00 53 56 BE 04 01 00 00 57 8D 85 94 FD FF FF 56 33 DB 50 53 FF 15 ?? ?? 40 00 8D 85 94 FD FF FF 56 50 8D 85 94 FD FF FF 50 FF 15 ?? ?? 40 00 8B 3D ?? 20 40 00 53 53 6A 03 53 6A 01 8D 85 94 FD FF FF 68 00 00 00 80 50 }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
	strings:
		$1 = { 55 8B EC 81 EC 7C 05 00 00 53 56 57 BE 04 01 00 00 56 8D 85 90 FD FF FF 33 DB 50 53 89 5D F4 FF 15 38 20 40 00 56 8D 85 90 FD FF FF 50 50 FF 15 34 20 40 00 8B 3D 30 20 40 00 53 53 6A 03 53 6A 01 68 00 00 00 80 8D 85 90 FD FF FF 50 FF D7 83 }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 D0 53 56 57 8D 75 FC 8B 44 24 30 25 00 00 FF FF 81 38 4D 5A 90 00 74 07 2D 00 10 00 00 EB F1 89 45 FC E8 C8 FF FF FF 2D 0F 05 00 00 89 45 F4 8B 06 8B 40 3C 03 06 8B 40 78 03 06 8B C8 8B 51 20 03 16 8B 59 24 03 1E 89 5D F0 8B 59 1C 03 1E 89 5D EC 8B 41 18 8B C8 49 85 C9 72 5A 41 33 C0 8B D8 C1 E3 02 03 DA 8B 3B 03 3E 81 3F 47 65 74 50 75 40 8B DF 83 C3 04 81 3B 72 6F 63 41 75 33 8B DF 83 C3 08 81 3B 64 64 72 65 75 26 83 C7 0C 66 81 3F 73 73 }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
		version = "1.0"
	strings:
		$1 = { 60 9C BE 00 10 40 00 8B FE B9 ?? ?? ?? ?? BB 78 56 34 12 AD 33 C3 AB E2 FA 9D 61 E9 ?? ?? ?? FF }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
		version = "2.0"
	strings:
		$1 = { EB 25 00 00 F7 BF 00 00 00 00 00 00 00 00 00 00 12 00 E8 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 00 00 E8 00 00 00 00 5D 81 ED 2C 10 40 00 8D B5 14 10 40 00 E8 33 00 00 00 89 85 10 10 40 00 BF 00 00 40 00 8B F7 03 7F 3C 8B 4F 54 51 56 8D 85 }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
		version = "3.0"
	strings:
		$1 = { E8 00 00 00 00 5D 81 ED 05 10 40 00 8D B5 24 10 40 00 8B FE B9 0F 00 00 00 BB ?? ?? ?? ?? AD 33 C3 E2 FA }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_31 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
		version = "3.1"
	strings:
		$1 = { E9 ?? ?? ?? 00 F0 0F C6 }
	condition:
		$1 at pe.entry_point
}

rule pencrypt_40b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEncrypt"
		version = "4.0b"
	strings:
		$1 = { 66 ?? ?? 00 66 83 ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule penguincrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEnguinCrypt"
		version = "1.0"
	strings:
		$1 = { B8 93 ?? ?? 00 55 50 67 64 FF 36 00 00 67 64 89 26 00 00 BD 4B 48 43 42 B8 04 00 00 00 CC 3C 04 75 04 90 90 C3 90 67 64 8F 06 00 00 58 5D BB 00 00 40 00 33 C9 33 C0 }
	condition:
		$1 at pe.entry_point
}

rule penightmare_13 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PENightMare"
		version = "1.3"
	strings:
		$1 = { 60 E8 00 00 00 00 5D B9 ?? ?? ?? ?? 80 31 15 41 81 F9 }
	condition:
		$1 at pe.entry_point
}

rule penightmare_2b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PENightMare"
		version = "2b"
	strings:
		$1 = { 60 E9 ?? ?? ?? ?? EF 40 03 A7 07 8F 07 1C 37 5D 43 A7 04 B9 2C 3A }
	condition:
		$1 at pe.entry_point
}

rule pequake_006 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEQuake"
		version = "0.06"
	strings:
		$1 = { E8 A5 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule peshit_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEShit"
	strings:
		$1 = { B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? 83 F9 00 7E 06 80 30 ?? 40 E2 F5 E9 ?? ?? ?? FF }
	condition:
		$1 at pe.entry_point
}

rule pespin_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "0.1"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 5C CB 46 00 0B E4 74 9E 75 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 ?? ?? ?? ?? ?? 5D 33 C9 41 E2 17 EB 07 ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 8B ?? ?? ?? ?? ?? 8B 42 3C 03 C2 89 ?? ?? ?? ?? ?? 41 C1 E1 07 8B 0C 01 03 CA 8B 59 10 03 DA 8B 1B 89 ?? ?? ?? ?? ?? 53 8F 85 ?? ?? ?? ?? BB ?? ?? ?? ?? B9 A5 08 00 00 8D ?? ?? ?? ?? ?? 4F 30 1C 39 FE CB E2 F9 68 2D 01 00 00 59 8D ?? ?? ?? ?? ?? C0 0C 39 02 E2 FA E8 02 00 00 00 FF 15 ?? ?? ?? ?? 4F 56 00 BB 54 13 0B 00 D1 E3 2B C3 FF E0 E8 01 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pespin_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "0.3"
	strings:
		$1 = { EB 01 68 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 B7 CD 46 00 0B E4 74 9E 75 01 C7 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 A3 68 72 01 FF 5D 33 C9 41 E2 17 EB 07 EA EB 01 EB EB 0D FF E8 01 00 00 00 EA 5A 83 EA 0B FF E2 8B 95 CB 2C 40 00 8B 42 3C 03 C2 89 85 D5 2C 40 00 41 C1 E1 07 8B 0C 01 03 CA 8B 59 10 03 DA 8B 1B 89 9D E9 2C 40 00 53 8F 85 B6 2B 40 00 BB ?? 00 00 00 B9 75 0A 00 00 8D BD 7E 2D 40 00 4F 30 1C 39 FE CB E2 F9 68 3C 01 00 00 59 8D BD B6 36 40 00 C0 0C 39 02 E2 FA E8 02 00 00 00 FF 15 5A 8D 85 1F 53 56 00 BB 54 13 0B 00 D1 E3 2B C3 FF E0 E8 01 00 00 00 68 E8 1A 00 00 00 8D 34 28 B9 08 00 00 00 B8 ?? ?? ?? ?? 2B C9 83 C9 15 0F A3 C8 0F 83 81 00 00 00 8D B4 0D DC 2C 40 00 }
	condition:
		$1 at pe.entry_point
}

rule pespin_041 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "0.41"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 02 D2 46 00 0B E4 74 9E 75 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 ?? ?? ?? ?? ?? 5D 33 C9 41 E2 17 EB 07 ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 8B ?? ?? ?? ?? ?? 8B 42 3C 03 C2 89 ?? ?? ?? ?? ?? 41 C1 E1 07 8B 0C 01 03 CA 8B 59 10 03 DA 8B 1B 89 ?? ?? ?? ?? ?? 53 8F ?? ?? ?? ?? ?? BB ?? ?? ?? ?? B9 ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 4F EB 01 AB 30 1C 39 FE CB E2 F9 EB 01 ?? 68 3C 01 00 00 59 8D ?? ?? ?? ?? ?? C0 0C 39 02 E2 FA E8 02 00 00 00 FF 15 ?? ?? ?? ?? 59 56 00 BB 54 13 0B 00 D1 E3 2B C3 FF E0 E8 01 00 00 00 ?? E8 1A 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule pespin_07 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "0.7"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 83 D5 46 00 0B E4 74 9E 75 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 ?? ?? ?? ?? ?? 5D 33 C9 41 E2 17 EB 07 ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 EB 04 ?? EB 04 00 EB FB FF 8B ?? ?? ?? ?? ?? 8B 42 3C 03 C2 89 ?? ?? ?? ?? ?? EB 01 ?? 41 C1 E1 07 8B 0C 01 03 CA E8 03 00 00 00 EB 04 ?? EB FB ?? 83 04 24 0C C3 }
	condition:
		$1 at pe.entry_point
}

rule pespin_09b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "0.9b"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 72 C8 46 00 0B E4 74 9E 75 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 ?? ?? ?? ?? ?? 5D 33 C9 41 E2 26 E8 01 00 00 00 ?? 5A 33 C9 ?? ?? ?? ?? ?? ?? 8B 42 3C 03 C2 89 ?? ?? ?? ?? ?? 41 C1 E1 07 8B 0C 01 03 CA 8B 59 10 03 DA 8B 1B ?? ?? ?? ?? ?? ?? 8B 59 24 03 DA 8B 1B ?? ?? ?? ?? ?? ?? 53 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 6A 0C 5B 6A 17 59 30 0C 03 02 CB 4B 75 F8 40 8D 9D 41 8F 4E 00 50 53 81 2C 24 01 78 0E 00 ?? ?? ?? ?? ?? ?? C3 92 EB 15 68 ?? ?? ?? ?? ?? B9 ?? 08 00 00 ?? ?? ?? ?? ?? ?? 4F 30 1C 39 FE CB E2 F9 68 1D 01 00 00 59 ?? ?? ?? ?? ?? ?? C0 0C 39 02 E2 FA 68 ?? ?? ?? ?? 50 01 6C 24 04 E8 BD 09 00 00 33 C0 0F 84 C0 08 00 00 ?? ?? ?? ?? ?? ?? 50 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? FF E0 C3 8D 64 24 04 E8 53 0A 00 00 D7 58 5B 51 C3 F7 F3 32 DA ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 81 2C 24 A3 00 00 00 58 ?? ?? ?? ?? ?? ?? 53 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule pespin_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "1.0"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 C8 DC 46 00 0B E4 74 9E 75 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 19 77 00 43 B7 F6 C3 ?? ?? ?? ?? ?? ?? ?? C9 C2 08 00 ?? ?? ?? ?? ?? 5D 33 C9 41 E2 17 EB 07 ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 EB 04 ?? EB 04 ?? EB FB FF 8B ?? ?? ?? ?? ?? 8B 42 3C 03 C2 89 ?? ?? ?? ?? ?? EB 02 ?? ?? F9 72 08 73 0E F9 83 04 24 17 C3 E8 04 00 00 00 0F F5 73 11 EB 06 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? FF 34 24 C3 41 C1 E1 07 8B 0C 01 03 CA E8 03 00 00 00 EB 04 ?? ?? ?? ?? 83 04 24 0C C3 }
	condition:
		$1 at pe.entry_point
}

rule pespin_1100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "1.100"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 7D DE 46 00 0B E4 74 9E 75 01 ?? 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 ?? ?? ?? ?? ?? 5D 33 C9 41 E2 17 EB 07 ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 EB 04 ?? EB 04 00 EB FB ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? EB 02 ?? ?? F9 72 08 73 0E F9 83 04 24 17 C3 E8 04 00 00 00 0F F5 73 11 EB 06 ?? ?? ?? ?? ?? ?? F5 72 0E F5 72 F8 68 EB EC 83 04 24 07 F5 FF 34 24 C3 41 C1 E1 07 8B 0C 01 03 CA E8 03 00 00 00 EB 04 ?? EB FB }
	condition:
		$1 at pe.entry_point
}

rule pespin_1300 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESPin"
		version = "1.300"
	strings:
		$1 = { EB 01 68 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 AC DF 46 00 0B E4 74 9E 75 01 C7 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 A3 68 72 01 FF 5D 33 C9 41 E2 17 EB 07 EA EB 01 EB EB 0D FF E8 01 00 00 00 EA 5A 83 EA 0B FF E2 EB 04 9A EB 04 00 EB FB FF 8B 95 0D 4F 40 00 8B 42 3C 03 C2 89 85 17 4F 40 00 EB 02 12 77 F9 72 08 73 0E F9 83 04 24 17 C3 E8 04 00 00 00 0F F5 73 11 EB 06 9A 72 ED 1F EB 07 F5 72 0E F5 72 F8 68 EB EC 83 04 24 }
	condition:
		$1 at pe.entry_point
}

rule pespin_1300b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "1.300b"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 71 DF 46 00 0B E4 74 9E 75 01 C7 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 A3 68 72 01 FF 5D 33 C9 41 E2 17 EB 07 ?? ?? ?? ?? ?? ?? ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 EB 04 ?? EB 04 ?? EB FB ?? ?? ?? ?? ?? ?? ?? 8B 42 3C 03 C2 ?? ?? ?? ?? ?? ?? EB 02 ?? ?? F9 72 08 73 0E F9 83 04 24 17 C3 E8 04 00 00 00 0F F5 73 11 EB 06 9A 72 ED 1F EB 07 F5 72 0E F5 72 F8 68 EB EC 83 04 24 07 F5 FF 34 24 C3 }
	condition:
		$1 at pe.entry_point
}

rule pespin_1304 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "1.304"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 88 DF 46 00 0B E4 74 9E 75 01 C7 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 A3 68 72 01 FF 5D 33 C9 41 E2 17 EB 07 ?? EB 01 ?? EB 0D ?? E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 EB 04 ?? EB 04 ?? EB FB ?? ?? ?? ?? ?? ?? ?? 8B 42 3C 03 C2 ?? ?? ?? ?? ?? ?? EB 02 ?? ?? F9 72 08 73 0E F9 83 04 24 17 C3 E8 04 00 00 00 ?? ?? ?? ?? EB 06 ?? ?? ?? ?? ?? ?? F5 72 0E F5 72 F8 68 EB EC 83 04 24 07 F5 FF 34 24 C3 }
	condition:
		$1 at pe.entry_point
}

rule pespin_1320 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "1.320"
	strings:
		$1 = { EB 01 ?? 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 17 E6 46 00 0B E4 74 9E 75 01 C7 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 A3 68 72 01 FF 5D 33 C9 41 E2 17 EB 07 ?? EB 01 ?? EB 0D FF E8 01 00 00 00 ?? 5A 83 EA 0B FF E2 EB 04 ?? EB 04 00 EB FB FF E8 02 00 00 00 ?? ?? 5A 81 ?? ?? ?? ?? ?? 83 EA FE 89 95 A9 57 40 00 2B C0 2B C9 83 F1 06 09 85 CB 57 40 00 9C D3 2C 24 80 C1 FB 21 0C 24 50 52 B8 36 C7 09 FF 05 FE 37 F6 00 F7 64 24 08 8D 84 28 B1 35 40 00 89 44 24 08 5A 58 8D 64 24 04 FF 64 24 FC CD 20 BB 69 74 58 0B C1 C3 }
	condition:
		$1 at pe.entry_point
}

rule pespin_1330 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PESpin"
		version = "1.330"
	strings:
		$1 = { EB 01 68 60 E8 00 00 00 00 8B 1C 24 83 C3 12 81 2B E8 B1 06 00 FE 4B FD 82 2C 24 77 E7 46 00 0B E4 74 9E 75 01 C7 81 73 04 D7 7A F7 2F 81 73 19 77 00 43 B7 F6 C3 6B B7 00 00 F9 FF E3 C9 C2 08 00 A3 68 72 01 FF 5D 33 C9 41 E2 17 EB 07 EA EB 01 EB EB 0D FF E8 01 00 00 00 EA 5A 83 EA 0B FF E2 EB 04 9A }
	condition:
		$1 at pe.entry_point
}

rule pestuboep_1x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PEStubOEP"
		version = "1.x"
	strings:
		$1 = { 40 48 BE 00 ?? ?? 00 40 48 60 33 C0 B8 ?? ?? ?? 00 FF E0 C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule pestuboep_1x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PeStubOEP"
		version = "1.x"
	strings:
		$1 = { 90 33 C9 33 D2 B8 ?? ?? ?? 00 B9 FF }
	condition:
		$1 at pe.entry_point
}

rule pestuboep_1x_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PeStubOEP"
		version = "1.x"
	strings:
		$1 = { E8 05 00 00 00 33 C0 40 48 C3 E8 05 }
	condition:
		$1 at pe.entry_point
}

rule petite__uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
	strings:
		$1 = { B8 ?? ?? ?? ?? 66 9C 60 50 }
	condition:
		$1 at pe.entry_point
}

rule petite_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.2"
	strings:
		$1 = { (66) 9C 60 E8 CA ?? ?? ?? 03 ?? 04 ?? 05 ?? 06 ?? 07 ?? 08 }
	condition:
		$1 at pe.entry_point
}

rule petite_13_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.3"
	strings:
		$1 = { ?? ?? ?? ?? ?? 66 9C 60 50 8D 88 ?? F? ?? ?? 8D 90 04 16 ?? ?? 8B DC 8B E1 68 ?? ?? ?? ?? 53 50 80 04 24 08 50 80 04 24 42 }
	condition:
		$1 at pe.entry_point
}

rule petite_13_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.3"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? 9C 60 50 8D 88 00 ?? ?? ?? 8D 90 ?? ?? 00 00 8B DC 8B E1 68 00 00 ?? ?? 53 50 80 04 24 08 50 80 04 24 42 50 80 04 24 61 50 80 04 24 9D 50 80 04 24 BB 83 3A 00 0F 84 DA 14 00 00 8B 44 24 18 F6 42 03 80 74 19 FD 80 72 03 80 8B F0 8B F8 03 72 04 03 7A 08 8B 0A F3 A5 83 C2 0C FC EB D4 8B 7A 08 03 F8 8B 5A 04 85 DB 74 13 52 53 57 03 02 50 E8 7B 00 00 00 85 C0 74 2E 5F 5F 58 5A 8B 4A 0C C1 F9 02 F3 AB 8B 4A 0C 83 E1 03 F3 AA 83 C2 10 EB A0 45 52 52 4F 52 21 00 43 6F 72 72 75 70 74 20 }
	condition:
		$1 at pe.entry_point
}

rule petite_13a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.3a"
	strings:
		$1 = { ?? ?? ?? ?? ?? 66 9C 60 50 8D 88 ?? ?? ?? ?? 8D 90 F8 15 ?? ?? 8B DC 8B E1 68 ?? ?? ?? ?? 53 50 80 04 24 08 50 80 04 24 42 }
	condition:
		$1 at pe.entry_point
}

rule petite_14_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.4"
	strings:
		$1 = { 66 9C 60 50 8B D8 03 ?? 68 54 BC ?? ?? 6A ?? FF 50 14 8B CC }
	condition:
		$1 at pe.entry_point
}

rule petite_14_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.4"
	strings:
		$1 = { B8 ?? ?? ?? ?? 66 9C 60 50 8B D8 03 00 68 ?? ?? ?? ?? 6A 00 }
	condition:
		$1 at pe.entry_point
}

rule petite_14_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.4"
	strings:
		$1 = { ?? ?? ?? ?? ?? 66 9C 60 50 8B D8 03 00 68 54 BC 00 00 6A 00 FF 50 14 8B CC 8D A0 54 BC 00 00 50 8B C3 8D 90 ?? 16 00 00 68 00 00 ?? ?? 51 50 80 04 24 08 50 80 04 24 42 50 80 04 24 61 50 80 04 24 9D 50 80 04 24 BB 83 3A 00 0F 84 D8 14 00 00 8B 44 24 18 F6 42 03 80 74 19 FD 80 72 03 80 8B F0 8B F8 03 72 04 03 7A 08 8B 0A F3 A5 83 C2 0C FC EB D4 8B 7A 08 03 F8 8B 5A 04 85 DB 74 13 52 53 57 03 02 50 E8 79 00 00 00 85 C0 74 30 5F 5F 58 5A 8B 4A 0C C1 F9 02 33 C0 F3 AB 8B 4A 0C 83 E1 03 F3 AA 83 C2 10 }
	condition:
		$1 at pe.entry_point
}

rule petite_14_04_or_higher {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "1.4 or higher"
	strings:
		$1 = { B8 ?? ?? ?? ?? 66 9C 60 50 8D ?? ?? ?? ?? ?? 68 ?? ?? ?? ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule petite_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.0"
	strings:
		$1 = { B8 ?? ?? ?? ?? 66 9C 60 50 8B D8 03 ?? 68 54 BC ?? ?? 6A ?? FF 50 18 8B CC 8D A0 54 BC ?? ?? 8B C3 8D 90 E0 15 ?? ?? 68 }
	condition:
		$1 at pe.entry_point
}

rule petite_21_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.1"
	strings:
		$1 = { B8 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 ?? ?? ?? ?? ?? ?? 64 ?? ?? ?? ?? ?? ?? 66 9C 60 50 }
	condition:
		$1 at pe.entry_point
}

rule petite_21_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.1"
	strings:
		$1 = { B8 ?? ?? ?? ?? 6A ?? 68 ?? ?? ?? ?? 64 FF 35 ?? ?? ?? ?? 64 89 25 ?? ?? ?? ?? 66 9C 60 50 }
	condition:
		$1 at pe.entry_point
}

rule petite_22_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.2"
	strings:
		$1 = { B8 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 FF 35 ?? ?? ?? ?? 64 89 25 ?? ?? ?? ?? 66 9C 60 50 }
	condition:
		$1 at pe.entry_point
}

rule petite_22_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.2"
	strings:
		$1 = { B8 00 ?? ?? ?? 6A 00 68 ?? ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 66 9C 60 50 8B D8 03 00 68 ?? ?? ?? 00 6A 00 FF 50 1C 8b cc 8d }
		$2 = { B8 00 ?? ?? ??       68 ?? ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 66 9C 60 50 68 00 00 ?? ?? 8b 3c 24 8b 30 66 81 c7 80 07 8d 74 }
	condition:
		$1 at pe.entry_point or
		$2 at pe.entry_point
}

rule petite_23 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.3"
	strings:
		$1 = { B8 00 ?? ?? ?? 6A 00 68 ?? ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 66 9C 60 50 8B D8 03 00 68 ?? ?? ?? 00 6A 00 FF 50 1C 89 43 08 }
		$2 = { B8 00 ?? ?? ??       68 ?? ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 66 9C 60 50 8B D8 03 00 68 ?? ?? ?? 00 6A 00 FF 50 1C 89 43 08 }
	condition:
		$1 at pe.entry_point or
		$2 at pe.entry_point
}

rule petite_24 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Petite"
		version = "2.4"
	strings:
		$1 = { B8 ?? ?? ?? ?? 60 8D A8 ?? ?? ?? F? 68 ?? ?? ?? ?? 6A 40 68 00 30 00 00 68 ?? ?? ?? 00 6A 00 ff 90 ?? ?? 00 00 89 44 24 1C bb ?? ?? 00 00 8d b5 ?? ?? ?? 00 8b f8 50 E8 0? 00 00 00 74 07 }
	condition:
		$1 at pe.entry_point
}

rule pex_099_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PeX"
		version = "0.99"
	strings:
		$1 = { 60 E8 01 ?? ?? ?? ?? 83 C4 04 E8 01 ?? ?? ?? ?? 5D 81 }
	condition:
		$1 at pe.entry_point
}

rule pex_099_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PeX"
		version = "0.99"
	strings:
		$1 = { E9 F5 00 00 00 0D 0A C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 C4 0D 0A 20 50 65 58 20 28 63 29 20 62 79 20 62 61 72 74 5E 43 72 61 63 6B 50 6C 20 62 65 74 61 20 72 65 6C 65 61 73 65 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 20 }
	condition:
		$1 at pe.entry_point
}

rule pi_cryptor_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pi Cryptor"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 56 57 31 C0 89 45 EC B8 40 1E 06 00 E8 48 FA FF FF 33 C0 55 68 36 1F 06 00 64 FF 30 64 89 20 6A 00 68 80 00 00 00 6A 03 6A 00 6A 01 68 00 00 00 80 8D 55 EC 31 C0 E8 4E F4 FF FF 8B 45 EC E8 F6 F7 FF FF 50 E8 CC FA FF FF 8B D8 83 FB FF 74 4E 6A 00 53 E8 CD FA FF FF 8B F8 81 EF AC 26 00 00 6A 00 6A 00 68 AC 26 00 00 53 E8 DE FA FF FF 89 F8 E8 E3 F1 FF FF 89 C6 6A 00 68 28 31 06 00 57 56 53 E8 AE FA FF FF 53 E8 80 FA FF FF 89 FA 81 EA 72 01 00 00 8B C6 E8 55 FE FF FF 89 C6 89 F0 09 C0 74 05 E8 A8 FB FF FF 31 C0 }
	condition:
		$1 at pe.entry_point
}

rule pi_cryptor_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pi Cryptor"
		version = "1.0"
	strings:
		$1 = { 89 55 F8 BB 01 00 00 00 8A 04 1F 24 0F 8B 55 FC 8A 14 32 80 E2 0F 32 C2 8A 14 1F 80 E2 F0 02 D0 88 14 1F 46 8D 45 F4 8B 55 FC E8 ?? ?? ?? ?? 8B 45 F4 E8 ?? ?? ?? ?? 3B F0 7E 05 BE 01 00 00 00 43 FF 4D F8 75 C2 ?? ?? ?? ?? 5A 59 59 64 89 10 68 ?? ?? ?? ?? 8D 45 F4 E8 ?? ?? ?? ?? C3 E9 }
	condition:
		$1 at pe.entry_point
}

rule pohernah_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pohernah"
		version = "1.0.0"
	strings:
		$1 = { 58 60 E8 00 00 00 00 5D 81 ED 20 25 40 00 8B BD 86 25 40 00 8B 8D 8E 25 40 00 6B C0 05 83 F0 04 89 85 92 25 40 00 83 F9 00 74 2D 81 7F 1C AB 00 00 00 75 1E 8B 77 0C 03 B5 8A 25 40 00 31 C0 3B 47 10 74 0E 50 8B 85 92 25 40 00 30 06 58 40 46 EB ED 83 C7 28 49 EB CE 8B 85 82 25 40 00 89 44 24 1C 61 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule pohernah_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pohernah"
		version = "1.0.1 Crypter"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED F1 26 40 00 8B BD 18 28 40 00 8B 8D 20 28 40 00 B8 38 28 40 00 01 E8 80 30 05 83 F9 00 74 71 81 7F 1C AB 00 00 00 75 62 8B 57 0C 03 95 1C 28 40 00 31 C0 51 31 C9 66 B9 FA 00 66 83 F9 00 74 49 8B 57 0C 03 95 1C 28 40 00 8B 85 24 28 40 00 83 F8 02 75 06 81 C2 00 02 00 00 51 8B 4F 10 83 F8 02 75 06 81 E9 00 02 00 00 57 BF C8 00 00 00 89 CE E8 27 00 00 00 89 C1 5F B8 38 28 40 00 01 E8 E8 24 00 00 00 59 49 EB B1 59 83 C7 28 49 EB 8A 8B 85 14 28 40 00 89 44 24 1C 61 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule pohernah_102 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pohernah"
		version = "1.0.2 Crypter"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED DE 26 40 00 8B BD 05 28 40 00 8B 8D 0D 28 40 00 B8 25 28 40 00 01 E8 80 30 05 83 F9 00 74 71 81 7F 1C AB 00 00 00 75 62 8B 57 0C 03 95 09 28 40 00 31 C0 51 31 C9 66 B9 F7 00 66 83 F9 00 74 49 8B 57 0C 03 95 09 28 40 00 8B 85 11 28 40 00 83 F8 02 75 06 81 C2 00 02 00 00 51 8B 4F 10 83 F8 02 75 06 81 E9 00 02 00 00 57 BF C8 00 00 00 89 CE E8 27 00 00 00 89 C1 5F B8 25 28 40 00 01 E8 E8 24 00 00 00 59 49 EB B1 59 83 C7 28 49 EB 8A 8B 85 01 28 40 00 89 44 24 1C 61 FF E0 }
	condition:
		$1 at pe.entry_point
}

rule pohernah_103 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Pohernah"
		version = "1.0.3 Crypter"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 2A 27 40 00 31 C0 40 83 F0 06 40 3D 40 1F 00 00 75 07 BE 6A 27 40 00 EB 02 EB EB 8B 85 9E 28 40 00 83 F8 01 75 17 31 C0 01 EE 3D 99 00 00 00 74 0C 8B 8D 86 28 40 00 30 0E 40 46 EB ED }
	condition:
		$1 at pe.entry_point
}

rule polycrypt_214b_215_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PolyCrypt PE"
		version = "2.1.4b, 2.1.5"
	strings:
		$1 = { 50 6F 6C 79 43 72 79 70 74 20 50 45 20 28 63 29 20 32 30 30 34 2D 32 30 30 35 2C 20 4A 4C 61 62 53 6F 66 74 77 61 72 65 2E 00 50 00 43 00 50 00 45 }
	condition:
		$1 at pe.entry_point
}

rule polycrypt_214b_215_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PolyCrypt PE"
		version = "2.1.4b, 2.1.5"
	strings:
		$1 = { 91 8B F4 AD FE C9 80 34 08 ?? E2 FA C3 60 E8 ED FF FF FF EB }
	condition:
		$1 at pe.entry_point
}

rule polycryptor_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PolyCryptor"
	strings:
		$1 = { EB ?? 28 50 6F 6C 79 53 63 72 79 70 74 20 ?? ?? ?? 20 62 79 20 53 4D 54 29 }
	condition:
		$1 at pe.entry_point
}

rule polyene_001_or_higher_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PolyEnE"
		version = "0.01 or higher"
	strings:
		$1 = { 50 6F 6C 79 45 6E 45 00 4D 65 73 73 61 67 65 42 6F 78 41 00 55 53 45 52 33 32 2E 64 6C 6C }
	condition:
		$1 at pe.entry_point
}

rule polyene_001_or_higher_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PolyEnE"
		version = "0.01 or higher"
	strings:
		$1 = { 60 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 60 00 00 E0 }
	condition:
		$1 at pe.entry_point
}

rule popa_001 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PoPa"
		version = "0.01"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 56 57 33 C0 89 45 EC B8 A4 3E 00 10 E8 30 F6 FF FF 33 C0 55 68 BE 40 00 10 ?? ?? ?? ?? 89 20 6A 00 68 80 00 00 00 6A 03 6A 00 6A 01 68 00 00 00 80 8D 55 EC 33 C0 E8 62 E7 FF FF 8B 45 EC E8 32 F2 FF FF 50 E8 B4 F6 FF FF A3 64 66 00 10 33 D2 55 68 93 40 00 10 64 FF 32 64 89 22 83 3D 64 66 00 10 FF 0F 84 3A 01 00 00 6A 00 6A 00 6A 00 A1 64 66 00 10 50 E8 9B F6 FF FF 83 E8 10 50 A1 64 66 00 10 50 E8 BC F6 FF FF 6A 00 68 80 66 00 10 6A 10 68 68 66 00 10 A1 64 66 00 10 50 E8 8B F6 FF FF }
	condition:
		$1 at pe.entry_point
}

rule ppc_protect_11x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PPC-PROTECT"
		version = "1.1x"
	strings:
		$1 = { FF 5F 2D E9 20 00 9F E5 00 00 90 E5 18 00 8F E5 18 00 9F E5 00 00 90 E5 10 00 8F E5 01 00 A0 E3 00 00 00 EB 02 00 00 EA 04 F0 1F E5 }
	condition:
		$1 at pe.entry_point
}

rule princesssandy_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PrincessSandy"
		version = "1.0"
	strings:
		$1 = { 68 27 11 40 00 E8 3C 01 00 00 6A 00 E8 41 01 00 00 A3 00 20 40 00 8B 58 3C 03 D8 0F B7 43 14 0F B7 4B 06 8D 7C 18 18 81 3F 2E 4C 4F 41 74 0B 83 C7 28 49 75 F2 E9 A7 00 00 00 8B 5F 0C 03 1D 00 20 40 00 89 1D 04 20 40 00 8B FB 83 C7 04 68 4C 20 40 00 68 08 20 40 00 6A 00 6A 00 6A 20 6A 00 6A 00 6A 00 57 6A 00 E8 CE 00 00 00 85 C0 74 78 BD 50 C3 00 00 8B 3D 04 20 40 00 8B 07 8D 3C 07 83 C7 04 89 3D 04 20 40 00 8B 0F 83 C7 04 8B 1F 83 C7 04 4D 85 ED 74 57 60 6A 00 51 68 5C 20 40 00 53 FF 35 4C 20 40 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_18_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "1.8"
	strings:
		$1 = { BB DC EE 0D 76 D9 D0 8D 16 85 D8 90 D9 D0 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_18_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "1.8"
	strings:
		$1 = { A4 B3 02 E8 6D 00 00 00 73 F6 31 C9 E8 64 00 00 00 73 1C 31 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 10 C0 73 F7 75 3F AA EB D4 E8 4D 00 00 00 29 D9 75 10 E8 42 00 00 00 EB 28 AC D1 E8 74 4D 11 C9 EB 1C 91 48 C1 E0 08 AC E8 2C 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 89 E8 B3 01 56 89 FE 29 C6 F3 A4 5E EB 8E 00 D2 75 05 8A 16 46 10 D2 C3 31 C9 41 E8 EE FF FF FF 11 C9 E8 E7 FF FF FF 72 F2 C3 31 FF 31 F6 C3 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_18_19 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "1.8 - 1.9"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? 00 00 00 00 00 00 45 78 69 74 50 72 6F 63 65 73 73 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_197 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "1.9.7"
	strings:
		$1 = { 55 8B EC 83 C4 F4 FC 53 57 56 8B 74 24 20 8B 7C 24 24 66 81 3E 4A 43 0F 85 A5 02 00 00 83 C6 0A 33 DB BA 00 00 00 80 C7 44 24 14 08 00 00 00 43 8D A4 24 00 00 00 00 8B FF 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 73 2C 8B 4C 24 10 33 C0 8D A4 24 00 00 00 00 05 00 00 00 00 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 13 C0 49 75 EF 02 44 24 0C 88 07 47 EB C6 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 0F 82 6E 01 00 00 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 0F 83 DC 00 00 00 B9 04 00 00 00 33 C0 8D A4 24 00 00 00 00 8D 64 24 00 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 13 C0 49 75 EF 48 74 B1 0F 89 EF 01 00 00 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 73 42 BD 00 01 00 00 B9 08 00 00 00 33 C0 8D A4 24 00 00 00 00 05 00 00 00 00 03 D2 75 08 8B 16 83 C6 04 F9 13 D2 13 C0 49 75 EF 88 07 47 4D 75 D6 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "1.x"
	strings:
		$1 = { B8 ?? ?? ?? ?? B9 ?? 90 01 ?? BE ?? 10 40 ?? 68 50 91 41 ?? 68 01 ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_20_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "2.0"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 ?? ?? ?? ?? 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_20_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "2.0"
	strings:
		$1 = { 89 ?? ?? 38 00 00 00 8B ?? 00 00 00 00 81 ?? ?? ?? ?? ?? 89 ?? 00 00 00 00 81 ?? 04 00 00 00 81 ?? 04 00 00 00 81 ?? 00 00 00 00 0F 85 D6 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_215_220 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "2.15 - 2.20"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_230_24x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "2.30 - 2.4x"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E0 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule private_exe_protector_25x_27x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private exe Protector"
		version = "2.5x - 2.7x"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 10 00 00 ?? ?? ?? ?? 00 04 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 04 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 E0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule private_personal_packer_102 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private Personal Packer"
		version = "1.0.2"
	strings:
		$1 = { E8 17 00 00 00 E8 68 00 00 00 FF 35 2C 37 00 10 E8 ED 01 00 00 6A 00 E8 2E 04 00 00 E8 41 04 00 00 A3 74 37 00 10 6A 64 E8 5F 04 00 00 E8 30 04 00 00 A3 78 37 00 10 6A 64 E8 4E 04 00 00 E8 1F 04 00 00 A3 7C 37 00 10 A1 74 37 00 10 8B 1D 78 37 00 10 2B D8 8B 0D 7C 37 00 10 2B C8 83 FB 64 73 0F 81 F9 C8 00 00 00 73 07 6A 00 E8 D9 03 00 00 C3 6A 0A 6A 07 6A 00 }
	condition:
		$1 at pe.entry_point
}

rule private_personal_packer_103 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Private Personal Packer"
		version = "1.0.3"
	strings:
		$1 = { E8 19 00 00 00 90 90 E8 68 00 00 00 FF 35 2C 37 00 10 E8 ED 01 00 00 6A 00 E8 2E 04 00 00 E8 41 04 00 00 A3 74 37 00 10 6A 64 E8 5F 04 00 00 E8 30 04 00 00 A3 78 37 00 10 6A 64 E8 4E 04 00 00 E8 1F 04 00 00 A3 7C 37 00 10 A1 74 37 00 10 8B 1D 78 37 00 10 2B D8 8B 0D 7C 37 00 10 2B C8 83 FB 64 73 0F 81 F9 C8 00 00 00 73 07 6A 00 E8 D9 03 00 00 C3 6A 0A 6A 07 6A 00 E8 D3 03 00 00 A3 20 37 00 10 50 6A 00 E8 DE 03 00 00 A3 24 37 00 10 FF 35 20 37 00 10 6A 00 E8 EA 03 00 00 A3 30 37 00 10 FF 35 24 37 00 10 E8 C2 03 00 00 A3 28 37 00 10 8B 0D 30 37 00 10 8B 3D 28 37 00 10 EB 09 49 C0 04 39 55 80 34 39 24 0B C9 }
	condition:
		$1 at pe.entry_point
}

rule privatexe_20a_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PrivateEXE"
		version = "2.0a"
	strings:
		$1 = { 53 E8 ?? ?? ?? ?? 5B 8B C3 2D }
	condition:
		$1 at pe.entry_point
}

rule privatexe_20a_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PrivateEXE"
		version = "2.0a"
	strings:
		$1 = { 06 60 C8 ?? ?? ?? 0E 68 ?? ?? 9A ?? ?? ?? ?? 3D ?? ?? 0F ?? ?? ?? 50 50 0E 68 ?? ?? 9A ?? ?? ?? ?? 0E }
	condition:
		$1 at pe.entry_point
}

rule proactivate_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PROACTIVATE"
	strings:
		$1 = { 55 8B EC B9 0E 00 00 00 6A 00 6A 00 49 75 F9 51 53 56 57 B8 ?? ?? ?? 00 90 90 90 90 90 33 C0 55 68 ?? ?? ?? ?? 64 FF 30 64 89 20 A1 ?? ?? ?? ?? 83 C0 05 A3 ?? ?? ?? ?? C7 05 ?? ?? ?? ?? 0D 00 00 00 E8 85 E2 FF FF 81 3D ?? ?? ?? ?? 21 7E 7E }
	condition:
		$1 at pe.entry_point
}

rule program_protector_xp_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Program Protector XP"
		version = "1.0"
	strings:
		$1 = { E8 ?? ?? ?? ?? 58 83 D8 05 89 C3 81 C3 ?? ?? ?? ?? 8B 43 64 50 }
	condition:
		$1 at pe.entry_point
}

rule protect_shareware_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Protect Shareware"
		version = "1.1"
	strings:
		$1 = { 53 00 74 00 72 00 69 00 6E 00 67 00 46 00 69 00 6C 00 65 00 49 00 6E 00 66 00 6F 00 00 00 ?? 01 00 00 01 00 30 00 34 00 30 00 39 00 30 00 34 00 42 00 30 00 00 00 34 00 ?? 00 01 00 43 00 6F 00 6D 00 70 00 61 00 6E 00 79 00 4E 00 61 00 6D 00 65 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule protection_plus_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Protection Plus"
	strings:
		$1 = { 50 60 29 C0 64 FF 30 E8 ?? ?? ?? ?? 5D 83 ED 3C 89 E8 89 A5 14 ?? ?? ?? 2B 85 1C ?? ?? ?? 89 85 1C ?? ?? ?? 8D 85 27 03 ?? ?? 50 8B ?? 85 C0 0F 85 C0 ?? ?? ?? 8D BD 5B 03 ?? ?? 8D B5 43 03 }
	condition:
		$1 at pe.entry_point
}

rule protext_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PROTEXT"
	strings:
		$1 = { E9 1D 01 00 00 E8 7D 00 00 00 05 E5 EB FF F7 C0 E9 40 8D 09 8D 00 C1 E7 20 C0 E6 20 0F 88 D2 01 00 00 79 04 68 94 9E AC 0F 89 C6 01 00 00 34 B8 21 C9 66 C1 E1 20 66 C1 ED 40 88 E4 71 03 E7 7D }
	condition:
		$1 at pe.entry_point
}

rule pscrambler_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "pscrambler"
		version = "1.2"
	strings:
		$1 = { 55 8B EC B9 04 00 00 00 6A 00 6A 00 49 75 F9 51 53 ?? ?? ?? ?? 10 E8 2D F3 FF FF 33 C0 55 68 E8 31 00 10 64 FF 30 64 89 20 8D 45 E0 E8 53 F5 FF FF 8B 45 E0 8D 55 E4 E8 30 F6 FF FF 8B 45 E4 8D 55 E8 E8 A9 F4 FF FF 8B 45 E8 8D 55 EC E8 EE F7 FF FF 8B 55 EC B8 C4 54 00 10 E8 D9 EC FF FF 83 3D C4 54 00 10 00 0F 84 05 01 00 00 80 3D A0 40 00 10 00 74 41 A1 C4 54 00 10 E8 D9 ED FF FF E8 48 E0 FF FF 8B D8 A1 C4 54 00 10 E8 C8 ED FF FF 50 B8 C4 54 00 10 E8 65 EF FF FF 8B D3 59 E8 69 E1 FF FF 8B C3 E8 12 FA FF FF 8B C3 E8 33 E0 FF FF E9 AD 00 00 00 B8 05 01 00 00 E8 0C E0 FF FF 8B D8 53 68 05 01 00 00 E8 57 F3 FF FF 8D 45 DC 8B D3 E8 39 ED FF FF 8B 55 DC B8 14 56 00 10 B9 00 32 00 10 E8 BB ED FF FF 8B 15 14 56 00 10 B8 C8 54 00 10 E8 53 E5 FF FF BA 01 00 00 00 B8 C8 54 00 10 E8 8C E8 FF FF E8 DF E0 FF FF 85 C0 75 52 6A 00 A1 C4 54 00 10 E8 3B ED FF FF 50 B8 C4 54 00 10 E8 D8 EE FF FF 8B D0 B8 C8 54 00 10 59 E8 3B E6 FF FF E8 76 E0 FF FF B8 C8 54 00 10 E8 4C E6 FF FF E8 67 E0 FF FF 6A 00 6A 00 6A 00 A1 14 56 00 10 E8 53 EE FF FF 50 6A 00 6A 00 E8 41 F3 FF FF 80 3D 9C 40 00 10 00 74 05 E8 EF FB FF FF 33 C0 5A 59 59 64 89 10 68 EF 31 00 10 8D 45 DC BA 05 00 00 00 E8 7D EB FF FF C3 E9 23 E9 FF FF EB EB 5B E8 63 EA FF FF 00 00 00 FF FF FF FF 08 00 00 00 74 65 6D 70 2E 65 78 65 }
	condition:
		$1 at pe.entry_point
}

rule punisher_15d {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PUNiSHER"
		version = "1.5d"
	strings:
		$1 = { EB 04 83 A4 BC CE 60 EB 04 80 BC 04 11 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule punisher_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PUNiSHER"
		version = "1.5"
	strings:
		$1 = { 3F 00 00 80 66 20 ?? 00 7E 20 ?? 00 92 20 ?? 00 A4 20 ?? 00 00 00 00 00 4B 45 52 4E 45 4C 33 32 }
	condition:
		$1 at pe.entry_point
}

rule punkmode_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "PuNkMoD"
		version = "1.x"
	strings:
		$1 = { 94 B9 ?? ?? 00 00 BC ?? ?? ?? ?? 80 34 0C }
	condition:
		$1 at pe.entry_point
}

rule qinwyingshieldlicense_10x_121 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "QinYingShieldLicense"
		version = "1.0x - 1.21"
	strings:
		$1 = { E8 00 00 00 00 58 05 ?? ?? ?? ?? 9C 50 C2 04 00 55 8B EC 56 57 53 34 99 47 49 34 33 EF 31 CD F5 B0 CB B5 B0 A3 A1 A3 A1 B9 FE B9 FE B9 FE B9 FE BF C9 CF A7 D1 BD A3 AC C4 E3 B2 BB D6 AA B5 C0 D5 E2 C0 EF B5 C4 D6 B8 C1 EE CA C7 CA B2 C3 B4 A3 A1 B9 FE B9 FE B9 FE 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule qryptor_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "QrYPt0r"
	strings:
		$1 = { EB 00 E8 B5 00 00 00 E9 2E 01 00 00 64 FF 35 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 64 89 25 00 00 00 00 8B 44 24 04 }
	condition:
		$1 at pe.entry_point
}

rule qryptor_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "QrYPt0r"
	strings:
		$1 = { 80 F9 00 0F 84 8D 01 00 00 8A C3 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 32 C1 3C F3 75 89 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? BA D9 04 00 00 E8 00 00 00 00 5F 81 C7 16 01 00 00 80 2C 3A 01 }
	condition:
		$1 at pe.entry_point
}

rule qryptor_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "QrYPt0r"
	strings:
		$1 = { 86 18 CC 64 FF 35 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 64 89 25 00 00 00 00 BB 00 00 F7 BF ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? B8 78 56 34 12 87 03 E8 CD FE FF FF E8 B3 }
	condition:
		$1 at pe.entry_point
}

rule riscs_process_patcher_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "R!SC's Process Patcher"
		version = "1.4"
	strings:
		$1 = { E8 E1 01 00 00 80 38 22 75 13 80 38 00 74 2E 80 38 20 75 06 80 78 FF 22 74 18 40 EB ED 80 38 00 74 1B EB 19 40 80 78 FF 20 75 F9 80 38 00 74 0D EB 0B 40 80 38 00 74 05 80 38 22 74 00 8B F8 B8 04 60 40 00 68 00 20 40 00 C7 05 A2 20 40 00 44 00 00 00 68 92 20 40 00 68 A2 20 40 00 6A 00 6A 00 6A 04 6A 00 6A 00 6A 00 57 50 E8 7C 01 00 00 85 C0 0F 84 2A 01 00 00 B8 00 60 40 00 8B 00 A3 1C 22 40 00 BE 40 60 40 00 83 7E FC 00 0F 84 F6 00 00 00 8B 3E 83 C6 04 85 FF 0F 84 83 00 00 00 81 FF 72 21 73 63 0F }
	condition:
		$1 at pe.entry_point
}

rule riscs_process_patcher_151 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "R!SC's Process Patcher"
		version = "1.5.1"
	strings:
		$1 = { 68 00 20 40 00 E8 C3 01 00 00 80 38 00 74 0D 66 81 78 FE 22 20 75 02 EB 03 40 EB EE 8B F8 B8 04 60 40 00 68 C4 20 40 00 68 D4 20 40 00 6A 00 6A 00 6A 04 6A 00 6A 00 6A 00 57 50 E8 9F 01 00 00 85 C0 0F 84 39 01 00 00 BE 00 60 40 00 8B 06 A3 28 21 40 00 83 C6 40 83 7E FC 00 0F 84 8F 00 00 00 8B 3E 83 C6 04 85 FF 0F 84 E5 00 00 00 81 FF 72 21 73 63 74 7A 0F B7 1E 8B CF 8D 7E 02 C7 05 24 21 40 00 00 00 00 00 83 05 24 21 40 00 01 50 A1 28 21 40 00 39 05 24 21 40 00 58 0F 84 D8 00 00 00 60 6A 00 53 68 }
	condition:
		$1 at pe.entry_point
}

rule ratpacker_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RatPacker (Glue)"
	strings:
		$1 = { 40 20 FF 00 00 00 00 00 00 00 ?? BE 00 60 40 00 8D BE 00 B0 FF FF }
	condition:
		$1 at pe.entry_point
}

rule razor_1911_encryptor_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RAZOR 1911 encryptor"
	strings:
		$1 = { E8 ?? ?? BF ?? ?? 3B FC 72 ?? B4 4C CD 21 BE ?? ?? B9 ?? ?? FD F3 A5 FC }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.1"
	strings:
		$1 = { 8B 04 24 83 E8 4F 68 ?? ?? ?? ?? FF D0 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_13b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.3b"
	strings:
		$1 = { 61 83 EF 4F 60 68 ?? ?? ?? ?? FF D7 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_13_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.3 - 1.4"
	strings:
		$1 = { 55 8B EC 8B 44 24 04 83 E8 4F 68 ?? ?? ?? ?? FF D0 58 59 50 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.5"
	strings:
		$1 = { 83 2C 24 4F 68 ?? ?? ?? ?? FF 54 24 04 83 44 24 04 4F }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_16v_16c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.6b - 1.6c"
	strings:
		$1 = { 8B C7 03 04 24 2B C7 80 38 50 0F 85 1B 8B 1F FF 68 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_16_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.6"
	strings:
		$1 = { 33 D0 68 ?? ?? ?? ?? FF D2 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_16_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.6"
	strings:
		$1 = { 60 90 61 61 80 7F F0 45 90 60 0F 85 1B 8B 1F FF 68 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "1.x"
	strings:
		$1 = { 90 58 90 50 90 8B 00 90 3C 50 90 58 0F 85 67 D6 EF 11 50 68 }
	condition:
		$1 at pe.entry_point
}

rule rcryptor_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RCryptor"
		version = "2.0"
	strings:
		$1 = { F7 D1 83 F1 FF 6A 00 F7 D1 83 F1 FF 81 04 24 ?? ?? ?? ?? F7 D1 83 F1 FF }
	condition:
		$1 at pe.entry_point
}

rule re_crypt_07x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RE-Crypt"
		version = "0.7x"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 55 81 04 24 0A 00 00 00 C3 8B F5 81 C5 ?? ?? 00 00 89 6D 34 89 75 38 8B 7D 38 81 E7 00 FF FF FF 81 C7 48 00 00 00 47 03 7D 60 8B 4D 5C 83 F9 00 7E 0F 8B 17 33 55 58 89 17 83 C7 04 83 C1 FC EB EC 8B }
	condition:
		$1 at pe.entry_point
}

rule re_crypt_07x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RE-Crypt"
		version = "0.7x"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED F3 1D 40 00 B9 7B 09 00 00 8D BD 3B 1E 40 00 8B F7 61 60 E8 00 00 00 00 5D 55 81 04 24 0A 00 00 00 C3 8B F5 81 C5 ?? ?? 00 00 89 6D 34 89 75 38 8B 7D 38 81 E7 00 FF FF FF 81 C7 48 00 00 00 47 03 7D 60 8B 4D 5C 83 F9 00 7E 0F 8B }
	condition:
		$1 at pe.entry_point
}

rule reflexive_arcade_wrapper_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Reflexive Arcade Wrapper"
	strings:
		$1 = { 55 8B EC 6A FF 68 98 68 42 00 68 14 FA 41 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 F8 50 42 00 33 D2 8A D4 89 15 3C E8 42 00 8B C8 81 E1 FF 00 00 00 89 0D 38 E8 42 00 C1 E1 08 03 CA 89 0D 34 E8 42 00 C1 E8 10 A3 30 E8 42 00 33 F6 56 E8 58 43 00 00 59 85 C0 75 08 6A 1C E8 B0 00 00 00 59 89 75 FC E8 23 40 00 00 FF 15 18 51 42 00 A3 44 FE 42 00 E8 E1 3E 00 00 A3 78 E8 42 00 E8 8A 3C 00 00 E8 CC 3B 00 00 E8 3E F5 FF FF 89 75 D0 8D 45 A4 50 FF 15 14 51 42 00 E8 5D }
	condition:
		$1 at pe.entry_point
}

rule res_crypt_102 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ResCrypt"
		version = "1.02"
	strings:
		$1 = { 55 E8 ?? ?? ?? ?? 5D 81 ED 06 ?? ?? ?? BE ?? ?? ?? ?? ?3 F5 8B DE BA 01 ?? ?? ?? 33 C9 66 8B 4E 0C 66 03 4E 0E 85 C9 74 54 83 C6 10 8B 06 83 FA 01 75 1B 25 ?? ?? ?? 7F 83 F8 03 74 0C 83 F8 0E 74 07 83 F8 10 74 02 EB 05 83 C6 08 EB 2D 8B 46 04 83 C6 08 A9 ?? ?? ?? 80 74 0E 51 56 25 ?? ?? ?? 7F 03 C3 8B F0 42 EB B2 51 03 C3 8B 38 03 FD 8B 48 04 D2 0F 30 0F 47 E2 F9 59 E2 AF 4A 74 04 5E 59 EB F7 8D 85 ?? ?? ?? ?? 5D FF E? }
	condition:
		$1 at pe.entry_point
}

rule reversinglabsprotector_074b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ReversingLabsProtector"
		version = "0.7.4b"
	strings:
		$1 = { 68 00 00 41 00 E8 01 00 00 00 C3 C3 }
	condition:
		$1 at pe.entry_point
}

rule rjoiner_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RJoiner"
		version = "1.2"
	strings:
		$1 = { 55 8B EC 81 EC 0C 02 00 00 8D 85 F4 FD FF FF 56 50 68 04 01 00 00 FF 15 14 10 40 00 90 8D 85 F4 FD FF FF 50 FF 15 10 10 40 00 90 BE 00 20 40 00 90 83 3E FF 0F 84 84 00 00 00 53 57 33 FF 8D 46 }
	condition:
		$1 at pe.entry_point
}

rule rjoiner_12a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RJoiner"
		version = "1.2a"
	strings:
		$1 = { 55 8B EC 81 EC 0C 01 00 00 8D 85 F4 FE FF FF 56 50 68 04 01 00 00 FF 15 0C 10 40 00 94 90 94 8D 85 F4 FE FF FF 50 FF 15 08 10 40 00 94 90 94 BE 00 20 40 00 94 90 94 83 3E FF 74 7D 53 57 33 DB 8D 7E 04 94 90 94 53 68 80 00 00 00 6A 02 53 6A 01 68 00 00 00 C0 57 FF 15 04 10 40 00 89 45 F8 94 90 94 8B 06 8D 74 06 04 94 90 94 8D 45 FC 53 50 8D 46 04 FF 36 50 FF 75 F8 FF 15 00 10 40 00 94 90 94 FF 75 F8 FF 15 10 10 40 00 94 90 94 8D 85 F4 FE FF FF 6A 0A 50 53 57 68 20 10 40 00 53 FF 15 18 10 40 00 94 90 94 8B 06 8D 74 06 04 94 90 94 83 3E FF 75 89 5F 5B 33 C0 5E C9 C2 10 00 CC CC 24 11 }
	condition:
		$1 at pe.entry_point
}
rule rjoiner_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RJoiner"
	strings:
		$1 = { E8 03 FD FF FF 6A 00 E8 0C 00 00 00 FF 25 6C 10 40 00 FF 25 70 10 40 00 FF 25 74 10 40 00 FF 25 78 10 40 00 FF 25 7C 10 40 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 8D B5 ?? ?? ?? ?? 8D 9D ?? ?? ?? ?? 33 FF E8 83 01 00 00 6A ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A ?? FF 95 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? EB 14 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 EB 03 ?? ?? ?? EB 03 ?? ?? ?? 8D B5 CB 22 00 00 8D 9D F0 02 00 00 33 FF E8 ?? ?? ?? ?? EB 03 ?? ?? ?? 6A 40 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 FF 95 9B 0A }
	condition:
		$1 at pe.entry_point
}

rule rlpack_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
	strings:
		$1 = { B8 00 00 00 00 60 0B C0 74 58 E8 00 00 00 00 58 05 43 00 00 00 80 38 E9 75 03 61 EB 35 E8 00 00 00 00 58 25 00 F0 FF FF 33 FF 66 BB 19 5A 66 83 C3 34 66 39 18 75 12 0F B7 50 3C 03 D0 BB E9 44 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_073b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "0.7.3b"
	strings:
		$1 = { 2E 72 6C 70 00 00 00 00 00 50 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 E0 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_073b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "0.7.3b"
	strings:
		$1 = { 60 8B DD E8 00 00 00 00 5D 95 32 C0 95 89 9D 80 00 00 00 B8 42 31 40 00 BB 41 30 40 00 2B C3 03 C5 33 D2 8A 10 40 B9 ?? ?? 00 00 8B F9 30 10 8A 10 40 49 75 F8 64 EF 86 3D 30 00 00 0F B9 FF 4B 89 52 5C 4C BD 77 C2 0C CE 88 4E 2D E8 00 00 00 5D 0D DB 5E 56 41 87 FC 0F F3 05 40 81 68 4B 93 71 40 BB 87 3C 40 40 8B 88 06 75 70 40 40 8B BB B3 43 C4 8F 93 2B F3 4A 88 06 07 30 F5 EA 2A 35 F0 4B 8A C3 07 C1 C6 02 C4 34 C0 74 74 32 02 C4 45 0B 3C 96 BE 0A 82 C3 DE 36 A9 7E 5A 51 A6 BC 63 A8 66 CB 30 58 20 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_10b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.0b"
	strings:
		$1 = { 60 E8 00 00 00 00 8D 64 24 04 8B 6C 24 FC 8D B5 4C 02 00 00 8D 9D 13 01 00 00 33 FF EB 0F FF 74 37 04 FF 34 37 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 EB }
	condition:
		$1 at pe.entry_point
}

rule rlpack_111_114 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.11 - 1.14"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 8D B5 ?? ?? ?? ?? 8D 9D ?? ?? ?? ?? 33 FF EB 0F FF ?? ?? ?? FF ?? ?? ?? D3 83 C4 ?? 83 C7 ?? 83 3C 37 00 75 EB }
	condition:
		$1 at pe.entry_point
}

rule rlpack_112_114_lzma_430 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.12 - 1.14 [LZMA 4.30]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 8D B5 ?? ?? ?? ?? 8D 9D ?? ?? ?? ?? 33 FF 6A ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A ?? FF 95 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? EB ?? 60 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_115_118_aplib_043 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.15 - 1.18 [aPLib 0.43]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 8D B5 ?? ?? ?? ?? 8D 9D ?? ?? ?? ?? 33 FF E8 45 01 00 00 EB 0F FF 74 37 04 FF 34 37 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 EB }
	condition:
		$1 at pe.entry_point
}

rule rlpack_115_118 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.15 - 1.18 DLL"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 E8 00 00 00 00 8B 2C 24 83 C4 ?? 8D B5 ?? ?? ?? ?? 8D 9D ?? ?? ?? ?? 33 FF E8 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_118_aplib_043 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.18 [aPLib 0.43]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 ?? 8D B5 1A 04 00 00 8D 9D C1 02 00 00 33 FF E8 61 01 00 00 EB 0F FF 74 37 04 FF 34 37 FF D3 83 C4 ?? 83 C7 ?? 83 3C 37 00 75 EB 83 BD 06 04 00 00 00 74 0E 83 BD 0A 04 00 00 00 74 05 E8 D7 01 00 00 8D 74 37 04 53 6A ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 FF 95 A7 03 00 00 89 85 16 04 00 00 5B FF B5 16 04 00 00 56 FF D3 83 C4 ?? 8B B5 16 04 00 00 8B C6 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_118_lzma_430 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.18 [LZMA 4.30]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 ?? 8D B5 21 0B 00 00 8D 9D FF 02 00 00 33 FF E8 9F 01 00 00 6A ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 FF 95 AA 0A 00 00 89 85 F9 0A 00 00 EB 14 60 FF B5 F9 0A 00 00 FF 34 37 FF 74 37 04 FF D3 61 83 C7 ?? 83 3C 37 00 75 E6 83 BD 0D 0B 00 00 00 74 0E 83 BD 11 0B 00 00 00 74 05 E8 F6 01 00 00 8D 74 37 04 53 6A ?? 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 FF 95 AA 0A 00 00 89 85 1D 0B 00 00 5B 60 FF B5 F9 0A 00 00 56 FF B5 1D 0B 00 00 FF D3 61 8B B5 1D 0B 00 00 8B C6 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_119_aplib_043_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.19 [aPlib 0.43]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 3C 04 00 00 EB 0C 8B 85 38 04 00 00 89 85 3C 04 00 00 8D B5 60 04 00 00 8D 9D EB 02 00 00 33 FF E8 52 01 00 00 EB 1B 8B 85 3C 04 00 00 FF 74 37 04 01 04 24 FF 34 37 01 04 24 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 DF 83 BD 48 04 00 00 00 74 0E 83 BD 4C 04 00 00 00 74 05 E8 B8 01 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 D1 03 00 00 89 85 5C 04 00 00 5B FF B5 5C 04 00 00 56 FF D3 83 C4 08 8B B5 5C 04 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD 3C 04 00 00 83 C0 04 89 85 58 04 00 00 E9 94 00 00 00 56 FF 95 C9 03 00 00 85 C0 0F 84 B4 00 00 00 89 85 54 04 00 00 8B C6 EB 5B 8B 85 58 04 00 00 8B 00 A9 00 00 00 80 74 14 35 00 00 00 80 50 8B 85 58 04 00 00 C7 00 20 20 20 00 EB 06 FF B5 58 04 00 00 FF B5 54 04 00 00 FF 95 CD 03 00 00 85 C0 74 71 89 07 83 C7 04 8B 85 58 04 00 00 EB 01 40 80 38 00 75 FA 40 89 85 58 04 00 00 66 81 78 02 00 80 74 A5 80 38 00 75 A0 EB 01 46 80 3E 00 75 FA 46 40 8B 38 03 BD 3C 04 00 00 83 C0 04 89 85 58 04 00 00 80 3E 01 0F 85 63 FF FF FF 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 5C 04 00 00 FF 95 D5 03 00 00 E8 3D 00 00 00 E8 24 01 00 00 61 E9 ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_119_aplib_043_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.19 [aPlib 0.43]"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 89 01 00 00 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 3C 04 00 00 EB 0C 8B 85 38 04 00 00 89 85 3C 04 00 00 8D B5 60 04 00 00 8D 9D EB 02 00 00 33 FF E8 52 01 00 00 EB 1B 8B 85 3C 04 00 00 FF 74 37 04 01 04 24 FF 34 37 01 04 24 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 DF 83 BD 48 04 00 00 00 74 0E 83 BD 4C 04 00 00 00 74 05 E8 B8 01 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 D1 03 00 00 89 85 5C 04 00 00 5B FF B5 5C 04 00 00 56 FF D3 83 C4 08 8B B5 5C 04 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD 3C 04 00 00 83 C0 04 89 85 58 04 00 00 E9 94 00 00 00 56 FF 95 C9 03 00 00 85 C0 0F 84 B4 00 00 00 89 85 54 04 00 00 8B C6 EB 5B 8B 85 58 04 00 00 8B 00 A9 00 00 00 80 74 14 35 00 00 00 80 50 8B 85 58 04 00 00 C7 00 20 20 20 00 EB 06 FF B5 58 04 00 00 FF B5 54 04 00 00 FF 95 CD 03 00 00 85 C0 74 71 89 07 83 C7 04 8B 85 58 04 00 00 EB 01 40 80 38 00 75 FA 40 89 85 58 04 00 00 66 81 78 02 00 80 74 A5 80 38 00 75 A0 EB 01 46 80 3E 00 75 FA 46 40 8B 38 03 BD 3C 04 00 00 83 C0 04 89 85 58 04 00 00 80 3E 01 0F 85 63 FF FF FF 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 5C 04 00 00 FF 95 D5 03 00 00 E8 3D 00 00 00 E8 24 01 00 00 61 E9 ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_119_lzma_430_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.19 [LZMA 4.30]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 49 0B 00 00 EB 0C 8B 85 45 0B 00 00 89 85 49 0B 00 00 8D B5 6D 0B 00 00 8D 9D 2F 03 00 00 33 FF 6A 40 68 00 10 00 00 68 00 20 0C 00 6A 00 FF 95 DA 0A 00 00 89 85 41 0B 00 00 E8 76 01 00 00 EB 20 60 8B 85 49 0B 00 00 FF B5 41 0B 00 00 FF 34 37 01 04 24 FF 74 37 04 01 04 24 FF D3 61 83 C7 08 83 3C 37 00 75 DA 83 BD 55 0B 00 00 00 74 0E 83 BD 59 0B 00 00 00 74 05 E8 D7 01 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 DA 0A 00 00 89 85 69 0B 00 00 5B 60 FF B5 41 0B 00 00 56 FF B5 69 0B 00 00 FF D3 61 8B B5 69 0B 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD 49 0B 00 00 83 C0 04 89 85 65 0B 00 00 E9 98 00 00 00 56 FF 95 D2 0A 00 00 89 85 61 0B 00 00 85 C0 0F 84 C8 00 00 00 8B C6 EB 5F 8B 85 65 0B 00 00 8B 00 A9 00 00 00 80 74 14 35 00 00 00 80 50 8B 85 65 0B 00 00 C7 00 20 20 20 00 EB 06 FF B5 65 0B 00 00 FF B5 61 0B 00 00 FF 95 D6 0A 00 00 85 C0 0F 84 87 00 00 00 89 07 83 C7 04 8B 85 65 0B 00 00 EB 01 40 80 38 00 75 FA 40 89 85 65 0B 00 00 66 81 78 02 00 80 74 A1 80 38 00 75 9C EB 01 46 80 3E 00 75 FA 46 40 8B 38 03 BD 49 0B 00 00 83 C0 04 89 85 65 0B 00 00 80 3E 01 0F 85 5F FF FF FF 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 69 0B 00 00 FF 95 DE 0A 00 00 68 00 40 00 00 68 00 20 0C 00 FF B5 41 0B 00 00 FF 95 DE 0A 00 00 E8 3D 00 00 00 E8 24 01 00 00 61 E9 ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_119_lzma_430_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.19 [LZMA 4.30]"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 C7 01 00 00 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 49 0B 00 00 EB 0C 8B 85 45 0B 00 00 89 85 49 0B 00 00 8D B5 6D 0B 00 00 8D 9D 2F 03 00 00 33 FF 6A 40 68 00 10 00 00 68 00 20 0C 00 6A 00 FF 95 DA 0A 00 00 89 85 41 0B 00 00 E8 76 01 00 00 EB 20 60 8B 85 49 0B 00 00 FF B5 41 0B 00 00 FF 34 37 01 04 24 FF 74 37 04 01 04 24 FF D3 61 83 C7 08 83 3C 37 00 75 DA 83 BD 55 0B 00 00 00 74 0E 83 BD 59 0B 00 00 00 74 05 E8 D7 01 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 DA 0A 00 00 89 85 69 0B 00 00 5B 60 FF B5 41 0B 00 00 56 FF B5 69 0B 00 00 FF D3 61 8B B5 69 0B 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD 49 0B 00 00 83 C0 04 89 85 65 0B 00 00 E9 98 00 00 00 56 FF 95 D2 0A 00 00 89 85 61 0B 00 00 85 C0 0F 84 C8 00 00 00 8B C6 EB 5F 8B 85 65 0B 00 00 8B 00 A9 00 00 00 80 74 14 35 00 00 00 80 50 8B 85 65 0B 00 00 C7 00 20 20 20 00 EB 06 FF B5 65 0B 00 00 FF B5 61 0B 00 00 FF 95 D6 0A 00 00 85 C0 0F 84 87 00 00 00 89 07 83 C7 04 8B 85 65 0B 00 00 EB 01 40 80 38 00 75 FA 40 89 85 65 0B 00 00 66 81 78 02 00 80 74 A1 80 38 00 75 9C EB 01 46 80 3E 00 75 FA 46 40 8B 38 03 BD 49 0B 00 00 83 C0 04 89 85 65 0B 00 00 80 3E 01 0F 85 5F FF FF FF 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 69 0B 00 00 FF 95 DE 0A 00 00 68 00 40 00 00 68 00 20 0C 00 FF B5 41 0B 00 00 FF 95 DE 0A 00 00 E8 3D 00 00 00 E8 24 01 00 00 61 E9 ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_basic_edition_aplib {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 Basic Edition [aPLib]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 92 05 00 00 EB 0C 8B 85 8E 05 00 00 89 85 92 05 00 00 8D B5 BA 05 00 00 8D 9D 41 04 00 00 33 FF E8 38 01 00 00 EB 1B 8B 85 92 05 00 00 FF 74 37 04 01 04 24 FF 34 37 01 04 24 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 DF 83 BD 9E 05 00 00 00 74 0E 83 BD A2 05 00 00 00 74 05 E8 D6 01 00 00 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_basic_edition_lzma {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 Basic Edition [LZMA]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 9C 0C 00 00 EB 0C 8B 85 98 0C 00 00 89 85 9C 0C 00 00 8D B5 C4 0C 00 00 8D 9D 82 04 00 00 33 FF 6A 40 68 00 10 00 00 68 00 20 0C 00 6A 00 FF 95 2D 0C 00 00 89 85 94 0C 00 00 E8 59 01 00 00 EB 20 60 8B 85 9C 0C 00 00 FF B5 94 0C 00 00 FF 34 37 01 04 24 FF 74 37 04 01 04 24 FF D3 61 83 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_aplib_043 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 [aPlib 0.43]"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 6F 01 00 00 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 92 05 00 00 EB 0C 8B 85 8E 05 00 00 89 85 92 05 00 00 8D B5 BA 05 00 00 8D 9D 41 04 00 00 33 FF E8 38 01 00 00 EB 1B 8B 85 92 05 00 00 FF 74 37 04 01 04 24 FF 34 37 01 04 24 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 DF 83 BD 9E 05 00 00 00 74 0E 83 BD A2 05 00 00 00 74 05 E8 D6 01 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 27 05 00 00 89 85 B6 05 00 00 5B FF B5 B6 05 00 00 56 FF D3 83 C4 08 8B B5 B6 05 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD 92 05 00 00 83 C0 04 89 85 B2 05 00 00 EB 6E 56 FF 95 1F 05 00 00 0B C0 75 05 E8 C9 02 00 00 85 C0 0F 84 94 00 00 00 89 85 AE 05 00 00 8B C6 EB 2A 8B 85 B2 05 00 00 8B 00 50 FF B5 AE 05 00 00 E8 11 02 00 00 85 C0 74 72 89 07 83 85 B2 05 00 00 04 83 C7 04 8B 85 B2 05 00 00 83 38 00 75 D1 EB 01 46 80 3E 00 75 FA 46 83 C0 04 8B 38 03 BD 92 05 00 00 83 C0 04 89 85 B2 05 00 00 80 3E 01 75 8D 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 B6 05 00 00 FF 95 2B 05 00 00 68 00 80 00 00 6A 00 FF B5 B6 05 00 00 FF 95 2B 05 00 00 E8 61 00 00 00 E8 5C 01 00 00 61 E9 ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_lzma_430 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 [LZMA 4.30]"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 AA 01 00 00 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 9C 0C 00 00 EB 0C 8B 85 98 0C 00 00 89 85 9C 0C 00 00 8D B5 C4 0C 00 00 8D 9D 82 04 00 00 33 FF 6A 40 68 00 10 00 00 68 00 20 0C 00 6A 00 FF 95 2D 0C 00 00 89 85 94 0C 00 00 E8 59 01 00 00 EB 20 60 8B 85 9C 0C 00 00 FF B5 94 0C 00 00 FF 34 37 01 04 24 FF 74 37 04 01 04 24 FF D3 61 83 C7 08 83 3C 37 00 75 DA 83 BD A8 0C 00 00 00 74 0E 83 BD AC 0C 00 00 00 74 05 E8 F2 01 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 2D 0C 00 00 89 85 C0 0C 00 00 5B 60 FF B5 94 0C 00 00 56 FF B5 C0 0C 00 00 FF D3 61 8B B5 C0 0C 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD 9C 0C 00 00 83 C0 04 89 85 BC 0C 00 00 EB 72 56 FF 95 25 0C 00 00 0B C0 75 05 E8 E6 02 00 00 85 C0 0F 84 AB 00 00 00 89 85 B8 0C 00 00 8B C6 EB 2E 8B 85 BC 0C 00 00 8B 00 50 FF B5 B8 0C 00 00 E8 2E 02 00 00 85 C0 0F 84 85 00 00 00 89 07 83 85 BC 0C 00 00 04 83 C7 04 8B 85 BC 0C 00 00 83 38 00 75 CD EB 01 46 80 3E 00 75 FA 46 83 C0 04 8B 38 03 BD 9C 0C 00 00 83 C0 04 89 85 BC 0C 00 00 80 3E 01 75 89 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 C0 0C 00 00 FF 95 31 0C 00 00 68 00 80 00 00 6A 00 FF B5 C0 0C 00 00 FF 95 31 0C 00 00 68 00 80 00 00 6A 00 FF B5 94 0C 00 00 FF 95 31 0C 00 00 E8 61 00 00 00 E8 5C 01 00 00 61 E9 ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_121_aplib_043_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.21 [aPlib 0.43]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 D6 05 00 00 EB 0C 8B 85 D2 05 00 00 89 85 D6 05 00 00 E8 4C 01 00 00 8D B5 FE 05 00 00 8D 9D 85 04 00 00 33 FF E8 77 01 00 00 EB 1B 8B 85 D6 05 00 00 FF 74 37 04 01 04 24 FF 34 37 01 04 24 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 DF 83 BD E2 05 00 00 00 74 0E 83 BD E6 05 00 00 00 74 05 E8 15 02 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 6B 05 00 00 89 85 FA 05 00 00 5B FF B5 FA 05 00 00 56 FF D3 83 C4 08 8B B5 FA 05 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD D6 05 00 00 83 C0 04 89 85 F6 05 00 00 EB 6E 56 FF 95 63 05 00 00 0B C0 75 05 E8 08 03 00 00 85 C0 0F 84 95 00 00 00 89 85 F2 05 00 00 8B C6 EB 2A 8B 85 F6 05 00 00 8B 00 50 FF B5 F2 05 00 00 E8 50 02 00 00 85 C0 74 73 89 07 83 85 F6 05 00 00 04 83 C7 04 8B 85 F6 05 00 00 83 38 00 75 D1 EB 01 46 80 3E 00 75 FA 46 83 C0 04 8B 38 03 BD D6 05 00 00 83 C0 04 89 85 F6 05 00 00 80 3E 01 75 8D 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? FF B5 FA 05 00 00 FF 95 6F 05 00 00 68 ?? ?? ?? ?? 6A 00 FF B5 FA 05 00 00 FF 95 6F 05 00 00 E8 A0 00 00 00 E8 9B 01 00 00 61 E9 ?? ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_121_aplib_043_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.21 [aPlib 0.43]"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 74 01 00 00 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 D6 05 00 00 EB 0C 8B 85 D2 05 00 00 89 85 D6 05 00 00 E8 4C 01 00 00 8D B5 FE 05 00 00 8D 9D 85 04 00 00 33 FF E8 77 01 00 00 EB 1B 8B 85 D6 05 00 00 FF 74 37 04 01 04 24 FF 34 37 01 04 24 FF D3 83 C4 08 83 C7 08 83 3C 37 00 75 DF 83 BD E2 05 00 00 00 74 0E 83 BD E6 05 00 00 00 74 05 E8 15 02 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 6B 05 00 00 89 85 FA 05 00 00 5B FF B5 FA 05 00 00 56 FF D3 83 C4 08 8B B5 FA 05 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD D6 05 00 00 83 C0 04 89 85 F6 05 00 00 EB 6E 56 FF 95 63 05 00 00 0B C0 75 05 E8 08 03 00 00 85 C0 0F 84 95 00 00 00 89 85 F2 05 00 00 8B C6 EB 2A 8B 85 F6 05 00 00 8B 00 50 FF B5 F2 05 00 00 E8 50 02 00 00 85 C0 74 73 89 07 83 85 F6 05 00 00 04 83 C7 04 8B 85 F6 05 00 00 83 38 00 75 D1 EB 01 46 80 3E 00 75 FA 46 83 C0 04 8B 38 03 BD D6 05 00 00 83 C0 04 89 85 F6 05 00 00 80 3E 01 75 8D 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? FF B5 FA 05 00 00 FF 95 6F 05 00 00 68 ?? ?? ?? ?? 6A 00 FF B5 FA 05 00 00 FF 95 6F 05 00 00 E8 A0 00 00 00 E8 9B 01 00 00 61 E9 ?? ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_121_lzma_430_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.21 [LZMA 4.30]"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 E0 0C 00 00 EB 0C 8B 85 DC 0C 00 00 89 85 E0 0C 00 00 E8 87 01 00 00 8D B5 08 0D 00 00 8D 9D C6 04 00 00 33 FF 6A 40 68 00 10 00 00 68 00 20 0C 00 6A 00 FF 95 71 0C 00 00 89 85 D8 0C 00 00 E8 98 01 00 00 EB 20 60 8B 85 E0 0C 00 00 FF B5 D8 0C 00 00 FF 34 37 01 04 24 FF 74 37 04 01 04 24 FF D3 61 83 C7 08 83 3C 37 00 75 DA 83 BD EC 0C 00 00 00 74 0E 83 BD F0 0C 00 00 00 74 05 E8 31 02 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 71 0C 00 00 89 85 04 0D 00 00 5B 60 FF B5 D8 0C 00 00 56 FF B5 04 0D 00 00 FF D3 61 8B B5 04 0D 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD E0 0C 00 00 83 C0 04 89 85 00 0D 00 00 EB 72 56 FF 95 69 0C 00 00 0B C0 75 05 E8 25 03 00 00 85 C0 0F 84 AC 00 00 00 89 85 FC 0C 00 00 8B C6 EB 2E 8B 85 00 0D 00 00 8B 00 50 FF B5 FC 0C 00 00 E8 6D 02 00 00 85 C0 0F 84 86 00 00 00 89 07 83 85 00 0D 00 00 04 83 C7 04 8B 85 00 0D 00 00 83 38 00 75 CD EB 01 46 80 3E 00 75 FA 46 83 C0 04 8B 38 03 BD E0 0C 00 00 83 C0 04 89 85 00 0D 00 00 80 3E 01 75 89 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? FF B5 04 0D 00 00 FF 95 75 0C 00 00 68 ?? ?? ?? ?? 6A 00 FF B5 04 0D 00 00 FF 95 75 0C 00 00 68 ?? ?? ?? ?? 6A 00 FF B5 D8 0C 00 00 FF 95 75 0C 00 00 E8 A0 00 00 00 E8 9B 01 00 00 61 E9 ?? ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_121_lzma_430_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.21 [LZMA 4.30]"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 AF 01 00 00 60 E8 00 00 00 00 8B 2C 24 83 C4 04 83 7C 24 28 01 75 0C 8B 44 24 24 89 85 E0 0C 00 00 EB 0C 8B 85 DC 0C 00 00 89 85 E0 0C 00 00 E8 87 01 00 00 8D B5 08 0D 00 00 8D 9D C6 04 00 00 33 FF 6A 40 68 00 10 00 00 68 00 20 0C 00 6A 00 FF 95 71 0C 00 00 89 85 D8 0C 00 00 E8 98 01 00 00 EB 20 60 8B 85 E0 0C 00 00 FF B5 D8 0C 00 00 FF 34 37 01 04 24 FF 74 37 04 01 04 24 FF D3 61 83 C7 08 83 3C 37 00 75 DA 83 BD EC 0C 00 00 00 74 0E 83 BD F0 0C 00 00 00 74 05 E8 31 02 00 00 8D 74 37 04 53 6A 40 68 00 10 00 00 68 ?? ?? ?? ?? 6A 00 FF 95 71 0C 00 00 89 85 04 0D 00 00 5B 60 FF B5 D8 0C 00 00 56 FF B5 04 0D 00 00 FF D3 61 8B B5 04 0D 00 00 8B C6 EB 01 40 80 38 01 75 FA 40 8B 38 03 BD E0 0C 00 00 83 C0 04 89 85 00 0D 00 00 EB 72 56 FF 95 69 0C 00 00 0B C0 75 05 E8 25 03 00 00 85 C0 0F 84 AC 00 00 00 89 85 FC 0C 00 00 8B C6 EB 2E 8B 85 00 0D 00 00 8B 00 50 FF B5 FC 0C 00 00 E8 6D 02 00 00 85 C0 0F 84 86 00 00 00 89 07 83 85 00 0D 00 00 04 83 C7 04 8B 85 00 0D 00 00 83 38 00 75 CD EB 01 46 80 3E 00 75 FA 46 83 C0 04 8B 38 03 BD E0 0C 00 00 83 C0 04 89 85 00 0D 00 00 80 3E 01 75 89 68 00 40 00 00 68 ?? ?? ?? ?? FF B5 04 0D 00 00 FF 95 75 0C 00 00 68 ?? ?? ?? ?? 6A 00 FF B5 04 0D 00 00 FF 95 75 0C 00 00 68 ?? ?? ?? ?? 6A 00 FF B5 D8 0C 00 00 FF 95 75 0C 00 00 E8 A0 00 00 00 E8 9B 01 00 00 61 E9 ?? ?? ?? ?? ?? 61 C3 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_117_full_edition {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.17 Full Edition"
	strings:
		$1 = { 60 E8 00 00 00 00 8B 2C 24 83 C4 04 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8D B5 ?? ?? ?? ?? 8D 9D ?? ?? ?? ?? 33 FF }
	condition:
		$1 at pe.entry_point
}

rule rlpack_11x_full_edition {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.1x Full Edition"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 10 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_121_full_edition_aplib_043 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 - 1.21 Full Edition [aPlib 0.43]"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 10 00 00 08 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_121_full_edition_lzma_430 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 - 1.21 Full Edition [LZMA 4.30]"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 10 00 00 08 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule rlpack_120_121_full_edition {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RLPack"
		version = "1.20 - 1.21 Full Edition"
		description = "basic edition stub"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 10 00 00 08 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule rod_high_tech_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ROD High TECH"
	strings:
		$1 = { 60 8B 15 1D 13 40 00 F7 E0 8D 82 83 19 00 00 E8 58 0C 00 00 }
	condition:
		$1 at pe.entry_point
}

rule rosasm_2050a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RosAsm"
		version = "2050a"
	strings:
		$1 = { 55 8B EC 60 8B 5D 08 B9 08 00 00 00 BF ?? ?? ?? ?? 83 C7 07 FD 8A C3 24 0F 04 30 3C 39 76 02 04 07 AA C1 EB 04 E2 EE FC 68 00 10 00 00 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 6A 00 FF 15 ?? ?? ?? ?? 61 8B E5 5D C2 04 00 }
	condition:
		$1 at pe.entry_point
}

rule rpolycrypt_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RPolyCrypt"
	strings:
		$1 = { 58 ?? ?? ?? ?? ?? ?? ?? E8 00 00 00 58 E8 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? 04 }
	condition:
		$1 at pe.entry_point
}

rule rpolycrypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "RPolyCrypt"
		version = "1.0"
	strings:
		$1 = { 50 58 97 97 60 61 8B 04 24 80 78 F3 6A E8 00 00 00 00 58 E8 00 00 00 00 58 91 91 EB 00 0F 85 6B F4 76 6F E8 00 00 00 00 83 C4 04 E8 00 00 00 00 58 90 E8 00 00 00 00 83 C4 04 8B 04 24 80 78 F1 }
	condition:
		$1 at pe.entry_point
}

rule safe_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Safe"
		version = "2.0"
	strings:
		$1 = { 83 EC 10 53 56 57 E8 C4 01 00 }
	condition:
		$1 at pe.entry_point
}

rule safedisc_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SafeDisc"
	strings:
		$1 = { 85 C9 74 0C B8 ?? ?? ?? ?? 2B C3 83 E8 05 EB 0E 51 B9 ?? ?? ?? ?? 8B C1 2B C3 03 41 01 59 C6 03 E9 89 43 01 }
	condition:
		$1 in (pe.entry_point + 17 .. pe.entry_point + 18)
}

rule safedisc_4 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SafeDisc"
		version = "4"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 42 6F 47 5F }
	condition:
		$1 at pe.entry_point
}

rule safedisc_450 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SafeDisc"
		version = "4.50"
	strings:
		$1 = { 55 8B EC 60 BB 6E ?? ?? ?? B8 0D ?? ?? ?? 33 C9 8A 08 85 C9 74 0C B8 E4 ?? ?? ?? 2B C3 83 E8 05 EB 0E 51 B9 2B ?? ?? ?? 8B C1 2B C3 03 41 01 59 C6 03 E9 89 43 01 51 68 D9 ?? ?? ?? 33 C0 85 C9 74 05 8B 45 08 EB 00 50 E8 25 FC FF FF 83 C4 08 59 83 F8 00 74 1C C6 03 C2 C6 43 01 0C 85 C9 74 09 61 5D B8 00 00 00 00 EB 96 50 B8 F9 ?? ?? ?? FF 10 61 5D EB 47 80 7C 24 08 00 75 40 51 8B 4C 24 04 89 0D ?? ?? ?? ?? B9 02 ?? ?? ?? 89 4C 24 04 59 EB 29 50 B8 FD ?? ?? ?? FF 70 08 8B 40 0C FF D0 B8 FD ?? ?? ?? FF 30 8B 40 04 FF D0 58 B8 25 ?? ?? ?? FF 30 C3 72 16 61 13 60 0D E9 ?? ?? ?? ?? 66 83 3D ?? ?? ?? ?? ?? 74 05 E9 91 FE FF FF C3 }
	condition:
		$1 at pe.entry_point
}

rule safeguard_10x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SafeGuard"
		version = "1.0x"
	strings:
		$1 = { E8 00 00 00 00 EB 29 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 59 9C 81 C1 E2 FF FF FF EB 01 ?? 9D FF E1 }
	condition:
		$1 at pe.entry_point
}

rule sc_obfuscator {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SC Obfuscator"
	strings:
		$1 = { 60 33 C9 8B 1D ?? ?? ?? ?? 03 1D ?? ?? ?? ?? 8A 04 19 84 C0 74 09 3C ?? 74 05 34 ?? 88 04 19 41 3B 0D ?? ?? ?? ?? 75 E7 A1 ?? ?? ?? ?? 01 05 ?? ?? ?? ?? 61 FF 25 }
	condition:
		$1 at pe.entry_point
}

rule scram_08a1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SCRAM!"
		version = "0.8a1"
	strings:
		$1 = { B4 30 CD 21 3C 02 77 ?? CD 20 BC ?? ?? B9 ?? ?? 8B FC B2 ?? 58 4C }
	condition:
		$1 at pe.entry_point
}

rule scram_c5 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SCRAM!"
		version = "C5"
	strings:
		$1 = { B8 ?? ?? 50 9D 9C 58 25 ?? ?? 75 ?? BA ?? ?? B4 09 CD 21 CD 20 }
	condition:
		$1 at pe.entry_point
}

rule sdc_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SDC"
		version = "1.2"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 01 00 00 00 FF 15 A0 91 40 00 E8 DB FE FF FF 55 89 E5 53 83 EC 14 8B 45 08 8B 00 8B 00 3D 91 00 00 C0 77 3B 3D 8D 00 00 C0 72 4B BB 01 00 00 00 C7 44 24 04 00 00 00 00 C7 04 24 08 00 00 00 E8 CE 24 00 00 83 F8 01 0F 84 C4 00 00 00 85 C0 0F 85 A9 00 00 00 31 C0 83 C4 14 5B 5D C2 04 00 3D 94 00 00 C0 74 56 3D 96 00 00 C0 74 1E 3D 93 00 00 C0 75 E1 EB B5 3D 05 00 00 C0 8D B4 26 00 00 00 00 74 43 3D 1D 00 00 C0 75 CA C7 44 24 04 00 00 00 00 C7 04 24 04 00 00 00 E8 73 24 00 00 83 F8 01 0F 84 99 00 00 00 85 C0 74 A9 C7 04 24 04 00 00 00 FF D0 B8 FF FF FF FF EB 9B 31 DB 8D 74 26 00 E9 69 FF FF FF C7 44 24 04 00 00 00 00 C7 04 24 0B 00 00 00 E8 37 24 00 00 83 F8 01 74 7F 85 C0 0F 84 6D FF FF FF C7 04 24 0B 00 00 00 8D 76 00 FF D0 B8 FF FF FF FF E9 59 FF FF FF C7 04 24 08 00 00 00 FF D0 B8 FF FF FF FF E9 46 FF FF FF C7 44 24 04 01 00 00 00 C7 04 24 08 00 00 00 E8 ED 23 00 00 B8 FF FF FF FF 85 DB 0F 84 25 FF FF FF E8 DB 15 00 00 B8 FF FF FF FF E9 16 FF FF FF C7 44 24 04 01 00 00 00 C7 04 24 04 00 00 00 E8 BD 23 00 00 B8 FF FF FF FF E9 F8 FE FF FF C7 44 24 04 01 00 00 00 C7 04 24 0B 00 00 00 E8 9F 23 00 00 B8 FF FF FF FF E9 DA FE FF FF }
	condition:
		$1 at pe.entry_point
}

rule sdprotector_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SDProtector"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 88 88 88 08 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 58 64 A3 00 00 00 00 58 58 58 58 8B E8 E8 3B 00 00 00 E8 01 00 00 00 FF 58 05 }
	condition:
		$1 at pe.entry_point
}

rule sdprotector_11x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SDProtector"
		version = "1.1x"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 88 88 88 08 64 A1 }
	condition:
		$1 at pe.entry_point
}

rule sdprotector_110_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SDProtector"
		version = "1.10 Basic or Pro Edition"
	strings:
		$1 = { 55 8B EC 6A FF 68 1D 32 13 05 68 88 88 88 08 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 58 64 A3 00 00 00 00 58 58 58 58 8B E8 50 83 EC 08 64 A1 00 00 00 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 83 C4 08 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 64 8F 05 00 00 00 00 64 A3 00 00 00 00 83 C4 08 58 74 07 75 05 19 32 67 E8 E8 74 27 75 25 EB 00 EB FC 68 39 44 CD 00 59 9C 50 74 0F 75 0D E8 59 C2 04 00 55 8B EC E9 FA FF FF 0E E8 EF FF FF FF 56 57 53 78 0F 79 0D E8 34 99 47 49 34 33 EF 31 34 52 47 }
	condition:
		$1 at pe.entry_point
}

rule sdprotector_110_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SDProtector"
		version = "1.10 Basic or Pro Edition"
	strings:
		$1 = { 55 8B EC 6A FF 68 1D 32 13 05 68 88 88 88 08 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 58 64 A3 00 00 00 00 58 58 58 58 8B E8 50 83 EC 08 64 A1 00 00 00 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 83 C4 08 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 64 }
	condition:
		$1 at pe.entry_point
}

rule secupack_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SecuPack"
		version = "1.5"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 56 57 33 C0 89 45 F0 B8 CC 3A 40 ?? E8 E0 FC FF FF 33 C0 55 68 EA 3C 40 ?? 64 FF 30 64 89 20 6A ?? 68 80 ?? ?? ?? 6A 03 6A ?? 6A 01 ?? ?? ?? 80 }
	condition:
		$1 at pe.entry_point
}

rule secureexe_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SecureEXE"
		version = "3.0"
	strings:
		$1 = { E9 B8 00 00 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule securepe_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SecurePE"
		version = "1.x"
	strings:
		$1 = { 8B 04 24 E8 00 00 00 00 5D 81 ED 4C 2F 40 00 89 85 61 2F 40 00 8D 9D 65 2F 40 00 53 C3 00 00 00 00 8D B5 BA 2F 40 00 8B FE BB 65 2F 40 00 B9 C6 01 00 00 AD 2B C3 C1 C0 03 33 C3 AB 43 81 FB 8E 2F 40 00 75 05 BB 65 2F 40 00 E2 E7 89 AD 1A 31 40 00 89 AD 55 34 40 00 89 AD 68 34 40 00 8D 85 BA 2F 40 00 50 C3 }
	condition:
		$1 at pe.entry_point
}


rule securom_7x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Securom"
		version = "7.x"
	strings:
		$1 = { 9C 9C 83 EC ?? C7 44 24 ?? ?? ?? ?? ?? C7 44 24 ?? ?? ?? ?? ?? 89 ?? 24 ?? ?? ?? ?? ?? ?? C1 4C 24 ?? 18 }
	condition:
		$1 at pe.entry_point
}

rule sen_debug_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SEN Debug Protector???"
	strings:
		$1 = { BB ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? 29 ?? ?? 4E E8 }
	condition:
		$1 at pe.entry_point
}

rule sexe_crypter_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Sexe Crypter"
		version = "1.1"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 56 57 33 C0 89 45 EC B8 D8 39 00 10 E8 30 FA FF FF 33 C0 55 68 D4 3A 00 10 64 FF 30 64 89 ?? ?? ?? ?? E4 3A 00 10 A1 00 57 00 10 50 E8 CC FA FF FF 8B D8 53 A1 00 57 00 10 50 E8 FE FA FF FF 8B F8 53 A1 00 57 00 10 50 E8 C8 FA FF FF 8B D8 53 E8 C8 FA FF FF 8B F0 85 F6 74 26 8B D7 4A B8 14 57 00 10 E8 AD F6 FF FF B8 14 57 00 10 E8 9B F6 FF FF 8B CF 8B D6 E8 DA FA FF FF 53 E8 84 FA FF FF 8D 4D EC BA F8 3A 00 10 A1 14 57 00 10 E8 0A FB FF FF 8B 55 EC B8 14 57 00 10 E8 65 F5 FF FF B8 14 57 00 10 E8 63 F6 FF FF E8 52 FC FF FF 33 C0 5A 59 59 64 89 10 68 DB 3A 00 10 8D 45 EC E8 ED F4 FF FF C3 E9 83 EF FF FF EB F0 5F 5E 5B E8 ED F3 FF FF 00 53 45 54 54 49 4E 47 53 00 00 00 00 FF FF FF FF 12 00 00 00 6B 75 74 68 37 36 67 62 62 67 36 37 34 76 38 38 67 79 }
	condition:
		$1 at pe.entry_point
}

rule shegerd_dongle_478 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shegerd Dongle"
		version = "4.78"
	strings:
		$1 = { E8 32 00 00 00 B8 ?? ?? ?? ?? 8B 18 C1 CB 05 89 DA 36 8B 4C 24 0C }
	condition:
		$1 at pe.entry_point
}

rule shellmodify_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ShellModify"
		version = "0.1"
	strings:
		$1 = { 55 8B EC 6A FF 68 98 66 41 00 68 3C 3D 41 00 64 A1 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule shrink_wrap_14 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrink Wrap"
		version = "1.4"
	strings:
		$1 = { 58 60 8B E8 55 33 F6 68 48 01 ?? ?? E8 49 01 ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule shrinker_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? 00 55 8B EC 56 57 75 65 68 00 01 00 00 E8 ?? ?? ?? 00 83 C4 04 8B 75 08 A3 ?? ?? ?? ?? 85 F6 74 1D 68 FF 00 00 00 50 56 FF 15 ?? ?? ?? ?? 85 C0 74 0C C7 05 }
	condition:
		$1 at pe.entry_point
}

rule shrinker_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? 00 55 8B EC 56 57 75 6B 68 00 01 00 00 E8 ?? ?? ?? 00 83 C4 04 8B 75 08 A3 ?? ?? ?? ?? 85 F6 74 23 83 7D 0C 03 77 1D 68 FF 00 00 00 50 56 FF 15 ?? ?? ?? ?? 85 C0 74 0C C7 05 }
	condition:
		$1 at pe.entry_point
}

rule shrinker_32_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.2"
	strings:
		$1 = { 55 8B EC 56 57 75 65 68 00 01 00 00 E8 F1 E6 FF FF 83 C4 04 }
	condition:
		$1 at pe.entry_point
}

rule shrinker_32_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.2"
	strings:
		$1 = { 83 3D ?? ?? ?? ?? ?? 55 8B EC 56 57 75 65 68 00 01 ?? ?? E8 ?? E6 FF FF 83 C4 04 8B 75 08 A3 ?? ?? ?? ?? 85 F6 74 1D 68 FF }
	condition:
		$1 at pe.entry_point
}

rule shrinker_33_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.3"
	strings:
		$1 = { 00 00 55 8B EC 56 57 75 65 68 00 01 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule shrinker_33_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.3"
	strings:
		$1 = { 83 3D ?? ?? ?? 00 00 55 8B EC 56 57 75 65 68 00 01 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule shrinker_34_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.4"
	strings:
		$1 = { 55 8B EC 56 57 75 6B 68 00 01 00 00 E8 11 0B 00 00 83 C4 04 }
	condition:
		$1 at pe.entry_point
}

rule shrinker_34_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.4"
	strings:
		$1 = { 83 3D B4 ?? ?? ?? ?? 55 8B EC 56 57 75 6B 68 00 01 00 00 E8 ?? 0B 00 00 83 C4 04 8B 75 08 A3 B4 ?? ?? ?? 85 F6 74 23 83 7D 0C 03 77 1D 68 FF }
	condition:
		$1 at pe.entry_point
}

rule shrinker_34_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Shrinker"
		version = "3.4"
	strings:
		$1 = { BB ?? ?? BA ?? ?? 81 C3 07 00 B8 40 B4 B1 04 D3 E8 03 C3 8C D9 49 8E C1 26 03 0E 03 00 2B }
	condition:
		$1 at pe.entry_point
}

rule silicon_realms_install_stub_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Silicon Realms Install Stub"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? 92 40 00 68 ?? ?? 40 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 ?? ?? 40 00 33 D2 8A D4 89 15 ?? ?? 40 00 8B C8 81 E1 FF 00 00 00 89 0D ?? ?? 40 00 C1 E1 08 03 CA 89 0D ?? ?? 40 00 C1 E8 10 A3 ?? ?? 40 00 33 F6 56 E8 ?? ?? 00 00 59 85 C0 75 08 6A 1C E8 B0 00 00 00 59 89 75 FC E8 ?? ?? 00 00 FF 15 ?? 91 40 00 A3 ?? ?? 40 00 E8 ?? ?? 00 00 A3 ?? ?? 40 00 E8 ?? ?? 00 00 E8 ?? ?? 00 00 E8 ?? ?? FF FF 89 75 D0 8D 45 A4 50 FF 15 ?? 91 40 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule simbioz_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimbiOZ"
	strings:
		$1 = { 50 60 E8 00 00 00 00 5D 81 ED 07 10 40 00 68 80 0B 00 00 8D 85 1F 10 40 00 50 E8 84 0B 00 00 }
	condition:
		$1 at pe.entry_point
}

rule simbioz_13_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimbiOZ"
		version = "1.3 - 2.xx"
	strings:
		$1 = { 57 57 8D 7C 24 04 50 B8 00 ?? ?? ?? AB 58 5F C3 }
	condition:
		$1 at pe.entry_point
}

rule simbioz_21_poly {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimbiOZ"
		version = "2.1 Poly"
	strings:
		$1 = { 55 50 8B C4 83 C0 04 C7 00 ?? ?? ?? ?? 58 C3 90 }
	condition:
		$1 at pe.entry_point
}

rule simbioz_polycryptor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimbiOZ"
		version = "PolyCryptor"
	strings:
		$1 = { 55 60 E8 00 00 00 00 5D 81 ED ?? ?? ?? ?? 8D 85 ?? ?? ?? ?? 68 ?? ?? ?? ?? 50 E8 }
	condition:
		$1 at pe.entry_point
}

rule simple_upx_cryptor_3042005 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Simple UPX Cryptor"
		version = "30.4.2005"
	strings:
		$1 = { 60 B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? ?? ?? ?? ?? E2 FA 61 68 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule simplepack_10x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimplePack"
		version = "1.0x"
	strings:
		$1 = { 60 E8 00 00 00 00 5B 8D 5B FA 6A 00 FF 93 ?? ?? 00 00 89 C5 8B 7D 3C 8D 74 3D 00 8D BE F8 00 00 00 8B 86 88 00 00 00 09 C0 }
	condition:
		$1 at pe.entry_point
}

rule simplepack_11x_2xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimplePack"
		version = "1.1x - 1.2x"
	strings:
		$1 = { 60 E8 00 00 00 00 5B 8D 5B FA BD ?? ?? ?? ?? 8B 7D 3C 8D 74 3D 00 8D BE F8 00 00 00 0F B7 76 06 4E 8B 47 10 09 C0 }
	condition:
		$1 at pe.entry_point
}

rule simplepack_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SimplePack"
		version = "1.x"
	strings:
		$1 = { 4D 5A 90 EB 01 00 52 E9 ?? 01 00 00 50 45 00 00 4C 01 02 00 }
	condition:
		$1 at pe.entry_point
}

rule skd_undetectabler_20_pro {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SkD Undetectabler"
		version = "2.0 Pro"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 FC 26 00 10 E8 EC F3 FF FF 6A 0F E8 15 F5 FF FF E8 64 FD FF FF E8 BB ED FF FF 8D 40 }
	condition:
		$1 at pe.entry_point
}

rule skd_undetectabler_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SkD Undetectabler"
		version = "3.0"
	strings:
		$1 = { 55 8B EC 81 EC 10 02 00 00 68 00 02 00 00 8D 85 F8 FD FF FF 50 6A 00 FF 15 38 10 00 01 50 FF 15 3C 10 00 01 8D 8D F8 FD FF FF 51 E8 4F FB FF FF 83 C4 04 8B 15 ?? 16 00 01 52 A1 ?? 16 00 01 50 E8 50 FF FF FF 83 C4 08 A3 ?? 16 00 01 C7 85 F4 FD FF FF 00 00 00 00 EB 0F 8B 8D F4 FD FF FF 83 C1 01 89 8D F4 FD FF FF 8B 95 F4 FD FF FF 3B 15 ?? 16 00 01 73 1C 8B 85 F4 FD FF FF 8B 0D ?? 16 00 01 8D 54 01 07 81 FA 74 10 00 01 75 02 EB 02 EB C7 8B 85 F4 FD FF FF 50 E8 ?? 00 00 00 83 C4 04 89 85 F0 FD FF FF 8B 8D F0 FD FF FF 89 4D FC C7 45 F8 00 00 00 00 EB 09 8B 55 F8 83 C2 01 89 55 F8 8B 45 F8 3B 85 F4 FD FF FF 73 15 8B 4D FC 03 4D F8 8B 15 ?? 16 00 01 03 55 F8 8A 02 88 01 EB D7 83 3D ?? 16 00 01 00 74 }
	condition:
		$1 at pe.entry_point
}

rule slvcodeprotector_060_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SLVc0deProtector"
		version = "0.60"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 49 00 00 00 69 E8 49 00 00 00 95 E8 4F 00 00 00 68 E8 1F 00 00 00 49 E8 E9 FF FF FF 67 E8 1F 00 00 00 93 E8 31 00 00 00 78 E8 DD FF FF FF 38 E8 E3 FF FF FF 66 E8 0D 00 00 00 04 E8 E3 FF FF FF 70 E8 CB FF FF FF 69 E8 DD FF FF FF 58 E8 DD FF FF FF 69 E8 E3 FF FF FF 79 E8 BF FF FF FF 69 83 C4 40 E8 00 00 00 00 5D 81 ED 97 11 40 00 8D B5 EF 11 40 00 B9 FE 2D 00 00 8B FE AC F8 ?? ?? ?? ?? ?? ?? 90 }
	condition:
		$1 at pe.entry_point
}

rule slvcodeprotector_060_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SLVc0deProtector"
		version = "0.60"
	strings:
		$1 = { EB 02 FA 04 E8 49 00 00 00 69 E8 49 00 00 00 95 E8 4F 00 00 00 68 E8 1F 00 00 00 49 E8 E9 FF FF FF 67 E8 1F 00 00 00 93 E8 31 00 00 00 78 E8 DD }
	condition:
		$1 at pe.entry_point
}

rule slvcodeprotector_061 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SLVc0deProtector"
		version = "0.61"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? EB 02 FA 04 E8 49 00 00 00 69 E8 49 00 00 00 95 E8 4F 00 00 00 68 E8 1F 00 00 00 49 E8 E9 FF FF FF 67 E8 1F 00 }
	condition:
		$1 at pe.entry_point
}

rule slvcodeprotector_11_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SLVc0deProtector"
		version = "1.1"
	strings:
		$1 = { E8 00 00 00 00 58 C6 00 EB C6 40 01 08 FF E0 E9 4C }
	condition:
		$1 at pe.entry_point
}

rule slvcodeprotector_11_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SLVc0deProtector"
		version = "1.1"
	strings:
		$1 = { E8 01 00 00 00 A0 5D EB 01 69 81 ED 5F 1A 40 00 8D 85 92 1A 40 00 F3 8D 95 83 1A 40 00 8B C0 8B D2 2B C2 83 E8 05 89 42 01 E8 FB FF FF FF 69 83 C4 08 E8 06 00 00 00 69 E8 F2 FF FF FF F3 B9 05 00 00 00 51 8D B5 BF 1A 40 00 8B FE B9 58 15 00 00 AC 32 C1 F6 D0 EB 01 00 D0 C0 FE C8 02 C1 AA E2 EF 59 E2 DE B7 FE AB E1 24 C8 0C 88 7A E1 B1 6A F7 95 83 1B A8 7F F8 A8 B0 1A 8B 08 91 47 6C 5A 88 6C 65 39 85 DB CB 54 3D B9 24 CF 4C AE C6 63 74 2C 63 F0 C8 18 0B 97 6B 79 63 A8 AB B8 78 A9 30 2F 2B DA 18 AC }
	condition:
		$1 at pe.entry_point
}

rule smarte_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SmartE"
	strings:
		$1 = { EB 15 03 00 00 00 ?? 00 00 00 00 00 00 00 00 00 00 00 68 00 00 00 00 55 E8 00 00 00 00 5D 81 ED 1D 00 00 00 8B C5 55 60 9C 2B 85 8F 07 00 00 89 85 83 07 00 00 FF 74 24 2C E8 BB 01 00 00 0F 82 2F 06 00 00 E8 8E 04 00 00 49 0F 88 23 06 }
	condition:
		$1 at pe.entry_point
}

rule smartloader {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SmartLoader"
	strings:
		$1 = { 55 56 57 E8 00 00 00 00 5D 81 ED E2 5F 00 10 EB 05 E9 67 01 00 00 8B 85 E5 61 00 10 85 C0 74 0A 8B 44 24 10 89 85 D9 61 00 10 8B 85 D9 61 00 10 03 40 3C 05 80 00 00 00 8B 08 03 8D D9 61 00 10 }
	condition:
		$1 at pe.entry_point
}

rule smokescrypt_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SmokesCrypt"
		version = "1.2"
	strings:
		$1 = { 60 B8 ?? ?? ?? ?? B8 ?? ?? ?? ?? 8A 14 08 80 F2 ?? 88 14 08 41 83 F9 ?? 75 F1 }
	condition:
		$1 at pe.entry_point
}

rule soft_defender_10_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Soft Defender"
		version = "1.0 - 1.1"
	strings:
		$1 = { 74 07 75 05 19 32 67 E8 E8 74 1F 75 1D E8 68 39 44 CD ?? 59 9C 50 74 0A 75 08 E8 59 C2 04 ?? 55 8B EC E8 F4 FF FF FF 56 57 53 78 0F 79 0D E8 34 99 47 49 34 33 EF 31 34 52 47 23 68 A2 AF 47 01 59 E8 ?? ?? ?? ?? 58 05 BA 01 ?? ?? 03 C8 74 BE 75 BC E8 }
	condition:
		$1 at pe.entry_point
}

rule soft_defender_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Soft Defender"
		version = "1.12"
	strings:
		$1 = { 74 07 75 05 19 32 67 E8 E8 74 1F 75 1D E8 68 39 44 CD 00 59 9C 50 74 0A 75 08 E8 59 C2 04 00 55 8B EC E8 F4 FF FF FF 56 57 53 78 0F 79 0D E8 34 99 47 49 34 33 EF 31 34 52 47 23 68 A2 AF 47 01 59 E8 01 00 00 00 FF 58 05 BE 01 00 00 03 C8 74 BD 75 BB E8 }
	condition:
		$1 at pe.entry_point
}

rule soft_defender_11x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Soft Defender"
		version = "1.1x"
	strings:
		$1 = { 74 07 75 05 ?? ?? ?? ?? ?? 74 1F 75 1D ?? 68 ?? ?? ?? 00 59 9C 50 74 0A 75 08 ?? 59 C2 04 00 ?? ?? ?? E8 F4 FF FF FF ?? ?? ?? 78 0F 79 0D }
	condition:
		$1 at pe.entry_point
}

rule soft_defender_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Soft Defender"
		version = "1.x"
	strings:
		$1 = { 74 07 75 05 19 32 67 E8 E8 74 1F 75 1D E8 68 39 44 CD 00 59 9C 50 74 0A 75 08 E8 59 C2 04 00 55 8B EC E8 F4 FF FF FF 56 57 53 78 0F 79 0D E8 34 99 47 49 34 33 EF 31 34 52 47 23 68 A2 AF 47 01 59 E8 01 00 00 00 FF 58 05 E6 01 00 00 03 C8 74 BD 75 BB E8 00 }
	condition:
		$1 at pe.entry_point
}

rule softcomp_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SoftComp"
		version = "1.x"
	strings:
		$1 = { E8 00 00 00 00 81 2C 24 3A 10 41 00 5D E8 00 00 00 00 81 2C 24 31 01 00 00 8B 85 2A 0F 41 00 29 04 24 8B 04 24 89 85 2A 0F 41 00 58 8B 85 2A 0F 41 00 }
	condition:
		$1 at pe.entry_point
}

rule softprotect_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SoftProtect"
	strings:
		$1 = { E8 ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? C7 00 00 00 00 00 E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 50 E8 ?? ?? ?? ?? 83 ?? ?? ?? ?? ?? 01 }
	condition:
		$1 at pe.entry_point
}

rule softprotect_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SoftProtect"
	strings:
		$1 = { EB 01 E3 60 E8 03 ?? ?? ?? D2 EB 0B 58 EB 01 48 40 EB 01 35 FF E0 E7 61 60 E8 03 ?? ?? ?? 83 EB 0E EB 01 0C 58 EB 01 35 40 EB 01 36 FF E0 0B 61 EB 01 83 9C EB 01 D5 EB 08 35 9D EB 01 89 EB 03 0B EB F7 E8 ?? ?? ?? ?? 58 E8 ?? ?? ?? ?? 59 83 01 01 80 39 5C 75 F2 33 C4 74 0C 23 C4 0B C4 C6 01 59 C6 01 59 EB E2 90 E8 44 14 ?? ?? 8D 85 CF 13 ?? ?? C7 ?? ?? ?? ?? ?? E8 61 0E ?? ?? E8 2E 14 ?? ?? 8D 85 E4 01 ?? ?? 50 E8 E2 15 ?? ?? 83 BD 23 01 ?? ?? 01 75 07 E8 21 0D ?? ?? EB 09 8D 85 CF 13 ?? ?? 83 08 }
	condition:
		$1 at pe.entry_point
}

rule softsentry_211 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SoftSentry"
		version = "2.11"
	strings:
		$1 = { 55 8B EC 83 EC ?? 53 56 57 E9 50 }
	condition:
		$1 at pe.entry_point
}

rule softsentry_300 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SoftSentry"
		version = "3.00"
	strings:
		$1 = { 55 8B EC 83 EC ?? 53 56 57 E9 B0 06 }
	condition:
		$1 at pe.entry_point
}

rule software_compress_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Software Compress"
		version = "1.2"
	strings:
		$1 = { E9 BE 00 00 00 60 8B 74 24 24 8B 7C 24 28 FC B2 80 33 DB A4 B3 02 E8 6D 00 00 }
	condition:
		$1 at pe.entry_point
}

rule software_compress_14_lite {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Software Compress"
		version = "1.4 LITE"
	strings:
		$1 = { E8 00 00 00 00 81 2C 24 AA 1A 41 00 5D E8 00 00 00 00 83 2C 24 6E 8B 85 5D 1A 41 00 29 04 24 8B 04 24 89 85 5D 1A 41 00 58 8B 85 5D 1A 41 00 8B 50 3C 03 D0 8B 92 80 00 00 00 03 D0 8B 4A 58 89 8D 49 1A 41 00 8B 4A 5C 89 8D 4D 1A 41 00 8B 4A 60 89 8D 55 1A }
	condition:
		$1 at pe.entry_point
}

rule softwrap_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SoftWrap"
	strings:
		$1 = { 52 53 51 56 57 55 E8 ?? ?? ?? ?? 5D 81 ED 36 ?? ?? ?? E8 ?? 01 ?? ?? 60 BA ?? ?? ?? ?? E8 ?? ?? ?? ?? 5F }
	condition:
		$1 at pe.entry_point
}

rule solidshield_protector_1x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Solidshield Protector"
		version = "1.x"
	strings:
		$1 = { 68 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? C3 00 60 89 00 0A 00 00 00 46 33 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule solidshield_protector_1x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Solidshield Protector"
		version = "1.x"
	strings:
		$1 = { 8B 44 24 08 48 75 0A FF 74 24 04 E8 ?? ?? ?? ?? 59 33 C0 40 C2 0C 00 55 8B EC 56 8B 75 08 85 F6 75 28 68 ?? ?? ?? ?? BE ?? ?? ?? ?? 56 FF 15 ?? ?? ?? ?? 59 59 6A ?? 68 ?? ?? ?? ?? 56 6A ?? FF ?? ?? ?? ?? ?? E9 80 00 00 00 83 FE 01 75 07 5E 5D E9 D2 F6 FF FF 83 FE 02 57 8B 7D 10 75 53 FF 75 24 FF 75 20 FF 75 1C FF 75 18 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? BE ?? ?? ?? ?? 56 57 E8 ?? ?? ?? ?? 83 C4 20 3C 01 75 04 8B C6 EB 6A 57 FF 75 0C E8 ?? ?? ?? ?? 57 E8 ?? ?? ?? ?? 56 57 E8 ?? ?? ?? ?? 83 C4 14 3C 01 74 DF 6A 03 5E 83 FE 03 75 1B 57 E8 ?? ?? ?? ?? C7 04 24 ?? ?? ?? ?? E8 ?? ?? ?? ?? 59 6A 00 FF 15 ?? ?? ?? ?? 83 FE 04 75 0D FF 75 2C FF 75 28 E8 ?? ?? ?? ?? 59 59 83 FE 05 75 11 FF 75 30 FF 75 2C FF 75 28 E8 ?? ?? ?? ?? 83 C4 0C 33 C0 5F 5E 5D C3 }
	condition:
		$1 at pe.entry_point
}

rule spec_b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SPEC"
		version = "b2"
	strings:
		$1 = { 55 57 51 53 E8 ?? ?? ?? ?? 5D 8B C5 81 ED ?? ?? ?? ?? 2B 85 ?? ?? ?? ?? 83 E8 09 89 85 ?? ?? ?? ?? 0F B6 }
	condition:
		$1 at pe.entry_point
}

rule spec_b3 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SPEC"
		version = "b3"
	strings:
		$1 = { 5B 53 50 45 43 5D E8 ?? ?? ?? ?? 5D 8B C5 81 ED 41 24 40 ?? 2B 85 89 26 40 ?? 83 E8 0B 89 85 8D 26 40 ?? 0F B6 B5 91 26 40 ?? 8B FD }
	condition:
		$1 at pe.entry_point
}

rule special_exe_password_protector_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Special EXE Pasword Protector"
		version = "1.01"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 00 00 00 89 AD 8C 01 00 00 8B C5 2B 85 FE 75 00 00 89 85 3E }
	condition:
		$1 at pe.entry_point
}

rule splash_bitmap_100_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Splash Bitmap"
		version = "1.00"
		description = "with unpack code"
	strings:
		$1 = { E8 00 00 00 00 60 8B 6C 24 20 55 81 ED ?? ?? ?? ?? 8D BD ?? ?? ?? ?? 8D 8D ?? ?? ?? ?? 29 F9 31 C0 FC F3 AA 8B 04 24 48 66 25 00 F0 66 81 38 4D 5A 75 F4 8B 48 3C 81 3C 01 50 45 00 00 75 E8 89 85 ?? ?? ?? ?? 6A 40 }
	condition:
		$1 at pe.entry_point
}

rule splash_bitmap_100_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Splash Bitmap"
		version = "1.00"
	strings:
		$1 = { E8 00 00 00 00 60 8B 6C 24 20 55 81 ED ?? ?? ?? ?? 8D BD ?? ?? ?? ?? 8D 8D ?? ?? ?? ?? 29 F9 31 C0 FC F3 AA 8B 04 24 48 66 25 00 F0 66 81 38 4D 5A 75 F4 8B 48 3C 81 3C 01 50 45 00 00 75 E8 89 85 ?? ?? ?? ?? 8D BD ?? ?? ?? ?? 6A 00 }
	condition:
		$1 at pe.entry_point
}

rule splasher_10_30 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Splasher"
		version = "1.0 - 3.0"
	strings:
		$1 = { 9C 60 8B 44 24 24 E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? 50 E8 ED 02 ?? ?? 8C C0 0F 84 }
	condition:
		$1 at pe.entry_point
}

rule splayer_008 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SPLayer"
		version = "0.08"
	strings:
		$1 = { 8D 40 00 B9 ?? ?? ?? ?? 6A ?? 58 C0 0C ?? ?? 48 ?? ?? 66 13 F0 91 3B D9 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule splice_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Splice"
		version = "1.1"
	strings:
		$1 = { 68 00 1A 40 00 E8 EE FF FF FF 00 00 00 00 00 00 30 00 00 00 40 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 01 00 00 00 ?? ?? ?? ?? ?? ?? 50 72 6F 6A 65 63 74 31 00 ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 06 00 00 00 AC 29 40 00 07 00 00 00 BC 28 40 00 07 00 00 00 74 28 40 00 07 00 00 00 2C 28 40 00 07 00 00 00 08 23 40 00 01 00 00 00 38 21 40 00 00 00 00 00 FF FF FF FF FF FF FF FF 00 00 00 00 8C 21 40 00 08 ?? 40 00 01 00 00 00 AC 19 40 00 00 00 00 00 00 00 00 00 00 00 00 00 AC 19 40 00 4F 00 43 00 50 00 00 00 E7 AF 58 2F 9A 4C 17 4D B7 A9 CA 3E 57 6F F7 76 }
	condition:
		$1 at pe.entry_point
}

rule st_protector_15 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ST Protector"
		version = "1.5"
	strings:
		$1 = { 00 00 00 00 4B 65 52 6E 45 6C 33 32 2E 64 4C 6C 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 }
	condition:
		$1 at pe.entry_point
}

rule stabstr_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "STABSTR"
	strings:
		$1 = { 55 89 E5 83 EC 14 53 8B 4D 08 8B 45 0C 8B 55 10 BB 01 00 00 00 83 F8 01 74 0E 72 44 83 F8 02 74 6F 83 F8 03 74 72 EB 7E 89 0D ?? ?? ?? ?? C7 05 ?? ?? ?? ?? 01 00 00 00 89 15 ?? ?? ?? ?? 83 C4 F8 }
	condition:
		$1 at pe.entry_point
}


rule starforce_protection_driver {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "StarForce"
		description = "Protection Driver"
	strings:
		$1 = { 57 68 ?? 0D 01 00 68 00 ?? ?? 00 E8 50 ?? FF FF 68 ?? ?? ?? 00 68 ?? ?? ?? 00 68 ?? ?? ?? 00 68 ?? ?? ?? 00 68 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule stealth_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ste@lth"
	strings:
		$1 = { ?? ?? ?? ?? ?? B8 ?? ?? ?? 00 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule stealth_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ste@lth"
	strings:
		$1 = { ?? ?? ?? ?? ?? B9 ?? ?? ?? 00 51 C3 }
	condition:
		$1 at pe.entry_point
}

rule stealth_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ste@lth"
	strings:
		$1 = { ?? ?? ?? ?? ?? BB ?? ?? ?? 00 53 C3 }
	condition:
		$1 at pe.entry_point
}

rule stealth_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ste@lth"
		version = "1.01"
	strings:
		$1 = { BA ?? ?? ?? 00 FF E2 BA ?? ?? ?? 00 B8 ?? ?? ?? ?? 89 02 83 C2 03 B8 ?? ?? ?? ?? 89 02 83 C2 FD FF E2 }
	condition:
		$1 at pe.entry_point
}

rule stealth_pe_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Stealth PE"
		version = "1.1"
	condition:
		stealth_101
}

rule stealth_210 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Ste@lth"
		version = "2.10"
	strings:
		$1 = { B8 ?? ?? ?? ?? BA ?? ?? ?? ?? B9 ?? ?? ?? ?? 31 01 83 C1 04 4A 75 F8 EB C0 }
	condition:
		$1 at pe.entry_point
}

rule stones_pe_encryptor_10_113 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Stone's PE Encryptor"
		version = "1.0 - 1.13"
	strings:
		$1 = { 55 57 56 52 51 53 E8 ?? ?? ?? ?? 5D 8B D5 81 }
	condition:
		$1 at pe.entry_point
}

rule stones_pe_encryptor_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Stone's PE Encryptor"
		version = "2.0"
	strings:
		$1 = { 53 51 52 56 57 55 E8 ?? ?? ?? ?? 5D 81 ED 42 30 40 ?? FF 95 32 35 40 ?? B8 37 30 40 ?? 03 C5 2B 85 1B 34 40 ?? 89 85 27 34 40 ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule stud_rc4_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "STUD RC4"
		version = "1.0 Jamie Edition"
	strings:
		$1 = { 68 2C 11 40 00 E8 F0 FF FF FF 00 00 00 00 00 00 30 00 00 00 38 00 00 00 00 00 00 00 37 BB 71 EC A4 E1 98 4C 9B FE 8F 0F FA 6A 07 F6 00 00 00 00 00 00 01 00 00 00 20 20 46 6F 72 20 73 74 75 64 00 20 54 6F 00 00 00 00 06 00 00 00 CC 1A 40 00 07 00 00 00 D4 18 40 00 07 00 00 00 7C 18 40 00 07 00 00 00 2C 18 40 00 07 00 00 00 E0 17 40 00 56 42 35 21 F0 1F 2A 00 00 00 00 00 00 00 00 00 00 00 00 00 7E 00 00 00 00 00 00 00 00 00 00 00 00 00 0A 00 09 04 00 00 00 00 00 00 E8 13 40 00 F4 13 40 00 00 F0 30 00 00 FF FF FF 08 00 00 00 01 00 00 00 00 00 00 00 E9 00 00 00 04 11 40 00 04 11 40 00 C8 10 40 00 78 00 00 00 7C 00 00 00 81 00 00 00 82 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 61 61 61 00 53 74 75 64 00 00 73 74 75 64 00 00 01 00 01 00 30 16 40 00 00 00 00 00 FF FF FF FF FF FF FF FF 00 00 00 00 B4 16 40 00 10 30 40 00 07 00 00 00 24 12 40 00 0E 00 20 00 00 00 00 00 1C 9E 21 00 EC 11 40 00 5C 10 40 00 E4 1A 40 00 2C 34 40 00 68 17 40 00 58 17 40 00 78 17 40 00 8C 17 40 00 8C 10 40 00 62 10 40 00 92 10 40 00 F8 1A 40 00 24 19 40 00 98 10 40 00 9E 10 40 00 77 04 18 FF 04 1C FF 05 00 00 24 01 00 0D 14 00 78 1C 40 00 48 21 40 00 }
	condition:
		$1 at pe.entry_point
}

rule superdat_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SuperDAT"
	strings:
		$1 = { 55 8B EC 6A FF 68 40 F3 42 00 68 A4 BF 42 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 58 53 56 57 89 65 E8 FF 15 08 F2 42 00 33 D2 8A D4 89 15 60 42 43 00 8B C8 81 E1 FF 00 00 00 89 0D }
	condition:
		$1 at pe.entry_point
}

rule svk_protector_1051 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SVK-Protector"
		version = "1.051"
	strings:
		$1 = { 60 EB 03 C7 84 E8 EB 03 C7 84 9A E8 00 00 00 00 5D 81 ED 10 00 00 00 EB 03 C7 84 E9 64 A0 23 00 00 00 EB }
	condition:
		$1 at pe.entry_point
}

rule svk_protector_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SVK-Protector"
		version = "1.11"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED 06 ?? ?? ?? 64 A0 23 }
	condition:
		$1 at pe.entry_point
}

rule svk_protector_13x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SVK-Protector"
		version = "1.3x"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 06 00 00 00 EB 05 B8 ?? ?? 42 00 64 A0 23 }
	condition:
		$1 at pe.entry_point
}

rule svk_protector_143 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SVK-Protector"
		version = "1.43"
	strings:
		$1 = { 78 4E 88 4C 0E B0 3C 78 4E 97 56 7B 94 90 00 00 08 DB 5C 50 20 00 05 6? }
	condition:
		$1 at pe.entry_point
}

rule svk_protector_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SVK-Protector"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED 06 00 00 00 EB 05 B8 ?? ?? ?? ?? 64 A0 23 00 00 00 EB 03 C7 84 E8 ?? ?? ?? ?? C7 84 E9 }
	condition:
		$1 at pe.entry_point
}

rule symantec_file_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SYMANTEC FILE"
	strings:
		$1 = { EB 08 ?? ?? ?? ?? 00 00 00 00 ?? 0B ?? ?? ?? ?? ?? 05 E8 ?? 00 00 00 52 FF 74 24 ?? FF 74 24 ?? FF 74 24 CC FF 74 24 ?? E8 06 00 00 00 ?? 08 90 ?? 05 ?? C2 10 00 }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 4)
}

rule symantec_file_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "SYMANTEC FILE"
	strings:
		$1 = { EB 08 ?? ?? ?? ?? 00 00 00 00 6A 17 E8 0D 00 00 00 6A 30 E8 06 00 00 00 7A 08 90 7B 05 69 C2 04 00 41 52 78 0B 51 52 5A 59 79 05 E8 02 00 00 00 53 FF 74 24 F4 FF 74 24 38 FF 74 24 6C }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 4)
}

rule tpack_05c_m1 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "T-PACK"
		version = "0.5c -m1"
	strings:
		$1 = { 68 ?? ?? FD 60 BE ?? ?? BF ?? ?? B9 ?? ?? F3 A4 8B F7 BF ?? ?? FC 46 E9 8E FE }
	condition:
		$1 at pe.entry_point
}

rule tpack_05c_m2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "T-PACK"
		version = "0.5c -m2"
	strings:
		$1 = { 68 ?? ?? FD 60 BE ?? ?? BF ?? ?? B9 ?? ?? F3 A4 8B F7 BF ?? ?? FC 46 E9 CE FD }
	condition:
		$1 at pe.entry_point
}

rule taishanziangyu_locksoft_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "TaiShanXiangYu LockSoft"
		version = "1.0"
	strings:
		$1 = { E8 03 00 00 00 EB 01 ?? BB 55 00 00 00 E8 03 00 00 00 EB 01 ?? E8 8F 00 00 00 E8 03 00 00 00 EB 01 ?? E8 82 00 00 00 E8 03 00 00 00 EB 01 ?? E8 B8 00 00 00 E8 03 00 00 00 EB 01 ?? E8 AB 00 00 00 E8 03 00 00 00 EB 01 ?? 83 FB 55 E8 03 00 00 00 EB 01 ?? 75 2E E8 03 00 00 00 EB 01 ?? C3 60 E8 00 00 00 00 5D 81 ED E3 00 42 00 8B D5 81 C2 32 01 42 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 4B CC C3 E8 03 00 00 00 EB 01 ?? 33 DB B9 ?? ?? ?? ?? 81 ?? ?? ?? ?? ?? 8B D5 81 ?? ?? ?? ?? ?? 8D 3A 8B F7 33 C0 E8 03 00 00 00 EB 01 ?? E8 17 00 00 00 ?? ?? ?? E9 ?? ?? ?? ?? 33 C0 64 FF 30 64 89 20 43 CC C3 }
	condition:
		$1 at pe.entry_point
}

rule taishanziangyu_locksoft_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "TaiShanXiangYu LockSoft"
		version = "1.0"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED E3 00 42 00 8B D5 81 C2 32 01 42 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 4B CC C3 E8 03 00 00 00 EB 01 ?? 33 DB B9 AF 28 42 00 81 E9 DD 01 42 00 8B D5 81 C2 DD 01 42 00 8D 3A 8B F7 33 C0 E8 03 00 00 00 EB 01 ?? E8 17 00 00 00 90 90 90 E9 78 20 00 00 33 C0 64 FF 30 64 89 20 43 CC C3 90 EB 01 ?? AC }
	condition:
		$1 at pe.entry_point
}

rule tarma_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "TARMA"
	strings:
		$1 = { 54 49 5A 31 }
	condition:
		$1 at pe.entry_point
}

rule telock_041x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.41x"
	strings:
		$1 = { 66 8B C0 8D 24 24 EB 01 EB 60 EB 01 EB 9C E8 00 00 00 00 5E 83 C6 50 8B FE 68 78 01 ?? ?? 59 EB 01 EB AC 54 E8 03 ?? ?? ?? 5C EB 08 }
	condition:
		$1 at pe.entry_point
}

rule telock_042 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.42"
	strings:
		$1 = { C1 EE 00 66 8B C9 EB 01 EB 60 EB 01 EB 9C E8 00 00 00 00 5E 83 C6 52 8B FE 68 79 01 59 EB 01 EB AC 54 E8 03 5C EB 08 }
	condition:
		$1 at pe.entry_point
}

rule telock_051_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.51"
	strings:
		$1 = { C1 EE 00 66 8B C9 EB 01 EB 60 EB 01 EB 9C E8 00 00 00 00 5E 83 C6 5E 8B FE 68 79 01 00 00 59 EB 01 EB AC 54 E8 03 00 00 00 5C EB 08 8D 64 24 04 FF 64 24 FC 6A 05 D0 2C 24 72 01 E8 01 24 24 5C F7 DC EB 02 CD 20 8D 64 24 FE F7 DC EB 02 CD 20 FE C8 E8 00 00 00 00 32 C1 EB 02 82 0D AA EB 03 82 0D 58 EB 02 1D 7A 49 EB 05 E8 01 00 00 00 7F AE 14 7E A0 77 76 75 74 }
	condition:
		$1 at pe.entry_point
}

rule telock_051_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.51"
	strings:
		$1 = { C1 EE 00 66 8B C9 EB 01 EB 60 EB 01 EB 9C E8 00 00 00 00 5E 83 C6 5E 8B FE 68 79 01 59 EB 01 EB AC 54 E8 03 5C EB 08 }
	condition:
		$1 at pe.entry_point
}

rule telock_060 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.60"
	strings:
		$1 = { E9 00 00 00 00 60 E8 00 00 00 00 58 83 C0 08 }
	condition:
		$1 at pe.entry_point
}

rule telock_070 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.70"
	strings:
		$1 = { 60 E8 BD 10 00 00 C3 83 E2 00 F9 75 FA 70 }
	condition:
		$1 at pe.entry_point
}

rule telock_071 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.71"
	strings:
		$1 = { 60 E8 ED 10 00 00 C3 83 }
	condition:
		$1 at pe.entry_point
}

rule telock_071b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.71b2"
	strings:
		$1 = { 60 E8 44 11 00 00 C3 83 }
	condition:
		$1 at pe.entry_point
}

rule telock_071b7 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.71b7"
	strings:
		$1 = { 60 E8 48 11 00 00 C3 83 }
	condition:
		$1 at pe.entry_point
}

rule telock_07x_084 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.7x - 0.84"
	strings:
		$1 = { 60 E8 00 00 C3 83 }
	condition:
		$1 at pe.entry_point
}

rule telock_80 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.80"
	strings:
		$1 = { 60 E8 F9 11 00 00 C3 83 }
	condition:
		$1 at pe.entry_point
}

rule telock_085f {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.85f"
	strings:
		$1 = { 60 E8 02 00 00 00 CD 20 E8 00 00 00 00 5E 2B C9 58 74 02 }
	condition:
		$1 at pe.entry_point
}

rule telock_090 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.90"
	strings:
		$1 = { ?? ?? E8 02 00 00 00 E8 00 E8 00 00 00 00 5E 2B }
	condition:
		$1 at pe.entry_point
}

rule telock_092a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.92a"
	strings:
		$1 = { E9 7E E9 FF FF 00 }
	condition:
		$1 at pe.entry_point
}

rule telock_095 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.95"
	strings:
		$1 = { E9 D5 E4 FF FF 00 }
	condition:
		$1 at pe.entry_point
}

rule telock_096 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.96"
	strings:
		$1 = { E9 59 E4 FF FF 00 00 00 00 00 00 00 ?? ?? ?? ?? EE ?? ?? 00 00 00 00 00 00 00 00 00 0E ?? ?? 00 FE ?? ?? 00 F6 ?? ?? 00 00 00 00 00 00 00 00 00 1B ?? ?? 00 06 ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 26 ?? ?? 00 00 00 00 00 39 ?? ?? 00 00 00 00 00 26 ?? ?? 00 00 00 00 00 39 ?? ?? 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C }
	condition:
		$1 at pe.entry_point
}

rule telock_098_special_build {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.98 special build"
	strings:
		$1 = { E9 99 D7 FF FF 00 00 00 ?? ?? ?? ?? AA ?? ?? 00 00 00 00 00 00 00 00 00 CA }
	condition:
		$1 at pe.entry_point
}

rule telock_098 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.98"
	strings:
		$1 = { E9 25 E4 FF FF 00 00 00 ?? ?? ?? ?? 1E ?? ?? 00 00 00 00 00 00 00 00 00 3E ?? ?? 00 2E ?? ?? 00 26 ?? ?? 00 00 00 00 00 00 00 00 00 4B ?? ?? 00 36 ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 56 ?? ?? 00 00 00 00 00 69 ?? ?? 00 00 00 00 00 56 ?? ?? 00 00 00 00 00 69 ?? ?? 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 75 73 65 }
	condition:
		$1 at pe.entry_point
}

rule telock_098b2 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.98b2"
	strings:
		$1 = { E9 1B E4 FF FF }
	condition:
		$1 at pe.entry_point
}

rule telock_099_special_build {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.99 special build"
	strings:
		$1 = { E9 5E DF FF FF 00 00 00 ?? ?? ?? ?? E5 ?? ?? 00 00 00 00 00 00 00 00 00 05 ?? ?? 00 F5 ?? ?? 00 ED ?? ?? 00 00 00 00 00 00 00 00 00 12 ?? ?? 00 FD ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 1D ?? ?? 00 00 00 00 00 30 ?? ?? 00 00 00 00 00 1D ?? ?? 00 00 00 00 00 30 ?? ?? 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule telock_099 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.99"
	strings:
		$1 = { E9 5E DF FF FF 00 00 00 ?? ?? ?? ?? E5 ?? ?? 00 00 00 00 00 00 00 00 00 05 }
	condition:
		$1 at pe.entry_point
}

rule telock_099c_private_eclipse {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "0.99c private ECLIPSE"
	strings:
		$1 = { E9 3F DF FF FF 00 00 00 ?? ?? ?? ?? 04 ?? ?? 00 00 00 00 00 00 00 00 00 24 ?? ?? 00 14 ?? ?? 00 0C ?? ?? 00 00 00 00 00 00 00 00 00 31 ?? ?? 00 1C ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 3C ?? ?? 00 00 00 00 00 4F ?? ?? 00 00 00 00 00 3C ?? ?? 00 00 00 00 00 4F ?? ?? 00 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 75 73 65 }
	condition:
		$1 at pe.entry_point
}

rule telock_100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "tElock"
		version = "1.00"
	strings:
		$1 = { E9 E5 E2 FF FF }
	condition:
		$1 at pe.entry_point
}

rule themida_1000_1800 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Themida"
		version = "1.0.0.0 - 1.8.0.0"
	strings:
		$1 = { B8 00 00 ?? ?? 60 0B C0 74 58 E8 ?? ?? ?? ?? 58 05 ?? 00 00 00 80 38 E9 ?? ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule themida_10x_18x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Themida"
		version = "1.0.x - 1.8.x"
	strings:
		$1 = { B8 ?? ?? ?? ?? 60 0B C0 74 58 E8 00 00 00 00 58 05 ?? ?? ?? ?? 80 38 E9 75 03 61 EB 35 E8 00 00 00 00 58 25 00 F0 FF FF 33 FF 66 BB ?? ?? 66 83 ?? ?? 66 39 18 75 12 0F B7 50 3C 03 D0 BB ?? ?? ?? ?? 83 C3 ?? 39 1A 74 07 2D 00 10 00 00 EB DA 8B F8 B8 ?? ?? ?? ?? 03 C7 B9 ?? ?? ?? ?? 03 CF EB 0A B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? 50 51 E8 84 00 00 00 E8 00 00 00 00 58 2D ?? ?? ?? ?? B9 ?? ?? ?? ?? C6 00 E9 83 E9 ?? 89 48 01 61 E9 }
	condition:
		$1 at pe.entry_point
}

rule themida_10x_10x_no_comp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Themida"
		version = "1.0.x - 1.8.x no compression"
	strings:
		$1 = { 55 8B EC 83 C4 D8 60 E8 00 00 00 00 5A 81 EA ?? ?? ?? ?? 8B DA C7 45 D8 00 00 00 00 8B 45 D8 40 89 45 D8 81 7D D8 80 00 00 00 74 0F 8B 45 08 89 83 ?? ?? ?? ?? FF 45 08 43 EB E1 89 45 DC 61 8B 45 DC C9 C2 04 00 55 8B EC 81 C4 7C FF FF FF 60 E8 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule themida_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Themida"
		version = "1.x"
	strings:
		$1 = { 8B C5 8B D4 60 E8 00 00 00 00 5D 81 ED ?? ?? ?? ?? 89 95 ?? ?? ?? ?? 89 B5 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? 83 BD ?? ?? ?? ?? ?? 74 0C 8B E8 8B E2 B8 01 00 00 00 C2 0C 00 8B 44 24 24 89 85 ?? ?? ?? ?? 6A 45 E8 A3 00 00 00 68 9A 74 83 07 E8 DF 00 00 00 68 25 4B 89 0A E8 D5 00 00 00 E9 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule themida_1802_winlicense {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Themida"
		version = "1.8.0.2 or higher WinLicense"
	strings:
		$1 = { B8 00 00 ?? ?? 60 0B C0 74 68 E8 ?? ?? ?? ?? 58 05 ?? 00 00 00 80 38 E9 ?? ?? ?? ?? ?? DB 2D ?? ?? ?? ?? ?? ?? ?? FF FF FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule themida_18x_2x_winlicense {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Themida"
		version = "1.8.x - 2.x WinLicense"
	strings:
		$1 = { B8 ?? ?? ?? ?? 60 0B C0 74 68 E8 00 00 00 00 58 05 53 00 00 00 80 38 E9 75 13 61 EB 45 DB 2D ?? ?? ?? ?? FF FF FF FF FF FF FF FF 3D ?? ?? ?? ?? 00 00 58 25 00 F0 FF FF 33 FF 66 BB ?? ?? 66 83 ?? ?? 66 39 18 75 12 0F B7 50 3C 03 D0 BB ?? ?? ?? ?? 83 C3 ?? 39 1A 74 07 2D ?? ?? ?? ?? EB DA 8B F8 B8 ?? ?? ?? ?? 03 C7 B9 ?? ?? ?? ?? 03 CF EB 0A B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? 50 51 E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 58 }
	condition:
		$1 at pe.entry_point
}

rule thewrap_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "theWRAP"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 56 57 33 C0 89 45 F0 B8 48 D2 4B 00 E8 BC 87 F4 FF BB 04 0B 4D 00 33 C0 55 68 E8 D5 4B 00 64 FF 30 64 89 20 E8 9C F4 FF FF E8 F7 FB FF FF 6A 40 8D 55 F0 A1 F0 ED 4B 00 8B 00 E8 42 2E F7 FF 8B 4D F0 B2 01 A1 F4 C2 40 00 E8 F7 20 F5 FF 8B F0 B2 01 A1 B4 C3 40 00 E8 F1 5B F4 FF 89 03 33 D2 8B 03 E8 42 1E F5 FF 66 B9 02 00 BA FC FF FF FF 8B C6 8B 38 FF 57 0C BA B8 A7 4D 00 B9 04 00 00 00 8B C6 8B 38 FF 57 04 83 3D B8 A7 4D 00 00 0F 84 5E 01 00 00 8B 15 B8 A7 4D 00 83 C2 04 F7 DA 66 B9 02 00 8B C6 8B 38 FF 57 0C 8B 0D B8 A7 4D 00 8B D6 8B 03 E8 2B 1F F5 FF 8B C6 E8 B4 5B F4 FF 33 D2 8B 03 E8 DF 1D F5 FF BA F0 44 4E 00 B9 01 00 00 00 8B 03 8B 30 FF 56 04 80 3D F0 44 4E 00 0A 75 3F BA B8 A7 4D 00 B9 04 00 00 00 8B 03 8B 30 FF 56 04 8B 15 B8 A7 }
	condition:
		$1 at pe.entry_point
}

rule thunderbolt_002 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Thunderbolt"
		version = "0.02"
	strings:
		$1 = { E9 00 00 00 00 60 E8 14 00 00 00 5D 81 ED 00 00 00 00 6A 45 E8 A3 00 00 00 68 00 00 00 00 E8 58 61 E8 AA 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 5D 68 00 FE 9F 07 53 E8 5D 00 00 00 EB FF 71 E8 C2 50 00 EB D6 5E F3 68 89 74 24 48 74 24 58 FF 8D 74 24 58 5E 83 C6 4C 75 F4 59 8D 71 E8 75 09 81 F6 EB FF 51 B9 01 00 83 EE FC 49 FF 71 C7 75 19 8B 74 24 00 00 81 36 50 56 8B 36 EB FF 77 C4 36 81 F6 EB 87 34 24 8B 8B 1C 24 83 EC FC EB 01 E8 83 EC FC E9 E7 00 00 00 5B EB FF F3 EB FF C3 }
	condition:
		$1 at pe.entry_point
}

rule tpav_cryptor_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "TPAV Cryptor"
		version = "1.1"
	strings:
		$1 = { 8D 85 08 FF FF FF 50 8D 85 C4 FE FF FF 50 6A 00 6A 00 6A 04 6A 00 6A 00 6A 00 8D 95 C0 FE FF FF 33 C0 E8 ?? ?? FF FF 8B 85 C0 FE FF FF E8 ?? ?? FF FF 50 6A 00 FF 15 2C ?? ?? 70 }
	condition:
		$1 at pe.entry_point
}

rule tpppack_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "TPPpack"
	strings:
		$1 = { E8 00 00 00 00 5D 81 ED F5 8F 40 00 60 33 ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule trainer_creation_kit_5 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Trainer Creation Kit"
		version = "5"
	strings:
		$1 = { 6A 00 68 80 00 00 00 6A 02 6A 00 6A 00 68 00 00 00 40 68 25 45 40 00 E8 3C 02 00 00 50 6A 00 68 40 45 40 00 68 00 10 00 00 68 00 30 40 00 50 E8 54 02 00 00 58 50 E8 17 02 00 00 6A 00 E8 2E 02 00 00 A3 70 45 40 00 68 25 45 40 00 E8 2B 02 00 00 A3 30 45 40 }
	condition:
		$1
}

rule trivial173_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Trivial173"
	strings:
		$1 = { EB ?? ?? 28 54 72 69 76 69 61 6C 31 37 33 20 62 79 20 53 4D 54 2F 53 4D 46 29 }
	condition:
		$1 at pe.entry_point
}

rule ug2002_cruncher_03b3 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UG2002 Cruncher"
		version = "0.3b3"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 5D 81 ED ?? ?? ?? ?? E8 0D ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 58 }
	condition:
		$1 at pe.entry_point
}

rule ultrapro_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UltraPro"
		version = "1.0"
	strings:
		$1 = { A1 ?? ?? ?? ?? 85 C0 0F 85 3B 06 00 00 55 56 C7 05 ?? ?? ?? ?? 01 00 00 00 FF 15 }
	condition:
		$1 at pe.entry_point
}

rule underdround_crypter_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UnderGround Crypter"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 74 3C 00 11 E8 94 F9 FF FF E8 BF FE FF FF E8 0A F3 FF FF 8B C0 }
	condition:
		$1 at pe.entry_point
}

rule unicops_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNICOPS"
	strings:
		$1 = { 68 F1 36 AD B6 87 1C 24 60 E8 00 00 00 00 5F 8D B7 EA F7 FF FF 81 C7 32 00 00 00 8B 0E 8A D1 83 C6 04 C1 E9 08 74 0B 8A 07 32 C3 2A F8 AA D3 D3 E2 F5 80 FA 00 74 07 01 1F 83 C7 04 EB DD 61 5B }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 56 33 C0 89 45 ?? ?? ?? ?? 40 00 E8 11 F4 FF FF BE 30 6B 40 00 33 C0 55 68 C9 42 40 00 64 FF 30 64 89 20 E8 C9 FA FF FF BA D8 42 40 00 8B ?? ?? ?? ?? FF FF 8B D8 B8 28 6B 40 00 8B 16 E8 37 F0 FF FF B8 2C 6B 40 00 8B 16 E8 2B F0 FF FF B8 28 6B 40 00 E8 19 F0 FF FF 8B D0 8B C3 8B 0E E8 42 E3 FF FF BA DC 42 40 00 8B C6 E8 2A FA FF FF 8B D8 B8 20 6B 40 00 8B 16 E8 FC EF FF FF B8 24 6B 40 00 8B 16 E8 F0 EF FF FF B8 20 6B 40 00 E8 DE EF FF FF 8B D0 8B C3 8B 0E E8 07 E3 FF FF 6A 00 6A 19 6A 00 6A 32 A1 28 6B 40 00 E8 59 EF FF FF 83 E8 05 03 C0 8D 55 EC E8 94 FE FF FF 8B 55 EC B9 24 6B 40 00 A1 20 6B 40 00 E8 E2 F6 FF FF 6A 00 6A 19 6A 00 6A 32 }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_11c {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "1.1c"
	strings:
		$1 = { 55 8B EC 83 C4 E4 53 56 33 C0 89 45 E4 89 45 E8 89 45 EC B8 C0 47 00 10 E8 4F F3 FF FF BE 5C 67 00 10 33 C0 55 68 D2 4A 00 10 64 FF 30 64 89 20 E8 EB DE FF FF E8 C6 F8 FF FF BA E0 4A 00 10 B8 CC 67 00 10 E8 5F F8 FF FF 8B D8 8B D6 8B C3 8B 0D CC 67 00 10 E8 3A DD FF FF 8B 46 50 8B D0 B8 D4 67 00 10 E8 5B EF FF FF B8 D4 67 00 10 E8 09 EF FF FF 8B D0 8D 46 14 8B 4E 50 E8 14 DD FF FF 8B 46 48 8B D0 B8 D8 67 00 ?? ?? ?? ?? ?? FF B8 D8 67 00 10 E8 E3 EE FF FF 8B D0 8B C6 8B 4E 48 E8 EF DC FF FF FF 76 5C FF 76 58 FF 76 64 FF 76 60 B9 D4 67 00 10 8B 15 D8 67 00 10 A1 D4 67 00 10 E8 76 F6 FF FF A1 D4 67 00 10 E8 5C EE FF FF 8B D0 B8 CC 67 00 10 E8 CC F7 FF FF 8B D8 B8 DC 67 00 10 }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_12b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "1.2b"
	strings:
		$1 = { 55 8B EC 83 C4 D8 53 56 57 33 C0 89 45 D8 89 45 DC 89 45 E0 89 45 E4 89 45 E8 B8 70 3A 40 00 E8 C4 EC FF FF 33 C0 55 68 5C 3F 40 00 64 FF 30 64 89 20 E8 C5 D7 FF FF E8 5C F5 FF FF B8 20 65 40 00 33 C9 BA 04 01 00 00 E8 D3 DB FF FF 68 04 01 00 00 68 20 65 40 00 6A 00 FF 15 10 55 40 00 BA 6C 3F 40 00 B8 14 55 40 00 E8 5A F4 FF FF 85 C0 0F 84 1B 04 00 00 BA 18 55 40 00 8B 0D 14 55 40 00 E8 16 D7 FF FF 8B 05 88 61 40 00 8B D0 B8 54 62 40 00 E8 D4 E3 FF FF B8 54 62 40 00 E8 F2 E2 FF FF 8B D0 B8 18 55 40 00 8B 0D 88 61 40 00 E8 E8 D6 FF FF FF 35 34 62 40 00 FF 35 30 62 40 00 FF 35 3C 62 40 00 FF 35 38 62 40 00 8D 55 E8 A1 88 61 40 00 E8 E3 F0 FF FF 8B 55 E8 }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_12c_12d {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "1.2c, 1.2d"
	strings:
		$1 = { 55 8B EC B9 05 00 00 00 6A 00 6A 00 49 75 F9 51 53 56 57 B8 ?? 3A ?? ?? E8 ?? EC FF FF 33 C0 55 68 ?? ?? ?? ?? 64 FF 30 64 89 20 E8 ?? D7 FF FF E8 ?? ?? FF FF B8 20 ?? ?? ?? 33 C9 BA 04 01 00 00 E8 ?? DB FF FF 68 04 01 00 00 68 20 ?? ?? ?? 6A 00 FF 15 10 ?? ?? ?? BA ?? ?? ?? ?? B8 14 ?? ?? ?? E8 ?? ?? FF FF 85 C0 0F 84 ?? 04 00 00 BA 18 ?? ?? ?? 8B 0D 14 ?? ?? ?? E8 ?? ?? FF FF 8B 05 88 ?? ?? ?? 8B D0 B8 54 ?? ?? ?? E8 ?? E3 FF FF B8 54 ?? ?? ?? E8 ?? E2 FF FF 8B D0 B8 18 ?? ?? ?? 8B 0D 88 ?? ?? ?? E8 ?? D6 FF FF FF 35 34 ?? ?? ?? FF 35 30 ?? ?? ?? FF 35 3C ?? ?? ?? FF 35 38 ?? ?? ?? 8D 55 E8 A1 88 ?? ?? ?? E8 ?? F0 FF FF 8B 55 E8 B9 54 }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_13b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "1.3b"
	strings:
		$1 = { 55 8B EC B9 08 00 00 00 6A 00 6A 00 49 75 F9 53 56 57 B8 98 56 00 10 E8 48 EB FF FF 33 C0 55 68 AC 5D 00 10 64 FF 30 64 89 20 6A 00 68 BC 5D 00 10 68 C4 5D 00 10 6A 00 E8 23 EC FF FF E8 C6 CE FF FF 6A 00 68 BC 5D 00 10 68 ?? ?? ?? ?? 6A 00 E8 0B EC FF FF E8 F2 F4 FF FF B8 08 BC 00 10 33 C9 BA 04 01 00 00 E8 C1 D2 FF FF 6A 00 68 BC 5D 00 10 68 E4 5D 00 10 6A 00 E8 E2 EB FF FF 68 04 01 00 00 68 08 BC 00 10 6A 00 FF 15 68 77 00 10 6A 00 68 BC 5D 00 10 68 FC 5D 00 10 6A 00 E8 BD EB FF FF BA 10 5E 00 10 B8 70 77 00 10 E8 CA F3 FF FF 85 C0 0F 84 F7 05 00 00 BA 74 77 00 10 8B 0D 70 77 00 10 E8 FE CD FF FF 6A 00 }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "2.0"
	strings:
		$1 = { 55 8B EC B9 0A 00 00 00 6A 00 6A 00 49 75 F9 53 56 57 B8 1C 2F 40 00 E8 C8 F1 FF FF 33 C0 55 68 FB 33 40 00 64 FF 30 64 89 20 BA 0C 34 40 00 B8 E4 54 40 00 E8 EF FE FF FF 8B D8 85 DB 75 07 6A 00 E8 5A F2 FF FF BA E8 54 40 00 8B C3 8B 0D E4 54 40 00 E8 74 E2 FF FF C7 05 20 6B 40 00 09 00 00 00 BB 98 69 40 00 C7 45 EC E8 54 40 00 C7 45 E8 31 57 40 00 C7 45 E4 43 60 40 00 BE D3 6A 40 00 BF E0 6A 40 00 83 7B 04 00 75 0B 83 3B 00 0F 86 AA 03 00 00 EB 06 0F 8E A2 03 00 00 8B 03 8B D0 B8 0C 6B 40 00 E8 C1 EE FF FF B8 0C 6B 40 00 E8 6F EE FF FF 8B D0 8B 45 EC 8B 0B E8 0B E2 FF FF 6A 00 6A 1E 6A 00 6A 2C A1 0C 6B 40 00 E8 25 ED FF FF 8D 55 E0 E8 15 FE FF FF 8B 55 E0 B9 10 6B 40 00 A1 0C 6B 40 00 }
	condition:
		$1 at pe.entry_point
}
rule unnamed_scrambler_211 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "2.1.1"
	strings:
		$1 = { 55 8B EC B9 15 00 00 00 6A 00 6A 00 49 75 F9 53 56 57 B8 ?? 3A ?? ?? E8 ?? EE FF FF 33 C0 55 68 ?? 43 ?? ?? 64 FF 30 64 89 20 BA ?? 43 ?? ?? B8 E4 64 ?? ?? E8 0F FD FF FF 8B D8 85 DB 75 07 6A 00 E8 ?? EE FF FF BA E8 64 ?? ?? 8B C3 8B 0D E4 64 ?? ?? E8 ?? D7 FF FF B8 F8 ?? ?? ?? BA 04 00 00 00 E8 ?? EF FF FF 33 C0 A3 F8 ?? ?? ?? BB ?? ?? ?? ?? C7 45 EC E8 64 ?? ?? C7 45 E8 ?? ?? ?? ?? C7 45 E4 ?? ?? ?? ?? BE ?? ?? ?? ?? BF ?? ?? ?? ?? B8 E0 ?? ?? ?? BA 04 00 00 00 E8 ?? EF FF FF 68 F4 01 00 00 E8 ?? EE FF FF 83 7B 04 00 75 0B 83 3B 00 0F 86 ?? 07 00 00 EB 06 0F 8E ?? 07 00 00 8B 03 8B D0 B8 E4 ?? ?? ?? E8 ?? E5 FF FF B8 E4 ?? ?? ?? E8 ?? E3 FF FF 8B D0 8B 45 EC 8B 0B E8 }
	condition:
		$1 at pe.entry_point
}
rule unnamed_scrambler_252 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "2.5.2"
	strings:
		$1 = { 55 8B EC B9 ?? 00 00 00 6A 00 6A 00 49 75 F9 53 56 57 B8 ?? ?? 40 00 E8 ?? EA FF FF 33 C0 55 68 ?? ?? 40 00 64 FF 30 64 89 20 BA ?? ?? 40 00 B8 ?? ?? 40 00 E8 63 F3 FF FF 8B D8 85 DB 75 07 6A 00 E8 ?? ?? FF FF BA ?? ?? 40 00 8B C3 8B 0D ?? ?? 40 00 E8 ?? ?? FF FF C7 05 ?? ?? 40 00 0A 00 00 00 BB ?? ?? 40 00 BE ?? ?? 40 00 BF ?? ?? 40 00 B8 ?? ?? 40 00 BA 04 00 00 00 E8 ?? EB FF FF 83 3B 00 74 04 33 C0 89 03 8B D7 8B C6 E8 0A F3 FF FF 89 03 83 3B 00 0F 84 F7 04 00 00 B8 ?? ?? 40 00 8B 16 E8 ?? E1 FF FF B8 ?? ?? 40 00 E8 ?? E0 FF FF 8B D0 8B 03 8B 0E E8 ?? ?? FF FF 8B C7 A3 ?? ?? 40 00 8D 55 EC 33 C0 E8 ?? D3 FF FF 8B 45 EC B9 ?? ?? 40 00 BA ?? ?? 40 00 E8 8B ED FF FF 3C 01 75 2B A1 }
	condition:
		$1 at pe.entry_point
}

rule unnamed_scrambler_25a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Unnamed Scrambler"
		version = "2.5a"
	strings:
		$1 = { 55 8B EC B9 0B 00 00 00 6A 00 6A 00 49 75 F9 51 53 56 57 B8 6C 3E 40 00 E8 F7 EA FF FF 33 C0 55 68 60 44 40 00 64 FF 30 64 89 20 BA 70 44 40 00 B8 B8 6C 40 00 E8 62 F3 FF FF 8B D8 85 DB 75 07 6A 00 E8 A1 EB FF FF BA E8 64 40 00 8B C3 8B 0D B8 6C 40 00 E8 37 D3 FF FF C7 05 BC 6C 40 00 0A 00 00 00 BB 68 6C 40 00 BE 90 6C 40 00 BF E8 64 40 00 B8 C0 6C 40 00 BA 04 00 00 00 E8 07 EC FF FF 83 3B 00 74 04 33 C0 89 03 8B D7 8B C6 E8 09 F3 FF FF 89 03 83 3B 00 0F 84 BB 04 00 00 B8 C0 6C 40 00 8B 16 E8 06 E2 FF FF B8 C0 6C 40 00 E8 24 E1 FF FF 8B D0 8B 03 8B 0E E8 D1 D2 FF FF 8B C7 A3 20 6E 40 00 8D 55 EC 33 C0 E8 0C D4 FF FF 8B 45 EC B9 1C 6E 40 00 BA 18 6E 40 00 }
	condition:
		$1 at pe.entry_point
}

rule unopix_075 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UnoPiX"
		version = "0.75"
	strings:
		$1 = { 60 E8 07 00 00 00 61 68 ?? ?? 40 00 C3 83 04 24 18 C3 20 83 B8 ED 20 37 EF C6 B9 79 37 9E 61 }
	condition:
		$1 at pe.entry_point
}

rule unopix_103_110 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UnoPiX"
		version = "1.03 - 1.10"
	strings:
		$1 = { 83 EC 04 C7 04 24 00 ?? ?? ?? C3 00 ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? 00 10 00 00 00 02 00 00 01 00 00 00 00 00 00 00 04 00 00 00 00 00 00 00 00 ?? ?? 00 00 10 00 00 00 00 00 00 02 00 00 ?? 00 00 ?? 00 00 ?? ?? 00 00 00 10 00 00 10 00 00 00 00 00 00 10 }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { 55 89 E5 83 EC 14 6A 01 FF 15 ?? ?? ?? ?? E8 ?? ?? ?? ?? 90 55 89 E5 53 83 EC 04 8B 45 08 8B 00 8B 00 3D 91 00 00 C0 77 37 3D 8D 00 00 C0 72 48 BB 01 00 00 00 83 EC 08 6A 00 6A 08 E8 }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { 60 E9 C5 00 00 00 60 8B 74 24 24 8B 7C 24 28 FC B2 80 33 DB A4 B3 02 E8 6D 00 00 00 73 F6 33 C9 E8 64 00 00 00 73 1C 33 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 12 C0 73 F7 75 3F }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { 9C 50 51 52 53 54 55 56 57 83 BC E4 2C 00 00 00 01 0F 85 8A 01 00 00 E8 00 00 00 00 5E 81 EE 65 00 00 00 89 F7 81 EF ?? ?? ?? ?? 89 F1 8B 09 01 F9 FF 31 68 ?? ?? ?? ?? B9 ?? ?? ?? ?? 01 F9 51 }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { 9C 60 60 E8 00 00 00 00 5E 81 C6 ?? ?? 00 00 56 64 67 FF 36 00 00 64 67 89 26 00 00 EA ?? ?? ?? ?? C3 E8 01 00 00 00 69 83 C4 04 FA E8 01 00 00 00 8B 83 C4 04 F0 0F C7 C8 EB 03 C7 84 8B 55 58 }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { 9C 68 ?? ?? 00 00 7? 1? 81 04 24 ?? ?? ?? ?? 90 81 04 24 C3 }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { 9C 68 ?? ?? 00 00 7? 1? 81 04 24 1F ?? ?? ?? ?? ?? 81 04 24 C2 00 00 }
	condition:
		$1 at pe.entry_point
}

rule unsorted_packer_uv_07 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UNSORTED PACKER"
	strings:
		$1 = { FC B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? 81 F9 ?? ?? ?? ?? 75 06 81 C1 27 00 00 00 30 01 C1 C0 03 41 81 F9 ?? ?? ?? ?? 75 E4 }
	condition:
		$1 at pe.entry_point
}

rule upack_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
	strings:
		$1 = { 81 3A 00 00 00 02 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upack_010_011 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.10 - 0.11"
	strings:
		$1 = { BE ?? ?? ?? ?? AD 8B F8 95 A5 33 C0 33 C9 AB 48 AB F7 D8 B1 ?? F3 AB C1 E0 ?? B5 ?? F3 AB AD 50 97 51 AD 87 F5 58 8D 54 86 5C FF D5 72 5A 2C ?? 73 ?? B0 ?? 3C ?? 72 02 2C ?? 50 0F B6 5F FF C1 E3 ?? B3 ?? 8D 1C 5B 8D ?? ?? ?? ?? ?? ?? B0 ?? 67 E3 29 8B D7 2B 56 0C 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 14 93 FF D5 }
	condition:
		$1 at pe.entry_point
}

rule upack_010_012 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.10 - 0.12"
	strings:
		$1 = { BE ?? ?? ?? ?? AD 8B F8 95 A5 33 C0 33 C9 AB 48 AB F7 D8 B1 04 F3 AB C1 E0 0A B5 ?? F3 AB AD 50 97 51 AD 87 F5 58 8D 54 86 5C FF D5 72 5A 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 E3 ?? B3 00 8D 1C 5B 8D 9C 9E 0C 10 00 00 B0 01 67 E3 29 8B D7 2B 56 0C 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 14 93 FF D5 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E4 B4 00 33 C9 B5 01 FF 55 CC 33 C9 E9 DF 00 00 00 8B 5E 0C 83 C2 30 FF D5 73 50 83 C2 30 FF D5 72 1B 83 C2 30 FF D5 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 46 0C B1 80 8A 00 EB CF 83 C2 60 FF D5 87 5E 10 73 0D 83 C2 30 FF D5 87 5E 14 73 03 87 5E 18 3C 07 B0 08 72 02 B0 0B 50 53 8D 96 7C 07 00 00 FF 55 D0 5B 91 EB 77 3C 07 B0 07 72 02 B0 0A 50 87 5E 10 87 5E 14 89 5E 18 8D 96 C4 0B 00 00 FF 55 D0 }
	condition:
		$1 at pe.entry_point
}

rule upack_011 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPack"
		version = "0.11"
	strings:
		$1 = { BE 48 01 40 00 AD 8B F8 95 A5 33 C0 33 C9 AB 48 AB F7 D8 B1 04 F3 AB C1 E0 0A B5 1C F3 AB AD 50 97 51 AD 87 F5 58 8D 54 86 5C FF D5 72 5A 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 E3 03 B3 00 8D 1C 5B 8D 9C 9E 0C 10 00 00 B0 01 67 E3 29 8B D7 }
	condition:
		$1 at pe.entry_point
}

rule upack_012b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.12b"
	strings:
		$1 = { BE 48 01 40 00 AD ?? ?? ?? A5 ?? C0 33 C9 ?? ?? ?? ?? ?? ?? ?? F3 AB ?? ?? 0A ?? ?? ?? ?? AD 50 97 51 ?? 87 F5 58 8D 54 86 5C ?? D5 72 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? B6 5F FF C1 }
	condition:
		$1 at pe.entry_point
}

rule upack_02b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.2b"
	strings:
		$1 = { BE 88 01 ?? ?? AD 8B F8 95 A5 33 C0 33 }
	condition:
		$1 at pe.entry_point
}

rule upack_020_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.20"
	strings:
		$1 = { BE ?? ?? ?? ?? AD 8B F8 95 A5 33 C0 33 C9 AB 48 AB F7 D8 B1 04 F3 AB C1 E0 0A ?? ?? F3 AB AD 50 97 51 58 8D 54 85 5C FF 16 72 5A 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 ?? ?? B3 00 8D 1C 5B 8D 9C 9D 0C 10 00 00 B0 01 67 E3 29 8B D7 2B 55 0C 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 14 93 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E4 B4 00 33 C9 B5 01 FF 56 08 33 C9 E9 07 01 00 00 8B 5D 0C 83 C2 30 FF 16 73 53 83 C2 30 FF 16 72 1B 83 C2 30 FF 16 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 45 0C B1 80 8A 00 EB CF 83 C2 60 FF 16 87 5D 10 73 0D 83 C2 30 FF 16 87 5D 14 73 03 87 5D 18 3C 07 B0 08 72 02 B0 0B 50 53 8D 95 7C 07 00 00 FF 56 0C 5B 91 E9 9C 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upack_020_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.20"
	strings:
		$1 = { E9 06 02 00 00 33 C9 5E 87 0E E3 F4 2B F1 8B DE AD 2B D8 AD 03 C3 50 97 AD 91 F3 A5 5E AD 56 91 01 1E AD E2 FB }
	condition:
		$1 at pe.entry_point
}

rule upack_021 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.21"
	strings:
		$1 = { BE ?? ?? ?? ?? AD 8B F8 ?? ?? ?? ?? 33 C0 AB 48 AB F7 D8 59 F3 AB C1 E0 0A ?? ?? F3 AB AD 50 97 51 58 8D 54 85 5C FF 16 72 5A 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF ?? ?? ?? B3 00 8D 1C 5B 8D 9C 9D 0C 10 00 00 B0 01 67 E3 29 8B D7 2B 55 0C 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 14 93 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E4 B4 00 33 C9 B5 01 FF 56 08 33 C9 E9 07 01 00 00 8B 5D 0C 83 C2 30 FF 16 73 53 83 C2 30 FF 16 72 1B 83 C2 30 FF 16 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 45 0C B1 80 8A 00 EB CF 83 C2 60 FF 16 87 5D 10 73 0D 83 C2 30 FF 16 87 5D 14 73 03 87 5D 18 3C 07 B0 08 72 02 B0 0B 50 53 8D 95 7C 07 00 00 FF 56 0C 5B 91 E9 9C 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upack_022b_023b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.22b - 0.23b"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? AD 8B F8 59 95 F3 A5 AD B5 ?? F3 AB AD 50 97 51 58 8D 54 85 5C FF 16 72 ?? 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 E3 ?? B3 00 8D 1C 5B 8D 9C 9D 0C 10 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upack_024a_028a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.24a - 0.28a"
	strings:
		$1 = { BE 88 01 40 00 AD ?? ?? 95 AD 91 F3 A5 AD }
	condition:
		$1 at pe.entry_point
}

rule upack_024_031 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.24 - 0.31"
	strings:
		$1 = { BE ?? ?? ?? ?? AD 8B F8 95 AD 91 F3 A5 AD ?? ?? F3 AB AD 50 97 51 58 8D 54 85 5C FF 16 72 57 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF ?? ?? ?? B3 00 8D 1C 5B 8D 9C 9D 0C 10 00 00 B0 01 E3 29 8B D7 2B 55 0C 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 14 93 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E4 B4 00 33 C9 B5 01 FF 56 08 33 C9 FF 66 24 B1 30 8B 5D 0C 03 D1 FF 16 73 4B 03 D1 FF 16 72 19 03 D1 FF 16 72 29 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 45 0C 8A 00 FF 66 20 83 C2 60 FF 16 87 5D 10 73 0C 03 D1 FF 16 87 5D 14 73 03 87 5D 18 3C 07 B0 08 72 02 B0 0B 50 53 8B D5 03 56 14 FF 56 0C 5B 91 FF 66 3C 07 B0 07 72 02 B0 0A 50 87 5D 10 87 5D 14 89 5D 18 8B D5 03 56 18 FF 56 0C 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D 9C 85 7C 03 00 00 FF 56 04 3C 04 8B D8 72 5F 33 DB D1 E8 13 DB 48 43 91 43 D3 E3 80 F9 05 8D 94 9D 7C 01 00 00 76 2E 80 E9 04 33 C0 8B 55 00 D1 6D 08 8B 12 0F CA 2B 55 04 03 C0 3B 55 08 72 07 8B 55 08 40 01 55 04 FF 56 10 }
	condition:
		$1 at pe.entry_point
}

rule upack_024b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.24b"
	strings:
		$1 = { BE 88 01 40 00 AD 8B F8 95 AD 91 F3 A5 AD B5 ?? F3 AB AD 50 97 51 58 8D 54 85 5C FF 16 72 57 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 E3 ?? B3 00 8D 1C 5B 8D 9C 9D 0C 10 00 00 B0 }
	condition:
		$1 at pe.entry_point
}

rule upack_029b_031bs {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.29b - 0.31b"
	strings:
		$1 = { BE 88 01 ?? ?? AD 8B F8 95 AD 91 F3 A5 AD B5 ?? F3 }
	condition:
		$1 at pe.entry_point
}

rule upack_029b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.29b"
	strings:
		$1 = { E9 ?? ?? ?? ?? 42 79 44 77 69 6E 67 40 00 00 00 50 45 00 00 4C 01 02 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 29 }
	condition:
		$1 at pe.entry_point
}

rule upack_029 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.29"
	strings:
		$1 = { BE 88 01 ?? ?? AD 8B F8 95 AD 91 F3 A5 AD B5 ?? F3 AB AD 50 97 51 58 8D 54 85 5C FF 16 72 57 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 E3 }
	condition:
		$1 at pe.entry_point
}

rule upack_030b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.30b"
	strings:
		$1 = { E9 ?? ?? ?? ?? 42 79 44 77 69 6E 67 40 00 00 00 50 45 00 00 4C 01 02 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 30 }
	condition:
		$1 at pe.entry_point
}

rule upack_031b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.31b"
	strings:
		$1 = { E9 ?? ?? ?? ?? 42 79 44 77 69 6E 67 40 00 00 00 50 45 00 00 4C 01 02 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 31 }
	condition:
		$1 at pe.entry_point
}

rule upack_032b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.32b"
	strings:
		$1 = { E9 ?? ?? ?? ?? 42 79 44 77 69 6E 67 40 00 00 00 50 45 00 00 4C 01 02 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 32 }
	condition:
		$1 at pe.entry_point
}

rule upack_032 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.32"
	strings:
		$1 = { BE ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8D 54 85 5C FF 16 72 57 2C 03 73 02 ?? ?? 3C 07 72 02 2C 03 50 0F B6 5F FF ?? ?? ?? ?? ?? 8D 1C 5B 8D 9C 9D 0C 10 00 00 B0 01 E3 29 8B ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? FF 16 5A 9F 12 C0 D0 E9 74 0E ?? ?? ?? ?? ?? ?? ?? ?? ?? B5 01 FF 56 08 ?? ?? FF 66 24 B1 30 8B 5D 0C 03 D1 FF 16 73 4B 03 D1 FF 16 72 19 03 D1 FF 16 72 29 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 45 0C 8A 00 FF 66 20 83 C2 60 FF 16 87 5D 10 73 0C 03 D1 FF 16 87 5D 14 73 03 87 5D 18 3C 07 B0 08 72 02 B0 0B 50 53 8B D5 03 56 14 FF 56 0C 5B 91 FF 66 3C 07 B0 07 72 02 B0 0A 50 87 5D 10 87 5D 14 89 5D 18 8B D5 03 56 18 FF 56 0C 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D 9C 85 7C 03 00 00 FF 56 04 3C 04 8B D8 72 5F ?? ?? D1 E8 13 DB 48 43 91 43 D3 E3 80 F9 05 8D 94 9D 7C 01 00 00 76 2E 80 E9 04 ?? ?? 8B 55 00 D1 6D 08 8B 12 0F CA 2B 55 04 03 C0 3B 55 08 72 07 8B 55 08 40 01 55 04 FF 56 10 E2 E0 }
	condition:
		$1 at pe.entry_point
}

rule upack_033b_034b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.33b - 0.34b"
	strings:
		$1 = { ?? ?? ?? ?? 59 F3 A5 83 C8 FF 8B DF AB 40 AB 40 }
	condition:
		$1 at pe.entry_point
}

rule upack_033_034 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.33 - 0.34"
	strings:
		$1 = { 57 51 58 8D 54 83 58 FF 16 72 5C 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 6F FF C1 ED ?? C1 E5 08 8D 6C 6D 00 8D AC AB 08 10 00 00 B0 01 E3 2A 8B D7 2B 53 08 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 54 95 00 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E3 B4 00 33 C9 B5 01 FF 56 AC 33 C9 E9 03 01 00 00 B1 30 8B 6B 08 03 D1 FF 16 73 51 03 D1 FF 16 72 1B 03 D1 FF 16 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 43 08 8A 00 E9 D5 00 00 00 83 C2 60 FF 16 87 6B 0C 73 0C 03 D1 FF 16 87 6B 10 73 03 87 6B 14 3C 07 B0 08 72 02 B0 0B 50 55 8D 93 78 07 00 00 FF 56 B0 5D 91 E9 99 00 00 00 3C 07 B0 07 72 02 B0 0A 50 87 6B 0C 87 6B 10 89 6B 14 8D 93 C0 0B 00 00 FF 56 B0 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D AC 83 78 03 00 00 FF 56 A8 3C 04 8B E8 72 5C 33 ED D1 E8 13 ED 48 45 91 45 D3 E5 80 F9 05 8D 94 AB 78 01 00 00 76 2B 80 E9 04 33 C0 8B 53 FC D1 2B 8B 12 0F CA 2B 53 04 03 C0 3B 13 72 06 8B 13 40 01 53 04 FF 56 88 E2 E3 }
	condition:
		$1 at pe.entry_point
}

rule upack_035a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.35a"
	strings:
		$1 = { 8B F2 8B CA 03 4C 19 1C 03 54 1A 20 }
	condition:
		$1 at pe.entry_point
}

rule upack_035 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.35"
	strings:
		$1 = { 58 8D 54 83 58 FF 16 72 5C 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 6F FF ?? ?? ?? ?? ?? ?? 8D 6C 6D 00 8D AC AB 08 10 00 00 B0 01 E3 2A 8B D7 2B 53 08 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 54 95 00 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E3 B4 00 33 C9 B5 01 FF 56 AC 33 C9 E9 03 01 00 00 B1 30 8B 6B 08 03 D1 FF 16 73 51 03 D1 FF 16 72 1B 03 D1 FF 16 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 43 08 8A 00 E9 D5 00 00 00 83 C2 60 FF 16 87 6B 0C 73 0C 03 D1 FF 16 87 6B 10 73 03 87 6B 14 3C 07 B0 08 72 02 B0 0B 50 55 8D 93 78 07 00 00 FF 56 B0 5D 91 E9 99 00 00 00 3C 07 B0 07 72 02 B0 0A 50 87 6B 0C 87 6B 10 89 6B 14 8D 93 C0 0B 00 00 FF 56 B0 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D AC 83 78 03 00 00 FF 56 A8 3C 04 8B E8 72 5C 33 ED D1 E8 13 ED 48 45 91 45 D3 E5 80 F9 05 8D 94 AB 78 01 00 00 76 2B 80 E9 04 33 C0 8B 53 FC D1 2B 8B 12 0F CA 2B 53 04 03 C0 3B 13 72 06 8B 13 40 01 53 04 FF 56 88 E2 E3 B1 04 D3 E0 03 E8 8D 53 18 33 C0 55 40 51 D3 E0 8B EA 91 FF 56 A8 }
	condition:
		$1 at pe.entry_point
}

rule upack_036a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.36a"
	strings:
		$1 = { AB E2 E5 5D 59 8B 76 68 51 59 46 AD 85 C0 }
	condition:
		$1 at pe.entry_point
}

rule upack_036b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.36b"
	strings:
		$1 = { BE E0 11 ?? ?? FF 36 E9 C3 00 00 00 48 01 ?? ?? 0B 01 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C }
	condition:
		$1 at pe.entry_point
}

rule upack_036_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.36"
	strings:
		$1 = { 0B 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 18 10 00 00 10 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 10 00 00 00 02 00 00 ?? ?? ?? ?? ?? ?? 36 ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 0A 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 14 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 FF 76 08 FF 76 0C BE 1C 01 }
	condition:
		$1 at pe.entry_point
}

rule upack_036_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.36"
	strings:
		$1 = { 58 8D 54 83 58 FF 16 72 5C 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 6F FF ?? ?? ?? C1 E5 08 8D 6C 6D 00 8D AC AB 08 10 00 00 B0 01 E3 2A 8B D7 2B 53 08 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 54 95 00 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E3 B4 00 33 C9 B5 01 FF 56 18 33 C9 E9 03 01 00 00 B1 30 8B 6B 08 03 D1 FF 16 73 51 03 D1 FF 16 72 1B 03 D1 FF 16 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 43 08 8A 00 E9 D5 00 00 00 83 C2 60 FF 16 87 6B 0C 73 0C 03 D1 FF 16 87 6B 10 73 03 87 6B 14 3C 07 B0 08 72 02 B0 0B 50 55 8D 93 78 07 00 00 FF 56 1C 5D 91 E9 99 00 00 00 3C 07 B0 07 72 02 B0 0A 50 87 6B 0C 87 6B 10 89 6B 14 8D 93 C0 0B 00 00 FF 56 1C 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D AC 83 78 03 00 00 FF 56 14 3C 04 8B E8 72 5C 33 ED D1 E8 13 ED 48 45 91 45 D3 E5 80 F9 05 8D 94 AB 78 01 00 00 76 2B 80 E9 04 33 C0 8B 53 FC D1 2B 8B 12 0F CA 2B 53 04 03 C0 3B 13 72 06 8B 13 40 01 53 04 FF 56 3C E2 E3 }
	condition:
		$1 at pe.entry_point
}

rule upack_036 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.36"
	strings:
		$1 = { BE ?? ?? ?? ?? FF 36 E9 C3 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upack_037b_038b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.37b - 0.38b"
		description = "strip base relocation table option"
	strings:
		$1 = { 53 18 33 C0 55 40 51 D3 E0 8B EA 91 FF 56 4C 33 }
	condition:
		$1 at pe.entry_point
}

rule upack_037b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.37b"
	strings:
		$1 = { BE B0 11 ?? ?? AD 50 FF 76 34 EB 7C 48 01 ?? ?? 0B 01 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 18 10 00 00 10 00 00 00 00 ?? ?? ?? 00 00 ?? ?? 00 10 00 00 00 02 00 00 04 00 00 00 00 00 37 }
	condition:
		$1 at pe.entry_point
}

rule upack_037_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.37"
	strings:
		$1 = { 0B 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 18 10 00 00 10 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 10 00 00 00 02 00 00 ?? ?? ?? ?? ?? ?? 37 ?? ?? ?? ?? ?? 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 0A 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? 14 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upack_037_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.37"
	strings:
		$1 = { 58 8D 54 83 58 FF 16 72 5C 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 6F FF ?? ?? ?? C1 E5 08 8D 6C 6D 00 8D AC AB 08 10 00 00 B0 01 E3 2A 8B D7 2B 53 08 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 54 95 00 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E3 B4 00 33 C9 B5 01 FF 56 50 33 C9 E9 03 01 00 00 B1 30 8B 6B 08 03 D1 FF 16 73 51 03 D1 FF 16 72 1B 03 D1 FF 16 72 2B 3C 07 B0 09 72 02 B0 0B 50 8B C7 2B 43 08 8A 00 E9 D5 00 00 00 83 C2 60 FF 16 87 6B 0C 73 0C 03 D1 FF 16 87 6B 10 73 03 87 6B 14 3C 07 B0 08 72 02 B0 0B 50 55 8D 93 78 07 00 00 FF 56 54 5D 91 E9 99 00 00 00 3C 07 B0 07 72 02 B0 0A 50 87 6B 0C 87 6B 10 89 6B 14 8D 93 C0 0B 00 00 FF 56 54 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D AC 83 78 03 00 00 FF 56 4C 3C 04 8B E8 72 5C 33 ED D1 E8 13 ED 48 45 91 45 D3 E5 80 F9 05 8D 94 AB 78 01 00 00 76 2B 80 E9 04 33 C0 8B 53 FC D1 2B 8B 12 0F CA 2B 53 04 03 C0 3B 13 72 06 8B 13 40 01 53 04 FF 56 10 }
	condition:
		$1 at pe.entry_point
}

rule upack_037_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.37"
	strings:
		$1 = { BE ?? ?? ?? ?? AD 50 FF ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule upack_038b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.38b"
	strings:
		$1 = { BE B0 11 ?? ?? AD 50 FF 76 34 EB 7C 48 01 ?? ?? 0B 01 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 18 10 00 00 10 00 00 00 00 ?? ?? ?? 00 00 ?? ?? 00 10 00 00 00 02 00 00 04 00 00 00 00 00 38 }
	condition:
		$1 at pe.entry_point
}

rule upack_038 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.38"
	strings:
		$1 = { 58 8D 54 83 58 FF 16 72 5B 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 6F FF ?? ?? ?? 69 ED 00 0C 00 00 8D AC 2B 08 10 00 00 B0 01 E3 2A 8B D7 2B 53 08 8A 2A 33 D2 84 E9 0F 95 C6 52 FE C6 8A D0 8D 54 95 00 FF 16 5A 9F 12 C0 D0 E9 74 0E 9E 1A F2 74 E3 B4 00 33 C9 B5 01 FF 56 50 33 C9 E9 FB 00 00 00 04 F9 1A C0 B1 30 8B 6B 08 03 D1 FF 16 73 49 03 D1 FF 16 72 17 03 D1 FF 16 72 27 24 02 04 09 50 8B C7 2B 43 08 8A 00 E9 CD 00 00 00 83 C2 60 FF 16 87 6B 0C 73 0C 03 D1 FF 16 87 6B 10 73 03 87 6B 14 24 03 04 08 50 55 8D 93 78 07 00 00 FF 56 54 5D 91 E9 95 00 00 00 24 03 04 07 50 87 6B 0C 87 6B 10 89 6B 14 8D 93 C0 0B 00 00 FF 56 54 6A 03 59 50 48 3B C1 72 02 8B C1 C1 E0 06 B1 40 8D AC 83 78 03 00 00 FF 56 4C 3C 04 8B E8 72 5C 33 ED D1 E8 13 ED 48 45 91 45 D3 E5 80 F9 05 8D 94 AB 78 01 00 00 76 2B 80 E9 04 33 C0 8B 53 FC D1 2B 8B 12 0F CA 2B 53 04 03 C0 3B 13 72 06 8B 13 40 01 53 04 FF 56 10 }
	condition:
		$1 at pe.entry_point
}

rule upack_039_0399 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.39 - 0.399"
	strings:
		$1 = { 58 8D 54 83 58 FF 16 72 4F 04 FD 1A D2 22 C2 3C 07 73 F6 50 0F B6 6F FF ?? ?? ?? 66 69 ED 00 03 8D AC AB 08 10 00 00 57 B0 01 E3 1F 2B 7B 08 84 0F 0F 95 C4 FE C4 8D 54 85 00 FF 16 12 C0 D0 E9 74 0E 2A E0 80 E4 01 75 E6 33 C9 B5 01 FF 56 50 33 C9 5F E9 F2 00 00 00 04 F9 1A C0 B1 30 24 03 8B 6B 08 04 08 03 D1 FF 16 73 42 03 D1 FF 16 72 14 03 D1 FF 16 72 24 0C 01 50 8B C7 2B 43 08 B1 80 8A 00 EB CE 83 C2 60 FF 16 87 6B 0C 73 0C 03 D1 FF 16 87 6B 10 73 03 87 6B 14 50 55 8D 93 78 07 00 00 FF 56 54 5D 91 E9 8F 00 00 00 48 87 6B 0C 50 87 6B 10 8D 93 C0 0B 00 00 89 6B 14 FF 56 54 6A 03 59 50 48 3B C1 72 02 8B C1 B1 40 F6 E1 8D AC 83 78 03 00 00 FF 56 4C 3C 04 8B E8 72 5A 33 ED D1 E8 83 D5 02 48 91 D3 E5 80 F9 05 8D 94 AB 78 01 00 00 76 2B 80 E9 04 33 C0 8B 53 FC D1 2B 8B 12 0F CA 2B 53 04 03 C0 3B 13 72 06 8B 13 40 01 53 04 FF }
	condition:
		$1 at pe.entry_point
}

rule upack_039f_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.39f"
	strings:
		$1 = { 56 10 E2 E3 B1 04 D3 E0 03 E8 8D 53 18 33 C0 55 40 51 D3 E0 8B EA 91 }
	condition:
		$1 at pe.entry_point
}

rule upack_039f_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.39f"
	strings:
		$1 = { FF 76 38 AD 50 8B 3E BE F0 ?? ?? ?? 6A 27 59 F3 A5 FF 76 04 83 C8 FF }
	condition:
		$1 at pe.entry_point
}

rule upack_039f_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.39f"
	strings:
		$1 = { BE B0 11 ?? ?? AD 50 FF 76 34 EB 7C 48 01 ?? ?? 0B 01 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 18 10 00 00 10 00 00 00 ?? ?? ?? ?? 00 00 ?? ?? 00 10 00 00 00 02 00 00 04 00 00 00 00 00 39 }
	condition:
		$1 at pe.entry_point
}

rule upack_039 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.39"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E9 06 02 00 00 33 C9 5E 87 0E E3 F4 2B F1 8B DE AD 2B D8 AD 03 C3 50 97 AD 91 F3 A5 5E AD 56 91 01 1E AD E2 FB }
	condition:
		$1 at pe.entry_point
}

rule upack_0399_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.399"
	strings:
		$1 = { 0B 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 18 10 00 00 10 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 10 00 00 00 02 00 00 ?? ?? ?? ?? ?? ?? 3A 00 04 00 00 00 00 00 00 00 00 ?? ?? 00 00 02 00 00 00 00 00 00 ?? 00 00 00 00 00 10 00 00 ?? 00 00 00 00 10 00 00 10 00 00 00 00 00 00 0A 00 00 00 00 00 00 00 00 00 00 00 EE ?? ?? 00 14 00 00 00 00 ?? ?? 00 ?? ?? 00 00 FF 76 38 AD 50 8B 3E BE F0 ?? ?? 00 6A 27 59 F3 A5 FF 76 04 83 C8 FF 8B DF AB EB 1C 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 ?? ?? ?? 00 ?? 00 00 00 40 AB 40 B1 04 F3 AB C1 E0 0A B5 }
	condition:
		$1 at pe.entry_point
}

rule upack_0399_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.399"
	strings:
		$1 = { BE B0 11 ?? ?? AD 50 FF 76 34 EB 7C 48 01 ?? ?? 0B 01 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 18 10 00 00 10 00 00 00 00 ?? ?? ?? 00 00 ?? ?? 00 10 00 00 00 02 00 00 04 00 00 00 00 00 3A }
	condition:
		$1 at pe.entry_point
}

rule upack_0399_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Upack"
		version = "0.399"
	strings:
		$1 = { 60 E8 09 00 00 00 ?? ?? ?? ?? E9 06 02 00 00 33 C9 5E 87 0E E3 F4 2B F1 8B DE AD 2B D8 AD 03 C3 50 97 AD 91 F3 A5 5E AD 56 91 01 1E AD E2 FB AD 8D 6E 10 01 5D 00 8D 7D 1C B5 ?? F3 AB 5E AD 53 50 51 97 58 8D 54 85 5C FF 16 72 57 2C 03 73 02 B0 00 3C 07 72 02 2C 03 50 0F B6 5F FF C1 E3 ?? B3 00 8D 1C }
	condition:
		$1 at pe.entry_point
}

rule upolyx_05_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPolyX"
		version = "0.5"
	strings:
		$1 = { 55 8B EC ?? 00 BD 46 00 8B ?? B9 ?? 00 00 00 80 ?? ?? 51 ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upolyx_05_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPolyX"
		version = "0.5"
	strings:
		$1 = { 83 EC 04 89 14 24 59 BA ?? 00 00 00 52 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule upolyx_05_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPolyX"
		version = "0.5"
	strings:
		$1 = { BB 00 BD 46 00 83 EC 04 89 1C 24 ?? B9 ?? 00 00 00 80 33 ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upolyx_05_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPolyX"
		version = "0.5"
	strings:
		$1 = { E8 00 00 00 00 59 83 C1 07 51 C3 C3 ?? 00 BD 46 00 83 EC 04 89 ?? 24 B9 ?? 00 00 00 81 ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 0? ?0 0B D4 60 0? ?B 9? ?0 00 00 0? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 0? }
	condition:
		$1 at pe.entry_point
}

rule upolyx_05_05 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPolyX"
		version = "0.5"
	strings:
		$1 = { E8 00 00 00 00 59 83 C1 07 51 C3 C3 ?? 00 BD 46 00 83 EC 04 89 ?? 24 B9 ?? 00 00 00 81 ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upolyx_05_06 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPolyX"
		version = "0.5"
	strings:
		$1 = { EB 01 C3 ?? 00 BD 46 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upx_eclipse_layer {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		description = "+ ECLiPSE layer"
	strings:
		$1 = { B8 ?? ?? ?? ?? B9 ?? ?? ?? ?? 33 D2 EB 01 0F 56 EB 01 0F E8 03 00 00 00 EB 01 0F EB 01 0F 5E EB 01 }
	condition:
		$1 at pe.entry_point
}

rule upx_060_061 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.60 - 0.61"
	strings:
		$1 = { 60 E8 00 00 00 00 58 83 E8 3D 50 8D B8 ?? ?? ?? FF 57 8D B0 E8 }
	condition:
		$1 at pe.entry_point
}

rule upx_062_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.62"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 95 01 00 00 60 E8 00 00 00 00 58 }
	condition:
		$1 at pe.entry_point
}

rule upx_062_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.62"
	strings:
		$1 = { 60 E8 00 00 00 00 58 83 E8 3D 50 8D B8 ?? ?? ?? FF 57 66 81 87 ?? ?? ?? ?? ?? ?? 8D B0 F0 01 ?? ?? 83 CD FF 31 DB 90 90 90 EB 08 90 90 8A 06 46 88 07 47 01 DB 75 07 }
	condition:
		$1 at pe.entry_point
}

rule upx_070 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.70"
	strings:
		$1 = { 60 E8 00 00 00 00 58 83 E8 3D 50 8D B8 ?? ?? ?? FF 57 66 81 87 ?? ?? ?? ?? ?? ?? 8D B0 EC 01 ?? ?? 83 CD FF 31 DB EB 07 90 8A 06 46 88 07 47 01 DB 75 07 }
	condition:
		$1 at pe.entry_point
}

rule upx_071 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.71"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 95 01 00 00 60 E8 00 00 00 00 83 }
	condition:
		$1 at pe.entry_point
}

rule upx_072 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.72"
	strings:
		$1 = { 60 E8 ?? ?? ?? ?? 83 ?? ?? 31 DB 5E 8D ?? ?? ?? ?? ?? 57 66 ?? ?? ?? ?? ?? ?? ?? ?? 81 ?? ?? ?? ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule upx_0761 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.76.1"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 66 ?? ?? ?? ?? ?? ?? 57 83 ?? ?? 31 DB EB }
	condition:
		$1 at pe.entry_point
}

rule upx_080_084 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.80 - 0.84"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 ?? ?? ?? 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 77 EF 75 09 8B 1E 83 EE FC }
	condition:
		$1 at pe.entry_point
}

rule upx_080_or_higher_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.80 or higher"
	strings:
		$1 = { 8A 06 ?? ?? ?? 47 0? DB 75 07 8B 1E 83 EE FC ?? ?? 72 E? ?? ?? ?? ?? ?? ?? DB 75 07 8B 1E 83 EE FC ?? ?? ?? C0 0? DB ?? ?? ?? ?? 8B 1E 83 EE FC }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 192)
}

rule upx_080_or_higher_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.80 or higher"
	strings:
		$1 = { 8A 06 ?? ?? ?? 47 01 DB 75 07 8B 1E 83 EE FC ?? ?? ?? ?? 72 E? ?? ?? ?? ?? ?? ?? DB 75 07 8B 1E 83 EE FC ?? ?? ?? C0 0? DB ?? ?? ?? ?? 8B 1E 83 EE FC }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 192)
}

rule upx_080_or_higher_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.80 or higher"
	strings:
		$1 = { 8A 06 ?? ?? ?? 47 01 DB 75 08 8B 1E 83 EE FC ?? ?? ?? 72 E? ?? ?? ?? ?? ?? ?? DB 75 08 8B 1E 83 EE FC ?? ?? ?? ?? C0 0? DB ?? ?? ?? ?? 8B 1E 83 EE FC }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 192)
}

rule upx_081_084_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.81 - 0.84 modified"
	strings:
		$1 = { 01 DB ?? 07 8B 1E 83 EE FC 11 DB ?? ED B8 01 00 00 00 01 DB ?? 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 77 EF }
	condition:
		$1 at pe.entry_point
}

rule upx_089_3xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.89 - 3.xx"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE }
	condition:
		$1 at pe.entry_point
}

rule upx_0896_102_105_122_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.89.6 - 1.02, 1.05 - 1.22"
		description = "Delphi stub"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? C7 87 ?? ?? ?? ?? ?? ?? ?? ?? 57 83 CD FF EB 0E ?? ?? ?? ?? 8A 06 46 88 07 47 01 DB 75 07 8B }
	condition:
		$1 at pe.entry_point
}

rule upx_0896_102_105_122_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.89.6 - 1.02, 1.05 - 1.22"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? 00 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF }
	condition:
		$1 at pe.entry_point
}

rule upx_0896_102_105_122_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.89.6 - 1.02, 1.05 - 1.22"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 ?? ?? ?? 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 ?? 75 ?? 8B 1E 83 EE FC }
	condition:
		$1 at pe.entry_point
}

rule upx_0896_102_105_122_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "0.89.6 - 1.02, 1.05 - 1.22 modified"
	strings:
		$1 = { 01 DB ?? 07 8B 1E 83 EE FC 11 DB ?? ED B8 01 00 00 00 01 DB ?? 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 ?? 75 }
	condition:
		$1 at pe.entry_point
}

rule upx_103_104_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.03 - 1.04 modified"
	strings:
		$1 = { 01 DB ?? 07 8B 1E 83 EE FC 11 DB 8A 07 ?? EB B8 01 00 00 00 01 DB ?? 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF }
	condition:
		$1 at pe.entry_point
}

rule upx_103_104 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.03 - 1.04"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 ?? 75 ?? 8B 1E 83 EE FC }
	condition:
		$1 at pe.entry_point
}

rule upx_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.2"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 05 A4 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 F2 31 C0 40 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 75 07 8B 1E 83 EE FC 11 DB 73 E6 31 C9 83 }
	condition:
		$1 at pe.entry_point
}

rule upx_121 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.21"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 66 81 87 ?? ?? ?? 00 0? 00 57 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 }
	condition:
		$1 at pe.entry_point
}

rule upx_12x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.2x"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 }
	condition:
		$1 at pe.entry_point
}

rule upx_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.0"
	strings:
		$1 = { 55 FF 96 ?? ?? ?? ?? 09 C0 74 07 89 03 83 C3 04 EB ?? FF 96 ?? ?? ?? ?? 8B AE ?? ?? ?? ?? 8D BE 00 F0 FF FF BB 00 10 00 00 50 54 6A 04 53 57 FF D5 8D 87 ?? ?? 00 00 80 20 7F 80 60 28 7F 58 50 54 50 53 57 FF D5 58 61 8D 44 24 80 6A 00 39 C4 75 FA 83 EC 80 E9 }
	condition:
		$1 at pe.entry_point
}

rule upx_290_lzma_delphi_stub {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.90 [LZMA]"
		description = "Delphi stub"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? C7 87 ?? ?? ?? ?? ?? ?? ?? ?? 57 83 CD FF 89 E5 8D 9C 24 ?? ?? ?? ?? 31 C0 50 39 DC 75 FB 46 46 53 68 ?? ?? ?? ?? 57 83 C3 04 53 68 ?? ?? ?? ?? 56 83 C3 04 }
	condition:
		$1 at pe.entry_point
}

rule upx_290_lzma_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.90 [LZMA]"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF 89 E5 8D 9C 24 ?? ?? ?? ?? 31 C0 50 39 DC 75 FB 46 46 53 68 ?? ?? ?? ?? 57 83 C3 04 53 68 ?? ?? ?? ?? 56 83 C3 04 53 50 C7 03 ?? ?? ?? ?? 90 90 }
	condition:
		$1 at pe.entry_point
}

rule upx_290_lzma_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.90 [LZMA]"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB }
	condition:
		$1 at pe.entry_point
}

rule upx_291_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.91"
	strings:
		$1 = { 68 00 04 F5 0F E8 02 00 00 00 50 C3 55 89 E5 81 EC 0C 02 00 00 C7 85 F4 FD FF FF 48 75 79 20 C7 85 F8 FD FF FF 76 61 6D 21 66 C7 85 FC FD FF FF 21 21 }
	condition:
		$1 at pe.entry_point
}

rule upx_291_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.91"
	strings:
		$1 = { E8 10 00 00 00 6A FF 6A 00 68 23 01 00 00 E8 0A 00 00 00 50 C3 C8 00 00 04 C9 58 EB E8 55 89 E5 81 EC F4 03 00 00 C7 85 0C FC FF FF 31 32 33 34 }
	condition:
		$1 at pe.entry_point
}

rule upx_293_300_lzma {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "2.93 - 3.00 [LZMA]"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 89 E5 8D 9C 24 ?? ?? ?? ?? 31 C0 50 39 DC 75 FB 46 46 53 68 ?? ?? ?? ?? 57 83 C3 04 53 68 ?? ?? ?? ?? 56 83 C3 04 53 50 C7 03 03 00 02 00 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule upx_30_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.0"
	strings:
		$1 = { 55 57 56 53 83 EC 7C 8B 94 24 90 00 00 00 C7 44 24 74 00 00 00 00 C6 44 24 73 00 8B AC 24 9C 00 00 00 8D 42 04 89 44 24 78 B8 01 00 00 00 0F B6 4A 02 89 C3 D3 E3 89 D9 49 89 4C 24 6C 0F B6 4A 01 D3 E0 48 89 44 24 68 8B 84 24 A8 00 00 00 0F B6 32 }
	condition:
		$1 in (pe.entry_point + 48 .. pe.entry_point + 80)
}

rule upx_30_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.0"
	strings:
		$1 = { E8 ?? ?? ?? ?? 58 83 D8 05 89 C3 83 C3 30 8B 43 39 05 00 00 40 00 8B 4B 3D 89 C6 89 C7 8C D8 8E C0 B4 00 AC 30 E0 88 C4 AA E2 F8 8B 43 08 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule upx_modofied_stub_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified stub"
	strings:
		$1 = { 01 DB 07 8B 1E 83 EE FC 11 DB ED B8 01 00 00 00 01 DB 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B }
	condition:
		$1 at pe.entry_point
}

rule upx_modofied_stub_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified stub"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF FC B2 80 31 DB A4 B3 02 E8 6D 00 00 00 73 F6 31 C9 E8 64 00 00 00 73 1C 31 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 10 C0 73 F7 75 3F AA EB D4 E8 4D 00 00 00 29 D9 75 10 E8 42 00 00 00 EB 28 AC D1 E8 74 4D 11 C9 EB 1C 91 48 C1 E0 08 AC E8 2C 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 89 E8 B3 01 56 89 FE 29 C6 F3 A4 5E EB 8E 00 D2 75 05 8A 16 46 10 D2 C3 31 C9 41 E8 EE FF FF FF 11 C9 E8 E7 FF FF FF 72 F2 C3 31 }
	condition:
		$1 at pe.entry_point
}

rule upx_modofied_stub_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified stub"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF FC B2 80 E8 00 00 00 00 5B 83 C3 66 A4 FF D3 73 FB 31 C9 FF D3 73 14 31 C0 FF D3 73 1D 41 B0 10 FF D3 10 C0 73 FA 75 3C AA EB E2 E8 4A 00 00 00 49 E2 10 E8 40 00 00 00 EB 28 AC D1 E8 74 45 11 C9 EB 1C 91 48 C1 E0 08 AC E8 2A 00 00 00 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 89 E8 56 89 FE 29 C6 F3 A4 5E EB 9F 00 D2 75 05 8A 16 46 10 D2 C3 31 C9 41 FF D3 11 C9 FF D3 72 F8 C3 31 C0 31 DB 31 C9 5E 89 F7 B9 ?? ?? ?? ?? 8A 07 47 2C E8 }
	condition:
		$1 at pe.entry_point
}

rule upx_modofied_stub_04 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified stub"
	strings:
		$1 = { 79 07 0F B7 07 47 50 47 B9 57 48 F2 AE 55 FF 96 84 ?? 00 00 09 C0 74 07 89 03 83 C3 04 EB D8 FF 96 88 ?? 00 00 61 E9 ?? ?? ?? FF }
	condition:
		$1 at pe.entry_point
}

rule upx_modified_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified"
	strings:
		$1 = { 55 8B EC 83 C4 F8 60 C6 45 FF 00 C7 45 F8 00 00 00 00 8B 7D 08 8B 75 0C 8B 55 10 8B 5D 1C 33 C9 EB 2C 8B C1 03 C3 3B 45 20 77 73 51 56 }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 192)
}

rule upx_modified_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? 01 00 00 60 BE 00 ?? ?? 10 8D BE 00 ?? ?? FF 57 83 CD FF EB 0F 90 90 90 8A 06 34 55 46 88 07 47 01 DB 75 09 50 B0 20 E8 ?? 00 00 00 58 72 E9 B8 01 00 00 00 50 B0 01 E8 }
	condition:
		$1 at pe.entry_point
}

rule upx_modified_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "modified"
	strings:
		$1 = { E8 00 00 00 00 55 8B 6C 24 04 81 6C 24 04 ?? ?? 00 00 E8 ?? ?? 00 00 8B C8 E8 ?? 01 00 00 2B C1 3D 00 01 00 00 0F 83 ?? 00 00 00 8B 5C 24 08 81 E3 00 F0 FF FF 81 ED 05 10 40 00 80 3B 4D 75 13 }
	condition:
		$1 at pe.entry_point
}

rule upx_10x_protector {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.0x Protector"
	strings:
		$1 = { EB ?? ?? ?? ?? ?? 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB }
	condition:
		$1 at pe.entry_point
}

rule upx_10_inliner {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.0 Inliner"
	strings:
		$1 = { 9C 60 E8 00 00 00 00 5D B8 B3 85 40 00 2D AC 85 40 00 2B E8 8D B5 D5 FE FF FF 8B 06 83 F8 00 74 11 8D B5 E1 FE FF FF 8B 06 83 F8 01 0F 84 F1 01 00 00 C7 06 01 00 00 00 8B D5 8B 85 B1 FE FF FF 2B D0 89 95 B1 FE FF FF 01 95 C9 FE FF FF 8D B5 E5 FE FF FF 01 16 8B 36 8B FD 60 6A 40 68 00 10 00 00 68 00 10 00 00 6A 00 FF 95 05 FF FF FF 85 C0 0F 84 06 03 00 00 89 85 C5 FE FF FF E8 00 00 00 00 5B B9 31 89 40 00 81 E9 2E 86 40 00 03 D9 50 53 E8 3D 02 00 00 61 03 BD A9 FE FF FF 8B DF 83 3F 00 75 0A 83 C7 }
	condition:
		$1 at pe.entry_point
}

rule upx_upxshit_001_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "UPX$HiT 0.0.1"
	strings:
		$1 = { 94 BC ?? ?? ?? 00 B9 ?? 00 00 00 80 34 0C ?? E2 FA 94 FF E0 61 }
	condition:
		$1
}

rule upx_upxshit_001_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "UPX$HiT 0.0.1"
	strings:
		$1 = { E2 FA 94 FF E0 61 00 00 00 00 00 00 00 }
	condition:
		$1
}

rule upx_upxshit_001_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "UPX$HiT 0.0.1"
	strings:
		$1 = { E8 ?? ?? ?? ?? 5E 83 C6 ?? AD 89 C7 AD 89 C1 AD 30 07 47 E2 ?? AD FF E0 C3 }
	condition:
		$1 at pe.entry_point
}

rule upx_upxshit_006 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "UPX$HiT 0.06"
	strings:
		$1 = { B8 ?? ?? 43 00 B9 15 00 00 00 80 34 08 ?? E2 FA E9 D6 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule upx_306_scrambler {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.06 Scrambler"
	strings:
		$1 = { E8 00 00 00 00 59 83 C1 07 51 C3 C3 BE ?? ?? ?? ?? 83 EC 04 89 34 24 B9 80 00 00 00 81 36 ?? ?? ?? ?? 50 B8 04 00 00 00 50 03 34 24 58 58 83 E9 03 E2 E9 EB D6 }
	condition:
		$1 at pe.entry_point
}

rule upx_1x_scrambler_rc {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "1.x Scrambler RC"
	strings:
		$1 = { 90 61 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF }
	condition:
		$1 at pe.entry_point
}

rule upx_upxcrypter {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "UPXcrypter"
	strings:
		$1 = { BF ?? ?? ?? 00 81 FF ?? ?? ?? 00 74 10 81 2F ?? 00 00 00 83 C7 04 BB 05 ?? ?? 00 FF E3 BE ?? ?? ?? 00 FF E6 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule upx_upxlock_10_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "UpxLock 1.0 - 1.2"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 48 12 40 00 60 E8 2B 03 00 00 61 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_lzma_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 89 E5 8D 9C 24 ?? ?? ?? ?? 31 C0 50 39 DC 75 FB 46 46 53 68 ?? ?? ?? ?? 57 83 C3 04 53 68 ?? ?? ?? ?? 56 83 C3 04 53 50 C7 03 ?? ?? ?? ?? 90 90 90 90 90 55 57 56 53 83 EC 7C 8B 94 24 90 00 00 00 C7 44 24 74 00 00 00 00 C6 44 24 73 00 8B AC 24 9C 00 00 00 8D 42 04 89 44 24 78 B8 01 00 00 00 0F B6 4A 02 89 C3 D3 E3 89 D9 49 89 4C 24 6C 0F B6 4A }
	condition:
		$1 at pe.entry_point
}

rule upx_391_lzma_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 89 E5 8D 9C 24 ?? ?? ?? ?? 31 C0 50 39 DC 75 FB 46 46 53 68 ?? ?? ?? ?? 57 83 C3 04 53 68 ?? ?? ?? ?? 56 83 C3 04 53 50 C7 03 ?? ?? ?? ?? 90 90 55 57 56 53 83 EC 7C 8B 94 24 90 00 00 00 C7 44 24 74 00 00 00 00 C6 44 24 73 00 8B AC 24 9C 00 00 00 8D 42 04 89 44 24 78 B8 01 00 00 00 0F B6 4A 02 89 C3 D3 E3 89 D9 49 89 4C 24 6C 0F B6 4A }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) EF 75 09 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 EB 10 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) EF 75 09 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2b_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 0D 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) EF 75 09 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2d_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 EB 0B 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) 0B 75 19 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2d_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) 0B 75 19 }
	condition:
		$1 at pe.entry_point
}
rule upx_391_nrv2d_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 0D 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) 0B 75 19 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2e_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 EB 0B 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) 0B 75 28 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2e_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) 0B 75 28 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2e_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? 57 83 CD FF EB 0D 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB ( 73 | 77 ) 0B 75 28 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2b_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2B] modified"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? C7 87 ?? ?? ?? ?? ?? ?? ?? ?? 57 EB 0E 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2d_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2D] modified"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? C7 87 ?? ?? ?? ?? ?? ?? ?? ?? 57 EB 0E 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 19 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2e_modf_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2E] modified"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? C7 87 ?? ?? ?? ?? ?? ?? ?? ?? 57 EB 0E 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 28 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2e_modf_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.9x [NRV2E] modified"
		source = "Made by Retdec Team"
	strings:
		$1 = { 80 7C 24 08 01 0F 85 ?? ?? ?? ?? 60 BE ?? ?? ?? ?? 8D BE ?? ?? ?? ?? C7 87 ?? ?? ?? ?? ?? ?? ?? ?? 57 EB 11 90 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 28 }
	condition:
		$1 at pe.entry_point
}

rule upx_394_lzma {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.94 [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE 00 ?0 4? 00 8D BE 00 ?0 F? FF 57 89 E5 8D 9C 24 80 C1 FF FF 31 C0 50 39 DC 75 FB 46 46 53 68 ?? ?? 0? 00 57 83 C3 04 53 68 ?? ?? 0? 00 56 83 C3 04 53 50 C7 03 03 00 02 00 55 57 56 53 83 EC 7C 8B 94 24 90 00 00 00 C7 44 24 74 00 00 00 00 C6 44 24 73 00 8B AC 24 9C 00 00 00 8D 42 04 89 44 24 78 }
	condition:
		$1 at pe.entry_point
}

rule upx_394_nrv2b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.94 [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE 00 ?0 4? 00 8D BE 00 ?0 F? FF 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 8B 1E 83 EE FC 11 DB 73 E4 31 C9 83 E8 03 72 0D C1 E0 08 8A 06 46 83 F0 FF 74 74 89 C5 01 DB 75 07 }
	condition:
		$1 at pe.entry_point
}

rule upx_394_nrv2b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPX"
		version = "3.94 [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 60 BE 00 ?0 4? 00 8D BE 00 ?0 F? FF 57 EB 0B 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 8B 1E 83 EE FC 11 DB 73 E4 31 C9 83 E8 03 72 0D C1 E0 08 8A 06 46 83 F0 FF 74 74 89 C5 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 }
	condition:
		$1 at pe.entry_point
}

rule upx_freak {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "UPXFreak"
		version = "0.1"
		description = "for Borland Delphi"
	strings:
		$1 = { BE ?? ?? ?? ?? 83 C6 01 FF E6 00 00 00 ?? ?? ?? 00 03 00 00 00 ?? ?? ?? ?? 00 10 00 00 00 00 ?? ?? ?? ?? 00 00 ?? F6 ?? 00 B2 4F 45 00 ?? F9 ?? 00 EF 4F 45 00 ?? F6 ?? 00 8C D1 42 00 ?? 56 ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? ?? ?? 00 ?? 24 ?? 00 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule ussr_031_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "USSR"
		version = "0.31"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 2E 55 53 53 52 00 00 00 00 10 00 00 ?? ?? ?? ?? 00 10 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ussr_031_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "USSR"
		version = "0.31"
	strings:
		$1 = { E8 00 00 00 00 5D 83 C5 12 55 C3 20 83 B8 ED 20 37 EF C6 B9 79 37 9E 8C C9 30 C9 E3 01 C3 BE 32 ?? ?? ?? B0 ?? 30 06 8A 06 46 81 FE 00 ?? ?? ?? 7C F3 }
	condition:
		$1 at pe.entry_point
}

rule vbox_42_mte {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VBOX"
		version = "4.2 MTE"
	strings:
		$1 = { 8C E0 0B C5 8C E0 0B C4 03 C5 74 00 74 00 8B C5 }
	condition:
		$1 at pe.entry_point
}

rule vbox_43_46_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VBOX"
		version = "4.3 - 4.6"
	strings:
		$1 = { ?? ?? ?? ?? 90 03 C4 33 C4 33 C5 2B C5 33 C5 8B C5 ?? ?? 2B C5 48 ?? ?? 0B C0 86 E0 8C E0 ?? ?? 8C E0 86 E0 03 C4 40 }
	condition:
		$1 at pe.entry_point
}

rule vbox_43_46_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VBOX"
		version = "4.3 - 4.6"
	strings:
		$1 = { 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 8B C5 }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_10e {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.0e"
	strings:
		$1 = { EB 0A 5B 56 50 72 6F 74 65 63 74 5D }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_10x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.0x"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 E8 03 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_10x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.0x"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 E8 03 00 00 00 C7 84 00 58 EB 01 E9 83 C0 07 50 }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_11_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.1 - 1.2"
	strings:
		$1 = { EB 0B 5B 56 50 72 6F 74 65 63 74 5D }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.1"
	strings:
		$1 = { B8 1A ED 41 00 B9 EC EB 41 00 50 51 E8 74 00 00 00 E8 51 6A 00 00 58 83 E8 10 B9 B3 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_11a_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.1a - 1.2"
	strings:
		$1 = { 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 00 00 76 63 61 73 6D 5F 70 72 6F 74 65 63 74 5F 32 30 30 35 5F 33 5F 31 38 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 33 F6 E8 10 00 00 00 8B 64 24 08 64 8F 05 00 00 00 00 58 EB 13 C7 83 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 AD CD 20 EB 01 0F 31 F0 EB 0C 33 C8 EB 03 EB 09 0F 59 74 05 75 F8 51 EB F1 B9 04 00 00 00 E8 1F 00 00 00 EB FA E8 16 00 00 00 E9 EB F8 00 00 58 EB 09 0F 25 E8 F2 FF FF FF 0F B9 49 75 F1 EB 05 EB F9 EB F0 D6 E8 07 00 00 00 C7 83 83 C0 13 EB 0B 58 EB 02 CD 20 83 C0 02 EB 01 E9 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_13x_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.3x"
	strings:
		$1 = { 00 00 00 00 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 60 8B B4 24 24 00 00 00 8B BC 24 28 00 00 00 FC C6 C2 80 33 DB A4 C6 C3 02 E8 A9 00 00 00 0F 83 F1 FF FF FF 33 C9 E8 9C 00 00 00 0F 83 2D 00 00 00 33 C0 E8 8F 00 00 00 0F 83 37 00 00 00 C6 C3 02 41 C6 C0 10 E8 7D 00 00 00 10 C0 0F 83 F3 FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_13x_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.3x"
	strings:
		$1 = { E9 B9 16 00 00 55 8B EC 81 EC 74 04 00 00 57 68 00 00 00 00 68 00 00 C2 14 68 FF FF 00 00 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 68 00 00 C2 10 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 68 FF FF C2 10 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 68 00 00 C2 14 68 FF FF 00 00 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 68 ?? ?? ?? ?? 9C 81 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 9D 54 FF 14 24 68 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule vcasm_protector_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vcasm Protector"
		version = "1.x"
	strings:
		$1 = { EB ?? 5B 56 50 72 6F 74 65 63 74 5D }
	condition:
		$1 at pe.entry_point
}

rule vfp_exenc_500 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "vfp&exeNc"
		version = "5.00"
	strings:
		$1 = { 60 E8 00 00 00 00 5D ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 CC }
	condition:
		$1 at pe.entry_point
}

rule vfp_exenc_600 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "vfp&exeNc"
		version = "6.00"
	strings:
		$1 = { 60 E8 01 00 00 00 63 58 E8 01 00 00 00 7A 58 2D 0D 10 40 00 8D 90 C1 10 40 00 52 50 8D 80 49 10 40 00 5D 50 8D 85 65 10 40 00 50 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 CC }
	condition:
		$1 at pe.entry_point
}

rule virogen_crypt_075 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Virogen Crypt"
		version = "0.75"
	strings:
		$1 = { 9C 55 E8 EC 00 00 00 87 D5 5D 60 87 D5 80 BD 15 27 40 00 01 }
	condition:
		$1 at pe.entry_point
}

rule virogens_pe_shrinker_014 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Virogen's PE Shrinker"
		version = "0.14"
	strings:
		$1 = { 9C 55 E8 ?? ?? ?? ?? 87 D5 5D 60 87 D5 8D ?? ?? ?? ?? ?? 8D ?? ?? ?? ?? ?? 57 56 AD 0B C0 74 }
	condition:
		$1 at pe.entry_point
}

rule viseman_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VISEMAN"
	strings:
		$1 = { 45 53 49 56 }
	condition:
		$1 at pe.entry_point
}

rule visual_protect_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Visual Protect"
	strings:
		$1 = { 55 8B EC 51 53 56 57 C7 05 ?? ?? ?? 00 00 00 00 00 68 ?? ?? ?? 00 FF 15 00 ?? ?? 00 A3 ?? ?? ?? 00 68 ?? ?? ?? 00 A1 ?? ?? ?? 00 50 FF 15 04 ?? ?? 00 A3 ?? ?? ?? 00 6A 00 FF 15 ?? ?? ?? 00 A3 ?? ?? ?? 00 8B 0D ?? ?? ?? 00 51 E8 ?? ?? 00 00 83 C4 04 89 45 FC 83 7D FC 00 74 03 FF 65 FC 5F }
	condition:
		$1 at pe.entry_point
}

rule vmprotect_07x_08 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VMProtect"
		version = "0.7x - 0.8"
	strings:
		$1 = { 5B 20 56 4D 50 72 6F 74 65 63 74 20 76 20 30 2E 38 20 28 43 29 20 50 6F 6C 79 54 65 63 68 20 5D }
	condition:
		$1 at pe.entry_point
}

rule vmprotect_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VMProtect"
		version = "1.x"
	strings:
		$1 = { 9C 60 68 00 00 00 00 8B 74 24 28 BF ?? ?? ?? ?? FC 89 F3 03 34 24 AC 00 D8 }
	condition:
		$1 at pe.entry_point
}

rule vmprotect_2x_xx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VMProtect"
		version = "2.04+"
	strings:
		$1 = { 50 F0 1F FD FD 8? ?7 92 6? ?? B4 ?? C2 ?? ?0 7? 4? ?? ?? C? C? ?F ?D 2? 6? ?1 9C BF 0? 99 12 ?7 17 ?? 36 35 CA 8A ?7 ?0 ?? ?F ?C ?D 7D 7? ?9 E5 ?1 ?8 4E 4? ?? 24 ?? D4 5? 5? C? 04 B9 E? D? 2? 15 ?8 9? ?6 ?7 84 ?? ?? ?D 9? ?1 ?1 ?E ?? 03 ?? ?? ?4 46 ?6 ?? ?3 EC 94 1E ?6 A? ?4 ?5 ?? ?? ?? ?? 8? C? ?8 ?? ?2 ?? ?0 C8 EB ?C 1? D? }
	condition:
		for any i in (0 .. pe.number_of_sections - 1): (
			pe.sections[i].characteristics & pe.SECTION_CNT_CODE and
			$1 in (pe.sections[i].raw_data_offset .. pe.sections[i].raw_data_offset + pe.sections[i].raw_data_size)
		)
}

rule vob_protectcd_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VOB ProtectCD"
	strings:
		$1 = { 5F 81 EF ?? ?? ?? ?? BE ?? ?? 40 ?? 8B 87 ?? ?? ?? ?? 03 C6 57 56 8C A7 ?? ?? ?? ?? FF 10 89 87 ?? ?? ?? ?? 5E 5F }
	condition:
		$1 at pe.entry_point
}

rule vpacker_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VPacker"
	strings:
		$1 = { 00 00 00 00 FF FF FF FF FF FF FF FF ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 00 00 48 65 61 70 43 72 65 61 74 65 00 00 00 48 65 61 70 41 6C 6C 6F 63 00 C3 8D 40 00 55 8B EC 51 E8 28 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule vpacker_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VPacker"
	strings:
		$1 = { 89 C6 C7 45 E0 01 00 00 00 F7 03 00 00 FF FF 75 18 0F B7 03 50 8B 45 D8 50 FF 55 F8 89 07 8B C3 E8 ?? FE FF FF 8B D8 EB 13 53 8B 45 D8 50 FF 55 F8 89 07 8B C3 E8 ?? FE FF FF 8B D8 83 C7 04 FF 45 E0 4E 75 C4 8B F3 83 3E 00 75 88 8B 45 E4 8B 40 10 03 45 DC 8B 55 14 83 C2 20 89 02 68 00 80 00 00 6A 00 8B 45 D4 50 FF 55 EC 8B 55 DC 8B 42 3C 03 45 DC 83 C0 04 8B D8 83 C3 14 8D 45 E0 50 6A 40 68 00 10 00 00 52 FF 55 E8 8D 43 60 }
	condition:
		$1 at pe.entry_point
}

rule vprotector_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VProtector"
	strings:
		$1 = { 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 64 6C 6C 00 00 55 53 45 52 33 32 2E 64 6C 6C 00 00 47 44 49 33 32 2E 64 6C 6C 00 00 00 00 00 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 47 65 74 4D 6F 64 75 6C 65 48 61 6E 64 6C 65 41 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 53 6C 65 65 70 00 00 00 47 65 74 56 65 72 73 69 6F 6E 00 00 00 47 65 74 43 6F 6D 6D 61 6E 64 4C 69 6E 65 41 00 00 00 47 65 74 53 74 61 72 74 75 70 49 6E 66 6F 41 00 00 00 47 65 74 41 43 50 00 00 00 43 72 65 61 74 65 54 68 72 65 61 64 00 00 00 44 65 66 57 69 6E 64 6F 77 50 72 6F 63 41 00 00 00 52 65 67 69 73 74 65 72 43 6C 61 73 73 45 78 41 00 00 00 43 72 65 61 74 65 57 69 6E 64 6F 77 45 78 41 00 00 00 47 65 74 53 79 73 74 65 6D 4D 65 74 72 69 63 73 00 00 00 53 68 6F 77 57 69 6E 64 6F 77 00 00 00 47 65 74 44 43 00 00 00 52 65 6C 65 61 73 65 44 43 00 00 00 46 69 6E 64 57 69 6E 64 6F 77 41 00 00 00 47 65 74 4D 65 73 73 61 67 65 41 00 00 00 44 65 73 74 72 6F 79 57 69 6E 64 6F 77 00 00 00 53 65 74 50 69 78 65 6C 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule vprotector_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VProtector"
	strings:
		$1 = { 00 00 00 00 55 73 65 72 33 32 2E 64 6C 6C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 47 64 69 33 32 2E 64 6C 6C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 4B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 44 65 66 57 69 6E 64 6F 77 50 72 6F 63 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 52 65 67 69 73 74 65 72 43 6C 61 73 73 45 78 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 43 72 65 61 74 65 57 69 6E 64 6F 77 45 78 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 47 65 74 53 79 73 74 65 6D 4D 65 74 72 69 63 73 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 53 68 6F 77 57 69 6E 64 6F 77 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 47 65 74 44 43 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 52 65 6C 65 61 73 65 44 43 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 08 00 46 69 6E 64 57 69 6E 64 6F 77 41 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 47 65 74 4D 65 73 73 61 67 65 41 00 }
	condition:
		$1 at pe.entry_point
}

rule vprotector_0x_12x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VProtector"
		version = "0.x - 1.2x"
	strings:
		$1 = { 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 00 00 76 63 61 73 6D 5F 70 72 6F 74 65 63 74 5F ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 33 F6 E8 10 00 00 00 8B 64 24 08 64 8F 05 00 00 00 00 58 EB 13 C7 83 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 AD CD 20 EB 01 0F 31 F0 EB 0C 33 C8 EB 03 EB 09 0F 59 74 05 75 F8 51 EB F1 B9 04 00 00 00 E8 1F 00 00 00 EB FA E8 16 00 00 00 E9 EB F8 00 00 58 EB 09 0F 25 E8 F2 FF FF FF 0F B9 49 75 F1 EB 05 EB F9 EB F0 D6 E8 07 00 00 00 C7 83 83 C0 13 EB 0B 58 EB 02 CD 20 83 C0 02 EB 01 E9 50 C3 }
	condition:
		$1 at pe.entry_point
}

rule vterminal_10x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vterminal"
		version = "1.0x"
	strings:
		$1 = { E8 00 00 00 00 58 05 ?? ?? ?? ?? 9C 50 C2 04 00 }
	condition:
		$1 at pe.entry_point
}

rule vx_acme {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "ACME (Clonewar Mutant)"
	strings:
		$1 = { FC AD 3D FF FF 74 20 E6 42 8A C4 E6 42 E4 61 0C 03 E6 61 AD B9 40 1F E2 FE }
	condition:
		$1 at pe.entry_point
}

rule vx_arcv4 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "ARCV.4"
	strings:
		$1 = { E8 00 00 5D 81 ED 06 01 81 FC 4F 50 74 0B 8D B6 86 01 BF 00 01 57 A4 EB 11 1E 06 }
	condition:
		$1 at pe.entry_point
}

rule vx_august_16th {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "August 16th (Iron Maiden)"
	strings:
		$1 = { BA 79 02 03 D7 B4 1A CD 21 B8 24 35 CD 21 5F 57 89 9D 4E 02 8C 85 50 02 }
	condition:
		$1 at pe.entry_point
}

rule vx_backfont900 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Backfont.900"
	strings:
		$1 = { E8 ?? ?? B4 30 CD 21 3C 03 ?? ?? B8 ?? ?? BA ?? ?? CD 21 81 FA ?? ?? ?? ?? BA ?? ?? 8C C0 48 8E C0 8E D8 80 ?? ?? ?? 5A ?? ?? 03 ?? ?? ?? 40 8E D8 80 ?? ?? ?? 5A ?? ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule vx_caz1024 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Caz.1204"
	strings:
		$1 = { E8 ?? ?? 5E 83 EE 03 1E 06 B8 FF FF CD 2F 3C 10 }
	condition:
		$1 at pe.entry_point
}

rule vx_cih_v12_ttit {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "CIH Version 1.2 TTIT (! WIN95CIH !)"
	strings:
		$1 = { 55 8D ?? ?? ?? 33 DB 64 87 03 E8 ?? ?? ?? ?? 5B 8D }
	condition:
		$1 at pe.entry_point
}

rule vx_compiler {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Compiler"
	strings:
		$1 = { 8C C3 83 C3 10 2E 01 1E ?? 02 2E 03 1E ?? 02 53 1E }
	condition:
		$1 at pe.entry_point
}

rule vx_danish_tiny {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Danish tiny"
	strings:
		$1 = { 33 C9 B4 4E CD 21 73 02 FF ?? BA ?? 00 B8 ?? 3D CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_doom666 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Doom.666"
	strings:
		$1 = { E8 ?? ?? ?? 5E 83 EE ?? B8 CF 7B CD 21 3D CF 7B ?? ?? 0E 1F 81 C6 ?? ?? BF ?? ?? B9 ?? ?? FC F3 A4 06 1F 06 B8 ?? ?? 50 CB B4 48 BB 2C 00 CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_eddie1028 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Eddie.1028"
	strings:
		$1 = { E8 ?? ?? 5E FC 83 ?? ?? 81 ?? ?? ?? 4D 5A ?? ?? FA 8B E6 81 C4 ?? ?? FB 3B ?? ?? ?? ?? ?? 50 06 56 1E B8 FE 4B CD 21 81 FF BB 55 ?? ?? 07 ?? ?? ?? 07 B4 49 CD 21 BB FF FF B4 48 CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_eddie1530 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Eddie.1530"
	strings:
		$1 = { E8 ?? ?? 5E 81 EE ?? ?? FC 2E ?? ?? ?? ?? 4D 5A ?? ?? FA 8B E6 81 C4 ?? ?? FB 3B ?? ?? ?? ?? ?? 2E ?? ?? ?? ?? 50 06 56 1E 33 C0 50 1F C4 ?? ?? ?? 2E ?? ?? ?? ?? 2E }
	condition:
		$1 at pe.entry_point
}

rule vx_eddie1800 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Eddie.1800"
	strings:
		$1 = { E8 ?? ?? 5E 81 EE ?? ?? FC 2E ?? ?? ?? ?? 4D 5A ?? ?? FA 8B E6 81 C4 ?? ?? FB 3B ?? ?? ?? ?? ?? 50 06 56 1E 8B FE 33 C0 50 8E D8 C4 ?? ?? ?? 2E ?? ?? ?? ?? 2E }
	condition:
		$1 at pe.entry_point
}

rule vx_eddie2000 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Eddie.2000"
	strings:
		$1 = { E8 ?? ?? 5E 81 EE ?? ?? FC 2E ?? ?? ?? ?? 2E ?? ?? ?? ?? 4D 5A ?? ?? FA 8B E6 81 C4 ?? ?? FB 3B ?? ?? ?? ?? ?? 50 06 56 1E 8B FE 33 C0 50 8E D8 C5 ?? ?? ?? B4 30 CD 21 }
	condition:
		$1 at pe.entry_point
}
rule vx_eddie2100 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Eddie.2100"
	strings:
		$1 = { E8 ?? ?? 4F 4F 0E E8 ?? ?? 47 47 1E FF ?? ?? CB E8 ?? ?? 84 C0 ?? ?? 50 53 56 57 1E 06 B4 51 CD 21 8E C3 ?? ?? ?? ?? ?? ?? ?? 8B F2 B4 2F CD 21 AC }
	condition:
		$1 at pe.entry_point
}

rule vx_eddiebased1745 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Eddie.based.1745"
	strings:
		$1 = { E8 ?? ?? 5E 81 EE ?? ?? FC ?? 2E ?? ?? ?? ?? 4D 5A ?? ?? FA ?? 8B E6 81 ?? ?? ?? FB ?? 3B ?? ?? ?? ?? ?? 50 06 ?? 56 1E 8B FE 33 C0 ?? 50 8E D8 }
	condition:
		$1 at pe.entry_point
}

rule vx_einstein {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Einstein"
	strings:
		$1 = { 00 42 CD 21 72 31 B9 6E 03 33 D2 B4 40 CD 21 72 19 3B C1 75 15 B8 00 42 }
	condition:
		$1 at pe.entry_point
}

rule vx_explosion1000 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Explosion.1000"
	strings:
		$1 = { E8 ?? ?? 5E 1E 06 50 81 ?? ?? ?? 56 FC B8 21 35 CD 21 2E ?? ?? ?? ?? 2E ?? ?? ?? ?? 26 ?? ?? ?? ?? ?? ?? 74 ?? 8C D8 48 8E D8 }
	condition:
		$1 at pe.entry_point
}

rule vx_faxfree_topo {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "FaxFree.Topo"
	strings:
		$1 = { FA 06 33 C0 8E C0 B8 ?? ?? 26 ?? ?? ?? ?? 50 8C C8 26 ?? ?? ?? ?? 50 CC 58 9D 58 26 ?? ?? ?? ?? 58 26 ?? ?? ?? ?? 07 FB }
	condition:
		$1 at pe.entry_point
}

rule vx_gotcha879 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Gotcha.879"
	strings:
		$1 = { E8 ?? ?? 5B 81 EB ?? ?? 9C FC 2E ?? ?? ?? ?? ?? ?? ?? 8C D8 05 ?? ?? 2E ?? ?? ?? ?? 50 2E ?? ?? ?? ?? ?? ?? 8B C3 05 ?? ?? 8B F0 BF 00 01 B9 20 00 F3 A4 0E B8 00 01 50 B8 DA DA CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_grazzie883 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Grazie.883"
	strings:
		$1 = { 1E 0E 1F 50 06 BF 70 03 B4 1A BA 70 03 CD 21 B4 47 B2 00 BE 32 04 CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_grunt1family {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "GRUNT.1.Family"
	strings:
		$1 = { 01 B9 ?? 00 31 17 }
	condition:
		$1 at pe.entry_point
}

rule vx_grunt2family {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "GRUNT.2.Family"
	strings:
		$1 = { 48 E2 F7 C3 51 53 52 E8 DD FF 5A 5B 59 C3 B9 00 00 E2 FE C3 }
	condition:
		$1 at pe.entry_point
}

rule vx_grunt4family {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "GRUNT.4.Family"
	strings:
		$1 = { E8 1C 00 8D 9E 41 01 40 3E 8B 96 14 03 B9 EA 00 87 DB F7 D0 31 17 83 C3 02 E2 F7 C3 }
	condition:
		$1 at pe.entry_point
}

rule vx_hafen1641 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Hafen.1641"
	strings:
		$1 = { E8 ?? ?? 01 ?? ?? ?? CE CC 25 ?? ?? 25 ?? ?? 25 ?? ?? 40 51 D4 ?? ?? ?? CC 47 CA ?? ?? 46 8A CC 44 88 CC }
	condition:
		$1 at pe.entry_point
}

rule vx_hafen809 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Hafen.809"
	strings:
		$1 = { E8 ?? ?? 1C ?? 81 EE ?? ?? 50 1E 06 8C C8 8E D8 06 33 C0 8E C0 26 ?? ?? ?? 07 3D }
	condition:
		$1 at pe.entry_point
}

rule vx_harynato {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Haryanto"
	strings:
		$1 = { 81 EB 2A 01 8B 0F 1E 5B 03 CB 0E 51 B9 10 01 51 CB }
	condition:
		$1 at pe.entry_point
}

rule vx_helloweem1172 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Heloween.1172"
	strings:
		$1 = { E8 ?? ?? 5E 81 EE ?? ?? 56 50 06 0E 1F 8C C0 01 ?? ?? 01 ?? ?? 80 ?? ?? ?? ?? 8B ?? ?? A3 ?? ?? 8A ?? ?? A2 ?? ?? B8 ?? ?? CD 21 3D }
	condition:
		$1 at pe.entry_point
}

rule vx_horse1776 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Horse.1776"
	strings:
		$1 = { E8 ?? ?? 5D 83 ?? ?? 06 1E 26 ?? ?? ?? ?? BF ?? ?? 1E 0E 1F 8B F7 01 EE B9 ?? ?? FC F3 A6 1F 1E 07 }
	condition:
		$1 at pe.entry_point
}

rule vx_hymn1865 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Hymn.1865"
	strings:
		$1 = { E8 ?? ?? 5E 83 EE 4C FC 2E ?? ?? ?? ?? 4D 5A ?? ?? FA 8B E6 81 ?? ?? ?? FB 3B ?? ?? ?? ?? ?? 2E ?? ?? ?? ?? ?? 50 06 56 1E 0E 1F B8 00 C5 CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_igor {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Igor"
	strings:
		$1 = { 1E B8 CD 7B CD 21 81 FB CD 7B 75 03 E9 87 00 33 DB 0E 1F 8C }
	condition:
		$1 at pe.entry_point
}
rule vx_involuntary1349 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Involuntary.1349"
	strings:
		$1 = { ?? BA ?? ?? B9 ?? ?? 8C DD ?? 8C C8 ?? 8E D8 8E C0 33 F6 8B FE FC ?? ?? AD ?? 33 C2 AB }
	condition:
		$1 at pe.entry_point
}

rule vx_kbdflags1024 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "KBDflags.1024"
	strings:
		$1 = { 8B EC 2E 89 2E 24 03 BC 00 04 8C D5 2E 89 2E 22 }
	condition:
		$1 at pe.entry_point
}

rule vx_keypress1212 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Keypress.1212"
	strings:
		$1 = { E8 ?? ?? E8 ?? ?? E8 ?? ?? E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? EA ?? ?? ?? ?? 1E 33 DB 8E DB BB }
	condition:
		$1 at pe.entry_point
}

rule vx_kuku448 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Kuku.448"
	strings:
		$1 = { AE 75 ED E2 F8 89 3E ?? ?? BA ?? ?? 0E 07 BF ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule vx_kuku886 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Kuku.886"
	strings:
		$1 = { 06 1E 50 8C C8 8E D8 BA 70 03 B8 24 25 CD 21 ?? ?? ?? ?? ?? 90 B4 2F CD 21 53 }
	condition:
		$1 at pe.entry_point
}

rule vx_hi924_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Modification of Hi.924"
	strings:
		$1 = { 50 53 51 52 1E 06 9C B8 21 35 CD 21 53 BB ?? ?? 26 ?? ?? 49 48 5B }
	condition:
		$1 at pe.entry_point
}

rule vx_mte {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "MTE (non-encrypted)"
	strings:
		$1 = { F7 D9 80 E1 FE 75 02 49 49 97 A3 ?? ?? 03 C1 24 FE 75 02 48 }
	condition:
		$1 at pe.entry_point
}

rule vx_nculi_1688 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Ncu-Li.1688"
	strings:
		$1 = { 0E 1E B8 55 AA CD 21 3D 49 4C 74 ?? 0E 0E 1F 07 E8 }
	condition:
		$1 at pe.entry_point
}

rule vx_necropolis {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Necropolis"
	strings:
		$1 = { 50 FC AD 33 C2 AB 8B D0 E2 F8 }
	condition:
		$1 at pe.entry_point
}

rule vx_necropolis1963 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Necropolis.1963"
	strings:
		$1 = { B4 30 CD 21 3C 03 ?? ?? B8 00 12 CD 2F 3C FF B8 ?? ?? ?? ?? B4 4A BB 40 01 CD 21 ?? ?? FA 0E 17 BC ?? ?? E8 ?? ?? FB A1 ?? ?? 0B C0 }
	condition:
		$1 at pe.entry_point
}

rule vx_noon1163 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Noon.1163"
	strings:
		$1 = { E8 ?? ?? 5B 50 56 B4 CB CD 21 3C 07 ?? ?? 81 ?? ?? ?? 2E ?? ?? 4D 5A ?? ?? BF 00 01 89 DE FC }
	condition:
		$1 at pe.entry_point
}

rule vx_november_17768 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "November 17.768"
	strings:
		$1 = { E8 ?? ?? 5E 81 EE ?? ?? 50 33 C0 8E D8 80 3E ?? ?? ?? 0E 1F ?? ?? FC }
	condition:
		$1 at pe.entry_point
}

rule vx_number_one {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Number One"
	strings:
		$1 = { F9 07 3C 53 6D 69 6C 65 3E E8 }
	condition:
		$1 at pe.entry_point
}

rule vx_phoenix_927 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Phoenix.927"
	strings:
		$1 = { E8 00 00 5E 81 C6 ?? ?? BF 00 01 B9 04 00 F3 A4 E8 }
	condition:
		$1 at pe.entry_point
}

rule vx_predator2448 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Predator.2448"
	strings:
		$1 = { 0E 1F BF ?? ?? B8 ?? ?? B9 ?? ?? 49 ?? ?? ?? ?? 2A C1 4F 4F ?? ?? F9 CC }
	condition:
		$1 at pe.entry_point
}

rule vx_quake518 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Quake.518"
	strings:
		$1 = { 1E 06 8C C8 8E D8 ?? ?? ?? ?? ?? ?? ?? B8 21 35 CD 21 81 }
	condition:
		$1 at pe.entry_point
}

rule vx_sk {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "SK"
	strings:
		$1 = { CD 20 B8 03 00 CD 10 51 E8 00 00 5E 83 EE 09 }
	condition:
		$1 at pe.entry_point
}

rule vx_slowload {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Slowload"
	strings:
		$1 = { 03 D6 B4 40 CD 21 B8 02 42 33 D2 33 C9 CD 21 8B D6 B9 78 01 }
	condition:
		$1 at pe.entry_point
}

rule vx_sonik_youth {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Sonik Youth"
	strings:
		$1 = { 8A 16 02 00 8A 07 32 C2 88 07 43 FE C2 81 FB }
	condition:
		$1 at pe.entry_point
}

rule vx_spanz {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Spanz"
	strings:
		$1 = { E8 00 00 5E 81 EE ?? ?? 8D 94 ?? ?? B4 1A CD 21 C7 84 }
	condition:
		$1 at pe.entry_point
}

rule vx_syp {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "SYP"
	strings:
		$1 = { 47 8B C2 05 1E 00 52 8B D0 B8 02 3D CD 21 8B D8 5A }
	condition:
		$1 at pe.entry_point
}

rule vx_tibs_zhelatin {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "VX:"
		version = "Tibs/Zhelatin (StormWorm) variant"
	strings:
		$1 = { FF 74 24 1C 58 8D 80 ?? ?? 77 04 50 68 62 34 35 04 E8 }
	condition:
		$1 at pe.entry_point
}

rule vx_travjack883 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "TravJack.883"
	strings:
		$1 = { EB ?? 9C 9E 26 ?? ?? 51 04 ?? 7D ?? 00 ?? 2E ?? ?? ?? ?? 8C C8 8E C0 8E D8 80 ?? ?? ?? ?? 74 ?? 8A ?? ?? ?? BB ?? ?? 8A ?? 32 C2 88 ?? FE C2 43 81 }
	condition:
		$1 at pe.entry_point
}

rule vx_trivial25 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Trivial.25"
	strings:
		$1 = { B4 4E FE C6 CD 21 B8 ?? 3D BA ?? 00 CD 21 93 B4 40 CD }
	condition:
		$1 at pe.entry_point
}

rule vx_trivial46 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Trivial.46"
	strings:
		$1 = { B4 4E B1 20 BA ?? ?? CD 21 BA ?? ?? B8 ?? 3D CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_trojanteflon {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Trojan.Telefoon"
	strings:
		$1 = { 60 1E E8 3B 01 BF CC 01 2E 03 3E CA 01 2E C7 05 }
	condition:
		$1 at pe.entry_point
}

rule vx_uddy2617 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "Uddy.2617"
	strings:
		$1 = { 2E ?? ?? ?? ?? ?? 2E ?? ?? ?? ?? ?? 2E ?? ?? ?? 8C C8 8E D8 8C ?? ?? ?? 2B ?? ?? ?? 03 ?? ?? ?? A3 ?? ?? A1 ?? ?? A3 ?? ?? A1 ?? ?? A3 ?? ?? 8C C8 2B ?? ?? ?? 03 ?? ?? ?? A3 ?? ?? B8 AB 9C CD 2F 3D 76 98 }
	condition:
		$1 at pe.entry_point
}

rule vx_vcl_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "VCL (encrypted)"
	strings:
		$1 = { 01 B9 ?? ?? 81 34 ?? ?? 46 46 E2 F8 C3 }
	condition:
		$1 at pe.entry_point
}

rule vx_vcl_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "VCL (encrypted)"
	strings:
		$1 = { 01 B9 ?? ?? 81 35 ?? ?? 47 47 E2 F8 C3 }
	condition:
		$1 at pe.entry_point
}

rule vx_vcl_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "VCL"
	strings:
		$1 = { AC B9 00 80 F2 AE B9 04 00 AC AE 75 ?? E2 FA 89 }
	condition:
		$1 at pe.entry_point
}

rule vx_virusconstructor_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "VirusConstructor(IVP).based"
	strings:
		$1 = { E9 ?? ?? E8 ?? ?? 5D ?? ?? ?? ?? ?? 81 ED ?? ?? ?? ?? ?? ?? E8 ?? ?? 81 FC ?? ?? ?? ?? 8D ?? ?? ?? BF ?? ?? 57 A4 A5 }
	condition:
		$1 at pe.entry_point
}

rule vx_virusconstructor_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "VirusConstructor.based"
	strings:
		$1 = { BB ?? ?? B9 ?? ?? 2E ?? ?? ?? ?? 43 43 ?? ?? 8B EC CC 8B ?? ?? 81 ?? ?? ?? 06 1E B8 ?? ?? CD 21 3D ?? ?? ?? ?? 8C D8 48 8E D8 }
	condition:
		$1 at pe.entry_point
}

rule vx_vx_virusconstructor_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "VirusConstructor.based"
	strings:
		$1 = { E8 ?? ?? 5D 81 ?? ?? ?? 06 1E E8 ?? ?? E8 ?? ?? ?? ?? 2E ?? ?? ?? ?? ?? ?? B4 4A BB FF FF CD 21 83 ?? ?? B4 4A CD 21 }
	condition:
		$1 at pe.entry_point
}

rule vx_xpeh4768 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "XPEH.4768"
	strings:
		$1 = { E8 ?? ?? 5B 81 ?? ?? ?? 50 56 57 2E ?? ?? ?? ?? ?? 2E ?? ?? ?? ?? ?? ?? B8 01 00 50 B8 ?? ?? 50 E8 }
	condition:
		$1 at pe.entry_point
}

rule vx_xrcv1015 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Vx:"
		version = "XRCV.1015"
	strings:
		$1 = { E8 ?? ?? 5E 83 ?? ?? 53 51 1E 06 B4 99 CD 21 80 FC 21 ?? ?? ?? ?? ?? 33 C0 50 8C D8 48 8E C0 1F A1 ?? ?? 8B }
	condition:
		$1 at pe.entry_point
}

rule webcops_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "WebCops"
	strings:
		$1 = { A8 BE 58 DC D6 CC C4 63 4A 0F E0 02 BB CE F3 5C 50 23 FB 62 E7 3D 2B }
	condition:
		$1 at pe.entry_point
}

rule webcops_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "WebCops"
	strings:
		$1 = { EB 03 05 EB 02 EB FC 55 EB 03 EB 04 05 EB FB EB 53 E8 04 00 00 00 72 }
	condition:
		$1 at pe.entry_point
}

rule werus_crypter_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Werus Crypter"
		version = "1.0"
	strings:
		$1 = { 68 98 11 40 00 6A 00 E8 50 00 00 00 C9 C3 ED B3 FE FF FF 6A 00 E8 0C 00 00 00 FF 25 80 10 40 00 FF 25 84 10 40 00 FF 25 88 10 40 00 FF 25 8C 10 40 00 FF 25 90 10 40 00 FF 25 94 10 40 00 FF 25 98 10 40 00 FF 25 9C 10 40 00 FF 25 A0 10 40 00 FF 25 A4 10 40 00 FF 25 A8 10 40 00 FF 25 B0 10 40 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 BB E8 12 40 00 80 33 05 E9 7D FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule werus_crypter_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Werus Crypter"
		version = "1.0"
	strings:
		$1 = { BB E8 12 40 00 80 33 05 E9 7D FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule winu_key_410a {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "WIBU-Key"
		version = "4.10a"
	strings:
		$1 = { F7 05 ?? ?? ?? ?? FF 00 00 00 75 12 }
	condition:
		$1 at pe.entry_point
}

rule wind_of_crypt_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Wind of Crypt"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 EC 53 ?? ?? ?? ?? 89 45 EC B8 64 40 00 10 E8 28 EA FF FF 33 C0 55 68 CE 51 00 10 64 ?? ?? ?? ?? 20 6A 00 68 80 00 00 00 6A 03 6A 00 6A 01 68 00 00 00 80 8D 55 EC 33 C0 E8 F6 DB FF FF 8B 45 EC E8 12 E7 FF FF 50 E8 3C EA FF FF 8B D8 83 FB FF 0F 84 A6 00 00 00 6A 00 53 E8 41 EA FF FF 8B F0 81 EE 00 5E 00 00 6A 00 6A 00 68 00 5E 00 00 53 E8 52 EA FF FF B8 F4 97 00 10 8B D6 E8 2E E7 FF FF B8 F8 97 00 10 8B D6 E8 22 E7 FF FF 8B C6 E8 AB D8 FF FF 8B F8 6A 00 68 F0 97 00 10 56 A1 F4 97 00 10 50 53 E8 05 EA FF FF 53 E8 CF E9 FF FF B8 FC 97 00 10 BA E8 51 00 10 E8 74 EA FF FF A1 F4 97 00 10 85 C0 74 05 83 E8 04 8B 00 50 B9 F8 97 00 10 B8 FC 97 00 10 8B 15 F4 97 00 10 E8 D8 EA FF FF B8 FC 97 00 10 E8 5A EB FF FF 8B CE 8B 15 F8 97 00 10 8B C7 E8 EB E9 FF FF 8B C7 85 C0 74 05 E8 E4 EB FF FF 33 C0 5A 59 59 64 89 10 68 D5 51 00 10 8D 45 EC E8 BB E5 FF FF C3 E9 A9 DF FF FF EB F0 5F 5E 5B E8 B7 E4 FF FF 00 00 00 FF FF FF FF 0A 00 00 00 63 5A 6C 56 30 55 6C 6B 70 4D }
	condition:
		$1 at pe.entry_point
}

rule winkript_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Winkript"
		version = "1.0"
	strings:
		$1 = { 33 C0 8B B8 00 ?? ?? ?? 8B 90 04 ?? ?? ?? 85 FF 74 1B 33 C9 50 EB 0C 8A 04 39 C0 C8 04 34 1B 88 04 39 41 3B CA 72 F0 58 }
	condition:
		$1 at pe.entry_point
}

rule wwpack32_1x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "WWPack32"
		version = "1.x"
	strings:
		$1 = { 53 55 8B E8 33 DB EB 60 }
	condition:
		$1 at pe.entry_point
}

rule xhider_10_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-Hider"
		version = "1.0"
	strings:
		$1 = { 55 8B EC 83 C4 EC 33 C0 89 45 EC B8 54 20 44 44 E8 DF F8 FF FF 33 C0 55 68 08 21 44 44 64 FF 30 64 89 20 8D 55 EC B8 1C 21 44 44 E8 E0 F9 FF FF 8B 55 EC B8 40 ?? ?? 44 E8 8B F5 FF FF 6A 00 6A 00 6A 02 6A 00 6A 01 68 00 00 00 40 A1 40 ?? ?? 44 E8 7E F6 FF FF 50 E8 4C F9 FF FF 6A 00 50 E8 4C F9 FF FF A3 28 ?? ?? 44 E8 CE FE FF FF 33 C0 5A 59 59 64 89 10 68 0F 21 44 44 8D 45 EC E8 F1 F4 FF FF C3 E9 BB F2 FF FF EB F0 E8 FC F3 FF FF FF FF FF FF 0E 00 00 00 63 3A 5C 30 30 30 30 30 30 31 2E 64 61 74 00 }
	condition:
		$1 at pe.entry_point
}

rule xhider_10_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-Hider"
		version = "1.0"
	strings:
		$1 = { 85 D2 74 23 8B 4A F8 41 7F 1A 50 52 8B 42 FC E8 30 00 00 00 89 C2 58 52 8B 48 FC E8 48 FB FF FF 5A 58 EB 03 FF 42 F8 87 10 85 D2 74 13 8B 4A F8 49 7C 0D FF 4A F8 75 08 8D 42 F8 E8 5C FA FF FF C3 8D 40 00 85 C0 7E 24 50 83 C0 0A 83 E0 FE 50 E8 2F FA FF FF 5A 66 C7 44 02 FE 00 00 83 C0 08 5A 89 50 FC C7 40 F8 01 00 00 00 C3 31 C0 C3 90 }
	condition:
		$1 at pe.entry_point
}

rule xpack_142 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-Pack"
		version = "1.4.2"
	strings:
		$1 = { 72 ?? C3 8B DE 83 ?? ?? C1 ?? ?? 8C D8 03 C3 8E D8 8B DF 83 ?? ?? C1 ?? ?? 8C C0 03 C3 8E C0 C3 }
	condition:
		$1 at pe.entry_point
}

rule xpack_152_164 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-Pack"
		version = "1.52 - 1.64"
	strings:
		$1 = { 8B EC FA 33 C0 8E D0 BC ?? ?? 2E ?? ?? ?? ?? 2E ?? ?? ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule xpack_167 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-Pack"
		version = "1.67"
	strings:
		$1 = { B8 8C D3 15 33 75 81 3E E8 0F 00 9A E8 F9 FF 9A 9C EB 01 9A 59 80 CD 01 51 9D EB }
	condition:
		$1 at pe.entry_point
}

rule xpeor_099b_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-PEOR"
		version = "0.99b"
	strings:
		$1 = { E8 ?? ?? ?? ?? 5D 8B CD 81 ED 7A 29 40 ?? 89 AD 0F 6D 40 }
	condition:
		$1 at pe.entry_point
}

rule xpeor_099b_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "X-PEOR"
		version = "0.99b"
	strings:
		$1 = { E8 00 00 00 00 5D 8B CD 81 ED 7A 29 40 00 89 AD 0F 6D 40 00 }
	condition:
		$1 at pe.entry_point
}

rule xcomp_xpack_097_098_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XComp/XPack"
		version = "0.97 - 0.98"
	strings:
		$1 = { 68 ?? ?? ?? ?? 9C 60 E8 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 }
	condition:
		$1 at pe.entry_point
}

rule xcomp_xpack_097_098_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XComp/XPack"
		version = "0.97 - 0.98"
	strings:
		$1 = { 68 ?? ?? ?? ?? 9C 60 E8 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 4B 45 52 4E 45 4C 33 32 2E 44 4C 4C 00 00 00 47 65 74 50 72 6F 63 41 64 64 72 65 73 73 00 00 00 4C 6F 61 64 4C 69 62 72 61 72 79 41 00 00 00 56 69 72 74 75 61 6C 50 72 6F 74 65 63 74 00 }
	condition:
		$1 at pe.entry_point
}

rule xcomp_xpack_09x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XComp/XPack"
		version = "0.9x"
	strings:
		$1 = { AC 84 C0 74 03 AA EB F8 E8 0B 00 00 00 20 6E 6F 74 20 66 6F 75 6E 64 00 5E AC AA 84 C0 75 FA 6A 00 57 52 6A 00 E8 06 00 00 00 45 72 72 6F 72 00 5E AC AA 84 C0 75 FA E8 0B 00 00 00 55 53 45 52 33 32 2E 44 4C 4C 00 FF 55 2C E8 0C 00 00 00 4D 65 73 73 61 67 65 42 6F 78 41 00 50 FF 55 28 FF D0 83 C4 7C 48 C3 }
	condition:
		$1 at pe.entry_point
}

rule xcr_012 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XCR"
		version = "0.12"
	strings:
		$1 = { 60 9C E8 ?? ?? ?? ?? 8B DD 5D 81 ED ?? ?? ?? ?? 89 9D }
	condition:
		$1 at pe.entry_point
}

rule xcr_013 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XCR"
		version = "0.13"
	strings:
		$1 = { 93 71 08 ?? ?? ?? ?? ?? ?? ?? ?? 8B D8 78 E2 ?? ?? ?? ?? 9C 33 C3 ?? ?? ?? ?? 60 79 CE ?? ?? ?? ?? E8 01 ?? ?? ?? ?? 83 C4 04 E8 AB FF FF FF ?? ?? ?? ?? 2B E8 ?? ?? ?? ?? 03 C5 FF 30 ?? ?? ?? ?? C6 ?? EB }
	condition:
		$1 at pe.entry_point
}

rule xenocode_811353 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Xenocode"
		version = "8.1.1353"
	strings:
		$1 = { 55 8B EC 83 E4 F8 81 EC 1C 09 00 00 53 56 57 E8 87 FB FF FF 8B 35 0C ?0 ?? ?? FF D6 83 E0 11 3D 11 01 00 00 0F 84 26 04 00 00 FF D6 8B 5C 24 28 A3 0C 50 ?? ?? E8 53 FC FF FF 8B C8 2B 0D 0C 50 ?? ?? 6A 03 33 D2 8B C1 5E F7 F6 F7 C1 00 80 FF FF 0F 85 86 02 00 00 33 C0 33 FF 89 BC 24 24 09 00 00 66 89 }
	condition:
		$1 at pe.entry_point
}

rule xj_xpal_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XJ or XPAL"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? 40 00 68 ?? ?? 40 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 44 53 56 57 66 9C }
	condition:
		$1 at pe.entry_point
}

rule xpep_03x {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "xPEP"
		version = "0.3x"
	strings:
		$1 = { 55 53 56 51 52 57 E8 16 00 00 00 }
	condition:
		$1 at pe.entry_point
}


rule extreme_protector_106 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Xtreme-Protector"
		version = "1.06"
	strings:
		$1 = { B8 ?? ?? ?? 00 B9 75 ?? ?? 00 50 51 E8 05 00 00 00 E9 4A 01 00 00 60 8B 74 24 24 8B 7C 24 28 FC B2 80 8A 06 46 88 07 47 BB 02 00 00 00 02 D2 75 05 8A 16 46 12 D2 73 EA 02 D2 75 05 8A 16 46 12 D2 73 4F 33 C0 02 D2 75 05 8A 16 46 12 D2 0F 83 DF 00 00 00 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A 16 46 12 D2 13 C0 74 06 57 2B F8 8A 07 5F 88 07 47 BB 02 00 00 00 EB 9B B8 01 00 00 00 02 D2 75 05 8A 16 46 12 D2 13 C0 02 D2 75 05 8A }
	condition:
		$1 at pe.entry_point
}

rule xtremlok_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XTREMLOK"
	strings:
		$1 = { 90 90 90 EB 29 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 53 54 41 54 49 43 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 52 53 51 56 57 55 E8 ?? ?? ?? ?? 5D 81 ED 36 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule xxpack_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "XXPack"
		version = "0.1"
	strings:
		$1 = { E8 04 00 00 00 83 60 EB 0C 5D EB 05 45 55 EB 04 B8 EB F9 00 C3 E8 00 00 00 00 5D EB 01 00 81 ED 5E 1F 40 00 EB 02 83 09 8D B5 EF 1F 40 00 EB 02 83 09 BA A3 11 00 00 EB 00 68 00 ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.0"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED E7 1A 40 00 E8 A1 00 00 00 E8 D1 00 00 00 E8 85 01 00 00 F7 85 }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.1"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED 8A 1C 40 00 B9 9E 00 00 00 8D BD 4C 23 40 00 8B F7 33 }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.2"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED F3 1D 40 00 B9 7B 09 00 00 8D BD 3B 1E 40 00 8B F7 AC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? AA E2 CC }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_13_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.3"
	strings:
		$1 = { 55 8B EC 53 56 57 60 E8 00 00 00 00 5D 81 ED 6C 28 40 00 B9 5D 34 40 00 81 E9 C6 28 40 00 8B D5 81 C2 C6 28 40 00 8D 3A 8B F7 33 C0 EB 04 90 EB 01 C2 AC }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_13_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.3"
	strings:
		$1 = { 55 8B EC 53 56 57 60 E8 00 00 00 00 5D 81 ED 8C 21 40 00 B9 51 2D 40 00 81 E9 E6 21 40 00 8B D5 81 C2 E6 21 40 00 8D 3A 8B F7 33 C0 EB 04 90 EB 01 C2 AC }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_13_03 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.3"
	strings:
		$1 = { 55 8B EC 81 EC C0 00 00 00 53 56 57 8D BD 40 FF FF FF B9 30 00 00 00 B8 CC CC CC CC F3 AB 60 E8 00 00 00 00 5D 81 ED 84 52 41 00 B9 75 5E 41 00 81 E9 DE 52 41 00 8B D5 81 C2 DE 52 41 00 8D 3A 8B F7 33 C0 EB 04 90 EB 01 C2 AC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? AA E2 CC }
	condition:
		$1 at pe.entry_point
}

rule yodas_crypter_1x_modf {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Crypter"
		version = "1.x modified"
	strings:
		$1 = { 60 E8 00 00 00 00 5D 81 ED ?? ?? ?? ?? B9 ?? ?? 00 00 8D BD ?? ?? ?? ?? 8B F7 AC }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_uv_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		start = 96
	strings:
		$1 = { E8 00 00 00 00 5D 81 ED ?? ?? 42 00 8B D5 81 C2 ?? ?? 42 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC }
	condition:
		$1 at pe.entry_point + 96
}

rule yodas_protector_uv_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
	strings:
		$1 = { E8 ?? ?? ?? 00 EB 01 E? ?? ?? ?? 00 00 E8 ?? ?? ?? 00 EB 01 ?? ?? ?? ?? ?? 00 E8 ?? ?? ?? 00 EB 01 ?? ?? ?? ?? ?? 00 E8 ?? ?? ?? 00 EB 01 ?? ?? ?? ?? 00 00 E8 ?? ?? ?? 00 EB 01 }
	condition:
		$1 in (pe.entry_point .. pe.entry_point + 12)
}

rule yodas_protector_10b {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.0b"
	strings:
		$1 = { 55 8B EC 53 56 57 60 E8 00 00 00 00 5D 81 ED 4C 32 40 00 E8 03 00 00 00 EB 01 ?? B9 EA 47 40 00 81 E9 E9 32 40 00 8B D5 81 C2 E9 32 40 00 8D 3A 8B F7 33 C0 E8 04 00 00 00 90 EB 01 ?? E8 03 00 00 00 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.01"
	strings:
		$1 = { 55 8B EC 53 56 57 E8 03 00 00 00 EB 01 ?? E8 86 00 00 00 E8 03 00 00 00 EB 01 ?? E8 79 00 00 00 E8 03 00 00 00 EB 01 ?? E8 A4 00 00 00 E8 03 00 00 00 EB 01 ?? E8 97 00 00 00 E8 03 00 00 00 EB 01 ?? E8 2D 00 00 00 E8 03 00 00 00 EB 01 ?? 60 E8 00 00 00 00 5D 81 ED }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_102 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.02"
	strings:
		$1 = { E8 03 00 00 00 EB 01 ?? BB 55 00 00 00 E8 03 00 00 00 EB 01 ?? E8 8F 00 00 00 E8 03 00 00 00 EB 01 ?? E8 82 00 00 00 E8 03 00 00 00 EB 01 ?? E8 B8 00 00 00 E8 03 00 00 00 EB 01 ?? E8 AB 00 00 00 E8 03 00 00 00 EB 01 ?? 83 FB 55 E8 03 00 00 00 EB 01 ?? 75 2E E8 03 00 00 00 EB 01 ?? C3 60 E8 00 00 00 00 5D 81 ED 23 3F 42 00 8B D5 81 C2 72 3F 42 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 4B CC C3 E8 03 00 00 00 EB 01 ?? 33 DB B9 3A 66 42 00 81 E9 1D 40 42 00 8B D5 81 C2 1D 40 42 00 8D 3A 8B F7 33 C0 E8 03 00 00 00 EB 01 ?? E8 17 00 00 00 90 90 90 E9 C3 1F 00 00 33 C0 64 FF 30 64 89 20 43 CC C3 }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_102_dll_ocx {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.02 DLL or OCX"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 60 E8 00 00 00 00 5D 81 ED 23 3F 42 00 8B D5 81 C2 72 3F 42 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_1033 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.03.3"
	strings:
		$1 = { E8 03 00 00 00 EB 01 ?? BB 55 00 00 00 E8 03 00 00 00 EB 01 ?? E8 8E 00 00 00 E8 03 00 00 00 EB 01 ?? E8 81 00 00 00 E8 03 00 00 00 EB 01 ?? E8 B7 00 00 00 E8 03 00 00 00 EB 01 ?? E8 AA 00 00 00 E8 03 00 00 00 EB 01 ?? 83 FB 55 E8 03 00 00 00 EB 01 ?? 75 2D E8 03 00 00 00 EB 01 ?? 60 E8 00 00 00 00 5D 81 ED 07 E2 40 00 8B D5 81 C2 56 E2 40 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 4B CC C3 }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_1032_dll_ocx_01 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.03.2 DLL or OCX"
	strings:
		$1 = { ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 60 E8 00 00 00 00 5D 81 ED 94 73 42 00 8B D5 81 C2 E3 73 42 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB }
	condition:
		$1 at pe.entry_point
}

rule yodas_protector_1032_dll_ocx_02 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yoda's Protector"
		version = "1.03.3 DLL or OCX"
	strings:
		$1 = { 60 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 00 00 00 00 5D 81 ED 07 E2 40 00 8B D5 81 C2 56 E2 40 00 52 E8 01 00 00 00 C3 C3 E8 03 00 00 00 EB 01 ?? E8 0E 00 00 00 E8 D1 FF FF FF C3 E8 03 00 00 00 EB 01 ?? 33 C0 64 FF 30 64 89 20 CC C3 E8 03 00 00 00 EB 01 }
	condition:
		$1 at pe.entry_point
}

rule yzpack_11 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yzpack"
		version = "1.1"
	strings:
		$1 = { 60 33 C0 8D 48 07 50 E2 FD 8B EC 64 8B 40 30 78 0C 8B 40 0C 8B 70 1C AD 8B 40 08 EB 09 8B 40 34 8D 40 7C 8B 40 3C 89 45 04 E8 F3 07 00 00 60 8B 5D 04 8B 73 3C 8B 74 33 78 03 F3 56 8B 76 20 03 F3 33 C9 49 92 41 AD 03 C3 52 33 FF 0F B6 10 38 F2 }
	condition:
		$1 at pe.entry_point
}

rule yzpack_112 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yzpack"
		version = "1.12"
	strings:
		$1 = { 5A 52 45 60 83 EC 18 8B EC 8B FC 33 C0 64 8B 40 30 78 0C 8B 40 0C 8B 70 1C AD 8B 40 08 EB 09 8B 40 34 83 C0 7C 8B 40 3C AB E9 ?? ?? ?? ?? B4 09 BA 00 00 1F CD 21 B8 01 4C CD 21 40 00 00 00 50 45 00 00 4C 01 02 00 ?? ?? ?? ?? 00 00 00 00 00 00 00 00 E0 00 ?? ?? 0B 01 ?? ?? ?? ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule yzpack_12 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yzpack"
		version = "1.2"
	strings:
		$1 = { 4D 5A 52 45 60 83 EC 18 8B EC 8B FC 33 C0 64 8B 40 30 78 0C 8B 40 0C 8B 70 1C AD 8B 40 08 EB 09 8B 40 34 83 C0 7C 8B 40 3C AB E9 }
	condition:
		$1 at pe.entry_point
}

rule yzpack_20 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "yzpack"
		version = "2.0"
	strings:
		$1 = { 25 ?? ?? ?? ?? 61 87 CC 55 45 45 55 81 ED CA 00 00 00 55 A4 B3 02 FF 14 24 73 F8 33 C9 FF 14 24 73 18 33 C0 FF 14 24 73 1F B3 02 41 B0 10 FF 14 24 12 C0 73 F9 75 3C AA EB DC FF 54 24 04 2B CB 75 0F FF 54 24 08 EB 27 AC D1 E8 74 30 13 C9 EB 1B 91 48 C1 E0 08 AC FF 54 24 08 3D 00 7D 00 00 73 0A 80 FC 05 73 06 83 F8 7F 77 02 41 41 95 8B C5 B3 01 56 8B F7 2B F0 F3 A4 5E EB 99 BD ?? ?? ?? ?? FF 65 28 }
	condition:
		$1 at pe.entry_point
}

rule zcode_101 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ZCode"
		version = "1.01"
	strings:
		$1 = { E9 12 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E9 FB FF FF FF C3 68 ?? ?? ?? ?? 64 FF 35 }
	condition:
		$1 at pe.entry_point
}

rule zealpack_10 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ZealPack"
		version = "1.0"
	strings:
		$1 = { C7 45 F4 00 00 40 00 C7 45 F0 ?? ?? ?? ?? 8B 45 F4 05 ?? ?? ?? ?? 89 45 F4 C7 45 FC 00 00 00 00 EB 09 8B 4D FC 83 C1 01 89 4D FC 8B 55 FC 3B 55 F0 7D 22 8B 45 F4 03 45 FC 8A 08 88 4D F8 0F BE 55 F8 83 F2 0F 88 55 F8 8B 45 F4 03 45 FC 8A 4D F8 88 08 EB CD FF 65 F4 }
	condition:
		$1 at pe.entry_point
}

rule zipworxsecureexe_25 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "ZipWorxSecureEXE"
		version = "2.5"
	strings:
		$1 = { E9 B8 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 53 65 63 75 72 65 45 58 45 20 45 78 65 63 75 74 61 62 6C 65 20 46 69 6C 65 20 50 72 6F 74 65 63 74 6F 72 0D 0A 43 6F 70 79 72 69 67 68 74 28 63 29 20 32 30 30 34 2D 32 30 30 37 20 5A 69 70 57 4F 52 58 20 54 65 63 68 6E 6F 6C 6F 67 69 65 73 2C 20 4C 4C 43 0D 0A 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 20 28 63 29 20 31 39 39 37 2D 32 30 30 31 20 4C 65 65 20 48 61 73 69 75 6B 0D 0A 41 6C }
	condition:
		$1 at pe.entry_point
}

rule zprotect_120_130 {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "Zprotect"
		version = "1.2.0 - 1.3.0"
	strings:
		$1 = { 00 00 00 00 00 00 00 00 2E 74 65 78 74 62 73 73 ?? ?? ?? ?? 00 10 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 E0 2E 74 65 78 74 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 20 00 00 E0 2E 64 61 74 61 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 40 2E 69 64 61 74 61 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 40 00 00 C0 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 00 00 00 00 00 00 00 00 ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule the_best_cryptor_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "The Best Cryptor"
	strings:
		$1 = { EB 06 56 52 55 4C 5A 00 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule the_guard_library_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "The Guard Library"
	strings:
		$1 = { 50 E8 ?? ?? ?? ?? 58 25 ?? F0 FF FF 8B C8 83 C1 60 51 83 C0 40 83 EA 06 52 FF 20 9D C3 }
	condition:
		$1 at pe.entry_point
}

rule thehypers_protector_uv {
	meta:
        author = "RetDec Team"
		category = "packer"
		name = "TheHyper's protector"
	strings:
		$1 = { 55 8B EC 83 EC 14 8B FC E8 14 00 00 00 ?? ?? 01 01 ?? ?? 01 01 ?? ?? ?? 00 ?? ?? 01 01 ?? ?? 02 01 5E E8 0D 00 00 00 6B 65 72 6E 65 6C 33 32 2E 64 6C 6C 00 8B 46 04 FF 10 8B D8 E8 0D 00 00 00 56 69 72 74 75 61 6C 41 6C 6C 6F 63 00 53 8B 06 FF 10 89 07 E8 0C 00 00 00 56 69 72 74 75 61 6C 46 72 65 65 00 53 8B 06 FF 10 89 47 04 E8 0F 00 00 00 47 65 74 50 72 6F 63 65 73 73 48 65 61 70 00 53 8B 06 FF 10 89 47 08 E8 0A 00 00 00 48 65 61 70 41 6C 6C 6F 63 00 53 8B 06 FF 10 89 47 0C E8 09 00 00 00 48 65 }
	condition:
		$1 at pe.entry_point
}
