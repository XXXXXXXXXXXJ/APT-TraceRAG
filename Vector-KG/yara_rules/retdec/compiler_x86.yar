import "pe"


rule fasm_15x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "FASM"
		version = "1.5x"
	strings:
		$1 = { 6A 00 FF 15 ?? ?? 40 00 A3 ?? ?? 40 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule masm_tasm_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MASM or TASM"
	strings:
		$1 = { 6A 00 E8 ?? 0? 00 00 A3 ?? ?? 40 00 ?? ?? ?? ?0 ?0 ?? ?? 00 00 00 ?? ?? 0? ?? ?? ?0 ?? ?? ?0 ?0 ?? ?? ?? ?0 ?? 0? ?? ?0 ?0 00 }
	condition:
		$1 at pe.entry_point
}

rule masm_tasm_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MASM or TASM"
	strings:
		$1 = { C2 ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule masm_tasm_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MASM or TASM"
	strings:
		$1 = { CC FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule masm_tasm_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MASM or TASM"
	strings:
		$1 = { FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 FF 25 ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule masm32 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MASM32"
	strings:
		$1 = { 6A ?? 68 00 30 40 00 68 ?? 30 40 00 6A 00 E8 07 00 00 00 6A 00 E8 06 00 00 00 FF 25 08 20 }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_264
{
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "2.64"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { 6A 60 68 08 FD 40 00 E8 73 3D 00 00 BF 94 00 00 00 8B C7 E8 4F F6 FF FF 89 65 E8 8B F4 89 3E 56 FF 15 58 F1 40 00 8B 4E 10 89 0D 6C 35 44 00 8B 46 04 A3 78 35 44 00 8B 56 08 89 15 7C 35 44 00 8B 76 0C 81 E6 FF 7F 00 00 89 35 70 35 44 00 83 F9 02 74 0C 81 CE 00 80 00 00 89 35 70 35 44 00 C1 E0 08 03 }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_3300
{
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.0.0"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 C4 AF 00 00 E9 79 FE FF FF 8B FF 55 8B EC 8B C1 8B 4D 08 C7 00 88 DA 47 00 8B 09 83 60 08 00 89 48 04 5D C2 08 00 8B FF 55 8B EC 53 8B 5D 08 56 8B F1 C7 06 88 DA 47 00 8B 43 08 89 46 08 85 C0 8B 43 04 57 74 31 85 C0 74 27 50 E8 EF D3 FF FF 8B F8 47 57 E8 10 D3 FF FF 59 59 89 46 04 85 C0 74 18 FF }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_338x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.8.x"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 16 90 00 00 E9 89 FE FF FF CC CC CC CC CC 55 8B EC 57 56 8B 75 0C 8B 4D 10 8B 7D 08 8B C1 8B D1 03 C6 3B FE 76 08 3B F8 0F 82 A0 01 00 00 81 F9 80 00 00 00 72 1C 83 3D 24 97 4A 00 00 74 13 57 56 83 E7 0F 83 E6 0F 3B FE 5E 5F 75 05 E9 DD 03 00 00 F7 C7 03 00 00 00 75 14 C1 E9 02 83 E2 03 83 F9 08 }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_33100 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.10.0"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 8A CF 00 00 E9 7F FE FF FF CC CC 57 56 8B 74 24 10 8B 4C 24 14 8B 7C 24 0C 8B C1 8B D1 03 C6 3B FE 76 08 3B F8 0F 82 68 03 00 00 0F BA 25 58 11 4C 00 01 73 07 F3 A4 E9 17 03 00 00 81 F9 80 00 00 00 0F 82 CE 01 00 00 8B C7 33 C6 A9 0F 00 00 00 75 0E 0F BA 25 70 B3 4B 00 01 0F 82 DA 04 00 00 0F BA }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_33102 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.10.2"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 97 CF 00 00 E9 7F FE FF FF CC CC CC CC CC CC CC CC CC CC CC CC CC CC CC 57 56 8B 74 24 10 8B 4C 24 14 8B 7C 24 0C 8B C1 8B D1 03 C6 3B FE 76 08 3B F8 0F 82 68 03 00 00 0F BA 25 58 01 4C 00 01 73 07 F3 A4 E9 17 03 00 00 81 F9 80 00 00 00 0F 82 CE 01 00 00 8B C7 33 C6 A9 0F 00 00 00 75 0E 0F BA 25 }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_33140 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.14.0"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 B5 D0 00 00 E9 7F FE FF FF CC CC CC CC CC CC CC CC CC 57 56 8B 74 24 10 8B 4C 24 14 8B 7C 24 0C 8B C1 8B D1 03 C6 3B FE 76 08 3B F8 0F 82 68 03 00 00 0F BA 25 FC 31 4C 00 01 73 07 F3 A4 E9 17 03 00 00 81 F9 80 00 00 00 0F 82 CE 01 00 00 8B C7 33 C6 A9 0F 00 00 00 75 0E 0F BA 25 24 E3 4B 00 01 0F }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_33142 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.14.2"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 B8 D0 00 00 E9 7F FE FF FF CC CC CC CC CC CC CC CC CC CC CC CC 57 56 8B 74 24 10 8B 4C 24 14 8B 7C 24 0C 8B C1 8B D1 03 C6 3B FE 76 08 3B F8 0F 82 68 03 00 00 0F BA 25 FC 31 4C 00 01 73 07 F3 A4 E9 17 03 00 00 81 F9 80 00 00 00 0F 82 CE 01 00 00 8B C7 33 C6 A9 0F 00 00 00 75 0E 0F BA 25 24 E3 4B }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_33143 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		version = "3.3.14.3"
		language = "AutoIt"
		bytecode = true
		source = "Made by RetDec Team"
	strings:
		$1 = { E8 C8 D0 00 00 E9 7F FE FF FF CC CC CC CC CC CC CC CC CC CC CC CC 57 56 8B 74 24 10 8B 4C 24 14 8B 7C 24 0C 8B C1 8B D1 03 C6 3B FE 76 08 3B F8 0F 82 68 03 00 00 0F BA 25 FC 41 4C 00 01 73 07 F3 A4 E9 17 03 00 00 81 F9 80 00 00 00 0F 82 CE 01 00 00 8B C7 33 C6 A9 0F 00 00 00 75 0E 0F BA 25 24 F3 4B }
	condition:
		$1 at pe.entry_point
}

rule aut2exe_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Aut2Exe"
		language = "AutoIt"
		bytecode = true
	strings:
		$1 = ">AUTOIT SCRIPT<"
		$2 = ">AUTOIT SCRIPT<" wide
		$3 = ">AUTOIT UNICODE SCRIPT<" wide
	condition:
		pe.is_32bit() and
		for 1 of them : (
			@ > pe.sections[pe.section_index(".rdata")].raw_data_offset and
			@ < pe.sections[pe.section_index(".rdata")].raw_data_offset +
			pe.sections[pe.section_index(".rdata")].raw_data_size
		)
}

rule autohotkey_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "AHK2Exe"
		language = "AutoHotKey"
		bytecode = true
	strings:
		$1 = ">AUTOHOTKEY SCRIPT<"
		$2 = ">AUTOHOTKEY SCRIPT<" wide
	condition:
		pe.is_32bit() and
		for 1 of them : (
			@ > pe.sections[pe.section_index(".rdata")].raw_data_offset and
			@ < pe.sections[pe.section_index(".rdata")].raw_data_offset +
			pe.sections[pe.section_index(".rdata")].raw_data_size
		) or
		for 1 i in (0 .. pe.number_of_resources) : (
			pe.resources[i].name_string matches />AUTOHOTKEY SCRIPT</
		)
}

rule borland_c {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C"
	strings:
		$1 = { 3B CF 76 05 2B CF FC F3 AA 59 }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
	strings:
		$1 = { A1 ?? ?? ?? ?? C1 E0 02 A3 }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		description = "DLL"
	strings:
		$1 = { EB 10 66 62 3A 43 2B 2B 48 4F 4F 4B 90 E9 A1 C1 E0 02 A3 8B }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
	strings:
		$1 = { EB 10 66 62 3A 43 2B 2B 48 4F 4F 4B 90 E9 ?? ?? ?? ?? A1 ?? ?? ?? ?? C1 E0 02 A3 ?? ?? ?? ?? 8B }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
	strings:
		$1 = { 57 51 33 C0 BF ?? ?? ?? ?? B9 ?? ?? ?? ?? 3B CF 76 15 8B CF 81 E1 ?? ?? ?? ?? E3 0B 81 E9 00 10 00 00 F7 D9 FC F3 AA }
	condition:
		for any of them : ( $ in (pe.entry_point .. pe.entry_point + 32) )
}

rule borland_cpp_1991 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "1991"
	strings:
		$1 = { 2E 8C 06 ?? ?? 2E 8C 1E ?? ?? BB ?? ?? 8E DB 1E E8 ?? ?? 1F }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_1991_1994 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "1992, 1994"
	strings:
		$1 = { 8C C8 8E D8 8C 1E ?? ?? 8C 06 ?? ?? 8C 06 ?? ?? 8C 06 }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_1994_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "1994"
	strings:
		$1 = { 8C CA 2E 89 ?? ?? ?? B4 30 CD 21 8B 2E ?? ?? 8B 1E ?? ?? 8E DA A3 ?? ?? 8C }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_1994_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "1994"
	strings:
		$1 = { A1 ?? ?? ?? ?? C1 ?? ?? A3 ?? ?? ?? ?? 83 ?? ?? ?? ?? 75 ?? 57 51 33 C0 BF }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_1995_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "1995"
	strings:
		$1 = { A1 ?? ?? ?? ?? C1 E0 02 A3 ?? ?? ?? ?? 57 51 33 C0 BF ?? ?? ?? ?? B9 ?? ?? ?? ?? 3B CF 76 05 2B CF FC F3 AA 59 5F }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_1999 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "1999"
	strings:
		$1 = { EB 10 66 62 3A 43 2B 2B 48 4F 4F 4B 90 E9 ?? ?? ?? ?? A1 ?? ?? ?? ?? C1 E0 02 A3 ?? ?? ?? ?? 52 }
	condition:
		$1 at pe.entry_point
}

rule borland_cpp_551 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland C++"
		version = "5.5.1"
	strings:
		$1 = { EB 10 66 62 3A 43 2B 2B 48 4F 4F 4B 90 E9 ?? ?? ?? ?? A1 ?? ?? ?? ?? C1 E0 02 A3 ?? ?? ?? ?? 52 6A 00 E8 ?? ?? 00 00 8B D0 E8 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
	strings:
		$1 = { 55 8B EC 83 C4 F4 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
	strings:
		$1 = { C3 E9 ?? ?? ?? FF 8D 40 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
	strings:
		$1 = { 55 8B EC 83 C4 ?? 53 56 57 33 C0 89 45 F0 89 45 D4 89 45 D0 E8 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
	strings:
		$1 = { 55 8B EC 83 C4 B4 B8 ?? ?? ?? ?? E8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8D 40 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_20 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "2.0"
	strings:
		$1 = { E8 ?? ?? ?? ?? 6A ?? E8 ?? ?? ?? ?? 89 05 ?? ?? ?? ?? E8 ?? ?? ?? ?? 89 05 ?? ?? ?? ?? C7 05 ?? ?? ?? ?? 0A ?? ?? ?? B8 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_30_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "3.0"
	strings:
		$1 = { 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 20 28 63 29 20 31 39 38 33 2C 39 36 20 42 6F 72 6C 61 6E 64 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_30_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "3.0"
	strings:
		$1 = { 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 20 28 63 29 20 31 39 38 33 2C 39 37 20 42 6F 72 6C 61 6E 64 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_30_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "3.0"
	strings:
		$1 = { 50 6A ?? E8 ?? ?? FF FF BA ?? ?? ?? ?? 52 89 05 ?? ?? ?? ?? 89 42 04 E8 ?? ?? ?? ?? 5A 58 E8 ?? ?? ?? ?? C3 55 8B EC 33 C0 }
	condition:
		$1 at pe.entry_point
}


rule borland_delphi_40_50 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "4.0 - 5.0"
	strings:
		$1 = { 50 6A ?? E8 ?? ?? FF FF BA ?? ?? ?? ?? 52 89 05 ?? ?? ?? ?? 89 42 04 C7 42 08 00 00 00 00 C7 42 0C 00 00 00 00 E8 ?? ?? ?? ?? 5A 58 E8 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_50 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "5.0"
	strings:
		$1 = { 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 20 28 63 29 20 31 39 38 33 2C 39 39 20 42 6F 72 6C 61 6E 64 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_50_kol {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "5.0"
		description = "with KOL"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 ?? ?? 40 00 E8 ?? ?? FF FF E8 ?? ?? FF FF E8 ?? ?? FF FF 8B C0 00 00 00 00 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_50_mck {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "5.0"
		description = "with MCK"
	strings:
		$1 = { 55 8B EC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? FF ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_60_70_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "6.0 - 7.0"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 ?? ?? ?? ?? E8 ?? ?? FB FF A1 ?? ?? ?? ?? 8B ?? E8 ?? ?? FF FF 8B 0D ?? ?? ?? ?? A1 ?? ?? ?? ?? 8B 00 8B 15 ?? ?? ?? ?? E8 ?? ?? FF FF A1 ?? ?? ?? ?? 8B ?? E8 ?? ?? FF FF E8 ?? ?? FB FF 8D 40 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_60_70_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "6.0 - 7.0"
	strings:
		$1 = { BA ?? ?? ?? ?? 83 7D 0C 01 75 ?? 50 52 C6 05 ?? ?? ?? ?? ?? 8B 4D 08 89 0D ?? ?? ?? ?? 89 4A 04 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_60_kol {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "6.0"
		description = "with Key Objects Library"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 ?? ?? 40 00 E8 ?? ?? FF FF A1 ?? 72 40 00 33 D2 E8 ?? ?? FF FF A1 ?? 72 40 00 8B 00 83 C0 14 E8 ?? ?? FF FF E8 ?? ?? FF FF }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_60_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "6.0"
	strings:
		$1 = { 53 8B D8 33 C0 A3 ?? ?? ?? ?? 6A 00 E8 ?? ?? ?? FF A3 ?? ?? ?? ?? A1 ?? ?? ?? ?? A3 ?? ?? ?? ?? 33 C0 A3 ?? ?? ?? ?? 33 C0 A3 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule borland_delphi_60_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Delphi"
		version = "6.0"
	strings:
		$1 = { 55 8B EC 83 C4 F0 B8 ?? ?? 45 00 E8 ?? ?? ?? FF A1 ?? ?? 45 00 8B 00 E8 ?? ?? FF FF 8B 0D }
	condition:
		$1 at pe.entry_point
}

rule borland_dotnet {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland .NET"
		bytecode = true
	strings:
		$1 = { 48 00 00 00 02 00 [61] 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 13 30 }
	condition:
		$1 at pe.entry_point
}

rule borland_pascal_70 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Pascal"
		version = "7.0"
	strings:
		$1 = { B8 ?? ?? 8E D8 8C ?? ?? ?? 8C D3 8C C0 2B D8 8B C4 05 ?? ?? C1 ?? ?? 03 D8 B4 ?? CD 21 0E }
	condition:
		$1 at pe.entry_point
}

rule borland_pascal_70_windows {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Pascal"
		version = "7.0 for Windows"
	strings:
		$1 = { 9A FF FF 00 00 9A FF FF 00 00 55 89 E5 31 C0 9A FF FF 00 00 }
	condition:
		$1 at pe.entry_point
}

rule borland_pascal_70_protected {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Borland Pascal"
		version = "7.0 protected mode"
	strings:
		$1 = { B8 ?? ?? BB ?? ?? 8E D0 8B E3 8C D8 8E C0 0E 1F A1 ?? ?? 25 ?? ?? A3 ?? ?? E8 ?? ?? 83 3E ?? ?? ?? 75 }
	condition:
		$1 at pe.entry_point
}

rule turbo_cpp {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Turbo C++"
		version = "3.0 1990"
	strings:
		$1 = { 8C CA 2E 89 16 ?? ?? B4 30 CD 21 8B 2E ?? ?? 8B ?? ?? ?? 8E DA A3 ?? ?? 8C 06 }
	condition:
		$1 at pe.entry_point
}

rule turbo_borland_pascal_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Turbo or Borland Pascal"
		version = "7.0"
	strings:
		$1 = { 9A ?? ?? ?? ?? C8 ?? ?? ?? 9A ?? ?? ?? ?? 09 C0 75 ?? EB ?? 8D ?? ?? ?? 16 57 6A ?? 9A ?? ?? ?? ?? BF ?? ?? 1E 57 68 }
	condition:
		$1 at pe.entry_point
}

rule turbo_borland_pascal_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Turbo or Borland Pascal"
		version = "7.x Unit"
	strings:
		$1 = { 54 50 55 51 00 }
	condition:
		$1 at pe.entry_point
}
import "pe"

rule bero_tiny_pascal {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "BeRo Tiny Pascal"
	strings:
		$1 = { E9 ?? ?? ?? ?? 20 43 6F 6D 70 69 6C 65 64 20 62 79 3A 20 42 65 52 6F 54 69 6E 79 50 61 73 63 61 6C 20 2D 20 28 43 29 20 43 6F 70 79 72 69 67 68 74 20 32 30 30 36 2C 20 42 65 6E 6A 61 6D 69 6E 20 27 42 65 52 6F 27 20 52 6F 73 73 65 61 75 78 20 }
	condition:
		$1 at pe.entry_point
}

rule tmt_pascal_040 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "TMT-Pascal"
		version = "0.40"
	strings:
		$1 = { 0E 1F 06 8C 06 ?? ?? 26 A1 ?? ?? A3 ?? ?? 8E C0 66 33 FF 66 33 C9 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule delphi_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Delphi"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 56 B8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 33 C0 55 68 ?? ?? ?? ?? 64 FF 30 64 89 20 B8 }
	condition:
		$1 at pe.entry_point
}

rule delphi_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Delphi"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 B8 ?? ?? ?? ?? E8 ?? ?? ?? ?? 33 C0 55 68 ?? ?? ?? ?? 64 FF 30 64 89 20 B8 ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule delphi_20 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Delphi"
		version = "2.0"
	strings:
		$1 = { 44 43 55 32 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_09910_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "0.99.10"
	strings:
		$1 = { E8 00 6E 00 00 55 89 E5 8B 7D 0C 8B 75 08 89 F8 8B 5D 10 29 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_09910_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "0.99.10"
		start = 19
	strings:
		$1 = { E8 00 6E 00 00 55 89 E5 8B 7D 0C 8B 75 08 89 F8 8B 5D 10 29 }
	condition:
		$1 at pe.entry_point + 19
}

rule free_pascal_104_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.4"
	strings:
		$1 = { 55 89 E5 C6 05 ?? ?? ?? ?? 00 E8 ?? ?? ?? ?? 55 31 ED 89 E0 A3 ?? ?? ?? ?? 66 8C D5 89 2D }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_104_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.4"
	strings:
		$1 = { C6 05 ?? ?? ?? ?? 00 55 89 E5 53 56 57 8B 7D 08 89 3D ?? ?? ?? ?? 8B 7D 0C 89 3D ?? ?? ?? ?? 8B 7D 10 89 3D }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_104_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.4"
	strings:
		$1 = { C6 05 ?? ?? ?? ?? 00 55 89 E5 53 56 57 8B 7D 08 89 3D ?? ?? ?? ?? 8B 7D 0C 89 3D ?? ?? ?? ?? 8B 7D 10 89 3D ?? ?? ?? ?? E8 ?? ?? ?? ?? 5F 5E 5B 5D C2 0C 00 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_104_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.4"
	strings:
		$1 = { 55 89 E5 C6 05 ?? ?? ?? ?? 00 E8 ?? ?? ?? ?? 55 31 ED 89 E0 A3 ?? ?? ?? ?? 66 8C D5 89 2D ?? ?? ?? ?? DB E3 D9 2D ?? ?? ?? ?? 31 ED E8 ?? ?? ?? ?? 5D E8 ?? ?? ?? ?? C9 C3 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_106 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.6"
	strings:
		$1 = { C6 05 ?? ?? 40 00 ?? E8 ?? ?? 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_1010_console {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.10"
		description = "console version"
	strings:
		$1 = { C6 05 ?? ?? ?? 00 01 E8 ?? ?? 00 00 C6 05 ?? ?? ?? 00 00 E8 ?? ?? 00 00 50 E8 00 00 00 00 FF 25 ?? ?? ?? 00 55 89 E5 ?? EC }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_1010_gui {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.10"
		description = "with GUI"
	strings:
		$1 = { C6 05 ?? ?? ?? 00 00 E8 ?? ?? 00 00 50 E8 00 00 00 00 FF 25 ?? ?? ?? 00 55 89 E5 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_1010_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.10"
	strings:
		$1 = { C6 05 ?? ?? ?? ?? 01 E8 ?? ?? 00 00 C6 05 ?? ?? ?? ?? 00 E8 ?? ?? 00 00 50 E8 00 00 00 00 FF 25 ?? ?? ?? ?? 55 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_1010_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "1.0.10"
	strings:
		$1 = { C6 05 ?? ?? ?? ?? 00 E8 ?? ?? 00 00 50 E8 00 00 00 00 FF 25 ?? ?? ?? ?? 55 89 E5 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_200_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "2.0.0"
	strings:
		$1 = { 55 89 E5 C6 05 ?? ?? ?? ?? 00 E8 ?? ?? ?? ?? 6A 00 64 FF 35 00 00 00 00 89 E0 A3 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_200_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "2.0.0"
	strings:
		$1 = { C6 05 ?? ?? ?? ?? 01 E8 74 00 00 00 C6 05 ?? ?? ?? ?? 00 E8 68 00 00 00 50 E8 00 00 00 00 FF 25 ?? ?? ?? ?? 90 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_200_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "2.0.0"
	strings:
		$1 = { 55 89 E5 C6 05 ?? ?? ?? ?? 00 E8 ?? ?? ?? ?? 6A 00 64 FF 35 00 00 00 00 89 E0 A3 ?? ?? ?? ?? 55 31 ED 89 E0 A3 ?? ?? ?? ?? 66 8C D5 89 2D ?? ?? ?? ?? E8 ?? ?? ?? ?? 31 ED E8 ?? ?? ?? ?? 5D E8 ?? ?? ?? ?? C9 C3 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_200_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "2.0.0"
	strings:
		$1 = { C6 05 00 80 40 00 01 E8 74 00 00 00 C6 05 00 80 40 00 00 E8 68 00 00 00 50 E8 00 00 00 00 FF 25 D8 A1 40 00 90 90 90 90 90 90 90 90 90 90 90 90 55 89 E5 83 EC 04 89 5D FC E8 92 00 00 00 E8 ED 00 00 00 89 C3 B9 ?? 70 40 00 89 DA B8 00 00 00 00 E8 0A 01 00 00 E8 C5 01 00 00 89 D8 E8 3E 02 00 00 E8 B9 01 00 00 E8 54 02 00 00 8B 5D FC C9 C3 8D 76 00 00 00 00 00 00 00 00 00 00 00 00 00 55 89 E5 C6 05 10 80 40 00 00 E8 D1 03 00 00 6A 00 64 FF 35 00 00 00 00 89 E0 A3 ?? 70 40 00 55 31 ED 89 E0 A3 20 80 }
	condition:
		$1 at pe.entry_point
}

rule free_pascal_260 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Free Pascal"
		version = "2.6.0"
	strings:
		$1 = { 55 89 E5 C6 05 ?? ?? ?? ?? 01 68 ?? ?? ?? ?? 6A F6 E8 ?? ?? ?? ?? 50 E8 }
	condition:
		$1 at pe.entry_point
}

rule free_basic_011 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "FreeBASIC"
		version = "0.11"
	strings:
		$1 = { E8 ?? ?? 00 00 E8 01 00 00 00 C3 55 89 E5 }
	condition:
		$1 at pe.entry_point
}

rule free_basic_014 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "FreeBASIC"
		version = "0.14"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 ?? 00 00 00 FF 15 ?? ?? ?? 00 E8 ?? FF FF FF 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 89 E5 83 EC 08 8B 45 08 89 04 24 FF 15 ?? ?? ?? 00 89 EC 5D C3 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 89 E5 }
	condition:
		$1 at pe.entry_point
}

rule free_basic_016b {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "FreeBASIC"
		version = "0.16b"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 01 00 00 00 FF 15 ?? ?? ?? 00 E8 88 FF FF FF 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 C7 04 24 02 00 00 00 FF 15 ?? ?? ?? 00 E8 68 FF FF FF 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 8B 45 08 89 04 24 FF 15 ?? ?? ?? 00 89 EC 5D C3 8D 76 00 8D BC 27 00 00 00 00 55 89 E5 83 EC 08 8B 45 08 89 04 24 FF 15 ?? ?? ?? 00 89 EC 5D C3 90 90 90 90 90 90 90 90 90 90 }
	condition:
		$1 at pe.entry_point
}

rule hot_soup_processor_31 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Hot Soup Processor"
		version = "3.1"
	strings:
		$1 = { 6A 60 68 30 08 42 00 E8 21 43 00 00 BF 94 00 00 00 8B C7 E8 FD F5 FF FF 89 65 E8 8B F4 89 3E 56 FF 15 F0 00 42 00 8B 4E 10 89 0D 88 45 42 00 8B 46 04 A3 94 45 42 00 8B 56 08 89 15 98 45 42 00 8B 76 0C 81 E6 FF 7F 00 00 89 35 8C 45 42 00 83 F9 02 74 0C 81 CE 00 80 00 00 89 35 8C 45 42 00 C1 E0 08 03 }
	condition:
		$1 at pe.entry_point
}

rule hot_soup_processor_32 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Hot Soup Processor"
		version = "3.2"
	strings:
		$1 = { 6A 60 68 ?? ?8 42 00 E8 ?? ?? 00 00 BF 94 00 00 00 8B C7 E8 ?? F? FF FF 89 65 E8 8B F4 89 3E 56 FF 15 ?0 ?? 42 00 8B 4E 10 89 0D ?? ?? 42 00 8B 46 04 A3 ?? ?? 42 00 8B 56 08 89 15 ?? ?? 42 00 8B 76 0C 81 E6 FF 7F 00 00 89 35 ?? ?? 42 00 83 F9 02 74 0C 81 CE 00 80 00 00 89 35 ?? ?? 42 00 C1 E0 08 03 }
	condition:
		$1 at pe.entry_point
}

rule intel_xe_13 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Intel XE"
		version = "13"
	strings:
		$1 = { E8 ?? ?? 00 00 E9 A4 FE FF FF }
	condition:
		$1 at pe.entry_point
}

rule lcclike_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "LCC or similar"
	strings:
		$1 = { 55 89 E5 53 }
	condition:
		$1 at pe.entry_point
}

rule lcclike_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "LCC or similar"
	strings:
		$1 = { 55 89 E5 55 }
	condition:
		$1 at pe.entry_point
}

rule lcc_1x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "LCC"
		version = "1.x"
	strings:
		$1 = { 64 A1 ?? ?? ?? ?? 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 9A 10 40 ?? 50 }
	condition:
		$1 at pe.entry_point
}

rule lcc_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "LCC"
	strings:
		$1 = { 55 89 E5 53 56 57 83 7D 0C 01 75 05 E8 17 ?? ?? ?? FF 75 10 FF 75 0C FF 75 08 A1 }
	condition:
		$1 at pe.entry_point
}

rule lahey_fortran_2001 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Lahey Fortran 90"
		version = "2001"
	strings:
		$1 = { 55 8B EC 8B 45 ?? 83 E8 ?? 72 ?? 74 ?? 48 74 ?? 48 74 ?? EB ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 59 E8 }
	condition:
		$1 at pe.entry_point
}

rule metaware_high_c_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MetaWare High C"
	strings:
		$1 = { B8 ?? ?? 8E D8 B8 ?? ?? CD 21 A3 ?? ?? 3C 03 7D ?? B4 09 }
	condition:
		$1 at pe.entry_point
}

rule metaware_high_c_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MetaWare High C"
	strings:
		$1 = { B8 ?? ?? 50 B8 ?? ?? 50 CB }
	condition:
		$1 at pe.entry_point
}

rule metrowerks_codewarrior_20_console {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Metrowerks CodeWarrior"
		version = "2.0"
		description = "console version"
	strings:
		$1 = { 55 89 E5 55 B8 FF FF FF FF 50 50 68 ?? ?? ?? ?? 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 ?? ?? 00 00 E8 ?? ?? 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule metrowerks_codewarrior_20_gui {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Metrowerks CodeWarrior"
		version = "2.0"
		description = "with GUI"
	strings:
		$1 = { 55 89 E5 53 56 83 EC 44 55 B8 FF FF FF FF 50 50 68 ?? ?? 40 00 64 FF 35 00 00 00 00 64 89 25 00 00 00 00 68 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 ?? ?? 00 00 E8 ?? ?? 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule metrowerks_codewarrior_20 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Metrowerks CodeWarrior"
		version = "2.0"
	strings:
		$1 = { 55 89 E5 53 56 57 8B 75 0C 8B 5D 10 83 FE 01 74 05 83 FE 02 75 12 53 56 FF 75 08 E8 6E FF FF FF 09 C0 75 04 31 C0 EB 21 53 56 FF 75 08 E8 ?? ?? ?? ?? 89 C7 09 F6 74 05 83 FE 03 75 0A 53 56 FF 75 08 E8 47 FF FF FF 89 F8 8D 65 F4 5F 5E 5B 5D C2 0C 00 C9 }
	condition:
		$1 at pe.entry_point
}

rule ms_fortran_1989 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft FORTRAN"
		version = "1989"
	strings:
		$1 = { B4 30 CD 21 86 E0 2E A3 ?? ?? 3D ?? ?? 73 }
	condition:
		$1 at pe.entry_point
}

rule ms_fortran_19xx {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft FORTRAN"
		version = "19xx"
	strings:
		$1 = { FC 1E B8 ?? ?? 8E D8 9A ?? ?? ?? ?? 81 ?? ?? ?? 8B EC 8C DB 8E C3 BB ?? ?? 9A ?? ?? ?? ?? 9B DB E3 9B D9 2E ?? ?? 33 C9 }
	condition:
		$1 at pe.entry_point
}

rule pelles_c_280_290 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Pelles C"
		version = "2.80 - 2.90"
	strings:
		$1 = { 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 FF 35 ?? ?? ?? ?? 64 89 25 ?? ?? ?? ?? 83 EC ?? 83 EC ?? 53 56 57 89 65 E8 68 00 00 00 ?? E8 ?? ?? ?? ?? 59 A3 }
	condition:
		$1 at pe.entry_point
}

rule pelles_c_290_300_400 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Pelles C"
		version = "2.90, 3.00, 4.00"
	strings:
		$1 = { 55 89 E5 53 56 57 8B 5D 0C 8B 75 10 BF 01 00 00 00 85 DB 75 10 83 3D ?? ?? ?? ?? 00 75 07 31 C0 E9 ?? ?? ?? ?? 83 FB 01 74 05 83 FB 02 75 ?? 85 FF 74 }
	condition:
		$1 at pe.entry_point
}

rule pelles_c_300_400_450_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Pelles C"
		version = "3.00, 4.00, 4.50"
	strings:
		$1 = { 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 FF 35 ?? ?? ?? ?? 64 89 25 ?? ?? ?? ?? 83 EC ?? 53 56 57 89 65 E8 C7 45 FC ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 59 BE ?? ?? ?? ?? EB }
	condition:
		$1 at pe.entry_point
}

rule pelles_c_300_400_450_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Pelles C"
		version = "3.00, 4.00, 4.50"
	strings:
		$1 = { 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 FF 35 ?? ?? ?? ?? 64 89 25 ?? ?? ?? ?? 83 EC ?? 53 56 57 89 65 E8 68 00 00 00 02 E8 ?? ?? ?? ?? 59 A3 }
	condition:
		$1 at pe.entry_point
}

rule pelles_c_450 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Pelles C"
		version = "4.50"
	strings:
		$1 = { 55 89 E5 53 56 57 8B 5D 0C 8B 75 10 85 DB 75 0D 83 3D ?? ?? ?? ?? 00 75 04 31 C0 EB 57 83 FB 01 74 05 83 FB 02 75 }
	condition:
		$1 at pe.entry_point
}

rule perlapp_602_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PerlApp"
		version = "6.0.2"
	strings:
		$1 = { 68 2C EA 40 00 FF D3 83 C4 0C 85 C0 0F 85 CD 00 00 00 6A 09 57 68 20 EA 40 00 FF D3 83 C4 0C 85 C0 75 12 8D 47 09 50 FF 15 1C D1 40 00 59 A3 B8 07 41 00 EB 55 6A 08 57 68 14 EA 40 00 FF D3 83 C4 0C 85 C0 75 11 8D 47 08 50 FF 15 1C D1 40 00 59 89 44 24 10 EB 33 6A 09 57 68 08 EA 40 00 FF D3 83 C4 0C 85 C0 74 22 6A 08 57 68 FC E9 40 00 FF D3 83 C4 0C 85 C0 74 11 6A 0B 57 68 F0 E9 40 00 FF D3 83 C4 0C 85 C0 75 55 }
	condition:
		$1 at pe.entry_point
}

rule perlapp_602_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PerlApp"
		version = "6.0.2"
	strings:
		$1 = { 68 9C E1 40 00 FF 15 A4 D0 40 00 85 C0 59 74 0F 50 FF 15 1C D1 40 00 85 C0 59 89 45 FC 75 62 6A 00 8D 45 F8 FF 75 0C F6 45 14 01 50 8D 45 14 50 E8 9B 01 00 00 83 C4 10 85 C0 0F 84 E9 00 00 00 8B 45 F8 83 C0 14 50 FF D6 85 C0 59 89 45 FC 75 0E FF 75 14 FF 15 78 D0 40 00 E9 C9 00 00 00 68 8C E1 40 00 FF 75 14 50 }
	condition:
		$1 at pe.entry_point
}

rule polybox_c_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PolyBox C"
	strings:
		$1 = { 55 8B EC 83 C4 F0 53 56 B8 E4 41 00 10 E8 3A E1 FF FF 33 C0 55 68 11 44 00 10 64 FF 30 64 89 20 EB 08 FC FC FC FC FC FC 27 54 6A 0A 68 20 44 00 10 A1 1C 71 00 10 50 E8 CC E1 ?? ?? ?? ?? 85 DB 0F 84 77 01 00 00 53 A1 1C 71 00 10 50 E8 1E E2 FF FF 8B F0 85 F6 0F 84 61 01 00 00 53 A1 1C 71 00 10 50 E8 E0 E1 FF FF 85 C0 0F 84 4D 01 00 00 50 E8 DA E1 FF FF 8B D8 85 DB 0F 84 3D 01 00 00 56 B8 70 80 00 10 B9 01 00 00 00 8B 15 98 41 00 10 E8 9E DE FF FF 83 C4 04 A1 70 80 00 10 8B CE 8B D3 E8 E1 E1 FF FF 6A 00 6A 00 A1 70 80 00 10 B9 30 44 00 10 8B D6 E8 F8 FD FF FF }
	condition:
		$1 at pe.entry_point
}

rule polybox_c_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PolyBox D"
	strings:
		$1 = { 55 8B EC 33 C9 51 51 51 51 51 53 33 C0 55 68 84 2C 40 00 64 FF 30 64 89 20 C6 45 FF 00 B8 B8 46 40 00 BA 24 00 00 00 E8 8C F3 FF FF 6A 24 BA B8 46 40 00 8B 0D B0 46 40 00 A1 94 46 40 00 E8 71 FB FF FF 84 C0 0F 84 6E 01 00 00 8B 1D D0 46 40 00 8B C3 83 C0 24 03 05 D8 46 40 00 3B 05 B4 46 40 00 0F 85 51 01 00 00 8D 45 F4 BA B8 46 40 00 B9 10 00 00 00 E8 A2 EC FF FF 8B 45 F4 BA 9C 2C 40 00 E8 F1 ED FF FF }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? ?? 00 66 2E F7 05 ?? ?? ?? ?? 04 00 0F 85 ?? 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? ?? 00 66 2E F7 05 ?? ?? ?? ?? 04 00 75 05 E9 ?? ?? ?? ?? E9 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? ?? 00 66 F7 05 ?? ?? ?? ?? 04 00 0F 85 ?? 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? ?? 00 66 F7 05 ?? ?? ?? ?? 04 00 75 05 E9 ?? ?? ?? ?? E9 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_30x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
		version = "3.0x"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? ?? 00 66 2E F7 05 ?? ?? ?? 00 04 00 0F 85 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_40x_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
		version = "4.0x"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? 40 00 66 2E F7 05 ?? ?? 40 00 04 00 75 05 E9 68 05 00 00 E9 6E 03 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_40x_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PureBasic"
		version = "4.0x"
	strings:
		$1 = { 83 7C 24 08 01 75 0E 8B 44 24 04 A3 ?? ?? ?? 10 E8 22 00 00 00 83 7C 24 08 02 75 00 83 7C 24 08 00 75 05 E8 ?? 00 00 00 83 7C 24 08 03 75 00 B8 01 00 00 00 C2 0C 00 68 00 00 00 00 68 00 10 00 00 68 00 00 00 00 E8 ?? 0F 00 00 A3 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_70x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
		version = "7.0x"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? 40 00 66 2E F7 05 ?? ?? 40 00 04 00 0F 85 DB 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule powerbasic_800 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PowerBASIC"
		version = "8.00"
	strings:
		$1 = { 55 8B EC 53 56 57 BB 00 ?? ?? 00 66 2E F7 05 ?? ?? 40 00 04 00 75 05 E9 14 04 00 00 E9 19 02 }
	condition:
		$1 at pe.entry_point
}

rule purebasic_4x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "PureBasic"
		version = "4.x"
	strings:
		$1 = { 68 ?? ?? 00 00 68 00 00 00 00 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 C4 0C 68 00 00 00 00 E8 ?? ?? ?? ?? A3 ?? ?? ?? ?? 68 00 00 00 00 68 00 10 00 00 68 00 00 00 00 E8 ?? ?? ?? ?? A3 }
	condition:
		$1 at pe.entry_point
}

rule symantec_c_zortech_c_210_400_30r1 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Symantec C or Zortech C"
		version = "2.10, 4.00 or 3.0r1"
	strings:
		$1 = { FA FC B8 ?? ?? 8E D8 }
	condition:
		$1 at pe.entry_point
}

rule symantec_c_400 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Symantec C"
		version = "4.00"
	strings:
		$1 = { FA B8 ?? ?? DB E3 8E D8 8C 06 ?? ?? 8B D8 2B 1E ?? ?? 89 1E ?? ?? 26 }
	condition:
		$1 at pe.entry_point
}

rule symantec_visual_cafe_30 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Symantec Visual Cafe"
		version = "3.0"
		bytecode = true
	strings:
		$1 = { 64 8B 05 ?? ?? ?? ?? 55 8B EC 6A FF 68 ?? ?? 40 ?? 68 ?? ?? 40 ?? 50 64 89 25 ?? ?? ?? ?? 83 EC 08 50 53 56 57 89 65 E8 C7 45 FC }
	condition:
		$1 at pe.entry_point
}

rule open_watcom_19 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Open Watcom"
		version = "1.9"
	strings:
		$1 = { E9 ?? ?? 00 00 03 10 40 00 4F 70 65 6E 20 57 61 74 63 6F 6D 20 43 2F 43 2B 2B 33 32 20 52 75 6E 2D 54 69 6D 65 20 73 79 73 74 65 6D 2E 20 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 20 28 43 29 20 53 79 62 61 73 65 2C 20 49 6E 63 2E 20 31 39 38 38 2D 32 30 30 32 2E }
	condition:
		$1 at pe.entry_point
}

rule watcom_c_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		start = 19
	strings:
		$1 = { 57 41 54 43 4F 4D 20 43 2F 43 2B 2B 33 32 20 52 75 6E 2D 54 69 6D 65 20 73 79 73 74 65 6D 2E 20 28 63 29 20 43 6F 70 79 72 69 67 68 74 }
	condition:
		$1 at pe.entry_point + 19
}

rule watcom_c_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		start = 9
	strings:
		$1 = { 4F 70 65 6E 20 57 61 74 63 6F 6D 20 43 2F 43 2B 2B 33 32 20 52 75 6E 2D 54 69 6D 65 20 73 79 73 74 65 6D 2E 20 50 6F 72 74 69 6F 6E 73 20 43 6F 70 79 72 69 67 68 74 }
	condition:
		$1 at pe.entry_point + 9
}

rule watcom_c_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		start = 9
	strings:
		$1 = { 57 41 54 43 4F 4D 20 43 2F 43 2B 2B 33 32 20 52 75 6E 2D 54 69 6D 65 20 73 79 73 74 65 6D 2E 20 28 63 29 20 43 6F 70 79 72 69 67 68 74 }
	condition:
		$1 at pe.entry_point + 9
}

rule watcom_c_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
	strings:
		$1 = { 53 56 57 55 8B 74 24 14 8B 7C 24 18 8B 6C 24 1C 83 FF 03 0F 87 }
	condition:
		$1 at pe.entry_point
}

rule watcom_c_uv_05 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
	strings:
		$1 = { E9 ?? ?? 00 00 03 10 40 00 57 41 54 43 4F 4D 20 43 2F 43 2B 2B 33 32 20 52 75 6E 2D 54 69 6D 65 20 73 79 73 74 65 6D 2E 20 28 63 29 20 43 6F 70 79 72 69 67 68 74 20 62 79 20 57 41 54 43 4F 4D 20 49 6E 74 65 72 6E 61 74 69 6F 6E 61 6C 20 43 6F 72 70 2E 20 31 39 38 38 2D 31 39 39 35 2E 20 41 6C 6C 20 72 69 67 68 74 73 20 72 65 73 65 72 76 65 64 2E 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule watcom_c_uv_06 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		version = "32 Run-Time System"
	strings:
		$1 = { FB 83 ?? ?? 89 E3 89 ?? ?? ?? ?? ?? 89 ?? ?? ?? ?? ?? 66 ?? ?? ?? 66 ?? ?? ?? ?? ?? BB ?? ?? ?? ?? 29 C0 B4 30 CD 21 }
	condition:
		$1 at pe.entry_point
}

rule watcom_c_uv_07 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		version = "32 Run-Time System"
	strings:
		$1 = { E9 ?? ?? ?? ?? ?? ?? ?? ?? 57 41 54 43 4F 4D 20 43 2F 43 2B 2B 33 32 20 52 75 6E 2D 54 }
	condition:
		$1 at pe.entry_point
}

rule watcom_c_uv_08 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		version = "32 Run-Time System"
	strings:
		$1 = { 0E 1F 8C C6 B4 ?? 50 BB ?? ?? CD 21 73 ?? 58 CD 21 72 }
	condition:
		$1 at pe.entry_point
}

rule watcom_c_uv_09 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "WATCOM C/C++"
		version = "Run-Time system + DOS4GW DOS Extender"
	strings:
		$1 = { BF ?? ?? 8E D7 81 C4 ?? ?? BE ?? ?? 2B F7 8B C6 B1 ?? D3 }
	condition:
		$1 at pe.entry_point
}

rule zortech_c {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Zortech C"
	strings:
		$1 = { E8 ?? ?? 2E FF ?? ?? ?? FC 06 }
	condition:
		$1 at pe.entry_point
}

rule zortech_c_20 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Zortech C"
		version = "2.0"
	strings:
		$1 = { FA B8 ?? ?? 8E D8 8C ?? ?? ?? 26 8B ?? ?? ?? 89 1E ?? ?? 8B D8 2B 1E ?? ?? 89 1E }
	condition:
		$1 at pe.entry_point
}

rule zortech_c_30 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Zortech C"
		version = "3.0"
	strings:
		$1 = { FA FC B8 ?? ?? ?? 8C C8 8E D8 }
	condition:
		$1 at pe.entry_point
}

rule gc
{
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "gc"
		language = "Go"
	strings:
		$1 = { ?? ?? ?? ?? ?? 24 0C 8D 5C 24 10 89 44 24 04 89 5C 24 08 C7 04 24 FF FF FF FF E9 01 00 00 00 ?? E9 ?B D? FF FF ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 8B 5C 24 04 64 C7 05 34 00 00 00 00 00 00 00 89 E5 8B 4B 04 89 C8 C1 E0 02 29 C4 89 E7 8B 73 08 FC F3 A5 FF 13 89 EC 8B 5C 24 04 89 43 0C 89 53 10 64 8B 05 34 00 00 00 89 43 14 C3 ?? ?? ?? ?? 83 EC 18 C7 04 24 F4 FF }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 55 89 E5 56 }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 55 89 E5 57 }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 55 89 E5 81 EC }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 55 89 E5 83 EC }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_05 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 55 89 E5 8B 45 08 A3 ?? ?? ?? ?? B8 01 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_06 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 83 EC 0C C7 04 24 02 00 00 00 FF 15 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8B 0D }
	condition:
		$1 at pe.entry_point
}

rule gcclike_uv_07 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "GCC or similar"
	strings:
		$1 = { 56 53 83 EC 14 8B ?? 24 24 83 F? 01 74 ?? 8B 44 24 28 89 ?? 24 04 89 44 24 08 8B 44 24 20 89 04 24 E8 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 ?? 00 00 00 FF 15 ?? ?? ?? 00 E8 ?? FE FF FF 90 8D B4 26 00 00 00 00 55 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_2x_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "2.x"
	strings:
		$1 = { 55 89 E5 E8 ?? ?? ?? ?? C9 C3 ?? ?? 45 58 45 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_2x_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "2.x"
	strings:
		$1 = { 55 89 E5 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_3x_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.x"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 ?? 00 00 00 FF 15 ?? ?? 40 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_3x_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.x"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 ?? 00 00 00 FF 15 ?? ?? ?? ?? E8 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 55 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_32x_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.2.x"
	strings:
		$1 = { 55 89 E5 83 EC 08 6A 00 6A 00 6A 00 6A 00 E8 0D 00 00 00 B8 00 00 00 00 C9 C3 90 90 90 90 90 90 FF 25 38 20 40 00 90 90 00 00 00 00 00 00 00 00 FF FF FF FF 00 00 00 00 FF FF FF FF 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_32x_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.2.x"
	strings:
		$1 = { 55 89 E5 83 EC 18 89 75 FC 8B 75 0C 89 5D F8 83 FE 01 74 5C 89 74 24 04 8B 55 10 89 54 24 08 8B 55 08 89 14 24 E8 96 01 00 00 83 EC 0C 83 FE 01 89 C3 74 2C 85 F6 75 0C 8B 0D 00 30 00 10 85 C9 75 10 31 DB 89 D8 8B 5D F8 8B 75 FC 89 EC 5D C2 0C 00 E8 59 00 00 00 EB EB 8D B4 26 00 00 00 00 85 C0 75 D0 E8 47 00 00 00 EB C9 90 8D 74 26 00 C7 04 24 80 00 00 00 E8 F4 05 00 00 A3 00 30 00 10 85 C0 74 1A C7 00 00 00 00 00 A3 10 30 00 10 E8 3B 02 00 00 E8 C6 01 00 00 E9 75 FF FF FF E8 BC 05 00 00 C7 00 0C }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_32x_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.2.x"
	strings:
		$1 = { 55 89 E5 83 EC 08 6A 00 6A 00 6A 00 6A 00 E8 0D 00 00 00 B8 00 00 00 00 C9 C3 90 90 90 90 90 90 FF 25 38 20 00 10 90 90 00 00 00 00 00 00 00 00 FF FF FF FF 00 00 00 00 FF FF FF FF 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_32x_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.2.x"
	strings:
		$1 = { 55 89 E5 83 EC 18 89 75 FC 8B 75 0C 89 5D F8 83 FE 01 74 5C 89 74 24 04 8B 55 10 89 54 24 08 8B 55 08 89 14 24 E8 76 01 00 00 83 EC 0C 83 FE 01 89 C3 74 2C 85 F6 75 0C 8B 0D 00 30 00 10 85 C9 75 10 31 DB 89 D8 8B 5D F8 8B 75 FC 89 EC 5D C2 0C 00 E8 59 00 00 00 EB EB 8D B4 26 00 00 00 00 85 C0 75 D0 E8 47 00 00 00 EB C9 90 8D 74 26 00 C7 04 24 80 00 00 00 E8 A4 05 00 00 A3 00 30 00 10 85 C0 74 1A C7 00 00 00 00 00 A3 10 30 00 10 E8 1B 02 00 00 E8 A6 01 00 00 E9 75 FF FF FF E8 6C 05 00 00 C7 00 0C }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_32x_05 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.2.x"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 01 00 00 00 FF 15 E4 40 40 00 E8 68 00 00 00 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 C7 04 24 02 00 00 00 FF 15 E4 40 40 00 E8 48 00 00 00 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 8B 55 08 89 14 24 FF 15 00 41 40 00 89 EC 5D C3 8D 76 00 8D BC 27 00 00 00 00 55 89 E5 83 EC 08 8B 55 08 89 14 24 FF 15 F4 40 40 00 89 EC 5D C3 8D 76 00 8D BC 27 00 00 00 00 55 89 E5 53 83 EC 24 C7 04 24 A0 11 40 00 E8 8D 07 00 00 83 EC 04 E8 85 02 00 00 C7 04 24 00 20 40 00 8B 15 10 20 40 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_32x_06 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "3.2.x"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 01 00 00 00 FF 15 FC 40 40 00 E8 68 00 00 00 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 C7 04 24 02 00 00 00 FF 15 FC 40 40 00 E8 48 00 00 00 89 EC 31 C0 5D C3 89 F6 55 89 E5 83 EC 08 8B 55 08 89 14 24 FF 15 18 41 40 00 89 EC 5D C3 8D 76 00 8D BC 27 00 00 00 00 55 89 E5 83 EC 08 8B 55 08 89 14 24 FF 15 0C 41 40 00 89 EC 5D C3 8D 76 00 8D BC 27 00 00 00 00 55 89 E5 53 83 EC 24 C7 04 24 A0 11 40 00 E8 5D 08 00 00 83 EC 04 E8 55 03 00 00 C7 04 24 00 20 40 00 8B 15 10 20 40 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_4x {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW G++"
		version = "4.x"
	strings:
		$1 = { 55 89 E5 83 EC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 55 89 E5 83 EC ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 55 ?? ?? ?? ?? ?? ?? ?? ?? ?? FF ?? ?? ?? ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_461 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "4.6.1"
	strings:
		$1 = { 55 89 E5 83 EC 18 C7 04 24 01 00 00 00 FF 15 ?? ?? ?? 00 E8 7C FD FF FF 55 89 E5 83 EC 18 C7 04 24 02 00 00 00 FF 15 ?? ?? ?? 00 E8 64 FD FF FF 55 89 E5 83 EC 08 A1 ?? ?? ?? 00 C9 FF E0 66 90 55 89 E5 83 EC 08 A1 ?? ?? ?? 00 C9 FF E0 90 90 55 89 E5 83 EC 18 C7 04 24 00 ?0 ?? 00 E8 ?? ?? ?? 00 52 85 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_473 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "4.7.3"
	strings:
		$1 = { 83 EC 0C C7 05 ?? ?? ?? 00 00 00 00 00 E8 ?E ?? 00 00 83 C4 0C E9 86 FC FF FF 90 90 90 90 90 90 A1 ?? ?? ?? 00 85 C0 74 43 55 89 E5 83 EC 18 C7 04 24 20 ?0 ?? 00 FF 15 ?? ?1 ?? 00 BA 00 00 00 00 83 EC 04 85 C0 74 16 C7 44 24 04 2E ?0 ?? 00 89 04 24 FF 15 ?? ?1 ?? 00 83 EC 08 89 C2 85 D2 74 09 C7 04 }
	condition:
		$1 at pe.entry_point
}

rule mingw_gcc_520 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MinGW GCC"
		version = "5.2.0"
	strings:
		$1 = { 83 EC 0C C7 05 ?? ?? ?? 00 00 00 00 00 E8 ?E ?? 00 00 83 C4 0C E9 76 FC FF FF 90 90 90 90 90 90 55 89 E5 57 56 53 83 EC 2C 8B 35 ?? ?1 ?? 00 C7 04 24 00 ?0 ?? 00 FF D6 83 EC 04 85 C0 0F 84 BD 00 00 00 89 C3 C7 04 24 00 ?0 ?? 00 FF 15 ?? ?1 ?? 00 8B 15 ?? ?1 ?? 00 83 EC 04 A3 ?? ?? ?? 00 C7 44 24 04 }
	condition:
		$1 at pe.entry_point
}

rule dev_cpp_gcc_4 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Dev-C++ GCC"
		version = "4"
	strings:
		$1 = { 55 89 E5 83 EC 08 83 C4 F4 6A ?? A1 ?? ?? ?? 00 FF D0 E8 ?? FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule dev_cpp_gcc_4992 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Dev-C++ GCC"
		version = "4.9.9.2"
	strings:
		$1 = { 55 89 E5 83 EC 08 C7 04 24 01 00 00 00 FF 15 ?? ?? ?? 00 E8 C8 FE FF FF 90 8D B4 26 00 00 00 00 55 89 E5 83 EC 08 C7 04 24 02 00 00 00 FF 15 ?? ?? ?? 00 E8 A8 FE FF FF 90 8D B4 26 00 00 00 00 55 8B 0D ?? ?? ?? 00 89 E5 5D FF E1 8D 74 26 00 55 8B 0D }
	condition:
		$1 at pe.entry_point
}

rule dev_cpp_gcc_5 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Dev-C++ GCC"
		description = "5"
	strings:
		$1 = { 55 89 E5 83 EC 14 6A ?? FF 15 ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ms_incremental_linker {
	meta:
        author = "RetDec Team"
		tool = "L"
		name = "Microsoft Incremental Linker"
		version = "5.12.8078"
		description = "WinASM Studio"
	strings:
		$1 = { 6A 00 68 00 30 40 00 68 1E 30 40 00 6A 00 E8 0D 00 00 00 6A 00 E8 00 00 00 00 FF 25 00 20 40 00 FF 25 08 20 40 }
	condition:
		$1 at pe.entry_point
}

rule ms_basic_560 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Basic"
		version = "5.60"
		description = "1982 - 1997"
	strings:
		$1 = { 9A ?? ?? ?? ?? 9A ?? ?? ?? ?? 9A ?? ?? ?? ?? 33 DB BA ?? ?? 9A ?? ?? ?? ?? C7 06 ?? ?? ?? ?? 33 DB }
	condition:
		$1 at pe.entry_point
}

rule ms_cpp_1990_1992 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft C++"
		description = "1990, 1992"
	strings:
		$1 = { B8 00 30 CD 21 3C 03 73 ?? 0E 1F BA ?? ?? B4 09 CD 21 06 33 C0 50 CB }
	condition:
		$1 at pe.entry_point
}

rule ms_fortran {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft FORTRAN"
	strings:
		$1 = { FC 1E B8 ?? ?? 8E D8 9A ?? ?? ?? ?? 81 ?? ?? ?? 8B EC 8C DB 8E C3 BB ?? ?? B9 ?? ?? 9A ?? ?? ?? ?? 80 ?? ?? ?? ?? 74 ?? E9 }
	condition:
		$1 at pe.entry_point
}

rule ms_visual_basic_50_60_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "5.0 - 6.0"
	strings:
		$1 = { 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 00 00 00 00 00 00 30 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ms_visual_basic_50_60_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "5.0 - 6.0"
	strings:
		$1 = { FF 25 ?? ?? ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule ms_visual_basic_50_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "5.0"
		start = 7
	strings:
		$1 = { FF FF FF 00 00 00 00 00 00 30 00 00 00 40 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point + 7
}

rule ms_visual_basic_50_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "5.0"
	strings:
		$1 = { FF FF FF 00 00 00 00 00 00 30 00 00 00 40 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ms_visual_basic_60_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "6.0"
	strings:
		$1 = { 5A 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 52 E9 ?? ?? FF }
	condition:
		$1 at pe.entry_point
}

rule ms_visual_basic_60_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "6.0"
	strings:
		$1 = { 5A 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 52 E9 ?? ?? ?? ?? 00 00 00 ?? 00 00 00 30 00 00 00 ?? 00 00 00 ?? 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule ms_visual_basic_60_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Microsoft Visual Basic"
		version = "6.0"
	strings:
		$1 = { FF 25 ?? ?? ?? ?? 68 ?? ?? ?? ?? E8 ?? FF FF FF ?? ?? ?? ?? ?? ?? 30 }
	condition:
		$1 at pe.entry_point
}

rule dotnet_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = ".NET"
		bytecode = true
	strings:
		$1 = { 00 00 00 00 00 00 00 00 5F 43 6F 72 44 6C 6C 4D 61 69 6E 00 6D 73 63 6F 72 65 65 2E 64 6C 6C 00 00 ?? 00 00 FF 25 }
	condition:
		$1 at pe.entry_point
}

rule dotnet_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = ".NET"
		bytecode = true
	strings:
		$1 = { 00 00 00 00 00 00 00 00 5F 43 6F 72 45 78 65 4D 61 69 6E 00 6D 73 63 6F 72 65 65 2E 64 6C 6C 00 00 00 00 00 FF 25 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 81 7C 24 04 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 83 7C 24 04 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 44 24 04  }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 44 24 08 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_05 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 44 24 0C }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_06 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 4C 24 04 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_07 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 4C 24 08 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_08 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 4C 24 0C }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_09 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B C0 55 8B EC }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_10 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 83 7D 0C }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_11 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8D 6C 24 94 81 EC }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_12 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B 44 24 ?? 8B EC }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_13 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 83 7C 24 0C 01 75 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_14 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { E8 ?? ?? 00 00 E9 95 FE FF FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_15 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { E8 ?? ?? 00 00 E9 41 FE FF FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_16 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { E8 ?? ?? 00 00 E9 87 FE FF FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_17 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 33 C0 40 39 44 24 08 75 ?? 8B ?? 24 04 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_18 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 33 C0 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_19 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 33 DB }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_20 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 33 F6 }
	condition:
		$1 at pe.entry_point
}


rule msvc_uv_22 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 65 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_23 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 7D }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_24 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8B 45 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_25 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 8D 45 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_26 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 55 56 8B 74 24 14 85 F6 57 B8 01 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_27 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 55 8B 6C 24 10 56 57 6A 01 5F B3 01 3B EF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_28 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 56 57 BB 01 00 00 00 8B 7C 24 14 55 3B FB }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_29 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 56 57 BB 01 00 00 00 8B 7C 24 14 55 85 FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_30 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? B? 94 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_31 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 56 8B 74 24 0C 83 FE 01 75 05 E8 ?? ?? ?? ?? 8B 44 24 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_32 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 33 C0 40 83 7C 24 08 00 75 05 A3 D4 50 A7 64 C2 0C 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_33 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 51 A1 E0 30 40 00 53 56 57 33 FF 3B C7 74 0A FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_34 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 51 C7 45 FC 01 00 00 00 83 7D 0C 00 75 10 83 3D }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_35 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B C0 53 55 56 8B 74 24 14 85 F6 57 B8 01 00 00 00 0F 84 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_36 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 66 81 3D ?? ?? ?? ?? 4D 5A }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_37 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 56 33 F6 56 56 56 56 FF 15 08 10 00 01 56 FF 15 00 10 00 01 5E C3 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_38 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 33 C0 83 7C 24 08 01 0F 94 C0 50 E8 ?? ?? ?? ?? 33 C0 59 40 C2 0C 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_39 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 6A ?? 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? E8 ?? ?? ?? ?? 59 A3 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_40 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 33 C0 50 50 50 50 FF 15 ?? ?? ?? ?? 50 E8 ?? ?? ?? ?? 50 FF 15 ?? ?? ?? ?? C3 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_41 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 ?? ?? ?? ?? ?? ?? 64 89 25 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_42 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 83 7C 24 08 01 75 05 E8 ?? ?? ?? ?? FF 74 24 04 8B 4C 24 10 8B 54 24 0C E8 ?? ?? ?? ?? 59 C2 0C 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_43 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?0 ?? 40 00 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 51 B8 ?? 7? 01 00 E8 ?D E? FF FF 53 56 57 89 65 F0 C? 85 ?? ?? F? FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_45 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 83 ?? ?? 6A 00 FF 15 F8 10 0B B0 8D ?? ?? ?? 51 6A 08 6A 00 6A 00 68 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_46 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 56 57 BF 01 00 00 00 8B 75 0C }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_47 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 55 56 8B ?? ?? ?? 85 F6 57 B8 ?? ?? ?? ?? 75 ?? 8B ?? ?? ?? ?? ?? 85 C9 75 ?? 33 C0 5F 5E 5D 5B C2 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_48 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 44 24 08 83 ?? ?? 74 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_49 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 8B 44 24 08 56 83 E8 ?? 74 ?? 48 75 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_50 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 56 57 BF ?? ?? ?? ?? 8B ?? ?? 3B F7 0F }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_51 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 56 57 55 BB 01 00 00 00 8B ?C 24 18 3B ?? 75 2A A1 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_52 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 56 57 64 A1 20 00 00 00 8B D8 8B 74 24 10 85 F6 0F 85 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_53 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 6A FF 68 68 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 EC 53 56 57 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_54 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 53 56 57 8B 7C 24 14 83 FF 01 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_55 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 ?? ?? ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_56 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 80 3D ?? ?? ?? ?? 00 75 12 E8 12 00 00 00 84 C0 B0 00 74 09 C6 05 ?? ?? ?? ?? 01 B0 01 C2 0C 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_57 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 68 ?? ?? ?? ?? 64 A1 00 00 00 00 50 8B 44 24 10 89 6C 24 10 8D 6C 24 10 2B E0 53 56 57 8B 45 F8 89 65 E8 50 8B 45 FC C7 45 FC FF FF FF FF 89 45 F8 8D 45 F0 64 A3 00 00 00 00 C3 8B 4D F0 64 89 0D 00 00 00 00 59 5F 5E 5B C9 51 C3 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_58 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { 0F BF 44 24 08 83 E8 00 74 25 48 75 39 8B 44 24 04 50 A3 ?? ?? ?? ?? E8 ?? ?? ?? ?? FF 35 ?? ?? ?? ?? A3 ?? ?? ?? ?? FF 15 ?? ?? ?? ?? EB 17 A1 ?? ?? ?? ?? 85 C0 74 07 50 FF 15 ?? ?? ?? ?? 83 }
	condition:
		$1 at pe.entry_point
}


rule msvc_uv_60 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { FF 25 00 ?0 40 00 00 00 0? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?0 ?0 0? ?? ?F ?? ?? ?? 0? 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?0 ?? 0? ?? ?F ?? ?? ?? 0? 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?0 ?0 ?0 ?? ?F ?? ?? ?? 0? 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?? ?? ?? ?? 0? 0? }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_61 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
	strings:
		$1 = { FF 25 ?? ?0 4? 00 CC CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 0C 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 02 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 07 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 0F 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_uv_debug {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "debug"
	strings:
		$1 = { E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 E9 ?? ?? 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_20_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "2.0"
	strings:
		$1 = { 64 A1 00 00 00 00 55 8B EC 6A FF 68 }
	condition:
		$1 at pe.entry_point
}

rule msvc_20_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "2.0"
	strings:
		$1 = { 53 56 57 BB ?? ?? ?? ?? 8B ?? ?? ?? 55 3B FB 75 }
	condition:
		$1 at pe.entry_point
}

rule msvc_30 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "3.0"
	strings:
		$1 = { 64 A1 00 00 00 00 55 ?? ?? 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 50 ?? ?? ?? ?? ?? 00 00 83 EC 10 }
	condition:
		$1 at pe.entry_point
}

rule msvc_42_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "4.2"
	strings:
		$1 = { 64 A1 00 00 00 00 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 50 64 ?? ?? ?? ?? ?? ?? 83 ?? ?? 53 56 57 89 }
	condition:
		$1 at pe.entry_point
}

rule msvc_42_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "4.2"
	strings:
		$1 = { 53 B8 01 00 00 00 8B 5C 24 0C 56 57 85 DB 55 75 }
	condition:
		$1 at pe.entry_point
}

rule msvc_4x_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "4.x"
	strings:
		$1 = { 64 A1 00 00 00 00 55 89 E5 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 50 64 89 25 00 00 00 00 83 EC ?? 53 56 57 }
	condition:
		$1 at pe.entry_point
}

rule msvc_4x_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "4.x"
	strings:
		$1 = { 64 A1 00 00 00 00 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 50 64 89 25 00 00 00 00 83 EC ?? 53 56 57 }
	condition:
		$1 at pe.entry_point
}

rule msvc_50_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "5.0"
		description = "Visual Studio 97"
	strings:
		$1 = { 64 A1 00 00 00 00 50 64 89 25 00 00 00 00 83 C4 A8 53 56 57 }
	condition:
		$1 at pe.entry_point
}

rule msvc_50_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "5.0"
		description = "Visual Studio 97"
	strings:
		$1 = { ?? ?? 24 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? 8B ?? 24 0C ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 8B 44 ?? 08 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 83 EC 44 56 FF 15 ?? ?? ?? ?? 6A 01 8B F0 FF 15 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 83 EC 44 56 FF 15 ?? ?? ?? ?? 8B F0 8A ?? 3C 22 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 83 7C 24 08 01 75 09 8B 44 24 04 A3 ?? ?? 00 10 E8 8B FF FF FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_05 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 83 EC 50 53 56 57 BE ?? ?? ?? ?? 8D 7D F4 A5 A5 66 A5 8B }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_06 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8D 6C ?? ?? 81 EC ?? ?? ?? ?? 8B 45 ?? 83 F8 01 56 0F 84 ?? ?? ?? ?? 85 C0 0F 84 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_07 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 6A FF 68 ?? ?? ?? ?? 68 ?? ?? ?? ?? 64 A1 ?? ?? ?? ?? 50 64 89 25 ?? ?? ?? ?? 83 EC ?? 53 56 57 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_08 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 53 8B 5D 08 56 8B 75 0C 57 8B 7D 10 85 F6 75 09 83 3D ?? ?? ?? ?? ?? EB 26 83 FE 01 74 05 83 FE 02 75 22 A1 ?? ?? ?? ?? 85 C0 74 09 57 56 53 FF D0 85 C0 74 0C 57 56 53 E8 15 FF FF FF 85 C0 75 04 33 C0 EB 4E }
	condition:
		$1 at pe.entry_point
}



rule msvc_60_debug_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0 debug"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 53 8B 5D 08 56 8B 75 0C 57 8B 7D 10 85 F6 ?? ?? 83 }
	condition:
		$1 at pe.entry_point
}

rule msvc_60_debug_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "6.0 debug"
		description = "Visual Studio 6.0"
	strings:
		$1 = { 55 8B EC 51 ?? ?? ?? 01 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 10 ?? ?? ?? ?? ?? ?? ?? ?? ?? E8 }
	condition:
		$1 at pe.entry_point
}

rule msvc_70_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.0"
		description = "Visual Studio .NET 2002"
	strings:
		$1 = { 55 8B EC 53 8B 5D 08 56 8B 75 0C 85 F6 57 8B 7D 10 }
	condition:
		$1 at pe.entry_point
}

rule msvc_70_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.0"
		description = "Visual Studio .NET 2002"
	strings:
		$1 = { 6A 0C 68 88 BF 01 10 E8 B8 1C 00 00 33 C0 40 89 45 E4 8B 75 0C 33 FF 3B F7 75 0C 39 3D 6C 1E 12 10 0F 84 B3 00 00 00 89 7D FC 3B F0 74 05 83 FE 02 75 31 A1 98 36 12 10 3B C7 74 0C FF 75 10 56 }
	condition:
		$1 at pe.entry_point
}

rule msvc_70_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.0"
		description = "Visual Studio .NET 2002"
	strings:
		$1 = { ?? ?? 41 00 00 00 00 00 00 00 63 00 00 00 00 00 ?? 00 ?? ?? ?? ?? ?? 00 ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? 00 ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? 20 ?? ?? 00 ?? 00 ?? ?? ?? ?? ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 ?? 00 ?? ?? ?? 00 ?? ?? ?? ?? ?? ?? ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? ?? ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? ?? ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? 00 ?? ?? ?? 00 ?? 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_71_debug_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.1 debug"
		description = "Visual Studio .NET 2003"
	strings:
		$1 = { 55 8B EC 53 8B 5D 08 56 8B 75 0C 85 F6 57 8B 7D 10 75 09 83 3D ?? ?? 40 00 00 EB 26 83 FE 01 74 05 83 FE 02 75 22 A1 }
	condition:
		$1 at pe.entry_point
}

rule msvc_71_debug_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.1 debug"
		description = "Visual Studio .NET 2003"
	strings:
		$1 = { 55 8B EC ?? ?? 0C 83 ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? ?? 00 00 00 8B }
	condition:
		$1 at pe.entry_point
}

rule msvc_71_debug_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.1 debug"
		description = "Visual Studio .NET 2003"
	strings:
		$1 = { 55 8B EC 83 EC 08 53 56 57 55 FC 8B 5D 0C 8B 45 08 F7 40 04 06 00 00 00 0F 85 AB 00 00 00 89 45 F8 8B 45 10 89 45 FC 8D 45 F8 89 43 FC 8B 73 0C 8B 7B 08 53 E8 ?? ?? ?? ?? 83 C4 04 0B C0 74 7B 83 FE FF 74 7D 8D 0C 76 8B 44 8F 04 0B C0 74 59 56 55 }
	condition:
		$1 at pe.entry_point
}

rule msvc_71_debug_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "7.1 debug"
		description = "Visual Studio .NET 2003"
	strings:
		$1 = { 8B FF 55 8B EC 56 33 F6 39 75 0C 75 0E 39 35 ?? ?? ?? ?? 7E 2D FF 0D ?? ?? ?? ?? 83 7D 0C 01 A1 ?? ?? ?? ?? 8B 00 A3 ?? ?? ?? ?? 75 3D 68 80 00 00 00 FF 15 ?? ?? ?? ?? 3B C6 59 A3 ?? ?? ?? ?? 75 04 33 C0 EB 67 89 30 A1 ?? ?? ?? ?? 68 ?? ?? ?? ?? 68 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { E8 ?? 0? 00 00 E9 36 FD FF FF }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { 83 EC 04 83 7C 24 0C 01 56 0F 85 ?? 00 00 00 68 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { 83 EC 44 56 FF 15 ?? ?? ?? ?? 8B F0 8A 0? 3C 22 75 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_04 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { 8B FF 55 8B EC 83 7D 0C 01 75 05 E8 ?? ?? ?? FF 5D E9 D6 FE FF FF CC CC CC CC CC }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_05 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { 8B FF 55 8B EC 53 8B 5D 08 56 8B 75 0C 85 F6 57 8B 7D 10 0F 84 ?? ?? 00 00 83 FE 01 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_06 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { 6A 14 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? BB 94 00 00 00 53 6A 00 8B ?? ?? ?? ?? ?? FF D7 50 FF ?? ?? ?? ?? ?? 8B F0 85 F6 75 0A 6A 12 E8 ?? ?? ?? ?? 59 EB 18 89 1E 56 FF ?? ?? ?? ?? ?? 56 85 C0 75 14 50 FF D7 50 FF ?? ?? ?? ?? ?? B8 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_07 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { E8 ?? 03 00 00 E9 9E FD FF FF 55 8B EC 81 EC 28 03 00 00 A3 ?? ?? 40 00 89 0D ?? ?? 40 00 89 15 ?? ?? 40 00 89 1D ?? ?? 40 00 89 35 ?? ?? 40 00 89 3D ?? ?? 40 00 66 8C 15 ?? ?? 40 00 66 8C 0D ?? ?? 40 00 66 8C 1D ?? ?? 40 00 66 8C 05 ?? ?? 40 00 66 8C 25 ?? ?? 40 00 66 8C 2D ?? ?? 40 00 9C 8F 05 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_08 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0 debug"
		description = "Visual Studio 2005"
	strings:
		$1 = { 55 8B EC E8 ?8 0? 00 00 E8 03 00 00 00 5D C3 CC 55 8B EC 6A FE 68 ?? ?? 40 00 68 ?0 ?? 40 00 64 A1 00 00 00 00 50 83 C4 E4 53 56 57 A1 ?? ?0 40 00 31 45 F8 33 C5 50 8D 45 F0 64 A3 00 00 00 00 89 65 E8 C7 45 FC 00 00 00 00 C7 45 DC 00 00 00 00 E8 0A 02 00 00 8B 40 04 89 45 E0 C7 45 E4 00 00 00 00 6A }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_10 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { FF 25 00 30 40 00 00 00 0? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?? ?? 00 00 00 00 00 ?0 ?? ?? ?? 0? 00 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?? ?? ?? ?? 00 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?? ?? ?? 0? 00 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?? ?? ?? 0? 00 0? ?? ?? ?? ?? ?? 0? }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_11 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0"
		description = "Visual Studio 2005"
	strings:
		$1 = { E8 ?? ?? ?? ?? E9 8D FE FF FF CC CC CC CC CC 66 81 3D 00 00 00 01 4D 5A 74 04 33 C0 EB 51 A1 3C 00 00 01 81 B8 00 00 00 01 50 45 00 00 75 EB 0F B7 88 18 00 00 01 81 F9 0B 01 00 00 74 1B 81 F9 0B 02 00 00 75 D4 83 B8 84 00 00 01 0E 76 CB 33 C9 39 88 F8 00 00 01 EB 11 83 B8 74 00 00 01 0E 76 B8 33 C9 39 88 E8 00 00 01 0F 95 C1 8B C1 6A 01 A3 ?? ?? ?? 01 E8 ?? ?? 00 00 50 FF ?? ?? ?? 00 01 83 0D ?? ?? ?? 01 FF 83 0D ?? ?? ?? 01 FF 59 59 FF 15 ?? ?? 00 01 8B 0D ?? ?? ?? 01 89 08 FF 15 ?? ?? 00 01 8B }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_debug_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0 debug"
		description = "Visual Studio 2005"
	strings:
		$1 = { FF 25 00 30 40 00 00 00 03 30 01 00 0F 00 00 00 00 00 00 00 23 00 00 00 00 00 00 F0 7F 80 07 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 12 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 04 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 0C 00 00 04 2A CC 03 30 01 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_80_debug_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "8.0 debug"
		description = "Visual Studio 2005"
	strings:
		$1 = { FF 25 8? ?0 4? 00 CC CC CC CC CC CC CC CC CC CC CC CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 0D 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 03 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 08 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 10 00 00 04 2A CC 03 30 }
	condition:
		$1 at pe.entry_point
}

rule msvc_90_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "9.0"
		description = "Visual Studio 2008"
	strings:
		$1 = { E8 ?? 04 00 00 E9 9F FD FF FF 8B FF 55 8B EC 81 EC 28 03 00 00 A3 ?? ?? 40 00 89 0D ?? ?? 40 00 89 15 ?? ?? 40 00 89 1D ?? ?? 40 00 89 35 ?? ?? 40 00 89 3D ?? ?? 40 00 66 8C 15 ?? ?? 40 00 66 8C 0D ?? ?? 40 00 66 8C 1D ?? ?? 40 00 66 8C 05 ?? ?? 40 00 66 8C 25 ?? ?? 40 00 66 8C 2D ?? ?? 40 00 9C 8F }
	condition:
		$1 at pe.entry_point
}

rule msvc_90_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "9.0"
		description = "Visual Studio 2008"
	strings:
		$1 = { FF 25 00 ?0 40 00 00 00 00 03 30 01 00 0B 00 00 00 00 00 00 00 20 00 01 FF 0F 80 ?? 00 00 04 2A CC 03 30 01 00 0B 00 00 00 00 00 00 00 20 01 00 FF 0F 80 ?? 00 00 04 2A CC 03 30 01 00 0B 00 00 00 00 00 00 00 20 10 00 FF 0F 80 ?? 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 ?? 00 00 04 2A }
	condition:
		$1 at pe.entry_point
}

rule msvc_90_debug_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "9.0 debug"
		description = "Visual Studio 2008"
	strings:
		$1 = { FF 25 00 ?0 40 00 00 00 0? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?0 ?0 0? ?? ?F ?? ?? 0? 00 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?0 ?? 0? ?? ?F ?? ?? 0? 00 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?0 ?0 ?0 ?? ?F ?? ?? 0? 00 0? ?? ?? ?? ?? ?? 0? 0? 0? 00 00 00 00 00 00 ?? ?? ?? 0? 00 0? }
	condition:
		$1 at pe.entry_point
}

rule msvc_90_debug_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "9.0 debug"
		description = "Visual Studio 2008"
	strings:
		$1 = { FF 25 ?? ?? ?0 00 CC CC CC CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 ?? 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 ?? 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 ?? 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 ?? 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 }
	condition:
		$1 at pe.entry_point
}

rule msvc_90_debug_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "9.0 debug"
		description = "Visual Studio 2008"
	strings:
		$1 = { 8B FF 55 8B EC E8 ?6 0? 00 00 E8 11 00 00 00 5D C3 CC CC CC CC CC CC CC CC CC CC CC CC CC CC CC 8B FF 55 8B EC 6A FE 68 ?? ?? 40 00 68 ?0 ?? 40 00 64 A1 00 00 00 00 50 83 C4 E4 53 56 57 A1 ?? ?0 40 00 31 45 F8 33 C5 50 8D 45 F0 64 A3 00 00 00 00 89 65 E8 C7 45 FC 00 00 00 00 C7 45 DC 00 00 00 00 E8 }
	condition:
		$1 at pe.entry_point
}

rule msvc_10_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "10.0"
		description = "Visual Studio 2010"
	strings:
		$1 = { E8 ?? 04 00 00 E9 B3 FD FF FF 8B FF 55 8B EC 81 EC 28 03 00 00 A3 ?? ?? 40 00 89 0D ?? ?? 40 00 89 15 ?? ?? 40 00 89 1D ?? ?? 40 00 89 35 ?? ?? 40 00 89 3D ?? ?? 40 00 66 8C 15 ?? ?? 40 00 66 8C 0D ?? ?? 40 00 66 8C 1D ?? ?? 40 00 66 8C 05 ?? ?? 40 00 66 8C 25 ?? ?? 40 00 66 8C 2D ?? ?? 40 00 9C 8F }
	condition:
		$1 at pe.entry_point
}

rule msvc_10_debug_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "10.0 debug"
		description = "Visual Studio 2010"
	strings:
		$1 = { 8B FF 55 8B EC E8 ?6 0? 00 00 E8 11 00 00 00 5D C3 CC CC CC CC CC CC CC CC CC CC CC CC CC CC CC 8B FF 55 8B EC 6A FE 68 ?? ?? 40 00 68 ?0 ?? 40 00 64 A1 00 00 00 00 50 83 C4 E4 53 56 57 A1 ?0 ?0 40 00 31 45 F8 33 C5 50 8D 45 F0 64 A3 00 00 00 00 89 65 E8 83 3D ?? ?? 40 00 00 75 0E 6A 00 6A 00 6A 01 }
	condition:
		$1 at pe.entry_point
}

rule msvc_10_debug_02 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "10.0 debug"
		description = "Visual Studio 2010"
	strings:
		$1 = { FF 25 00 ?0 4? 00 00 00 03 30 01 00 0B 00 00 00 00 00 00 00 20 00 01 FF 0F 80 ?? 00 00 04 2A CC 03 30 01 00 0B 00 00 00 00 00 00 00 20 01 00 FF 0F 80 ?? 00 00 04 2A CC 03 30 01 00 0B 00 00 00 00 00 00 00 20 10 00 FF 0F 80 ?? 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 ?? 00 00 04 2A CC }
	condition:
		$1 at pe.entry_point
}

rule msvc_10_debug_03 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MSVC"
		version = "10.0 debug"
		description = "Visual Studio 2010"
	strings:
		$1 = { FF 25 8? ?0 4? 00 CC CC CC CC CC CC CC CC CC CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 0F 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 05 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 0A 00 00 04 2A CC 03 30 01 00 07 00 00 00 00 00 00 00 16 80 12 00 00 04 2A CC 03 30 01 00 }
	condition:
		$1 at pe.entry_point
}

rule metalang_uv {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "MetaLang"
	strings:
		$1 = { E9 FF 00 00 00 60 8B 74 24 24 8B 7C 24 28 FC B2 80 33 DB A4 B3 02 E8 6D 00 00 00 73 F6 33 C9 E8 64 00 00 00 73 1C 33 C0 E8 5B 00 00 00 73 23 B3 02 41 B0 10 E8 4F 00 00 00 12 C0 73 F7 75 3F AA }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule ocbat2exe {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "ocBat2Exe"
		version = "1.0"
	strings:
		$1 = { 55 8B EC B9 08 00 00 00 6A 00 6A 00 49 75 F9 53 56 57 B8 58 3C 40 00 E8 6C FA FF FF 33 C0 55 68 8A 3F 40 00 64 FF 30 64 89 20 6A 00 6A 00 6A 03 6A 00 6A 01 68 00 00 00 80 8D 55 EC 33 C0 E8 81 E9 FF FF 8B 45 EC E8 41 F6 FF FF 50 E8 F3 FA FF FF 8B F8 83 FF FF 0F 84 83 02 00 00 6A 02 6A 00 6A EE 57 E8 FC FA FF FF 6A 00 68 60 99 4F 00 6A 12 68 18 57 40 00 57 E8 E0 FA FF FF 83 3D 60 99 4F 00 12 0F 85 56 02 00 00 8D 45 E4 50 8D 45 E0 BA 18 57 40 00 B9 40 42 0F 00 E8 61 F4 FF FF 8B 45 E0 B9 12 00 00 00 BA 01 00 00 00 E8 3B F6 FF FF 8B 45 E4 8D 55 E8 E8 04 FB ?? ?? ?? ?? E8 B8 58 99 4F 00 E8 67 F3 FF FF 33 C0 A3 60 99 4F 00 8D 45 DC 50 B9 05 00 00 00 BA 01 00 00 00 A1 58 99 4F 00 E8 04 F6 FF FF 8B 45 DC BA A4 3F 40 00 E8 E3 F4 FF FF }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule f2ko_bat2exe_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "F2KO Bat2Exe"
	strings:
		$1 = { 68 ?? ?? 00 00 68 00 00 00 00 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 83 C4 0C 68 00 00 00 00 E8 ?? ?? ?? ?? A3 ?? ?? ?? ?? 68 00 00 00 00 68 00 10 00 00 68 00 00 00 00 E8 ?? ?? ?? ?? A3 }
	condition:
		$1 at pe.entry_point
}

rule adv_bat_to_exe_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "Advanced BAT to EXE Converter"
	strings:
		$1 = { B9 4F C3 00 00 33 C0 8D BD ?? ?? ?? ?? F3 AB 66 AB AA A0 ?? ?? ?? ?? 88 85 ?? ?? ?? ?? B9 69 18 00 00 33 C0 8D BD ?? ?? ?? ?? F3 AB 66 AB AA 8A 0D ?? ?? ?? ?? 88 8D ?? ?? ?? ?? B9 40 00 00 00 33 C0 8D BD ?? ?? ?? ?? F3 AB 66 AB AA 8A 15 ?? ?? ?? ?? 88 95 ?? ?? ?? ?? B9 59 00 00 00 33 C0 8D }
	condition:
		$1
}

rule exescript_uv_01 {
	meta:
        author = "RetDec Team"
		category = "compiler"
		name = "ExeScript"
	strings:
		$1 = { 55 8B EC 81 EC ?? ?? ?? ?? 56 6A 00 FF 15 ?? ?? ?? ?? 89 85 ?? ?? ?? ?? C7 85 ?? ?? ?? ?? 00 00 00 00 C7 45 ?? ?? ?? ?? 00 FF 15 ?? ?? ?? ?? A3 }
		$2 = "<!-- ----- ExeScript Options Begin -----"
	condition:
		$1 at pe.entry_point and
		@2 > pe.sections[pe.section_index(".rdata")].raw_data_offset and
		@2 < pe.sections[pe.section_index(".rdata")].raw_data_offset +
			pe.sections[pe.section_index(".rdata")].raw_data_size
}

rule plugintoexe_100 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "PluginToExe"
		version = "1.00"
		description = "for PEiD"
	strings:
		$1 = { E8 00 00 00 00 29 C0 5D 81 ED D1 40 40 00 50 FF 95 B8 40 40 00 89 85 09 40 40 00 FF 95 B4 40 40 00 89 85 11 40 40 00 50 FF 95 C0 40 40 00 8A 08 80 F9 22 75 07 50 FF 95 C4 40 40 00 89 85 0D 40 40 00 8B 9D 09 40 40 00 60 6A 00 6A 01 53 81 C3 ?? ?? ?? 00 FF D3 61 6A 00 68 44 69 45 50 FF B5 0D 40 40 00 6A 00 81 C3 ?? ?? ?? 00 FF D3 83 C4 10 FF 95 B0 40 40 00 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule plugintoexe_101 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "PluginToExe"
		version = "1.01"
		description = "for PEiD"
	strings:
		$1 = { E8 00 00 00 00 29 C0 5D 81 ED C6 41 40 00 50 8F 85 71 40 40 00 50 FF 95 A5 41 40 00 89 85 6D 40 40 00 FF 95 A1 41 40 00 50 FF 95 B5 41 40 00 80 38 00 74 16 8A 08 80 F9 22 75 07 50 FF 95 B9 41 40 00 89 85 75 40 40 00 EB 6C 6A 01 8F 85 71 40 40 00 6A 58 6A 40 FF 95 A9 41 40 00 89 85 69 40 40 00 89 C7 68 00 08 00 00 6A 40 FF 95 A9 41 40 00 89 47 1C C7 07 58 00 00 00 C7 47 20 00 08 00 00 C7 47 18 01 00 00 00 C7 47 34 04 10 88 00 8D 8D B9 40 40 00 89 4F 0C 8D 8D DB 40 40 00 89 4F 30 FF B5 69 40 40 00 FF 95 95 41 40 00 FF 77 1C 8F 85 75 40 40 00 8B 9D 6D 40 40 00 60 6A 00 6A 01 53 81 C3 ?? ?? ?? 00 FF D3 61 6A 00 68 44 69 45 50 FF B5 75 40 40 00 6A 00 81 C3 ?? ?? 00 00 FF D3 83 C4 10 83 BD 71 40 40 00 00 74 10 FF 77 1C FF 95 AD 41 40 00 57 FF 95 AD 41 40 00 6A 00 FF 95 9D 41 40 00 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule plugintoexe_102 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "PluginToExe"
		version = "1.02"
		description = "for PEiD"
	strings:
		$1 = { E8 00 00 00 00 29 C0 5D 81 ED 32 42 40 00 50 8F 85 DD 40 40 00 50 FF 95 11 42 40 00 89 85 D9 40 40 00 FF 95 0D 42 40 00 50 FF 95 21 42 40 00 80 38 00 74 16 8A 08 80 F9 22 75 07 50 FF 95 25 42 40 00 89 85 E1 40 40 00 EB 6C 6A 01 8F 85 DD 40 40 00 6A 58 6A 40 FF 95 15 42 40 00 89 85 D5 40 40 00 89 C7 68 00 08 00 00 6A 40 FF 95 15 42 40 00 89 47 1C C7 07 58 00 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule reg2exe_220_221 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "Reg2Exe"
		version = "2.20, 2.21"
	strings:
		$1 = { 6A 00 E8 7D 12 00 00 A3 A0 44 40 00 E8 79 12 00 00 6A 0A 50 6A 00 FF 35 A0 44 40 00 E8 0F 00 00 00 50 E8 69 12 00 00 CC CC CC CC CC CC CC CC CC 68 2C 02 00 00 68 00 00 00 00 68 B0 44 40 00 E8 3A 12 00 00 83 C4 0C 8B 44 24 04 A3 B8 44 40 00 68 00 00 00 00 68 A0 0F 00 00 68 00 00 00 00 E8 32 12 00 00 A3 B0 44 40 00 68 F4 01 00 00 68 BC 44 40 00 FF 35 B8 44 40 00 E8 1E 12 00 00 B8 BC 44 40 00 89 C1 8A 30 40 80 FE 5C 75 02 89 C1 80 FE 00 75 F1 C6 01 00 E8 EC 18 00 00 E8 28 16 00 00 E8 4A 12 00 00 68 00 FA 00 00 68 08 00 00 00 FF 35 B0 44 40 00 E8 E7 11 00 00 A3 B4 44 40 00 8B 15 D4 46 40 00 E8 65 0A 00 00 BB 00 00 10 00 B8 01 00 00 00 E8 72 0A 00 00 74 09 C7 00 01 00 00 00 83 C0 04 A3 D4 46 40 00 FF 35 B4 44 40 00 E8 26 05 00 00 8D 0D B8 46 40 00 5A E8 CF 0F 00 00 FF 35 B4 44 40 00 FF 35 B8 46 40 00 E8 EE 06 00 00 8D 0D B4 46 40 00 5A E8 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule reg2exe_222_223 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "Reg2Exe"
		version = "2.22, 2.23"
	strings:
		$1 = { 6A 00 E8 2F 1E 00 00 A3 C4 35 40 00 E8 2B 1E 00 00 6A 0A 50 6A 00 FF 35 C4 35 40 00 E8 07 00 00 00 50 E8 1B 1E 00 00 CC 68 48 00 00 00 68 00 00 00 00 68 C8 35 40 00 E8 76 16 00 00 83 C4 0C 8B 44 24 04 A3 CC 35 40 00 68 00 00 00 00 68 A0 0F 00 00 68 00 00 00 00 E8 EC 1D 00 00 A3 C8 35 40 00 E8 62 1D 00 00 E8 92 1A 00 00 E8 80 16 00 00 E8 13 14 00 00 68 01 00 00 00 68 08 36 40 00 68 00 00 00 00 8B 15 08 36 40 00 E8 71 3F 00 00 B8 00 00 10 00 BB 01 00 00 00 E8 82 3F 00 00 FF 35 48 31 40 00 B8 00 01 00 00 E8 0D 13 00 00 8D 0D EC 35 40 00 5A E8 F2 13 00 00 68 00 01 00 00 FF 35 EC 35 40 00 E8 84 1D 00 00 A3 F4 35 40 00 FF 35 48 31 40 00 FF 35 F4 35 40 00 FF 35 EC 35 40 00 E8 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule reg2exe_224 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "Reg2Exe"
		version = "2.24"
	strings:
		$1 = { 6A 00 E8 CF 20 00 00 A3 F4 45 40 00 E8 CB 20 00 00 6A 0A 50 6A 00 FF 35 F4 45 40 00 E8 07 00 00 00 50 E8 BB 20 00 00 CC 68 48 00 00 00 68 00 00 00 00 68 F8 45 40 00 E8 06 19 00 00 83 C4 0C 8B 44 24 04 A3 FC 45 40 00 68 00 00 00 00 68 A0 0F 00 00 68 00 00 00 00 E8 8C 20 00 00 A3 F8 45 40 00 E8 02 20 00 00 E8 32 1D 00 00 E8 20 19 00 00 E8 A3 16 00 00 68 01 00 00 00 68 38 46 40 00 68 00 00 00 00 8B 15 38 46 40 00 E8 71 4F 00 00 B8 00 00 10 00 BB 01 00 00 00 E8 82 4F 00 00 FF 35 48 41 40 00 B8 00 01 00 00 E8 9D 15 00 00 8D 0D 1C 46 40 00 5A E8 82 16 00 00 68 00 01 00 00 FF 35 1C 46 40 00 E8 24 20 00 00 A3 24 46 40 00 FF 35 48 41 40 00 FF 35 24 46 40 00 FF 35 1C 46 40 00 E8 DC 10 00 00 8D 0D 14 46 40 00 5A E8 4A 16 }
	condition:
		for any of them : ( $ at pe.entry_point )
}

rule reg2exe_225 {
	meta:
        author = "RetDec Team"
		category = "framework"
		name = "Reg2Exe"
		version = "2.25"
	strings:
		$1 = { 68 68 00 00 00 68 00 00 00 00 68 70 7D 40 00 E8 AE 20 00 00 83 C4 0C 68 00 00 00 00 E8 AF 52 00 00 A3 74 7D 40 00 68 00 00 00 00 68 00 10 00 00 68 00 00 00 00 E8 9C 52 00 00 A3 70 7D 40 00 E8 24 50 00 00 E8 E2 48 00 00 E8 44 34 00 00 E8 54 28 00 00 E8 98 27 00 00 E8 93 20 00 00 68 01 00 00 00 68 D0 7D 40 00 68 00 00 00 00 8B 15 D0 7D 40 00 E8 89 8F 00 00 B8 00 00 10 00 68 01 00 00 00 E8 9A 8F 00 00 FF 35 A4 7F 40 00 68 00 01 00 00 E8 3A 23 00 00 8D 0D A8 7D 40 00 5A E8 5E 1F 00 00 FF 35 A8 7D 40 00 68 00 01 00 00 E8 2A 52 00 00 A3 B4 7D 40 00 FF 35 A4 7F 40 00 FF 35 B4 7D 40 00 FF 35 A8 7D 40 00 E8 5C 0C 00 00 8D 0D A0 7D 40 00 5A E8 26 1F 00 00 FF 35 }
	condition:
		for any of them : ( $ at pe.entry_point )
}



