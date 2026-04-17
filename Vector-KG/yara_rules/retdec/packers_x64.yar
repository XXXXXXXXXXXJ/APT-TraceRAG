/*
 * YARA rules for x64 PE packer detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "pe"

rule upx_39x_lzma_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 53 56 57 55 48 8D 35 ?? ?? ?? ?? 48 8D BE ?? ?? ?? ?? 57 B8 ?? ?? ?? ?? 50 48 89 E1 48 89 FA 48 89 F7 BE ?? ?? ?? ?? 55 48 89 E5 44 8B 09 49 89 D0 48 89 F2 48 8D 77 02 56 8A 07 FF CA 88 C1 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2b_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.91 [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 53 56 57 55 48 8D 35 ?? ?? ?? ?? [154] 73 EB 83 E8 03 72 17 C1 E0 08 0F B6 D2 09 D0 48 FF C6 83 F0 FF 0F 84 3A 00 00 00 48 63 E8 8D 41 01 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2d_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.91 [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 53 56 57 55 48 8D 35 ?? ?? ?? ?? [163] 73 E4 83 E8 03 72 1B C1 E0 08 0F B6 D2 09 D0 48 FF C6 83 F0 FF 0F 84 3F 00 00 00 D1 F8 48 63 E8 EB 03 41 FF D3 11 C9 41 FF D3 11 C9 75 18 FF C1 }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2e_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.91 [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { 53 56 57 55 48 8D 35 ?? ?? ?? ?? [163] 73 E4 83 E8 03 72 1D C1 E0 08 0F B6 D2 09 D0 48 FF C6 83 F0 FF 0F 84 58 00 00 00 D1 F8 48 63 E8 72 38 EB 0E 01 DB 75 08 8B 1E }
	condition:
		$1 at pe.entry_point
}

rule upx_391_nrv2b_modf_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.91 [NRV2B] modified"
		source = "Made by Retdec Team"
	strings:
		$1 = { BE ?? ?? ?? ?? 60 8D BE ?? ?? ?? ?? 57 83 CD FF EB 10 90 90 90 90 90 90 8A 06 88 07 46 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 72 ED B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 }
	condition:
		$1 at pe.entry_point
}

rule upx_39x_nrv2b_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
		start = 118
	strings:
		$1 = { FF C6 88 17 48 FF C7 8A 16 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 72 E6 8D 41 01 41 FF D3 11 C0 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 73 EB 83 E8 03 72 13 C1 E0 08 0F B6 D2 09 D0 48 FF C6 83 F0 FF 74 3A 48 63 E8 8D 41 01 41 FF D3 11 C9 41 FF D3 11 C9 75 18 89 C1 83 C0 02 41 FF D3 11 C9 01 DB 75 08 8B 1E 48 83 EE FC 11 DB 73 ED 48 81 FD 00 F3 FF FF 11 C1 E8 3E FF FF FF EB 87 5E 48 8? }
	condition:
		$1 at pe.entry_point + 118
}

rule upx_39x_nrv2d_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
		start = 118
	strings:
		$1 = { FF C6 88 17 48 FF C7 8A 16 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 72 E6 8D 41 01 EB 07 FF C8 41 FF D3 11 C0 41 FF D3 11 C0 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 73 E4 83 E8 03 72 17 C1 E0 08 0F B6 D2 09 D0 48 FF C6 83 F0 FF 74 3F D1 F8 48 63 E8 EB 03 41 FF D3 11 C9 41 FF D3 11 C9 75 18 FF C1 41 FF D3 11 C9 01 DB 75 08 8B 1E 48 83 EE FC 11 DB 73 ED 83 C1 02 48 81 FD 00 FB FF FF 83 D1 01 E8 33 FF FF FF E9 79 FF FF FF 5E 48 8? }
	condition:
		$1 at pe.entry_point + 118
}

rule upx_39x_nrv2e_x64 {
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
		start = 118
	strings:
		$1 = { FF C6 88 17 48 FF C7 8A 16 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 72 E6 8D 41 01 EB 07 FF C8 41 FF D3 11 C0 41 FF D3 11 C0 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 73 E4 83 E8 03 72 19 C1 E0 08 0F B6 D2 09 D0 48 FF C6 83 F0 FF 74 58 D1 F8 48 63 E8 72 38 EB 0E 01 DB 75 08 8B 1E 48 83 EE FC 11 DB 72 28 FF C1 01 DB 75 08 8B 1E 48 83 EE FC 11 DB 72 18 41 FF D3 11 C9 01 DB 75 08 8B 1E 48 83 EE FC 11 DB 73 ED 83 C1 02 EB 05 41 FF D3 11 C9 }
	condition:
		$1 at pe.entry_point + 118
}
