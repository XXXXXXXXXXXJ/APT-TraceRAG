/*
 * YARA rules for x64 ELF packer detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "elf"

rule upx_39x_lzma_x64_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? 55 53 51 52 48 01 FE 56 41 80 F8 0E 0F 85 6C 0A 00 00 55 48 89 E5 44 8B 09 49 89 D0 48 89 F2 48 8D 77 02 56 8A 07 FF CA 88 C1 }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2b_x64_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { FC 41 5B 41 80 F8 02 74 0D E9 ?? ?? ?? ?? 48 FF C6 88 17 48 FF C7 8A 16 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 72 E6 8D 41 01 41 FF D3 }
	condition:
		$1 at elf.entry_point + 112
}

rule upx_39x_nrv2d_x64_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { FC 41 5B 41 80 F8 05 74 0D E9 ?? ?? ?? ?? 48 FF C6 88 17 48 FF C7 8A 16 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 72 E6 8D 41 01 EB 07 FF C8 41 FF D3 }
	condition:
		$1 at elf.entry_point + 112
}

rule upx_39x_nrv2e_x64_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { FC 41 5B 41 80 F8 08 74 0D E9 ?? ?? ?? ?? 48 FF C6 88 17 48 FF C7 8A 16 01 DB 75 0A 8B 1E 48 83 EE FC 11 DB 8A 16 72 E6 8D 41 01 EB 07 FF C8 41 FF D3 }
	condition:
		$1 at elf.entry_point + 112
}

rule upx_394_lzma_x64_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.94 [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 11 0B 00 00 60 8B 74 24 24 8B 7C 24 2C 83 CD FF 89 E5 8B 55 28 AC 4A 88 C1 24 07 C0 E9 03 BB 00 FD FF FF D3 E3 8D A4 5C 90 F1 FF FF 83 E4 E0 6A 00 6A }
	condition:
		$1 at elf.entry_point
}

rule upx_394_lzma_2_x64_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? 55 53 51 52 48 01 FE 56 41 80 F8 0E 0F 85 65 0A 00 00 55 48 89 E5 44 8B 09 49 89 D0 48 89 F2 48 8D 77 02 56 8A 07 FF CA 88 C1 24 07 C0 E9 03 48 C7 C3 00 FD FF FF 48 D3 E3 88 C1 48 8D 9C 5C 88 }
	condition:
		$1 at elf.entry_point
}
