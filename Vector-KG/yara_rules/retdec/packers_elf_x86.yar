/*
 * YARA rules for x86 ELF packer detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "elf"

rule elfcrypt_10_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "ELFCrypt"
		version = "1.0"
		source = "from Detect It Easy signatures"
	strings:
		$1 = { EB 02 06 C6 60 9C BE }
	condition:
		$1 at elf.entry_point
}

rule upx_3xx_01_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.xx"
		source = "from Detect It Easy signatures"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 ?? ?? ?? ?? 60 }
	condition:
		$1 at elf.entry_point
}

rule upx_3xx_02_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.xx"
	strings:
		$1 = { E8 ?? ?? ?? ?? 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB }
	condition:
		$1 at elf.entry_point
}

rule upx_3xx_03_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.xx"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 ?? ?? ?? ?? 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_lzma_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [LZMA]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 ?? 0B 00 00 60 8B 74 24 24 8B 7C 24 2C 83 CD FF 89 E5 8B 55 28 AC 4A 88 C1 24 07 C0 E9 03 BB 00 FD FF FF D3 E3 8D A4 5C 90 F1 FF FF 83 E4 E0 6A 00 6A 00 89 E3 53 83 C3 04 8B 4D 30 FF 31 57 53 83 C3 04 88 43 02 AC 4A 88 C1 24 0F 88 03 C0 E9 04 88 4B 01 52 }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2b_01_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 EE 00 00 00 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 8B 1E 83 EE FC 11 DB 73 E4 31 C9 }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2b_02_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2B]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 EF 75 09 }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2d_01_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 02 01 00 00 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 19 8B 1E 83 EE FC 11 DB 72 10 48 01 DB }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2d_02_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2D]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 19 }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2e_01_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? EB 0E 5A 58 59 97 60 8A 54 24 20 E9 12 01 00 00 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 28 8B 1E 83 EE FC 11 DB 72 1F 48 01 DB }
	condition:
		$1 at elf.entry_point
}

rule upx_39x_nrv2e_02_elf
{
	meta:
        category = "packer"
        author = "RetDec Team"
		name = "UPX"
		version = "3.9x [NRV2E]"
		source = "Made by Retdec Team"
	strings:
		$1 = { E8 ?? ?? ?? ?? 60 8B 74 24 24 8B 7C 24 2C 83 CD FF EB 0F 90 90 90 90 90 8A 06 46 88 07 47 01 DB 75 07 8B 1E 83 EE FC 11 DB 8A 07 72 EB B8 01 00 00 00 01 DB 75 07 8B 1E 83 EE FC 11 DB 11 C0 01 DB 73 0B 75 28 }
	condition:
		$1 at elf.entry_point
}
