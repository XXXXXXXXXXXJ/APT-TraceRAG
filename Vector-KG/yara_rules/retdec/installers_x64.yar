/*
 * YARA rules for x64 PE installer detection.
 * Copyright (c) 2017 Avast Software, licensed under the MIT license
 */

import "pe"

rule winrar_sfx_392b1_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "3.92b1"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 9B FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 48 39 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 09 8D FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_393_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "3.93"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 9B FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 48 39 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 E5 8C FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_400_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "4.00"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 97 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 10 3B 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 D1 83 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_401_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "4.01"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 9B FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 84 3B 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 99 83 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_410_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "4.10"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 9B FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 F8 3B 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 81 84 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_411_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "4.11"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 9B FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 F8 3B 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 0D 82 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_420_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "4.20"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 97 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 58 3F 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 0D 92 FF FF 48 89 47 08 8B 83 2C 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 2C 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_501_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.01"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 2F 57 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_51x_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.1x"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 93 66 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_520_530_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.20, 5.30"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 2F 65 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_521_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.21"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 33 65 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_531_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.31"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 E7 67 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_540_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.40"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 D3 04 00 00 48 83 C4 28 E9 72 FE FF FF CC CC 33 C0 48 89 41 10 48 8D 05 CB 35 01 00 48 89 41 08 48 8D 05 68 42 01 00 48 89 01 48 8B C1 C3 CC 40 53 48 83 EC 20 48 8B D9 48 8B C2 48 8D 0D 35 42 01 00 48 89 0B 48 8D 53 08 33 C9 48 89 0A 48 89 4A 08 48 8D 48 08 E8 D8 29 00 00 48 8D 05 45 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_550_x64
{
  meta:
        category = "installer"
        author = "RetDec Team"
		name = "WinRAR SFX"
		version = "5.50"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 EB 04 00 00 48 83 C4 28 E9 82 FE FF FF CC CC 33 C0 48 89 41 10 48 8D 05 63 36 01 00 48 89 41 08 48 8D 05 18 43 01 00 48 89 01 48 8B C1 C3 CC 40 53 48 83 EC 20 48 8B D9 48 8B C2 48 8D 0D E5 42 01 00 48 89 0B 48 8D 53 08 33 C9 48 89 0A 48 89 4A 08 48 8D 48 08 E8 C0 2A 00 00 48 8D 05 F5 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_392b1_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "3.92b1"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 C3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 A8 37 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 85 88 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_393_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "3.93"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 C3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 A8 37 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 61 88 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_400_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.00"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 C3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 A4 38 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 5D 81 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_401_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.01"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 C3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 18 39 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 11 81 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_410_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.10"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 C3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 8C 39 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 B5 81 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_411_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.11"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 C3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 8C 39 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 41 7F FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 24 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_420_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.20"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 A3 FE FF FF 45 33 C9 45 33 C0 33 D2 33 C9 48 83 C4 28 E9 74 3C 00 00 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 A9 8F FF FF 48 89 47 08 8B 83 2C 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC 48 83 EC 28 4C 8B 09 41 8B 81 2C 0C 00 00 39 41 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_501_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.01"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 27 58 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D 51 12 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_510_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.10"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 93 67 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D E9 51 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_511_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.11"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 9B 67 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D 7D 4F 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_520_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.20"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 37 66 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D 2D 4F 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_521_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.21"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 2B 66 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D 71 4E 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_530_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.30"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 37 66 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D 4D 4C 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_531_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.31"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 E3 68 00 00 48 83 C4 28 E9 12 FE FF FF CC CC 4C 8D 0D 75 4A 01 00 33 C0 49 8B D1 44 8D 40 08 3B 0A 74 2B FF C0 49 03 D0 83 F8 2D 72 F2 8D 41 ED 83 F8 11 77 06 B8 0D 00 00 00 C3 81 C1 44 FF FF FF B8 16 00 00 00 83 F9 0E 41 0F 46 C0 C3 48 98 41 8B 44 C1 04 C3 CC 48 83 EC 28 E8 F3 02 00 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_540_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.40"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 73 05 00 00 48 83 C4 28 E9 72 FE FF FF CC CC E9 CF 3A 00 00 CC CC CC 48 89 5C 24 10 55 48 8B EC 48 83 EC 20 83 65 E8 00 33 C9 33 C0 C7 05 F9 00 02 00 02 00 00 00 0F A2 44 8B C1 C7 05 E6 00 02 00 01 00 00 00 41 81 F0 6E 74 65 6C 44 8B CA 41 81 F1 69 6E 65 49 44 8B D2 45 0B C8 8B D3 81 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_zip_550_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.50"
    source = "Made by RetDec Team"
    extra = "with ZIP payload"
	strings:
		$1 = { 48 83 EC 28 E8 BF 05 00 00 48 83 C4 28 E9 82 FE FF FF CC CC E9 FF 36 00 00 CC CC CC 48 89 5C 24 10 48 89 7C 24 18 55 48 8B EC 48 83 EC 20 83 65 E8 00 33 C9 33 C0 C7 05 14 FE 01 00 02 00 00 00 0F A2 44 8B C1 C7 05 01 FE 01 00 01 00 00 00 81 F1 63 41 4D 44 44 8B CA 44 8B D2 41 81 F1 65 6E 74 69 41 81 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_392b1_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "392b1"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 9A 64 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 D4 C7 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 19 97 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_393_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "3.93"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 7E 64 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 D4 C7 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 F5 96 FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_400_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.00"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 2A 67 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 E4 C2 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 D9 8D FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_401_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.01"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 26 67 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 E0 C2 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 CD 8D FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_410_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.10"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 52 65 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 B8 C1 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 19 8C FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_411_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.11"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 4E 65 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 B8 C1 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 19 8C FF FF 48 89 47 08 8B 83 24 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_420_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "4.20"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 48 8D 05 FE 70 00 00 48 89 44 24 30 E8 E7 FE FF FF 48 8D 54 24 30 B9 01 00 00 00 E8 B8 C1 FF FF 48 83 C4 28 C3 CC CC CC 48 89 5C 24 08 57 48 83 EC 20 48 8B F9 48 89 11 48 8B CA 48 8B DA E8 65 8A FF FF 48 89 47 08 8B 83 2C 0C 00 00 48 8B 5C 24 30 89 47 10 48 8B C7 48 83 C4 20 5F C3 CC CC }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_501_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.01"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 27 64 00 00 48 83 C4 28 E9 56 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_51x_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.1x"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 93 65 00 00 48 83 C4 28 E9 56 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_520_530_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.20 - 5.30"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 2F 64 00 00 48 83 C4 28 E9 56 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_531_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.31"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 E7 66 00 00 48 83 C4 28 E9 56 FE FF FF CC CC 48 89 5C 24 08 48 89 6C 24 10 48 89 74 24 18 57 48 83 EC 20 49 8B E8 48 8B F2 48 8B D9 48 85 C9 75 05 E8 81 1A 00 00 48 63 43 18 8B 7B 14 48 03 46 08 75 05 E8 6F 1A 00 00 33 C9 85 FF 74 33 4C 8B 4E 08 4C 63 43 18 4B 8D 14 01 48 63 02 49 03 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_540_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.40"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 E3 04 00 00 48 83 C4 28 E9 66 FE FF FF CC CC 33 C0 48 89 41 10 48 8D 05 EF 5E 01 00 48 89 41 08 48 8D 05 4C 60 01 00 48 89 01 48 8B C1 C3 CC 40 53 48 83 EC 20 48 8B D9 48 8B C2 48 8D 0D 19 60 01 00 48 89 0B 48 8D 53 08 33 C9 48 89 0A 48 89 4A 08 48 8D 48 08 E8 7C 2A 00 00 48 8D 05 29 }
	condition:
		$1 at pe.entry_point
}

rule winrar_sfx_console_550_x64
{
	meta:
        category = "installer"
        author = "RetDec Team"
    name = "WinRAR SFX"
		version = "5.50"
		extra = "console version"
    source = "Made by RetDec Team"
	strings:
		$1 = { 48 83 EC 28 E8 A3 04 00 00 48 83 C4 28 E9 76 FE FF FF CC CC 33 C0 48 89 41 10 48 8D 05 27 5E 01 00 48 89 41 08 48 8D 05 9C 5F 01 00 48 89 01 48 8B C1 C3 CC 40 53 48 83 EC 20 48 8B D9 48 8B C2 48 8D 0D 69 5F 01 00 48 89 0B 48 8D 53 08 33 C9 48 89 0A 48 89 4A 08 48 8D 48 08 E8 0C 2B 00 00 48 8D 05 79 }
	condition:
		$1 at pe.entry_point
}
