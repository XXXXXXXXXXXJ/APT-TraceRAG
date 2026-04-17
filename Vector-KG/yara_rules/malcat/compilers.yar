import "pe"

rule FASM {
    meta:
        category = "compiler"
        description = "detects fasm using DOS stub"
        author = "malcat"
        reliability = 70

    strings:
        $stub = { 4D5A80000100000004001000FFFF00004001000000000000400000000000000000000000000000000000000000000000000000000000000000000000800000000E1FBA0E00B409CD21B8014CCD21546869732070726F6772616D2063616E6E6F742062652072756E20696E20444F53206D6F64652E0D0A24000000000000000050450000 }

    condition:
        $stub at 0
}

///////////////////////// OTHER LINKERS

rule TurboLinker {
    meta:
        name        = "TurboLinker"
        category    = "compiler"
        description = "Linked with TurboLinker"
        author      = "malcat"
        created     = "2023-05-12"
        reliability = 80
        tlp         = "TLP:WHITE"
        sample      = "2a64b59654dbf1a12b93a4892e1312807d2be8b396a0adc73ecbd51af2689be2"
    strings:
        $dosstub = { 546869732070726F6772616D206D7573742062652072756E20756E6465722057696E????0D0A24??00 } private
        
    condition:
        $dosstub in (0..1000)
}


///////////////////////// OTHER C COMPILERS

rule BorlandCpp {
    meta:
        category = "compiler"
        description = "detects Borland C/Cpp compiler"
        author = "malcat"
        reliability = 40

    strings:
        $ = {EB1066623A432B2B484F4F4B}
        $ = "borlndmm" ascii fullword

    condition:
        pe.section_index(".tls") >= 0 and all of them and pe.characteristics & 0xC == 0xC
}

rule Golang {
    meta:
        category = "compiler"
        description = "detects golang compiler"
        author = "malcat"
        reliability = 60

    strings:
        $ = "runtime.gopanic" ascii fullword
        $ = "runtime.rt0_go" ascii fullword
        $ = "runtime.sysmontick" ascii fullword
        $ = { (FB | FA | F0) FF FF FF 00 00 ?? (04 | 08) }
        $ = "runtime.GOMAXPROCS" ascii fullword
        $ = "runtime.GOROOT" ascii fullword

    condition:
         4 of them
}

rule MinGW {
    meta:
        category = "compiler"
        description = "detects mingw compiler"
        author = "malcat"
        reliability = 60

    strings:
        $ = "Mingw runtime failure" ascii
        $ = "Mingw-w64 runtime failure" ascii
        $ = "../../gcc/gcc/config/i386/w32-shared-ptr.c" ascii

    condition:
        pe.section_index(".bss") >= 0 and
        any of them and 
        (
            for all i in (0..pe.number_of_sections - 1): (
                pe.sections[i].characteristics & 0x700000 != 0
            ) or
            pe.characteristics & 0xC == 0xC
        )
}

rule PellesC {
    meta:
        category = "compiler"
        description = "detects pellesC compiler"
        author = "DIE (Jason Hood <jadoxa@yahoo.com.au>)"
        reliability = 60

    strings:
        $ep = { 5589E56AFF68????????68????????64FF35000000006489250000000083EC0C }

    condition:
        $ep at pe.entry_point and
        pe.linker_version.major == 2 and pe.linker_version.minor == 50
}

///////////////////////// MSVC 6.0

rule MSVC_6_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 6 and pe.linker_version.minor == 0
}

rule MSVC_6_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        pe.rich_signature.version(0x2636) or
        pe.rich_signature.version(0x2306) or
        pe.rich_signature.version(0x2354) or
        pe.rich_signature.version(0x1fe8)
}

///////////////////////// MSVC 2002

rule MSVC_2002_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 7 and pe.linker_version.minor < 10
}

rule MSVC_2002_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        pe.rich_signature.version(0x24fa) or
        pe.rich_signature.toolid(0x3d) and (pe.rich_signature.toolid(0x1c) or pe.rich_signature.toolid(0x1d))
}

///////////////////////// MSVC 2003

rule MSVC_2003_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 7 and pe.linker_version.minor >= 10
}

rule MSVC_2003_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x0c05) or 
            pe.rich_signature.version(0x0fc3) or 
            pe.rich_signature.version(0x0883) or 
            pe.rich_signature.version(0x178e)
        ) and 
        (pe.rich_signature.toolid(0x5f) or pe.rich_signature.toolid(0x60))
}

///////////////////////// MSVC 2005

rule MSVC_2005_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    strings:
        $dosstub = "This program cannot" ascii private

    condition:
        pe.linker_version.major == 8 and pe.linker_version.minor == 0 and $dosstub in (0..1000)
}

rule MSVC_2005_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        pe.rich_signature.version(0xc627) and 
        (pe.rich_signature.toolid(0x6d) or pe.rich_signature.toolid(0x6e))
}

///////////////////////// MSVC 2008

rule MSVC_2008_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 9
}

rule MSVC_2008_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x521e) or 
            pe.rich_signature.version(0x7809)
        ) and
        pe.rich_signature.toolid(0x91) and (pe.rich_signature.toolid(0x83) or pe.rich_signature.toolid(0x84) or pe.rich_signature.toolid(0x89))
}

///////////////////////// MSVC 2010

rule MSVC_2010_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 10 and pe.linker_version.minor == 0
}

rule MSVC_2010_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        pe.rich_signature.version(0x766f) or 
        pe.rich_signature.version(0x9d1b) or
        pe.rich_signature.toolid(0x9d) and (pe.rich_signature.toolid(0xaa) or pe.rich_signature.toolid(0xab))
}

///////////////////////// MSVC 2012

rule MSVC_2012_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 11 and pe.linker_version.minor == 0
}

rule MSVC_2012_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0xc627) or 
            pe.rich_signature.version(0xeb9b) or 
            pe.rich_signature.version(0xc7a2) or 
            pe.rich_signature.version(0xecc2) or 
            pe.rich_signature.version(0xee66)
        ) and // and because conflict with msvc 2005
            pe.rich_signature.toolid(0xcc) and (pe.rich_signature.toolid(0xce) or pe.rich_signature.toolid(0xcf))
}

///////////////////////// MSVC 2013

rule MSVC_2013_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 12 and pe.linker_version.minor == 0
}

rule MSVC_2013_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x520d) or 
            pe.rich_signature.version(0x9eb5) or 
            pe.rich_signature.version(0x7725) or 
            pe.rich_signature.version(0x797d) or 
            pe.rich_signature.version(0x7803) or 
            pe.rich_signature.version(0x9eb5)
        ) and
        pe.rich_signature.toolid(0xde) and (pe.rich_signature.toolid(0xe0) or pe.rich_signature.toolid(0xe1))
}

///////////////////////// MSVC 2015

rule MSVC_2015_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 14 and pe.linker_version.minor == 0
}

rule MSVC_2015_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x59f2) or 
            pe.rich_signature.version(0x5bd2) or 
            pe.rich_signature.version(0x5d6e) or 
            pe.rich_signature.version(0x5e3b) or 
            pe.rich_signature.version(0x5e95) or 
            pe.rich_signature.version(0x5e97) or  
            pe.rich_signature.version(0x5ead) or
            pe.rich_signature.version(0x5eb5)
        ) and
        pe.rich_signature.toolid(0x102) and (pe.rich_signature.toolid(0x104) or pe.rich_signature.toolid(0x105) or pe.rich_signature.toolid(0x108))
}

///////////////////////// MSVC 2017

rule MSVC_2017_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 14 and pe.linker_version.minor > 0 and pe.linker_version.minor <= 16
}

rule MSVC_2017_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x6093) or 
            pe.rich_signature.version(0x61b9) or 
            pe.rich_signature.version(0x62d9) or 
            pe.rich_signature.version(0x646f) or 
            pe.rich_signature.version(0x64ea) or 
            pe.rich_signature.version(0x6610) or 
            pe.rich_signature.version(0x6665) or 
            pe.rich_signature.version(0x6852) or 
            pe.rich_signature.version(0x685b) or 
            pe.rich_signature.version(0x699b) or 
            pe.rich_signature.version(0x6997)
        ) and
        pe.rich_signature.toolid(0x102) and (pe.rich_signature.toolid(0x104) or pe.rich_signature.toolid(0x105) or pe.rich_signature.toolid(0x108))
}

///////////////////////// MSVC 2019

rule MSVC_2019_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x6b74) or 
            pe.rich_signature.version(0x6dc9) or 
            pe.rich_signature.version(0x6e9b) or
            pe.rich_signature.version(0x6f0b) or 
            pe.rich_signature.version(0x7086) or 
            pe.rich_signature.version(0x71b7) or 
            pe.rich_signature.version(0x74d9) or 
            pe.rich_signature.version(0x7556) or 
            pe.rich_signature.version(0x75c2) or 
            pe.rich_signature.version(0x75b5) or 
            pe.rich_signature.version(0x75b9) or 
            pe.rich_signature.version(0x75ba) or 
            pe.rich_signature.version(0x75bb) or 
            pe.rich_signature.version(0x75bc) or 
            pe.rich_signature.version(0x75bd) or 
            pe.rich_signature.version(0x75be) or 
            pe.rich_signature.version(0x75bf)
        ) and
        pe.rich_signature.toolid(0x102) and (pe.rich_signature.toolid(0x104) or pe.rich_signature.toolid(0x105) or pe.rich_signature.toolid(0x108))
}

rule MSVC_2019_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 14 and pe.linker_version.minor > 20 and pe.linker_version.minor < 30
}

///////////////////////// MSVC 2022

rule MSVC_2022_rich {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on rich header information"
        author = "malcat"
        reliability = 80

    condition:
        (
            pe.rich_signature.version(0x7b8d) or 
            pe.rich_signature.version(0x7a61) or 
            pe.rich_signature.version(0x7a64) or
            pe.rich_signature.version(0x77a1) or
            pe.rich_signature.version(0x77f2) or
            pe.rich_signature.version(0x7a60)
        ) and
        pe.rich_signature.toolid(0x102) and (pe.rich_signature.toolid(0x104) or pe.rich_signature.toolid(0x105) or pe.rich_signature.toolid(0x108))
}

rule MSVC_2022_linker {
    meta:
        category = "compiler"
        description = "detects used visual studio version based on linker information"
        author = "malcat"
        reliability = 60

    condition:
        pe.linker_version.major == 14 and pe.linker_version.minor >= 30 
}

