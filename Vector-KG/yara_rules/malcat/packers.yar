import "pe"
import "math"

rule VMProtect_sections {
    meta:
        category = "packer"
        description = "Detect VMProtect based on section names (very unreliable)"
        reliability = 10
        author = "malcat"

    condition:
        pe.section_index(".vmp") >= 0 or pe.section_index(".vmp0") >= 0 or pe.section_index(".vmp1") >= 0
}

rule PECompact2 {
    meta:
        category = "packer"
        description = "Detect PECompact based on section artifacts"
        reliability = 60
        author = "malcat"

    condition:
        pe.sections[0].pointer_to_relocations == 0x32434550
}

rule UPX {
    meta:
        category = "packer"
        description = "Detect UPX based on section artifacts and EP"
        reliability = 40
        author = "malcat"

    condition:
        pe.section_index("UPX0") == 0 and pe.sections[0].raw_data_size == 0 and 
        pe.section_index("UPX1") == 1 and pe.sections[1].raw_data_size != 0 and 
        math.entropy(pe.sections[1].raw_data_offset, pe.sections[1].raw_data_size) > 7 and
        (uint8(pe.entry_point) == 0x60 or uint8(pe.entry_point) == 0x53)
}


rule Aspack_sections {
    meta:
        category = "packer"
        description = "Detect Aspack based on section artifacts"
        reliability = 60
        author = "malcat"

    condition:
        pe.section_index(".aspack") >= 0 and pe.section_index(".adata") >= 0
}

rule Themida {
    meta:
        category = "packer"
        description = "Detect Themida"
        reliability = 60

    condition:
        pe.section_index(pe.entry_point) >= 3 and 
        pe.sections[pe.section_index(pe.entry_point)].raw_data_offset == pe.entry_point and
        pe.section_index(pe.rva_to_offset(pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_RESOURCE].virtual_address)) == 1 and
        pe.section_index(pe.rva_to_offset(pe.data_directories[pe.IMAGE_DIRECTORY_ENTRY_IMPORT].virtual_address)) == 2 and
        pe.sections[0].characteristics & pe.SECTION_MEM_EXECUTE and
        pe.sections[0].characteristics & pe.SECTION_MEM_WRITE and
        math.entropy(pe.sections[0].raw_data_offset, pe.sections[0].raw_data_size) > 7
}

rule SmartAssembly : odd {
    meta:
        category = "packer"
        description = "Detect SmartAssembly"
        reliability = 60

    strings:
        $ = "SmartAssembly.Attributes" ascii fullword

    condition:
        dotnet.number_of_streams > 0 and any of them
}

rule EzrinProtector : odd {
    meta:
        category = "packer"
        description = "Ezrin .NET Protector"
        author = "DIE"
        reliability = 90
    
    strings:
        $ = "<PrivateImplementationDetails>{842D7503-493E-4AEE-9AF4-6E02B7840C65}" ascii

    condition:
        dotnet.number_of_streams > 0 and all of them
}


rule Dotfuscator : odd {
    meta:
        category = "packer"
        description = "Dotfuscator"
        author = "DIE"
        reliability = 50

    strings:
        $ = "DotfuscatorAttribute" ascii

    condition:
        dotnet.number_of_streams > 0 and all of them
}


rule DNGuard : odd {
    meta:
        category = "packer"
        description = "DNGuard .NET packer"
        author = "DIE"
        reliability = 80

    strings:
        $ = "ZYXDNGuarder" ascii fullword
        $ = "HVMRuntm.dll" ascii fullword

    condition:
        dotnet.number_of_streams > 0 and any of them
}


rule BabelNet : odd {
    meta:
        category = "packer"
        description = "Babel .NET packer"
        author = "DIE"
        reliability = 80

    strings:
        $ = "BabelAttribute" ascii fullword
        $ = "BabelObfuscatorAttribute" ascii fullword

    condition:
        dotnet.number_of_streams > 0 and any of them
}

rule YanoNet : odd {
    meta:
        category = "packer"
        description = "Yano .NET packer"
        author = "DIE"
        reliability = 80

    strings:
        $ = "YanoAttribute" ascii fullword

    condition:
        dotnet.number_of_streams > 0 and any of them
}

rule RyanProtector : odd {
    meta:
        category = "packer"
        description = "Ryan Borland Protector .NET packer"
        author = "malcat"
        reliability = 80

    strings:
        $ = "ProtectedBy_RyanBorland" ascii
        $ = "@Ryan-_-Borland+_+Was_HERE" ascii
        $ = "@Ryan-_-Borland+_+Protector_v" ascii

    condition:
        dotnet.number_of_streams > 0 and any of them
}


rule Confuser : odd {
    meta:
        category = "packer"
        description = "Confuser packer"
        author = "malcat"
        reliability = 80

    strings:
        $ = "ConfusedByAttribute" fullword
        $ = "Confuser.Core" fullword

    condition:
        dotnet.number_of_streams > 0 and 2 of them
}

rule EzirzDotnetReactor : odd {
    meta:
        category = "packer"
        description = "Eziriz .NET reactor"
        author = "malcat"
        reliability = 90

    strings:
        $private = { 003C50726976617465496D706C656D656E746174696F6E44657461696C733E7B????????????????????????????????????????????????????????????????????????7D00 }

    condition:
        dotnet.number_of_streams > 0 and $private
}