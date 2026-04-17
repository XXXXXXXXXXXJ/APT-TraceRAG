
rule ListInstalledAntivirus : suspect {
    meta:
        category = "fingerprint"
        description = "tries to enumerate installed antivirus programs"
        author = "malcat"
        reliability = 70

    strings:
        $query1 = "Select * From AntiVirusProduct" ascii wide fullword nocase

        $fav1 = "360TotalSecurity" ascii wide fullword
        $fav2 = "ESET" ascii wide fullword
        $fav3 = "Windows Defender" ascii wide fullword
        $fav4 = "Kaspersky Lab" ascii wide fullword
        $fav5 = "Avira" ascii wide fullword
        $fav6 = "AVG Antivirus" ascii wide fullword
        $fav7 = "Panda Security" ascii wide fullword
        $fav8 = "AVAST" ascii wide fullword
        $fav9 = "Doctor Web" ascii wide fullword
        $fav10 = "Malwarebytes" ascii wide fullword
        $fav11 = "Bitdefender" ascii wide fullword
        $fav12 = "AVAST Software" ascii wide fullword
        $fav13 = "%ProgramData%\\Avg" ascii wide fullword
        $fav14 = "avgsvc" ascii wide fullword
        $fav15 = "avgui" ascii wide fullword
        $fav16 = "avgsvca" ascii wide fullword
        $fav17 = "avgidsagent" ascii wide fullword
        $fav18 = "avgsvcx" ascii wide fullword
        $fav19 = "avgwdsvcx" ascii wide fullword
        $fav20 = "avgadminclientservice" ascii wide fullword
        $fav21 = "afwserv" ascii wide fullword
        $fav22 = "avastui" ascii wide fullword
        $fav23 = "avastsvc" ascii wide fullword
        $fav24 = "aswidsagent" ascii wide fullword
        $fav25 = "aswidsagenta" ascii wide fullword
        $fav26 = "aswengsrv" ascii wide fullword
        $fav27 = "avastavwrapper" ascii wide fullword
        $fav28 = "bccavsvc" ascii wide fullword
        $fav29 = "avp" ascii wide fullword
        $fav30 = "avpui" ascii wide fullword
        $fav31 = "AvastSvc.exe" ascii wide fullword
        $fav32 = "nod32krn.exe" ascii wide fullword
        $fav33 = "savadminservice.exe" ascii wide fullword
        $fav34 = "mcvsescn.exe" ascii wide fullword

    condition:
        any of ($query*) or 4 of ($fav*)
}


rule WmiQuery : odd {
    meta:
        category = "fingerprint"
        description = "uses WMI queries"
        author = "malcat"
        reliability = 60

    strings:
        $ = "root\\CIMV2" ascii wide fullword nocase
        $ = "ROOT\\SecurityCenter2" ascii wide fullword nocase
        $ = "SELECT * FROM Win32_Processor" ascii wide nocase
        $ = "SELECT * FROM Win32_VideoController" ascii wide nocase
        $ = "Select * From AntiVirusProduct" ascii wide nocase
        $ = "SELECT * FROM Win32_NetworkAdapter" ascii wide nocase
        $ = "SELECT * FROM Win32_process" ascii wide nocase
        $ = "github.com/StackExchange/wmi" ascii fullword
        $ = {99DC56958C82CF11A37E00AA003240C7}
        $ = {E147790231D7CE11A357000000000001}
        $ = {75A6AC44FCE8D011A07C00C04FB68820}

    condition:
        any of them
}

rule FingerprintHardware : odd {
    meta:
        category = "fingerprint"
        description = "tries to enumerate installed hardware"
        author = "malcat"
        reliability = 50

    strings:
        $ = "ControlSet001\\ENUM\\PCI" ascii wide fullword nocase
        $ = "ControlSet001\\SERVICES\\MSSMBIOS\\DATA" ascii wide fullword nocase
        $ = "ControlSet001\\SERVICES\\DISK\\ENUM" ascii wide fullword nocase
        $ = "System\\CentralProcessor\\0" ascii wide fullword nocase
        $ = "SYSTEM\\ControlSet001\\Enum\\IDE" ascii wide fullword nocase
        $ = /ipconfig .{0,25}\/all/ ascii wide fullword nocase
        $ = "systeminfo" ascii wide fullword nocase
        $ = "processorid" ascii wide fullword nocase
        $ = "GetVolumeInformation" ascii wide
		$ = "WNetGetConnectionA" ascii fullword
		$ = "WNetGetConnectionW" ascii fullword
		$ = "GetLogicalDrives" ascii fullword
        $ = "ProcessorNameString" ascii wide fullword
        $ = "get_TotalPhysicalMemory" ascii fullword
        $ = "EnumDisplayMonitors" ascii wide
        $ = "github.com/denisbrodbeck/machineid.machineID" ascii fullword   // GO package
        $ = "HARDWARE\\Description\\System" ascii fullword nocase
        $ = /(SELECT|select|Select) .{1,64} (FROM|from|From) +Win32_DisplayConfiguration/ ascii wide fullword
        $ = /(SELECT|select|Select) .{1,64} (FROM|from|From) +Win32_ComputerSystem/ ascii wide fullword
        $ = /(SELECT|select|Select) .{1,64} (FROM|from|From) +Win32_Bios/ ascii wide fullword
        $ = /(SELECT|select|Select) .{1,64} (FROM|from|From) +Win32_VideoController/ ascii wide fullword
        $ = /(SELECT|select|Select) .{1,64} (FROM|from|From) +Win32_Processor/ ascii wide fullword

    condition:
        2 of them or (WmiQuery and any of them)
}


rule FingerprintSoftware : odd {
    meta:
        category = "fingerprint"
        description = "tries to enumerate installed software"
        author = "malcat"
        reliability = 30

    strings:
        $ = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall" ascii wide fullword
        $ = "DisplayName" ascii wide fullword nocase
        $ = "SELECT * FROM FirewallProduct" ascii wide nocase
        $ = "Select * from AntiVirusProduct" ascii wide nocase
        $ = "Select * from win32_process" ascii wide nocase
        $ = "github.com/digitalocean/go-smbios/smbios" ascii

    condition:
        2 of them 
}


rule FingerprintEnvironment : odd {
    meta:
        category    = "fingerprint"
        description = "tries to assess the O.S environment"
        author      = "malcat"
        created     = "2022-10-27"
        reliability = 50
        tlp         = "TLP:WHITE"
        sample      = "aa2316ff30647295efac2c884b30b6e83a8515c2d2ff1df3c5d6091b404c73a9"

	strings:
        $ = /GetComputerName(Ex)?[AW]/ ascii fullword
        $ = /GetUserName(Ex)?[AW]/ ascii fullword
        $ = "GetSystemInfo" ascii fullword
        $ = /GetVersion(Ex[AW])?/ ascii fullword

    condition:
        3 of them
}
