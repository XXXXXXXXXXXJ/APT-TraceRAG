import "pe"



rule ProcessInjectionTargets : odd {
    meta:
        category = "evasion"
        description = "contains a list of process names often used as injection target in Windows"
        author = "malcat"
        reliability = 20

    strings:
        $ = "svchost.exe" ascii wide
        $ = "vbc.exe" ascii wide fullword
        $ = "explorer.exe" ascii wide fullword
        $ = "wmihost.exe" ascii wide fullword
        $ = "iexplore.exe" ascii wide fullword
        $ = "regsvr32.exe" ascii wide fullword
        $ = "rundll32.exe" ascii wide fullword
        $ = "Regasm" ascii wide fullword

    condition:
        2 of them
}



rule BlacklistAnalysisTool : suspect {
    meta:
        category = "evasion"
        description = "contains a list of common malware analysis tools"
        author = "malcat"
        reliability = 40

    strings:

        $ = "apimonitor-x64" ascii wide fullword
        $ = "apimonitor-x86" ascii wide fullword
        $ = "atrsdfw.sys" ascii wide fullword
        $ = "autopsy" ascii wide fullword
        $ = "autopsy64" ascii wide fullword
        $ = "autoruns" ascii wide fullword
        $ = "autoruns64" ascii wide fullword
        $ = "autorunsc" ascii wide fullword
        $ = "autorunsc64" ascii wide fullword
        $ = "binaryninja" ascii wide fullword
        $ = "blacklight" ascii wide fullword
        $ = "brcow_x_x_x_x.sys" ascii wide fullword
        $ = "brfilter.sys" ascii wide fullword
        $ = "crexecprev.sys" ascii wide fullword
        $ = "cutter" ascii wide fullword
        $ = "cve.sys" ascii wide fullword
        $ = "cybkerneltracker.sys" ascii wide fullword
        $ = "de4dot" ascii wide fullword
        $ = "debugview" ascii wide fullword
        $ = "dgdmk.sys" ascii wide fullword
        $ = "diskmon" ascii wide fullword
        $ = "dnsd" ascii wide fullword
        $ = "dnSpy" ascii wide fullword nocase
        $ = "dotpeek32" ascii wide fullword
        $ = "dotpeek64" ascii wide fullword
        $ = "dumpcap" ascii wide fullword
        $ = "eaw.sys" ascii wide fullword
        $ = "exeinfope" ascii wide fullword
        $ = "fakedns" ascii wide fullword
        $ = "fakenet" ascii wide fullword
        $ = "ffdec" ascii wide fullword
        $ = "fiddler" ascii wide fullword
        $ = "fileinsight" ascii wide fullword
        $ = "floss" ascii wide fullword
        $ = "groundling32.sys" ascii wide fullword
        $ = "groundling64.sys" ascii wide fullword
        $ = "hexisfsmonitor.sys" ascii wide fullword
        $ = "hiew32" ascii wide fullword
        $ = "hiew32demo" ascii wide fullword
        $ = "hollows_hunter" ascii wide fullword
        $ = "idag" ascii wide fullword
        $ = "idaq" ascii wide fullword
        $ = "idaq64" ascii wide fullword
        $ = "ildasm" ascii wide fullword
        $ = "ilspy" ascii wide fullword nocase
        $ = "jd-gui" ascii wide fullword
        $ = "ksde" ascii wide fullword
        $ = "ksdeui" ascii wide fullword
        $ = "libwamf.sys" ascii wide fullword
        $ = "lordpe" ascii wide fullword
        $ = "lragentmf.sys" ascii wide fullword
        $ = "officemalscanner" ascii wide fullword
        $ = "ollydbg" ascii wide fullword
        $ = "pdfstreamdumper" ascii wide fullword
        $ = "pe-bear" ascii wide fullword
        $ = "pe-sieve32" ascii wide fullword
        $ = "pe-sieve64" ascii wide fullword
        $ = "pebrowse64" ascii wide fullword
        $ = "peid" ascii wide fullword
        $ = "pestudio" ascii wide fullword
        $ = "peview" ascii wide fullword
        $ = "pexplorer" ascii wide fullword
        $ = "procdump" ascii wide fullword
        $ = "procdump64" ascii wide fullword
        $ = "processhacker" ascii wide fullword
        $ = "procexp" ascii wide fullword
        $ = "procexp64" ascii wide fullword
        $ = "procmon" ascii wide fullword
        $ = "prodiscoverbasic" ascii wide fullword
        $ = "psanhost" ascii wide fullword
        $ = "psepfilter.sys" ascii wide fullword
        $ = "psuamain" ascii wide fullword
        $ = "psuaservice" ascii wide fullword
        $ = "py2exedecompiler" ascii wide fullword
        $ = "r2agent" ascii wide fullword
        $ = "rabin2" ascii wide fullword
        $ = "radare2" ascii wide fullword
        $ = "ramcapture" ascii wide fullword
        $ = "ramcapture64" ascii wide fullword
        $ = "redcloak" ascii wide fullword
        $ = "reflector" ascii wide fullword
        $ = "regmon" ascii wide fullword
        $ = "resourcehacker" ascii wide fullword
        $ = "retdec" ascii wide fullword
        $ = "rundotnetdll" ascii wide fullword
        $ = "rvsavd.sys" ascii wide fullword
        $ = "safe-agent.sys" ascii wide fullword
        $ = "scdbg" ascii wide fullword
        $ = "scylla_x64" ascii wide fullword
        $ = "scylla_x86" ascii wide fullword
        $ = "sentinelmonitor.sys" ascii wide fullword
        $ = "shellcode_launcher" ascii wide fullword
        $ = "solarwindsdiagnostics" ascii wide fullword
        $ = "sysmon" ascii wide fullword
        $ = "sysmon64" ascii wide fullword
        $ = "tanium" ascii wide fullword
        $ = "taniumclient" ascii wide fullword
        $ = "taniumdetectengine" ascii wide fullword
        $ = "taniumendpointindex" ascii wide fullword
        $ = "taniumtracecli" ascii wide fullword
        $ = "taniumtracewebsocketclient64" ascii wide fullword
        $ = "tcpdump" ascii wide fullword
        $ = "tcpvcon" ascii wide fullword
        $ = "tcpview" ascii wide fullword nocase
        $ = "win32_remote" ascii wide fullword
        $ = "win64_remotex64" ascii wide fullword
        $ = "windbg" ascii wide fullword
        $ = "windump" ascii wide fullword
        $ = "winhex" ascii wide fullword
        $ = "winhex64" ascii wide fullword
        $ = "winobj" ascii wide fullword
        $ = "wireshark" ascii wide fullword
        $ = "x32dbg" ascii wide fullword
        $ = "x64dbg" ascii wide fullword
        $ = "xwforensics" ascii wide fullword
        $ = "xwforensics64" ascii wide fullword
    condition:
        2 of them
}

rule BlacklistSandbox : suspect {
    meta:
        category = "evasion"
        description = "contains a list of common sandbox programs"
        author = "malcat"
        reliability = 60

    strings:
        $ = "SbieDll.dll" ascii wide fullword
        $ = "snxhk.dll" ascii wide fullword
        $ = "vmware" ascii wide fullword nocase
        $ = "VBOX" ascii wide fullword
        $ = "Sandboxie" ascii wide fullword
        $ = "VirtualBox" ascii wide fullword
        $ = "vboxservice" ascii wide fullword
        $ = "sbiesvc" ascii wide fullword
        $ = "vboxhook.dll" ascii wide fullword
        $ = "Xen" ascii wide fullword
		$ = "wine_get_unix_file_name" ascii wide fullword
		$ = "vboxhook" ascii wide fullword
		$ = "wpespy" ascii wide fullword nocase
		$ = "vmcheck" ascii wide fullword nocase

    condition:
        2 of them
}

rule DelayBatch : suspect {
    meta:
        category = "evasion"
        description = "uses batch commands to delay execution"
        author = "malcat"
        reliability = 20

    strings:
        $ = /ping .{0,32}(-w|-n) \d{1,8}/ ascii wide fullword
        $ = /timeout \/t \d+/ ascii wide fullword
        $ = "cmd.exe /k ping 0" ascii wide fullword
    condition:
        any of them
}

rule SelfDeleteBatch : suspect {
    meta:
        category = "evasion"
        description = "uses batch commands to delete itself"
        author = "malcat"
        reliability = 30

    strings:
        $echooff = "@echo off" ascii wide fullword
		$s1 = "del /F" ascii wide fullword

    condition:
        $echooff and 1 of ($s*)
}


rule DisableAntivirus : suspect {
    meta:
        category = "evasion"
        description = "disables installed antivirus product"
        author = "malcat"
        reliability = 60

    strings:
        $ = "DisableAntiSpyware" ascii wide fullword
        $ = "net stop WinDefend" ascii wide fullword

    condition:
        any of them
}
