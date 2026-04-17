rule ElevateUsingCOM : odd {
    meta:
        category = "lateral movement"
        description = "elevate privileges using COM moniker"
        author = "malcat"
        details = "https://docs.microsoft.com/en-us/windows/win32/com/the-com-elevation-moniker"
        reliability = 70

    strings:
        $ = "Elevation:Administrator!new" ascii wide fullword
        $ = "{F885120E-3789-4FD9-865E-DC9B4A6412D2}" ascii wide fullword
        $ = "{FCC74B77-EC3E-4DD8-A80B-008A702075A9}" ascii wide fullword

    condition:
        any of them
}

rule ElevatePrivileges : odd {
    meta:
        category = "lateral movement"
        description = "elevate privileges using Windows API"
        author = "malcat"
        reliability = 70

    strings:
        $ = "AdjustTokenPrivileges" ascii fullword
        $ = "SeDebugPrivilege" ascii wide fullword

    condition:
        any of them
}

rule RunShell : odd {
    meta:
        category = "lateral movement"
        description = "starts a shell"
        author = "malcat"
        reliability = 70

    strings:
        $ = "cmd.exe" ascii wide fullword
		$ = "/c start" ascii wide fullword
        $ = "powershell.exe" ascii wide fullword

    condition:
        any of them
}


rule Wscript : suspicious {
    meta:
        category = "lateral movement"
        description = "runs a wscript script (vbs, js, ..)"
        author = "malcat"
        reliability = 30

    strings:
        $ = "wscript.exe" ascii wide fullword
        $ = "WScript.Shell" ascii wide fullword
        $ = "On Error Resume Next" ascii wide fullword
        $ = "CreateObject("
        $ = "Scripting.FileSystemObject" ascii wide fullword
        $ = "536372697074696E672E46696C6553797374656D4F626A656374" ascii wide fullword  // Scripting.FileSystemObject hexencoded
        $ = "winmgmts:{impersonationLevel=impersonate}" ascii wide

    condition:
        any of them
}

rule Powershell : suspicious {
    meta:
        category = "lateral movement"
        description = "runs a powershell script"
        author = "malcat"
        reliability = 30

    strings:
        $ = "powershell.exe" ascii wide fullword
        $ = /-ExecutionPolicy\s+Bypass/ ascii wide fullword

    condition:
        any of them
}

rule WmiProcessCreate : suspicious {
    meta:
        category = "lateral movement"
        description = "starts a process using WMI, sometimes used to alter process filiation"
        author = "malcat"
        reliability = 90

    strings:
        $ = "WMIC Process Call Create" ascii wide fullword nocase

    condition:
        any of them
}

rule CreateMediaAutorunFile : suspicious {
    meta:
        category = "lateral movement"
        description = "may write a autorun.inf file"
        author = "malcat"
        reliability = 30

    strings:
        $ = {5B6175746F72756E5D0A6F70656E} // [autorun]\nopen=
        $ = {5B006100750074006F00720075006E005D000D000A006F00700065006E003D00} // [autorun]\nopen= wide

        $ = "autorun.inf" ascii wide fullword

    condition:
        any of them
}

rule CreateService : suspicious {
    meta:
        category = "lateral movement"
        description = "creates a service"
        author = "malcat"
        reliability = 70

    strings:
        $ = "sc create" ascii wide fullword nocase
        $ = "CreateServiceA" ascii wide fullword
        $ = "CreateServiceW" ascii wide fullword

    condition:
        any of them
}




rule ReflectiveLoader : suspicious {
    meta:
        category = "evasion"
        description = "Export typical of reflective loader-loaded dlls"
        author = "malcat"
        reliability = 80
        sample = "bb26724c27361a5881ebf646166423b9668fd4089cf50e4e493641d471d30fa9"

    condition:
        pe.exports("ReflectiveLoader") or pe.exports("_ReflectiveLoader@4")
}
