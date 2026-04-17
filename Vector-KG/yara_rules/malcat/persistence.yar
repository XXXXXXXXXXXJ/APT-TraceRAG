import "pe"


rule CreateScheduledTask : odd {
    meta:
        category = "persistence"
        description = "can create a scheduled task"
        author = "malcat"
        reliability = 60

    strings:
        $ = "schtasks /create" ascii wide fullword
        $ = { 4D623D4C6BFDA349B9B709CB3CD3F047 }
        $ = { C7A4AB2FA94D1340969720CC3FD40F85 }

    condition:
        any of them
}

rule CreateRegistryEntryUsingBatch : odd {
    meta:
        category = "persistence"
        description = "create a registry entry using batch commands (reg.exe ..). Often used by malware"
        author = "malcat"
        reliability = 30

    strings:
        $ = /reg(\.exe)? .{0,32}add/ ascii wide fullword
        $ = "/t REG_SZ" ascii wide fullword

    condition:
        any of them
}


rule AutorunKey : odd {
    meta:
        category = "persistence"
        description = "file contains path of an autorun key"
        author = "malcat"
        reliability = 20

    strings:
        $ = "Software\\Microsoft\\Windows\\CurrentVersion\\Run" ascii wide fullword
        $ = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run" ascii wide fullword
        $ = "Software\\Microsoft\\Windows\\CurrentVersion\\RunOnce" ascii wide fullword
        $ = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnce" ascii wide fullword
        $ = "Software\\Microsoft\\Windows\\CurrentVersion\\RunOnceEx" ascii wide fullword
        $ = "SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\RunOnceEx" ascii wide fullword
        $ = "Software\\Microsoft\\Windows NT\\CurrentVersion\\Winlogon" ascii wide fullword
        $ = "Software\\Microsoft\\Windows\\CurrentVersion\\Policies\\Explorer\\Run" ascii wide fullword
        $ = "Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\Load" ascii wide fullword
        $ = "CurrentControlSet\\Control\\Session Manager\\BootExecute" ascii wide fullword
        $ = "Software\\Microsoft\\Windows NT\\CurrentVersion\\Windows\\AppInit_DLLs" ascii wide fullword

    condition:
        any of them
}
