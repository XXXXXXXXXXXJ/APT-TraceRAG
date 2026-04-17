rule DeletesVssShadowCopy : suspect {
    meta:
        category = "destruction"
        description = "attempts to remove vss shadow copies, a classical ransomware move"
        author = "malcat"
        reliability = 80

    strings:
        $ = /vssadmin(\.exe)? .{0,64}[Dd]elete .{0,64}[sS]hadows/ ascii wide fullword
        $ = "wmic.exe SHADOWCOPY" ascii wide fullword
		$ = "Win32_ShadowCopy" ascii wide fullword

    condition:
        any of them
}

rule ValuableFileExtensions : suspect {
    meta:
        category = "destruction"
        description = "embeds a list of file extensions often targeted by ransomwares"
        author = "malcat"
        reliability = 20

    strings:
        $ = "csv" ascii wide fullword
        $ = "ps1" ascii wide fullword
        $ = "backup" ascii wide fullword
        $ = "pdf" ascii wide fullword
        $ = "ods" ascii wide fullword
        $ = "odf" ascii wide fullword
        $ = "odp" ascii wide fullword
        $ = "ppt" ascii wide fullword
        $ = "pptx" ascii wide fullword
        $ = "doc" ascii wide fullword
        $ = "docm" ascii wide fullword
        $ = "docx" ascii wide fullword
        $ = "xls" ascii wide fullword
        $ = "xlsm" ascii wide fullword
        $ = "xlsx" ascii wide fullword
        $ = "jpg" ascii wide fullword
        $ = "jpeg" ascii wide fullword
        $ = "png" ascii wide fullword
        $ = "mp3" ascii wide fullword
        $ = "mp4" ascii wide fullword
        $ = "avi" ascii wide fullword
        $ = "mov" ascii wide fullword
        $ = "rtf" ascii wide fullword
        $ = "7z" ascii wide fullword
        $ = "tar" ascii wide fullword
        $ = "zip" ascii wide fullword
        $ = "rar" ascii wide fullword
        $ = "sql" ascii wide fullword
        $ = "sqlite" ascii wide fullword
        $ = "sqlite3" ascii wide fullword
        $ = "sqlitedb" ascii wide fullword
        $ = "pem" ascii wide fullword
        $ = "crt" ascii wide fullword
        $ = "der" ascii wide fullword
        $ = "3ds" ascii wide fullword
        $ = "vmdk" ascii wide fullword
        $ = "tgz" ascii wide fullword
        $ = "keychain" ascii wide fullword
        $ = "sdf" ascii wide fullword
        $ = "myd" ascii wide fullword
        $ = "backupdb" ascii wide fullword
        $ = "vbm" ascii wide fullword
        $ = "db3" ascii wide fullword
        $ = "cad" ascii wide fullword
        $ = "avhdx" ascii wide fullword
        $ = "001" ascii wide fullword

    condition:
        15 of them
}


rule RansomNote : suspicious {
    meta:
        category = "destruction"
        description = "match standard ransomware notifications"
        author = "malcat"
        reliability = 60

    strings:
        $e1 = "onionmail.org" ascii wide fullword
        $e2 = "protonmail.com" ascii wide fullword
        $e3 = "keemail.me" ascii wide fullword
        $e4 = ".onion" ascii wide
		$e5 = ".wallet" ascii wide

        $n1 = "sell them on the darknet" ascii wide fullword
        $n2 = "get all data back" ascii wide fullword
        $n3 = "have been encrypted" ascii wide fullword
        $n4 = "what happened to your files" ascii wide fullword nocase
        $n5 = "files are encrypted" ascii wide fullword
        $n6 = "decrypt program" ascii wide fullword
        $n7 = "return your files" ascii wide fullword
        $n8 = "strongest military algorithm" ascii wide 
        $n9 = "pay for decryption" ascii wide fullword
        $n10 = "do not rename" ascii wide fullword nocase
        $n11 = "recover them" ascii wide fullword nocase
        $n12 = "to recover your files" ascii wide fullword nocase
        $n13 = "was encrypted" ascii wide fullword nocase
        $n14 = "follow instructions" ascii wide fullword nocase
        $n15 = "How To Decrypt" ascii wide fullword nocase
        $n16 = "ALL YOUR FILES ARE" ascii wide fullword nocase
        $n17 = "ransomware" ascii wide fullword nocase
		$n18 = "your_files_are_encrypted" ascii wide fullword nocase
        $n19 = "HOW TO RECOVER" ascii wide fullword nocase


    condition:
        any of ($e*) and any of ($n*) or 2 of ($n*)
}


rule DeletesBackups : suspect {
    meta:
        category = "destruction"
        description = "destroys Windows backups, a strategy used by some ransomware"
        author = "malcat"
        reliability = 90

    strings:
        $ = "bcdedit.exe /set {default} recoveryenabled No" ascii wide fullword nocase
        $ = "wbadmin DELETE SYSTEMSTATEBACKUP" ascii wide fullword nocase

    condition:
        any of them
}



