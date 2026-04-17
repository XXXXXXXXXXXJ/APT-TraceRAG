
rule LowerInternetSecurity : suspect {
    meta:
        category = "tampering"
        description = "may lower Windows internet security"
        author = "malcat"
        reliability = 60

    strings:
        $ = "WarnOnZoneCrossing" ascii wide fullword

    condition:
        any of them
}


rule ChangeBrowserPreference : suspect {
    meta:
        category = "tampering"
        description = "may change browser preference, often used by adware"
        author = "malcat"
        reliability = 40

    strings:
        $ = "user_pref(" ascii wide fullword
        $ = "Automatic Proxy Configuration" ascii wide fullword
        $ = "network.proxy.autoconfig_url" ascii wide fullword
        $ = "OperaPrefs.ini" ascii wide fullword
        $ = "Microsoft\\Windows\\CurrentVersion\\Internet Settings" ascii wide fullword

    condition:
        any of them
}
