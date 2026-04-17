
rule PowershellEncodedCommand : suspect {
    meta:
        category = "obfuscation"
        description = "a powershell b64 encoded command line"
        author = "malcat"
        reliability = 70

    strings:
        $ = /powershell .{0,64}(hidden|encod) .{0,64}[a-zA-Z0-9=+\/]{32,}/ nocase ascii wide
      

    condition:
        any of them
}