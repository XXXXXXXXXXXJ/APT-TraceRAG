rule PublicIP : suspect {
    meta:
        category = "network"
        description = "program tries to get its public IP address using well-known services"
        author = "malcat"
        reliability = 90

    strings:
        $ = "icanhazip.com" ascii wide fullword
        $ = "iplogger.org" ascii wide fullword
        $ = "myexternalip.com" ascii wide fullword
        $ = "ip-api.com" ascii wide fullword
        $ = "wipmania.com" ascii wide fullword
        $ = "apia.ipify.org" ascii wide fullword
        $ = "api.ipstack.com/" ascii wide fullword

    condition:
        any of them
}

rule ZoneAlternateStream : odd {
    meta:
        category = "network"
        description = "program tries to manipulate internet alternate streams"
        author = "malcat"
        reliability = 60

    strings:
        $ = "ZoneTransfer" ascii wide fullword
        $ = "Zone.Identifier" ascii wide fullword

    condition:
        any of them
}

rule TorUsage : suspect {
    meta:
        category = "network"
        description = "connects to TOR website or uses TOR library / client"
        author = "malcat"
        reliability = 60

    strings:
        $ = "tor.exe" ascii wide fullword
        $ = "defaults-torrc" ascii wide fullword
        $ = ".onion" ascii wide
        $ = "onion.to" ascii wide
        $ = "tor2web" ascii wide

    condition:
        any of them
}

rule HideInternetActivity : odd {
    meta:
        category = "network"
        description = "tries to hide recent internet activity"
        author = "malcat"
        reliability = 60

    strings:
        $ = "DeleteUrlCacheEntry" ascii fullword

    condition:
        any of them
}

rule DownloadUsingWininet : odd {
    meta:
        category = "network"
        description = "can download files from internet using wininet API"
        author = "malcat"
        reliability = 60

    strings:
        $ = "InternetReadFile" ascii fullword
        $ = "URLDownloadToFileA" ascii fullword
        $ = "URLDownloadToFileW" ascii fullword

    condition:
        any of them
}

rule DownloadUsingWinHttp : odd {
    meta:
        category = "network"
        description = "can download files from internet using Winhttp API"
        author = "malcat"
        reliability = 60

    strings:
        $ = "WinHttpReadData" ascii fullword

    condition:
        any of them
}

rule DownloadUsingPowershell : suspicious {
    meta:
        category = "network"
        description = "can download files from internet using powershell commandlets"
        author = "malcat"
        reliability = 70

    strings:
        $ = /New-Object\s+Net.Webclient/ ascii wide fullword

    condition:
        any of them
}


rule CustomUserAgent : odd {
    meta:
        category = "network"
        description = "embeds a user agent string"
        author = "malcat"
        reliability = 30

    strings:
        $ = /Mozilla\/\d+\.\d+( {0,2}\([a-zA-Z0-9 ;_\/:.-]{4,256}\))?/ ascii wide fullword


    condition:
        any of them
}

rule MultipleUserAgent : suspect {
    meta:
        category = "network"
        description = "embeds more than 2 user agent strings, sometimes used by spammers"
        author = "malcat"
        reliability = 30

    strings:
        $ua = /Mozilla\/\d+\.\d+( {0,2}\([a-zA-Z0-9 ;_\/:.-]{4,256}\))?/ ascii wide fullword

    condition:
        #ua > 2
}

rule PostHttpForm : odd {
    meta:
        category = "network"
        description = "post data using http form"
        author = "malcat"
        reliability = 70

    strings:
        $ = "multipart/form-data" ascii wide
        $ = "application/x-www-form-urlencoded" ascii wide

    condition:
        any of them
}


rule AccessNetworkShares : suspicious {
    meta:
        category = "network"
        description = "may access network shares"
        author = "malcat"
        reliability = 70

    strings:
        $ = "WNetEnumResourceA" ascii fullword
        $ = "WNetEnumResourceW" ascii fullword
        $ = "WNetOpenEnumA" ascii fullword
        $ = "WNetOpenEnumW" ascii fullword

    condition:
        2 of them
}

rule MiningProtocol : suspicious {
    meta:
        category = "network"
        description = "use cryptomining protocols/domains"
        author = "malcat"
        reliability = 90

    strings:
        $ = "stratum+tcp://" ascii wide 
        $ = "pool.minergate.com" ascii wide fullword
        $ = "stratum+ssl://" ascii wide
        $ = "api.xmrig.com" ascii wide

    condition:
        any of them
}

rule DisableFirewall : suspicious {
    meta:
        category = "network"
        description = "add an exception to the windows firewall"
        author = "malcat"
        reliability = 90

    strings:
        $ = "netsh firewall add allowedprogram" ascii wide 
        $ = "netsh firewall set opmode disable" ascii wide

    condition:
        any of them
}

rule CloudFileHosting : suspicious {
    meta:
        category = "network"
        description = "contains a typical file hosting service url"
        author = "malcat"
        reliability = 90

    strings:
        $ = "dl.dropbox.com" ascii wide fullword
        $ = "content.dropboxapi.com" ascii wide fullword
        $ = "justpaste.it" ascii wide fullword

    condition:
        any of them
}


rule NetworkTime : odd {
    meta:
        category = "network"
        description = "query network time services"
        author = "malcat"
        reliability = 70

    strings:
        $ = "utcnist.colorado.edu" ascii wide fullword
        $ = "time-a-g.nist.gov" ascii wide fullword
        $ = "time-b-g.nist.gov" ascii wide fullword
        $ = "time-d-g.nist.gov" ascii wide fullword
        $ = "time.google.com" ascii wide fullword
        $ = "time.nist.gov" ascii wide fullword
        $ = "pool.ntp.org" ascii wide fullword

    condition:
        any of them
}

rule BruteforcePassword : suspicious {
    meta:
        category = "network"
        description = "contains a list of most-used passwords"
        author = "malcat"
        reliability = 90

    strings:
        $ = "admin1" ascii wide fullword
        $ = "666666" ascii wide fullword
        $ = "888888" ascii wide fullword
        $ = "ubnt" ascii wide fullword
        $ = "quser" ascii wide fullword
        $ = "tech" ascii wide fullword
        $ = "111111" ascii wide fullword
        $ = "password1" ascii wide fullword
        $ = "D-Link" ascii wide fullword
        $ = "dlink" ascii wide fullword
        $ = "!root" ascii wide fullword
        $ = "cablecom" ascii wide fullword
        $ = "netopia" ascii wide fullword
        $ = "sysadm" ascii wide fullword
        $ = "diag" ascii wide fullword
        $ = "netgear" ascii wide fullword
        $ = "vt100" ascii wide fullword
        $ = "vyatta" ascii wide fullword
        $ = "micros" ascii wide fullword
        $ = "comcast" ascii wide fullword
        $ = "netman" ascii wide fullword
        $ = "daemon" ascii wide fullword
        $ = "demo" ascii wide fullword
        $ = "arris" ascii wide fullword
        $ = "qwerty" ascii wide fullword
        $ = "7654321" ascii wide fullword
        $ = "adsl" ascii wide fullword
        $ = "mg3500" ascii wide fullword
        $ = "bbsd-client" ascii wide fullword
        $ = "adminttd" ascii wide fullword
        $ = "PlcmSpIp" ascii wide fullword
        $ = "11111111" ascii wide fullword
        $ = "22222222" ascii wide fullword
        $ = "mountsys" ascii wide fullword
        $ = "memotec" ascii wide fullword
        $ = "museadmin" ascii wide fullword
        $ = "storwatch" ascii wide fullword
        $ = "adminpldt" ascii wide fullword
        $ = "pldtadmin" ascii wide fullword
        $ = "telecomadmin" ascii wide fullword
        $ = "xc3511" ascii wide fullword
        $ = "vizxv" ascii wide fullword
        $ = "xmhdipc" ascii wide fullword
        $ = "juantech" ascii wide fullword
        $ = "123456" ascii wide fullword
        $ = "54321" ascii wide fullword
        $ = "1234" ascii wide fullword
        $ = "12345" ascii wide fullword
        $ = "admin1234" ascii wide fullword
        $ = "1111" ascii wide fullword
        $ = "smcadmin" ascii wide fullword
        $ = "klv123" ascii wide fullword
        $ = "klv1234" ascii wide fullword
        $ = "Zte521" ascii wide fullword
        $ = "hi3518" ascii wide fullword
        $ = "jvbzd" ascii wide fullword
        $ = "anko" ascii wide fullword
        $ = "zlxx." ascii wide fullword
        $ = "changeme" ascii wide fullword
        $ = "7ujMko0vizxv" ascii wide fullword
        $ = "7ujMko0admin" ascii wide fullword
        $ = "ikwb" ascii wide fullword
        $ = "dreambox" ascii wide fullword
        $ = "realtek" ascii wide fullword
        $ = "1111111" ascii wide fullword
        $ = "admin123" ascii wide fullword
        $ = "meinsm" ascii wide fullword
        $ = "99999999" ascii wide fullword
        $ = "Meins" ascii wide fullword
        $ = "4321" ascii wide fullword
        $ = "utstar" ascii wide fullword
        $ = "zoomadsl" ascii wide fullword
        $ = "ip20" ascii wide fullword
        $ = "ip3000" ascii wide fullword
        $ = "ip400" ascii wide fullword
        $ = "bintec" ascii wide fullword
        $ = "tsunami" ascii wide fullword
        $ = "abc123" ascii wide fullword
        $ = "2601hx" ascii wide fullword
        $ = "synnet" ascii wide fullword
        $ = "bayandsl" ascii wide fullword
        $ = "1234567" ascii wide fullword
        $ = "friend" ascii wide fullword
        $ = "oelinux123" ascii wide fullword
        $ = "tini" ascii wide fullword
        $ = "router" ascii wide fullword
        $ = "anicust" ascii wide fullword
        $ = "d.e.b.u.g" ascii wide fullword
        $ = "switch" ascii wide fullword
        $ = "1234567890" ascii wide fullword
        $ = "toor" ascii wide fullword
        $ = "calvin" ascii wide fullword
        $ = "1234qwer" ascii wide fullword
        $ = "root123" ascii wide fullword
        $ = "ahetzip8" ascii wide fullword
        $ = "696969" ascii wide fullword
        $ = "pa55w0rd" ascii wide fullword
        $ = "123123" ascii wide fullword
        $ = "root500" ascii wide fullword
        $ = "alpine" ascii wide fullword
        $ = "zte9x15" ascii wide fullword
        $ = "b120root" ascii wide fullword
        $ = "atc456" ascii wide fullword
        $ = "CISCO" ascii wide fullword
        $ = "surt" ascii wide fullword
        $ = "adsl1234" ascii wide fullword
        $ = "merlin" ascii wide fullword
        $ = "netadmin" ascii wide fullword
        $ = "hewlpack" ascii wide fullword
        $ = "NetICs" ascii wide fullword
        $ = "LSiuY7pOmZG2s" ascii wide fullword
        $ = "3UJUh2VemEfUte" ascii wide fullword
        $ = "specialist" ascii wide fullword
        $ = "changeme2" ascii wide fullword
        $ = "admintelecom" ascii wide fullword
        $ = "xc3611" ascii wide fullword
        $ = "antslq" ascii wide fullword
        $ = "zyad1234" ascii wide fullword

    condition:
        15 of them
}
