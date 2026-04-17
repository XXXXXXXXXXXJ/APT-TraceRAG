import "pe"


rule PasswordStealer : suspect {
    meta:
        category = "stealer"
        description = "program is likely to steal passwords from local databases/files/registry"
        author = "malcat"
        reliability = 60

    strings:
        $ = "UseMasterPassword" ascii wide fullword
        $ = "files\\passwords.txt" ascii wide fullword
        $ = "wcx_ftp.ini" ascii wide fullword
        $ = "SELECT action_url, username_value, password_value FROM logins" ascii wide fullword
        $ = "WinSCP 2\\Sessions" ascii wide fullword
        $ = "sitemanager.xml" ascii wide fullword
        $ = "Authy Desktop\\Local Storage\\*.localstorage" ascii wide fullword
        $ = "Outlook\\Profiles\\Outlook\\9375CFF0413111d3B88A00104B2A6676" ascii wide fullword
        $ = "VaultGetItem" ascii wide fullword
        $ = "QIP.Online\\accounts" ascii wide fullword
        $ = "UltraVNC\\UltraVNC.ini" ascii wide fullword
        $ = "ClassicFTP\\FTPAccounts" ascii wide fullword
        $ = "GoFTP\\settings\\Connections.txt" ascii wide fullword
        $ = "Software\\VanDyke\\SecureFX" ascii wide fullword
        $ = "oZone3D\\MyFTP\\myftp.ini" ascii wide fullword
        $ = "GHISLER\\wcx_ftp.ini" ascii wide fullword
        $ = "FTPShell\\ftpshell.fsi" ascii wide fullword
        $ = "NetDrive2\\drives.dat" ascii wide fullword
        $ = "{9BDD5314-20A6-4d98-AB30-8325A95771EE}\\data" wide
        $ = "Data\\AccCfg\\Accounts.tdat" ascii wide fullword
        $ = "Microsoft\\Credentials" ascii wide fullword
        $ = "BitKinex\\bitkinex.ds" ascii wide fullword
        $ = "Software\\SimonTatham\\PuTTY\\Sessions" ascii wide fullword
        $ = "QupZilla\\profiles\\default\\browsedata.db" ascii wide fullword
        $ = "8pecxstudios\\Cyberfox86" ascii wide fullword
        $ = "FlashPeak\\BlazeFtp\\Settings" ascii wide fullword
        $ = "IncrediMail\\Identities" ascii wide fullword
        $ = "Bitvise\\BvSshClient" ascii wide fullword
        $ = "Cyberfox\\profiles.ini" ascii wide fullword
        $ = "LsaICryptUnprotectData" ascii fullword
        $ = "form_password_control" ascii wide fullword
		$ = "KeePass" ascii wide fullword
		$ = "KeeFarce" ascii wide fullword
    condition:
        2 of them
}

rule BrowserStealer : suspect {
    meta:
        category = "stealer"
        description = "program is likely to steal browser information like cookies, passwords or sessions"
        author = "malcat"
        reliability = 60

    strings:
        // ??
        $ = "logins.json" ascii wide
        $ = "Account.CFN" ascii wide
        $ = "files\\cookie_list.txt" ascii wide
        $ = "encrypted_value from cookies" ascii wide
        $ = "files\\Autofill" ascii wide
        $ = "QtWeb Internet Browser\\AutoComplete" ascii wide
        $ = "YandexBrowser\\User Data" ascii wide
        $ = "Brave-Browser\\User Data" ascii wide
        // FF
        $ = "key4.db" ascii wide fullword
        $ = "places.sqlite" ascii wide
        $ = "cookies.sqlite" ascii wide
        $ = "signons.sqlite" ascii wide
        $ = "from moz_cookies" ascii wide
        $ = "FROM moz_logins" ascii wide
        $ = "from moz_logins" ascii wide
        $ = "encryptedPassword" ascii wide
        $ = "PK11SDR_Decrypt" ascii wide fullword
        // chrome
        $ = "Chrome\\User Data" ascii wide
        $ = "Chromium\\User Data" ascii wide
        $ = "Local Storage\\leveldb" ascii wide
        // IE
        $ = "Internet Explorer\\IntelliForms\\Storage2" ascii wide
        $ = "Microsoft\\Windows\\Cookies\\Low" ascii wide
        $ = "MicrosoftEdge\\Cookies" ascii wide
        $ = "\\Microsoft\\Edge\\User Data" ascii wide
        $ = "Microsoft.MicrosoftEdge_8wekyb3d8bbwe" ascii wide
        $ = "Cookies\\IE_Cookies.txt" ascii wide
        // Opera
        $ = "Opera Stable\\Login Data" ascii wide
        $ = "Vivaldi\\User Data" ascii wide
        $ = "Opera Software\\Opera Stable" ascii wide 

        $ = "shortcuts-custom.json"  ascii wide 
        $ = "\\History\\%s_%s.txt" ascii wide 
        $ = "\\Autofill\\%s_%s.txt" ascii wide 
        

    condition:
        2 of them
}

rule CryptoWalletStealer : suspect {
    meta:
        category = "stealer"
        description = "program is likely to steal cryptocurrency wallets"
        author = "malcat"
        reliability = 60

    strings:
        $ = "Electrum-LTC" ascii wide fullword
        $ = "Electrum" ascii wide fullword
        $ = "Exodus" ascii wide fullword
        $ = "exodus.conf.json" ascii wide fullword
        $ = "ElectronCash" ascii wide fullword
        $ = "MultiDoge" ascii wide fullword
        $ = "*allet*.dat" ascii wide fullword
        $ = "wallet.dat" ascii wide fullword
        $ = "Ethereum\\keystore" ascii wide fullword
        $ = "*wallet*.txt" ascii wide fullword
        $ = "*coin*.txt" ascii wide fullword
        $ = "*monero*.txt" ascii wide fullword
        $ = "*bittrex*.txt" ascii wide fullword
        $ = "*blockchain*.txt" ascii wide fullword
        $ = "*bitcoin*.txt" ascii wide fullword
        $ = "*bitmex*.txt" ascii wide fullword
        $ = ".wallet" ascii wide
        $ = "Feathercoin" ascii wide
        $ = "Coinomi" ascii wide fullword
        $ = "Zcash" ascii wide fullword
        $ = "LitecoinCore" ascii wide fullword
        $ = "MoneroCore" ascii wide fullword
        $ = "DashcoinCore" ascii wide fullword
        $ = "DogecoinCore" ascii wide fullword
        $ = "GeroWallet" ascii wide fullword
        $ = "Opera Crypto Stable" ascii wide fullword
        $ = "Bitwarden" ascii wide fullword
        $ = "Petra Wallet" ascii wide fullword

    condition:
        4 of them
}

rule KeyloggerApi : suspect {
    meta:
        category = "stealer"
        description = "program includes typical keylogger API under Windows"
        author = "malcat"
        reliability = 60

    strings:
        $ = "WH_KEYBOARD_LL" ascii fullword
        $ = "SetWindowsHookEx" ascii fullword
        $ = "MapVirtualKey" ascii fullword
        $ = "GetKeyboardState" ascii fullword
        $ = "GetKeyState" ascii fullword
        $ = "GetForegroundWindow" ascii fullword
        $ = "GetWindowText" ascii fullword

    condition:
        3 of them
}


rule SpecialKeyNames : suspect {
    meta:
        category = "stealer"
        description = "program includes special key names like [HOME] or [ENTER], strings which can often be found in keyloggers"
        author = "malcat"
        reliability = 40

    strings:
        $ = "[INSERT]" ascii wide fullword nocase
        $ = "[END]" ascii wide fullword nocase
        $ = "[PRINT]" ascii wide fullword nocase
        $ = "[DEL]" ascii wide fullword nocase
        $ = "[INSERT]" ascii wide fullword nocase
        $ = "[BACK]" ascii wide fullword nocase
        $ = "[LEFT]" ascii wide fullword nocase
        $ = "[UP]" ascii wide fullword nocase
        $ = "[RIGHT]" ascii wide fullword nocase
        $ = "[DOWN]" ascii wide fullword nocase
        $ = "[ALT]" ascii wide fullword nocase
        $ = "[ESC]" ascii wide fullword nocase
        $ = "[BACKSPACE]" ascii wide fullword nocase
        $ = "[ENTER]" ascii wide fullword nocase
        $ = "[TAB]" ascii wide fullword nocase
        $ = "[HOME]" ascii wide fullword nocase
        $ = "[PAGEUP]" ascii wide fullword nocase
        $ = "[PAGEDOWN]" ascii wide fullword nocase
        $ = "[SPACE]" ascii wide fullword nocase
        $ = "[WIN]" ascii wide fullword nocase
        $ = "[CTRL]" ascii wide fullword nocase
        $ = "[Shift]" ascii wide fullword nocase

    condition:
        5 of them
}

rule ATMStealer : suspect {
    meta:
        category = "stealer"
        description = "program which most likely targets ATM machines"
        author = "malcat"
        reliability = 60

    strings:
        $ = "WFSExecute" ascii fullword
        $ = "WFSStartup" ascii fullword
        $ = "WFSLock" ascii fullword
        $ = "msxfs.dll" ascii fullword

    condition:
        2 of them
}

rule LsaStealer : suspect {
    meta:
        category = "stealer"
        description = "program which most likely steals credentials from LSA"
        author = "malcat"
        reliability = 90

    strings:
        $ = "sekurlsa" ascii wide fullword
        $ = "lsasrv" ascii wide fullword
        $ = "LSACREDKEY" ascii wide fullword
        $ = "lsadump" ascii wide fullword
        $ = "LsaQuerySecret" ascii fullword
        $ = "LsaRetrievePrivateData" ascii fullword
        $ = "LsaOpenPolicy" ascii fullword

    condition:
        2 of them
}


rule DiscordStealer : suspicious {
    meta:
        category = "stealer"
        description = "program attempts to steal discord artifacts"
        author = "malcat"
        reliability = 70

    strings:
        $ = "discordcanary" ascii wide fullword
        $ = "discordptb" ascii wide fullword

    condition:
        any of them
}
