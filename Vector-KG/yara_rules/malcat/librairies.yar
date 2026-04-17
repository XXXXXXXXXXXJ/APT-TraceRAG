

rule Sqlite {
    meta:
        category = "library"
        description = "embeds sqlite library, sqlite is often used by password stealers"
        author = "malcat"
        reliability = 80

    strings:
        $ = "sqlite_version" ascii fullword
        $ = "sqlite_attach" ascii fullword
        $ = "sqlite3_get_table" ascii fullword
        $ = "sqlite3_exec" ascii fullword
        $ = "sqlite3_open" ascii fullword
        $ = "sqlite3_close" ascii fullword
        $ = "SELECT *" ascii fullword nocase
        $ = "naturaleftouterightfullinnercross" ascii fullword
        $ = "CREATE TEMP TABLE sqlite_temp_master(" ascii fullword
        $ = "there is already a table named %s" ascii fullword
		$ = "sqlite3_get_table() called with two or more incompatible queries" ascii fullword

    condition:
        4 of them
}


rule Boost {
    meta:
        category = "library"
        description = "embeds boost c++ library"
        author = "malcat"
        reliability = 70

    strings:
        $ = "class boost::exception_ptr" ascii fullword
        $ = "boost::lock_error" ascii fullword
        $ = "boost shared_lock owns already the mutex" ascii fullword
        $ = "@boost@@" ascii

    condition:
        2 of them
}


rule Zlib {
    meta:
        category    = "library"
        description = "Uses zlib algortihm"
        author      = "malcat"
        created     = "2022-10-26"
        reliability = 80
        tlp         = "TLP:WHITE"
        sample      = "ed3a685ca65de70b79faf95bbd94c343e73a150e83184f67e0bdb35b11d05791"

    strings:
        $ = { 03000400050006000700080009000A000B000D000F001100130017001B001F0023002B0033003B0043005300630073008300A300C300E3000201 }
        $ = { 01000200030004000500070009000D001100190021003100410061008100C1000101810101020103010401060108010C011001180120013001400160 }
        $ = /inflate .{0,32} Copyright/

    condition:
        2 of them
}


rule Libcurl {
    meta:
        category    = "library"
        description = "Linked against libcurl"
        author      = "malcat"
        created     = "2022-10-26"
        reliability = 80
        tlp         = "TLP:WHITE"
        sample      = "ed3a685ca65de70b79faf95bbd94c343e73a150e83184f67e0bdb35b11d05791"

    strings:
        $ = "CLIENT libcurl" ascii fullword

    condition:
        any of them
}


rule CryptoPP {
    meta:
        category    = "library"
        description = "link against CryptoPP"
        author      = "malcat"
        created     = "2022-10-26"
        reliability = 80
        tlp         = "TLP:WHITE"
        sample      = "ed3a685ca65de70b79faf95bbd94c343e73a150e83184f67e0bdb35b11d05791"

	strings:
		$ = "Cryptographic algorithms are disabled after a power-up self test failed" ascii fullword
		$ = "this object requires an IV" ascii fullword
		$ = "BER decode error" ascii fullword
		$ = ".?AVException@CryptoPP@@" ascii
		$ = "FileStore: error reading file" ascii fullword
		$ = "StreamTransformationFilter: PKCS_PADDING cannot be used with" ascii fullword
    condition:
		2 of them
}


rule PolarSsl {
    meta:
        category    = "library"
        description = "links against polarssl library"
        author      = "malcat"
        created     = "2022-10-27"
        reliability = 80
        tlp         = "TLP:WHITE"
        sample      = "232b0a8546035d9017fadf68398826edb0a1e055566bc1d356d6c9fdf1d7e485"

	strings:
		$ = "PolarSSLTest" ascii fullword
		$ = "mbedtls_cipher_setup" ascii fullword
		$ = "mbedtls_pk_verify" ascii fullword
		$ = "mbedtls_ssl_write_record" ascii fullword
		$ = "mbedtls_ssl_fetch_input" ascii fullword
    condition:
        3 of them
}


rule OpenSSL {
    meta:
        name        = "OpenSSL"
        category    = "library"
        description = "links aginst OpenSSL library"
        author      = "malcat"
        created     = "2023-03-10"
        reliability = 85
        tlp         = "TLP:WHITE"
        sample      = "776b274e88d253394b58136f330fc470100a94023099624c87c61b91013d2ed8"

    strings:
        $ = "EVP_OpenInit" ascii fullword
        $ = "Montgomery Multiplication for x86, CRYPTOGAMS by <appro@openssl.org>"  ascii fullword
        $ = "openssl.cnf"  ascii fullword
        $ = ".\\crypto\\ui\\ui_openssl.c" ascii fullword
        $ = "DSA-SHA1-old" ascii fullword
        $ = "Eric Young's PKCS#1 RSA" ascii fullword

    condition:
        3 of them
}


rule BoringSSL {
    meta:
        name        = "BoringSSL"
        category    = "library"
        description = "embeds BoringSSL library"
        author      = "malcat"
        created     = "2023-03-14"
        reliability = 95
        tlp         = "TLP:WHITE"
        sample      = "13e773e1c2e77fe1a14e74d7fb2ca089478a6a624d55461a21354843244fd6b8"

    strings:
        /*
         * C744242C5C627569         | mov dword ptr [esp+0x2C], 0x6975625C
         * C74424306C645C73         | mov dword ptr [esp+0x30], 0x735C646C
         * C74424346C617665         | mov dword ptr [esp+0x34], 0x6576616C
         * C74424385C77696E         | mov dword ptr [esp+0x38], 0x6E69775C
         * C744243C5C627569         | mov dword ptr [esp+0x3C], 0x6975625C
         * C74424406C645C73         | mov dword ptr [esp+0x40], 0x735C646C
         * C744244472635C74         | mov dword ptr [esp+0x44], 0x745C6372
         * C744244868697264         | mov dword ptr [esp+0x48], 0x64726968
         * C744244C5F706172         | mov dword ptr [esp+0x4C], 0x7261705F
         * C744245074795C62         | mov dword ptr [esp+0x50], 0x625C7974
         * C74424546F72696E         | mov dword ptr [esp+0x54], 0x6E69726F
         * C74424586773736C         | mov dword ptr [esp+0x58], 0x6C737367
         * C744245C5C737263         | mov dword ptr [esp+0x5C], 0x6372735C
         * C74424605C73736C         | mov dword ptr [esp+0x60], 0x6C73735C
         * C74424645C73736C         | mov dword ptr [esp+0x64], 0x6C73735C
         * C74424685F6C6962         | mov dword ptr [esp+0x68], 0x62696C5F
        */
        $ = { C744242C5C627569 C74424306C645C73 C74424346C617665 C74424385C77696E C744243C5C627569 C74424406C645C73 C744244472635C74 C744244868697264 C744244C5F706172 C744245074795C62 C74424546F72696E C74424586773736C C744245C5C737263 C74424605C73736C C74424645C73736C C74424685F6C6962 }

    condition:
        any of them
}
