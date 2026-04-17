
rule %RULE_NAME% %RULE_TAG%{
    meta:
        name        = "%RULE_NAME%"
        category    = "%RULE_CATEGORY%"
        description = "%RULE_DESCRIPTION%"
        author      = "%RULE_AUTHOR%"
        created     = "%DATE%"
        reliability = %RULE_RELIABILITY%
        tlp         = "TLP:%RULE_TLP%"
        sample      = "%SHA256%"

    condition:
        %RULE_CONDITION%
}
