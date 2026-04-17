You can add your own user-defined yara rules by adding yara files (*.yar) in this directory. Malcat uses internally the yara library version 3.8 and will be udpated as needed. 

You can use every yara features (e.g. modules) as you see fit. There are only four (optional) syntaxic point to take into consideration to makes the best use of your yara rules inside Malcat. 
Each yara rule should feature the following elements:
- one of the yara rule tags should specify which level of detection it is (i.e which color it will be displayed in). Supported tags are malware/adware/apt/rat/ransom/banker(red), tool/pua/hacktool(orange), suspect/suspicious/heuristic(yellow) and odd/rare/weird/unusual/misc(grey). Any other tag (or the absence of tag) will be given the color blue
- a "description" metadata entry which describes what the rule detects
- a "category" metadata: matching yara rules are grouped by category in Malcat 
- a reliability metadata: a number between 0 and 100. It should represent your level of confidence in this rule, 100 meaning it has no false positive/false negative whatsoever. In malcat the reliability is represented by the green gauge right to the rule name. If not specified, a default reliability of 50 is assumed. Why this field ? Because most of the publicly available yara rules are trash and/or false positive magnets.

import "pe"
rule WinrarSelfExtractor : tool {
    meta:
        category = "sfx"
        description = "WINRAR self extractor"
        reliability = 80
    strings:
        $c1 = "RarHtmlClassName" wide
        $c2 = "GETPASSWORD1" wide
        $c3 = "RarSFX" wide
        $rar = { 526172211A07 }
    condition:
        pe.overlay.size > 64 and $rar at pe.overlay.offset and all of ($c*)
}

Note that every *.yar file in this directory will be imported separately. So if your yara rules are split into several files and include each other, the correct way to store your rules is:
- put a single *.yar file in this directory that will include all your rules (in the correct order)
- create a subdirectory (e.g. myrules/) that contains all the rules to be included. Subdirectories are not scanned by malcat, so these .yar won't be imported twice.

