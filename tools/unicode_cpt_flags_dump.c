// Dumps unicode_cpt_flags(cp) for every codepoint 0..0x10FFFF, one hex value per line, for an
// exhaustive cross-check against an independently-computed Python dump (D-tok Phase 2 oracle).
#include "unicode_cpt_flags.h"
#include <stdio.h>

int main(void) {
    for (uint32_t cp = 0; cp <= 0x10FFFF; cp++) {
        printf("%04x\n", unicode_cpt_flags(cp));
    }
    return 0;
}
