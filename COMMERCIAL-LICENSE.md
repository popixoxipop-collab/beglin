# Licensing

vdsp-engine is dual-licensed:

1. **Open source: GNU Affero General Public License v3.0 (AGPL-3.0-or-later)**
   See [`LICENSE`](./LICENSE) for the full text. Under this license you may
   use, modify, and redistribute vdsp-engine freely, including in a network
   service, **provided that**:
   - Any modified version you run as a network service must offer its
     complete corresponding source code to users of that service
     (AGPL-3.0 §13).
   - Any distributed derivative work must also be licensed under
     AGPL-3.0-or-later.

2. **Commercial license**
   If AGPL-3.0's copyleft/source-disclosure obligations do not work for
   your use case — for example, you want to embed vdsp-engine in a
   closed-source product, ship it inside a proprietary SaaS backend
   without disclosing your modifications, or otherwise use it without
   triggering AGPL's terms — a separate commercial license is available
   for a fee.

   Contact **popixoxipop@gmail.com** to discuss commercial licensing
   terms and pricing. The actual contract text used to license
   vdsp-engine commercially is [`COMMERCIAL-LICENSE-AGREEMENT.md`](./COMMERCIAL-LICENSE-AGREEMENT.md)
   (adapted from a freely reusable public template — not yet filled
   in with a specific licensee's deal terms).

## Why dual licensing

This engine's SME2-accelerated MoE serving path has been benchmarked
against pure-scalar CPU inference on the same hardware and shown a real,
measured throughput advantage (2.38× at B=16, with higher accuracy than
the naive int8-activation SME2 kernel and no re-verification overhead) —
see [`RESULTS.md`](./RESULTS.md) for the full methodology and numbers.
Dual licensing lets individuals, researchers, and AGPL-compliant
open-source projects use the engine freely, while companies that want to
use it commercially without AGPL's obligations fund continued development
through a paid license.

## Not legal advice

This file describes the licensing *offer*; it is not itself a commercial
license agreement. Actual commercial terms (scope, price, support,
warranty, liability) are set out in a separate contract negotiated per
licensee. If you are evaluating vdsp-engine for use inside a company,
please also have your own legal/compliance team review the AGPL-3.0 terms
in `LICENSE` before choosing between the open-source and commercial paths.
