# Learnt Lessons

- Notion attachment preservation requires downloading the actual bytes and recording size/hash provenance; a page snapshot is not enough.
- Reviewer-facing refits must read the exact historical Git blobs behind the submitted figures. Mixing in the later held-out/SP500 sweep would silently change the protocol.
- A successful Notion write is not sufficient verification. Re-fetching caught malformed LaTeX caused by JavaScript escape handling; plain, unambiguous notation was safer for this callout.
- Raw token counts are not cross-model data units when tokenizers encode 26 tokens/message versus one token/event. Message semantics, repeated exposure, parameter convention, and architecture-specific compute must all be aligned before declaring a model data-limited.
- A large fixed-`beta=0.28` fit penalty supports only a finite-range difference from that LM reference. The much smaller fixed-`beta=1.0` penalty prevents treating `beta>1` as tightly established.
- IsoFLOP point-estimate stability and identifiability are distinct. A modest slope change on 5 retained slices does not rescue the loss of 4 unbracketed slices.
- Throughput profiling supplies the compute function, not a loss surface. It cannot substitute for completed, matched Transformer trajectories.
