# Hybrid model portfolio research

Audit time: `2026-08-11T16:41:37Z`

## Decision

Use three progressively larger local inference targets and keep Kimi K3 behind an explicit capacity gate:

1. **Jamba2 3B** is the first smoke target. Its checkpoint is 6,394,271,296 bytes and its 28 layers place attention every 14 layers with Mamba elsewhere. It is the smallest clean proof that the local runtime can execute an SSM-Transformer hybrid.
2. **NVIDIA Nemotron Nano 9B v2** is the second target. Its checkpoint is 17,776,492,512 bytes and its official configuration contains 56 layers with a hybrid Mamba/MLP/attention pattern and only four attention layers.
3. **Kimi Linear 48B-A3B Instruct** is the Moonshot deployment target. Its checkpoint is 98,248,224,120 bytes. The architecture uses 20 KDA layers and 7 global MLA layers (3:1) plus sparse MoE. Raw weights almost fill one 96 GB-class device, so the safe local plan uses at least two physically empty GPUs.
4. **Kimi K3** is captured and planned, not falsely labeled deployed. Its official checkpoint is 1,560,936,091,448 bytes across 96 shards. The eight GPUs found empty during the live probe provide less aggregate VRAM than the weights alone, before runtime state or activations. A real K3 launch therefore needs a later multi-node window with substantially more simultaneously empty devices and a supported distributed engine.

## Evidence table

| Target | Hybrid mechanism | Total/active parameters | Context | Pinned model revision | Weight bytes | License | Local decision |
|---|---|---:|---:|---|---:|---|---|
| AI21 Jamba2 3B | Mamba + attention | 3B / 3B | 256K | `525c6c8e1d9f5bddedfbdc1dbb0ade2df84230c9` | 6,394,271,296 | Apache-2.0 | One-GPU smoke first |
| NVIDIA Nemotron Nano 9B v2 | Mamba-2 + MLP + 4 attention layers | 9B / 9B | 128K | `6533e8de2c68e4536bf7c411d7a3ce5734111476` | 17,776,492,512 | NVIDIA Open Model License | One-GPU smoke second |
| Moonshot Kimi Linear 48B-A3B | KDA + global MLA + MoE | 48B / 3B | 1M | `e1df551a447157d4658b573f9a695d57658590e9` | 98,248,224,120 | MIT | Two-or-more-GPU smoke after small targets |
| Moonshot Kimi K3 | KDA + gated MLA + AttnRes + latent MoE | 2.8T / 104B | 1M | `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` | 1,560,936,091,448 | Kimi K3 License | Source/recipe capture now; inference deferred by capacity gate |

## Why these targets

The four systems are not interchangeable:

- Jamba2 and Nemotron-H test the SSM-Transformer branch of the design space.
- Kimi Linear tests linear recurrent attention mixed with periodic global attention.
- Kimi K3 extends the Kimi branch with AttnRes, a much larger expert pool, native vision, and MXFP4 weights. It is the frontier reference, but using its API or cloning its README would not prove local inference.

This staged order separates runtime bring-up from scale. A successful Jamba2 load does not count as a Kimi deployment, and a downloaded checkpoint does not count as an inference result.

## Primary sources

- [AI21 Jamba2 3B model card](https://huggingface.co/ai21labs/AI21-Jamba2-3B)
- [NVIDIA Nemotron Nano 9B v2 model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2)
- [Moonshot Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear)
- [Moonshot Kimi Linear model card](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)
- [Moonshot Kimi K3 repository](https://github.com/MoonshotAI/Kimi-K3)
- [Moonshot Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)

## Live GPU evidence

Allocation `5980502` was `RUNNING` for user `kangli.u6gb`. A read-only four-node `nvidia-smi` probe at `2026-08-11T16:39Z` found eight physically empty devices:

- `nid010053`: GPU 3
- `nid010371`: GPUs 0 and 1
- `nid010473`: GPUs 0 and 2
- `nid011179`: GPUs 1, 2, and 3

The remaining devices held existing Python processes using about 78.6 GB each. This is point-in-time evidence only. Every launch must repeat the process/memory probe immediately before loading a model and restrict `CUDA_VISIBLE_DEVICES` to the revalidated physical indices.
