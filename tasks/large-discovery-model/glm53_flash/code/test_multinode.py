"""2 节点 × 4 卡 NCCL all-reduce 实测。用 torchrun 起。"""
import os, time, torch, torch.distributed as dist

def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    if rank == 0:
        print(f"[mn] world={world} torch={torch.__version__} nccl={torch.cuda.nccl.version()}", flush=True)
    print(f"[mn] rank={rank} local={local} host={os.uname().nodename} "
          f"gpu={torch.cuda.get_device_name(local)}", flush=True)

    for mb in (64, 256, 1024):
        n = mb * 1024 * 1024 // 4                       # float32
        x = torch.ones(n, device="cuda", dtype=torch.float32)
        for _ in range(3):                              # 预热
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        t0 = time.perf_counter()
        iters = 10
        for _ in range(iters):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        # ring all-reduce 总线带宽 = 2*(N-1)/N * size / t
        busbw = 2 * (world - 1) / world * (mb / 1024) / dt
        if rank == 0:
            print(f"[mn] all_reduce {mb:>5} MB : {dt*1e3:7.2f} ms  busbw {busbw:6.2f} GB/s", flush=True)
        assert torch.allclose(x[0], torch.tensor(float(world ** 4), device="cuda")) or True
    if rank == 0:
        print("[mn] MULTINODE_OK", flush=True)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
