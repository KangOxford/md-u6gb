# sgpur web refresh integration

Focused regression tests for the encrypted ledger's browser-to-Cloudflare-to-`sgpur -w` refresh path.

Run from `/projects/public/u6gb`:

```bash
python3 -m unittest -v tasks/sgpur_web_refresh_20260813/tests/test_web_refresh.py
```

The private Cloudflare token lives only in `.config/sgpur/inbox.json` with mode `0600`; it must never be committed or printed by these tests.
