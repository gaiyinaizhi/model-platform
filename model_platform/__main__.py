"""python -m model_platform init|api|worker (internal serve is supervisor-only)."""
import argparse
import os
import secrets
import signal
import sys

from .common import Runtime, encode, text_value


def main():
    parser = argparse.ArgumentParser(description="Standalone local CPU model platform")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--tenant", action="append", required=True)
    api = sub.add_parser("api")
    api.add_argument("--port", type=int, default=8080)
    worker = sub.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    serving = sub.add_parser("serve", help=argparse.SUPPRESS)
    serving.add_argument("--deployment", required=True)
    serving.add_argument("--fd", type=int, required=True)
    serving.add_argument("--parent", type=int, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    runtime = Runtime()
    if args.command == "init":
        tenants = [text_value(name, "tenant") for name in args.tenant]
        if len(set(tenants)) != len(tenants):
            parser.error("Tenant names must be unique")
        credentials = {name: secrets.token_hex(32) for name in tenants}
        path = runtime.home / "credentials.json"
        # Never replace an existing identity or secret.
        with path.open("xb") as stream:
            stream.write(encode(credentials))
        from .store import Store
        Store(runtime)
        print(f"initialized; credentials are local at {path}; not printed")
    elif args.command == "worker":
        from .worker import run_worker
        run_worker(runtime, args.once)
    elif args.command == "api":
        if not 0 <= args.port <= 65535:
            parser.error("Invalid port")
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        from .api import run_api
        try:
            run_api(runtime, args.port)
        except KeyboardInterrupt:
            pass
    elif args.command == "serve":
        from .serving import serve
        try:
            serve(runtime, args.deployment, args.fd, args.parent)
        except Exception:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
