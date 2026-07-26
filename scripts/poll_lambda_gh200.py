#!/usr/bin/env python3
"""Poll Lambda Cloud until a GH200 is available, then launch it.

Uses the Lambda Cloud API (https://cloud.lambdalabs.com/api/v1):
  GET  /instance-types              — capacity check
  POST /instance-operations/launch  — reserve / launch
  GET  /ssh-keys                    — discover default SSH key
  GET  /instances                   — confirm launch

Auth: HTTP basic with API key as username and empty password
      (same as ``curl -u "$LAMBDA_API_KEY:"``).

Reads ``LAMBDA_API_KEY`` from the environment or a local ``.env`` file.

Examples:
  # Poll every 30s and launch when available (uses first SSH key on account)
  python scripts/poll_lambda_gh200.py

  # Dry-run: only report availability
  python scripts/poll_lambda_gh200.py --dry-run

  # Faster poll, prefer a region, custom name
  python scripts/poll_lambda_gh200.py --interval 15 --region us-east-1 \\
      --ssh-key lambda --name mathllm-gh200
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

API_BASE = "https://cloud.lambdalabs.com/api/v1"
DEFAULT_INSTANCE_TYPE = "gpu_1x_gh200"
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file into os.environ (no overwrite)."""
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


class LambdaClient:
    def __init__(self, api_key: str, timeout: float = 30.0) -> None:
        token = base64.b64encode(f"{api_key}:".encode()).decode()
        self._headers = {
            "Authorization": f"Basic {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "mathllm-poll-lambda-gh200/1.0",
        }
        self._timeout = timeout

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{API_BASE}{path}"
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=self._headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            try:
                parsed = json.loads(detail)
                detail = json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                pass
            raise RuntimeError(f"HTTP {exc.code} {method} {path}:\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error {method} {path}: {exc.reason}") from exc

    def instance_types(self) -> dict[str, Any]:
        return self.request("GET", "/instance-types").get("data", {})

    def ssh_keys(self) -> list[dict[str, Any]]:
        return self.request("GET", "/ssh-keys").get("data", [])

    def instances(self) -> list[dict[str, Any]]:
        return self.request("GET", "/instances").get("data", [])

    def launch(
        self,
        *,
        region_name: str,
        instance_type_name: str,
        ssh_key_names: list[str],
        quantity: int = 1,
        name: str | None = None,
        file_system_names: list[str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "region_name": region_name,
            "instance_type_name": instance_type_name,
            "ssh_key_names": ssh_key_names,
            "quantity": quantity,
        }
        if name:
            body["name"] = name
        if file_system_names:
            body["file_system_names"] = file_system_names
        return self.request("POST", "/instance-operations/launch", body)


def regions_with_capacity(types: dict[str, Any], instance_type: str) -> list[str]:
    info = types.get(instance_type)
    if not info:
        known = ", ".join(sorted(types)) or "(none)"
        raise KeyError(f"Unknown instance type {instance_type!r}. Known: {known}")
    regions = info.get("regions_with_capacity_available") or []
    names: list[str] = []
    for region in regions:
        if isinstance(region, dict):
            name = region.get("name")
            if name:
                names.append(name)
        elif isinstance(region, str):
            names.append(region)
    return names


def pick_region(available: list[str], preferred: str | None) -> str:
    if preferred:
        if preferred not in available:
            raise RuntimeError(
                f"Preferred region {preferred!r} not in available capacity: {available}"
            )
        return preferred
    return available[0]


def resolve_ssh_key(client: LambdaClient, ssh_key: str | None) -> str:
    keys = client.ssh_keys()
    if not keys:
        raise RuntimeError(
            "No SSH keys on this Lambda account. Add one at "
            "https://cloud.lambda.ai/ssh-keys before launching."
        )
    names = [k["name"] for k in keys if k.get("name")]
    if ssh_key:
        if ssh_key not in names:
            raise RuntimeError(f"SSH key {ssh_key!r} not found. Available: {names}")
        return ssh_key
    if len(names) == 1:
        return names[0]
    raise RuntimeError(
        f"Multiple SSH keys found ({names}); pass --ssh-key to choose one."
    )


def ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Poll Lambda Cloud for GH200 capacity and launch when available.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=f"Lambda instance type name (default: {DEFAULT_INSTANCE_TYPE})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Seconds between polls (default: 30)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Only launch in this region when it has capacity (default: first available)",
    )
    parser.add_argument(
        "--ssh-key",
        default=None,
        help="SSH key name on the Lambda account (default: sole key, or required if multiple)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional display name for the launched instance",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Number of instances to launch (default: 1)",
    )
    parser.add_argument(
        "--file-system",
        action="append",
        default=[],
        help="Optional filesystem name to attach (repeatable; API currently allows one)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Stop after N failed launch races / polls with capacity (0 = unlimited)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only poll and print availability; never launch",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Check once and exit (launch if available unless --dry-run)",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=REPO_ROOT / ".env",
        help="Path to .env containing LAMBDA_API_KEY (default: repo .env)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.interval < 1:
        print("error: --interval must be >= 1", file=sys.stderr)
        return 2

    load_dotenv(args.env_file)
    api_key = os.environ.get("LAMBDA_API_KEY", "").strip()
    if not api_key:
        print(
            f"error: LAMBDA_API_KEY not set (env or {args.env_file})",
            file=sys.stderr,
        )
        return 2

    client = LambdaClient(api_key)
    ssh_key = resolve_ssh_key(client, args.ssh_key)

    print(f"[{ts()}] Lambda GH200 poller")
    print(f"  instance_type = {args.instance_type}")
    print(f"  ssh_key       = {ssh_key}")
    print(f"  region        = {args.region or '(any available)'}")
    print(f"  interval      = {args.interval}s")
    print(f"  dry_run       = {args.dry_run}")
    print()

    attempt = 0
    while True:
        try:
            types = client.instance_types()
            available = regions_with_capacity(types, args.instance_type)
        except Exception as exc:  # noqa: BLE001 — keep polling on transient API errors
            print(f"[{ts()}] poll error: {exc}", file=sys.stderr)
            if args.once:
                return 1
            time.sleep(args.interval)
            continue

        if not available:
            print(f"[{ts()}] no capacity for {args.instance_type}")
            if args.once:
                return 1
            time.sleep(args.interval)
            continue

        print(f"[{ts()}] CAPACITY: {args.instance_type} in {available}")

        if args.dry_run:
            if args.once:
                return 0
            time.sleep(args.interval)
            continue

        try:
            region = pick_region(available, args.region)
        except RuntimeError as exc:
            print(f"[{ts()}] {exc}")
            if args.once:
                return 1
            time.sleep(args.interval)
            continue

        print(f"[{ts()}] launching {args.quantity}x {args.instance_type} in {region} ...")
        try:
            result = client.launch(
                region_name=region,
                instance_type_name=args.instance_type,
                ssh_key_names=[ssh_key],
                quantity=args.quantity,
                name=args.name,
                file_system_names=args.file_system or None,
            )
        except RuntimeError as exc:
            attempt += 1
            print(f"[{ts()}] launch failed (likely raced): {exc}", file=sys.stderr)
            if args.once:
                return 1
            if args.max_attempts and attempt >= args.max_attempts:
                print(f"[{ts()}] giving up after {attempt} launch attempts", file=sys.stderr)
                return 1
            time.sleep(min(args.interval, 5.0))
            continue

        instance_ids = (result.get("data") or {}).get("instance_ids") or []
        print(f"[{ts()}] LAUNCHED instance_ids={instance_ids}")
        print(json.dumps(result, indent=2))

        try:
            running = client.instances()
            if running:
                print(f"[{ts()}] current instances:")
                print(json.dumps(running, indent=2))
        except RuntimeError as exc:
            print(f"[{ts()}] warning: could not list instances: {exc}", file=sys.stderr)

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
