#!/usr/bin/env python3
"""
wifi_setup.py — TANGLAWAN WiFi Static IP Manager
Run on the Jetson via USB SSH to add/update WiFi networks.
Usage:
    python3 wifi_setup.py add    SSID PASSWORD
    python3 wifi_setup.py list
    python3 wifi_setup.py status
"""

import subprocess
import sys
import re
import time

STATIC_HOST = 224       # Always use .224 on any subnet
STATIC_DNS  = "8.8.8.8"

def run(cmd, check=True):
    # FIXED: capture_output and text replaced with Python 3.6 equivalents
    result = subprocess.run(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )
    if check and result.returncode != 0:
        print(f"ERROR: {result.stderr.strip()}")
    return result.stdout.strip(), result.returncode

def get_gateway_and_subnet():
    """Wait up to 15s for wlan0 to get a DHCP address, return (gateway, subnet_base)."""
    for _ in range(15):
        out, _ = run("ip route show dev wlan0", check=False)
        match = re.search(r"default via (\d+\.\d+\.\d+)\.\d+", out)
        if match:
            return match.group(1)
        time.sleep(1)
    return None

def add_network(ssid, password):
    print(f"\n[1/4] Connecting to '{ssid}'...")
    out, code = run(f'sudo nmcli device wifi connect "{ssid}" password "{password}"', check=False)
    if code != 0:
        # Already saved — just bring it up
        out2, code2 = run(f'sudo nmcli connection up "{ssid}"', check=False)
        if code2 != 0:
            print(f"Failed to connect: {out}\n{out2}")
            return

    print("[2/4] Waiting for DHCP address...")
    subnet = get_gateway_and_subnet()
    if not subnet:
        print("Could not detect subnet. Is the router reachable?")
        return

    gateway    = f"{subnet}.1"
    static_ip  = f"{subnet}.{STATIC_HOST}"

    print(f"[3/4] Setting static IP {static_ip} (gateway {gateway})...")
    run(f'sudo nmcli connection modify "{ssid}" '
        f'ipv4.addresses {static_ip}/24 '
        f'ipv4.gateway {gateway} '
        f'ipv4.dns {STATIC_DNS} '
        f'ipv4.method manual '
        f'connection.autoconnect yes')

    print("[4/4] Applying...")
    run(f'sudo nmcli connection up "{ssid}"', check=False)
    time.sleep(2)

    # Verify
    out, _ = run("ip addr show wlan0", check=False)
    if static_ip in out:
        print(f"\nDone. SSH via WiFi: ssh tanglawan@{static_ip}")
    else:
        print(f"\nApplied but IP not confirmed yet. Try: ssh tanglawan@{static_ip}")

def list_networks():
    print("\nSaved WiFi connections:")
    out, _ = run("nmcli -f NAME,TYPE,DEVICE connection show", check=False)
    for line in out.splitlines():
        if "wifi" in line:
            print(" ", line)

    print("\nCurrent wlan0 IP:")
    out, _ = run("ip addr show wlan0 | grep 'inet '", check=False)
    print(" ", out.strip() if out else "(not connected)")

def status():
    print("\nNetwork status:")
    out, _ = run("nmcli device status", check=False)
    print(out)
    print("\nCurrent routes:")
    out, _ = run("ip route show", check=False)
    print(out)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "add":
        if len(sys.argv) < 4:
            print("Usage: python3 wifi_setup.py add SSID PASSWORD")
            sys.exit(1)
        add_network(sys.argv[2], sys.argv[3])

    elif cmd == "list":
        list_networks()

    elif cmd == "status":
        status()

    else:
        print(__doc__)