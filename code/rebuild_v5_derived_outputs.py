#!/usr/bin/env python3
"""Rebuild V5 derived summaries, run tests, and regenerate SHA256SUMS.

This does not rerun the full Monte Carlo campaign. It rebuilds V5 summary
tables and figures from the locked raw outputs, executes the automated tests,
and refreshes the integrity manifest.
"""
from __future__ import annotations
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main() -> None:
    subprocess.run([sys.executable,str(ROOT/'code'/'test_step3.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'code'/'build_v5_summaries.py')],cwd=ROOT,check=True)
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p.name=='SHA256SUMS.txt' or '__pycache__' in p.parts:
            continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        entries.append(f'{h}  {p.relative_to(ROOT).as_posix()}')
    (ROOT/'SHA256SUMS.txt').write_text('\n'.join(entries)+'\n')
    print(f'Wrote {len(entries)} SHA-256 entries.')

if __name__=='__main__':
    main()
