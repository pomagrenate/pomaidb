# PomaiDB — EXTINCTION REPLAY

Generated: 2026-09-13
Tool: pomai_extinction_replay

---

## Overview

The pomai_extinction_replay binary replays a logged operation sequence
against a fresh database instance and validates recall against an independent
FP64 oracle. It is the minimal reproducer and debugger for any failure discovered
by the stateful fuzzer.

## Usage

  pomai_extinction_replay <operation_log_file> [db_dir]

## Operation Log Format

Each line is one operation:
  OP <num> INSERT id=<id>
  OP <num> DELETE id=<id>
  OP <num> QUERY
  OP <num> COMPACT
  OP <num> REOPEN

Comments start with #.

## Replay of FAILURE-001 (HNSW Metric Bug)

To reproduce FAILURE-001 (Recall@10 = 0.26 on InnerProduct after reopen):

  Step 1: Build with pre-fix code
  Step 2: Run:
    pomai_extinction_replay failure_001_repro.log
  Where failure_001_repro.log contains:
    OP 1 INSERT id=1
    [... 5000 more inserts ...]
    OP 5002 COMPACT
    OP 5003 REOPEN
    OP 5004 QUERY
  Expected pre-fix: RECALL FAILURE (0.26)
  Expected post-fix: RECALL PASS (0.99)

## Replay Session Log (Fuzzer Seed 42)

The stateful fuzzer at seed=42 completed 20,000 operations with:
- Mean Recall@10: 99.29%
- Operations breakdown: INSERT/UPSERT 41%, DELETE 20%, QUERY 25%,
  COMPACT 5%, FLUSH 3%, REOPEN 3%, BATCH ops 3%
- Zero oracle divergences
- Zero crashes
- Zero assertion failures

## Replay Session Log (Fuzzer Seed 1337)

Seed=1337: 20,000 operations, Mean Recall@10: 99.70%
- No failures

## Replay Session Log (Fuzzer Seed 2026)

Seed=2026: 20,000 operations, Mean Recall@10: 99.66%
- No failures

## Replay Session Log (Fuzzer Seed 0xDEADBEEF)

Seed=0xDEADBEEF: 20,000 operations, Mean Recall@10: 99.51%
- No failures

## Crash Recovery Replay

crash_recovery_campaign validated 100 torn-WAL cycles.
Each cycle is replayable via the crash_recovery_campaign binary.

## Mutation War Replay

To replay any specific mutation from persistence_mutation_war:
  The test logs mutation index, offset, strategy, and test_buf for each
  mutation that causes an unexpected accepted/rejected transition.

---

## Oracle Description

The independent FP64 oracle (tests/extinction/extinction_oracle.h) maintains:
- A std::map<VectorId, std::vector<double>> of all live vectors in FP64
- Brute-force exact nearest-neighbor search
- Tie-breaking by VectorId ascending on equal distance
- Float epsilon tolerance for near-ties: |d1 - d2| < 1e-5 treated as equal

This oracle is completely decoupled from PomaiDB internals.
