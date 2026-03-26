# Backup and Restore

PomaiDB backup flow uses portable `tar.zst` snapshots.

## Export
```bash
pomaictl snapshot --mode export --path /data/pomaidb --output /backup/pomaidb-$(date +%F).tar.zst
```

## Import
```bash
pomaictl snapshot --mode import --path /unused --output /backup/pomaidb.tar.zst --target /restore
```

## Verify/Repair
```bash
pomaictl doctor-repair --path /data/pomaidb
pomaictl doctor-repair --path /data/pomaidb --repair
```
