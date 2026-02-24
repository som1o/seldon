# Enabling Excel Import

Seldon can now ingest `.xlsx` and `.xls` inputs by converting them to CSV before typed loading.

## Required tools

- `.xlsx`: `xlsx2csv`
- `.xls`: `xls2csv` (from `catdoc` / `libxls` toolchain)

Example install (Ubuntu/Debian):

```bash
sudo apt-get update
sudo apt-get install -y xlsx2csv catdoc
```

## How it works

- Input sources `.xlsx`, `.xls`, `.csv.gz`, `.csv.zip` are converted to a temporary CSV file.
- The core typed parser then runs exactly as for native CSV.
- Temporary conversion files are removed automatically after load.

## Notes

- If required converters are missing, Seldon returns a clear error explaining what to install.
- `.csv.gz` uses `gzip -cd`; `.csv.zip` uses `unzip -p`.
- Conversion is streamed to disk temp files (not fully kept in RAM).
