# Mayan → Paperless Migration

`migration.py` migrates documents from [Mayan EDMS](https://www.mayan-edms.com/) into [Paperless‑ngx](https://docs.paperless-ngx.com/). It transfers documents by document type, tags them in Paperless, and marks the originals in Mayan as moved to prevent duplicates.

## Features
- Migrates documents grouped by Mayan document types defined in `config.json`.
- Skips documents already tagged as moved in Mayan.
- Ensures Paperless tags exist (creates them when missing).
- Robust download pipeline: follows redirects, falls back to document version exports, retries empty downloads with exponential backoff.
- Modes for dry runs (`--count-only`) and sample downloads (`--download-first`).
- Debug logging (`--debug`) to trace Mayan/Paperless API calls.

## Prerequisites
1. **Python 3.10+** and dependencies:
   ```bash
   pip install requests
   ```
2. **Configuration file** (`config.json`). Copy `config.example.json` to `config.json` and fill in real values:
   ```json
   {
     "mayan_base": "https://mayan.example.com/",
     "paperless_base": "https://paperless.example.com/",
     "mayan_user": "admin",
     "mayan_pass": "secret",
     "mayan_moved_tag": "moved",
     "mappings": {
       "ExampleDocType": {
         "paperless_token": "paperless-api-token",
         "paperless_tag_name": "PaperlessTag"
       }
     },
     "export_poll_interval": 1.0,
     "export_poll_timeout": 60.0,
     "download_retry_limit": 5,
     "download_backoff_min": 0.5,
     "download_backoff_max": 5.0,
     "download_backoff_multiplier": 2.0
   }
   ```
   - One mapping entry per Mayan document type `ExampleDocType`. Each defines the Paperless API token and tag to apply.
3. **Mayan preparation**
   - Ensure each document type exists and contains the documents you need.
   - Create the `mayan_moved_tag` (default `moved`). The script attaches it after migrating a document.
   - Use a Mayan user with API access to documents, tags, and downloads.
4. **Paperless preparation**
   - Generate API tokens for each Paperless account in `mappings`.
   - Create the Paperless tags referenced by `paperless_tag_name` (the script also attempts creation if missing).

## Usage
1. Edit `config.json` with your environment details.
2. Dry run (count only):
   ```bash
   python migration.py --count-only --debug
   ```
   - Confirms credentials/connectivity and prints how many documents would migrate.
3. Verify a single download:
   ```bash
   python migration.py --download-first --debug
   ```
   - Saves the first eligible document as `sample_<doc_id>_<name>` locally without uploading.
4. Run the migration:
   ```bash
   python migration.py --debug
   ```
   - Uses `config.json` by default. Override with `--config path/to/custom.json`.
   - Additional overrides available: `--mayan-base`, `--paperless-base`, `--export-poll-interval`, `--export-poll-timeout`, `--download-retry-limit`, and backoff settings. See `python migration.py --help`.

## Operational Notes
- The script caches Mayan document details and tags per run to minimize API calls.
- Download retries use exponential backoff. Exported files are polled via `/downloads/` until ready.
- Failures on individual documents are logged; the script continues with the rest and reports the number of errors at the end.
- After migrating, verify documents in Paperless by searching for the target tags.

## Troubleshooting
- **Missing config/stanza**: The script raises an error if required keys are absent in `config.json`.
- **404 downloads**: Ensure documents have files or allow exports (default fallback). Check Mayan permissions.
- **400 from Paperless**: Typically indicates empty/unsupported files. Use `--debug` and/or `--download-first` to inspect the source.
- **Duplicates**: Confirm the Mayan moved tag exists and matches `mayan_moved_tag`.

Adjust mappings and rerun as needed to migrate additional document types.
