# File Monitor and Copy Script

This script monitors `/tmp/openai-2025-07-22-22-29-00-760380` and copies new files to `/om/user/akiruga/diffsplatimg3d/checkpoints-plenoxels-diffusion/20250722_222854/model_log` every 30 minutes.

## Features

- **Smart Detection**: Only copies new or modified files (based on modification time and file size)
- **State Persistence**: Remembers what files have been copied to avoid duplicates
- **Robust Logging**: Logs all operations to both console and `file_monitor.log`
- **Error Handling**: Continues running even if individual copy operations fail
- **Directory Creation**: Automatically creates destination directories as needed

## Usage

### Option 1: Run Directly

**With default directories:**
```bash
python3 file_monitor_copy.py
```

**With custom directories:**
```bash
# Custom source and destination
python3 file_monitor_copy.py -s /tmp/my-source -d /path/to/destination

# Custom source only (uses default destination)
python3 file_monitor_copy.py --source /tmp/my-source

# Custom destination only (uses default source)
python3 file_monitor_copy.py --dest /path/to/destination

# Custom interval (15 minutes instead of 30)
python3 file_monitor_copy.py --interval 15

# Custom state and log files
python3 file_monitor_copy.py --state-file my_state.json --log-file my_monitor.log
```

**View help:**
```bash
python3 file_monitor_copy.py --help
```

The script will:
1. Run one copy cycle immediately
2. Then wait for the specified interval (default: 30 minutes) between subsequent cycles
3. Log all activity to the specified log file (default: `file_monitor.log`)

### Option 2: Run as Background Service (Recommended)

1. **Copy the service file to systemd:**
   ```bash
   sudo cp file_monitor_copy.service /etc/systemd/system/
   ```

2. **Reload systemd and enable the service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable file_monitor_copy.service
   ```

3. **Start the service:**
   ```bash
   sudo systemctl start file_monitor_copy.service
   ```

4. **Check service status:**
   ```bash
   sudo systemctl status file_monitor_copy.service
   ```

5. **View logs:**
   ```bash
   sudo journalctl -u file_monitor_copy.service -f
   ```

### Option 3: Run with nohup (Alternative background method)

```bash
nohup python3 file_monitor_copy.py > file_monitor_output.log 2>&1 &
```

## Configuration

You can configure the script in two ways:

### 1. Command-line Arguments (Recommended)

Use command-line arguments to specify directories and options:

- `--source` / `-s`: Source directory to monitor
- `--dest` / `-d`: Destination directory for copied files  
- `--interval`: Time between copy cycles in minutes (default: 30)
- `--state-file`: File to store tracking information (default: `file_monitor_state.json`)
- `--log-file`: Log file location (default: `file_monitor.log`)

### 2. Edit Default Values

Alternatively, edit the default constants at the top of `file_monitor_copy.py`:

- `DEFAULT_SOURCE_DIR`: Default source directory to monitor
- `DEFAULT_DEST_DIR`: Default destination directory for copied files
- `INTERVAL_MINUTES`: Default time between copy cycles (30 minutes)
- `STATE_FILE`: Default file to store tracking information
- `LOG_FILE`: Default log file location

## Files Created

- `file_monitor_state.json`: Tracks which files have been copied
- `file_monitor.log`: Activity log file

## Stopping the Service

```bash
sudo systemctl stop file_monitor_copy.service
sudo systemctl disable file_monitor_copy.service  # To prevent auto-start on boot
```

## Troubleshooting

1. **Check if source directory exists:**
   ```bash
   ls -la /tmp/openai-2025-07-22-22-29-00-760380
   ```

2. **Check permissions on destination directory:**
   ```bash
   ls -la /om/user/akiruga/diffsplatimg3d/checkpoints-plenoxels-diffusion/20250722_222854/
   ```

3. **View recent logs:**
   ```bash
   tail -f file_monitor.log
   ```

4. **Test run a single cycle:**
   ```bash
   # With default directories
   python3 file_monitor_copy.py --help
   
   # Test with custom directories (replace with your paths)
   python3 file_monitor_copy.py -s /tmp/openai-2025-07-22-22-29-00-760380 -d /path/to/dest --interval 1
   ```

5. **Run with custom arguments via service:**
   If you need to modify the service to use different directories, edit the service file's `ExecStart` line:
   ```bash
   ExecStart=/usr/bin/python3 /om/user/akiruga/diffsplatimg3d/file_monitor_copy.py -s /custom/source -d /custom/dest
   ``` 