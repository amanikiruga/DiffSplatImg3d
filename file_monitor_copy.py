#!/usr/bin/env python3
"""
File Monitor and Copy Script
Runs every 30 minutes and copies new files from source to destination directory.
"""

import os
import shutil
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Default Configuration
DEFAULT_SOURCE_DIR = "/tmp/openai-2025-07-22-22-29-00-760380"
DEFAULT_DEST_DIR = "/om/user/akiruga/diffsplatimg3d/checkpoints-plenoxels-diffusion/20250722_222854/model_log"
STATE_FILE = "file_monitor_state.json"
INTERVAL_MINUTES = 30
LOG_FILE = "file_monitor.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FileMonitor:
    def __init__(self, source_dir, dest_dir, state_file):
        self.source_dir = Path(source_dir)
        self.dest_dir = Path(dest_dir)
        self.state_file = Path(state_file)
        self.copied_files = self.load_state()
        
        # Ensure destination directory exists
        self.dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Destination directory: {self.dest_dir}")
        logger.info(f"Source directory: {self.source_dir}")
    
    def load_state(self):
        """Load the state of previously copied files."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                return data.get('copied_files', {})
            except (json.JSONDecodeError, KeyError):
                logger.warning("Could not load state file, starting fresh")
                return {}
        return {}
    
    def save_state(self):
        """Save the current state of copied files."""
        state = {
            'copied_files': self.copied_files,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_file_info(self, file_path):
        """Get file modification time and size."""
        stat = file_path.stat()
        return {
            'mtime': stat.st_mtime,
            'size': stat.st_size
        }
    
    def find_new_files(self):
        """Find files that are new or have been modified."""
        new_files = []
        
        if not self.source_dir.exists():
            logger.warning(f"Source directory does not exist: {self.source_dir}")
            return new_files
        
        try:
            for file_path in self.source_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = str(file_path.relative_to(self.source_dir))
                    current_info = self.get_file_info(file_path)
                    
                    # Check if file is new or modified
                    if (relative_path not in self.copied_files or 
                        self.copied_files[relative_path]['mtime'] != current_info['mtime'] or
                        self.copied_files[relative_path]['size'] != current_info['size']):
                        
                        new_files.append((file_path, relative_path, current_info))
                        
        except Exception as e:
            logger.error(f"Error scanning source directory: {e}")
        
        return new_files
    
    def copy_file(self, source_file, relative_path):
        """Copy a single file to the destination directory."""
        dest_file = self.dest_dir / relative_path
        
        # Create parent directories if they don't exist
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            shutil.copy2(source_file, dest_file)
            logger.info(f"Copied: {relative_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy {relative_path}: {e}")
            return False
    
    def run_copy_cycle(self):
        """Run one cycle of finding and copying new files."""
        logger.info("Starting copy cycle...")
        
        new_files = self.find_new_files()
        
        if not new_files:
            logger.info("No new files found")
            return
        
        logger.info(f"Found {len(new_files)} new/modified files")
        
        copied_count = 0
        for source_file, relative_path, file_info in new_files:
            if self.copy_file(source_file, relative_path):
                self.copied_files[relative_path] = file_info
                copied_count += 1
        
        if copied_count > 0:
            self.save_state()
            logger.info(f"Successfully copied {copied_count} files")
        
        logger.info("Copy cycle completed")
    
    def run_forever(self, interval_minutes):
        """Run the monitor continuously."""
        logger.info(f"Starting file monitor (interval: {interval_minutes} minutes)")
        logger.info(f"Source: {self.source_dir}")
        logger.info(f"Destination: {self.dest_dir}")
        
        while True:
            try:
                self.run_copy_cycle()
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in copy cycle: {e}")
            
            # Wait for the specified interval
            logger.info(f"Waiting {interval_minutes} minutes until next cycle...")
            time.sleep(interval_minutes * 60)
        
        logger.info("File monitor stopped")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monitor a directory and copy new files to a destination every 30 minutes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default directories
  %(prog)s -s /tmp/source -d /home/dest       # Custom source and destination
  %(prog)s --source /tmp/source               # Custom source, default destination
  %(prog)s --dest /home/dest                  # Default source, custom destination
        """
    )
    
    parser.add_argument(
        '-s', '--source',
        default=DEFAULT_SOURCE_DIR,
        help=f'Source directory to monitor (default: {DEFAULT_SOURCE_DIR})'
    )
    
    parser.add_argument(
        '-d', '--dest',
        default=DEFAULT_DEST_DIR,
        help=f'Destination directory for copied files (default: {DEFAULT_DEST_DIR})'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=INTERVAL_MINUTES,
        help=f'Interval between copy cycles in minutes (default: {INTERVAL_MINUTES})'
    )
    
    parser.add_argument(
        '--state-file',
        default=STATE_FILE,
        help=f'State file to track copied files (default: {STATE_FILE})'
    )
    
    parser.add_argument(
        '--log-file',
        default=LOG_FILE,
        help=f'Log file location (default: {LOG_FILE})'
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Update logging configuration with custom log file
    global LOG_FILE
    LOG_FILE = args.log_file
    
    # Reconfigure logging with the specified log file
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ],
        force=True  # Override existing configuration
    )
    
    logger.info(f"Starting file monitor with:")
    logger.info(f"  Source: {args.source}")
    logger.info(f"  Destination: {args.dest}")
    logger.info(f"  Interval: {args.interval} minutes")
    logger.info(f"  State file: {args.state_file}")
    logger.info(f"  Log file: {args.log_file}")
    
    monitor = FileMonitor(args.source, args.dest, args.state_file)
    
    # Run once immediately, then start the regular schedule
    monitor.run_copy_cycle()
    
    # Start the continuous monitoring
    monitor.run_forever(args.interval)

if __name__ == "__main__":
    main() 