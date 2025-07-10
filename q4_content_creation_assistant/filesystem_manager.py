import os
import shutil
import json
import datetime
from pathlib import Path
from typing import List, Dict, Optional

class FilesystemManager:
    def __init__(self):
        """Initialize the filesystem manager with workspace paths"""
        self.workspace_path = Path("content-workspace")
        self.ideas_path = self.workspace_path / "ideas"
        self.generated_path = self.workspace_path / "generated"
        self.published_path = self.workspace_path / "published"
        self.templates_path = self.workspace_path / "templates"
        
        # Ensure all directories exist
        self.setup_workspace()
    
    def setup_workspace(self):
        """Create the required workspace directories"""
        directories = [
            self.workspace_path,
            self.ideas_path,
            self.generated_path,
            self.published_path,
            self.templates_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def write_file(self, file_path: str, content: str, metadata: Optional[Dict] = None) -> bool:
        """
        Write content to a file with optional metadata
        
        Args:
            file_path: Path to the file (relative to workspace)
            content: Content to write
            metadata: Optional metadata to include
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure the file path is within the workspace
            full_path = self.workspace_path / file_path
            
            # Create parent directories if they don't exist
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # If metadata is provided, create a metadata file
            if metadata:
                metadata_path = full_path.with_suffix('.json')
                metadata['last_modified'] = datetime.datetime.now().isoformat()
                metadata['file_path'] = str(file_path)
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error writing file {file_path}: {str(e)}")
            return False
    
    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read content from a file
        
        Args:
            file_path: Path to the file (relative to workspace)
            
        Returns:
            str: File content or None if error
        """
        try:
            full_path = self.workspace_path / file_path
            
            if not full_path.exists():
                print(f"File not found: {file_path}")
                return None
            
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None
    
    def edit_file(self, file_path: str, content: str) -> bool:
        """
        Edit/update an existing file
        
        Args:
            file_path: Path to the file (relative to workspace)
            content: New content to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        return self.write_file(file_path, content)
    
    def move_file(self, source_path: str, destination_path: str) -> bool:
        """
        Move a file from source to destination
        
        Args:
            source_path: Source file path (relative to workspace)
            destination_path: Destination file path (relative to workspace)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            source_full = self.workspace_path / source_path
            dest_full = self.workspace_path / destination_path
            
            if not source_full.exists():
                print(f"Source file not found: {source_path}")
                return False
            
            # Create destination directory if it doesn't exist
            dest_full.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the file
            shutil.move(str(source_full), str(dest_full))
            
            # Move metadata file if it exists
            metadata_source = source_full.with_suffix('.json')
            if metadata_source.exists():
                metadata_dest = dest_full.with_suffix('.json')
                shutil.move(str(metadata_source), str(metadata_dest))
            
            return True
            
        except Exception as e:
            print(f"Error moving file from {source_path} to {destination_path}: {str(e)}")
            return False
    
    def list_directory(self, directory_path: str = "") -> List[Dict]:
        """
        List contents of a directory
        
        Args:
            directory_path: Directory path (relative to workspace)
            
        Returns:
            List[Dict]: List of files and directories with metadata
        """
        try:
            full_path = self.workspace_path / directory_path
            
            if not full_path.exists():
                print(f"Directory not found: {directory_path}")
                return []
            
            contents = []
            
            for item in full_path.iterdir():
                item_info = {
                    'name': item.name,
                    'path': str(item.relative_to(self.workspace_path)),
                    'type': 'directory' if item.is_dir() else 'file',
                    'size': item.stat().st_size if item.is_file() else None,
                    'modified': datetime.datetime.fromtimestamp(item.stat().st_mtime).isoformat()
                }
                contents.append(item_info)
            
            # Sort by type (directories first) then by name
            contents.sort(key=lambda x: (x['type'] == 'file', x['name'].lower()))
            
            return contents
            
        except Exception as e:
            print(f"Error listing directory {directory_path}: {str(e)}")
            return []
    
    def delete_file(self, file_path: str) -> bool:
        """
        Delete a file
        
        Args:
            file_path: Path to the file (relative to workspace)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            full_path = self.workspace_path / file_path
            
            if not full_path.exists():
                print(f"File not found: {file_path}")
                return False
            
            # Delete the file
            full_path.unlink()
            
            # Delete metadata file if it exists
            metadata_path = full_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()
            
            return True
            
        except Exception as e:
            print(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    def get_file_metadata(self, file_path: str) -> Optional[Dict]:
        """
        Get metadata for a file
        
        Args:
            file_path: Path to the file (relative to workspace)
            
        Returns:
            Dict: File metadata or None if not found
        """
        try:
            metadata_path = (self.workspace_path / file_path).with_suffix('.json')
            
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error reading metadata for {file_path}: {str(e)}")
            return None
    
    def update_file_metadata(self, file_path: str, metadata: Dict) -> bool:
        """
        Update metadata for a file
        
        Args:
            file_path: Path to the file (relative to workspace)
            metadata: New metadata to write
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            metadata_path = (self.workspace_path / file_path).with_suffix('.json')
            
            # Load existing metadata if it exists
            existing_metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    existing_metadata = json.load(f)
            
            # Update with new metadata
            existing_metadata.update(metadata)
            existing_metadata['last_modified'] = datetime.datetime.now().isoformat()
            
            # Write updated metadata
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(existing_metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error updating metadata for {file_path}: {str(e)}")
            return False
    
    def get_workspace_stats(self) -> Dict:
        """
        Get statistics about the workspace
        
        Returns:
            Dict: Workspace statistics
        """
        try:
            stats = {
                'total_files': 0,
                'total_directories': 0,
                'ideas_count': 0,
                'generated_count': 0,
                'published_count': 0,
                'templates_count': 0,
                'total_size': 0
            }
            
            # Count files in each directory
            for root, dirs, files in os.walk(self.workspace_path):
                stats['total_directories'] += len(dirs)
                stats['total_files'] += len(files)
                
                # Count files by directory type
                rel_path = Path(root).relative_to(self.workspace_path)
                if rel_path == Path('ideas'):
                    stats['ideas_count'] += len(files)
                elif rel_path == Path('generated'):
                    stats['generated_count'] += len(files)
                elif rel_path == Path('published'):
                    stats['published_count'] += len(files)
                elif rel_path == Path('templates'):
                    stats['templates_count'] += len(files)
                
                # Calculate total size
                for file in files:
                    file_path = Path(root) / file
                    if file_path.is_file():
                        stats['total_size'] += file_path.stat().st_size
            
            return stats
            
        except Exception as e:
            print(f"Error getting workspace stats: {str(e)}")
            return {}
    
    def create_backup(self, backup_name: str = None) -> bool:
        """
        Create a backup of the workspace
        
        Args:
            backup_name: Name for the backup (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not backup_name:
                backup_name = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            backup_path = Path(f"backups/{backup_name}")
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy workspace to backup
            shutil.copytree(self.workspace_path, backup_path / "content-workspace", dirs_exist_ok=True)
            
            return True
            
        except Exception as e:
            print(f"Error creating backup: {str(e)}")
            return False 