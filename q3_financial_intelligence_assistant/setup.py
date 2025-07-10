#!/usr/bin/env python3
"""
Setup script for Financial Intelligence RAG System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "uploads",
        "logs",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 11):
        print("✗ Python 3.11 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    
    print(f"✓ Python version {sys.version_info.major}.{sys.version_info.minor} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies"""
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    return True


def create_env_file():
    """Create .env file from template"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if env_file.exists():
        print("✓ .env file already exists")
        return True
    
    if env_example.exists():
        shutil.copy(env_example, env_file)
        print("✓ Created .env file from template")
        print("⚠️  Please edit .env file with your configuration")
        return True
    else:
        print("✗ env.example file not found")
        return False


def check_services():
    """Check if required services are available"""
    services = {
        "redis": "redis-cli ping",
        "postgres": "psql --version"
    }
    
    for service, command in services.items():
        try:
            subprocess.run(command, shell=True, check=True, capture_output=True)
            print(f"✓ {service} is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"⚠️  {service} is not available or not in PATH")
            print(f"   Please install and start {service} before running the application")


def main():
    """Main setup function"""
    print("🚀 Setting up Financial Intelligence RAG System")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("✗ Failed to install dependencies")
        sys.exit(1)
    
    # Create environment file
    if not create_env_file():
        print("✗ Failed to create environment file")
        sys.exit(1)
    
    # Check services
    check_services()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your configuration")
    print("2. Start Redis and PostgreSQL services")
    print("3. Run: python main.py")
    print("4. Access the API at: http://localhost:8000")
    print("5. View documentation at: http://localhost:8000/docs")
    print("\nFor load testing:")
    print("locust -f load_tests/locustfile.py --host=http://localhost:8000")


if __name__ == "__main__":
    main() 