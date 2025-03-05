#!/usr/bin/env python3
from ast import arg
import os
import shutil
import subprocess
import argparse
from pathlib import Path
import platform
import logging
from typing import List

from definition_scripts import MODEL_DEFINITIONS_SCRIPT, DATA_IMPORT_DEFINITION_SCRIPT


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ProjectSetup:
    def __init__(self, project_dir: str, database_name: str, port: str, source_dir: str, db_user: str, db_password: str, skip_copy: bool = False, skip_import: bool = False) -> None:
        self.project_dir = Path(project_dir).absolute()
        parent_dir_name = self.project_dir.name
        self.venv_path = self.project_dir / f".{parent_dir_name}-venv"
        self.database_name = database_name
        self.db_port = port
        self.source_dir = source_dir
        self.db_user = db_user
        self.db_password = db_password
        self.skip_copy = skip_copy
        self.skip_import = skip_import
        
        self.is_windows = platform.system() == "Windows"
        self.python_cmd = "python" if self.is_windows else "python3"
        self.pip_cmd = str(self.venv_path / "Scripts" / "pip") if self.is_windows else str(self.venv_path / "bin" / "pip")
        self.alembic_cmd = str(self.venv_path / "Scripts" / "alembic") if self.is_windows else str(self.venv_path / "bin" / "alembic")

        os.environ["DB_PORT"] = self.db_port
        os.environ["DB_NAME"] = self.database_name
        os.environ["DB_USER"] = self.db_user
        os.environ["DB_PASSWORD"] = self.db_password
        os.environ["SOURCE_DIR"] = self.source_dir
        
        logger.info(f"Virtual environment will be created as: {self.venv_path}")
        logger.info(f"Database name will be: {self.database_name}")
        logger.info(f"Database port will be: {self.db_port}")
        logger.info(f"Database user will be: {self.db_user}")
        logger.info(f"Database password: {'*' * len(self.db_password)}")
        logger.info(f"Source directory for CSV files: {self.source_dir}")
        if self.skip_copy:
            logger.info("CSV file copying will be skipped as requested")
        if self.skip_import:
            logger.info("Data import will be skipped as requested")


    def run_command(self, command: List[str], cwd: str = None) -> None:
        """Run a command and log its output"""
        try:
            logger.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=cwd or str(self.project_dir),
                check=True,
                text=True,
                capture_output=True
            )
            if result.stdout:
                logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e.stderr}")
            raise


    def create_virtual_environment(self) -> None:
        """Create virtual environment"""
        if not self.venv_path.exists():
            logger.info("Creating virtual environment...")
            self.run_command([self.python_cmd, "-m", "venv", str(self.venv_path)])


    def install_dependencies(self) -> None:
        """Install project dependencies"""
        requirements = [
            "sqlalchemy",
            "alembic",
            "psycopg2-binary",
            "python-dotenv",
            "tqdm",
            "fastapi",
            "uvicorn",
            "pandas",  # For data manipulation
            "numpy",   # Required by pandas
            "openpyxl",  # For Excel file support if needed
            'pytest',
            'pytest-cov',
            'pytest-postgresql',
        ]
        
        logger.info("Installing dependencies...")
    
        python_executable = str(self.venv_path / "Scripts" / "python") if self.is_windows else str(self.venv_path / "bin" / "python")

        self.run_command([python_executable, "-m", "pip", "install", "--upgrade", "pip"])
        self.run_command([python_executable, "-m", "pip", "install"] + requirements)

        with open(self.project_dir / "requirements.txt", "w") as f:
            f.write("\n".join(requirements))


    def create_project_structure(self) -> None:
        """Create project directory structure"""
        directories = [
            "migrations/versions",
            "scripts",
            "data",
            "models",
            "tests",
        ]

        logger.info("Creating project structure...")
        for directory in directories:
            (self.project_dir / directory).mkdir(parents=True, exist_ok=True)

    def create_files(self) -> None:
        """Create necessary project files"""
        # Create model files
        logger.info("Creating model files...")
        for filepath, content in MODEL_DEFINITIONS_SCRIPT.items():
            full_path = self.project_dir / filepath
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            logger.info(f"Created {filepath}")

        # Create data import files
        logger.info("Creating data import files...")
        for filepath, content in DATA_IMPORT_DEFINITION_SCRIPT.items():
            full_path = self.project_dir / filepath
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            logger.info(f"Created {filepath}")


        # Define the content for db.py
        db_content = (
            'from sqlalchemy import create_engine\n'
            'from sqlalchemy.orm import sessionmaker, declarative_base\n'
            'from sqlalchemy.pool import QueuePool\n'
            'import os\n'
            'from dotenv import load_dotenv\n\n'
            'load_dotenv()\n\n'
            '# Database configuration\n'
            f'DB_USER = os.getenv(\'DB_USER\', \'{self.db_user}\')\n'
            f'DB_PASSWORD = os.getenv(\'DB_PASSWORD\', \'{self.db_password}\')\n'
            'DB_HOST = os.getenv(\'DB_HOST\', \'localhost\')\n'
            f'DB_NAME = os.getenv(\'DB_NAME\', \'{self.database_name}\')\n\n'
            f'DB_PORT = os.getenv(\'DB_PORT\', \'{self.db_port}\')\n'
            '# Determine environment\n'
            'IS_PROD = os.getenv(\'ENVIRONMENT\', \'development\') == \'production\'\n\n'
            '# Create database URL\n'
            'if IS_PROD:\n'
            '    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"\n'
            'else:\n'
            f'    DATABASE_URL = f"postgresql://{self.db_user}:{self.db_password}@localhost:{self.db_port}/{self.database_name}"\n\n'
            '# Create engine with connection pooling\n'
            'engine = create_engine(\n'
            '    DATABASE_URL,\n'
            '    poolclass=QueuePool,\n'
            '    pool_size=5,\n'
            '    max_overflow=10,\n'
            '    pool_timeout=30,\n'
            '    pool_pre_ping=True\n'
            ')\n\n'
            '# Create session factory\n'
            'SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\n\n'
            '# Create declarative base\n'
            'Base = declarative_base()\n\n'
            'def get_db():\n'
            '    """Dependency for FastAPI to get DB session"""\n'
            '    db = SessionLocal()\n'
            '    try:\n'
            '        yield db\n'
            '    finally:\n'
            '        db.close()\n\n'
            'def init_db():\n'
            '    """Initialize the database, creating all tables"""\n'
            '    Base.metadata.create_all(bind=engine)\n'
        )

        # Define the content for .env
        env_content = (
            f'DB_USER={self.db_user}\n'
            f'DB_PASSWORD={self.db_password}\n'
            'DB_HOST=localhost\n'
            f'DB_NAME={self.database_name}\n'
            f'DB_PORT={self.db_port}\n'
            'ENVIRONMENT=development\n'
        )

        # Define the content for .gitignore
        gitignore_content = (
            '# Python\n'
            '__pycache__/\n'
            '*.py[cod]\n'
            '*$py.class\n'
            '*.so\n'
            '.Python\n'
            'build/\n'
            'develop-eggs/\n'
            'dist/\n'
            'downloads/\n'
            'eggs/\n'
            '.eggs/\n'
            'lib/\n'
            'lib64/\n'
            'parts/\n'
            'sdist/\n'
            'var/\n'
            'wheels/\n'
            '*.egg-info/\n'
            '.installed.cfg\n'
            '*.egg\n\n'
            '# Virtual Environment\n'
            '.dme-env/\n'
            'venv/\n'
            'ENV/\n\n'
            '# Environment Variables\n'
            '.env\n\n'
            '# IDE\n'
            '.idea/\n'
            '.vscode/\n'
            '*.swp\n'
            '*.swo\n\n'
            '# Database\n'
            '*.sqlite3\n'
            '*.db\n\n'
            '# Logs\n'
            '*.log\n'
            'logs/\n'
        )

        # Create dictionary of files
        files = {
            "db.py": db_content,
            ".env": env_content,
            ".gitignore": gitignore_content
        }

        # Create each file
        logger.info("Creating project files...")
        for filename, content in files.items():
            filepath = self.project_dir / filename
            with open(filepath, "w") as f:
                f.write(content)
            logger.info(f"Created {filename}")

    def initialize_alembic(self) -> None:
        """Initialize Alembic for database migrations"""
        logger.info("Initializing Alembic...")
        
        try:
            # First try using module execution
            self.run_command([
                str(self.venv_path / "Scripts" / "python") if self.is_windows else str(self.venv_path / "bin" / "python"),
                "-m", "alembic.config", "init", "migrations"
            ])
        except subprocess.CalledProcessError as e:
            logger.warning(f"First alembic initialization attempt failed: {e}")
        try:
            # Second attempt: Create alembic directory and files manually
            logger.info("Attempting manual alembic initialization...")
            
            # Create alembic directory structure
            alembic_dir = self.project_dir / "migrations"
            versions_dir = alembic_dir / "versions"
            versions_dir.mkdir(parents=True, exist_ok=True)
            
            # Create alembic.ini in root directory
            alembic_ini_content = '''# A generic, single database configuration.

[alembic]
# path to migration scripts
script_location = migrations

# template used to generate migration files
file_template = %%(year)d%%(month).2d%%(day).2d_%%(hour).2d%%(minute).2d%%(second).2d_%%(rev)s_%%(slug)s

# timezone to use when rendering the date
# within the migration file as well as the filename.
# string value is passed to dateutil.tz.gettz()
# leave blank for localtime
# timezone =

# max length of characters to apply to the
# "slug" field
# truncate_slug_length = 40

# set to 'true' to run the environment during
# the 'revision' command, regardless of autogenerate
# revision_environment = false

# set to 'true' to allow .pyc and .pyo files without
# a source .py file to be detected as revisions in the
# versions/ directory
# sourceless = false

# version location specification; this defaults
# to alembic/versions.  When using multiple version
# directories, initial revisions must be specified with --version-path
# version_locations = %(here)s/bar %(here)s/bat alembic/versions

# the output encoding used when revision files
# are written from script.py.mako
# output_encoding = utf-8

sqlalchemy.url = postgresql://{self.db_user}:{self.db_password}@localhost:{self.db_port}/{self.database_name}


[post_write_hooks]
# post_write_hooks defines scripts or Python functions that are run
# on newly generated revision scripts.  See the documentation for further
# detail and examples

# format using "black" - use the console_scripts runner, against the "black" entrypoint
# hooks=black
# black.type=console_scripts
# black.entrypoint=black
# black.options=-l 79

# Logging configuration
[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S'''

            # Write alembic.ini to root directory
            with open(self.project_dir / "alembic.ini", "w") as f:
                f.write(alembic_ini_content)
                
            # Create env.py with correct indentation
            env_py_content = '''from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

import os
import sys
import subprocess
import logging
from pathlib import Path
import re
from datetime import datetime
import inspect

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import your models here
from db import Base
from models import *

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# Get a logger
logger = logging.getLogger("alembic.env")

# Store the migration operation type globally
MIGRATION_OPERATION = None
# Flag to determine if we're running autogenerate
IS_AUTOGENERATE = False

def determine_operation_type():
    """Determine operation type and if we're running autogenerate"""
    stack = inspect.stack()
    is_autogenerate = False
    operation_type = 'upgrade'  # Default
    
    # Check the command line arguments first for the clearest signal
    cli_args = ' '.join(sys.argv)
    if 'revision' in cli_args and '--autogenerate' in cli_args:
        is_autogenerate = True
        operation_type = 'autogenerate'
        
    # Check the stack for more clues
    for frame in stack:
        if 'command.py' in frame.filename:
            # Look for the function name
            if 'downgrade' in frame.function.lower():
                operation_type = 'downgrade'
            elif 'upgrade' in frame.function.lower():
                operation_type = 'upgrade'
            elif 'revision' in frame.function.lower() and not is_autogenerate:
                # Double check for autogenerate in the frame locals
                frame_locals = frame.frame.f_locals
                if 'options' in frame_locals and hasattr(frame_locals['options'], 'cmd'):
                    command = frame_locals['options'].cmd
                    if 'revision' in command and '--autogenerate' in command:
                        is_autogenerate = True
                        operation_type = 'autogenerate'
            
            # Check the command line arguments in the frame locals
            frame_locals = frame.frame.f_locals
            if 'options' in frame_locals and hasattr(frame_locals['options'], 'cmd'):
                command = frame_locals['options'].cmd
                if isinstance(command, list) and command:
                    cmd_str = ' '.join(command)
                    if 'downgrade' in cmd_str:
                        operation_type = 'downgrade'
                    elif 'upgrade' in cmd_str:
                        operation_type = 'upgrade'
                    elif 'revision' in cmd_str and '--autogenerate' in cmd_str:
                        is_autogenerate = True
                        operation_type = 'autogenerate'
    
    return operation_type, is_autogenerate

# Set the operation type and autogenerate flag at module load time
MIGRATION_OPERATION, IS_AUTOGENERATE = determine_operation_type()
logger.info(f"Detected operation: {MIGRATION_OPERATION}, autogenerate: {IS_AUTOGENERATE}")

def get_url():
    """Get database URL from environment variable"""
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "mydb")
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"

def get_current_migration_name():
    """Extract the current migration name from the script context"""
    try:
        # Try to get the migration script from context
        if hasattr(context, '_script') and context._script:
            script = context._script
            if hasattr(context, 'get_revision_argument') and context.get_revision_argument():
                rev = context.get_revision_argument()
            elif hasattr(context, '_revision_context') and context._revision_context and context._revision_context.up_revisions:
                rev = context._revision_context.up_revisions[0].revision
            else:
                rev = None
                
            if rev and rev != 'head':
                # Get the migration script for this revision
                migration = script.get_revision(rev)
                if migration and migration.doc:
                    # Extract the first line of the docstring as the migration name
                    migration_name = migration.doc.split('\\n')[0].strip('"').strip()
                    # Clean up the name for use in a filename
                    return re.sub(r'[^\w\s-]', '', migration_name).strip().lower().replace(' ', '_')
                elif migration:
                    # Use script filename if docstring is not available
                    script_path = Path(migration.module.__file__)
                    # Extract the migration name from filename (removing timestamp and revision hash)
                    filename = script_path.stem
                    # Typical format: "20250224_174733_ab201b44ee76_initial_migration"
                    match = re.search(r'\d+_\d+_[a-f0-9]+_(.+)$', filename)
                    if match:
                        return match.group(1)
                    else:
                        return filename
        
        # Alternative approach: try to get from revision context
        if hasattr(context, '_revision_context') and context._revision_context:
            rev_ctx = context._revision_context
            if hasattr(rev_ctx, 'up_revisions') and rev_ctx.up_revisions:
                for rev in rev_ctx.up_revisions:
                    if rev.doc:
                        migration_name = rev.doc.split('\\n')[0].strip('"').strip()
                        return re.sub(r'[^\w\s-]', '', migration_name).strip().lower().replace(' ', '_')
                    elif hasattr(rev, 'module') and hasattr(rev.module, '__file__'):
                        script_path = Path(rev.module.__file__)
                        filename = script_path.stem
                        match = re.search(r'\d+_\d+_[a-f0-9]+_(.+)$', filename)
                        if match:
                            return match.group(1)
                        else:
                            return filename
    except Exception as e:
        logger.warning(f"Could not determine migration name: {str(e)}")
    
    # If we get here, try to extract from the migration filenames directly
    try:
        if os.environ.get('ALEMBIC_MIGRATION_FILENAME'):
            # If the environment has set the migration filename
            filename = os.environ.get('ALEMBIC_MIGRATION_FILENAME')
            match = re.search(r'\d+_\d+_[a-f0-9]+_(.+)\.py$', filename)
            if match:
                return match.group(1)
        else:
            # Try to find the migration directory
            project_root = Path(__file__).parent.parent
            versions_dir = project_root / 'migrations' / 'versions'
            if versions_dir.exists():
                # Find the most recently modified migration file
                migration_files = list(versions_dir.glob('*.py'))
                if migration_files:
                    latest_file = max(migration_files, key=lambda p: p.stat().st_mtime)
                    match = re.search(r'\d+_\d+_[a-f0-9]+_(.+)\.py$', latest_file.name)
                    if match:
                        return match.group(1)
    except Exception as e:
        logger.warning(f"Could not extract migration name from files: {str(e)}")
    
    return "migration"  # Default name if we can't determine it

def generate_erd_after_migration(migration_type):
    """Generate ERD after a successful migration"""
    # Skip ERD generation for autogenerate operations
    if IS_AUTOGENERATE:
        logger.info("Skipping ERD generation for autogenerate operation")
        return
        
    try:
        logger.info(f"Generating ERD after successful {migration_type}...")
        project_root = Path(__file__).parent.parent
        erd_script = project_root / "scripts" / "generate_erd.py"
        
        # Get migration name for the filename
        migration_name = get_current_migration_name()
        logger.info(f"Detected migration name: {migration_name}")
        
        # Check if the path exists before running
        if erd_script.exists():
            logger.info(f"ERD script found at: {erd_script}")
            
            # Create timestamp directly without subprocess
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create ERD filename with migration name and migration type
            erd_filename = f"{migration_name}_{migration_type}_{timestamp}_erd.md"
            output_path = project_root / "docs" / "erds" / erd_filename
            
            # Create the output directory if it doesn't exist
            (project_root / "docs" / "erds").mkdir(parents=True, exist_ok=True)
            
            # Run the ERD generation script with the output path
            cmd = [sys.executable, str(erd_script)]
            if output_path:
                cmd.extend(["--output", str(output_path)])
            
            logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                cwd=str(project_root),
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                logger.info(f"ERD generation output: {result.stdout}")
            if result.stderr:
                logger.warning(f"ERD generation warnings: {result.stderr}")
                
            logger.info(f"ERD generation completed successfully: {erd_filename}")
        else:
            logger.error(f"ERD script not found at expected path: {erd_script}")
            
            # Try an alternative path in case the casing is different
            alternative_path = project_root / "Scripts" / "generate_erd.py"
            if alternative_path.exists():
                logger.info(f"Found ERD script at alternative path: {alternative_path}")
                
                # Create timestamp directly
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Create ERD filename with migration name and migration type
                erd_filename = f"{migration_name}_{migration_type}_{timestamp}_erd.md"
                output_path = project_root / "docs" / "erds" / erd_filename
                
                # Create the output directory if it doesn't exist
                (project_root / "docs" / "erds").mkdir(parents=True, exist_ok=True)
                
                # Run the ERD generation script with the output path
                cmd = [sys.executable, str(alternative_path)]
                if output_path:
                    cmd.extend(["--output", str(output_path)])
                
                logger.info(f"Running command: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd,
                    check=True,
                    cwd=str(project_root),
                    capture_output=True,
                    text=True
                )
                
                if result.stdout:
                    logger.info(f"ERD generation output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"ERD generation warnings: {result.stderr}")
                    
                logger.info(f"ERD generation completed successfully: {erd_filename}")
            else:
                logger.error(f"ERD script not found at any checked paths")
    except Exception as e:
        logger.error(f"ERD generation failed: {str(e)}")
        # We don't want to fail the migration if ERD generation fails
        # so we just log the error and continue

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
    
    # Use the globally determined migration type
    generate_erd_after_migration(MIGRATION_OPERATION)

def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_url()
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()
        
        # Use the globally determined migration type
        generate_erd_after_migration(MIGRATION_OPERATION)

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()''' 

            with open(alembic_dir / "env.py", "w") as f:
                f.write(env_py_content)
                
            # Create script.py.mako
            script_mako_content = '''"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

"""
from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
'''
            with open(alembic_dir / "script.py.mako", "w") as f:
                f.write(script_mako_content)
                    
            logger.info("Manual alembic initialization completed successfully")
                
        except Exception as manual_error:
            logger.error(f"Manual alembic initialization failed: {manual_error}")
            raise

    def initialize_git(self) -> None:
        """Initialize Git repository"""
        logger.info("Initializing Git repository...")
        if not (self.project_dir / ".git").exists():
            self.run_command(["git", "init"])
            self.run_command(["git", "add", "."])
            self.run_command(["git", "commit", "-m", "Initial commit"])

    def create_database(self) -> None:
        """Create PostgreSQL database if it doesn't exist"""
        logger.info("Creating database...")
        
        db_creation_code = '''import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import dotenv

dotenv.load_dotenv()

try:
    conn = psycopg2.connect(
        dbname="postgres",
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    db_name = os.getenv("DB_NAME", "authdb")
    
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
    exists = cur.fetchone()
    
    if not exists:
        try:
            cur.execute(f'CREATE DATABASE "{db_name}"')
            print(f"Database '{db_name}' created successfully")
        except Exception as e:
            print(f"Error creating database: {e}")
            raise
    else:
        print(f"Database '{db_name}' already exists")
    
    cur.close()
    conn.close()
    
except psycopg2.Error as e:
    if "password authentication failed" in str(e):
        print("Database authentication failed. Please check your PostgreSQL credentials in .env file")
    elif "could not connect to server" in str(e):
        print("Could not connect to PostgreSQL server. Please make sure PostgreSQL is installed and running")
    else:
        print(f"Database error: {str(e)}")
    raise
except Exception as e:
    print(f"Unexpected error during database creation: {str(e)}")
    raise'''
        
        python_executable = str(self.venv_path / "Scripts" / "python") if self.is_windows else str(self.venv_path / "bin" / "python")
        try:
            self.run_command([python_executable, "-c", db_creation_code])
        except subprocess.CalledProcessError as e:
            if "No module named 'psycopg2'" in str(e):
                logger.error("psycopg2 module not found. Make sure it was installed correctly.")
            raise
    
    def copy_csv_files(self) -> bool:
        """Copy CSV files from auth_db and mapping subdirectories to project's data directory with user confirmation."""
        if self.skip_copy:
            logger.info("Skipping CSV file copying as requested (--skip_copy flag is set)")
            return False
            
        try:
            data_dir = self.project_dir / "data" / "import"
            if not data_dir.exists():
                data_dir.mkdir(parents=True)
                logger.info(f"Created data directory: {data_dir}")
                    
            # Define base directory path from source_dir
            base_dir = Path(self.source_dir).absolute()
            
            # Define auth_db and mapping directory paths
            auth_db_dir = base_dir / "auth_db"
            mapping_dir = base_dir / "mapping"
            
            files_copied = False
            
            # Copy auth_db CSV files
            if not auth_db_dir.exists():
                logger.warning(f"Auth DB directory does not exist: {auth_db_dir}")
            else:
                csv_files = list(auth_db_dir.glob("*.csv"))
                if not csv_files:
                    logger.warning(f"No CSV files found in {auth_db_dir}")
                else:
                    for csv_file in csv_files:
                        dest_file = data_dir / csv_file.name
                        shutil.copy2(csv_file, dest_file)
                        logger.info(f"Copied {csv_file.name} to data directory")
                    files_copied = True
            
            # Copy mapping CSV files
            if not mapping_dir.exists():
                logger.warning(f"Mapping directory does not exist: {mapping_dir}")
            else:
                mapping_files = list(mapping_dir.glob("*.csv"))
                if not mapping_files:
                    logger.warning(f"No mapping CSV files found in {mapping_dir}")
                else:
                    for mapping_file in mapping_files:
                        dest_file = data_dir / mapping_file.name
                        shutil.copy2(mapping_file, dest_file)
                        logger.info(f"Copied mapping file {mapping_file.name} to data directory")
                    files_copied = True
            
            if not files_copied:
                logger.warning("No CSV files were copied from either source directory")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error copying CSV files: {str(e)}")
            return False
        
    def import_csv_data(self) -> None:
        """Import CSV files into the database using the data_import script"""
        if self.skip_import:
            logger.info("Skipping data import as requested (--skip_import flag is set)")
            return
            
        logger.info("Starting CSV import process...")
        python_executable = str(self.venv_path / "Scripts" / "python") if self.is_windows else str(self.venv_path / "bin" / "python")
        
        try:
            self.run_command([python_executable, str(self.project_dir / "scripts" / "data_import.py")])
            logger.info("Data import completed successfully!")
        except subprocess.CalledProcessError as e:
            logger.error(f"Data import failed: {str(e)}")
            raise
    
    def add_erd_generation_function(self):
        """Add ERD generation functionality directly in the setup module"""
        logger.info("Adding ERD generation functionality...")
    
        erd_script_content = '''#!/usr/bin/env python3
        """
        ERD generator for PostgreSQL databases - creates a Mermaid diagram
        with proper entity relationship display and formatting
        """

        import os
        import sys
        from pathlib import Path
        import logging
        import argparse
        from sqlalchemy import create_engine, MetaData, inspect
        from sqlalchemy.dialects.postgresql import UUID
        from datetime import datetime
        from dotenv import load_dotenv
        import traceback

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)

        def get_db_url_from_env():
            """Get database URL from environment variables."""
            load_dotenv()
            
            db_user = os.getenv('DB_USER', 'postgres')
            db_password = os.getenv('DB_PASSWORD', 'postgres')
            db_host = os.getenv('DB_HOST', 'localhost')
            db_port = os.getenv('DB_PORT', '5432')
            db_name = os.getenv('DB_NAME', 'mydb')
            
            logger.info(f"Using database connection: {db_host}:{db_port}/{db_name} (user: {db_user})")
            
            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

        def format_column_type(column_type):
            """Format column type to be more readable in the ERD"""
            type_str = str(column_type).lower()
            
            # Map SQLAlchemy types to simpler display types
            if 'varchar' in type_str or 'character varying' in type_str:
                return 'string'
            elif 'text' in type_str:
                return 'string'
            elif 'int' in type_str:
                if 'bigint' in type_str:
                    return 'bigint'
                return 'integer'
            elif 'bool' in type_str:
                return 'boolean'
            elif 'datetime' in type_str or 'timestamp' in type_str:
                return 'timestamp'
            elif 'double' in type_str or 'float' in type_str:
                return 'double'
            elif 'uuid' in type_str:
                return 'uuid'
            
            # Default case
            return type_str

        def generate_erd(db_url, output_path=None, exclude_tables=None):
            """Generate ERD diagram using SQLAlchemy metadata reflection"""
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = Path('docs/erds')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"database_erd_{timestamp}.md"
            
            exclude_tables = exclude_tables or ['alembic_version']
            
            logger.info("Connecting to database...")
            engine = create_engine(db_url)
            
            try:
                # Reflect database structure
                logger.info("Reflecting database structure...")
                metadata = MetaData()
                metadata.reflect(bind=engine)
                inspector = inspect(engine)
                
                # Filter out excluded tables
                tables = [t for t in metadata.tables.values() 
                        if t.name not in exclude_tables]
                
                # Start generating Mermaid diagram
                mermaid_lines = ["# Database ERD Diagram", "", 
                                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
                                "```mermaid", "erDiagram"]
                
                # Add tables with columns
                for table in tables:
                    table_name = table.name
                    mermaid_lines.append(f"    {table_name} {{")
                    
                    # Determine primary key columns
                    pk_columns = []
                    try:
                        # First, try to get primary key from inspector
                        pk_constraint = inspector.get_pk_constraint(table_name)
                        
                        # Handle different possible return types of pk_constraint
                        if isinstance(pk_constraint, dict):
                            pk_columns = pk_constraint.get('constrained_columns', [])
                        elif isinstance(pk_constraint, list):
                            pk_columns = pk_constraint
                    except Exception as e:
                        logger.warning(f"Could not retrieve primary key for {table_name} via inspector: {e}")
                    
                    # If no primary keys found, try to find UUID column
                    if not pk_columns:
                        pk_columns = [
                            col.name for col in table.columns 
                            if isinstance(col.type, UUID)
                        ]
                    
                    # Add columns
                    for column in table.columns:
                        col_name = column.name
                        col_type = format_column_type(column.type)
                        
                        # Format the column entry
                        if col_name in pk_columns:
                            mermaid_lines.append(f"        {col_type} {col_name} PK")
                        else:
                            mermaid_lines.append(f"        {col_type} {col_name}")
                    
                    mermaid_lines.append("    }")
                
                # Add relationships
                for table in tables:
                    table_name = table.name
                    for fk in table.foreign_keys:
                        target_table = fk.column.table.name
                        if target_table in exclude_tables:
                            continue
                        
                        source_col = fk.parent.name
                        target_col = fk.column.name
                        
                        # Create relationship with the correct format
                        rel_line = f"    {table_name} }}o--|| {target_table} : \\"FK {source_col} -> {target_col}\\""
                        mermaid_lines.append(rel_line)
                
                # Write to file
                logger.info(f"Writing ERD to {output_path}")
                with open(output_path, 'w') as f:
                    f.write("\\n".join(mermaid_lines))
                
                logger.info("ERD generation completed successfully!")
                return output_path
            
            except Exception as e:
                logger.error(f"Error generating ERD: {str(e)}")
                # Log the full traceback for debugging
                logger.error(traceback.format_exc())
                raise
            finally:
                engine.dispose()

        if __name__ == "__main__":
            parser = argparse.ArgumentParser(description='Generate an ERD diagram from a PostgreSQL database')
            parser.add_argument('--output', '-o', type=str, help='Output file path')
            parser.add_argument('--exclude', '-e', type=str, nargs='+', default=['alembic_version'],
                                help='Tables to exclude from the ERD')
            args = parser.parse_args()
            
            try:
                db_url = get_db_url_from_env()
                output_path = generate_erd(db_url, args.output, args.exclude)
                print(f"ERD diagram saved to: {output_path}")
            except Exception as e:
                print(f"Failed to generate ERD: {str(e)}")
                sys.exit(1)
        '''
    
        scripts_dir = self.project_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        with open(scripts_dir / "generate_erd.py", "w") as f:
            f.write(erd_script_content)
        
        logger.info("ERD generation functionality added successfully")

    def setup(self) -> None:
        """Run complete setup process"""
        try:
            logger.info(f"Starting project setup in: {self.project_dir}")
            
            # Create project structure
            self.create_project_structure()
            
            # Create virtual environment
            self.create_virtual_environment()
            
            # Install dependencies
            self.install_dependencies()
            
            # Create project files
            self.create_files()

            # Add ERD generation functionality
            self.add_erd_generation_function()
            
            # Initialize Alembic
            self.initialize_alembic()
            
            # Create database
            self.create_database()

            # Copy CSV files if not skipped
            self.copy_csv_files()
            
            # Import data if not skipped
            if not self.skip_import:
                # Check if we have CSV files before attempting import
                data_dir = self.project_dir / "data" / "import"
                has_csv_files = list(data_dir.glob("*.csv")) if data_dir.exists() else []
                
                if has_csv_files:
                    self.import_csv_data()
                else:
                    logger.warning("No CSV files found in data directory. Skipping import.")
            
            # Initialize Git
            self.initialize_git()
            
            logger.info("Project setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Activate virtual environment:")
            logger.info(f"   source {os.path.basename(self.venv_path)}/bin/activate  # Unix")
            logger.info(f"   .\\{os.path.basename(self.venv_path)}\\Scripts\\activate  # Windows")
            logger.info("2. Review and edit .env file with your database credentials")
            logger.info("3. Run initial migration:")
            logger.info("   alembic revision --autogenerate -m 'Initial migration'")
            logger.info("   Review, and if necessary edit, the generated migration file in migrations/versions")
            logger.info("   alembic upgrade head")
            
        except Exception as e:
            logger.error(f"Setup failed: {str(e)}")
            raise

def main():
    """Main function to run setup"""
    parser = argparse.ArgumentParser(description='Setup a new project with specified database name')
    parser.add_argument('--directory', '-d', type=str, default=os.getcwd(),
                       help='Project directory (default: current directory)')
    
    parser.add_argument('--database_name', '-db', type=str, default='mydb',
                       help='Name of the database to create (default: mydb)')
    
    parser.add_argument('--port', '-p', type=str, default='5432',
                       help='PostgreSQL port number (default: 5432)')
    
    parser.add_argument('--db_user', '-u', type=str, default='postgres',
                       help='PostgreSQL username (default: postgres)')
    
    parser.add_argument('--db_password', '-pw', type=str, default='postgres',
                       help='PostgreSQL password (default: postgres)')
    
    parser.add_argument('--source_dir', '-s', type=str, default='../csvs/',
                       help='Directory containing CSV files to copy to project data directory (default: ../csvs/)') # update parse arg with correct directory
    
    parser.add_argument('--skip_copy', '-sc', action='store_true',
                       help='Skip copying CSV files (default: False)')
    
    parser.add_argument('--skip_import', '-si', action='store_true',
                       help='Skip the data import step (default: False)')
    
    args = parser.parse_args()
    setup = ProjectSetup(args.directory, args.database_name, args.port, args.source_dir, 
                         args.db_user, args.db_password, args.skip_copy, args.skip_import)
    setup.setup()

if __name__ == "__main__":
    main()