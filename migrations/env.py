from logging.config import fileConfig

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
                    migration_name = migration.doc.split('\n')[0].strip('"').strip()
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
                        migration_name = rev.doc.split('\n')[0].strip('"').strip()
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
    run_migrations_online()