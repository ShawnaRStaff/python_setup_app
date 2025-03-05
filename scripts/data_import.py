import os
import sys
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import argparse
from tqdm import tqdm

# Add the project root directory to the Python path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

# Import models
from models import User, UserPost
from db import Base, engine, SessionLocal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_import.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define file-specific encodings
FILE_ENCODINGS = {
    'default': 'utf-8'
}

def parse_timestamp(ts_str):
    """Convert string timestamp to datetime object."""
    if pd.isna(ts_str) or ts_str == '' or ts_str.lower() in ['null', 'none']:
        return None
    try:
        # Add more formats here if you encounter different timestamp formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%d/%m/%Y %H:%M:%S'  # Added common alternative format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse timestamp: {ts_str}")
        return None
    except Exception as e:
        logger.error(f"Error parsing timestamp {ts_str}: {str(e)}")
        return None

def clean_dataframe(df):
    """Clean dataframe by converting empty strings to None and handling timestamps."""
    # Remove any unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Convert timestamps if they exist
    timestamp_fields = ['created_at', 'updated_at']
    for field in timestamp_fields:
        if field in df.columns:
            df[field] = df[field].apply(parse_timestamp)
    
    # Convert empty strings and 'null' values to None
    df = df.replace(['null', 'NULL', ''], None)
    
    return df

def import_data(csv_dir=None, truncate=False):
    """
    Import CSV files into database in correct dependency order.
    
    Args:
        csv_dir (str, optional): Directory containing CSV files to import.
        truncate (bool, optional): Whether to truncate tables before importing.
    """
    # Get database configuration from environment
    db_user = os.getenv('DB_USER', 'postgres')
    db_pass = os.getenv('DB_PASSWORD', 'postgres')
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'mydb')
    
    logger.info(f"Using database: {db_name} on {db_host}:{db_port} (user: {db_user})")
    
    # Use the existing engine if available, otherwise create a new one
    try:
        Session = sessionmaker(bind=engine)
    except NameError:
        # Create database connection if not already available
        db_url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(db_url)
        Session = sessionmaker(bind=engine)
        
    session = Session()

    # Define the data directory
    if csv_dir:
        data_dir = Path(csv_dir)
    else:
        data_dir = Path(project_root) / "data" / "import"
    
    logger.info(f"Looking for CSV files in: {data_dir}")

    # Define import order based on model dependencies
    # In this example, User must be imported before UserPost due to foreign key constraints
    import_order = [
        # First level - no foreign key dependencies
        ['user'],
        
        # Second level - depends on first level
        ['user_post']
    ]
    
    # Map CSV file names to model classes
    model_map = {
        'user': User,
        'user_post': UserPost
    }
    
    # Map CSV columns to model attributes, with any necessary transformations
    column_mappings = {
        'user': {
            'id': 'id',
            'name': 'name',
            'username': 'username',
            'password': 'password',
            'email': 'email',
            'is_active': 'is_active',
            'created_at': 'created_at',
            'updated_at': 'updated_at'
        },
        'user_post': {
            'id': 'id',
            'user_id': 'user_id',
            'title': 'title',
            'content': 'content',
            'is_active': 'is_active',
            'created_at': 'created_at',
            'updated_at': 'updated_at'
        }
    }

    try:
        # Optionally truncate tables before import
        if truncate:
            logger.info("Truncating tables before import...")
            # Truncate in reverse order to respect foreign key constraints
            for level in reversed(import_order):
                for table in level:
                    logger.info(f"Truncating table: {table}")
                    try:
                        session.execute(f'TRUNCATE TABLE "{table}" CASCADE')
                        session.commit()
                    except Exception as e:
                        session.rollback()
                        logger.error(f"Error truncating {table}: {str(e)}")
        
        # Import each level in order
        for level in import_order:
            for table in level:
                csv_file = data_dir / f"{table}.csv"
                if not csv_file.exists():
                    logger.warning(f"Missing CSV file: {csv_file}")
                    continue
                    
                logger.info(f"Importing {table} from {csv_file}")
                
                try:
                    # Get the correct encoding for the file
                    encoding = FILE_ENCODINGS.get(f"{table}.csv", FILE_ENCODINGS['default'])
                    logger.info(f"Using {encoding} encoding for {table}.csv")
                    
                    # Read the CSV file
                    df = pd.read_csv(
                        csv_file, 
                        keep_default_na=False, 
                        na_values=[''],
                        encoding=encoding
                    )
                    
                    # Clean the dataframe
                    df = clean_dataframe(df)
                    
                    # Get the model class for this table
                    model_class = model_map.get(table)
                    if not model_class:
                        logger.warning(f"No model class defined for {table}, skipping")
                        continue
                    
                    # Get column mappings for this table
                    mapping = column_mappings.get(table, {})
                    
                    # Convert boolean columns if needed
                    if 'is_active' in df.columns:
                        df['is_active'] = df['is_active'].astype(bool)
                    
                    # Insert in chunks to handle large files
                    chunk_size = 1000
                    total_chunks = (len(df) + chunk_size - 1) // chunk_size
                    
                    with tqdm(total=total_chunks, desc=f"Importing {table}") as pbar:
                        for i in range(0, len(df), chunk_size):
                            chunk = df.iloc[i:i + chunk_size]
                            
                            # Option 1: Using SQLAlchemy ORM
                            # More flexible but slower for large datasets
                            objects = []
                            for _, row in chunk.iterrows():
                                # Create model instance with mapped columns
                                obj_data = {}
                                for csv_col, model_attr in mapping.items():
                                    if csv_col in row:
                                        obj_data[model_attr] = row[csv_col]
                                
                                obj = model_class(**obj_data)
                                objects.append(obj)
                            
                            session.bulk_save_objects(objects)
                            
                            session.commit()
                            pbar.update(1)
                    
                    logger.info(f"Successfully imported {len(df)} rows to {table}")
                
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error importing {table}: {str(e)}")
                    raise

        logger.info("Data import completed successfully!")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Import failed: {str(e)}")
        raise
    finally:
        session.close()
        if 'engine' in locals():
            engine.dispose()

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description='Import CSV data into the database')
    parser.add_argument('--csv_dir', '-d', type=str, 
                        help='Directory containing CSV files (default: project_root/data/import)')
    parser.add_argument('--truncate', '-t', action='store_true',
                        help='Truncate tables before importing (default: False)')
    args = parser.parse_args()
    
    try:
        import_data(args.csv_dir, args.truncate)
    except Exception as e:
        logger.error(f"Data import failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
