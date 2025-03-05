#!/usr/bin/env python3
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
                        rel_line = f"    {table_name} }}o--|| {target_table} : \"FK {source_col} -> {target_col}\""
                        mermaid_lines.append(rel_line)
                
                # Write to file
                logger.info(f"Writing ERD to {output_path}")
                with open(output_path, 'w') as f:
                    f.write("\n".join(mermaid_lines))
                
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
        