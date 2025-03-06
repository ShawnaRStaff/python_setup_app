# Definition Scripts

This directory contains script modules that define the structure of the database models and data import processes. These scripts are used by the setup process to initialize your project. And can be omitted.

## Contents

### `MODEL_DEFINITIONS_SCRIPT`

This dictionary contains the file paths and content for all model-related files. During setup, these files are created in your project directory according to these definitions, including:

- Base model class definitions
- User model
- UserPost model
- Any other models defined for your application

The model definitions include:

- Table structures
- Column definitions
- Relationships between models
- Indexes and constraints
- SQLAlchemy configuration

### `DATA_IMPORT_DEFINITION_SCRIPT`

This dictionary contains scripts for importing data into the models from CSV files. These scripts are responsible for:

- Reading CSV data from the data directory
- Mapping CSV data to your model structures
- Handling data transformations and cleaning
- Inserting the data into your database
- Managing import validation and error handling

## Usage

These definition scripts are imported by `setup.py` and used during project initialization. They allow for a consistent project structure and database setup without requiring manual file creation.

### How It Works

1. When `setup.py` runs, it imports these definition scripts
2. It creates files based on the paths and content defined in these dictionaries
3. The resulting files define your database models and import processes

## Customization

To customize your project structure:

1. Modify the `MODEL_DEFINITIONS_SCRIPT` dictionary to add, remove, or change model definitions
2. Modify the `DATA_IMPORT_DEFINITION_SCRIPT` dictionary to adjust how data is imported

Each entry in these dictionaries is a key-value pair where:

- The key is the file path relative to your project root
- The value is the file content as a string

## Notes

- The example models in `MODEL_DEFINITIONS_SCRIPT` include User and UserPost models
- The integration tests in the test directory are designed to validate these example models
- When adding new models, be sure to update the model imports in `__init__.py`
