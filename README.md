# Database ORM Project

A comprehensive project template for building SQLAlchemy-based applications with FastAPI, JWT authentication, and automated setup processes.

## Project Overview

This project provides a complete scaffold for developing database-driven applications with the following features:

- **SQL Database Layer**: PostgreSQL support with SQLAlchemy ORM
- **API Framework**: FastAPI integration ready
- **Authentication**: JWT token infrastructure with bcrypt and Python-Jose
- **Migration Management**: Alembic for version-controlled database changes
- **Visualization**: Automatic ERD generation for database structure visualization
- **Data Import**: Streamlined CSV import system for populating databases
- **Testing**: Built-in pytest infrastructure with SQLite for testing

## Getting Started

### Prerequisites

- Python 3.8+
- PostgreSQL database server
- Git (optional, for version control)

### Setup

1. Clone this repository:

   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. Run the setup script with your desired configuration:

   ```bash
   python setup.py --database_name my_database --db_user postgres --db_password secretpassword
   ```

   - Running will set args to default values ```bash
     python setup.py

   ```

   ```

3. Activate the virtual environment:

   ```bash
   # Unix/macOS
   source .<project-name>-venv/bin/activate

   # Windows
   .\.<project-name>-venv\Scripts\activate
   ```

4. Run initial migration:
   ```bash
   alembic revision --autogenerate -m "Initial migration"
   alembic upgrade head
   ```

### Setup Script Options

The `setup.py` script accepts the following arguments:

| Argument          | Short | Default           | Description             |
| ----------------- | ----- | ----------------- | ----------------------- |
| `--directory`     | `-d`  | Current directory | Project directory       |
| `--database_name` | `-db` | `mydb`            | Database name           |
| `--port`          | `-p`  | `5432`            | PostgreSQL port         |
| `--db_user`       | `-u`  | `postgres`        | Database username       |
| `--db_password`   | `-pw` | `postgres`        | Database password       |
| `--source_dir`    | `-s`  | `../csvs/`        | Directory for CSV files |
| `--skip_copy`     | `-sc` | `False`           | Skip copying CSV files  |
| `--skip_import`   | `-si` | `False`           | Skip data import        |

Example:

```bash
python setup.py -db user_service -u dbadmin -pw secure123 -s /path/to/csvs -p 5433
```

## Project Structure

```
project/
├── .env                      # Environment variables
├── alembic.ini               # Alembic configuration
├── data/                     # Data directory
│   ├── db.py                 # Database connection setup
│   └── import/               # CSV files for import
├── migrations/               # Alembic migrations
│   ├── env.py                # Migration environment
│   └── versions/             # Migration versions
├── models/                   # SQLAlchemy models
│   ├── __init__.py           # Model imports
│   ├── base.py               # Base model class
│   ├── user.py               # User model
│   └── user_post.py          # UserPost model
├── scripts/                  # Utility scripts
│   ├── data_import.py        # CSV import script
│   └── generate_erd.py       # ERD generation script
└── tests/                    # Test directory
    ├── integration_tests/    # Integration tests
    └── unit_tests/           # Unit tests
```

## Model Definitions

The project includes example model definitions that demonstrate how to create SQLAlchemy models with proper relationships:

### User Model

The `User` model demonstrates:

- UUID primary keys
- Unique constraints
- Password fields (for authentication)
- Relationship definition with cascade deletion
- Timestamp tracking

```python
class User(Base):
    __tablename__ = "user"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    username = Column(String(255), nullable=False, unique=True)
    password = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, unique=True)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationship with UserPost
    posts = relationship("UserPost", back_populates="user", cascade="all, delete-orphan")
```

### UserPost Model

The `UserPost` model demonstrates:

- Foreign key relationships
- One-to-many relationships
- Soft deletion via is_active flag
- Bidirectional relationship with User

```python
class UserPost(Base):
    __tablename__ = "user_post"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("user.id"), nullable=False)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    # Relationship with User
    user = relationship("User", back_populates="posts")
```

## Authentication Setup

The project includes all dependencies for implementing JWT-based authentication:

- **bcrypt**: For password hashing and verification
- **python-jose**: For JWT token creation and validation
- **passlib**: For password hashing utilities
- **python-multipart**: For form data handling

The model structure is already designed to support authentication with:

- User model with username, password and email fields
- Active/inactive flag for account disabling
- Timestamp tracking for security auditing

## Data Import System

The project includes a robust data import system in `scripts/data_import.py` that can:

1. Read CSV files from a specified directory
2. Map CSV columns to model attributes
3. Handle different timestamp formats
4. Clean and validate data
5. Import data in the correct order to respect foreign key constraints
6. Provide progress bars for large imports

Example usage:

```bash
python scripts/data_import.py --csv_dir data/import --truncate
```

The import system is customizable through the mapping definitions in `column_mappings` and file encodings in `FILE_ENCODINGS`.

## Database ERD Generation

The project can automatically generate Entity Relationship Diagrams (ERDs) using Mermaid:

```bash
python scripts/generate_erd.py --output docs/erds/my_diagram.md
```

ERDs are automatically generated after migrations to document database changes.

## Testing

The project includes a comprehensive testing framework with:

- SQLite-based test database for speed and isolation
- Pytest fixtures for database setup and teardown
- Integration tests for models and relationships
- Separate test database from the production database

See the [tests/README.md](tests/README.md) for detailed information on the testing approach.

## Extending the Project

### Adding New Models

1. Create a new model file in the `models/` directory:

   ```python
   # models/new_model.py
   from sqlalchemy import Column, String, ForeignKey
   from sqlalchemy.dialects.postgresql import UUID
   from sqlalchemy.orm import relationship
   import uuid
   from .base import Base

   class NewModel(Base):
       __tablename__ = "new_model"

       id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
       name = Column(String(255), nullable=False)
       # Add more columns...
   ```

2. Import the model in `models/__init__.py`:

   ```python
   from .base import Base
   from .user import User
   from .user_post import UserPost
   from .new_model import NewModel  # Add this line
   ```

3. Generate and run a migration:
   ```bash
   alembic revision --autogenerate -m "Add new model"
   alembic upgrade head
   ```

### Adding CSV Import Support for New Models

Update the `import_order` and `model_map` in `scripts/data_import.py`:

```python
# Define import order based on model dependencies
import_order = [
    # First level - no foreign key dependencies
    ['user'],

    # Second level - depends on first level
    ['user_post', 'new_model']  # Add your new model
]

# Map CSV file names to model classes
model_map = {
    'user': User,
    'user_post': UserPost,
    'new_model': NewModel  # Add your new model
}
```

### Configuring Custom Environment Variables

Update the `.env` file with additional environment variables:

```
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_NAME=mydb
DB_PORT=5432
ENVIRONMENT=development
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Conclusion

This project provides a solid foundation for developing database-driven applications with proper ORM modeling, authentication, and testing. By following the established patterns, you can quickly extend the models and functionality while maintaining good software engineering practices.

## Contributing

Contributions are welcome! This is an open-source project, and we appreciate any help in making it better.

### How to Contribute

1. **Fork the repository**: Start by forking the repository to your GitHub account.

2. **Clone the forked repository**:

   ```bash
   git clone https://github.com/your-username/project-name.git
   cd project-name
   ```

3. **Create a new branch**:

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes**: Implement your feature or bug fix.

5. **Run tests**: Make sure your changes don't break existing functionality.

   ```bash
   pytest
   ```

6. **Commit your changes**:

   ```bash
   git commit -m "Add your meaningful commit message here"
   ```

7. **Push to your fork**:

   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**: Open a pull request from your forked repository to the main repository.

### Contribution Guidelines

- Follow the existing code style and conventions
- Write or update tests for the changes you make
- Update documentation as needed
- Keep pull requests focused on a single topic
- Be respectful and constructive in discussions

## Contact

For questions, suggestions, or collaboration opportunities, please contact:

- Email: shawnastaff@gmail.com
- GitHub: [GitHubUsername](https://github.com/shawnarstaff)
- LinkedIn: [LinkedIn Profile](https://www.linkedin.com/in/shawnastaff/)

## License

MIT License

Copyright (c) 2025 Shawna R. Staff

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
