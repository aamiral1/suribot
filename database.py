import sqlite3
import uuid

class Database:
    def __init__(self, database_path, doc_table_name):
        self.db_path = database_path
        self.doc_table = doc_table_name

    def init_schema(self):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        create_table_command = f"""
            CREATE TABLE IF NOT EXISTS 
            {self.doc_table} (doc_id TEXT PRIMARY_KEY, file_path TEXT NOT NULL UNIQUE)
        """

        cursor.execute(create_table_command)

        connection.commit()
        connection.close()
    
    def create(self, file_path):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        doc_id = str(uuid.uuid4())

        command = f"INSERT INTO {self.doc_table} VALUES (?, ?)"
        try:
            cursor.execute(command, (doc_id, file_path))
            connection.commit()
        except sqlite3.IntegrityError:
            raise Exception("File record already exists")
        except Exception as e:
            raise Exception(f"Error: {e}")
        finally:
            connection.close()
        
        return doc_id
        
    def get_path(self, doc_id):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        command = f"SELECT * FROM {self.doc_table} WHERE doc_id = ?"

        rows = cursor.execute(command, (doc_id, )).fetchone()

        if not rows:
            err = "No document exists for the given document id"
            raise Exception(err)
        
        file_path = rows[1]

        connection.commit()
        connection.close()

        return file_path
    