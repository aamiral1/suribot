import psycopg2 as pg2
from psycopg2.pool import ThreadedConnectionPool
import uuid
from enums import DocumentStatus
import custom_exceptions as ex


class Database:
    def __init__(self, database_path, doc_table_name):
        self.db_path = database_path
        self.doc_table = doc_table_name

        # Create database connection pool
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host="localhost",
            dbname="postgres",
            user="postgres",
            password="9999",
            port=5432
        )

        # pool.getconn()
        # pool.putconn(conn_name)

    def init_schema(self):
        connection = self._get_conn()
        cursor = connection.cursor()

        try:
            # create table with appropriate columns
            create_table_command = f"""
                DO $$ 
                BEGIN 
                    CREATE TYPE format_type AS ENUM('PDF', 'DOCX', 'TXT', 'MD', 'PNG'); 
                EXCEPTION 
                    WHEN duplicate_object THEN null; 
                END $$;

                DO $$ 
                BEGIN 
                    CREATE TYPE extraction_status AS ENUM(
                        'created', 
                        'processing', 
                        'success',
                        'failed'
                    );
                EXCEPTION 
                    WHEN duplicate_object THEN null; 
                END $$;

                DO $$
                BEGIN
                    CREATE TYPE source_format AS ENUM(
                        'upload',
                        'crawl'
                    );
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;

                DO $$
                BEGIN
                    CREATE TYPE kb_status_type AS ENUM(
                        'none',
                        'processing',
                        'success',
                        'failed'
                    );
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;

                CREATE TABLE IF NOT EXISTS {self.doc_table} (
                    doc_id TEXT PRIMARY KEY,
                    source_type source_format NOT NULL,
                    file_name TEXT,
                    file_size TEXT NOT NULL,
                    file_type format_type NOT NULL,
                    uploaded_date DATE NOT NULL,
                    s3_file_bucket TEXT NOT NULL,
                    s3_file_key TEXT NOT NULL,
                    status extraction_status NOT NULL,
                    s3_extracted_text_bucket TEXT,
                    s3_extracted_text_key TEXT,
                    error_msg TEXT,
                    in_kb BOOLEAN DEFAULT FALSE
                );

                ALTER TABLE {self.doc_table} ADD COLUMN IF NOT EXISTS in_kb BOOLEAN DEFAULT FALSE;
                ALTER TABLE {self.doc_table} ADD COLUMN IF NOT EXISTS kb_status kb_status_type NOT NULL DEFAULT 'none';
                ALTER TABLE {self.doc_table} ADD COLUMN IF NOT EXISTS doc_type TEXT NOT NULL DEFAULT 'knowledge_base';
                ALTER TABLE {self.doc_table} ADD COLUMN IF NOT EXISTS doc_structure TEXT NOT NULL DEFAULT 'free_flow';

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id       SERIAL PRIMARY KEY,
                    doc_id   TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    text     TEXT NOT NULL,
                    heading  TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_doc_chunks_doc_id
                    ON document_chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_doc_chunks_chunk
                    ON document_chunks(doc_id, chunk_id);
            """

            cursor.execute(create_table_command)

            connection.commit()
        except Exception as e:
            connection.rollback()
            print(e)
            raise
        finally:
            cursor.close()
            self._put_conn(connection)

# creates record for a document with CREATED as initial value
    def create(self, source_type, name, size, type, upload_date, s3_file_bucket, s3_file_key, s3_extracted_text_bucket, s3_extracted_text_key, doc_type="knowledge_base", doc_structure="free_flow"):
        connection = self._get_conn()
        cursor = connection.cursor()

        doc_id = str(uuid.uuid4())

        command = f"""INSERT INTO {self.doc_table}
            (doc_id, source_type, file_name, file_size, file_type, uploaded_date,
             s3_file_bucket, s3_file_key, status, s3_extracted_text_bucket,
             s3_extracted_text_key, error_msg, doc_type, doc_structure)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

        try:
            cursor.execute(
                command,
                (
                    doc_id,
                    source_type,
                    name, size,
                    type,
                    upload_date,
                    s3_file_bucket,
                    s3_file_key,
                    DocumentStatus.CREATED.value,
                    s3_extracted_text_bucket,
                    s3_extracted_text_key,
                    None,
                    doc_type,
                    doc_structure,
                ),
            )
            connection.commit()
        except pg2.IntegrityError:
            connection.rollback()
            # return doc id of existing file
            raise Exception("File record already exists")
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

        return doc_id
    
    # return s3 file bucket and key for a given doc id
    def get_file_path(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT s3_file_bucket, s3_file_key FROM {self.doc_table} WHERE doc_id = %s"

        try:
            cursor.execute(command, (doc_id,))
            row = cursor.fetchone()

            if not row:
                err = "No document exists for the given document id"
                raise Exception(err)

            s3_file_bucket = row[0]
            s3_file_key = row[1]

        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

        return [s3_file_bucket, s3_file_key]

    # retrieves extraction status for a given doc id
    def get_status(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT status FROM {self.doc_table} WHERE doc_id = %s"

        try:
            cursor.execute(command, (doc_id,))
            row = cursor.fetchone()

            if row is None:
                raise ex.InvalidDocumentID("No document available")

            status = DocumentStatus(row[0])

        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")

        finally:
            cursor.close()
            self._put_conn(connection)

        return status

    # changes status for a given doc_id according to state machine diagram
    def transition_status(self, doc_id, new_status: DocumentStatus):
        connection = self._get_conn()
        cursor = connection.cursor()

        get_status_command = f"SELECT status FROM {self.doc_table} WHERE doc_id = %s"

        update_status_command = f"UPDATE {self.doc_table} SET status=%s WHERE doc_id=%s"

        # update database with new status
        try:
            # check if current status and new status are valid
            cursor.execute(get_status_command, (doc_id,))
            row = cursor.fetchone()
            if not row:
                raise Exception("Document does not exist")

            curr_status = DocumentStatus(row[0])

            # state machine
            if (
                curr_status == DocumentStatus.CREATED
                and new_status == DocumentStatus.PROCESSING
            ):
                pass
            elif curr_status == DocumentStatus.PROCESSING and (
                new_status == DocumentStatus.SUCCESS
            ):
                pass
            elif (
                curr_status == DocumentStatus.FAILED
                and new_status == DocumentStatus.PROCESSING
            ):
                pass
            elif new_status == DocumentStatus.FAILED:
                pass
            else:
                raise ex.InvalidDocumentStatusTransition("Invalid status transition.")

            # update document status
            cursor.execute(update_status_command, (new_status.value, doc_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # sets S3 paths of extracted text for a given doc id
    def set_extraction_text_path(self, doc_id, s3_extracted_text_bucket, s3_extracted_text_key):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"UPDATE {self.doc_table} SET s3_extracted_text_bucket=%s, s3_extracted_text_key=%s WHERE doc_id=%s"

        try:
            cursor.execute(command, (s3_extracted_text_bucket, s3_extracted_text_key, doc_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # retrieves S3 extracted text paths for a given doc id
    def get_extracted_text_file_path(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT s3_extracted_text_bucket, s3_extracted_text_key from {self.doc_table} WHERE doc_id=%s"

        try:
            cursor.execute(command, (doc_id,))
            row = cursor.fetchone()

            if not row:
                raise ex.InvalidDocumentID("No document available")

            s3_bucket = row[0]
            s3_key = row[1]

        except Exception as e:
            connection.rollback()
            raise

        finally:
            cursor.close()
            self._put_conn(connection)

        return [s3_bucket, s3_key]

    # returns all rows from the documents table
    def get_all_documents(self):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT * FROM {self.doc_table}"

        try:
            cursor.execute(command)
            rows = cursor.fetchall()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

        return rows

    # marks a document as added to the knowledge base
    def set_in_kb(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"UPDATE {self.doc_table} SET in_kb = TRUE WHERE doc_id = %s"

        try:
            cursor.execute(command, (doc_id,))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # retrieves kb_status for a given doc id
    def get_kb_status(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT kb_status FROM {self.doc_table} WHERE doc_id = %s"

        try:
            cursor.execute(command, (doc_id,))
            row = cursor.fetchone()

            if not row:
                raise ex.InvalidDocumentID("No document available")

            return row[0]
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # sets kb_status for a given doc id
    def set_kb_status(self, doc_id, status: str):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"UPDATE {self.doc_table} SET kb_status = %s WHERE doc_id = %s"

        try:
            cursor.execute(command, (status, doc_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # returns doc_type for a given doc_id
    def get_doc_type(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT doc_type FROM {self.doc_table} WHERE doc_id = %s"

        try:
            cursor.execute(command, (doc_id,))
            row = cursor.fetchone()
            if not row:
                raise ex.InvalidDocumentID("No document available")
            return row[0]
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # returns (s3_extracted_text_bucket, s3_extracted_text_key) for all system prompt docs in KB
    def get_system_prompt_docs(self):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"""SELECT s3_extracted_text_bucket, s3_extracted_text_key
                      FROM {self.doc_table}
                      WHERE doc_type = 'system_prompt' AND in_kb = TRUE"""

        try:
            cursor.execute(command)
            return cursor.fetchall()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # returns doc_structure ('free_flow' or 'structured') for a given doc_id
    def get_doc_structure(self, doc_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"SELECT doc_structure FROM {self.doc_table} WHERE doc_id = %s"

        try:
            cursor.execute(command, (doc_id,))
            row = cursor.fetchone()
            if not row:
                raise ex.InvalidDocumentID("No document available")
            return row[0]
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # bulk inserts chunk rows into document_chunks
    def insert_chunks(self, doc_id, rows):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = """INSERT INTO document_chunks
            (doc_id, chunk_id, text, heading)
            VALUES (%s, %s, %s, %s)"""

        try:
            cursor.executemany(
                command,
                [
                    (
                        doc_id,
                        row["chunk_id"],
                        row["text"],
                        row.get("heading"),
                    )
                    for row in rows
                ],
            )
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # returns all chunks across all documents (used to rebuild BM25 encoder)
    def get_all_chunks(self):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = """SELECT doc_id, chunk_id, text FROM document_chunks"""

        try:
            cursor.execute(command)
            rows = cursor.fetchall()
            return [
                {
                    "doc_id": row[0],
                    "chunk_id": row[1],
                    "text": row[2],
                }
                for row in rows
            ]
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)
            

    # private helper functions
    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)