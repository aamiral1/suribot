import psycopg2 as pg2
from psycopg2.pool import ThreadedConnectionPool
import uuid
from enums import DocumentStatus
import exceptions as ex


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
    def create(self, source_type, name, size, type, upload_date, s3_file_bucket, s3_file_key, s3_extracted_text_bucket, s3_extracted_text_key):
        connection = self._get_conn()
        cursor = connection.cursor()

        doc_id = str(uuid.uuid4())

        command = f"""INSERT INTO {self.doc_table} VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""

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
                    None),
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

    # private helper functions
    def _get_conn(self):
        return self.pool.getconn()
    
    def _put_conn(self, conn):
        self.pool.putconn(conn)