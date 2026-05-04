import json

import psycopg2 as pg2
from psycopg2.pool import ThreadedConnectionPool
import uuid
from enums import DocumentStatus
import custom_exceptions as ex
from sales_controller import SalesController

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

                CREATE TABLE IF NOT EXISTS webpages (
                    url_id         TEXT PRIMARY KEY,
                    url            TEXT NOT NULL,
                    domain         TEXT,
                    crawled_date   DATE NOT NULL,
                    crawl_status   extraction_status NOT NULL,
                    kb_status      kb_status_type NOT NULL DEFAULT 'none',
                    in_kb          BOOLEAN DEFAULT FALSE,
                    error_msg      TEXT,
                    s3_text_bucket TEXT,
                    s3_text_key    TEXT
                );

                CREATE TABLE IF NOT EXISTS webpage_chunks (
                    id       SERIAL PRIMARY KEY,
                    url_id   TEXT NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    text     TEXT NOT NULL,
                    heading  TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_webpage_chunks_url_id
                    ON webpage_chunks(url_id);
                CREATE INDEX IF NOT EXISTS idx_webpage_chunks_chunk
                    ON webpage_chunks(url_id, chunk_id);

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

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    message TEXT NOT NULL,
                    used_action TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS lead_profiles (
                    session_id TEXT PRIMARY KEY,
                    profile JSONB NOT NULL DEFAULT '{{}}',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS bookings (
                    booking_id  TEXT PRIMARY KEY,
                    session_id  TEXT NOT NULL,
                    event_id    TEXT NOT NULL,
                    slot_iso    TEXT NOT NULL,
                    name        TEXT,
                    email       TEXT,
                    status      TEXT NOT NULL DEFAULT 'confirmed',
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_bookings_session
                    ON bookings(session_id);

                CREATE INDEX IF NOT EXISTS idx_bookings_email
                    ON bookings(email);

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

# records a message interaction
    def save_message(self, session_id, role, content, used_action=None):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = f"""
        INSERT INTO conversation_messages (session_id, role, message, used_action)
        VALUES (%s, %s, %s, %s)
        """

        try:
            cursor.execute(
                command,
                (
                    session_id,
                    role,
                    content,
                    used_action
                )
            )

            connection.commit()

        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

# get conversation history for a particular session_id
    def get_conversation_history(self, session_id, limit=10):
        connection = self._get_conn()
        cursor = connection.cursor()

        query = """
        SELECT role, message
        FROM conversation_messages
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """

        rows = []  

        try:
            cursor.execute(query, (session_id, limit))
            rows = cursor.fetchall()

        except Exception as e:
            raise Exception(f"Error fetching conversation history: {e}")

        finally:
            cursor.close()
            self._put_conn(connection)

        # reverse so oldest → newest
        return [
            {"role": row[0], "content": row[1]}
            for row in reversed(rows)
        ]

# returns total number of assistant turns for a session (used for sales_turn threshold)
    def get_sales_turn_count(self, session_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        query = """
        SELECT COUNT(*) FROM conversation_messages
        WHERE session_id = %s AND role = 'assistant'
        """

        try:
            cursor.execute(query, (session_id,))
            row = cursor.fetchone()
            return row[0] if row else 0
        except Exception as e:
            raise Exception(f"Error fetching sales turn count: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

# gets last used actions for a particular session_id
    def get_last_actions(self, session_id, limit=3):
        connection = self._get_conn()
        cursor = connection.cursor()

        query = """
        SELECT used_action
        FROM conversation_messages
        WHERE session_id = %s
        AND role = 'assistant'
        AND used_action IS NOT NULL
        ORDER BY created_at DESC
        LIMIT %s
        """

        try:
            cursor.execute(query, (session_id, limit))
            rows = cursor.fetchall()
        except Exception as e:
            raise Exception(f"Error fetching used actions: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

        return [row[0] for row in rows]


# retrieves a lead profile for a particular session_id
    def get_lead_profile(self, session_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        query = f"""
            SELECT profile FROM lead_profiles
            WHERE session_id=%s
        """

        try:
            cursor.execute(query, (session_id,))
            row = cursor.fetchone()

            schema = SalesController.create_new_profile()

            if not row:
                return schema

            profile = row[0]
            return {k: profile.get(k, v) for k, v in schema.items()}
            
        except Exception as e:
            raise Exception(f"Error fetching lead profile: {e}")

        finally:
            cursor.close()
            self._put_conn(connection)

# saves a lead profile
    def save_lead_profile(self, session_id, profile):
        connection = self._get_conn()
        cursor = connection.cursor()

        query = """
        INSERT INTO lead_profiles (session_id, profile)
        VALUES (%s, %s)
        ON CONFLICT (session_id)
        DO UPDATE SET
            profile = EXCLUDED.profile,
            updated_at = CURRENT_TIMESTAMP
        """

        try:
            cursor.execute(query, (session_id, json.dumps(profile)))
            connection.commit()

        except Exception as e:
            connection.rollback()
            raise Exception(f"Error saving lead profile: {e}")

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

    # returns all chunks across documents and webpages (used to rebuild BM25 encoder)
    def get_all_chunks(self):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = """
            SELECT doc_id, chunk_id, text FROM document_chunks
            UNION ALL
            SELECT url_id, chunk_id, text FROM webpage_chunks
        """

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

    # creates a record for a crawled URL
    def create_url(self, url, domain):
        connection = self._get_conn()
        cursor = connection.cursor()

        url_id = str(uuid.uuid4())

        command = """INSERT INTO webpages
            (url_id, url, domain, crawled_date, crawl_status)
            VALUES (%s, %s, %s, %s, %s)"""

        try:
            cursor.execute(command, (url_id, url, domain, "now()", DocumentStatus.CREATED.value))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

        return url_id

    def set_url_crawl_status(self, url_id, status):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = "UPDATE webpages SET crawl_status = %s WHERE url_id = %s"

        try:
            cursor.execute(command, (status, url_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def set_url_text_path(self, url_id, bucket, key):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = "UPDATE webpages SET s3_text_bucket = %s, s3_text_key = %s WHERE url_id = %s"

        try:
            cursor.execute(command, (bucket, key, url_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def get_url_text_path(self, url_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = "SELECT s3_text_bucket, s3_text_key FROM webpages WHERE url_id = %s"

        try:
            cursor.execute(command, (url_id,))
            row = cursor.fetchone()
            if not row:
                raise Exception("No webpage record for given url_id")
            return [row[0], row[1]]
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def set_url_in_kb(self, url_id):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = "UPDATE webpages SET in_kb = TRUE WHERE url_id = %s"

        try:
            cursor.execute(command, (url_id,))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def set_url_kb_status(self, url_id, status):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = "UPDATE webpages SET kb_status = %s WHERE url_id = %s"

        try:
            cursor.execute(command, (status, url_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def get_webpage_status(self, url_id: str) -> dict:
        connection = self._get_conn()
        cursor = connection.cursor()
        try:
            cursor.execute(
                "SELECT crawl_status, kb_status, error_msg FROM webpages WHERE url_id = %s",
                (url_id,)
            )
            row = cursor.fetchone()
        finally:
            cursor.close()
            self._put_conn(connection)
        if not row:
            return {"crawl_status": "not_found", "kb_status": "not_found", "error_msg": None}
        return {"crawl_status": row[0], "kb_status": row[1], "error_msg": row[2]}

    def set_url_error(self, url_id, error_msg):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = "UPDATE webpages SET error_msg = %s WHERE url_id = %s"

        try:
            cursor.execute(command, (error_msg, url_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def insert_webpage_chunks(self, url_id, rows):
        connection = self._get_conn()
        cursor = connection.cursor()

        command = """INSERT INTO webpage_chunks
            (url_id, chunk_id, text, heading)
            VALUES (%s, %s, %s, %s)"""

        try:
            cursor.executemany(
                command,
                [
                    (
                        url_id,
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
            

    def save_booking(self, session_id, event_id, slot_iso, name, email) -> str:
        connection = self._get_conn()
        cursor = connection.cursor()
        booking_id = str(uuid.uuid4())
        command = """
        INSERT INTO bookings (booking_id, session_id, event_id, slot_iso, name, email)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            cursor.execute(command, (booking_id, session_id, event_id, slot_iso, name, email))
            connection.commit()
            return booking_id
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error saving booking: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def get_booking_by_session(self, session_id) -> dict | None:
        connection = self._get_conn()
        cursor = connection.cursor()
        query = """
        SELECT booking_id, event_id, slot_iso, name, email, status
        FROM bookings
        WHERE session_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        try:
            cursor.execute(query, (session_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "booking_id": row[0],
                "event_id": row[1],
                "slot_iso": row[2],
                "name": row[3],
                "email": row[4],
                "status": row[5],
            }
        except Exception as e:
            raise Exception(f"Error fetching booking: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def get_booking_by_email(self, email: str) -> dict | None:
        connection = self._get_conn()
        cursor = connection.cursor()
        query = """
        SELECT booking_id, event_id, slot_iso, name, email, status
        FROM bookings
        WHERE LOWER(email) = LOWER(%s) AND status != 'cancelled'
        ORDER BY created_at DESC
        LIMIT 1
        """
        try:
            cursor.execute(query, (email.strip(),))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "booking_id": row[0],
                "event_id": row[1],
                "slot_iso": row[2],
                "name": row[3],
                "email": row[4],
                "status": row[5],
            }
        except Exception as e:
            raise Exception(f"Error fetching booking by email: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def update_booking_slot(self, booking_id, new_slot_iso) -> None:
        connection = self._get_conn()
        cursor = connection.cursor()
        command = """
        UPDATE bookings SET slot_iso = %s, status = 'rescheduled'
        WHERE booking_id = %s
        """
        try:
            cursor.execute(command, (new_slot_iso, booking_id))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error updating booking slot: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    def cancel_booking(self, booking_id) -> None:
        connection = self._get_conn()
        cursor = connection.cursor()
        command = "UPDATE bookings SET status = 'cancelled' WHERE booking_id = %s"
        try:
            cursor.execute(command, (booking_id,))
            connection.commit()
        except Exception as e:
            connection.rollback()
            raise Exception(f"Error cancelling booking: {e}")
        finally:
            cursor.close()
            self._put_conn(connection)

    # private helper functions
    def _get_conn(self):
        return self.pool.getconn()

    def _put_conn(self, conn):
        self.pool.putconn(conn)