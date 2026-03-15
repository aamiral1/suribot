import sqlite3
import uuid
from enums import DocumentStatus
import exceptions as ex


class Database:
    def __init__(self, database_path, doc_table_name):
        self.db_path = database_path
        self.doc_table = doc_table_name

    def init_schema(self):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        create_table_command = f"""
            CREATE TABLE IF NOT EXISTS 
            {self.doc_table} (doc_id TEXT PRIMARY KEY, file_path TEXT NOT NULL UNIQUE, status TEXT NOT NULL, extracted_text_path TEXT)
        """

        cursor.execute(create_table_command)

        connection.commit()
        connection.close()

# creates record for a document with CREATED as initial value
    def create(self, file_path):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        doc_id = str(uuid.uuid4())

        command = f"INSERT INTO {self.doc_table} VALUES (?, ?, ?, ?)"

        try:
            cursor.execute(
                command,
                (doc_id, file_path, DocumentStatus.CREATED.value, None),
            )
            connection.commit()
        except sqlite3.IntegrityError:
            # return doc id of existing file
            raise Exception("File record already exists")
        except Exception as e:
            raise Exception(f"Error: {e}")
        finally:
            connection.close()

        return doc_id

    def get_path(self, doc_id):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        command = f"SELECT file_path FROM {self.doc_table} WHERE doc_id = ?"

        try:
            rows = cursor.execute(command, (doc_id,)).fetchone()

            if not rows:
                err = "No document exists for the given document id"
                raise Exception(err)

            file_path = rows[0]

        except Exception as e:
            raise Exception(f"Error: {e}")
        finally:
            connection.close()

        return file_path

    def get_status(self, doc_id):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        command = f"SELECT status FROM {self.doc_table} WHERE doc_id = ?"

        try:
            row = cursor.execute(command, (doc_id,)).fetchone()
            if row is None:
                raise ex.InvalidDocumentID("No document available")

            status = DocumentStatus(row[0])

        except Exception as e:
            raise Exception(f"Error: {e}")

        finally:
            connection.close()

        return status

    def transition_status(self, doc_id, new_status: DocumentStatus):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        get_status_command = f"SELECT status FROM {self.doc_table} WHERE doc_id = ?"

        update_status_command = f"UPDATE {self.doc_table} SET status=? WHERE doc_id=?"

        # update database with new status
        try:
            # check if current status and new status are valid
            row = cursor.execute(get_status_command, (doc_id,)).fetchone()
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
            raise Exception(f"Error: {e}")
        finally:
            connection.close()

    def set_extraction_text_path(self, doc_id, extracted_text_path):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        command = f"UPDATE {self.doc_table} SET extracted_text_path=? WHERE doc_id=?"

        try:
            cursor.execute(command, (extracted_text_path, doc_id))
            connection.commit()
        except Exception as e:
            raise Exception(f"Error: {e}")
        finally:
            connection.close()

    def get_extracted_text_file_path(self, doc_id):
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()

        command = f"SELECT extracted_text_path from {self.doc_table} WHERE doc_id=?"

        try:
            row = cursor.execute(command, (doc_id,)).fetchone()

            if not row:
                raise ex.InvalidDocumentID("No document available")

            path = row[0]

        except Exception as e:
            raise

        finally:
            connection.close()

        return path
