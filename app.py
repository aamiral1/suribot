from flask import Flask, render_template, jsonify, request
from flask_restful import Api, Resource, reqparse
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from database import Database
from doc_parser import extract_doc_info
from enums import DocumentStatus, SourceType, AllowedFileTypes
import boto3
import botocore
import threading
import exceptions as ex
import datetime
import os
import tempfile

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["OCR_PROCESSING_FOLDER"] = "static/ocr"
app.config["EXTRACTED_TEXT_FOLDER"] = "static/extracted_text"
api = Api(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=20)

response = client

# Initialise Postgres Database
db = Database("document_database.db", "documents")
db.init_schema()

# Initialise AWS S3 Client
s3 = boto3.resource(
    service_name="s3",
    region_name="eu-north-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)

bucket_name = os.getenv("AWS_S3_BUCKET")


@app.route("/")
def home():
    return render_template("index.html")


# REST API for backend
chatbot_args = reqparse.RequestParser()
chatbot_args.add_argument("message", type=str, help="Message for LLM", required=True)


# API that handles communication with Open AI
class ChatbotAPI(Resource):
    def post(self):
        args = chatbot_args.parse_args()

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": args["message"]}]
        )

        # print(response.choices[0].message.content)
        return {"response": response.choices[0].message.content}, 200


api.add_resource(ChatbotAPI, "/api")


# REST API to poll document status (status, has_text)
class DocumentStatusAPI(Resource):
    def get(self, doc_id):

        # get status and extracted file path if available
        status = db.get_status(doc_id)

        if status == DocumentStatus.PROCESSING:
            msg = {"status": "processing", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp

        elif status == DocumentStatus.CREATED:
            msg = {"status": "created", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp

        elif status == DocumentStatus.FAILED:
            msg = {"status": "failed", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp

        elif status == DocumentStatus.SUCCESS:
            # get extracted text s3 bucket and key of document
            try:
                s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)

            except Exception as e:

                return jsonify(
                    {
                        "status": "error",
                        "has_text": False,
                        "message": "DB Error: " + str(e),
                    }
                )

            # check if file exists in S3
            if _file_exists(s3, s3_bucket, s3_key):
                msg = {"status": "success", "has_text": True}
                resp = jsonify(msg)
                resp.status_code = 200

                return resp
            else:
                msg = {"status": "error", "has_text": False}
                resp = jsonify(msg)
                resp.status_code = 200

                return resp

        else:
            # generic error
            msg = {"status": "failed", "has_text": False}
            resp = jsonify(msg)
            resp.status_code = 200

            return resp


api.add_resource(DocumentStatusAPI, "/document/<string:doc_id>/status")


# REST API to get extracted text from document
class ExtractedDocumentTextAPI(Resource):
    def get(self, doc_id):
        # get extracted text file path of document
        try:
            s3_bucket, s3_key = db.get_extracted_text_file_path(doc_id)

        except Exception as e:
            return jsonify({"text": None})

        # if extracted text file exists
        text = None

        if _file_exists(s3, s3_bucket, s3_key):
            # retrieve text from the file
            text = _get_file_text(s3, s3_bucket, s3_key)

        # return extracted text in JSON response
        return jsonify({"text": text})


api.add_resource(ExtractedDocumentTextAPI, "/document/<string:doc_id>/text")


# REST API to get all documents
@app.route("/documents", methods=["GET"])
def get_documents():
    try:
        rows = db.get_all_documents()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    documents = [
        {
            "doc_id":        row[0],
            "source_type":   row[1],
            "file_name":     row[2],
            "file_size":     row[3],
            "file_type":     row[4],
            "uploaded_date": str(row[5]),
            "status":        row[8],
            "in_kb":         row[12],
        }
        for row in rows
    ]

    return jsonify({"documents": documents}), 200


# REST API to mark a document as added to the knowledge base
@app.route("/document/<string:doc_id>/add-to-kb", methods=["POST"])
def add_to_kb(doc_id):
    try:
        db.set_in_kb(doc_id)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "ok"}), 200


# Admin Dashboard
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    form = UploadFileForm()

    if request.method == "GET":
        return render_template("admin-revised.html", form=form)

    if form.validate_on_submit():
        file = form.file.data  # get the uploaded file data sent from frontend
        file_name = secure_filename(file.filename)
        file_size = str(_format_size(_get_file_size(file)))
        file_type = AllowedFileTypes.from_filename(file_name)

        # check if file already exists in S3 bucket
        if _file_exists(s3, bucket_name, file_name):
            print("File already exits")

            msg = {"status": "fail", "error": "File already exists"}

            return jsonify(msg), 400

        # upload file to S3 storage
        s3.meta.client.upload_fileobj(file, bucket_name, file_name)

        # Register file entry in database
        doc_id = db.create(
            source_type=SourceType.UPLOAD.value,
            name=file_name,
            size=file_size,
            type=file_type.value,
            upload_date=datetime.datetime.now(),
            s3_file_bucket=bucket_name,
            s3_file_key=file_name,
            s3_extracted_text_bucket="na",
            s3_extracted_text_key="na"
        )

        msg = {
            "status": "ok",
            "doc_id": doc_id,
            "s3_file_bucket": bucket_name,
            "s3_file_key": file_name,
        }

        print("File has been uploaded succesfully.")
        print(f"Current status of file is {db.get_status(doc_id)}")

        return jsonify(msg), 200

    else:
        return jsonify({"status": "fail", "error": "validation failed"}), 400


# Route to extract text from an uploaded document
@app.route("/extract-text", methods=["POST"])
def extract_text():
    # wrapper function to thread extraction: Replace with Celery later

    # AWS S3 - CONTINUE FROM HERE!
    def extraction_job(doc_id, file_path):
        file_name = doc_id + ".txt"
        isSuccess = False

        try:
            extracted_text = extract_doc_info(
                client=client,
                file_path=file_path,
            )

            # upload extracted text as file to S3 bucket
            s3.meta.client.put_object(
                Bucket=os.getenv("AWS_S3_BUCKET"),
                Key=secure_filename(file_name),
                Body=extracted_text.encode("utf-8"),
                ContentType="text/plain",
            )

            # register extracted text file path in database
            try:
                db.set_extraction_text_path(
                    doc_id=doc_id,
                    s3_extracted_text_bucket=bucket_name,
                    s3_extracted_text_key=secure_filename(file_name),
                )

                isSuccess = True

            except Exception as e:
                print("Failed to update extracted file path of document")
                raise

            try:
                # update document status based on whether extracted text file exists
                if isSuccess:
                    db.transition_status(
                        doc_id=doc_id, new_status=DocumentStatus.SUCCESS
                    )
                else:
                    db.transition_status(
                        doc_id=doc_id, new_status=DocumentStatus.FAILED
                    )

            except ex.InvalidDocumentStatusTransition as e:
                db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
                raise

            except Exception as e:
                db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
                raise

        except ex.InvalidDocumentStatusTransition as e:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(e)

        except ex.ExtractionTimeOut as e:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(e)

        except Exception as e:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(f"Error: {e}")

        finally:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    # get doc id from POST request
    data = request.get_json()
    print(data)
    doc_id = data["doc_id"]

    # Check if document is valid to be processed
    doc_status = db.get_status(doc_id)

    if doc_status == DocumentStatus.CREATED or doc_status == DocumentStatus.FAILED:
        # update doc status to Processing and perform extraction
        try:
            db.transition_status(
                doc_id=doc_id, new_status=DocumentStatus.PROCESSING
            )  # set doc status to processing

            # extract file path from doc_id based on DB
            s3_bucket, s3_key = db.get_file_path(doc_id)

            local_file_path = _download_s3_file_to_temp(s3, s3_bucket, s3_key)

            # successfully extraction starts - replace with Celery
            threading.Thread(
                target=extraction_job, args=(doc_id, local_file_path)
            ).start()  # begin extraction job on thread

        except ex.InvalidDocumentStatusTransition as e:
            msg = {
                "status": "failed",
                "message": "Unable to update document status to PROCESSING on database (IDST)",
            }

            return jsonify(msg), 500

        except Exception as e:
            msg = {"status": "failed", "message": str(e)}
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.FAILED)
            print(f"Error: {e}")
            return jsonify(msg), 500


        msg = {"status": "began processing"}

        return jsonify(msg), 200

    elif doc_status == DocumentStatus.PROCESSING:
        msg = {"status": "already processing"}
        return jsonify(msg), 200

    elif doc_status == DocumentStatus.SUCCESS:
        msg = {"status": "already extracted"}
        return jsonify(msg), 200

    else:
        msg = {"status": "error"}
        return jsonify(msg), 500



# AWS S3 Helper Functions
def _file_exists(resource, bucket, key):
    try:
        resource.meta.client.head_object(Bucket=bucket, Key=key)
        print(f"File: '{key}' found!")
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            print(f"File'{key}' File does not exist!")
            return False
        else:
            print("Something else went wrong")
            return False


def _get_file_text(resource, bucket, key):
    # get the object
    response = resource.meta.client.get_object(Bucket=bucket, Key=key)

    # Read the file contents
    file_content = response["Body"].read()

    return file_content.decode("utf-8")


def _get_file_size(file):
    pos = file.stream.tell()
    file.stream.seek(0, 2)
    size = file.stream.tell()
    file.stream.seek(pos)
    return size


def _format_size(size_bytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024


def _get_file_type(filename: str) -> AllowedFileTypes:
    if not filename or "." not in filename:
        raise ex.InvalidFileType("Invalid file name")

    ext = os.path.splitext(filename)[1].lower().strip(".")  # 'pdf'
    ext_upper = ext.upper()  # 'PDF'

    try:
        return AllowedFileTypes(ext_upper)
    except ValueError:
        raise ex.InvalidFileType(f"Unsupported file type: {ext_upper}")

def _download_s3_file_to_temp(s3_resource, bucket, key):
    suffix = os.path.splitext(key)[1] or ".tmp"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    tmp.close()

    s3_resource.meta.client.download_file(bucket, key, tmp_path)
    return tmp_path


if __name__ == "__main__":
    app.run(debug=True)
