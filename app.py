from flask import Flask, render_template, jsonify, request
from flask_restful import Api, Resource, reqparse
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from database import Database
from doc_parser import extract_doc_info
from enums import DocumentStatus
import threading
import exceptions as ex
import os

load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"
app.config["UPLOAD_FOLDER"] = "static/files"
app.config["OCR_PROCESSING_FOLDER"] = "static/ocr"
app.config["EXTRACTED_TEXT_FOLDER"] = "static/extracted_text"
api = Api(app)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=20)

response = client

db = Database("document_database.db", "documents")
db.init_schema()


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
            # get extracted text file path of document
            try:
                path = db.get_extracted_text_file_path(doc_id)

            except Exception as e:

                return jsonify({"status": "error", "has_text": False, "message": "DB Error: " + str(e)})
                

            # final check
            if path and os.path.exists(path):
                msg = {"status": "success", "has_text": True}
                resp = jsonify(msg)
                resp.status_code = 200

                return resp
            else:
                msg = {"status": "success", "has_text": False}
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
            file_path = db.get_extracted_text_file_path(doc_id)

        except Exception as e:
            return jsonify({"text": None})

        # if file exists
        text = None

        if file_path and os.path.exists(file_path):
            # open and extract text
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

        # return extracted text in JSON response
        return jsonify({"text": text})


api.add_resource(ExtractedDocumentTextAPI, "/document/<string:doc_id>/text")


# Admin Dashboard
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    form = UploadFileForm()

    if request.method == "GET":
        return render_template("admin.html", form=form)

    if form.validate_on_submit():
        file = form.file.data  # get the uploaded file data sent from frontend
        file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),
            app.config["UPLOAD_FOLDER"],
            secure_filename(file.filename),
        )  # get root dir path
        if os.path.exists(file_path):
            print("File already exits")

            msg = {"status": "fail", "error": "File already exists"}

            return jsonify(msg), 400

        file.save(file_path)  # Then save the file in local directory

        # Register file in database
        doc_id = db.create(file_path) # automatically sets status of doc to CREATED

        msg = {"status": "ok", "doc_id": doc_id, "doc_path": file_path}

        print("File has been uploaded succesfully.")
        print(f"Current status of file is {db.get_status(doc_id)}")
        return jsonify(msg), 200

    else:
        return jsonify({"status": "fail", "error": "validation failed"}), 400


# Route to extract text from an uploaded document
@app.route("/extract-text", methods=["POST"])
def extract_text():
    # wrapper function to thread extraction: Replace with Celery later
    def extraction_job(doc_id, pdf_file_path):
        file_name = doc_id + ".txt"
        try:
            # begin OpenAI extraction
            extracted_text = extract_doc_info(
                client=client,
                pdf_file_path=pdf_file_path,
                images_dir_name=app.config["OCR_PROCESSING_FOLDER"],
            )

            # create directory to store extracted text
            dir_path = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                app.config["EXTRACTED_TEXT_FOLDER"],
            )
            os.makedirs(dir_path, exist_ok=True)

            # save extracted file as .txt file in above directory
            file_path = os.path.join(dir_path, file_name)
            isSuccess = False
            with open(file_path, "w") as f:
                f.write(extracted_text)
                isSuccess = True

            # store extracted text file path in database
            try:
                db.set_extraction_text_path(
                    doc_id=doc_id, extracted_text_path=file_path
                )
            except Exception as e:
                print("Failed to updated extracted file path of document")
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

    # get doc id from POST request
    data = request.get_json()
    print(data)
    doc_id = data["doc_id"]

    # extract file path from doc_id based on DB
    pdf_file_path = db.get_path(doc_id)

    doc_status = db.get_status(doc_id=doc_id)

    if doc_status == DocumentStatus.CREATED or doc_status == DocumentStatus.FAILED:
        # update doc status to Processing
        try:
            db.transition_status(doc_id=doc_id, new_status=DocumentStatus.PROCESSING) # set doc status to processing

        except ex.InvalidDocumentStatusTransition as e:
            msg = {
                "status": "failed",
                "message": "Unable to update document status to PROCESSING on database (IDST)",
            }

            return jsonify(msg), 500

        except Exception as e:
            msg = {"status": "failed", "message": str(e)}

            return jsonify(msg), 500

        # successfully extraction starts
        threading.Thread(
            target=extraction_job, args=(doc_id, pdf_file_path)
        ).start()  # begin extraction job on thread

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


if __name__ == "__main__":
    app.run(debug=True)
