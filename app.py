from flask import Flask, render_template, jsonify, request
from flask_restful import Api, Resource, reqparse
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from dotenv import load_dotenv
from openai import OpenAI
from werkzeug.utils import secure_filename
from database import Database
from doc_parser import extract_doc_info
import os

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['OCR_PROCESSING_FOLDER'] = 'static/ocr'
api=Api(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client

chatbot_args = reqparse.RequestParser()
chatbot_args.add_argument("message", type=str, help="Message for LLM", required=True)

db = Database("document_database.db", "documents")
db.init_schema()

@app.route("/")
def home():
    return render_template("index.html")

# REST API for backend
class ChatbotAPI(Resource):
    def post(self):
        args = chatbot_args.parse_args()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": args['message']}
            ]
        )

        # print(response.choices[0].message.content)
        return {"response": response.choices[0].message.content}, 200

api.add_resource(ChatbotAPI, "/api")

# Admin Dashboard
class UploadFileForm(FlaskForm):
    file = FileField("File")
    submit = SubmitField("Upload File")

@app.route("/admin", methods=["GET", "POST"])
def admin():
    form = UploadFileForm()

    if request.method == "GET":
        return render_template("admin.html", form=form)
    
    # issue: allowing multiple submits for the same file incrementing iterator indefinitely
    if form.validate_on_submit():
        file = form.file.data # get the uploaded file data sent from frontend
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)) # get root dir path
        if os.path.exists(file_path):
            print("File already exits")
            
            msg = {
                'status': 'fail',
                'error': 'File already exists'
            }

            return jsonify(msg), 400
        
        file.save(file_path)  # Then save the file in local directory

        # Register file in database
        doc_id = db.create(file_path)

        msg = {
            'status': 'ok',
            'doc_id': doc_id,
            'doc_path': file_path
        }

        print("File has been uploaded succesfully.")
        return jsonify(msg), 200
    
    else:
        return jsonify({
            'status': 'fail',
            'error': 'validation failed'
        }), 400
    
# Route to extract text from an uploaded document
@app.route("/extract-text", methods=['POST'])
def extract_text():
    data = request.get_json()
    doc_id = data['doc_id']
    
    # extract text from doc_id
    file_path = db.get_path(doc_id)
    extracted_text = extract_doc_info(client=client, pdf_file_path=file_path, images_dir_name=app.config['OCR_PROCESSING_FOLDER'])

    msg = {
        'status': 'ok',
        'extracted_text': extracted_text
    }

    return jsonify(msg), 200
    
    

if __name__ == "__main__":
    app.run(debug=True)