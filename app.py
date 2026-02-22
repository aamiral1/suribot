from flask import Flask, render_template
from flask_restful import Api, Resource, reqparse
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
api=Api(app)

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client

chatbot_args = reqparse.RequestParser()
chatbot_args.add_argument("message", type=str, help="Message for LLM", required=True)

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

@app.route("/admin")
def admin():
    return render_template("admin.html")

if __name__ == "__main__":
    app.run(debug=True)