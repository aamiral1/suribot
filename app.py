from flask import Flask, request, render_template
from flask_restful import Api, Resource, reqparse

app = Flask(__name__)
api=Api(app)

chatbot_args = reqparse.RequestParser()
chatbot_args.add_argument("message", type=str, help="Message for LLM", required=True)

@app.route("/")
def home():
    return render_template("index.html")

# REST API for backend
class ChatbotAPI(Resource):
    def post(self):
        args = chatbot_args.parse_args()
        print(args)

        return {"response": "successful"}, 200

api.add_resource(ChatbotAPI, "/api")


# @app.route("/api", methods=["POST"])
# def api():
#     data = request.json
#     print(data)
#     return "Connection Successful"

if __name__ == "__main__":
    app.run(debug=True)