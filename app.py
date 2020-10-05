from flask import Flask,request,jsonify
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app) 
        
@app.route('/startup', methods=['GET'])
def recommend_startups():
    res = recommendation.results(request.args.get('Comapny'))
    return jsonify(res)

if __name__=='__main__':
    app.run(port = 5000, debug = True)