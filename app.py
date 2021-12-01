from flask import Flask, request, render_template
import pickle

from numpy.lib.utils import safe_eval
from ret import search

app = Flask(__name__)

@app.route('/')
def hello():
	return render_template("ui.html")

@app.route('/ui')
def xxx():
	return render_template("ui.html")

@app.route('/search', methods=['POST', 'GET'])
def findSong():
	search_query = request.form['query']
	ans = search(search_query)
	print(len(ans), type(ans), ans)

	return render_template('result.html', xx=ans)

@app.route('/similar/<name>')
def similarquery(name):
	ans = search(name)
	return render_template('similarui.html', xx=ans)

if __name__ == '__main__':
	app.run()