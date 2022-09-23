
import movie
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = "secretkey"

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        movie_name = request.form['mvname']
        return render_template('recommend.html', movies = movie.recommend_movies(movie_name), mo=movie_name)
    return render_template('search.html')

if __name__ == '__main__':
    app.run(debug=True)
