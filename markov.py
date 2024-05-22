from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def markov_chain(transitions, state, steps):
    state = np.array(state)
    transitions = np.array(transitions)
    for _ in range(steps):
        state = np.dot(state, transitions)
    return state

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            matrix = request.form['matrix']
            state = request.form['state']
            steps = int(request.form['steps'])

            transitions = [list(map(float, row.split())) for row in matrix.strip().split('\n')]
            initial_state = list(map(float, state.split()))

            result = markov_chain(transitions, initial_state, steps)
        except ValueError:
            result = "Please enter valid numbers."
    return render_template('markov.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
