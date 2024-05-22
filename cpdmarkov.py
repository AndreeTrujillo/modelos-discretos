from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def value_iteration(transitions, rewards, gamma, threshold=1e-6):
    num_states, num_actions = len(rewards), len(rewards[0])
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            V[s] = max(sum(transitions[s][a][s1] * (rewards[s][a] + gamma * V[s1]) for s1 in range(num_states)) for a in range(num_actions))
            delta = max(delta, abs(v - V[s]))
        if delta < threshold:
            break
    
    for s in range(num_states):
        policy[s] = np.argmax([sum(transitions[s][a][s1] * (rewards[s][a] + gamma * V[s1]) for s1 in range(num_states)) for a in range(num_actions)])
    
    return policy, V

@app.route('/', methods=['GET', 'POST'])
def index():
    policy, value_function = None, None
    if request.method == 'POST':
        try:
            rewards = request.form['rewards']
            transitions = request.form['transitions']
            gamma = float(request.form['gamma'])

            rewards = [[float(num) for num in row.split()] for row in rewards.strip().split('\n')]
            transitions = [[[float(num) for num in col.split()] for col in row.split(';')] for row in transitions.strip().split('\n')]

            policy, value_function = value_iteration(transitions, rewards, gamma)
        except ValueError:
            policy, value_function = "Invalid input", "Invalid input"
    
    return render_template('cpdmarkov.html', policy=policy, value_function=value_function)

if __name__ == '__main__':
    app.run(debug=True)
