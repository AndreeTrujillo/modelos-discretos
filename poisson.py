from flask import Flask, render_template, request, jsonify
import math

app = Flask(__name__)

def poisson_probability(lmbda, k):
    """Calcula la probabilidad de k eventos en un intervalo dado el promedio lmbda usando la distribución de Poisson."""
    return (lmbda ** k * math.exp(-lmbda)) / math.factorial(k)

@app.route('/')
def index():
    return render_template('poisson.html')

@app.route('/calcular', methods=['POST'])
def calcular():
    data = request.json
    lmbda = float(data.get('lmbda', 1))
    k = int(data.get('k', 0))
    probabilidad = poisson_probability(lmbda, k)
    return jsonify(probabilidad=probabilidad)

if __name__ == '__main__':
    app.run(debug=True)
