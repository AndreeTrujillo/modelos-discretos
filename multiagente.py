from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# Clase para representar a un agente
class Agente:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def mover(self):
        # Ejemplo de una regla de comportamiento simple: moverse aleatoriamente
        self.x += random.choice([-1, 0, 1])
        self.y += random.choice([-1, 0, 1])

    def to_dict(self):
        return {'id': self.id, 'x': self.x, 'y': self.y}

# Inicializar agentes
def inicializar_agentes(num_agentes):
    agentes = []
    for i in range(num_agentes):
        agentes.append(Agente(i, random.randint(0, 10), random.randint(0, 10)))
    return agentes

@app.route('/')
def index():
    return render_template('multiagente.html')

@app.route('/simular', methods=['POST'])
def simular():
    data = request.json
    num_agentes = int(data.get('num_agentes', 10))
    pasos = int(data.get('pasos', 10))

    agentes = inicializar_agentes(num_agentes)
    historia = []

    for _ in range(pasos):
        estado_actual = [agente.to_dict() for agente in agentes]
        historia.append(estado_actual)
        for agente in agentes:
            agente.mover()

    return jsonify(historia)

if __name__ == '__main__':
    app.run(debug=True)
