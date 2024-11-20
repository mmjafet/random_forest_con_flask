from flask import Flask, render_template, jsonify, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from graphviz import Source

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/generate-tree", methods=["POST"])
def generate_tree():
    # Cargar el dataset
    df = pd.read_csv("datasets/TotalFeatures-ISCXFlowMeter.csv")
    
    # Especificar la columna objetivo
    label_name = 'calss'  # El nombre correcto de la columna objetivo
    
    # Separar características (X) y etiquetas (y)
    X = df.drop(label_name, axis=1)
    y = df[label_name]
    
    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.4, random_state=42)
    
    # Crear y entrenar el árbol de decisión
    clf_tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    clf_tree.fit(X_train, y_train)
    
    # Generar el archivo de visualización
    dot_data = export_graphviz(
        clf_tree,
        out_file=None,
        feature_names=X.columns,
        class_names=y.unique(),
        rounded=True,
        filled=True
    )
    
    # Guardar el árbol como SVG
    graph = Source(dot_data)
    graph.render("static/tree", format="svg", cleanup=True)
    
    return jsonify({"message": "Árbol generado correctamente", "svg_path": "/static/tree.svg"})

if __name__ == "__main__":
    app.run(debug=True)
