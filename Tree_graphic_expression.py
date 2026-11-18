import ast
import math
import networkx as nx
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Formato de etiquetas
# ---------------------------------------------------------

def format_label(value):

    if isinstance(value, float):
        return f"{value:.1f}"

    return str(value)


# ---------------------------------------------------------
# Construcción del árbol a partir de una expresión GP
# ---------------------------------------------------------

def build_graph(node, graph, labels, counter):

    node_id = counter[0]
    counter[0] += 1

    graph.add_node(node_id)

    # Llamadas a funciones: gp_add(...), Tan(...), etc.
    if isinstance(node, ast.Call):

        if isinstance(node.func, ast.Name):
            label = node.func.id

        elif isinstance(node.func, ast.Attribute):
            label = node.func.attr

        else:
            label = "Function"

        labels[node_id] = label

        for argument in node.args:

            child_id = build_graph(
                argument, graph, labels, counter
            )

            graph.add_edge(node_id, child_id)

    # Variables y terminales
    elif isinstance(node, ast.Name):

        labels[node_id] = node.id

    # Constantes numéricas
    elif isinstance(node, ast.Constant):

        labels[node_id] = format_label(node.value)

    # Operaciones binarias
    elif isinstance(node, ast.BinOp):

        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Pow: "**"
        }

        labels[node_id] = operators.get(
            type(node.op), "?"
        )

        left_id = build_graph(
            node.left, graph, labels, counter
        )

        right_id = build_graph(
            node.right, graph, labels, counter
        )

        graph.add_edge(node_id, left_id)
        graph.add_edge(node_id, right_id)

    # Operaciones unarias
    elif isinstance(node, ast.UnaryOp):

        labels[node_id] = "-"

        child_id = build_graph(
            node.operand, graph, labels, counter
        )

        graph.add_edge(node_id, child_id)

    else:

        labels[node_id] = str(node)

    return node_id


# ---------------------------------------------------------
# Distribución jerárquica del árbol
# ---------------------------------------------------------

def calculate_tree_positions(
    graph,
    labels,
    root,
    horizontal_spacing=1.0,
    vertical_spacing=2.0
):

    positions = {}
    leaf_counter = [0]

    def calculate_subtree(node, depth):

        children = list(graph.successors(node))

        # Caso 1: nodo hoja
        if not children:

            x = leaf_counter[0]
            leaf_counter[0] += 1

            positions[node] = (
                x * horizontal_spacing,
                -depth * vertical_spacing
            )

            return x, x

        # Caso 2: nodo con hijos
        intervals = []

        for child in children:

            interval = calculate_subtree(
                child, depth + 1
            )

            intervals.append(interval)

        # Intervalo horizontal ocupado por el subárbol
        left = intervals[0][0]
        right = intervals[-1][1]

        # El padre se coloca en el centro de sus descendientes
        x = (left + right) / 2

        positions[node] = (
            x * horizontal_spacing,
            -depth * vertical_spacing
        )

        return left, right

    calculate_subtree(root, 0)

    return positions, leaf_counter[0]



