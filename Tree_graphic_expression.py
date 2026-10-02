import re
import matplotlib.pyplot as plt
import networkx as nx


# ============================================================
# 1. Expresión GP a visualizar
# ============================================================

expression = """
gp_subtract(gp_add(gp_subtract(POINT, RECTANGLE), gp_convolution(gp_closing(gp_dilation(CIRCLE)))), gp_add(LINE_PARAM(6.56, 5.0, 7.0), SQUARE))
"""


# ============================================================
# 2. Parser de la expresión
# ============================================================

def tokenize(expression):
    """
    Divide la expresión en tokens.
    """
    pattern = r"""
        [A-Za-z_][A-Za-z0-9_]*   # nombres de funciones/terminales
        |
        [-+]?\d*\.\d+            # números decimales
        |
        [-+]?\d+                 # números enteros
        |
        [(),]                    # paréntesis y comas
    """

    return re.findall(pattern, expression, re.VERBOSE)


def parse_expression(tokens, index=0):
    """
    Construye recursivamente un árbol a partir de los tokens.
    """

    name = tokens[index]
    index += 1

    # Si es una función
    if index < len(tokens) and tokens[index] == "(":

        index += 1  # consumir '('

        children = []

        while tokens[index] != ")":

            child, index = parse_expression(tokens, index)
            children.append(child)

            if tokens[index] == ",":
                index += 1

        index += 1  # consumir ')'

        return {
            "name": name,
            "children": children
        }, index

    # Si es un terminal
    return {
        "name": name,
        "children": []
    }, index


# ============================================================
# 3. Construcción del grafo
# ============================================================

def build_graph(tree):

    graph = nx.DiGraph()

    counter = [0]

    def add_node(node, parent=None):

        node_id = counter[0]
        counter[0] += 1

        graph.add_node(
            node_id,
            label=node["name"]
        )

        if parent is not None:
            graph.add_edge(parent, node_id)

        for child in node["children"]:
            add_node(child, node_id)

        return node_id

    add_node(tree)

    return graph


# ============================================================
# 4. Layout jerárquico
# ============================================================

def hierarchy_pos(graph, root=0, width=1.0, vert_gap=0.25,
                  vert_loc=0, xcenter=0.5):

    pos = {}

    def _hierarchy_pos(
        graph,
        root,
        left,
        right,
        vert_loc,
        pos,
        parent=None
    ):

        children = list(graph.neighbors(root))

        if not children:

            pos[root] = ((left + right) / 2, vert_loc)

        else:

            dx = (right - left) / len(children)

            for i, child in enumerate(children):

                child_left = left + i * dx
                child_right = left + (i + 1) * dx

                _hierarchy_pos(
                    graph,
                    child,
                    child_left,
                    child_right,
                    vert_loc - vert_gap,
                    pos,
                    root
                )

            pos[root] = (
                (left + right) / 2,
                vert_loc
            )

    _hierarchy_pos(
        graph,
        root,
        0,
        width,
        vert_loc,
        pos
    )

    return pos


# ============================================================
# 5. Crear el árbol
# ============================================================

tokens = tokenize(expression)

tree, index = parse_expression(tokens)

graph = build_graph(tree)

pos = hierarchy_pos(graph)


# ============================================================
# 6. Dibujar
# ============================================================

plt.figure(figsize=(16, 8))

labels = nx.get_node_attributes(graph, "label")

nx.draw(
    graph,
    pos,
    labels=labels,
    with_labels=True,
    node_size=3500,
    node_color="lightblue",
    node_shape="o",
    font_size=9,
    font_weight="bold",
    arrows=True,
    arrowsize=20,
    edge_color="gray"
)

plt.title(
    "Árbol sintáctico del individuo GP",
    fontsize=16
)

plt.axis("off")

plt.tight_layout()

# Mostrar la imagen.
# NO se guarda automáticamente.
plt.show()