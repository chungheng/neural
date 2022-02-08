"""Utilities for Networks"""
import typing as tp
import numpy as np
import pydot
from ..network.network import Symbol, Input
from .. import errors as err


def to_graph(network, png: bool = False, svg: bool = False, prog="dot") -> tp.Union[bytes, dict]:
    """Visualize Network instance as Graph

    Arguments:
        network: network to visualize

    Keyword Arguments:
        png : whether to return png image as output
        svg : whether to return svg image as output
            - png takes precedence over svg
        prog: program used to optimize graph layout
    """
    graph = pydot.Dot(
        graph_type="digraph", rankdir="LR", splines="ortho", decorate=True
    )

    nodes = {}
    for c in list(network.containers.values()) + list(network.inputs.values()):
        node = pydot.Node(c.name, shape="rect")
        nodes[c.name] = node
        graph.add_node(node)

    edges = []
    for c in network.containers.values():
        target = c.name
        v = nodes[target]
        for key, val in c.inputs.items():
            if isinstance(val, Symbol):
                source = val.container.name
                label = val.key
            elif isinstance(val, Input):
                source = val.name
                label = ""
            else:
                raise err.NeuralNetworkError(
                    f"Container wrapping [{c.obj}] input {key} value {val} not "
                    "understood"
                )
            u = nodes[source]
            graph.add_edge(pydot.Edge(u, v, label=label))
            edges.append((source, target, label))

    if png:  # return PNG Directly
        png_str = graph.create_png(prog="dot")  # pylint: disable=no-member
        return png_str
    if svg:
        svg_str = graph.create_svg(prog="dot")  # pylint: disable=no-member
        return svg_str
    
    # return dot
    D_bytes = graph.create_dot(prog="dot")  # pylint: disable=no-member

    D = str(D_bytes, encoding="utf-8")

    if D == "":  # no data returned
        print(
            f"""Graphviz layout with {prog} failed
            To debug what happened try:
            >>> P = nx.nx_pydot.to_pydot(G)
            >>> P.write_dot("file.dot")
            And then run {prog} on file.dot"""
        )

    # List of "pydot.Dot" instances deserialized from this string.
    Q_list = pydot.graph_from_dot_data(D)
    assert len(Q_list) == 1
    Q = Q_list[0]
    # return Q

    def get_node(Q, n):
        node = Q.get_node(n)

        if isinstance(node, list) and len(node) == 0:
            node = Q.get_node(f'"{n}"')
            assert node

        return node[0]

    def get_label_xy(x, y, ex, ey):
        min_dist = np.inf
        min_ex, min_ey = [0, 0], [0, 0]
        for _ex, _ey in zip(zip(ex, ex[1:]), zip(ey, ey[1:])):
            dist = (np.mean(_ex) - x) ** 2 + (np.mean(_ey) - y) ** 2
            if dist < min_dist:
                min_dist = dist
                min_ex[:] = _ex[:]
                min_ey[:] = _ey[:]
        if min_ex[0] == min_ex[1]:
            _x = min_ex[0]
            _x = np.sign(x - _x) * 10 + _x
            _y = y
        else:
            _x = x
            _y = min_ey[0]
            _y = np.sign(y - _y) * 10 + _y
        return _x, _y - 3

    elements = []
    bb = Q.get_bb()
    viewbox = bb[1:-1].replace(",", " ")

    for n in nodes.keys():

        node = get_node(Q, n)

        # strip leading and trailing double quotes
        pos = node.get_pos()[1:-1]

        if pos is not None:
            obj = network.get_obj(n)
            w = float(node.get_width())
            h = float(node.get_height())

            x, y = map(float, pos.split(","))
            attrs = {
                "width": w,
                "height": h,
                "rx": 5,
                "ry": 5,
                "x": x,
                "y": y,
                "stroke-width": 1.5,
                "fill": "none",
                "stroke": "#48caf9",
            }

            elements.append(
                {
                    "label": [n, x, y],
                    "shape": "rect",
                    "attrs": attrs,
                    "latex": obj.latex_src,
                    "graph": obj.graph_src,
                }
            )

    min_x, min_y, scale_w, scale_h = np.inf, np.inf, 0, 0
    for el in elements:
        if min_x > el["attrs"]["x"]:
            min_x = el["attrs"]["x"]
            scale_w = 2 * min_x / el["attrs"]["width"]
        if min_y > el["attrs"]["y"]:
            min_y = el["attrs"]["y"]
            scale_h = 2 * min_y / el["attrs"]["height"]
    for el in elements:
        w = scale_w * el["attrs"]["width"]
        h = scale_h * el["attrs"]["height"]
        el["attrs"]["x"] = el["attrs"]["x"] - w / 2
        el["attrs"]["y"] = el["attrs"]["y"] - h / 2
        el["attrs"]["width"] = w
        el["attrs"]["height"] = h

    for edge in Q.get_edge_list():
        pos = (edge.get_pos()[1:-1]).split(" ")
        ax, ay = [float(v) for v in pos[0].split(",")[1:]]
        pos = [v.split(",") for v in pos[1:]]

        xx = [float(v[0]) for v in pos] + [ax]
        yy = [float(v[1]) for v in pos] + [ay]
        x, y, _x, _y = [], [], 0, 0
        for __x, __y in zip(xx, yy):
            if not (__x == _x and __y == _y):
                x.append(__x)
                y.append(__y)
            _x = __x
            _y = __y
        path = [f"{_x} {_y}" for _x, _y in zip(x, y)]
        p = "M" + " L".join(path)
        attrs = {"d": p, "stroke-width": 1.5, "fill": "none", "stroke": "black"}
        lp = edge.get_lp()
        if lp:
            lx, ly = [float(v) for v in lp[1:-1].split(",")]
            lx, ly = get_label_xy(lx, ly, x, y)
            label = [edge.get_label() or "", lx, ly]
        elements.append({"label": label, "shape": "path", "attrs": attrs})
    output = {"elements": elements, "viewbox": viewbox}
    return output
