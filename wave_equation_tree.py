"""
Экспрессионные деревья (суперпозиции) для 3D волнового уравнения.
Уравнение: ∂²u/∂t² = c² * ∇²u + f(x,y,z,t)
Где:
  u - смещение (наблюдаемая переменная)
  c - скорость волны (параметр)
  f - источник (параметр/наблюдаемая)
  ∇²u - лапласиан (латентная операция)
"""

import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum

class NodeColor(Enum):
    OBSERVED = "observed"    # u, f
    LATENT = "latent"        # операции
    PARAMETER = "parameter"  # c

class WaveEquationDAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self.colors = {}
        self.labels = {}

    def _add_node(self, name, color, label=None):
        node_id = self.node_counter
        self.graph.add_node(node_id)
        self.colors[node_id] = color
        self.labels[node_id] = label if label else name
        self.node_counter += 1
        return node_id

    def _add_edge(self, u, v):
        self.graph.add_edge(u, v)

    def build(self):
        # 1. Наблюдаемые переменные (Входы)
        u = self._add_node("u", NodeColor.OBSERVED)
        f = self._add_node("f", NodeColor.OBSERVED)

        # 2. Параметры
        c = self._add_node("c", NodeColor.PARAMETER)
        c_sq = self._add_node("c^2", NodeColor.LATENT, "square")

        # 3. Латентные операции (Лапласиан)
        # d2u/dx2, d2u/dy2, d2u/dz2
        d2u_dx2 = self._add_node("d2u/dx2", NodeColor.LATENT, "d2/dx2")
        d2u_dy2 = self._add_node("d2u/dy2", NodeColor.LATENT, "d2/dy2")
        d2u_dz2 = self._add_node("d2u/dz2", NodeColor.LATENT, "d2/dz2")

        # Сумма вторых производных (Лапласиан)
        laplacian_sum1 = self._add_node("sum_d2u_dx2_dy2", NodeColor.LATENT, "+")
        laplacian = self._add_node("laplacian_u", NodeColor.LATENT, "+")

        # 4. Правая часть уравнения: c^2 * Laplacian + f
        rhs_term1 = self._add_node("c2_times_lap", NodeColor.LATENT, "*")
        rhs = self._add_node("rhs", NodeColor.LATENT, "+")

        # Связи (Edges) - входы для узлов
        # c -> c^2
        self._add_edge(c, c_sq)

        # u -> производные
        self._add_edge(u, d2u_dx2)
        self._add_edge(u, d2u_dy2)
        self._add_edge(u, d2u_dz2)

        # Производные -> Лапласиан
        self._add_edge(d2u_dx2, laplacian_sum1)
        self._add_edge(d2u_dy2, laplacian_sum1)
        self._add_edge(laplacian_sum1, laplacian)
        self._add_edge(d2u_dz2, laplacian)

        # c^2 и Лапласиан -> RHS
        self._add_edge(c_sq, rhs_term1)
        self._add_edge(laplacian, rhs_term1)
        
        # f и rhs_term1 -> RHS
        self._add_edge(rhs_term1, rhs)
        self._add_edge(f, rhs)

        return self.graph

    def get_adjacency_matrix(self):
        return nx.to_numpy_array(self.graph)

    def get_node_info(self):
        return [(self.labels[n], self.colors[n].value) for n in self.graph.nodes]

def visualize_graph(dag, output_path="wave_dag.png"):
    pos = nx.spring_layout(dag.graph, seed=42)
    colors = [dag.colors[n].value for n in dag.graph.nodes]
    
    # Маппинг цветов для отрисовки
    color_map = {'observed': '#4C9AFF', 'latent': '#8ED18E', 'parameter': '#FFAB4C'}
    node_colors = [color_map[c] for c in colors]
    
    plt.figure(figsize=(12, 8))
    nx.draw(dag.graph, pos, with_labels=True, labels=dag.labels, 
            node_color=node_colors, node_size=2000, font_size=10, 
            font_weight='bold', arrowsize=20)
    plt.title("DAG волнового уравнения")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Граф сохранен в {output_path}")

if __name__ == "__main__":
    dag = WaveEquationDAG()
    dag.build()
    visualize_graph(dag)
    print("Матрица смежности:\n", dag.get_adjacency_matrix())
