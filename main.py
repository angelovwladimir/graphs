from idlelib.pyshell import usage_msg
from collections import deque
from collections import defaultdict
from pprint import pprint
import sys
import heapq
class Graph:
    def __init__(self, path, type_file):
        self._matrix = None
        self._is_directed = None
        self._size = None
        self._adjacency_list = None
        match type_file:
            case 'list_of_adjacency':
                self.read_list_of_adjacency(path)

            case 'list_of_edges':
                self.read_list_of_edges(path)

            case 'matrix':
                self.read_matrix(path)

    def read_list_of_adjacency(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().split('\n')
            self._size = int(lines[0])
            self._adjacency_list = [[] for _ in range(self._size)]
            self._matrix = [[0] * (self._size + 1) for _ in range(self._size + 1)]
            for i, line in enumerate(lines[1:]):
                if line.strip():
                    parts = line.split()
                    for p in parts:
                        if ':' in p:
                            v_str, w_str = p.split(':')
                            v, w = int(v_str), int(w_str)
                        else:
                            v = int(p)
                            w = 1
                        self._adjacency_list[i].append((v, w))
                        self._matrix[i + 1][v] = w

        except FileNotFoundError:
            print("Файл не найден!")
        except PermissionError:
            print("Нет прав доступа к файлу!")
        except IOError as e:
            print(f"Произошла ошибка ввода/вывода: {e}")
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")

    def read_list_of_edges(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().split('\n')
            self._size = int(lines[0])
            self._matrix = [[0] * self._size for _ in range(self._size)]
            self._adjacency_list = [[] for _ in range(self._size)]
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) == 2:
                    u, v = map(int, parts)
                    w = 1
                elif len(parts) == 3:
                    u, v, w = map(int, parts)
                else:
                    raise ValueError(f"Некорректная строка с ребром: '{line}'")
                self._matrix[u - 1][v - 1] = w
                self._adjacency_list[u - 1].append((v, w))
        except FileNotFoundError:
            print("Файл не найден!")
        except PermissionError:
            print("Нет прав доступа к файлу!")
        except IOError as e:
            print(f"Произошла ошибка ввода/вывода: {e}")
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")

    def read_matrix(self, path):
        try:
            with open(path, 'r') as f:
                lines = f.read().strip().split('\n')
                self._size = int(lines[0])
                matrix = [list(map(int, line.split())) for line in lines[1:]]
                self._matrix = matrix
                self._adjacency_list = [[] for _ in range(self._size)]
                for i in range(self._size):
                    for j in range(self._size):
                        if matrix[i][j] != 0:
                            self._adjacency_list[i].append((j + 1, matrix[i][j]))



        except FileNotFoundError:
            print("Файл не найден!")
        except PermissionError:
            print("Нет прав доступа к файлу!")
        except IOError as e:
            print(f"Произошла ошибка ввода/вывода: {e}")
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")



    def is_directed(self):
        return 'Graph is directed' if self._is_directed else 'Graph is not directed'

    def size(self):
        return self._size

    def adjacency_matrix(self):
        return self._matrix

    def adjacency_list(self, top_number):
        row = self._matrix[top_number]
        adjacency_tops = []
        for i in range(len(row)):
            if row[i] != 0:
                adjacency_tops.append(i)
        return adjacency_tops

    def list_of_edges(self, top_number = None):
        if top_number is None:
            edges = []
            for i in range(1, self.size() + 1):
                for j in range(1, self.size() + 1):
                    if self._matrix[i][j] != 0:
                        edges.append([i, j])
            return edges
        else:
            edges = []
            for i in range(1, self.size() + 1):
                if self._matrix[top_number][i] != 0:
                    edges.append([top_number, i])
                    '''
                if matrix[i][top_number] != 0:
                    edges.append([i, top_number])
                    '''
            return edges

    def is_edge(self, start, end):
        if self._matrix[start][end] != 0:
            return True
        else:
            return False

    def weight(self, start, end):
        if self.is_edge(start, end):
            return self._matrix[start][end]
        else:
            return "Edge is not found"
    #1
    def connected_components(self):
        visited = [False] * self._size
        components = []

        adjacency = [[] for _ in range(self._size)]
        if self._is_directed:
            for u in range(self._size):
                for v, _ in self._adjacency_list[u]:
                    adjacency[u].append(v - 1)
                    adjacency[v - 1].append(u)
        else:
            for u in range(self._size):
                for v, _ in self._adjacency_list[u]:
                    adjacency[u].append(v - 1)

        def dfs(u, component):
            visited[u] = True
            component.append(u + 1)
            for v in adjacency[u]:
                if not visited[v]:
                    dfs(v, component)

        for i in range(self._size):
            if not visited[i]:
                component = []
                dfs(i, component)
                components.append(component)

        if self._is_directed:
            if len(components) == 1:
                print("Digraph is weakly connected")
            else:
                print("Digraph is not connected")
        else:
            if len(components) == 1:
                print("Graph is connected")
            else:
                print("Graph is not connected")

        print("\nConnected components:")
        for comp in components:
            comp.sort()
            print(comp)

        return components


    #2-8 TODO проверить еще раз
    def find_bridges_and_articulations(self):
        self._time = 0
        tin = [-1] * self._size
        low = [-1] * self._size
        visited = [False] * self._size
        bridges = []
        articulations = set()

        def dfs(u, parent):
            visited[u] = True
            tin[u] = low[u] = self._time
            self._time += 1
            children = 0

            for v, _ in self._adjacency_list[u]:
                v -= 1
                if v == parent:
                    continue
                if visited[v]:
                    low[u] = min(low[u], tin[v])
                else:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    if low[v] > tin[u]:
                        bridges.append((u + 1, v + 1))
                    if low[v] >= tin[u] and parent != -1:
                        articulations.add(u + 1)
                    children += 1

            if parent == -1 and children > 1:
                articulations.add(u + 1)

        for i in range(self._size):
            if not visited[i]:
                dfs(i, -1)

        bridges.sort()
        articulations = sorted(articulations)

        print("Bridges:")
        print(bridges)
        print("Cut vertices:")
        print(articulations)

        return bridges, articulations
    #4-10
    def floyd_warshall(self):
        n = self._size
        INF = float('inf')

        dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for u in range(n):
            for v, w in self._adjacency_list[u]:
                dist[u][v - 1] = w

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        self._dist = dist
        return dist

    def degree_vector(self):
        return [len(self._adjacency_list[i]) for i in range(self._size)]

    def eccentricities(self):
        if not hasattr(self, '_dist'):
            self.floyd_warshall()

        n = self._size
        ecc = []
        INF = float('inf')
        for i in range(n):
            max_dist = max(self._dist[i][j] for j in range(n) if self._dist[i][j] != INF)
            ecc.append(max_dist)
        return ecc

    def diameter_and_peripheral(self):
        ecc = self.eccentricities()
        diameter = max(ecc)
        peripheral = [i + 1 for i, e in enumerate(ecc) if e == diameter]
        return diameter, peripheral

    def radius_and_central(self):
        ecc = self.eccentricities()
        radius = min(ecc)
        central = [i + 1 for i, e in enumerate(ecc) if e == radius]
        return radius, central

    def reconstruct_path(self, u, v):
        u -= 1
        v -= 1
        if not hasattr(self, '_next'):
            self.floyd_warshall()

        if self._next[u][v] is None:
            return None

        path = [u]
        while u != v:
            u = self._next[u][v]
            path.append(u)

        return [x + 1 for x in path]

    def component_characteristics(self):
        if not hasattr(self, '_dist'):
            self.floyd_warshall()

        components = self.connected_components()
        INF = float('inf')

        for idx, comp in enumerate(components):
            print(f"\nComponent {idx + 1}:" if len(components) > 1 else "")
            sub_indices = [v - 1 for v in comp]  # 0-based индексы вершин
            degrees = [len(self._adjacency_list[i]) for i in sub_indices]
            print("Vertices degrees:")
            print(degrees)

            ecc = []
            for u in sub_indices:
                max_dist = max(
                    self._dist[u][v] for v in sub_indices if self._dist[u][v] != INF
                )
                ecc.append(max_dist)
            print("Eccentricity:")
            print(ecc)

            radius = min(ecc)
            diameter = max(ecc)

            central = [comp[i] for i, e in enumerate(ecc) if e == radius]
            peripheral = [comp[i] for i, e in enumerate(ecc) if e == diameter]

            print(f"R = {radius}")
            print("Central vertices:")
            print(sorted(central))
            print(f"D = {diameter}")
            print("Peripherial vertices:")
            print(sorted(peripheral))
    #3-9
    def minimum_spanning_tree_kruskal(self):
        if not self._matrix or len(self._matrix) < 2 or len(self._matrix[0]) < 2:
            raise ValueError("Некорректная матрица смежности")

        n = len(self._matrix) - 1

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if self._matrix[i][j] != self._matrix[j][i]:
                    raise ValueError("Граф ориентированный — минимальное остовное дерево не определено")

        parent = list(range(n + 1))
        rank = [0] * (n + 1)

        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]

        def union(u, v):
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            if rank[ru] < rank[rv]:
                parent[ru] = rv
            else:
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
            return True

        edges = []
        for i in range(1, n + 1):
            for j in range(i + 1, n + 1):
                if self._matrix[i][j] != 0:
                    edges.append((i, j, self._matrix[i][j]))

        edges.sort(key=lambda x: x[2])

        mst = []
        total_weight = 0
        for u, v, w in edges:
            if union(u, v):
                mst.append((u, v, w))
                total_weight += w
                if len(mst) == n - 1:
                    break


        if len(mst) < n - 1:
            raise ValueError("Граф несвязный — минимальное остовное дерево не существует")

        print("Минимальное остовное дерево (ребра):")
        for u, v, w in mst:
            print(f"{u} - {v}: {w}")
        print(f"Общий вес: {total_weight}")

        return mst, total_weight
    #5-13
    def is_bipartite(self):
        color = [-1] * self._size
        for start in range(self._size):
            if color[start] == -1:
                queue = deque([start])
                color[start] = 0
                while queue:
                    u = queue.popleft()
                    for v, w in self._adjacency_list[u]:
                        v_index = v - 1
                        if color[v_index] == -1:
                            color[v_index] = 1 - color[u]
                            queue.append(v_index)
                        elif color[v_index] == color[u]:
                            return False, None
        part1 = [i + 1 for i in range(self._size) if color[i] == 0]
        part2 = [i + 1 for i in range(self._size) if color[i] == 1]
        return True, (part1, part2)

    def maximum_bipartite_matching(self):
        is_bipartite, parts = self.is_bipartite()
        if not is_bipartite:
            return None, None

        left, right = parts
        right_set = set(right)

        adj = {u: [] for u in left}
        for u in left:
            for v, w in self._adjacency_list[u - 1]:
                if v in right_set:
                    adj[u].append(v)
            adj[u].sort()

        pair_u = {u: None for u in left}
        pair_v = {v: None for v in right}
        dist = {}

        def bfs():
            queue = deque()
            for u in left:
                if pair_u[u] is None:
                    dist[u] = 0
                    queue.append(u)
                else:
                    dist[u] = float('inf')
            dist[None] = float('inf')

            while queue:
                u = queue.popleft()
                if dist[u] < dist[None]:
                    for v in adj[u]:
                        pv = pair_v[v]
                        if pv is None:
                            dist[None] = dist[u] + 1
                        elif dist[pv] == float('inf'):
                            dist[pv] = dist[u] + 1
                            queue.append(pv)
            return dist[None] != float('inf')

        def dfs(u):
            if u is not None:
                for v in adj[u]:
                    pv = pair_v[v]
                    if pv is None or (dist[pv] == dist[u] + 1 and dfs(pv)):
                        pair_u[u] = v
                        pair_v[v] = u
                        return True
                dist[u] = float('inf')
                return False
            return True

        matching = 0
        while bfs():
            for u in sorted(left):
                if pair_u[u] is None and dfs(u):
                    matching += 1

        result = [(u, pair_u[u], None) for u in sorted(left) if pair_u[u] is not None]
        return matching, result

    #14 TODO изменить и доделать
    def ford_fulkerson(self, source, sink):
        if not self._directed:
            raise ValueError("Метод Форда-Фалкерсона применяется только к ориентированным графам.")

        source -= 1
        sink -= 1
        size = self._size

        residual = [row[:] for row in self._adjacency_matrix]
        original = [row[:] for row in self._adjacency_matrix]  # для восстановления потока
        max_flow = 0
        parent = [-1] * size

        def bfs():
            visited = [False] * size
            queue = []
            queue.append(source)
            visited[source] = True
            while queue:
                u = queue.pop(0)
                for v in range(size):
                    if not visited[v] and residual[u][v] > 0:
                        queue.append(v)
                        visited[v] = True
                        parent[v] = u
                        if v == sink:
                            return True
            return False

        while bfs():
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                v = u

            v = sink
            while v != source:
                u = parent[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                v = parent[v]

            max_flow += path_flow

        # Вычислим итоговый поток по рёбрам
        flow_result = {}
        for u in range(size):
            for v in range(size):
                if original[u][v] > 0 and residual[u][v] < original[u][v]:
                    flow_value = original[u][v] - residual[u][v]
                    flow_result[(u + 1, v + 1)] = flow_value

        return max_flow, flow_result


    #7 TODO ИЗМЕНИТЬ!
    def find_sccs_kosaraju(self):
        if not self._is_directed:
            print("Предупреждение: Алгоритм Косараджу обычно применяется к ориентированным графам.")
            print("Для неориентированного графа каждая связная компонента является SCC.")

        num_vertices = self._size
        visited = [False] * num_vertices
        finish_order_stack = []

        def dfs1(u_0_indexed):
            visited[u_0_indexed] = True
            for v_1_indexed, _ in self._adjacency_list[u_0_indexed]:
                v_0_indexed = v_1_indexed - 1
                if not visited[v_0_indexed]:
                    dfs1(v_0_indexed)
            finish_order_stack.append(u_0_indexed)

        for i in range(num_vertices):
            if not visited[i]:
                dfs1(i)

        g_transpose_adj_list = [[] for _ in range(num_vertices)]
        for u_0_indexed in range(num_vertices):
            for v_1_indexed, _ in self._adjacency_list[u_0_indexed]:
                v_0_indexed = v_1_indexed - 1
                g_transpose_adj_list[v_0_indexed].append(u_0_indexed)

        visited = [False] * num_vertices
        sccs = []

        while finish_order_stack:
            u_0_indexed = finish_order_stack.pop()
            if not visited[u_0_indexed]:
                current_scc = []

                def dfs2(v_0_indexed):
                    visited[v_0_indexed] = True
                    current_scc.append(v_0_indexed + 1)
                    for neighbor_0_indexed in g_transpose_adj_list[v_0_indexed]:
                        if not visited[neighbor_0_indexed]:
                            dfs2(neighbor_0_indexed)

                dfs2(u_0_indexed)
                sccs.append(current_scc)

        return sccs
    #11
    def Bellman_Ford_Moor(self, start):
        start -= 1
        dist = [float('inf')] * self._size
        dist[start] = 0

        edges = []
        for u in range(self._size):
            for v, w in self._adjacency_list[u]:
                edges.append((u, v - 1, w))

        for _ in range(self._size - 1):
            for u, v, w in edges:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in edges:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                print("Graph contains a negative-weight cycle")
                return

        result = {i + 1: (int(dist[i]) if dist[i] != float('inf') else float('inf')) for i in range(self._size)}
        print(f"Shotest paths lengths from {start + 1}:\n{result}")
        return result

class Map:
    def __init__(self, filepath: str):
        with open(filepath, 'r') as f:
            lines = f.read().strip().split('\n')
        if len(lines[0].split()) == 2 and all(x.isdigit() for x in lines[0].split()):
            lines = lines[1:]

        self._matrix = [list(map(int, line.split())) for line in lines]
        self._rows = len(self._matrix)
        self._cols = len(self._matrix[0]) if self._rows else 0

    def __getitem__(self, pos):
        i, j = pos
        return self._matrix[i][j]

    def size(self):
        return self._rows, self._cols

    def neighbors(self, i, j):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # вверх, вниз, влево, вправо
        result = []
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < self._rows and 0 <= nj < self._cols:
                if self._matrix[ni][nj] > 0:
                    result.append((ni, nj))
        return result

    @staticmethod
    def heuristic_manhattan(i, j, s, p):
        return abs(s - i) + abs(p - j)

    @staticmethod
    def heuristic_chebyshev(i, j, s, p):
        return max(abs(s - i), abs(p - j))

    @staticmethod
    def heuristic_euclidean(i, j, s, p):
        return ((s - i) ** 2 + (p - j) ** 2) ** 0.5

    @staticmethod
    def distance(aij, akl, i, j, k, l):
        return abs(k - i) + abs(l - j) + abs(akl - aij)

    #6
    def find_path_bfs(self, start, goal):
        rows, cols = self.size()
        queue = deque([start])
        came_from = {start: None}

        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path

            i, j = current
            for ni, nj in self.neighbors(i, j):
                if (ni, nj) not in came_from:
                    came_from[(ni, nj)] = current
                    queue.append((ni, nj))

        return None
    #12
    def find_path_a_star(self, start, goal, heuristic):
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {start: None}
        g_score = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                total_cost = g_score[current]
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                path.reverse()

                print(f"{total_cost} - length of path between {start} and {goal} points.")
                print("Path:")
                print(path)
                return path

            i, j = current
            for ni, nj in self.neighbors(i, j):
                tentative_g = g_score[current] + self.distance(self[i, j], self[ni, nj], i, j, ni, nj)

                if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                    g_score[(ni, nj)] = tentative_g
                    f_score = tentative_g + heuristic(ni, nj, goal[0], goal[1])
                    heapq.heappush(open_set, (f_score, (ni, nj)))
                    came_from[(ni, nj)] = current

        print("No path found")
        return None
    def find_path_a_star_ecm(self, start, goal):
        print('Manhattan heuristic:')
        print(self.find_path_a_star(start, goal, Map.heuristic_manhattan))
        print('-' * 30)
        print('Chebyshev heuristic:')
        print(self.find_path_a_star(start, goal, Map.heuristic_chebyshev))
        print('-' * 30)
        print('Euclidean heuristic:')
        print(self.find_path_a_star(start, goal, Map.heuristic_euclidean))





obj = Graph(r"C:\Users\angel\Downloads\list_of_adjacency_t1_008.txt", 'list_of_adjacency')
obj.connected_components()


