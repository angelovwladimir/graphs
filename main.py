import heapq
import math
from collections import deque


class Graph:
    def __init__(self, file_path, file_type):
        self.graph = {}
        self.directed = False
        self.size = 0
        self._read_graph(file_path, file_type)

    def _read_graph(self, file_path, file_type):
        try:
            with open(file_path, 'r') as f:
                content = [line.strip() for line in f if line.strip()]

            if not content:
                raise ValueError("Файл пуст")

            if file_type == 'matrix':
                self._read_adjacency_matrix(content)
            elif file_type == 'list':
                self._read_adjacency_list(content)
            elif file_type == 'edges':
                self._read_edge_list(content)
            else:
                raise ValueError(f"Неизвестный тип файла: {file_type}")

        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден")
        except Exception as e:
            raise ValueError(f"Ошибка чтения файла: {str(e)}")

    def _read_adjacency_matrix(self, lines):
        self.size = len(lines)
        if self.size == 0:
            return

        # Проверка на квадратность матрицы
        for line in lines:
            if len(line.split()) != self.size:
                raise ValueError("Матрица смежности должна быть квадратной")

        # Проверка ориентированности
        self.directed = False
        for i in range(self.size):
            row = list(map(int, lines[i].split()))
            for j in range(i + 1, self.size):
                if row[j] != int(lines[j].split()[i]):
                    self.directed = True
                    break
            if self.directed:
                break

        # Заполнение графа
        for i in range(self.size):
            row = list(map(int, lines[i].split()))
            self.graph[i] = []
            for j in range(self.size):
                weight = row[j]
                if weight != 0:
                    if not self.directed and i == j and weight != 0:
                        raise ValueError("Петли не допускаются в неориентированном графе")
                    self.graph[i].append((j, weight))

    def _read_adjacency_list(self, lines):
        self.size = len(lines)
        self.directed = False

        # Проверка ориентированности
        for i in range(self.size):
            parts = lines[i].split()
            for neighbor in parts:
                if ':' in neighbor:
                    node, _ = neighbor.split(':')
                else:
                    node = neighbor
                if not self.directed and int(node) < i:
                    # Проверяем есть ли обратное ребро
                    found = False
                    for item in lines[int(node)].split():
                        if ':' in item:
                            n, _ = item.split(':')
                        else:
                            n = item
                        if int(n) == i:
                            found = True
                            break
                    if not found:
                        self.directed = True
                        break
            if self.directed:
                break

        # Заполнение графа
        for i in range(self.size):
            self.graph[i] = []
            parts = lines[i].split()
            for neighbor in parts:
                if ':' in neighbor:
                    node, weight = map(int, neighbor.split(':'))
                else:
                    node, weight = int(neighbor), 1
                self.graph[i].append((node, weight))

    def _read_edge_list(self, lines):
        nodes = set()
        edges = []

        for line in lines:
            parts = line.split()
            if len(parts) < 2:
                continue

            u = int(parts[0])
            v = int(parts[1])
            weight = int(parts[2]) if len(parts) > 2 else 1

            nodes.add(u)
            nodes.add(v)
            edges.append((u, v, weight))

        self.size = len(nodes)
        if self.size == 0:
            return

        # Проверка ориентированности
        self.directed = False
        edge_set = {(u, v, w) for u, v, w in edges}
        for u, v, w in edges:
            if (v, u, w) not in edge_set:
                self.directed = True
                break

        # Заполнение графа
        for i in range(self.size):
            self.graph[i] = []

        for u, v, weight in edges:
            self.graph[u].append((v, weight))
            if not self.directed:
                self.graph[v].append((u, weight))

    def size(self):
        return self.size

    def weight(self, u, v):
        if u < 0 or u >= self.size or v < 0 or v >= self.size:
            raise IndexError("Неверные индексы вершин")

        for node, weight in self.graph.get(u, []):
            if node == v:
                return weight
        raise ValueError(f"Ребро между {u} и {v} не существует")

    def is_edge(self, u, v):
        if u < 0 or u >= self.size or v < 0 or v >= self.size:
            raise IndexError("Неверные индексы вершин")

        for node, _ in self.graph.get(u, []):
            if node == v:
                return True
        return False

    def adjacency_matrix(self):
        matrix = [[0] * self.size for _ in range(self.size)]
        for u in self.graph:
            for v, weight in self.graph[u]:
                matrix[u][v] = weight
        return matrix

    def adjacency_list(self, u):
        if u < 0 or u >= self.size:
            raise IndexError("Неверный индекс вершины")
        return [node for node, _ in self.graph.get(u, [])]

    def list_of_edges(self, u=None):
        if u is not None:
            if u < 0 or u >= self.size:
                raise IndexError("Неверный индекс вершины")
            return [(u, v, w) for v, w in self.graph.get(u, [])]

        edges = []
        if self.directed:
            for u in range(self.size):
                for v, w in self.graph.get(u, []):
                    edges.append((u, v, w))
        else:
            added = set()
            for u in range(self.size):
                for v, w in self.graph.get(u, []):
                    if (u, v) not in added and (v, u) not in added:
                        edges.append((u, v, w))
                        added.add((u, v))
        return edges

    def is_directed(self):
        return self.directed

    def find_connected_components(self):
        if self.size == 0:
            return []

        visited = [False] * self.size
        components = []

        for node in range(self.size):
            if not visited[node]:
                component = []
                stack = [node]
                visited[node] = True

                while stack:
                    current = stack.pop()
                    component.append(current)

                    for neighbor, _ in self.graph.get(current, []):
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)

                components.append(component)

        return components

    def find_weakly_connected_components(self):
        if not self.directed:
            return self.find_connected_components()

        # Создаем неориентированную версию графа
        undirected_graph = {i: [] for i in range(self.size)}
        for u in range(self.size):
            for v, w in self.graph.get(u, []):
                undirected_graph[u].append((v, w))
                undirected_graph[v].append((u, w))

        # Используем стандартный алгоритм для компонент связности
        visited = [False] * self.size
        components = []

        for node in range(self.size):
            if not visited[node]:
                component = []
                stack = [node]
                visited[node] = True

                while stack:
                    current = stack.pop()
                    component.append(current)

                    for neighbor, _ in undirected_graph.get(current, []):
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)

                components.append(component)

        return components

    def find_bridges(self):
        if self.directed:
            raise ValueError("Мосты ищутся только для неориентированных графов")

        visited = [False] * self.size
        disc = [float('inf')] * self.size
        low = [float('inf')] * self.size
        parent = [-1] * self.size
        bridges = []
        time = 0

        def dfs(u):
            nonlocal time
            visited[u] = True
            disc[u] = low[u] = time
            time += 1

            for v, _ in self.graph.get(u, []):
                if not visited[v]:
                    parent[v] = u
                    dfs(v)

                    low[u] = min(low[u], low[v])

                    if low[v] > disc[u]:
                        bridges.append((u, v))
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(self.size):
            if not visited[i]:
                dfs(i)

        return bridges

    def find_articulation_points(self):
        if self.directed:
            raise ValueError("Шарниры ищутся только для неориентированных графов")

        visited = [False] * self.size
        disc = [float('inf')] * self.size
        low = [float('inf')] * self.size
        parent = [-1] * self.size
        ap = [False] * self.size
        time = 0

        def dfs(u):
            nonlocal time
            children = 0
            visited[u] = True
            disc[u] = low[u] = time
            time += 1

            for v, _ in self.graph.get(u, []):
                if not visited[v]:
                    parent[v] = u
                    children += 1
                    dfs(v)

                    low[u] = min(low[u], low[v])

                    if parent[u] == -1 and children >= 2:
                        ap[u] = True

                    if parent[u] != -1 and low[v] >= disc[u]:
                        ap[u] = True
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])

        for i in range(self.size):
            if not visited[i]:
                dfs(i)

        return [i for i in range(self.size) if ap[i]]

    def spanning_tree_bfs(self, start=0):
        if start < 0 or start >= self.size:
            raise IndexError("Неверный индекс стартовой вершины")

        visited = [False] * self.size
        queue = deque([start])
        visited[start] = True
        tree_edges = []

        while queue:
            u = queue.popleft()
            for v, _ in self.graph.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    tree_edges.append((u, v))
                    queue.append(v)

        return tree_edges

    def spanning_tree_dfs(self, start=0):
        if start < 0 or start >= self.size:
            raise IndexError("Неверный индекс стартовой вершины")

        visited = [False] * self.size
        stack = [start]
        visited[start] = True
        tree_edges = []

        while stack:
            u = stack.pop()
            for v, _ in self.graph.get(u, []):
                if not visited[v]:
                    visited[v] = True
                    tree_edges.append((u, v))
                    stack.append(v)

        return tree_edges

    def floyd_warshall(self):
        dist = [[float('inf')] * self.size for _ in range(self.size)]

        for u in range(self.size):
            dist[u][u] = 0
            for v, w in self.graph.get(u, []):
                dist[u][v] = w

        for k in range(self.size):
            for i in range(self.size):
                for j in range(self.size):
                    if dist[i][k] != float('inf') and dist[k][j] != float('inf'):
                        if dist[i][j] > dist[i][k] + dist[k][j]:
                            dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    def vertex_degrees(self):
        if not self.directed:
            return [len(self.graph.get(i, [])) for i in range(self.size)]
        else:
            in_degree = [0] * self.size
            out_degree = [len(self.graph.get(i, [])) for i in range(self.size)]

            for u in range(self.size):
                for v, _ in self.graph.get(u, []):
                    in_degree[v] += 1

            return (in_degree, out_degree)

    def eccentricities(self, dist_matrix=None):
        if dist_matrix is None:
            dist_matrix = self.floyd_warshall()

        ecc = []
        for row in dist_matrix:
            max_dist = max(d for d in row if d != float('inf'))
            ecc.append(max_dist if max_dist != float('inf') else 0)

        return ecc

    def diameter_and_periphery(self, ecc=None):
        if ecc is None:
            ecc = self.eccentricities()

        diameter = max(ecc)
        periphery = [i for i, e in enumerate(ecc) if e == diameter]

        return diameter, periphery

    def radius_and_center(self, ecc=None):
        if ecc is None:
            ecc = self.eccentricities()

        radius = min(ecc)
        center = [i for i, e in enumerate(ecc) if e == radius]

        return radius, center

    def is_bipartite(self):
        color = [-1] * self.size
        queue = deque()
        bipartition = ([], [])

        for i in range(self.size):
            if color[i] == -1:
                queue.append(i)
                color[i] = 0
                bipartition[0].append(i)

                while queue:
                    u = queue.popleft()

                    for v, _ in self.graph.get(u, []):
                        if color[v] == -1:
                            color[v] = color[u] ^ 1
                            if color[v] == 0:
                                bipartition[0].append(v)
                            else:
                                bipartition[1].append(v)
                            queue.append(v)
                        elif color[v] == color[u]:
                            return (False, None)

        return (True, bipartition)


class Map:
    def __init__(self, file_path):
        try:
            with open(file_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                raise ValueError("Файл карты пуст")

            self.map = []
            cols = len(lines[0].split())

            for line in lines:
                row = list(map(int, line.split()))
                if len(row) != cols:
                    raise ValueError("Все строки карты должны иметь одинаковую длину")
                self.map.append(row)

            self.rows = len(self.map)
            self.cols = cols

        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {file_path} не найден")
        except Exception as e:
            raise ValueError(f"Ошибка чтения карты: {str(e)}")

    def __getitem__(self, indices):
        i, j = indices
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError("Координаты за пределами карты")
        return self.map[i][j]

    def size(self):
        return (self.rows, self.cols)

    def neighbors(self, i, j):
        if not (0 <= i < self.rows and 0 <= j < self.cols):
            raise IndexError("Координаты за пределами карты")

        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.rows and 0 <= nj < self.cols:
                if self.map[ni][nj] != 0:
                    neighbors.append((ni, nj))
        return neighbors

    def find_path(self, start, end, heuristic='manhattan'):
        if not isinstance(start, tuple) or len(start) != 2:
            raise ValueError("Начальная точка должна быть кортежем (i,j)")
        if not isinstance(end, tuple) or len(end) != 2:
            raise ValueError("Конечная точка должна быть кортежем (i,j)")

        si, sj = start
        ei, ej = end

        if not (0 <= si < self.rows and 0 <= sj < self.cols):
            raise IndexError("Начальная точка за пределами карты")
        if not (0 <= ei < self.rows and 0 <= ej < self.cols):
            raise IndexError("Конечная точка за пределами карты")
        if self.map[si][sj] == 0:
            raise ValueError("Начальная точка непроходима")
        if self.map[ei][ej] == 0:
            raise ValueError("Конечная точка непроходима")
        if start == end:
            return [start]

        # Определяем эвристическую функцию
        if heuristic.lower() == 'manhattan':
            def h(ij, sp):
                return abs(ij[0] - sp[0]) + abs(ij[1] - sp[1])
        elif heuristic.lower() == 'chebyshev':
            def h(ij, sp):
                return max(abs(ij[0] - sp[0]), abs(ij[1] - sp[1]))
        elif heuristic.lower() == 'euclidean':
            def h(ij, sp):
                return math.sqrt((ij[0] - sp[0]) ** 2 + (ij[1] - sp[1]) ** 2)
        else:
            raise ValueError("Неизвестная эвристика. Допустимые значения: 'manhattan', 'chebyshev', 'euclidean'")

        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: h(start, end)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self.neighbors(*current):
                tentative_g_score = g_score[current] + self._distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + h(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _distance(self, ij, kl):
        i, j = ij
        k, l = kl
        return abs(k - i) + abs(l - j) + abs(self.map[k][l] - self.map[i][j])
