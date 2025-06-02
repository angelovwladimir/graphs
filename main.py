from idlelib.pyshell import usage_msg
from collections import deque
from collections import defaultdict
from pprint import pprint
import sys
class Graph:
    def __init__(self, path, type_file):
        self._matrix = None
        self._is_directed = None
        self._size = None
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
            with open(path, 'r') as file:
                graph = []
                self._is_direct = False
                tops_cnt = int(file.readline())
                self._size = tops_cnt
                for string in file:
                    string = list(map(int, string.split()))
                    if len(string) == 3:
                        graph.append([string[0], string[1], string[2]])
                    elif len(string) == 2:
                        graph.append([string[0], string[1], 1])

                for u, v, weight in graph:
                    if [v, u, weight] not in graph:
                        self._is_direct = True
                '''
                стороим матрицу смежности
                '''
                matrix = [[0 for _ in range(tops_cnt + 1)] for _ in range(tops_cnt + 1)]
                for string in graph:
                    u = string[0]
                    v = string[1]
                    weight = string[2]
                    matrix[u][v] = weight
            self._matrix = matrix


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
            with open(path, 'r') as file:
                self._is_direct = False
                tops_cnt = int(file.readline())
                self._size = tops_cnt
                '''строим матрицу смежности'''
                matrix = [[0 for _ in range(tops_cnt + 1)] for _ in range(tops_cnt + 1)]
                row_number = 1
                for string in file:
                    string = list(map(int, string.split()))
                    col_number = 1
                    for column in string:
                        matrix[row_number][col_number] = column
                        col_number += 1
                    row_number += 1
                for i in range(1, tops_cnt + 1):
                    for j in range(1, tops_cnt + 1):
                        if matrix[i][j] != matrix[j][i]:
                            self._is_direct = True
            self._matrix = matrix



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
    def components(self):
        def dfs_components_undirected(adj_matrix):
            n = len(adj_matrix)
            visited = [False] * n
            components = []

            for node in range(1, n):
                if not visited[node]:
                    stack = [node]
                    visited[node] = True
                    component = []

                    while stack:
                        current = stack.pop()
                        component.append(current)

                        for neighbor in range(1, n):
                            if adj_matrix[current][neighbor] and not visited[neighbor]:
                                visited[neighbor] = True
                                stack.append(neighbor)

                    components.append(component)

            return components

        def weak_components_directed(adj_matrix):
            n = len(adj_matrix)
            undirected_matrix = [[False] * n for _ in range(n)]

            for i in range(1, n):
                for j in range(1, n):
                    if adj_matrix[i][j] or adj_matrix[j][i]:
                        undirected_matrix[i][j] = True

            return dfs_components_undirected(undirected_matrix)

        def is_strongly_connected(adj_matrix):
            n = len(adj_matrix)
            if n <= 1:
                return True

            def dfs(start):
                visited = [False] * n
                stack = [start]
                visited[start] = True
                count = 0

                while stack:
                    current = stack.pop()
                    count += 1

                    for neighbor in range(1, n):
                        if adj_matrix[current][neighbor] and not visited[neighbor]:
                            visited[neighbor] = True
                            stack.append(neighbor)

                return count == n - 1  # Минус 1, т.к. нулевая вершина не учитывается

            # Проверяем, что из вершины 1 достижимы все остальные
            if not dfs(1):
                return False

            # Транспонируем матрицу для проверки обратных путей
            transposed = [[False] * n for _ in range(n)]
            for i in range(1, n):
                for j in range(1, n):
                    transposed[j][i] = adj_matrix[i][j]

            # Восстанавливаем исходную матрицу для DFS
            global temp_matrix
            temp_matrix = adj_matrix
            adj_matrix = transposed

            # Проверяем, что в транспонированном графе из вершины 1 достижимы все
            if not dfs(1):
                return False

            return True

        if self._is_directed:
            weak_components = weak_components_directed(self._matrix)
            if len(weak_components) == 1:
                if is_strongly_connected(self._matrix):
                    print("Graph is connected")
                else:
                    print("Graph is weakly connected")
            else:
                print("Graph is not connected")
            return weak_components
        else:
            components = dfs_components_undirected(self._matrix)
            if len(components) == 1:
                print("Graph is connected")
            else:
                print("Graph is not connected")
            return components
    #2
    def find_bridges(self):
        n = len(self._matrix)
        adj = [[] for _ in range(n)]

        # Преобразуем матрицу в список смежности (неориентированный граф)
        for i in range(1, n):
            for j in range(i + 1, n):  # Проверяем только верхний треугольник
                if self._matrix[i][j] != 0:
                    adj[i].append(j)
                    adj[j].append(i)

        tin = [-1] * n
        low = [-1] * n
        visited = [False] * n
        bridges = []
        timer = 0

        def dfs(u, parent=-1):
            nonlocal timer
            visited[u] = True
            tin[u] = low[u] = timer
            timer += 1

            for v in adj[u]:
                if v == parent:  # Пропускаем родителя
                    continue
                if visited[v]:  # Обратное ребро
                    low[u] = min(low[u], tin[v])
                else:  # Ребро DFS-дерева
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    # Проверка на мост
                    if low[v] > tin[u]:
                        bridges.append((u, v))

        for i in range(1, n):
            if not visited[i]:
                dfs(i)

        return bridges

    def find_articulation_points(self):
        n = len(self._matrix)
        adj = [[] for _ in range(n)]

        for i in range(1, n):
            for j in range(i + 1, n):
                if self._matrix[i][j] != 0:
                    adj[i].append(j)
                    adj[j].append(i)

        tin = [-1] * n
        low = [-1] * n
        visited = [False] * n
        is_articulation = [False] * n
        timer = 0

        def dfs(u, parent=-1):
            nonlocal timer
            visited[u] = True
            tin[u] = low[u] = timer
            timer += 1
            children = 0

            for v in adj[u]:
                if v == parent:
                    continue
                if visited[v]:
                    low[u] = min(low[u], tin[v])
                else:
                    dfs(v, u)
                    low[u] = min(low[u], low[v])
                    # Проверка на точку сочленения
                    if low[v] >= tin[u] and parent != -1:
                        is_articulation[u] = True
                    children += 1


            if parent == -1 and children > 1:
                is_articulation[u] = True

        for i in range(1, n):
            if not visited[i]:
                dfs(i)

        articulation_points = [i for i in range(1, n) if is_articulation[i]]

        return articulation_points
    #4
    def floyd_warshall(self):
        n = len(self._matrix)
        dist = [[0] * (n - 1) for _ in range(n - 1)]


        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    dist[i - 1][j - 1] = 0
                elif self._matrix[i][j] != 0:
                    dist[i - 1][j - 1] = self._matrix[i][j]
                else:
                    dist[i - 1][j - 1] = sys.maxsize

        for k in range(n - 1):
            for i in range(n - 1):
                for j in range(n - 1):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    def calculate(self):
        n = len(self._matrix)
        degrees = [0] * (n - 1)

        for i in range(1, n):
            for j in range(1, n):
                if self._matrix[i][j] != 0 and i != j:
                    degrees[i - 1] += 1
        return degrees

    def analyze_graph(self):
        dist = self.floyd_warshall()
        n = len(dist)

        # Эксцентриситеты вершин (максимальные расстояния от каждой вершины)
        eccentricities = [max(row) for row in dist]

        # Диаметр графа (максимальный эксцентриситет)
        diameter = max(eccentricities)

        # Радиус графа (минимальный эксцентриситет)
        radius = min(eccentricities)

        # Периферийные вершины (с эксцентриситетом = диаметру)
        peripheral = [i + 1 for i, e in enumerate(eccentricities) if e == diameter]

        # Центральные вершины (с эксцентриситетом = радиусу)
        central = [i + 1 for i, e in enumerate(eccentricities) if e == radius]

        return {
            'degrees': self.calculate(),
            'eccentricities': eccentricities,
            'diameter': diameter,
            'peripheral_vertices': peripheral,
            'radius': radius,
            'central_vertices': central
        }
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


    #переделать
    def check_bipartite_and_print(self, adj_matrix):
        if not adj_matrix or len(adj_matrix) < 2 or len(adj_matrix[0]) < 2:
            print("Ошибка: некорректная матрица смежности")
            return False

        n = len(adj_matrix) - 1
        color = [-1] * (n + 1)
        partition_A = []
        partition_B = []
        is_bipartite = True

        for start in range(1, n + 1):
            if color[start] == -1:
                queue = deque([start])
                color[start] = 0
                partition_A.append(start)
                while queue:
                    u = queue.popleft()

                    for v in range(1, n + 1):
                        if adj_matrix[u][v] != 0:
                            if color[v] == -1:
                                color[v] = color[u] ^ 1
                                if color[v] == 0:
                                    partition_A.append(v)
                                else:
                                    partition_B.append(v)
                                queue.append(v)
                            elif color[v] == color[u]:
                                print("Граф НЕ является двудольным!")
                                return False

        partition_A = sorted(list(set(partition_A)))
        partition_B = sorted(list(set(partition_B)))

        print("\nПроверка завершена успешно. Граф является двудольным!")
        print(f"Доля A: {partition_A}")
        print(f"Доля B: {partition_B}")
        return True









