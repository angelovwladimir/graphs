from idlelib.pyshell import usage_msg
from collections import deque
from collections import defaultdict
from pprint import pprint
import sys
class Graph:
    def __init__(self, path, type_file):
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
            self.matrix = [[0] * (self._size + 1) for _ in range(self._size + 1)]
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
                        self.matrix[i + 1][v] = w

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
                self.is_direct = False
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
                        self.is_direct = True
                '''
                стороим матрицу смежности
                '''
                matrix = [[0 for _ in range(tops_cnt + 1)] for _ in range(tops_cnt + 1)]
                for string in graph:
                    u = string[0]
                    v = string[1]
                    weight = string[2]
                    matrix[u][v] = weight
            self.matrix = matrix


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
                self.is_direct = False
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
                            self.is_direct = True
            self.matrix = matrix



        except FileNotFoundError:
            print("Файл не найден!")
        except PermissionError:
            print("Нет прав доступа к файлу!")
        except IOError as e:
            print(f"Произошла ошибка ввода/вывода: {e}")
        except Exception as e:
            print(f"Неизвестная ошибка: {e}")

    def is_directed(self):
        return 'Graph is directed' if self.is_directed else 'Graph is not directed'

    def size(self):
        return self._size

    def adjacency_matrix(self):
        return self.matrix

    def adjacency_list(self, top_number):
        matrix = self.matrix
        row = matrix[top_number]
        adjacency_tops = []
        for i in range(len(row)):
            if row[i] != 0:
                adjacency_tops.append(i)
        return adjacency_tops

    def list_of_edges(self, top_number = None):
        if top_number is None:
            matrix = self.matrix
            edges = []
            for i in range(1, self.size() + 1):
                for j in range(1, self.size() + 1):
                    if matrix[i][j] != 0:
                        edges.append([i, j])
            return edges
        else:
            matrix = self.matrix
            edges = []
            for i in range(1, self.size() + 1):
                if matrix[top_number][i] != 0:
                    edges.append([top_number, i])
                    '''
                if matrix[i][top_number] != 0:
                    edges.append([i, top_number])
                    '''
            return edges

    def is_edge(self, start, end):
        matrix = self.matrix
        if matrix[start][end] != 0:
            return True
        else:
            return False

    def weight(self, start, end):
        if self.is_edge(start, end):
            return self.matrix[start][end]
        else:
            return "Edge is not found"

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

        if self.is_directed():
            weak_components = weak_components_directed(self.matrix)
            if len(weak_components) == 1:
                if is_strongly_connected(self.matrix):
                    print("Graph is connected")
                else:
                    print("Graph is weakly connected")
            else:
                print("Graph is not connected")
            return weak_components
        else:
            components = dfs_components_undirected(self.matrix)
            if len(components) == 1:
                print("Graph is connected")
            else:
                print("Graph is not connected")
            return components

    def find_bridges(self, adj_matrix):
        n = len(adj_matrix)
        adj = [[] for _ in range(n)]

        # Преобразуем матрицу в список смежности (неориентированный граф)
        for i in range(1, n):
            for j in range(i + 1, n):  # Проверяем только верхний треугольник
                if adj_matrix[i][j] != 0:
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

    def floyd_warshall_ignored_first(self, graph):
        n = len(graph)
        dist = [[0] * (n - 1) for _ in range(n - 1)]  # Игнорируем первую строку и столбец

        # Инициализация матрицы расстояний (без первой строки и столбца)
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    dist[i - 1][j - 1] = 0
                elif graph[i][j] != 0:
                    dist[i - 1][j - 1] = graph[i][j]
                else:
                    dist[i - 1][j - 1] = sys.maxsize

        # Алгоритм Флойда-Уоршелла (для уменьшенной матрицы)
        for k in range(n - 1):
            for i in range(n - 1):
                for j in range(n - 1):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    def calculate_degrees_ignored_first(self, graph):
        n = len(graph)
        degrees = [0] * (n - 1)  # Игнорируем первую вершину

        for i in range(1, n):
            for j in range(1, n):
                if graph[i][j] != 0 and i != j:
                    degrees[i - 1] += 1
        return degrees

    def analyze_graph_ignored_first(self, dist_matrix):
        n = len(dist_matrix)

        eccentricities = [max(row) for row in dist_matrix]
        diameter = max(eccentricities)
        radius = min(eccentricities)
        peripheral = [i + 1 for i, e in enumerate(eccentricities) if e == diameter]  # +1, т.к. вершины нумеруются с 1
        central = [i + 1 for i, e in enumerate(eccentricities) if e == radius]

        return {
            'eccentricities': eccentricities,
            'diameter': diameter,
            'peripheral_vertices': peripheral,
            'radius': radius,
            'central_vertices': central
        }



obj1 = Graph(r"C:\Users\vladimir\Downloads\matrix_t1_016.txt", 'matrix')
obj2 = Graph(r"C:\Users\vladimir\Downloads\list_of_adjacency_t1_016.txt", 'list_of_adjacency')
obj3 = Graph(r"C:\Users\vladimir\Downloads\list_of_edges_t1_016.txt", 'list_of_edges')
#print(obj.adjacency_list(6))
# pprint(obj1.matrix)
# print('-' * 30)
# pprint(obj2.matrix)
# print('-' * 30)
# pprint(obj3.matrix)
# print('-' * 30)
# print(obj1.matrix == obj3.matrix == obj2.matrix)
#print(obj.list_of_edges())
#print(obj.list_of_edges(2))
#print(obj.weight(6, 1))
#print(obj.components())
#print(obj.find_bridges(obj.matrix))
print(obj1.components())
print(obj2.components())
print(obj3.components())


