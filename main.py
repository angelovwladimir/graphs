from idlelib.pyshell import usage_msg

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
            with open(path, 'r') as file:
                tops_cnt = int(file.readline())
                self._size = tops_cnt
                str_number = 1
                graph = {}
                edges = []
                self.is_direct = False
                for string in file:
                    string = list(map(int, string.split()))
                    if len(string) % 2 == 0:
                        graph[str_number] = string
                    else:
                        string += [1]
                        graph[str_number] = string

                    for i in range(len(string)):
                        if i % 2 == 0:
                            edges.append((str_number, string[i], string[i + 1]))
                        else:
                            continue
                    '''
                    edges список ребер в виде начало - конец - вес
                    '''
                    str_number += 1

                for u, v, weight in edges:
                    if (v, u, weight) not in edges:
                        self.is_direct = True
                '''
                строим матрицу смежности
                '''
                matrix = [[0 for _ in range(tops_cnt + 1)] for _ in range(tops_cnt + 1)]

                for point in graph:
                    for i in range(len(graph[point]) - 1):
                        if i % 2 == 0:
                            matrix[point][graph[point][i]] = graph[point][i + 1]

                self.matrix = matrix
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

obj = Graph(r"C:\Users\vladimir\Downloads\list_of_adjacency_t1_004.txt", 'list_of_adjacency')
print(obj.adjacency_list(6))
print(obj.matrix)
print(obj.list_of_edges())
print(obj.list_of_edges(2))
print(obj.weight(6, 1))



