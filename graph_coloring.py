import argparse
import os
import random

def read_obj_file(file_path):
    vertices = []
    triangles = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):  # vertex
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):  # face (triangle)
                face_vertices = [int(v.split('/')[0]) - 1 for v in line.strip().split()[1:]]
                triangles.append(face_vertices)

    return vertices, triangles

def graph_coloring(constraints):
    constraints_groups = []
    done = [False] * len(constraints)
    num_done = 0
    while num_done < len(constraints):
        constraints_groups.append([])
        mark = [False] * len(vertices)
        indices = list(range(len(constraints)))
        # random.shuffle(indices)
        for i in indices:
            if not done[i]:
                if not mark[constraints[i][0]] and not mark[constraints[i][1]]:
                    mark[constraints[i][0]] = True
                    mark[constraints[i][1]] = True
                    done[i] = True
                    num_done += 1
                    constraints_groups[-1].append(i)
    constraints_groups.sort(key=lambda a: -len(a))
    print("Len =", len(constraints_groups))
    for i in range(len(constraints_groups)):
        print(len(constraints_groups[i]))
    return constraints_groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read .obj file and extract vertices and triangles.')
    parser.add_argument('obj_file_path', type=str, help='Path to the .obj file')

    args = parser.parse_args()

    obj_file_path = args.obj_file_path
    vertices, triangles = read_obj_file(obj_file_path)

    stretching_constraints = []
    bending_constraints = []

    # stretching contraints
    for tri in triangles:
        stretching_constraints.append((tri[0], tri[1]))
        stretching_constraints.append((tri[0], tri[2]))
        stretching_constraints.append((tri[1], tri[2]))

    # bending contraints
    edges_tri = []
    for tri in triangles:
        tri.sort()
        edges_tri.append((tri[0], tri[1], tri[2]))
        edges_tri.append((tri[0], tri[2], tri[1]))
        edges_tri.append((tri[1], tri[2], tri[0]))
    edges_tri.sort()
    i = 0
    while i < len(edges_tri):
        i += 1
        if i < len(edges_tri) and edges_tri[i][0] == edges_tri[i - 1][0] and edges_tri[i][1] == edges_tri[i - 1][1]:
            assert(edges_tri[i][2] != edges_tri[i - 1][2])
            if edges_tri[i][2] < edges_tri[i - 1][2]:
                bending_constraints.append((edges_tri[i][2], edges_tri[i - 1][2]))
            else:
                bending_constraints.append((edges_tri[i - 1][2], edges_tri[i][2]))
            i += 1
    print("Number of constraints =", len(stretching_constraints) + len(bending_constraints))

    # stretching_groups = graph_coloring(stretching_constraints)
    # bending_groups = graph_coloring(bending_constraints)
    constraints = stretching_constraints + bending_constraints
    constraints_groups = graph_coloring(constraints)

    output_file = obj_file_path.split('.')[0] + "_constraints_groups"
    while os.path.exists(output_file + ".txt"):
        output_file += "_1"

    with open(output_file + ".txt", "w") as file:
        file.write(str(len(vertices)) + " " + str(len(triangles)) + " " + str(len(stretching_constraints) + len(bending_constraints)) + "\n")
        file.write(str(len(constraints_groups)) + "\n")
        for i in range(len(constraints_groups)):
            file.write(str(len(constraints_groups[i])) + "\n")
            for c in constraints_groups[i]:
                file.write(str(constraints[c][0]) + " " + str(constraints[c][1]) + "\n")
        # file.write(str(len(stretching_groups)) + "\n")
        # for i in range(len(stretching_groups)):
        #     file.write(str(len(stretching_groups[i])) + "\n")
        #     for c in stretching_groups[i]:
        #         file.write(str(stretching_constraints[c][0]) + " " + str(stretching_constraints[c][1]) + "\n")
        # file.write(str(len(bending_groups)) + "\n")
        # for i in range(len(bending_groups)):
        #     file.write(str(len(bending_groups[i])) + "\n")
        #     for c in bending_groups[i]:
        #         file.write(str(bending_constraints[c][0]) + " " + str(bending_constraints[c][1]) + "\n")
