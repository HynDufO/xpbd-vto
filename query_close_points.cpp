#include <bits/stdc++.h>

using namespace std;

vector<array<double, 3>> vertices;
vector<vector<int>> triangles;

void readOBJFile(const string &file_path) {
    vertices.clear();
    triangles.clear();
    ifstream file(file_path);
    if (!file.is_open()) {
        cerr << "Error: Couldn't open file." << endl;
        return;
    }

    string line;
    while (getline(file, line)) {
        if (line.find("v ") == 0) { // vertex
            istringstream iss(line);
            char prefix;
            float x, y, z;
            iss >> prefix >> x >> y >> z;
            vertices.push_back({x, y, z});
        } else if (line.find("f ") == 0) { // face (triangle)
            istringstream iss(line);
            char prefix;
            vector<int> face_vertices;
            iss >> prefix;
            int vertex_index;
            while (iss >> vertex_index) {
                face_vertices.push_back(vertex_index - 1);
                iss.ignore(256, ' '); // ignore the rest
            }
            triangles.push_back(face_vertices);
        }
    }
}

double getDist2(double x, double y, double z, double x1, double y1, double z1) {
    return (x - x1) * (x - x1) + (y - y1) * (y - y1) * 0.5 + (z - z1) * (z - z1);
}

void queryAllCloseBodyParticles(const vector<int> &triIds,
                                const vector<double> &triPos,
                                const vector<double> &pos, int numPointsEach) {
    int num = 0;
    int numTris = triIds.size() / 3;
    int numPoints = pos.size();
    vector<int> adjIds, adjIdsTri;
    adjIds.resize(numPointsEach * numTris);
    adjIdsTri.resize(numPointsEach * numTris);

    for (int iTri = 0; iTri < numTris; ++iTri) {
        if (iTri % 1000 == 0) {
            cerr << "iTri = " << iTri << '\n';
        }
        priority_queue<pair<double, int>> heap;

        for (int iPoint = 0; iPoint < numPoints; ++iPoint) {
            double sumDist2 = 0;
            double x = pos[3 * iPoint], y = pos[3 * iPoint + 1],
                   z = pos[3 * iPoint + 2];
            if (y >= 0.56379 || y <= -0.24046) {
                continue;
            }

            for (int j = 0; j < 3; ++j) {
                int p = triIds[3 * iTri + j];
                double dist2 = getDist2(triPos[3 * p], triPos[3 * p + 1],
                                        triPos[3 * p + 2], x, y, z);
                sumDist2 += dist2;
            }

            heap.push(std::make_pair(sumDist2, iPoint));

            if (heap.size() > numPointsEach) {
                heap.pop();
            }
        }

        assert(heap.size() == numPointsEach);

        while (!heap.empty()) {
            adjIds[num] = heap.top().second;
            adjIdsTri[num] = iTri;
            heap.pop();
            num++;
        }
    }
    cout << num << '\n';
    for (int i = 0; i < num; ++i) {
        cout << adjIdsTri[i] << ' ' << adjIds[i] << '\n';
    }
}
int main() {
    freopen("tri_point_pairs.txt", "w", stdout);
    readOBJFile("starlight_tri.obj");
    int numParticles = (int)vertices.size();
    cerr << "numParticles = " << numParticles << '\n';
    vector<double> pos(3 * numParticles);

    for (int i = 0; i < numParticles; ++i) {
        pos[3 * i] = vertices[i][0];
        pos[3 * i + 1] = vertices[i][1];
        pos[3 * i + 2] = vertices[i][2];
    }

    int numTris = triangles.size();
    vector<int> hostTriIds(3 * numTris);
    for (int i = 0; i < numTris; ++i) {
        hostTriIds[3 * i] = triangles[i][0];
        hostTriIds[3 * i + 1] = triangles[i][1];
        hostTriIds[3 * i + 2] = triangles[i][2];
    }

    readOBJFile("body.obj");
    auto bodyVertices = vertices;
    vector<double> bodyPos(3 * bodyVertices.size());
    for (int i = 0; i < (int)bodyVertices.size(); ++i) {
        bodyPos[3 * i] = bodyVertices[i][0];
        bodyPos[3 * i + 1] = bodyVertices[i][1];
        bodyPos[3 * i + 2] = bodyVertices[i][2];
    }
    queryAllCloseBodyParticles(hostTriIds, pos, bodyPos, 20);
}
