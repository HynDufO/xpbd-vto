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
    return (x - x1) * (x - x1) + (y - y1) * (y - y1) * 0.8 + (z - z1) * (z - z1);
}

void queryAllCloseBodyParticles(const vector<int> &triIds,
                                const vector<double> &triPos,
                                const vector<double> &pos, int numPointsEach, int type) {
    int numTris = triIds.size() / 3;
    int numPoints = pos.size() / 3;
    vector<int> adjIds, adjIdsTri;

    for (int iTri = 0; iTri < numTris; ++iTri) {
        if (iTri % 1000 == 0) {
            cerr << "iTri = " << iTri << '\n';
        }

        if (type == 1) {
            bool ok = false;
            for (int j = 0; j < 3; ++j) {
                int p = triIds[3 * iTri + j];
                if (triPos[3 * p + 1] < 0.33 && triPos[3 * p + 1] > -0.434) {
                    ok = true;
                    break;
                }
            }
            if (!ok) {
                continue;
            }
        }
        int pp = triIds[3 * iTri];
        if (triPos[3 * pp] < -0.6 || triPos[3 * pp] > 0.6) {
            numPointsEach = 150;
        } else if (triPos[3 * pp] < -0.5 || triPos[3 * pp] > 0.5) {
            numPointsEach = 150;
        } else if (triPos[3 * pp] < -0.4 || triPos[3 * pp] > 0.4) {
            numPointsEach = 150;
        } else if (triPos[3 * pp + 1] > 0.2 && triPos[3 * pp + 1] < 0.27) {
            numPointsEach = 150;
        } else {
            numPointsEach = 30;
        }
        priority_queue<pair<double, int>> heap;

        for (int iPoint = 0; iPoint < numPoints; ++iPoint) {
            double sumDist2 = 0;
            double x = pos[3 * iPoint], y = pos[3 * iPoint + 1],
                   z = pos[3 * iPoint + 2];
            if (type == 0) {
                if (y >= 0.35 || y <= -0.454) {
                    continue;
                }
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
            adjIds.push_back(heap.top().second);
            adjIdsTri.push_back(iTri);
            heap.pop();
        }
    }
    cout << adjIds.size() << '\n';
    for (int i = 0; i < adjIds.size(); ++i) {
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

    numTris = triangles.size();
    vector<int> hostBodyTriIds(3 * numTris);
    for (int i = 0; i < numTris; ++i) {
        hostBodyTriIds[3 * i] = triangles[i][0];
        hostBodyTriIds[3 * i + 1] = triangles[i][1];
        hostBodyTriIds[3 * i + 2] = triangles[i][2];
    }
    queryAllCloseBodyParticles(hostTriIds, pos, bodyPos, 20, 0);
    queryAllCloseBodyParticles(hostBodyTriIds, bodyPos, pos, 20, 1);
}
