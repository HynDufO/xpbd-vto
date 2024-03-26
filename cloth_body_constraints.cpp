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
    return (x - x1) * (x - x1) + (y - y1) * (y - y1) +
           (z - z1) * (z - z1);
}

void queryClothBodyConstraints(const vector<double> &clothPos,
                               const vector<double> &pos, int numPointsEach,
                               int type) {
    int numPointsCloth = clothPos.size() / 3;
    int numPoints = pos.size() / 3;
    vector<pair<int, int>> vec;

    for (int i = 0; i < numPointsCloth; ++i) {
        if (i == 4692 || i == 13768 || i == 9806 || i == 9812 || i == 19471 || i == 19464 || i == 2500 || i == 19133 || i == 13737 || i == 13748 || i == 13418 || i == 8188 || i == 3100 || i == 12064 || i == 12044 || i == 11951 || i == 1366 || i == 11651 || i == 12521) {
            cerr << "i = " << i << '\n';
            double mnDist = 1e9;
            int mnID = 0;

            for (int iPoint = 0; iPoint < numPoints; ++iPoint) {
                double dist = getDist2(clothPos[3 * i], clothPos[3 * i + 1],
                                       clothPos[3 * i + 2], pos[3 * iPoint],
                                       pos[3 * iPoint + 1], pos[3 * iPoint + 2]);
                if (dist < mnDist) {
                    mnDist = dist;
                    mnID = iPoint;
                }
            }
            vec.push_back({i, mnID});
        }
    }
    cout << vec.size() << '\n';
    for (auto v : vec) {
        cout << v.first << ' ' << v.second << '\n';
    }
}
int main() {
    freopen("cloth_body_constraints.txt", "w", stdout);
    readOBJFile("starlight_tri.obj");
    int numParticles = (int)vertices.size();
    cerr << "numParticles = " << numParticles << '\n';
    vector<double> pos(3 * numParticles);

    for (int i = 0; i < numParticles; ++i) {
        pos[3 * i] = vertices[i][0];
        pos[3 * i + 1] = vertices[i][1];
        pos[3 * i + 2] = vertices[i][2];
    }

    readOBJFile("body.obj");
    auto bodyVertices = vertices;
    vector<double> bodyPos(3 * bodyVertices.size());
    for (int i = 0; i < (int)bodyVertices.size(); ++i) {
        bodyPos[3 * i] = bodyVertices[i][0];
        bodyPos[3 * i + 1] = bodyVertices[i][1];
        bodyPos[3 * i + 2] = bodyVertices[i][2];
    }

    queryClothBodyConstraints(pos, bodyPos, 30, 0);
}
