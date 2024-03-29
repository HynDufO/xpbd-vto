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

double getDist2(double x, double y, double z, double x1, double y1, double z1, double xc = 1.0, double yc = 1.0, double zc = 1.0) {
    return (x - x1) * (x - x1) * xc + (y - y1) * (y - y1) * yc + (z - z1) * (z - z1) * zc;
}

void getClosestBodyPoint(const vector<double> &pos,
                                const vector<double> &bodyPos) {
    for (int i = 0; i < (int) pos.size(); i += 3) {
        int mnID = -1;
        double mnDist = 1e9;
        for (int j = 0; j < (int) bodyPos.size(); j += 3) {
            double dist = 0;

            if (abs(pos[i]) > 0.26) {
                dist = getDist2(pos[i], pos[i + 1], pos[i + 2], bodyPos[j], bodyPos[j + 1], bodyPos[j + 2], 1, 0.1, 1);
            } else {
                dist = getDist2(pos[i], pos[i + 1], pos[i + 2], bodyPos[j], bodyPos[j + 1], bodyPos[j + 2]);
            }
            if (mnDist > dist) {
                mnDist = dist;
                mnID = j;
            }
        }
        cout << i / 3 << ' ' << mnID / 3 << '\n';
    }

}
int main() {
    freopen("corr_weights.txt", "w", stdout);
    readOBJFile("starlight_tri_before.obj");
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

    getClosestBodyPoint(pos, bodyPos);
}
